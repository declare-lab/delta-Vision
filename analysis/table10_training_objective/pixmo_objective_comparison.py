"""PixMo-only objective ablations; backbone frozen, exact on-policy token trajectories."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


class FullVocabKL(torch.autograd.Function):
    """Exact teacher->student KL, token-chunked FP32 normalizers and analytical gradient."""
    @staticmethod
    def forward(ctx, student, teacher, temperature, chunk):
        if student.shape != teacher.shape or student.ndim != 2 or not len(student):
            raise ValueError('KL requires nonempty, equally shaped [answer_tokens,vocabulary] logits')
        ctx.save_for_backward(student, teacher)
        ctx.temperature,ctx.chunk = temperature,chunk
        loss=student.new_zeros((),dtype=torch.float32)
        for i in range(0,len(student),chunk):
            s=F.log_softmax(student[i:i+chunk].float()/temperature,dim=-1)
            t=F.log_softmax(teacher[i:i+chunk].float()/temperature,dim=-1)
            loss+=(t.exp()*(t-s)).sum()
        return loss*(temperature**2/len(student))

    @staticmethod
    def backward(ctx, scale):
        student,teacher=ctx.saved_tensors
        grad=torch.empty_like(student)
        for i in range(0,len(student),ctx.chunk):
            p=F.softmax(student[i:i+ctx.chunk].float()/ctx.temperature,dim=-1)
            q=F.softmax(teacher[i:i+ctx.chunk].float()/ctx.temperature,dim=-1)
            grad[i:i+ctx.chunk]=(p-q)*(scale*ctx.temperature/len(student))
        return grad,None,None,None


def trajectory_inputs(prompt, token_columns, active_columns):
    """Append exact sampled IDs, never decode/re-tokenize or append an artificial EOS."""
    result=dict(prompt)
    tokens=torch.stack(token_columns,dim=1)
    active=torch.stack(active_columns,dim=1).to(prompt['attention_mask'].dtype)
    result['input_ids']=torch.cat([prompt['input_ids'],tokens],dim=1)
    result['attention_mask']=torch.cat([prompt['attention_mask'],active],dim=1)
    result['mm_token_type_ids']=torch.cat([prompt['mm_token_type_ids'],torch.zeros_like(active)],dim=1)
    answer_mask=torch.cat([torch.zeros_like(prompt['attention_mask']),active],dim=1).bool()
    return result,answer_mask


@torch.no_grad()
def sample_trajectory(model,adapter,processor,prompt,max_new_tokens):
    from src.model import (build_qwen_initial_context,qwen_embedding_adapter_prefill_cache,
                           qwen_embedding_adapter_decode_step_shape_exact)
    from src.evaluate import _next_token_logits_batch, _eos_token_ids
    hidden,pos=build_qwen_initial_context(model,prompt)
    logits,text_mask,cache=qwen_embedding_adapter_prefill_cache(
        model,adapter,prompt['input_ids'],prompt['attention_mask'],prompt['mm_token_type_ids'],hidden,pos,logits_to_keep=1)
    active=torch.ones(len(prompt['input_ids']),dtype=torch.bool,device=logits.device)
    eos=_eos_token_ids(processor.tokenizer)
    pad=processor.tokenizer.pad_token_id
    if pad is None:pad=next(iter(eos))
    tokens,masks=[],[]
    for step in range(max_new_tokens):
        # Temperature 1, no top-k/p truncation: the student's current sampling policy.
        nxt=torch.multinomial(F.softmax(_next_token_logits_batch(logits,text_mask).float(),-1),1).squeeze(1)
        nxt=torch.where(active,nxt,torch.full_like(nxt,pad))
        tokens.append(nxt);masks.append(active.clone())
        for end in eos:active=active & nxt.ne(end)
        if not active.any() or step+1==max_new_tokens:break
        logits,cache=qwen_embedding_adapter_decode_step_shape_exact(
            model,adapter,nxt[:,None],cache,logits_to_keep=1,token_active_mask=masks[-1])
        text_mask=torch.ones((len(nxt),1),device=logits.device,dtype=torch.bool)
    return trajectory_inputs(prompt,tokens,masks),float(active.float().mean())


def opd_loss(args,processor,model,adapter,rows,device,dtype,num_layers):
    from src.model import (prepare_qwen3vl_batch_inputs,qwen_position_ids,get_qwen_text_image_positions,
                           gather_batched_positions,qwen_embedding_adapter_logits)
    from src.training.engine import masked_topk_kl
    start=time.perf_counter()
    prompt,_,_,paths=prepare_qwen3vl_batch_inputs(processor,rows,Path(args.image_root),device,include_answers=False)
    (inputs,answer_full),truncated=sample_trajectory(model,adapter,processor,prompt,args.opd_max_new_tokens)
    rollout_s=time.perf_counter()-start
    with torch.no_grad():
        pos=qwen_position_ids(model,inputs)
        text_positions,_,_,text_mask,_,_=get_qwen_text_image_positions(
            inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],pos)
        teacher=model(**inputs,return_dict=True,use_cache=False)
        teacher_logits=gather_batched_positions(teacher.logits,text_positions,text_mask)
        del teacher
        ids=gather_batched_positions(inputs['input_ids'].unsqueeze(-1),text_positions,text_mask).squeeze(-1)
        mask=gather_batched_positions(answer_full.unsqueeze(-1),text_positions,text_mask).squeeze(-1).bool()
    student_logits,student_mask,_=qwen_embedding_adapter_logits(model,adapter,inputs)
    if not torch.equal(student_mask.bool(),text_mask.bool()):raise RuntimeError('OPD text masks differ')
    selected=mask[:,1:] & text_mask[:,1:].bool()
    if args.opd_vocab=='full':
        loss=FullVocabKL.apply(student_logits[:,:-1][selected],teacher_logits[:,:-1][selected].detach(),args.temperature,32)
    else:
        loss,_,_=masked_topk_kl(student_logits,teacher_logits,ids,mask,args.temperature,1024,normalization='token',calibration='none')
    loss=args.lambda_logit*loss
    metrics=dict(loss=float(loss.detach()),ce=0.,logit_kl=float(loss.detach()),
                 answer_tokens=float(selected.sum())/len(rows),text_tokens=float(text_mask.sum())/len(rows),
                 opd_rollout_s=rollout_s,opd_truncated_fraction=truncated,batch_size=len(rows),
                 image=paths[0],supervision_loss='opd',opd_vocab=args.opd_vocab)
    return loss,metrics


def main():
    from src.training import engine as train
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    opts=p.parse_args()
    config=json.loads(Path(opts.config).read_text())
    args=argparse.Namespace(**config)
    # Current shared trainer requires this flag; the project policy is DeepStack off.
    args.teacher_deepstack=getattr(args, "teacher_deepstack", False)
    if args.experiment=='opd':
        if args.output_mode!='embedding_adapter' or args.adapter_start_layer!=0:
            raise ValueError('OPD currently requires the all-layer static adapter')
        train.compute_loss_for_rows=opd_loss
    train.run_qwen(args)


if __name__=='__main__':main()
