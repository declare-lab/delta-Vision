"""Same Adapter, compact text execution vs independent full-sequence HF layers.

Trace separator tokens through norm, attention, residual and FFN. No ablation:
both implementations receive identical adapter memories at identical positions.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_text_trajectory_parity_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
                                                          torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.adapter_start_layer==adapter.active_adapter_layers==0
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    lm=model.model.language_model
    lm.register_forward_pre_hook(lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    lm._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
                           data_root=str(ROOT/'data/benchmarks/muirbench'),prompt_layout='media_first_v1')
    index=next(i for i,r in enumerate(ds.rows) if len(r['images'])==shard+2)
    inputs=_qwen_inputs_from_item(ds[index],torch.device('cuda'))
    visual=inputs['mm_token_type_ids'][0].ne(0);vp=visual.nonzero().flatten();tp=(~visual).nonzero().flatten()
    cuts=[0]+(torch.where(vp[1:]-vp[:-1]!=1)[0]+1).tolist()+[len(vp)]
    separator=torch.zeros(len(tp),device='cuda',dtype=torch.bool)
    ids=inputs['input_ids'][0].tolist(); inverse={p:j for j,p in enumerate(tp.tolist())}
    for g,(a,b) in enumerate(zip(cuts[:-1],cuts[1:]),1):
        start=int(vp[a])-1;end=int(vp[b-1])+1
        separator[inverse[start]]=True
        after=processor.tokenizer.encode(f'<|vision_end|>\n[End of Image {g}]\n',add_special_tokens=False)
        assert ids[end:end+len(after)]==after
        separator[[inverse[p] for p in range(end,end+len(after))]]=True
    report=dict(index=index,images=shard+2,separator_tokens=int(separator.sum()),phases={})
    with torch.inference_mode():
        for precision in ('bf16','fp32'):
            if precision=='fp32':
                lm.float();model.lm_head.float();adapter.float();adapter.precompute_stacked_weights()
            initial,pos=ref.build_qwen_initial_context(model,inputs)
            memory=adapter.all_visual_memories_batched(initial[:,visual])
            current={'mode':'compact'};captured={};counts={};handles=[]
            def save(layer,stage,value):
                mode=current['mode']; key=(mode,layer,stage)
                counts[key]=counts.get(key,0)+1
                if stage=='input' and mode=='compact' and counts[key]==2:return
                assert key not in captured,(key,counts[key])
                selected=value[:,tp] if mode=='full' else value
                assert selected.shape[1]==len(tp)
                captured[key]=selected.clone()
            def pre(layer,stage):
                def hook(module,args):save(layer,stage,args[0])
                return hook
            def post(layer,stage):
                def hook(module,args,output):save(layer,stage,output)
                return hook
            for l,layer in enumerate(lm.layers):
                handles.extend([
                    layer.input_layernorm.register_forward_pre_hook(pre(l,'input')),
                    layer.self_attn.o_proj.register_forward_hook(post(l,'attention_output')),
                    layer.post_attention_layernorm.register_forward_pre_hook(pre(l,'attention_residual')),
                    layer.mlp.register_forward_hook(post(l,'ffn_output'))])
            original_memories=adapter.all_visual_memories_batched
            adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,**k:memory,adapter)
            try:
                compact=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,
                                                         position_ids=pos,logits_to_keep=1)[0].float()
                current['mode']='full';swap_handles=[]
                def swap(l):
                    def hook(module,args,kwargs):
                        old=args[0] if args else kwargs['hidden_states']
                        if l==0:assert torch.equal(old,initial),'Different initial sequence'
                        h=old.clone();h[:,visual]=memory[l].to(h)
                        assert torch.equal(h[:,tp],old[:,tp]),'Text/markers overwritten by visual replacement'
                        return ((h,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=h))
                    return hook
                try:
                    for l,layer in enumerate(lm.layers):
                        swap_handles.append(layer.register_forward_pre_hook(swap(l),with_kwargs=True))
                    model.model.rope_deltas=None
                    full=model(**inputs,use_cache=False,logits_to_keep=1).logits.float()
                finally:
                    for h in swap_handles:h.remove()
            finally:
                adapter.all_visual_memories_batched=original_memories
                for h in handles:h.remove()
            checks=[]
            for l in range(36):
                for stage in ('input','attention_output','attention_residual','ffn_output'):
                    assert counts['compact',l,stage]==(2 if stage=='input' else 1)
                    assert counts['full',l,stage]==1
                    a=captured['compact',l,stage].float();b=captured['full',l,stage].float()
                    relative=float((a-b).norm()/b.norm().clamp_min(1e-12))
                    sep_relative=float((a[:,separator]-b[:,separator]).norm()/b[:,separator].norm().clamp_min(1e-12))
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    if precision=='fp32':assert max(relative,sep_relative)<.001,(l,stage,relative,sep_relative)
                    checks.append(dict(layer=l,stage=stage,all_text_rel_error=relative,separator_rel_error=sep_relative))
            logit_error=float((full-compact).norm()/full.norm().clamp_min(1e-12))
            report['phases'][precision]=dict(stages=checks,logit_rel_error=logit_error,
                same_prediction=bool(full.argmax()==compact.argmax()),
                compact_prediction=processor.tokenizer.decode([int(compact.argmax())]),
                full_prediction=processor.tokenizer.decode([int(full.argmax())]),
                initial_sequence_exact=True,visual_replacement_preserves_text_exactly=True,
                max_separator_rel_error=max(r['separator_rel_error'] for r in checks))
            del captured,memory,initial
            print('PASS',index,precision,'logit rel',logit_error,flush=True)
    (OUT/f'row_{shard}.json').write_text(json.dumps(report,indent=2)+'\n')


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for s in range(8):
        log=(OUT/f'worker{s}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_text_trajectory_parity',str(s)],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
                    stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads((OUT/f'row_{s}.json').read_text()) for s in range(8)]
    result=dict(samples=8,image_counts=[r['images'] for r in rows],layers_per_precision=288,
        precisions={p:dict(max_separator_rel_error=max(r['phases'][p]['max_separator_rel_error'] for r in rows),
                          max_logit_rel_error=max(r['phases'][p]['logit_rel_error'] for r in rows),
                          same_prediction_count=sum(r['phases'][p]['same_prediction'] for r in rows))
                    for p in ('bf16','fp32')})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
