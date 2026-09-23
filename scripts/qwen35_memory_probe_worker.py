"""Workers for the three Qwen3.5 recurrent-memory questions. No training."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35_experiment import load_model,initial_context,prepare_inputs,generate_evaluation_answer,sha
from src.benchmarks import build_benchmark_prompt,get_benchmark_spec
from src.qwen35_memory_probe import (RANKS,matrix_similarity,spectrum_metrics,decompose,truncate,
    subspace_overlap,next_token_kl,span,project_linear,kernel,boundary_states,linear_state_output,
    finish_linear,text_effect_rank,mixer_output,layer_finish,capture,tail_logits,state_intervention)


def cpu_json(x):
    return json.dumps(x,ensure_ascii=False,allow_nan=False)


def scalar_difference(a,b):
    return matrix_similarity(a.float().reshape(1,-1),b.float().reshape(1,-1))


def full_attention_blocked(layer,h,kwargs,mask):
    text=(~mask[0]).nonzero().flatten()
    pos=tuple(x.index_select(-2,text) for x in kwargs['position_embeddings'])
    ids=kwargs.get('position_ids')
    if ids is not None: ids=ids.index_select(-1,text)
    # FA2 over text-only Q/K/V, preserving original rotary positions. Visual
    # query outputs are irrelevant to this local text-readout comparison.
    out,_=layer.self_attn(layer.input_layernorm(h.index_select(1,text)),
        position_embeddings=pos,position_ids=ids,attention_mask=None,past_key_values=None)
    return out,text


@torch.inference_mode()
def analyze(model,controller,context,mask,predictions):
    traces={}; logits={}
    for method in ['native','adapter']:
        traces[method],logits[method]=capture(model,controller,context,mask,method)
    start,end=span(mask)
    records=[]
    for i,layer in enumerate(model.model.language_model.layers):
        teacher_delta=teacher_svd=None
        for method in ['native','adapter']:
            trace=traces[method][i];h=trace['hidden']; kw=trace['kwargs']
            norm=layer.input_layernorm(h)
            rec={'method':method,'layer':i,'block_type':layer.block_type,'visual_tokens':end-start,
                 'post_visual_text_tokens':h.shape[1]-end}
            if layer.block_type=='linear_attention':
                parts=project_linear(layer.linear_attn,norm)
                core,_=kernel(parts)
                full=finish_linear(layer.linear_attn,core,parts['z'])
                sin,sout=boundary_states(parts,start,end)
                delta=(sout-sin)[0];svd=decompose(delta)
                rec['state_rank']=spectrum_metrics(svd[1],128)
                rec['state_in_frobenius']=sin.float().norm().item()
                rec['state_out_frobenius']=sout.float().norm().item()
                rec['state_delta_frobenius']=delta.norm().item()
                # Suppressing beta alone keeps forgetting: document this distinct
                # counterfactual rather than silently calling it frozen state.
                decay_only=sin*parts['g'][:,start:end].sum(1).exp()[...,None,None]
                rec['freeze_vs_decay_only']=scalar_difference(decay_only,sin)
                blocked,_=linear_state_output(layer.linear_attn,parts,start,end,sin,core)
                restored,_=linear_state_output(layer.linear_attn,parts,start,end,sout,core)
                rec['segmented_full_vs_native']=scalar_difference(restored[:,end:],full[:,end:])
                rec['text_effect_rank']=text_effect_rank(full[:,end:].float()-blocked[:,end:].float())
                rec['text_effect_frobenius']=(full[:,end:].float()-blocked[:,end:].float()).norm().item()
                no_conv=project_linear(layer.linear_attn,norm,erase_visual_conv=(start,end))
                both_blocked,_=linear_state_output(layer.linear_attn,no_conv,start,end,sin)
                rec['text_effect_rank_state_and_conv_ablation']=text_effect_rank(full[:,end:].float()-both_blocked[:,end:].float())
                rec['conv_bypass_effect_frobenius']=(both_blocked[:,end:].float()-blocked[:,end:].float()).norm().item()
                rec['svd_reconstruction']={str(r):matrix_similarity(truncate(svd,r),delta) for r in RANKS}
                if method=='native':
                    teacher_delta,teacher_svd=delta,svd
                    matched=h.index_copy(1,mask[0].nonzero().flatten(),predictions[i].to(h.dtype))
                    ap=project_linear(layer.linear_attn,layer.input_layernorm(matched))
                    ain,aout=boundary_states(ap,start,end)
                    torch.testing.assert_close(ain,sin,rtol=0,atol=0)
                    adelta=(aout-ain)[0]; asvd=decompose(adelta)
                    rec['representation_pre_norm']=matrix_similarity(predictions[i][0],h[0,start:end])
                    rec['representation_post_norm']=matrix_similarity(layer.input_layernorm(predictions[i])[0],norm[0,start:end])
                    rec['adapter_state_matched_context']=matrix_similarity(adelta,delta)
                    rec['adapter_state_subspace_overlap']=subspace_overlap(svd,asvd)
                else:
                    rec['adapter_state_actual_trajectory']=matrix_similarity(delta,teacher_delta)
                    rec['adapter_actual_subspace_overlap']=subspace_overlap(teacher_svd,svd)
            else:
                full=mixer_output(layer,h,kw)
                blocked,ti=full_attention_blocked(layer,h,kw,mask)
                after=ti>=end
                effect=full.index_select(1,ti)[:,after].float()-blocked[:,after].float()
                rec['text_effect_rank']=text_effect_rank(effect)
                rec['text_effect_frobenius']=effect.float().norm().item()
                if method=='native':
                    rec['representation_pre_norm']=matrix_similarity(predictions[i][0],h[0,start:end])
                    rec['representation_post_norm']=matrix_similarity(layer.input_layernorm(predictions[i])[0],norm[0,start:end])
            records.append(rec)
    return {'layers':records,'native_adapter_next_token_kl':next_token_kl(logits['adapter'],logits['native']).item()}


@torch.inference_mode()
def sensitivity(model,controller,context,mask,predictions,seed,magnitudes):
    start,end=span(mask);vi=mask[0].nonzero().flatten();ti=(~mask[0]).nonzero().flatten()
    generator=torch.Generator(device=context['inputs_embeds'].device).manual_seed(seed)
    direction=torch.randn(predictions[0].shape,device=context['inputs_embeds'].device,dtype=torch.float32,generator=generator)
    direction=direction/direction.norm()
    records=[]
    for method in ['native','adapter']:
        traces,original_logits=capture(model,controller,context,mask,method)
        for i,layer in enumerate(model.model.language_model.layers):
            h=traces[i]['hidden']; base_visual=h[:,start:end].float()
            outputs=[]; input_hidden=[]; actual=[];mixed_outputs=[]
            for magnitude in magnitudes:
                visual=(base_visual+direction*(magnitude*base_visual.norm())).to(h.dtype)
                changed=h.index_copy(1,vi,visual)
                mixed=mixer_output(layer,changed,traces[i]['kwargs'])
                out=layer_finish(layer,changed,mixed,text_idx=ti if method=='adapter' else None)
                outputs.append(out);mixed_outputs.append(mixed[:,end:]);input_hidden.append(changed)
                actual.append(((visual.float()-base_visual).norm()/base_visual.norm().clamp_min(1e-30)).item())
            # Batch only the suffix replays, including an epsilon=0 replay to
            # control BF16 GEMM changes caused by batch size.
            combined=torch.cat(outputs)
            replay=tail_logits(model,traces,i,combined,predictions if method=='adapter' else None,mask)
            kls=next_token_kl(replay,replay[:1].expand_as(replay))
            zero_kl=next_token_kl(replay[:1],original_logits).item()
            for j,magnitude in enumerate(magnitudes):
                records.append(dict(method=method,layer=i,block_type=layer.block_type,
                    magnitude=magnitude,actual_magnitude=actual[j],
                    text_attention_error=scalar_difference(mixed_outputs[j],mixed_outputs[0]),
                    text_hidden_error=scalar_difference(outputs[j][:,end:],outputs[0][:,end:]),
                    next_token_kl=kls[j].item(),zero_replay_vs_original_kl=zero_kl))
    return {'perturbations':records}


@torch.inference_mode()
def evaluate(model,processor,controller,inputs,context,mask,predictions,row,spec,config,info):
    with controller.activate('native',mask):
        ref=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
    records=[]
    cases=[(m,None) for m in ['native','adapter']]+[(m,r) for m in ['native','adapter'] for r in RANKS]+[('state_adapter',None)]
    for method,rank in cases:
        adapter_mode='adapter' if method=='adapter' else 'native'
        intervention=(state_intervention(model,mask,memory=predictions) if method=='state_adapter' else
                      state_intervention(model,mask,rank=rank) if rank is not None else nullcontext())
        with controller.activate(adapter_mode,mask), intervention:
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            generated=generate_evaluation_answer(model,processor,inputs,row,spec,config,max_new_tokens=info['max_new_tokens'])
        records.append(dict(method=method,rank=rank,kl_to_native=next_token_kl(logits,ref).item(),**generated))
    return {'variants':records}


@torch.inference_mode()
def validate(model,processor,controller,inputs,context,mask,predictions,row,spec,config,info):
    """Native decomposition, state boundaries, r0/r128 and cached-generation gates."""
    start,end=span(mask); records=[]
    for method in ['native','adapter']:
        traces,ref=capture(model,controller,context,mask,method)
        with controller.activate(method,mask):
            original=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        torch.testing.assert_close(original,ref,rtol=0,atol=0)
        for i,layer in enumerate(model.model.language_model.layers):
            if layer.block_type!='linear_attention':continue
            h=traces[i]['hidden'];norm=layer.input_layernorm(h)
            parts=project_linear(layer.linear_attn,norm); core,_=kernel(parts)
            full=finish_linear(layer.linear_attn,core,parts['z'])
            native=mixer_output(layer,h,traces[i]['kwargs'])
            torch.testing.assert_close(full,native,rtol=0,atol=0)
            sin,sout=boundary_states(parts,start,end)
            reconstructed,_=linear_state_output(layer.linear_attn,parts,start,end,sout,core)
            error=scalar_difference(reconstructed[:,end:],full[:,end:])
            assert error['normalized_frobenius'] < .03,(method,i,error)
            # Identity transitions are g=0 AND beta=0 (not just beta=0).
            frozen=dict(parts)
            frozen['g']=parts['g'].clone();frozen['beta']=parts['beta'].clone()
            frozen['g'][:,start:end]=0;frozen['beta'][:,start:end]=0
            _,freeze_state=kernel(frozen,start,end,sin)
            torch.testing.assert_close(freeze_state,sin,rtol=2e-5,atol=2e-5)
            frozen_core,_=kernel(frozen)
            direct_zero=finish_linear(layer.linear_attn,frozen_core,parts['z'])
            paired_zero,_=linear_state_output(layer.linear_attn,parts,start,end,sin,core)
            zero_error=scalar_difference(paired_zero[:,end:],direct_zero[:,end:])
            assert zero_error['normalized_frobenius'] < .03,(method,i,'rank0 vs direct identity transitions',zero_error)
            delta=sout-sin
            svd_restored=truncate(decompose(delta),128)
            svd_error=scalar_difference(svd_restored,delta)
            # A scale-free reconstruction bound is meaningful for large state
            # entries; elementwise relative error near zero is not.
            assert svd_error['normalized_frobenius'] < 2e-5,(method,i,svd_error)
            records.append(dict(method=method,layer=i,segmentation=error,full_svd_error=svd_error,rank0_vs_direct=zero_error))
        with controller.activate(method,mask),state_intervention(model,mask,rank=128):
            restored=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        kl=next_token_kl(restored,ref).item()
        assert kl<.01,(method,'rank128 logits KL',kl)
        records.append(dict(method=method,rank128_kl=kl,rank128_argmax_equal=bool(torch.equal(restored.argmax(-1),ref.argmax(-1)))))
    generated=evaluate(model,processor,controller,inputs,context,mask,predictions,row,spec,config,info)
    for method in ['native','adapter']:
        baseline=next(v for v in generated['variants'] if v['method']==method and v['rank'] is None)
        full=next(v for v in generated['variants'] if v['method']==method and v['rank']==128)
        assert baseline['generated_token_ids']==full['generated_token_ids'],(method,'full-rank generation mismatch',baseline,full)
    return {'checks':records,'generation_checks':generated}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['validate','analysis','accuracy','sensitivity'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);p.add_argument('--limit',type=int);args=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(44)
    config=json.loads((args.run/'config.json').read_text())
    processor,model,adapter,controller=load_model(config,torch.device('cuda:0'))
    # FP32 SVD reconstruction/overlap must not use TF32. Backbone remains BF16.
    torch.set_float32_matmul_precision('highest')
    assert sha(config['adapter_checkpoint'])==config['adapter_checkpoint_sha256']
    ckpt=torch.load(config['adapter_checkpoint'],map_location='cpu',weights_only=False)
    assert ckpt['global_step']==2000
    adapter.load_state_dict(ckpt['state_dict'],strict=True);adapter.eval().requires_grad_(False);del ckpt
    for benchmark,info in config['evaluation'].items():
        rows=[json.loads(line) for line in Path(info['path']).read_text().splitlines()]
        assert len(rows)==info['samples'] and sha(info['path'])==info['sha256']
        path=args.run/args.stage/f'{benchmark}.shard{args.shard}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        old=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        indices=list(range(args.shard,len(rows),args.shards))
        if args.limit is not None:indices=indices[:args.limit]
        assert [r['index'] for r in old]==indices[:len(old)]
        with path.open('a') as handle,torch.inference_mode():
            for index in indices[len(old):]:
                begin=time.time();row=rows[index];spec=get_benchmark_spec(benchmark)
                inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
                context=initial_context(model,inputs);mask=inputs['mm_token_type_ids'].eq(1);start,end=span(mask)
                predictions=adapter(context['inputs_embeds'][:,start:end])
                if args.stage=='analysis':result=analyze(model,controller,context,mask,predictions)
                elif args.stage=='sensitivity':result=sensitivity(model,controller,context,mask,predictions,44+index,config['perturbation_magnitudes'])
                else:
                    fn=validate if args.stage=='validate' else evaluate
                    result=fn(model,processor,controller,inputs,context,mask,predictions,row,spec,config,info)
                result.update(index=index,benchmark=benchmark,elapsed_s=time.time()-begin,
                    input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                    visual_tokens=end-start,sequence_length=mask.shape[1],image_grid_thw=inputs['image_grid_thw'].tolist())
                handle.write(cpu_json(result)+'\n');handle.flush()
                print(cpu_json(dict(stage=args.stage,benchmark=benchmark,index=index,elapsed_s=result['elapsed_s'])),flush=True)
                del inputs,context,predictions,result
    controller.close()

if __name__=='__main__':main()
