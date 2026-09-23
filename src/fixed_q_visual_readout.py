"""Fixed native text-query readout, full embedding adapter, MMStar 1500.

No training or label-derived subspace. All nonvisual K/V and query states stay
native; only visual K/V differ. Full causal softmax denominator is preserved.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from src.embedding_task_subspace import (MODEL,CHECKPOINT,REFERENCE_SRC,DATA,LAYERS,
    Capture,Stats,reference_module,inputs_for,native_forward,sha,dump)

ROOT=Path(__file__).resolve().parents[1]

def compare_rows(pred,native):
    p,y=pred.double().flatten(1),native.double().flatten(1)
    denom=p.norm(dim=1)*y.norm(dim=1);valid=denom>1e-20
    return dict(sse=float((p-y).square().sum()),energy=float(y.square().sum()),
                cosine_sum=float(((p*y).sum(1)[valid]/denom[valid]).sum()),valid=int(valid.sum()))

def visible_top_overlap(scores_t,scores_p,visible,top=10):
    """Rank logits, not underflowed probabilities; never select a future key."""
    overlap=[]
    for j in range(scores_t.shape[0]):
        k=min(top,int(visible[j].sum()))
        assert k>0
        t=scores_t[j].masked_fill(~visible[j],-float('inf'))
        p=scores_p[j].masked_fill(~visible[j],-float('inf'))
        it=t.topk(k,dim=-1).indices;ip=p.topk(k,dim=-1).indices
        assert visible[j][it].all() and visible[j][ip].all()
        overlap.append((it[:,:,None]==ip[:,None,:]).any(-1).float().mean())
    return torch.stack(overlap)

def fixed_readout(layer,hidden,pred,visual,positions):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,repeat_kv
    attention=layer.self_attn
    norm=layer.input_layernorm(hidden)
    altered=norm.clone();altered[:,visual]=layer.input_layernorm(pred.unsqueeze(0))
    shape=(*hidden.shape[:-1],-1,attention.head_dim)
    q=attention.q_norm(attention.q_proj(norm).view(shape)).transpose(1,2)
    kt=attention.k_norm(attention.k_proj(norm).view(shape)).transpose(1,2)
    kp=attention.k_norm(attention.k_proj(altered).view(shape)).transpose(1,2)
    vt=attention.v_proj(norm).view(shape).transpose(1,2)
    vp=attention.v_proj(altered).view(shape).transpose(1,2)
    qt,kt=apply_rotary_pos_emb(q,kt,*positions)
    _,kp=apply_rotary_pos_emb(q,kp,*positions)
    other=torch.ones(hidden.shape[1],device=hidden.device,dtype=torch.bool);other[visual]=False
    # The two projection calls see identical native text states; check bitwise.
    assert torch.equal(kt[:,:,other],kp[:,:,other]) and torch.equal(vt[:,:,other],vp[:,:,other])
    kv_metrics=(kt[0,:,visual].transpose(0,1).flatten(1),kp[0,:,visual].transpose(0,1).flatten(1),
                vt[0,:,visual].transpose(0,1).flatten(1),vp[0,:,visual].transpose(0,1).flatten(1))
    heads=qt.shape[1];groups=heads//kt.shape[1]
    kt,kp,vt,vp=[repeat_kv(x,groups).float() for x in (kt,kp,vt,vp)]
    # Only text queries with at least one causal visual key. Include final
    # assistant answer-boundary token; report it separately as well.
    queries=(other & (torch.arange(len(other),device=hidden.device)>visual[0])).nonzero().flatten()
    assert len(queries)>0 and queries[-1]==hidden.shape[1]-1
    output=[]
    for ids in queries.split(64):
        qfixed=qt[:,:,ids].float()
        st=(qfixed@kt.transpose(-1,-2))*attention.scaling
        sp=(qfixed@kp.transpose(-1,-2))*attention.scaling
        causal=torch.arange(hidden.shape[1],device=hidden.device)[None,:]<=ids[:,None]
        lt=st.masked_fill(~causal,float('-inf')).log_softmax(-1)
        lp=sp.masked_fill(~causal,float('-inf')).log_softmax(-1)
        at,ap=lt.exp(),lp.exp()
        diff=torch.where(causal,lt-lp,0.)
        kl=(at*diff).sum(-1)[0].transpose(0,1)
        atv,apv=at[:,:,:,visual],ap[:,:,:,visual]
        ct=atv@vt[:,:,visual];cp=apv@vp[:,:,visual]
        ct,cp=[x[0].transpose(0,1).flatten(1) for x in (ct,cp)]
        fullt=(at@vt)[0].transpose(0,1).flatten(1)
        # o_proj bias is NOT part of visual contribution.
        w=attention.o_proj.weight.float()
        outt,outp=ct@w.T,cp@w.T
        # Scores on visible visual entries only; masking to zero excludes
        # future keys from cosine/error without treating -inf as data.
        visible=causal[:,visual]
        sqt=st[0,:,:,visual].transpose(0,1).masked_fill(~visible[:,None,:],0.)
        sqp=sp[0,:,:,visual].transpose(0,1).masked_fill(~visible[:,None,:],0.)
        av_t,av_p=[x[0].transpose(0,1) for x in (atv,apv)]
        overlap=visible_top_overlap(sqt,sqp,visible)
        assert torch.isfinite(kl).all() and torch.isfinite(cp).all() and torch.isfinite(ct).all()
        output.append(dict(ids=ids,ct=ct,cp=cp,outt=outt,outp=outp,sqt=sqt,sqp=sqp,fullt=fullt,
            avt=av_t,avp=av_p,kl=kl,top=overlap,
            mass_t=av_t.sum(-1),mass_p=av_p.sum(-1)))
    return kv_metrics,output

def worker(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    torch.set_num_threads(4);torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(args.output);plan=json.loads((root/'plan.json').read_text())
    assert sha(CHECKPOINT)==plan['checkpoint_sha256'] and sha(__file__)==plan['source_sha256']
    assert sha(REFERENCE_SRC/'model.py')==plan['reference_model_source_sha256']
    assert sha(DATA)==plan['dataset_sha256']
    processor,model=load_frozen_qwen3vl(MODEL,torch.bfloat16,torch.device('cuda:0'),'flash_attention_2')
    adapter,meta=reference_module().load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda:0'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.visual_adapter_rank==128 and not adapter.native_ffn_carriers
    assert len(model.model.language_model.layers)==36
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    data=QwenBenchmarkDataset(str(DATA),processor,'mmstar',max_samples=args.samples)
    assert len(data)==args.samples
    cap=Capture(model);stats={};native_heads={};start=time.time()
    def capture_heads(l):
        def hook(module,a):native_heads[l]=a[0].detach()
        return hook
    handles=[cap.layers[l].self_attn.o_proj.register_forward_pre_hook(capture_heads(l)) for l in LAYERS]
    def add(key,p,y):
        if key not in stats:stats[key]=Stats(y.shape[-1])
        stats[key].add_vectors(p,y)
    with torch.inference_mode(),(root/f'rows_{args.shard}.jsonl').open('w',buffering=1) as file:
        for index in range(args.shard,args.samples,args.world):
            inputs=inputs_for(data[index]);visual=inputs['mm_token_type_ids'][0].eq(1).nonzero().flatten()
            assert inputs['input_ids'].shape[0]==1 and inputs['attention_mask'].bool().all(), 'Only unpadded single requests supported'
            assert len(visual)>0
            cap.reset(visual);native_forward(model,inputs);cap.enabled=False
            memories=adapter.all_visual_memories_batched(cap.initial[:,visual])
            for l in LAYERS:
                hidden=cap.h[l];pred=memories[l,0]
                add(f'{l}/hidden',pred,hidden[0,visual])
                kv,blocks=fixed_readout(cap.layers[l],hidden,pred,visual,cap.kw[l]['position_embeddings'])
                native_error=sum(float((b['fullt'].double()-native_heads[l][0,b['ids']].double()).square().sum()) for b in blocks)
                native_energy=sum(float(native_heads[l][0,b['ids']].double().square().sum()) for b in blocks)
                native_relative_error=math.sqrt(native_error/max(native_energy,1e-30))
                # Offline FP32 math vs fused BF16 attention: tolerate numerical
                # rounding, but abort on a substantive reconstruction mismatch.
                assert native_relative_error<.02,(index,l,native_relative_error)
                if index==args.shard:
                    _,identity=fixed_readout(cap.layers[l],hidden,hidden[0,visual],visual,cap.kw[l]['position_embeddings'])
                    for b in identity:
                        assert torch.equal(b['sqt'],b['sqp']) and torch.equal(b['ct'],b['cp'])
                        assert b['kl'].abs().max()==0
                kt,kp,vt,vp=kv;add(f'{l}/key',kp,kt);add(f'{l}/value',vp,vt)
                for group in ('all_text','answer_boundary'):
                    row=dict(index=index,layer=l,group=group,native_reconstruction_relative_error=native_relative_error,queries=0,heads=0,kl_sum=0.,top_sum=0.,mass_t=0.,mass_p=0.,
                        qk=dict(sse=0.,energy=0.,cosine_sum=0.,valid=0),attn=dict(sse=0.,energy=0.,cosine_sum=0.,valid=0))
                    for b in blocks:
                        take=torch.ones_like(b['ids'],dtype=torch.bool) if group=='all_text' else b['ids']==hidden.shape[1]-1
                        if not take.any():continue
                        add(f'{l}/{group}/readout',b['cp'][take],b['ct'][take])
                        add(f'{l}/{group}/readout_wo',b['outp'][take],b['outt'][take])
                        for key,pk,tk in [('qk','sqp','sqt'),('attn','avp','avt')]:
                            val=compare_rows(b[pk][take].flatten(0,1),b[tk][take].flatten(0,1))
                            for name,v in val.items():row[key][name]+=v
                        row['queries']+=int(take.sum());row['heads']+=b['kl'][take].numel()
                        row['kl_sum']+=float(b['kl'][take].sum());row['top_sum']+=float(b['top'][take].sum())
                        row['mass_t']+=float(b['mass_t'][take].sum());row['mass_p']+=float(b['mass_p'][take].sum())
                    file.write(json.dumps(row)+'\n')
            cap.h.clear();cap.kw.clear();cap.initial=None;native_heads.clear()
            if index%40==args.shard:print(index,'seconds',round(time.time()-start),flush=True)
    dump(root/f'stats_{args.shard}.json',{k:v.state() for k,v in stats.items()})
    for h in handles:h.remove()

def merge(args):
    root=Path(args.output);states=[json.loads((root/f'stats_{s}.json').read_text()) for s in range(args.world)]
    stats={k:Stats.merge([s[k] for s in states]) for k in states[0]}
    for value in stats.values():
        value['vector_count']=value.pop('visual_tokens')
        value['samples']=args.samples
        value.pop('images')  # Accumulator updates may be query chunks, not images.
    rows=[json.loads(line) for s in range(args.world) for line in (root/f'rows_{s}.jsonl').read_text().splitlines()]
    result={};text=['# Fixed-native-Q visual readout','',
      f'MMStar {args.samples} samples, full embedding adapter checkpoint: `{CHECKPOINT}`. Layers are zero-based inputs 33/34/35. No training, no answer labels used.', '',
      'Native RMSNorm, Q/K head normalization, M-RoPE and GQA replication retained. Q and all nonvisual K/V stay native. FP32 offline scores/softmax/value multiplication use the complete causal sequence denominator. Text queries before all visual keys are excluded. Predicted memory is the exact fixed-anchor adapter output, not a separately fitted probe.', '',
      'The teacher is the native model with DeepStack disabled under the project-wide policy. All nonvisual prompt positions after the first visible image key count as text queries, including separators/template tokens; the final answer boundary is also reported separately. Top-k ranks causally visible logits to avoid probability underflow ties.', '',
      'Every sample/layer reconstructs full native AV and checks it against the captured original attention input to W_O (FP32 vs fused BF16, relative-error threshold 0.02). Identity visual replacement is checked on the first sample of each shard. This is a fixed-native-query diagnostic, not evidence that the adapter produces the same queries or identical end-to-end accuracy.', '',
      'Readout is concatenated per-head visual contribution before W_O; the companion JSON also reports the W_O-transformed contribution (without bias). Cosine averages valid vectors; relative error and R² use pooled sums. QK and visual attention cosine average query/head rows; KL averages all query/head distributions. Readout/hidden cosine average token vectors. The complement of the visual contribution is not deleted from the softmax denominator.', '',
      '## All text queries with visible visual keys','',
      '| Layer | Hidden cos | K cos | V cos | QK cos | QK relative error | Attention KL | Visual top-10 overlap | Readout cos | Readout R² | Readout relative error |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('all_text','answer_boundary'):
        if group=='answer_boundary':text+=['','## Final answer-boundary query','', '| Layer | Hidden cos | K cos | V cos | QK cos | QK relative error | Attention KL | Visual top-10 overlap | Readout cos | Readout R² | Readout relative error |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for l in LAYERS:
            selected=[r for r in rows if r['layer']==l and r['group']==group]
            assert sorted(r['index'] for r in selected)==list(range(args.samples))
            qk={k:sum(r['qk'][k] for r in selected) for k in selected[0]['qk']}
            attn={k:sum(r['attn'][k] for r in selected) for k in selected[0]['attn']}
            count=sum(r['heads'] for r in selected);queries=sum(r['queries'] for r in selected)
            res=dict(qk_cos=qk['cosine_sum']/qk['valid'] if qk['valid'] else None,
                qk_relative_error=math.sqrt(qk['sse']/qk['energy']) if qk['energy']>1e-30 else None,
                native_reconstruction_max_relative_error=max(r['native_reconstruction_relative_error'] for r in selected),
                attention_kl=sum(r['kl_sum'] for r in selected)/count,visual_attn_cos=attn['cosine_sum']/attn['valid'] if attn['valid'] else None,
                top10_overlap=sum(r['top_sum'] for r in selected)/queries,
                native_visual_mass=sum(r['mass_t'] for r in selected)/count,pred_visual_mass=sum(r['mass_p'] for r in selected)/count,
                **{k:stats[f'{l}/{k}'] for k in ('hidden','key','value')},
                readout=stats[f'{l}/{group}/readout'],readout_wo=stats[f'{l}/{group}/readout_wo'])
            result[f'{l}/{group}']=res
            vals=[res[k]['cosine'] for k in ('hidden','key','value')]+[res['qk_cos'],res['qk_relative_error'],res['attention_kl'],res['top10_overlap'],res['readout']['cosine'],res['readout']['r2'],res['readout']['relative_error']]
            text.append('| '+str(l)+' | '+' | '.join(f'{v:.6f}' if v is not None else 'undefined' for v in vals)+' |')
    dump(root/'results.json',result);(root/'README.md').write_text('\n'.join(text)+'\n')

def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    dump(root/'plan.json',dict(samples=args.samples,layers=LAYERS,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
        model=MODEL,source_sha256=sha(__file__),reference_model_source_sha256=sha(REFERENCE_SRC/'model.py'),dataset_sha256=sha(DATA)))
    jobs=[];start=time.time()
    for s in range(args.world):
        log=(root/f'worker{s}.log').open('w')
        cmd=[sys.executable,'-m','src.fixed_q_visual_readout','worker','--output',str(root),'--samples',str(args.samples),'--world',str(args.world),'--shard',str(s)]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log))
    while any(p.poll() is None for p,_ in jobs):
        dump(root/'status.json',dict(state='running',elapsed=time.time()-start,pids=[p.pid for p,_ in jobs if p.poll() is None]));time.sleep(10)
    for p,log in jobs:log.close()
    codes=[p.returncode for p,_ in jobs]
    if any(codes):dump(root/'status.json',dict(state='failed',codes=codes));raise RuntimeError(codes)
    merge(args);dump(root/'status.json',dict(state='complete',elapsed=time.time()-start))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('worker','run','merge'));p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=1500);p.add_argument('--world',type=int,default=8);p.add_argument('--shard',type=int,default=0)
    args=p.parse_args();globals()[args.mode](args)

if __name__=='__main__':main()
