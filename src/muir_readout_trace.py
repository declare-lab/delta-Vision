"""Trace actual native/adapter answer-boundary attention by image and layer.

Read-only hooks on Q/K/V projections. Reconstructs the real full softmax
denominator, retains text keys and native RoPE. Not an inference intervention.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_readout_trace_validated_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    def off(module,args,kwargs):return args,dict(kwargs,deepstack_visual_embeds=None)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off,with_kwargs=True)
    model.model.language_model._deepstack_process=reject
    dataset=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',
        max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(dataset.rows) if r['task']=='Image-Text Matching']
    cache={};state={};records=[];handles=[]

    def callback(layer,kind):
        attn=model.model.language_model.layers[layer].self_attn
        def hook(module,args,raw):
            parts=cache.setdefault(layer,{'k':[],'v':[]})
            if kind=='q':parts['q']=raw[:,-1:];return
            parts[kind].append(raw)
            if kind!='v' or len(parts['v'])!=(1 if state['method']=='base' else 2):return
            shape=lambda x:x.view(1,x.shape[1],-1,attn.head_dim)
            rope=lambda x,emb:ref._apply_rope_one_from_embeddings(x,emb)
            q=attn.q_norm(shape(parts['q'])).transpose(1,2)
            q=rope(q,state['q_emb'])
            if state['method']=='base':
                k=rope(attn.k_norm(shape(parts['k'][0])).transpose(1,2),state['all_emb'])
                v=shape(parts['v'][0]).transpose(1,2)
                groups=state['native_image_indices']
            else:
                kt,kv=parts['k'];vt,vv=parts['v']
                kt=rope(attn.k_norm(shape(kt)).transpose(1,2),state['text_emb'])
                kv=rope(attn.k_norm(shape(kv)).transpose(1,2),state['visual_emb'])
                k=torch.cat([kv,kt],dim=2)
                v=torch.cat([shape(vv).transpose(1,2),shape(vt).transpose(1,2)],dim=2)
                groups=state['compact_image_indices']
            k=k.repeat_interleave(attn.num_key_value_groups,dim=1)
            v=v.repeat_interleave(attn.num_key_value_groups,dim=1)
            weights=(q.float()@k.float().transpose(-1,-2)*attn.scaling).softmax(-1)
            parts['reconstructed']=(weights@v.float()).transpose(1,2).reshape(1,1,-1)
            mass=[];norms=[];perhead=[]
            for group in groups:
                a=weights[:,:,:,group]
                mass.append(float(a.sum(-1).mean()))
                perhead.append(a.sum(-1).flatten().tolist())
                contribution=(a@v[:,:,group].float()).transpose(1,2).reshape(1,1,-1)
                # Native output projection without its optional bias.
                projected=torch.nn.functional.linear(contribution,attn.o_proj.weight.float())
                norms.append(float(projected.norm()))
            records.append(dict(index=state['index'],method=state['method'],layer=layer,
                image_mass=mass,image_readout_norm=norms,image_mass_per_head=perhead))
            state.setdefault('reconstructed',{})[layer]=parts['reconstructed']
            cache.pop(layer)
        return hook

    for l,layer in enumerate(model.model.language_model.layers):
        for kind in ['q','k','v']:
            handles.append(getattr(layer.self_attn,kind+'_proj').register_forward_hook(callback(l,kind)))
    # Used only to validate the reconstructed AV, after the normal computation.
    def verify_output(layer):
        def hook(module,args):
            assert args[0].shape[0]==1
            expected=state['reconstructed'].pop(layer)
            actual=args[0][:,-1:].float()
            error=float((expected-actual).norm()/actual.norm().clamp_min(1e-12))
            assert error<.025,(state['index'],state['method'],layer,error)
            assert records[-1]['layer']==layer
            records[-1]['av_reconstruction_relative_error']=error
        return hook
    for l,layer in enumerate(model.model.language_model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(verify_output(l)))
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            item=dataset[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            initial,pos=ref.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].ne(0)
            counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
            assert len(counts)==3 and sum(counts)==int(visual.sum())
            all_visual=visual.nonzero().flatten();start=0;native_groups=[];compact_groups=[]
            for n in counts:
                native_groups.append(all_visual[start:start+n]);compact_groups.append(torch.arange(start,start+n,device='cuda'));start+=n
            emb=model.model.language_model.rotary_emb(initial,pos)
            state.update(index=index,all_emb=emb,q_emb=tuple(x[:,-1:] for x in emb),
                text_emb=tuple(x[:,~visual] for x in emb),visual_emb=tuple(x[:,visual] for x in emb),
                native_image_indices=native_groups,compact_image_indices=compact_groups)
            for method in ['base','static_kl']:
                state['method']=method;records.clear();cache.clear();model.model.rope_deltas=None
                if method=='base':logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                else:logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,position_ids=pos,logits_to_keep=1)[0]
                assert len(records)==36 and not cache
                prediction=processor.tokenizer.decode([int(logits[0,-1].argmax())])
                for r in records:out.write(json.dumps(dict(r,prediction=prediction,deepstack_enabled=False))+'\n')
            print('DONE',index,flush=True)
    for h in handles:h.remove()


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    logs=[];jobs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_readout_trace',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({(r['index'],r['method'],r['layer']) for r in rows})==6048
    result={}
    for method in ['base','static_kl']:
        result[method]={}
        for l in range(36):
            rs=[r for r in rows if r['method']==method and r['layer']==l]
            result[method][l]={key:[sum(r[key][j] for r in rs)/len(rs) for j in range(3)]
                for key in ['image_mass','image_readout_norm']}
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
