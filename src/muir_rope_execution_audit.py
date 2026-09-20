"""Audit actual Adapter rotary Q/K and causal masks, not only their indices.

1000 original multi-image inputs + 1000 synthetic video inputs, all 36 layers.
Independent coordinate construction from modality runs and grids. Compare actual
Q/K to native rotary applied to identical pre-rotation normalized projections.
No production model changes; instrumentation removed when each worker exits.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/muir_rope_execution_20260914'
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'


def independent_positions(inputs,merge):
    import numpy as np
    import torch
    types=inputs['mm_token_type_ids'][0].cpu().numpy()
    assert bool(inputs['attention_mask'].all())
    grids={1:[],2:[]}
    if 'image_grid_thw' in inputs:
        grids[1]=inputs['image_grid_thw'].cpu().tolist()
    if 'video_grid_thw' in inputs:
        for t,h,w in inputs['video_grid_thw'].cpu().tolist():
            grids[2].extend([[1,h,w]]*t)
    cursors={1:0,2:0}
    result=np.zeros((3,len(types)),dtype=np.int64)
    start=0;offset=0;groups=[]
    while start<len(types):
        kind=int(types[start]);end=start+1
        while end<len(types) and types[end]==kind:end+=1
        if kind==0:
            result[:,start:end]=np.arange(offset,offset+end-start)[None,:]
            offset+=end-start
        else:
            t,h,w=grids[kind][cursors[kind]];cursors[kind]+=1
            h//=merge;w//=merge
            assert t*h*w==end-start
            local=np.arange(end-start)
            result[0,start:end]=offset+local//(h*w)
            result[1,start:end]=offset+(local//w)%h
            result[2,start:end]=offset+local%w
            groups.append(dict(kind=kind,start=start,end=end,grid=[t,h,w],offset=offset))
            offset+=max(h,w)
        start=end
    assert all(cursors[k]==len(grids[k]) for k in cursors)
    return torch.from_numpy(result).unsqueeze(1).to(inputs['input_ids'].device),groups


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL,ADAPTER_CHECKPOINTS
    from src.muir_images_as_video_audit import prepare as video_prepare
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    model.eval().requires_grad_(False)
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(ADAPTER_CHECKPOINTS['embedding_adapter'],
        model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_attention_backend=='efficient'
    assert adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0 and not adapter.native_ffn_carriers
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
        data_root=str(ROOT/'data/benchmarks/muirbench'),max_samples=1000,prompt_layout='media_first_v1')
    state={'active':False};raw={};handles=[]
    def capture(layer,kind):
        def hook(module,args,output):
            if state['active']:
                raw.setdefault(layer,{}).setdefault(kind,[]).append(output.transpose(1,2))
        return hook
    for i,layer in enumerate(model.model.language_model.layers):
        handles.append(layer.self_attn.q_norm.register_forward_hook(capture(i,'q')))
        handles.append(layer.self_attn.k_norm.register_forward_hook(capture(i,'k')))
    original_attention=ref._efficient_prefix_causal_attention_heads
    def attention(query,visual_key,visual_value,text_key,text_value,*,scaling,attention_mask):
        if state['active']:
            layer=state['layer'];saved=raw.pop(layer)
            assert len(saved['q'])==1 and len(saved['k'])==2
            q=saved['q'][0];kt,kv=saved['k']
            q_ref,kt_ref=apply_rotary_pos_emb(q,kt,*state['text_rope'])
            _,kv_ref=apply_rotary_pos_emb(kv,kv,*state['visual_rope'])
            for name,actual,expected in [('Q',query,q_ref),('text_K',text_key,kt_ref),('visual_K',visual_key,kv_ref)]:
                error=float((actual-expected).abs().max())
                state['max_error'][name]=max(state['max_error'][name],error)
                assert torch.equal(actual,expected),(state['index'],state['mode'],layer,name,error)
            assert torch.equal(attention_mask,state['mask']),(state['index'],layer,'causal mask mismatch')
            state['layer']+=1
        return original_attention(query,visual_key,visual_value,text_key,text_value,scaling=scaling,attention_mask=attention_mask)
    ref._efficient_prefix_causal_attention_heads=attention
    with torch.inference_mode(),(OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in range(shard,1000,8):
            for mode in ('original_images','as_video'):
                item=ds[index] if mode=='original_images' else video_prepare(ds,ds.rows[index],'as_video')[0]
                inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
                hidden,pos=ref.build_qwen_initial_context(model,inputs)
                independent,groups=independent_positions(inputs,model.model.visual.spatial_merge_size)
                assert torch.equal(pos,independent),(index,mode,'independent coordinate mismatch')
                pack=ref.prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],
                    inputs['mm_token_type_ids'],hidden,pos)
                visual=inputs['mm_token_type_ids'][0]!=0
                vi=visual.nonzero().flatten();ti=(~visual).nonzero().flatten()
                assert len(groups)==len(ds.rows[index]['images'])
                assert torch.equal(pack['image_positions'][0],vi) and torch.equal(pack['text_positions'][0],ti)
                assert torch.equal(pack['visual_position_ids'],independent[:,:,vi])
                assert torch.equal(pack['text_position_ids'],independent[:,:,ti])
                assert torch.equal(pack['visual_memory'],hidden[:,vi])
                full_rope=model.model.language_model.rotary_emb(hidden,independent)
                tr=tuple(x[:,ti] for x in full_rope);vr=tuple(x[:,vi] for x in full_rope)
                assert all(torch.equal(a,b) for a,b in zip(tr,pack['text_position_embeddings']))
                assert all(torch.equal(a,b) for a,b in zip(vr,pack['visual_position_embeddings']))
                # Construct full-order causal permissions and then reindex columns;
                # no use of the Adapter's mask helper or M-RoPE coordinate values.
                columns=torch.cat([vi,ti])
                mask=(ti[:,None]>=columns[None,:])[None,None]
                assert torch.equal(mask,pack['prefix_attention_mask'])
                counts=[int(mask[0,0,-1,:len(vi)][(vi>=g['start'])&(vi<g['end'])].sum()) for g in groups]
                assert counts==[g['end']-g['start'] for g in groups]
                state.update(active=True,index=index,mode=mode,layer=0,text_rope=tr,visual_rope=vr,mask=mask,
                    max_error=dict(Q=0.,text_K=0.,visual_K=0.))
                model.model.rope_deltas=None
                logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                state['active']=False
                assert state['layer']==36 and not raw
                # Audit itself must not change the model's result.
                noop=None
                if index==shard:
                    control=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                    noop=bool(torch.equal(logits,control));assert noop
                # Four appended generated tokens: check incremental positions against
                # fresh native reconstruction, independently of accuracy/early EOS.
                extended={k:v.clone() for k,v in inputs.items() if k not in ('pixel_values','pixel_values_videos')}
                inc=pos.clone();token=logits[:,-1].argmax(-1).view(1,1)
                for step in range(4):
                    extended['input_ids']=torch.cat([extended['input_ids'],token],1)
                    extended['attention_mask']=torch.ones_like(extended['input_ids'])
                    extended['mm_token_type_ids']=torch.cat([extended['mm_token_type_ids'],torch.zeros_like(token)],1)
                    inc=torch.cat([inc,inc[:,:,-1:]+1],2)
                    fresh,_=model.model.get_rope_index(input_ids=extended['input_ids'],mm_token_type_ids=extended['mm_token_type_ids'],
                        image_grid_thw=extended.get('image_grid_thw'),video_grid_thw=extended.get('video_grid_thw'),attention_mask=extended['attention_mask'])
                    assert torch.equal(inc,fresh),(index,mode,'decode position',step)
                out.write(json.dumps(dict(index=index,mode=mode,images=len(groups),sequence_tokens=inputs['input_ids'].shape[1],
                    visual_tokens=len(vi),groups=groups,last_query_visible_counts=counts,position_exact=True,
                    rotary_max_abs_error=state['max_error'],layers_checked=36,causal_mask_exact=True,
                    incremental_position_steps=4,noop_exact=noop))+'\n')
            if index//8%10==0:print('DONE',shard,index,flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs,logs=[],[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_rope_execution_audit',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert not any(codes) and len(rows)==len({(r['index'],r['mode']) for r in rows})==2000,codes
    summary=dict(completed=len(rows),exit_codes=codes,layer_checks=sum(r['layers_checked'] for r in rows),
        position_mismatches=sum(not r['position_exact'] for r in rows),mask_mismatches=sum(not r['causal_mask_exact'] for r in rows),
        rotary_max_abs_error={k:max(r['rotary_max_abs_error'][k] for r in rows) for k in ('Q','text_K','visual_K')},
        incremental_position_checks=sum(r['incremental_position_steps'] for r in rows),
        no_op_controls=sum(r['noop_exact'] is True for r in rows),
        image_counts=sorted({r['images'] for r in rows}),max_visual_tokens=max(r['visual_tokens'] for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
