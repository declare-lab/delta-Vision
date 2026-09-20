"""Isolate native visual evolution from preceding-image context, without training.

Each image keeps its original vision-encoder output and is passed alone through
the native LLM with the SAME prefix as image 1. Its layer-input states are then
read at the original multi-image positions. Original task/text/masks stay fixed.
These oracle controls are not deployable benchmark scores.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_independent_native_memory_20260914'
CONTEXT_AUDIT=os.environ.get('MUIR_NATIVE_CONTEXT_AUDIT')=='1'
if CONTEXT_AUDIT:
    OUTPUT=ROOT/'artifacts/diagnostics/muir_native_context_factors_20260914'


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
    lm=model.model.language_model
    lm.register_forward_pre_hook(lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    lm._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(ds.rows) if r['task']=='Image-Text Matching']
    original_memories=adapter.all_visual_memories_batched
    def capture_forward(hidden,positions,mask):
        captured={}
        def hook(i):
            def save(module,args,kwargs):
                h=args[0] if args else kwargs['hidden_states']
                captured[i]=h[:,mask].clone()
            return save
        handles=[l.register_forward_pre_hook(hook(i),with_kwargs=True) for i,l in enumerate(lm.layers)]
        try:
            result=lm(inputs_embeds=hidden,position_ids=positions,
                attention_mask=torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.long),use_cache=False)
            logits=model.lm_head(result.last_hidden_state[:,-1:]).float()
        finally:
            for h in handles:h.remove()
        assert len(captured)==36
        return torch.stack([captured[i] for i in range(36)]),logits
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            item=ds[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            initial,pos=ref.build_qwen_initial_context(model,inputs)
            mask=inputs['mm_token_type_ids'][0].ne(0)
            counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
            image_indices=list(mask.nonzero().flatten().split(counts))
            assert len(image_indices)==3
            full,base_logits=capture_forward(initial,pos,mask)
            predicted=original_memories(initial[:,mask])
            prefix_end=int(image_indices[0][0])
            prefix_indices=torch.arange(prefix_end,device='cuda')
            independent=[];original_position=[];original_prefix=[]
            for j,positions in enumerate(image_indices):
                select=torch.cat([prefix_indices,positions])
                single={k:inputs[k][:,select] for k in ['input_ids','attention_mask','mm_token_type_ids']}
                single['image_grid_thw']=inputs['image_grid_thw'][j:j+1]
                single_hidden=initial[:,select]
                single_pos=ref.qwen_position_ids(model,single,inputs_embeds=single_hidden)
                memory,_=capture_forward(single_hidden,single_pos,single['mm_token_type_ids'][0].ne(0))
                assert torch.equal(memory[0],initial[:,positions])
                independent.append(memory)
                if CONTEXT_AUDIT:
                    # Same isolated image and common prefix, restore only its
                    # original M-RoPE coordinates from the multi-image sequence.
                    own_pos_memory,_=capture_forward(single_hidden,pos[:,:,select],single['mm_token_type_ids'][0].ne(0))
                    original_position.append(own_pos_memory)
                    # Restore all original preceding TEXT, including image IDs,
                    # but not preceding image tokens. Keep original coordinates.
                    all_indices=torch.arange(initial.shape[1],device='cuda')
                    text_prefix=((~mask)&(all_indices<int(positions[0]))).nonzero().flatten()
                    prefix_select=torch.cat([text_prefix,positions])
                    prefix_memory,_=capture_forward(initial[:,prefix_select],pos[:,:,prefix_select],mask[prefix_select])
                    original_prefix.append(prefix_memory)
            isolated=torch.cat(independent,dim=2)
            assert isolated.shape==full.shape==predicted.shape
            first_image_error=float((isolated[:,:,:counts[0]].float()-full[:,:,:counts[0]].float()).norm()/full[:,:,:counts[0]].float().norm())
            result=dict(index=index,gold=item['answer'],first_image_native_replay_relative_error=first_image_error,
                original_images=item['row']['images'],conditions={})
            def save(name,logits):
                text=processor.tokenizer.decode([int(logits[0,-1].argmax())]).strip()
                reference=base_logits[0,-1]
                current=logits[0,-1].float()
                result['conditions'][name]=dict(prediction=text,correct=text==item['answer'],
                    native_kl=float((reference.softmax(-1)*(reference.log_softmax(-1)-current.log_softmax(-1))).sum()))
            save('native',base_logits)
            conditions=[('adapter',predicted,0,0),('joint_native_middle',full,12,18),
                    ('independent_native_middle',isolated,12,18),('joint_native_all',full,0,36),
                    ('independent_native_all',isolated,0,36)]
            if CONTEXT_AUDIT:
                own_position=torch.cat(original_position,dim=2)
                own_prefix=torch.cat(original_prefix,dim=2)
                conditions.extend([('original_position_middle',own_position,12,18),('original_position_all',own_position,0,36),
                    ('original_text_prefix_middle',own_prefix,12,18),('original_text_prefix_all',own_prefix,0,36)])
            for name,source,start,end in conditions:
                memory=predicted.clone();memory[start:end]=source[start:end]
                adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
                try:
                    logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,position_ids=pos,logits_to_keep=1)[0]
                finally:adapter.all_visual_memories_batched=original_memories
                save(name,logits)
            out.write(json.dumps(result)+'\n');print('DONE',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_independent_native_memory',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==84
    summary={name:dict(samples=84,correct=sum(r['conditions'][name]['correct'] for r in rows),
        accuracy=100*sum(r['conditions'][name]['correct'] for r in rows)/84,
        mean_native_kl=sum(r['conditions'][name]['native_kl'] for r in rows)/84) for name in rows[0]['conditions']}
    summary['max_first_image_native_replay_relative_error']=max(r['first_image_native_replay_relative_error'] for r in rows)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
