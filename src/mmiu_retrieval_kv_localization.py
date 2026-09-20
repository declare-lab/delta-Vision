"""Exploratory K/V localization on four complete random-subset task groups.

Native memory patches are oracle diagnostics, NOT deployable benchmark scores.
The original images, their order, prompt, masks and RoPE remain unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/mmiu_random1000_seed42_all_methods_20260914'
OUT=ROOT/'artifacts/diagnostics/mmiu_retrieval_kv_localization_20260914'
TASKS={'threeD_Scene_Reconstruction','person_reid','text2image_retrieval','image2image_retrieval'}
CONDITIONS={'adapter':('kv',0,0), 'native_kv_all':('kv',0,36),
    'native_kv_0_11':('kv',0,12), 'native_kv_12_17':('kv',12,18),
    'native_kv_18_35':('kv',18,36), 'native_k_all':('k',0,36), 'native_v_all':('v',0,36)}


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    from src.mmiu_binding_protocol_audit import MODEL,CHECKPOINT
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    model.eval().requires_grad_(False)
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*args,**kwargs):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(SOURCE/'mmiu_random1000.jsonl'),processor,'mmiu',
        data_root=str(ROOT/'data/benchmarks/mmiu'),max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(ds.rows) if r['task'] in TASKS]
    assert len(indices)==77
    previous={}
    for method in ('base','embedding_adapter'):
        for p in SOURCE.glob(method+'_shard*.jsonl'):
            for line in p.open():
                r=json.loads(line)
                if r['index'] in indices:
                    assert r['generated_tokens']==1
                    previous[method,r['index']]=r
    original=adapter.all_visual_memories_batched
    layers=model.model.language_model.layers
    with torch.inference_mode(),(OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            item=ds[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            digest=hashlib.sha256()
            for name in ('input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
                if torch.is_tensor(item.get(name)):
                    value=item[name].contiguous().cpu()
                    digest.update(str((name,tuple(value.shape),str(value.dtype))).encode())
                    digest.update(value.view(torch.uint8).numpy().tobytes())
            assert digest.hexdigest()==previous['base',index]['input_sha256']==previous['embedding_adapter',index]['input_sha256']
            model.model.rope_deltas=None
            initial,pos=ref.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].ne(0)
            nv=int(visual.sum());nt=int((~visual).sum());assert nv!=nt
            memories=original(initial[:,visual])
            capture={}
            def hook(layer):
                def save(m,args,kwargs):
                    hidden=args[0] if args else kwargs['hidden_states']
                    capture[layer]=hidden[:,visual].clone()
                return save
            handles=[layer.register_forward_pre_hook(hook(i),with_kwargs=True) for i,layer in enumerate(layers)]
            try:
                model.model.rope_deltas=None
                native=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            finally:
                for handle in handles:handle.remove()
            assert len(capture)==36 and torch.equal(capture[0],initial[:,visual])
            kv={}
            for l,layer in enumerate(layers):
                normalized=layer.input_layernorm(capture[l])
                kv[l,'k']=layer.self_attn.k_proj(normalized)
                kv[l,'v']=layer.self_attn.v_proj(normalized)
            del capture
            option_ids=[processor.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(item['choices']))]
            assert all(len(x)==1 for x in option_ids)
            option_ids=[x[0] for x in option_ids]
            def record(condition,logits):
                text=processor.tokenizer.decode([int(logits.argmax())]).strip()
                if condition in ('native','adapter'):
                    method='base' if condition=='native' else 'embedding_adapter'
                    assert text==previous[method,index]['text'].strip(),(index,condition,'Did not reproduce')
                gold=ord(item['answer'])-65
                scores=logits[option_ids];other=scores.clone();other[gold]=-float('inf')
                out.write(json.dumps(dict(index=index,source_index=item['index'],task=item['row']['task'],
                    condition=condition,prediction=text,gold=item['answer'],correct=text==item['answer'],
                    option_logits=scores.tolist(),gold_margin=float(scores[gold]-other.max()),
                    native_kl=float((native.softmax(-1)*(native.log_softmax(-1)-logits.log_softmax(-1))).sum()),
                    input_sha256=digest.hexdigest(),deepstack=False,diagnostic_only=True))+'\n')
            record('native',native)
            adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memories,**kw:_m,adapter)
            try:
                for condition,(channels,start,end) in CONDITIONS.items():
                    patch=[];counts={}
                    for l in range(start,end):
                        for channel in channels:
                            key=(l,channel);counts[key]=0
                            def replace(module,args,output,key=key):
                                if output.shape[1]==nv:
                                    counts[key]+=1
                                    assert output.shape==kv[key].shape
                                    return kv[key]
                                assert output.shape[1]==nt
                                return output
                            patch.append(getattr(layers[l].self_attn,channel+'_proj').register_forward_hook(replace))
                    try:
                        model.model.rope_deltas=None
                        logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                            initial_hidden=initial,position_ids=pos,logits_to_keep=1)[0][0,-1].float()
                        assert all(n==1 for n in counts.values()),counts
                        assert torch.isfinite(logits).all()
                        record(condition,logits)
                    finally:
                        for handle in patch:handle.remove()
            finally:
                adapter.all_visual_memories_batched=original
            print('DONE',index,flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not list(OUT.glob('rows_*.jsonl')), 'Use a fresh directory if diagnostic rows already exist'
    (OUT/'plan.json').write_text(json.dumps(dict(tasks=sorted(TASKS),n=77,conditions=CONDITIONS,
        source_manifest=str(SOURCE/'mmiu_random1000.jsonl'),layer_numbering='zero-based',
        diagnostic_only=True,selection='All sampled rows from four tasks identified by paired accuracy gaps; exploratory, not a held-out benchmark',
        deepstack=False),indent=2)+'\n')
    # Do not steal GPUs from the still-authorized benchmark run.
    while True:
        status=json.loads((SOURCE/'status.json').read_text())
        # The scheduler only writes a terminal state after all workers exit.
        # An unrelated pruning failure does not invalidate completed controls.
        if status['state'] in ('complete','failed'):break
        time.sleep(5)
    jobs=[];logs=[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.mmiu_retrieval_kv_localization',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({(r['index'],r['condition']) for r in rows})==77*8
    result=[]
    for task in ['ALL',*sorted(TASKS)]:
        for condition in ['native',*CONDITIONS]:
            rs=[r for r in rows if r['condition']==condition and (task=='ALL' or r['task']==task)]
            result.append(dict(task=task,condition=condition,n=len(rs),correct=sum(r['correct'] for r in rs),
                accuracy=100*sum(r['correct'] for r in rs)/len(rs),
                mean_native_kl=sum(r['native_kl'] for r in rs)/len(rs)))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
