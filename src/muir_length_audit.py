"""Read-only paired diagnostic: continue every previously capped Muir answer.

Identical weights, media, layout, greedy decoder and DeepStack-off policy.
Only generation budget changes from 8 to 128. Original results stay untouched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913'
OUTPUT = ROOT/'artifacts/diagnostics/muir_length_audit_20260914'


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    from src.adapter_nodeepstack_suite import checkpoints, MODEL
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rows = [json.loads(l) for p in sorted(SOURCE.glob('rows_*.jsonl')) for l in p.read_text().splitlines()]
    jobs = sorted([r for r in rows if r['benchmark']=='muirbench' and r['generated_tokens']==8],
                  key=lambda r:(r['method'],r['index']))[shard::8]
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    model.eval().requires_grad_(False)
    adapters = {}
    for method in sorted({r['method'] for r in jobs}-{'base'}):
        adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(checkpoints()[method],
            model.model.language_model, torch.device('cuda'), torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected']
        adapters[method] = adapter.eval().requires_grad_(False)
    def off(module,args,kwargs): return args, dict(kwargs,deepstack_visual_embeds=None)
    def reject(*args,**kwargs): raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off,with_kwargs=True)
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward
    vision_cache={}
    def cached_vision(*args,**kwargs):
        if not vision_cache:
            result=original_vision(*args,**kwargs)
            vision_cache['value']=(type(result),dict(result))
        cls, fields=vision_cache['value']
        return cls(**fields)
    model.model.visual.forward=cached_vision
    dataset=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,
        'muirbench',max_samples=1000,prompt_layout='media_first_v1')
    with torch.inference_mode(), (OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for old in jobs:
            vision_cache.clear()
            item=dataset[old['index']]
            inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            hidden,pos=ref.build_qwen_initial_context(model,inputs)
            adapter=adapters.get(old['method'])
            original_memories=None
            if adapter is not None:
                memory=adapter.all_visual_memories_batched(hidden[:,inputs['mm_token_type_ids'][0].ne(0)])
                original_memories=adapter.all_visual_memories_batched
                adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
            generated=[]
            eos=model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            prefix_matches=None
            try:
                for step in range(128):
                    model.model.rope_deltas=None
                    if adapter is None: logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                    else: logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                        initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                    token=int(logits[0,-1].argmax()); generated.append(token)
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                    if len(generated)==8:
                        prefix_matches=text==old['text']
                        assert prefix_matches,(old['index'],old['method'],old['text'],text)
                    if token in eos or text in [chr(65+j) for j in range(len(item['choices']))]:break
                    new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                    inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                    inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                    if adapter is not None:
                        hidden=torch.cat([hidden,model.model.get_input_embeddings()(new)],1)
                        pos=torch.cat([pos,pos[:,:,-1:]+1],2)
            finally:
                if original_memories is not None:adapter.all_visual_memories_batched=original_memories
            scored=score_prediction(metric=dataset.spec.metric,prediction_text=text,answer=item['answer'],
                choices=item['choices'],question=item['row'].get('question'))
            out.write(json.dumps(dict(index=old['index'],method=old['method'],old_text=old['text'],
                old_score=old['score'],old_prediction=old['prediction'],text=text,**scored,
                generated_tokens=len(generated),prefix_matches=prefix_matches,deepstack_enabled=False),ensure_ascii=False)+'\n')
            print('DONE',old['method'],old['index'],len(generated),flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[]; logs=[]
    for shard in range(8):
        log=(OUTPUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_length_audit',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.read_text().splitlines()]
    result={}
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method]
        result[method]=dict(capped_samples=len(rs),old_correct=sum(r['old_score'] for r in rs),
            new_correct=sum(r['score'] for r in rs),accuracy_change_points=sum(r['score']-r['old_score'] for r in rs)/10,
            still_capped=sum(r['generated_tokens']==128 for r in rs))
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
