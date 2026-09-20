"""One frozen KL embedding adapter: repaired MMIU + native-path parity checks.

Preserves first-1000 IDs, 8 video frames and media_first_v1 to isolate changes.
No production annotations or historical results are overwritten.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent/'vision-kv-inject-attention-sink/src'
CONVERTER = REFERENCE.parent/'scripts/prepare_multimodal_benchmarks.py'
DEFAULT_OUTPUT = ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914'
BENCHES = ('mmiu','muirbench','videomme','mvbench')
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'


def sha(path):
    import hashlib
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def prompt_builder():
    # Pure helper is shared with the maintained converter, without importing
    # download/extraction code or its optional dependencies.
    tree=ast.parse(CONVERTER.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mmiu_question_with_context')
    scope={'Any':object}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(CONVERTER),'exec'),scope)
    return scope[fn.name]


def prepare(output,limit):
    from datasets import load_dataset,DownloadConfig
    output.mkdir(parents=True,exist_ok=True)
    raw=load_dataset('FanqingM/MMIU-Benchmark',split='test',download_config=DownloadConfig(local_files_only=True))
    old=[json.loads(l) for l in (ROOT/'data/benchmarks/mmiu/test.jsonl').open() if l.strip()]
    assert len(raw)==len(old)
    build=prompt_builder(); new=[]
    for i,row in enumerate(old):
        doc=raw[i]
        assert row['task']==doc['task'] and row['answer']==doc['output'].strip()
        assert row['raw_options']==doc['options']
        assert len(row['images'])==len(doc['input_image_path'])
        assert all(Path(a).name==Path(b).name for a,b in zip(row['images'],doc['input_image_path']))
        new.append(dict(row,question=build(doc),source_question=doc['question'],source_context=doc['context'],
            prompt_schema='mmiu_context_and_question_v2'))
        assert doc['question'].strip().casefold() in new[-1]['question'].casefold()
    manifest=output/'mmiu_context_and_question_v2.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new)
    if manifest.exists():assert manifest.read_text()==text
    else:manifest.write_text(text)
    plan=dict(model=MODEL,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),limit=limit,
        datasets={b:dict(path=str(manifest if b=='mmiu' else ROOT/f'data/benchmarks/{b}/test.jsonl'),
            sha256=sha(manifest if b=='mmiu' else ROOT/f'data/benchmarks/{b}/test.jsonl')) for b in BENCHES},
        source_hashes={str(p):sha(p) for p in (Path(__file__),CONVERTER,REFERENCE/'model.py',REFERENCE/'data.py',
            REFERENCE/'benchmark_video_sampling.py',REFERENCE/'benchmarks.py')},
        deepstack=False,prompt_layout='media_first_v1',max_new_tokens=128,video_frames=8,
        mmiu_source_fingerprint=raw._fingerprint,method='Embedding Adapter + KL',shards=8)
    if (output/'plan.json').exists():assert json.loads((output/'plan.json').read_text())==plan
    else:dump(output/'plan.json',plan)


def worker(output,shard):
    import src
    src.__path__.insert(0,str(REFERENCE))
    import torch
    import types
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction,build_benchmark_prompt
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    plan=json.loads((output/'plan.json').read_text());plan_hash=sha(output/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    assert sha(CHECKPOINT)==plan['checkpoint_sha256']
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.native_prefix_memory=='legacy' and not adapter.native_ffn_carriers
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    def off(module,args,kwargs):return args,dict(kwargs,deepstack_visual_embeds=None)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off,with_kwargs=True)
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward;vision_cache={}
    def cached_vision(*args,**kwargs):
        if not vision_cache:
            value=original_vision(*args,**kwargs)
            vision_cache['value']=(type(value),dict(value))
        cls,fields=vision_cache['value'];return cls(**fields)
    model.model.visual.forward=cached_vision
    path=output/f'rows_{shard}.jsonl';done=set()
    if path.exists():
        for line in path.open():
            r=json.loads(line);assert r['plan_sha256']==plan_hash
            assert (r['benchmark'],r['index']) not in done
            done.add((r['benchmark'],r['index']))
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for benchmark in BENCHES:
            dataset=QwenBenchmarkDataset(plan['datasets'][benchmark]['path'],processor,benchmark,
                data_root=str(ROOT/f'data/benchmarks/{benchmark}'),max_samples=plan['limit'],prompt_layout=plan['prompt_layout'])
            audited=False
            for index in range(shard,len(dataset),8):
                if (benchmark,index) in done:continue
                started=time.time();item=dataset[index]
                if benchmark=='mmiu':
                    assert item['row']['source_question'].strip() in build_benchmark_prompt(item['row'],dataset.spec)
                inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
                assert inputs0['attention_mask'].bool().all()
                assert not ('pixel_values' in inputs0 and 'pixel_values_videos' in inputs0)
                vision_cache.clear();hidden,pos=ref.build_qwen_initial_context(model,inputs0)
                visual=inputs0['mm_token_type_ids'][0].ne(0)
                memory=adapter.all_visual_memories_batched(hidden[:,visual])
                assert memory.shape[:3]==(36,1,int(visual.sum()))
                original_memories=adapter.all_visual_memories_batched
                adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
                audit=None;generated=[];inputs=dict(inputs0)
                try:
                    first_logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                        initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                    if not audited:
                        # Independent full native sequence with the exact same
                        # adapter memory substituted at each decoder-layer input.
                        handles=[]
                        def make_hook(layer):
                            def replace(module,args,kwargs):
                                h=(args[0] if args else kwargs['hidden_states']).clone()
                                h[:,visual]=memory[layer].to(h)
                                return ((h,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=h))
                            return replace
                        try:
                            for l,layer in enumerate(model.model.language_model.layers):
                                handles.append(layer.register_forward_pre_hook(make_hook(l),with_kwargs=True))
                            model.model.rope_deltas=None
                            full=model(**inputs0,use_cache=False,logits_to_keep=1).logits.float()
                        finally:
                            for handle in handles:handle.remove()
                        actual=first_logits.float()
                        rel=float((actual-full).norm()/full.norm().clamp_min(1e-12))
                        kl=float((full.softmax(-1)*(full.log_softmax(-1)-actual.log_softmax(-1))).sum())
                        audit=dict(native_scatter_relative_logit_error=rel,native_scatter_kl=kl,
                            same_argmax=bool(full.argmax()==actual.argmax()),visual_tokens=int(visual.sum()))
                        assert rel<.04 and kl<.05,audit
                        audited=True
                    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                    for step in range(plan['max_new_tokens']):
                        model.model.rope_deltas=None
                        logits=first_logits if step==0 else ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                            initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                        token=int(logits[0,-1].argmax());generated.append(token)
                        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+j) for j in range(len(item['choices']))]:break
                        new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                        inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                        hidden=torch.cat([hidden,model.model.get_input_embeddings()(new)],1)
                        pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                finally:adapter.all_visual_memories_batched=original_memories
                scored=score_prediction(metric=dataset.spec.metric,prediction_text=text,answer=item['answer'],
                    choices=item['choices'],answers=item.get('answers'),question=item['row']['question'])
                out.write(json.dumps(dict(benchmark=benchmark,index=index,source_index=item['index'],
                    task=item['row'].get('task'),method='static_kl',text=text,**scored,generated_tokens=len(generated),
                    deepstack_enabled=False,prompt_layout=plan['prompt_layout'],plan_sha256=plan_hash,audit=audit,
                    visual_tokens=int(visual.sum()),seconds=time.time()-started),ensure_ascii=False)+'\n')
                del memory;vision_cache.clear()
                if index%80==shard:print('PROGRESS',benchmark,index,'parity',audit,flush=True)
    print('COMPLETE',shard,flush=True)


def aggregate(output,complete=False):
    rows=[json.loads(l) for p in output.glob('rows_*.jsonl') for l in p.open() if l.strip()]
    plan=json.loads((output/'plan.json').read_text());keys=[(r['benchmark'],r['index']) for r in rows]
    assert len(keys)==len(set(keys))
    if complete:assert set(keys)=={(b,i) for b in BENCHES for i in range(plan['limit'])}
    result={}
    for b in BENCHES:
        selected=[r for r in rows if r['benchmark']==b]
        if not selected:continue
        result[b]=dict(samples=len(selected),accuracy=100*sum(r['score'] for r in selected)/len(selected),
            invalid=sum(not r['prediction'] for r in selected),capped=sum(r['generated_tokens']==128 for r in selected),
            tasks={t:dict(samples=len(rs),accuracy=100*sum(r['score'] for r in rs)/len(rs))
                for t in sorted({r['task'] for r in selected}) if (rs:=[r for r in selected if r['task']==t])})
    dump(output/'results.json',result)
    audits=[dict(benchmark=r['benchmark'],index=r['index'],**r['audit']) for r in rows if r['audit']]
    dump(output/'parity_checks.json',audits)
    return len(rows)


def run(output,limit):
    prepare(output,limit)
    import fcntl
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs={};logs=[];retries={i:0 for i in range(8)}
    def start(i):
        log=(output/f'worker{i}.log').open('a');logs.append(log)
        jobs[i]=subprocess.Popen([sys.executable,'-m','src.embedding_adapter_corrected_eval','--shard',str(i),
            '--output',str(output)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT)
    for i in range(8):start(i)
    started=time.time()
    while jobs:
        for i,p in list(jobs.items()):
            code=p.poll()
            if code is None:continue
            if code and retries[i]<1:
                retries[i]+=1;start(i)
            elif code:
                dump(output/'status.json',dict(state='failed',shard=i,exit_code=code,retries=retries))
                raise RuntimeError(f'Shard {i} failed twice; inspect log')
            else:del jobs[i]
        dump(output/'status.json',dict(state='running',active_shards=list(jobs),retries=retries,seconds=time.time()-started))
        time.sleep(5)
    count=aggregate(output,True)
    for log in logs:log.close()
    dump(output/'status.json',dict(state='complete',records=count,retries=retries,seconds=time.time()-started))
    print((output/'results.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--limit',type=int,default=1000);p.add_argument('--shard',type=int)
    a=p.parse_args()
    run(a.output,a.limit) if a.shard is None else worker(a.output,a.shard)
