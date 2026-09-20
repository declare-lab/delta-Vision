"""Matched, stratified video evaluation; no training or historical score reuse."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT.parent/'vision-kv-inject-attention-sink/src'
OUTPUT=ROOT/'artifacts/diagnostics/video_balanced_base_adapter_20260914'
MODEL='/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT=ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
BENCHES=('videomme','mvbench')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def select_rows(rows,bench):
    rng=random.Random(42)
    if bench=='videomme':
        groups=defaultdict(lambda:defaultdict(list))
        for r in rows:groups[r['duration']][r['videoID']].append(r)
        selected=[]
        for duration in ('short','medium','long'):
            ids=rng.sample(sorted(groups[duration]),111)
            for ident in ids:
                assert len(groups[duration][ident])==3
                selected.extend(groups[duration][ident])
    else:
        groups=defaultdict(list)
        for r in rows:groups[r['task']].append(r)
        assert len(groups)==19 and 'fine_grained_pose' not in groups
        selected=[r for task in sorted(groups) for r in rng.sample(sorted(groups[task],key=lambda r:r['index']),50)]
    return sorted(selected,key=lambda r:r['index'])


def prepare(output):
    import av
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(model=MODEL,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),seed=42,
        deepstack=False,video_frames=8,video_subtitles=False,video_sampling='full_timestamp_v1',
        prompt_layout='media_first_v1',max_new_tokens=128,shards=8,methods=['base','embedding_adapter_kl'],
        missing_mvbench_tasks={'fine_grained_pose':'Licensed NTU RGB+D media not available; not evaluated'},
        datasets={},source_sha256={str(p):sha(p) for p in [Path(__file__),REF/'model.py',REF/'data.py',REF/'benchmarks.py',REF/'benchmark_video_sampling.py']})
    for bench in BENCHES:
        source=ROOT/f'data/benchmarks/{bench}/test.jsonl'
        rows=[json.loads(l) for l in source.open()]
        chosen=select_rows(rows,bench)
        def check(r):
            path=ROOT/f'data/benchmarks/{bench}'/r['videos'][0]
            if path.is_dir():
                frames=sorted(p for p in path.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
                assert frames and r.get('fps'),str(path)
                fps=float(r['fps']);duration=len(frames)/fps
                indices=[int(p.stem) for p in frames]
                assert all(b-a==1 for a,b in zip(indices,indices[1:])),str(path)
            else:
                with av.open(str(path)) as c:
                    s=c.streams.video[0];fps=float(s.average_rate or s.base_rate or 30)
                    duration=float(s.duration*s.time_base) if s.duration is not None else c.duration/av.time_base
            start=max(0,float(r.get('start') or 0))
            end=min(float(r['end']) if r.get('end') is not None else duration,max(0,duration-1/fps))
            if start>end:raise ValueError(f'Invalid selected clip; do not drop it: {r["index"]}, {start}, {end}, {path}')
            return dict(source_index=r['index'],path=str(path),duration=duration,fps=fps,
                annotated_start=r.get('start'),annotated_end=r.get('end'),sample_start=start,sample_end=end,
                end_exceeds_media=r.get('end') is not None and float(r['end'])>duration+1/fps+0.01)
        with ThreadPoolExecutor(max_workers=8) as pool:checks=list(pool.map(check,chosen))
        manifest=output/f'{bench}_selected.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)
        if manifest.exists():assert manifest.read_text()==content,'Selection changed; use a new output directory'
        else:manifest.write_text(content)
        dump(output/f'{bench}_media_checks.json',checks)
        plan['datasets'][bench]=dict(path=str(manifest),sha256=sha(manifest),source_sha256=sha(source),
            samples=len(chosen),groups=dict(Counter(r['duration'] if bench=='videomme' else r['task'] for r in chosen)),
            flagged_media_ends=[r['source_index'] for r in checks if r['end_exceeds_media']],
            boundary_policy='Clamp to available frames and report; no sample removed or substituted')
        print('PREPARED',bench,len(chosen),'flagged ends',plan['datasets'][bench]['flagged_media_ends'],flush=True)
    if (output/'plan.json').exists():assert json.loads((output/'plan.json').read_text())==plan,'Plan changed; use a new directory'
    else:dump(output/'plan.json',plan)


def worker(output,shard):
    import src
    src.__path__.insert(0,str(REF))
    import torch
    import types
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    plan=json.loads((output/'plan.json').read_text());plan_hash=sha(output/'plan.json')
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    assert sha(CHECKPOINT)==plan['checkpoint_sha256']
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.native_prefix_memory=='legacy' and not adapter.native_ffn_carriers
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    calls=[0]
    def off(m,a,k):calls[0]+=1;return a,dict(k,deepstack_visual_embeds=None)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off,with_kwargs=True)
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward;vision_cache={}
    def cached_vision(*a,**k):
        if not vision_cache:
            v=original_vision(*a,**k);vision_cache['result']=(type(v),dict(v))
        cls,fields=vision_cache['result'];return cls(**fields)
    model.model.visual.forward=cached_vision
    path=output/f'rows_{shard}.jsonl';done=set()
    if path.exists():
        for line in path.open():
            r=json.loads(line);assert r['plan_sha256']==plan_hash
            key=(r['benchmark'],r['index'],r['method']);assert key not in done;done.add(key)
    with torch.inference_mode(),path.open('a',buffering=1) as file:
        for bench in BENCHES:
            info=plan['datasets'][bench];assert sha(info['path'])==info['sha256']
            ds=QwenBenchmarkDataset(info['path'],processor,bench,data_root=str(ROOT/f'data/benchmarks/{bench}'),
                                    prompt_layout=plan['prompt_layout'],cache_dir=None)
            for i in range(shard,len(ds),8):
                if all((bench,i,m) in done for m in plan['methods']):continue
                item=ds[i];inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
                assert 'pixel_values_videos' in inputs0 and 'pixel_values' not in inputs0
                assert inputs0['attention_mask'].bool().all()
                vision_cache.clear();initial,positions=ref.build_qwen_initial_context(model,inputs0)
                visual=inputs0['mm_token_type_ids'][0].ne(0)
                memories=adapter.all_visual_memories_batched(initial[:,visual])
                assert memories.shape[:3]==(36,1,int(visual.sum()))
                choices=[chr(65+j) for j in range(len(item['choices']))]
                input_hash=hashlib.sha256(inputs0['input_ids'].cpu().numpy().tobytes()+positions.cpu().numpy().tobytes()).hexdigest()
                original=adapter.all_visual_memories_batched
                adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memories,**k:_m,adapter)
                try:
                    for method in plan['methods']:
                        if (bench,i,method) in done:continue
                        start=time.monotonic();inputs=dict(inputs0);h=initial;p=positions;generated=[]
                        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                        for step in range(plan['max_new_tokens']):
                            model.model.rope_deltas=None
                            if method=='base':
                                before=calls[0];logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                                assert calls[0]==before+1
                            else:
                                logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=h,position_ids=p,logits_to_keep=1)[0]
                            token=int(logits[0,-1].argmax());generated.append(token)
                            text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                            if token in eos or text in choices:break
                            new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                            inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                            inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                            if method!='base':
                                h=torch.cat([h,model.model.get_input_embeddings()(new)],1)
                                p=torch.cat([p,p[:,:,-1:]+1],2)
                        scored=score_prediction(metric=ds.spec.metric,prediction_text=text,answer=item['answer'],
                            choices=item['choices'],answers=item.get('answers'),question=item['row']['question'])
                        file.write(json.dumps(dict(benchmark=bench,index=i,source_index=item['row']['index'],
                            group=item['row'].get('duration') if bench=='videomme' else item['row']['task'],method=method,
                            text=text,tokens=generated,**scored,seconds=time.monotonic()-start,
                            visual_tokens=int(visual.sum()),input_sha256=input_hash,plan_sha256=plan_hash,deepstack=False))+'\n')
                finally:adapter.all_visual_memories_batched=original
                del memories;vision_cache.clear()
                if i%80==shard:print('PROGRESS',bench,i,flush=True)
    print('COMPLETE',shard,flush=True)


def aggregate(output,complete=False):
    plan=json.loads((output/'plan.json').read_text());rows=[]
    for p in output.glob('rows_*.jsonl'):
        content=p.read_text();lines=content.splitlines()
        if content and not content.endswith('\n'):lines=lines[:-1]
        rows.extend(json.loads(l) for l in lines)
    keys=[(r['benchmark'],r['index'],r['method']) for r in rows];assert len(keys)==len(set(keys))
    if complete:assert set(keys)=={(b,i,m) for b in BENCHES for i in range(plan['datasets'][b]['samples']) for m in plan['methods']}
    inputs=defaultdict(set)
    for r in rows:
        assert r['plan_sha256']==sha(output/'plan.json')
        inputs[(r['benchmark'],r['index'])].add(r['input_sha256'])
    assert all(len(v)==1 for v in inputs.values()),'Methods saw different prompts or positions'
    results={}
    for bench in BENCHES:
        results[bench]={}
        for method in plan['methods']:
            sub=[r for r in rows if r['benchmark']==bench and r['method']==method]
            per_group={g:100*sum(r['score'] for r in sub if r['group']==g)/sum(r['group']==g for r in sub)
                       for g in sorted({r['group'] for r in sub})}
            results[bench][method]=dict(samples=len(sub),expected=plan['datasets'][bench]['samples'],
                accuracy=100*sum(r['score'] for r in sub)/len(sub) if sub else None,groups=per_group,
                invalid=sum(r['invalid'] for r in sub),capped=sum(len(r['tokens'])==128 for r in sub))
    dump(output/'results.json',results)
    return len(rows)


def run(output):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(output)
    running={};attempts=Counter();logs=[];start=time.monotonic()
    def launch(shard):
        attempts[shard]+=1;log=(output/f'worker{shard}.log').open('a');logs.append(log)
        running[shard]=subprocess.Popen([sys.executable,'-m','src.video_balanced_base_adapter_eval','--output',str(output),'--shard',str(shard)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
            stdout=log,stderr=subprocess.STDOUT)
    try:
        for shard in range(8):launch(shard)
        while running:
            for shard,p in list(running.items()):
                code=p.poll()
                if code is None:continue
                if code and attempts[shard]<2:
                    print('RETRY',shard,code,flush=True);launch(shard)
                elif code:raise RuntimeError(f'Worker {shard} failed twice; inspect its log')
                else:del running[shard]
            count=aggregate(output)
            dump(output/'status.json',dict(state='running',rows=count,seconds=time.monotonic()-start,
                workers={k:p.pid for k,p in running.items()},attempts=dict(attempts)))
            if running:time.sleep(10)
        count=aggregate(output,True)
        dump(output/'status.json',dict(state='complete',rows=count,seconds=time.monotonic()-start))
        print((output/'results.json').read_text(),flush=True)
    finally:
        for p in running.values():
            if p.poll() is None:p.terminate()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--shard',type=int)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare(a.output)
    elif a.shard is None:run(a.output)
    else:worker(a.output,a.shard)
