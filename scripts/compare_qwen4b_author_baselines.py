"""Paired 20% baseline comparison on the existing seed44 nine-image manifests."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
METHODS=['fastv','visionzip','zoo','sparsevlm']
BURN=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/baselines','data','rows','logs']:(run/d).mkdir(parents=True)
    old=json.loads((ROOT/'artifacts/eval/native_initial_visual_random44_20260921/config.json').read_text())
    config=dict(model_path=old['model_path'],evaluation={},seed=44,shards=8,methods=METHODS,
        variants=['before','after'],retention=.2,attention='flash_attention_2',deepstack=False,dtype='bfloat16',
        scorer_commit='7f266415a28b3801339da93211a8fd9de2ff319e',
        scoring='POPE/MME question accuracy; VQAv2 soft score; mean nine unrounded scores; EOS or cap, no extra no-EOS penalty',
        before='Frozen current Qwen implementations, not LLaVA norm-based selectors',
        after='Explicit author-logic Qwen adaptations; see source/src/qwen_baseline_author_comparison.py',
        sparse_budget='Author v1 [303,110,36] profile rescaled to mean 20% over layers 3..35, including recycling; all compulsory full layers excluded',
        visionzip_adaptation='Qwen native final vision features and received-attention proxy retained; merge before native spatial merger',
        zoo_adaptation='Finite differences on grouped premerger visual features through native Qwen merger; 64 directions, noise .01')
    for b,info in old['evaluation'].items():
        dest=run/'data'/f'{b}.jsonl';shutil.copy2(info['path'],dest)
        config['evaluation'][b]=dict(info,path=str(dest),sha256=sha(dest))
    files=list((ROOT/'src').glob('*.py'))+[ROOT/'baselines'/n for n in ['eval_baselines.py','llava_hf_baselines.py','multimodal_pruning_utils.py']]
    for method in METHODS:files+=list((ROOT/'baselines'/method/'qwen3_vl').glob('*.py'))
    for p in files:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (run/'source/src/scoring_reference.py').write_bytes(subprocess.check_output(['git','show',config['scorer_commit']+':src/benchmarks.py'],cwd=ROOT))
    shutil.copy2(__file__,run/'source/worker.py')
    config['author_commits']={m:subprocess.check_output(['git','-C',str(ROOT/'baselines'/m),'rev-parse','HEAD'],text=True).strip() for m in METHODS}
    dump(run/'config.json',config)
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    dump(run/'status.json',dict(state='prepared',expected_predictions=8*8765))

def fingerprint(inputs):
    import torch
    d=hashlib.sha256()
    for k,v in sorted(inputs.items()):
        v=v.cpu().contiguous();d.update(str((k,list(v.shape),str(v.dtype))).encode());d.update(v.view(torch.uint8).numpy().tobytes())
    return d.hexdigest()

def worker(run,method,shard,smoke=False):
    sys.path.insert(0,str(run/'source'))
    import torch
    from baselines.eval_baselines import load_baseline_model,configure_baseline
    from src.data import QwenBenchmarkDataset
    from src.native_initial_visual_eval import inputs_from_item
    from src.scoring_reference import get_benchmark_spec,score_prediction
    from src.qwen_baseline_author_comparison import AuthorCorrections,sparse_budgets
    for p,d in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==d,p
    cfg=json.loads((run/'config.json').read_text());torch.set_num_threads(4);torch.manual_seed(44)
    model,processor=load_baseline_model(method,cfg['model_path'],torch.bfloat16,'cuda:0',.2,'flash_attention_2')
    model.eval();lm=model.model.language_model
    assert lm.config._attn_implementation=='flash_attention_2' and model.model.visual.deepstack_visual_indexes==[]
    module=sys.modules[type(model).__module__];correction=AuthorCorrections(model,method,module)
    lengths=[]
    def capture(layer,args,kw):
        h=kw.get('hidden_states',args[0] if args else None)
        if h.shape[1]>1:lengths.append(h.shape[1])
    handles=[l.register_forward_pre_hook(capture,with_kwargs=True) for l in lm.layers]
    tag=f'{method}_'+('smoke' if smoke else f'shard{shard}')
    path=run/'rows'/f'{tag}.jsonl';done=set()
    if path.exists():
        done={(r['benchmark'],r['sample'],r['variant']) for r in map(json.loads,path.read_text().splitlines())}
    count=0;start=time.time()
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for bi,(b,info) in enumerate(cfg['evaluation'].items()):
            assert sha(info['path'])==info['sha256']
            ds=QwenBenchmarkDataset(info['path'],processor,b,data_root=info['image_root'])
            assert len(ds)==info['samples']
            indices=range(min(2,len(ds))) if smoke else range(shard,len(ds),cfg['shards'])
            for i in indices:
                if all((b,i,v) in done for v in cfg['variants']):continue
                item=ds[i];inputs=inputs_from_item(item,torch.device('cuda:0'));row=item['row'];digest=fingerprint(inputs)
                vis=inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten();n=len(vis);nt=inputs['input_ids'].shape[1]-n
                paired={}
                for variant in cfg['variants']:
                    if (b,i,variant) in done:continue
                    torch.manual_seed(44+bi*10000+i)
                    configure_baseline(model,method,.2,int(vis[0]),n)
                    if method=='sparsevlm':correction.before_sparse=lm.config.sparse_config
                    correction.set_variant(variant)
                    lm._pruning_audit=[];model.model._pruning_audit=[];lengths.clear();model.model.rope_deltas=None
                    t=time.time()
                    result=model.generate(**inputs,do_sample=False,max_new_tokens=info['max_new_tokens'],use_cache=True,
                                          return_dict_in_generate=True)
                    tokens=result.sequences[0,inputs['input_ids'].shape[1]:].tolist()
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    actual=[v-nt for v in lengths]
                    if method=='sparsevlm' and variant=='after':
                        a,c,d=sparse_budgets(n,len(lm.layers),.2);expected=[n]*3+[a]*4+[c]*9+[d]*(len(lm.layers)-16);excluded=3
                        pruning=correction.sparse.audit
                    else:
                        excluded=2 if method in ('fastv','sparsevlm') else 0
                        expected=[n]*excluded+[max(1,round(n*.2))]*(len(lm.layers)-excluded)
                        pruning=getattr(lm,'_pruning_audit',[]) or getattr(model.model,'_pruning_audit',[])
                    assert actual==expected,(method,variant,b,i,n,actual,expected)
                    cache=[result.past_key_values.get_seq_length(j) for j in range(len(lm.layers))]
                    assert cache==[x+nt+len(tokens)-1 for x in actual],(cache,actual,len(tokens))
                    audit=dict(original_visual=n,layer_visual=actual,excluded_full_layers=list(range(excluded)),
                        prunable_visual_ratio=sum(actual[excluded:])/(n*(len(actual)-excluded)),
                        all_layer_visual_ratio=sum(actual)/(n*len(actual)),cache_lengths=cache,
                        stage_counts=[{k:v for k,v in p.items() if k not in ('visual_positions','selected_positions')} for p in pruning])
                    score=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),
                        answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(method=method,variant=variant,benchmark=b,sample=i,input_sha256=digest,
                        generated_token_ids=tokens,prediction_text=text,token_audit=audit,seconds=time.time()-t,**score)
                    out.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1;paired[variant]=tokens
                    del result
                    correction.raw=None
                if method=='fastv' and len(paired)==2:assert paired['before']==paired['after'],'FastV negative control changed'
                if count%100==0:
                    dump(run/f'progress_{tag}.json',dict(count=count,benchmark=b,sample=i,elapsed=time.time()-start))
                    print('PROGRESS',tag,b,i,count,round(time.time()-start),flush=True)
    dump(run/'rows'/f'{tag}.done.json',dict(complete=True,rows=count,elapsed=time.time()-start))

def report(run,complete=False):
    cfg=json.loads((run/'config.json').read_text());groups={};keys=set();pairs={}
    for p in (run/'rows').glob('*_shard*.jsonl'):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if complete:raise
                continue
            key=(r['method'],r['variant'],r['benchmark'],r['sample']);assert key not in keys;keys.add(key)
            groups.setdefault(key[:3],[]).append(r)
            pk=(r['method'],r['benchmark'],r['sample']);pairs.setdefault(pk,{})[r['variant']]=r
    changed={m:0 for m in METHODS}
    for (m,b,i),pair in pairs.items():
        if len(pair)==2:
            assert pair['before']['input_sha256']==pair['after']['input_sha256']
            changed[m]+=int(pair['before']['generated_token_ids']!=pair['after']['generated_token_ids'])
    table=[];names=list(cfg['evaluation'])
    for m in METHODS:
        for v in ('before','after'):
            rr=dict(method=m,variant=v)
            for b in names:
                rows=groups.get((m,v,b),[])
                if len(rows)==cfg['evaluation'][b]['samples']:rr[b]=100*sum(r['score'] for r in rows)/len(rows)
            if all(b in rr for b in names):rr['AVG']=sum(rr[b] for b in names)/9
            table.append(rr)
    dump(run/'summary.json',dict(rows=table,changed_answers=changed,predictions=len(keys)))
    lines=['# Qwen3-VL-4B — 20% paired author-logic audit','',
        'Same seed44 manifests, FA2, DeepStack off, BF16; scorer pinned to 7f266415. Before and after regenerated on identical inputs.',
        'After is an explicit Qwen adaptation, not an official Qwen implementation. Details: config.json and source/src/qwen_baseline_author_comparison.py.','',
        '| Method | Version | '+' | '.join(names+['AVG'])+' |','|---|---|'+'---:|'*10]
    for r in table:lines.append('| '+r['method']+' | '+r['variant']+' | '+' | '.join(f'{r[b]:.2f}' if b in r else 'pending' for b in names+['AVG'])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if complete:assert len(keys)==8*8765 and all('AVG'in r for r in table)
    return len(keys)

def queue(run):
    cfg=json.loads((run/'config.json').read_text());jobs=[]
    try:
        for m in METHODS:
            smoke=run/'rows'/f'{m}_smoke.done.json'
            if not smoke.exists():
                log=(run/'logs'/f'{m}_smoke.log').open('w')
                p=subprocess.Popen([sys.executable,str(run/'source/worker.py'),'worker','--run',str(run),'--method',m,'--smoke'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                jobs=[p]
                while p.poll() is None:
                    dump(run/'status.json',dict(state='smoke',method=m,pid=p.pid));time.sleep(5)
                log.close();assert p.returncode==0,(m,'smoke failed');jobs=[]
            jobs=[];logs=[]
            for shard in range(cfg['shards']):
                if (run/'rows'/f'{m}_shard{shard}.done.json').exists():continue
                log=(run/'logs'/f'{m}_{shard}.log').open('w');logs.append(log)
                p=subprocess.Popen([sys.executable,str(run/'source/worker.py'),'worker','--run',str(run),'--method',m,'--shard',str(shard)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append(p)
            while any(p.poll() is None for p in jobs):
                assert not any(p.poll() not in (0,None) for p in jobs),(m,'worker failed')
                dump(run/'status.json',dict(state='running',method=m,pids=[p.pid for p in jobs if p.poll() is None],predictions=report(run)))
                time.sleep(15)
            assert all(p.returncode==0 for p in jobs),(m,'worker failed')
            for log in logs:log.close()
        n=report(run,True);dump(run/'status.json',dict(state='complete',predictions=n))
    except BaseException as error:
        for p in jobs:
            if p.poll() is None:p.terminate()
        for p in jobs:p.wait()
        dump(run/'status.json',dict(state='failed',error=repr(error),predictions=report(run)))
        raise
    finally:
        if BURN.exists():
            r=subprocess.run([sys.executable,str(BURN),'start'],capture_output=True,text=True)
            dump(run/'burn_resume.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','queue','report']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--method',choices=METHODS);p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare(a.run)
    elif a.mode=='worker':worker(a.run,a.method,a.shard,a.smoke)
    elif a.mode=='queue':queue(a.run)
    else:print(report(a.run))
