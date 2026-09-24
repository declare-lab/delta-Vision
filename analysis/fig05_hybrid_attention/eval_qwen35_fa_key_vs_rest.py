"""Full RWQA765/MMStar1500: block FA visual reads at11/15 or other six.

LA remains native in both new conditions. Max8 generation; aggregate alongside
exactly matched native and historical boundary-rank0 paired records.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'artifacts/experiments/qwen35_memory/qwen35_memory_mechanism_20260922_102800'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def read(p):
    if not p.exists():return []
    return [json.loads(s) for s in p.read_text().splitlines(keepends=True) if s.endswith('\n')]

PAIRED=ROOT/'artifacts/experiments/qwen35_memory/boundary_rank0_full_rwqa765_mmstar1500_cap64_20260923'
CONDITIONS={'fa_key_11_15':[11,15],'fa_other_six':[3,7,19,23,27,31]}

def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/scripts','source/test/diagnostics','data','rows','logs','audits']:(run/d).mkdir(parents=True)
    prior=json.loads((PAIRED/'config.json').read_text())
    c={k:prior[k] for k in ['model_path','model_revision','rank','evaluation_generation']}
    c.update(original_root=str(ROOT),paired_reference=str(PAIRED),primary_cap=8,seed=44,deepstack=False,
        attention='flash_attention_2',dtype='bfloat16',conditions=CONDITIONS,evaluation={},
        intervention='Only text-query access to visual KV blocked in selected FA layers during prefill AND decode; full visual KV and visual-query outputs kept',
        linear_attention='All24 LA layers unmodified; no boundary state intervention',sampling='All765 RWQA and1500 MMStar; source order',
        scoring='Same historical scorer and prompts; greedy max8; unfinished/no-EOS invalid zero')
    for name,info in prior['evaluation'].items():
        assert sha(info['path'])==info['sha256'];dest=run/'data'/f'{name}.jsonl';shutil.copy2(info['path'],dest)
        c['evaluation'][name]=dict(info,path=str(dest),max_new_tokens=8)
    c['evaluation_generation']['max_new_tokens']=8
    sources=(list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py')))+[Path(__file__).resolve(),ROOT/'test/diagnostics/test_qwen35_full_attention_ablation.py']
    for p in sources:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(run/'config.json',c);dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    dump(run/'status.json',dict(state='prepared',expected=2265))

def worker(run,shard,smoke=False):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root'])
    sys.path.insert(0,str(run/'source'));sys.path.insert(0,str(root/'artifacts/dependencies/qwen35_python'))
    import torch
    from src.qwen35 import load_model,prepare_inputs,generate_evaluation_answer
    from analysis.fig05_hybrid_attention.qwen35_full_attention_ablation import remove_full_attention_visual_effect
    from src.benchmarks import build_benchmark_prompt,get_benchmark_spec
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h
    torch.set_num_threads(2);torch.manual_seed(44);torch.set_float32_matmul_precision('highest')
    processor,model,unused_adapter,controller=load_model(c,torch.device('cuda:0'))
    assert [i for i,l in enumerate(model.model.language_model.layers) if l.block_type=='full_attention']==[3,7,11,15,19,23,27,31]
    if smoke:
        import importlib.util
        spec=importlib.util.spec_from_file_location('fa_tests',run/'source/test/diagnostics/test_qwen35_full_attention_ablation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from transformers.integrations.flash_attention import flash_attention_forward
        module.check_readout('cuda',flash_attention_forward,torch.bfloat16)
    refs={(r['benchmark'],r['index']):r for f in (Path(c['paired_reference'])/'rows').glob('shard*.jsonl') for r in read(f)}
    assert len(refs)==2265
    tag='smoke' if smoke else f'shard{shard}';path=run/'rows'/f'{tag}.jsonl'
    done={(r['benchmark'],r['index']) for r in read(path)};started=time.time();checks=[]
    la_layers=[model.model.language_model.layers[i].linear_attn for i in range(32) if i%4!=3]
    la_forwards=[m.forward for m in la_layers]
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for benchmark,info in c['evaluation'].items():
            assert sha(info['path'])==info['sha256'];data=read(Path(info['path']));assert len(data)==info['samples']
            spec=get_benchmark_spec(benchmark);indices=[0] if smoke else list(range(shard,len(data),8))
            for ordinal,index in enumerate(indices):
                if (benchmark,index) in done:continue
                begin=time.time();row=data[index];ref=refs[benchmark,index]
                inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
                digest=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest();assert digest==ref['input_ids_sha256']
                mask=inputs['mm_token_type_ids'].eq(1)
                native=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=8)
                assert native==ref['variants']['native']['cap8'],(benchmark,index,'native reference changed')
                variants={'native':native};audits={}
                for name,layers in c['conditions'].items():
                    if ordinal==0:
                        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
                        with remove_full_attention_visual_effect(model,mask,layers,block=False):
                            check_logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
                            control=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=8)
                        assert torch.equal(logits,check_logits) and control==native
                        checks.append(dict(benchmark=benchmark,index=index,condition=name,noop_logits_exact=True,noop_generation_exact=True))
                    with remove_full_attention_visual_effect(model,mask,layers) as audit:
                        answer=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=8)
                    assert [m.forward for m in la_layers]==la_forwards
                    assert {v['layer'] for v in audit['calls']}==set(layers)
                    for layer in layers:
                        calls=[v for v in audit['calls'] if v['layer']==layer]
                        assert len(calls)==answer['generated_tokens']
                        assert calls[0]['query_length']==mask.shape[1]
                        assert [v['cache_length'] for v in calls]==list(range(mask.shape[1],mask.shape[1]+len(calls)))
                        assert all(v['query_length']==1 for v in calls[1:])
                    variants[name]=answer;audits[name]=dict(layers=layers,prefill_and_decode_blocked=True,full_kv_cache_preserved=True)
                record=dict(benchmark=benchmark,index=index,source_index=row.get('index',index),input_ids_sha256=digest,
                    question=row['question'],gold=row.get('answer'),variants=variants,audits=audits,seconds=time.time()-begin)
                out.write(json.dumps(record,ensure_ascii=False)+'\n');done.add((benchmark,index))
                dump(run/f'progress_{tag}.json',dict(completed=len(done),benchmark=benchmark,index=index,elapsed_s=time.time()-started))
                if smoke or len(done)%20==0:print(tag,benchmark,index,len(done),round(time.time()-started),flush=True)
    controller.close();dump(run/'audits'/f'{tag}.json',dict(passed=True,samples=len(done),checks=checks))

def report(run):
    c=json.loads((run/'config.json').read_text())
    rows={(r['benchmark'],r['index']):r for f in (run/'rows').glob('shard*.jsonl') for r in read(f)}
    old={(r['benchmark'],r['index']):r for f in (Path(c['paired_reference'])/'rows').glob('shard*.jsonl') for r in read(f)}
    assert len(rows)==2265 and set(rows)==set(old)
    results=[];cohorts={}
    for benchmark,n in [('realworldqa',765),('mmstar',1500)]:
        for method in ['native','boundary_rank0']+list(c['conditions']):
            groups={k:[] for k in ['both_correct','base_only','intervention_only','both_wrong']};invalid=limited=0
            for i in range(n):
                r=rows[benchmark,i];ref=old[benchmark,i];a=r['variants']['native']
                assert a==ref['variants']['native']['cap8'] and r['input_ids_sha256']==ref['input_ids_sha256']
                b=ref['variants']['boundary_rank0']['cap8'] if method=='boundary_rank0' else r['variants'][method]
                key='both_correct' if a['score'] and b['score'] else 'base_only' if a['score'] else 'intervention_only' if b['score'] else 'both_wrong'
                groups[key].append(i);invalid+=int(b.get('invalid',False));limited+=int(b['hit_generation_limit'])
            counts={k:len(v) for k,v in groups.items()};assert sum(counts.values())==n
            correct=counts['both_correct']+counts['intervention_only'];base_correct=counts['both_correct']+counts['base_only']
            results.append(dict(benchmark=benchmark,method=method,samples=n,correct=correct,accuracy=100*correct/n,
                delta_pp=100*(correct-base_correct)/n,**counts,invalid=invalid,hit_generation_limit=limited,
                source='historical matched-input cap8' if method=='boundary_rank0' else 'fresh max8'))
            cohorts[benchmark+'_'+method]=groups
    dump(run/'RESULTS.json',dict(complete=True,max_new_tokens=8,results=results));dump(run/'paired_cohorts.json',cohorts)
    labels={'native':'Base','boundary_rank0':'LA boundary rank0','fa_key_11_15':'FA visual OFF:11,15','fa_other_six':'FA visual OFF:3,7,19,23,27,31'}
    lines=['# Qwen3.5-4B: LA boundary rank0 vs FA visual-readout ablation','',
        'RealWorldQA765 and MMStar1500; full datasets, unchanged prompts, max8, greedy; unfinished/no-EOS invalid zero. DeepStack off, FA2/FLA. FA ablations leave all LA layers untouched and preserve visual queries and full KV caches; only text access to visual KV is blocked, including cached decode. Native is freshly checked against every prior sample. Boundary rank0 results reused only after exact input/native-output parity checks.','',
        '| Dataset | Method | Correct/N | ACC | Delta pp | Base correct → intervention wrong | Base wrong → intervention correct | Both correct | Both wrong | Unfinished at8 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['benchmark']} | {labels[r['method']]} | {r['correct']}/{r['samples']} | {r['accuracy']:.2f}% | {r['delta_pp']:+.2f} | {r['base_only']} | {r['intervention_only']} | {r['both_correct']} | {r['both_wrong']} | {r['hit_generation_limit']} |")
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def queue(run):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root']);active=[];started=time.time();restore=False
    burn=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')
    def stop(*args):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if burn.exists() and (burn.parent/'pid').exists():
            pid=(burn.parent/'pid').read_text().strip();stat=Path('/proc')/pid/'stat'
            restore=stat.exists() and stat.read_text().split()[2]!='Z'
            if restore:subprocess.run([str(root/'.venv/bin/python'),str(burn),'stop'],check=True)
        for smoke in [True,False]:
            active=[]
            for shard in ([0] if smoke else range(8)):
                tag='smoke' if smoke else f'shard{shard}'
                log=(run/'logs'/f'{tag}.log').open('a')
                cmd=[str(root/'.venv/bin/python'),'-u',str(run/'source/analysis/fig05_hybrid_attention/eval_qwen35_fa_key_vs_rest.py'),'worker','--run',str(run),'--shard',str(shard)]
                if smoke:cmd+=['--smoke']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false')
                p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();active.append(p)
            while any(p.poll() is None for p in active):
                assert all(p.poll() in (None,0) for p in active),'Worker failed, see logs'
                done=sum(json.loads(p.read_text())['completed'] for p in run.glob('progress_shard*.json'))
                dump(run/'status.json',dict(state='validation' if smoke else 'running',completed=done,expected=2265,pids=[p.pid for p in active if p.poll() is None],elapsed_s=time.time()-started));time.sleep(5)
            assert all(p.returncode==0 for p in active)
        report(run);dump(run/'status.json',dict(state='complete',completed=2265,expected=2265,elapsed_s=time.time()-started))
    except BaseException as e:dump(run/'status.json',dict(state='failed',error=repr(e)));raise
    finally:
        for p in active:
            if p.poll() is None:p.terminate()
        for p in active:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        if restore:
            r=subprocess.run([str(root/'.venv/bin/python'),str(burn),'start','--coexist'],capture_output=True,text=True)
            dump(run/'burn_restoration.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','queue','report']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.run=a.run.resolve()
    if a.action=='worker':worker(a.run,a.shard,a.smoke)
    else:globals()[a.action](a.run)
