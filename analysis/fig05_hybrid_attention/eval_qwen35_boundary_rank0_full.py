"""Paired native vs boundary rank0: full RWQA765 and MMStar1500.

Primary cap64 follows the user's updated generation limit. The cap8 prefix
is retained only to verify reproduction of historical outputs.
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

def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/scripts','data','rows','logs','audits']:(run/d).mkdir(parents=True)
    old=json.loads((REFERENCE/'config.json').read_text())
    c={k:old[k] for k in ['model_path','model_revision','rank','evaluation_generation']}
    c.update(original_root=str(ROOT),reference=str(REFERENCE),seed=44,deepstack=False,
        attention='flash_attention_2',dtype='bfloat16',methods=['native','boundary_rank0'],
        scopes='All24 linear-attention layers; full-attention/visual outputs/FFNs/convolution unchanged',
        boundary='Immediately before first image patch and after last image patch, before vision_end',
        state='At visual boundary restore Sin; preserve visual readouts; correct native suffix and final cache using paired replay',
        primary_cap=64,historical_validation_cap=8,evaluation={},sampling='All examples, source order; no sampling',
        scoring='Same historical score_evaluation_prediction; unfinished/no-EOS invalid zero at each cap separately')
    for name,n in [('realworldqa',765),('mmstar',1500)]:
        info=old['evaluation'][name];source=Path(info['source']);assert sha(source)==info['source_sha256']
        assert len(read(source))==n
        dest=run/'data'/f'{name}.jsonl';shutil.copy2(source,dest)
        c['evaluation'][name]=dict(path=str(dest),sha256=sha(dest),source=str(source),samples=n,
            image_root=info['image_root'],metric=info['metric'],max_new_tokens=64)
    for p in (list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py')))+[Path(__file__).resolve()]:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(run/'config.json',c)
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    dump(run/'status.json',dict(state='prepared',expected=2265))

def worker(run,shard,smoke=False):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root'])
    sys.path.insert(0,str(run/'source'));sys.path.insert(0,str(root/'artifacts/dependencies/qwen35_python'))
    import torch
    from src.qwen35 import load_model,prepare_inputs,generate_evaluation_answer,score_evaluation_prediction
    from analysis.fig05_hybrid_attention.qwen35_memory_probe import state_intervention
    from src.benchmarks import get_benchmark_spec,build_benchmark_prompt
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h
    torch.set_num_threads(2);torch.manual_seed(44);torch.set_float32_matmul_precision('highest')
    processor,model,unused_adapter,controller=load_model(c,torch.device('cuda:0'))
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    tag='smoke' if smoke else f'shard{shard}'
    output=run/'rows'/f'{tag}.jsonl';done={(r['benchmark'],r['index']) for r in read(output)}
    checks=[];matched=0;started=time.time()
    with torch.inference_mode(),output.open('a',buffering=1) as out:
        for name,info in c['evaluation'].items():
            assert sha(info['path'])==info['sha256'];data=read(Path(info['path']));assert len(data)==info['samples']
            spec=get_benchmark_spec(name)
            refroot=Path(c['reference'])
            refinfo=json.loads((refroot/'config.json').read_text())['evaluation'][name]
            refdata=read(Path(refinfo['path']))
            # Identical questions and image-grid shapes can share input-token
            # hashes across distinct pictures. Join by the original source ID.
            refs={refdata[r['index']]['index']:r for f in (refroot/'accuracy').glob(f'{name}.shard*.jsonl') for r in read(f)}
            indices=[0] if smoke else list(range(shard,len(data),8))
            for ordinal,index in enumerate(indices):
                if (name,index) in done:continue
                row=data[index];begin=time.time()
                inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
                digest=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
                mask=inputs['mm_token_type_ids'].eq(1);end=int(mask[0].nonzero()[-1])+1
                assert int(inputs['input_ids'][0,end])==model.config.vision_end_token_id
                variants={}
                for method in c['methods']:
                    context=state_intervention(model,mask,rank=0) if method=='boundary_rank0' else nullcontext()
                    with context:
                        full=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=64)
                    tokens=full['generated_token_ids'][:8];finished=bool(tokens and tokens[-1] in eos)
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
                    short=dict(prediction_text=text,**score_evaluation_prediction(dict(prediction_text=text,stopped_by_eos=finished),row,spec.metric),
                        generated_token_ids=tokens,generated_tokens=len(tokens),max_new_tokens=8,stopped_by_eos=finished,
                        hit_generation_limit=not finished and len(tokens)>=8)
                    variants[method]=dict(cap8=short,cap64=full)
                # Reproduce all historical paired outputs present in this full dataset.
                if row['index'] in refs:
                    ref=refs[row['index']]
                    assert digest==ref['input_ids_sha256'],(name,index,'historical input mismatch')
                    for method,rank in [('native',None),('boundary_rank0',0)]:
                        previous=next(v for v in ref['variants'] if v['method']=='native' and v['rank']==rank)
                        assert variants[method]['cap8']['generated_token_ids']==previous['generated_token_ids'],(name,index,method,'historical tokens')
                        assert variants[method]['cap8']['score']==previous['score'],(name,index,method,'historical score')
                    matched+=1
                if ordinal==0:
                    # Verify cap64 prefix actually equals a separate cap8 rollout,
                    # and full state rank128 is an identity for this intervention.
                    for method in c['methods']:
                        context=state_intervention(model,mask,rank=0) if method=='boundary_rank0' else nullcontext()
                        with context:short=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=8)
                        assert short==variants[method]['cap8'],(name,index,method,'cap prefix')
                    with state_intervention(model,mask,rank=128):
                        identity=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=8)
                    assert identity==variants['native']['cap8'],(name,index,'rank128 identity')
                    checks.append(dict(benchmark=name,index=index,cap_prefix_exact=True,full_rank_identity_exact=True))
                record=dict(benchmark=name,index=index,source_index=row.get('index',index),input_ids_sha256=digest,
                    question=row['question'],gold=row.get('answer'),variants=variants,seconds=time.time()-begin)
                out.write(json.dumps(record,ensure_ascii=False)+'\n');done.add((name,index))
                dump(run/f'progress_{tag}.json',dict(completed=len(done),benchmark=name,index=index,historical_matches=matched,elapsed_s=time.time()-started))
                if smoke or len(done)%20==0:print(tag,name,index,len(done),round(time.time()-started),flush=True)
    controller.close()
    dump(run/'audits'/f'{tag}.json',dict(passed=True,samples=len(done),historical_matches=matched,checks=checks))

def report(run):
    rows=[r for f in (run/'rows').glob('shard*.jsonl') for r in read(f)]
    assert len(rows)==2265 and len({(r['benchmark'],r['index']) for r in rows})==2265
    summaries=[];cohorts={}
    for benchmark,n in [('realworldqa',765),('mmstar',1500)]:
        subset=[r for r in rows if r['benchmark']==benchmark];assert {r['index'] for r in subset}==set(range(n))
        for cap in ['cap64','cap8']:
            groups={k:[] for k in ['both_correct','base_only','rank0_only','both_wrong']};invalid={'native':0,'boundary_rank0':0};limited=dict(invalid)
            for r in subset:
                a=r['variants']['native'][cap];b=r['variants']['boundary_rank0'][cap]
                assert a['score'] in (0,1) and b['score'] in (0,1)
                key='both_correct' if a['score'] and b['score'] else 'base_only' if a['score'] else 'rank0_only' if b['score'] else 'both_wrong'
                groups[key].append(r['index'])
                for method in invalid:
                    v=r['variants'][method][cap];invalid[method]+=int(v.get('invalid',False));limited[method]+=int(v['hit_generation_limit'])
            counts={k:len(v) for k,v in groups.items()};assert sum(counts.values())==n
            a=counts['both_correct']+counts['base_only'];b=counts['both_correct']+counts['rank0_only']
            summaries.append(dict(benchmark=benchmark,cap=cap,samples=n,**counts,base_correct=a,rank0_correct=b,
                base_accuracy=100*a/n,rank0_accuracy=100*b/n,delta_pp=100*(b-a)/n,invalid=invalid,hit_generation_limit=limited))
            cohorts[f'{benchmark}_{cap}']=groups
    dump(run/'RESULTS.json',dict(complete=True,results=summaries));dump(run/'paired_cohorts.json',cohorts)
    lines=['# Qwen3.5-4B native vs boundary rank0: full datasets','',
        'RealWorldQA765 and MMStar1500, all examples, unchanged prompts and historical scorer. DeepStack off; FA2/FLA; greedy generation. Boundary rank0 restores recurrent memory to before the patch span in all24 LA layers while preserving visual outputs/convolution/full attention.','',
        'Primary results use max64 generated tokens for BOTH models, as requested. Prompts and historical scorer unchanged; unfinished/no-EOS is still invalid zero. The cap8 prefix is retained ONLY for historical validation, not the main reported accuracy. Each native/rank0 pair uses identical inputs. Historical overlapping samples must match cap8 tokens and scores exactly.','']
    for cap,label in [('cap64','Primary:64-token limit'),('cap8','Historical8-token prefix validation only')]:
        lines+=['## '+label,'','| Dataset | N | Both correct | Base correct / rank0 wrong | Base wrong / rank0 correct | Both wrong | Base ACC | Rank0 ACC | Delta pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['cap']==cap:lines.append(f"| {r['benchmark']} | {r['samples']} | {r['both_correct']} | {r['base_only']} | {r['rank0_only']} | {r['both_wrong']} | {r['base_accuracy']:.2f}% | {r['rank0_accuracy']:.2f}% | {r['delta_pp']:+.2f} |")
        lines+=['']
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
                cmd=[str(root/'.venv/bin/python'),'-u',str(run/'source/analysis/fig05_hybrid_attention/eval_qwen35_boundary_rank0_full.py'),'worker','--run',str(run),'--shard',str(shard)]
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
