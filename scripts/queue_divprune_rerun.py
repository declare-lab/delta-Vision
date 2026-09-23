"""Eight-GPU queue: three fixed multimodal sets and five seed44 image suites."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.divprune_rerun import dump


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run):
    import torch
    run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir();(run/'data').mkdir()
    parent=ROOT/'artifacts/experiments/qwen35_pruning/qwen35_random44_corrected_20260920_101318'
    prior=json.loads((parent/'config.json').read_text())
    config=dict(repository=str(ROOT),created=time.time(),method='divprune',retentions=[.05,.2],dtype='bfloat16',attention='flash_attention_2',deepstack='off',seed=44,
        models={
            'qwen3-vl-4b':dict(kind='qwen',path='/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',shards=8),
            'qwen3-vl-30b-a3b':dict(kind='qwen',path=str(ROOT/'model/Qwen3-VL-30B-A3B-Instruct'),shards=2),
            'qwen3-vl-8b':dict(kind='qwen',path=str(ROOT/'model/Qwen3-VL-8B-Instruct'),shards=2),
            'llava-1.5-7b':dict(kind='llava',path='/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf',shards=2),
            'llava-1.5-13b':dict(kind='llava',path=str(ROOT/'model/llava-1.5-13b-hf'),shards=2),
            'llava-1.6-mistral-7b':dict(kind='llava',path=str(ROOT/'model/llava-v1.6-mistral-7b-hf'),shards=2)},
        single_image={},multimodal={},
        scoring='100*mean(per-question score); MME/POPE accuracy, VQAv2 soft accuracy; AVG arithmetic mean of unrounded benchmark scores; no F1',
        output_limit_scoring='Score generated text with pinned corrected parser; do not impose Qwen3.5-only no-EOS zero rule',
        retention_definition='All visual tokens of the request; pruning before layer0; all-layer visual-only budget; text unchanged',
        input_policy='Multimodal original fixed manifests and exact historical processed inputs; single-image same random44 manifests as latest paired Qwen3.5 run')
    for model,info in config['models'].items():assert Path(info['path'],'config.json').exists(),(model,info['path'])
    for name,info in prior['evaluation'].items():
        dest=run/'data'/f'{name}.jsonl';shutil.copy2(info['path'],dest)
        config['single_image'][name]=dict(path=str(dest),sha256=sha(dest),samples=info['samples'],image_root=info['image_root'],
            max_new_tokens=info['max_new_tokens'],sampling='uniform_without_replacement',seed=44,source_indices=info['source_indices'],reference_manifest=info['path'])
    fixed={
        'muirbench':(ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914', 'muirbench_random1000.jsonl',1000,8),
        'videomme':(ROOT/'artifacts/diagnostics/video_balanced_base_adapter_20260914','videomme_selected.jsonl',999,128),
        'mvbench':(ROOT/'artifacts/diagnostics/video_balanced_base_adapter_20260914','mvbench_selected.jsonl',950,128)}
    all_cache={}
    for name,(folder,filename,n,cap) in fixed.items():
        source=folder/filename;target=run/'data'/f'{name}.jsonl';shutil.copy2(source,target)
        rows=[json.loads(s) for s in target.read_text().splitlines() if s];assert len(rows)==n
        history=folder if name=='muirbench' else ROOT/'artifacts/diagnostics/mmiu_video_all_methods_matched_20260914'
        references={}
        for p in history.glob('base_shard*.jsonl'):
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if r['benchmark']==name:references[str(r['source_index'])]=r['input_sha256']
        assert len(references)==n,(name,len(references),n)
        cache_by_index={}
        for p in (history/'processed'/name).rglob('*.pt'):
            item=torch.load(p,map_location='cpu',weights_only=False,mmap=True)['item']
            key=str(item['index']);assert key not in cache_by_index,(name,key)
            cache_by_index[key]=(str(p),item['row']['question'],item['answer'],item.get('choices'))
        entries=[]
        for row in rows:
            key=str(row['index']);path,question,answer,choices=cache_by_index[key]
            assert question==row['question'] and answer==row.get('answer') and choices==row.get('choices'),(name,key)
            entries.append(dict(path=path,input_sha256=references[key]))
        all_cache[name]=entries
        config['multimodal'][name]=dict(path=str(target),sha256=sha(target),samples=n,max_new_tokens=cap,original_manifest=str(source),
            processed_reference=str(history),sampling='historical fixed subset, unchanged',prompt_layout='media_first_v1',video_frames=8 if name!='muirbench' else None)
        print('MATCHED_HISTORICAL_INPUTS',name,n,flush=True)
    dump(run/'multimodal_input_cache.json',all_cache);dump(run/'config.json',config)
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'baselines').rglob('*.py'))+[Path(__file__).resolve()]
    hashes={}
    for p in files:
        rel=p.relative_to(ROOT);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(rel)]=sha(dest)
    dump(run/'plan.json',dict(source_sha256=hashes,cache_index_sha256=sha(run/'multimodal_input_cache.json'),total_predictions=5*2*8765+2*(1000+999+950)))
    dump(run/'status.json',dict(state='prepared'))


def report(run):
    config=json.loads((run/'config.json').read_text());records=[]
    tables=[]
    for multi,title in [(True,'Qwen3-VL-4B: fixed multi-image/video subsets'),(False,'Five models: seed44 nine single-image benchmarks')]:
        if not any((m=='qwen3-vl-4b')==multi for m in config['models']):continue
        if not multi:title='Single-image benchmarks: seed44'
        datasets=config['multimodal'] if multi else config['single_image']
        names=list(datasets)
        lines=['# '+title,'','Scores (%); MME/POPE question accuracy; VQAv2 soft accuracy. AVG uses unrounded scores.','',
            '| Model | Retention | '+' | '.join(names+['AVG'])+' |','|---|---:|'+'---:|'*(len(names)+1)]
        for model,mi in config['models'].items():
            if (model=='qwen3-vl-4b')!=multi:continue
            complete_files=[run/'full'/model/f'shard{s}.jsonl' for s in range(mi['shards']) if (run/'full'/model/f'shard{s}.done.json').exists()]
            predictions=[json.loads(l) for p in complete_files for l in p.read_text().splitlines() if l]
            for retention in [.05,.2]:
                result=dict(model=model,retention=retention)
                for name,info in datasets.items():
                    rows=[r for r in predictions if r['benchmark']==name and r['retention']==retention]
                    if len(rows)!=info['samples']:continue
                    assert sorted(r['sample'] for r in rows)==list(range(info['samples']))
                    result[name]=100*sum(r['score'] for r in rows)/len(rows)
                if all(n in result for n in names):result['AVG']=sum(result[n] for n in names)/len(names)
                records.append(result)
                lines.append('| '+model+f' | {retention:.0%} | '+' | '.join(f'{result[n]:.2f}' if n in result else 'pending' for n in names+['AVG'])+' |')
        tables+=lines+['']
    (run/'RESULTS.md').write_text('\n'.join(tables)+'\n');dump(run/'summary.json',records)
    return records


def execute(run):
    config=json.loads((run/'config.json').read_text());plan=json.loads((run/'plan.json').read_text());root=Path(config['repository'])
    for name,digest in plan['source_sha256'].items():assert sha(run/'source'/name)==digest,name
    assert sha(run/'multimodal_input_cache.json')==plan['cache_index_sha256']
    for info in list(config['single_image'].values())+list(config['multimodal'].values()):assert sha(info['path'])==info['sha256']
    env=dict(os.environ,PYTHONPATH=str(root/'artifacts/dependencies/qwen35_python')+':'+str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    active={}
    jobs=[(phase,m,s) for phase in ['smoke','full'] for m,info in config['models'].items() for s in (range(info['shards']) if phase=='full' else [0])]
    def marker(job):
        phase,m,s=job;return run/phase/m/f'shard{s}.done.json'
    def status(state,**kw):dump(run/'status.json',dict(state=state,pid=os.getpid(),updated=time.time(),**kw))
    try:
        while True:
            for gpu,(job,proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();assert proc.returncode==0 and marker(job).exists(),(job,proc.returncode)
                del active[gpu];report(run)
            pending=[j for j in jobs if not marker(j).exists() and j not in [x[0] for x in active.values()]]
            if not pending and not active:break
            for gpu in range(8):
                if gpu in active:continue
                job=next((j for j in pending if j[0]=='smoke' or marker(('smoke',j[1],0)).exists()),None)
                if job is None:break
                used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used>=1024:continue
                phase,m,s=job;log=(run/'logs'/f'{phase}_{m}_{s}.log').open('a')
                cmd=[str(root/'.venv/bin/python'),'-m','src.divprune_rerun','--run-dir',str(run),'--model',m,'--phase',phase,'--shard',str(s),'--shards',str(config['models'][m]['shards'])]
                proc=subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(job,proc,log);pending.remove(job)
            status('running',completed=sum(marker(j).exists() for j in jobs),total=len(jobs),assignments=[dict(gpu=g,job=j,pid=p.pid) for g,(j,p,l) in active.items()])
            time.sleep(10)
        records=report(run);assert len(records)==2*len(config['models']) and all('AVG' in r for r in records)
        status('complete',report=str(run/'RESULTS.md'))
    except Exception as exc:
        for job,proc,log in active.values():
            if proc.poll() is None:proc.terminate()
            proc.wait();log.close()
        status('failed',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--start-prepared',action='store_true');a=p.parse_args()
    run=a.run_dir.resolve() if a.run_dir else ROOT/'artifacts/eval'/('divprune_fixed_multimodal_random44_five_models_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if not a.start_prepared:prepare(run)
    print('RUN_DIR='+str(run),flush=True)
    if not a.prepare_only:execute(run)
