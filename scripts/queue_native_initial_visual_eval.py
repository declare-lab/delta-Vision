"""Paired native/constant-E evaluation on frozen seed44 nine-benchmark manifests."""
import argparse
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
from src.native_initial_visual_eval import dump


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for name in ('logs','rows','audits','data','source/src'):(run/name).mkdir(parents=True)
    parent=ROOT/'artifacts/eval/divprune_fixed_multimodal_random44_five_models_20260920_112257/config.json'
    prior=json.loads(parent.read_text())
    config=dict(model_path='/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
        methods=['native','initial_embedding'],shards=8,seed=44,evaluation={},attention='flash_attention_2',
        deepstack=False,adapter_checkpoint=None,fast_path=False,visual_tokens='all retained',
        intervention='Before every LM layer, replace visual hidden by the layer0 input E; text continues unchanged. Native full attention/FFN and generate/KV-cache.',
        scoring='Per-question score average x100; POPE/MME accuracy, VQAv2 soft accuracy; nine-benchmark unrounded mean. No no-EOS zero rule.')
    for name,info in prior['single_image'].items():
        dest=run/'data'/f'{name}.jsonl';shutil.copy2(info['path'],dest)
        config['evaluation'][name]=dict(info,path=str(dest),reference_manifest=info['path'])
    dump(run/'config.json',config)
    hashes={}
    for path in (ROOT/'src').glob('*.py'):
        dest=run/'source/src'/path.name;shutil.copy2(path,dest);hashes[str(dest.relative_to(run))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    dump(run/'source_hashes.json',hashes)
    dump(run/'status.json',dict(state='prepared',predictions=2*sum(x['samples'] for x in config['evaluation'].values())))


def report(run):
    config=json.loads((run/'config.json').read_text());records=[]
    for path in (run/'rows').glob('shard*.jsonl'):
        for line in path.read_text().splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:continue
    keys=[(r['benchmark'],r['sample'],r['method']) for r in records];assert len(keys)==len(set(keys))
    lookup={(r['benchmark'],r['sample'],r['method']):r for r in records}
    for (b,i,m),r in lookup.items():
        other=lookup.get((b,i,'native' if m=='initial_embedding' else 'initial_embedding'))
        if other:assert other['input_sha256']==r['input_sha256']
    results={};lines=['# Native Qwen3-VL-4B vs constant initial visual embedding','',
        'FA2, DeepStack off, native full forward; no adapter, no fast-path. Seed44 fixed manifests. POPE/MME question accuracy; VQAv2 soft score.','',
        '| Benchmark | Native | Initial embedding at every layer | Difference (pp) |','|---|---:|---:|---:|']
    for name,info in config['evaluation'].items():
        values={}
        for method in config['methods']:
            rows=[r for r in records if r['benchmark']==name and r['method']==method]
            if len(rows)==info['samples']:
                assert sorted(r['sample'] for r in rows)==list(range(info['samples']))
                values[method]=100*sum(r['score'] for r in rows)/len(rows)
        results[name]=values
        a=values.get('native');b=values.get('initial_embedding')
        lines.append('| '+name+' | '+' | '.join('pending' if x is None else f'{x:.2f}' for x in (a,b,None if a is None or b is None else b-a))+' |')
    if all(len(v)==2 for v in results.values()):
        avgs={m:sum(v[m] for v in results.values())/9 for m in config['methods']}
        results['AVG']=avgs;lines.append(f"| AVG | {avgs['native']:.2f} | {avgs['initial_embedding']:.2f} | {avgs['initial_embedding']-avgs['native']:.2f} |")
    dump(run/'summary.json',results);(run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return len(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    run=a.run_dir.resolve()
    if not run.exists():prepare(run)
    if a.prepare_only:return
    env=dict(os.environ,PYTHONPATH=str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
    children=[];started=time.time()
    for shard in range(8):
        log=(run/'logs'/f'shard{shard}.log').open('a')
        cmd=[sys.executable,'-m','src.native_initial_visual_eval','--run-dir',str(run),'--shard',str(shard)]
        child=subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT)
        log.close();children.append(child)
    while True:
        codes=[c.poll() for c in children];count=report(run)
        state='running' if any(c is None for c in codes) else ('complete' if all(c==0 for c in codes) else 'failed')
        dump(run/'status.json',dict(state=state,launcher_pid=os.getpid(),worker_pids=[c.pid for c in children],exit_codes=codes,predictions=count,elapsed_seconds=time.time()-started))
        if state!='running':break
        time.sleep(15)
    if state=='complete':assert count==17530


if __name__=='__main__':main()
