"""Eight independent data shards, sequential analysis/accuracy/sensitivity stages."""
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
SOURCES=['src/qwen35_memory_probe.py','scripts/qwen35_memory_probe_worker.py','scripts/report_qwen35_memory_probe.py',
 'scripts/queue_qwen35_memory_probe.py','src/qwen35_embedding.py','src/qwen35_experiment.py','src/qwen_deepstack.py','src/benchmarks.py',
 'test/diagnostics/test_qwen35_memory_probe.py','docs/qwen35_memory_mechanism.md']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def counts(run,stage,config):
 result={}
 for b,info in config['evaluation'].items():
  indices=[]
  for p in (run/stage).glob(f'{b}.shard*.jsonl'):
   # A worker may be appending; only count completed newline-terminated rows.
   raw=p.read_bytes()
   for line in raw.splitlines(keepends=True):
    if line.endswith(b'\n'):indices.append(json.loads(line)['index'])
  assert len(indices)==len(set(indices)),(stage,b,'duplicate index')
  assert set(indices).issubset(range(info['samples']))
  result[b]=len(indices)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
 config=json.loads((run/'config.json').read_text())
 for b in config['evaluation']:
  records=[json.loads(line) for f in (run/'validate').glob(f'{b}.shard*.jsonl') for line in f.read_text().splitlines()]
  assert records,('Validation missing',b)
  for record in records:
   for x in record['checks']:
    if 'rank128_argmax_equal' in x:assert x['rank128_argmax_equal']
    if 'full_svd_error' in x:assert x['full_svd_error']['normalized_frobenius']<2e-5
 source_hashes={name:digest(ROOT/name) for name in SOURCES}
 plan=run/'plan.json'
 if plan.exists():assert json.loads(plan.read_text())['source_sha256']==source_hashes,'Source changed since launch'
 else:
  for name in SOURCES:
   dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
  dump(plan,dict(source_sha256=source_hashes,config_sha256=digest(run/'config.json'),
    adapter_checkpoint_sha256=config['adapter_checkpoint_sha256'],stages=['analysis','accuracy','sensitivity'],shards=8))
 status={'state':'running','stage':None,'completed_stages':[],'started':time.time(),'failed':[]}
 running=[]
 try:
  for stage in ['analysis','accuracy','sensitivity']:
   assert {name:digest(ROOT/name) for name in SOURCES}==source_hashes,'Source changed while running'
   status['stage']=stage;running=[]
   for gpu in range(8):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    log=(run/'logs'/f'{stage}_{gpu}.log').open('a')
    cmd=[str(ROOT/'.venv/bin/python'),'-u','scripts/qwen35_memory_probe_worker.py','--run',str(run),'--stage',stage,'--shard',str(gpu),'--shards','8']
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    running.append((gpu,proc,log))
   while True:
    failures=[{'gpu':gpu,'returncode':proc.poll()} for gpu,proc,_ in running if proc.poll() not in (None,0)]
    if failures:raise RuntimeError(str(failures))
    status.update(progress=counts(run,stage,config),workers=[{'gpu':gpu,'pid':proc.pid,'returncode':proc.poll()} for gpu,proc,_ in running],elapsed_s=time.time()-status['started'])
    dump(run/'status.json',status)
    if all(proc.poll() is not None for _,proc,_ in running):break
    time.sleep(10)
   for _,_,log in running:log.close()
   assert status['progress']=={b:info['samples'] for b,info in config['evaluation'].items()}
   status['completed_stages'].append(stage);dump(run/'status.json',status)
   subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/report_qwen35_memory_probe.py','--run',str(run)],cwd=ROOT,check=True)
  status.update(state='complete',stage=None,elapsed_s=time.time()-status['started']);dump(run/'status.json',status)
 except BaseException as e:
  for _,proc,log in running:
   if proc.poll() is None:proc.terminate()
  for _,proc,log in running:
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill()
   log.close()
  status.update(state='failed',failed=[repr(e)],elapsed_s=time.time()-status['started']);dump(run/'status.json',status)
  raise

if __name__=='__main__':main()
