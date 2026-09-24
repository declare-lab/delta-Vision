"""Eight-GPU ABBA comparison on a fixed 48-item stratified Video-MME subset.

Cases run serially. Stops only the known RAM-only load, restores it in finally.
Each worker checks the archived input/token/logit/KV reference before timing.
"""
import json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'artifacts/maintenance/video_benchmark_unification_20260924'
PREVIOUS=ROOT/'artifacts/diagnostics/video_adapter_layer_ablation_999_20260923'
BURN=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')

def dump(name,value):
 p=AUDIT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def report():
 records=[];selection=json.loads((AUDIT/'selection.json').read_text())['indices']
 for case in ['base','adapter']:
  data={}
  for name in ['old0','new0','new1','old1']:
   rows=[json.loads(s) for p in (AUDIT/'runs'/f'{case}_{name}'/case).glob('shard*.jsonl') for s in p.read_text().splitlines()]
   assert sorted(x['index'] for x in rows)==sorted(selection),(case,name,len(rows));data[name]={x['index']:x for x in rows}
  for i in selection:
   for name in data:
    for field in ['input_sha256','tokens','layer_lengths','prefill_kv_sha256','final_kv_sha256']:
     assert data[name][i][field]==data['old0'][i][field],(case,name,i,field)
    assert data[name][i]['timed_encoder_calls']==data[name][i]['timed_captures']==data[name][i]['timed_fallbacks']==0
  result={'case':case,'samples':len(selection),'pairing_exact':True,'modes':{}}
  for mode in ['trials','continuous_trials']:
   stages={}
   for stage in ['prefill','decode','total']:
    means={name:1000*statistics.mean(statistics.median(t[stage+'_s'] for t in row[mode]) for row in data[name].values()) for name in data}
    old=(means['old0']+means['old1'])/2;new=(means['new0']+means['new1'])/2
    stages[stage]={'old_ms':old,'new_ms':new,'change_pct':100*(new/old-1),'round_means_ms':means,'within_5pct':abs(new/old-1)<=.05}
   result['modes'][mode]=stages
  result['peak_GiB']={label:max(t['peak_GiB'] for name in data if name.startswith(label) for row in data[name].values() for t in row['continuous_trials']) for label in ['old','new']}
  records.append(result)
 dump('speed_comparison.json',records)
 lines=['# Video-MME测速目录迁移回归','','48条：短/中/长各16，seed44，来自固定999条清单。8卡，base/adapter串行，ABBA旧/新/新/旧；每输入3次计时。FA2/DeepStack-off/CUDA Graph/adapter fast-path；视觉编码不计时；固定8输出token，7次decode forward。主表沿用逐forward计时，continuous另记JSON。','',
 '| Model | Stage | Before ms | After ms | Change | Within ±5% |','|---|---|---:|---:|---:|---|']
 for r in records:
  for stage,v in r['modes']['trials'].items():lines.append(f"| {r['case']} | {stage} | {v['old_ms']:.4f} | {v['new_ms']:.4f} | {v['change_pct']:+.2f}% | {v['within_5pct']} |")
 lines+=['','所有48条、两轮、两个方法：输入hash、生成token、prefill/final KV hash完全一致；每个worker独立验证8步logits对历史参考一致。计时内没有视觉编码、graph capture或fallback。','小样本重构回归，不是999条全量重测。没有为得到相同时间而改计时范围。']
 (AUDIT/'SPEED.md').write_text('\n'.join(lines)+'\n');return records

def main():
 indices=json.loads((AUDIT/'selection.json').read_text())['indices'];jobs=[];restore=False;completed=[]
 def state(status,**extra):dump('status.json',{'state':status,'completed':completed,**extra})
 def interrupted(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  if (BURN.parent/'pid').exists() and alive(int((BURN.parent/'pid').read_text())):
   restore=True;subprocess.run([sys.executable,str(BURN),'stop'],check=True)
  deadline=time.time()+120
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
   if time.time()>deadline:raise RuntimeError('Unrelated GPU jobs still running; none were stopped')
   state('waiting_for_gpu_release');time.sleep(2)
  env=dict(os.environ,RESOURCE_REPO=str(ROOT),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
  for key in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']:env.pop(key,None)
  for case in ['base','adapter']:
   for name in ['old0','new0','new1','old1']:
    run=AUDIT/'runs'/f'{case}_{name}';run.mkdir(parents=True,exist_ok=False);jobs=[]
    for gpu in range(8):
     command=([sys.executable,str(AUDIT/'reference/scripts/measure_video_no_visual_resources.py')] if name.startswith('old') else [sys.executable,'-m','src.benchmarking','videomme','llm','--'])
     command+=['worker','--run',str(run),'--previous',str(PREVIOUS),'--case',case,'--shard',str(gpu),'--indices',*[str(i) for i in indices[gpu::8]]]
     with (run/f'gpu{gpu}.log').open('w') as log:jobs.append(subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in jobs):
     if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError(f'{case}/{name} failed; inspect {run}')
     state('running',case=case,round=name,pids=[p.pid for p in jobs if p.poll() is None]);time.sleep(3)
    if any(p.returncode for p in jobs):raise RuntimeError(f'{case}/{name} failed')
    completed.append(f'{case}/{name}');state('running')
  results=report();state('complete',all_stages_within_5pct=all(v['within_5pct'] for r in results for m in r['modes'].values() for v in m.values()))
 except BaseException as e:state('failed',error=repr(e));raise
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  if restore:
   done=subprocess.run([sys.executable,str(BURN),'start','--coexist'],capture_output=True,text=True);dump('burn_restoration.json',{'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr})

if __name__=='__main__':main()
