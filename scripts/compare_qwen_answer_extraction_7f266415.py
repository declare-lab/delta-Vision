"""Replay preserved answers through the exact requested Git scoring implementation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.benchmarks import score_prediction,get_benchmark_spec
COMMIT='7f266415a28b3801339da93211a8fd9de2ff319e'
RUN=ROOT/'artifacts/eval/qwen4b_adapter_image_reproduction_20260922'
DEST=RUN/'extraction_7f266415';DEST.mkdir(exist_ok=True)
source=subprocess.check_output(['git','show',f'{COMMIT}:src/benchmarks.py'],text=True,cwd=ROOT)
(DEST/'benchmarks_7f266415.py').write_text(source)
legacy=types.ModuleType('benchmarks_7f266415');sys.modules[legacy.__name__]=legacy
exec(compile(source,str(DEST/'benchmarks_7f266415.py'),'exec'),legacy.__dict__)
cfg=json.loads((RUN/'config.json').read_text());data={b:[json.loads(l) for l in Path(v['path']).read_text().splitlines()] for b,v in cfg['evaluation'].items()}
records=[]
for f in (RUN/'rows').glob('shard*.jsonl'):
 for l in f.read_text().splitlines():
  try:records.append(json.loads(l))
  except json.JSONDecodeError:continue
changes=[];all_scores=[]
for r in records:
 b=r['benchmark'];row=data[b][r['sample']]
 kw=dict(prediction_text=r['prediction_text'],answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
 current=score_prediction(metric=get_benchmark_spec(b).metric,**kw)
 old=legacy.score_prediction(metric=legacy.get_benchmark_spec(b).metric,**kw)
 assert current['score']==r['score']
 item=dict(benchmark=b,sample=r['sample'],method=r['method'],members=r['members'],prediction_text=r['prediction_text'],current=current,reference=old)
 all_scores.append(item)
 if current['prediction']!=old['prediction'] or current['gold']!=old['gold'] or current['score']!=old['score']:
  changes.append(dict(item,question=row.get('question'),answer=row.get('answer'),choices=row.get('choices')))
summary={};lines=['# Identical generated answers: current extraction vs Git 7f266415','',f'Exact reference commit: {COMMIT}. Inputs and generated answer strings are held fixed. POPE/MME remain per-question accuracy; VQAv2 remains soft accuracy.','']
wide=[]
for cohort in cfg['cohorts']:
 summary[cohort]={};lines += [f'## {cohort}','','| Benchmark | Method | N | Current | 7f266415 | Change pp | Changed scores |','|---|---|---:|---:|---:|---:|---:|']
 for b,info in cfg['evaluation'].items():
  summary[cohort][b]={}
  for m in cfg['methods']:
   subset=[r for r in all_scores if r['benchmark']==b and r['method']==m for mem in r['members'] if mem['cohort']==cohort]
   if len(subset)!=info['cohort_counts'][cohort]:continue
   new=100*sum(r['current']['score'] for r in subset)/len(subset);old=100*sum(r['reference']['score'] for r in subset)/len(subset)
   changed=sum(r['current']['score']!=r['reference']['score'] for r in subset)
   entry=dict(benchmark=b,method=m,cohort=cohort,samples=len(subset),current=new,reference=old,delta_pp=old-new,changed_scores=changed)
   wide.append(entry);summary[cohort][b][m]=entry
   lines.append(f'| {b} | {m} | {len(subset)} | {new:.2f} | {old:.2f} | {old-new:+.2f} | {changed} |')
 lines.append('')
for filename,obj in [('changes.json',changes),('summary.json',summary),('scores.json',all_scores),('audit.json',dict(commit=COMMIT,source_sha256=hashlib.sha256(source.encode()).hexdigest(),predictions=len(records),changed_scores=sum(r['current']['score']!=r['reference']['score'] for r in all_scores),changed_extraction=len(changes)))]:
 (DEST/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
(DEST/'RESULTS.md').write_text('\n'.join(lines)+'\n')
with (DEST/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(wide[0]));w.writeheader();w.writerows(wide)
print((DEST/'audit.json').read_text());print('\n'.join(lines))
