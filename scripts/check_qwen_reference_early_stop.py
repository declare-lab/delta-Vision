"""Replay 7f266415 adapter answer stopping on saved generated token prefixes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import typing
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/eval/qwen4b_adapter_image_reproduction_20260922'
DEST=RUN/'extraction_7f266415';COMMIT='7f266415a28b3801339da93211a8fd9de2ff319e'
legacy=types.ModuleType('reference_benchmarks');sys.modules[legacy.__name__]=legacy
source=(DEST/'benchmarks_7f266415.py').read_text();exec(compile(source,'reference_benchmarks.py','exec'),legacy.__dict__)
entry_source=subprocess.check_output(['git','show',f'{COMMIT}:src/eval_benchmarks.py'],cwd=ROOT,text=True)
(DEST/'eval_benchmarks_7f266415.py').write_text(entry_source)
node=next(n for n in ast.parse(entry_source).body if isinstance(n,ast.FunctionDef) and n.name=='_structured_answer_ready')
scope=dict(Any=typing.Any,extract_choice=legacy.extract_choice,extract_yes_no=legacy.extract_yes_no)
exec(compile(ast.Module(body=[node],type_ignores=[]),'reference_early_stop','exec'),scope)
from transformers import AutoTokenizer
cfg=json.loads((RUN/'config.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model_path'])
data={b:[json.loads(l) for l in Path(v['path']).read_text().splitlines()] for b,v in cfg['evaluation'].items()}
records=[]
for f in (RUN/'rows').glob('shard*.jsonl'):
 for l in f.read_text().splitlines():
  try:records.append(json.loads(l))
  except json.JSONDecodeError:continue
out=[];changed=[]
for r in records:
 b=r['benchmark'];row=data[b][r['sample']];metric=legacy.get_benchmark_spec(b).metric;text=r['prediction_text'];stop_len=len(r['generated_token_ids'])
 if r['method']=='adapter':
  for n in range(1,len(r['generated_token_ids'])+1):
   prefix=tok.decode(r['generated_token_ids'][:n],skip_special_tokens=True)
   if scope['_structured_answer_ready'](metric,prefix,row.get('choices')):
    text=prefix;stop_len=n;break
 kwargs=dict(metric=metric,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
 full=legacy.score_prediction(prediction_text=r['prediction_text'],**kwargs)
 stopped=legacy.score_prediction(prediction_text=text,**kwargs)
 new=dict(benchmark=b,sample=r['sample'],method=r['method'],members=r['members'],full_text=r['prediction_text'],reference_text=text,stop_tokens=stop_len,full_eval=full,stopped_eval=stopped)
 out.append(new)
 if full['score']!=stopped['score']:changed.append(new)
summary={}
for cohort in cfg['cohorts']:
 summary[cohort]={}
 for b,info in cfg['evaluation'].items():
  summary[cohort][b]={}
  for m in cfg['methods']:
   rs=[r for r in out if r['benchmark']==b and r['method']==m for mem in r['members'] if mem['cohort']==cohort]
   if len(rs)==info['cohort_counts'][cohort]:summary[cohort][b][m]=100*sum(r['stopped_eval']['score'] for r in rs)/len(rs)
result=dict(commit=COMMIT,records=len(records),changed_scores=len(changed),changes=changed,summary=summary,
    adapter_stopped_earlier=sum(r['method']=='adapter' and r['stop_tokens']<len(original['generated_token_ids']) for r,original in zip(out,records)),
    implementation='Exact reference _structured_answer_ready AST; native reference generate ignores early_stop_metric, adapter applies it each generated token. Prefix replay preserves greedy generated tokens.')
(DEST/'early_stop_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('summary','changes')},indent=2))
