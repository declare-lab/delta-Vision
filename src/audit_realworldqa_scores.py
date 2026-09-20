"""Rescore saved predictions without overwriting historical results."""
import json,collections,runpy
from pathlib import Path
from src.benchmarks import score_realworldqa_prediction,_realworldqa_choices_from_question
root=Path('artifacts/diagnostics');out=root/'realworldqa_scoring_audit_20260917';data=[json.loads(s) for s in (root/'channel_native_cache_20260916/realworldqa_eval.jsonl').read_text().splitlines()];summary={};changes=[]
for suite in ['native_visual_self_attention_20260917','native_visual_self_attention_long_20260917']:
 rows=[json.loads(s) for f in (root/suite).glob('rows*.jsonl') for s in f.read_text().splitlines()];summary[suite]={}
 for mode in ['native','visual_self_only']:
  rr=[r for r in rows if r['dataset']=='realworldqa' and r['mode']==mode];new=[]
  for r in rr:
   d=data[r['sample']];v=score_realworldqa_prediction(r['answer'],d['answer'],d.get('choices'),d['question']);new.append(v['score'])
   if v['score']!=r['score']:changes.append({'suite':suite,'sample':r['sample'],'mode':mode,'answer':r['answer'],'gold':d['answer'],'old_score':r['score'],'new_score':v['score']})
  summary[suite][mode]={'n':len(rr),'old_correct':sum(r['score'] for r in rr),'new_correct':sum(new),'accuracy_pct':100*sum(new)/len(rr)}
counts=collections.Counter(len(_realworldqa_choices_from_question(r['question'])) for r in data);summary['choice_counts']=dict(counts);summary['mcq_uniform_random_pct']=100*sum(n/k for k,n in counts.items() if k)/sum(n for k,n in counts.items() if k)
(out/'results.json').write_text(json.dumps(summary,indent=2));(out/'changed_scores.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))
