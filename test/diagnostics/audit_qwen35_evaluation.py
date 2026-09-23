"""Rescore saved outputs without overwriting the original experimental record."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.benchmarks import score_prediction


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args();run=args.run_dir
 cfg=json.loads((run/'config.json').read_text());ref=Path(cfg['reference_run'])
 report={'sampling':{},'methods':{},'score_changes':[]}
 for name,info in cfg['evaluation'].items():
  subset=[json.loads(s) for s in Path(info['path']).read_text().splitlines()];full=[json.loads(s) for s in Path(info['source']).read_text().splitlines()]
  subcats=collections.Counter(r.get('category','unspecified') for r in subset);fullcats=collections.Counter(r.get('category','unspecified') for r in full)
  report['sampling'][name]=dict(selected=len(subset),full=len(full),is_ordered_prefix=subset==full[:len(subset)],selected_categories=dict(subcats),full_categories=dict(fullcats),missing_categories=sorted(set(fullcats)-set(subcats)))
  for method in ('native','adapter','divprune_5','divprune_20','dart_5','dart_20'):
   root=ref if method in ('native','adapter') else run
   preds=sorted([json.loads(s) for f in (root/'eval'/method).glob(f'{name}.shard*.jsonl') for s in f.read_text().splitlines()],key=lambda p:p['index'])
   assert [p['index'] for p in preds]==list(range(len(subset)))
   old=[];new=[];invalid=0;atcap=0
   for p,r in zip(preds,subset):
    sc=score_prediction(metric=info['metric'],prediction_text=p['prediction_text'],answer=r.get('answer'),answers=r.get('answers'),choices=r.get('choices'),question=r.get('question'))
    old.append(p['score']);new.append(sc['score']);invalid+=sc.get('invalid',False);atcap+=p['generated_tokens']==info['max_new_tokens']
    if p['prediction']!=sc['prediction'] or p['score']!=sc['score']:
     report['score_changes'].append(dict(method=method,benchmark=name,index=p['index'],text=p['prediction_text'],old_prediction=p['prediction'],new_prediction=sc['prediction'],old_score=p['score'],new_score=sc['score'],gold=sc['gold']))
   report['methods'].setdefault(method,{})[name]=dict(n=len(subset),old_score=100*sum(old)/len(old),rescored=100*sum(new)/len(new),changed_score=sum(a!=b for a,b in zip(old,new)),invalid_after_rescore=invalid,at_generation_limit=atcap,limit=info['max_new_tokens'])
 (run/'evaluation_protocol_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 names=list(cfg['evaluation'])
 lines=['# Saved-answer rescoring after fixing choice extraction','',
 'Diagnostic only: the original 8/16-token generations and ordered-prefix sample remain unchanged.',
 'This is not a complete corrected evaluation: truncated answers require new generation.',
 '', '| Method | '+' | '.join(names)+' | AVG |','|---|'+'---:|'*(len(names)+1)]
 for method,bench in report['methods'].items():
  scores=[bench[n]['rescored'] for n in names]
  lines.append('| '+method+' | '+' | '.join(f'{s:.2f}' for s in scores+[sum(scores)/len(scores)])+' |')
 (run/'RESCORED_SHORT_OUTPUTS.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
 print('score_changes',sum(x['old_score']!=x['new_score'] for x in report['score_changes']))

if __name__=='__main__':main()
