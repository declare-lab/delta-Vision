"""Paired checkpoint-template rerun of LLaVA-Mistral DivPrune on the same lists."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from scripts import queue_divprune_rerun as queue

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'llava-1.6-mistral-7b'


def prepare(reference, run):
    config = json.loads((reference/'config.json').read_text())
    old_plan = json.loads((reference/'plan.json').read_text())
    run.mkdir(parents=True, exist_ok=False)
    for name in ['logs','data','source/src','source/scripts']:(run/name).mkdir(parents=True)
    config.update(created=time.time(),paired_reference=str(reference),llava_prompt_template='checkpoint',
                  input_policy='Same seed44 questions and images; checkpoint chat template instead of legacy USER/ASSISTANT')
    config['models'] = {MODEL:dict(config['models'][MODEL],shards=8)}
    config['multimodal'] = {}
    for name,info in config['single_image'].items():
        target=run/'data'/f'{name}.jsonl';shutil.copy2(info['path'],target)
        assert queue.sha(target)==info['sha256']
        info['path']=str(target)
    # Reuse immutable prior dependencies; snapshot only the changed files anew.
    (run/'source/baselines').symlink_to(reference/'source/baselines',target_is_directory=True)
    hashes={}
    for rel,digest in old_plan['source_sha256'].items():
        if rel.startswith('baselines/'):
            hashes[rel]=digest
        elif rel.startswith('src/'):
            (run/'source'/rel).symlink_to(reference/'source'/rel)
            hashes[rel]=digest
    for rel in ['src/data.py','src/divprune_rerun.py','scripts/queue_divprune_rerun.py','scripts/queue_llava_mistral_prompt.py']:
        dest=run/'source'/rel
        if dest.is_symlink():dest.unlink()
        shutil.copy2(ROOT/rel,dest);hashes[rel]=queue.sha(dest)
    queue.dump(run/'config.json',config)
    queue.dump(run/'multimodal_input_cache.json',{})
    queue.dump(run/'plan.json',dict(source_sha256=hashes,cache_index_sha256=queue.sha(run/'multimodal_input_cache.json'),
        total_predictions=2*sum(i['samples'] for i in config['single_image'].values()),
        changed_variable='LLaVA-Mistral chat template; scorer/retention/data/generation caps unchanged'))
    queue.dump(run/'status.json',dict(state='prepared'))


def comparison(run, records):
    config=json.loads((run/'config.json').read_text())
    reference=Path(config['paired_reference'])
    old_rows=[json.loads(l) for p in (reference/'full'/MODEL).glob('*.jsonl') for l in p.read_text().splitlines()]
    old={(r['benchmark'],r['sample'],r['retention']):r for r in old_rows}
    names=list(config['single_image'])
    old_scores={rate:{name:100*sum(r['score'] for r in old_rows if r['benchmark']==name and r['retention']==rate)/config['single_image'][name]['samples']
                     for name in names} for rate in [.05,.2]}
    for rate in old_scores:old_scores[rate]['AVG']=sum(old_scores[rate].values())/len(names)
    new={row['retention']:row for row in records}
    lines=['# LLaVA-1.6-Mistral-7B: prompt comparison','',
           'Same seed44 lists, FA2, DivPrune, generation caps and per-question scoring. Only the chat template changes.',
           'Old: USER / ASSISTANT. New: checkpoint [INST] ... [/INST]. Delta is percentage points.','',
           '| Benchmark | Old 5% | New 5% | Delta | Old 20% | New 20% | Delta |',
           '|---|---:|---:|---:|---:|---:|---:|']
    changes=[]
    for name in names+['AVG']:
        cells=[]
        for rate in [.05,.2]:
            a=old_scores[rate][name];b=new.get(rate,{}).get(name)
            cells.extend([f'{a:.2f}',f'{b:.2f}' if b is not None else 'pending',f'{b-a:+.2f}' if b is not None else 'pending'])
            if b is not None:changes.append(dict(benchmark=name,retention=rate,old=a,new=b,delta_pp=b-a))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    (run/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    queue.dump(run/'comparison.json',changes)
    if all('AVG' in row for row in records):
        new_rows=[json.loads(l) for p in (run/'full'/MODEL).glob('*.jsonl') for l in p.read_text().splitlines()]
        paired=[]
        for r in new_rows:
            previous=old[(r['benchmark'],r['sample'],r['retention'])]
            assert previous['source_index']==r['source_index'] and previous['gold']==r['gold']
            if previous['text']!=r['text'] or previous['score']!=r['score']:
                paired.append(dict(benchmark=r['benchmark'],sample=r['sample'],retention=r['retention'],
                    old_text=previous['text'],new_text=r['text'],old_score=previous['score'],new_score=r['score'],gold=r['gold']))
        queue.dump(run/'changed_answers.json',paired)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--start-prepared',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    run=a.run_dir.resolve() if a.run_dir else ROOT/'artifacts/eval'/('llava_mistral_checkpoint_prompt_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if not a.start_prepared:prepare(a.reference.resolve(),run)
    print('RUN_DIR='+str(run),flush=True)
    if not a.prepare_only:
        original_report=queue.report
        def report(r):
            records=original_report(r);comparison(r,records);return records
        queue.report=report
        queue.execute(run)
