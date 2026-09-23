"""Audit and collect the paired 7B/13B DART runs excluding full layers 0/1."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAMES=['mmstar','realworldqa','gqa','mmb','mmb-cn','mme','pope','sqa','vqav2']


def main():
    runs=[Path(Path(f'/tmp/llava{s}_dart_active_run').read_text().strip()) for s in ['7b','13b']]
    report=ROOT/'artifacts/reports/llava_dart_7b_13b_prunable_layers_seed44_20260922'
    report.mkdir(parents=True,exist_ok=True)
    tables=[];audits=[];configs=[]
    for size,run in zip(['7B','13B'],runs):
        status=json.loads((run/'status.json').read_text());assert status['state']=='complete',(size,status)
        c=json.loads((run/'config.json').read_text());configs.append(c)
        rs=[json.loads(l) for f in (run/'full').glob('shard*.jsonl') for l in f.read_text().splitlines()]
        lookup={(r['benchmark'],r['sample'],r['retention']):r for r in rs};assert len(lookup)==len(rs)==17530
        for name,info in c['data'].items():
            for index in range(info['samples']):
                a,b=[lookup[name,index,r] for r in [.05,.2]]
                assert a['input_sha256']==b['input_sha256']
                assert a['max_new_tokens']==b['max_new_tokens']==info['max_new_tokens']
        for r in rs:
            k=29 if r['retention']==.05 else 115;a=r['token_audit']
            assert a['visual_tokens_per_layer']==[576]*2+[k]*(c['layers']-2)
            assert a['excluded_compulsory_full_layers']==[0,1]
            assert abs(a['prunable_layer_retention']-k/576)<1e-12
            assert len(a['selected_indices'])==len(set(a['selected_indices']))==k
            assert set(a['image_pivots'])<=set(a['selected_indices'])
        saved=json.loads((run/'summary.json').read_text())
        for retention in [.05,.2]:
            row=dict(model=f'LLaVA-1.5-{size}',retention=retention)
            for name in NAMES:
                vals=[r['score'] for r in rs if r['retention']==retention and r['benchmark']==name]
                assert len(vals)==c['data'][name]['samples']
                row[name]=100*sum(vals)/len(vals)
            row['AVG']=sum(row[n] for n in NAMES)/9
            reference=next(r for r in saved if r['retention']==retention)
            assert all(abs(row[n]-reference[n])<1e-9 for n in [*NAMES,'AVG'])
            tables.append(row)
        audits.append(dict(model=size,run=str(run),unique_predictions=17530,paired_inputs_equal=True,
            actual_layer_counts_verified=True,image_pivots_included=True,averages_recomputed=True))
    assert all(configs[0]['data'][n]['sha256']==configs[1]['data'][n]['sha256'] for n in NAMES)
    lines=['# DART: LLaVA-1.5-7B and 13B, corrected retention definition','',
        'Both compulsory full-retention layers (0 and 1) are excluded. Retention is summed over layers 2–31 (7B) or 2–39 (13B). Original visual count: 576. Retained count: 29 (5.0347%) or 115 (19.9653%).','',
        'Identical seed44 manifests across models and retentions. RealWorldQA: 765; all other datasets: 1000. FA2, BF16, greedy; generation caps 8, except GQA/VQAv2 16. MME/POPE per-question accuracy, VQAv2 soft accuracy, arithmetic mean of nine unrounded scores. No extra no-EOS-zero rule.','',
        '| Model | Retention | '+' | '.join(NAMES+['AVG'])+' |','|---|---:|'+'---:|'*10]
    for r in tables:lines.append('| '+r['model']+f" | {r['retention']:.0%} | "+' | '.join(f'{r[n]:.2f}' for n in NAMES+['AVG'])+' |')
    lines+=['','Previous runs excluding only layer 0 are superseded and use a different budget.']
    for size,run in zip(['7B','13B'],runs):lines.append(f'- [{size} run]({run}/RESULTS.md); [validation]({run}/validation.json)')
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (report/'summary.json').write_text(json.dumps(tables,indent=2))
    (report/'audit.json').write_text(json.dumps(audits,indent=2))
    with (report/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['model','retention',*NAMES,'AVG']);w.writeheader();w.writerows(tables)
    print((report/'RESULTS.md').read_text())


if __name__=='__main__':main()
