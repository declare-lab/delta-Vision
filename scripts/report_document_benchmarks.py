"""Independently rescore saved document answers and validate the complete table."""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_document_benchmarks import OUT, BENCHES, METHODS, ratios, manifest, sha, dump, load_rows
from src.benchmarks import score_prediction, get_benchmark_spec

NAMES={'base':'Qwen3-VL-4B (base)','fastv':'FastV','dart':'DART','visionzip':'VisionZip',
       'sparsevlm':'SparseVLM','divprune':'DivPrune','zoo':'Zoo-Prune',
       'embedding_adapter':'Embedding adapter','recurrent_adapter':'Recurrent adapter'}


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    plan_hash=sha(OUT/'plan.json')
    records=[r for p in (OUT/'full').glob('*.jsonl') for r in load_rows(p)]
    assert len(records)==45000, f'Incomplete: {len(records)}/45000'
    assert all(r['plan_sha256']==plan_hash for r in records)
    expected={(m,ret,b,i) for m in METHODS for ret in ratios(m) for b in BENCHES for i in range(1000)}
    assert {(r['method'],r['retention'],r['benchmark'],r['index']) for r in records}==expected
    data={b:[json.loads(l) for l in manifest(b).read_text().splitlines()] for b in BENCHES}
    for b in BENCHES:
        assert sha(manifest(b))==plan['data'][b]['manifest_sha256']
    hashes={};groups={};changed=0
    for r in records:
        row=data[r['benchmark']][r['index']]
        assert (r['source_index'],r['question_id'])==(row['index'],row['question_id'])
        assert r['deepstack'] is False and r['attention']=='flash_attention_2'
        result=score_prediction(metric=get_benchmark_spec(r['benchmark']).metric,
            prediction_text=r['prediction_text'],answer=row['answer'],answers=row['answers'],question=row['question'])
        changed+=abs(result['score']-r['score'])>1e-12
        key=r['benchmark'],r['index']
        assert hashes.setdefault(key,r['input_sha256'])==r['input_sha256']
        groups.setdefault((r['method'],r['retention'],r['benchmark']),[]).append(r)
    assert changed==0
    rows=[];budget=[]
    for method in METHODS:
        for retention in ratios(method):
            row={'Method':NAMES[method],'Retention':f'{retention:.0%}' if method in METHODS[1:7] else '—'}
            for bench in BENCHES:
                rs=groups[(method,retention,bench)]
                assert len(rs)==1000
                row[bench]=100*statistics.mean(r['score'] for r in rs)
                budget.append(dict(method=method,retention=retention,benchmark=bench,
                    actual_summed_visual_retention=sum(sum(r['layer_visual_tokens']) for r in rs)/(36*sum(r['visual_tokens'] for r in rs)),
                    mean_visual_tokens=statistics.mean(r['visual_tokens'] for r in rs),
                    mean_generated_tokens=statistics.mean(len(r['generated_token_ids']) for r in rs),
                    empty=sum(r['invalid'] for r in rs),at_generation_limit=sum(r['truncated'] for r in rs)))
            row['AVG']=statistics.mean(row[b] for b in BENCHES)
            rows.append(row)
    with (OUT/'FINAL.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ChartQA / DocVQA / InfographicVQA','',
        'Qwen3-VL-4B-Instruct; BF16, FA2, DeepStack off. Each cell contains exactly 1000 questions. All methods use identical sampled questions, images, processor settings and prompts.',
        'ChartQA: Relaxed Accuracy on 500 human + 500 augmented test questions. DocVQA and InfographicVQA: ANLS on seeded validation subsets. All values x100. AVG is the arithmetic mean of the three unrounded scores.',
        '5% and 20% denote the existing post-pruning visual retention parameter; actual layer-summed ratios are in token_budget_audit.json. Adapters keep all visual K/V tokens.',
        'Adapters: PixMo static KL / recurrent KL, step 2000. Greedy cached decoding, 128-token limit, native processor resolution. No training was performed.','',
        '| Method | Retention | ChartQA | DocVQA | InfographicVQA | AVG |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join([row['Method'],row['Retention'],*[f'{row[b]:.2f}' for b in BENCHES],f"{row['AVG']:.2f}"])+' |')
    lines+=['','Metric references: [ChartQA](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/chartqa/utils.py), [ANLS evaluation](https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py). Downloaded reference source hashes and 2000 parity checks are saved in metric_reference/.']
    (OUT/'FINAL.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'token_budget_audit.json',budget)
    dump(OUT/'final_scoring_audit.json',dict(predictions=len(records),cells=45,samples_per_cell=1000,
        rescored_from_prediction_text=True,changed_scores=changed,all_input_hashes_match=True,
        plan_sha256=plan_hash,scorer_sha256=sha(ROOT/'src/document_metrics.py'),
        metric_reference_parity=json.loads((OUT/'metric_reference/parity.json').read_text()),
        adapter_cached_vs_uncached_comparisons=len(json.loads((OUT/'adapter_decode_parity.json').read_text()))))
    print('\n'.join(lines))


if __name__=='__main__':
    main()
