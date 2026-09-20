"""Rescore saved answers and aggregate the nine `score` fields, as in the old runner."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarks import get_benchmark_spec, score_prediction, summarize_metric

OUT = ROOT / 'artifacts/eval/dart_divprune_ret20_5models_9bench_20260918'
MODELS = ['llava-1.5-7b-hf', 'llava-1.5-13b-hf', 'llava-v1.6-mistral-7b-hf', 'qwen3-vl-8b', 'qwen3-vl-30b-a3b']
NAMES = ['LLaVA-1.5-7B', 'LLaVA-1.5-13B', 'LLaVA-1.6-Mistral-7B', 'Qwen3-VL-8B', 'Qwen3-VL-30B-A3B']
BENCHMARKS = 'mmstar,gqa,mmb,mmb-cn,mme,pope,sqa,vqav2,realworldqa'.split(',')
plan = json.loads((OUT / 'plan.json').read_text())
scorer = ROOT / 'src/benchmarks.py'
assert hashlib.sha256(scorer.read_bytes()).hexdigest() == plan['source_sha256']['src/benchmarks.py'], 'Scorer changed since the run'
data = {}
for name in BENCHMARKS:
    info = plan['data'][name]
    raw = Path(info['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info['sha256'], name
    data[name] = [json.loads(line) for line in raw.splitlines() if line.strip()][:info['samples']]

long_rows, wide_rows = [], []
audit = {'protocol': 'mean of nine benchmark score fields; percent display',
         'pope': 'question accuracy; F1 retained as auxiliary only',
         'mme': 'question accuracy; raw/normalized MME retained as auxiliary only',
         'vqa': 'original soft-consensus score', 'cells': 0, 'predictions': 0,
         'changed_prediction_scores': 0, 'changed_aggregate_scores': 0,
         'historical_5pct_results_verified': False,
         'reference_aggregation': 'baselines/run_all_baselines.sh: score=sum(score)/len(scores)',
         'scorer_sha256': hashlib.sha256(scorer.read_bytes()).hexdigest()}

for model in MODELS:
    for method in ('dart', 'divprune'):
        wide = dict(model=model, method=method)
        for benchmark in BENCHMARKS:
            folder = OUT / 'full' / model / method / 'ret20' / benchmark
            predictions = json.loads((folder / 'predictions.json').read_text())
            original = json.loads((folder / 'results.json').read_text())
            rows = data[benchmark]
            assert len(predictions) == len(rows) == original['samples']
            spec = get_benchmark_spec(benchmark)
            rescored = []
            for i, (pred, row) in enumerate(zip(predictions, rows)):
                expected_index = i if original['model_kind'] == 'qwen' else row.get('index', i)
                assert pred['index'] == expected_index, (model, method, benchmark, i)
                result = score_prediction(metric=spec.metric, prediction_text=pred['prediction_text'],
                                          answer=row.get('answer'), answers=row.get('answers'),
                                          choices=row.get('choices'), question=row.get('question'))
                audit['changed_prediction_scores'] += result['score'] != pred['score']
                rescored.append(dict(index=pred['index'], prediction_text=pred['prediction_text'], **result))
            summary = summarize_metric(spec.metric, rescored, rows)
            assert summary['samples'] == len(rows)
            audit['changed_aggregate_scores'] += abs(summary['score'] - original['score']) > 1e-12
            audit['cells'] += 1
            audit['predictions'] += len(rows)
            dest = OUT / 'rescored_legacy_score' / model / method / benchmark
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'results.json').write_text(json.dumps(summary, indent=2))
            (dest / 'predictions.json').write_text(json.dumps(rescored, ensure_ascii=False))
            wide[benchmark] = 100 * summary['score']
            long_rows.append(dict(model=model, method=method, retention=.2, benchmark=benchmark,
                                  samples=len(rows), score=summary['score'], score_pct=wide[benchmark],
                                  table_metric='question_accuracy' if benchmark in ('mme', 'pope') else spec.metric,
                                  auxiliary_pope_f1=summary.get('f1'), auxiliary_mme_score=summary.get('mme_score')))
        wide['avg'] = sum(wide[b] for b in BENCHMARKS) / len(BENCHMARKS)
        wide_rows.append(wide)

assert audit['cells'] == 90 and audit['predictions'] == 87650
for filename in ('RESULTS.md', 'summary.csv'):
    old = OUT / filename
    archive = OUT / ('previous_mixed_metrics_' + filename)
    if old.exists() and not archive.exists():
        archive.write_bytes(old.read_bytes())
for filename, values in [('summary.csv', long_rows), ('summary_with_avg.csv', wide_rows)]:
    with (OUT / filename).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
lines = ['# DART / DivPrune — 20% retention, legacy score aggregation', '',
         'FA2, BF16; Qwen DeepStack off. First 1000 per benchmark; RealWorldQA all 765.',
         'All displayed values are percentages. POPE and MME use question accuracy; VQAv2 uses soft accuracy.',
         'AVG is the unrounded arithmetic mean of the nine score percentages, matching run_all_baselines.sh.',
         'Recomputed from all 87,650 saved prediction texts. The historical 5% result files remain unavailable.', '',
         '| Model | Method | MMStar | GQA | MMB-EN | MMB-CN | MME Acc | POPE Acc | SQA | VQAv2 | RealWorldQA | AVG |',
         '|---|---|' + '---:|' * 10]
for row in wide_rows:
    values = [NAMES[MODELS.index(row['model'])], row['method']] + [f'{row[b]:.2f}' for b in BENCHMARKS + ['avg']]
    lines.append('| ' + ' | '.join(values) + ' |')
(OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
(OUT / 'rescoring_audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
print('\n'.join(lines[8:]))
