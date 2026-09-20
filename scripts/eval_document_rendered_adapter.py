"""Evaluate the existing all-layer rendered-QA adapter on frozen document samples."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_document_benchmarks as suite

OUT = ROOT / 'artifacts/eval/document_rendered_adapter_1000_20260919'
TRAIN = ROOT / 'artifacts/experiments/render_sequential_joint/sequential_vs_joint_2ep_20260911/sequential_stage1_adapter'
CHECKPOINT = TRAIN / 'checkpoints/qwen_embedding_adapter_final.pt'
PREVIOUS = ROOT / 'artifacts/eval/document_benchmarks_1000_20260918'
RECURRENT = ROOT / 'artifacts/experiments/pixmo_recurrent_rank1024/qwen3vl4b_pixmo_recurrent_kl_rank1024_2000_20260919_111920'


def configure():
    suite.OUT = OUT
    suite.METHODS = ['embedding_adapter']
    suite.CHECKPOINTS = {'embedding_adapter': CHECKPOINT}


def report():
    from src.benchmarks import get_benchmark_spec, score_prediction
    records = [r for p in (OUT/'full').glob('*.jsonl') for r in suite.load_rows(p)]
    assert len(records) == 3000
    assert {(r['benchmark'], r['index']) for r in records} == {(b, i) for b in suite.BENCHES for i in range(1000)}
    plan = json.loads((OUT/'plan.json').read_text())
    oldplan = json.loads((PREVIOUS/'plan.json').read_text())
    for key in ['data', 'model', 'attention', 'dtype', 'deepstack', 'prompt', 'max_new_tokens', 'decoding', 'image_resolution', 'metrics']:
        assert plan[key] == oldplan[key], key
    old = {(r['benchmark'], r['index']): r for p in (PREVIOUS/'full').glob('embedding_adapter_*.jsonl') for r in suite.load_rows(p)}
    data = {b: suite.load_rows(suite.manifest(b)) for b in suite.BENCHES}
    for r in records:
        row = data[r['benchmark']][r['index']]
        assert r['input_sha256'] == old[(r['benchmark'], r['index'])]['input_sha256']
        assert (r['source_index'], r['question_id']) == (row['index'], row['question_id'])
        score = score_prediction(metric=get_benchmark_spec(r['benchmark']).metric,
            prediction_text=r['prediction_text'], answer=row['answer'], answers=row['answers'])['score']
        assert abs(score-r['score']) < 1e-12
    with (RECURRENT/'RESULTS.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in [*suite.BENCHES, 'AVG']:
            row[key] = float(row[key])
    row = {'Method': 'Rendered-QA embedding adapter, rank512, all36'}
    for b in suite.BENCHES:
        chosen = [r for r in records if r['benchmark'] == b]
        assert len(chosen) == 1000
        row[b] = sum(r['score'] for r in chosen)/10
    row['AVG'] = sum(row[b] for b in suite.BENCHES)/3
    rows.append(row)
    with (OUT/'RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Rendered-QA adapter document evaluation', '',
        'Qwen3-VL-4B-Instruct; BF16; FA2; DeepStack disabled. Identical frozen 1000 samples per benchmark and processor settings as the previous document experiment.',
        'ChartQA: relaxed accuracy; DocVQA and InfographicVQA: ANLS. All scores x100; AVG is the arithmetic mean of unrounded scores.',
        'New checkpoint: static embedding adapter, rank512, all36 layers, 2896 steps (one epoch), rendered QA training, frozen native raw-text teacher, gold-prefix top1024 KL. No LoRA. Existing comparison rows were not rerun.',
        'Training data, rank, teacher input modality, and training duration differ from PixMo runs; this comparison does not isolate the effect of training data.', '',
        '| Method | ChartQA | DocVQA | InfographicVQA | AVG |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join([row['Method'], *[f'{row[b]:.2f}' for b in suite.BENCHES], f"{row['AVG']:.2f}"])+' |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    suite.dump(OUT/'scoring_audit.json', dict(predictions=3000, samples_each=1000,
        all_input_hashes_match_previous=True, all_scores_recomputed_and_equal=True,
        empty=sum(r['invalid'] for r in records), at_generation_limit=sum(r['truncated'] for r in records)))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--stage', choices=['smoke', 'full'], default='full')
    args = parser.parse_args()
    configure()
    if args.worker:
        suite.worker(argparse.Namespace(method='embedding_adapter', shard=args.shard, shards=8, stage=args.stage))
        return
    if args.verify_only:
        from scripts import verify_document_adapter_decode as verify
        verify.OUT = OUT
        verify.CHECKPOINTS = suite.CHECKPOINTS
        verify.main()
        return
    import torch
    torch.set_num_threads(4)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = ckpt['adapter_config']
    assert ckpt['global_step'] == 2896
    assert cfg['output_mode'] == 'embedding_adapter' and cfg['visual_adapter_rank'] == 512
    assert cfg['adapter_start_layer'] == 0 and cfg['active_adapter_layers'] == 0
    assert len(ckpt['state_dict']) == 72
    assert all(torch.isfinite(t).all() for t in ckpt['state_dict'].values())
    del ckpt
    suite.preflight()
    suite.dump(OUT/'training_provenance.json', dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=suite.sha(CHECKPOINT),
        training_args=json.loads((TRAIN/'checkpoints/args.json').read_text()), adapter_config=cfg,
        global_step=2896, wrapper_sha256=suite.sha(__file__)))
    (OUT/'source/scripts/eval_document_rendered_adapter.py').write_bytes(Path(__file__).read_bytes())
    for stage in ['smoke', 'full']:
        jobs = []
        for shard in range(1 if stage == 'smoke' else 8):
            handle = (OUT/'logs'/f'{stage}_{shard}.log').open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_DISABLE_PROGRESS_BARS='1')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--stage', stage, '--shard', str(shard)]
            jobs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
        while any(p.poll() is None for p, _ in jobs):
            count = suite.aggregate(stage)
            suite.dump(OUT/'status.json', dict(state='running', stage=stage, predictions=count, updated=time.time()))
            time.sleep(15)
        for p, handle in jobs:
            handle.close()
            assert p.returncode == 0, f'{stage} worker failed: {p.returncode}'
        suite.aggregate(stage, require_complete=True)
        if stage == 'smoke':
            with (OUT/'logs/decode_parity.log').open('w') as f:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--verify-only'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4'), stdout=f, stderr=subprocess.STDOUT, check=True)
    report()
    suite.dump(OUT/'status.json', dict(state='complete', predictions=3000, finished=time.time()))


if __name__ == '__main__':
    main()
