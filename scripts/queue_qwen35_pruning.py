"""Freeze, validate, run four pruning configurations on eight GPUs, and rescore."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qwen35_experiment import dump, sha, score_evaluation_prediction
from src.benchmarks import score_prediction

REFERENCE = ROOT/'artifacts/experiments/qwen35_pixmo/qwen35_4b_embedding128_pixmo2000_20260920_064827'
CASES = [('divprune', .05), ('divprune', .2), ('dart', .05), ('dart', .2)]


def prepare(run, reference=REFERENCE):
    reference = Path(reference).resolve()
    config = json.loads((reference/'config.json').read_text())
    # A new random subset cannot reuse the historical first-1000 reference.
    # Refuse to silently produce another mismatched native/baseline comparison.
    if config.get('evaluation_sampling_seed') != 44 or config.get('evaluation_sampling') != 'uniform_without_replacement':
        raise ValueError('Reference uses the retired sampling protocol. Evaluate native and adapter on the shared random seed=44 manifests first, then pass --reference-run.')
    if (reference/'source/src/benchmarks.py').read_bytes() != (ROOT/'src/benchmarks.py').read_bytes():
        raise ValueError('Reference scoring code differs. Native, adapter and pruning results must use the same corrected scorer.')
    for name, info in config['evaluation'].items():
        assert info.get('seed') == 44 and info.get('sampling') == 'uniform_without_replacement', name
        assert sha(info['path']) == info['sha256'], name
    run.mkdir(parents=True, exist_ok=False)
    config['experiment'] = 'Qwen3.5 DART/DivPrune 5% and 20%, nine single-image benchmarks'
    config['reference_run'] = str(reference)
    config['pruning'] = dict(dart_prune_before_layer=4, dart_ratio='post-pruning layers only',
                             divprune_prune_before_layer=0, divprune_ratio='all layers',
                             cases=CASES, shards_per_case=2)
    (run/'eval_data').mkdir()
    for name, info in config['evaluation'].items():
        target = run/'eval_data'/f'{name}.jsonl'
        shutil.copy2(info['path'], target)
        assert sha(target) == info['sha256']
        info['path'] = str(target)
    dump(run/'config.json', config)
    source = run/'source'
    names = ['src/__init__.py', 'src/qwen35_embedding.py', 'src/qwen35_experiment.py', 'src/qwen_deepstack.py',
             'src/qwen35_pruning.py', 'src/benchmarks.py', 'scripts/qwen35_worker.py',
             'scripts/qwen35_pruning_worker.py', 'scripts/queue_qwen35_pruning.py',
             'test/diagnostics/test_qwen35_pruning.py', 'configs/qwen35_adapter_requirements.txt']
    hashes = {}
    for name in names:
        dest = source/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = sha(dest)
    # Same model weights and isolated dependencies as the completed reference.
    plan = json.loads((reference/'plan.json').read_text())
    dump(run/'plan.json', dict(source_sha256=hashes, model_files_sha256=plan['model_files_sha256'],
        dependencies=plan['dependencies'], reference_run=str(reference),
        selector_sources={name:sha(ROOT/name) for name in [
            'baselines/dart/qwen3_vl/modeling_qwen3_vl_dart.py',
            'baselines/divprune/qwen3_vl/modeling_qwen3_vl_divprune.py']}))


def report(run):
    config = json.loads((run/'config.json').read_text())
    reference = Path(config['reference_run'])
    results, native_rows = [], {}
    for method in ['native', 'adapter']+[f'{m}_{round(r*100)}' for m,r in CASES]:
        row = dict(method=method)
        for name, info in config['evaluation'].items():
            root = reference if method in ('native', 'adapter') else run
            predictions = sorted([json.loads(line) for path in (root/'eval'/method).glob(f'{name}.shard*.jsonl')
                                  for line in path.read_text().splitlines() if line], key=lambda p:p['index'])
            assert [p['index'] for p in predictions] == list(range(info['samples'])), (method, name)
            data = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
            scores = []
            for pred, original in zip(predictions, data):
                score = score_evaluation_prediction(pred, original, info['metric'])
                assert score['score'] == pred['score']
                scores.append(score['score'])
                if method != 'native':
                    native = native_rows[name][pred['index']]
                    assert native['input_ids_sha256'] == pred['input_ids_sha256']
                    assert native['image_grid_thw'] == pred['image_grid_thw']
            if method == 'native':
                native_rows[name] = predictions
            row[name] = 100*sum(scores)/len(scores)
            if method not in ('native', 'adapter'):
                audits = [p['token_audit'] for p in predictions]
                total = sum(a['original_visual_tokens'] for a in audits)
                kept = sum(a['retained_visual_tokens'] for a in audits)
                layer_sum = sum(sum(a['visual_tokens_per_layer']) for a in audits)
                row.setdefault('token_retention', {})[name] = dict(
                    post_pruning=kept/total, all_layers=layer_sum/(32*total))
        row['avg'] = sum(row[name] for name in config['evaluation'])/9
        results.append(row)
    dump(run/'summary.json', results)
    names = list(config['evaluation'])+['avg']
    lines = ['# Qwen3.5-4B: nine single-image benchmarks', '',
        'Same frozen questions, native processor resolution, FA2, FLA, no DeepStack, no thinking.',
        'DART: layers 0–3 retain all visual tokens; 5%/20% applies to layers 4–31.',
        'DivPrune: 5%/20% applies to all 32 layers. Integer counts are rounded per image.',
        'Scores (%): MME/POPE question accuracy, VQAv2 soft score, AVG of nine unrounded scores.', '',
        '| Method | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for row in results:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[n]:.2f}' for n in names)+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def execute(run):
    plan = json.loads((run/'plan.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(run/'source'/name) == digest, name
    logs = run/'logs'
    logs.mkdir(exist_ok=True)
    def status(state, **kwargs):
        dump(run/'status.json', dict(state=state, pid=os.getpid(), updated=time.time(), **kwargs))
    env = dict(os.environ, PYTHONPATH=str(ROOT/'artifacts/dependencies/qwen35_python')+os.pathsep+str(run/'source'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    worker = str(run/'source/scripts/qwen35_pruning_worker.py')
    try:
        if not (run/'validation.json').exists():
            status('validating')
            with (logs/'validation.log').open('a') as handle:
                subprocess.run([sys.executable, worker, 'validate', '--run-dir', str(run)],
                    cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES='0'),
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
        assert json.loads((run/'validation.json').read_text())['passed']
        # Do not overlap another workload; no unrelated process is interrupted.
        status('waiting_gpus')
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            if len(memory) == 8 and all(int(m) < 1024 for m in memory):
                break
            time.sleep(20)
        active = []
        try:
            for case, (method, ratio) in enumerate(CASES):
                for shard in range(2):
                    gpu = case*2+shard
                    handle = (logs/f'{method}_{round(ratio*100)}_{shard}.log').open('a')
                    command = [sys.executable, worker, 'eval', '--run-dir', str(run), '--method', method,
                               '--retention', str(ratio), '--shard', str(shard), '--shards', '2']
                    process = subprocess.Popen(command, cwd=run/'source',
                        env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=handle, stderr=subprocess.STDOUT)
                    active.append((process, handle))
            status('evaluating', workers=[p.pid for p,h in active])
            while any(p.poll() is None for p,h in active):
                assert all(p.poll() in (None, 0) for p,h in active), 'Evaluation worker failed; inspect logs'
                time.sleep(10)
            assert all(p.returncode == 0 for p,h in active)
        finally:
            for p,h in active:
                if p.poll() is None:
                    p.terminate()
                p.wait()
                h.close()
        status('rescoring')
        report(run)
        status('complete', report=str(run/'RESULTS.md'))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--start-prepared', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/qwen35_pruning'/(
        'qwen35_4b_dart_divprune_5_20_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if args.report_only:
        report(run)
        return
    if not args.start_prepared:
        prepare(run, args.reference_run)
    print('RUN_DIR='+str(run), flush=True)
    if not args.prepare_only:
        execute(run)


if __name__ == '__main__':
    main()
