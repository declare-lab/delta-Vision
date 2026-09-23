"""Corrected seed44 evaluation: four pruning cases, then fresh native/adapter references."""
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
from src.qwen35_pruning import visual_budget

PARENT = ROOT/'artifacts/experiments/qwen35_pixmo/qwen35_4b_embedding128_pixmo2000_20260920_064827'
MANIFEST = ROOT/'artifacts/experiments/qwen35_pruning/random44_shared_eval_data_20260920/manifest.json'
METHODS = ['dart_5', 'dart_20', 'divprune_5', 'divprune_20', 'native', 'adapter']


def prepare(run, max_new_tokens=None):
    manifests = json.loads(MANIFEST.read_text())
    assert manifests['seed'] == 44 and manifests['sampling'] == 'uniform_without_replacement'
    config = json.loads((PARENT/'config.json').read_text())
    caps = {name: int(info['max_new_tokens'] if max_new_tokens is None else max_new_tokens)
            for name, info in config['evaluation'].items()}
    config.update(experiment='Corrected Qwen3.5 four pruning cases plus paired native/adapter',
        checkpoint_parent=str(PARENT), evaluation_sampling_seed=44,
        evaluation_sampling='uniform_without_replacement',
        evaluation_generation=dict(max_new_tokens=max_new_tokens, max_new_tokens_by_benchmark=caps,
            token_limit_reference=str(PARENT/'config.json') if max_new_tokens is None else 'explicit override',
            do_sample=False, unfinished_response='invalid_zero'),
        methods=METHODS, eval_shards=8)
    run.mkdir(parents=True, exist_ok=False)
    (run/'eval_data').mkdir()
    for name, selected in manifests['benchmarks'].items():
        dest = run/'eval_data'/f'{name}.jsonl'
        assert sha(selected['path']) == selected['sha256']
        shutil.copy2(selected['path'], dest)
        config['evaluation'][name].update(selected, path=str(dest),
            seed=44, sampling='uniform_without_replacement', max_new_tokens=caps[name])
    config['pruning'] = dict(dart_prune_before_layer=4, dart_ratio='post-pruning layers only',
                             divprune_prune_before_layer=0, divprune_ratio='all layers')
    (run/'checkpoints').mkdir()
    checkpoint = PARENT/'checkpoints/qwen35_embedding_adapter_step2000.pt'
    (run/'checkpoints'/checkpoint.name).symlink_to(checkpoint)
    dump(run/'config.json', config)
    names = ['src/__init__.py', 'src/qwen35_embedding.py', 'src/qwen35_experiment.py', 'src/qwen_deepstack.py',
        'src/qwen35_pruning.py', 'src/benchmarks.py', 'src/evaluation_sampling.py',
        'scripts/qwen35_worker.py', 'scripts/qwen35_pruning_worker.py',
        'scripts/queue_qwen35_random44_eval.py', 'configs/qwen35_adapter_requirements.txt',
        'scripts/resume_qwen35_eval_pipeline.py',
        'test/diagnostics/test_choice_scoring.py', 'test/diagnostics/test_evaluation_sampling.py',
        'test/diagnostics/test_qwen35_evaluation_stop.py']
    hashes = {}
    for name in names:
        path = run/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
        hashes[name] = sha(path)
    parent_plan = json.loads((PARENT/'plan.json').read_text())
    dump(run/'plan.json', dict(source_sha256=hashes, checkpoint_sha256=sha(checkpoint),
        model_files_sha256=parent_plan['model_files_sha256'], dependencies=parent_plan['dependencies'],
        shared_manifest=str(MANIFEST), methods=METHODS, predictions=6*8765,
        notes='No historical scores reused; all six methods generate fresh answers on shared seed44 data.'))


def report(run, complete=False):
    config = json.loads((run/'config.json').read_text())
    results, inputs = [], {}
    for method in ['native', 'adapter', 'dart_5', 'dart_20', 'divprune_5', 'divprune_20']:
        if method not in config['methods']:
            continue
        row = dict(method=method, unfinished={}, invalid={}, token_retention={})
        for name, info in config['evaluation'].items():
            predictions = sorted([json.loads(line) for path in (run/'eval'/method).glob(f'{name}.shard*.jsonl')
                for line in path.read_text().splitlines() if line], key=lambda p:p['index'])
            if len(predictions) != info['samples']:
                assert not complete, (method, name, len(predictions), info['samples'])
                continue
            assert [p['index'] for p in predictions] == list(range(info['samples']))
            data = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
            scores = []
            original_visual = retained_visual = summed_visual = 0
            for pred, question in zip(predictions, data):
                rescored = score_evaluation_prediction(pred, question, info['metric'])
                assert rescored['score'] == pred['score'] and rescored['invalid'] == pred['invalid']
                assert pred['max_new_tokens'] == info['max_new_tokens']
                assert pred['generated_tokens'] <= pred['max_new_tokens']
                assert len(pred['generated_token_ids']) == pred['generated_tokens']
                assert pred['stopped_by_eos'] or pred['hit_generation_limit']
                assert pred['stopped_by_eos'] or pred['score'] == 0
                scores.append(pred['score'])
                key = (name, pred['index'])
                actual = (pred['input_ids_sha256'], pred['image_grid_thw'])
                if key in inputs:
                    assert inputs[key] == actual, (method, key)
                else:
                    inputs[key] = actual
                if method not in ('native', 'adapter'):
                    audit = pred['token_audit']
                    ratio = int(method.rsplit('_', 1)[1])/100
                    v, k = audit['original_visual_tokens'], audit['retained_visual_tokens']
                    assert k == visual_budget(v, ratio)
                    layer = 4 if method.startswith('dart') else 0
                    assert audit['visual_tokens_per_layer'] == [v]*layer+[k]*(32-layer)
                    assert len(set(audit['selected_visual_indices'])) == k
                    original_visual += v
                    retained_visual += k
                    summed_visual += sum(audit['visual_tokens_per_layer'])
            row[name] = 100*sum(scores)/len(scores)
            row['unfinished'][name] = sum(not p['stopped_by_eos'] for p in predictions)
            row['invalid'][name] = sum(p['invalid'] for p in predictions)
            if original_visual:
                row['token_retention'][name] = dict(post_pruning=retained_visual/original_visual,
                                                    all_layers=summed_visual/(32*original_visual))
        if all(n in row for n in config['evaluation']):
            row['avg'] = sum(row[n] for n in config['evaluation'])/9
        results.append(row)
    dest = 'summary.json' if complete else 'partial_summary.json'
    dump(run/dest, results)
    names = list(config['evaluation'])+['avg']
    lines = ['# Qwen3.5-4B corrected evaluation', '',
        'Uniform random sampling without replacement, seed=44; same questions for all methods.',
        '1000 questions per benchmark; RealWorldQA 765. FA2 / FLA; no DeepStack; thinking disabled.',
        'Greedy decoding, max_new_tokens by benchmark: '+', '.join(
            f"{name}={info['max_new_tokens']}" for name, info in config['evaluation'].items())+
            '. EOS/length flags and raw token IDs saved.',
        'Unfinished responses count as invalid/zero; no question is dropped from the denominator.',
        'DART retention applies to layers 4–31 (layers 0–3 full); DivPrune to all 32 layers.',
        'MME/POPE: question accuracy. VQAv2: soft accuracy. AVG: mean of nine unrounded scores.', '',
        '| Method | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for row in results:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[n]:.2f}' if n in row else 'pending' for n in names)+' |')
    lines += ['', '## Responses reaching the length limit without EOS', '',
              '| Method | '+' | '.join(config['evaluation'])+' |', '|---|'+'---:|'*9]
    for row in results:
        lines.append('| '+row['method']+' | '+' | '.join(str(row['unfinished'].get(n, 'pending')) for n in config['evaluation'])+' |')
    if config.get('historical_reference_summary'):
        historical_path = Path(config['historical_reference_summary'])
        assert sha(historical_path) == config['historical_reference_sha256']
        historical = {row['method']: row for row in json.loads(historical_path.read_text())}
        lines += ['', '## New evaluation and historical reference', '',
            'Historical scores are preserved as originally reported: ordered subsets and the old scorer.',
            'New scores use seed44 random subsets and the corrected scorer, including zero for responses without EOS.',
            'Both use the original adapter generation limits (8/16 tokens); old and new scores are not a paired comparison.', '',
            '| Method | Run | '+' | '.join(names)+' |', '|---|---|'+'---:|'*len(names)]
        for row in results:
            for label, values in [('New', row), ('Historical reference', historical[row['method']])]:
                lines.append('| '+row['method']+' | '+label+' | '+
                    ' | '.join(f'{values[n]:.2f}' if n in values else 'pending' for n in names)+' |')
    (run/('RESULTS.md' if complete else 'PARTIAL_RESULTS.md')).write_text('\n'.join(lines)+'\n')


def execute(run):
    config = json.loads((run/'config.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(run/'source'/name) == digest, name
    logs = run/'logs'
    logs.mkdir(exist_ok=True)
    def status(state, **kwargs):
        dump(run/'status.json', dict(state=state, pid=os.getpid(), updated=time.time(), **kwargs))
    env = dict(os.environ, PYTHONPATH=str(ROOT/'artifacts/dependencies/qwen35_python')+os.pathsep+str(run/'source'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    def free_gpus():
        status('waiting_gpus')
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            if len(memory) == 8 and all(int(m) < 1024 for m in memory):
                return
            time.sleep(20)
    try:
        free_gpus()
        if not (run/'validation.json').exists():
            status('validating')
            with (logs/'validation.log').open('a') as handle:
                subprocess.run([sys.executable, str(run/'source/scripts/qwen35_pruning_worker.py'),
                    'validate', '--run-dir', str(run)], cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES='0'),
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
        assert json.loads((run/'validation.json').read_text())['passed']
        for method in config['methods']:
            marker = run/f'{method}.complete.json'
            if marker.exists():
                continue
            free_gpus()
            active = []
            try:
                for shard in range(8):
                    worker = 'qwen35_worker.py' if method in ('native','adapter') else 'qwen35_pruning_worker.py'
                    command = [sys.executable, str(run/'source/scripts'/worker), 'eval', '--run-dir', str(run),
                               '--shard', str(shard)]
                    if method in ('native','adapter'):
                        command += ['--method', method]
                    else:
                        algo, percent = method.rsplit('_',1)
                        command += ['--method', algo, '--retention', str(int(percent)/100), '--shards', '8']
                    handle = (logs/f'{method}_{shard}.log').open('a')
                    p = subprocess.Popen(command, cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES=str(shard)),
                                         stdout=handle, stderr=subprocess.STDOUT)
                    active.append((p,handle))
                status('evaluating', method=method, workers=[p.pid for p,h in active])
                while any(p.poll() is None for p,h in active):
                    assert all(p.poll() in (None,0) for p,h in active), 'Evaluation worker failed; inspect logs'
                    time.sleep(10)
                assert all(p.returncode==0 for p,h in active)
            finally:
                for p,h in active:
                    if p.poll() is None:
                        p.terminate()
                    p.wait()
                    h.close()
            report(run)
            dump(marker, dict(completed=time.time()))
        report(run, complete=True)
        status('complete', report=str(run/'RESULTS.md'))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=None,
                        help='Override all benchmarks; default matches original adapter per-benchmark limits.')
    parser.add_argument('--start-prepared',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    run=args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/qwen35_pruning'/(
        'qwen35_random44_corrected_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if args.report_only:
        configured_methods=json.loads((run/'config.json').read_text())['methods']
        report(run,complete=all((run/f'{m}.complete.json').exists() for m in configured_methods))
        return
    if not args.start_prepared:
        assert args.max_new_tokens is None or args.max_new_tokens > 0
        prepare(run, args.max_new_tokens)
    print('RUN_DIR='+str(run),flush=True)
    if not args.prepare_only:
        execute(run)


if __name__=='__main__':main()
