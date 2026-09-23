"""Wait for the existing memory experiment, validate, then evaluate eight shards."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['src/qwen35_full_attention_ablation.py', 'scripts/qwen35_full_attention_ablation_worker.py',
           'scripts/queue_qwen35_full_attention_ablation.py', 'test/diagnostics/test_qwen35_full_attention_ablation.py',
           'src/qwen35_experiment.py', 'src/qwen35_embedding.py', 'src/qwen_deepstack.py', 'src/benchmarks.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def records(run, benchmark):
    rows = []
    for path in (run / 'accuracy').glob(f'{benchmark}.shard*.jsonl'):
        for line in path.read_bytes().splitlines(keepends=True):
            if line.endswith(b'\n'):
                rows.append(json.loads(line))
    assert len(rows) == len({r['index'] for r in rows})
    return rows


def report(run, config):
    scores = []
    condition = config.get('ablation_condition', 'block_two_fa')
    for benchmark, info in config['evaluation'].items():
        rows = records(run, benchmark)
        assert {r['index'] for r in rows} == set(range(info['samples']))
        grouped = defaultdict(list)
        for row in rows:
            assert row['layers'] == config['ablated_layers']
            assert len(row['variants']) == 2 * len(config['methods'])
            for v in row['variants']:
                grouped[v['method'], v['condition']].append(v['score'])
        for method in config['methods']:
            base = 100 * sum(grouped[method, 'unmodified']) / len(rows)
            blocked = 100 * sum(grouped[method, condition]) / len(rows)
            scores.append(dict(benchmark=benchmark, samples=len(rows), method=method,
                               unmodified=base, **{condition: blocked}, difference_pp=blocked-base))
    dump(run / 'summary.json', dict(layers=config['ablated_layers'], seed=config['layer_selection_seed'], scores=scores))
    lines = ['# Full-attention layers: text-to-visual access removed', '',
             f"Fixed zero-based layers: {config['ablated_layers']}. FA2; DeepStack off.",
             'Linear attention remains native. Full visual KV caches and visual query computation remain intact.',
             'Same prompt, manifests and scoring; max_new_tokens=8. Fresh paired controls.', '',
             '| Dataset | N | Model | Unmodified (%) | Selected FA visual access blocked (%) | Difference (pp) |',
             '|---|---:|---|---:|---:|---:|']
    for r in scores:
        lines.append(f"| {r['benchmark']} | {r['samples']} | {r['method']} | {r['unmodified']:.2f} | {r[condition]:.2f} | {r['difference_pp']:+.2f} |")
    (run / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    config = json.loads((run / 'config.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    status = dict(state='waiting', stage=None, dependency=config['wait_for_run'], started=time.time())
    running = []
    try:
        while True:
            dependency = json.loads((Path(config['wait_for_run']) / 'status.json').read_text())
            status['dependency_stage'] = dependency.get('stage')
            status['dependency_state'] = dependency['state']
            dump(run / 'status.json', status)
            if dependency['state'] == 'complete':
                break
            if dependency['state'] == 'failed':
                raise RuntimeError('Preceding memory experiment failed; not starting GPU work')
            time.sleep(20)
        for stage in ['validate', 'accuracy']:
            assert {name: sha(ROOT / name) for name in SOURCES} == plan['source_sha256'], 'Source changed while queued'
            assert sha(run / 'config.json') == plan['config_sha256']
            status.update(state='running', stage=stage)
            running = []
            for gpu in range(1 if stage == 'validate' else 8):
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
                log = (run / 'logs' / f'{stage}_{gpu}.log').open('a')
                proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u',
                    'scripts/qwen35_full_attention_ablation_worker.py', '--run', str(run),
                    '--stage', stage, '--shard', str(gpu), '--shards', '8'],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append((gpu, proc, log))
            while True:
                failures = [(gpu, proc.poll()) for gpu, proc, _ in running if proc.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(str(failures))
                status.update(workers=[dict(gpu=gpu, pid=proc.pid, returncode=proc.poll()) for gpu, proc, _ in running],
                              progress={b: len(records(run, b)) for b in config['evaluation']})
                dump(run / 'status.json', status)
                if all(proc.poll() is not None for _, proc, _ in running):
                    break
                time.sleep(10)
            for _, _, log in running:
                log.close()
            if stage == 'validate':
                for b in config['evaluation']:
                    rows = [json.loads(l) for l in (run / 'validate' / f'{b}.shard0.jsonl').read_text().splitlines()]
                    assert len(rows) == 1 and len(rows[0]['checks']) == len(config['methods'])
                    assert all(c['no_op_logits_exact'] and c['no_op_generation_exact'] for c in rows[0]['checks'])
        report(run, config)
        status.update(state='complete', stage=None, elapsed_s=time.time()-status['started'])
        dump(run / 'status.json', status)
    except BaseException as exc:
        for _, proc, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, log in running:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        status.update(state='failed', error=repr(exc))
        dump(run / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
