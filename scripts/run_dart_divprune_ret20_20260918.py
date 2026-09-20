"""Eight-GPU, resumable 5-model x 2-method x 9-benchmark evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarks import get_benchmark_spec

OUT = ROOT / 'artifacts/eval/dart_divprune_ret20_5models_9bench_20260918'
BENCHMARKS = 'mmstar,gqa,mmb,mmb-cn,mme,pope,sqa,vqav2,realworldqa'.split(',')
# Start the larger models first; single GPU per process fits H200 memory.
MODELS = ['qwen3-vl-30b-a3b', 'llava-v1.6-mistral-7b-hf', 'llava-1.5-13b-hf',
          'qwen3-vl-8b', 'llava-1.5-7b-hf']
METHODS = ['dart', 'divprune']
SOURCES = ['baselines/eval_baselines.py', 'baselines/llava_hf_baselines.py', 'src/model.py',
           'src/data.py', 'src/benchmarks.py', 'baselines/dart/qwen3_vl/modeling_qwen3_vl_dart.py',
           'baselines/divprune/qwen3_vl/modeling_qwen3_vl_divprune.py']


class AttachedProcess:
    """Reconnect the scheduler to an existing evaluator without restarting it."""
    def __init__(self, info):
        self.pid = info['pid']
        self.model = info['model']
        self.method = info['method']

    def poll(self):
        path = Path(f'/proc/{self.pid}/cmdline')
        try:
            cmd = path.read_bytes().decode().split('\0')
        except FileNotFoundError:
            return 0
        return None if self.model in cmd and self.method in cmd and 'baselines/eval_baselines.py' in cmd else 0

    def terminate(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def result_path(stage, model, method, benchmark):
    return OUT / stage / model / method / 'ret20' / benchmark / 'results.json'


def complete(stage, model, method, benchmark, plan):
    path = result_path(stage, model, method, benchmark)
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    n = min(2, plan['data'][benchmark]['samples']) if stage == 'smoke' else plan['data'][benchmark]['samples']
    return (result['samples'] == n and result['retention'] == .2
            and result['method'] == method and result['model_label'] == model)


def report(plan):
    rows = []
    for model in MODELS:
        for method in METHODS:
            for benchmark in BENCHMARKS:
                if not complete('full', model, method, benchmark, plan):
                    continue
                result = json.loads(result_path('full', model, method, benchmark).read_text())
                rows.append(dict(model=model, method=method, retention=.2, benchmark=benchmark,
                                 samples=result['samples'], metric=result['metric'], score=result['score'],
                                 pope_f1=result.get('f1'),
                                 mme_score=result.get('mme_score'), invalid_rate=result.get('invalid_rate')))
    if rows:
        with (OUT / 'summary.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# DART / DivPrune, retention 20%', '',
             'FA2, BF16, Qwen DeepStack off; first up to 1000 examples per benchmark.',
             'Cells: benchmark score x100, except MME reports the raw category-summed MME score.',
             'POPE uses F1; VQAv2 uses soft accuracy. Missing cells are pending.', '',
             '| Model | Method | ' + ' | '.join(BENCHMARKS) + ' |',
             '|---|---|' + '---:|' * len(BENCHMARKS)]
    for model in MODELS:
        for method in METHODS:
            cells = []
            for benchmark in BENCHMARKS:
                rr = [r for r in rows if (r['model'], r['method'], r['benchmark']) == (model, method, benchmark)]
                if not rr:
                    cells.append('—')
                else:
                    value = (rr[0]['mme_score'] if benchmark == 'mme' else
                             100 * rr[0]['pope_f1'] if benchmark == 'pope' else 100 * rr[0]['score'])
                    cells.append(f'{value:.2f}')
            lines.append('| ' + ' | '.join([model, method, *cells]) + ' |')
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return len(rows)


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    data = {}
    for name in BENCHMARKS:
        path = ROOT / get_benchmark_spec(name).default_data
        raw = path.read_bytes()
        selected = [json.loads(s) for s in raw.splitlines() if s.strip()][:1000]
        assert selected
        for row in selected:
            base = Path(row.get('image_root') or path.parent)
            for image in row.get('images', [row.get('image')]):
                assert image is not None and (base / str(image)).is_file(), (name, image)
        data[name] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), samples=len(selected),
                          max_new_tokens=get_benchmark_spec(name).max_new_tokens)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    plan = dict(models=MODELS, methods=METHODS, benchmarks=BENCHMARKS, retention=.2,
                attention='flash_attention_2', qwen_deepstack='off', dtype='bfloat16', seed=42,
                max_samples=1000, gpus=list(range(8)), data=data, source_sha256=hashes,
                metrics='project benchmark-specific scoring; MME first1000 subset, not full-MME',
                source_root=str(ROOT), created=time.time())
    if (OUT / 'plan.json').exists():
        old = json.loads((OUT / 'plan.json').read_text())
        assert old['data'] == data, 'Evaluation data changed during resume'
        if old['source_sha256'] != hashes:
            write(OUT / f'plan_revision_{time.time_ns()}.json', plan)
    write(OUT / 'plan.json', plan)
    snapshot = OUT / 'source'
    for name in SOURCES:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((ROOT / name).read_bytes())
    return plan


def run_stage(stage, plan, status):
    active = {}
    old_path = OUT / 'status.json'
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if old.get('stage') == stage:
            for info in old.get('active', []):
                process = AttachedProcess(info)
                if process.poll() is None:
                    active[info['gpu']] = (process, open(os.devnull, 'w'), info)
                    print('ATTACHED', stage, info['model'], info['method'], process.pid, flush=True)
    pending = []
    for model in MODELS:
        for method in METHODS:
            missing = [b for b in BENCHMARKS if not complete(stage, model, method, b, plan)]
            already_running = any(v[2]['model'] == model and v[2]['method'] == method for v in active.values())
            if missing and not already_running:
                pending.append((model, method, missing))
    failed = []
    status.update(stage=stage, failed=[])
    try:
        while pending or active:
            for gpu in range(8):
                if gpu in active or not pending:
                    continue
                model, method, missing = pending.pop(0)
                cmd = [sys.executable, '-u', 'baselines/eval_baselines.py', '--model-label', model,
                       '--method', method, '--retention', '0.20', '--benchmark', ','.join(missing),
                       '--max-samples', '2' if stage == 'smoke' else '1000', '--attn-implementation',
                       'flash_attention_2', '--deepstack', 'off', '--dtype', 'bfloat16', '--seed', '42',
                       '--data-root', str(ROOT), '--output-dir', str(OUT / stage), '--log-every', '50']
                tag = f'{stage}_{model}_{method}_{time.time_ns()}'
                logpath = OUT / 'logs' / f'{tag}.log'
                handle = logpath.open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                info = dict(model=model, method=method, gpu=gpu, pid=process.pid, log=str(logpath),
                            command=cmd, started=time.time())
                write(OUT / 'logs' / f'{tag}.json', info)
                active[gpu] = (process, handle, info)
                print('START', stage, model, method, 'GPU', gpu, 'PID', process.pid, flush=True)
            for gpu, (process, handle, info) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                info.update(returncode=rc, finished=time.time())
                valid = rc == 0 and all(complete(stage, info['model'], info['method'], b, plan) for b in BENCHMARKS)
                if not valid:
                    failed.append(info)
                print('DONE' if valid else 'FAILED', stage, info['model'], info['method'], rc, flush=True)
                del active[gpu]
            status.update(active=[v[2] for v in active.values()], pending=len(pending), failed=failed,
                          completed_full_cells=report(plan), total_full_cells=90, updated=time.time())
            write(OUT / 'status.json', status)
            if pending or active:
                time.sleep(10)
        if failed:
            raise RuntimeError(f'{len(failed)} {stage} jobs failed; inspect status.json and logs')
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


if __name__ == '__main__':
    state = dict(state='running', started=time.time(), launcher_pid=os.getpid())
    try:
        plan = preflight()
        run_stage('smoke', plan, state)
        run_stage('full', plan, state)
        assert report(plan) == 90
        state.update(state='complete', finished=time.time())
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        write(OUT / 'status.json', state)
