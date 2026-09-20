"""Collect warmed process memory peaks on all MMStar 200 inputs, one model/process."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test/results/deepstack_off_20260915/memory200'
FULL = OUT.parent / 'full'


def run(gpu, method, retention):
    name = 'base' if method == 'base' else f'{method}_ret{int(retention*100):02d}'
    template = json.loads((FULL / 'divprune_ret05.command.json').read_text())['argv']
    command = list(template)
    for flag, value in [('--method', method), ('--retention', str(retention)), ('--output-dir', str(OUT / name))]:
        command[command.index(flag)+1] = value
    command.append('--measure-peak-memory')
    record = dict(argv=command, gpu=gpu, status='running')
    path = OUT / (name + '.command.json')
    path.write_text(json.dumps(record, indent=2))
    with (OUT / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'))
    record.update(exit_code=result.returncode, status='complete' if result.returncode == 0 else 'failed')
    path.write_text(json.dumps(record, indent=2))
    result.check_returncode()
    print(name, 'complete', flush=True)


def worker(gpu, method):
    for retention in [.05, .2]:
        run(gpu, method, retention)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ['src/generation_timing.py', 'src/peak_memory.py', 'src/qwen_native_graph.py',
        'src/qwen_attention_metadata.py', 'src/qwen_deepstack.py', 'baselines/eval_baselines.py']
    sources += [f'baselines/{m}/qwen3_vl/modeling_qwen3_vl_{m}.py' for m in ['fastv','dart','divprune','zoo','sparsevlm','visionzip']]
    protocol = dict(samples=200, deepstack='off', attention='flash_attention_2', shared_with_training=True,
        metric='Maximum warmed CUDA allocated memory across all 200 generation requests; model weights and resident graph pools included; each method/retention in a fresh process; graph capture excluded; MiB',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, status='running')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    errors=[]
    with ThreadPoolExecutor(max_workers=7) as pool:
        jobs=[pool.submit(worker,gpu,m) for gpu,m in [(0,'fastv'),(1,'dart'),(2,'divprune'),(5,'zoo'),(6,'sparsevlm'),(7,'visionzip')]]
        jobs.append(pool.submit(run,4,'base',1.))
        for job in as_completed(jobs):
            try:job.result()
            except Exception as e:errors.append(repr(e))
    protocol.update(status='failed' if errors else 'complete', errors=errors)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if errors:raise RuntimeError(errors)


if __name__ == '__main__':main()
