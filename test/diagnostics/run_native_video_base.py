"""Launch the fixed native FA2 base on eight GPU shards and retain provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmark_adapter_optimizations import dump, file_sha


def main():
    output = ROOT/'test/results/video_base_native_fa2_20260915/videomme999'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('native_*.jsonl')):
        raise FileExistsError(output)
    sources = ['src/benchmark_video_base.py', 'src/generation_timing.py', 'baselines/eval_baselines.py',
        'src/qwen_deepstack.py', 'src/data.py', 'src/benchmarks.py', 'src/video_benchmark_inputs.py',
        'src/benchmark_video_sampling.py', 'src/benchmark_comparison.py']
    hashes = {p:file_sha(ROOT/p) for p in sources}
    dump(output/'source_start.json', hashes)
    running = {}
    try:
        for shard in range(8):
            command = [sys.executable, '-m', 'src.benchmark_video_base', '--variant', 'native',
                '--shard', str(shard), '--shards', '8', '--runs', '3', '--tokens', '8',
                '--input-cache', 'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme',
                '--output', str(output)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
            with (output/f'native_{shard}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[shard] = process
            print('START', shard, process.pid, flush=True)
        while running:
            for shard, process in list(running.items()):
                if process.poll() is None:
                    continue
                if process.returncode:
                    raise RuntimeError((output/f'native_{shard}.log').read_text()[-3500:])
                print('DONE', shard, flush=True)
                del running[shard]
            time.sleep(2)
    finally:
        for process in running.values(): process.terminate()
        for process in running.values(): process.wait()
    changed = [p for p,h in hashes.items() if file_sha(ROOT/p) != h]
    dump(output/'source_validation.json', dict(all_sources_unchanged=not changed, changed=changed))
    assert not changed, changed
    subprocess.run([sys.executable, '-m', 'src.benchmark_video_base', '--aggregate', '--variant', 'native',
        '--runs', '3', '--tokens', '8', '--output', str(output)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
