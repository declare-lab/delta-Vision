"""Full MMStar 200 runs: FA2 throughout, warmed native graphs on base/pruning."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/all_fa2_20260915/full"
MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"


def run_gpu(gpu, methods):
    commands = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    for retention in [.05, .2]:
        name = f"gpu{gpu}_ret{int(retention * 100):02d}"
        command = [sys.executable, "-m", "baselines.eval_baselines", "--model-path", MODEL,
            "--method", ",".join(methods), "--retention", str(retention), "--benchmark", "mmstar",
            "--data", str(ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"),
            "--max-samples", "200", "--max-new-tokens", "8", "--measure-prefill", "--measure-decode",
            "--native-cuda-graphs", "--optimize-attention-metadata", "--speed-warmup", "1",
            "--attn-implementation", "flash_attention_2", "--dtype", "bfloat16", "--seed", "42",
            "--log-every", "25", "--output-dir", str(OUT / name)]
        commands.append(dict(argv=command, gpu=gpu, shared_gpu=True))
        (OUT / f"gpu{gpu}.commands.json").write_text(json.dumps(commands, indent=2))
        with (OUT / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name, "complete", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ["src/model.py", "src/qwen_native_graph.py", "src/qwen_attention_metadata.py",
        "src/generation_timing.py", "baselines/eval_baselines.py",
        "baselines/visionzip/qwen3_vl/modeling_qwen3_vl_visionzip.py",
        "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"]
    protocol = dict(samples=200, attention="flash_attention_2", shared_gpu=True,
        correctness="each sample compared with native eager: exact tokens, logits and KV",
        preparation="per-input untimed warmup/capture and validation, separately reported",
        decode="native generate, measured cached forwards, EOS or max 8 tokens",
        comparisons="cross-run full-set totals; interleaved same-GPU pairs are stored separately",
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(run_gpu, gpu, methods) for gpu, methods in
            [(0, ["fastv"]), (1, ["base"]), (2, ["dart"]), (4, ["divprune"]),
             (5, ["zoo"]), (6, ["sparsevlm"]), (7, ["visionzip"])]]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(repr(error))
    protocol.update(status="failed" if errors else "complete", errors=errors)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
