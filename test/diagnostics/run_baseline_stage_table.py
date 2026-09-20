"""Run the canonical baseline entry with corrected FA metadata and stage timing."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/prefill_decode_corrected_20260915"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="0")
    protocol = {"shared_gpu_with_training": True, "samples": 200, "generation": "native HF generate, EOS or max 8 tokens",
                "prefill": "directly timed first model forward inside generate",
                "decode": "directly timed subsequent cached one-token model forwards; no vision recomputation",
                "overhead": "generate total minus observed model forwards; includes token choice, EOS checks, instrumentation bookkeeping",
                "legacy_prefill": "separate standalone prefill retained as prefilling_time_s for comparison",
                "metadata_fix": "cache FA2 sequence metadata per pruned position tensor; preserve original varlen/dense kernel choice and rotary embeddings",
                "commands": [], "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                    "src/generation_timing.py", "src/qwen_attention_metadata.py", "baselines/eval_baselines.py",
                    "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"]}}
    for retention in [.05, .2]:
        output = OUT / f"ret{int(retention*100):02d}"
        command = [str(ROOT/".venv/bin/python"), "-m", "baselines.eval_baselines",
                   "--model-path", "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct",
                   "--method", "base,fastv,dart,divprune,zoo,sparsevlm,visionzip", "--retention", str(retention),
                   "--benchmark", "mmstar", "--data", str(ROOT/"data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"),
                   "--max-samples", "200", "--max-new-tokens", "8", "--measure-prefill", "--measure-decode",
                   "--optimize-attention-metadata", "--speed-warmup", "1", "--attn-implementation", "flash_attention_2",
                   "--dtype", "bfloat16", "--seed", "42", "--log-every", "25", "--output-dir", str(output)]
        protocol["commands"].append({"argv": command, "shell": shlex.join(command)})
        (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2))
        print("Running", retention, flush=True)
        with (OUT/f"ret{int(retention*100):02d}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    protocol["status"] = "complete"
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
