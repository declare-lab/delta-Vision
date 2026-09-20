"""MMStar 200, native FA2 + graphs, DeepStack off for every method."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/deepstack_off_20260915/full"
MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DATA = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
CHECKPOINT = ROOT / "artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt"


def run(command, name, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    record = dict(argv=command, gpu=gpu, shared_with_training=True)
    path = OUT / f"{name}.command.json"
    path.write_text(json.dumps(record, indent=2))
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["exit_code"] = result.returncode
    path.write_text(json.dumps(record, indent=2))
    result.check_returncode()
    print(name, "complete", flush=True)


def native(gpu, methods, retention, name):
    run([sys.executable, "-m", "baselines.eval_baselines", "--model-path", MODEL,
        "--method", ",".join(methods), "--retention", str(retention), "--benchmark", "mmstar",
        "--data", str(DATA), "--data-root", str(DATA.parent), "--max-samples", "200",
        "--max-new-tokens", "8", "--measure-prefill", "--measure-decode", "--native-cuda-graphs",
        "--optimize-attention-metadata", "--speed-warmup", "1", "--attn-implementation", "flash_attention_2",
        "--deepstack", "off", "--dtype", "bfloat16", "--seed", "42", "--log-every", "25",
        "--output-dir", str(OUT / name)], name, gpu)


def baseline_worker(gpu, method):
    for retention in [.05, .2]:
        native(gpu, ["base", method], retention, f"{method}_ret{int(100*retention):02d}")


def adapter_worker():
    native(3, ["base"], 1., "adapter_reference")
    run([sys.executable, "-m", "src.benchmark_prefill", "--model-path", MODEL,
        "--checkpoint", str(CHECKPOINT), "--metric-table", "--benchmark", "mmstar",
        "--sample-jsonl", str(DATA), "--data-root", str(DATA.parent), "--metric-samples", "200",
        "--max-new-tokens", "8", "--cuda-graph", "--cuda-graph-context", "--adapter-decode-cache-mode", "fast",
        "--comparison-deepstack", "off", "--attn-implementation", "flash_attention_2", "--measure-decode-steps",
        "--metric-prefill-warmup", "1", "--seed", "42", "--log-every", "25",
        "--output-json", str(OUT / "adapter200.json")], "adapter200", 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ["src/model.py", "src/benchmark_prefill.py", "src/eval_benchmarks.py", "src/generation_timing.py",
        "src/qwen_deepstack.py", "src/qwen_adapter_fa2.py", "src/qwen_adapter_graph.py", "src/qwen_native_graph.py",
        "src/qwen_adapter_native_graph.py", "src/qwen_adapter_shared_graph.py",
        "src/qwen_attention_metadata.py", "baselines/eval_baselines.py", "baselines/multimodal_pruning_utils.py"]
    sources += [f"baselines/{m}/qwen3_vl/modeling_qwen3_vl_{m}.py" for m in ["fastv","dart","divprune","zoo","sparsevlm","visionzip"]]
    protocol = dict(samples=200, attention="flash_attention_2", deepstack="off: no vision side mergers or language injection",
        adapter="genuine fast decode; vision-context and prefill-cache graphs; whole decode graph",
        native="vision and prefill-layer graphs plus whole-forward decode graphs, with exact eager checks per input",
        stop="historical MMStar stop rules: native EOS/max8; adapter structured-answer/EOS/max8",
        reference="each method and retention has a same-GPU base run; adapter uses optimized native base on GPU3",
        limitations="concurrent training remains running; separate full-set runs are not alternating request pairs",
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(), checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        status="running")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    errors = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        jobs = [pool.submit(baseline_worker, gpu, method) for gpu, method in
            [(0,"fastv"),(1,"dart"),(2,"divprune"),(5,"zoo"),(6,"sparsevlm"),(7,"visionzip")]]
        jobs.append(pool.submit(adapter_worker))
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as error:
                errors.append(repr(error))
    protocol.update(status="failed" if errors else "complete", errors=errors)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
