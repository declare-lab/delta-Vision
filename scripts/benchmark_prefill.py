"""Minimal prefill benchmark runner for base VLMs and visual-delta adapters."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path(__file__).resolve().parents[1] / "artifacts/torch_compile_cache"),
)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_benchmark_utils import (
    benchmark,
    collect_checkpoint_specs,
    configure_torch_runtime,
    dtype_from_name,
    fmt_ms,
    fmt_speedup,
    maybe_compile,
    read_jsonl_sample,
    run_qwen3vl,
    run_qwen3vl_batch_prefill,
    write_outputs,
)


def _resolve_repo_or_data_path(path: str, data_root: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    root_dir = Path(__file__).resolve().parents[1]
    repo_path = root_dir / raw
    if repo_path.exists():
        return repo_path
    return Path(data_root).expanduser() / raw


def _fmt_flops(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}G"
    return f"{value:.0f}"


def _fmt_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _metric_value(summary: dict[str, Any], side: str, metric: str) -> float | None:
    values = summary.get(side) or {}
    if metric == "pope_f1":
        return values.get("f1")
    return values.get("score")


def run_qwen_metric_table(args: argparse.Namespace) -> None:
    from src.benchmarks import get_benchmark_spec
    from src.data import QwenBenchmarkDataset
    from src.eval_benchmarks import build_qwen_adapter_logits_fn, evaluate_qwen_benchmark_shard
    from src.model import load_frozen_qwen3vl, load_qwen_visual_delta_checkpoint

    checkpoint_specs = collect_checkpoint_specs(args)
    if len(checkpoint_specs) != 1:
        raise ValueError("--metric-table expects exactly one --checkpoint for the adapter-only row")

    spec = get_benchmark_spec(args.benchmark)
    data_path = _resolve_repo_or_data_path(args.sample_jsonl or spec.default_data, args.data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"benchmark data does not exist: {data_path}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    label, checkpoint = checkpoint_specs[0]
    adapter, meta = load_qwen_visual_delta_checkpoint(checkpoint, model.model.language_model, device, dtype)
    if meta["missing"] or meta["unexpected"]:
        print(f"checkpoint load missing={meta['missing']} unexpected={meta['unexpected']}", flush=True)

    adapter_logits_fn = build_qwen_adapter_logits_fn(
        model,
        adapter,
        compile_adapter=bool(args.compile),
        compile_mode=args.compile_mode,
        compile_dynamic=bool(args.compile_dynamic),
        last_logits_only=bool(args.last_logits_only),
    )
    dataset = QwenBenchmarkDataset(
        str(data_path),
        processor,
        spec.name,
        data_root=args.data_root,
        max_samples=args.metric_samples,
        answer_instruction=args.answer_instruction,
        cache_dir=args.input_cache_dir or None,
    )
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else spec.max_new_tokens
    result = evaluate_qwen_benchmark_shard(
        model,
        processor,
        adapter,
        dataset,
        device,
        log_every=max(1, int(args.log_every)),
        max_new_tokens=max_new_tokens,
        benchmark=spec.name,
        measure_prefill=True,
        dtype=dtype,
        adapter_logits_fn=adapter_logits_fn,
        compile_warmup=bool(args.compile_warmup),
        context_cache_dir=args.context_cache_dir or None,
        structured_answer_early_stop=bool(args.structured_answer_early_stop),
        last_logits_only=bool(args.last_logits_only),
    )
    summary = result["summary"]
    timing = summary["timing"]
    resources = summary["resources"]
    rows = [
        {
            "method": "qwen_base",
            "benchmark": spec.name,
            "samples": int(args.metric_samples),
            "total_time_s": timing["teacher_total_s"],
            "total_time": timing["teacher_total_minsec"],
            "prefilling_time_s": timing["teacher_prefill_s"],
            "prefilling_time": timing["teacher_prefill_minsec"],
            "flops": resources["teacher_prefill_flops_avg"],
            "kv_cache_mb": resources["teacher_kv_cache_mb_avg"],
            "score": _metric_value(summary, "teacher", spec.metric),
            "speedup_total": 1.0,
            "speedup_prefilling": 1.0,
        },
        {
            "method": "adapter_only",
            "checkpoint": str(checkpoint),
            "checkpoint_label": label,
            "mode": result.get("output_mode"),
            "benchmark": spec.name,
            "samples": int(args.metric_samples),
            "total_time_s": timing["adapter_total_s"],
            "total_time": timing["adapter_total_minsec"],
            "prefilling_time_s": timing["adapter_prefill_s"],
            "prefilling_time": timing["adapter_prefill_minsec"],
            "flops": resources["adapter_prefill_flops_avg"],
            "kv_cache_mb": resources["adapter_kv_cache_mb_avg"],
            "score": _metric_value(summary, "adapter", spec.metric),
            "speedup_total": timing["speedup_total"],
            "speedup_prefilling": timing["speedup_prefill"],
        },
    ]

    print()
    print("=== Qwen Base vs Adapter-Only Metric Table ===")
    print(f"benchmark={spec.display_name} metric={spec.metric} samples={args.metric_samples} max_new_tokens={max_new_tokens}")
    print(f"data={data_path}")
    print(f"checkpoint={checkpoint}")
    header = (
        f"{'method':14} {'Total Time':>12} {'Prefilling':>12} {'FLOPs':>10} "
        f"{'KV Cache MB':>12} {'score/F1':>9} {'Speedup T':>10} {'Speedup P':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method'][:14]:14} {row['total_time']:>12} {row['prefilling_time']:>12} "
            f"{_fmt_flops(row['flops']):>10} {row['kv_cache_mb']:12.2f} {_fmt_score(row['score']):>9} "
            f"{fmt_speedup(1.0, 1.0 / row['speedup_total']) if row['speedup_total'] else '':>10} "
            f"{fmt_speedup(1.0, 1.0 / row['speedup_prefilling']) if row['speedup_prefilling'] else '':>10}"
        )
    write_outputs(rows, args)


def run_llava(args: argparse.Namespace) -> None:
    from src.model import (
        extract_vision_kv,
        load_adapter_checkpoint,
        load_frozen_llava,
        student_forward_with_visual_kv,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_llava(args.model_path, dtype=dtype, device=device)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    row = read_jsonl_sample(args.sample_jsonl, args.sample_index)
    image_path = Path(str(row["image"]))
    if not image_path.is_absolute():
        image_path = Path(args.data_root) / image_path
    with Image.open(image_path) as image:
        img = image.convert("RGB")
    prompt = "USER: <image>\n" + str(row["question"]).strip() + "\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)

    checkpoint_specs = collect_checkpoint_specs(args)
    adapter = None
    checkpoint_label_value = "random"
    if checkpoint_specs:
        checkpoint_label_value, checkpoint = checkpoint_specs[0]
        adapter, _, _ = load_adapter_checkpoint(checkpoint, device, language_model=model.model.language_model)
    else:
        raise ValueError("LLaVA benchmark requires --checkpoint")

    with torch.inference_mode():
        source_k, source_v = extract_vision_kv(model, inputs.pixel_values, [22, 23])

    teacher_fn = maybe_compile(
        lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values).logits,
        args,
        enabled=bool(args.compile_teacher),
    )
    e2e_fn = maybe_compile(
        lambda: student_forward_with_visual_kv(
            model,
            inputs.input_ids,
            adapter,
            *extract_vision_kv(model, inputs.pixel_values, [22, 23]),
            image_token_id,
        ),
        args,
        enabled=bool(args.compile),
    )
    cached_fn = maybe_compile(
        lambda: student_forward_with_visual_kv(model, inputs.input_ids, adapter, source_k, source_v, image_token_id),
        args,
        enabled=bool(args.compile),
    )
    teacher_s = benchmark(teacher_fn, warmup=args.warmup, n_runs=args.n_runs)
    e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)
    cached_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)

    n_vis = int(source_k.shape[2])
    n_text = int((inputs.input_ids[0] != image_token_id).sum().item())
    rows = [
        {
            "kind": "base",
            "name": "base_teacher_full",
            "model_path": args.model_path,
            "visual_tokens": n_vis,
            "text_tokens": n_text,
            "seconds": teacher_s,
            "ms": teacher_s * 1000.0,
        },
        {
            "kind": "adapter",
            "name": checkpoint_label_value,
            "e2e_s": e2e_s,
            "e2e_ms": e2e_s * 1000.0,
            "e2e_speedup_vs_teacher": teacher_s / e2e_s,
            "cached_s": cached_s,
            "cached_ms": cached_s * 1000.0,
            "cached_speedup_vs_teacher": teacher_s / cached_s,
        },
    ]
    print()
    print(f"=== LLaVA Prefill Benchmark ({args.n_runs} runs) ===")
    print(f"visual_tokens={n_vis} text_tokens={n_text}")
    print(f"base_teacher_full={fmt_ms(teacher_s)} ms")
    print(f"{checkpoint_label_value} e2e={fmt_ms(e2e_s)} ms ({fmt_speedup(teacher_s, e2e_s)})")
    print(f"{checkpoint_label_value} cached={fmt_ms(cached_s)} ms ({fmt_speedup(teacher_s, cached_s)})")
    write_outputs(rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-kind", choices=("auto", "qwen", "llava"), default="auto")
    parser.add_argument("--metric-table", action="store_true", help="Run a small base-vs-adapter metric table from benchmark samples.")
    parser.add_argument("--benchmark", default="pope", help="Benchmark name for --metric-table. Defaults to POPE for F1.")
    parser.add_argument("--metric-samples", type=int, default=10, help="Number of benchmark samples for --metric-table.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override metric-table generation length.")
    parser.add_argument("--answer-instruction", default=None, help="Override benchmark answer instruction for --metric-table.")
    parser.add_argument("--input-cache-dir", default="", help="Optional cache for processed benchmark inputs in --metric-table.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint", action="append", default=[], help="Adapter checkpoint. Can repeat. Use name=/path to label.")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob for adapter checkpoints. Can repeat.")
    parser.add_argument("--auto-qwen-checkpoints", action="store_true", help="Benchmark all step500 Qwen checkpoints under --qwen-run-root.")
    parser.add_argument("--qwen-run-root", default="artifacts/experiments/qwen_topk1024_freezeqkv")
    parser.add_argument("--sample-jsonl", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1, help="Number of samples for batched Qwen prefill benchmarking.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for batched Qwen prefill benchmarking.")
    parser.add_argument("--bucket-by-length", action=argparse.BooleanOptionalAction, default=True, help="Sort batched samples by a rough length key before batching.")
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile for adapter measured functions. Use --no-compile for quick debugging.",
    )
    parser.add_argument(
        "--compile-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also compile base teacher/v0 forwards. This is slow for HF Qwen3-VL and disabled by default.",
    )
    parser.add_argument(
        "--compile-e2e",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the full Qwen adapter e2e tensor forward, including vision context build.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--last-logits-only", action=argparse.BooleanOptionalAction, default=True, help="Only compute logits for the next-token position.")
    parser.add_argument("--context-cache-dir", default="", help="Optional cache for Qwen initial_hidden/position_ids after vision encoder/merger.")
    parser.add_argument("--structured-answer-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-e2e", action="store_true", help="Only benchmark cached adapter path.")
    parser.add_argument("--skip-v0", action="store_true", help="Skip Qwen V0 source build timing.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_runtime()

    model_kind = args.model_kind
    if model_kind == "auto":
        lower = args.model_path.lower()
        if "qwen" in lower:
            model_kind = "qwen"
        elif "llava" in lower:
            model_kind = "llava"
        else:
            raise ValueError("Could not infer model kind from --model-path; set --model-kind qwen or llava")

    if model_kind == "qwen":
        if args.metric_table:
            run_qwen_metric_table(args)
        elif int(args.sample_count) > 1 or int(args.batch_size) > 1:
            args.sample_jsonl = args.sample_jsonl or "../delta-vision/data/mmstar/mmstar_val.jsonl"
            run_qwen3vl_batch_prefill(args)
        else:
            args.sample_jsonl = args.sample_jsonl or "../delta-vision/data/mmstar/mmstar_val.jsonl"
            run_qwen3vl(args)
    elif model_kind == "llava":
        args.sample_jsonl = args.sample_jsonl or "../delta-vision/data/mmstar/mmstar_val.jsonl"
        run_llava(args)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")


if __name__ == "__main__":
    main()
