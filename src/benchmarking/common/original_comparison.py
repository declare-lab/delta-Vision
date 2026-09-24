"""Orchestrate the existing speed entry points without replacing their decoders."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from src.benchmarking.common.comparison import parse_methods


def run_comparison(args):
    from src.benchmarking.common.prefill import collect_checkpoint_specs, _resolve_repo_or_data_path
    from src.benchmarks import get_benchmark_spec

    root = Path(__file__).resolve().parents[3]
    methods = parse_methods(args.compare_methods)
    # Match the adapter's default execution optimizations. Otherwise the default
    # table compares graph replay against Python-dispatched native forwards.
    if args.native_cuda_graphs is None:
        args.native_cuda_graphs = args.attn_implementation in ('auto', 'flash_attention_2') and bool(args.cuda_graph)
    if args.optimize_attention_metadata is None:
        args.optimize_attention_metadata = args.attn_implementation in ('auto', 'flash_attention_2')
    if args.context_cache_dir or args.skip_e2e:
        raise ValueError("Original entry-point protocol requires fresh vision on every request")
    if args.sample_index or args.eval_batch_size != 1 or args.batch_size != 1:
        raise ValueError("Original comparison uses the first metric-samples dataset rows, batch size 1")
    if args.answer_instruction is not None or args.compile or args.compile_teacher or args.compile_e2e:
        raise ValueError("Original comparison preserves the original prompts and uncompiled execution")
    if not args.adapter_decode_cache or not args.last_logits_only:
        raise ValueError("Original comparison requires adapter decode cache and last-logits-only")
    if args.metric_samples < 1 or any(not 0 < r <= 1 for r in args.retentions):
        raise ValueError("Invalid sample count or retention")
    spec = get_benchmark_spec(args.benchmark)
    data = _resolve_repo_or_data_path(args.sample_jsonl or spec.default_data, args.data_root).resolve()
    checkpoint_specs = collect_checkpoint_specs(args)
    if "embedding_adapter" in methods and len(checkpoint_specs) != 1:
        raise ValueError("Specify exactly one adapter checkpoint")
    output = Path(args.output_json or "test/results/original_comparison/table.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.parent / (output.stem + "_raw")
    raw.mkdir(parents=True, exist_ok=True)
    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else spec.max_new_tokens
    attention = "flash_attention_2" if args.attn_implementation == "auto" else args.attn_implementation
    common = ["--model-path", args.model_path, "--benchmark", spec.name,
              "--dtype", args.dtype, "--device", args.device,
              "--attn-implementation", attention, "--max-new-tokens", str(max_tokens),
              "--seed", str(args.seed), "--log-every", str(args.log_every)]
    protocol = {
        "protocol": "original_entry_points", "model_path": args.model_path,
        "data_path": str(data), "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "requested_samples": args.metric_samples, "max_new_tokens": max_tokens,
        "dtype": args.dtype, "attention": attention, "device": args.device,
        "adapter_prefill": "original metric-table context + prefill-cache CUDA graphs",
        "adapter_decode_cache_mode": args.adapter_decode_cache_mode,
        "adapter_fast_fa2_decode": "native Qwen decoder; owned adapter KV constructed once inside graph prefill; masked prefixes use manual FA2 fallback",
        "deepstack": args.comparison_deepstack,
        "cuda_graph": args.cuda_graph, "cuda_graph_context": args.cuda_graph_context,
        "structured_answer_early_stop": args.structured_answer_early_stop,
        "measure_decode_steps": bool(args.measure_decode_steps),
        "adapter_exact_optimizations": bool(getattr(args, 'adapter_exact_optimizations', False)),
        "native_cuda_graphs": bool(args.native_cuda_graphs),
        "optimize_attention_metadata": bool(args.optimize_attention_metadata),
        "generation": "base/pruning: original HF generate EOS; adapter: original metric-table structured stop",
        "timing": "sum of original per-sample measured seconds; warmup/capture/preprocessing excluded",
        "speedup": "paired base total / method total; paired base prefill / method prefill",
        "kv": "observed adapter prefill K/V when available, with legacy estimate in analytic_kv_cache_mb; full cache includes metadata",
        "peak_memory": "dataset maximum of per-request warmed CUDA allocated peak in MiB; includes model weights and retained graph pools; excludes other processes; reserved separately",
        "flops": "original analytic decoder prefill core, 2 FLOPs/MAC; excludes vision/selectors/head/norm/softmax",
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "commands": [],
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in (
            "src/benchmarking/common/prefill.py", "src/evaluate.py", "src/model.py", "src/benchmarks.py",
            "src/benchmarking/common/original_comparison.py", "src/benchmarking/common/generation_timing.py",
            "src/attention.py", "src/graphs.py", "src/attention.py", "src/graphs.py", "src/model_setup.py",
            "src/graphs.py", "src/graphs.py",
            "src/benchmarking/common/peak_memory.py",
            "baselines/visionzip/qwen3_vl/modeling_qwen3_vl_visionzip.py", "baselines/eval_baselines.py")},
    }
    import torch
    import transformers
    protocol.update(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(args.device))
    protocol_path = output.with_suffix(".protocol.json")

    def run(command, name):
        log = raw / (name + ".log")
        protocol["commands"].append({"argv": command, "shell": shlex.join(command), "log": str(log)})
        protocol_path.write_text(json.dumps(protocol, indent=2))
        print(f"Running {name}: {shlex.join(command)}\nLog: {log}", flush=True)
        with log.open("w") as handle:
            subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)

    rows = []

    def save():
        output.write_text(json.dumps(rows, indent=2))
        csv_path = Path(args.output_csv).resolve() if args.output_csv else output.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# Qwen3-VL-4B · MMStar · original entry points", "",
                 f"DeepStack: {args.comparison_deepstack}. Times sum all samples. Adapter KV is observed; FLOPs are mean analytic decoder prefill FLOPs.",
                 "Each group uses its own measured base. Adapter and baseline retain their original stopping rules.", "",
                 "Peak Memory is the maximum warmed CUDA allocated peak across requests (MiB), including weights and retained graphs.", "",
                 "| Group | Method | Retention | Total s | Prefill s | KV MiB | Peak Memory MiB | FLOPs | Total speedup | Prefill speedup |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            peak = f"{row['peak_memory_mb']:.2f}" if row.get('peak_memory_mb') is not None else 'N/A'
            lines.append(f"| {row['reference_group']} | {row['method']} | {row.get('retention', '—')} | "
                         f"{row['total_time_s']:.4f} | {row['prefilling_time_s']:.4f} | {row['kv_cache_mb']:.2f} | "
                         f"{peak} | {row['flops']:.4e} | {row['speedup_total']:.4f} | {row['speedup_prefilling']:.4f} |")
        if args.measure_decode_steps:
            lines += ["", "## Direct generation stages", "",
                      "Baseline prefill below is the first forward inside generate; the table above retains standalone prefill.",
                      "Decode counts actual cached forwards. Zero steps means ms/step is N/A. Total includes remaining generation overhead.", "",
                      "| Group | Method | Prefill s | Decode forward s | Overhead s | Decode steps | Decode ms/step | Actual KV MiB |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                steps = row.get("decode_steps")
                if steps is None:
                    continue
                adapter_row = row["method"] == "embedding_adapter"
                prefill = row["prefilling_time_s"] if adapter_row else row["generation_prefill_time_s"]
                decode = row["decode_forward_time_s"] if adapter_row else row["decode_time_s"]
                per_step = f"{1000 * decode / steps:.3f}" if steps else "N/A"
                lines.append(f"| {row['reference_group']} | {row['method']} | {prefill:.4f} | {decode:.4f} | "
                             f"{row['generation_overhead_s']:.4f} | {steps} | {per_step} | {row['actual_prefill_kv_cache_mb']:.2f} |")
        output.with_suffix(".md").write_text("\n".join(lines) + "\n")

    if "embedding_adapter" in methods:
        checkpoint = Path(checkpoint_specs[0][1]).resolve()
        protocol.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        adapter_output = raw / "adapter.json"
        command = [sys.executable, "-m", "src.benchmarking.common.prefill", *common,
                   "--metric-table", "--checkpoint", str(checkpoint), "--sample-jsonl", str(data),
                   "--data-root", str(data.parent), "--metric-samples", str(args.metric_samples),
                   "--metric-prefill-warmup", str(args.metric_prefill_warmup),
                   "--adapter-decode-cache-mode", args.adapter_decode_cache_mode,
                   "--comparison-deepstack", args.comparison_deepstack,
                   "--cuda-graph-warmup", str(args.cuda_graph_warmup),
                   "--compile-max-diff", str(args.compile_max_diff),
                   "--output-json", str(adapter_output)]
        for flag in ("cuda_graph", "cuda_graph_context", "compile_verify", "structured_answer_early_stop"):
            command.append("--" + ("" if getattr(args, flag) else "no-") + flag.replace("_", "-"))
        if args.measure_decode_steps:
            command.append("--measure-decode-steps")
        if args.optimize_attention_metadata:
            command.append("--optimize-attention-metadata")
        if getattr(args, 'adapter_exact_optimizations', False):
            command.append('--adapter-exact-optimizations')
        run(command, "adapter")
        detail = json.loads(adapter_output.with_suffix(".details.json").read_text())
        adapter_rows = json.loads(adapter_output.read_text())
        ref = adapter_rows[0]
        for row in adapter_rows:
            row.update(reference_group="adapter", samples=detail["summary"]["total_samples"],
                       reference_total_s=ref["total_time_s"], reference_prefill_s=ref["prefilling_time_s"],
                       source=str(adapter_output))
        rows.extend([r for r in adapter_rows if r["method"] == "embedding_adapter"] if args.native_cuda_graphs else adapter_rows)
        save()

    pruning = [m for m in methods if m not in ("embedding_adapter", "base")]
    budgets = list(dict.fromkeys(args.retentions)) if pruning else ([1.0] if args.native_cuda_graphs or not rows else [])
    for retention in budgets:
        group = f"baseline_r{retention:g}"
        out_dir = raw / group
        command = [sys.executable, "-m", "baselines.eval_baselines", *common,
                   "--deepstack", args.comparison_deepstack,
                   "--method", ",".join(["base", *pruning]), "--retention", str(retention),
                   "--data", str(data), "--data-root", str(data.parent),
                   "--max-samples", str(args.metric_samples), "--measure-prefill",
                   "--speed-warmup", str(args.metric_prefill_warmup), "--output-dir", str(out_dir)]
        if args.measure_decode_steps:
            command.append("--measure-decode")
        command.append('--measure-peak-memory')
        if args.native_cuda_graphs:
            command.append("--native-cuda-graphs")
        if args.optimize_attention_metadata:
            command.append("--optimize-attention-metadata")
        run(command, group)
        summaries = {r["method"]: (r, p) for p in out_dir.rglob("results.json")
                     for r in [json.loads(p.read_text())]}
        ref = summaries["base"][0]
        for method in ["base", *pruning]:
            result, source = summaries[method]
            predictions = json.loads(source.with_name("predictions.json").read_text())
            rows.append(dict(method=method, reference_group=group, retention=1.0 if method == "base" else retention,
                             samples=len(predictions), total_time_s=result["total_time_s"],
                             prefilling_time_s=result["prefilling_time_s"], kv_cache_mb=result["kv_cache_mb"],
                             flops=result["flops"], score=result["score"],
                             speedup_total=result["speedup_total"], speedup_prefilling=result["speedup_prefilling"],
                             reference_total_s=ref["total_time_s"], reference_prefill_s=ref["prefilling_time_s"],
                             source=str(source)))
            rows[-1].update({key: result.get(key) for key in ('peak_memory_mb', 'peak_memory_mb_mean',
                'peak_reserved_mb', 'prefill_peak_memory_mb', 'decode_peak_memory_mb', 'peak_memory_samples')})
            if args.measure_decode_steps:
                rows[-1].update({key: result.get(key) for key in (
                    "generation_prefill_time_s", "decode_time_s", "generation_overhead_s", "decode_steps",
                    "generated_tokens", "decode_ms_per_step", "actual_prefill_kv_cache_mb")})
                rows[-1]["speedup_generation_prefill"] = ref["generation_prefill_time_s"] / result["generation_prefill_time_s"]
                rows[-1]["speedup_decode_per_step"] = ref["decode_ms_per_step"] / result["decode_ms_per_step"] if result["decode_ms_per_step"] else None
        save()
    if args.native_cuda_graphs:
        ref = next(row for row in rows if row["method"] == "base")
        for row in rows:
            if row["method"] == "embedding_adapter":
                row.update(reference_group=ref["reference_group"],
                    reference_total_s=ref["total_time_s"], reference_prefill_s=ref["prefilling_time_s"],
                    speedup_total=ref["total_time_s"] / row["total_time_s"],
                    speedup_prefilling=ref["prefilling_time_s"] / row["prefilling_time_s"])
        save()
    protocol["status"] = "complete"
    protocol_path.write_text(json.dumps(protocol, indent=2))
    print(output.with_suffix(".md").read_text(), flush=True)
