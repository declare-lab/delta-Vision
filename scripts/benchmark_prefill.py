"""Benchmark prefill speed for base VLMs and visual-delta adapters."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "artifacts/torch_compile_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def benchmark(fn: Callable[[], Any], *, warmup: int, n_runs: int) -> float:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, n_runs)):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / max(1, n_runs)


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def maybe_compile(fn: Callable[[], Any], args: argparse.Namespace, *, enabled: bool) -> Callable[[], Any]:
    if not enabled:
        return fn
    return torch.compile(fn, mode=args.compile_mode, dynamic=args.compile_dynamic)


def read_jsonl_sample(path: str | Path, sample_index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index={sample_index} is out of range for {path}")


def read_jsonl_samples(path: str | Path, start_index: int, sample_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip() or idx < start_index:
                continue
            rows.append(json.loads(line))
            if len(rows) >= sample_count:
                break
    if not rows:
        raise IndexError(f"no samples starting at sample_index={start_index} in {path}")
    return rows


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def rough_qwen_length_key(row: dict[str, Any]) -> int:
    return len(str(row.get("question", ""))) + len(str(row.get("image", ""))) // 8


def checkpoint_label(path: Path) -> str:
    if path.parent.name == "checkpoints":
        return path.parent.parent.name
    return path.stem


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return label.strip(), Path(raw_path).expanduser()
    path = Path(spec).expanduser()
    return checkpoint_label(path), path


def collect_checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for spec in args.checkpoint:
        specs.append(parse_checkpoint_spec(spec))
    for pattern in args.checkpoint_glob:
        for match in sorted(glob.glob(pattern)):
            path = Path(match).expanduser()
            specs.append((checkpoint_label(path), path))
    if args.auto_qwen_checkpoints:
        pattern = (
            Path(args.qwen_run_root).expanduser()
            / "*"
            / "checkpoints"
            / "qwen_visual_delta_step500.pt"
        )
        for match in sorted(glob.glob(str(pattern))):
            path = Path(match)
            specs.append((checkpoint_label(path), path))

    deduped: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for label, path in specs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, path))
    return deduped


def fmt_ms(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds * 1000.0:.2f}"


def fmt_speedup(reference_s: float, value_s: float | None) -> str:
    return "" if value_s is None else f"{reference_s / value_s:.2f}x"


def fmt_delta_ms(reference_s: float, value_s: float | None) -> str:
    if value_s is None:
        return ""
    delta = (value_s - reference_s) * 1000.0
    return f"{delta:+.2f}"


def count_active_qwen_params(adapter: torch.nn.Module) -> int:
    total = 0
    mode = getattr(adapter, "mode", "")
    for name, param in adapter.named_parameters():
        active = name.startswith("visual_adapter_")
        if mode == "native_visual_kv_split":
            active = active or name == "gate" or name.startswith("reader_") or name.startswith("mass_head")
        if active:
            total += param.numel()
    return total


def count_total_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def build_qwen_benchmark_delta_fn(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    *,
    args: argparse.Namespace,
    logits_to_keep: int,
    qwen_visual_delta_logits: Callable[..., Any],
) -> Callable[[dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    def eager(inputs: dict[str, torch.Tensor], initial_hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            compact_no_padding=True,
            fast_split_text=bool(args.fast_split_text),
            fast_injection_prefix=bool(args.fast_injection_prefix),
            fast_prefix_kvcache=bool(args.fast_prefix_kvcache),
            logits_to_keep=logits_to_keep,
        )[0]

    if not args.compile:
        return eager

    def cached_forward(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        initial_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            compact_no_padding=True,
            fast_split_text=bool(args.fast_split_text),
            fast_injection_prefix=bool(args.fast_injection_prefix),
            fast_prefix_kvcache=bool(args.fast_prefix_kvcache),
            logits_to_keep=logits_to_keep,
        )[0]

    compiled_cached = torch.compile(cached_forward, mode=args.compile_mode, dynamic=args.compile_dynamic)

    def compiled(inputs: dict[str, torch.Tensor], initial_hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return compiled_cached(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
        )

    return compiled


def build_qwen_benchmark_e2e_fn(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    *,
    args: argparse.Namespace,
    logits_to_keep: int,
    build_qwen_initial_context: Callable[..., Any],
    qwen_visual_delta_logits: Callable[..., Any],
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    def e2e_forward(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
        return qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            compact_no_padding=True,
            fast_split_text=bool(args.fast_split_text),
            fast_injection_prefix=bool(args.fast_injection_prefix),
            fast_prefix_kvcache=bool(args.fast_prefix_kvcache),
            logits_to_keep=logits_to_keep,
        )[0]

    compiled_e2e = (
        torch.compile(e2e_forward, mode=args.compile_mode, dynamic=args.compile_dynamic)
        if args.compile_e2e
        else e2e_forward
    )

    def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return compiled_e2e(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["mm_token_type_ids"],
        )

    return run


@torch.inference_mode()
def verify_qwen_fast_path(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    initial_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    logits_to_keep: int,
    qwen_visual_delta_logits: Callable[..., Any],
    fast_split_text: bool,
    atol: float,
) -> dict[str, Any]:
    dense_logits, dense_mask, _ = qwen_visual_delta_logits(
        model,
        adapter,
        dict(inputs),
        initial_hidden=initial_hidden,
        position_ids=position_ids,
        collect_states=False,
        compact_no_padding=True,
        fast_split_text=fast_split_text,
        fast_injection_prefix=False,
        logits_to_keep=logits_to_keep,
    )
    fast_logits, fast_mask, _ = qwen_visual_delta_logits(
        model,
        adapter,
        dict(inputs),
        initial_hidden=initial_hidden,
        position_ids=position_ids,
        collect_states=False,
        compact_no_padding=True,
        fast_split_text=fast_split_text,
        fast_injection_prefix=True,
        logits_to_keep=logits_to_keep,
    )
    diff = (dense_logits.float() - fast_logits.float()).abs()
    dense_next = dense_logits[0, -1] if dense_logits.shape[1] == 1 else dense_logits[0, int(dense_mask[0].sum().item()) - 1]
    fast_next = fast_logits[0, -1] if fast_logits.shape[1] == 1 else fast_logits[0, int(fast_mask[0].sum().item()) - 1]
    result = {
        "fast_path_max_abs_diff": float(diff.max().item()),
        "fast_path_mean_abs_diff": float(diff.mean().item()),
        "fast_path_dense_argmax": int(torch.argmax(dense_next).item()),
        "fast_path_fast_argmax": int(torch.argmax(fast_next).item()),
    }
    if result["fast_path_dense_argmax"] != result["fast_path_fast_argmax"] or result["fast_path_max_abs_diff"] > float(atol):
        raise RuntimeError(
            "fast injection prefix path failed equivalence check: "
            f"max_abs_diff={result['fast_path_max_abs_diff']:.6g}, "
            f"dense_argmax={result['fast_path_dense_argmax']}, "
            f"fast_argmax={result['fast_path_fast_argmax']}"
        )
    return result


def _qwen_next_logits(logits: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return logits[0, -1]
    return logits[0, int(text_mask[0].sum().item()) - 1]


@torch.inference_mode()
def verify_qwen_decode_cache(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    initial_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    logits_to_keep: int,
    qwen_visual_delta_logits: Callable[..., Any],
    qwen_native_injection_prefill_cache: Callable[..., Any],
    qwen_native_injection_decode_logits: Callable[..., Any],
    steps: int,
    atol: float,
    fast_prefix_mask: bool,
) -> dict[str, Any]:
    full_ids = inputs["input_ids"].clone()
    full_mask = inputs["attention_mask"].clone()
    full_mm_ids = inputs["mm_token_type_ids"].clone()
    full_hidden = initial_hidden.clone()
    full_position_ids = position_ids.clone()
    token_embeddings = model.model.get_input_embeddings()

    full_logits, full_text_mask, _ = qwen_visual_delta_logits(
        model,
        adapter,
        dict(inputs),
        initial_hidden=full_hidden,
        position_ids=full_position_ids,
        collect_states=False,
        compact_no_padding=True,
        fast_injection_prefix=False,
        logits_to_keep=logits_to_keep,
    )
    cache_logits, cache_text_mask, decode_cache = qwen_native_injection_prefill_cache(
        model,
        adapter,
        dict(inputs),
        initial_hidden=initial_hidden,
        position_ids=position_ids,
        compact_no_padding=True,
        logits_to_keep=logits_to_keep,
        fast_prefix_mask=fast_prefix_mask,
        fast_prefix_kvcache=False,
    )

    max_diff = float((full_logits.float() - cache_logits.float()).abs().max().item())
    mean_diff = float((full_logits.float() - cache_logits.float()).abs().mean().item())
    full_argmax = int(torch.argmax(_qwen_next_logits(full_logits, full_text_mask)).item())
    cache_argmax = int(torch.argmax(_qwen_next_logits(cache_logits, cache_text_mask)).item())
    if full_argmax != cache_argmax or max_diff > float(atol):
        raise RuntimeError(
            "decode cache prefill failed equivalence check: "
            f"max_abs_diff={max_diff:.6g}, full_argmax={full_argmax}, cache_argmax={cache_argmax}"
        )

    last_pos_idx = full_mask.long().sum(dim=1).sub(1).view(1, -1, 1).expand(full_position_ids.shape[0], -1, 1)
    token_position_ids = full_position_ids.gather(2, last_pos_idx)
    checked_steps = 0
    for _ in range(max(0, steps)):
        next_token = int(torch.argmax(_qwen_next_logits(cache_logits, cache_text_mask)).item())
        token = torch.tensor([[next_token]], dtype=full_ids.dtype, device=full_ids.device)
        full_ids = torch.cat([full_ids, token], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(token)], dim=1)
        full_mm_ids = torch.cat([full_mm_ids, torch.zeros_like(token)], dim=1)
        token_position_ids = token_position_ids + 1
        full_position_ids = torch.cat([full_position_ids, token_position_ids], dim=2)
        full_hidden = torch.cat(
            [full_hidden, token_embeddings(token).to(device=full_hidden.device, dtype=full_hidden.dtype)],
            dim=1,
        )
        full_inputs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "mm_token_type_ids": full_mm_ids,
        }
        full_logits, full_text_mask, _ = qwen_visual_delta_logits(
            model,
            adapter,
            full_inputs,
            initial_hidden=full_hidden,
            position_ids=full_position_ids,
            collect_states=False,
            compact_no_padding=True,
            fast_injection_prefix=False,
            logits_to_keep=1,
        )
        cache_logits = qwen_native_injection_decode_logits(
            model,
            token_embeddings(token).to(device=full_hidden.device, dtype=full_hidden.dtype),
            token_position_ids,
            decode_cache,
        )
        cache_text_mask = torch.ones((1, 1), dtype=torch.bool, device=cache_logits.device)
        step_diff = (full_logits.float() - cache_logits.float()).abs()
        max_diff = max(max_diff, float(step_diff.max().item()))
        mean_diff = max(mean_diff, float(step_diff.mean().item()))
        full_argmax = int(torch.argmax(_qwen_next_logits(full_logits, full_text_mask)).item())
        cache_argmax = int(torch.argmax(_qwen_next_logits(cache_logits, cache_text_mask)).item())
        checked_steps += 1
        if full_argmax != cache_argmax or max_diff > float(atol):
            raise RuntimeError(
                "decode cache failed equivalence check: "
                f"step={checked_steps}, max_abs_diff={max_diff:.6g}, "
                f"full_argmax={full_argmax}, cache_argmax={cache_argmax}"
            )

    return {
        "decode_cache_verify_steps": checked_steps,
        "decode_cache_max_abs_diff": max_diff,
        "decode_cache_mean_abs_diff": mean_diff,
    }


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def print_qwen_results(
    *,
    model_path: str,
    sample_path: str,
    sample_index: int,
    visual_tokens: int,
    text_tokens: int,
    teacher_s: float,
    v0_s: float | None,
    last_logits_only: bool,
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("=== Qwen3-VL Prefill Benchmark ===")
    print(f"model={model_path}")
    print(f"sample={sample_path} index={sample_index}")
    print(f"visual_tokens={visual_tokens} text_tokens={text_tokens}")
    teacher_label = "base_teacher_next_logits" if last_logits_only else "base_teacher_full"
    print(f"{teacher_label}={fmt_ms(teacher_s)} ms (1.00x)")
    if v0_s is not None:
        print(f"qwen_v0_build={fmt_ms(v0_s)} ms ({fmt_speedup(teacher_s, v0_s)})")
    print()
    compare_header = f"{'path':52} {'ms':>9} {'vs base':>8} {'delta ms':>9} {'note'}"
    print("Base comparison (same run)")
    print(compare_header)
    print("-" * len(compare_header))
    print(f"{'base Qwen3-VL full prefill'[:52]:52} {fmt_ms(teacher_s):>9} {'1.00x':>8} {'+0.00':>9} official")
    if v0_s is not None:
        print(
            f"{'qwen_v0_build component'[:52]:52} {fmt_ms(v0_s):>9} "
            f"{fmt_speedup(teacher_s, v0_s):>8} {fmt_delta_ms(teacher_s, v0_s):>9} vision/context only"
        )
    for row in rows:
        if row["kind"] != "adapter":
            continue
        print(
            f"{(str(row['name']) + ' complete e2e')[:52]:52} {fmt_ms(row['e2e_s']):>9} "
            f"{fmt_speedup(teacher_s, row['e2e_s']):>8} {fmt_delta_ms(teacher_s, row['e2e_s']):>9} full method"
        )
        print(
            f"{(str(row['name']) + ' cached adapter')[:52]:52} {fmt_ms(row['cached_s']):>9} "
            f"{fmt_speedup(teacher_s, row['cached_s']):>8} {fmt_delta_ms(teacher_s, row['cached_s']):>9} diagnostic only"
        )
    print()
    header = (
        f"{'name':48} {'mode':27} {'active/total M':>18} {'ckpt GB':>8} "
        f"{'e2e ms':>9} {'e2e':>7} {'cached ms':>10} {'cached':>8} "
        f"{'decodeKV ms':>11} {'decodeKV':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["kind"] != "adapter":
            continue
        params = f"{row['active_params_m']:.2f}/{row['total_params_m']:.2f}"
        print(
            f"{row['name'][:48]:48} {row['mode'][:27]:27} {params:>18} "
            f"{row['checkpoint_gb']:8.2f} {fmt_ms(row['e2e_s']):>9} "
            f"{fmt_speedup(teacher_s, row['e2e_s']):>7} {fmt_ms(row['cached_s']):>10} "
            f"{fmt_speedup(teacher_s, row['cached_s']):>8} {fmt_ms(row.get('decode_cache_s')):>11} "
            f"{fmt_speedup(teacher_s, row.get('decode_cache_s')):>8}"
        )


def run_qwen3vl(args: argparse.Namespace) -> None:
    from src.model import (
        build_qwen_initial_context,
        load_or_build_qwen_initial_context,
        load_frozen_qwen3vl,
        load_qwen_visual_delta_checkpoint,
        prepare_qwen3vl_batch_inputs,
        qwen_native_injection_prefill_cache,
        qwen_visual_delta_logits,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)

    row = read_jsonl_sample(args.sample_jsonl, args.sample_index)
    inputs, _, _, image_paths = prepare_qwen3vl_batch_inputs(
        processor,
        [row],
        Path(args.data_root).expanduser(),
        device,
        include_answers=False,
    )
    attention_mask = inputs["attention_mask"].bool()
    mm_ids = inputs["mm_token_type_ids"]
    visual_tokens = int(((mm_ids == 1) & attention_mask).sum().item())
    text_tokens = int(((mm_ids == 0) & attention_mask).sum().item())
    logits_to_keep = 1 if args.last_logits_only else 0

    teacher_fn = maybe_compile(
        lambda: model(**inputs, logits_to_keep=logits_to_keep).logits,
        args,
        enabled=bool(args.compile_teacher),
    )
    teacher_s = benchmark(teacher_fn, warmup=args.warmup, n_runs=args.n_runs)

    v0_s: float | None = None
    if not args.skip_v0:
        v0_fn = maybe_compile(
            lambda: build_qwen_initial_context(model, dict(inputs)),
            args,
            enabled=bool(args.compile_teacher),
        )
        v0_s = benchmark(v0_fn, warmup=args.warmup, n_runs=args.n_runs)

    with torch.inference_mode():
        initial_hidden, position_ids = load_or_build_qwen_initial_context(
            model,
            dict(inputs),
            cache_dir=args.context_cache_dir or None,
            dtype=dtype,
        )

    result_rows: list[dict[str, Any]] = [
        {
            "kind": "base",
            "name": "base_teacher_next_logits" if args.last_logits_only else "base_teacher_full",
            "model_path": args.model_path,
            "sample_jsonl": args.sample_jsonl,
            "sample_index": args.sample_index,
            "image": image_paths[0] if image_paths else "",
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "last_logits_only": bool(args.last_logits_only),
            "seconds": teacher_s,
            "ms": teacher_s * 1000.0,
        }
    ]
    if v0_s is not None:
        result_rows.append(
            {
                "kind": "base",
                "name": "qwen_v0_build",
                "seconds": v0_s,
                "ms": v0_s * 1000.0,
                "speedup_vs_teacher": teacher_s / v0_s,
            }
        )

    checkpoint_specs = collect_checkpoint_specs(args)
    if not checkpoint_specs:
        print_qwen_results(
            model_path=args.model_path,
            sample_path=args.sample_jsonl,
            sample_index=args.sample_index,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            teacher_s=teacher_s,
            v0_s=v0_s,
            last_logits_only=bool(args.last_logits_only),
            rows=result_rows,
        )
        write_outputs(result_rows, args)
        return

    for label, checkpoint in checkpoint_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        print(f"Loading checkpoint: {label} -> {checkpoint}", flush=True)
        adapter, meta = load_qwen_visual_delta_checkpoint(
            checkpoint,
            model.model.language_model,
            device,
            dtype,
        )
        mode = str(meta.get("args", {}).get("output_mode", getattr(adapter, "mode", "")))
        total_params = count_total_params(adapter)
        active_params = count_active_qwen_params(adapter)
        checkpoint_gb = checkpoint.stat().st_size / (1024.0**3)
        fast_verify: dict[str, Any] | None = None
        if args.fast_injection_prefix and args.verify_fast_path and mode == "native_visual_kv_injection":
            fast_verify = verify_qwen_fast_path(
                model,
                adapter,
                dict(inputs),
                initial_hidden,
                position_ids,
                logits_to_keep=logits_to_keep,
                qwen_visual_delta_logits=qwen_visual_delta_logits,
                fast_split_text=bool(args.fast_split_text),
                atol=float(args.fast_path_atol),
            )
            print(
                "fast_path_verified "
                f"max_abs={fast_verify['fast_path_max_abs_diff']:.6g} "
                f"mean_abs={fast_verify['fast_path_mean_abs_diff']:.6g}",
                flush=True,
            )
        adapter_forward = build_qwen_benchmark_delta_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            qwen_visual_delta_logits=qwen_visual_delta_logits,
        )
        e2e_forward = build_qwen_benchmark_e2e_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            build_qwen_initial_context=build_qwen_initial_context,
            qwen_visual_delta_logits=qwen_visual_delta_logits,
        )

        e2e_s: float | None = None
        if not args.skip_e2e:
            def e2e_fn() -> torch.Tensor:
                return e2e_forward(dict(inputs))

            e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)

        def cached_fn() -> torch.Tensor:
            return adapter_forward(dict(inputs), initial_hidden, position_ids)

        cached_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)
        decode_cache_s: float | None = None
        if mode == "native_visual_kv_injection" and args.run_decode_cache and not args.skip_decode_cache:
            decode_verify: dict[str, Any] | None = None
            if args.verify_decode_cache:
                from src.model import qwen_native_injection_decode_logits

                decode_verify = verify_qwen_decode_cache(
                    model,
                    adapter,
                    dict(inputs),
                    initial_hidden,
                    position_ids,
                    logits_to_keep=logits_to_keep,
                    qwen_visual_delta_logits=qwen_visual_delta_logits,
                    qwen_native_injection_prefill_cache=qwen_native_injection_prefill_cache,
                    qwen_native_injection_decode_logits=qwen_native_injection_decode_logits,
                    steps=int(args.decode_verify_steps),
                    atol=float(args.decode_cache_atol),
                    fast_prefix_mask=bool(args.fast_injection_prefix),
                )
                print(
                    "decode_cache_verified "
                    f"steps={decode_verify['decode_cache_verify_steps']} "
                    f"max_abs={decode_verify['decode_cache_max_abs_diff']:.6g} "
                    f"mean_abs={decode_verify['decode_cache_mean_abs_diff']:.6g}",
                    flush=True,
                )
            decode_cache_fn = maybe_compile(
                lambda: qwen_native_injection_prefill_cache(
                    model,
                    adapter,
                    dict(inputs),
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    compact_no_padding=True,
                    logits_to_keep=logits_to_keep,
                    fast_prefix_mask=bool(args.fast_injection_prefix),
                    fast_prefix_kvcache=bool(args.fast_prefix_kvcache),
                )[0],
                args,
                enabled=bool(args.compile_decode_cache),
            )
            decode_cache_s = benchmark(decode_cache_fn, warmup=args.warmup, n_runs=args.n_runs)
        else:
            decode_verify = None

        result_rows.append(
            {
                "kind": "adapter",
                "name": label,
                "checkpoint": str(checkpoint),
                "checkpoint_gb": checkpoint_gb,
                "mode": mode,
                "active_params_m": active_params / 1_000_000.0,
                "total_params_m": total_params / 1_000_000.0,
                "last_logits_only": bool(args.last_logits_only),
                "e2e_s": e2e_s,
                "e2e_ms": None if e2e_s is None else e2e_s * 1000.0,
                "e2e_speedup_vs_teacher": None if e2e_s is None else teacher_s / e2e_s,
                "cached_s": cached_s,
                "cached_ms": cached_s * 1000.0,
                "cached_speedup_vs_teacher": teacher_s / cached_s,
                "decode_cache_s": decode_cache_s,
                "decode_cache_ms": None if decode_cache_s is None else decode_cache_s * 1000.0,
                "decode_cache_speedup_vs_teacher": None if decode_cache_s is None else teacher_s / decode_cache_s,
                **(fast_verify or {}),
                **(decode_verify or {}),
            }
        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_qwen_results(
        model_path=args.model_path,
        sample_path=args.sample_jsonl,
        sample_index=args.sample_index,
        visual_tokens=visual_tokens,
        text_tokens=text_tokens,
        teacher_s=teacher_s,
        v0_s=v0_s,
        last_logits_only=bool(args.last_logits_only),
        rows=result_rows,
    )
    write_outputs(result_rows, args)


def run_qwen3vl_batch_prefill(args: argparse.Namespace) -> None:
    from src.model import (
        build_qwen_initial_context,
        load_or_build_qwen_initial_context,
        load_frozen_qwen3vl,
        load_qwen_visual_delta_checkpoint,
        prepare_qwen3vl_batch_inputs,
        qwen_visual_delta_logits,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    rows = read_jsonl_samples(args.sample_jsonl, args.sample_index, int(args.sample_count))
    if args.bucket_by_length:
        rows = sorted(rows, key=rough_qwen_length_key)
    batches = chunked(rows, int(args.batch_size))
    logits_to_keep = 1 if args.last_logits_only else 0

    checkpoint_specs = collect_checkpoint_specs(args)
    adapters: list[tuple[str, Path, torch.nn.Module, dict[str, Any]]] = []
    adapter_forwards: dict[str, Callable[[dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    for label, checkpoint in checkpoint_specs:
        adapter, meta = load_qwen_visual_delta_checkpoint(checkpoint, model.model.language_model, device, dtype)
        adapters.append((label, checkpoint, adapter, meta))
        adapter_forwards[label] = build_qwen_benchmark_delta_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            qwen_visual_delta_logits=qwen_visual_delta_logits,
        )

    rows_out: list[dict[str, Any]] = []
    teacher_batch_seconds: list[float] = []
    teacher_sample_seconds: list[float] = []
    adapter_batch_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}
    adapter_sample_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}

    for batch_idx, batch_rows in enumerate(batches):
        inputs, _, _, image_paths = prepare_qwen3vl_batch_inputs(
            processor,
            batch_rows,
            Path(args.data_root).expanduser(),
            device,
            include_answers=False,
        )
        attention_mask = inputs["attention_mask"].bool()
        mm_ids = inputs["mm_token_type_ids"]
        visual_tokens = int(((mm_ids == 1) & attention_mask).sum().item())
        text_tokens = int(((mm_ids == 0) & attention_mask).sum().item())

        teacher_fn = maybe_compile(
            lambda: model(**inputs, logits_to_keep=logits_to_keep).logits,
            args,
            enabled=bool(args.compile_teacher),
        )
        teacher_s = benchmark(teacher_fn, warmup=args.warmup, n_runs=args.n_runs)
        teacher_batch_seconds.append(teacher_s)
        teacher_sample_seconds.append(teacher_s / max(1, len(batch_rows)))
        rows_out.append(
            {
                "kind": "batch_base",
                "name": "base_teacher_next_logits" if args.last_logits_only else "base_teacher_full",
                "batch_index": batch_idx,
                "batch_size": len(batch_rows),
                "visual_tokens": visual_tokens,
                "text_tokens": text_tokens,
                "seconds": teacher_s,
                "seconds_per_sample": teacher_s / max(1, len(batch_rows)),
                "images": image_paths,
            }
        )

        with torch.inference_mode():
            initial_hidden, position_ids = load_or_build_qwen_initial_context(
                model,
                dict(inputs),
                cache_dir=args.context_cache_dir or None,
                dtype=dtype,
            )

        for label, checkpoint, adapter, meta in adapters:
            mode = str(meta.get("args", {}).get("output_mode", getattr(adapter, "mode", "")))

            def cached_fn() -> torch.Tensor:
                return adapter_forwards[label](dict(inputs), initial_hidden, position_ids)

            adapter_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)
            adapter_batch_seconds[label].append(adapter_s)
            adapter_sample_seconds[label].append(adapter_s / max(1, len(batch_rows)))
            rows_out.append(
                {
                    "kind": "batch_adapter",
                    "name": label,
                    "checkpoint": str(checkpoint),
                    "mode": mode,
                    "batch_index": batch_idx,
                    "batch_size": len(batch_rows),
                    "visual_tokens": visual_tokens,
                    "text_tokens": text_tokens,
                    "seconds": adapter_s,
                    "seconds_per_sample": adapter_s / max(1, len(batch_rows)),
                    "speedup_vs_teacher": teacher_s / adapter_s,
                }
            )

    print()
    print("=== Qwen3-VL Batched Prefill Benchmark ===")
    print(f"model={args.model_path}")
    print(f"samples={len(rows)} batches={len(batches)} batch_size={args.batch_size} bucket_by_length={bool(args.bucket_by_length)}")
    teacher_avg = sum(teacher_batch_seconds) / max(1, len(teacher_batch_seconds))
    teacher_sample_avg = sum(teacher_sample_seconds) / max(1, len(teacher_sample_seconds))
    print(f"teacher_batch_avg={fmt_ms(teacher_avg)} ms teacher_sample_avg={fmt_ms(teacher_sample_avg)} ms")
    for label, _, _, _ in adapters:
        batch_avg = sum(adapter_batch_seconds[label]) / max(1, len(adapter_batch_seconds[label]))
        sample_avg = sum(adapter_sample_seconds[label]) / max(1, len(adapter_sample_seconds[label]))
        print(
            f"{label}: adapter_batch_avg={fmt_ms(batch_avg)} ms "
            f"adapter_sample_avg={fmt_ms(sample_avg)} ms speedup_vs_teacher_batch={fmt_speedup(teacher_avg, batch_avg)}"
        )

    write_outputs(rows_out, args)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-kind", choices=("auto", "qwen", "llava"), default="auto")
    parser.add_argument("--checkpoint", action="append", default=[], help="Adapter checkpoint. Can repeat. Use name=/path to label.")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob for adapter checkpoints. Can repeat.")
    parser.add_argument("--auto-qwen-checkpoints", action="store_true", help="Benchmark all step500 Qwen checkpoints under --qwen-run-root.")
    parser.add_argument("--qwen-run-root", default="artifacts/experiments/qwen_topk1024_freezeqkv")
    parser.add_argument("--sample-jsonl", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
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
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-split-text", action="store_true", help="Experimental split speed path; default off because logits drift slightly.")
    parser.add_argument("--fast-injection-prefix", action="store_true", help="Experimental native-injection prefix mask path; default off because logits drift slightly.")
    parser.add_argument(
        "--fast-prefix-kvcache",
        action="store_true",
        help="Use flash_attn_with_kvcache for native-injection prefix. Experimental and not enabled by --fast-injection-prefix.",
    )
    parser.add_argument("--verify-fast-path", action=argparse.BooleanOptionalAction, default=True, help="Compare fast prefix path against dense reference before benchmarking.")
    parser.add_argument("--fast-path-atol", type=float, default=1e-2, help="Max absolute logit tolerance for --verify-fast-path.")
    parser.add_argument("--last-logits-only", action=argparse.BooleanOptionalAction, default=True, help="Only compute logits for the next-token position.")
    parser.add_argument("--context-cache-dir", default="", help="Optional cache for Qwen initial_hidden/position_ids after vision encoder/merger.")
    parser.add_argument("--skip-e2e", action="store_true", help="Only benchmark cached adapter path.")
    parser.add_argument("--run-decode-cache", action="store_true", help="Run experimental native-injection decode-cache prefill benchmark.")
    parser.add_argument("--skip-decode-cache", action="store_true", help="Deprecated alias to keep decode-cache benchmark disabled.")
    parser.add_argument("--verify-decode-cache", action=argparse.BooleanOptionalAction, default=True, help="Compare native-injection decode cache against full forward before timing it.")
    parser.add_argument("--decode-verify-steps", type=int, default=2)
    parser.add_argument("--decode-cache-atol", type=float, default=1e-2)
    parser.add_argument("--skip-v0", action="store_true", help="Skip Qwen V0 source build timing.")
    parser.add_argument(
        "--compile-decode-cache",
        action="store_true",
        help="Try torch.compile on native-injection decode-cache prefill. Experimental; this path may compile very slowly.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()
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
        if int(args.sample_count) > 1 or int(args.batch_size) > 1:
            run_qwen3vl_batch_prefill(args)
        else:
            run_qwen3vl(args)
    elif model_kind == "llava":
        run_llava(args)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")


if __name__ == "__main__":
    main()
