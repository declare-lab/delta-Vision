"""Benchmark prefill speed for base VLMs and visual-delta adapters."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("=== Qwen3-VL Prefill Benchmark ===")
    print(f"model={model_path}")
    print(f"sample={sample_path} index={sample_index}")
    print(f"visual_tokens={visual_tokens} text_tokens={text_tokens}")
    print(f"base_teacher_full={fmt_ms(teacher_s)} ms (1.00x)")
    if v0_s is not None:
        print(f"qwen_v0_build={fmt_ms(v0_s)} ms ({fmt_speedup(teacher_s, v0_s)})")
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

    teacher_fn = maybe_compile(lambda: model(**inputs).logits, args, enabled=bool(args.compile_teacher))
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
        initial_hidden, position_ids = build_qwen_initial_context(model, dict(inputs))

    result_rows: list[dict[str, Any]] = [
        {
            "kind": "base",
            "name": "base_teacher_full",
            "model_path": args.model_path,
            "sample_jsonl": args.sample_jsonl,
            "sample_index": args.sample_index,
            "image": image_paths[0] if image_paths else "",
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
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

        e2e_s: float | None = None
        if not args.skip_e2e:
            e2e_fn = maybe_compile(
                lambda: qwen_visual_delta_logits(
                    model,
                    adapter,
                    dict(inputs),
                    compact_no_padding=True,
                    fast_split_text=bool(args.fast_split_text),
                    fast_injection_prefix=bool(args.fast_injection_prefix),
                )[0],
                args,
                enabled=bool(args.compile),
            )
            e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)

        cached_fn = maybe_compile(
            lambda: qwen_visual_delta_logits(
                model,
                adapter,
                dict(inputs),
                initial_hidden=initial_hidden,
                position_ids=position_ids,
                compact_no_padding=True,
                fast_split_text=bool(args.fast_split_text),
                fast_injection_prefix=bool(args.fast_injection_prefix),
            )[0],
            args,
            enabled=bool(args.compile),
        )
        cached_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)
        decode_cache_s: float | None = None
        if mode == "native_visual_kv_injection" and not args.skip_decode_cache:
            decode_cache_fn = maybe_compile(
                lambda: qwen_native_injection_prefill_cache(
                    model,
                    adapter,
                    dict(inputs),
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    compact_no_padding=True,
                )[0],
                args,
                enabled=bool(args.compile_decode_cache),
            )
            decode_cache_s = benchmark(decode_cache_fn, warmup=args.warmup, n_runs=args.n_runs)

        result_rows.append(
            {
                "kind": "adapter",
                "name": label,
                "checkpoint": str(checkpoint),
                "checkpoint_gb": checkpoint_gb,
                "mode": mode,
                "active_params_m": active_params / 1_000_000.0,
                "total_params_m": total_params / 1_000_000.0,
                "e2e_s": e2e_s,
                "e2e_ms": None if e2e_s is None else e2e_s * 1000.0,
                "e2e_speedup_vs_teacher": None if e2e_s is None else teacher_s / e2e_s,
                "cached_s": cached_s,
                "cached_ms": cached_s * 1000.0,
                "cached_speedup_vs_teacher": teacher_s / cached_s,
                "decode_cache_s": decode_cache_s,
                "decode_cache_ms": None if decode_cache_s is None else decode_cache_s * 1000.0,
                "decode_cache_speedup_vs_teacher": None if decode_cache_s is None else teacher_s / decode_cache_s,
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
        rows=result_rows,
    )
    write_outputs(result_rows, args)


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
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-split-text", action="store_true", help="Experimental split speed path; default off because logits drift slightly.")
    parser.add_argument("--fast-injection-prefix", action="store_true", help="Experimental native-injection prefix mask path; default off because logits drift slightly.")
    parser.add_argument("--skip-e2e", action="store_true", help="Only benchmark cached adapter path.")
    parser.add_argument("--skip-decode-cache", action="store_true", help="Skip native-injection prefill that builds decode KV cache.")
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
        run_qwen3vl(args)
    elif model_kind == "llava":
        run_llava(args)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")


if __name__ == "__main__":
    main()
