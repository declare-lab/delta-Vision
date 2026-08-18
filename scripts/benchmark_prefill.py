"""Minimal prefill benchmark runner for base VLMs and visual-delta adapters."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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
    assert_same_logits,
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


@contextmanager
def sdpa_backend_context(name: str):
    if name == "default":
        with nullcontext():
            yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    mapping = {
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    with sdpa_kernel([mapping[name]]):
        yield


def cuda_graph_zero_arg(
    name: str,
    fn,
    *,
    warmup: int,
    verify: bool,
    max_diff: float,
):
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-graph requires CUDA")
    graph = None
    output = None
    verified = False

    def run():
        nonlocal graph, output, verified
        if graph is None:
            eager_out = fn()
            device = eager_out.device
            with torch.cuda.device(device):
                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream):
                    for _ in range(max(0, int(warmup))):
                        fn()
                torch.cuda.current_stream().wait_stream(side_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn()
                graph.replay()
            if verify and not verified:
                assert output is not None
                assert_same_logits(name, eager_out, output, max_diff)
                verified = True
        else:
            graph.replay()
        assert output is not None
        return output

    return run


def _graph_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    device = tensor.device
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, device.type, device.index)


def _graph_value_signature(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _graph_tensor_signature(value)
    if isinstance(value, tuple):
        return tuple(_graph_value_signature(item) for item in value)
    raise TypeError(f"unsupported CUDA graph value type: {type(value)!r}")


def _clone_graph_value(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return torch.empty_like(value).copy_(value)
    if isinstance(value, tuple):
        return tuple(_clone_graph_value(item) for item in value)
    raise TypeError(f"unsupported CUDA graph value type: {type(value)!r}")


def _copy_graph_value_(target: Any, source: Any) -> None:
    if target is None and source is None:
        return
    if torch.is_tensor(target) and torch.is_tensor(source):
        target.copy_(source)
        return
    if isinstance(target, tuple) and isinstance(source, tuple):
        if len(target) != len(source):
            raise RuntimeError("CUDA graph tuple length changed")
        for target_item, source_item in zip(target, source):
            _copy_graph_value_(target_item, source_item)
        return
    raise RuntimeError(f"CUDA graph value type changed: {type(target)!r} vs {type(source)!r}")


class PreparedCudaGraphRunner:
    def __init__(
        self,
        name: str,
        fn,
        *,
        warmup: int,
        verify: bool,
        max_diff: float,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-graph requires CUDA")
        self.name = name
        self.fn = fn
        self.warmup = int(max(0, warmup))
        self.verify = bool(verify)
        self.max_diff = float(max_diff)
        self.graph = None
        self.static_prepared: dict[str, Any] | None = None
        self.signature: Any = None
        self.output: torch.Tensor | None = None
        self.verified = False

    def _signature(self, prepared: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((key, _graph_value_signature(prepared[key])) for key in sorted(prepared))

    def _clone_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return {key: _clone_graph_value(value) for key, value in prepared.items()}

    def _copy_prepared_(self, prepared: dict[str, Any]) -> None:
        if self.static_prepared is None:
            raise RuntimeError("CUDA graph static tensors are not initialized")
        for key, value in prepared.items():
            _copy_graph_value_(self.static_prepared[key], value)

    def _static_forward(self) -> torch.Tensor:
        if self.static_prepared is None:
            raise RuntimeError("CUDA graph static tensors are not initialized")
        return self.fn(self.static_prepared)

    def _capture(self, prepared: dict[str, Any], signature: Any) -> None:
        eager_out = self.fn(prepared)
        device = eager_out.device
        self.static_prepared = self._clone_prepared(prepared)
        with torch.cuda.device(device):
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(self.warmup):
                    self._static_forward()
            torch.cuda.current_stream().wait_stream(side_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._static_forward()
            self.graph.replay()
        if self.verify and not self.verified:
            if self.output is None:
                raise RuntimeError("CUDA graph did not produce an output")
            assert_same_logits(self.name, eager_out, self.output, self.max_diff)
            self.verified = True
        self.signature = signature

    def __call__(self, prepared: dict[str, Any]) -> torch.Tensor:
        signature = self._signature(prepared)
        if self.graph is None or signature != self.signature:
            self._capture(prepared, signature)
        else:
            self._copy_prepared_(prepared)
            self.graph.replay()
        if self.output is None:
            raise RuntimeError("CUDA graph did not produce an output")
        return self.output


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
        compile_verify=bool(args.compile_verify),
        compile_max_diff=float(args.compile_max_diff),
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
        eval_batch_size=int(args.eval_batch_size),
        eval_max_batch_tokens=int(args.eval_max_batch_tokens),
        eval_bucket_by_length=bool(args.eval_bucket_by_length),
        verify_batched_generation=int(args.verify_batched_generation),
        adapter_decode_cache=bool(args.adapter_decode_cache),
        adapter_decode_cache_mode=str(args.adapter_decode_cache_mode),
        verify_decode_cache_generation=int(args.verify_decode_cache_generation),
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
        llava_projected_image_features,
        prepare_llava_adapter_only_inputs,
        prepare_llava_injection_inputs,
        student_forward_llava_injection,
        student_forward_llava_injection_prepared,
        student_forward_llava_injection_prepared_hf_attention,
        student_forward_with_visual_kv,
        student_forward_with_visual_kv_prepared,
        student_forward_with_visual_kv_prepared_hf_attention,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_llava(
        args.model_path,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
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
        adapter, _, metadata = load_adapter_checkpoint(checkpoint, device, language_model=model.model.language_model, dtype=dtype)
    else:
        raise ValueError("LLaVA benchmark requires --checkpoint")
    output_mode = args.output_mode or str(metadata.get("output_mode") or "adapter_only")
    if output_mode not in ("adapter_only", "native_visual_kv_injection"):
        raise ValueError(f"LLaVA benchmark only supports adapter_only/native_visual_kv_injection, got {output_mode!r}")
    if bool(args.cuda_graph) and bool(args.compile_e2e):
        raise RuntimeError("--cuda-graph and --compile-e2e are separate LLaVA e2e fast paths; enable only one")
    adapter_only_prepared_fn = (
        student_forward_with_visual_kv_prepared_hf_attention
        if bool(args.hf_attn_injection)
        else student_forward_with_visual_kv_prepared
    )
    injection_prepared_fn = (
        student_forward_llava_injection_prepared_hf_attention
        if bool(args.hf_attn_injection)
        else student_forward_llava_injection_prepared
    )

    attention_mask = inputs.get("attention_mask", None)
    with torch.inference_mode():
        if output_mode == "native_visual_kv_injection":
            visual_memory = llava_projected_image_features(model, inputs.pixel_values)
            source_k = source_v = None
        else:
            source_k, source_v = extract_vision_kv(model, inputs.pixel_values, [22, 23])
            visual_memory = None

    e2e_graph_runner = None
    if bool(args.cuda_graph):
        if output_mode == "native_visual_kv_injection":
            e2e_graph_runner = PreparedCudaGraphRunner(
                "llava_native_visual_kv_e2e_cuda_graph",
                lambda prepared: injection_prepared_fn(model, adapter, **prepared),
                warmup=int(args.cuda_graph_warmup),
                verify=bool(args.compile_verify),
                max_diff=float(args.compile_max_diff),
            )
        else:
            e2e_graph_runner = PreparedCudaGraphRunner(
                "llava_adapter_only_e2e_cuda_graph",
                lambda prepared: adapter_only_prepared_fn(model, adapter, **prepared),
                warmup=int(args.cuda_graph_warmup),
                verify=bool(args.compile_verify),
                max_diff=float(args.compile_max_diff),
            )

    def adapter_forward_with_extraction() -> torch.Tensor:
        if output_mode == "native_visual_kv_injection":
            if e2e_graph_runner is not None:
                next_visual_memory = llava_projected_image_features(model, inputs.pixel_values)
                prepared = prepare_llava_injection_inputs(
                    model,
                    inputs.input_ids,
                    inputs.pixel_values,
                    image_token_id,
                    attention_mask=attention_mask,
                    visual_memory=next_visual_memory,
                )
                return e2e_graph_runner(prepared)
            if bool(args.hf_attn_injection):
                next_visual_memory = llava_projected_image_features(model, inputs.pixel_values)
                prepared = prepare_llava_injection_inputs(
                    model,
                    inputs.input_ids,
                    inputs.pixel_values,
                    image_token_id,
                    attention_mask=attention_mask,
                    visual_memory=next_visual_memory,
                )
                return injection_prepared_fn(model, adapter, **prepared)
            return student_forward_llava_injection(
                model,
                inputs.input_ids,
                inputs.pixel_values,
                adapter,
                image_token_id,
                attention_mask=attention_mask,
            )
        next_source_k, next_source_v = extract_vision_kv(model, inputs.pixel_values, [22, 23])
        if e2e_graph_runner is not None:
            prepared = prepare_llava_adapter_only_inputs(
                model,
                inputs.input_ids,
                next_source_k,
                next_source_v,
                image_token_id,
                attention_mask=attention_mask,
            )
            return e2e_graph_runner(prepared)
        if bool(args.hf_attn_injection):
            prepared = prepare_llava_adapter_only_inputs(
                model,
                inputs.input_ids,
                next_source_k,
                next_source_v,
                image_token_id,
                attention_mask=attention_mask,
            )
            return adapter_only_prepared_fn(model, adapter, **prepared)
        return student_forward_with_visual_kv(
            model,
            inputs.input_ids,
            adapter,
            next_source_k,
            next_source_v,
            image_token_id,
            attention_mask=attention_mask,
        )

    if output_mode == "native_visual_kv_injection":
        assert visual_memory is not None
        cached_prepared = prepare_llava_injection_inputs(
            model,
            inputs.input_ids,
            inputs.pixel_values,
            image_token_id,
            attention_mask=attention_mask,
            visual_memory=visual_memory,
        )

        def cached_prepared_eager() -> torch.Tensor:
            return injection_prepared_fn(model, adapter, **cached_prepared)

        def cached_eager(
            input_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            attention_mask_tensor: torch.Tensor | None,
            visual_memory_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return student_forward_llava_injection(
                model,
                input_ids,
                pixel_values,
                adapter,
                image_token_id,
                attention_mask=attention_mask_tensor,
                visual_memory=visual_memory_tensor,
            )

        cached_graph = (
            cuda_graph_zero_arg(
                "llava_native_visual_kv_cached_cuda_graph",
                cached_prepared_eager,
                warmup=int(args.cuda_graph_warmup),
                verify=bool(args.compile_verify),
                max_diff=float(args.compile_max_diff),
            )
            if bool(args.cuda_graph)
            else None
        )
        compiled_cached = (
            torch.compile(cached_eager, mode=args.compile_mode, dynamic=args.compile_dynamic)
            if args.compile and cached_graph is None
            else None
        )
        compile_verified = False
        if bool(args.hf_attn_injection) and bool(args.hf_attn_verify):
            old_out = student_forward_llava_injection_prepared(model, adapter, **cached_prepared)
            new_out = student_forward_llava_injection_prepared_hf_attention(model, adapter, **cached_prepared)
            assert_same_logits("llava_native_visual_kv_hf_attn", old_out, new_out, float(args.hf_attn_max_diff))

        def adapter_forward_cached() -> torch.Tensor:
            nonlocal compile_verified
            if cached_graph is not None:
                return cached_graph()
            if compiled_cached is None:
                return cached_eager(inputs.input_ids, inputs.pixel_values, attention_mask, visual_memory)
            if bool(args.compile_verify) and not compile_verified:
                eager_out = cached_eager(inputs.input_ids, inputs.pixel_values, attention_mask, visual_memory)
                compiled_out = compiled_cached(inputs.input_ids, inputs.pixel_values, attention_mask, visual_memory)
                assert_same_logits("llava_native_visual_kv_cached", eager_out, compiled_out, args.compile_max_diff)
                compile_verified = True
                return compiled_out
            return compiled_cached(inputs.input_ids, inputs.pixel_values, attention_mask, visual_memory)
    else:
        assert source_k is not None and source_v is not None
        cached_prepared = prepare_llava_adapter_only_inputs(
            model,
            inputs.input_ids,
            source_k,
            source_v,
            image_token_id,
            attention_mask=attention_mask,
        )

        def cached_prepared_eager() -> torch.Tensor:
            return adapter_only_prepared_fn(model, adapter, **cached_prepared)

        def cached_eager(
            input_ids: torch.Tensor,
            attention_mask_tensor: torch.Tensor | None,
            source_k_tensor: torch.Tensor,
            source_v_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return student_forward_with_visual_kv(
                model,
                input_ids,
                adapter,
                source_k_tensor,
                source_v_tensor,
                image_token_id,
                attention_mask=attention_mask_tensor,
            )

        cached_graph = (
            cuda_graph_zero_arg(
                "llava_adapter_only_cached_cuda_graph",
                cached_prepared_eager,
                warmup=int(args.cuda_graph_warmup),
                verify=bool(args.compile_verify),
                max_diff=float(args.compile_max_diff),
            )
            if bool(args.cuda_graph)
            else None
        )
        compiled_cached = (
            torch.compile(cached_eager, mode=args.compile_mode, dynamic=args.compile_dynamic)
            if args.compile and cached_graph is None
            else None
        )
        compile_verified = False
        if bool(args.hf_attn_injection) and bool(args.hf_attn_verify):
            old_out = student_forward_with_visual_kv_prepared(model, adapter, **cached_prepared)
            new_out = student_forward_with_visual_kv_prepared_hf_attention(model, adapter, **cached_prepared)
            assert_same_logits("llava_adapter_only_hf_attn", old_out, new_out, float(args.hf_attn_max_diff))

        def adapter_forward_cached() -> torch.Tensor:
            nonlocal compile_verified
            if cached_graph is not None:
                return cached_graph()
            if compiled_cached is None:
                return cached_eager(inputs.input_ids, attention_mask, source_k, source_v)
            if bool(args.compile_verify) and not compile_verified:
                eager_out = cached_eager(inputs.input_ids, attention_mask, source_k, source_v)
                compiled_out = compiled_cached(inputs.input_ids, attention_mask, source_k, source_v)
                assert_same_logits("llava_adapter_only_cached", eager_out, compiled_out, args.compile_max_diff)
                compile_verified = True
                return compiled_out
            return compiled_cached(inputs.input_ids, attention_mask, source_k, source_v)

    teacher_fn = maybe_compile(
        lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values, attention_mask=attention_mask).logits,
        args,
        enabled=bool(args.compile_teacher),
    )
    compiled_e2e = torch.compile(adapter_forward_with_extraction, mode=args.compile_mode, dynamic=args.compile_dynamic) if args.compile_e2e else None
    e2e_compile_verified = False

    def e2e_fn() -> torch.Tensor:
        nonlocal e2e_compile_verified
        if compiled_e2e is None:
            return adapter_forward_with_extraction()
        if bool(args.compile_verify) and not e2e_compile_verified:
            eager_out = adapter_forward_with_extraction()
            compiled_out = compiled_e2e()
            assert_same_logits("llava_e2e_adapter", eager_out, compiled_out, args.compile_max_diff)
            e2e_compile_verified = True
            return compiled_out
        return compiled_e2e()

    teacher_s = benchmark(teacher_fn, warmup=args.warmup, n_runs=args.n_runs)
    e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)
    cached_s: float | None = None
    if bool(args.measure_cached):
        cached_s = benchmark(adapter_forward_cached, warmup=args.warmup, n_runs=args.n_runs)

    n_vis = int(visual_memory.shape[1]) if visual_memory is not None else int(source_k.shape[2])
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
            "mode": output_mode,
            "e2e_s": e2e_s,
            "e2e_ms": e2e_s * 1000.0,
            "e2e_speedup_vs_teacher": teacher_s / e2e_s,
            "cached_s": cached_s,
            "cached_ms": None if cached_s is None else cached_s * 1000.0,
            "cached_speedup_vs_teacher": None if cached_s is None else teacher_s / cached_s,
            "cuda_graph": bool(args.cuda_graph),
            "hf_attn_injection": bool(args.hf_attn_injection),
        },
    ]
    print()
    print(f"=== LLaVA Prefill Benchmark ({args.n_runs} runs) ===")
    print(
        f"visual_tokens={n_vis} text_tokens={n_text} output_mode={output_mode} "
        f"hf_attn_injection={bool(args.hf_attn_injection)}"
    )
    print(f"base_teacher_full={fmt_ms(teacher_s)} ms")
    print(f"{checkpoint_label_value} e2e={fmt_ms(e2e_s)} ms ({fmt_speedup(teacher_s, e2e_s)})")
    if cached_s is not None:
        print(f"{checkpoint_label_value} cached={fmt_ms(cached_s)} ms ({fmt_speedup(teacher_s, cached_s)}) [diagnostic]")
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
    parser.add_argument("--exact-bucket-lengths", action=argparse.BooleanOptionalAction, default=False, help="Use processed Qwen token lengths for batch bucketing. Setup is slower; measured prefill is unchanged.")
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help="Cap padded tokens per Qwen benchmark batch as max_seq_len * batch_size. 0 disables.",
    )
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument(
        "--sdpa-backend",
        choices=("default", "math", "efficient", "flash", "cudnn"),
        default="default",
        help="Force a PyTorch SDPA backend for diagnostics. Default leaves backend selection unchanged.",
    )
    parser.add_argument("--output-mode", choices=("adapter_only", "native_visual_kv_injection"), default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile for adapter measured functions. This is opt-in because exact BF16 compile paths may be slower.",
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
    parser.add_argument("--compile-verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-max-diff", type=float, default=0.0)
    parser.add_argument("--compile-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA graph replay for adapter layer loops on fixed-shape CUDA benchmark inputs. Use --no-cuda-graph for eager debugging.",
    )
    parser.add_argument(
        "--cuda-graph-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA graph replay for Qwen vision/context build when --cuda-graph is enabled.",
    )
    parser.add_argument("--cuda-graph-warmup", type=int, default=3)
    parser.add_argument(
        "--hf-attn-injection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark the project-local HF SDPA attention-interface path for external visual KV injection.",
    )
    parser.add_argument(
        "--hf-attn-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --hf-attn-injection is enabled, compare logits against the current canonical path before timing.",
    )
    parser.add_argument("--hf-attn-max-diff", type=float, default=0.0)
    parser.add_argument("--last-logits-only", action=argparse.BooleanOptionalAction, default=True, help="Only compute logits for the next-token position.")
    parser.add_argument("--context-cache-dir", default="", help="Optional cache for Qwen initial_hidden/position_ids after vision encoder/merger.")
    parser.add_argument("--structured-answer-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure-cached", action=argparse.BooleanOptionalAction, default=False, help="Also benchmark cached adapter-only path; diagnostic, not full e2e method speed.")
    parser.add_argument("--skip-e2e", action="store_true", help="Only benchmark cached adapter path.")
    parser.add_argument("--skip-v0", action="store_true", help="Skip Qwen V0 source build timing.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Batch Qwen adapter generation for --metric-table.")
    parser.add_argument("--eval-max-batch-tokens", type=int, default=0, help="Optional rough token budget for Qwen --metric-table eval batches.")
    parser.add_argument(
        "--eval-bucket-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort Qwen --metric-table eval samples by rough length to reduce padding when eval_batch_size > 1.",
    )
    parser.add_argument("--verify-batched-generation", type=int, default=0, help="Verify this many batched Qwen generations against single-sample generation.")
    parser.add_argument(
        "--adapter-decode-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use shape-exact adapter decode cache for Qwen --metric-table generation.",
    )
    parser.add_argument(
        "--adapter-decode-cache-mode",
        choices=("shape_exact", "fast"),
        default="shape_exact",
        help="Qwen metric-table decode cache mode. fast is diagnostic and not logits-exact.",
    )
    parser.add_argument("--verify-decode-cache-generation", type=int, default=0, help="Verify this many decode-cache generations against full recompute.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_e2e:
        args.measure_cached = True
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

    with sdpa_backend_context(args.sdpa_backend):
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
