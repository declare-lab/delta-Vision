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


def token_budget_batches(
    keyed_items: list[tuple[int, Any]],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[list[Any]]:
    max_batch_size = max(1, int(max_batch_size))
    max_batch_tokens = int(max_batch_tokens)
    if max_batch_tokens <= 0:
        return chunked([item for _, item in keyed_items], max_batch_size)

    batches: list[list[Any]] = []
    current: list[tuple[int, Any]] = []
    current_max_len = 0
    for length, item in keyed_items:
        item_len = max(1, int(length))
        next_max_len = max(current_max_len, item_len)
        next_size = len(current) + 1
        would_overflow_size = next_size > max_batch_size
        would_overflow_tokens = current and (next_max_len * next_size > max_batch_tokens)
        if would_overflow_size or would_overflow_tokens:
            batches.append([entry for _, entry in current])
            current = []
            current_max_len = 0
        current.append((item_len, item))
        current_max_len = max(current_max_len, item_len)
    if current:
        batches.append([entry for _, entry in current])
    return batches


def rough_qwen_length_key(row: dict[str, Any]) -> int:
    return len(str(row.get("question", ""))) + len(str(row.get("image", ""))) // 8


def qwen_batch_padding_stats(attention_mask: torch.Tensor) -> dict[str, float | int]:
    batch_size, seq_len = attention_mask.shape
    token_slots = int(batch_size) * int(seq_len)
    valid_tokens = int(attention_mask.bool().sum().item())
    pad_tokens = max(0, token_slots - valid_tokens)
    return {
        "seq_len": int(seq_len),
        "token_slots": token_slots,
        "valid_tokens": valid_tokens,
        "pad_tokens": pad_tokens,
        "padding_waste_ratio": float(pad_tokens) / float(max(1, token_slots)),
    }


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
    for name, param in adapter.named_parameters():
        active = name.startswith("visual_adapter_")
        if active:
            total += param.numel()
    return total


def count_total_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def assert_same_logits(name: str, reference: torch.Tensor, candidate: torch.Tensor, max_diff: float) -> None:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"{name} compile changed shape: eager={tuple(reference.shape)} compiled={tuple(candidate.shape)}")
    diff = (reference.float() - candidate.float()).abs()
    observed = float(diff.max().item()) if diff.numel() else 0.0
    if observed > float(max_diff):
        raise RuntimeError(f"{name} compile changed logits: max_diff={observed:.8g} allowed={float(max_diff):.8g}")


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    device = tensor.device
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, device.type, device.index)


def _tree_signature(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _tensor_signature(value)
    if isinstance(value, tuple):
        return tuple(_tree_signature(item) for item in value)
    if isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"unsupported CUDA graph value type: {type(value)!r}")


def _clone_tree_for_cuda_graph(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return torch.empty_like(value).copy_(value)
    if isinstance(value, tuple):
        return tuple(_clone_tree_for_cuda_graph(item) for item in value)
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported CUDA graph value type: {type(value)!r}")


def _copy_tree_(target: Any, source: Any) -> None:
    if target is None and source is None:
        return
    if torch.is_tensor(target) and torch.is_tensor(source):
        target.copy_(source)
        return
    if isinstance(target, tuple) and isinstance(source, tuple):
        if len(target) != len(source):
            raise RuntimeError("CUDA graph tuple length changed")
        for target_item, source_item in zip(target, source):
            _copy_tree_(target_item, source_item)
        return
    if isinstance(target, (bool, int, float, str)) and isinstance(source, type(target)) and target == source:
        return
    raise RuntimeError(f"CUDA graph value type changed: {type(target)!r} vs {type(source)!r}")


class QwenAdapterCudaGraphRunner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        adapter: torch.nn.Module,
        prepare_qwen_visual_delta_inputs: Callable[..., Any],
        qwen_visual_delta_logits_prepared: Callable[..., Any],
        logits_to_keep: int,
        graph_warmup: int,
        verify: bool,
        max_diff: float,
        reuse_static_input: bool,
        name: str,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-graph requires CUDA")
        self.model = model
        self.adapter = adapter
        self.prepare_qwen_visual_delta_inputs = prepare_qwen_visual_delta_inputs
        self.qwen_visual_delta_logits_prepared = qwen_visual_delta_logits_prepared
        self.logits_to_keep = int(logits_to_keep)
        self.graph_warmup = int(max(0, graph_warmup))
        self.verify = bool(verify)
        self.max_diff = float(max_diff)
        self.reuse_static_input = bool(reuse_static_input)
        self.name = name
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_prepared: dict[str, Any] | None = None
        self.signature: Any = None
        self.static_input_key: tuple[int, ...] | None = None
        self.output: torch.Tensor | None = None
        self.verified = False

    def _prepare(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        initial_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> dict[str, Any]:
        return self.prepare_qwen_visual_delta_inputs(
            self.model,
            self.adapter,
            input_ids,
            attention_mask,
            mm_token_type_ids,
            initial_hidden,
            position_ids,
            reuse_position_embeddings=True,
        )

    def _forward_prepared(self, prepared: dict[str, Any]) -> torch.Tensor:
        return self.qwen_visual_delta_logits_prepared(
            self.model,
            self.adapter,
            h=prepared["h"],
            visual_memory=prepared["visual_memory"],
            text_mask=prepared["text_mask"],
            text_position_ids=prepared["text_position_ids"],
            visual_position_ids=prepared["visual_position_ids"],
            prefix_attention_mask=prepared["prefix_attention_mask"],
            text_position_embeddings=prepared["text_position_embeddings"],
            visual_position_embeddings=prepared["visual_position_embeddings"],
            logits_to_keep=self.logits_to_keep,
        )[0]

    def _graph_signature(self, prepared: dict[str, Any]) -> tuple[Any, ...]:
        return (
            _tree_signature(prepared["h"]),
            _tree_signature(prepared["visual_memory"]),
            _tree_signature(prepared["text_mask"]),
            _tree_signature(prepared["text_position_ids"]),
            _tree_signature(prepared["visual_position_ids"]),
            _tree_signature(prepared["prefix_attention_mask"]),
            _tree_signature(prepared["text_position_embeddings"]),
            _tree_signature(prepared["visual_position_embeddings"]),
        )

    def _clone_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return {
            "h": _clone_tree_for_cuda_graph(prepared["h"]),
            "visual_memory": _clone_tree_for_cuda_graph(prepared["visual_memory"]),
            "text_mask": _clone_tree_for_cuda_graph(prepared["text_mask"]),
            "text_position_ids": _clone_tree_for_cuda_graph(prepared["text_position_ids"]),
            "visual_position_ids": _clone_tree_for_cuda_graph(prepared["visual_position_ids"]),
            "prefix_attention_mask": _clone_tree_for_cuda_graph(prepared["prefix_attention_mask"]),
            "text_position_embeddings": _clone_tree_for_cuda_graph(prepared["text_position_embeddings"]),
            "visual_position_embeddings": _clone_tree_for_cuda_graph(prepared["visual_position_embeddings"]),
        }

    def _copy_prepared_(self, prepared: dict[str, Any]) -> None:
        if self.static_prepared is None:
            raise RuntimeError("CUDA graph static tensors are not initialized")
        for key in self.static_prepared:
            _copy_tree_(self.static_prepared[key], prepared[key])

    def _static_forward(self) -> torch.Tensor:
        if self.static_prepared is None:
            raise RuntimeError("CUDA graph static tensors are not initialized")
        return self._forward_prepared(self.static_prepared)

    def _capture(self, prepared: dict[str, Any], signature: Any) -> None:
        eager_out = self._forward_prepared(prepared)
        device = eager_out.device
        self.static_prepared = self._clone_prepared(prepared)
        with torch.cuda.device(device):
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(self.graph_warmup):
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

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        initial_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_key = (
            input_ids.data_ptr(),
            attention_mask.data_ptr(),
            mm_token_type_ids.data_ptr(),
            initial_hidden.data_ptr(),
            position_ids.data_ptr(),
        )
        if self.reuse_static_input and self.graph is not None and input_key == self.static_input_key:
            self.graph.replay()
            if self.output is None:
                raise RuntimeError("CUDA graph did not produce an output")
            return self.output

        prepared = self._prepare(input_ids, attention_mask, mm_token_type_ids, initial_hidden, position_ids)
        signature = self._graph_signature(prepared)
        if self.graph is None or signature != self.signature:
            self._capture(prepared, signature)
        else:
            self._copy_prepared_(prepared)
            self.graph.replay()
        self.static_input_key = input_key
        if self.output is None:
            raise RuntimeError("CUDA graph did not produce an output")
        return self.output


class QwenContextCudaGraphRunner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        build_qwen_initial_context: Callable[..., Any],
        qwen_position_ids: Callable[..., Any],
        qwen_visual_grid_metadata: Callable[..., Any],
        graph_warmup: int,
        verify: bool,
        max_diff: float,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-graph-context requires CUDA")
        self.model = model
        self.build_qwen_initial_context = build_qwen_initial_context
        self.qwen_position_ids = qwen_position_ids
        self.qwen_visual_grid_metadata = qwen_visual_grid_metadata
        self.graph_warmup = int(max(0, graph_warmup))
        self.verify = bool(verify)
        self.max_diff = float(max_diff)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_inputs: dict[str, torch.Tensor] | None = None
        self.static_position_ids: torch.Tensor | None = None
        self.static_visual_grid_metadata: dict[str, Any] | None = None
        self.signature: Any = None
        self.output_hidden: torch.Tensor | None = None
        self.output_position_ids: torch.Tensor | None = None
        self.verified = False

    def _inputs_dict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

    def _prepare_static_context(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        qwen_model = self.model.model
        position_inputs_embeds = qwen_model.get_input_embeddings()(inputs["input_ids"])
        position_ids = self.qwen_position_ids(self.model, inputs, inputs_embeds=position_inputs_embeds)
        visual_grid_metadata = self.qwen_visual_grid_metadata(self.model, inputs["image_grid_thw"])
        return position_ids, visual_grid_metadata

    def _signature(
        self,
        inputs: dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        visual_grid_metadata: dict[str, Any],
    ) -> tuple[Any, ...]:
        return (
            tuple((key, _tree_signature(inputs[key])) for key in sorted(inputs)),
            _tree_signature(position_ids),
            tuple((key, _tree_signature(visual_grid_metadata[key])) for key in sorted(visual_grid_metadata)),
        )

    def _clone_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: torch.empty_like(value).copy_(value) for key, value in inputs.items()}

    def _clone_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {key: _clone_tree_for_cuda_graph(value) for key, value in metadata.items()}

    def _copy_inputs_(self, inputs: dict[str, torch.Tensor]) -> None:
        if self.static_inputs is None:
            raise RuntimeError("CUDA graph static context inputs are not initialized")
        for key, value in inputs.items():
            self.static_inputs[key].copy_(value)

    def _copy_metadata_(self, position_ids: torch.Tensor, visual_grid_metadata: dict[str, Any]) -> None:
        if self.static_position_ids is None or self.static_visual_grid_metadata is None:
            raise RuntimeError("CUDA graph static context metadata is not initialized")
        self.static_position_ids.copy_(position_ids)
        for key, value in visual_grid_metadata.items():
            _copy_tree_(self.static_visual_grid_metadata[key], value)

    def _static_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.static_inputs is None or self.static_position_ids is None or self.static_visual_grid_metadata is None:
            raise RuntimeError("CUDA graph static context inputs are not initialized")
        return self.build_qwen_initial_context(
            self.model,
            self.static_inputs,
            position_ids=self.static_position_ids,
            visual_grid_metadata=self.static_visual_grid_metadata,
        )

    def _capture(
        self,
        inputs: dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        visual_grid_metadata: dict[str, Any],
        signature: Any,
    ) -> None:
        eager_hidden, eager_position_ids = self.build_qwen_initial_context(
            self.model,
            inputs,
            position_ids=position_ids,
            visual_grid_metadata=visual_grid_metadata,
        )
        device = eager_hidden.device
        self.static_inputs = self._clone_inputs(inputs)
        self.static_position_ids = torch.empty_like(position_ids).copy_(position_ids)
        self.static_visual_grid_metadata = self._clone_metadata(visual_grid_metadata)
        with torch.cuda.device(device):
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(self.graph_warmup):
                    self._static_forward()
            torch.cuda.current_stream().wait_stream(side_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output_hidden, self.output_position_ids = self._static_forward()
            self.graph.replay()
        if self.verify and not self.verified:
            if self.output_hidden is None or self.output_position_ids is None:
                raise RuntimeError("CUDA graph did not produce Qwen context outputs")
            assert_same_logits("qwen_context_cuda_graph_hidden", eager_hidden, self.output_hidden, self.max_diff)
            assert_same_logits("qwen_context_cuda_graph_position_ids", eager_position_ids, self.output_position_ids, 0.0)
            self.verified = True
        self.signature = signature

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs_dict(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
        position_ids, visual_grid_metadata = self._prepare_static_context(inputs)
        signature = self._signature(inputs, position_ids, visual_grid_metadata)
        if self.graph is None or signature != self.signature:
            self._capture(inputs, position_ids, visual_grid_metadata, signature)
        else:
            self._copy_inputs_(inputs)
            self._copy_metadata_(position_ids, visual_grid_metadata)
            self.graph.replay()
        if self.output_hidden is None or self.output_position_ids is None:
            raise RuntimeError("CUDA graph did not produce Qwen context outputs")
        return self.output_hidden, self.output_position_ids


def build_qwen_benchmark_delta_fn(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    *,
    args: argparse.Namespace,
    logits_to_keep: int,
    qwen_visual_delta_logits_from_tensors: Callable[..., Any],
    prepare_qwen_visual_delta_inputs: Callable[..., Any] | None = None,
    qwen_visual_delta_logits_prepared: Callable[..., Any] | None = None,
) -> Callable[[dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    if bool(getattr(args, "cuda_graph", False)) and bool(getattr(args, "compile", False)):
        raise RuntimeError("--cuda-graph and --compile are separate adapter fast paths; enable only one")

    cuda_graph_runner: QwenAdapterCudaGraphRunner | None = None
    if bool(getattr(args, "cuda_graph", False)):
        if prepare_qwen_visual_delta_inputs is None or qwen_visual_delta_logits_prepared is None:
            raise RuntimeError("Qwen CUDA graph path requires prepared Qwen adapter functions")
        cuda_graph_runner = QwenAdapterCudaGraphRunner(
            model=model,
            adapter=adapter,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=qwen_visual_delta_logits_prepared,
            logits_to_keep=logits_to_keep,
            graph_warmup=int(getattr(args, "cuda_graph_warmup", 3)),
            verify=bool(getattr(args, "compile_verify", True)),
            max_diff=float(getattr(args, "compile_max_diff", 0.0)),
            reuse_static_input=True,
            name="qwen_cached_adapter_cuda_graph",
        )

    def prepared_forward(prepared: dict[str, Any]) -> torch.Tensor:
        if qwen_visual_delta_logits_prepared is None:
            raise RuntimeError("prepared Qwen adapter function is required")
        return qwen_visual_delta_logits_prepared(
            model,
            adapter,
            h=prepared["h"],
            visual_memory=prepared["visual_memory"],
            text_mask=prepared["text_mask"],
            text_position_ids=prepared["text_position_ids"],
            visual_position_ids=prepared["visual_position_ids"],
            prefix_attention_mask=prepared["prefix_attention_mask"],
            text_position_embeddings=prepared["text_position_embeddings"],
            visual_position_embeddings=prepared["visual_position_embeddings"],
            logits_to_keep=logits_to_keep,
        )[0]

    def eager(inputs: dict[str, torch.Tensor], initial_hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if cuda_graph_runner is not None:
            return cuda_graph_runner(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                initial_hidden,
                position_ids,
            )
        if bool(getattr(args, "hf_attn_injection", False)):
            prepared = prepare_qwen_visual_delta_inputs(
                model,
                adapter,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                initial_hidden,
                position_ids,
                reuse_position_embeddings=True,
            )
            return prepared_forward(prepared)
        return qwen_visual_delta_logits_from_tensors(
            model,
            adapter,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
            compact_no_padding=True,
            logits_to_keep=logits_to_keep,
        )[0]

    if not args.compile:
        return eager

    if prepare_qwen_visual_delta_inputs is None or qwen_visual_delta_logits_prepared is None:
        raise RuntimeError("Qwen compile path requires prepared Qwen adapter functions")

    def cached_forward_prepared(
        h: torch.Tensor,
        visual_memory: torch.Tensor,
        text_mask: torch.Tensor,
        text_position_ids: torch.Tensor,
        visual_position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        text_cos: torch.Tensor,
        text_sin: torch.Tensor,
        visual_cos: torch.Tensor,
        visual_sin: torch.Tensor,
    ) -> torch.Tensor:
        return qwen_visual_delta_logits_prepared(
            model,
            adapter,
            h=h,
            visual_memory=visual_memory,
            text_mask=text_mask,
            text_position_ids=text_position_ids,
            visual_position_ids=visual_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            text_position_embeddings=(text_cos, text_sin),
            visual_position_embeddings=(visual_cos, visual_sin),
            logits_to_keep=logits_to_keep,
        )[0]

    compiled_cached = torch.compile(cached_forward_prepared, mode=args.compile_mode, dynamic=args.compile_dynamic)
    verified = False

    def compiled(inputs: dict[str, torch.Tensor], initial_hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        nonlocal verified
        prepared = prepare_qwen_visual_delta_inputs(
            model,
            adapter,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
            reuse_position_embeddings=True,
        )
        text_cos, text_sin = prepared["text_position_embeddings"]
        visual_cos, visual_sin = prepared["visual_position_embeddings"]
        tensor_args = (
            prepared["h"],
            prepared["visual_memory"],
            prepared["text_mask"],
            prepared["text_position_ids"],
            prepared["visual_position_ids"],
            prepared["prefix_attention_mask"],
            text_cos,
            text_sin,
            visual_cos,
            visual_sin,
        )
        if bool(getattr(args, "compile_verify", True)) and not verified:
            eager_out = prepared_forward(prepared)
            compiled_out = compiled_cached(*tensor_args)
            assert_same_logits(
                "qwen_cached_adapter",
                eager_out,
                compiled_out,
                float(getattr(args, "compile_max_diff", 0.0)),
            )
            verified = True
            return compiled_out
        return compiled_cached(*tensor_args)

    return compiled


def build_qwen_benchmark_e2e_fn(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    *,
    args: argparse.Namespace,
    logits_to_keep: int,
    build_qwen_initial_context: Callable[..., Any],
    qwen_visual_delta_logits: Callable[..., Any],
    prepare_qwen_visual_delta_inputs: Callable[..., Any] | None = None,
    qwen_visual_delta_logits_prepared: Callable[..., Any] | None = None,
    qwen_position_ids: Callable[..., Any] | None = None,
    qwen_visual_grid_metadata: Callable[..., Any] | None = None,
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    if bool(getattr(args, "cuda_graph", False)) and bool(getattr(args, "compile_e2e", False)):
        raise RuntimeError("--cuda-graph and --compile-e2e are separate Qwen e2e fast paths; enable only one")

    cuda_graph_runner: QwenAdapterCudaGraphRunner | None = None
    if bool(getattr(args, "cuda_graph", False)):
        if prepare_qwen_visual_delta_inputs is None or qwen_visual_delta_logits_prepared is None:
            raise RuntimeError("Qwen CUDA graph path requires prepared Qwen adapter functions")
        cuda_graph_runner = QwenAdapterCudaGraphRunner(
            model=model,
            adapter=adapter,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=qwen_visual_delta_logits_prepared,
            logits_to_keep=logits_to_keep,
            graph_warmup=int(getattr(args, "cuda_graph_warmup", 3)),
            verify=bool(getattr(args, "compile_verify", True)),
            max_diff=float(getattr(args, "compile_max_diff", 0.0)),
            reuse_static_input=False,
            name="qwen_e2e_adapter_cuda_graph",
        )
    context_graph_runner: QwenContextCudaGraphRunner | None = None
    if bool(getattr(args, "cuda_graph", False)) and bool(getattr(args, "cuda_graph_context", False)):
        if qwen_position_ids is None or qwen_visual_grid_metadata is None:
            raise RuntimeError("Qwen context CUDA graph path requires qwen_position_ids and qwen_visual_grid_metadata")
        context_graph_runner = QwenContextCudaGraphRunner(
            model=model,
            build_qwen_initial_context=build_qwen_initial_context,
            qwen_position_ids=qwen_position_ids,
            qwen_visual_grid_metadata=qwen_visual_grid_metadata,
            graph_warmup=int(getattr(args, "cuda_graph_warmup", 3)),
            verify=bool(getattr(args, "compile_verify", True)),
            max_diff=float(getattr(args, "compile_max_diff", 0.0)),
        )

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
        if context_graph_runner is None:
            initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
        else:
            initial_hidden, position_ids = context_graph_runner(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                mm_token_type_ids,
            )
        if cuda_graph_runner is not None:
            return cuda_graph_runner(input_ids, attention_mask, mm_token_type_ids, initial_hidden, position_ids)
        if bool(getattr(args, "hf_attn_injection", False)):
            if prepare_qwen_visual_delta_inputs is None or qwen_visual_delta_logits_prepared is None:
                raise RuntimeError("Qwen HF-attn injection path requires prepared Qwen adapter functions")
            prepared = prepare_qwen_visual_delta_inputs(
                model,
                adapter,
                input_ids,
                attention_mask,
                mm_token_type_ids,
                initial_hidden,
                position_ids,
                reuse_position_embeddings=True,
            )
            return qwen_visual_delta_logits_prepared(
                model,
                adapter,
                h=prepared["h"],
                visual_memory=prepared["visual_memory"],
                text_mask=prepared["text_mask"],
                text_position_ids=prepared["text_position_ids"],
                visual_position_ids=prepared["visual_position_ids"],
                prefix_attention_mask=prepared["prefix_attention_mask"],
                text_position_embeddings=prepared["text_position_embeddings"],
                visual_position_embeddings=prepared["visual_position_embeddings"],
                logits_to_keep=logits_to_keep,
            )[0]
        return qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            compact_no_padding=True,
            logits_to_keep=logits_to_keep,
        )[0]

    compiled_e2e = (
        torch.compile(e2e_forward, mode=args.compile_mode, dynamic=args.compile_dynamic)
        if args.compile_e2e
        else None
    )
    verified = False

    def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal verified
        tensor_args = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["mm_token_type_ids"],
        )
        if compiled_e2e is None:
            return e2e_forward(*tensor_args)
        if bool(getattr(args, "compile_verify", True)) and not verified:
            eager_out = e2e_forward(*tensor_args)
            compiled_out = compiled_e2e(*tensor_args)
            assert_same_logits(
                "qwen_e2e_adapter",
                eager_out,
                compiled_out,
                float(getattr(args, "compile_max_diff", 0.0)),
            )
            verified = True
            return compiled_out
        return compiled_e2e(*tensor_args)

    return run


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
    hf_attn = any(row.get("hf_attn_injection") for row in rows)
    print(f"hf_attn_injection={hf_attn}")
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
        if row.get("cached_s") is not None:
            print(
                f"{(str(row['name']) + ' cached adapter')[:52]:52} {fmt_ms(row['cached_s']):>9} "
                f"{fmt_speedup(teacher_s, row['cached_s']):>8} {fmt_delta_ms(teacher_s, row['cached_s']):>9} diagnostic only"
            )
    print()
    show_cached = any(row["kind"] == "adapter" and row.get("cached_s") is not None for row in rows)
    cached_header = f" {'cached ms':>10} {'cached':>8}" if show_cached else ""
    header = f"{'name':48} {'mode':27} {'active/total M':>18} {'ckpt GB':>8} {'e2e ms':>9} {'e2e':>7}{cached_header}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["kind"] != "adapter":
            continue
        params = f"{row['active_params_m']:.2f}/{row['total_params_m']:.2f}"
        line = (
            f"{row['name'][:48]:48} {row['mode'][:27]:27} {params:>18} "
            f"{row['checkpoint_gb']:8.2f} {fmt_ms(row['e2e_s']):>9} {fmt_speedup(teacher_s, row['e2e_s']):>7}"
        )
        if show_cached:
            line += f" {fmt_ms(row.get('cached_s')):>10} {fmt_speedup(teacher_s, row.get('cached_s')):>8}"
        print(line)


def run_qwen3vl(args: argparse.Namespace) -> None:
    from src.model import (
        build_qwen_initial_context,
        load_or_build_qwen_initial_context,
        load_frozen_qwen3vl,
        load_qwen_visual_delta_checkpoint,
        prepare_qwen_visual_delta_inputs,
        prepare_qwen3vl_batch_inputs,
        qwen_position_ids,
        qwen_visual_delta_logits,
        qwen_visual_delta_logits_from_tensors,
        qwen_visual_delta_logits_prepared,
        qwen_visual_delta_logits_prepared_hf_attention,
        qwen_visual_grid_metadata,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    prepared_logits_fn = (
        qwen_visual_delta_logits_prepared_hf_attention
        if bool(getattr(args, "hf_attn_injection", False))
        else qwen_visual_delta_logits_prepared
    )

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

    initial_hidden = position_ids = None
    if bool(getattr(args, "measure_cached", False)):
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
        adapter_forward = build_qwen_benchmark_delta_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            qwen_visual_delta_logits_from_tensors=qwen_visual_delta_logits_from_tensors,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=prepared_logits_fn,
        )
        e2e_forward = build_qwen_benchmark_e2e_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            build_qwen_initial_context=build_qwen_initial_context,
            qwen_visual_delta_logits=qwen_visual_delta_logits,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=prepared_logits_fn,
            qwen_position_ids=qwen_position_ids,
            qwen_visual_grid_metadata=qwen_visual_grid_metadata,
        )
        if bool(getattr(args, "hf_attn_injection", False)) and bool(getattr(args, "hf_attn_verify", True)):
            verify_hidden, verify_position_ids = load_or_build_qwen_initial_context(
                model,
                dict(inputs),
                cache_dir=args.context_cache_dir or None,
                dtype=dtype,
            )
            verify_prepared = prepare_qwen_visual_delta_inputs(
                model,
                adapter,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                verify_hidden,
                verify_position_ids,
                reuse_position_embeddings=True,
            )
            old_out = qwen_visual_delta_logits_prepared(
                model,
                adapter,
                h=verify_prepared["h"],
                visual_memory=verify_prepared["visual_memory"],
                text_mask=verify_prepared["text_mask"],
                text_position_ids=verify_prepared["text_position_ids"],
                visual_position_ids=verify_prepared["visual_position_ids"],
                prefix_attention_mask=verify_prepared["prefix_attention_mask"],
                text_position_embeddings=verify_prepared["text_position_embeddings"],
                visual_position_embeddings=verify_prepared["visual_position_embeddings"],
                logits_to_keep=logits_to_keep,
            )[0]
            new_out = qwen_visual_delta_logits_prepared_hf_attention(
                model,
                adapter,
                h=verify_prepared["h"],
                visual_memory=verify_prepared["visual_memory"],
                text_mask=verify_prepared["text_mask"],
                text_position_ids=verify_prepared["text_position_ids"],
                visual_position_ids=verify_prepared["visual_position_ids"],
                prefix_attention_mask=verify_prepared["prefix_attention_mask"],
                text_position_embeddings=verify_prepared["text_position_embeddings"],
                visual_position_embeddings=verify_prepared["visual_position_embeddings"],
                logits_to_keep=logits_to_keep,
            )[0]
            assert_same_logits("qwen_hf_attn_injection", old_out, new_out, float(getattr(args, "hf_attn_max_diff", 0.0)))

        e2e_s: float | None = None
        if not args.skip_e2e:
            def e2e_fn() -> torch.Tensor:
                return e2e_forward(dict(inputs))

            e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)

        cached_s: float | None = None
        if bool(getattr(args, "measure_cached", False)):
            assert initial_hidden is not None and position_ids is not None

            def cached_fn() -> torch.Tensor:
                return adapter_forward(dict(inputs), initial_hidden, position_ids)

            cached_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)

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
                "cached_ms": None if cached_s is None else cached_s * 1000.0,
                "cached_speedup_vs_teacher": None if cached_s is None else teacher_s / cached_s,
                "cuda_graph": bool(getattr(args, "cuda_graph", False)),
                "hf_attn_injection": bool(getattr(args, "hf_attn_injection", False)),
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
        prepare_qwen_visual_delta_inputs,
        prepare_qwen3vl_batch_inputs,
        qwen_position_ids,
        qwen_visual_delta_logits,
        qwen_visual_delta_logits_from_tensors,
        qwen_visual_delta_logits_prepared,
        qwen_visual_delta_logits_prepared_hf_attention,
        qwen_visual_grid_metadata,
    )

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    prepared_logits_fn = (
        qwen_visual_delta_logits_prepared_hf_attention
        if bool(getattr(args, "hf_attn_injection", False))
        else qwen_visual_delta_logits_prepared
    )
    rows = read_jsonl_samples(args.sample_jsonl, args.sample_index, int(args.sample_count))
    if args.bucket_by_length:
        data_root = Path(args.data_root).expanduser()
        if bool(getattr(args, "exact_bucket_lengths", False)):
            keyed_rows: list[tuple[int, dict[str, Any]]] = []
            for row in rows:
                row_inputs, _, _, _ = prepare_qwen3vl_batch_inputs(
                    processor,
                    [row],
                    data_root,
                    device,
                    include_answers=False,
                )
                keyed_rows.append((int(row_inputs["attention_mask"].bool().sum().item()), row))
            keyed_rows = sorted(keyed_rows, key=lambda item: item[0])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            keyed_rows = sorted((rough_qwen_length_key(row), row) for row in rows)
    else:
        keyed_rows = [(rough_qwen_length_key(row), row) for row in rows]
    batches = token_budget_batches(
        keyed_rows,
        max_batch_size=int(args.batch_size),
        max_batch_tokens=int(getattr(args, "max_batch_tokens", 0)),
    )
    logits_to_keep = 1 if args.last_logits_only else 0

    checkpoint_specs = collect_checkpoint_specs(args)
    adapters: list[tuple[str, Path, torch.nn.Module, dict[str, Any]]] = []
    adapter_cached_forwards: dict[str, Callable[[dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    adapter_e2e_forwards: dict[str, Callable[[dict[str, torch.Tensor]], torch.Tensor]] = {}
    for label, checkpoint in checkpoint_specs:
        adapter, meta = load_qwen_visual_delta_checkpoint(checkpoint, model.model.language_model, device, dtype)
        adapters.append((label, checkpoint, adapter, meta))
        adapter_cached_forwards[label] = build_qwen_benchmark_delta_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            qwen_visual_delta_logits_from_tensors=qwen_visual_delta_logits_from_tensors,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=prepared_logits_fn,
        )
        adapter_e2e_forwards[label] = build_qwen_benchmark_e2e_fn(
            model,
            adapter,
            args=args,
            logits_to_keep=logits_to_keep,
            build_qwen_initial_context=build_qwen_initial_context,
            qwen_visual_delta_logits=qwen_visual_delta_logits,
            prepare_qwen_visual_delta_inputs=prepare_qwen_visual_delta_inputs,
            qwen_visual_delta_logits_prepared=prepared_logits_fn,
            qwen_position_ids=qwen_position_ids,
            qwen_visual_grid_metadata=qwen_visual_grid_metadata,
        )

    rows_out: list[dict[str, Any]] = []
    teacher_batch_seconds: list[float] = []
    teacher_sample_seconds: list[float] = []
    adapter_e2e_batch_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}
    adapter_e2e_sample_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}
    adapter_cached_batch_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}
    adapter_cached_sample_seconds: dict[str, list[float]] = {label: [] for label, _, _, _ in adapters}
    total_pad_tokens = 0
    total_token_slots = 0

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
        padding_stats = qwen_batch_padding_stats(attention_mask)
        total_pad_tokens += int(padding_stats["pad_tokens"])
        total_token_slots += int(padding_stats["token_slots"])

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
                **padding_stats,
                "seconds": teacher_s,
                "seconds_per_sample": teacher_s / max(1, len(batch_rows)),
                "images": image_paths,
            }
        )

        initial_hidden = position_ids = None
        if bool(getattr(args, "measure_cached", False)):
            initial_hidden, position_ids = load_or_build_qwen_initial_context(
                model,
                dict(inputs),
                cache_dir=args.context_cache_dir or None,
                dtype=dtype,
            )

        for label, checkpoint, adapter, meta in adapters:
            mode = str(meta.get("args", {}).get("output_mode", getattr(adapter, "mode", "")))
            if not bool(getattr(args, "skip_e2e", False)):
                def e2e_fn() -> torch.Tensor:
                    return adapter_e2e_forwards[label](dict(inputs))

                adapter_e2e_s = benchmark(e2e_fn, warmup=args.warmup, n_runs=args.n_runs)
                adapter_e2e_batch_seconds[label].append(adapter_e2e_s)
                adapter_e2e_sample_seconds[label].append(adapter_e2e_s / max(1, len(batch_rows)))
                rows_out.append(
                    {
                        "kind": "batch_adapter_e2e",
                        "name": label,
                        "checkpoint": str(checkpoint),
                        "mode": mode,
                        "batch_index": batch_idx,
                        "batch_size": len(batch_rows),
                        "visual_tokens": visual_tokens,
                        "text_tokens": text_tokens,
                        **padding_stats,
                        "seconds": adapter_e2e_s,
                        "seconds_per_sample": adapter_e2e_s / max(1, len(batch_rows)),
                        "speedup_vs_teacher": teacher_s / adapter_e2e_s,
                        "cuda_graph": bool(getattr(args, "cuda_graph", False)),
                    }
                )
            if bool(getattr(args, "measure_cached", False)):
                assert initial_hidden is not None and position_ids is not None

                def cached_fn() -> torch.Tensor:
                    return adapter_cached_forwards[label](dict(inputs), initial_hidden, position_ids)

                adapter_cached_s = benchmark(cached_fn, warmup=args.warmup, n_runs=args.n_runs)
                adapter_cached_batch_seconds[label].append(adapter_cached_s)
                adapter_cached_sample_seconds[label].append(adapter_cached_s / max(1, len(batch_rows)))
                rows_out.append(
                    {
                        "kind": "batch_adapter_cached",
                        "name": label,
                        "checkpoint": str(checkpoint),
                        "mode": mode,
                        "batch_index": batch_idx,
                        "batch_size": len(batch_rows),
                        "visual_tokens": visual_tokens,
                        "text_tokens": text_tokens,
                        **padding_stats,
                        "seconds": adapter_cached_s,
                        "seconds_per_sample": adapter_cached_s / max(1, len(batch_rows)),
                        "speedup_vs_teacher": teacher_s / adapter_cached_s,
                        "cuda_graph": bool(getattr(args, "cuda_graph", False)),
                    }
                )

    print()
    print("=== Qwen3-VL Batched Prefill Benchmark ===")
    print(f"model={args.model_path}")
    padding_waste = float(total_pad_tokens) / float(max(1, total_token_slots))
    print(
        f"samples={len(rows)} batches={len(batches)} batch_size={args.batch_size} "
        f"max_batch_tokens={int(getattr(args, 'max_batch_tokens', 0))} "
        f"bucket_by_length={bool(args.bucket_by_length)} exact_bucket_lengths={bool(getattr(args, 'exact_bucket_lengths', False))} "
        f"padding_waste={padding_waste:.4f} cuda_graph={bool(getattr(args, 'cuda_graph', False))}"
    )
    teacher_avg = sum(teacher_batch_seconds) / max(1, len(teacher_batch_seconds))
    teacher_sample_avg = sum(teacher_sample_seconds) / max(1, len(teacher_sample_seconds))
    print(f"teacher_batch_avg={fmt_ms(teacher_avg)} ms teacher_sample_avg={fmt_ms(teacher_sample_avg)} ms")
    for label, _, _, _ in adapters:
        if adapter_e2e_batch_seconds[label]:
            batch_avg = sum(adapter_e2e_batch_seconds[label]) / max(1, len(adapter_e2e_batch_seconds[label]))
            sample_avg = sum(adapter_e2e_sample_seconds[label]) / max(1, len(adapter_e2e_sample_seconds[label]))
            print(
                f"{label}: adapter_e2e_batch_avg={fmt_ms(batch_avg)} ms "
                f"adapter_e2e_sample_avg={fmt_ms(sample_avg)} ms speedup_vs_teacher_batch={fmt_speedup(teacher_avg, batch_avg)}"
            )
        if adapter_cached_batch_seconds[label]:
            batch_avg = sum(adapter_cached_batch_seconds[label]) / max(1, len(adapter_cached_batch_seconds[label]))
            sample_avg = sum(adapter_cached_sample_seconds[label]) / max(1, len(adapter_cached_sample_seconds[label]))
            print(
                f"{label}: adapter_cached_batch_avg={fmt_ms(batch_avg)} ms "
                f"adapter_cached_sample_avg={fmt_ms(sample_avg)} ms speedup_vs_teacher_batch={fmt_speedup(teacher_avg, batch_avg)} [diagnostic]"
            )

    write_outputs(rows_out, args)
