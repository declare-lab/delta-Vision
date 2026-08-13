#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt,
    load_sidecar,
    no_visual_logits,
    qwen_logits,
    sidecar_logits,
)
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
)
from PIL import Image


def bytes_to_mib(value: float) -> float:
    return float(value) / 1024.0 / 1024.0


def dtype_bytes(dtype: torch.dtype) -> int:
    return {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }[dtype]


def cuda_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def count_prompt_tokens(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
) -> dict[str, int]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0, position_ids, _visual_pos_masks, _deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, _text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    del hidden0, position_ids, text_pos, image_pos
    return {
        "full_tokens": int(full_mask.sum().item()),
        "text_tokens": int(text_mask.sum().item()),
        "image_tokens": int(image_mask.sum().item()),
    }


def qwen_visual_kv_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_layers = len(language_model.layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return num_layers * int(image_tokens) * 2 * num_kv_heads * head_dim * dtype_bytes(dtype)


def sidecar_static_visual_kv_bytes(sidecar: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    sidecar_dim = int(getattr(sidecar, "sidecar_dim"))
    return int(image_tokens) * 2 * sidecar_dim * dtype_bytes(dtype)


def qwen_native_sidecar_temp_kv_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_query_heads = int(getattr(config, "num_attention_heads"))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(image_tokens) * 2 * num_query_heads * head_dim * dtype_bytes(dtype)


def qwen_native_one_layer_visual_kv_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(image_tokens) * 2 * num_kv_heads * head_dim * dtype_bytes(dtype)


def sidecar_layer_kv_adapter_param_bytes(sidecar: torch.nn.Module) -> int:
    names = (
        "visual_adapter_down",
        "visual_adapter_up",
        "visual_depth_embed",
        "visual_recurrent",
        "visual_stage_down",
        "visual_stage_up",
        "visual_stage_film",
    )
    total = 0
    for name, param in sidecar.named_parameters():
        if any(name.startswith(prefix) for prefix in names):
            total += param.numel() * param.element_size()
    return int(total)


def sidecar_total_param_bytes(sidecar: torch.nn.Module) -> int:
    return int(sum(param.numel() * param.element_size() for param in sidecar.parameters()))


def hidden_memory_bytes(num_layers: int, image_tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
    return int(num_layers) * int(image_tokens) * int(hidden_size) * dtype_bytes(dtype)


def measure_peak(
    name: str,
    device: torch.device,
    fn: Callable[[], torch.Tensor],
) -> dict[str, Any]:
    cuda_reset(device)
    start = time.perf_counter()
    logits = fn()
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    return {
        "mode": name,
        "elapsed_sec": elapsed,
        "max_allocated_mib": bytes_to_mib(allocated),
        "max_reserved_mib": bytes_to_mib(reserved),
        "logit_rms": float(logits.float().pow(2).mean().sqrt().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Measure Qwen visual KV cache accounting and prompt-logit peak memory.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), default="mmstar")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="artifacts/bench/qwen_visual_kv_memory.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--visual-memory-mode", default="v0")
    parser.add_argument("--skip-peak", action="store_true", help="Only compute accounting; skip prompt-logit peak runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.row_index + max(1, args.max_samples))[args.row_index : args.row_index + max(1, args.max_samples)]
    row = rows[0]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)

    sidecar_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        rank=512,
        sidecar_dim=1024,
        num_heads=8,
        state_tokens=0,
        reader_mlp_ratio=4.0,
        reader_activation="gelu",
        layer_adapter_rank=128,
        shared_basis=True,
        output_mode="residual",
        visual_memory_mode=args.visual_memory_mode,
        sidecar_scale=args.sidecar_scale,
        use_rope=False,
        sidecar_query_source="sidecar",
        sidecar_visual_kv_source="sidecar",
        factorized_mass_mode="learned",
        fixed_visual_mass=0.12,
    )
    sidecar = load_sidecar(sidecar_args, int(language_model.config.hidden_size), len(language_model.layers), device, dtype)

    sample_accounting = []
    for offset, sample_row in enumerate(rows):
        token_counts = count_prompt_tokens(processor, model, sample_row, args.benchmark, device)
        image_tokens = token_counts["image_tokens"]
        qwen_one_layer_kv_bytes = qwen_native_one_layer_visual_kv_bytes(language_model, image_tokens, dtype)
        qwen_vis_bytes = qwen_visual_kv_bytes(language_model, image_tokens, dtype)
        qwen_native_sidecar_temp_bytes = qwen_native_sidecar_temp_kv_bytes(language_model, image_tokens, dtype)
        sidecar_vis_bytes = sidecar_static_visual_kv_bytes(sidecar, image_tokens, dtype)
        persistent_v0_bytes = hidden_memory_bytes(1, image_tokens, int(language_model.config.hidden_size), dtype)
        layerwise_visual_memory_bytes = hidden_memory_bytes(
            len(language_model.layers),
            image_tokens,
            int(language_model.config.hidden_size),
            dtype,
        )
        one_layer_visual_memory_bytes = hidden_memory_bytes(
            1,
            image_tokens,
            int(language_model.config.hidden_size),
            dtype,
        )
        layerwise_sidecar_kv_bytes = qwen_native_sidecar_temp_bytes * len(language_model.layers)
        sample_accounting.append(
            {
                "row_index": args.row_index + offset,
                "image": sample_row.get("image"),
                "token_counts": token_counts,
                "qwen_main_visual_kv_cache_mib_per_request": bytes_to_mib(qwen_vis_bytes),
                "qwen_main_one_layer_visual_kv_mib": bytes_to_mib(qwen_one_layer_kv_bytes),
                "sidecar_persistent_v0_hidden_mib_per_request": bytes_to_mib(persistent_v0_bytes),
                "sidecar_single_layer_temp_visual_kv_mib_qwen_native_repeated_heads": bytes_to_mib(qwen_native_sidecar_temp_bytes),
                "sidecar_single_layer_temp_visual_kv_mib_sidecar_projection": bytes_to_mib(sidecar_vis_bytes),
                "layer_kv_adapter_one_layer_v_l_temp_mib": bytes_to_mib(one_layer_visual_memory_bytes),
                "layer_kv_adapter_cache_all_v_l_mib_if_materialized": bytes_to_mib(layerwise_visual_memory_bytes),
                "cache_all_external_qwen_native_sidecar_kv_l_mib_if_materialized": bytes_to_mib(layerwise_sidecar_kv_bytes),
                "saved_persistent_request_mib_vs_qwen_main_visual_kv": bytes_to_mib(qwen_vis_bytes - persistent_v0_bytes),
                "cached_external_qwen_native_kv_minus_qwen_main_visual_kv_mib": bytes_to_mib(layerwise_sidecar_kv_bytes - qwen_vis_bytes),
                "qwen_main_visual_kv_over_persistent_v0": qwen_vis_bytes / max(persistent_v0_bytes, 1),
                "cached_external_qwen_native_kv_over_qwen_main_visual_kv": layerwise_sidecar_kv_bytes / max(qwen_vis_bytes, 1),
            }
        )

    token_counts = sample_accounting[0]["token_counts"]
    qwen_vis_bytes = int(sample_accounting[0]["qwen_main_visual_kv_cache_mib_per_request"] * 1024 * 1024)
    sidecar_vis_bytes = int(sample_accounting[0]["sidecar_single_layer_temp_visual_kv_mib_sidecar_projection"] * 1024 * 1024)
    layerwise_visual_memory_bytes = hidden_memory_bytes(
        len(language_model.layers),
        token_counts["image_tokens"],
        int(language_model.config.hidden_size),
        dtype,
    )
    one_layer_visual_memory_bytes = hidden_memory_bytes(
        1,
        token_counts["image_tokens"],
        int(language_model.config.hidden_size),
        dtype,
    )
    layerwise_sidecar_kv_bytes = sidecar_vis_bytes * len(language_model.layers)
    layer_adapter_bytes = sidecar_layer_kv_adapter_param_bytes(sidecar)
    sidecar_param_bytes = sidecar_total_param_bytes(sidecar)

    measurements = []
    if not args.skip_peak:
        measurements = [
            measure_peak(
                "qwen_full_prompt_logits_no_cache",
                device,
                lambda: qwen_logits(processor, model, row, args.benchmark, device),
            ),
            measure_peak(
                "no_visual_prompt_logits_no_cache",
                device,
                lambda: no_visual_logits(processor, model, language_model, row, args.benchmark, device, dtype),
            ),
            measure_peak(
                "sidecar_only_prompt_logits_no_cache",
                device,
                lambda: sidecar_logits(
                    processor,
                    model,
                    language_model,
                    sidecar,
                    row,
                    args.benchmark,
                    device,
                    dtype,
                    args.sidecar_scale,
                    sidecar_args.visual_memory_mode,
                    sidecar_args.sidecar_query_source,
                    sidecar_args.sidecar_visual_kv_source,
                    sidecar_args.factorized_mass_mode,
                    sidecar_args.fixed_visual_mass,
                ),
            ),
        ]

    avg = {}
    if sample_accounting:
        numeric_keys = [
            key
            for key, value in sample_accounting[0].items()
            if isinstance(value, (int, float)) and key != "row_index"
        ]
        for key in numeric_keys:
            avg[key] = sum(float(item[key]) for item in sample_accounting) / len(sample_accounting)

    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "row_index": args.row_index,
        "max_samples": args.max_samples,
        "image": row.get("image"),
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "token_counts": token_counts,
        "sample_accounting": sample_accounting,
        "average_accounting": avg,
        "visual_kv_accounting": {
            "qwen_main_visual_kv_cache_mib_per_request": bytes_to_mib(qwen_vis_bytes),
            "sidecar_static_visual_kv_cache_mib_per_request": bytes_to_mib(sidecar_vis_bytes),
            "layer_kv_adapter_one_layer_v_l_temp_mib": bytes_to_mib(one_layer_visual_memory_bytes),
            "layer_kv_adapter_cache_all_v_l_mib_if_materialized": bytes_to_mib(layerwise_visual_memory_bytes),
            "layer_kv_adapter_cache_all_sidecar_kv_l_mib_if_materialized": bytes_to_mib(layerwise_sidecar_kv_bytes),
            "sidecar_layer_kv_adapter_param_mib": bytes_to_mib(layer_adapter_bytes),
            "sidecar_total_param_mib": bytes_to_mib(sidecar_param_bytes),
            "saved_visual_kv_mib_per_request_vs_qwen": bytes_to_mib(qwen_vis_bytes - sidecar_vis_bytes),
            "saved_visual_kv_mib_if_cache_all_sidecar_kv_l_vs_qwen": bytes_to_mib(qwen_vis_bytes - layerwise_sidecar_kv_bytes),
            "qwen_visual_kv_over_sidecar_static_visual_kv": qwen_vis_bytes / max(sidecar_vis_bytes, 1),
            "qwen_visual_kv_over_cached_all_sidecar_kv_l": qwen_vis_bytes / max(layerwise_sidecar_kv_bytes, 1),
            "break_even_concurrent_requests_for_total_sidecar_params": sidecar_param_bytes
            / max(qwen_vis_bytes - sidecar_vis_bytes, 1),
        },
        "measured_peak_memory": measurements,
        "note": (
            "Prompt-logit measurements use no generation KV cache because MMStar scoring is prompt-logit based. "
            "The sample_accounting section is the decode-time visual KV cache accounting. "
            "Current Qwen learned sidecar with qwen_native visual K/V can discard the main LLM visual KV cache only "
            "if external visual K/V is generated per layer and discarded. If all external per-layer K/V is cached, "
            "the repeated query-head layout is larger than Qwen's native GQA visual KV cache."
        ),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
