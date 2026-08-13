#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prefix_visual_memory_by_layer,
    qwen3vl_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose adjacent-layer Qwen3-VL visual K/V correlation.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--modes", default="v0,teacher,prefix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def build_prompt(processor: Any, row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    question = f"{question}\nAnswer directly with only the letter of the correct option."
    return qwen3vl_prompt(processor, question)


@torch.inference_mode()
def prompt_inputs(processor: Any, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def visual_position_ids(full_position_ids: torch.Tensor, image_positions: torch.Tensor) -> torch.Tensor:
    index = image_positions.to(device=full_position_ids.device).unsqueeze(0).expand(full_position_ids.shape[0], -1, -1)
    return torch.gather(full_position_ids, dim=2, index=index)


def qwen_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    visual_hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(visual_hidden)
    hidden_shape = (*normed.shape[:-1], -1, attn.head_dim)
    key_states = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2).contiguous()
    value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2).contiguous()
    _, key_states = apply_rotary_pos_emb(
        key_states,
        key_states,
        *language_model.rotary_emb(normed, position_ids),
    )
    return key_states, value_states


def flatten_valid_heads(x: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    # [B, heads, tokens, dim] -> [valid_tokens, heads * dim]
    x = x.transpose(1, 2).contiguous()
    flat = x.reshape(x.shape[0], x.shape[1], -1).float()
    return flat[image_mask.to(device=x.device).bool()]


def flatten_valid_tokens(x: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    return x.float()[image_mask.to(device=x.device).bool()]


def update_pair_stats(
    stats: dict[str, list[float] | list[int]],
    idx: int,
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    if left.numel() == 0:
        return
    cos = F.cosine_similarity(left, right, dim=-1)
    left_norm = left.norm(dim=-1).clamp_min(1e-8)
    right_norm = right.norm(dim=-1)
    stats["cos_sum"][idx] += float(cos.sum().item())
    stats["norm_ratio_sum"][idx] += float((right_norm / left_norm).sum().item())
    stats["count"][idx] += int(cos.numel())


def init_stats(num_pairs: int) -> dict[str, list[float] | list[int]]:
    return {
        "cos_sum": [0.0 for _ in range(num_pairs)],
        "norm_ratio_sum": [0.0 for _ in range(num_pairs)],
        "count": [0 for _ in range(num_pairs)],
    }


def finalize_stats(stats: dict[str, list[float] | list[int]]) -> dict[str, list[float]]:
    cos_sum = stats["cos_sum"]
    norm_sum = stats["norm_ratio_sum"]
    counts = stats["count"]
    return {
        "adjacent_cos": [float(c) / max(int(n), 1) for c, n in zip(cos_sum, counts)],
        "adjacent_norm_ratio": [float(s) / max(int(n), 1) for s, n in zip(norm_sum, counts)],
        "counts": [int(n) for n in counts],
    }


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "early_mean": float(tensor[: min(8, tensor.numel())].mean().item()),
        "mid_mean": float(tensor[8: min(24, tensor.numel())].mean().item()) if tensor.numel() > 8 else float("nan"),
        "late_mean": float(tensor[24:].mean().item()) if tensor.numel() > 24 else float("nan"),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    allowed = {"v0", "teacher", "prefix"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise ValueError(f"unknown modes: {bad}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    num_pairs = num_layers - 1

    stats = {
        mode: {
            "raw_visual": init_stats(num_pairs),
            "key": init_stats(num_pairs),
            "value": init_stats(num_pairs),
        }
        for mode in modes
    }
    token_counts: list[int] = []

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = None
        if "teacher" in modes:
            teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(
            model,
            inputs,
        )
        _, image_positions, _, _, image_mask, _ = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        vpos = visual_position_ids(full_position_ids, image_positions)
        token_counts.append(int(image_mask.sum().item()))

        memories: dict[str, list[torch.Tensor]] = {}
        v0 = gather_batched_positions(hidden0, image_positions, image_mask).to(dtype=dtype)
        if "v0" in modes:
            memories["v0"] = [v0 for _ in range(num_layers)]
        if "teacher" in modes:
            assert teacher is not None
            memories["teacher"] = [
                gather_batched_positions(state.detach().to(dtype=dtype), image_positions, image_mask)
                for state in teacher.hidden_states[:num_layers]
            ]
        if "prefix" in modes:
            memories["prefix"] = qwen3vl_prefix_visual_memory_by_layer(
                language_model,
                hidden0.to(dtype=dtype),
                full_position_ids,
                inputs["attention_mask"],
                image_positions,
                image_mask,
                visual_pos_masks,
                deepstack_visual_embeds,
            )

        for mode in modes:
            prev_memory = memories[mode][0]
            prev_key, prev_value = qwen_visual_kv(language_model, 0, prev_memory, vpos)
            for layer_idx in range(1, num_layers):
                memory = memories[mode][layer_idx]
                key, value = qwen_visual_kv(language_model, layer_idx, memory, vpos)
                pair_idx = layer_idx - 1
                update_pair_stats(
                    stats[mode]["raw_visual"],
                    pair_idx,
                    flatten_valid_tokens(prev_memory, image_mask),
                    flatten_valid_tokens(memory, image_mask),
                )
                update_pair_stats(
                    stats[mode]["key"],
                    pair_idx,
                    flatten_valid_heads(prev_key, image_mask),
                    flatten_valid_heads(key, image_mask),
                )
                update_pair_stats(
                    stats[mode]["value"],
                    pair_idx,
                    flatten_valid_heads(prev_value, image_mask),
                    flatten_valid_heads(value, image_mask),
                )
                prev_memory = memory
                prev_key = key
                prev_value = value

        if (idx + 1) % 10 == 0:
            print(f"processed {idx + 1}/{len(rows)} avg_image_tokens={sum(token_counts)/len(token_counts):.1f}", flush=True)

    results: dict[str, Any] = {
        "data": args.data,
        "model_path": args.model_path,
        "num_samples": len(rows),
        "modes": modes,
        "image_tokens": {
            "mean": sum(token_counts) / max(len(token_counts), 1),
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        },
        "layers": num_layers,
        "adjacent_pairs": [[idx, idx + 1] for idx in range(num_pairs)],
        "results": {},
    }
    for mode in modes:
        mode_result = {}
        for key in ("raw_visual", "key", "value"):
            finalized = finalize_stats(stats[mode][key])
            finalized["summary"] = summarize(finalized["adjacent_cos"])
            mode_result[key] = finalized
        results["results"][mode] = mode_result

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
