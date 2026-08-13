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
    qwen3vl_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Diagnose whether text hidden deltas can predict layer-specific Qwen visual K/V changes."
    )
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--ridge", type=float, default=1e-3)
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


def mean_cos(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).mean().item())


def norm_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return float("nan")
    denom = left.float().norm(dim=-1).clamp_min(1e-8)
    return float((right.float().norm(dim=-1) / denom).mean().item())


def add_metric(bucket: dict[str, list[float]], name: str, value: float) -> None:
    if value == value:
        bucket.setdefault(name, []).append(float(value))


def summarize_bucket(bucket: dict[str, list[float]]) -> dict[str, float]:
    result = {}
    for key, values in sorted(bucket.items()):
        t = torch.tensor(values, dtype=torch.float32)
        result[key] = float(t.mean().item()) if t.numel() else float("nan")
    return result


def kernel_ridge_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Dual form works when feature dimension is much larger than the sample count.
    x_train = F.normalize(x_train.float(), dim=-1)
    x_eval = F.normalize(x_eval.float(), dim=-1)
    y_train = y_train.float()
    k_train = x_train @ x_train.T
    scale = float(k_train.diag().mean().item()) if k_train.numel() else 1.0
    eye = torch.eye(k_train.shape[0], dtype=k_train.dtype)
    alpha = torch.linalg.solve(k_train + ridge * scale * eye, y_train)
    y_train_hat = k_train @ alpha
    y_eval_hat = (x_eval @ x_train.T) @ alpha
    return y_train_hat, y_eval_hat


@torch.inference_mode()
def collect_features(
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    xs: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    ys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    token_counts: list[int] = []
    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_pos, image_pos, _, text_mask, image_mask, _ = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        token_counts.append(int(image_mask.sum().item()))
        v0 = gather_batched_positions(hidden0.to(dtype=dtype), image_pos, image_mask)
        h0 = gather_batched_positions(teacher.hidden_states[0].detach().to(dtype=dtype), text_pos, text_mask)
        v0_mean = v0.float().mean(dim=1).squeeze(0).cpu()
        h0_mean = h0.float().mean(dim=1).squeeze(0).cpu()
        for layer_idx in range(num_layers):
            state = teacher.hidden_states[layer_idx].detach().to(dtype=dtype)
            hl = gather_batched_positions(state, text_pos, text_mask)
            vl = gather_batched_positions(state, image_pos, image_mask)
            hl_mean = hl.float().mean(dim=1).squeeze(0).cpu()
            target = (vl.float().mean(dim=1).squeeze(0) - v0.float().mean(dim=1).squeeze(0)).cpu()
            feature = torch.cat([v0_mean, hl_mean, hl_mean - h0_mean], dim=0)
            xs[layer_idx].append(feature)
            ys[layer_idx].append(target)
        if (idx + 1) % 10 == 0:
            print(f"collected {idx + 1}/{len(rows)} avg_image_tokens={sum(token_counts)/len(token_counts):.1f}", flush=True)
    return [torch.stack(v, dim=0) for v in xs], [torch.stack(v, dim=0) for v in ys], token_counts


@torch.inference_mode()
def evaluate_mapping(
    rows: list[dict[str, Any]],
    y_hat_by_layer: list[torch.Tensor],
    y_zero_by_layer: list[torch.Tensor],
    eval_offset: int,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    per_layer: list[dict[str, list[float]]] = [dict() for _ in range(num_layers)]
    for local_idx, row in enumerate(rows):
        sample_idx = eval_offset + local_idx
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        _, image_pos, _, _, image_mask, _ = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        vpos = visual_position_ids(full_position_ids, image_pos)
        v0 = gather_batched_positions(hidden0.to(dtype=dtype), image_pos, image_mask)
        for layer_idx in range(num_layers):
            vl = gather_batched_positions(
                teacher.hidden_states[layer_idx].detach().to(dtype=dtype),
                image_pos,
                image_mask,
            )
            shift = y_hat_by_layer[layer_idx][sample_idx].to(device=device, dtype=dtype).view(1, 1, -1)
            zero = y_zero_by_layer[layer_idx][sample_idx].to(device=device, dtype=dtype).view(1, 1, -1)
            v_pred = v0 + shift
            v_zero = v0 + zero

            target_key, target_value = qwen_visual_kv(language_model, layer_idx, vl, vpos)
            native_zero_key, native_zero_value = qwen_visual_kv(language_model, layer_idx, v_zero, vpos)
            native_pred_key, native_pred_value = qwen_visual_kv(language_model, layer_idx, v_pred, vpos)
            shared_zero_key, shared_zero_value = qwen_visual_kv(language_model, 0, v_zero, vpos)
            shared_pred_key, shared_pred_value = qwen_visual_kv(language_model, 0, v_pred, vpos)

            target_tokens = flatten_valid_tokens(vl, image_mask)
            zero_tokens = flatten_valid_tokens(v_zero, image_mask)
            pred_tokens = flatten_valid_tokens(v_pred, image_mask)
            target_k = flatten_valid_heads(target_key, image_mask)
            target_v = flatten_valid_heads(target_value, image_mask)
            native_zero_k = flatten_valid_heads(native_zero_key, image_mask)
            native_zero_v = flatten_valid_heads(native_zero_value, image_mask)
            native_pred_k = flatten_valid_heads(native_pred_key, image_mask)
            native_pred_v = flatten_valid_heads(native_pred_value, image_mask)
            shared_zero_k = flatten_valid_heads(shared_zero_key, image_mask)
            shared_zero_v = flatten_valid_heads(shared_zero_value, image_mask)
            shared_pred_k = flatten_valid_heads(shared_pred_key, image_mask)
            shared_pred_v = flatten_valid_heads(shared_pred_value, image_mask)

            bucket = per_layer[layer_idx]
            add_metric(bucket, "raw_v0_cos", mean_cos(target_tokens, zero_tokens))
            add_metric(bucket, "raw_pred_cos", mean_cos(target_tokens, pred_tokens))
            add_metric(bucket, "raw_pred_norm_ratio", norm_ratio(target_tokens, pred_tokens))
            add_metric(bucket, "native_layer_v0_key_cos", mean_cos(target_k, native_zero_k))
            add_metric(bucket, "native_layer_pred_key_cos", mean_cos(target_k, native_pred_k))
            add_metric(bucket, "native_layer_v0_value_cos", mean_cos(target_v, native_zero_v))
            add_metric(bucket, "native_layer_pred_value_cos", mean_cos(target_v, native_pred_v))
            add_metric(bucket, "shared_l0_v0_key_cos", mean_cos(target_k, shared_zero_k))
            add_metric(bucket, "shared_l0_pred_key_cos", mean_cos(target_k, shared_pred_k))
            add_metric(bucket, "shared_l0_v0_value_cos", mean_cos(target_v, shared_zero_v))
            add_metric(bucket, "shared_l0_pred_value_cos", mean_cos(target_v, shared_pred_v))
        print(f"evaluated {local_idx + 1}/{len(rows)}", flush=True)
    layer_results = [summarize_bucket(bucket) for bucket in per_layer]
    summary: dict[str, float] = {}
    for key in layer_results[0]:
        vals = torch.tensor([row[key] for row in layer_results], dtype=torch.float32)
        summary[f"{key}_mean"] = float(vals.mean().item())
        summary[f"{key}_early"] = float(vals[: min(8, vals.numel())].mean().item())
        summary[f"{key}_mid"] = float(vals[8: min(24, vals.numel())].mean().item()) if vals.numel() > 8 else float("nan")
        summary[f"{key}_late"] = float(vals[24:].mean().item()) if vals.numel() > 24 else float("nan")
    return {"summary": summary, "per_layer": layer_results}


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    if len(rows) < 4:
        raise ValueError("need at least 4 samples for train/eval split")
    split = max(1, min(len(rows) - 1, int(round(len(rows) * args.train_fraction))))
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    num_layers = len(get_language_model(model).layers)

    xs, ys, token_counts = collect_features(rows, processor, model, device, dtype)
    yhat_layers: list[torch.Tensor] = []
    yzero_layers: list[torch.Tensor] = []
    fit_results = []
    for layer_idx in range(num_layers):
        x = xs[layer_idx]
        y = ys[layer_idx]
        y_train_hat, y_eval_hat = kernel_ridge_predict(x[:split], y[:split], x[split:], args.ridge)
        yhat = torch.cat([y_train_hat, y_eval_hat], dim=0)
        zero = torch.zeros_like(yhat)
        yhat_layers.append(yhat)
        yzero_layers.append(zero)
        fit_results.append(
            {
                "layer": layer_idx,
                "train_shift_cos": mean_cos(y[:split], y_train_hat),
                "eval_shift_cos": mean_cos(y[split:], y_eval_hat),
                "train_shift_norm_ratio": norm_ratio(y[:split], y_train_hat),
                "eval_shift_norm_ratio": norm_ratio(y[split:], y_eval_hat),
            }
        )
    eval_results = evaluate_mapping(
        rows[split:],
        yhat_layers,
        yzero_layers,
        split,
        processor,
        model,
        device,
        dtype,
    )
    fit_summary: dict[str, float] = {}
    for key in ("train_shift_cos", "eval_shift_cos", "train_shift_norm_ratio", "eval_shift_norm_ratio"):
        vals = torch.tensor([row[key] for row in fit_results], dtype=torch.float32)
        fit_summary[f"{key}_mean"] = float(vals.mean().item())
        fit_summary[f"{key}_early"] = float(vals[: min(8, vals.numel())].mean().item())
        fit_summary[f"{key}_mid"] = float(vals[8: min(24, vals.numel())].mean().item()) if vals.numel() > 8 else float("nan")
        fit_summary[f"{key}_late"] = float(vals[24:].mean().item()) if vals.numel() > 24 else float("nan")
    results: dict[str, Any] = {
        "data": args.data,
        "model_path": args.model_path,
        "num_samples": len(rows),
        "train_samples": split,
        "eval_samples": len(rows) - split,
        "ridge": args.ridge,
        "image_tokens": {
            "mean": sum(token_counts) / max(len(token_counts), 1),
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        },
        "feature": "concat(mean(V0), mean(H_l), mean(H_l-H_0))",
        "target_shift": "mean(V_l^teacher - V0)",
        "shared_kv_probe": "compare layer-0 native visual K/V on V0+predicted_shift against layer-l native visual K/V on teacher V_l",
        "fit_summary": fit_summary,
        "fit_per_layer": fit_results,
        "kv_eval": eval_results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
