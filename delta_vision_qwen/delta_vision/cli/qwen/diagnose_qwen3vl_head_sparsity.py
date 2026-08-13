#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prompt,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)
from delta_vision.cli.qwen.diagnose_qwen3vl_factorized_deep import (
    causal_valid_mask,
    gather_heads,
    gather_probs_keys,
    native_qkv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose Qwen3-VL visual head sparsity on MMStar.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--importance-samples", type=int, default=200)
    parser.add_argument("--top-fracs", default="0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--importance", choices=("delta_rms", "mass_mean", "mass_delta"), default="delta_rms")
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


def parse_top_fracs(spec: str) -> list[float]:
    vals = sorted({float(x) for x in spec.split(",") if x.strip()})
    if not vals:
        raise ValueError("--top-fracs cannot be empty")
    if vals[0] <= 0 or vals[-1] > 1:
        raise ValueError("--top-fracs entries must be in (0, 1]")
    return vals


def update_metrics(metrics: dict[str, float | int], pred: str, teacher_pred: str, gold: str, kl: float) -> None:
    metrics["correct"] = int(metrics["correct"]) + int(pred == gold)
    metrics["agree"] = int(metrics["agree"]) + int(pred == teacher_pred)
    metrics["teacher_correct_and_agree"] = int(metrics["teacher_correct_and_agree"]) + int(
        teacher_pred == gold and pred == teacher_pred
    )
    metrics["output_kl_sum"] = float(metrics["output_kl_sum"]) + float(kl)
    metrics["scored"] = int(metrics["scored"]) + 1


def finalize_metrics(metrics: dict[str, float | int], teacher_correct: int) -> dict[str, float | int]:
    scored = max(int(metrics["scored"]), 1)
    teacher_correct = max(int(teacher_correct), 1)
    return {
        "scored": int(metrics["scored"]),
        "correct": int(metrics["correct"]),
        "accuracy": float(metrics["correct"]) / scored,
        "teacher_agreement": float(metrics["agree"]) / scored,
        "teacher_correct_retention": float(metrics["teacher_correct_and_agree"]) / teacher_correct,
        "output_kl": float(metrics["output_kl_sum"]) / scored,
    }


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=x.device, dtype=torch.bool)
    while valid.ndim < x.ndim:
        valid = valid.unsqueeze(-1)
    return x.masked_select(valid.expand_as(x)).float().mean()


@torch.inference_mode()
def teacher_head_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden: torch.Tensor,
    text_hidden: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    full_mask: torch.Tensor,
    text_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-o visual delta per head and visual mass.

    head_delta: [B, heads, text_len, head_dim]
    mass: [B, heads, text_len]
    """
    full_effect_state = scatter_batched_positions(full_hidden, text_positions, text_hidden, text_mask)
    q_full, k_full, v_full = native_qkv(language_model, layer_idx, full_effect_state, full_position_ids)
    attn = language_model.layers[layer_idx].self_attn
    k_vis = gather_heads(k_full, image_positions, image_mask)
    v_vis = gather_heads(v_full, image_positions, image_mask)
    v_text = gather_heads(v_full, text_positions, text_mask)

    scores = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) * float(attn.scaling)
    scores = scores.masked_fill(~causal_valid_mask(full_mask), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    query_idx = text_positions.to(device=probs.device, dtype=torch.long)[:, None, :, None].expand(
        probs.shape[0], probs.shape[1], text_positions.shape[1], probs.shape[-1]
    )
    probs_text_queries = torch.gather(probs, dim=2, index=query_idx)
    image_probs = gather_probs_keys(probs_text_queries, image_positions) * image_mask[:, None, None, :].to(
        dtype=probs_text_queries.dtype
    )
    text_probs = gather_probs_keys(probs_text_queries, text_positions) * text_mask[:, None, None, :].to(
        dtype=probs_text_queries.dtype
    )
    m_vis = image_probs.sum(dim=-1).clamp_min(0.0)
    m_text = text_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    avis = torch.matmul(image_probs / m_vis.unsqueeze(-1).clamp_min(1e-12), v_vis.float())
    atext = torch.matmul(text_probs / m_text, v_text.float())
    head_delta = m_vis.unsqueeze(-1) * (avis - atext)
    head_delta = head_delta * text_mask[:, None, :, None].to(device=head_delta.device, dtype=head_delta.dtype)
    return head_delta, m_vis * text_mask[:, None, :].to(device=m_vis.device, dtype=m_vis.dtype)


def head_delta_to_hidden(
    language_model: torch.nn.Module,
    layer_idx: int,
    head_delta: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    attn = language_model.layers[layer_idx].self_attn
    merged = head_delta.transpose(1, 2).reshape(head_delta.shape[0], head_delta.shape[2], -1)
    return attn.o_proj(merged.to(dtype=dtype))


@torch.inference_mode()
def collect_importance(
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    num_layers = len(language_model.layers)
    num_heads = int(language_model.config.num_attention_heads)
    mass_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    delta_sq_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    delta_abs_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    count = torch.zeros(num_layers, num_heads, dtype=torch.float64)

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, _, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
            for state in teacher_states
        ]
        valid_tokens = float(text_mask.sum().item())
        for layer_idx in range(num_layers):
            head_delta, mass = teacher_head_delta(
                language_model,
                layer_idx,
                teacher_states[layer_idx],
                teacher_text_states[layer_idx],
                full_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            mass_sum[layer_idx] += mass.float().sum(dim=(0, 2)).double().cpu()
            delta_sq_sum[layer_idx] += head_delta.float().pow(2).sum(dim=(0, 2, 3)).double().cpu()
            delta_abs_sum[layer_idx] += head_delta.float().abs().sum(dim=(0, 2, 3)).double().cpu()
            count[layer_idx] += valid_tokens
        if (idx + 1) % 25 == 0:
            print(f"importance {idx + 1}/{len(rows)}", flush=True)

    head_dim = int(language_model.layers[0].self_attn.head_dim)
    denom = count.clamp_min(1.0)
    mass_mean = mass_sum / denom
    delta_rms = (delta_sq_sum / (denom * head_dim)).sqrt()
    delta_abs = delta_abs_sum / (denom * head_dim)
    mass_delta = mass_mean * delta_rms
    return {
        "mass_mean": mass_mean.tolist(),
        "delta_rms": delta_rms.tolist(),
        "delta_abs": delta_abs.tolist(),
        "mass_delta": mass_delta.tolist(),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def build_head_masks(importance_payload: dict[str, Any], top_fracs: list[float], score_name: str, device: torch.device) -> dict[str, torch.Tensor]:
    scores = torch.tensor(importance_payload[score_name], dtype=torch.float32, device=device)
    num_layers, num_heads = scores.shape
    masks: dict[str, torch.Tensor] = {}
    for frac in top_fracs:
        k = max(1, int(round(num_heads * frac)))
        top_idx = torch.topk(scores, k=k, dim=1).indices
        mask = torch.zeros(num_layers, num_heads, dtype=torch.bool, device=device)
        mask.scatter_(1, top_idx, True)
        masks[f"top_{int(round(frac * 100))}pct"] = mask
    return masks


@torch.inference_mode()
def evaluate_top_head_oracles(
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    head_masks: dict[str, torch.Tensor],
    predictions_jsonl: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    option_ids = option_token_id_lists(processor.tokenizer)
    num_layers = len(language_model.layers)
    metrics = {
        name: {"correct": 0, "agree": 0, "teacher_correct_and_agree": 0, "output_kl_sum": 0.0, "scored": 0}
        for name in ("no_visual", "full_heads", *head_masks.keys())
    }
    teacher_correct = 0
    predictions: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
            for state in teacher_states
        ]
        head_deltas = [
            teacher_head_delta(
                language_model,
                layer_idx,
                teacher_states[layer_idx],
                teacher_text_states[layer_idx],
                full_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )[0]
            for layer_idx in range(num_layers)
        ]
        last_text_idx = int(text_mask[0].sum().item()) - 1
        teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_correct += int(teacher_pred == gold)
        sample_pred = {"index": row.get("index", idx), "gold": gold, "teacher": teacher_pred}

        for name in metrics:
            h = teacher_text_states[0]
            for layer_idx in range(num_layers):
                delta = None
                if name != "no_visual":
                    head_delta = head_deltas[layer_idx]
                    if name != "full_heads":
                        mask = head_masks[name][layer_idx].to(device=head_delta.device, dtype=head_delta.dtype)
                        head_delta = head_delta * mask.view(1, -1, 1, 1)
                    delta = head_delta_to_hidden(language_model, layer_idx, head_delta, dtype)
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    padding_mask=~text_mask,
                )
            logits = model.lm_head(language_model.norm(h))[0, last_text_idx]
            pred = predict_option(logits, option_ids)
            dist = option_distribution(logits, option_ids)
            kl = F.kl_div(dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
            update_metrics(metrics[name], pred, teacher_pred, gold, kl)
            sample_pred[name] = pred
        predictions.append(sample_pred)
        if (idx + 1) % 25 == 0:
            print(f"oracle {idx + 1}/{len(rows)} teacher_correct={teacher_correct}", flush=True)

    if predictions_jsonl:
        out = Path(predictions_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    results = {
        "teacher": {
            "correct": teacher_correct,
            "accuracy": teacher_correct / max(len(rows), 1),
        },
        "oracles": {name: finalize_metrics(metrics[name], teacher_correct) for name in metrics},
    }
    return results, predictions


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    top_fracs = parse_top_fracs(args.top_fracs)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)

    importance_rows = rows[: min(args.importance_samples, len(rows))]
    print(f"collecting head importance on {len(importance_rows)} samples", flush=True)
    importance = collect_importance(importance_rows, processor, model, language_model, device, dtype)
    head_masks = build_head_masks(importance, top_fracs, args.importance, device)
    print(f"evaluating top-head oracles on {len(rows)} samples", flush=True)
    oracle_results, _ = evaluate_top_head_oracles(
        rows,
        processor,
        model,
        language_model,
        device,
        dtype,
        head_masks,
        args.predictions_jsonl,
    )

    scores = torch.tensor(importance[args.importance], dtype=torch.float32)
    top_heads = {}
    for layer_idx in range(scores.shape[0]):
        order = torch.argsort(scores[layer_idx], descending=True).tolist()
        top_heads[str(layer_idx)] = order
    payload = {
        "benchmark": "mmstar",
        "data": args.data,
        "model_path": args.model_path,
        "max_samples": len(rows),
        "importance_samples": len(importance_rows),
        "importance_score": args.importance,
        "top_fracs": top_fracs,
        "head_importance": importance,
        "top_heads_by_layer": top_heads,
        **oracle_results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
