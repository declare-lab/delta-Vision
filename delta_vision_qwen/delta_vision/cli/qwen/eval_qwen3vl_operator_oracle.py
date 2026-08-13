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
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prompt,
    run_qwen3vl_layer_text_with_attention_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL image-conditioned effect-operator oracles.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--modes",
        default="no_visual,full_target,constant,film,linear",
        help="Comma-separated modes. Also supports linear_rank_<r>, e.g. linear_rank_32.",
    )
    parser.add_argument(
        "--fit-token-split",
        choices=("all", "first_half", "even"),
        default="all",
        help="Token subset used to fit per-image operators.",
    )
    parser.add_argument(
        "--metric-token-split",
        choices=("all", "fit", "heldout"),
        default="all",
        help="Token subset used to report effect metrics. Rollout always applies the fitted operator to all text tokens.",
    )
    parser.add_argument(
        "--effect-input",
        choices=("rollout", "teacher"),
        default="rollout",
        help="Hidden states used when reporting effect metrics. Rollout quality is always evaluated with rollout states.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def build_prompt(processor: Any, row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    return qwen3vl_prompt(processor, f"{question}\nAnswer directly with only the letter of the correct option.")


def masked_metric(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.to(device=pred.device).bool()
    pred_v = pred.float()[valid]
    target_v = target.float()[valid]
    if pred_v.numel() == 0:
        return {"cos": 0.0, "nmse": 0.0, "norm_ratio": 0.0}
    cos = F.cosine_similarity(pred_v, target_v, dim=-1, eps=1e-6).mean()
    nmse = (pred_v - target_v).pow(2).sum() / target_v.pow(2).sum().clamp_min(1e-6)
    pred_rms = pred_v.pow(2).mean().sqrt()
    target_rms = target_v.pow(2).mean().sqrt()
    return {
        "cos": float(cos.item()),
        "nmse": float(nmse.item()),
        "norm_ratio": float((pred_rms / target_rms.clamp_min(1e-6)).item()),
    }


def constant_operator_delta(
    teacher_h: torch.Tensor,
    target_delta: torch.Tensor,
    rollout_h: torch.Tensor,
    fit_mask: torch.Tensor,
    output_mask: torch.Tensor,
) -> torch.Tensor:
    del teacher_h
    fit_valid = fit_mask.to(device=target_delta.device).bool()
    output_valid = output_mask.to(device=target_delta.device).bool()
    delta = torch.zeros((rollout_h.shape[0], target_delta.shape[-1]), device=rollout_h.device, dtype=rollout_h.dtype)
    if bool(fit_valid.any()) and bool(output_valid.any()):
        bias = target_delta.float()[fit_valid].mean(dim=0)
        delta[output_valid] = bias.to(dtype=target_delta.dtype)
    return delta.to(device=rollout_h.device, dtype=rollout_h.dtype)


def film_operator_delta(
    teacher_h: torch.Tensor,
    target_delta: torch.Tensor,
    rollout_h: torch.Tensor,
    fit_mask: torch.Tensor,
    output_mask: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    fit_valid = fit_mask.to(device=teacher_h.device).bool()
    output_valid = output_mask.to(device=teacher_h.device).bool()
    out = torch.zeros((rollout_h.shape[0], target_delta.shape[-1]), device=rollout_h.device, dtype=rollout_h.dtype)
    if not bool(fit_valid.any()) or not bool(output_valid.any()):
        return out
    x = teacher_h.float()[fit_valid]
    y = target_delta.float()[fit_valid]
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    x_center = x - x_mean
    y_center = y - y_mean
    gamma = (x_center * y_center).sum(dim=0) / (x_center.pow(2).sum(dim=0) + float(ridge))
    beta = y_mean - gamma * x_mean
    pred = rollout_h.float()[output_valid] * gamma + beta
    out[output_valid] = pred.to(dtype=out.dtype)
    return out


def linear_operator_delta(
    teacher_h: torch.Tensor,
    target_delta: torch.Tensor,
    rollout_h: torch.Tensor,
    fit_mask: torch.Tensor,
    output_mask: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    fit_valid = fit_mask.to(device=teacher_h.device).bool()
    output_valid = output_mask.to(device=teacher_h.device).bool()
    out = torch.zeros((rollout_h.shape[0], target_delta.shape[-1]), device=rollout_h.device, dtype=rollout_h.dtype)
    if not bool(fit_valid.any()) or not bool(output_valid.any()):
        return out
    x = teacher_h.float()[fit_valid]
    y = target_delta.float()[fit_valid]
    x_new = rollout_h.float()[output_valid]
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=-1)
    x_new_aug = torch.cat([x_new, torch.ones((x_new.shape[0], 1), device=x_new.device, dtype=x_new.dtype)], dim=-1)
    gram = x_aug @ x_aug.transpose(0, 1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    system = gram + float(ridge) * eye
    try:
        alpha = torch.linalg.solve(system, y)
    except RuntimeError:
        alpha = torch.linalg.pinv(system) @ y
    pred = (x_new_aug @ x_aug.transpose(0, 1)) @ alpha
    out[output_valid] = pred.to(dtype=out.dtype)
    return out


def lowrank_linear_operator_delta(
    teacher_h: torch.Tensor,
    target_delta: torch.Tensor,
    rollout_h: torch.Tensor,
    fit_mask: torch.Tensor,
    output_mask: torch.Tensor,
    ridge: float,
    rank: int,
) -> torch.Tensor:
    fit_valid = fit_mask.to(device=teacher_h.device).bool()
    output_valid = output_mask.to(device=teacher_h.device).bool()
    out = torch.zeros((rollout_h.shape[0], target_delta.shape[-1]), device=rollout_h.device, dtype=rollout_h.dtype)
    if not bool(fit_valid.any()) or not bool(output_valid.any()):
        return out
    x = teacher_h.float()[fit_valid]
    y = target_delta.float()[fit_valid]
    x_new = rollout_h.float()[output_valid]
    x_mean = x.mean(dim=0, keepdim=True)
    x_center = x - x_mean
    x_new_center = x_new - x_mean
    effective_rank = min(int(rank), x.shape[0], x.shape[1])
    if effective_rank <= 0:
        return constant_operator_delta(teacher_h, target_delta, rollout_h, fit_mask, output_mask)
    _, _s, vh = torch.linalg.svd(x_center, full_matrices=False)
    basis = vh[:effective_rank].transpose(0, 1).contiguous()
    z = x_center @ basis
    z_new = x_new_center @ basis
    ones = torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
    z_aug = torch.cat([z, ones], dim=-1)
    z_new_aug = torch.cat([z_new, torch.ones((z_new.shape[0], 1), device=z_new.device, dtype=z_new.dtype)], dim=-1)
    gram = z_aug.transpose(0, 1) @ z_aug
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    system = gram + float(ridge) * eye
    rhs = z_aug.transpose(0, 1) @ y
    try:
        weight = torch.linalg.solve(system, rhs)
    except RuntimeError:
        weight = torch.linalg.pinv(system) @ rhs
    pred = z_new_aug @ weight
    out[output_valid] = pred.to(dtype=out.dtype)
    return out


def parse_mode(mode: str) -> tuple[str, int | None]:
    if mode.startswith("linear_rank_"):
        return "linear_rank", int(mode.rsplit("_", 1)[1])
    return mode, None


def token_split_mask(mask: torch.Tensor, split: str, fit_split: str | None = None) -> torch.Tensor:
    valid = mask.bool()
    if split == "all":
        return valid
    positions = torch.arange(valid.shape[0], device=valid.device)
    valid_positions = positions[valid]
    selected = torch.zeros_like(valid)
    if valid_positions.numel() == 0:
        return selected
    if split == "first_half":
        keep = valid_positions[: max(1, int(valid_positions.numel()) // 2)]
    elif split == "even":
        keep = valid_positions[::2]
    elif split == "fit":
        if fit_split is None:
            raise ValueError("fit_split is required for split='fit'")
        return token_split_mask(mask, fit_split)
    elif split == "heldout":
        if fit_split is None:
            raise ValueError("fit_split is required for split='heldout'")
        return valid & ~token_split_mask(mask, fit_split)
    else:
        raise ValueError(f"unsupported token split: {split}")
    selected[keep] = True
    return selected


def update_metric(
    metrics: dict[str, dict[str, float | int]],
    name: str,
    pred: str,
    teacher_pred: str,
    gold: str,
    dist: torch.Tensor,
    teacher_dist: torch.Tensor,
) -> None:
    row = metrics[name]
    row["correct"] = int(row["correct"]) + int(pred == gold)
    row["agree"] = int(row["agree"]) + int(pred == teacher_pred)
    row["teacher_correct_and_agree"] = int(row["teacher_correct_and_agree"]) + int(
        teacher_pred == gold and pred == teacher_pred
    )
    row["kl"] = float(row["kl"]) + float(F.kl_div(dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item())
    row["scored"] = int(row["scored"]) + 1


def finalize(row: dict[str, float | int], teacher_correct: int) -> dict[str, float | int]:
    scored = max(int(row["scored"]), 1)
    return {
        "scored": int(row["scored"]),
        "correct": int(row["correct"]),
        "accuracy": float(row["correct"]) / scored,
        "teacher_agreement": float(row["agree"]) / scored,
        "teacher_correct_retention": float(row["teacher_correct_and_agree"]) / max(int(teacher_correct), 1),
        "output_kl": float(row["kl"]) / scored,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, None)[args.start_index : args.start_index + args.max_samples]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    valid_modes = {"no_visual", "full_target", "constant", "film", "linear", "linear_rank"}
    parsed_modes = {mode: parse_mode(mode) for mode in modes}
    unknown = {mode for mode, (kind, _rank) in parsed_modes.items() if kind not in valid_modes}
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    option_ids = option_token_id_lists(processor.tokenizer)
    num_layers = len(language_model.layers)
    metrics = {
        name: {"correct": 0, "agree": 0, "teacher_correct_and_agree": 0, "kl": 0.0, "scored": 0}
        for name in modes
    }
    effect_metrics = {
        name: {"cos": 0.0, "nmse": 0.0, "norm_ratio": 0.0, "count": 0}
        for name in modes
        if parsed_modes[name][0] not in {"no_visual", "full_target"}
    }
    teacher_correct = 0
    predictions: list[dict[str, Any]] = []

    for sample_idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _visual_pos_masks, _deepstack_visual_embeds = build_qwen3vl_initial_context(
            model,
            inputs,
        )
        del hidden0
        text_positions, _image_positions, text_position_ids, text_mask, _image_mask, full_mask = (
            get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                full_position_ids,
            )
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
            for state in teacher_states
        ]
        target_deltas = [
            compute_qwen3vl_attention_effect_batched(
                language_model,
                layer_idx,
                teacher_states[layer_idx],
                teacher_text_states[layer_idx],
                full_position_ids,
                text_position_ids,
                text_positions,
                full_mask,
                text_mask,
            ).detach()
            for layer_idx in range(num_layers)
        ]
        last_text_idx = int(text_mask[0].sum().item()) - 1
        teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_correct += int(teacher_pred == gold)
        sample_out = {"index": row.get("index", sample_idx), "gold": gold, "teacher": teacher_pred}

        for mode in modes:
            mode_kind, mode_rank = parsed_modes[mode]
            h = teacher_text_states[0].masked_fill((~text_mask).unsqueeze(-1), 0.0)
            for layer_idx in range(num_layers):
                target = target_deltas[layer_idx]
                if mode_kind == "no_visual":
                    delta = None
                elif mode_kind == "full_target":
                    delta = target
                elif mode_kind == "constant":
                    fit_mask = token_split_mask(text_mask[0], args.fit_token_split)
                    delta = constant_operator_delta(
                        teacher_text_states[layer_idx][0],
                        target[0],
                        h[0],
                        fit_mask,
                        text_mask[0],
                    ).unsqueeze(0)
                elif mode_kind == "film":
                    fit_mask = token_split_mask(text_mask[0], args.fit_token_split)
                    delta = film_operator_delta(
                        teacher_text_states[layer_idx][0],
                        target[0],
                        h[0],
                        fit_mask,
                        text_mask[0],
                        args.ridge,
                    ).unsqueeze(0)
                elif mode_kind == "linear":
                    fit_mask = token_split_mask(text_mask[0], args.fit_token_split)
                    delta = linear_operator_delta(
                        teacher_text_states[layer_idx][0],
                        target[0],
                        h[0],
                        fit_mask,
                        text_mask[0],
                        args.ridge,
                    ).unsqueeze(0)
                elif mode_kind == "linear_rank":
                    if mode_rank is None:
                        raise AssertionError(mode)
                    fit_mask = token_split_mask(text_mask[0], args.fit_token_split)
                    delta = lowrank_linear_operator_delta(
                        teacher_text_states[layer_idx][0],
                        target[0],
                        h[0],
                        fit_mask,
                        text_mask[0],
                        args.ridge,
                        mode_rank,
                    ).unsqueeze(0)
                else:
                    raise AssertionError(mode)
                if mode in effect_metrics:
                    metric_mask = token_split_mask(text_mask[0], args.metric_token_split, args.fit_token_split)
                    metric_delta = delta[0]
                    if args.effect_input == "teacher":
                        fit_mask = token_split_mask(text_mask[0], args.fit_token_split)
                        teacher_h_l = teacher_text_states[layer_idx][0]
                        if mode_kind == "constant":
                            metric_delta = constant_operator_delta(
                                teacher_h_l,
                                target[0],
                                teacher_h_l,
                                fit_mask,
                                text_mask[0],
                            )
                        elif mode_kind == "film":
                            metric_delta = film_operator_delta(
                                teacher_h_l,
                                target[0],
                                teacher_h_l,
                                fit_mask,
                                text_mask[0],
                                args.ridge,
                            )
                        elif mode_kind == "linear":
                            metric_delta = linear_operator_delta(
                                teacher_h_l,
                                target[0],
                                teacher_h_l,
                                fit_mask,
                                text_mask[0],
                                args.ridge,
                            )
                        elif mode_kind == "linear_rank":
                            if mode_rank is None:
                                raise AssertionError(mode)
                            metric_delta = lowrank_linear_operator_delta(
                                teacher_h_l,
                                target[0],
                                teacher_h_l,
                                fit_mask,
                                text_mask[0],
                                args.ridge,
                                mode_rank,
                            )
                    metric = masked_metric(metric_delta, target[0], metric_mask)
                    for key in ("cos", "nmse", "norm_ratio"):
                        effect_metrics[mode][key] += metric[key]
                    effect_metrics[mode]["count"] += 1
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
            update_metric(metrics, mode, pred, teacher_pred, gold, dist, teacher_dist)
            sample_out[mode] = pred
        predictions.append(sample_out)
        if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(rows):
            print(f"operator oracle {sample_idx + 1}/{len(rows)} teacher_acc={teacher_correct / (sample_idx + 1):.3f}", flush=True)

    finalized_effect = {}
    for name, row in effect_metrics.items():
        count = max(int(row["count"]), 1)
        finalized_effect[name] = {
            "effect_cos": float(row["cos"]) / count,
            "effect_nmse": float(row["nmse"]) / count,
            "effect_norm_ratio": float(row["norm_ratio"]) / count,
        }
    results = {
        "benchmark": "mmstar",
        "data": args.data,
        "model_path": args.model_path,
        "num_samples": len(rows),
        "ridge": args.ridge,
        "fit_token_split": args.fit_token_split,
        "metric_token_split": args.metric_token_split,
        "effect_input": args.effect_input,
        "teacher": {"correct": teacher_correct, "accuracy": teacher_correct / max(len(rows), 1)},
        "oracles": {name: finalize(metrics[name], teacher_correct) for name in modes},
        "effect_metrics": finalized_effect,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions_jsonl:
        pred_out = Path(args.predictions_jsonl)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
