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
    parser = argparse.ArgumentParser("Evaluate focused Qwen3-VL attention-effect Oracle modes on MMStar.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--modes", default="no_visual,full,sparse,last_token")
    parser.add_argument("--active-layers", default="0,8,16,24,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_layers(spec: str) -> set[int]:
    return {int(x) for x in spec.split(",") if x.strip()}


def parse_modes(spec: str) -> list[str]:
    modes = [x.strip() for x in spec.split(",") if x.strip()]
    allowed = {"no_visual", "full", "sparse", "last_token"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise ValueError(f"unknown modes: {bad}")
    if not modes:
        raise ValueError("--modes cannot be empty")
    return modes


def build_prompt(processor: Any, row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    question = f"{question}\nAnswer directly with only the letter of the correct option."
    return qwen3vl_prompt(processor, question)


@torch.inference_mode()
def prompt_inputs(processor: Any, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def last_token_only(delta: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(delta)
    lengths = text_mask.to(device=delta.device).long().sum(dim=1)
    for batch_idx, length in enumerate(lengths.tolist()):
        if length > 0:
            out[batch_idx, int(length) - 1] = delta[batch_idx, int(length) - 1]
    return out


def relative_hidden_distance(student: torch.Tensor, teacher: torch.Tensor, text_mask: torch.Tensor) -> float:
    mask = text_mask.to(device=student.device, dtype=torch.bool).unsqueeze(-1)
    diff = (student.float() - teacher.float()).masked_fill(~mask, 0.0)
    ref = teacher.float().masked_fill(~mask, 0.0)
    return float(diff.norm().item() / max(ref.norm().item(), 1e-8))


def update_metrics(metrics: dict[str, float | int], pred: str, teacher_pred: str, gold: str, kl: float) -> None:
    metrics["correct"] = int(metrics["correct"]) + int(pred == gold)
    metrics["agree"] = int(metrics["agree"]) + int(pred == teacher_pred)
    metrics["teacher_correct_and_agree"] = int(metrics["teacher_correct_and_agree"]) + int(
        teacher_pred == gold and pred == teacher_pred
    )
    metrics["output_kl_sum"] = float(metrics["output_kl_sum"]) + float(kl)
    metrics["scored"] = int(metrics["scored"]) + 1


def finalize_metrics(
    metrics: dict[str, float | int],
    teacher_correct: int,
    hidden_distance_sums: list[float],
) -> dict[str, Any]:
    scored = max(int(metrics["scored"]), 1)
    teacher_correct = max(int(teacher_correct), 1)
    return {
        "num_samples": int(metrics["scored"]),
        "correct": int(metrics["correct"]),
        "accuracy": float(metrics["correct"]) / scored,
        "teacher_agreement": float(metrics["agree"]) / scored,
        "teacher_correct_and_agree": int(metrics["teacher_correct_and_agree"]),
        "teacher_correct_retention": float(metrics["teacher_correct_and_agree"]) / teacher_correct,
        "output_kl": float(metrics["output_kl_sum"]) / scored,
        "hidden_distance_by_layer": [x / scored for x in hidden_distance_sums],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    modes = parse_modes(args.modes)
    active_layers = parse_layers(args.active_layers)

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    option_ids = option_token_id_lists(processor.tokenizer)

    metrics = {
        mode: {"correct": 0, "agree": 0, "teacher_correct_and_agree": 0, "output_kl_sum": 0.0, "scored": 0}
        for mode in modes
    }
    hidden_distance_sums = {mode: [0.0 for _ in range(num_layers)] for mode in modes}
    teacher_correct = 0
    predictions = []

    need_deltas = any(mode in {"full", "sparse", "last_token"} for mode in modes)

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen3vl_text_image_positions(
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
        last_text_idx = int(text_mask[0].sum().item()) - 1
        teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_correct += int(teacher_pred == gold)

        deltas: list[torch.Tensor | None] = [None for _ in range(num_layers)]
        if need_deltas:
            for layer_idx in range(num_layers):
                deltas[layer_idx] = compute_qwen3vl_attention_effect_batched(
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

        sample_preds = {"index": row.get("index", idx), "gold": gold, "teacher": teacher_pred}
        for mode in modes:
            h = teacher_text_states[0]
            for layer_idx in range(num_layers):
                if mode == "no_visual":
                    delta = None
                elif mode == "full":
                    delta = deltas[layer_idx]
                elif mode == "sparse":
                    delta = deltas[layer_idx] if layer_idx in active_layers else None
                elif mode == "last_token":
                    assert deltas[layer_idx] is not None
                    delta = last_token_only(deltas[layer_idx], text_mask)
                else:
                    raise ValueError(mode)
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    padding_mask=~text_mask,
                )
                hidden_distance_sums[mode][layer_idx] += relative_hidden_distance(
                    h,
                    teacher_text_states[layer_idx + 1],
                    text_mask,
                )

            logits = model.lm_head(language_model.norm(h))[0, last_text_idx]
            pred = predict_option(logits, option_ids)
            dist = option_distribution(logits, option_ids)
            kl = F.kl_div(dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
            update_metrics(metrics[mode], pred, teacher_pred, gold, kl)
            sample_preds[mode] = pred
        predictions.append(sample_preds)

        if (idx + 1) % 25 == 0:
            print(f"evaluated {idx + 1}/{len(rows)} teacher_correct={teacher_correct}", flush=True)

    results = {
        "benchmark": "mmstar",
        "model_path": args.model_path,
        "data": args.data,
        "num_samples": len(rows),
        "active_layers": sorted(active_layers),
        "teacher": {
            "correct": teacher_correct,
            "accuracy": teacher_correct / max(len(rows), 1),
        },
        "oracles": {
            mode: finalize_metrics(metrics[mode], teacher_correct, hidden_distance_sums[mode]) for mode in modes
        },
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
