#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt,
    candidate_ids,
    candidate_kind,
    distribution,
    normalize_gold,
    predict,
    qwen_logits,
)
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_attention_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL M-RoPE layer ablations.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--mrope-layers",
        default="0",
        help="Comma-separated layer ids that use real M-RoPE. Other layers use identity RoPE.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_layers(spec: str) -> set[int]:
    return {int(x) for x in spec.split(",") if x.strip()}


@torch.inference_mode()
def qwen_logits_mrope_ablation(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
    mrope_layers: set[int],
) -> torch.Tensor:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    hidden_states, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    hidden_states = hidden_states.to(dtype=dtype)
    text_positions, _, _, text_mask, _, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    attention_mask_2d = inputs.get("attention_mask")
    causal_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    identity_position_ids = torch.zeros_like(position_ids)

    for layer_idx, layer in enumerate(language_model.layers):
        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask_2d,
            past_key_values=None,
            position_ids=causal_position_ids,
        )
        rope_position_ids = position_ids if layer_idx in mrope_layers else identity_position_ids
        position_embeddings = language_model.rotary_emb(hidden_states, rope_position_ids)

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        attn_out = qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = language_model._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    logits = model.lm_head(language_model.norm(hidden_states))
    last_text = int(text_positions[0, int(text_mask[0].sum().item()) - 1].item())
    return logits[0, last_text]


def update(stats: dict[str, Any], name: str, pred: str, gold: str, teacher_pred: str, teacher_correct: bool, kl: float) -> None:
    row = stats[name]
    row["scored"] += 1
    row["correct"] += int(pred == gold)
    row["agree"] += int(pred == teacher_pred)
    row["ret"] += int(teacher_correct and pred == gold)
    row["kl"] += float(kl)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    mrope_layers = parse_layers(args.mrope_layers)

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    stats: dict[str, Any] = {
        "qwen": {"scored": 0, "correct": 0},
        "mrope_ablation": {"scored": 0, "correct": 0, "agree": 0, "ret": 0, "kl": 0.0},
    }
    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        teacher_logits = qwen_logits(processor, model, row, args.benchmark, device)
        ablated_logits = qwen_logits_mrope_ablation(
            processor,
            model,
            language_model,
            row,
            args.benchmark,
            device,
            dtype,
            mrope_layers,
        )
        teacher_pred = predict(teacher_logits, ids)
        ablated_pred = predict(ablated_logits, ids)
        teacher_correct = teacher_pred == gold
        teacher_dist = distribution(teacher_logits, ids)
        ablated_dist = distribution(ablated_logits, ids)
        kl = F.kl_div(ablated_dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()

        stats["qwen"]["scored"] += 1
        stats["qwen"]["correct"] += int(teacher_correct)
        update(stats, "mrope_ablation", ablated_pred, gold, teacher_pred, teacher_correct, kl)
        predictions.append(
            {
                "index": row.get("index", args.start_index + idx),
                "gold": gold,
                "qwen": teacher_pred,
                "mrope_ablation": ablated_pred,
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"evaluated {idx + 1}/{len(rows)}", flush=True)

    q_scored = max(1, int(stats["qwen"]["scored"]))
    q_correct = max(1, int(stats["qwen"]["correct"]))
    a_scored = max(1, int(stats["mrope_ablation"]["scored"]))
    results = {
        "benchmark": args.benchmark,
        "data": args.data,
        "model_path": args.model_path,
        "num_samples": int(stats["qwen"]["scored"]),
        "mrope_layers": sorted(mrope_layers),
        "results": [
            {
                "setting": "qwen",
                "scored": int(stats["qwen"]["scored"]),
                "correct": int(stats["qwen"]["correct"]),
                "accuracy": int(stats["qwen"]["correct"]) / q_scored,
            },
            {
                "setting": "mrope_ablation",
                "scored": int(stats["mrope_ablation"]["scored"]),
                "correct": int(stats["mrope_ablation"]["correct"]),
                "accuracy": int(stats["mrope_ablation"]["correct"]) / a_scored,
                "qwen_agreement": int(stats["mrope_ablation"]["agree"]) / a_scored,
                "qwen_correct_retention": int(stats["mrope_ablation"]["ret"]) / q_correct,
                "output_kl_to_qwen": float(stats["mrope_ablation"]["kl"]) / a_scored,
            },
        ],
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
