#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.evaluation.metrics import option_token_id_lists
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    run_qwen3vl_full_layer_with_text_delta,
)
from delta_vision.models.visual_prefix import VisualPrefixCompressor, grid_position_ids
from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt,
    candidate_ids,
    candidate_kind,
    distribution,
    no_visual_logits,
    normalize_gold,
    predict,
    qwen_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL small visual-prefix baseline.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def load_prefix(args: argparse.Namespace, hidden_size: int, device: torch.device, dtype: torch.dtype) -> VisualPrefixCompressor:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    module = VisualPrefixCompressor(
        hidden_size=hidden_size,
        num_tokens=int(ckpt_args.get("num_prefix_tokens", 128)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
    ).to(device=device, dtype=dtype)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


@torch.inference_mode()
def prefix_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    compressor: VisualPrefixCompressor,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0, position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    text_h0 = gather_batched_positions(hidden0.to(dtype=dtype), text_pos, text_mask)
    v0 = gather_batched_positions(hidden0.to(dtype=dtype), image_pos, image_mask)
    prefix = compressor(v0, image_mask)
    batch, prefix_len = prefix.shape[:2]
    h = torch.cat([prefix, text_h0], dim=1)
    prefix_pos = grid_position_ids(batch, prefix_len, device).to(dtype=text_position_ids.dtype)
    full_pos = torch.cat([prefix_pos, text_position_ids], dim=2).contiguous()
    full_mask = torch.cat(
        [
            torch.ones(batch, prefix_len, device=device, dtype=torch.long),
            text_mask.to(dtype=torch.long),
        ],
        dim=1,
    )
    full_text_positions = torch.arange(text_h0.shape[1], device=device).view(1, -1).expand(batch, -1) + prefix_len
    for layer_idx in range(len(language_model.layers)):
        h = run_qwen3vl_full_layer_with_text_delta(
            language_model,
            layer_idx,
            h,
            full_pos,
            full_mask,
            full_text_positions,
            text_delta=None,
        )
    text_h = h[:, prefix_len : prefix_len + text_h0.shape[1]]
    logits = model.lm_head(language_model.norm(text_h))
    return logits[0, int(text_mask[0].sum().item()) - 1]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    compressor = load_prefix(args, int(language_model.config.hidden_size), device, dtype)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    stats = {
        "qwen": {"correct": 0, "scored": 0},
        "no_visual": {"correct": 0, "scored": 0, "agree": 0, "ret": 0, "kl": 0.0},
        "visual_prefix": {"correct": 0, "scored": 0, "agree": 0, "ret": 0, "kl": 0.0},
    }
    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        q_logits = qwen_logits(processor, model, row, args.benchmark, device)
        nv_logits = no_visual_logits(processor, model, language_model, row, args.benchmark, device, dtype)
        vp_logits = prefix_logits(processor, model, language_model, compressor, row, args.benchmark, device, dtype)
        q_pred = predict(q_logits, ids)
        q_correct = q_pred == gold
        stats["qwen"]["scored"] += 1
        stats["qwen"]["correct"] += int(q_correct)
        q_dist = distribution(q_logits, ids)
        for name, logits in (("no_visual", nv_logits), ("visual_prefix", vp_logits)):
            pred = predict(logits, ids)
            stats[name]["scored"] += 1
            stats[name]["correct"] += int(pred == gold)
            stats[name]["agree"] += int(pred == q_pred)
            stats[name]["ret"] += int(q_correct and pred == gold)
            dist = distribution(logits, ids)
            stats[name]["kl"] += float(F.kl_div(dist.log(), q_dist, reduction="sum").item())
        if (idx + 1) % 100 == 0:
            print(f"eval {idx + 1}/{len(rows)}", flush=True)

    results = []
    q_scored = max(1, stats["qwen"]["scored"])
    q_correct = max(1, stats["qwen"]["correct"])
    results.append(
        {
            "setting": "qwen",
            "scored": stats["qwen"]["scored"],
            "correct": stats["qwen"]["correct"],
            "accuracy": stats["qwen"]["correct"] / q_scored,
        }
    )
    for name in ("no_visual", "visual_prefix"):
        scored = max(1, stats[name]["scored"])
        results.append(
            {
                "setting": name,
                "scored": stats[name]["scored"],
                "correct": stats[name]["correct"],
                "accuracy": stats[name]["correct"] / scored,
                "qwen_agreement": stats[name]["agree"] / scored,
                "qwen_correct_retention": stats[name]["ret"] / q_correct,
                "output_kl_to_qwen": stats[name]["kl"] / scored,
            }
        )
    output = {
        "benchmark": args.benchmark,
        "data": args.data,
        "max_samples": args.max_samples,
        "checkpoint": args.checkpoint,
        "results": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
