#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from delta_vision.data import JsonlDataset, collate_rows
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.modeling import build_rollout_model, image_token_id, load_frozen_llava, load_rollout_checkpoint
from delta_vision.grounding.pointing import (
    PointHead,
    parse_points,
    point_distance,
    point_in_masks_xy100,
    sidecar_point_hidden_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate shared Sidecar pointing on PixMo-Points-Eval.")
    parser.add_argument("--data", default="data/pixmo_points/eval_test.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="artifacts/eval/pointing/pixmo_points_eval_shared.json")
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--point-head-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    img_token = image_token_id(model, processor)
    image_seq_length = int(getattr(model.config, "image_seq_length", 0))
    if image_seq_length <= 0:
        raise RuntimeError("LLaVA config must define image_seq_length")

    rollout_model = build_rollout_model(args, dtype, device)
    load_rollout_checkpoint(rollout_model, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    point_head = PointHead(args.hidden_size, args.point_head_dim).to(device=device)
    if "point_head" not in checkpoint:
        raise RuntimeError(f"checkpoint does not contain point_head: {args.checkpoint}")
    point_head.load_state_dict(checkpoint["point_head"], strict=True)
    rollout_model.eval()
    point_head.eval()

    dataset = JsonlDataset(args.data, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
        pin_memory=True,
    )
    correct = 0
    total = 0
    dist_sum = 0.0
    predictions: list[dict] = []
    for batch_idx, rows in enumerate(loader):
        hidden = sidecar_point_hidden_batch(
            processor,
            model,
            language_model,
            rollout_model,
            img_token,
            image_seq_length,
            rows,
            device,
            dtype,
        )
        preds = point_head(hidden).float().cpu()
        for pred, row in zip(preds, rows, strict=True):
            x = float(pred[0].item())
            y = float(pred[1].item())
            points = parse_points(row["points"])
            dist = float(point_distance(pred.to(device), points.to(device)).cpu().item())
            in_mask = point_in_masks_xy100((x, y), row["mask_path"]) if row.get("mask_path") else False
            total += 1
            correct += int(in_mask)
            dist_sum += dist
            predictions.append(
                {
                    "index": row.get("index", total - 1),
                    "label": row["label"],
                    "prediction": {"x": x, "y": y},
                    "points": row["points"],
                    "point_distance": dist,
                    "point_in_mask": in_mask,
                    "image": row["image"],
                }
            )
        done = total
        if done % 25 == 0 or batch_idx == len(loader) - 1:
            print(f"evaluated {done}/{len(dataset)} point_acc={correct / max(total, 1):.4f}", flush=True)

    metrics = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "num_samples": total,
        "point_in_mask_correct": correct,
        "point_in_mask_accuracy": correct / max(total, 1),
        "mean_point_distance_xy100": dist_sum / max(total, 1),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
