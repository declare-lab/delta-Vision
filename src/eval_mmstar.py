"""8-GPU sharded MMStar evaluation for vision KV adapter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import (
    PerLayerKVAdapter,
    extract_vision_kv,
    load_frozen_llava,
    student_forward_with_visual_kv,
    teacher_forward,
)
from src.data import MMStarDataset


OPTION_LETTERS = ["A", "B", "C", "D"]


def get_option_token_ids(tokenizer) -> dict[str, list[int]]:
    """Get token IDs for option letters."""
    result = {}
    for letter in OPTION_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        # Only use bare letter token (single token)
        result[letter] = ids
    return result


def predict_option(logits: torch.Tensor, option_ids: dict[str, list[int]]) -> str:
    """Predict the most likely option letter from logits."""
    best_letter = "A"
    best_score = float("-inf")
    for letter, ids in option_ids.items():
        score = max(logits[tid].item() for tid in ids)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def load_adapter(checkpoint_path: str, device: torch.device) -> PerLayerKVAdapter:
    """Load trained adapter from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = PerLayerKVAdapter(
        num_llm_layers=32,
        num_source_layers=2,
        source_dim=1024,
        num_heads=32,
        head_dim=128,
    )
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.to(device=device, dtype=torch.bfloat16)
    adapter.eval()
    return adapter


@torch.inference_mode()
def evaluate_shard(
    model,
    processor,
    adapter: PerLayerKVAdapter,
    dataset: MMStarDataset,
    device: torch.device,
    image_token_id: int,
    log_every: int = 25,
) -> dict:
    """Evaluate adapter on a shard of MMStar."""
    option_ids = get_option_token_ids(processor.tokenizer)

    stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "agree": 0,
    }
    predictions = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        gold = item["gold"]

        source_k, source_v = extract_vision_kv(model, pixel_values, source_layer_indices=[22, 23])

        teacher_logits = teacher_forward(model, input_ids, pixel_values, attention_mask)
        teacher_last = teacher_logits[0, -1]

        student_logits = student_forward_with_visual_kv(
            model, input_ids, adapter, source_k, source_v, image_token_id
        )
        student_last = student_logits[0, -1]

        teacher_pred = predict_option(teacher_last, option_ids)
        adapter_pred = predict_option(student_last, option_ids)

        stats["scored"] += 1
        stats["teacher_correct"] += int(teacher_pred == gold)
        stats["adapter_correct"] += int(adapter_pred == gold)
        stats["agree"] += int(adapter_pred == teacher_pred)

        predictions.append({
            "index": item["index"],
            "gold": gold,
            "teacher": teacher_pred,
            "adapter": adapter_pred,
        })

        if (idx + 1) % log_every == 0:
            acc = stats["adapter_correct"] / stats["scored"]
            print(f"[{idx+1}/{len(dataset)}] adapter_acc={acc:.4f}", flush=True)

    return {"stats": stats, "predictions": predictions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving image paths")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None, help="If set, only run this shard")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def run_single_shard(args, shard_id: int, num_shards: int):
    """Run evaluation on a single shard (one GPU)."""
    device = torch.device("cuda:0")
    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device="cuda:0")
    adapter = load_adapter(args.checkpoint, device)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    full_dataset = MMStarDataset(args.data, processor, data_root=args.data_root, max_samples=args.max_samples)
    total = len(full_dataset)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)

    full_dataset.rows = full_dataset.rows[start:end]
    print(f"Shard {shard_id}: samples [{start}, {end}) = {len(full_dataset)} items", flush=True)

    result = evaluate_shard(model, processor, adapter, full_dataset, device, image_token_id, args.log_every)

    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
    return result


def merge_shards(output_dir: str, num_shards: int) -> dict:
    """Merge results from all shards."""
    all_stats = {"scored": 0, "teacher_correct": 0, "adapter_correct": 0, "agree": 0}
    all_predictions = []

    for shard_id in range(num_shards):
        path = Path(output_dir) / f"shard_{shard_id}.json"
        with open(path) as f:
            data = json.load(f)
        for key in all_stats:
            all_stats[key] += data["stats"][key]
        all_predictions.extend(data["predictions"])

    n = max(all_stats["scored"], 1)
    tc = max(all_stats["teacher_correct"], 1)
    merged = {
        "total_samples": all_stats["scored"],
        "teacher_accuracy": all_stats["teacher_correct"] / n,
        "adapter_accuracy": all_stats["adapter_correct"] / n,
        "agreement": all_stats["agree"] / n,
        "retention": all_stats["agree"] / tc,
    }

    out_path = Path(output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(json.dumps(merged, indent=2), flush=True)
    return merged


def main():
    args = parse_args()

    if args.shard_id is not None:
        run_single_shard(args, args.shard_id, args.num_shards)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
