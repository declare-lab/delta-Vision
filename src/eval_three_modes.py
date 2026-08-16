"""Evaluate 3 modes on MMStar: teacher (original), adapter-only, mixed (original + adapter KV)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, ".")
from src.model import (
    extract_vision_kv, load_adapter_checkpoint, load_frozen_llava,
    student_forward_with_visual_kv, teacher_forward, student_forward_mixed,
)
from src.eval_benchmarks import get_option_token_ids, predict_option
from src.data import MMStarDataset


def load_adapter_from_ckpt(checkpoint_path: str, model, device: torch.device):
    adapter, source_layers, _ = load_adapter_checkpoint(
        checkpoint_path,
        device=device,
        language_model=model.model.language_model,
        dtype=torch.bfloat16,
    )
    return adapter, source_layers


@torch.inference_mode()
def evaluate(args):
    device = torch.device("cuda:0")
    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device="cuda:0")
    adapter, source_layers = load_adapter_from_ckpt(args.checkpoint, model, device)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    option_ids = get_option_token_ids(processor.tokenizer)

    dataset = MMStarDataset(args.data, processor, data_root=args.data_root, max_samples=args.max_samples)

    # Shard
    total = len(dataset)
    per_shard = (total + args.num_shards - 1) // args.num_shards
    start = args.shard_id * per_shard
    end = min(start + per_shard, total)
    dataset.rows = dataset.rows[start:end]
    print(f"Shard {args.shard_id}: [{start}, {end}) = {len(dataset)} samples", flush=True)

    stats = {"teacher": 0, "adapter": 0, "mixed": 0, "scored": 0, "mixed_scored": 0}
    mixed_supported = not hasattr(model.model, "image_newline")

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        gold = item["gold"]

        source_k, source_v = extract_vision_kv(model, pixel_values, source_layers)

        # 1. Teacher (original LLaVA)
        image_sizes = item.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.unsqueeze(0).to(device) if torch.is_tensor(image_sizes) else image_sizes
        t_logits = teacher_forward(model, input_ids, pixel_values, attention_mask=attention_mask, image_sizes=image_sizes)
        full_last_idx = int(attention_mask[0].sum().item()) - 1
        t_pred = predict_option(t_logits[0, full_last_idx], option_ids)

        # 2. Adapter only (vision KV inject, no image embeddings)
        a_logits = student_forward_with_visual_kv(
            model, input_ids, adapter, source_k, source_v, image_token_id, attention_mask=attention_mask
        )
        a_pred = predict_option(a_logits[0, -1], option_ids)

        # 3. Mixed (original + adapter KV added)
        m_pred = None
        if mixed_supported and pixel_values.ndim == 4:
            m_logits = student_forward_mixed(model, input_ids, pixel_values, adapter, source_k, source_v, image_token_id)
            m_pred = predict_option(m_logits[0, full_last_idx], option_ids)

        stats["scored"] += 1
        stats["teacher"] += int(t_pred == gold)
        stats["adapter"] += int(a_pred == gold)
        if m_pred is not None:
            stats["mixed"] += int(m_pred == gold)
            stats["mixed_scored"] += 1

        if (idx + 1) % 25 == 0:
            n = stats["scored"]
            mixed_n = max(stats["mixed_scored"], 1)
            ta, aa, ma = stats["teacher"]/n, stats["adapter"]/n, stats["mixed"]/mixed_n
            print(f"[{idx+1}/{len(dataset)}] teacher={ta:.3f} adapter={aa:.3f} mixed={ma:.3f}", flush=True)

    out_path = Path(args.output_dir) / f"shard_{args.shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f)
    print(f"Shard {args.shard_id} done: {json.dumps(stats)}", flush=True)


def merge(args):
    total = {"teacher": 0, "adapter": 0, "mixed": 0, "scored": 0, "mixed_scored": 0}
    for shard in range(args.num_shards):
        path = Path(args.output_dir) / f"shard_{shard}.json"
        with open(path) as f:
            s = json.load(f)
        for k in total:
            total[k] += s[k]
    n = max(total["scored"], 1)
    result = {
        "total_samples": total["scored"],
        "teacher_accuracy": total["teacher"] / n,
        "adapter_only_accuracy": total["adapter"] / n,
        "mixed_accuracy": total["mixed"] / max(total["mixed_scored"], 1),
        "mixed_scored": total["mixed_scored"],
    }
    out = Path(args.output_dir) / "results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    args = parser.parse_args()

    if args.shard_id is not None:
        evaluate(args)
    else:
        merge(args)
