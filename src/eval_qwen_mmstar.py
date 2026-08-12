"""Sharded MMStar generation evaluation for Qwen3-VL vision KV adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.data import QwenMMStarDataset
from src.eval_mmstar import _eos_token_ids, extract_option_from_text
from src.model import (
    QWEN_SOURCE_RAW_SPATIAL_CONCAT,
    PerLayerKVAdapter,
    extract_vision_kv_qwen,
    load_adapter_checkpoint,
    qwen_source_dim,
    student_forward_qwen,
)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def validate_qwen_source_checkpoint(model, adapter_config: dict) -> None:
    source_mode = QWEN_SOURCE_RAW_SPATIAL_CONCAT
    saved_mode = adapter_config.get("qwen_source_mode", adapter_config.get("source_mode"))
    if saved_mode is not None and saved_mode != source_mode:
        raise ValueError(
            f"Checkpoint uses unsupported Qwen source_mode={saved_mode!r}; "
            f"expected {source_mode!r}."
        )
    expected_source_dim = qwen_source_dim(model, source_mode)
    if int(adapter_config["source_dim"]) != expected_source_dim:
        raise ValueError(
            f"Checkpoint source_dim={adapter_config['source_dim']} does not match "
            f"{source_mode} source_dim={expected_source_dim}."
        )


@torch.inference_mode()
def generate_teacher_qwen(
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str | None, str]:
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None

    eos_ids = sorted(_eos_token_ids(processor.tokenizer))
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id

    generated = model.generate(**kwargs)
    text = processor.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def generate_adapter_qwen(
    model,
    processor,
    adapter: PerLayerKVAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    max_new_tokens: int,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    full_mm_ids = mm_token_type_ids.clone()
    generated: list[int] = []
    eos_ids = _eos_token_ids(processor.tokenizer)

    for _ in range(max_new_tokens):
        text_mask = (full_ids[0] != image_token_id) & full_mask[0].bool()
        text_ids = full_ids[:, text_mask]
        logits = student_forward_qwen(
            model,
            text_ids,
            adapter,
            source_k,
            source_v,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=spatial_merge_size,
            full_input_ids=full_ids,
            mm_token_type_ids=full_mm_ids,
            attention_mask=full_mask,
        )
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_token)

        token = torch.tensor([[next_token]], dtype=full_ids.dtype, device=full_ids.device)
        full_ids = torch.cat([full_ids, token], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(token)], dim=1)
        full_mm_ids = torch.cat([full_mm_ids, torch.zeros_like(token)], dim=1)
        if next_token in eos_ids:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def evaluate_shard(
    model,
    processor,
    adapter: PerLayerKVAdapter,
    dataset: QwenMMStarDataset,
    device: torch.device,
    source_layers: list[int],
    source_mode: str,
    log_every: int,
    max_new_tokens: int,
) -> dict:
    image_token_id = int(getattr(processor, "image_token_id", getattr(model.config, "image_token_id", 151655)))
    spatial_merge_size = int(getattr(model.model.visual, "spatial_merge_size", 2))
    expected_source_dim = qwen_source_dim(model, source_mode)

    stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "adapter_correct_when_teacher_correct": 0,
        "agree": 0,
        "teacher_invalid": 0,
        "adapter_invalid": 0,
    }
    predictions = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].to(device)
        image_grid_thw = item["image_grid_thw"].to(device)
        mm_token_type_ids = item["mm_token_type_ids"].unsqueeze(0).to(device)
        gold = item["gold"]

        source_k, source_v = extract_vision_kv_qwen(
            model,
            pixel_values,
            image_grid_thw,
            source_layers,
            source_mode=source_mode,
        )
        if source_k.shape[-1] != expected_source_dim:
            raise RuntimeError(f"source dim mismatch: got {source_k.shape[-1]}, expected {expected_source_dim}")
        n_image_tokens = int(((input_ids[0] == image_token_id) & attention_mask[0].bool()).sum().item())
        if source_k.shape[2] != n_image_tokens:
            raise RuntimeError(f"source token count {source_k.shape[2]} != image token count {n_image_tokens}")

        teacher_pred, teacher_text = generate_teacher_qwen(
            model,
            processor,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            max_new_tokens,
        )
        adapter_pred, adapter_text = generate_adapter_qwen(
            model,
            processor,
            adapter,
            input_ids,
            attention_mask,
            mm_token_type_ids,
            source_k,
            source_v,
            image_grid_thw,
            image_token_id,
            spatial_merge_size,
            max_new_tokens,
        )

        stats["scored"] += 1
        stats["teacher_correct"] += int(teacher_pred == gold)
        stats["adapter_correct"] += int(adapter_pred == gold)
        stats["adapter_correct_when_teacher_correct"] += int(teacher_pred == gold and adapter_pred == gold)
        stats["agree"] += int(teacher_pred == adapter_pred)
        stats["teacher_invalid"] += int(teacher_pred is None)
        stats["adapter_invalid"] += int(adapter_pred is None)

        predictions.append(
            {
                "index": item["index"],
                "gold": gold,
                "teacher": teacher_pred,
                "adapter": adapter_pred,
                "teacher_text": teacher_text,
                "adapter_text": adapter_text,
            }
        )

        if (idx + 1) % log_every == 0:
            n = max(stats["scored"], 1)
            print(
                f"[{idx+1}/{len(dataset)}] teacher={stats['teacher_correct']/n:.4f} "
                f"adapter={stats['adapter_correct']/n:.4f} agreement={stats['agree']/n:.4f}",
                flush=True,
            )

    return {
        "stats": stats,
        "predictions": predictions,
        "source_mode": source_mode,
        "source_layers": source_layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL adapter on MMStar by short greedy generation.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--answer-instruction",
        default="Answer directly with only the letter of the correct option.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    return parser.parse_args()


def run_single_shard(args, shard_id: int, num_shards: int) -> dict:
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    adapter, source_layers, adapter_config = load_adapter_checkpoint(
        args.checkpoint,
        device=device,
        language_model=model.model.language_model,
        dtype=dtype,
    )
    validate_qwen_source_checkpoint(model, adapter_config)
    source_mode = QWEN_SOURCE_RAW_SPATIAL_CONCAT

    dataset = QwenMMStarDataset(
        args.data,
        processor,
        data_root=args.data_root,
        max_samples=args.max_samples,
        answer_instruction=args.answer_instruction,
    )
    total = len(dataset)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)
    dataset.rows = dataset.rows[start:end]
    print(
        f"Shard {shard_id}: samples [{start}, {end}) = {len(dataset)} items; "
        f"source_mode={source_mode} source_layers={source_layers}",
        flush=True,
    )

    result = evaluate_shard(
        model,
        processor,
        adapter,
        dataset,
        device,
        source_layers,
        source_mode,
        args.log_every,
        args.max_new_tokens,
    )

    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
    return result


def merge_shards(output_dir: str, num_shards: int) -> dict:
    all_stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "adapter_correct_when_teacher_correct": 0,
        "agree": 0,
        "teacher_invalid": 0,
        "adapter_invalid": 0,
    }
    all_predictions = []
    source_modes = set()
    source_layers = None

    for shard_id in range(num_shards):
        path = Path(output_dir) / f"shard_{shard_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard result: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in all_stats:
            all_stats[key] += data["stats"].get(key, 0)
        all_predictions.extend(data["predictions"])
        if "source_mode" in data:
            source_modes.add(data["source_mode"])
        if source_layers is None:
            source_layers = data.get("source_layers")

    n = max(all_stats["scored"], 1)
    teacher_correct = all_stats["teacher_correct"]
    merged = {
        "total_samples": all_stats["scored"],
        "teacher_accuracy": all_stats["teacher_correct"] / n,
        "adapter_accuracy": all_stats["adapter_correct"] / n,
        "agreement": all_stats["agree"] / n,
        "retention": (
            all_stats["adapter_correct_when_teacher_correct"] / teacher_correct
            if teacher_correct
            else 0.0
        ),
        "teacher_invalid_rate": all_stats["teacher_invalid"] / n,
        "adapter_invalid_rate": all_stats["adapter_invalid"] / n,
        "source_modes": sorted(source_modes),
        "source_layers": source_layers,
    }

    out_path = Path(output_dir) / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(json.dumps(merged, indent=2, ensure_ascii=False), flush=True)
    return merged


def main():
    args = parse_args()
    if args.shard_id is not None:
        run_single_shard(args, args.shard_id, args.num_shards)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
