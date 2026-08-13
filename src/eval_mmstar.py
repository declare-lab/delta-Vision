"""8-GPU sharded MMStar evaluation for vision KV adapter."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from src.model import (
    extract_vision_kv,
    load_frozen_llava,
    load_adapter_checkpoint,
    PerLayerKVAdapter,
    student_forward_with_visual_kv,
    teacher_forward,
    dtype_from_name,
    load_frozen_qwen3vl,
    load_qwen_visual_delta_checkpoint,
    qwen_visual_delta_logits,
)
from src.data import MMStarDataset, QwenMMStarDataset


OPTION_LETTERS = ["A", "B", "C", "D"]


def extract_option_from_text(text: str) -> str | None:
    """Extract an A/B/C/D answer from generated text."""
    clean = text.strip().upper()
    if not clean:
        return None

    patterns = [
        r"(?:ANSWER|OPTION|CHOICE|答案|选项)\s*(?:IS|是|:|：)?\s*[\(\[]?\s*([ABCD])(?:\b|[\)\]\.。,:：])",
        r"^[\s\(\[]*([ABCD])(?:[\)\]\.。,:：\s]|$)",
        r"(?<![A-Z])([ABCD])(?![A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    return None


def get_option_token_ids(tokenizer) -> dict[str, list[int]]:
    """Get single-token IDs for common option-letter renderings."""
    result = {}
    for letter in OPTION_LETTERS:
        ids = []
        for text in (letter, f" {letter}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
        if not ids:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            ids.append(encoded[-1])
        result[letter] = sorted(set(ids))
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


def _eos_token_ids(tokenizer) -> set[int]:
    eos_ids = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        eos_ids.update(int(x) for x in eos_token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("<|im_end|>", "</s>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                eos_ids.add(token_id)
    return eos_ids


@torch.inference_mode()
def generate_teacher_llava(
    model,
    processor,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    image_sizes=None,
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if image_sizes is not None:
        kwargs["image_sizes"] = image_sizes
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id

    generated = model.generate(**kwargs)
    new_tokens = generated[0, input_ids.shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def generate_adapter_llava(
    model,
    processor,
    adapter: PerLayerKVAdapter,
    input_ids: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    generated = []
    eos_ids = _eos_token_ids(processor.tokenizer)

    for _ in range(max_new_tokens):
        logits = student_forward_with_visual_kv(
            model,
            full_ids,
            adapter,
            source_k,
            source_v,
            image_token_id,
            attention_mask=full_mask,
        )
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_token)
        token_tensor = torch.tensor([[next_token]], dtype=full_ids.dtype, device=full_ids.device)
        full_ids = torch.cat([full_ids, token_tensor], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(token_tensor)], dim=1)
        if next_token in eos_ids:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return extract_option_from_text(text), text


def load_adapter(checkpoint_path: str, model, device: torch.device) -> tuple[PerLayerKVAdapter, list[int]]:
    """Load trained adapter from checkpoint."""
    adapter, source_layers, _ = load_adapter_checkpoint(
        checkpoint_path,
        device=device,
        language_model=model.model.language_model,
        dtype=torch.bfloat16,
    )
    return adapter, source_layers


@torch.inference_mode()
def evaluate_llava_shard(
    model,
    processor,
    adapter: PerLayerKVAdapter,
    dataset: MMStarDataset,
    device: torch.device,
    image_token_id: int,
    source_layers: list[int],
    log_every: int = 25,
    max_new_tokens: int = 8,
    eval_mode: str = "generate",
) -> dict:
    """Evaluate adapter on a shard of MMStar."""
    option_ids = get_option_token_ids(processor.tokenizer)

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
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        image_sizes = item.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.unsqueeze(0).to(device) if torch.is_tensor(image_sizes) else image_sizes
        gold = item["gold"]

        source_k, source_v = extract_vision_kv(model, pixel_values, source_layer_indices=source_layers)
        if eval_mode == "logits":
            teacher_logits = teacher_forward(model, input_ids, pixel_values, attention_mask, image_sizes=image_sizes)
            full_last_idx = int(attention_mask[0].sum().item()) - 1
            teacher_last = teacher_logits[0, full_last_idx]

            student_logits = student_forward_with_visual_kv(
                model, input_ids, adapter, source_k, source_v, image_token_id, attention_mask=attention_mask
            )
            student_last = student_logits[0, -1]

            teacher_pred = predict_option(teacher_last, option_ids)
            adapter_pred = predict_option(student_last, option_ids)
            teacher_text = ""
            adapter_text = ""
        elif eval_mode == "generate":
            teacher_pred, teacher_text = generate_teacher_llava(
                model,
                processor,
                input_ids,
                pixel_values,
                attention_mask,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
            )
            adapter_pred, adapter_text = generate_adapter_llava(
                model,
                processor,
                adapter,
                input_ids,
                source_k,
                source_v,
                image_token_id,
                attention_mask,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown eval_mode={eval_mode!r}")

        stats["scored"] += 1
        stats["teacher_correct"] += int(teacher_pred == gold)
        stats["adapter_correct"] += int(adapter_pred == gold)
        stats["agree"] += int(adapter_pred == teacher_pred)
        stats["teacher_invalid"] += int(teacher_pred is None)
        stats["adapter_invalid"] += int(adapter_pred is None)
        stats["adapter_correct_when_teacher_correct"] += int(teacher_pred == gold and adapter_pred == gold)

        predictions.append({
            "index": item["index"],
            "gold": gold,
            "teacher": teacher_pred,
            "adapter": adapter_pred,
            "teacher_text": teacher_text,
            "adapter_text": adapter_text,
        })

        if (idx + 1) % log_every == 0:
            acc = stats["adapter_correct"] / stats["scored"]
            print(f"[{idx+1}/{len(dataset)}] adapter_acc={acc:.4f}", flush=True)

    return {"stats": stats, "predictions": predictions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified MMStar evaluation for LLaVA KV adapters and Qwen3-VL visual-delta adapters.")
    parser.add_argument("--model-kind", choices=("llava", "qwen"), default="llava")
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving image paths")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None, help="If set, only run this shard")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--eval-mode", choices=("generate", "logits"), default="generate")
    parser.add_argument("--answer-instruction", default="Answer directly with only the letter of the correct option.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def run_llava_single_shard(args, shard_id: int, num_shards: int):
    """Run evaluation on a single shard (one GPU)."""
    device = torch.device("cuda:0")
    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device="cuda:0")
    adapter, source_layers = load_adapter(args.checkpoint, model, device)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    full_dataset = MMStarDataset(
        args.data,
        processor,
        data_root=args.data_root,
        max_samples=args.max_samples,
        answer_instruction=args.answer_instruction,
    )
    total = len(full_dataset)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)

    full_dataset.rows = full_dataset.rows[start:end]
    print(f"Shard {shard_id}: samples [{start}, {end}) = {len(full_dataset)} items", flush=True)

    result = evaluate_llava_shard(
        model,
        processor,
        adapter,
        full_dataset,
        device,
        image_token_id,
        source_layers,
        args.log_every,
        args.max_new_tokens,
        args.eval_mode,
    )

    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
    return result



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
    text = processor.tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def generate_adapter_qwen(
    model,
    processor,
    adapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    full_mm_ids = mm_token_type_ids.clone()
    generated: list[int] = []
    eos_ids = _eos_token_ids(processor.tokenizer)
    for _ in range(max_new_tokens):
        inputs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": full_mm_ids,
        }
        logits, text_mask, _ = qwen_visual_delta_logits(model, adapter, inputs, collect_states=False)
        next_token = int(torch.argmax(logits[0, int(text_mask[0].sum().item()) - 1]).item())
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
def evaluate_qwen_shard(
    model,
    processor,
    adapter,
    dataset: QwenMMStarDataset,
    device: torch.device,
    log_every: int,
    max_new_tokens: int,
) -> dict:
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
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
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
    return {"stats": stats, "predictions": predictions, "output_mode": adapter.mode}


def run_qwen_single_shard(args: argparse.Namespace, shard_id: int, num_shards: int) -> dict:
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter, meta = load_qwen_visual_delta_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    if meta["missing"] or meta["unexpected"]:
        print(f"checkpoint load missing={meta['missing']} unexpected={meta['unexpected']}", flush=True)

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
    print(f"Shard {shard_id}: samples [{start}, {end}) = {len(dataset)} items; mode={adapter.mode}", flush=True)
    result = evaluate_qwen_shard(model, processor, adapter, dataset, device, args.log_every, args.max_new_tokens)
    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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
    output_modes = set()

    for shard_id in range(num_shards):
        path = Path(output_dir) / f"shard_{shard_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard result: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in all_stats:
            all_stats[key] += int(data["stats"].get(key, 0))
        all_predictions.extend(data.get("predictions", []))
        if data.get("output_mode"):
            output_modes.add(data["output_mode"])

    n = max(all_stats["scored"], 1)
    tc = all_stats["teacher_correct"]
    merged = {
        "total_samples": all_stats["scored"],
        "teacher_accuracy": all_stats["teacher_correct"] / n,
        "adapter_accuracy": all_stats["adapter_correct"] / n,
        "agreement": all_stats["agree"] / n,
        "retention": (all_stats["adapter_correct_when_teacher_correct"] / tc) if tc else 0.0,
        "teacher_invalid_rate": all_stats["teacher_invalid"] / n,
        "adapter_invalid_rate": all_stats["adapter_invalid"] / n,
    }
    if output_modes:
        merged["output_modes"] = sorted(output_modes)

    out_dir = Path(output_dir)
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(all_predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2), flush=True)
    return merged


def main() -> None:
    args = parse_args()
    if args.shard_id is not None:
        if args.model_kind == "qwen":
            run_qwen_single_shard(args, args.shard_id, args.num_shards)
        else:
            run_llava_single_shard(args, args.shard_id, args.num_shards)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
