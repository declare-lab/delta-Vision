"""Evaluate OCR/rendered-text copy/transcription rows.

Metrics are generation based:
  - CER: character edit distance / reference length, lower is better
  - token_f1: whitespace-token overlap F1, higher is better
  - exact: normalized-whitespace exact match, higher is better
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.eval_benchmarks import generate_adapter_qwen_decode_cache, generate_teacher_qwen
from src.model import dtype_from_name, load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint, prepare_qwen3vl_batch_inputs


COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Eval Qwen OCR/rendered-text copy/transcription data.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/rendered_text_copy_2048/paired_eval.jsonl")
    parser.add_argument("--image-root", default="data/train/rendered_text_copy_2048")
    parser.add_argument("--checkpoint", default="", help="Optional adapter checkpoint. If empty, adapter eval is skipped.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=360)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--eval-text-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-image-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-decode-cache-mode", choices=("shape_exact", "fast"), default="shape_exact")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def normalized(text: str) -> str:
    return " ".join(str(text).strip().split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalized(prediction).split()
    ref_tokens = normalized(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in ref_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_text(prediction: str, reference: str) -> dict[str, float]:
    ref = str(reference)
    pred = str(prediction)
    cer = levenshtein(pred, ref) / max(1, len(ref))
    return {
        "cer": float(cer),
        "char_acc": float(max(0.0, 1.0 - cer)),
        "token_f1": float(token_f1(pred, ref)),
        "exact": float(normalized(pred) == normalized(ref)),
        "pred_chars": float(len(pred)),
        "ref_chars": float(len(ref)),
    }


def add_generation_stats(score: dict[str, float], processor: Any, text: str, max_new_tokens: int) -> dict[str, float]:
    pred_tokens = len(processor.tokenizer(str(text), add_special_tokens=False).input_ids)
    score = dict(score)
    score["pred_tokens"] = float(pred_tokens)
    score["hit_max_new_tokens"] = float(pred_tokens >= int(max_new_tokens))
    return score


def mean_metric(items: list[dict[str, float]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item[key]) for item in items) / len(items))


def summarize(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "samples": float(len(items)),
        "cer": mean_metric(items, "cer"),
        "char_acc": mean_metric(items, "char_acc"),
        "token_f1": mean_metric(items, "token_f1"),
        "exact": mean_metric(items, "exact"),
        "pred_chars": mean_metric(items, "pred_chars"),
        "ref_chars": mean_metric(items, "ref_chars"),
        "pred_tokens": mean_metric(items, "pred_tokens") if "pred_tokens" in items[0] else 0.0,
        "hit_max_new_tokens": mean_metric(items, "hit_max_new_tokens") if "hit_max_new_tokens" in items[0] else 0.0,
    }


def text_teacher_prompt(processor: Any, row: dict[str, Any]) -> str:
    context = str(row.get("text_context") or row.get("answer") or "").strip()
    question = str(row.get("question") or COPY_TRANSCRIPTION_INSTRUCTION).strip()
    user_text = f"Text:\n{context}\n\n{question}"
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_text_teacher(model: Any, processor: Any, row: dict[str, Any], device: torch.device, max_new_tokens: int) -> str:
    prompt = text_teacher_prompt(processor, row)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", eos_token_id)
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    generated = model.generate(**kwargs)
    return processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def eval_one_image_teacher(model: Any, processor: Any, row: dict[str, Any], image_root: Path, device: torch.device, max_new_tokens: int) -> str:
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, [row], image_root, device, include_answers=False)
    _, text = generate_teacher_qwen(
        model,
        processor,
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        inputs["mm_token_type_ids"],
        max_new_tokens=max_new_tokens,
    )
    return text.strip()


def eval_one_adapter(
    model: Any,
    processor: Any,
    adapter: Any,
    row: dict[str, Any],
    image_root: Path,
    device: torch.device,
    max_new_tokens: int,
    decode_cache_mode: str,
) -> str:
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, [row], image_root, device, include_answers=False)
    _, texts = generate_adapter_qwen_decode_cache(
        model,
        processor,
        adapter,
        inputs,
        max_new_tokens=max_new_tokens,
        decode_cache_mode=decode_cache_mode,
    )
    return texts[0].strip()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.data), args.max_samples)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter = None
    adapter_meta: dict[str, Any] | None = None
    if args.checkpoint and args.eval_adapter:
        adapter, adapter_meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, dtype)

    image_root = Path(args.image_root) if args.image_root else None
    predictions: list[dict[str, Any]] = []
    scores: dict[str, list[dict[str, float]]] = {"text_teacher": [], "image_teacher": [], "adapter": []}
    timings: dict[str, float] = {"text_teacher": 0.0, "image_teacher": 0.0, "adapter": 0.0}

    for idx, row in enumerate(rows, start=1):
        reference = str(row.get("answer") or "")
        record: dict[str, Any] = {
            "index": row.get("index", idx - 1),
            "id": row.get("id", idx - 1),
            "source_dataset": row.get("source_dataset"),
            "answer": reference,
        }
        if args.eval_text_teacher:
            start = time.perf_counter()
            text = generate_text_teacher(model, processor, row, device, args.max_new_tokens)
            timings["text_teacher"] += time.perf_counter() - start
            score = add_generation_stats(score_text(text, reference), processor, text, args.max_new_tokens)
            scores["text_teacher"].append(score)
            record["text_teacher_text"] = text
            record["text_teacher_eval"] = score
        if args.eval_image_teacher:
            start = time.perf_counter()
            text = eval_one_image_teacher(model, processor, row, image_root, device, args.max_new_tokens)
            timings["image_teacher"] += time.perf_counter() - start
            score = add_generation_stats(score_text(text, reference), processor, text, args.max_new_tokens)
            scores["image_teacher"].append(score)
            record["image_teacher_text"] = text
            record["image_teacher_eval"] = score
        if adapter is not None:
            start = time.perf_counter()
            text = eval_one_adapter(model, processor, adapter, row, image_root, device, args.max_new_tokens, args.adapter_decode_cache_mode)
            timings["adapter"] += time.perf_counter() - start
            score = add_generation_stats(score_text(text, reference), processor, text, args.max_new_tokens)
            scores["adapter"].append(score)
            record["adapter_text"] = text
            record["adapter_eval"] = score
        predictions.append(record)

        if idx % args.log_every == 0 or idx == len(rows):
            parts = [f"[{idx}/{len(rows)}]"]
            for name in ("text_teacher", "image_teacher", "adapter"):
                if scores[name]:
                    summary = summarize(scores[name])
                    parts.append(f"{name}: cer={summary['cer']:.4f} f1={summary['token_f1']:.4f} exact={summary['exact']:.4f}")
            print(" ".join(parts), flush=True)

    result = {
        "task": "ocr_copy",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "adapter_meta": adapter_meta,
        "max_new_tokens": int(args.max_new_tokens),
        "total_samples": len(predictions),
        "metrics": {name: summarize(values) for name, values in scores.items() if values},
        "timing": {
            name: {
                "total_s": value,
                "avg_s": value / max(1, len(scores[name])),
            }
            for name, value in timings.items()
            if scores[name]
        },
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
