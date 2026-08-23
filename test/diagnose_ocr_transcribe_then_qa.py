"""Diagnose whether OCR adapter failures come from memory or answer extraction.

The experiment runs three paths on rendered-context QA rows:

1. gold_context_qa: put row["text_context"] into a text-only LLM prompt.
2. adapter_image_qa: ask the adapter to answer directly from the image.
3. transcribed_context_qa: ask the adapter to transcribe the image first, then
   put that generated transcription into a text-only LLM prompt and answer.
4. read_then_answer_qa: ask the adapter in one image prompt to transcribe the
   relevant text first, then answer.

If path 3 recovers path 1 while path 2 fails, the adapter can expose readable
context but the image-memory query path is not robust to the QA prompt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.benchmarks import normalize_answer, score_prediction
from src.eval_benchmarks import generate_adapter_qwen_decode_cache, generate_teacher_qwen
from src.model import (
    dtype_from_name,
    load_frozen_qwen3vl,
    load_qwen_embedding_adapter_checkpoint,
    prepare_qwen3vl_batch_inputs,
)


COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."
DEFAULT_ANSWER_INSTRUCTION = "Answer directly with a short phrase."
STRICT_ANSWER_INSTRUCTION = "Return only the final answer, with no explanation."
RENDERED_PAGE_INSTRUCTION = "Read the ordered page images and answer using only their text."
READ_THEN_ANSWER_TEMPLATE = """First transcribe the relevant text from the image that helps answer the question.
Then answer the question using only that transcribed text.

Question: {question}

Use this exact output format:
Relevant text:
<transcription>

Final answer:
<answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose OCR transcribe-then-text-QA behavior.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--data",
        default="data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_span_100.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/"
            "qwen_embedding_adapter_step500.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/eval/qwen/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/"
            "eager/diagnose_transcribe_then_qa"
        ),
    )
    parser.add_argument("--image-root", default="", help="Optional image root override; row image_root is used by default.")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-transcribe-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--max-read-then-answer-tokens", type=int, default=512)
    parser.add_argument(
        "--answer-instruction",
        default=STRICT_ANSWER_INSTRUCTION,
        help="Instruction appended to text-QA and direct adapter-QA prompts.",
    )
    parser.add_argument("--metric", default="relaxed_exact")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--adapter-decode-cache-mode", choices=("shape_exact", "fast"), default="shape_exact")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-gold-context", action="store_true")
    parser.add_argument("--skip-direct-adapter", action="store_true")
    parser.add_argument("--skip-transcribe-then-qa", action="store_true")
    parser.add_argument("--skip-read-then-answer", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def cleaned_question(row: dict[str, Any]) -> str:
    question = str(row.get("raw_question") or row.get("question") or row.get("rendered_question") or "").strip()
    prefixes = (
        RENDERED_PAGE_INSTRUCTION,
        "The attached page images are consecutive pages in order.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix) :].lstrip("\n ").strip()
                changed = True
    return question


def make_text_qa_prompt(processor: Any, context: str, question: str, answer_instruction: str) -> str:
    user_text = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}"
    if answer_instruction.strip():
        user_text = f"{user_text}\n\n{answer_instruction.strip()}"
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_text_answer(
    model: Any,
    processor: Any,
    *,
    context: str,
    question: str,
    answer_instruction: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = make_text_qa_prompt(processor, context, question, answer_instruction)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", eos_token_id)
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    generated = model.generate(**kwargs)
    return processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_values(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    answers = row.get("answers")
    if isinstance(answers, list):
        values.extend(answers)
    elif answers is not None:
        values.append(answers)
    answer = row.get("answer")
    if isinstance(answer, list):
        values.extend(answer)
    elif answer is not None:
        values.append(answer)

    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("answer", value.get("text", value.get("label")))
        text = str(value or "").strip()
        if text:
            result.append(text)
    return result


def score_token_f1(row: dict[str, Any], text: str) -> dict[str, Any]:
    pred_norm = normalize_answer(text)
    refs = answer_values(row)
    if not refs:
        return {"prediction": pred_norm, "gold": "", "score": 0.0, "invalid": not bool(pred_norm)}
    scored = [(token_f1(text, ref), ref) for ref in refs]
    best_score, best_ref = max(scored, key=lambda item: item[0])
    return {
        "prediction": pred_norm,
        "gold": normalize_answer(best_ref),
        "score": float(best_score),
        "invalid": not bool(pred_norm),
    }


def score_row(row: dict[str, Any], text: str, metric: str) -> dict[str, Any]:
    if metric in {"token_f1", "f1"}:
        return score_token_f1(row, text)
    return score_prediction(
        metric=metric,
        prediction_text=text,
        answer=row.get("answer"),
        answers=row.get("answers"),
        choices=row.get("choices"),
        question=row.get("question"),
    )


def extract_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    marker = "Final answer:"
    lower = raw.lower()
    marker_lower = marker.lower()
    pos = lower.rfind(marker_lower)
    if pos >= 0:
        return raw[pos + len(marker) :].strip()
    for fallback in ("Answer:", "Final:"):
        pos = lower.rfind(fallback.lower())
        if pos >= 0:
            return raw[pos + len(fallback) :].strip()
    return raw


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    invalid = sum(1 for item in scored if item.get("invalid"))
    score = sum(float(item.get("score", 0.0)) for item in scored) / max(1, total)
    return {
        "samples": total,
        "score": score,
        "accuracy": score,
        "invalid_rate": invalid / max(1, total),
    }


def prepare_row_for_question(row: dict[str, Any], question: str) -> dict[str, Any]:
    copy = dict(row)
    copy["question"] = question
    return copy


@torch.inference_mode()
def generate_adapter_text(
    model: Any,
    processor: Any,
    adapter: Any,
    row: dict[str, Any],
    image_root: Path | None,
    device: torch.device,
    *,
    max_new_tokens: int,
    decode_cache_mode: str,
) -> str:
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, [row], image_root, device, include_answers=False)
    _, texts = generate_adapter_qwen_decode_cache(
        model,
        processor,
        adapter,
        inputs,
        max_new_tokens=int(max_new_tokens),
        decode_cache_mode=decode_cache_mode,
    )
    return texts[0].strip()


@torch.inference_mode()
def generate_image_teacher_text(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    image_root: Path | None,
    device: torch.device,
    *,
    max_new_tokens: int,
) -> str:
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, [row], image_root, device, include_answers=False)
    _, text = generate_teacher_qwen(
        model,
        processor,
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        inputs["mm_token_type_ids"],
        max_new_tokens=int(max_new_tokens),
    )
    return text.strip()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.data), int(args.max_samples))
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= int(args.shard_id) < int(args.num_shards)):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    total_rows_before_shard = len(rows)
    rows = rows[int(args.shard_id) :: int(args.num_shards)]
    if not rows:
        raise RuntimeError(
            f"empty shard {args.shard_id}/{args.num_shards} after loading {total_rows_before_shard} rows"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter, adapter_meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    image_root = Path(args.image_root) if args.image_root.strip() else None

    predictions: list[dict[str, Any]] = []
    scores: dict[str, list[dict[str, Any]]] = {
        "gold_context_qa": [],
        "image_teacher_qa": [],
        "adapter_image_qa": [],
        "transcribed_context_qa": [],
        "read_then_answer_qa": [],
    }
    timings: dict[str, float] = {key: 0.0 for key in scores}
    timings["adapter_transcription"] = 0.0

    for idx, row in enumerate(rows, start=1):
        question = cleaned_question(row)
        answer_instruction = str(args.answer_instruction or "").strip()
        qa_question = f"{question}\n{answer_instruction}" if answer_instruction else question
        direct_row = prepare_row_for_question(row, qa_question)
        transcribe_row = prepare_row_for_question(row, COPY_TRANSCRIPTION_INSTRUCTION)
        read_then_answer_row = prepare_row_for_question(row, READ_THEN_ANSWER_TEMPLATE.format(question=question))

        record: dict[str, Any] = {
            "index": row.get("index", idx - 1),
            "id": row.get("id", idx - 1),
            "question": question,
            "answer": row.get("answer"),
            "answers": row.get("answers"),
            "image": row.get("image"),
            "images": row.get("images"),
        }

        if not args.skip_gold_context:
            start = time.perf_counter()
            text = generate_text_answer(
                model,
                processor,
                context=str(row.get("text_context") or ""),
                question=question,
                answer_instruction=answer_instruction,
                device=device,
                max_new_tokens=args.max_answer_tokens,
            )
            timings["gold_context_qa"] += time.perf_counter() - start
            score = score_row(row, text, args.metric)
            scores["gold_context_qa"].append(score)
            record["gold_context_text"] = text
            record["gold_context_eval"] = score

        start = time.perf_counter()
        image_teacher_text = generate_image_teacher_text(
            model,
            processor,
            direct_row,
            image_root,
            device,
            max_new_tokens=args.max_answer_tokens,
        )
        timings["image_teacher_qa"] += time.perf_counter() - start
        image_teacher_score = score_row(row, image_teacher_text, args.metric)
        scores["image_teacher_qa"].append(image_teacher_score)
        record["image_teacher_text"] = image_teacher_text
        record["image_teacher_eval"] = image_teacher_score

        if not args.skip_direct_adapter:
            start = time.perf_counter()
            text = generate_adapter_text(
                model,
                processor,
                adapter,
                direct_row,
                image_root,
                device,
                max_new_tokens=args.max_answer_tokens,
                decode_cache_mode=args.adapter_decode_cache_mode,
            )
            timings["adapter_image_qa"] += time.perf_counter() - start
            score = score_row(row, text, args.metric)
            scores["adapter_image_qa"].append(score)
            record["adapter_image_text"] = text
            record["adapter_image_eval"] = score

        if not args.skip_transcribe_then_qa:
            start = time.perf_counter()
            transcription = generate_adapter_text(
                model,
                processor,
                adapter,
                transcribe_row,
                image_root,
                device,
                max_new_tokens=args.max_transcribe_tokens,
                decode_cache_mode=args.adapter_decode_cache_mode,
            )
            timings["adapter_transcription"] += time.perf_counter() - start
            start = time.perf_counter()
            text = generate_text_answer(
                model,
                processor,
                context=transcription,
                question=question,
                answer_instruction=answer_instruction,
                device=device,
                max_new_tokens=args.max_answer_tokens,
            )
            timings["transcribed_context_qa"] += time.perf_counter() - start
            score = score_row(row, text, args.metric)
            scores["transcribed_context_qa"].append(score)
            record["adapter_transcription"] = transcription
            record["transcribed_context_text"] = text
            record["transcribed_context_eval"] = score

        if not args.skip_read_then_answer:
            start = time.perf_counter()
            raw_text = generate_adapter_text(
                model,
                processor,
                adapter,
                read_then_answer_row,
                image_root,
                device,
                max_new_tokens=args.max_read_then_answer_tokens,
                decode_cache_mode=args.adapter_decode_cache_mode,
            )
            timings["read_then_answer_qa"] += time.perf_counter() - start
            answer_text = extract_final_answer(raw_text)
            score = score_row(row, answer_text, args.metric)
            scores["read_then_answer_qa"].append(score)
            record["read_then_answer_raw_text"] = raw_text
            record["read_then_answer_text"] = answer_text
            record["read_then_answer_eval"] = score

        predictions.append(record)
        if idx % int(args.log_every) == 0 or idx == len(rows):
            parts = [f"[{idx}/{len(rows)}]"]
            for name, values in scores.items():
                if values:
                    parts.append(f"{name}={summarize(values)['score']:.4f}")
            print(" ".join(parts), flush=True)

    result = {
        "task": "diagnose_ocr_transcribe_then_qa",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "adapter_meta": adapter_meta,
        "metric": args.metric,
        "max_samples": int(args.max_samples),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "total_rows_before_shard": int(total_rows_before_shard),
        "max_transcribe_tokens": int(args.max_transcribe_tokens),
        "max_answer_tokens": int(args.max_answer_tokens),
        "max_read_then_answer_tokens": int(args.max_read_then_answer_tokens),
        "answer_instruction": str(args.answer_instruction),
        "total_samples": len(predictions),
        "metrics": {name: summarize(values) for name, values in scores.items() if values},
        "timing": {
            name: {"total_s": value, "avg_s": value / max(1, len(predictions))}
            for name, value in timings.items()
            if value > 0
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
