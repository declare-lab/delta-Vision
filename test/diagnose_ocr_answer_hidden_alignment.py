"""Compare final-layer hidden states at the final Answer: marker.

For each rendered QA sample this diagnostic builds three prefill paths:

1. gold_text_context: text-only prompt with row["text_context"].
2. adapter_image_direct: image prompt using the adapter.
3. transcribed_context: adapter transcription fed back as text-only context.

All three prompts share the same question suffix ending in "Answer:". The script
finds the last token of the last "Answer:" marker and compares the final-layer
hidden vector at that position.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.eval_benchmarks import generate_adapter_qwen_decode_cache
from src.model import (
    dtype_from_name,
    build_qwen_initial_context,
    load_frozen_qwen3vl,
    load_qwen_embedding_adapter_checkpoint,
    prepare_qwen_embedding_adapter_inputs,
    prepare_qwen3vl_batch_inputs,
    qwen_embedding_adapter_logits,
)


COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."
QA_IMAGE_INSTRUCTION = "Use the image text to answer the question."
STRICT_ANSWER_INSTRUCTION = "Return only the final answer, with no explanation."
RENDERED_PAGE_INSTRUCTION = "Read the ordered page images and answer using only their text."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose hidden-state alignment at Answer: marker.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/ocr_overlap_qa_only_1024/paired_eval.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="", help="Optional image root override; row image_root is used by default.")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-transcribe-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--adapter-decode-cache-mode", choices=("shape_exact", "fast"), default="shape_exact")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
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


def text_chat_prompt(processor: Any, context: str, question: str) -> str:
    user_text = (
        f"Context:\n{context.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"{STRICT_ANSWER_INSTRUCTION}\n\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def image_question(question: str) -> str:
    return (
        f"{QA_IMAGE_INSTRUCTION}\n"
        f"{STRICT_ANSWER_INSTRUCTION}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )


def row_with_question(row: dict[str, Any], question: str) -> dict[str, Any]:
    out = dict(row)
    out["question"] = question
    return out


def find_last_subsequence(ids: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty token subsequence")
    last = -1
    width = len(needle)
    for idx in range(0, len(ids) - width + 1):
        if ids[idx : idx + width] == needle:
            last = idx + width - 1
    return last


def answer_marker_id_variants(tokenizer: Any) -> list[list[int]]:
    variants = ("Answer:", " Answer:", "\nAnswer:", "\n\nAnswer:")
    results: list[list[int]] = []
    for text in variants:
        ids = list(tokenizer(text, add_special_tokens=False).input_ids)
        if ids and ids not in results:
            results.append(ids)
    if not results:
        raise RuntimeError("tokenizer produced empty Answer: ids")
    return results


def find_answer_marker_position(ids: list[int], marker_variants: list[list[int]], tokenizer: Any) -> int:
    best = -1
    for marker_ids in marker_variants:
        pos = find_last_subsequence(ids, marker_ids)
        best = max(best, pos)
    if best >= 0:
        return best

    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    marker = "Answer:"
    marker_start = decoded.rfind(marker)
    if marker_start < 0:
        marker = "Answer"
        marker_start = decoded.rfind(marker)
    if marker_start < 0:
        raise ValueError(f"could not find Answer marker in decoded prompt tail={decoded[-500:]!r}")
    marker_end = marker_start + len(marker)
    for end_idx in range(len(ids)):
        prefix = tokenizer.decode(ids[: end_idx + 1], skip_special_tokens=False)
        if len(prefix) >= marker_end:
            return end_idx
    raise ValueError("could not map decoded Answer: marker to token index")


@torch.inference_mode()
def text_answer_hidden(
    model: Any,
    processor: Any,
    *,
    context: str,
    question: str,
    marker_variants: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = text_chat_prompt(processor, context, question)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    outputs = model(**inputs, return_dict=True, use_cache=False, output_hidden_states=True)
    valid_len = int(inputs["attention_mask"][0].sum().item())
    ids = [int(x) for x in inputs["input_ids"][0, :valid_len].tolist()]
    answer_pos = find_answer_marker_position(ids, marker_variants, processor.tokenizer)
    hidden = outputs.hidden_states[-1][0, answer_pos].detach().float().cpu()
    meta = {
        "prompt_tokens": valid_len,
        "answer_marker_pos": answer_pos,
        "answer_marker_id_variants": marker_variants,
    }
    return hidden, meta


@torch.inference_mode()
def adapter_answer_hidden(
    model: Any,
    processor: Any,
    adapter: Any,
    row: dict[str, Any],
    image_root: Path | None,
    marker_variants: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs, _, _, image_paths = prepare_qwen3vl_batch_inputs(processor, [row], image_root, device, include_answers=False)
    initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    prepared = prepare_qwen_embedding_adapter_inputs(
        model,
        adapter,
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        initial_hidden,
        position_ids,
    )
    text_positions = prepared["text_positions"]
    text_mask = prepared["text_mask"]
    text_ids = torch.gather(inputs["input_ids"], dim=1, index=text_positions)
    _, _, states = qwen_embedding_adapter_logits(
        model,
        adapter,
        inputs,
        initial_hidden=initial_hidden,
        position_ids=position_ids,
        collect_states=True,
        collect_state_indices={len(model.model.language_model.layers)},
    )
    if states is None:
        raise RuntimeError("adapter did not return hidden states")
    final_hidden = states[-1]
    valid_len = int(text_mask[0].sum().item())
    ids = [int(x) for x in text_ids[0, :valid_len].tolist()]
    answer_pos = find_answer_marker_position(ids, marker_variants, processor.tokenizer)
    hidden = final_hidden[0, answer_pos].detach().float().cpu()
    meta = {
        "text_tokens": valid_len,
        "answer_marker_pos": answer_pos,
        "image_paths": image_paths,
    }
    return hidden, meta


@torch.inference_mode()
def generate_adapter_transcription(
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
    transcribe_row = row_with_question(row, COPY_TRANSCRIPTION_INSTRUCTION)
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, [transcribe_row], image_root, device, include_answers=False)
    _, texts = generate_adapter_qwen_decode_cache(
        model,
        processor,
        adapter,
        inputs,
        max_new_tokens=int(max_new_tokens),
        decode_cache_mode=decode_cache_mode,
    )
    return texts[0].strip()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item())


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b).item())


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "cos_gold_adapter",
        "cos_gold_transcribed",
        "cos_adapter_transcribed",
        "l2_gold_adapter",
        "l2_gold_transcribed",
        "l2_adapter_transcribed",
    )
    out: dict[str, float] = {"samples": float(len(rows))}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = sum(vals) / max(1, len(vals))
        out[f"{key}_min"] = min(vals) if vals else 0.0
        out[f"{key}_max"] = max(vals) if vals else 0.0
    return out


def main() -> None:
    args = parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= int(args.shard_id) < int(args.num_shards)):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.data), int(args.max_samples))
    total_rows_before_shard = len(rows)
    rows = rows[int(args.shard_id) :: int(args.num_shards)]
    if not rows:
        raise RuntimeError(f"empty shard {args.shard_id}/{args.num_shards}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter, adapter_meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    image_root = Path(args.image_root) if args.image_root.strip() else None
    marker_variants = answer_marker_id_variants(processor.tokenizer)

    records: list[dict[str, Any]] = []
    for local_idx, row in enumerate(rows, start=1):
        start_s = time.perf_counter()
        question = cleaned_question(row)
        transcription = generate_adapter_transcription(
            model,
            processor,
            adapter,
            row,
            image_root,
            device,
            max_new_tokens=args.max_transcribe_tokens,
            decode_cache_mode=args.adapter_decode_cache_mode,
        )

        gold_h, gold_meta = text_answer_hidden(
            model,
            processor,
            context=str(row.get("text_context") or ""),
            question=question,
            marker_variants=marker_variants,
            device=device,
        )
        transcribed_h, transcribed_meta = text_answer_hidden(
            model,
            processor,
            context=transcription,
            question=question,
            marker_variants=marker_variants,
            device=device,
        )
        adapter_h, adapter_meta_row = adapter_answer_hidden(
            model,
            processor,
            adapter,
            row_with_question(row, image_question(question)),
            image_root,
            marker_variants,
            device,
        )

        record = {
            "index": row.get("index", local_idx - 1),
            "id": row.get("id", local_idx - 1),
            "question": question,
            "answer": row.get("answer"),
            "answers": row.get("answers"),
            "transcription_chars": len(transcription),
            "transcription_preview": transcription[:500],
            "cos_gold_adapter": cosine(gold_h, adapter_h),
            "cos_gold_transcribed": cosine(gold_h, transcribed_h),
            "cos_adapter_transcribed": cosine(adapter_h, transcribed_h),
            "l2_gold_adapter": l2(gold_h, adapter_h),
            "l2_gold_transcribed": l2(gold_h, transcribed_h),
            "l2_adapter_transcribed": l2(adapter_h, transcribed_h),
            "gold_meta": gold_meta,
            "transcribed_meta": transcribed_meta,
            "adapter_meta": adapter_meta_row,
            "elapsed_s": time.perf_counter() - start_s,
        }
        records.append(record)
        if local_idx % int(args.log_every) == 0 or local_idx == len(rows):
            summary = summarize(records)
            print(
                f"[{local_idx}/{len(rows)}] "
                f"cos_gold_adapter={summary['cos_gold_adapter_mean']:.4f} "
                f"cos_gold_transcribed={summary['cos_gold_transcribed_mean']:.4f} "
                f"cos_adapter_transcribed={summary['cos_adapter_transcribed_mean']:.4f}",
                flush=True,
            )

    result = {
        "task": "diagnose_ocr_answer_hidden_alignment",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "adapter_meta": adapter_meta,
        "max_samples": int(args.max_samples),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "total_rows_before_shard": int(total_rows_before_shard),
        "max_transcribe_tokens": int(args.max_transcribe_tokens),
        "answer_marker_id_variants": marker_variants,
        "metrics": summarize(records),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
