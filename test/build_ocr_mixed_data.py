"""Build mixed OCR-copy and native QA training JSONL.

The output references existing rendered images; it does not copy or rerender
image files. Each row keeps its own image_root so the trainer can resolve mixed
sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


QA_PREFIXES = (
    "Read the ordered page images and answer using only their text.",
    "The attached page images are consecutive pages in order.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build mixed OCR copy + native QA data.")
    parser.add_argument("--copy-train", default="data/train/rendered_text_copy_2048/paired_train.jsonl")
    parser.add_argument("--copy-eval", default="data/train/rendered_text_copy_2048/paired_eval.jsonl")
    parser.add_argument("--qa-train", default="data/train/rendered_text/paired_train.jsonl")
    parser.add_argument("--qa-eval", default="data/train/rendered_text/paired_eval.jsonl")
    parser.add_argument("--output-dir", default="data/train/ocr_mixed_copy70_qa30_2048")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--qa-ratio", type=float, default=0.3)
    parser.add_argument("--max-qa-answer-tokens", type=int, default=32)
    parser.add_argument("--max-qa-context-tokens", type=int, default=0)
    parser.add_argument("--max-copy-answer-tokens", type=int, default=2048)
    parser.add_argument("--min-copy-answer-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def norm_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_question(value: Any) -> str:
    question = norm_text(value)
    changed = True
    while changed:
        changed = False
        for prefix in QA_PREFIXES:
            if question.startswith(prefix):
                question = question[len(prefix) :].lstrip("\n ").strip()
                changed = True
    return question


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def answer_visible(row: dict[str, Any]) -> bool:
    if bool(row.get("answer_visible")):
        return True
    answer = norm_text(row.get("answer"))
    context = norm_text(row.get("text_context"))
    return bool(answer and answer.lower() in context.lower())


def normalize_copy_row(row: dict[str, Any], idx: int, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    answer = norm_text(row.get("answer"))
    if not answer:
        return None
    answer_tokens = int(row.get("answer_tokens") or token_len(tokenizer, answer))
    if answer_tokens < int(args.min_copy_answer_tokens) or answer_tokens > int(args.max_copy_answer_tokens):
        return None
    out = dict(row)
    dataset_name = Path(args.output_dir).name
    out["id"] = f"{dataset_name}:copy:{idx:06d}:{stable_id(row.get('id'), answer[:128])}"
    out["index"] = out["id"]
    out["task_type"] = "copy"
    out["answer"] = answer
    out["answers"] = [answer]
    out["raw_question"] = clean_question(row.get("question") or row.get("rendered_question"))
    out["question"] = "Transcribe all visible text in the image exactly. Preserve line breaks."
    out["rendered_question"] = out["question"]
    out["answer_tokens"] = answer_tokens
    return out


def normalize_qa_row(
    row: dict[str, Any],
    idx: int,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    image_root: Path,
) -> dict[str, Any] | None:
    answer = norm_text(row.get("answer"))
    question = clean_question(row.get("question") or row.get("rendered_question"))
    context = norm_text(row.get("text_context"))
    if not answer or not question or not context:
        return None
    if not answer_visible(row):
        return None
    context_tokens = token_len(tokenizer, context)
    if int(args.max_qa_context_tokens) > 0 and context_tokens > int(args.max_qa_context_tokens):
        return None
    answer_tokens = token_len(tokenizer, answer)
    if answer_tokens <= 0 or answer_tokens > int(args.max_qa_answer_tokens):
        return None
    out = dict(row)
    dataset_name = Path(args.output_dir).name
    out["id"] = f"{dataset_name}:qa:{idx:06d}:{stable_id(row.get('id'), question, answer)}"
    out["index"] = out["id"]
    out["task_type"] = "qa"
    out["answer"] = answer
    out["answers"] = [answer]
    out["raw_question"] = question
    out["question"] = question
    out["rendered_question"] = question
    out["text_context"] = context
    out["image_root"] = str(row.get("image_root") or image_root)
    out["answer_tokens"] = answer_tokens
    out["context_tokens"] = context_tokens
    return out


def sample_balanced(rows: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("source_dataset") or row.get("source") or "unknown")].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < int(size) and any(buckets.values()):
        key = keys[cursor % len(keys)]
        cursor += 1
        if buckets[key]:
            selected.append(buckets[key].pop())
    while len(selected) < int(size):
        selected.append(rng.choice(rows))
    rng.shuffle(selected)
    return selected[: int(size)]


def build_split(
    *,
    split: str,
    copy_path: Path,
    qa_path: Path,
    size: int,
    tokenizer: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    qa_size = int(round(int(size) * float(args.qa_ratio)))
    copy_size = int(size) - qa_size

    copy_rows = [
        out
        for idx, row in enumerate(load_jsonl(copy_path))
        if (out := normalize_copy_row(row, idx, tokenizer, args)) is not None
    ]
    qa_image_root = qa_path.parent
    qa_rows = [
        out
        for idx, row in enumerate(load_jsonl(qa_path))
        if (out := normalize_qa_row(row, idx, tokenizer, args, image_root=qa_image_root)) is not None
    ]
    if not copy_rows:
        raise RuntimeError(f"no usable copy rows from {copy_path}")
    if not qa_rows:
        raise RuntimeError(f"no usable qa rows from {qa_path}")
    selected = sample_balanced(copy_rows, copy_size, rng) + sample_balanced(qa_rows, qa_size, rng)
    rng.shuffle(selected)
    print(
        f"{split}: copy_pool={len(copy_rows)} qa_pool={len(qa_rows)} "
        f"selected_copy={copy_size} selected_qa={qa_size}",
        flush=True,
    )
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source_dataset") or "unknown") for row in rows)
    answer_tokens = [int(row.get("answer_tokens") or 0) for row in rows]
    context_tokens = [int(row.get("context_tokens") or 0) for row in rows if str(row.get("task_type")) == "qa"]
    return {
        "size": len(rows),
        "task_counts": dict(task_counts),
        "source_counts": dict(source_counts),
        "answer_tokens_min": min(answer_tokens) if answer_tokens else 0,
        "answer_tokens_max": max(answer_tokens) if answer_tokens else 0,
        "answer_tokens_mean": sum(answer_tokens) / max(1, len(answer_tokens)),
        "qa_context_tokens_min": min(context_tokens) if context_tokens else 0,
        "qa_context_tokens_max": max(context_tokens) if context_tokens else 0,
        "qa_context_tokens_mean": sum(context_tokens) / max(1, len(context_tokens)),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    train_rows = build_split(
        split="train",
        copy_path=Path(args.copy_train),
        qa_path=Path(args.qa_train),
        size=int(args.train_size),
        tokenizer=tokenizer,
        args=args,
        rng=rng,
    )
    eval_rows = build_split(
        split="eval",
        copy_path=Path(args.copy_eval),
        qa_path=Path(args.qa_eval),
        size=int(args.eval_size),
        tokenizer=tokenizer,
        args=args,
        rng=rng,
    )
    write_jsonl(out_dir / "paired_train.jsonl", train_rows)
    write_jsonl(out_dir / "paired_eval.jsonl", eval_rows)
    manifest = {
        "name": out_dir.name,
        "copy_train": str(Path(args.copy_train)),
        "copy_eval": str(Path(args.copy_eval)),
        "qa_train": str(Path(args.qa_train)),
        "qa_eval": str(Path(args.qa_eval)),
        "qa_ratio": float(args.qa_ratio),
        "max_qa_answer_tokens": int(args.max_qa_answer_tokens),
        "max_qa_context_tokens": int(args.max_qa_context_tokens),
        "max_copy_answer_tokens": int(args.max_copy_answer_tokens),
        "min_copy_answer_tokens": int(args.min_copy_answer_tokens),
        "train": summarize(train_rows),
        "eval": summarize(eval_rows),
    }
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
