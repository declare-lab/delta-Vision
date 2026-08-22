"""Build overlapping OCR-copy and QA data from the same rendered contexts.

Every QA row is derived from a context that also appears as a copy row. QA
contexts are not cropped; rows are filtered by full context token length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


COPY_QUESTION = "Transcribe all visible text in the image exactly. Preserve line breaks."
QA_PREFIXES = (
    "Read the ordered page images and answer using only their text.",
    "The attached page images are consecutive pages in order.",
)
_WORKER_TOKENIZER: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build overlap OCR copy + QA data.")
    parser.add_argument("--qa-train", default="data/train/rendered_text/paired_train.jsonl")
    parser.add_argument("--qa-eval", default="data/train/rendered_text/paired_eval.jsonl")
    parser.add_argument("--output-dir", default="data/train/ocr_overlap_copy70_qa30_1024")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--use-all-copy-contexts", action="store_true")
    parser.add_argument("--qa-ratio", type=float, default=0.3)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--min-context-tokens", type=int, default=1)
    parser.add_argument("--max-qa-answer-tokens", type=int, default=32)
    parser.add_argument("--tokenize-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
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


def answer_visible(row: dict[str, Any], answer: str, context: str) -> bool:
    if bool(row.get("answer_visible")):
        return True
    return bool(answer and answer.lower() in context.lower())


def token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = texts[start : start + int(batch_size)]
        encoded = tokenizer(batch, add_special_tokens=False)
        lengths.extend(len(ids) for ids in encoded.input_ids)
    return lengths


def init_tokenizer_worker(model_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)


def filter_candidate_batch(
    task: tuple[
        list[tuple[int, dict[str, Any], str, str, str]],
        str,
        int,
        int,
        int,
        int,
    ],
) -> list[dict[str, Any]]:
    batch, image_root, min_context_tokens, max_context_tokens, max_qa_answer_tokens, batch_size = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("tokenizer worker was not initialized")
    contexts = [item[2] for item in batch]
    answers = [item[3] for item in batch]
    context_lengths = token_lengths(tokenizer, contexts, batch_size)
    answer_lengths = token_lengths(tokenizer, answers, batch_size)
    candidates: list[dict[str, Any]] = []
    for (idx, row, context, answer, question), context_tokens, answer_tokens in zip(
        batch, context_lengths, answer_lengths, strict=True
    ):
        if context_tokens < min_context_tokens or context_tokens > max_context_tokens:
            continue
        if answer_tokens <= 0 or answer_tokens > max_qa_answer_tokens:
            continue
        out = dict(row)
        out["base_index"] = idx
        out["raw_question"] = question
        out["question"] = question
        out["rendered_question"] = question
        out["answer"] = answer
        out["answers"] = [answer]
        out["text_context"] = context
        out["context_tokens"] = int(context_tokens)
        out["answer_tokens"] = int(answer_tokens)
        out["image_root"] = image_root
        candidates.append(out)
    return candidates


def load_candidates(path: Path, tokenizer: Any, args: argparse.Namespace, *, target_size: int, rng: random.Random) -> list[dict[str, Any]]:
    raw_items: list[tuple[int, dict[str, Any], str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            context = norm_text(row.get("text_context"))
            answer = norm_text(row.get("answer"))
            question = clean_question(row.get("question") or row.get("rendered_question"))
            if not context or not answer or not question:
                continue
            if not answer_visible(row, answer, context):
                continue
            # Conservative character prefilter only; exact token filtering below
            # still uses the complete, uncropped context.
            if len(context) > int(args.max_context_tokens) * 12:
                continue
            raw_items.append((idx, row, context, answer, question))

    rng.shuffle(raw_items)
    image_root = str(path.parent)
    candidates: list[dict[str, Any]] = []
    batch_size = max(1, int(args.tokenize_batch_size))
    tasks = [
        (
            raw_items[start : start + batch_size],
            image_root,
            int(args.min_context_tokens),
            int(args.max_context_tokens),
            int(args.max_qa_answer_tokens),
            batch_size,
        )
        for start in range(0, len(raw_items), batch_size)
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        init_tokenizer_worker(str(args.model_path))
        for task in tasks:
            candidates.extend(filter_candidate_batch(task))
            if len(candidates) >= int(target_size):
                return candidates[: int(target_size)]
    else:
        with Pool(processes=workers, initializer=init_tokenizer_worker, initargs=(str(args.model_path),)) as pool:
            for batch_candidates in pool.imap_unordered(filter_candidate_batch, tasks, chunksize=1):
                candidates.extend(batch_candidates)
                if len(candidates) >= int(target_size):
                    pool.terminate()
                    pool.join()
                    return candidates[: int(target_size)]
    return candidates


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


def make_copy_row(base: dict[str, Any], dataset_name: str, idx: int) -> dict[str, Any]:
    out = dict(base)
    context = str(base["text_context"])
    out["id"] = f"{dataset_name}:copy:{idx:06d}:{stable_id(base.get('id'), context[:256])}"
    out["index"] = out["id"]
    out["task_type"] = "copy"
    out["question"] = COPY_QUESTION
    out["rendered_question"] = COPY_QUESTION
    out["raw_question"] = str(base.get("raw_question") or "")
    out["answer"] = context
    out["answers"] = [context]
    out["answer_tokens"] = int(base["context_tokens"])
    out["paired_context_id"] = stable_id(base.get("id"), context)
    return out


def make_qa_row(base: dict[str, Any], dataset_name: str, idx: int) -> dict[str, Any]:
    out = dict(base)
    answer = str(base["answer"])
    question = str(base["raw_question"])
    context = str(base["text_context"])
    out["id"] = f"{dataset_name}:qa:{idx:06d}:{stable_id(base.get('id'), question, answer)}"
    out["index"] = out["id"]
    out["task_type"] = "qa"
    out["question"] = question
    out["rendered_question"] = question
    out["answer"] = answer
    out["answers"] = [answer]
    out["paired_context_id"] = stable_id(base.get("id"), context)
    return out


def build_split(
    *,
    split: str,
    path: Path,
    size: int,
    tokenizer: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    requested_qa_size = int(round(int(size) * float(args.qa_ratio)))
    requested_copy_size = int(size) - requested_qa_size
    target_candidates = 10**12 if bool(args.use_all_copy_contexts) else requested_copy_size
    candidates = load_candidates(path, tokenizer, args, target_size=target_candidates, rng=rng)
    if not candidates:
        raise RuntimeError(f"no usable overlap candidates from {path}")
    if bool(args.use_all_copy_contexts):
        copy_size = len(candidates)
        qa_size = int(round(copy_size * float(args.qa_ratio) / max(1e-12, 1.0 - float(args.qa_ratio))))
        copy_bases = candidates
        rng.shuffle(copy_bases)
    else:
        copy_size = requested_copy_size
        qa_size = requested_qa_size
        copy_bases = sample_balanced(candidates, copy_size, rng)
    qa_bases = [rng.choice(copy_bases) for _ in range(qa_size)]
    dataset_name = Path(args.output_dir).name
    rows = [make_copy_row(row, dataset_name, idx) for idx, row in enumerate(copy_bases)]
    rows.extend(make_qa_row(row, dataset_name, idx) for idx, row in enumerate(qa_bases))
    rng.shuffle(rows)
    print(
        f"{split}: candidate_contexts={len(candidates)} selected_copy={copy_size} selected_qa={qa_size}",
        flush=True,
    )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source_dataset") or row.get("source") or "unknown") for row in rows)
    answer_tokens = [int(row.get("answer_tokens") or 0) for row in rows]
    context_tokens = [int(row.get("context_tokens") or 0) for row in rows]
    qa_context_ids = {str(row.get("paired_context_id")) for row in rows if row.get("task_type") == "qa"}
    copy_context_ids = {str(row.get("paired_context_id")) for row in rows if row.get("task_type") == "copy"}
    return {
        "size": len(rows),
        "task_counts": dict(task_counts),
        "source_counts": dict(source_counts),
        "answer_tokens_min": min(answer_tokens) if answer_tokens else 0,
        "answer_tokens_max": max(answer_tokens) if answer_tokens else 0,
        "answer_tokens_mean": sum(answer_tokens) / max(1, len(answer_tokens)),
        "context_tokens_min": min(context_tokens) if context_tokens else 0,
        "context_tokens_max": max(context_tokens) if context_tokens else 0,
        "context_tokens_mean": sum(context_tokens) / max(1, len(context_tokens)),
        "qa_contexts": len(qa_context_ids),
        "copy_contexts": len(copy_context_ids),
        "qa_contexts_missing_copy": len(qa_context_ids - copy_context_ids),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    train_rows = build_split(
        split="train",
        path=Path(args.qa_train),
        size=int(args.train_size),
        tokenizer=tokenizer,
        args=args,
        rng=rng,
    )
    eval_rows = build_split(
        split="eval",
        path=Path(args.qa_eval),
        size=int(args.eval_size),
        tokenizer=tokenizer,
        args=args,
        rng=rng,
    )
    write_jsonl(out_dir / "paired_train.jsonl", train_rows)
    write_jsonl(out_dir / "paired_eval.jsonl", eval_rows)
    manifest = {
        "name": out_dir.name,
        "qa_train": str(Path(args.qa_train)),
        "qa_eval": str(Path(args.qa_eval)),
        "qa_ratio": float(args.qa_ratio),
        "max_context_tokens": int(args.max_context_tokens),
        "max_qa_answer_tokens": int(args.max_qa_answer_tokens),
        "overlap_rule": "Every QA paired_context_id appears in copy rows.",
        "train": summarize(train_rows),
        "eval": summarize(eval_rows),
    }
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
