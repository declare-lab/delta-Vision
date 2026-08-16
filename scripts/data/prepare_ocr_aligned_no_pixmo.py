#!/usr/bin/env python3
"""Build an OCR-aligned short-answer train JSONL without changing prompts."""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import string
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "pixmo_clean_ocrmix_300k" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "ocr_aligned_no_pixmo_filtered"

EXCLUDED_SOURCES = {"pixmo_clean"}
BINARY_ANSWERS = {"yes", "no", "true", "false"}
UNANSWERABLE_EXACT = {
    "unknown",
    "n/a",
    "na",
    "none",
}
UNANSWERABLE_PHRASES = (
    "unanswerable",
    "not answerable",
    "cannot answer",
    "can't answer",
    "can not answer",
    "cannot be determined",
    "can't be determined",
    "not enough information",
    "no answer",
)
OCR_DUMP_PATTERNS = (
    "ocr detected words",
    "**ocr detected words:**",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Filter the existing OCR mix into a short-answer aligned dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--max-answer-words", type=int, default=30)
    parser.add_argument("--target-samples", type=int, default=0, help="0 keeps every row that passes filters.")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--data-root", default=str(ROOT_DIR))
    return parser.parse_args()


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalize_answer(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return normalized.strip(" \t\r\n" + string.punctuation)


def word_count(value: str) -> int:
    return len(re.findall(r"\S+", value.strip()))


def resolve_image(row: dict[str, Any], data_root: Path) -> Path:
    image = Path(str(row.get("image", "")))
    if image.is_absolute():
        return image
    image_root_value = clean_text(row.get("image_root"))
    if image_root_value:
        image_root = Path(image_root_value)
        if not image_root.is_absolute():
            image_root = data_root / image_root
        return image_root / image
    return data_root / image


def reject_reason(row: dict[str, Any], *, max_answer_words: int, check_images: bool, data_root: Path) -> str:
    source = clean_text(row.get("source"))
    question = clean_text(row.get("question"))
    answer = clean_text(row.get("answer"))
    answer_norm = normalize_answer(answer)
    joined_norm = normalize_answer(f"{question}\n{answer}")

    if source in EXCLUDED_SOURCES:
        return "excluded_source"
    if not question or not answer:
        return "empty_text"
    if any(pattern in joined_norm for pattern in OCR_DUMP_PATTERNS):
        return "ocr_dump"
    if word_count(answer) > max_answer_words:
        return "long_answer"
    if answer_norm in BINARY_ANSWERS:
        return "binary_answer"
    if answer_norm in UNANSWERABLE_EXACT or any(pattern in answer_norm for pattern in UNANSWERABLE_PHRASES):
        return "unanswerable"
    if check_images and not resolve_image(row, data_root).exists():
        return "missing_image"
    return ""


def select_rows(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if target <= 0 or len(rows) <= target:
        return rows
    return random.Random(seed).sample(rows, target)


def atomic_write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    data_root = Path(args.data_root).expanduser()

    kept: list[dict[str, Any]] = []
    source_counts = Counter()
    rejected = Counter()
    rejected_by_source = Counter()
    answer_words: list[int] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = clean_text(row.get("source"))
            source_counts[source] += 1
            reason = reject_reason(
                row,
                max_answer_words=int(args.max_answer_words),
                check_images=bool(args.check_images),
                data_root=data_root,
            )
            if reason:
                rejected[reason] += 1
                rejected_by_source[f"{source}:{reason}"] += 1
                continue
            kept.append(row)
            answer_words.append(word_count(clean_text(row.get("answer"))))

    selected = select_rows(kept, int(args.target_samples), int(args.seed))
    random.Random(int(args.seed)).shuffle(selected)
    atomic_write_jsonl(selected, output_path)

    latest = output_dir / "latest.jsonl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(output_path.name)
    except OSError:
        shutil.copyfile(output_path, latest)

    selected_source_counts = Counter(clean_text(row.get("source")) for row in selected)
    selected_answer_words = [word_count(clean_text(row.get("answer"))) for row in selected]
    selected_answer_words.sort()
    manifest = {
        "input": str(input_path),
        "output_jsonl": str(output_path),
        "filter": {
            "excluded_sources": sorted(EXCLUDED_SOURCES),
            "max_answer_words": int(args.max_answer_words),
            "removed_binary_answers": sorted(BINARY_ANSWERS),
            "removed_unanswerable_exact": sorted(UNANSWERABLE_EXACT),
            "removed_unanswerable_phrases": list(UNANSWERABLE_PHRASES),
            "removed_ocr_dump_patterns": list(OCR_DUMP_PATTERNS),
            "prompt_policy": "preserve original question text; no prompt unification",
        },
        "raw_samples": sum(source_counts.values()),
        "kept_before_target": len(kept),
        "selected_samples": len(selected),
        "raw_source_counts": dict(source_counts),
        "selected_source_counts": dict(selected_source_counts),
        "rejected": dict(rejected),
        "rejected_by_source": dict(rejected_by_source),
        "args": vars(args),
    }
    if selected_answer_words:
        manifest["selected_answer_word_stats"] = {
            "avg": sum(selected_answer_words) / len(selected_answer_words),
            "p50": selected_answer_words[len(selected_answer_words) // 2],
            "p90": selected_answer_words[int(len(selected_answer_words) * 0.9)],
            "max": selected_answer_words[-1],
        }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
