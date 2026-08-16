#!/usr/bin/env python3
"""Build an OCRBench-targeted OCR training mix.

The existing OCR-aligned dataset is kept as-is. This script only appends
task-targeted formula and KIE/receipt field samples, while avoiding full-page
OCR dumps and unified prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_JSONL = ROOT_DIR / "data" / "ocr_aligned_no_pixmo_filtered" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "ocrbench_target_mix_v1"

HME_DATASET = "SoyVitou/Latex-Math-HME100K"
CORD_DATASET = "mychen76/receipt_cord_ocr_v2"

FORMULA_PROMPTS = (
    "Convert the handwritten formula in the image to LaTeX.",
    "What formula is written in the image? Answer in LaTeX.",
    "Read the mathematical expression in the image.",
)

FIELD_QUESTIONS = {
    "total.total": (
        "What is the total amount on the receipt?",
        "Read the receipt total.",
    ),
    "total.cash": (
        "What cash amount is shown on the receipt?",
        "Read the cash amount on the receipt.",
    ),
    "total.change": (
        "What change amount is shown on the receipt?",
        "Read the change amount on the receipt.",
    ),
    "subtotal.subtotal": (
        "What is the subtotal on the receipt?",
        "Read the receipt subtotal.",
    ),
    "subtotal.service": (
        "What service charge is shown on the receipt?",
        "Read the service charge on the receipt.",
    ),
    "subtotal.tax": (
        "What tax amount is shown on the receipt?",
        "Read the tax amount on the receipt.",
    ),
    "subtotal.etc": (
        "What other charge is shown on the receipt?",
        "Read the other charge on the receipt.",
    ),
}

ITEM_VALUE_QUESTIONS = (
    "What is the price of {item_name}?",
    "Read the value for {item_name}.",
)
ITEM_QUANTITY_QUESTIONS = (
    "What quantity is shown for {item_name}?",
    "Read the quantity for {item_name}.",
)

BINARY_ANSWERS = {"yes", "no", "true", "false"}
UNANSWERABLE_EXACT = {"unknown", "n/a", "na", "none", "null"}
UNANSWERABLE_PHRASES = (
    "unanswerable",
    "not answerable",
    "cannot answer",
    "can't answer",
    "can not answer",
    "cannot be determined",
    "not enough information",
    "no answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build OCRBench-targeted Qwen training JSONL.")
    parser.add_argument("--base-jsonl", default=str(DEFAULT_BASE_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--base-samples", type=int, default=0, help="0 keeps all base rows.")
    parser.add_argument(
        "--base-source-targets",
        default="",
        help="Optional comma-separated source=count targets for base rows; overrides --base-samples.",
    )
    parser.add_argument("--hme-samples", type=int, default=50_000)
    parser.add_argument("--cord-samples", type=int, default=30_000)
    parser.add_argument("--cord-max-items-per-doc", type=int, default=8)
    parser.add_argument("--hme-dataset", default=HME_DATASET)
    parser.add_argument("--cord-dataset", default=CORD_DATASET)
    parser.add_argument("--cord-split", default="train")
    parser.add_argument("--max-answer-words", type=int, default=30)
    parser.add_argument("--compact-hme-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_answer(value: str) -> str:
    return clean_text(value).lower().strip(" \t\r\n" + string.punctuation)


def word_count(value: str) -> int:
    return len(re.findall(r"\S+", value.strip()))


def valid_short_answer(answer: str, max_answer_words: int) -> bool:
    answer = clean_text(answer)
    answer_norm = normalize_answer(answer)
    if not answer:
        return False
    if answer_norm in BINARY_ANSWERS or answer_norm in UNANSWERABLE_EXACT:
        return False
    if any(phrase in answer_norm for phrase in UNANSWERABLE_PHRASES):
        return False
    return word_count(answer) <= max_answer_words


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if target <= 0 or len(rows) <= target:
        return list(rows)
    return random.Random(seed).sample(rows, target)


def parse_count_map(spec: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected source=count item, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = int(raw_value.strip())
        if not key:
            raise ValueError(f"empty source name in {item!r}")
        if value < 0:
            raise ValueError(f"negative target for source {key!r}: {value}")
        targets[key] = value
    return targets


def select_base_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter, dict[str, int]]:
    rows = load_jsonl(Path(args.base_jsonl).expanduser())
    targets = parse_count_map(str(args.base_source_targets))
    if not targets:
        return select_rows(rows, int(args.base_samples), int(args.seed)), Counter(), {}

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(clean_text(row.get("source")), []).append(row)

    selected: list[dict[str, Any]] = []
    stats = Counter()
    for offset, (source, target) in enumerate(targets.items()):
        available = by_source.get(source, [])
        picked = select_rows(available, int(target), int(args.seed) + offset * 1009)
        selected.extend(picked)
        stats[f"base_source_available:{source}"] = len(available)
        stats[f"base_source_selected:{source}"] = len(picked)
        if len(picked) < int(target):
            stats[f"base_source_shortfall:{source}"] = int(target) - len(picked)
    return selected, stats, targets


def image_output_path(output_dir: Path, source: str, index: int, suffix: str = ".png") -> Path:
    return output_dir / "images" / source / f"{index:08d}{suffix}"


def save_pil_image(image: Image.Image, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        image.convert("RGB").save(tmp_path, format="PNG")
        tmp_path.replace(path)
        return True
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def json_load_repeated(value: object) -> Any:
    current = value
    for _ in range(4):
        if not isinstance(current, str):
            return current
        stripped = current.strip()
        if not stripped:
            return None
        current = json.loads(stripped)
    return current


def compact_formula_label(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    compact = re.sub(r"(\\(?:sin|cos|tan|log|ln|lim|max|min))([A-Za-z])", r"\1 \2", compact)
    return compact


def hme_answer(row: dict[str, Any], compact: bool) -> str:
    for key in ("text", "answer"):
        value = clean_text(row.get(key))
        if value:
            return compact_formula_label(value) if compact else value
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and str(turn.get("role", "")).lower() == "assistant":
                value = clean_text(turn.get("content"))
                if value:
                    return compact_formula_label(value) if compact else value
    return ""


def collect_hme_samples(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], Counter]:
    from datasets import load_dataset

    samples: list[dict[str, Any]] = []
    stats = Counter()
    rng = random.Random(int(args.seed) + 1009)
    dataset = load_dataset(args.hme_dataset, split="train", streaming=True)
    for row_idx, row in enumerate(dataset):
        if len(samples) >= int(args.hme_samples):
            break
        image = row.get("image")
        answer = hme_answer(row, bool(args.compact_hme_labels))
        if image is None or not valid_short_answer(answer, max(120, int(args.max_answer_words))):
            stats["hme_rejected"] += 1
            continue
        image_path = image_output_path(output_dir, "hme100k", len(samples))
        if not save_pil_image(image, image_path, bool(args.overwrite_images)):
            stats["hme_bad_image"] += 1
            continue
        rel_image = image_path.relative_to(output_dir)
        samples.append(
            {
                "image": str(rel_image),
                "image_root": str(output_dir.resolve()),
                "question": rng.choice(FORMULA_PROMPTS),
                "answer": answer,
                "source": "hme100k",
                "source_dataset": args.hme_dataset,
                "source_index": row.get("id", row_idx),
            }
        )
        if len(samples) % 5000 == 0:
            print(f"hme100k: selected={len(samples)}/{args.hme_samples}", flush=True)
    stats["hme_selected"] = len(samples)
    return samples, stats


def parse_cord_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        outer = json_load_repeated(row.get("parsed_data"))
        if not isinstance(outer, dict):
            return None
        parsed = json_load_repeated(outer.get("json"))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def iter_field_samples(
    parsed: dict[str, Any],
    rng: random.Random,
    max_items_per_doc: int,
    max_answer_words: int,
) -> Iterable[tuple[str, str, str]]:
    for group_name in ("total", "subtotal"):
        group = parsed.get(group_name)
        if not isinstance(group, dict):
            continue
        for key, value in group.items():
            answer = clean_text(value)
            field_key = f"{group_name}.{key}"
            questions = FIELD_QUESTIONS.get(field_key)
            if questions and valid_short_answer(answer, max_answer_words):
                yield field_key, rng.choice(questions), answer

    items = parsed.get("line_items")
    if isinstance(items, dict):
        items_list = [items]
    elif isinstance(items, list):
        items_list = [item for item in items if isinstance(item, dict)]
    else:
        items_list = []
    if len(items_list) > max_items_per_doc:
        items_list = rng.sample(items_list, max_items_per_doc)

    for item_idx, item in enumerate(items_list):
        item_name = clean_text(item.get("item_name"))
        if not valid_short_answer(item_name, max_answer_words):
            continue
        item_value = clean_text(item.get("item_value"))
        if valid_short_answer(item_value, max_answer_words):
            yield "line_items.item_value", rng.choice(ITEM_VALUE_QUESTIONS).format(item_name=item_name), item_value
        item_quantity = clean_text(item.get("item_quantity"))
        if valid_short_answer(item_quantity, max_answer_words):
            yield "line_items.item_quantity", rng.choice(ITEM_QUANTITY_QUESTIONS).format(item_name=item_name), item_quantity
        if item_idx >= max_items_per_doc - 1:
            break


def collect_cord_samples(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], Counter]:
    from datasets import load_dataset

    samples: list[dict[str, Any]] = []
    stats = Counter()
    rng = random.Random(int(args.seed) + 2027)
    dataset = load_dataset(args.cord_dataset, split=args.cord_split, streaming=True)

    for row_idx, row in enumerate(dataset):
        if len(samples) >= int(args.cord_samples):
            break
        image = row.get("image")
        parsed = parse_cord_payload(row)
        if image is None or parsed is None:
            stats["cord_bad_row"] += 1
            continue
        image_path = image_output_path(output_dir, "cord", row_idx)
        if not save_pil_image(image, image_path, bool(args.overwrite_images)):
            stats["cord_bad_image"] += 1
            continue
        rel_image = image_path.relative_to(output_dir)
        emitted = 0
        for field_key, question, answer in iter_field_samples(
            parsed,
            rng,
            int(args.cord_max_items_per_doc),
            int(args.max_answer_words),
        ):
            if len(samples) >= int(args.cord_samples):
                break
            samples.append(
                {
                    "image": str(rel_image),
                    "image_root": str(output_dir.resolve()),
                    "question": question,
                    "answer": answer,
                    "source": "cord_receipt_kie",
                    "source_dataset": args.cord_dataset,
                    "source_index": row.get("id", row_idx),
                    "field": field_key,
                }
            )
            stats[f"cord_field:{field_key}"] += 1
            emitted += 1
        if emitted == 0:
            stats["cord_no_fields"] += 1
        if len(samples) and len(samples) % 5000 == 0:
            print(f"cord: selected={len(samples)}/{args.cord_samples}", flush=True)
    stats["cord_selected"] = len(samples)
    return samples, stats


def atomic_write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def answer_word_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    counts = sorted(word_count(clean_text(row.get("answer"))) for row in rows)
    if not counts:
        return {}
    return {
        "avg": sum(counts) / len(counts),
        "p50": counts[len(counts) // 2],
        "p90": counts[int(len(counts) * 0.9)],
        "max": counts[-1],
    }


def write_latest(output_dir: Path, output_path: Path) -> None:
    latest = output_dir / "latest.jsonl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(output_path.name)
    except OSError:
        shutil.copyfile(output_path, latest)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    base_rows, base_stats, base_source_targets = select_base_rows(args)
    print(f"base: selected={len(base_rows)}", flush=True)

    hme_rows, hme_stats = collect_hme_samples(args, output_dir) if int(args.hme_samples) > 0 else ([], Counter())
    cord_rows, cord_stats = collect_cord_samples(args, output_dir) if int(args.cord_samples) > 0 else ([], Counter())

    rows = list(base_rows) + hme_rows + cord_rows
    if not args.no_shuffle:
        random.Random(int(args.seed)).shuffle(rows)

    atomic_write_jsonl(rows, output_path)
    write_latest(output_dir, output_path)

    source_counts = Counter(clean_text(row.get("source")) for row in rows)
    manifest = {
        "output_jsonl": str(output_path),
        "total_samples": len(rows),
        "source_counts": dict(source_counts),
        "base_jsonl": str(Path(args.base_jsonl).expanduser()),
        "base_source_targets": base_source_targets,
        "data_policy": {
            "base_rows": "preserve existing questions and answers",
            "new_rows": "task-specific prompts only; no unified prompt",
            "excluded": "full-page OCR dumps, binary yes/no, unanswerable labels, long generic answers",
            "purpose": "cover OCRBench formula and receipt/KIE gaps",
        },
        "answer_word_stats": answer_word_stats(rows),
        "stats": {**dict(base_stats), **dict(hme_stats), **dict(cord_stats)},
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
