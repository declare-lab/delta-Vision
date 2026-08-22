"""Build rendered-text copy/transcription data.

Rows render a short text span as an image and ask the model to transcribe the
visible text exactly. The answer is the rendered span itself, capped by tokenizer
tokens so the supervision is dense but not long.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


QUESTION = "Transcribe all visible text in the image exactly. Preserve line breaks."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build rendered text copy/transcription data.")
    parser.add_argument("--source-train", default="data/train/rendered_text/paired_train.jsonl")
    parser.add_argument("--source-eval", default="data/train/rendered_text/paired_eval.jsonl")
    parser.add_argument("--output-dir", default="data/train/rendered_text_copy_300")
    parser.add_argument("--dataset-name", default="", help="Dataset id/source name. Defaults to rendered_text_copy_${max_answer_tokens}.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--max-answer-tokens", type=int, default=300)
    parser.add_argument("--min-answer-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--page-width", type=int, default=1344)
    parser.add_argument("--page-height", type=int, default=1792)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--jpeg-optimize", action="store_true")
    parser.add_argument("--preview", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--max-build-attempt-factor", type=float, default=20.0)
    return parser.parse_args()


def stable_id(*parts: Any) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def norm_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            context = norm_text(row.get("text_context"))
            if context:
                row["text_context"] = context
                rows.append(row)
    if bad:
        print(f"skipped {bad} malformed lines from {path}", flush=True)
    if not rows:
        raise RuntimeError(f"no usable rows found in {path}")
    return rows


def balanced_sample(rows: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
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
    rng.shuffle(selected)
    return selected


def token_capped_span(
    tokenizer: Any,
    text: str,
    *,
    max_tokens: int,
    min_tokens: int,
    rng: random.Random,
) -> tuple[str, int]:
    # Avoid tokenizing very long contexts in full. Sample a local character window first, then
    # use the tokenizer only for exact capping inside that window.
    max_chars = max(1600, int(max_tokens) * 8)
    if len(text) > max_chars:
        start_char = rng.randint(0, max(0, len(text) - max_chars))
        clean_start = text.find("\n", start_char, min(len(text), start_char + 300))
        if clean_start >= 0 and clean_start + 1 < len(text):
            start_char = clean_start + 1
        end_char = min(len(text), start_char + max_chars)
        clean_end = text.rfind("\n", max(start_char, end_char - 300), end_char)
        if clean_end > start_char:
            end_char = clean_end
        text = text[start_char:end_char].strip()

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"] if int(b) > int(a)]
    if not offsets:
        return "", 0
    if len(offsets) <= int(max_tokens):
        span = text[offsets[0][0] : offsets[-1][1]].strip()
        return span, len(offsets)

    window = int(max_tokens)
    max_start = max(0, len(offsets) - window)
    start_tok = rng.randint(0, max_start)
    end_tok = start_tok + window
    start_char = offsets[start_tok][0]
    end_char = offsets[end_tok - 1][1]

    # Prefer clean line boundaries inside the token window when possible.
    clean_start = text.find("\n", start_char, min(end_char, start_char + 240))
    if clean_start >= 0 and clean_start + 1 < end_char:
        start_char = clean_start + 1
    clean_end = text.rfind("\n", max(start_char, end_char - 240), end_char)
    if clean_end > start_char:
        end_char = clean_end
    span = text[start_char:end_char].strip()
    count = sum(1 for left, right in offsets[start_tok:end_tok] if right > start_char and left < end_char)
    if count < int(min_tokens) and len(offsets) >= int(min_tokens):
        start_tok = min(start_tok, max(0, len(offsets) - int(min_tokens)))
        end_tok = min(len(offsets), start_tok + int(min_tokens))
        span = text[offsets[start_tok][0] : offsets[end_tok - 1][1]].strip()
        count = end_tok - start_tok
    return span, count


def sample_text_window(text: str, *, max_tokens: int, rng: random.Random) -> str:
    max_chars = max(1600, int(max_tokens) * 8)
    if len(text) <= max_chars:
        return text.strip()
    start_char = rng.randint(0, max(0, len(text) - max_chars))
    clean_start = text.find("\n", start_char, min(len(text), start_char + 300))
    if clean_start >= 0 and clean_start + 1 < len(text):
        start_char = clean_start + 1
    end_char = min(len(text), start_char + max_chars)
    clean_end = text.rfind("\n", max(start_char, end_char - 300), end_char)
    if clean_end > start_char:
        end_char = clean_end
    return text[start_char:end_char].strip()


def render_image(
    text: str,
    out_dir: Path,
    rel: Path,
    *,
    page_width: int,
    page_height: int,
    font_size: int,
    jpeg_quality: int,
    jpeg_optimize: bool,
) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    margin = 38
    line_height = int(font_size * 1.35)
    chars_per_line = max(20, (page_width - margin * 2) // max(1, int(font_size * 0.62)))
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=chars_per_line, replace_whitespace=False) or [""])
    max_lines = max(1, (page_height - margin * 2) // line_height)
    lines = lines[:max_lines]

    image = Image.new("RGB", (page_width, page_height), (252, 251, 248))
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=(28, 28, 28), font=font)
        y += line_height
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=int(jpeg_quality), optimize=bool(jpeg_optimize))


def render_worker(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, cfg = task
    out_dir = Path(str(cfg["out_dir"]))
    rel = Path(str(row["image"]))
    render_image(
        str(row["answer"]),
        out_dir,
        rel,
        page_width=int(cfg["page_width"]),
        page_height=int(cfg["page_height"]),
        font_size=int(cfg["font_size"]),
        jpeg_quality=int(cfg["jpeg_quality"]),
        jpeg_optimize=bool(cfg["jpeg_optimize"]),
    )
    return row


def build_rows(
    tokenizer: Any,
    source_rows: list[dict[str, Any]],
    *,
    split: str,
    size: int,
    out_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    target_size = int(size)
    sampled = (
        [rng.choice(source_rows) for _ in range(target_size)]
        if bool(args.sample_with_replacement)
        else balanced_sample(source_rows, target_size, rng)
    )
    windows = [sample_text_window(str(source["text_context"]), max_tokens=int(args.max_answer_tokens), rng=rng) for source in sampled]
    batch_size = max(1, int(args.tokenize_batch_size))
    rows: list[dict[str, Any]] = []
    skipped = 0
    dataset_name = str(args.dataset_name or f"rendered_text_copy_{int(args.max_answer_tokens)}")

    def append_row(source: dict[str, Any], span: str, token_count: int) -> None:
        nonlocal skipped
        if not span or token_count < int(args.min_answer_tokens):
            skipped += 1
            return
        source_dataset = str(source.get("source_dataset") or source.get("source") or "unknown")
        source_id = str(source.get("source_id") or source.get("id") or source.get("index") or len(rows))
        stem = f"{split}_{len(rows):06d}_{stable_id(source_dataset, source_id, span)}"
        image = str(Path("pages") / f"{stem}.jpg")
        row = {
            "id": f"{dataset_name}:{split}:{len(rows):06d}",
            "index": f"{dataset_name}:{split}:{len(rows):06d}",
            "image": image,
            "images": [image],
            "image_root": str(out_dir),
            "question": QUESTION,
            "rendered_question": QUESTION,
            "answer": span,
            "answers": [span],
            "text_context": span,
            "source": dataset_name,
            "task_type": "copy_transcription",
            "source_dataset": source_dataset,
            "source_split": source.get("source_split"),
            "source_id": source_id,
            "source_row_id": source.get("id") or source.get("index"),
            "answer_tokens": int(token_count),
            "max_answer_tokens": int(args.max_answer_tokens),
            "page_width": int(args.page_width),
            "page_height": int(args.page_height),
            "font_size": int(args.font_size),
        }
        rows.append(row)

    for batch_start in range(0, len(windows), batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=int(args.max_answer_tokens),
        )
        for local_idx, (text, offsets) in enumerate(zip(batch, encoded["offset_mapping"], strict=False)):
            clean_offsets = [(int(a), int(b)) for a, b in offsets if int(b) > int(a)]
            if not clean_offsets:
                append_row(sampled[batch_start + local_idx], "", 0)
                continue
            span = text[clean_offsets[0][0] : clean_offsets[-1][1]].strip()
            append_row(sampled[batch_start + local_idx], span, len(clean_offsets))
        print(f"{split}: tokenized {min(batch_start + len(batch), len(windows))}/{len(windows)}", flush=True)

    attempts = len(sampled)
    max_attempts = max(attempts, int(target_size * float(args.max_build_attempt_factor)))
    while bool(args.sample_with_replacement) and len(rows) < target_size and attempts < max_attempts:
        need = min(batch_size, target_size - len(rows))
        candidate_sources = [rng.choice(source_rows) for _ in range(need)]
        candidate_windows = [
            sample_text_window(str(source["text_context"]), max_tokens=int(args.max_answer_tokens), rng=rng)
            for source in candidate_sources
        ]
        encoded = tokenizer(
            candidate_windows,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=int(args.max_answer_tokens),
        )
        attempts += len(candidate_windows)
        for source, text, offsets in zip(candidate_sources, candidate_windows, encoded["offset_mapping"], strict=False):
            clean_offsets = [(int(a), int(b)) for a, b in offsets if int(b) > int(a)]
            if not clean_offsets:
                append_row(source, "", 0)
                continue
            span = text[clean_offsets[0][0] : clean_offsets[-1][1]].strip()
            append_row(source, span, len(clean_offsets))
            if len(rows) >= target_size:
                break
        print(f"{split}: accepted {len(rows)}/{target_size} attempts={attempts}", flush=True)

    if len(rows) > target_size:
        rows = rows[:target_size]
    if len(rows) < target_size:
        print(f"{split}: warning accepted only {len(rows)}/{target_size} after {attempts} attempts", flush=True)
    if skipped:
        print(f"{split}: skipped {skipped} empty spans", flush=True)
    return rows


def write_split(rows: list[dict[str, Any]], out_dir: Path, split: str, args: argparse.Namespace) -> Counter:
    cfg = {
        "out_dir": str(out_dir),
        "page_width": int(args.page_width),
        "page_height": int(args.page_height),
        "font_size": int(args.font_size),
        "jpeg_quality": int(args.jpeg_quality),
        "jpeg_optimize": bool(args.jpeg_optimize),
    }
    tasks = [(row, cfg) for row in rows]
    path = out_dir / f"{split}.jsonl"
    counts: Counter = Counter()
    with path.open("w", encoding="utf-8") as handle:
        if int(args.workers) <= 1:
            iterator = map(render_worker, tasks)
        else:
            pool = Pool(processes=int(args.workers))
            iterator = pool.imap(render_worker, tasks, chunksize=16)
        try:
            for done, row in enumerate(iterator, start=1):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[str(row.get("source_dataset") or "unknown")] += 1
                if done % 1000 == 0:
                    print(f"{split}: rendered {done}/{len(rows)}", flush=True)
        finally:
            if int(args.workers) > 1:
                pool.close()
                pool.join()
    return counts


def make_preview(out_dir: Path, split: str, limit: int) -> None:
    rows: list[dict[str, Any]] = []
    with (out_dir / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= int(limit):
                break
    thumbs = []
    for row in rows:
        with Image.open(out_dir / row["image"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((260, 340))
            thumbs.append((thumb.copy(), row))
    if not thumbs:
        return
    cols = 4
    cell_w, cell_h = 300, 390
    grid = Image.new("RGB", (cols * cell_w, ((len(thumbs) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, (thumb, row) in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(thumb, (x, y))
        draw.text((x + 4, y + 344), f"{row['source_dataset']} tokens={row['answer_tokens']}", fill=(20, 20, 20))
        draw.text((x + 4, y + 362), str(row["answer"]).replace("\n", " ")[:42], fill=(20, 20, 20))
    grid.save(out_dir / f"{split}_preview_grid.jpg", quality=92)


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"loading tokenizer {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    train_source = load_jsonl(Path(args.source_train))
    eval_source = load_jsonl(Path(args.source_eval))
    train_rows = build_rows(tokenizer, train_source, split="train", size=args.train_size, out_dir=out_dir, args=args, rng=rng)
    eval_rows = build_rows(tokenizer, eval_source, split="eval", size=args.eval_size, out_dir=out_dir, args=args, rng=rng)
    train_counts = write_split(train_rows, out_dir, "paired_train", args)
    eval_counts = write_split(eval_rows, out_dir, "paired_eval", args)
    make_preview(out_dir, "paired_train", args.preview)
    make_preview(out_dir, "paired_eval", args.preview)
    manifest = {
        "name": out_dir.name,
        "task_type": "copy_transcription",
        "paired_train_jsonl": str(out_dir / "paired_train.jsonl"),
        "paired_eval_jsonl": str(out_dir / "paired_eval.jsonl"),
        "image_root": str(out_dir),
        "train_size": sum(train_counts.values()),
        "eval_size": sum(eval_counts.values()),
        "train_source_counts": dict(train_counts),
        "eval_source_counts": dict(eval_counts),
        "max_answer_tokens": int(args.max_answer_tokens),
        "min_answer_tokens": int(args.min_answer_tokens),
        "question": QUESTION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
