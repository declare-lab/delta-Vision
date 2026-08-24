"""Render HotpotQA fullwiki context fields into page images for QA CE training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build rendered HotpotQA fullwiki QA data.")
    parser.add_argument(
        "--train",
        nargs="+",
        default=[
            "data/train/rendered_text/rendered_text_sources/hotpotqa__hotpot_qa/fullwiki/train-00000-of-00002.parquet",
            "data/train/rendered_text/rendered_text_sources/hotpotqa__hotpot_qa/fullwiki/train-00001-of-00002.parquet",
        ],
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        default=["data/train/rendered_text/rendered_text_sources/hotpotqa__hotpot_qa/fullwiki/validation-00000-of-00001.parquet"],
    )
    parser.add_argument("--output-dir", default="data/train/hotpotqa_fullwiki_rendered")
    parser.add_argument("--dataset-name", default="hotpotqa_fullwiki_rendered")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--page-width", type=int, default=1344)
    parser.add_argument("--page-height", type=int, default=1792)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--margin", type=int, default=38)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--jpeg-optimize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
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


def context_to_text(context: dict[str, Any]) -> str:
    titles = list(context.get("title") or [])
    sentence_groups = list(context.get("sentences") or [])
    blocks: list[str] = []
    for title, sentences in zip(titles, sentence_groups):
        title_text = norm_text(title)
        sentence_text = norm_text(" ".join(str(sentence) for sentence in (sentences or [])))
        if title_text and sentence_text:
            blocks.append(f"{title_text}\n{sentence_text}")
        elif title_text:
            blocks.append(title_text)
        elif sentence_text:
            blocks.append(sentence_text)
    return "\n\n".join(blocks).strip()


def wrap_lines(text: str, *, page_width: int, font_size: int, margin: int) -> list[str]:
    chars_per_line = max(20, (int(page_width) - int(margin) * 2) // max(1, int(font_size * 0.62)))
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=chars_per_line, replace_whitespace=False) or [""])
    return lines


def render_page(
    lines: list[str],
    path: Path,
    *,
    page_width: int,
    page_height: int,
    font_size: int,
    margin: int,
    jpeg_quality: int,
    jpeg_optimize: bool,
) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", int(font_size))
    line_height = int(font_size * 1.35)
    image = Image.new("RGB", (int(page_width), int(page_height)), (252, 251, 248))
    draw = ImageDraw.Draw(image)
    y = int(margin)
    for line in lines:
        draw.text((int(margin), y), line, fill=(28, 28, 28), font=font)
        y += line_height
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=int(jpeg_quality), optimize=bool(jpeg_optimize))


def build_task(row: dict[str, Any], split: str, index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    context_text = context_to_text(row.get("context") or {})
    question = norm_text(row.get("question"))
    answer = norm_text(row.get("answer"))
    source_id = str(row.get("id") or "")
    if not context_text or not question or not answer:
        return None
    row_id = f"{args.dataset_name}:{split}:{index:06d}:{stable_id(source_id, question, answer)}"
    rel_prefix = f"pages/{split}_{index:06d}_{stable_id(source_id, context_text[:512])}"
    return {
        "id": row_id,
        "index": row_id,
        "source_id": source_id,
        "source_dataset": "hotpotqa_fullwiki",
        "source_split": split,
        "source": args.dataset_name,
        "task_type": "qa",
        "question": question,
        "raw_question": question,
        "rendered_question": question,
        "answer": answer,
        "answers": [answer],
        "text_context": context_text,
        "hotpot_type": row.get("type"),
        "hotpot_level": row.get("level"),
        "supporting_facts": row.get("supporting_facts"),
        "_rel_prefix": rel_prefix,
    }


def render_worker(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, cfg = payload
    out_dir = Path(str(cfg["out_dir"]))
    lines = wrap_lines(
        str(row["text_context"]),
        page_width=int(cfg["page_width"]),
        font_size=int(cfg["font_size"]),
        margin=int(cfg["margin"]),
    )
    line_height = int(int(cfg["font_size"]) * 1.35)
    max_lines = max(1, (int(cfg["page_height"]) - int(cfg["margin"]) * 2) // line_height)
    pages = [lines[idx : idx + max_lines] for idx in range(0, len(lines), max_lines)] or [[""]]
    rel_paths = [f"{row['_rel_prefix']}_{page_idx:02d}.jpg" for page_idx in range(len(pages))]
    if bool(cfg["overwrite"]) or not all((out_dir / rel).exists() for rel in rel_paths):
        for rel, page_lines in zip(rel_paths, pages):
            render_page(
                page_lines,
                out_dir / rel,
                page_width=int(cfg["page_width"]),
                page_height=int(cfg["page_height"]),
                font_size=int(cfg["font_size"]),
                margin=int(cfg["margin"]),
                jpeg_quality=int(cfg["jpeg_quality"]),
                jpeg_optimize=bool(cfg["jpeg_optimize"]),
            )
    row = dict(row)
    row.pop("_rel_prefix", None)
    row["image"] = rel_paths[0]
    row["images"] = rel_paths
    row["image_root"] = str(out_dir)
    row["num_pages"] = len(rel_paths)
    row["page_width"] = int(cfg["page_width"])
    row["page_height"] = int(cfg["page_height"])
    row["font_size"] = int(cfg["font_size"])
    return row


def iter_rows(paths: list[str], split: str, args: argparse.Namespace):
    index = 0
    max_rows = int(args.max_train if split == "train" else args.max_eval)
    columns = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    for raw_path in paths:
        parquet = pq.ParquetFile(raw_path)
        for batch in parquet.iter_batches(batch_size=int(args.batch_size), columns=columns):
            for raw in batch.to_pylist():
                if max_rows > 0 and index >= max_rows:
                    return
                row = build_task(raw, split, index, args)
                index += 1
                if row is not None:
                    yield row


def build_split(paths: list[str], split: str, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    jsonl_path = out_dir / f"paired_{split}.jsonl"
    cfg = {
        "out_dir": str(out_dir),
        "page_width": int(args.page_width),
        "page_height": int(args.page_height),
        "font_size": int(args.font_size),
        "margin": int(args.margin),
        "jpeg_quality": int(args.jpeg_quality),
        "jpeg_optimize": bool(args.jpeg_optimize),
        "overwrite": bool(args.overwrite),
    }
    count = 0
    page_count = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        tasks = ((row, cfg) for row in iter_rows(paths, split, args))
        if int(args.workers) <= 1:
            iterator = map(render_worker, tasks)
            pool = None
        else:
            pool = Pool(processes=int(args.workers))
            iterator = pool.imap(render_worker, tasks, chunksize=8)
        try:
            for rendered in iterator:
                handle.write(json.dumps(rendered, ensure_ascii=False) + "\n")
                count += 1
                page_count += int(rendered.get("num_pages") or 0)
                if count == 1 or count % 1000 == 0:
                    print(f"{split}: rendered {count} rows pages={page_count}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return {"rows": count, "pages": page_count, "jsonl": str(jsonl_path)}


def main() -> None:
    args = parse_args()
    train = build_split([str(path) for path in args.train], "train", args)
    eval_stats = build_split([str(path) for path in args.eval], "eval", args)
    metadata = {
        "dataset_name": args.dataset_name,
        "train_sources": [str(path) for path in args.train],
        "eval_sources": [str(path) for path in args.eval],
        "train": train,
        "eval": eval_stats,
        "render": {
            "workers": int(args.workers),
            "page_width": int(args.page_width),
            "page_height": int(args.page_height),
            "font_size": int(args.font_size),
            "margin": int(args.margin),
        },
    }
    Path(args.output_dir, "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
