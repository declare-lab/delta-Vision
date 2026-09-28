from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/benchmarks/hotpotqa/validation.jsonl"
DEFAULT_OUT = ROOT / "data/benchmarks/hotpotqa/rendered_validation"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct")
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_FONT_PATH = DEFAULT_FONT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render HotpotQA validation context_texts into compact Pillow images.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-batch-size", type=int, default=2048)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_worker(font_path: str) -> None:
    global _FONT_PATH
    _FONT_PATH = font_path


def qwen_visual_tokens_estimate(width: int, height: int) -> int:
    # Matches measured Qwen3-VL image-token behavior for still images:
    # 512x512 -> 16*16=256, 1024x1024 -> 32*32=1024.
    return max(1, int(round(width / 32.0)) * int(round(height / 32.0)))


def line_height(font: ImageFont.FreeTypeFont, line_spacing: int) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1] + line_spacing)


def wrap_paragraph(paragraph: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not paragraph:
        return []
    words = paragraph.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
            continue
        if font.getlength(word) <= max_width:
            lines.append(current)
            current = word
            continue
        if current:
            lines.append(current)
            current = ""
        chunk = ""
        for ch in word:
            candidate = chunk + ch
            if font.getlength(candidate) <= max_width or not chunk:
                chunk = candidate
            else:
                lines.append(chunk)
                chunk = ch
        current = chunk
    if current:
        lines.append(current)
    return lines


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines.extend(wrap_paragraph(paragraph, font, max_width))
    return lines or [""]


def layout_info(text: str, *, width: int, font_size: int, padding: int, line_spacing: int) -> dict[str, Any]:
    font = ImageFont.truetype(_FONT_PATH, font_size)
    lines = wrap_text(text, font, width - 2 * padding)
    lh = line_height(font, line_spacing)
    height = 2 * padding + len(lines) * lh
    visual_tokens = qwen_visual_tokens_estimate(width, height)
    return {
        "width": width,
        "height": height,
        "font_size": font_size,
        "padding": padding,
        "line_spacing": line_spacing,
        "line_count": len(lines),
        "visual_tokens": visual_tokens,
        "lines": lines,
    }


def choose_layout(text: str, text_tokens: int) -> dict[str, Any]:
    target_visual = max(1.0, text_tokens / 1.5)
    target_side = int(math.sqrt(target_visual * 1024.0))
    widths = sorted(
        {
            max(256, min(2304, int(round((target_side * scale) / 64.0) * 64)))
            for scale in (0.85, 1.0, 1.15, 1.35, 1.6)
        }
    )
    font_sizes = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    paddings = [16, 20, 24, 28]
    line_spacings = [1, 2, 3]
    best: dict[str, Any] | None = None
    best_score = float("inf")
    for width in widths:
        for font_size in font_sizes:
            for padding in paddings:
                for line_spacing in line_spacings:
                    info = layout_info(
                        text,
                        width=width,
                        font_size=font_size,
                        padding=padding,
                        line_spacing=line_spacing,
                    )
                    ratio = text_tokens / max(1, int(info["visual_tokens"]))
                    aspect = info["height"] / info["width"]
                    ratio_penalty = abs(ratio - 1.5)
                    aspect_penalty = max(0.0, aspect - 1.8) * 0.25 + max(0.0, 0.45 - aspect) * 0.25
                    score = ratio_penalty + aspect_penalty
                    if score < best_score:
                        best_score = score
                        best = {
                            **info,
                            "text_tokens": text_tokens,
                            "ratio": ratio,
                            "target_visual_tokens": target_visual,
                            "aspect": aspect,
                            "score": score,
                        }
    if best is None:
        raise RuntimeError("no layout candidates")
    return best


def render_task(task: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(task["image_path"])
    if image_path.exists() and not task["overwrite"]:
        return {**task["meta"], "image": str(image_path), "skipped": True}

    text = task["text"]
    layout = choose_layout(text, int(task["text_tokens"]))
    font = ImageFont.truetype(_FONT_PATH, int(layout["font_size"]))
    image = Image.new("RGB", (int(layout["width"]), int(layout["height"])), "white")
    draw = ImageDraw.Draw(image)
    y = int(layout["padding"])
    lh = line_height(font, int(layout["line_spacing"]))
    for line in layout.pop("lines"):
        draw.text((int(layout["padding"]), y), line, fill=(10, 10, 10), font=font)
        y += lh
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return {
        **task["meta"],
        "image": str(image_path),
        "text": text,
        **layout,
        "skipped": False,
    }


def iter_units(data_path: Path, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    metas: list[dict[str, Any]] = []
    texts: list[str] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit and row_index >= limit:
                break
            row = json.loads(line)
            for unit_index, text in enumerate(row.get("context_texts") or []):
                metas.append(
                    {
                        "source": "hotpotqa",
                        "split": "validation",
                        "row_index": row_index,
                        "unit_index": unit_index,
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "title": (row.get("context") or {}).get("title", [None])[unit_index],
                        "support_facts": row.get("support_facts"),
                    }
                )
                texts.append(str(text))
    return metas, texts


def batched_token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
        if start and start % (batch_size * 25) == 0:
            print(f"tokenized {start}/{len(texts)}", flush=True)
    return lengths


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading units from {data_path}", flush=True)
    metas, texts = iter_units(data_path, int(args.limit))
    print(f"units={len(texts)} rows_limit={args.limit or 'all'}", flush=True)

    print(f"loading tokenizer {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    token_lengths = batched_token_lengths(tokenizer, texts, int(args.token_batch_size))

    tasks: list[dict[str, Any]] = []
    for meta, text, text_tokens in zip(metas, texts, token_lengths, strict=True):
        image_name = f"row_{meta['row_index']:06d}_ctx_{meta['unit_index']:02d}.png"
        tasks.append(
            {
                "text": text,
                "text_tokens": int(text_tokens),
                "image_path": str(images_dir / image_name),
                "overwrite": bool(args.overwrite),
                "meta": meta,
            }
        )

    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "summary.json"
    start = time.time()
    count = 0
    ratios: list[float] = []
    text_token_values: list[int] = []
    visual_token_values: list[int] = []
    with metadata_path.open("w", encoding="utf-8") as meta_out:
        with Pool(processes=int(args.workers), initializer=init_worker, initargs=(args.font,)) as pool:
            for item in pool.imap_unordered(render_task, tasks, chunksize=int(args.chunksize)):
                meta_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                ratios.append(float(item["ratio"]))
                text_token_values.append(int(item["text_tokens"]))
                visual_token_values.append(int(item["visual_tokens"]))
                if count % 1000 == 0:
                    elapsed = time.time() - start
                    print(f"rendered {count}/{len(tasks)} elapsed={elapsed:.1f}s rate={count/max(elapsed,1e-6):.1f}/s", flush=True)

    def stats(values: list[float]) -> dict[str, float]:
        values = sorted(values)
        if not values:
            return {}
        def pct(p: int) -> float:
            return values[min(len(values) - 1, round((p / 100.0) * (len(values) - 1)))]
        return {
            "min": values[0],
            "p50": pct(50),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    summary = {
        "data": str(data_path),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "metadata": str(metadata_path),
        "workers": int(args.workers),
        "count": count,
        "elapsed_s": time.time() - start,
        "text_tokens": stats([float(x) for x in text_token_values]),
        "visual_tokens_estimated": stats([float(x) for x in visual_token_values]),
        "ratio_estimated": stats(ratios),
        "outside_1p4_1p6": sum(1 for ratio in ratios if ratio < 1.4 or ratio > 1.6),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
