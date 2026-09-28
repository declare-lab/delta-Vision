from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUT = ROOT / "data/benchmarks/hotpotqa/rendered_validation_grouped"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct")
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_FONT_PATH = DEFAULT_FONT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render grouped HotpotQA validation pages.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default="hotpotqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--workers", type=int, default=224)
    parser.add_argument("--target-page-tokens", type=int, default=512)
    parser.add_argument("--soft-max-page-tokens", type=int, default=768)
    parser.add_argument("--min-page-tokens", type=int, default=256)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=320)
    parser.add_argument("--min-font-size", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-batch-size", type=int, default=2048)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--layout-version", default="hotpotqa_grouped_v3_compact")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_json(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def init_worker(font_path: str) -> None:
    global _FONT_PATH
    _FONT_PATH = font_path


def qwen_visual_tokens_estimate(width: int, height: int) -> int:
    return max(1, int(round(width / 32.0)) * int(round(height / 32.0)))


def line_height(font: ImageFont.FreeTypeFont, line_spacing: int) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1] + line_spacing)


def wrap_paragraph(paragraph: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
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


def layout_info(
    text: str,
    *,
    width: int,
    font_size: int,
    padding: int,
    line_spacing: int,
    min_height: int,
) -> dict[str, Any]:
    font = ImageFont.truetype(_FONT_PATH, font_size)
    lines = wrap_text(text, font, width - 2 * padding)
    lh = line_height(font, line_spacing)
    content_height = 2 * padding + len(lines) * lh
    height = max(min_height, int(math.ceil(content_height / 32.0) * 32))
    visual_tokens = qwen_visual_tokens_estimate(width, height)
    blank_px = max(0, height - content_height)
    return {
        "width": width,
        "height": height,
        "font_size": font_size,
        "padding": padding,
        "line_spacing": line_spacing,
        "line_count": len(lines),
        "visual_tokens": visual_tokens,
        "blank_px": blank_px,
        "lines": lines,
    }


def choose_layout(text: str, text_tokens: int, args: dict[str, Any]) -> dict[str, Any]:
    target_visual = max(1.0, text_tokens / 1.5)
    target_side = int(math.sqrt(target_visual * 1024.0))
    min_width = int(args["min_width"])
    min_height = int(args["min_height"])
    if text_tokens < int(args.get("min_page_tokens", 256)):
        min_width = 320
        min_height = 160
    widths = sorted(
        {
            max(
                min_width,
                min(2304, int(round((target_side * scale) / 64.0) * 64)),
            )
            for scale in (1.0, 1.2, 1.4, 1.7, 2.0)
        }
    )
    font_sizes = list(range(int(args["min_font_size"]), 19))
    paddings = [10, 12, 16]
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
                        min_height=min_height,
                    )
                    ratio = text_tokens / max(1, int(info["visual_tokens"]))
                    aspect = info["height"] / info["width"]
                    ratio_error = abs(ratio - 1.5)
                    ratio_penalty = min(ratio_error, 0.35) * 0.25
                    aspect_penalty = max(0.0, aspect - 1.8) * 0.5 + max(0.0, 0.55 - aspect) * 0.4
                    blank_penalty = min(1.0, info["blank_px"] / max(1, info["height"])) * 0.35
                    score = ratio_penalty + aspect_penalty + blank_penalty
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
    if task.get("reuse_metadata") is not None:
        reused = dict(task["reuse_metadata"])
        reused["skipped"] = True
        return reused
    layout = choose_layout(task["text"], int(task["text_tokens"]), task["layout_args"])
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
        "text": task["text"],
        "item_key": task["item_key"],
        "text_hash": task["text_hash"],
        "layout_version": task["layout_version"],
        "layout_config_hash": task["layout_config_hash"],
        **layout,
        "skipped": False,
    }


def merge_short_groups(groups: list[list[int]], unit_tokens: list[int], min_tokens: int, soft_max: int) -> list[list[int]]:
    if len(groups) <= 1:
        return groups
    changed = True
    while changed and len(groups) > 1:
        changed = False
        token_sums = [sum(unit_tokens[idx] for idx in group) for group in groups]
        short_indices = [idx for idx, total in enumerate(token_sums) if total < min_tokens]
        if not short_indices:
            break
        group_idx = min(short_indices, key=lambda idx: token_sums[idx])
        candidates: list[tuple[int, int, int]] = []
        if group_idx > 0:
            combined = token_sums[group_idx - 1] + token_sums[group_idx]
            overflow = max(0, combined - soft_max)
            candidates.append((overflow, combined, group_idx - 1))
        if group_idx + 1 < len(groups):
            combined = token_sums[group_idx] + token_sums[group_idx + 1]
            overflow = max(0, combined - soft_max)
            candidates.append((overflow, combined, group_idx + 1))
        if not candidates:
            break
        _, _, neighbor_idx = min(candidates, key=lambda item: (item[0], item[1]))
        if neighbor_idx < group_idx:
            groups[neighbor_idx].extend(groups[group_idx])
            del groups[group_idx]
        else:
            groups[group_idx].extend(groups[neighbor_idx])
            del groups[neighbor_idx]
        changed = True
    return groups


def split_oversized_groups(groups: list[list[int]], unit_tokens: list[int], hard_max: int, min_tokens: int) -> list[list[int]]:
    result: list[list[int]] = []
    for group in groups:
        total = sum(unit_tokens[idx] for idx in group)
        if total <= hard_max or len(group) <= 1:
            result.append(group)
            continue
        current: list[int] = []
        current_tokens = 0
        for idx in group:
            count = unit_tokens[idx]
            if current and current_tokens + count > hard_max and current_tokens >= min_tokens:
                result.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += count
        if current:
            result.append(current)
    return result


def group_units(unit_tokens: list[int], target: int, soft_max: int, min_tokens: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, count in enumerate(unit_tokens):
        if current and current_tokens + count > target:
            if current_tokens >= int(target * 0.65) or current_tokens + count > soft_max:
                groups.append(current)
                current = []
                current_tokens = 0
        current.append(idx)
        current_tokens += count
    if current:
        if groups and current_tokens < min_tokens:
            prev_tokens = sum(unit_tokens[i] for i in groups[-1])
            if prev_tokens + current_tokens <= soft_max:
                groups[-1].extend(current)
            else:
                groups.append(current)
        else:
            groups.append(current)
    groups = merge_short_groups(groups, unit_tokens, min_tokens, soft_max)
    groups = split_oversized_groups(groups, unit_tokens, hard_max=max(soft_max, target + min_tokens), min_tokens=min_tokens)
    groups = merge_short_groups(groups, unit_tokens, min_tokens, soft_max)
    return groups


def read_rows(data_path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def batched_token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def layout_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_page_tokens": int(args.target_page_tokens),
        "soft_max_page_tokens": int(args.soft_max_page_tokens),
        "min_page_tokens": int(args.min_page_tokens),
        "min_width": int(args.min_width),
        "min_height": int(args.min_height),
        "min_font_size": int(args.min_font_size),
        "font": str(args.font),
        "padding_candidates": [10, 12, 16],
        "line_spacing_candidates": [1, 2, 3],
        "short_page_min_width": 320,
        "short_page_min_height": 160,
    }


def metadata_paths(output_dir: Path, num_shards: int, shard_id: int) -> tuple[Path, Path, Path]:
    if num_shards > 1:
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard_{shard_id:05d}_of_{num_shards:05d}"
        path = shard_dir / f"{stem}.jsonl"
        tmp_path = shard_dir / f"{stem}.tmp.jsonl"
        done_path = shard_dir / f"{stem}.done"
        return path, tmp_path, done_path
    path = output_dir / "metadata.jsonl"
    tmp_path = output_dir / "metadata.tmp.jsonl"
    done_path = output_dir / "metadata.done"
    return path, tmp_path, done_path


def load_existing_metadata(output_dir: Path) -> dict[str, dict[str, Any]]:
    paths = [output_dir / "metadata.jsonl"]
    shard_dir = output_dir / "shards"
    if shard_dir.exists():
        paths.extend(sorted(shard_dir.glob("shard_*_of_*.jsonl")))
    existing: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_key = row.get("item_key")
                if item_key:
                    existing[str(item_key)] = row
    return existing


def reusable_metadata(
    task: dict[str, Any],
    existing: dict[str, dict[str, Any]],
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any] | None:
    if overwrite or not resume:
        return None
    old = existing.get(str(task["item_key"]))
    if old is None:
        return None
    if old.get("text_hash") != task["text_hash"]:
        return None
    if old.get("layout_version") != task["layout_version"]:
        return None
    if old.get("layout_config_hash") != task["layout_config_hash"]:
        return None
    image = old.get("image")
    if not image or not Path(str(image)).exists():
        return None
    return old


def build_tasks(rows: list[dict[str, Any]], unit_token_lengths: list[list[int]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    images_dir = Path(args.output_dir) / "images"
    layout_args = {
        "min_width": int(args.min_width),
        "min_height": int(args.min_height),
        "min_font_size": int(args.min_font_size),
        "min_page_tokens": int(args.min_page_tokens),
    }
    config = layout_config(args)
    config_hash = sha1_json(config)
    for row_index, (row, tokens) in enumerate(zip(rows, unit_token_lengths, strict=True)):
        units = [str(text) for text in row.get("context_texts") or []]
        groups = group_units(
            tokens,
            int(args.target_page_tokens),
            int(args.soft_max_page_tokens),
            int(args.min_page_tokens),
        )
        for page_index, group in enumerate(groups):
            text = "\n".join(units[idx] for idx in group)
            text_tokens = sum(tokens[idx] for idx in group)
            image_name = f"row_{row_index:06d}_page_{page_index:02d}.png"
            item_key = f"{args.source}:{args.split}:row={row_index}:page={page_index}:contexts={','.join(str(idx) for idx in group)}"
            titles = []
            context = row.get("context") or {}
            all_titles = context.get("title") or []
            for idx in group:
                titles.append(all_titles[idx] if idx < len(all_titles) else None)
            tasks.append(
                {
                    "text": text,
                    "text_tokens": int(text_tokens),
                    "image_path": str(images_dir / image_name),
                    "overwrite": bool(args.overwrite),
                    "layout_args": layout_args,
                    "item_key": item_key,
                    "text_hash": sha1_text(text),
                    "layout_version": str(args.layout_version),
                    "layout_config_hash": config_hash,
                    "meta": {
                        "source": str(args.source),
                        "split": str(args.split),
                        "row_index": row_index,
                        "page_index": page_index,
                        "num_pages": len(groups),
                        "context_indices": group,
                        "titles": titles,
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "supporting_facts": row.get("supporting_facts"),
                        "support_facts": row.get("support_facts"),
                    },
                }
            )
    return tasks


def value_stats(values: list[float]) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"reading rows from {args.data}", flush=True)
    rows = read_rows(Path(args.data), int(args.limit))
    all_units = [str(text) for row in rows for text in (row.get("context_texts") or [])]
    per_row_counts = [len(row.get("context_texts") or []) for row in rows]
    print(f"rows={len(rows)} units={len(all_units)}", flush=True)

    print(f"loading tokenizer {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    flat_lengths = batched_token_lengths(tokenizer, all_units, int(args.token_batch_size))
    unit_lengths: list[list[int]] = []
    offset = 0
    for count in per_row_counts:
        unit_lengths.append(flat_lengths[offset : offset + count])
        offset += count

    all_tasks = build_tasks(rows, unit_lengths, args)
    tasks = [
        task
        for task_idx, task in enumerate(all_tasks)
        if task_idx % int(args.num_shards) == int(args.shard_id)
    ]
    existing = load_existing_metadata(output_dir)
    reusable = 0
    for task in tasks:
        old = reusable_metadata(task, existing, resume=bool(args.resume), overwrite=bool(args.overwrite))
        if old is not None:
            task["reuse_metadata"] = old
            reusable += 1
    metadata_path, tmp_metadata_path, done_path = metadata_paths(output_dir, int(args.num_shards), int(args.shard_id))
    summary_path = output_dir / "summary.json"
    args_path = output_dir / "args.json"
    args_payload = {**vars(args), "layout_config": layout_config(args), "layout_config_hash": sha1_json(layout_config(args))}
    args_path.write_text(json.dumps(args_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    start = time.time()
    count = 0
    ratios: list[float] = []
    text_tokens: list[float] = []
    visual_tokens: list[float] = []
    heights: list[float] = []
    widths: list[float] = []
    with tmp_metadata_path.open("w", encoding="utf-8") as meta_out:
        with Pool(processes=int(args.workers), initializer=init_worker, initargs=(args.font,)) as pool:
            for item in pool.imap_unordered(render_task, tasks, chunksize=int(args.chunksize)):
                meta_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                ratios.append(float(item["ratio"]))
                text_tokens.append(float(item["text_tokens"]))
                visual_tokens.append(float(item["visual_tokens"]))
                heights.append(float(item["height"]))
                widths.append(float(item["width"]))
                if count % 1000 == 0:
                    elapsed = time.time() - start
                    print(f"rendered {count}/{len(tasks)} elapsed={elapsed:.1f}s rate={count/max(elapsed,1e-6):.1f}/s", flush=True)
    os.replace(tmp_metadata_path, metadata_path)
    done_path.write_text(json.dumps({"metadata": str(metadata_path), "count": count, "elapsed_s": time.time() - start}), encoding="utf-8")

    summary = {
        "data": str(args.data),
        "output_dir": str(output_dir),
        "images_dir": str(output_dir / "images"),
        "metadata": str(metadata_path),
        "rows": len(rows),
        "context_units": len(all_units),
        "all_rendered_pages": len(all_tasks),
        "selected_pages": len(tasks),
        "metadata_rows": count,
        "reused_pages": reusable,
        "workers": int(args.workers),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "layout_version": str(args.layout_version),
        "layout_config_hash": sha1_json(layout_config(args)),
        "target_page_tokens": int(args.target_page_tokens),
        "soft_max_page_tokens": int(args.soft_max_page_tokens),
        "min_page_tokens": int(args.min_page_tokens),
        "elapsed_s": time.time() - start,
        "text_tokens": value_stats(text_tokens),
        "visual_tokens_estimated": value_stats(visual_tokens),
        "ratio_estimated": value_stats(ratios),
        "width": value_stats(widths),
        "height": value_stats(heights),
        "outside_1p4_1p6": sum(1 for ratio in ratios if ratio < 1.4 or ratio > 1.6),
    }
    if int(args.num_shards) == 1:
        tmp_summary_path = output_dir / "summary.tmp.json"
        tmp_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_summary_path, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
