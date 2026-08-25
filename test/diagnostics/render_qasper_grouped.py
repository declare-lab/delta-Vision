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
DEFAULT_DATA = ROOT / "data/train/qasper-data/agent_memory_qasper_ctx8192_episode_safe_seed42.sectioned.jsonl"
DEFAULT_OUT = ROOT / "data/train/render-data/qasper"
DEFAULT_MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_FONT_PATH = DEFAULT_FONT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render Qasper sectioned user context blocks into grouped pages.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--workers", type=int, default=224)
    parser.add_argument("--target-page-tokens", type=int, default=1024)
    parser.add_argument("--soft-max-page-tokens", type=int, default=1536)
    parser.add_argument("--min-page-tokens", type=int, default=512)
    parser.add_argument("--min-width", type=int, default=768)
    parser.add_argument("--min-height", type=int, default=384)
    parser.add_argument("--min-font-size", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--layout-version", default="qasper_grouped_v1_compact")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_worker(font_path: str) -> None:
    global _FONT_PATH
    _FONT_PATH = font_path


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_json(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


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


def layout_info(text: str, *, width: int, font_size: int, padding: int, line_spacing: int, min_height: int) -> dict[str, Any]:
    font = ImageFont.truetype(_FONT_PATH, font_size)
    lines = wrap_text(text, font, width - 2 * padding)
    lh = line_height(font, line_spacing)
    content_height = 2 * padding + len(lines) * lh
    height = max(min_height, int(math.ceil(content_height / 32.0) * 32))
    visual_tokens = qwen_visual_tokens_estimate(width, height)
    return {
        "width": width,
        "height": height,
        "font_size": font_size,
        "padding": padding,
        "line_spacing": line_spacing,
        "line_count": len(lines),
        "visual_tokens": visual_tokens,
        "blank_px": max(0, height - content_height),
        "lines": lines,
    }


def choose_layout(text: str, text_tokens: int, args: dict[str, Any]) -> dict[str, Any]:
    target_visual = max(1.0, text_tokens / 1.5)
    target_side = int(math.sqrt(target_visual * 1024.0))
    min_width = int(args["min_width"])
    min_height = int(args["min_height"])
    if text_tokens < int(args.get("min_page_tokens", 512)):
        min_width = 480
        min_height = 240
    widths = sorted(
        {
            max(min_width, min(2560, int(round((target_side * scale) / 64.0) * 64)))
            for scale in (0.9, 1.05, 1.2, 1.4, 1.7)
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
                    info = layout_info(text, width=width, font_size=font_size, padding=padding, line_spacing=line_spacing, min_height=min_height)
                    ratio = text_tokens / max(1, int(info["visual_tokens"]))
                    aspect = info["height"] / info["width"]
                    ratio_penalty = min(abs(ratio - 1.5), 0.35) * 0.25
                    aspect_penalty = max(0.0, aspect - 1.8) * 0.5 + max(0.0, 0.55 - aspect) * 0.4
                    blank_penalty = min(1.0, info["blank_px"] / max(1, info["height"])) * 0.35
                    score = ratio_penalty + aspect_penalty + blank_penalty
                    if score < best_score:
                        best_score = score
                        best = {**info, "text_tokens": text_tokens, "ratio": ratio, "target_visual_tokens": target_visual, "aspect": aspect, "score": score}
    if best is None:
        raise RuntimeError("no layout candidates")
    return best


def merge_short_groups(groups: list[list[int]], token_counts: list[int], min_tokens: int, soft_max: int) -> list[list[int]]:
    while len(groups) > 1:
        totals = [sum(token_counts[idx] for idx in group) for group in groups]
        shorts = [idx for idx, total in enumerate(totals) if total < min_tokens]
        if not shorts:
            break
        group_idx = min(shorts, key=lambda idx: totals[idx])
        candidates: list[tuple[int, int, int]] = []
        if group_idx > 0:
            combined = totals[group_idx - 1] + totals[group_idx]
            candidates.append((max(0, combined - soft_max), combined, group_idx - 1))
        if group_idx + 1 < len(groups):
            combined = totals[group_idx] + totals[group_idx + 1]
            candidates.append((max(0, combined - soft_max), combined, group_idx + 1))
        if not candidates:
            break
        _, _, neighbor_idx = min(candidates, key=lambda item: (item[0], item[1]))
        if neighbor_idx < group_idx:
            groups[neighbor_idx].extend(groups[group_idx])
            del groups[group_idx]
        else:
            groups[group_idx].extend(groups[neighbor_idx])
            del groups[neighbor_idx]
    return groups


def group_units(token_counts: list[int], target: int, soft_max: int, min_tokens: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, count in enumerate(token_counts):
        if current and current_tokens + count > target:
            if current_tokens >= int(target * 0.65) or current_tokens + count > soft_max:
                groups.append(current)
                current = []
                current_tokens = 0
        current.append(idx)
        current_tokens += count
    if current:
        groups.append(current)
    return merge_short_groups(groups, token_counts, min_tokens, soft_max)


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def qasper_context(row: dict[str, Any]) -> tuple[list[str], str, str]:
    users = [str(msg.get("content", "")) for msg in row.get("messages", []) if msg.get("role") == "user"]
    question = ""
    if users and users[-1].startswith("Question:"):
        question = users[-1]
        users = users[:-1]
    answer = ""
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            answer = str(msg.get("content", ""))
            break
    return users, question, answer


def batched_token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], add_special_tokens=False, return_attention_mask=False)
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
        "short_page_min_width": 480,
        "short_page_min_height": 240,
    }


def metadata_paths(output_dir: Path, num_shards: int, shard_id: int) -> tuple[Path, Path, Path]:
    if num_shards > 1:
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard_{shard_id:05d}_of_{num_shards:05d}"
        return shard_dir / f"{stem}.jsonl", shard_dir / f"{stem}.tmp.jsonl", shard_dir / f"{stem}.done"
    return output_dir / "metadata.jsonl", output_dir / "metadata.tmp.jsonl", output_dir / "metadata.done"


def load_existing_metadata(output_dir: Path) -> dict[str, dict[str, Any]]:
    paths = [output_dir / "metadata.jsonl"]
    shard_dir = output_dir / "shards"
    if shard_dir.exists():
        paths.extend(sorted(shard_dir.glob("shard_*_of_*.jsonl")))
    existing: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("item_key"):
                existing[str(row["item_key"])] = row
    return existing


def render_task(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("reuse_metadata") is not None:
        reused = dict(task["reuse_metadata"])
        reused["skipped"] = True
        return reused
    image_path = Path(task["image_path"])
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
    return {**task["meta"], "image": str(image_path), "text": task["text"], "item_key": task["item_key"], "text_hash": task["text_hash"], "layout_version": task["layout_version"], "layout_config_hash": task["layout_config_hash"], **layout, "skipped": False}


def reusable(task: dict[str, Any], existing: dict[str, dict[str, Any]], resume: bool, overwrite: bool) -> dict[str, Any] | None:
    if overwrite or not resume:
        return None
    old = existing.get(str(task["item_key"]))
    if old is None:
        return None
    if old.get("text_hash") != task["text_hash"] or old.get("layout_version") != task["layout_version"] or old.get("layout_config_hash") != task["layout_config_hash"]:
        return None
    if not old.get("image") or not Path(str(old["image"])).exists():
        return None
    return old


def build_tasks(rows: list[dict[str, Any]], token_lengths: list[list[int]], args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    config_hash = sha1_json(layout_config(args))
    layout_args = {"min_width": args.min_width, "min_height": args.min_height, "min_font_size": args.min_font_size, "min_page_tokens": args.min_page_tokens}
    tasks: list[dict[str, Any]] = []
    for row_index, (row, counts) in enumerate(zip(rows, token_lengths, strict=True)):
        blocks, question, answer = qasper_context(row)
        groups = group_units(counts, args.target_page_tokens, args.soft_max_page_tokens, args.min_page_tokens)
        for page_index, group in enumerate(groups):
            text = "\n".join(blocks[idx] for idx in group)
            item_key = f"qasper:train:row={row_index}:page={page_index}:blocks={','.join(str(idx) for idx in group)}"
            tasks.append(
                {
                    "text": text,
                    "text_tokens": int(sum(counts[idx] for idx in group)),
                    "image_path": str(images_dir / f"row_{row_index:06d}_page_{page_index:02d}.png"),
                    "layout_args": layout_args,
                    "item_key": item_key,
                    "text_hash": sha1_text(text),
                    "layout_version": args.layout_version,
                    "layout_config_hash": config_hash,
                    "meta": {
                        "source": "qasper",
                        "split": "train",
                        "row_index": row_index,
                        "page_index": page_index,
                        "num_pages": len(groups),
                        "block_indices": group,
                        "block_token_counts": [counts[idx] for idx in group],
                        "paper_id": row.get("paper_id"),
                        "question_id": row.get("question_id"),
                        "question": question,
                        "answer": answer,
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
    return {"min": values[0], "p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99), "max": values[-1], "mean": sum(values) / len(values)}


def main() -> None:
    args = parse_args()
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"reading rows from {args.data}", flush=True)
    rows = read_rows(Path(args.data), args.limit)
    contexts = [qasper_context(row)[0] for row in rows]
    flat_texts = [block for blocks in contexts for block in blocks]
    per_row_counts = [len(blocks) for blocks in contexts]
    print(f"rows={len(rows)} blocks={len(flat_texts)}", flush=True)

    print(f"loading tokenizer {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    flat_lengths = batched_token_lengths(tokenizer, flat_texts, args.token_batch_size)
    token_lengths: list[list[int]] = []
    offset = 0
    for count in per_row_counts:
        token_lengths.append(flat_lengths[offset : offset + count])
        offset += count

    all_tasks = build_tasks(rows, token_lengths, args)
    tasks = [task for idx, task in enumerate(all_tasks) if idx % args.num_shards == args.shard_id]
    existing = load_existing_metadata(output_dir)
    reused = 0
    for task in tasks:
        old = reusable(task, existing, args.resume, args.overwrite)
        if old is not None:
            task["reuse_metadata"] = old
            reused += 1

    metadata_path, tmp_metadata_path, done_path = metadata_paths(output_dir, args.num_shards, args.shard_id)
    args_payload = {**vars(args), "layout_config": layout_config(args), "layout_config_hash": sha1_json(layout_config(args))}
    (output_dir / "args.json").write_text(json.dumps(args_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    start = time.time()
    values: dict[str, list[float]] = {"ratio": [], "text": [], "visual": [], "width": [], "height": []}
    count = 0
    with tmp_metadata_path.open("w", encoding="utf-8") as meta_out:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(args.font,)) as pool:
            for item in pool.imap_unordered(render_task, tasks, chunksize=args.chunksize):
                meta_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                values["ratio"].append(float(item["ratio"]))
                values["text"].append(float(item["text_tokens"]))
                values["visual"].append(float(item["visual_tokens"]))
                values["width"].append(float(item["width"]))
                values["height"].append(float(item["height"]))
                if count % 500 == 0:
                    elapsed = time.time() - start
                    print(f"rendered {count}/{len(tasks)} elapsed={elapsed:.1f}s rate={count/max(elapsed,1e-6):.1f}/s", flush=True)
    os.replace(tmp_metadata_path, metadata_path)
    done_path.write_text(json.dumps({"metadata": str(metadata_path), "count": count, "elapsed_s": time.time() - start}), encoding="utf-8")

    summary = {
        "data": str(args.data),
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "rows": len(rows),
        "blocks": len(flat_texts),
        "all_pages": len(all_tasks),
        "selected_pages": len(tasks),
        "metadata_rows": count,
        "reused_pages": reused,
        "workers": args.workers,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "layout_version": args.layout_version,
        "layout_config_hash": sha1_json(layout_config(args)),
        "elapsed_s": time.time() - start,
        "text_tokens": value_stats(values["text"]),
        "visual_tokens_estimated": value_stats(values["visual"]),
        "ratio_estimated": value_stats(values["ratio"]),
        "width": value_stats(values["width"]),
        "height": value_stats(values["height"]),
        "outside_1p4_1p6": sum(1 for ratio in values["ratio"] if ratio < 1.4 or ratio > 1.6),
    }
    if args.num_shards == 1:
        tmp_summary = output_dir / "summary.tmp.json"
        tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
