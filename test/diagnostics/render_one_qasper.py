from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/train/qasper-data/agent_memory_qasper_ctx8192_episode_safe_seed42.sectioned.jsonl"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct")
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render one Qasper context image with compact layout search.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--groups", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(ROOT / "test/results/rendered_text/qasper_one"))
    parser.add_argument("--font", default=DEFAULT_FONT)
    return parser.parse_args()


def read_row(path: Path, row_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index out of range: {row_index}")


def user_context_blocks(row: dict[str, Any]) -> tuple[list[str], str]:
    users = [str(msg.get("content", "")) for msg in row["messages"] if msg.get("role") == "user"]
    question = ""
    if users and users[-1].startswith("Question:"):
        question = users[-1]
        users = users[:-1]
    return users, question


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def balanced_groups(blocks: list[str], token_counts: list[int], n_groups: int) -> list[list[int]]:
    total = sum(token_counts)
    target = total / max(1, n_groups)
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    remaining_groups = n_groups

    for idx, count in enumerate(token_counts):
        remaining_items = len(blocks) - idx
        should_cut = (
            current
            and remaining_groups > 1
            and current_tokens + count > target
            and abs(current_tokens - target) <= abs(current_tokens + count - target)
            and remaining_items >= remaining_groups
        )
        if should_cut:
            groups.append(current)
            current = []
            current_tokens = 0
            remaining_groups -= 1
        current.append(idx)
        current_tokens += count
    if current:
        groups.append(current)
    while len(groups) < n_groups:
        groups.append([])
    return groups


def visual_tokens(processor: Any, image: Image.Image) -> int:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "x"}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    if "mm_token_type_ids" in inputs:
        return int((inputs["mm_token_type_ids"] == 1).sum().item())
    grid = inputs["image_grid_thw"][0].tolist()
    return int(math.prod(grid) // 4)


def line_height(font: ImageFont.FreeTypeFont, line_spacing: int) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1] + line_spacing)


def wrap_paragraph(paragraph: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not paragraph:
        return [""]
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


def render_text(text: str, *, width: int, font_size: int, padding: int, line_spacing: int, font_path: str) -> tuple[Image.Image, dict[str, int]]:
    font = ImageFont.truetype(font_path, font_size)
    max_width = width - 2 * padding
    lines = wrap_text(text, font, max_width)
    lh = line_height(font, line_spacing)
    height = max(width // 3, 2 * padding + len(lines) * lh)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(10, 10, 10), font=font)
        y += lh
    return image, {"line_count": len(lines), "height": height}


def choose_render(processor: Any, text: str, text_tokens: int, font_path: str) -> tuple[Image.Image, dict[str, Any]]:
    target_visual = max(1.0, text_tokens / 1.5)
    target_side = int(math.sqrt(target_visual * 1024))
    widths = sorted({max(512, min(2560, int(round((target_side * scale) / 64) * 64))) for scale in (0.9, 1.05, 1.2, 1.4, 1.6)})
    font_sizes = [11, 12, 13, 14, 15, 16, 17, 18]
    paddings = [18, 22, 26]
    line_spacings = [1, 2, 3]

    best: tuple[float, Image.Image, dict[str, Any]] | None = None
    for width in widths:
        for font_size in font_sizes:
            for padding in paddings:
                for line_spacing in line_spacings:
                    image, info = render_text(
                        text,
                        width=width,
                        font_size=font_size,
                        padding=padding,
                        line_spacing=line_spacing,
                        font_path=font_path,
                    )
                    vt = visual_tokens(processor, image)
                    ratio = text_tokens / max(1, vt)
                    aspect = image.height / image.width
                    ratio_penalty = abs(ratio - 1.5)
                    aspect_penalty = max(0.0, aspect - 1.8) * 0.25 + max(0.0, 0.65 - aspect) * 0.25
                    score = ratio_penalty + aspect_penalty
                    meta = {
                        **info,
                        "width": image.width,
                        "font_size": font_size,
                        "padding": padding,
                        "line_spacing": line_spacing,
                        "text_tokens": text_tokens,
                        "visual_tokens": vt,
                        "ratio": ratio,
                        "target_visual_tokens": target_visual,
                        "aspect": aspect,
                        "score": score,
                    }
                    if best is None or score < best[0]:
                        best = (score, image, meta)
    if best is None:
        raise RuntimeError("no render candidates produced")
    return best[1], best[2]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)

    row = read_row(Path(args.data), args.row_index)
    blocks, question = user_context_blocks(row)
    token_counts = [token_len(tokenizer, block) for block in blocks]
    groups = balanced_groups(blocks, token_counts, args.groups)
    selected = groups[args.group_index]
    text = "\n\n".join(blocks[idx] for idx in selected)
    text_tokens = token_len(tokenizer, text)
    image, meta = choose_render(processor, text, text_tokens, args.font)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"qasper_row{args.row_index:06d}_group{args.group_index:02d}_of{args.groups}.png"
    meta_path = image_path.with_suffix(".json")
    image.save(image_path)
    meta.update(
        {
            "image": str(image_path),
            "row_index": args.row_index,
            "groups": args.groups,
            "group_index": args.group_index,
            "block_indices": selected,
            "block_token_counts": [token_counts[idx] for idx in selected],
            "question": question,
            "paper_id": row.get("paper_id"),
            "question_id": row.get("question_id"),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
