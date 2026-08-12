"""Prepare Pixmo-clean + FineVision LLaVA-150K mixed VQA data.

Output JSONL schema matches the training datasets:
  {"image": str, "question": str, "answer": str, "source": str}

Pixmo images are written as absolute paths resolved against --pixmo-data-root.
FineVision images are exported under --output-dir and referenced relatively, so
training can use --data-root equal to --output-dir.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError


DEFAULT_PIXMO_JSONL = (
    "/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/"
    "pixmo_ama_full_valid.clean.jsonl"
)
DEFAULT_PIXMO_ROOT = "/lustre-data/leijingdi/code/delta-vision"
DEFAULT_FINEVISION_LLAVA_DIR = (
    "~/.cache/huggingface/hub/datasets--HuggingFaceM4--FineVision/snapshots/"
    "3c380a731a3429c1d04693d6ec16d7e683def84c/LLaVA_Instruct_150K"
)
DEFAULT_OUTPUT_DIR = (
    "/lustre-data/leijingdi/code/delta-vision/data/"
    "pixmo_clean_finevision_llava150k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Mix Pixmo clean JSONL with FineVision/LLaVA-Instruct-150K parquet shards."
    )
    parser.add_argument("--pixmo-jsonl", default=DEFAULT_PIXMO_JSONL)
    parser.add_argument("--pixmo-data-root", default=DEFAULT_PIXMO_ROOT)
    parser.add_argument("--finevision-llava-dir", default=DEFAULT_FINEVISION_LLAVA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixmo", type=int, default=None)
    parser.add_argument("--max-llava", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--conversation-mode",
        choices=["first_turn", "all_turns"],
        default="first_turn",
        help="first_turn keeps one sample per LLaVA conversation; all_turns emits every valid QA turn.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--reencode-images",
        action="store_true",
        help="Re-encode FineVision images as RGB JPEG. By default raw image bytes are written unchanged.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def clean_text(value) -> str:
    return str(value or "").strip()


def resolved_pixmo_image(row: dict, data_root: Path) -> str:
    image = Path(str(row["image"]))
    if not image.is_absolute():
        image = data_root / image
    return str(image)


def load_pixmo_samples(path: Path, data_root: Path, max_samples: int | None) -> tuple[list[dict], Counter]:
    samples: list[dict] = []
    stats = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            question = clean_text(row.get("question"))
            answer = clean_text(row.get("answer"))
            image = resolved_pixmo_image(row, data_root)
            if not question or not answer:
                stats["pixmo_empty_text"] += 1
                continue
            if not Path(image).exists():
                stats["pixmo_missing_image"] += 1
                continue
            samples.append(
                {
                    "image": image,
                    "question": question,
                    "answer": answer,
                    "source": "pixmo_clean",
                }
            )
    stats["pixmo_kept"] = len(samples)
    return samples, stats


def iter_llava_turns(texts: list[dict], mode: str):
    if mode == "first_turn":
        iterable = texts[:1]
    else:
        iterable = texts
    for turn_idx, turn in enumerate(iterable):
        question = clean_text(turn.get("user"))
        answer = clean_text(turn.get("assistant"))
        if question and answer:
            yield turn_idx, question, answer


def save_llava_image(
    image_record: dict,
    output_path: Path,
    jpeg_quality: int,
    overwrite: bool,
    reencode: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return True

    image_bytes = image_record.get("bytes")
    if not image_bytes:
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        if reencode:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                img.save(tmp_path, "JPEG", quality=jpeg_quality)
        else:
            with tmp_path.open("wb") as f:
                f.write(image_bytes)
        tmp_path.replace(output_path)
    except (OSError, UnidentifiedImageError):
        return False
    return True


def load_llava_samples(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict], Counter]:
    parquet_dir = Path(args.finevision_llava_dir).expanduser()
    shards = sorted(parquet_dir.glob("*.parquet"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {parquet_dir}")

    samples: list[dict] = []
    stats = Counter({"llava_shards": len(shards)})
    global_row_idx = 0

    for shard_idx, shard_path in enumerate(shards):
        table = pq.read_table(shard_path, columns=["texts", "images"])
        for row_idx in range(len(table)):
            if args.max_llava is not None and len(samples) >= args.max_llava:
                break

            texts = table["texts"][row_idx].as_py() or []
            images = table["images"][row_idx].as_py() or []
            if len(images) != 1:
                stats["llava_non_single_image"] += 1
                global_row_idx += 1
                continue

            rel_image = Path("images") / "llava_instruct_150k" / f"{global_row_idx:08d}.jpg"
            abs_image = output_dir / rel_image
            if not save_llava_image(
                images[0],
                abs_image,
                args.jpeg_quality,
                args.overwrite_images,
                args.reencode_images,
            ):
                stats["llava_bad_image"] += 1
                global_row_idx += 1
                continue

            kept_turns = 0
            for turn_idx, question, answer in iter_llava_turns(texts, args.conversation_mode):
                if args.max_llava is not None and len(samples) >= args.max_llava:
                    break
                samples.append(
                    {
                        "image": str(rel_image),
                        "question": question,
                        "answer": answer,
                        "source": "finevision_llava_150k",
                        "conversation_turn": turn_idx,
                    }
                )
                kept_turns += 1
            if kept_turns == 0:
                stats["llava_empty_text"] += 1

            global_row_idx += 1

        print(
            f"Processed shard {shard_idx + 1}/{len(shards)}: "
            f"llava_samples={len(samples)}",
            flush=True,
        )
        del table
        if args.max_llava is not None and len(samples) >= args.max_llava:
            break

    stats["llava_kept"] = len(samples)
    return samples, stats


def atomic_write_jsonl(samples: list[dict], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def write_manifest(output_dir: Path, output_path: Path, args: argparse.Namespace, stats: Counter, samples: list[dict]) -> None:
    manifest = {
        "output_jsonl": str(output_path),
        "num_samples": len(samples),
        "source_counts": Counter(sample["source"] for sample in samples),
        "stats": dict(stats),
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    print("Loading Pixmo clean...", flush=True)
    pixmo_samples, pixmo_stats = load_pixmo_samples(
        Path(args.pixmo_jsonl).expanduser(),
        Path(args.pixmo_data_root).expanduser(),
        args.max_pixmo,
    )
    print(f"  Pixmo kept: {len(pixmo_samples)}", flush=True)

    print("Loading FineVision LLaVA-150K...", flush=True)
    llava_samples, llava_stats = load_llava_samples(args, output_dir)
    print(f"  LLaVA kept: {len(llava_samples)}", flush=True)

    samples = pixmo_samples + llava_samples
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(samples)

    print(f"Writing {len(samples)} samples to {output_path}", flush=True)
    atomic_write_jsonl(samples, output_path)

    stats = pixmo_stats + llava_stats
    write_manifest(output_dir, output_path, args, stats, samples)

    # Keep a stable alias for shell scripts and quick inspection.
    latest = output_dir / "latest.jsonl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(output_path.name)
    except OSError:
        shutil.copyfile(output_path, latest)

    print(json.dumps({
        "output": str(output_path),
        "samples": len(samples),
        "source_counts": dict(Counter(sample["source"] for sample in samples)),
        "stats": dict(stats),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
