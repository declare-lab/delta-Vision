"""Prepare a 300K Pixmo + OCR-heavy training mix for Qwen.

Output schema:
  {"image": str, "image_root": str, "question": str, "answer": str, "source": str}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "pixmo_clean_ocrmix_300k"
DEFAULT_PIXMO_JSONL = (
    "/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/"
    "pixmo_ama_full_valid.clean.jsonl"
)
DEFAULT_PIXMO_ROOT = "/lustre-data/leijingdi/code/delta-vision"
FINEVISION_REPO = "HuggingFaceM4/FineVision"


BUCKET_WEIGHTS = {
    "pixmo_clean": 0.45,
    "docvqa": 0.15,
    "text_scene": 0.15,
    "info_chart": 0.15,
    "receipt_form": 0.05,
    "ocrvqa": 0.05,
}

SUBSOURCE_WEIGHTS = {
    "docvqa": {
        "docvqa": 0.50,
        "pdfvqa": 0.25,
        "ureader_qa_processed": 0.25,
    },
    "text_scene": {
        "textvqa": 0.55,
        "st_vqa": 0.45,
    },
    "info_chart": {
        "infographic_vqa": 0.25,
        "chartqa": 0.45,
        "plotqa": 0.30,
    },
    "receipt_form": {
        "sroie": 0.60,
        "invoices_receipts": 0.25,
        "funsd": 0.15,
    },
    "ocrvqa": {
        "ocrvqa": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build Pixmo-clean + OCR mix JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--total-samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pixmo-jsonl", default=DEFAULT_PIXMO_JSONL)
    parser.add_argument("--pixmo-root", default=DEFAULT_PIXMO_ROOT)
    parser.add_argument("--hf-repo", default=FINEVISION_REPO)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def clean_text(value: object) -> str:
    return str(value or "").strip()


def split_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {name: float(weight) * int(total) for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = int(total) - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def select_or_repeat(samples: list[dict], target: int, seed: int) -> tuple[list[dict], int]:
    if target <= 0:
        return [], 0
    rng = random.Random(seed)
    if len(samples) >= target:
        return rng.sample(samples, target), 0
    if not samples:
        return [], target
    selected = list(samples)
    shortfall = target - len(samples)
    selected.extend(rng.choice(samples) for _ in range(shortfall))
    rng.shuffle(selected)
    return selected, shortfall


def load_pixmo_samples(path: Path, pixmo_root: Path, target: int, seed: int) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question = clean_text(row.get("question"))
            answer = clean_text(row.get("answer"))
            if not question or not answer:
                stats["pixmo_empty_text"] += 1
                continue
            image = str(row.get("image"))
            image_root = str(row.get("image_root") or pixmo_root)
            image_path = Path(image) if Path(image).is_absolute() else Path(image_root) / image
            if not image_path.exists():
                stats["pixmo_missing_image"] += 1
                continue
            rows.append(
                {
                    "image": image,
                    "image_root": image_root,
                    "question": question,
                    "answer": answer,
                    "source": "pixmo_clean",
                }
            )
    selected, shortfall = select_or_repeat(rows, target, seed)
    stats["pixmo_available"] = len(rows)
    stats["pixmo_shortfall_repeated"] = shortfall
    stats["pixmo_selected"] = len(selected)
    return selected, stats


def image_extension(image_bytes: bytes, fallback_path: str | None = None) -> str:
    if image_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if fallback_path:
        suffix = Path(fallback_path).suffix.lower()
        if suffix:
            return suffix
    return ".jpg"


def write_image_bytes(image_record: dict, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return True
    image_bytes = image_record.get("bytes")
    if not image_bytes:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        handle.write(image_bytes)
    tmp_path.replace(output_path)
    return True


def list_config_shards(api: HfApi, repo: str, config: str, seed: int) -> list[str]:
    files = [
        name
        for name in api.list_repo_files(repo_id=repo, repo_type="dataset")
        if name.startswith(f"{config}/") and name.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"No parquet shards found for {repo}/{config}")
    random.Random(seed).shuffle(files)
    return files


def collect_finevision_config(
    *,
    repo: str,
    config: str,
    target: int,
    output_dir: Path,
    seed: int,
    overwrite_images: bool,
    api: HfApi,
) -> tuple[list[dict], Counter]:
    samples: list[dict] = []
    stats = Counter()
    shards = list_config_shards(api, repo, config, seed)

    for shard_num, filename in enumerate(shards, start=1):
        local_path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename))
        table = pq.read_table(local_path, columns=["texts", "images"])
        row_order = list(range(len(table)))
        random.Random(seed + shard_num).shuffle(row_order)

        for local_row_idx in row_order:
            if len(samples) >= target:
                break
            texts = table["texts"][local_row_idx].as_py() or []
            images = table["images"][local_row_idx].as_py() or []
            if len(images) != 1:
                stats[f"{config}_non_single_image"] += 1
                continue

            global_image_id = f"{Path(filename).stem}_{local_row_idx:06d}"
            rel_image, image_root = save_finevision_image(
                config,
                global_image_id,
                images[0],
                output_dir,
                overwrite_images,
            )
            if not rel_image:
                stats[f"{config}_bad_image"] += 1
                continue

            for turn_idx, turn in enumerate(texts):
                if len(samples) >= target:
                    break
                question = clean_text(turn.get("user") if isinstance(turn, dict) else "")
                answer = clean_text(turn.get("assistant") if isinstance(turn, dict) else "")
                if not question or not answer:
                    stats[f"{config}_empty_text"] += 1
                    continue
                samples.append(
                    {
                        "image": rel_image,
                        "image_root": image_root,
                        "question": question,
                        "answer": answer,
                        "source": config,
                        "conversation_turn": turn_idx,
                    }
                )

        print(
            f"{config}: shard {shard_num}/{len(shards)} selected={len(samples)}/{target}",
            flush=True,
        )
        del table
        if len(samples) >= target:
            break

    selected, shortfall = select_or_repeat(samples, target, seed)
    stats[f"{config}_selected"] = len(selected)
    stats[f"{config}_unique_before_repeat"] = len(samples)
    stats[f"{config}_shortfall_repeated"] = shortfall
    return selected, stats


def save_finevision_image(
    config: str,
    image_id: str,
    image_record: dict,
    output_dir: Path,
    overwrite: bool,
) -> tuple[str, str]:
    image_bytes = image_record.get("bytes")
    if not image_bytes:
        return "", ""
    suffix = image_extension(image_bytes, image_record.get("path"))
    rel_path = Path("images") / config / f"{image_id}{suffix}"
    abs_path = output_dir / rel_path
    if not write_image_bytes(image_record, abs_path, overwrite):
        return "", ""
    return str(rel_path), str(output_dir)


def atomic_write_jsonl(samples: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    bucket_counts = split_counts(args.total_samples, BUCKET_WEIGHTS)
    api = HfApi()
    all_samples: list[dict] = []
    stats = Counter()
    source_targets: dict[str, int] = {}

    print("Targets:", json.dumps(bucket_counts, indent=2), flush=True)
    pixmo, pixmo_stats = load_pixmo_samples(
        Path(args.pixmo_jsonl).expanduser(),
        Path(args.pixmo_root).expanduser(),
        bucket_counts["pixmo_clean"],
        args.seed,
    )
    all_samples.extend(pixmo)
    stats.update(pixmo_stats)
    source_targets["pixmo_clean"] = bucket_counts["pixmo_clean"]
    print(f"pixmo_clean: selected={len(pixmo)}/{bucket_counts['pixmo_clean']}", flush=True)

    for bucket, configs in SUBSOURCE_WEIGHTS.items():
        config_counts = split_counts(bucket_counts[bucket], configs)
        print(f"{bucket} targets: {json.dumps(config_counts, ensure_ascii=False)}", flush=True)
        for config, target in config_counts.items():
            if target <= 0:
                continue
            samples, cfg_stats = collect_finevision_config(
                repo=args.hf_repo,
                config=config,
                target=target,
                output_dir=output_dir,
                seed=args.seed + len(source_targets) * 1009,
                overwrite_images=args.overwrite_images,
                api=api,
            )
            all_samples.extend(samples)
            stats.update(cfg_stats)
            source_targets[config] = target
            print(f"{config}: selected={len(samples)}/{target}", flush=True)

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(all_samples)

    atomic_write_jsonl(all_samples, output_path)
    latest = output_dir / "latest.jsonl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(output_path.name)
    except OSError:
        shutil.copyfile(output_path, latest)

    source_counts = Counter(sample["source"] for sample in all_samples)
    manifest = {
        "output_jsonl": str(output_path),
        "total_samples": len(all_samples),
        "bucket_targets": bucket_counts,
        "source_targets": source_targets,
        "source_counts": dict(source_counts),
        "stats": dict(stats),
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
