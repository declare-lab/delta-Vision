#!/usr/bin/env python3
"""Download HF benchmark datasets and convert them to the project JSONL schema."""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

from src.benchmarks import parse_benchmark_names


@dataclass(frozen=True)
class HFBenchmarkSource:
    name: str
    repo: str
    config: str | None
    split: str
    output_rel: str
    kind: str = "generic"
    data_files: str | None = None
    image_config: str | None = None
    instruction_config: str | None = None
    image_data_files: str | None = None
    instruction_data_files: str | None = None


DEFAULT_SOURCES: dict[str, HFBenchmarkSource] = {
    "gqa": HFBenchmarkSource(
        name="gqa",
        repo="lmms-lab-encoder/GQA",
        config=None,
        split="testdev",
        output_rel="gqa/testdev_balanced.jsonl",
        kind="gqa_pair",
        data_files=None,
        image_config="testdev_balanced_images",
        instruction_config="testdev_balanced_instructions",
        image_data_files="testdev_balanced_images/testdev-*.parquet",
        instruction_data_files="testdev_balanced_instructions/testdev-*.parquet",
    ),
    "mmb": HFBenchmarkSource("mmb", "lmms-lab/MMBench_EN", "default", "dev", "mmb/dev.jsonl", data_files="data/dev-*.parquet"),
    "mmb-cn": HFBenchmarkSource("mmb-cn", "lmms-lab/MMBench_CN", "default", "dev", "mmb-cn/dev.jsonl", data_files="data/dev-*.parquet"),
    "mme": HFBenchmarkSource("mme", "lmms-lab-encoder/MME", "default", "test", "mme/test.jsonl", data_files="data/test-*.parquet"),
    "pope": HFBenchmarkSource("pope", "lmms-lab-encoder/POPE", "default", "test", "pope/test.jsonl", data_files="data/test-*.parquet"),
    "sqa": HFBenchmarkSource(
        "sqa",
        "lmms-lab-encoder/ScienceQA",
        "ScienceQA-IMG",
        "test",
        "sqa/test.jsonl",
        data_files="ScienceQA-IMG/test-*.parquet",
    ),
    "vqav2": HFBenchmarkSource(
        "vqav2",
        "lmms-lab-encoder/VQAv2",
        "default",
        "validation",
        "vqav2/validation.jsonl",
        data_files="data/validation-*.parquet",
    ),
    "textvqa": HFBenchmarkSource(
        "textvqa",
        "lmms-lab-encoder/textvqa",
        "default",
        "validation",
        "textvqa/validation.jsonl",
        data_files="data/validation-*.parquet",
    ),
    "vizwiz": HFBenchmarkSource(
        "vizwiz",
        "lmms-lab-encoder/VizWiz-VQA",
        "default",
        "val",
        "vizwiz/val.jsonl",
        data_files="data/val-*.parquet",
    ),
    "ocrbench": HFBenchmarkSource("ocrbench", "echo840/OCRBench", "default", "test", "ocrbench/test.jsonl", data_files="data/test-*.parquet"),
}


def parse_benchmarks(value: str) -> list[str]:
    if not value.strip() or value.strip().lower() == "all":
        return list(DEFAULT_SOURCES)
    names = parse_benchmark_names(value)
    missing = [name for name in names if name not in DEFAULT_SOURCES]
    if missing:
        raise ValueError(
            f"no HF download source for {missing}; downloadable benchmarks are {sorted(DEFAULT_SOURCES)}"
        )
    for name in names:
        if name not in DEFAULT_SOURCES:
            raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(DEFAULT_SOURCES)}")
    return names


def load_hf_dataset(repo: str, config: str | None, split: str, data_files: str | None = None):
    if data_files:
        data_files_arg = {split: f"hf://datasets/{repo}/{data_files}"}
        return load_dataset("parquet", split=split, data_files=data_files_arg)
    if config:
        return load_dataset(repo, config, split=split)
    return load_dataset(repo, split=split)


def clean_filename(value: Any, fallback: str) -> str:
    raw = str(value if value is not None else fallback)
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return raw or fallback


def image_extension(image: Image.Image) -> str:
    fmt = (getattr(image, "format", None) or "").lower()
    if fmt in {"jpeg", "jpg"}:
        return ".jpg"
    if fmt in {"png", "webp"}:
        return f".{fmt}"
    return ".jpg"


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image
    fmt = (getattr(img, "format", None) or "").upper()
    if path.suffix.lower() == ".jpg":
        fmt = "JPEG"
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
    elif path.suffix.lower() == ".png":
        fmt = "PNG"
    elif path.suffix.lower() == ".webp":
        fmt = "WEBP"
    img.save(path, format=fmt or "JPEG")


def relative_image_path(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def get_first(example: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = example.get(key)
        if not is_blank(value):
            return value
    return None


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def collect_choices(example: dict[str, Any]) -> list[Any]:
    choices = example.get("choices")
    if isinstance(choices, list):
        return [choice for choice in choices if not is_blank(choice)]
    if isinstance(choices, dict):
        return [choices[key] for key in sorted(choices) if not is_blank(choices[key])]
    result = []
    for letter in string.ascii_uppercase[:8]:
        value = example.get(letter)
        if not is_blank(value):
            result.append(value)
    return result


def collect_answers(example: dict[str, Any]) -> tuple[Any, Any]:
    answer = get_first(example, ["answer", "multiple_choice_answer", "label", "gt_answer", "gt"])
    answers = example.get("answers")
    if answer is None and isinstance(answers, list) and answers:
        first = answers[0]
        answer = first.get("answer") if isinstance(first, dict) else first
    return answer, answers


def standardize_example(
    *,
    benchmark: str,
    example: dict[str, Any],
    image_rel: str,
    index: int,
    source: HFBenchmarkSource,
) -> dict[str, Any] | None:
    question = get_first(example, ["question", "query", "prompt", "text"])
    answer, answers = collect_answers(example)
    choices = collect_choices(example)
    if question is None or (is_blank(answer) and answers is None):
        return None
    row: dict[str, Any] = {
        "index": get_first(example, ["index", "question_id", "id"]) or index,
        "benchmark": benchmark,
        "source_repo": source.repo,
        "source_config": source.config,
        "source_split": source.split,
        "image": image_rel,
        "question": str(question).strip(),
        "answer": answer,
    }
    if answers is not None:
        row["answers"] = answers
    if choices:
        row["choices"] = choices
    for key in ("hint", "category", "question_type", "image_id", "imageId", "dataset", "source"):
        if key in example and example[key] is not None:
            out_key = "image_id" if key == "imageId" else key
            row[out_key] = example[key]
    return row


def convert_generic(source: HFBenchmarkSource, data_root: Path, output_root: Path, max_samples: int | None) -> dict[str, Any]:
    dataset = load_hf_dataset(source.repo, source.config, source.split, source.data_files)
    out_path = output_root / source.output_rel
    image_dir = out_path.parent / "images"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(dataset):
            if max_samples is not None and written >= max_samples:
                break
            image = example.get("image")
            if image is None:
                skipped += 1
                continue
            if not isinstance(image, Image.Image):
                skipped += 1
                continue
            raw_id = get_first(example, ["image_id", "imageId", "question_id", "index", "id"]) or idx
            image_name = clean_filename(raw_id, f"{idx:08d}") + image_extension(image)
            image_path = image_dir / image_name
            if not image_path.exists():
                save_image(image, image_path)
            row = standardize_example(
                benchmark=source.name,
                example=example,
                image_rel=relative_image_path(image_path, data_root),
                index=idx,
                source=source,
            )
            if row is None:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return {"benchmark": source.name, "path": str(out_path), "samples": written, "skipped": skipped}


def convert_gqa_pair(source: HFBenchmarkSource, data_root: Path, output_root: Path, max_samples: int | None) -> dict[str, Any]:
    if source.image_config is None or source.instruction_config is None:
        raise ValueError("GQA paired source requires image_config and instruction_config")
    images = load_hf_dataset(source.repo, source.image_config, source.split, source.image_data_files)
    instructions = load_hf_dataset(source.repo, source.instruction_config, source.split, source.instruction_data_files)
    out_path = output_root / source.output_rel
    image_dir = out_path.parent / "images"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_by_id = {str(item["id"]): item["image"] for item in images}
    image_path_by_id: dict[str, Path] = {}
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(instructions):
            if max_samples is not None and written >= max_samples:
                break
            image_id = str(example.get("imageId", ""))
            image = image_by_id.get(image_id)
            if image is None:
                skipped += 1
                continue
            if image_id not in image_path_by_id:
                image_path = image_dir / f"{clean_filename(image_id, f'image_{idx:08d}')}{image_extension(image)}"
                save_image(image, image_path)
                image_path_by_id[image_id] = image_path
            row = standardize_example(
                benchmark=source.name,
                example=example,
                image_rel=relative_image_path(image_path_by_id[image_id], data_root),
                index=idx,
                source=source,
            )
            if row is None:
                skipped += 1
                continue
            row["source_config"] = source.instruction_config
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return {"benchmark": source.name, "path": str(out_path), "samples": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="all", help="Comma/space-separated names, or all.")
    parser.add_argument("--data-root", default="/lustre-data/leijingdi/code/delta-vision")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else data_root / "data" / "benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name in parse_benchmarks(args.benchmarks):
        source = DEFAULT_SOURCES[name]
        out_path = output_root / source.output_rel
        if out_path.exists() and not args.overwrite:
            samples = sum(1 for line in out_path.open(encoding="utf-8") if line.strip())
            result = {"benchmark": name, "path": str(out_path), "samples": samples, "skipped": 0, "status": "exists"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            manifest.append(result)
            continue

        print(f"Downloading {name}: {source.repo} config={source.config or source.instruction_config} split={source.split}", flush=True)
        if source.kind == "gqa_pair":
            result = convert_gqa_pair(source, data_root, output_root, args.max_samples)
        else:
            result = convert_generic(source, data_root, output_root, args.max_samples)
        result["status"] = "downloaded"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        manifest.append(result)

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
