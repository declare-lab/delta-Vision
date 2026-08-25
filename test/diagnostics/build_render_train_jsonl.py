from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build mixed rendered QA/copy training JSONL from render metadata.")
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=[
            "data/train/render-data/qasper/metadata.jsonl",
            "data/train/render-data/hotpot/metadata.jsonl",
        ],
    )
    parser.add_argument("--output", default="data/train/render-data/render_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-min-pages", type=int, default=1)
    parser.add_argument("--copy-max-pages", type=int, default=2)
    return parser.parse_args()


def read_metadata(paths: list[str]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("source") or ""), str(row.get("split") or ""), int(row["row_index"]))
                groups[key].append(row)
    for pages in groups.values():
        pages.sort(key=lambda item: int(item.get("page_index", 0)))
    return groups


def base_id(source: str, split: str, row_index: int) -> str:
    return f"{source}:{split}:row={row_index}"


def qa_row(source: str, split: str, row_index: int, pages: list[dict[str, Any]]) -> dict[str, Any]:
    first = pages[0]
    return {
        "id": str(first.get("id") or base_id(source, split, row_index)),
        "source_dataset": source,
        "split": split,
        "task_type": "qa",
        "row_index": row_index,
        "images": [str(page["image"]) for page in pages],
        "text_context": "\n\n".join(str(page.get("text") or "") for page in pages).strip(),
        "question": str(first.get("question") or "").strip(),
        "raw_question": str(first.get("question") or "").strip(),
        "answer": str(first.get("answer") or "").strip(),
        "num_pages": len(pages),
        "page_indices": [int(page.get("page_index", idx)) for idx, page in enumerate(pages)],
        "render_item_keys": [str(page.get("item_key") or "") for page in pages],
    }


def copy_row(source: str, split: str, row_index: int, page: dict[str, Any]) -> dict[str, Any]:
    page_index = int(page.get("page_index", 0))
    text = str(page.get("text") or "").strip()
    image = str(page["image"])
    return {
        "id": f"{base_id(source, split, row_index)}:copy:page={page_index}",
        "source_dataset": source,
        "split": split,
        "task_type": "copy_transcription",
        "row_index": row_index,
        "page_index": page_index,
        "image": image,
        "images": [image],
        "text_context": text,
        "question": COPY_TRANSCRIPTION_INSTRUCTION,
        "raw_question": COPY_TRANSCRIPTION_INSTRUCTION,
        "answer": text,
        "render_item_key": str(page.get("item_key") or ""),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    groups = read_metadata(list(args.metadata))

    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = defaultdict(int)
    for source, split, row_index in sorted(groups):
        pages = groups[(source, split, row_index)]
        if not pages:
            continue
        qa = qa_row(source, split, row_index, pages)
        if qa["question"] and qa["answer"] and qa["text_context"]:
            rows.append(qa)
            stats[f"{source}_qa"] += 1

        max_pages = min(len(pages), int(args.copy_max_pages))
        min_pages = min(max_pages, int(args.copy_min_pages))
        if max_pages <= 0:
            continue
        copy_count = min_pages if max_pages == min_pages else rng.randint(min_pages, max_pages)
        for page in rng.sample(pages, copy_count):
            item = copy_row(source, split, row_index, page)
            if item["answer"]:
                rows.append(item)
                stats[f"{source}_copy"] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)

    summary = {
        "output": str(output),
        "rows": len(rows),
        "seed": int(args.seed),
        "copy_min_pages": int(args.copy_min_pages),
        "copy_max_pages": int(args.copy_max_pages),
        "stats": dict(sorted(stats.items())),
        "metadata": list(args.metadata),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
