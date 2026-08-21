from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch.distributed as dist
import yaml

from epic_qwen.eval import get_adapter
from epic_qwen.eval.runner import QwenEPICRunner, choice_correct, normalize_answer


def json_safe_meta(meta):
    keep = (
        "index", "image_name", "question", "category", "l2_category", "bench",
        "source", "split", "task", "grade", "subject", "topic", "skill",
    )
    return {key: meta[key] for key in keep if key in meta and isinstance(meta[key], (str, int, float, bool))}


def append_jsonl(path: Path, row: dict):
    """Append and fsync every completed sample for crash-safe live results."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def clean_mme_field(value):
    return str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def append_mme_txt(root: Path, rank: int, row: dict):
    rank_root = root / f"rank{rank}"
    rank_root.mkdir(parents=True, exist_ok=True)
    fields = [row["meta"]["image_name"], row["meta"]["question"], row["answer"], row["response"]]
    with (rank_root / f"{row['category']}.txt").open("a", encoding="utf-8") as handle:
        handle.write("\t".join(clean_mme_field(value) for value in fields) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path):
    if not path.exists():
        return [], set()
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records, {row["id"] for row in records}


def write_json_atomic(path: Path, value: dict):
    """Replace a live metrics snapshot atomically so readers never see partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_mme_merged(output_root: Path, name: str, records: list[dict]):
    official_root = output_root / f"{name}_mme_results"
    official_root.mkdir(parents=True, exist_ok=True)
    by_category = {}
    for row in records:
        by_category.setdefault(row["category"], []).append(row)
    for category, rows in by_category.items():
        rows.sort(key=lambda row: int(row["id"].rsplit(":", 1)[-1]))
        with (official_root / f"{category}.txt").open("w", encoding="utf-8") as handle:
            for row in rows:
                fields = [row["meta"]["image_name"], row["meta"]["question"],
                          row["answer"], row["response"]]
                handle.write("\t".join(clean_mme_field(value) for value in fields) + "\n")
    return official_root


def write_vizwiz_submission(output_root: Path, name: str, records: list[dict]):
    submission = []
    for row in sorted(records, key=lambda item: item["id"]):
        image_name = row["id"] if Path(row["id"]).suffix else f"{row['id']}.jpg"
        submission.append({"image": image_name, "answer": clean_mme_field(row["response"])})
    path = output_root / f"{name}_submission.json"
    write_json_atomic(path, submission)
    return path


def evaluate_dataset(runner, spec, output_root: Path, rank: int, world: int):
    adapter = get_adapter(spec["type"], data=spec["data"], split=spec.get("split", "test"),
                          image_root=spec.get("image_root"))
    name = spec.get("name", spec["type"])
    shard = output_root / f"{name}.rank{rank}.jsonl"
    resume = bool(spec.get("resume", True))
    if not resume and shard.exists():
        shard.unlink()
    records, completed_ids = load_completed(shard) if resume else ([], set())
    shard.touch(exist_ok=True)
    live_metrics = output_root / f"{name}.rank{rank}.metrics.live.json"
    mme_live_root = output_root / f"{name}_mme_live"
    if spec["type"] == "mme" and not resume:
        rank_root = mme_live_root / f"rank{rank}"
        if rank_root.exists():
            for path in rank_root.glob("*.txt"):
                path.unlink()
    limit = spec.get("limit")
    for index, sample in enumerate(adapter):
        if limit is not None and index >= int(limit):
            break
        if index % world != rank:
            continue
        if sample.uid in completed_ids:
            continue
        if sample.choices:
            prediction, scores = runner.score_choices(sample.question, sample.images, sample.choices)
            correct = choice_correct(prediction, sample.answer, sample.choices)
            choice_index = ord(prediction.strip().upper()[:1]) - ord("A")
            response = sample.choices[choice_index] if 0 <= choice_index < len(sample.choices) else prediction
        else:
            prediction = runner.generate(sample.question, sample.images,
                                         int(spec.get("max_new_tokens", 64)))
            scores = None
            response = prediction
            if hasattr(adapter, "score_prediction"):
                correct = adapter.score_prediction(prediction, sample)
            else:
                correct = (normalize_answer(prediction) == normalize_answer(sample.answer)
                           if sample.answer is not None else None)
        row = {
            "id": sample.uid, "prediction": prediction, "answer": sample.answer,
            "response": response,
            "correct": correct, "category": sample.category, "choices": sample.choices,
            "scores": scores, "meta": json_safe_meta(sample.meta),
            "keep_ratio": runner.keep_ratio, "pruning_layer": runner.pruning_layer,
        }
        records.append(row)
        completed_ids.add(sample.uid)
        append_jsonl(shard, row)
        if spec["type"] == "mme":
            append_mme_txt(mme_live_root, rank, row)
        write_json_atomic(live_metrics, {
            "dataset": name, "rank": rank, "world_size": world,
            "completed_on_rank": len(records), "metrics": adapter.summarize(records),
        })
        print(f"[{name}] rank={rank} saved={len(records)} id={sample.uid} pred={response}", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for shard_rank in range(world):
            with (output_root / f"{name}.rank{shard_rank}.jsonl").open(encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle)
        merged.sort(key=lambda row: row["id"])
        with (output_root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = adapter.summarize(merged)
        if spec["type"] == "mme":
            summary["official_results_dir"] = str(write_mme_merged(output_root, name, merged))
        elif spec["type"] == "vizwiz":
            summary["submission_file"] = str(write_vizwiz_submission(output_root, name, merged))
        with (output_root / f"{name}.metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps({"dataset": name, **summary}, ensure_ascii=False), flush=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL + EPIC LCD")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    output_root = Path(cfg.get("output_dir", "eval_outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    runner = QwenEPICRunner(**cfg["model"])
    for spec in cfg["datasets"]:
        evaluate_dataset(runner, spec, output_root, rank, world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
