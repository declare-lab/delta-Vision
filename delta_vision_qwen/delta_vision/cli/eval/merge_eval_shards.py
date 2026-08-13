#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_mmstar(shards: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(s.get("num_samples", 0)) for s in shards)
    correct = sum(int(s.get("correct", 0)) for s in shards)
    agree = sum(int(s.get("agree", 0)) for s in shards)
    teacher_correct = sum(int(s.get("teacher_correct", 0)) for s in shards)
    teacher_correct_agree = sum(int(s.get("teacher_correct_and_agree", 0)) for s in shards)
    output_kl_sum = sum(float(s.get("output_kl_sum", float(s.get("output_kl", 0.0)) * int(s.get("num_samples", 0)))) for s in shards)

    hidden_sums: dict[str, float] = {}
    hidden_counts: dict[str, int] = {}
    for shard in shards:
        n = int(shard.get("num_samples", 0))
        for layer, value in shard.get("mean_hidden_mse", {}).items():
            hidden_sums[layer] = hidden_sums.get(layer, 0.0) + float(value) * n
            hidden_counts[layer] = hidden_counts.get(layer, 0) + n

    active_layers = shards[0].get("active_layers", []) if shards else []
    return {
        "num_samples": total,
        "correct": correct,
        "agree": agree,
        "accuracy": correct / max(total, 1),
        "teacher_agreement": agree / max(total, 1),
        "teacher_correct_and_agree": teacher_correct_agree,
        "teacher_correct_retention": teacher_correct_agree / max(teacher_correct, 1),
        "output_kl_sum": output_kl_sum,
        "output_kl": output_kl_sum / max(total, 1),
        "teacher_correct": teacher_correct,
        "active_layers": active_layers,
        "mean_hidden_mse": {
            layer: hidden_sums[layer] / max(hidden_counts[layer], 1)
            for layer in sorted(hidden_sums, key=lambda x: int(x) if x.isdigit() else x)
        },
        "merged_shards": len(shards),
    }


def merge_generation(shards: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(s.get("num_samples", 0)) for s in shards)
    correct = sum(int(s.get("correct", 0)) for s in shards)
    choice_scored = sum(int(s.get("choice_logit_scored", 0)) for s in shards)
    generated_scored = sum(int(s.get("generation_scored", 0)) for s in shards)
    by_type: dict[str, list[int]] = {}
    for shard in shards:
        for key, value in shard.get("by_question_type", {}).items():
            by_type.setdefault(key, [0, 0])
            by_type[key][0] += int(value.get("correct", 0))
            by_type[key][1] += int(value.get("total", 0))
    first = shards[0] if shards else {}
    return {
        "benchmark": first.get("benchmark"),
        "model_kind": first.get("model_kind"),
        "num_samples": total,
        "choice_logit_scored": choice_scored,
        "generation_scored": generated_scored,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "by_question_type": {
            key: {"correct": val[0], "total": val[1], "accuracy": val[0] / max(val[1], 1)}
            for key, val in sorted(by_type.items())
        },
        "merged_shards": len(shards),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Merge delta-vision evaluation shard JSON files.")
    parser.add_argument("--kind", choices=("mmstar", "generation"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()

    shards = [_load(path) for path in args.shards]
    merged = merge_mmstar(shards) if args.kind == "mmstar" else merge_generation(shards)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
