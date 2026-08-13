#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


OPTION_RE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_option(text: str) -> str:
    text = text.strip()
    if text[:1].upper() in {"A", "B", "C", "D"}:
        return text[:1].upper()
    match = OPTION_RE.search(text.upper())
    return match.group(1) if match else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate MMStar multiple-choice accuracy.")
    parser.add_argument("--predictions", required=True, help="JSONL with index and prediction.")
    parser.add_argument("--references", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = {int(row["index"]): row for row in read_jsonl(args.references)}
    preds = read_jsonl(args.predictions)
    total = 0
    correct = 0
    by_category = defaultdict(lambda: [0, 0])
    by_l2 = defaultdict(lambda: [0, 0])

    for pred in preds:
        index = int(pred["index"])
        ref = refs[index]
        pred_answer = extract_option(str(pred.get("prediction", "")))
        gold = str(ref["answer"]).strip().upper()
        ok = pred_answer == gold
        total += 1
        correct += int(ok)
        by_category[ref["category"]][0] += int(ok)
        by_category[ref["category"]][1] += 1
        by_l2[ref["l2_category"]][0] += int(ok)
        by_l2[ref["l2_category"]][1] += 1

    metrics = {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "by_category": {
            key: {"accuracy": val[0] / max(val[1], 1), "correct": val[0], "total": val[1]}
            for key, val in sorted(by_category.items())
        },
        "by_l2_category": {
            key: {"accuracy": val[0] / max(val[1], 1), "correct": val[0], "total": val[1]}
            for key, val in sorted(by_l2.items())
        },
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
