#!/usr/bin/env python3
"""Compare Qwen3-VL sidecar repro outputs with the 20260807_100141 reference run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REFERENCE_TRAIN = {
    5: {"loss": 5.250278, "trajectory": 0.070994, "logit_kl": 1.303695},
    100: {"loss": 1.774879, "trajectory": 0.033078, "logit_kl": 0.439585},
    250: {"loss": 2.146816, "trajectory": 0.043469, "logit_kl": 0.531270},
    500: {"loss": 1.955866, "trajectory": 0.048810, "logit_kl": 0.482865},
}

REFERENCE_EVAL = {
    "qwen": {"scored": 1000, "correct": 653, "accuracy": 0.653},
    "no_visual": {
        "scored": 1000,
        "correct": 252,
        "accuracy": 0.252,
        "qwen_agreement": 0.338,
        "qwen_correct_retention": 0.3108728943338438,
        "output_kl_to_qwen": 2.7673553557973625,
    },
    "sidecar_only": {
        "scored": 1000,
        "correct": 549,
        "accuracy": 0.549,
        "qwen_agreement": 0.695,
        "qwen_correct_retention": 0.7366003062787136,
        "output_kl_to_qwen": 1.0019933164109807,
    },
    "hybrid": {
        "scored": 1000,
        "correct": 650,
        "accuracy": 0.65,
        "qwen_agreement": 0.838,
        "qwen_correct_retention": 0.8958652373660031,
        "output_kl_to_qwen": 0.44958348669239784,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metrics", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing train_metrics.jsonl.")
    parser.add_argument("--loss-atol", type=float, default=0.15)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--accuracy-atol", type=float, default=0.002)
    return parser.parse_args()


def resolve_train_metrics(args: argparse.Namespace) -> Path | None:
    if args.train_metrics is not None:
        return args.train_metrics
    if args.run_dir is None:
        return None
    candidates = [
        args.run_dir / "train_metrics.jsonl",
        args.run_dir / "checkpoints" / "train_metrics.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_train_metrics(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            rows[step] = row
    return rows


def compare_float(name: str, got: float, ref: float, atol: float) -> bool:
    ok = math.isfinite(got) and abs(got - ref) <= atol
    status = "OK" if ok else "DIFF"
    print(f"{status} {name}: got={got:.6f} ref={ref:.6f} diff={got - ref:+.6f}")
    return ok


def compare_train(path: Path, loss_atol: float) -> bool:
    print(f"train_metrics={path}")
    rows = load_train_metrics(path)
    ok = True
    for step, expected in REFERENCE_TRAIN.items():
        if step not in rows:
            print(f"DIFF train step {step}: missing")
            ok = False
            continue
        for key, ref_value in expected.items():
            ok = compare_float(f"train/{key}@{step}", float(rows[step][key]), ref_value, loss_atol) and ok
    return ok


def compare_eval(path: Path, metric_atol: float, accuracy_atol: float) -> bool:
    print(f"eval_json={path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    got_by_setting = {row["setting"]: row for row in payload["results"]}
    ok = True
    for setting, expected in REFERENCE_EVAL.items():
        if setting not in got_by_setting:
            print(f"DIFF eval/{setting}: missing")
            ok = False
            continue
        got = got_by_setting[setting]
        for key, ref_value in expected.items():
            if key in {"scored", "correct"}:
                same = int(got.get(key, -1)) == int(ref_value)
                print(
                    f"{'OK' if same else 'DIFF'} eval/{setting}/{key}: "
                    f"got={int(got.get(key, -1))} ref={int(ref_value)}"
                )
                ok = same and ok
            else:
                atol = accuracy_atol if key == "accuracy" else metric_atol
                ok = compare_float(f"eval/{setting}/{key}", float(got[key]), float(ref_value), atol) and ok
    return ok


def main() -> None:
    args = parse_args()
    ok = True
    train_metrics = resolve_train_metrics(args)
    if train_metrics is not None:
        if not train_metrics.exists():
            raise SystemExit(f"missing train metrics: {train_metrics}")
        ok = compare_train(train_metrics, args.loss_atol) and ok
    if args.eval_json is not None:
        if not args.eval_json.exists():
            raise SystemExit(f"missing eval json: {args.eval_json}")
        ok = compare_eval(args.eval_json, args.metric_atol, args.accuracy_atol) and ok
    if train_metrics is None and args.eval_json is None:
        raise SystemExit("provide --run-dir/--train-metrics and/or --eval-json")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
