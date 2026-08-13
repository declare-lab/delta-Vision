from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Log Delta-Vision evaluation JSON files to W&B.")
    parser.add_argument("--json", nargs="+", required=True, help="Evaluation JSON files to log.")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--project", default="delta-vision")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def metric_prefix(payload: dict[str, Any], path: Path) -> str:
    benchmark = str(payload.get("benchmark") or "diagnosis")
    if "aggregate" in payload:
        return f"eval/{benchmark}/diagnosis"
    scale = payload.get("sidecar_scale")
    scale_part = f"scale{scale:g}" if isinstance(scale, (int, float)) else path.stem
    return f"eval/{benchmark}/{scale_part}"


def collect_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prefix = metric_prefix(payload, path)
    metrics: dict[str, float] = {}
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        for key, value in aggregate.items():
            if isinstance(value, (int, float)):
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics

    for result in payload.get("results", []):
        setting = str(result.get("setting", "unknown"))
        for key, value in result.items():
            if key == "setting":
                continue
            if isinstance(value, (int, float)):
                metrics[f"{prefix}/{setting}/{key}"] = float(value)
    skipped = payload.get("skipped")
    if isinstance(skipped, (int, float)):
        metrics[f"{prefix}/skipped"] = float(skipped)
    return metrics


def main() -> None:
    args = parse_args()
    if args.mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        id=args.run_id,
        resume="allow",
        mode=args.mode,
    )
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    metrics: dict[str, float | int] = {"eval/step": int(args.step)}
    for item in args.json:
        path = Path(item)
        if path.exists():
            metrics.update(collect_metrics(path))
    if len(metrics) > 1:
        wandb.log(metrics, step=int(args.step))
    run.finish()


if __name__ == "__main__":
    main()
