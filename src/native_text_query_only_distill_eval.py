from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from src.native_text_query_only_distill import ROOT, OUT, MODEL, VisualFFNOnly
from src.initial_token_mlp_probe import runtime
from src.model import load_frozen_qwen3vl
from src.qwen_deepstack import disable_qwen_deepstack
from src.visual_cross_token_ablation import prepare
from src.data import QwenBenchmarkDataset
from src.benchmarks import get_benchmark_spec, score_prediction
from src.eval_benchmarks import generate_teacher_qwen

DATASETS = ("realworldqa", "mmstar", "sqa")


def worker(stage: str, rank: int) -> None:
    runtime()
    source = MODEL if stage == "before" else str(OUT / "checkpoint-2000")
    processor, model = load_frozen_qwen3vl(source, torch.bfloat16, torch.device("cuda:0"), "flash_attention_2")
    disable_qwen_deepstack(model)
    hook = VisualFFNOnly(model)
    depth = len(model.model.language_model.layers)
    folder = OUT / f"eval_{stage}"
    with torch.inference_mode(), (folder / f"rows{rank}.jsonl").open("w", buffering=1) as handle:
        for dataset_name in DATASETS:
            ds = QwenBenchmarkDataset(
                str(ROOT / f"artifacts/diagnostics/channel_native_cache_20260916/{dataset_name}_eval.jsonl"),
                processor,
                dataset_name,
            )
            for i in range(rank, len(ds), 8):
                item = ds[i]
                inputs = prepare(item, model.device)
                hook.mask = inputs["input_ids"] == model.config.image_token_id
                modes = ("native", "text_query_only_untrained") if stage == "before" else ("text_query_only_trained",)
                for mode in modes:
                    hook.enabled = mode != "native"
                    hook.calls = 0
                    _, answer = generate_teacher_qwen(
                        model,
                        processor,
                        **inputs,
                        max_new_tokens=get_benchmark_spec(dataset_name).max_new_tokens,
                    )
                    expected = 0 if mode == "native" else depth
                    assert hook.calls == expected, (dataset_name, i, mode, hook.calls, expected)
                    score = score_prediction(
                        metric=get_benchmark_spec(dataset_name).metric,
                        prediction_text=answer,
                        answer=item.get("answer"),
                        answers=item.get("answers"),
                        choices=item.get("choices"),
                        question=item.get("row", {}).get("question"),
                    )
                    handle.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "sample": i,
                                "mode": mode,
                                "answer": answer,
                                **score,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if i // 8 % 30 == 0:
                    print(dataset_name, i, flush=True)


def launch(stage: str) -> None:
    folder = OUT / f"eval_{stage}"
    folder.mkdir(parents=True, exist_ok=False)
    jobs = []
    try:
        for rank in range(8):
            log = (folder / f"gpu{rank}.log").open("w")
            p = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.native_text_query_only_distill_eval", stage, str(rank)],
                cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((p, log))
        while any(p.poll() is None for p, _ in jobs):
            if any(p.poll() not in (None, 0) for p, _ in jobs):
                raise RuntimeError("eval failed")
            time.sleep(3)
        rows = []
        for rank in range(8):
            rows.extend(json.loads(line) for line in (folder / f"rows{rank}.jsonl").read_text().splitlines() if line.strip())
        result = {}
        for dataset_name, total in (("realworldqa", 765), ("mmstar", 1000), ("sqa", 1000)):
            result[dataset_name] = {}
            modes = ("native", "text_query_only_untrained") if stage == "before" else ("text_query_only_trained",)
            for mode in modes:
                selected = [r for r in rows if r["dataset"] == dataset_name and r["mode"] == mode]
                assert len(selected) == total and {r["sample"] for r in selected} == set(range(total)), (
                    dataset_name,
                    mode,
                    len(selected),
                    total,
                )
                result[dataset_name][mode] = {
                    "samples": total,
                    "accuracy_pct": 100 * sum(float(r["score"]) for r in selected) / total,
                    "invalid_pct": 100 * sum(1 for r in selected if r.get("invalid")) / total,
                }
        (folder / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        for p, log in jobs:
            if p.poll() is None:
                p.terminate()
            log.close()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        worker(sys.argv[1], int(sys.argv[2]))
    else:
        launch(sys.argv[1])
