#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmarks import normalize_answer  # noqa: E402
from src.eval_benchmarks import configure_torch_runtime, generate_adapter_qwen, generate_teacher_qwen  # noqa: E402
from src.model import dtype_from_name, load_frozen_qwen3vl, load_qwen_visual_delta_checkpoint  # noqa: E402


DEFAULT_MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_DATA = ROOT_DIR / "data" / "rendered_context_qa_eval_v1" / "paired.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate raw-text context QA vs rendered ordered-page QA.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="paired.jsonl from prepare_rendered_context_qa_eval.py")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default="", help="Optional adapter checkpoint for rendered-adapter mode.")
    parser.add_argument(
        "--modes",
        default="text-base,rendered-base,rendered-adapter",
        help="Comma-separated: text-base, rendered-base, rendered-adapter. rendered-adapter is skipped without --checkpoint.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--source", default="", help="Optional source_dataset filter, e.g. locomo or hotpotqa.")
    parser.add_argument("--answer-visible", choices=("all", "visible", "not-visible"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def read_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    if args.source.strip():
        rows = [row for row in rows if str(row.get("source_dataset")) == args.source.strip()]
    if args.answer_visible == "visible":
        rows = [row for row in rows if bool(row.get("answer_visible"))]
    elif args.answer_visible == "not-visible":
        rows = [row for row in rows if not bool(row.get("answer_visible"))]
    if args.max_samples and args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("no rows left after filters")
    return rows


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "artifacts" / "evals" / "rendered_context_qa" / stamp


def split_modes(value: str, checkpoint: str) -> list[str]:
    modes = []
    for item in re.split(r"[\s,]+", value.strip()):
        if not item:
            continue
        if item not in {"text-base", "rendered-base", "rendered-adapter"}:
            raise ValueError(f"unknown mode: {item}")
        if item == "rendered-adapter" and not checkpoint.strip():
            continue
        modes.append(item)
    if not modes:
        raise ValueError("no eval modes selected")
    return modes


def resolve_image_paths(row: dict[str, Any], data_path: Path) -> list[Path]:
    raw_paths = row.get("images")
    if raw_paths is None:
        raw_paths = [row["image"]]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("row must contain non-empty images")
    root = Path(str(row.get("image_root") or data_path.parent)).expanduser()
    out = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        out.append(path)
    return out


def text_prompt(row: dict[str, Any]) -> str:
    return (
        "Use the context below to answer the question.\n\n"
        f"Context:\n{row['text_context']}\n\n"
        f"Question: {row['question']}\n"
        "Answer directly with a short phrase."
    )


def rendered_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("rendered_question") or row.get("question") or "").strip()
    return f"{question}\nAnswer directly with a short phrase."


def eos_kwargs(processor: Any) -> dict[str, Any]:
    eos_ids = {
        int(token_id)
        for token_id in [
            getattr(processor.tokenizer, "eos_token_id", None),
            getattr(processor.tokenizer, "pad_token_id", None),
        ]
        if token_id is not None
    }
    kwargs: dict[str, Any] = {}
    if eos_ids:
        kwargs["eos_token_id"] = sorted(eos_ids)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    return kwargs


@torch.inference_mode()
def generate_text_base(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, dict[str, int]]:
    messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt(row)}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **eos_kwargs(processor),
    )
    text = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text, {"input_tokens": int(inputs["attention_mask"].sum().item()), "image_tokens": 0}


def build_rendered_inputs(
    processor: Any,
    row: dict[str, Any],
    data_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    image_paths = resolve_image_paths(row, data_path)
    images = [Image.open(path).convert("RGB") for path in image_paths]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": rendered_prompt(row)})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    for image in images:
        image.close()
    if "mm_token_type_ids" not in inputs:
        raise ValueError("Qwen processor did not return mm_token_type_ids for rendered pages")
    counts = {
        "input_tokens": int(inputs["attention_mask"].sum().item()),
        "image_tokens": int(((inputs["mm_token_type_ids"] == 1) & inputs["attention_mask"].bool()).sum().item()),
    }
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}, counts


def token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def rouge_l(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if not gold_tokens:
            continue
        dp = [0] * (len(gold_tokens) + 1)
        for token in pred_tokens:
            prev = 0
            for idx, gold in enumerate(gold_tokens, start=1):
                cur = dp[idx]
                if token == gold:
                    dp[idx] = prev + 1
                else:
                    dp[idx] = max(dp[idx], dp[idx - 1])
                prev = cur
        lcs = dp[-1]
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score_text(prediction: str, answers: list[str]) -> dict[str, Any]:
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(answer) for answer in answers if normalize_answer(answer)]
    exact = any(pred_norm and pred_norm == gold for gold in gold_norms)
    contains_gold = any(gold and gold in pred_norm for gold in gold_norms)
    pred_in_gold = any(pred_norm and pred_norm in gold for gold in gold_norms)
    return {
        "prediction_norm": pred_norm,
        "gold_norms": gold_norms,
        "exact": float(exact),
        "contains_gold": float(contains_gold),
        "bidirectional_contains": float(contains_gold or pred_in_gold),
        "token_f1": token_f1(prediction, answers),
        "rouge_l": rouge_l(prediction, answers),
        "invalid": float(not bool(pred_norm)),
        "completion_chars": len(prediction.strip()),
        "completion_words": len(prediction.strip().split()),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"samples": 0}
    keys = [
        "exact",
        "contains_gold",
        "bidirectional_contains",
        "token_f1",
        "rouge_l",
        "invalid",
        "completion_chars",
        "completion_words",
        "total_s",
        "input_tokens",
        "image_tokens",
        "context_tokens",
        "compression_ratio",
        "num_pages",
    ]
    out: dict[str, Any] = {"samples": len(items)}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in items if item.get(key) is not None]
        if values:
            out[f"mean_{key}"] = sum(values) / len(values)
    return out


def grouped_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"overall": summarize(predictions), "by_source": {}, "by_answer_visible": {}}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_visible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_source[str(item.get("source_dataset", ""))].append(item)
        by_visible["visible" if item.get("answer_visible") else "not_visible"].append(item)
    result["by_source"] = {key: summarize(value) for key, value in sorted(by_source.items())}
    result["by_answer_visible"] = {key: summarize(value) for key, value in sorted(by_visible.items())}
    return result


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    rows = read_rows(data_path, args)
    modes = split_modes(args.modes, args.checkpoint)
    out_dir = output_dir_from_args(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter = None
    adapter_meta = None
    if "rendered-adapter" in modes:
        adapter, adapter_meta = load_qwen_visual_delta_checkpoint(args.checkpoint, model.model.language_model, device, dtype)

    mode_predictions: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for idx, row in enumerate(rows):
        answers = [str(item) for item in (row.get("answers") or [row.get("answer", "")]) if str(item).strip()]
        for mode in modes:
            start = time.perf_counter()
            if mode == "text-base":
                output, counts = generate_text_base(model, processor, row, device, int(args.max_new_tokens))
            else:
                inputs, counts = build_rendered_inputs(processor, row, data_path, device)
                if mode == "rendered-base":
                    _, output = generate_teacher_qwen(model, processor, max_new_tokens=int(args.max_new_tokens), **inputs)
                elif mode == "rendered-adapter":
                    assert adapter is not None
                    _, output = generate_adapter_qwen(
                        model,
                        processor,
                        adapter,
                        max_new_tokens=int(args.max_new_tokens),
                        early_stop_metric=None,
                        **inputs,
                    )
                else:
                    raise AssertionError(mode)
            elapsed = time.perf_counter() - start
            scored = score_text(output, answers)
            record = {
                "index": row.get("index", idx),
                "mode": mode,
                "source_dataset": row.get("source_dataset"),
                "source_id": row.get("source_id"),
                "answer_visible": bool(row.get("answer_visible")),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "answers": answers,
                "prediction": output.strip(),
                "total_s": elapsed,
                "input_tokens": counts["input_tokens"],
                "image_tokens": counts["image_tokens"],
                "context_tokens": row.get("context_tokens"),
                "compression_ratio": row.get("compression_ratio"),
                "num_pages": row.get("num_pages"),
                **scored,
            }
            mode_predictions[mode].append(record)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            short = []
            for mode in modes:
                summary = summarize(mode_predictions[mode])
                short.append(f"{mode}:f1={summary.get('mean_token_f1', 0.0):.3f}")
            print(f"[{idx + 1}/{len(rows)}] " + " ".join(short), flush=True)

    summary = {
        "data": str(data_path),
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "adapter_meta": adapter_meta,
        "samples": len(rows),
        "modes": modes,
        "filters": {
            "source": args.source,
            "answer_visible": args.answer_visible,
            "max_samples": int(args.max_samples),
        },
        "results": {mode: grouped_summary(predictions) for mode, predictions in mode_predictions.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, predictions in mode_predictions.items():
        with (out_dir / f"predictions_{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
