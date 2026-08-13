#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import UserDict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from delta_vision.evaluation.metrics import OPTIONS, option_scores, option_token_id_lists
from delta_vision.models.gemma4 import (
    build_gemma4_initial_context,
    gather_batched_positions,
    gemma4_lm_logits,
    gemma4_prompt,
    gemma4_text_attention_masks,
    get_gemma4_language_model,
    get_gemma4_text_image_positions,
    load_frozen_gemma4,
    prepare_gemma4_eval_inputs,
    run_gemma4_full_layer_with_text_delta,
    run_gemma4_layer_text_with_attention_delta,
)
from delta_vision.models.llava import dtype_from_name, read_jsonl
from delta_vision.models.sidecar import DeltaVisionModule


CHOICE_RE = re.compile(r"(?m)(?:^|\b)([A-D])(?:[.)：:]|\\s*:)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Gemma4 delta-vision sidecar/hybrid prompt logits.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--shared-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def build_question(row: dict[str, Any], benchmark: str) -> str:
    question = str(row["question"]).strip()
    if benchmark == "mmstar":
        return f"{question}\nAnswer directly with only the letter of the correct option."
    return f"{question}\nAnswer directly with the final answer only."


def candidate_kind(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "mmstar":
        return "abcd"
    question = str(row.get("question", ""))
    answer = str(row.get("answer", "")).strip()
    if CHOICE_RE.search(question) or answer.upper()[:1] in OPTIONS:
        return "abcd"
    if answer.lower() in {"yes", "no"}:
        return "yesno"
    return "skip"


def candidate_ids(tokenizer: Any, kind: str) -> dict[str, list[int]]:
    if kind == "abcd":
        return option_token_id_lists(tokenizer)
    if kind == "yesno":
        out: dict[str, list[int]] = {}
        for key, variants in {
            "Yes": ("Yes", " Yes", "yes", " yes"),
            "No": ("No", " No", "no", " no"),
        }.items():
            ids = set()
            for text in variants:
                encoded = tokenizer(text, add_special_tokens=False).input_ids
                if encoded:
                    ids.add(int(encoded[-1]))
            out[key] = sorted(ids)
        return out
    raise ValueError(f"unsupported candidate kind: {kind}")


def normalize_gold(row: dict[str, Any], kind: str) -> str:
    answer = str(row.get("answer", "")).strip()
    if kind == "abcd":
        return answer.upper()[:1]
    if kind == "yesno":
        return "Yes" if answer.lower().startswith("yes") else "No"
    return answer


def score_candidates(logits: torch.Tensor, ids: dict[str, list[int]]) -> torch.Tensor:
    if set(ids.keys()) == set(OPTIONS):
        return option_scores(logits, ids)
    scores = []
    for key in ids:
        idx = torch.tensor(ids[key], device=logits.device, dtype=torch.long)
        scores.append(logits.index_select(0, idx).max())
    return torch.stack(scores)


def predict(logits: torch.Tensor, ids: dict[str, list[int]]) -> str:
    keys = list(ids)
    scores = score_candidates(logits.float(), ids)
    return keys[int(scores.argmax().item())]


def distribution(logits: torch.Tensor, ids: dict[str, list[int]]) -> torch.Tensor:
    return F.softmax(score_candidates(logits.float(), ids), dim=0)


def load_sidecar(args: argparse.Namespace, hidden_size: int, num_layers: int, device: torch.device, dtype: torch.dtype) -> DeltaVisionModule:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for name in (
        "rank",
        "sidecar_dim",
        "num_heads",
        "reader_mlp_ratio",
        "layer_adapter_rank",
        "shared_basis",
    ):
        if name in checkpoint_args:
            setattr(args, name, checkpoint_args[name])
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=True,
        reader_mlp_ratio=args.reader_mlp_ratio,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=args.shared_basis,
        output_mode="residual",
    ).to(device=device, dtype=dtype)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    sidecar.load_state_dict(state_dict, strict=True)
    sidecar.eval()
    for param in sidecar.parameters():
        param.requires_grad_(False)
    sidecar.runtime_fold_output_basis = True
    sidecar.prepare_inference_cache(device, dtype)
    return sidecar


@torch.inference_mode()
def teacher_logits(processor: Any, model: torch.nn.Module, row: dict[str, Any], benchmark: str, device: torch.device) -> torch.Tensor:
    inputs = prepare_gemma4_eval_inputs(processor, row, build_question(row, benchmark), device)
    out = model(**inputs, return_dict=True, use_cache=False)
    hidden0, position_ids, _, image_mask_raw, _ = build_gemma4_initial_context(model, inputs)
    text_pos, _, _, text_mask, _, _ = get_gemma4_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        image_mask_raw,
        position_ids,
        None,
    )
    del hidden0
    last_text = int(text_pos[0, int(text_mask[0].sum().item()) - 1].item())
    return out.logits[0, last_text]


@torch.inference_mode()
def rollout_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
    sidecar: DeltaVisionModule | None,
    sidecar_scale: float,
    hybrid: bool,
) -> torch.Tensor:
    inputs = prepare_gemma4_eval_inputs(processor, row, build_question(row, benchmark), device)
    hidden0, position_ids, per_layer_inputs, image_mask_raw, full_masks = build_gemma4_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_gemma4_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        image_mask_raw,
        position_ids,
        None,
    )
    visual_memory = gather_batched_positions(hidden0, image_pos, image_mask).to(dtype=dtype)
    visual_kv = sidecar.prepare_visual_kv(visual_memory, ~image_mask) if sidecar is not None else None
    num_layers = len(language_model.layers)
    if hybrid:
        h_full = hidden0.to(dtype=dtype)
        shared_kv: UserDict = UserDict()
        for layer_idx in range(num_layers):
            text_h = gather_batched_positions(h_full, text_pos, text_mask)
            delta = None
            if sidecar is not None:
                assert visual_kv is not None
                delta = sidecar.decode_no_state_layer(text_h, layer_idx, visual_kv) * sidecar_scale
            per_layer_input = per_layer_inputs[:, :, layer_idx, :].to(dtype=dtype) if per_layer_inputs is not None else None
            h_full = run_gemma4_full_layer_with_text_delta(
                language_model,
                layer_idx,
                h_full,
                full_masks,
                position_ids,
                shared_kv,
                text_pos,
                delta,
                text_mask,
                per_layer_input,
            )
        final_text = gather_batched_positions(language_model.norm(h_full), text_pos, text_mask)
    else:
        h = gather_batched_positions(hidden0, text_pos, text_mask).to(dtype=dtype)
        text_masks = gemma4_text_attention_masks(language_model, h, text_mask, text_position_ids)
        shared_kv = UserDict()
        for layer_idx in range(num_layers):
            delta = None
            if sidecar is not None:
                assert visual_kv is not None
                delta = sidecar.decode_no_state_layer(h, layer_idx, visual_kv) * sidecar_scale
            per_layer_input = None
            if per_layer_inputs is not None:
                per_layer_input = gather_batched_positions(per_layer_inputs[:, :, layer_idx, :], text_pos, text_mask).to(dtype=dtype)
            h = run_gemma4_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h,
                text_masks,
                text_position_ids,
                shared_kv,
                delta,
                per_layer_input,
            )
        final_text = language_model.norm(h)
    logits = gemma4_lm_logits(model, final_text)
    last_idx = int(text_mask[0].sum().item()) - 1
    return logits[0, last_idx]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_gemma4(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_gemma4_language_model(model)
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    sidecar = load_sidecar(args, hidden_size, num_layers, device, dtype) if args.checkpoint else None

    rows = read_jsonl(args.data, None)[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    metrics: dict[str, dict[str, float]] = {
        "gemma4": {"scored": 0, "correct": 0},
        "no_visual": {"scored": 0, "correct": 0, "agree": 0, "teacher_correct_retention": 0, "kl": 0.0},
    }
    if sidecar is not None:
        metrics["sidecar_only"] = {"scored": 0, "correct": 0, "agree": 0, "teacher_correct_retention": 0, "kl": 0.0}
        metrics["hybrid"] = {"scored": 0, "correct": 0, "agree": 0, "teacher_correct_retention": 0, "kl": 0.0}
    predictions: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            skipped += 1
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        try:
            t_logits = teacher_logits(processor, model, row, args.benchmark, device)
            nv_logits = rollout_logits(processor, model, language_model, row, args.benchmark, device, dtype, None, 0.0, False)
            sc_logits = None
            hy_logits = None
            if sidecar is not None:
                sc_logits = rollout_logits(processor, model, language_model, row, args.benchmark, device, dtype, sidecar, args.sidecar_scale, False)
                hy_logits = rollout_logits(processor, model, language_model, row, args.benchmark, device, dtype, sidecar, args.sidecar_scale, True)
        except Exception as exc:
            skipped += 1
            predictions.append({"index": args.start_index + idx, "error": repr(exc)})
            continue
        teacher_pred = predict(t_logits, ids)
        teacher_dist = distribution(t_logits, ids)
        teacher_correct = teacher_pred == gold
        metrics["gemma4"]["scored"] += 1
        metrics["gemma4"]["correct"] += int(teacher_correct)
        row_pred: dict[str, Any] = {
            "index": args.start_index + idx,
            "gold": gold,
            "gemma4": teacher_pred,
        }
        for name, logits in (("no_visual", nv_logits), ("sidecar_only", sc_logits), ("hybrid", hy_logits)):
            if logits is None or name not in metrics:
                continue
            pred = predict(logits, ids)
            dist = distribution(logits, ids)
            metrics[name]["scored"] += 1
            metrics[name]["correct"] += int(pred == gold)
            metrics[name]["agree"] += int(pred == teacher_pred)
            metrics[name]["teacher_correct_retention"] += int(teacher_correct and pred == teacher_pred)
            metrics[name]["kl"] += float(F.kl_div(dist.log(), teacher_dist, reduction="sum").item())
            row_pred[name] = pred
        predictions.append(row_pred)
        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(rows)}", flush=True)

    teacher_scored = max(int(metrics["gemma4"]["scored"]), 1)
    teacher_correct_total = int(metrics["gemma4"]["correct"])
    results = [
        {
            "setting": "gemma4",
            "scored": int(metrics["gemma4"]["scored"]),
            "correct": int(metrics["gemma4"]["correct"]),
            "accuracy": float(metrics["gemma4"]["correct"]) / teacher_scored,
        }
    ]
    for key in ("no_visual", "sidecar_only", "hybrid"):
        if key not in metrics:
            continue
        row_metrics = metrics[key]
        n = max(int(row_metrics["scored"]), 1)
        results.append(
            {
                "setting": key,
                "scored": int(row_metrics["scored"]),
                "correct": int(row_metrics["correct"]),
                "accuracy": float(row_metrics["correct"]) / n,
                "gemma4_agreement": float(row_metrics["agree"]) / n,
                "gemma4_correct_retention": float(row_metrics["teacher_correct_retention"]) / max(teacher_correct_total, 1),
                "output_kl_to_gemma4": float(row_metrics["kl"]) / n,
            }
        )
    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "rank": args.rank,
        "sidecar_dim": args.sidecar_dim,
        "num_heads": args.num_heads,
        "layer_adapter_rank": args.layer_adapter_rank,
        "shared_basis": args.shared_basis,
        "sidecar_scale": args.sidecar_scale,
        "skipped": skipped,
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
