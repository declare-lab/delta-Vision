#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from PIL import Image

from delta_vision.runtime.basis import load_layer_basis, project_delta_to_coefficients, reconstruct_delta
from delta_vision.models.llava import (
    compute_llama_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    read_jsonl,
    run_llama_layer_text_with_attention_delta,
)
from delta_vision.evaluation.metrics import OPTIONS, option_distribution, option_scores, option_token_id_lists
from delta_vision.models.modeling import image_token_id, load_frozen_llava


CHOICE_RE = re.compile(r"(?m)^\s*([A-D])[.)：:]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate RealWorldQA online attention-effect oracle with low-rank PCA residuals.")
    parser.add_argument("--data", default="data/realworldqa/test.jsonl")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--output-json", default="artifacts/eval/oracle/realworldqa_attention_oracle_lowrank.json")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ranks", default="32,64,128,256,512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def build_prompt(row: dict[str, Any]) -> str:
    return llava15_prompt(str(row["question"]).strip())


def candidate_kind(row: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    answer = str(row.get("answer", "")).strip().lower()
    if CHOICE_RE.search(question) or str(row.get("answer", "")).strip().upper()[:1] in OPTIONS:
        return "abcd"
    if answer in {"yes", "no"}:
        return "yesno"
    return "text"


def candidate_ids(tokenizer: Any, kind: str) -> dict[str, list[int]]:
    if kind == "abcd":
        return option_token_id_lists(tokenizer)
    if kind == "yesno":
        out = {}
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
    raise ValueError(f"unsupported candidate kind {kind}")


def score_candidates(logits: torch.Tensor, ids: dict[str, list[int]]) -> torch.Tensor:
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


def normalize_gold(row: dict[str, Any], kind: str) -> str:
    answer = str(row.get("answer", "")).strip()
    if kind == "abcd":
        return answer.upper()[:1]
    if kind == "yesno":
        return "Yes" if answer.lower().startswith("yes") else "No"
    return answer


@torch.inference_mode()
def rollout_with_oracle(
    language_model: torch.nn.Module,
    teacher_hidden_states: tuple[torch.Tensor, ...],
    text_positions: torch.Tensor,
    mode: str,
    attention_deltas: list[torch.Tensor] | None = None,
    basis: torch.Tensor | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    num_layers = len(get_lm_layers(language_model))
    h = teacher_hidden_states[0].index_select(1, text_positions)
    position_ids = text_positions.unsqueeze(0)
    for layer_idx in range(num_layers):
        attn_delta = None
        if mode.startswith("attention"):
            if attention_deltas is None:
                attn_delta = compute_llama_attention_effect(
                    language_model,
                    layer_idx,
                    teacher_hidden_states[layer_idx],
                    text_positions,
                ).to(dtype=h.dtype)
            else:
                attn_delta = attention_deltas[layer_idx].to(dtype=h.dtype)
            if mode == "attention_rank":
                if basis is None or rank is None:
                    raise ValueError("attention_rank needs basis/rank")
                layer_basis = basis[layer_idx : layer_idx + 1, :rank].expand(attn_delta.shape[0], -1, -1)
                coeff = project_delta_to_coefficients(attn_delta, layer_basis)
                attn_delta = reconstruct_delta(coeff, layer_basis)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=attn_delta,
        )
    return h


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.max_samples)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_layers = len(get_lm_layers(language_model))
    img_token = image_token_id(model, processor)
    basis = load_layer_basis(args.basis, max(ranks), num_layers, 4096).to(device=device, dtype=dtype)

    settings: list[tuple[str, int | None]] = [("teacher", None), ("no_visual", None), ("attention_full", None)]
    settings += [("attention_rank", rank) for rank in ranks]
    metrics = {
        f"{mode}{'' if rank is None else f'_{rank}'}": {
            "correct": 0,
            "agree": 0,
            "teacher_correct_retention": 0,
            "kl": 0.0,
            "scored": 0,
        }
        for mode, rank in settings
    }
    teacher_correct = 0
    skipped = 0
    predictions: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        kind = candidate_kind(row)
        if kind == "text":
            skipped += 1
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        image = Image.open(row["image"]).convert("RGB")
        inputs = processor(text=build_prompt(row), images=image, return_tensors="pt")
        image.close()
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        teacher = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = tuple(x.detach().to(dtype=dtype) for x in teacher.hidden_states[:-1])
        text_pos, _, _ = get_text_and_image_positions(inputs["input_ids"], hidden_states[0].shape[1], img_token)
        text_pos = text_pos.to(device)
        last_text_idx = int(text_pos[-1].item())
        teacher_logits = teacher.logits[0, last_text_idx]
        teacher_pred = predict(teacher_logits, ids)
        teacher_dist = distribution(teacher_logits, ids)
        is_teacher_correct = teacher_pred == gold
        teacher_correct += int(is_teacher_correct)
        attention_deltas = [
            compute_llama_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx],
                text_pos,
            ).detach()
            for layer_idx in range(num_layers)
        ]

        sample_pred = {"index": row.get("index", idx), "gold": gold, "teacher": teacher_pred, "kind": kind}
        for mode, rank in settings:
            key = f"{mode}{'' if rank is None else f'_{rank}'}"
            if mode == "teacher":
                logits = teacher_logits
            else:
                h = rollout_with_oracle(
                    language_model,
                    hidden_states,
                    text_pos,
                    "no_visual" if mode == "no_visual" else mode,
                    attention_deltas=attention_deltas,
                    basis=basis,
                    rank=rank,
                )
                logits = model.lm_head(get_lm_norm(language_model)(h))[0, -1]
            pred = predict(logits, ids)
            dist = distribution(logits, ids)
            metrics[key]["correct"] += int(pred == gold)
            metrics[key]["agree"] += int(pred == teacher_pred)
            metrics[key]["teacher_correct_retention"] += int(is_teacher_correct and pred == teacher_pred)
            metrics[key]["kl"] += float(F.kl_div(dist.log(), teacher_dist, reduction="sum").item())
            metrics[key]["scored"] += 1
            sample_pred[key] = pred
        predictions.append(sample_pred)
        if (idx + 1) % 25 == 0:
            print(f"processed {idx + 1}/{len(rows)} scored={metrics['teacher']['scored']} skipped={skipped}", flush=True)

    results = []
    for mode, rank in settings:
        key = f"{mode}{'' if rank is None else f'_{rank}'}"
        row_metrics = metrics[key]
        n = max(int(row_metrics["scored"]), 1)
        results.append(
            {
                "setting": key,
                "mode": mode,
                "rank": rank,
                "scored": row_metrics["scored"],
                "correct": row_metrics["correct"],
                "accuracy": row_metrics["correct"] / n,
                "teacher_agreement": row_metrics["agree"] / n,
                "teacher_correct_retention": row_metrics["teacher_correct_retention"] / max(teacher_correct, 1),
                "output_kl": row_metrics["kl"] / n,
            }
        )

    payload = {
        "data": args.data,
        "basis": args.basis,
        "num_rows": len(rows),
        "skipped_text_answers": skipped,
        "teacher_correct": teacher_correct,
        "scoring": "last-prompt candidate-token scoring; A-D or Yes/No candidates only",
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
