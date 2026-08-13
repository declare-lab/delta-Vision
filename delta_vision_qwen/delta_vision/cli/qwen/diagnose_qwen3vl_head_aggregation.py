#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prompt,
    run_qwen3vl_layer_text_with_attention_delta,
)
from delta_vision.cli.qwen.diagnose_qwen3vl_head_sparsity import (
    head_delta_to_hidden,
    teacher_head_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate static head-aggregation oracles for Qwen3-VL visual effects.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--basis-output", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--basis-samples", type=int, default=200)
    parser.add_argument("--groups", default="4,8,16,24,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_groups(spec: str, num_heads: int) -> list[int]:
    groups = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not groups:
        raise ValueError("--groups cannot be empty")
    if groups[0] <= 0 or groups[-1] > num_heads:
        raise ValueError(f"--groups must be in [1, {num_heads}]")
    return groups


def build_prompt(processor: Any, row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    question = f"{question}\nAnswer directly with only the letter of the correct option."
    return qwen3vl_prompt(processor, question)


@torch.inference_mode()
def prompt_inputs(processor: Any, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def update_metrics(metrics: dict[str, float | int], pred: str, teacher_pred: str, gold: str, kl: float) -> None:
    metrics["correct"] = int(metrics["correct"]) + int(pred == gold)
    metrics["agree"] = int(metrics["agree"]) + int(pred == teacher_pred)
    metrics["teacher_correct_and_agree"] = int(metrics["teacher_correct_and_agree"]) + int(
        teacher_pred == gold and pred == teacher_pred
    )
    metrics["output_kl_sum"] = float(metrics["output_kl_sum"]) + float(kl)
    metrics["scored"] = int(metrics["scored"]) + 1


def finalize_metrics(metrics: dict[str, float | int], teacher_correct: int) -> dict[str, float | int]:
    scored = max(int(metrics["scored"]), 1)
    teacher_correct = max(int(teacher_correct), 1)
    return {
        "scored": int(metrics["scored"]),
        "correct": int(metrics["correct"]),
        "accuracy": float(metrics["correct"]) / scored,
        "teacher_agreement": float(metrics["agree"]) / scored,
        "teacher_correct_retention": float(metrics["teacher_correct_and_agree"]) / teacher_correct,
        "output_kl": float(metrics["output_kl_sum"]) / scored,
    }


def update_head_cov(cov: torch.Tensor, head_delta: torch.Tensor, text_mask: torch.Tensor) -> None:
    # head_delta: [B, H, T, D]. Build X=[valid_tokens*D, H].
    valid = text_mask.to(device=head_delta.device, dtype=torch.bool)
    x = head_delta.float().permute(0, 2, 3, 1)[valid]  # [tokens, D, H]
    if x.numel() == 0:
        return
    x = x.reshape(-1, head_delta.shape[1])
    cov += torch.matmul(x.transpose(0, 1), x).double().cpu()


@torch.inference_mode()
def collect_head_bases(
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    num_layers = len(language_model.layers)
    num_heads = int(language_model.config.num_attention_heads)
    covs = [torch.zeros(num_heads, num_heads, dtype=torch.float64) for _ in range(num_layers)]

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, _, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
            for state in teacher_states
        ]
        for layer_idx in range(num_layers):
            head_delta, _ = teacher_head_delta(
                language_model,
                layer_idx,
                teacher_states[layer_idx],
                teacher_text_states[layer_idx],
                full_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            update_head_cov(covs[layer_idx], head_delta, text_mask)
        if (idx + 1) % 25 == 0:
            print(f"basis {idx + 1}/{len(rows)}", flush=True)

    bases = []
    explained = []
    for layer_idx, cov in enumerate(covs):
        evals, evecs = torch.linalg.eigh(cov.float())
        order = torch.argsort(evals, descending=True)
        evals = evals[order].clamp_min(0)
        evecs = evecs[:, order]
        bases.append(evecs)
        denom = evals.sum().clamp_min(1e-12)
        explained.append((evals.cumsum(dim=0) / denom).tolist())
    return {
        "basis": torch.stack(bases, dim=0),
        "explained": explained,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


def project_head_delta(head_delta: torch.Tensor, basis: torch.Tensor, groups: int) -> torch.Tensor:
    # head_delta: [B,H,T,D], basis: [H,H].
    b = basis[:, :groups].to(device=head_delta.device, dtype=head_delta.dtype)
    coeff = torch.einsum("bhtd,hg->bgtd", head_delta, b)
    return torch.einsum("bgtd,hg->bhtd", coeff, b)


@torch.inference_mode()
def evaluate_group_oracles(
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    bases: torch.Tensor,
    groups: list[int],
    predictions_jsonl: str,
) -> dict[str, Any]:
    option_ids = option_token_id_lists(processor.tokenizer)
    num_layers = len(language_model.layers)
    names = ["no_visual", "full_heads"] + [f"group_{g}" for g in groups]
    metrics = {
        name: {"correct": 0, "agree": 0, "teacher_correct_and_agree": 0, "output_kl_sum": 0.0, "scored": 0}
        for name in names
    }
    teacher_correct = 0
    predictions: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
            for state in teacher_states
        ]
        head_deltas = [
            teacher_head_delta(
                language_model,
                layer_idx,
                teacher_states[layer_idx],
                teacher_text_states[layer_idx],
                full_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )[0]
            for layer_idx in range(num_layers)
        ]
        last_text_idx = int(text_mask[0].sum().item()) - 1
        teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_correct += int(teacher_pred == gold)
        sample_pred = {"index": row.get("index", idx), "gold": gold, "teacher": teacher_pred}

        for name in names:
            h = teacher_text_states[0]
            for layer_idx in range(num_layers):
                delta = None
                if name != "no_visual":
                    head_delta = head_deltas[layer_idx]
                    if name.startswith("group_"):
                        group_count = int(name.rsplit("_", 1)[1])
                        head_delta = project_head_delta(head_delta, bases[layer_idx], group_count)
                    delta = head_delta_to_hidden(language_model, layer_idx, head_delta, dtype)
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    padding_mask=~text_mask,
                )
            logits = model.lm_head(language_model.norm(h))[0, last_text_idx]
            pred = predict_option(logits, option_ids)
            dist = option_distribution(logits, option_ids)
            kl = F.kl_div(dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
            update_metrics(metrics[name], pred, teacher_pred, gold, kl)
            sample_pred[name] = pred
        predictions.append(sample_pred)
        if (idx + 1) % 25 == 0:
            print(f"oracle {idx + 1}/{len(rows)} teacher_correct={teacher_correct}", flush=True)

    if predictions_jsonl:
        pred_out = Path(predictions_jsonl)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "teacher": {"correct": teacher_correct, "accuracy": teacher_correct / max(len(rows), 1)},
        "oracles": {name: finalize_metrics(metrics[name], teacher_correct) for name in metrics},
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_heads = int(language_model.config.num_attention_heads)
    groups = parse_groups(args.groups, num_heads)

    basis_rows = rows[: min(args.basis_samples, len(rows))]
    print(f"collecting head aggregation basis on {len(basis_rows)} samples", flush=True)
    basis_payload = collect_head_bases(basis_rows, processor, model, language_model, device, dtype)
    bases = basis_payload["basis"].to(device=device, dtype=dtype)
    if args.basis_output:
        basis_out = Path(args.basis_output)
        basis_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data": args.data,
                "model_path": args.model_path,
                "basis_samples": len(basis_rows),
                "basis": basis_payload["basis"].cpu(),
                "explained": basis_payload["explained"],
            },
            basis_out,
        )

    print(f"evaluating head aggregation oracles on {len(rows)} samples", flush=True)
    results = evaluate_group_oracles(
        rows,
        processor,
        model,
        language_model,
        device,
        dtype,
        bases,
        groups,
        args.predictions_jsonl,
    )
    explained_at_groups = {
        str(layer_idx): {str(g): basis_payload["explained"][layer_idx][g - 1] for g in groups}
        for layer_idx in range(basis_payload["num_layers"])
    }
    payload = {
        "benchmark": "mmstar",
        "data": args.data,
        "model_path": args.model_path,
        "max_samples": len(rows),
        "basis_samples": len(basis_rows),
        "groups": groups,
        "explained_at_groups": explained_at_groups,
        **results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
