#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.cli.qwen.eval_qwen3vl_operator_oracle import (
    build_prompt,
    constant_operator_delta,
    film_operator_delta,
    linear_operator_delta,
    lowrank_linear_operator_delta,
    masked_metric,
    parse_mode,
    token_split_mask,
)
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    run_qwen3vl_layer_text_with_attention_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose Qwen3-VL same-image unseen-query operator oracle.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--start-group", type=int, default=0)
    parser.add_argument("--fit-queries", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--modes", default="no_visual,full_target,linear_rank_32,linear_rank_64,linear")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def grouped_examples(
    rows: list[dict[str, Any]],
    start_group: int,
    max_groups: int,
    fit_queries: int,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        image = str(row.get("image") or row.get("image_path") or "")
        if image:
            groups[image].append(row)
    need = int(fit_queries) + 1
    examples = [(items[:fit_queries], items[fit_queries]) for _image, items in groups.items() if len(items) >= need]
    return examples[start_group : start_group + max_groups]


@torch.inference_mode()
def trace_sample(
    processor: Any,
    model: Any,
    language_model: Any,
    row: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    _hidden0, full_position_ids, _visual_pos_masks, _deepstack_visual_embeds = build_qwen3vl_initial_context(
        model,
        inputs,
    )
    text_positions, _image_positions, text_position_ids, text_mask, _image_mask, full_mask = (
        get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
    )
    teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
    teacher_text_states = [
        gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype)
        for state in teacher_states
    ]
    target_deltas = [
        compute_qwen3vl_attention_effect_batched(
            language_model,
            layer_idx,
            teacher_states[layer_idx],
            teacher_text_states[layer_idx],
            full_position_ids,
            text_position_ids,
            text_positions,
            full_mask,
            text_mask,
        ).detach()
        for layer_idx in range(len(language_model.layers))
    ]
    last_text_idx = int(text_mask[0].sum().item()) - 1
    teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())].detach()
    return {
        "teacher_text_states": teacher_text_states,
        "target_deltas": target_deltas,
        "text_position_ids": text_position_ids,
        "text_mask": text_mask,
        "teacher_logits": teacher_logits,
    }


def update(metrics: dict[str, dict[str, float | int]], name: str, logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    row = metrics[name]
    student_logp = F.log_softmax(logits.float(), dim=-1)
    teacher_p = F.softmax(teacher_logits.float(), dim=-1)
    row["kl"] = float(row["kl"]) + float(F.kl_div(student_logp, teacher_p, reduction="sum").item())
    row["top1_agree"] = int(row["top1_agree"]) + int(int(logits.argmax().item()) == int(teacher_logits.argmax().item()))
    row["count"] = int(row["count"]) + 1


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, None)
    examples = grouped_examples(rows, args.start_group, args.max_groups, args.fit_queries)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    parsed_modes = {mode: parse_mode(mode) for mode in modes}

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    metrics = {mode: {"kl": 0.0, "top1_agree": 0, "count": 0} for mode in modes}
    effect_metrics = {
        mode: {"cos": 0.0, "nmse": 0.0, "norm_ratio": 0.0, "count": 0}
        for mode in modes
        if parsed_modes[mode][0] not in {"no_visual", "full_target"}
    }

    for pair_idx, (fit_rows, eval_row) in enumerate(examples):
        fit_traces = [trace_sample(processor, model, language_model, fit_row, dtype, device) for fit_row in fit_rows]
        eval_trace = trace_sample(processor, model, language_model, eval_row, dtype, device)
        eval_mask = token_split_mask(eval_trace["text_mask"][0], "all")
        fit_text_states = [
            torch.cat([trace["teacher_text_states"][layer_idx][0] for trace in fit_traces], dim=0)
            for layer_idx in range(num_layers)
        ]
        fit_target_deltas = [
            torch.cat([trace["target_deltas"][layer_idx][0] for trace in fit_traces], dim=0)
            for layer_idx in range(num_layers)
        ]
        fit_masks = [
            torch.cat([token_split_mask(trace["text_mask"][0], "all") for trace in fit_traces], dim=0)
            for _layer_idx in range(num_layers)
        ]
        last_eval_idx = int(eval_trace["text_mask"][0].sum().item()) - 1

        for mode in modes:
            mode_kind, mode_rank = parsed_modes[mode]
            h = eval_trace["teacher_text_states"][0].masked_fill((~eval_trace["text_mask"]).unsqueeze(-1), 0.0)
            for layer_idx in range(num_layers):
                target = eval_trace["target_deltas"][layer_idx]
                if mode_kind == "no_visual":
                    delta = None
                elif mode_kind == "full_target":
                    delta = target
                elif mode_kind == "constant":
                    delta = constant_operator_delta(
                        fit_text_states[layer_idx],
                        fit_target_deltas[layer_idx],
                        h[0],
                        fit_masks[layer_idx],
                        eval_mask,
                    ).unsqueeze(0)
                elif mode_kind == "film":
                    delta = film_operator_delta(
                        fit_text_states[layer_idx],
                        fit_target_deltas[layer_idx],
                        h[0],
                        fit_masks[layer_idx],
                        eval_mask,
                        args.ridge,
                    ).unsqueeze(0)
                elif mode_kind == "linear":
                    delta = linear_operator_delta(
                        fit_text_states[layer_idx],
                        fit_target_deltas[layer_idx],
                        h[0],
                        fit_masks[layer_idx],
                        eval_mask,
                        args.ridge,
                    ).unsqueeze(0)
                elif mode_kind == "linear_rank":
                    if mode_rank is None:
                        raise AssertionError(mode)
                    delta = lowrank_linear_operator_delta(
                        fit_text_states[layer_idx],
                        fit_target_deltas[layer_idx],
                        h[0],
                        fit_masks[layer_idx],
                        eval_mask,
                        args.ridge,
                        mode_rank,
                    ).unsqueeze(0)
                else:
                    raise AssertionError(mode)
                if mode in effect_metrics:
                    metric = masked_metric(delta[0], target[0], eval_mask)
                    for key in ("cos", "nmse", "norm_ratio"):
                        effect_metrics[mode][key] += metric[key]
                    effect_metrics[mode]["count"] += 1
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    eval_trace["text_position_ids"],
                    delta,
                    padding_mask=~eval_trace["text_mask"],
                )
            logits = model.lm_head(language_model.norm(h))[0, last_eval_idx]
            update(metrics, mode, logits, eval_trace["teacher_logits"])
        if (pair_idx + 1) % 5 == 0 or pair_idx + 1 == len(examples):
            print(f"same-image operator {pair_idx + 1}/{len(examples)}", flush=True)

    results = {
        "data": args.data,
        "model_path": args.model_path,
        "num_pairs": len(examples),
        "ridge": args.ridge,
        "fit_queries": args.fit_queries,
        "modes": modes,
        "metrics": {
            mode: {
                "output_kl": row["kl"] / max(int(row["count"]), 1),
                "top1_agreement": row["top1_agree"] / max(int(row["count"]), 1),
                "count": int(row["count"]),
            }
            for mode, row in metrics.items()
        },
        "effect_metrics": {
            mode: {
                "effect_cos": row["cos"] / max(int(row["count"]), 1),
                "effect_nmse": row["nmse"] / max(int(row["count"]), 1),
                "effect_norm_ratio": row["norm_ratio"] / max(int(row["count"]), 1),
            }
            for mode, row in effect_metrics.items()
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
