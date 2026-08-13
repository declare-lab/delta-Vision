#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from delta_vision.models.llava import (
    compute_llama_attention_effect,
    compute_llama_factorized_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_text_and_image_positions,
    read_jsonl,
)
from delta_vision.runtime.rollout import prepare_sample_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate factorized attention-effect equivalence on MMStar.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output-json", default="artifacts/eval/oracle/mmstar_factorized_attention_oracle_100.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def _flat_valid(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1, x.shape[-1])


def _metric_pack(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_f = _flat_valid(pred)
    target_f = _flat_valid(target)
    diff = pred_f - target_f
    diff_norm = torch.linalg.vector_norm(diff)
    target_norm = torch.linalg.vector_norm(target_f).clamp_min(1e-12)
    pred_norm = torch.linalg.vector_norm(pred_f).clamp_min(1e-12)
    cosine = torch.sum(pred_f * target_f) / (pred_norm * target_norm)
    return {
        "rel_l2": float((diff_norm / target_norm).item()),
        "nmse": float((diff.pow(2).mean() / target_f.pow(2).mean().clamp_min(1e-12)).item()),
        "cosine": float(cosine.item()),
        "max_abs": float(diff.abs().max().item()),
        "target_rms": float(target_f.pow(2).mean().sqrt().item()),
        "pred_rms": float(pred_f.pow(2).mean().sqrt().item()),
    }


def _zero_metric_pack(x: torch.Tensor) -> dict[str, float]:
    x_f = _flat_valid(x)
    return {
        "rms": float(x_f.pow(2).mean().sqrt().item()),
        "mean_abs": float(x_f.abs().mean().item()),
        "max_abs": float(x_f.abs().max().item()),
    }


def _accumulate_metric(stats: dict[str, dict[str, float]], prefix: str, metrics: dict[str, float]) -> None:
    bucket = stats.setdefault(prefix, {"count": 0.0})
    bucket["count"] += 1.0
    for key, value in metrics.items():
        bucket[key] = bucket.get(key, 0.0) + float(value)


def _finalize(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, values in stats.items():
        count = max(values.get("count", 0.0), 1.0)
        out[key] = {metric: total / count for metric, total in values.items() if metric != "count"}
        out[key]["count"] = count
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, None)[args.start_index :]
    rows = rows[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    language_model = get_language_model(model)
    num_layers = len(get_lm_layers(language_model))
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    per_layer: dict[int, dict[str, dict[str, float]]] = {idx: {} for idx in range(num_layers)}
    global_stats: dict[str, dict[str, float]] = {}
    mass_sums = {idx: 0.0 for idx in range(num_layers)}
    mass_counts = {idx: 0 for idx in range(num_layers)}

    for sample_idx, row in enumerate(rows):
        inputs, _, _, image_path = prepare_sample_inputs(
            processor,
            row,
            "image",
            "question",
            "answer",
            None,
            image_token_id,
            device,
        )
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden_states = tuple(x.detach() for x in outputs.hidden_states[:-1])
        merged_len = hidden_states[0].shape[1]
        text_positions, image_positions, _ = get_text_and_image_positions(inputs["input_ids"], merged_len, image_token_id)
        text_positions = text_positions.to(device)
        image_positions = image_positions.to(device)

        for layer_idx in range(num_layers):
            factor = compute_llama_factorized_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx].to(dtype=dtype),
                text_positions,
                image_positions,
            )
            direct_sdpa = compute_llama_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx].to(dtype=dtype),
                text_positions,
            )
            comparisons = {
                "factorized_vs_direct_explicit": _metric_pack(factor["factorized_delta"], factor["direct_delta"]),
                "direct_explicit_vs_sdpa": _metric_pack(factor["direct_delta"], direct_sdpa),
                "factorized_vs_sdpa": _metric_pack(factor["factorized_delta"], direct_sdpa),
                "text_renorm_vs_text_only_zero": _zero_metric_pack(factor["text_equiv_delta"]),
            }
            for name, metrics in comparisons.items():
                _accumulate_metric(per_layer[layer_idx], name, metrics)
                _accumulate_metric(global_stats, name, metrics)
            mass = factor["visual_mass"].detach().float()
            mass_sums[layer_idx] += float(mass.mean().item())
            mass_counts[layer_idx] += 1

        if (sample_idx + 1) % 5 == 0 or sample_idx + 1 == len(rows):
            print(f"processed {sample_idx + 1}/{len(rows)} image={image_path}", flush=True)

    finalized_layers = {
        str(layer_idx): {
            **_finalize(stats),
            "visual_mass_mean": mass_sums[layer_idx] / max(mass_counts[layer_idx], 1),
        }
        for layer_idx, stats in per_layer.items()
    }
    result = {
        "data": args.data,
        "model_path": args.model_path,
        "num_samples": len(rows),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "global": _finalize(global_stats),
        "per_layer": finalized_layers,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["global"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
