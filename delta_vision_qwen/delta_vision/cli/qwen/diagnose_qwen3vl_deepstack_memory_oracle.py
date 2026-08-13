#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from delta_vision.data import JsonlDataset
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.evaluation.metrics import masked_kl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_sample_inputs,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL visual-memory attention oracle diagnostic.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if any(x < 0 or x >= num_layers for x in layers):
        raise ValueError(f"layers must be in [0, {num_layers - 1}]")
    return layers


def masked_nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    num = ((pred.float() - target.float()).pow(2) * valid).sum()
    den = (target.float().pow(2) * valid).sum().clamp_min(1e-6)
    return float((num / den).item())


def masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return 0.0
    return float(F.cosine_similarity(pred.float()[valid], target.float()[valid], dim=-1, eps=1e-6).mean().item())


def masked_rms(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.to(device=tensor.device).bool()
    if not bool(valid.any()):
        return 0.0
    return float(tensor.float()[valid].pow(2).mean().sqrt().item())


def build_full_from_text_and_memory(
    reference_full: torch.Tensor,
    text_positions: torch.Tensor,
    text_hidden: torch.Tensor,
    text_mask: torch.Tensor,
    image_positions: torch.Tensor,
    image_memory: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    full = scatter_batched_positions(reference_full, text_positions, text_hidden, text_mask)
    full = scatter_batched_positions(full, image_positions, image_memory, image_mask)
    return full


def memory_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    reference_full: torch.Tensor,
    text_hidden: torch.Tensor,
    image_memory: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    full_mask: torch.Tensor,
    text_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    full = build_full_from_text_and_memory(
        reference_full,
        text_positions,
        text_hidden,
        text_mask,
        image_positions,
        image_memory,
        image_mask,
    )
    return compute_qwen3vl_attention_effect_batched(
        language_model,
        layer_idx,
        full,
        text_hidden,
        full_position_ids,
        text_position_ids,
        text_positions,
        full_mask,
        text_mask,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    num_layers = len(language_model.layers)
    layers = parse_layers(args.layers, num_layers)
    dataset = JsonlDataset(args.data, max_samples=args.max_samples)

    layer_sums = {
        layer: {
            "v0_nmse": 0.0,
            "v0_cos": 0.0,
            "vdeep_nmse": 0.0,
            "vdeep_cos": 0.0,
            "vcum_nmse": 0.0,
            "vcum_cos": 0.0,
            "vteacher_nmse": 0.0,
            "vteacher_cos": 0.0,
            "target_rms": 0.0,
            "v0_rms": 0.0,
            "vdeep_rms": 0.0,
            "vcum_rms": 0.0,
            "vteacher_rms": 0.0,
            "count": 0,
        }
        for layer in layers
    }
    rollout_sums = {
        "v0_logit_kl": 0.0,
        "vdeep_logit_kl": 0.0,
        "vcum_logit_kl": 0.0,
        "vteacher_logit_kl": 0.0,
        "full_target_logit_kl": 0.0,
        "v0_final_hidden_nmse": 0.0,
        "vdeep_final_hidden_nmse": 0.0,
        "vcum_final_hidden_nmse": 0.0,
        "vteacher_final_hidden_nmse": 0.0,
        "full_target_final_hidden_nmse": 0.0,
        "count": 0,
    }
    samples = []

    for sample_idx, row in enumerate(dataset):
        inputs, text_ids, answer_mask, image_path = prepare_qwen3vl_sample_inputs(
            processor,
            row,
            "image",
            "question",
            "answer",
            None,
            device,
        )
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = (
            get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                full_position_ids,
            )
        )
        teacher_states = [state.detach() for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach()
            for state in teacher_states
        ]
        teacher_logits = gather_batched_positions(teacher.logits, text_positions, text_mask).detach()
        v0 = gather_batched_positions(hidden0, image_positions, image_mask).to(dtype=dtype)
        deep_sum = torch.stack([x.to(device=device, dtype=dtype) for x in deepstack_visual_embeds], dim=0).sum(dim=0)
        vdeep = v0 + deep_sum.unsqueeze(0)
        vcum_by_layer = []
        running_deep = torch.zeros_like(v0)
        for layer_idx in range(num_layers):
            vcum_by_layer.append(v0 + running_deep)
            if layer_idx < len(deepstack_visual_embeds):
                running_deep = running_deep + deepstack_visual_embeds[layer_idx].to(device=device, dtype=dtype).unsqueeze(0)
        vteacher_by_layer = [
            gather_batched_positions(state, image_positions, image_mask).to(dtype=dtype)
            for state in teacher_states[:num_layers]
        ]

        sample_record = {"sample": sample_idx, "image": image_path, "layers": {}}
        for layer_idx in layers:
            target = compute_qwen3vl_attention_effect_batched(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                full_position_ids,
                text_position_ids,
                text_positions,
                full_mask,
                text_mask,
            )
            v0_delta = memory_attention_effect(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                v0,
                full_position_ids,
                text_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            vdeep_delta = memory_attention_effect(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                vdeep,
                full_position_ids,
                text_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            vcum_delta = memory_attention_effect(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                vcum_by_layer[layer_idx],
                full_position_ids,
                text_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            vteacher_delta = memory_attention_effect(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                vteacher_by_layer[layer_idx],
                full_position_ids,
                text_position_ids,
                text_positions,
                image_positions,
                full_mask,
                text_mask,
                image_mask,
            )
            metrics = {
                "v0_nmse": masked_nmse(v0_delta, target, text_mask),
                "v0_cos": masked_cos(v0_delta, target, text_mask),
                "vdeep_nmse": masked_nmse(vdeep_delta, target, text_mask),
                "vdeep_cos": masked_cos(vdeep_delta, target, text_mask),
                "vcum_nmse": masked_nmse(vcum_delta, target, text_mask),
                "vcum_cos": masked_cos(vcum_delta, target, text_mask),
                "vteacher_nmse": masked_nmse(vteacher_delta, target, text_mask),
                "vteacher_cos": masked_cos(vteacher_delta, target, text_mask),
                "target_rms": masked_rms(target, text_mask),
                "v0_rms": masked_rms(v0_delta, text_mask),
                "vdeep_rms": masked_rms(vdeep_delta, text_mask),
                "vcum_rms": masked_rms(vcum_delta, text_mask),
                "vteacher_rms": masked_rms(vteacher_delta, text_mask),
            }
            sample_record["layers"][str(layer_idx)] = metrics
            acc = layer_sums[layer_idx]
            for key, value in metrics.items():
                acc[key] += float(value)
            acc["count"] += 1

        rollout_metrics = {}
        for name, memory_or_list in (("v0", v0), ("vdeep", vdeep), ("vcum", vcum_by_layer), ("vteacher", vteacher_by_layer)):
            h = teacher_text_states[0].to(dtype=dtype)
            for layer_idx in range(num_layers):
                memory = memory_or_list[layer_idx] if isinstance(memory_or_list, list) else memory_or_list
                delta = memory_attention_effect(
                    language_model,
                    layer_idx,
                    teacher_states[layer_idx].to(dtype=dtype),
                    h,
                    memory,
                    full_position_ids,
                    text_position_ids,
                    text_positions,
                    image_positions,
                    full_mask,
                    text_mask,
                    image_mask,
                )
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    padding_mask=~text_mask,
                )
            final_h = language_model.norm(h)
            student_logits = model.lm_head(final_h)
            rollout_metrics[f"{name}_logit_kl"] = float(masked_kl(student_logits, teacher_logits, answer_mask, 2.0).item())
            rollout_metrics[f"{name}_final_hidden_nmse"] = masked_nmse(final_h, teacher_text_states[-1].to(dtype=dtype), text_mask)
        h = teacher_text_states[0].to(dtype=dtype)
        for layer_idx in range(num_layers):
            target = compute_qwen3vl_attention_effect_batched(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                full_position_ids,
                text_position_ids,
                text_positions,
                full_mask,
                text_mask,
            )
            h = run_qwen3vl_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                target,
                padding_mask=~text_mask,
            )
        final_h = language_model.norm(h)
        student_logits = model.lm_head(final_h)
        rollout_metrics["full_target_logit_kl"] = float(masked_kl(student_logits, teacher_logits, answer_mask, 2.0).item())
        rollout_metrics["full_target_final_hidden_nmse"] = masked_nmse(final_h, teacher_text_states[-1].to(dtype=dtype), text_mask)
        sample_record["rollout"] = rollout_metrics
        for key, value in rollout_metrics.items():
            rollout_sums[key] += float(value)
        rollout_sums["count"] += 1
        samples.append(sample_record)
        print(
            f"sample={sample_idx} "
            f"v0_kl={rollout_metrics['v0_logit_kl']:.6f} "
            f"vdeep_kl={rollout_metrics['vdeep_logit_kl']:.6f} "
            f"vcum_kl={rollout_metrics['vcum_logit_kl']:.6f} "
            f"vteacher_kl={rollout_metrics['vteacher_logit_kl']:.6f} "
            f"full_target_kl={rollout_metrics['full_target_logit_kl']:.6f} "
            f"v0_h={rollout_metrics['v0_final_hidden_nmse']:.6f} "
            f"vdeep_h={rollout_metrics['vdeep_final_hidden_nmse']:.6f} "
            f"vcum_h={rollout_metrics['vcum_final_hidden_nmse']:.6f} "
            f"vteacher_h={rollout_metrics['vteacher_final_hidden_nmse']:.6f} "
            f"full_target_h={rollout_metrics['full_target_final_hidden_nmse']:.6f}",
            flush=True,
        )

    layer_mean = {}
    for layer_idx, values in layer_sums.items():
        count = max(int(values["count"]), 1)
        layer_mean[str(layer_idx)] = {
            key: float(value) / count for key, value in values.items() if key != "count"
        }
    rollout_count = max(int(rollout_sums["count"]), 1)
    rollout_mean = {
        key: float(value) / rollout_count for key, value in rollout_sums.items() if key != "count"
    }
    payload = {
        "model_path": args.model_path,
        "data": args.data,
        "max_samples": args.max_samples,
        "layers": layers,
        "layer_mean": layer_mean,
        "rollout_mean": rollout_mean,
        "samples": samples,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"layer_mean": layer_mean, "rollout_mean": rollout_mean}, indent=2), flush=True)


if __name__ == "__main__":
    main()
