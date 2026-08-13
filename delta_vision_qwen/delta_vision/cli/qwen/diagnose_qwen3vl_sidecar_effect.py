#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt,
    load_sidecar,
    qwen3vl_visual_position_ids,
    sidecar_rope_visual_kv,
)
from delta_vision.cli.qwen.train_qwen3vl_sidecar import (
    qwen_native_sidecar_query,
    qwen_native_visual_kv,
    qwen_trainable_native_sidecar_query,
    qwen_trainable_native_visual_kv,
)
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_text_attention_heads,
    qwen3vl_text_attention_output,
    run_qwen3vl_layer_text_from_attention_output,
    run_qwen3vl_layer_text_with_attention_delta,
)
from delta_vision.runtime.qwen_analytic_sidecar import analytic_attention_delta, qwen3vl_visual_memories_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose Qwen3-VL learned Sidecar local effect prediction.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), default="mmstar")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer ids")
    parser.add_argument(
        "--state-mode",
        choices=("teacher", "rollout"),
        default="teacher",
        help="teacher diagnoses local prediction on H_l^T; rollout diagnoses prediction on the actual student hidden trajectory.",
    )
    parser.add_argument(
        "--target-memory",
        choices=("visible_memory", "teacher_visual"),
        default="visible_memory",
        help=(
            "visible_memory targets the effect produced by the selected sidecar-visible memory mode; "
            "teacher_visual targets the original joint Qwen visual-token states, matching older teacher_visual training."
        ),
    )
    parser.add_argument("--visual-memory-mode", choices=("v0", "vdeep", "vcum", "vprefix"), default="vprefix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers = [int(x) for x in spec.split(",") if x.strip()]
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"layer out of range: {layer}")
    return layers


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return {"nmse": 0.0, "cos": 0.0, "pred_rms": 0.0, "target_rms": 0.0, "norm_ratio": 0.0}
    pred_valid = pred.float()[valid]
    target_valid = target.float()[valid]
    nmse = (pred_valid - target_valid).pow(2).sum() / target_valid.pow(2).sum().clamp_min(1e-6)
    cos = F.cosine_similarity(pred_valid, target_valid, dim=-1, eps=1e-6).mean()
    pred_rms = pred_valid.pow(2).mean().sqrt()
    target_rms = target_valid.pow(2).mean().sqrt()
    return {
        "nmse": float(nmse.item()),
        "cos": float(cos.item()),
        "pred_rms": float(pred_rms.item()),
        "target_rms": float(target_rms.item()),
        "norm_ratio": float((pred_rms / target_rms.clamp_min(1e-6)).item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, None)[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    sidecar_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        rank=512,
        sidecar_dim=1024,
        num_heads=8,
        state_tokens=0,
        reader_mlp_ratio=4.0,
        reader_activation="gelu",
        layer_adapter_rank=128,
        shared_basis=True,
        output_mode="residual",
        corrector_layers="",
        corrector_dim=0,
        block_corrector_groups="",
        block_corrector_dim=0,
        use_rope=False,
        sidecar_query_source="sidecar",
        sidecar_visual_kv_source="sidecar",
        factorized_mass_mode="learned",
        fixed_visual_mass=0.12,
        visual_memory_mode=args.visual_memory_mode,
    )
    sidecar = load_sidecar(sidecar_args, language_model, device, dtype)
    args.visual_memory_mode = sidecar_args.visual_memory_mode
    layers = parse_layers(args.layers, len(language_model.layers))

    totals = {
        layer: {"nmse": 0.0, "cos": 0.0, "pred_rms": 0.0, "target_rms": 0.0, "norm_ratio": 0.0, "count": 0}
        for layer in layers
    }
    samples = []
    for sample_idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(
                text=build_prompt(processor, row, args.benchmark),
                images=image.convert("RGB"),
                return_tensors="pt",
            )
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        hidden0, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
        text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            position_ids,
        )
        teacher = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        teacher_text_states = [
            gather_batched_positions(state, text_pos, text_mask).detach().to(dtype=dtype)
            for state in teacher.hidden_states
        ]
        visual_memories = qwen3vl_visual_memories_for_mode(
            language_model,
            hidden0.to(dtype=dtype),
            position_ids,
            inputs["attention_mask"],
            image_pos,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
            args.visual_memory_mode,
        )
        shared_visual_kv = None
        shared_text_pos_emb = None
        visual_position_ids = qwen3vl_visual_position_ids(position_ids, image_pos, image_mask)
        sidecar_query_source = getattr(sidecar_args, "sidecar_query_source", "sidecar")
        sidecar_visual_kv_source = getattr(sidecar_args, "sidecar_visual_kv_source", "sidecar")
        if args.visual_memory_mode in {"v0", "vdeep"} and sidecar_visual_kv_source == "sidecar":
            shared_visual_kv, shared_text_pos_emb = sidecar_rope_visual_kv(
                sidecar,
                language_model,
                visual_memories[0],
                position_ids,
                text_position_ids,
                image_pos,
                image_mask,
                dtype,
            )
        elif sidecar_visual_kv_source == "qwen_first_layer":
            shared_visual_kv = qwen_native_visual_kv(
                language_model,
                0,
                visual_memories[0].to(dtype=dtype),
                visual_position_ids,
                ~image_mask,
            )
            shared_text_pos_emb = None
        sidecar_state = (
            sidecar.initial_state(visual_memories[0].to(dtype=dtype), ~image_mask)
            if sidecar.state_tokens > 0
            else None
        )
        sample_metrics = {"index": row.get("index", sample_idx), "layers": {}}
        h_rollout = teacher_text_states[0].to(dtype=dtype).masked_fill((~text_mask).unsqueeze(-1), 0.0)
        text_padding_mask = ~text_mask
        for layer_idx in range(len(language_model.layers)):
            h = teacher_text_states[layer_idx] if args.state_mode == "teacher" else h_rollout
            layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
            visual_kv = shared_visual_kv
            text_pos_emb = shared_text_pos_emb
            if visual_kv is None:
                if sidecar_visual_kv_source == "qwen_native":
                    visual_kv = qwen_native_visual_kv(
                        language_model,
                        layer_idx,
                        visual_memories[layer_idx].to(dtype=dtype),
                        visual_position_ids,
                        ~image_mask,
                    )
                    text_pos_emb = None
                elif sidecar_visual_kv_source == "qwen_trainable_native":
                    visual_kv = qwen_trainable_native_visual_kv(
                        sidecar,
                        language_model,
                        layer_idx,
                        visual_memories[layer_idx].to(dtype=dtype),
                        visual_position_ids,
                        ~image_mask,
                    )
                    text_pos_emb = None
                elif sidecar_visual_kv_source == "qwen_first_layer":
                    visual_kv = shared_visual_kv
                    text_pos_emb = None
                else:
                    visual_kv, text_pos_emb = sidecar_rope_visual_kv(
                        sidecar,
                        language_model,
                        visual_memories[layer_idx],
                        position_ids,
                        text_position_ids,
                        image_pos,
                        image_mask,
                        dtype,
                    )
            query_override = None
            if sidecar_query_source == "qwen_native":
                query_override = qwen_native_sidecar_query(language_model, layer_idx, h, text_position_ids)
            elif sidecar_query_source == "qwen_trainable_native":
                query_override = qwen_trainable_native_sidecar_query(
                    sidecar,
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                )
            text_attention = None
            if sidecar.output_mode.startswith("factorized"):
                text_attention = qwen3vl_text_attention_output(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    padding_mask=~text_mask,
                )
            text_attention_heads = None
            if sidecar.output_mode in {"factorized_native_head_o", "factorized_native_head_o_residual"}:
                text_attention_heads = qwen3vl_text_attention_heads(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    padding_mask=~text_mask,
                )
            if sidecar.state_tokens > 0:
                pred_delta, sidecar_state = sidecar(
                    h,
                    visual_memories[layer_idx].to(dtype=dtype),
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=visual_kv,
                    text_attention=text_attention,
                    output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                    text_attention_heads=text_attention_heads,
                    position_embeddings=text_pos_emb,
                    query_states=query_override,
                    return_state=True,
                )
            else:
                pred_delta = sidecar(
                    h,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    text_attention=text_attention,
                    output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                    text_attention_heads=text_attention_heads,
                    position_embeddings=text_pos_emb,
                    query_states=query_override,
                )
            if layer_idx in layers:
                if args.target_memory == "visible_memory":
                    target_delta = analytic_attention_delta(
                        language_model,
                        layer_idx,
                        hidden0.to(dtype=dtype),
                        h,
                        visual_memories[layer_idx].to(dtype=dtype),
                        position_ids,
                        text_position_ids,
                        text_pos,
                        image_pos,
                        full_mask,
                        text_mask,
                        image_mask,
                    )
                else:
                    if args.state_mode == "teacher":
                        full_effect_state = teacher.hidden_states[layer_idx].detach().to(dtype=dtype)
                        target_text_state = teacher_text_states[layer_idx].to(dtype=dtype)
                    else:
                        full_effect_state = teacher.hidden_states[layer_idx].detach().to(dtype=dtype).clone()
                        for batch_idx in range(full_effect_state.shape[0]):
                            full_effect_state[batch_idx, text_pos[batch_idx, text_mask[batch_idx]]] = h[
                                batch_idx, text_mask[batch_idx]
                            ].to(dtype=full_effect_state.dtype)
                        target_text_state = h
                    target_delta = compute_qwen3vl_attention_effect_batched(
                        language_model,
                        layer_idx,
                        full_effect_state,
                        target_text_state,
                        position_ids,
                        text_position_ids,
                        text_pos,
                        full_mask,
                        text_mask,
                    ).detach()
                metric = masked_metrics(pred_delta, target_delta, text_mask)
                target_h = teacher_text_states[layer_idx].to(dtype=dtype)
                hidden_metric = masked_metrics(h, target_h, text_mask)
                metric["hidden_nmse_to_teacher"] = hidden_metric["nmse"]
                metric["hidden_cos_to_teacher"] = hidden_metric["cos"]
                sample_metrics["layers"][str(layer_idx)] = metric
                total = totals[layer_idx]
                for key in ("nmse", "cos", "pred_rms", "target_rms", "norm_ratio"):
                    total[key] += metric[key]
                total["count"] += 1
            if args.state_mode == "rollout":
                if sidecar.output_mode.startswith("factorized"):
                    assert text_attention is not None
                    h_rollout = run_qwen3vl_layer_text_from_attention_output(
                        language_model,
                        layer_idx,
                        h_rollout,
                        text_attention,
                        pred_delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                    )
                else:
                    h_rollout = run_qwen3vl_layer_text_with_attention_delta(
                        language_model,
                        layer_idx,
                        h_rollout,
                        text_position_ids,
                        pred_delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                        padding_mask=text_padding_mask,
                    )
        samples.append(sample_metrics)
        print(f"diagnosed {sample_idx + 1}/{len(rows)}", flush=True)

    layer_results = []
    for layer_idx in layers:
        total = totals[layer_idx]
        count = max(int(total["count"]), 1)
        layer_results.append(
            {
                "layer": layer_idx,
                **{key: float(total[key]) / count for key in ("nmse", "cos", "pred_rms", "target_rms", "norm_ratio")},
            }
        )
    aggregate = {
        key: sum(item[key] for item in layer_results) / max(len(layer_results), 1)
        for key in ("nmse", "cos", "pred_rms", "target_rms", "norm_ratio")
    }
    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "visual_memory_mode": args.visual_memory_mode,
        "state_mode": args.state_mode,
        "target_memory": args.target_memory,
        "max_samples": args.max_samples,
        "layers": layers,
        "aggregate": aggregate,
        "layer_results": layer_results,
        "samples": samples,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("checkpoint", "visual_memory_mode", "aggregate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
