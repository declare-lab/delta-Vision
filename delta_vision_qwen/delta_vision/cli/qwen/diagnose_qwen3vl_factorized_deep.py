#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from delta_vision.data import JsonlDataset
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    qwen3vl_visual_memory_by_layer,
    scatter_batched_positions,
)
from delta_vision.runtime.ops import cross_attention
from delta_vision.cli.qwen.train_qwen3vl_sidecar import qwen_native_visual_kv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose Qwen3-VL factorized attention effect on deep layers.")
    parser.add_argument("--data", default="artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--layers", default="30,31,32,33,34,35")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def parse_layers(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def metric(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.to(device=pred.device).bool()
    pred_v = pred.float()[valid]
    target_v = target.float()[valid]
    diff = pred_v - target_v
    denom = target_v.pow(2).sum().clamp_min(1e-12)
    pred_rms = pred_v.pow(2).mean().sqrt()
    target_rms = target_v.pow(2).mean().sqrt()
    return {
        "nmse": float(diff.pow(2).sum().div(denom).item()),
        "cos": float(F.cosine_similarity(pred_v, target_v, dim=-1, eps=1e-6).mean().item()),
        "pred_rms": float(pred_rms.item()),
        "target_rms": float(target_rms.item()),
        "norm_ratio": float((pred_rms / target_rms.clamp_min(1e-12)).item()),
    }


def head_metric(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    # pred/target: [B, H, T, D]
    valid = mask.to(device=pred.device).bool()
    pred_v = pred.float().permute(0, 2, 1, 3)[valid]
    target_v = target.float().permute(0, 2, 1, 3)[valid]
    pred_v = pred_v.reshape(-1, pred.shape[-1])
    target_v = target_v.reshape(-1, target.shape[-1])
    diff = pred_v - target_v
    denom = target_v.pow(2).sum().clamp_min(1e-12)
    pred_rms = pred_v.pow(2).mean().sqrt()
    target_rms = target_v.pow(2).mean().sqrt()
    return {
        "nmse": float(diff.pow(2).sum().div(denom).item()),
        "cos": float(F.cosine_similarity(pred_v, target_v, dim=-1, eps=1e-6).mean().item()),
        "pred_rms": float(pred_rms.item()),
        "target_rms": float(target_rms.item()),
        "norm_ratio": float((pred_rms / target_rms.clamp_min(1e-12)).item()),
    }


def gather_heads(x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bsz, heads, _, dim = x.shape
    idx = positions.to(device=x.device, dtype=torch.long)[:, None, :, None].expand(bsz, heads, positions.shape[1], dim)
    out = torch.gather(x, dim=2, index=idx)
    return out * mask.to(device=x.device, dtype=out.dtype)[:, None, :, None]


def gather_probs_keys(probs: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
    bsz, heads, q_len, _ = probs.shape
    idx = key_positions.to(device=probs.device, dtype=torch.long)[:, None, None, :].expand(
        bsz, heads, q_len, key_positions.shape[1]
    )
    return torch.gather(probs, dim=-1, index=idx)


def causal_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    seq_len = valid_mask.shape[1]
    causal = torch.ones((seq_len, seq_len), device=valid_mask.device, dtype=torch.bool).tril()
    key_valid = valid_mask.to(dtype=torch.bool).view(valid_mask.shape[0], 1, 1, seq_len)
    return causal.view(1, 1, seq_len, seq_len) & key_valid


def native_qkv(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(hidden)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *language_model.rotary_emb(normed, position_ids))
    k = repeat_kv(k, int(attn.num_key_value_groups))
    v = repeat_kv(v, int(attn.num_key_value_groups))
    return q.contiguous(), k.contiguous(), v.contiguous()


def visual_position_ids(full_position_ids: torch.Tensor, image_positions: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        3,
        image_positions.shape[0],
        image_positions.shape[1],
        device=full_position_ids.device,
        dtype=full_position_ids.dtype,
    )
    valid = image_mask.bool()
    batch_idx = torch.arange(image_positions.shape[0], device=image_positions.device).unsqueeze(1).expand_as(image_positions)
    for dim_idx in range(3):
        source = full_position_ids[dim_idx]
        out[dim_idx][valid] = source[batch_idx[valid], image_positions[valid].long()]
    return out


def average_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: sum(item[key] for item in items) / len(items) for key in items[0]}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    layers = parse_layers(args.layers)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    dataset = JsonlDataset(args.data, max_samples=args.max_samples, decode_images=False)
    rows = [dataset[idx] for idx in range(len(dataset))]
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, rows, "image", "question", "answer", None, device)
    teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden0, full_position_ids, _, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"], inputs["attention_mask"], inputs["mm_token_type_ids"], full_position_ids
    )
    v0_memories = qwen3vl_visual_memory_by_layer(
        hidden0.to(dtype=dtype), image_positions, image_mask, deepstack_visual_embeds, "v0", len(language_model.layers)
    )
    vis_pos_ids = visual_position_ids(full_position_ids, image_positions, image_mask)

    per_layer: dict[str, dict[str, object]] = {}
    for layer_idx in layers:
        full_hidden = teacher.hidden_states[layer_idx].to(dtype=dtype)
        text_hidden = gather_batched_positions(full_hidden, text_positions, text_mask).to(dtype=dtype)
        full_effect_state = scatter_batched_positions(full_hidden, text_positions, text_hidden, text_mask)
        attn = language_model.layers[layer_idx].self_attn

        target_delta = compute_qwen3vl_attention_effect_batched(
            language_model,
            layer_idx,
            full_effect_state,
            text_hidden,
            full_position_ids,
            text_position_ids,
            text_positions,
            full_mask,
            text_mask,
        )

        q_full, k_full, v_full = native_qkv(language_model, layer_idx, full_effect_state, full_position_ids)
        q_text, k_text, v_text = native_qkv(language_model, layer_idx, text_hidden, text_position_ids)
        k_vis_teacher = gather_heads(k_full, image_positions, image_mask)
        v_vis_teacher = gather_heads(v_full, image_positions, image_mask)
        k_text_from_full = gather_heads(k_full, text_positions, text_mask)
        v_text_from_full = gather_heads(v_full, text_positions, text_mask)
        q_text_from_full = gather_heads(q_full, text_positions, text_mask)

        scores_full = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) * float(attn.scaling)
        scores_full = scores_full.masked_fill(~causal_valid_mask(full_mask), torch.finfo(scores_full.dtype).min)
        probs_full = torch.softmax(scores_full, dim=-1)
        query_idx = text_positions.to(device=device, dtype=torch.long)[:, None, :, None].expand(
            probs_full.shape[0], probs_full.shape[1], text_positions.shape[1], probs_full.shape[-1]
        )
        probs_text_queries = torch.gather(probs_full, dim=2, index=query_idx)
        image_probs = gather_probs_keys(probs_text_queries, image_positions) * image_mask[:, None, None, :].to(
            dtype=probs_text_queries.dtype
        )
        text_probs = gather_probs_keys(probs_text_queries, text_positions) * text_mask[:, None, None, :].to(
            dtype=probs_text_queries.dtype
        )
        m_vis = image_probs.sum(dim=-1, keepdim=True).clamp_min(0.0)
        m_text = text_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        avis_teacher = torch.matmul(image_probs / m_vis.clamp_min(1e-12), v_vis_teacher.float())
        atext_joint = torch.matmul(text_probs / m_text, v_text_from_full.float())
        oracle_pre = m_vis * (avis_teacher - atext_joint)
        oracle_delta = attn.o_proj(
            oracle_pre.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1).to(dtype=dtype)
        )

        native_v0 = qwen_native_visual_kv(
            language_model,
            layer_idx,
            v0_memories[layer_idx].to(dtype=dtype),
            vis_pos_ids,
            ~image_mask,
        )
        avis_v0 = cross_attention(
            q_text.transpose(1, 2).contiguous(),
            native_v0.key,
            native_v0.value,
            native_v0.padding_mask,
        )
        num_heads = q_text.shape[1]
        head_dim = q_text.shape[-1]
        avis_v0_heads = avis_v0.view(avis_v0.shape[0], avis_v0.shape[1], num_heads, head_dim).transpose(1, 2)
        v0_with_teacher_mass_pre = m_vis * (avis_v0_heads.float() - atext_joint)
        v0_with_teacher_mass_delta = attn.o_proj(
            v0_with_teacher_mass_pre.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1).to(dtype=dtype)
        )

        text_pre = F.scaled_dot_product_attention(
            q_text,
            k_text,
            v_text,
            attn_mask=causal_valid_mask(text_mask),
            is_causal=False,
            scale=float(attn.scaling),
        )
        text_direct = attn.o_proj(text_pre.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1))

        per_layer[str(layer_idx)] = {
            "target_rms": float(target_delta.float()[text_mask].pow(2).mean().sqrt().item()),
            "mass_mean": float(m_vis[text_mask[:, None, :, None].expand_as(m_vis)].float().mean().item()),
            "mass_max": float(m_vis[text_mask[:, None, :, None].expand_as(m_vis)].float().max().item()),
            "oracle_formula_vs_target": metric(oracle_delta, target_delta, text_mask),
            "v0_visual_teacher_mass_vs_target": metric(v0_with_teacher_mass_delta, target_delta, text_mask),
            "v0_visual_head_vs_teacher_visual_head": head_metric(avis_v0_heads, avis_teacher, text_mask),
            "q_text_direct_vs_teacher_full_q": head_metric(q_text, q_text_from_full, text_mask),
            "k_text_direct_vs_teacher_full_k": head_metric(k_text, k_text_from_full, text_mask),
            "text_attention_direct_rms": float(text_direct.float()[text_mask].pow(2).mean().sqrt().item()),
        }
        print(f"layer {layer_idx}: {json.dumps(per_layer[str(layer_idx)], ensure_ascii=False)}", flush=True)

    aggregate = {}
    keys = [
        "oracle_formula_vs_target",
        "v0_visual_teacher_mass_vs_target",
        "v0_visual_head_vs_teacher_visual_head",
        "q_text_direct_vs_teacher_full_q",
        "k_text_direct_vs_teacher_full_k",
    ]
    for key in keys:
        aggregate[key] = average_dicts([per_layer[str(layer)][key] for layer in layers])  # type: ignore[index]
    payload = {
        "data": args.data,
        "model_path": args.model_path,
        "samples": len(rows),
        "layers": layers,
        "per_layer": per_layer,
        "aggregate": aggregate,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "aggregate": aggregate}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
