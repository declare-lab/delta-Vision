#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser("Diagnose Qwen3-VL layer-0 native visual alignment.")
    parser.add_argument("--data", default="artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def metric(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        while mask.ndim < pred_f.ndim:
            mask = mask.unsqueeze(-1)
        pred_f = pred_f[mask.expand_as(pred_f)].view(-1, pred.shape[-1])
        target_f = target_f[mask.expand_as(target_f)].view(-1, target.shape[-1])
    diff = pred_f - target_f
    denom = target_f.pow(2).mean().clamp_min(1e-12)
    return {
        "nmse": float(diff.pow(2).mean().div(denom).item()),
        "rel_l2": float(diff.pow(2).sum().sqrt().div(target_f.pow(2).sum().sqrt().clamp_min(1e-12)).item()),
        "cos": float(F.cosine_similarity(pred_f.reshape(-1, pred.shape[-1]), target_f.reshape(-1, target.shape[-1]), dim=-1).mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "pred_rms": float(pred_f.pow(2).mean().sqrt().item()),
        "target_rms": float(target_f.pow(2).mean().sqrt().item()),
    }


def gather_heads(x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D], positions/mask: [B, T] -> [B, H, T, D]
    bsz, heads, _, dim = x.shape
    gather_idx = positions.to(device=x.device, dtype=torch.long)[:, None, :, None].expand(bsz, heads, positions.shape[1], dim)
    out = torch.gather(x, dim=2, index=gather_idx)
    return out * mask.to(device=x.device, dtype=out.dtype)[:, None, :, None]


def gather_mask_from_full(mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.gather(mask.to(dtype=torch.bool), dim=1, index=positions.to(dtype=torch.long))


def gather_probs_keys(probs: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
    # probs: [B, H, Tq, S], key_positions: [B, Tk] -> [B, H, Tq, Tk]
    bsz, heads, q_len, _ = probs.shape
    idx = key_positions.to(device=probs.device, dtype=torch.long)[:, None, None, :].expand(
        bsz, heads, q_len, key_positions.shape[1]
    )
    return torch.gather(probs, dim=-1, index=idx)


def sdpa_bool_causal_mask(valid_mask: torch.Tensor) -> torch.Tensor:
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


def visual_pos_ids(full_position_ids: torch.Tensor, image_positions: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
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
        dim_positions = full_position_ids[dim_idx]
        out[dim_idx][valid] = dim_positions[batch_idx[valid], image_positions[valid].long()]
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
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

    layer_idx = int(args.layer)
    full_hidden = teacher.hidden_states[layer_idx].to(dtype=dtype)
    text_hidden = gather_batched_positions(full_hidden, text_positions, text_mask).to(dtype=dtype)
    full_effect_state = scatter_batched_positions(full_hidden, text_positions, text_hidden, text_mask)
    target_delta_train = compute_qwen3vl_attention_effect_batched(
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
    layer = language_model.layers[layer_idx]
    full_normed = layer.input_layernorm(full_effect_state)
    text_normed = layer.input_layernorm(text_hidden)
    full_attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=full_normed,
        attention_mask=full_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=full_position_ids[0],
    )
    text_attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=text_normed,
        attention_mask=text_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=text_position_ids[0],
    )
    self_joint_out, _ = layer.self_attn(
        hidden_states=full_normed,
        position_embeddings=language_model.rotary_emb(full_normed, full_position_ids),
        attention_mask=full_attention_mask,
        past_key_values=None,
    )
    self_text_out, _ = layer.self_attn(
        hidden_states=text_normed,
        position_embeddings=language_model.rotary_emb(text_normed, text_position_ids),
        attention_mask=text_attention_mask,
        past_key_values=None,
    )
    self_joint_text = gather_batched_positions(self_joint_out, text_positions, text_mask)

    q_full, k_full, v_full = native_qkv(language_model, layer_idx, full_effect_state, full_position_ids)
    q_text_direct, k_text_direct, v_text_direct = native_qkv(language_model, layer_idx, text_hidden, text_position_ids)
    q_text_from_full = gather_heads(q_full, text_positions, text_mask).transpose(1, 2).contiguous()
    k_vis_from_full = gather_heads(k_full, image_positions, image_mask)
    v_vis_from_full = gather_heads(v_full, image_positions, image_mask)

    memories = qwen3vl_visual_memory_by_layer(hidden0.to(dtype=dtype), image_positions, image_mask, deepstack_visual_embeds, "v0", len(language_model.layers))
    native_kv = qwen_native_visual_kv(
        language_model,
        layer_idx,
        memories[layer_idx].to(dtype=dtype),
        visual_pos_ids(full_position_ids, image_positions, image_mask),
        ~image_mask,
    )
    q_text_for_sidecar = q_text_direct.transpose(1, 2).contiguous()
    sidecar_vis = cross_attention(q_text_for_sidecar, native_kv.key, native_kv.value, native_kv.padding_mask)
    manual_vis = cross_attention(q_text_from_full, k_vis_from_full, v_vis_from_full, ~image_mask)

    # Explicit text/joint attention using the same native Q/K/V tensors.
    text_pre_o = F.scaled_dot_product_attention(
        q_text_direct,
        k_text_direct,
        v_text_direct,
        attn_mask=sdpa_bool_causal_mask(text_mask),
        is_causal=False,
        scale=float(language_model.layers[layer_idx].self_attn.scaling),
    )
    full_pre_o = F.scaled_dot_product_attention(
        q_full,
        k_full,
        v_full,
        attn_mask=sdpa_bool_causal_mask(full_mask),
        is_causal=False,
        scale=float(language_model.layers[layer_idx].self_attn.scaling),
    )
    joint_text_pre_o = gather_heads(full_pre_o, text_positions, text_mask)
    attn = language_model.layers[layer_idx].self_attn
    text_out = attn.o_proj(text_pre_o.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1))
    joint_out = attn.o_proj(joint_text_pre_o.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1))
    target_delta_explicit = joint_out - text_out

    # Manual probability decomposition:
    # A_joint - A_text = m_vis * (A_vis - A_text), before the shared o_proj.
    scale = float(attn.scaling)
    scores_full = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) * scale
    scores_full = scores_full.masked_fill(~sdpa_bool_causal_mask(full_mask), torch.finfo(scores_full.dtype).min)
    probs_full = torch.softmax(scores_full, dim=-1)
    text_query_idx = text_positions.to(device=probs_full.device, dtype=torch.long)[:, None, :, None].expand(
        probs_full.shape[0], probs_full.shape[1], text_positions.shape[1], probs_full.shape[-1]
    )
    probs_for_text_queries = torch.gather(probs_full, dim=2, index=text_query_idx)
    image_probs = gather_probs_keys(probs_for_text_queries, image_positions)
    text_probs = gather_probs_keys(probs_for_text_queries, text_positions)
    image_probs = image_probs * image_mask[:, None, None, :].to(dtype=image_probs.dtype)
    text_probs = text_probs * text_mask[:, None, None, :].to(dtype=text_probs.dtype)
    m_vis = image_probs.sum(dim=-1, keepdim=True).clamp_min(0.0)
    vis_readout_pre_o = torch.matmul(
        image_probs / m_vis.clamp_min(1e-12),
        v_vis_from_full.float(),
    )
    m_text = text_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    text_readout_from_joint_pre_o = torch.matmul(
        text_probs / m_text,
        gather_heads(v_full, text_positions, text_mask).float(),
    )
    formula_pre_o = m_vis * (vis_readout_pre_o - text_readout_from_joint_pre_o)
    formula_delta = attn.o_proj(
        formula_pre_o.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1).to(dtype=text_hidden.dtype)
    )
    visual_only_delta = attn.o_proj(
        vis_readout_pre_o.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1).to(dtype=text_hidden.dtype)
    )

    payload = {
        "model_path": args.model_path,
        "data": args.data,
        "samples": len(rows),
        "layer": layer_idx,
        "shapes": {
            "text_hidden": list(text_hidden.shape),
            "image_memory": list(memories[layer_idx].shape),
            "q_text": list(q_text_for_sidecar.shape),
            "native_k": list(native_kv.key.shape),
        },
        "kv_native_helper_vs_full_gather": {
            "key": metric(native_kv.key.transpose(1, 2), k_vis_from_full.transpose(1, 2), image_mask),
            "value": metric(native_kv.value.transpose(1, 2), v_vis_from_full.transpose(1, 2), image_mask),
        },
        "q_text_direct_vs_full_gather": metric(q_text_for_sidecar, q_text_from_full, text_mask),
        "sidecar_cross_attention_vs_manual_visual_attention": metric(sidecar_vis, manual_vis, text_mask),
        "self_attn_delta_vs_train_target": metric(self_joint_text - self_text_out, target_delta_train, text_mask),
        "manual_text_out_vs_self_attn_text_out": metric(text_out, self_text_out, text_mask),
        "manual_joint_text_out_vs_self_attn_joint_text_out": metric(joint_out, self_joint_text, text_mask),
        "target_train_vs_explicit_qkv": metric(target_delta_train, target_delta_explicit, text_mask),
        "manual_formula_delta_vs_explicit_delta": metric(formula_delta, target_delta_explicit, text_mask),
        "manual_formula_delta_vs_train_target": metric(formula_delta, target_delta_train, text_mask),
        "manual_visual_only_readout_vs_train_target": metric(visual_only_delta, target_delta_train, text_mask),
        "visual_attention_mass": {
            "mean": float(m_vis[text_mask[:, None, :, None].expand_as(m_vis)].float().mean().item()),
            "max": float(m_vis[text_mask[:, None, :, None].expand_as(m_vis)].float().max().item()),
            "min": float(m_vis[text_mask[:, None, :, None].expand_as(m_vis)].float().min().item()),
        },
        "visual_only_post_o_vs_target": metric(
            language_model.layers[layer_idx].self_attn.o_proj(sidecar_vis.reshape(sidecar_vis.shape[0], sidecar_vis.shape[1], -1)),
            target_delta_train,
            text_mask,
        ),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
