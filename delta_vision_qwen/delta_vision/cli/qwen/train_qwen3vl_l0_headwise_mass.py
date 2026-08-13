#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from delta_vision.cli.qwen.diagnose_qwen3vl_l0_alignment import gather_heads, gather_probs_keys, native_qkv
from delta_vision.data import JsonlDataset
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    scatter_batched_positions,
)
from delta_vision.runtime.ops import cross_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a strict L0 headwise mass predictor for Qwen3-VL.")
    parser.add_argument("--data", default="artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", default="artifacts/experiments/qwen_l0_headwise_mass/checkpoints")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--mass-loss-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-vision")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


class HeadwiseMassPredictor(nn.Module):
    def __init__(self, q_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.v_norm = nn.LayerNorm(q_dim)
        self.net = nn.Sequential(
            nn.Linear(q_dim * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.3)

    def forward(self, q_merged: torch.Tensor, visual_merged: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.q_norm(q_merged), self.v_norm(visual_merged)], dim=-1)
        return torch.sigmoid(self.net(x))


def masked_nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    pred_f = pred.float()[valid]
    target_f = target.float()[valid]
    return (pred_f - target_f).pow(2).mean() / target_f.pow(2).mean().clamp_min(1e-12)


def masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    pred_f = pred.float()[valid]
    target_f = target.float()[valid]
    return F.cosine_similarity(pred_f, target_f, dim=-1).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    return (pred.float()[valid] - target.float()[valid]).pow(2).mean()


@torch.no_grad()
def build_l0_targets(
    processor: object,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rows: list[dict[str, object]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor, rows, "image", "question", "answer", None, device)
    teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    _, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )

    layer_idx = 0
    full_hidden = teacher.hidden_states[layer_idx].to(dtype=dtype)
    text_hidden = gather_batched_positions(full_hidden, text_positions, text_mask).to(dtype=dtype)
    full_effect_state = scatter_batched_positions(full_hidden, text_positions, text_hidden, text_mask)
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
    ).detach()

    q_full, k_full, v_full = native_qkv(language_model, layer_idx, full_effect_state, full_position_ids)
    q_text, k_text, v_text = native_qkv(language_model, layer_idx, text_hidden, text_position_ids)
    k_vis = gather_heads(k_full, image_positions, image_mask)
    v_vis = gather_heads(v_full, image_positions, image_mask)
    visual_head = cross_attention(
        q_text.transpose(1, 2).contiguous(),
        k_vis,
        v_vis,
        ~image_mask,
    ).detach()

    layer = language_model.layers[layer_idx]
    text_normed = layer.input_layernorm(text_hidden)
    text_attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=text_normed,
        attention_mask=text_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=text_position_ids[0],
    )
    text_head = F.scaled_dot_product_attention(
        q_text,
        k_text,
        v_text,
        attn_mask=text_attention_mask,
        is_causal=False,
        scale=float(layer.self_attn.scaling),
    ).transpose(1, 2).contiguous().detach()

    scores_full = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) * float(layer.self_attn.scaling)
    full_attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=layer.input_layernorm(full_effect_state),
        attention_mask=full_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=full_position_ids[0],
    )
    if full_attention_mask is not None:
        scores_full = scores_full + full_attention_mask.float()
    probs_full = torch.softmax(scores_full, dim=-1)
    text_query_idx = text_positions.to(device=probs_full.device, dtype=torch.long)[:, None, :, None].expand(
        probs_full.shape[0],
        probs_full.shape[1],
        text_positions.shape[1],
        probs_full.shape[-1],
    )
    probs_for_text_queries = torch.gather(probs_full, dim=2, index=text_query_idx)
    image_probs = gather_probs_keys(probs_for_text_queries, image_positions)
    image_probs = image_probs * image_mask[:, None, None, :].to(dtype=image_probs.dtype)
    target_mass = image_probs.sum(dim=-1).transpose(1, 2).contiguous().detach()

    q_merged = q_text.transpose(1, 2).reshape(text_hidden.shape[0], text_hidden.shape[1], -1).detach()
    visual_merged = visual_head.reshape(text_hidden.shape[0], text_hidden.shape[1], -1).detach()
    return q_merged, visual_merged, visual_head, text_head, target_mass, target_delta, text_mask


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    layer0 = language_model.layers[0]
    num_heads = int(layer0.self_attn.config.num_attention_heads)
    head_dim = int(layer0.self_attn.head_dim)
    q_dim = num_heads * head_dim
    predictor = HeadwiseMassPredictor(q_dim=q_dim, num_heads=num_heads, hidden_dim=args.hidden_dim).to(device=device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = JsonlDataset(args.data, max_samples=args.max_samples, decode_images=False)
    if len(dataset) == 0:
        raise RuntimeError("empty dataset")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    for step in range(1, args.max_steps + 1):
        base = (step - 1) * args.batch_size
        rows = [dataset[(base + idx) % len(dataset)] for idx in range(args.batch_size)]
        q_merged, visual_merged, visual_head, text_head, target_mass, target_delta, text_mask = build_l0_targets(
            processor,
            model,
            language_model,
            rows,
            device,
            dtype,
        )
        pred_mass = predictor(q_merged.float(), visual_merged.float()).to(dtype=visual_head.dtype)
        pred_head_delta = pred_mass.unsqueeze(-1) * (visual_head - text_head)
        pred_delta = layer0.self_attn.o_proj(
            pred_head_delta.transpose(1, 2).reshape(target_delta.shape[0], target_delta.shape[1], -1)
        )
        effect = masked_nmse(pred_delta, target_delta, text_mask)
        effect_cos = masked_cos(pred_delta, target_delta, text_mask)
        mass_mse = masked_mse(pred_mass, target_mass, text_mask)
        loss = effect + args.mass_loss_weight * mass_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            valid = text_mask.bool()
            metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "effect": float(effect.detach()),
                "effect_cos": float(effect_cos.detach()),
                "mass_mse": float(mass_mse.detach()),
                "pred_mass": float(pred_mass.float()[valid].mean().detach()),
                "target_mass": float(target_mass.float()[valid].mean().detach()),
                "pred_effect_rms": float(pred_delta.float()[valid].pow(2).mean().sqrt().detach()),
                "target_effect_rms": float(target_delta.float()[valid].pow(2).mean().sqrt().detach()),
                "text_tokens": float(text_mask.sum().item()) / max(1, len(rows)),
            }
            print(
                " ".join(
                    f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in metrics.items()
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        if step % args.save_every == 0 or step == args.max_steps:
            torch.save({"state_dict": predictor.state_dict(), "args": vars(args), "step": step}, output_dir / f"step_{step}.pt")


if __name__ == "__main__":
    main()
