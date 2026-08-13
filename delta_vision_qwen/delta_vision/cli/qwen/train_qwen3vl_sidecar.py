#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from delta_vision.data import JsonlDataset
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.evaluation.metrics import masked_topk_kl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    compute_qwen3vl_visual_attention_mass_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    qwen3vl_text_attention_heads,
    qwen3vl_text_attention_output,
    qwen3vl_visual_memory_by_layer,
    qwen3vl_prefix_visual_memory_by_layer,
    run_qwen3vl_layer_text_from_attention_output,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)
from delta_vision.models.sidecar import DeltaVisionModule
from delta_vision.runtime.basis import reconstruct_delta
from delta_vision.runtime.ops import VisualKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL delta-vision trainer.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument(
        "--image-root",
        default=os.environ.get("DELTA_VISION_IMAGE_ROOT", ""),
        help="Root used to resolve relative image paths in the JSONL rows.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional sidecar checkpoint to initialize from. Missing new modules are allowed.",
    )
    parser.add_argument(
        "--resume-deepspeed-dir",
        default="",
        help="Optional DeepSpeed checkpoint directory to resume model/optimizer state from.",
    )
    parser.add_argument(
        "--resume-deepspeed-tag",
        default="",
        help="Optional DeepSpeed checkpoint tag, e.g. step1500. Defaults to DeepSpeed latest.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=0)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--reader-activation", choices=("gelu", "silu", "swiglu", "situ_glu"), default="situ_glu")
    parser.add_argument("--layer-adapter-rank", type=int, default=128)
    parser.add_argument(
        "--corrector-layers",
        default="",
        help="Comma-separated late layers that receive an extra coefficient corrector, e.g. 29,30,31,32,33,35.",
    )
    parser.add_argument("--corrector-dim", type=int, default=0)
    parser.add_argument(
        "--block-corrector-groups",
        default="",
        help="Semicolon-separated layer groups for hidden residual correctors, e.g. '12-17;18-29;30-35'.",
    )
    parser.add_argument("--block-corrector-dim", type=int, default=0)
    parser.add_argument("--shared-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-mode",
        choices=(
            "residual",
            "residual_full",
            "factorized_lowrank",
            "factorized_full",
            "factorized_native_o",
            "factorized_native_head_o",
            "factorized_native_head_o_pure",
            "factorized_native_head_o_residual",
            "native_cross_attention",
        ),
        default="residual",
        help="residual predicts low-rank basis coefficients; residual_full directly predicts the hidden residual.",
    )
    parser.add_argument("--effect-layers-per-sample", type=int, default=8)
    parser.add_argument(
        "--sidecar-active-layers",
        default="",
        help=(
            "Comma-separated layers where Sidecar is allowed to read visual memory. "
            "For residual output, inactive layers reuse the latest active-layer coefficients "
            "and do not run visual K/V projection or CrossAttn."
        ),
    )
    parser.add_argument("--visual-memory-mode", choices=("v0", "vdeep", "vcum", "vprefix"), default="v0")
    parser.add_argument(
        "--effect-target",
        choices=("teacher_visual", "static_memory", "native_vprefix"),
        default="teacher_visual",
        help=(
            "teacher_visual matches Qwen's dynamic visual-token effect from the joint teacher state; "
            "static_memory matches the Sidecar-visible memory oracle for the selected visual-memory-mode; "
            "native_vprefix explicitly trains a learned predictor against Qwen native attention effect "
            "computed from prefix-dynamic visual memory V_l^prefix."
        ),
    )
    parser.add_argument(
        "--teacher-force-steps",
        type=int,
        default=0,
        help="For the first N optimizer steps, reset each layer input to the Teacher text state before predicting delta.",
    )
    parser.add_argument(
        "--teacher-force-mix",
        type=float,
        default=1.0,
        help="Teacher-state mix ratio used before --teacher-force-steps; 1.0 is pure teacher forcing.",
    )
    parser.add_argument(
        "--teacher-mix-end-step",
        type=int,
        default=0,
        help=(
            "If > teacher-force-steps, linearly anneal teacher-state mixing from 1 to 0 between "
            "teacher-force-steps and this step. This keeps training closer to rollout at the end."
        ),
    )
    parser.add_argument(
        "--effect-layer-sampling",
        choices=("uniform", "all", "hard_mixed"),
        default="uniform",
        help="How to select layers for online effect supervision on each batch.",
    )
    parser.add_argument(
        "--hard-effect-layers",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35",
        help="Comma-separated priority layer ids used by --effect-layer-sampling hard_mixed.",
    )
    parser.add_argument(
        "--trajectory-layers",
        default="4,8,12,16,20,24,28,32,36",
        help="'all' or comma-separated state ids in [1, num_layers] for trajectory supervision.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--basis-lr-mult", type=float, default=0.2)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="constant",
        help="Learning-rate schedule. constant preserves legacy behavior.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of max_steps used for linear warmup.",
    )
    parser.add_argument(
        "--warmup-start-lr-ratio",
        type=float,
        default=0.0,
        help="Warmup starts at this fraction of each parameter group's base LR.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine decay floor as a fraction of each parameter group's base LR.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-effect", type=float, default=0.5)
    parser.add_argument("--lambda-effect-cos", type=float, default=0.0)
    parser.add_argument(
        "--lambda-effect-rms",
        type=float,
        default=0.0,
        help="Weight for matching the RMS scale of predicted and target attention effects.",
    )
    parser.add_argument("--lambda-trajectory", type=float, default=4.0)
    parser.add_argument("--lambda-trajectory-rms", type=float, default=0.0)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument(
        "--lambda-ce",
        type=float,
        default=0.0,
        help="Optional answer-token next-token cross entropy weight against the dataset answer.",
    )
    parser.add_argument(
        "--late-loss-layers",
        default="",
        help=(
            "Comma-separated layer ids whose local losses receive --late-loss-weight. "
            "Applies to effect/mass at layer l and trajectory at state l+1."
        ),
    )
    parser.add_argument(
        "--late-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for effect/mass/trajectory terms on --late-loss-layers.",
    )
    parser.add_argument(
        "--effect-start-step",
        type=int,
        default=0,
        help="Disable effect/cos/rms/trajectory/logit losses before this optimizer step; useful for mass warmup.",
    )
    parser.add_argument(
        "--lambda-mass",
        type=float,
        default=0.0,
        help="Optional BCE supervision for learned headwise visual mass against analytic Qwen mass.",
    )
    parser.add_argument(
        "--lambda-mass-after-effect-start",
        type=float,
        default=None,
        help="Mass supervision weight after --effect-start-step. Defaults to --lambda-mass.",
    )
    parser.add_argument("--mass-positive-threshold", type=float, default=0.01)
    parser.add_argument(
        "--mass-positive-weight",
        type=float,
        default=0.0,
        help="Extra BCE weight for token/head entries whose analytic mass exceeds --mass-positive-threshold.",
    )
    parser.add_argument("--use-rope", action="store_true", default=False, help="Apply M-RoPE to Sidecar Q/K.")
    parser.add_argument(
        "--visual-transform-mode",
        choices=(
            "none",
            "layer_kv_adapter",
            "stage_kv_adapter",
            "stage_kv_film",
            "depth_pos_mixing",
            "recurrent_adapter",
            "full_cascade_adapter",
            "latent_compressor",
            "context_shift",
        ),
        default="none",
        help="Layer-conditioned transform applied to visual memory before Sidecar/native visual K/V projection.",
    )
    parser.add_argument(
        "--visual-transform-rank",
        type=int,
        default=128,
        help="Bottleneck rank for layer_kv_adapter/recurrent_adapter/full_cascade_adapter visual-memory transforms.",
    )
    parser.add_argument(
        "--visual-transform-activation",
        choices=("gelu", "silu"),
        default="gelu",
        help="Activation used inside layer_kv_adapter visual-memory transforms.",
    )
    parser.add_argument(
        "--sidecar-query-source",
        choices=("sidecar", "qwen_native", "qwen_trainable_native"),
        default="sidecar",
        help="Use Sidecar q_proj or frozen Qwen layer q_proj/q_norm/M-RoPE for Sidecar queries.",
    )
    parser.add_argument(
        "--sidecar-visual-kv-source",
        choices=(
            "sidecar",
            "qwen_native",
            "qwen_first_layer",
            "qwen_first_layer_film",
            "qwen_trainable_native",
            "qwen_anchor_trainable",
        ),
        default="sidecar",
        help=(
            "Use Sidecar k/v projections or frozen Qwen layer input_layernorm+k_proj/k_norm/v_proj+M-RoPE "
            "for the visual K/V read by the Sidecar. qwen_first_layer computes Qwen layer-0 visual K/V once "
            "from V0 and reuses it at every Sidecar layer. qwen_first_layer_film adds a light per-layer FiLM "
            "adapter on top of that shared layer-0 visual K/V. qwen_anchor_trainable uses a small set of trainable "
            "Qwen-native K/V clones initialized from --qwen-anchor-layers and routes each layer to its nearest anchor."
        ),
    )
    parser.add_argument(
        "--qwen-anchor-layers",
        default="4,16,30",
        help="Comma-separated Qwen layer ids used by sidecar_visual_kv_source=qwen_anchor_trainable.",
    )
    parser.add_argument(
        "--layer-condition-mode",
        choices=("query", "none", "post_film", "hdelta_film"),
        default="query",
        help=(
            "Where explicit layer conditioning enters the Sidecar. query is the historical behavior; "
            "none removes layer embeddings from the reader; post_film applies FiLM after reader features; "
            "hdelta_film applies token-wise FiLM from [H_l, H_l-H_0]."
        ),
    )
    parser.add_argument(
        "--reader-mode",
        choices=("cross_attention", "pooled"),
        default="cross_attention",
        help=(
            "How the Sidecar reads visual memory. cross_attention is the normal token reader; "
            "pooled removes token cross-attention and conditions on a masked mean visual summary."
        ),
    )
    parser.add_argument(
        "--latent-tokens",
        type=int,
        default=64,
        help="Number of compressed visual tokens used by visual_transform_mode=latent_compressor.",
    )
    parser.add_argument("--output-init-std",
        type=float,
        default=1e-4,
        help="Qwen random-basis warm start for coefficient heads; 0 keeps exact zero-init.",
    )
    parser.add_argument(
        "--factorized-mass-mode",
        choices=("learned", "analytic", "oracle", "fixed"),
        default="learned",
        help=(
            "Mass used by factorized outputs. learned is the original mass_head; "
            "analytic computes deployable Qwen native visual mass from current text state and visible memory; "
            "oracle computes Teacher-state Qwen native visual mass from Q/K for diagnosis; "
            "fixed uses --fixed-visual-mass."
        ),
    )
    parser.add_argument("--fixed-visual-mass", type=float, default=0.12)
    parser.add_argument(
        "--native-qkv-init",
        choices=("none", "first", "mean"),
        default="none",
        help=(
            "Initialize Sidecar q/k/v projections from Qwen native attention projections. "
            "Requires sidecar_dim/num_heads to match Qwen query heads; native KV heads are repeated to query heads."
        ),
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument(
        "--distributed-engine",
        choices=("deepspeed", "torch_grad_sync"),
        default="deepspeed",
        help="Distributed optimizer backend. torch_grad_sync avoids DeepSpeed comm initialization.",
    )
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2_coeff.json")
    parser.add_argument("--required-world-size", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument(
        "--batch-sampling",
        choices=("sequential", "pixel_bucket"),
        default="pixel_bucket",
        help="Training sample order. pixel_bucket groups similarly sized images to reduce Qwen visual-token padding.",
    )
    parser.add_argument(
        "--pixel-bucket-size",
        type=int,
        default=512,
        help="Number of samples per size bucket when --batch-sampling pixel_bucket.",
    )
    parser.add_argument(
        "--pixel-area-cache",
        default="",
        help="Optional JSON cache for per-row image pixel areas. Defaults to <data>.pixel_areas.json.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of single-sample micro-steps per optimizer step on each rank.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-vision")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Synchronize CUDA and log per-section timing. This is slower and should be used for profiling only.",
    )
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def debug_print(message: str) -> None:
    if os.environ.get("QWEN_SIDECAR_DEBUG", "0") != "1":
        return
    rank = dist.get_rank() if distributed_is_initialized() else 0
    print(f"[debug rank={rank}] {message}", flush=True)


def distributed_barrier(device: torch.device | None = None) -> None:
    if not distributed_is_initialized():
        return
    if dist.get_backend() == "nccl" and device is not None and device.type == "cuda":
        token = torch.ones((), device=device, dtype=torch.float32)
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
    else:
        dist.barrier()


def sync_module_state(module: nn.Module) -> None:
    if not distributed_is_initialized():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=0)


def unwrap_sidecar(sidecar: nn.Module) -> DeltaVisionModule:
    return sidecar  # type: ignore[return-value]


def reduce_metric_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed_is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, values)}


def masked_directional_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    pred_norm = pred.float() / pred.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    target_norm = target.float() / target.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return ((pred_norm - target_norm).pow(2) * valid).sum() / valid.sum().mul(pred.shape[-1]).clamp_min(1.0)


def masked_nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    return ((pred.float() - target.float()).pow(2) * valid).sum() / (
        target.float().pow(2).mul(valid).sum().clamp_min(1e-6)
    )


def masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    return F.cosine_similarity(pred.float()[valid], target.float()[valid], dim=-1, eps=1e-6).mean()


def masked_rms_abs(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_rms = pred.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    target_rms = target.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return torch.log(pred_rms[valid] / target_rms[valid]).abs().mean()


def masked_mass_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_threshold: float = 0.01,
    positive_weight: float = 0.0,
) -> torch.Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_valid = pred.float()[valid].clamp(1e-5, 1.0 - 1e-5)
    target_valid = target.float()[valid].clamp(0.0, 1.0)
    loss = F.binary_cross_entropy(pred_valid, target_valid, reduction="none")
    if positive_weight > 0.0:
        weights = torch.ones_like(loss)
        weights = weights + float(positive_weight) * (target_valid > float(positive_threshold)).to(dtype=loss.dtype)
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def masked_mass_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    return (pred.float()[valid] - target.float()[valid]).pow(2).mean()


def answer_token_ce(logits: torch.Tensor, text_ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Causal CE on answer tokens: logits at t-1 predict token id at t."""
    if logits.shape[1] < 2:
        return logits.new_zeros(())
    shifted_logits = logits[:, :-1, :]
    shifted_labels = text_ids[:, 1:]
    shifted_mask = answer_mask[:, 1:].to(device=logits.device).bool()
    if not bool(shifted_mask.any()):
        return logits.new_zeros(())
    return F.cross_entropy(shifted_logits.float()[shifted_mask], shifted_labels.long()[shifted_mask])


def sample_layers(num_layers: int, count: int, device: torch.device) -> list[int]:
    count = min(int(count), int(num_layers))
    if count <= 0:
        return []
    perm = torch.randperm(num_layers, device=device)[:count].sort().values
    return [int(x.item()) for x in perm]


def parse_int_set(spec: str, upper_exclusive: int | None = None) -> set[int]:
    values: set[int] = set()
    if not spec:
        return values
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if upper_exclusive is not None and (value < 0 or value >= upper_exclusive):
            continue
        values.add(value)
    return values


def parse_trajectory_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(1, num_layers + 1))
    values = parse_int_set(spec, num_layers + 1)
    return {value for value in values if value > 0}


def sample_effect_layers(args: argparse.Namespace, num_layers: int, device: torch.device) -> list[int]:
    if args.effect_layer_sampling == "all":
        return list(range(num_layers))
    count = min(int(args.effect_layers_per_sample), int(num_layers))
    if count <= 0:
        return []
    if args.effect_layer_sampling == "uniform":
        return sample_layers(num_layers, count, device)
    hard_layers = sorted(parse_int_set(args.hard_effect_layers, num_layers))
    selected = hard_layers[:count]
    if len(selected) < count:
        selected_set = set(selected)
        remaining = [layer for layer in range(num_layers) if layer not in selected_set]
        perm = torch.randperm(len(remaining), device=device)[: count - len(selected)].sort().values
        selected.extend(remaining[int(x.item())] for x in perm)
    return sorted(selected)


def weighted_mean(terms: list[torch.Tensor], weights: list[float], fallback: torch.Tensor) -> torch.Tensor:
    if not terms:
        return fallback.new_zeros(())
    if len(terms) != len(weights):
        raise ValueError("terms and weights must have the same length")
    weighted = [term * float(weight) for term, weight in zip(terms, weights)]
    return torch.stack(weighted).sum() / max(1e-6, float(sum(weights)))


def teacher_mix_ratio(args: argparse.Namespace, global_step: int) -> float:
    teacher_force_steps = int(args.teacher_force_steps)
    mix_end = int(args.teacher_mix_end_step)
    force_mix = max(0.0, min(1.0, float(args.teacher_force_mix)))
    if global_step < teacher_force_steps:
        return force_mix
    if mix_end <= teacher_force_steps or global_step >= mix_end:
        return 0.0
    progress = float(global_step - teacher_force_steps) / float(max(1, mix_end - teacher_force_steps))
    return max(0.0, min(1.0, force_mix * (1.0 - progress)))


def image_pixel_area(row: dict[str, object], image_key: str = "image", image_root: Path | None = None) -> int:
    image_value = row.get(image_key)
    if not image_value:
        return 0
    image_path = Path(str(image_value))
    if image_root is not None and not image_path.is_absolute():
        image_path = image_root / image_path
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return max(1, int(width) * int(height))
    except Exception:
        return 0


def load_or_build_pixel_areas(args: argparse.Namespace, dataset: JsonlDataset) -> list[int]:
    dataset_size = len(dataset)
    cache_path = Path(args.pixel_area_cache) if args.pixel_area_cache else Path(str(args.data) + ".pixel_areas.json")
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            areas = cached.get("areas") if isinstance(cached, dict) else cached
            if isinstance(areas, list) and len(areas) == dataset_size:
                return [int(x) for x in areas]
        except Exception:
            pass
    rows = getattr(dataset, "rows", None)
    if rows is None:
        rows = [dataset[idx] for idx in range(dataset_size)]
    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    areas = [image_pixel_area(row, image_root=image_root) for row in rows]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"data": str(args.data), "count": dataset_size, "areas": areas}),
            encoding="utf-8",
        )
    except Exception:
        pass
    return areas


def build_training_order(args: argparse.Namespace, dataset: JsonlDataset, world_size: int) -> list[int]:
    dataset_size = len(dataset)
    if args.batch_sampling == "sequential":
        return list(range(dataset_size))

    min_bucket = max(1, int(args.micro_batch_size_per_gpu) * max(1, int(world_size)))
    bucket_size = max(min_bucket, int(args.pixel_bucket_size))
    areas = load_or_build_pixel_areas(args, dataset)

    sized = [(area, idx) for idx, area in enumerate(areas)]
    sized.sort(key=lambda x: x[0])
    buckets = [
        [idx for _, idx in sized[start : start + bucket_size]]
        for start in range(0, dataset_size, bucket_size)
    ]
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    bucket_order = torch.randperm(len(buckets), generator=generator).tolist()
    order: list[int] = []
    for bucket_idx in bucket_order:
        bucket = buckets[int(bucket_idx)]
        if len(bucket) > 1:
            perm = torch.randperm(len(bucket), generator=generator).tolist()
            order.extend(bucket[int(i)] for i in perm)
        else:
            order.extend(bucket)
    return order


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
    return scatter_batched_positions(full, image_positions, image_memory, image_mask)


def save_sidecar_checkpoint(sidecar: DeltaVisionModule, output_path: Path, args: argparse.Namespace, global_step: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in sidecar.state_dict().items()},
            "args": vars(args),
            "global_step": int(global_step),
        },
        output_path,
    )


class SimpleEngine:
    """Single-process fallback with the subset of DeepSpeedEngine used here."""

    def __init__(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.module = module
        self.optimizer = optimizer

    def train(self) -> None:
        self.module.train()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, output_dir: str, tag: str) -> None:
        if distributed_is_initialized() and dist.get_rank() != 0:
            return
        path = Path(output_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"module": self.module.state_dict()}, path / "model_states.pt")


class TorchGradSyncEngine(SimpleEngine):
    """Torch distributed engine that synchronizes gradients without DeepSpeed."""

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        world_size = dist.get_world_size() if distributed_is_initialized() else 1
        if world_size <= 1:
            return
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(float(world_size))


def optimizer_param_groups(engine: object) -> list[dict]:
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        return []
    if hasattr(optimizer, "param_groups"):
        return optimizer.param_groups
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "param_groups"):
        return inner.param_groups
    return []


def lr_multiplier(args: argparse.Namespace, step: int) -> float:
    if args.lr_scheduler == "constant":
        return 1.0
    if args.lr_scheduler != "cosine":
        raise ValueError(f"unsupported lr scheduler: {args.lr_scheduler}")
    total_steps = max(1, int(args.max_steps))
    warmup_steps = int(math.ceil(float(args.warmup_ratio) * total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps))
    start_ratio = max(0.0, float(args.warmup_start_lr_ratio))
    min_ratio = max(0.0, float(args.min_lr_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        progress = float(step + 1) / float(warmup_steps)
        return start_ratio + (1.0 - start_ratio) * progress
    decay_steps = max(1, total_steps - warmup_steps)
    decay_progress = float(step - warmup_steps + 1) / float(decay_steps)
    decay_progress = min(1.0, max(0.0, decay_progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_engine_lr(engine: object, base_lrs: list[float], multiplier: float) -> float:
    groups = optimizer_param_groups(engine)
    if not groups:
        return 0.0
    for group, base_lr in zip(groups, base_lrs, strict=False):
        group["lr"] = float(base_lr) * float(multiplier)
    return float(groups[0].get("lr", 0.0))


def _repeat_native_kv_to_query_heads(
    weight: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    out_features: int,
) -> torch.Tensor:
    expected_kv_out = int(kv_heads) * int(head_dim)
    if int(weight.shape[0]) != expected_kv_out:
        raise ValueError(f"native KV projection out_features={weight.shape[0]} does not match {expected_kv_out}")
    repeat_factor = int(q_heads) // int(kv_heads)
    if repeat_factor * int(kv_heads) != int(q_heads):
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    repeated = weight.view(kv_heads, head_dim, weight.shape[1]).repeat_interleave(repeat_factor, dim=0)
    repeated = repeated.reshape(out_features, weight.shape[1])
    if int(repeated.shape[0]) != int(out_features):
        raise ValueError(f"repeated KV out_features={repeated.shape[0]} does not match {out_features}")
    return repeated


@torch.no_grad()
def initialize_sidecar_from_qwen_native_qkv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    mode: str,
) -> None:
    if mode == "none":
        return
    config = language_model.config
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    native_q_out = q_heads * head_dim
    native_kv_out = kv_heads * head_dim
    hidden_size = int(config.hidden_size)
    if int(sidecar.sidecar_dim) != native_q_out or int(sidecar.num_heads) != q_heads:
        raise ValueError(
            "native_qkv_init requires sidecar_dim/num_heads to match Qwen native query layout: "
            f"expected sidecar_dim={native_q_out}, num_heads={q_heads}; "
            f"got sidecar_dim={sidecar.sidecar_dim}, num_heads={sidecar.num_heads}"
        )
    if int(sidecar.hidden_size) != hidden_size:
        raise ValueError(f"hidden_size mismatch: sidecar={sidecar.hidden_size}, qwen={hidden_size}")

    layers = list(language_model.layers)
    if not layers:
        raise ValueError("Qwen language model has no layers")
    if mode == "first":
        q_weight = layers[0].self_attn.q_proj.weight.detach().float().cpu()
        k_weight = layers[0].self_attn.k_proj.weight.detach().float().cpu()
        v_weight = layers[0].self_attn.v_proj.weight.detach().float().cpu()
    elif mode == "mean":
        q_weight = torch.zeros((native_q_out, hidden_size), dtype=torch.float32)
        k_weight = torch.zeros((native_kv_out, hidden_size), dtype=torch.float32)
        v_weight = torch.zeros((native_kv_out, hidden_size), dtype=torch.float32)
        for layer in layers:
            q_weight.add_(layer.self_attn.q_proj.weight.detach().float().cpu())
            k_weight.add_(layer.self_attn.k_proj.weight.detach().float().cpu())
            v_weight.add_(layer.self_attn.v_proj.weight.detach().float().cpu())
        denom = float(len(layers))
        q_weight.div_(denom)
        k_weight.div_(denom)
        v_weight.div_(denom)
    else:
        raise ValueError(f"unsupported native_qkv_init mode: {mode}")

    k_weight = _repeat_native_kv_to_query_heads(
        k_weight,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        out_features=native_q_out,
    )
    v_weight = _repeat_native_kv_to_query_heads(
        v_weight,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        out_features=native_q_out,
    )
    sidecar.q_proj.weight.copy_(q_weight.to(device=sidecar.q_proj.weight.device, dtype=sidecar.q_proj.weight.dtype))
    sidecar.k_proj.weight.copy_(k_weight.to(device=sidecar.k_proj.weight.device, dtype=sidecar.k_proj.weight.dtype))
    sidecar.v_proj.weight.copy_(v_weight.to(device=sidecar.v_proj.weight.device, dtype=sidecar.v_proj.weight.dtype))


@torch.no_grad()
def attach_qwen_trainable_native_qkv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    *,
    train_query: bool,
    train_visual_kv: bool,
) -> None:
    """Attach per-layer trainable Qwen-native q/k/v clones to the Sidecar.

    These modules keep the frozen Qwen backbone unchanged. They are used only
    for the Sidecar reader path, while Qwen layernorm, q_norm/k_norm and M-RoPE
    semantics are still reused around the cloned projections.
    """
    config = language_model.config
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    hidden_size = int(config.hidden_size)
    native_q_out = q_heads * head_dim
    native_kv_out = kv_heads * head_dim
    if int(sidecar.hidden_size) != hidden_size:
        raise ValueError(f"hidden_size mismatch: sidecar={sidecar.hidden_size}, qwen={hidden_size}")
    if int(sidecar.sidecar_dim) != native_q_out or int(sidecar.num_heads) != q_heads:
        raise ValueError(
            "qwen_trainable_native requires sidecar_dim/num_heads to match Qwen native query layout: "
            f"expected sidecar_dim={native_q_out}, num_heads={q_heads}; "
            f"got sidecar_dim={sidecar.sidecar_dim}, num_heads={sidecar.num_heads}"
        )
    layers = list(language_model.layers)
    device = sidecar.q_proj.weight.device
    dtype = sidecar.q_proj.weight.dtype
    if train_query:
        q_modules = nn.ModuleList([nn.Linear(hidden_size, native_q_out, bias=False) for _ in layers])
        for module, layer in zip(q_modules, layers, strict=True):
            module.weight.copy_(layer.self_attn.q_proj.weight.detach().to(device=device, dtype=dtype))
        sidecar.trainable_native_q_proj = q_modules.to(device=device, dtype=dtype)
    if train_visual_kv:
        k_modules = nn.ModuleList([nn.Linear(hidden_size, native_kv_out, bias=False) for _ in layers])
        v_modules = nn.ModuleList([nn.Linear(hidden_size, native_kv_out, bias=False) for _ in layers])
        for k_module, v_module, layer in zip(k_modules, v_modules, layers, strict=True):
            k_module.weight.copy_(layer.self_attn.k_proj.weight.detach().to(device=device, dtype=dtype))
            v_module.weight.copy_(layer.self_attn.v_proj.weight.detach().to(device=device, dtype=dtype))
        sidecar.trainable_native_k_proj = k_modules.to(device=device, dtype=dtype)
        sidecar.trainable_native_v_proj = v_modules.to(device=device, dtype=dtype)


@torch.no_grad()
def attach_qwen_anchor_trainable_visual_kv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    anchor_layers: str,
) -> None:
    """Attach trainable Qwen-native visual K/V clones for a few anchor layers."""
    config = language_model.config
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    hidden_size = int(config.hidden_size)
    native_q_out = q_heads * head_dim
    native_kv_out = kv_heads * head_dim
    if int(sidecar.hidden_size) != hidden_size:
        raise ValueError(f"hidden_size mismatch: sidecar={sidecar.hidden_size}, qwen={hidden_size}")
    if int(sidecar.sidecar_dim) != native_q_out or int(sidecar.num_heads) != q_heads:
        raise ValueError(
            "qwen_anchor_trainable requires sidecar_dim/num_heads to match Qwen native query layout: "
            f"expected sidecar_dim={native_q_out}, num_heads={q_heads}; "
            f"got sidecar_dim={sidecar.sidecar_dim}, num_heads={sidecar.num_heads}"
        )
    layers = list(language_model.layers)
    anchors = sorted(parse_int_set(anchor_layers, len(layers)))
    if not anchors:
        raise ValueError("--qwen-anchor-layers must contain at least one valid layer id")
    device = sidecar.q_proj.weight.device
    dtype = sidecar.q_proj.weight.dtype
    k_modules = nn.ModuleList([nn.Linear(hidden_size, native_kv_out, bias=False) for _ in anchors])
    v_modules = nn.ModuleList([nn.Linear(hidden_size, native_kv_out, bias=False) for _ in anchors])
    for k_module, v_module, anchor_layer in zip(k_modules, v_modules, anchors, strict=True):
        layer = layers[anchor_layer]
        k_module.weight.copy_(layer.self_attn.k_proj.weight.detach().to(device=device, dtype=dtype))
        v_module.weight.copy_(layer.self_attn.v_proj.weight.detach().to(device=device, dtype=dtype))
    sidecar.qwen_anchor_layers = tuple(anchors)
    sidecar.qwen_anchor_k_proj = k_modules.to(device=device, dtype=dtype)
    sidecar.qwen_anchor_v_proj = v_modules.to(device=device, dtype=dtype)


def nearest_qwen_anchor_index(sidecar: DeltaVisionModule, layer_idx: int) -> tuple[int, int]:
    anchors = tuple(getattr(sidecar, "qwen_anchor_layers", ()))
    if not anchors:
        raise RuntimeError("sidecar is missing qwen_anchor_layers")
    best_idx, best_layer = min(enumerate(anchors), key=lambda item: (abs(int(item[1]) - int(layer_idx)), int(item[1])))
    return int(best_idx), int(best_layer)


def residual_from_cached_coeff(
    sidecar: DeltaVisionModule,
    coeff: torch.Tensor | None,
    layer_idx_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if coeff is None:
        return hidden_states.new_zeros(hidden_states.shape)
    basis = sidecar.layer_basis(layer_idx_tensor, hidden_states.device, coeff.dtype)
    residual = reconstruct_delta(coeff, basis)
    gate = sidecar.gate[layer_idx_tensor].view(hidden_states.shape[0], 1, 1).to(
        device=hidden_states.device,
        dtype=residual.dtype,
    )
    return residual * gate


def qwen_native_sidecar_query(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Return Qwen-native Q after input norm, q_proj, q_norm and M-RoPE."""
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    position_embeddings = language_model.rotary_emb(normed, position_ids)
    query_states, _ = apply_rotary_pos_emb(query_states, query_states, *position_embeddings)
    return query_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()


def qwen_trainable_native_sidecar_query(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Return trainable-clone Q after frozen Qwen input norm, q_norm and M-RoPE."""
    if not hasattr(sidecar, "trainable_native_q_proj"):
        raise RuntimeError("sidecar is missing trainable_native_q_proj")
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(sidecar.trainable_native_q_proj[layer_idx](normed).view(hidden_shape)).transpose(1, 2)
    position_embeddings = language_model.rotary_emb(normed, position_ids)
    query_states, _ = apply_rotary_pos_emb(query_states, query_states, *position_embeddings)
    return query_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()


def qwen_native_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    vision_states: torch.Tensor,
    visual_position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> VisualKVCache:
    """Return Qwen-native visual K/V after input norm, k_norm and M-RoPE.

    The cache is converted to query-head layout because DeltaVisionModule's
    cross_attention expects key/value heads to match query heads.
    """
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    position_embeddings = language_model.rotary_emb(normed, visual_position_ids)
    _, key_states = apply_rotary_pos_emb(key_states, key_states, *position_embeddings)
    num_key_value_groups = int(self_attn.num_key_value_groups)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    return VisualKVCache(
        key=key_states.contiguous(),
        value=value_states.contiguous(),
        padding_mask=padding_mask,
    )


@torch.no_grad()
def attach_qwen_first_layer_film_visual_kv(sidecar: DeltaVisionModule, language_model: torch.nn.Module) -> None:
    """Register lightweight per-layer FiLM parameters over shared layer-0 Qwen visual K/V."""
    first_attn = language_model.layers[0].self_attn
    num_heads = int(sidecar.num_heads)
    head_dim = int(first_attn.head_dim)
    shape = (len(language_model.layers), num_heads, head_dim)
    device = sidecar.q_proj.weight.device
    dtype = sidecar.q_proj.weight.dtype
    sidecar.qwen_first_layer_k_gamma = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
    sidecar.qwen_first_layer_k_beta = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
    sidecar.qwen_first_layer_v_gamma = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
    sidecar.qwen_first_layer_v_beta = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))


def qwen_first_layer_film_visual_kv(sidecar: DeltaVisionModule, base_visual_kv: VisualKVCache, layer_idx: int) -> VisualKVCache:
    if not hasattr(sidecar, "qwen_first_layer_k_gamma"):
        raise RuntimeError("sidecar is missing qwen_first_layer_film parameters")
    k_gamma = sidecar.qwen_first_layer_k_gamma[layer_idx].to(device=base_visual_kv.key.device, dtype=base_visual_kv.key.dtype)
    k_beta = sidecar.qwen_first_layer_k_beta[layer_idx].to(device=base_visual_kv.key.device, dtype=base_visual_kv.key.dtype)
    v_gamma = sidecar.qwen_first_layer_v_gamma[layer_idx].to(device=base_visual_kv.value.device, dtype=base_visual_kv.value.dtype)
    v_beta = sidecar.qwen_first_layer_v_beta[layer_idx].to(device=base_visual_kv.value.device, dtype=base_visual_kv.value.dtype)
    return VisualKVCache(
        key=base_visual_kv.key * (1.0 + k_gamma.view(1, *k_gamma.shape[:1], 1, k_gamma.shape[-1]))
        + k_beta.view(1, *k_beta.shape[:1], 1, k_beta.shape[-1]),
        value=base_visual_kv.value * (1.0 + v_gamma.view(1, *v_gamma.shape[:1], 1, v_gamma.shape[-1]))
        + v_beta.view(1, *v_beta.shape[:1], 1, v_beta.shape[-1]),
        padding_mask=base_visual_kv.padding_mask,
    )


def qwen_trainable_native_visual_kv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    vision_states: torch.Tensor,
    visual_position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> VisualKVCache:
    """Return trainable-clone visual K/V with frozen Qwen norm/k_norm/M-RoPE."""
    if not hasattr(sidecar, "trainable_native_k_proj") or not hasattr(sidecar, "trainable_native_v_proj"):
        raise RuntimeError("sidecar is missing trainable_native_k_proj/trainable_native_v_proj")
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    key_states = self_attn.k_norm(sidecar.trainable_native_k_proj[layer_idx](normed).view(hidden_shape)).transpose(1, 2)
    value_states = sidecar.trainable_native_v_proj[layer_idx](normed).view(hidden_shape).transpose(1, 2)
    position_embeddings = language_model.rotary_emb(normed, visual_position_ids)
    _, key_states = apply_rotary_pos_emb(key_states, key_states, *position_embeddings)
    num_key_value_groups = int(self_attn.num_key_value_groups)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    return VisualKVCache(
        key=key_states.contiguous(),
        value=value_states.contiguous(),
        padding_mask=padding_mask,
    )


def qwen_anchor_trainable_visual_kv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    vision_states: torch.Tensor,
    visual_position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> VisualKVCache:
    """Return visual K/V from the nearest trainable Qwen-native anchor clone."""
    if not hasattr(sidecar, "qwen_anchor_k_proj") or not hasattr(sidecar, "qwen_anchor_v_proj"):
        raise RuntimeError("sidecar is missing qwen_anchor_k_proj/qwen_anchor_v_proj")
    anchor_idx, anchor_layer = nearest_qwen_anchor_index(sidecar, layer_idx)
    layer = language_model.layers[anchor_layer]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    key_states = self_attn.k_norm(sidecar.qwen_anchor_k_proj[anchor_idx](normed).view(hidden_shape)).transpose(1, 2)
    value_states = sidecar.qwen_anchor_v_proj[anchor_idx](normed).view(hidden_shape).transpose(1, 2)
    position_embeddings = language_model.rotary_emb(normed, visual_position_ids)
    _, key_states = apply_rotary_pos_emb(key_states, key_states, *position_embeddings)
    num_key_value_groups = int(self_attn.num_key_value_groups)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    return VisualKVCache(
        key=key_states.contiguous(),
        value=value_states.contiguous(),
        padding_mask=padding_mask,
    )


def compute_loss_for_row(
    args: argparse.Namespace,
    processor: object,
    teacher_model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: torch.nn.Module,
    rows: list[dict[str, object]],
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    sidecar_module = unwrap_sidecar(sidecar)
    timings: dict[str, float] = {}

    def mark_start() -> float:
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def mark(name: str, start: float) -> float:
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + end - start
        return end

    section_start = mark_start()
    inputs, text_ids, answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
        processor,
        rows,
        "image",
        "question",
        "answer",
        Path(args.image_root) if str(args.image_root).strip() else None,
        device,
    )
    section_start = mark("prepare_s", section_start)
    with torch.no_grad():
        teacher = teacher_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        section_start = mark("teacher_forward_s", section_start)
        hidden0, full_position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(
            teacher_model,
            inputs,
        )
        section_start = mark("context_s", section_start)

    (
        text_positions,
        image_positions,
        text_position_ids,
        text_mask,
        image_mask,
        full_mask,
    ) = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    teacher_states = [state.detach() for state in teacher.hidden_states]
    teacher_text_states = [gather_batched_positions(state, text_positions, text_mask).detach() for state in teacher_states]
    if args.visual_memory_mode == "vprefix":
        visual_memories = qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            hidden0.to(dtype=dtype),
            full_position_ids,
            inputs["attention_mask"],
            image_positions,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
    else:
        visual_memories = qwen3vl_visual_memory_by_layer(
            hidden0.to(dtype=dtype),
            image_positions,
            image_mask,
            deepstack_visual_embeds,
            args.visual_memory_mode,
            num_layers,
        )
    section_start = mark("visual_memory_s", section_start)
    teacher_logits = gather_batched_positions(teacher.logits, text_positions, text_mask).detach()

    sidecar_token_mask = text_mask
    text_padding_mask = ~text_mask
    # RoPE/native visual positions. Qwen visual positions are gathered from the
    # original multimodal position_ids; using text-like positions here is a
    # silent mismatch for native visual attention.
    _vis_pos_ids = None
    if getattr(args, "use_rope", False) or args.sidecar_visual_kv_source in {
        "qwen_native",
        "qwen_first_layer",
        "qwen_first_layer_film",
        "qwen_trainable_native",
        "qwen_anchor_trainable",
    }:
        _vis_pos_ids = torch.zeros(
            3,
            image_positions.shape[0],
            image_positions.shape[1],
            device=device,
            dtype=full_position_ids.dtype,
        )
        _vis_valid = image_mask.bool()
        for dim_idx in range(3):
            _dim_positions = full_position_ids[dim_idx]  # [batch, full_seq_len]
            _batch_idx = torch.arange(_dim_positions.shape[0], device=device).unsqueeze(1).expand_as(image_positions)
            _vis_pos_ids[dim_idx][_vis_valid] = _dim_positions[
                _batch_idx[_vis_valid], image_positions[_vis_valid].long()
            ]

    # RoPE: precompute Sidecar visual KV and text position embeddings.
    _rope_visual_kv = None
    _rope_text_pos_emb = None
    _dynamic_visual_pos_emb = None
    if (
        getattr(args, "use_rope", False)
        and args.sidecar_visual_kv_source == "sidecar"
        and getattr(sidecar_module, "visual_transform_mode", "none") == "none"
    ):
        assert _vis_pos_ids is not None
        _vis_pos_emb = language_model.rotary_emb(visual_memories[0].to(dtype=dtype), _vis_pos_ids)
        _rope_visual_kv = sidecar_module.prepare_visual_kv(
            visual_memories[0].to(dtype=dtype), ~image_mask, position_embeddings=_vis_pos_emb
        )
        _rope_text_pos_emb = language_model.rotary_emb(visual_memories[0].to(dtype=dtype), text_position_ids)
    elif getattr(args, "use_rope", False) and args.sidecar_visual_kv_source == "sidecar":
        if _vis_pos_ids is None:
            raise RuntimeError("visual position ids are required for Sidecar RoPE visual K/V")
        _rope_text_pos_emb = language_model.rotary_emb(visual_memories[0].to(dtype=dtype), text_position_ids)
    _first_layer_visual_kv = None
    if args.sidecar_visual_kv_source in {"qwen_first_layer", "qwen_first_layer_film"}:
        if _vis_pos_ids is None:
            raise RuntimeError("visual position ids are required for qwen_first_layer visual K/V")
        with torch.no_grad():
            _first_layer_visual_kv = qwen_native_visual_kv(
                language_model,
                0,
                visual_memories[0].to(dtype=dtype),
                _vis_pos_ids,
                ~image_mask,
            )
    sidecar_state = (
        sidecar_module.initial_state(visual_memories[0].to(dtype=dtype), ~image_mask)
        if args.state_tokens > 0
        else None
    )
    visual_memory_state = sidecar_module.initial_visual_memory_state(visual_memories[0].to(dtype=dtype))
    h = teacher_text_states[0].to(dtype=dtype).masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
    initial_text_hidden = h
    effect_layers = set(sample_effect_layers(args, num_layers, device))
    trajectory_layers = parse_trajectory_layers(args.trajectory_layers, num_layers)
    sidecar_active_layers = (
        parse_int_set(args.sidecar_active_layers, num_layers)
        if str(args.sidecar_active_layers).strip()
        else set(range(num_layers))
    )
    if sidecar_active_layers != set(range(num_layers)) and sidecar_module.output_mode != "residual":
        raise ValueError("--sidecar-active-layers currently requires --output-mode residual")
    late_loss_layers = parse_int_set(args.late_loss_layers, num_layers)
    late_loss_weight = max(1.0, float(args.late_loss_weight))
    effect_terms: list[torch.Tensor] = []
    effect_loss_terms: list[torch.Tensor] = []
    effect_loss_weights: list[float] = []
    effect_cos_terms: list[torch.Tensor] = []
    effect_cos_loss_terms: list[torch.Tensor] = []
    effect_cos_loss_weights: list[float] = []
    effect_rms_terms: list[torch.Tensor] = []
    effect_rms_loss_terms: list[torch.Tensor] = []
    effect_rms_loss_weights: list[float] = []
    pred_effect_rms_terms: list[torch.Tensor] = []
    target_effect_rms_terms: list[torch.Tensor] = []
    mass_terms: list[torch.Tensor] = []
    mass_loss_terms: list[torch.Tensor] = []
    mass_loss_weights: list[float] = []
    mass_mse_terms: list[torch.Tensor] = []
    mass_pred_mean_terms: list[torch.Tensor] = []
    mass_target_mean_terms: list[torch.Tensor] = []
    mass_target_pos_rate_terms: list[torch.Tensor] = []
    mass_target_pos_mean_terms: list[torch.Tensor] = []
    mass_target_max_terms: list[torch.Tensor] = []
    traj_terms: list[torch.Tensor] = []
    traj_loss_terms: list[torch.Tensor] = []
    traj_loss_weights: list[float] = []
    traj_rms_terms: list[torch.Tensor] = []
    traj_rms_loss_terms: list[torch.Tensor] = []
    traj_rms_loss_weights: list[float] = []
    teacher_mix = teacher_mix_ratio(args, global_step)
    cached_anchor_coeff: torch.Tensor | None = None
    for layer_idx in range(num_layers):
        if teacher_mix > 0.0:
            teacher_h = teacher_text_states[layer_idx].to(dtype=dtype).masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
            if teacher_mix >= 1.0:
                h = teacher_h
            else:
                h = (h * (1.0 - teacher_mix) + teacher_h * teacher_mix).masked_fill(
                    text_padding_mask.unsqueeze(-1),
                    0.0,
        )
        layer_tensor = torch.full((h.shape[0],), layer_idx, device=device, dtype=torch.long)
        sidecar_layer_active = layer_idx in sidecar_active_layers
        raw_vision_states = visual_memories[layer_idx].to(dtype=dtype)
        vision_states, visual_memory_state = sidecar_module.visual_memory_for_layer(
            raw_vision_states,
            layer_idx,
            hidden_states=h,
            initial_hidden_states=initial_text_hidden,
            current_visual_memory=visual_memory_state,
            vision_padding_mask=~image_mask,
        )
        if (
            getattr(args, "use_rope", False)
            and args.sidecar_visual_kv_source == "sidecar"
            and _rope_visual_kv is None
        ):
            _dynamic_visual_pos_emb = language_model.rotary_emb(vision_states, _vis_pos_ids)
        else:
            _dynamic_visual_pos_emb = None
        text_attention = None
        if sidecar_module.output_mode.startswith("factorized"):
            op_start = mark_start()
            text_attention = qwen3vl_text_attention_output(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
            mark("text_attention_s", op_start)
        text_attention_heads = None
        if sidecar_module.output_mode in {
            "factorized_native_head_o",
            "factorized_native_head_o_pure",
            "factorized_native_head_o_residual",
        }:
            op_start = mark_start()
            text_attention_heads = qwen3vl_text_attention_heads(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
            mark("text_attention_heads_s", op_start)
        visual_mass_override = None
        if sidecar_module.output_mode.startswith("factorized") and args.factorized_mass_mode != "learned":
            if args.factorized_mass_mode == "fixed":
                visual_mass_override = torch.full(
                    (h.shape[0], h.shape[1], 1),
                    float(args.fixed_visual_mass),
                    device=device,
                    dtype=dtype,
                )
            else:
                if args.factorized_mass_mode == "analytic":
                    mass_full_state = build_full_from_text_and_memory(
                        teacher_states[layer_idx].to(dtype=dtype),
                        text_positions,
                        h,
                        text_mask,
                        image_positions,
                        vision_states,
                        image_mask,
                    )
                elif args.effect_target == "teacher_visual":
                    mass_full_state = scatter_batched_positions(
                        teacher_states[layer_idx].to(dtype=dtype),
                        text_positions,
                        h,
                        text_mask,
                    )
                else:
                    mass_full_state = build_full_from_text_and_memory(
                        teacher_states[layer_idx].to(dtype=dtype),
                        text_positions,
                        h,
                        text_mask,
                        image_positions,
                        vision_states,
                        image_mask,
                    )
                with torch.no_grad():
                    visual_mass_override = compute_qwen3vl_visual_attention_mass_batched(
                        language_model,
                        layer_idx,
                        mass_full_state,
                        full_position_ids,
                        text_positions,
                        image_positions,
                        full_mask,
                        text_mask,
                        image_mask,
                        reduce_heads=(
                            "none"
                            if sidecar_module.output_mode
                            in {
                                "factorized_native_head_o",
                                "factorized_native_head_o_pure",
                                "factorized_native_head_o_residual",
                            }
                            else "mean"
                        ),
                    ).detach()
        op_start = mark_start()
        if not sidecar_layer_active:
            pred_delta = residual_from_cached_coeff(sidecar_module, cached_anchor_coeff, layer_tensor, h)
        else:
            if args.sidecar_visual_kv_source == "qwen_native":
                if _vis_pos_ids is None:
                    raise RuntimeError("visual position ids are required for qwen_native visual K/V")
                _call_kv = qwen_native_visual_kv(
                    language_model,
                    layer_idx,
                    vision_states,
                    _vis_pos_ids,
                    ~image_mask,
                )
            elif args.sidecar_visual_kv_source == "qwen_trainable_native":
                if _vis_pos_ids is None:
                    raise RuntimeError("visual position ids are required for qwen_trainable_native visual K/V")
                _call_kv = qwen_trainable_native_visual_kv(
                    sidecar_module,
                    language_model,
                    layer_idx,
                    vision_states,
                    _vis_pos_ids,
                    ~image_mask,
                )
            elif args.sidecar_visual_kv_source == "qwen_anchor_trainable":
                if _vis_pos_ids is None:
                    raise RuntimeError("visual position ids are required for qwen_anchor_trainable visual K/V")
                _call_kv = qwen_anchor_trainable_visual_kv(
                    sidecar_module,
                    language_model,
                    layer_idx,
                    vision_states,
                    _vis_pos_ids,
                    ~image_mask,
                )
            elif args.sidecar_visual_kv_source == "qwen_first_layer":
                _call_kv = _first_layer_visual_kv
            elif args.sidecar_visual_kv_source == "qwen_first_layer_film":
                if _first_layer_visual_kv is None:
                    raise RuntimeError("qwen_first_layer_film requires precomputed layer-0 visual K/V")
                _call_kv = qwen_first_layer_film_visual_kv(sidecar_module, _first_layer_visual_kv, layer_idx)
            else:
                _call_kv = _rope_visual_kv if _rope_visual_kv is not None else None
            _call_pos = _rope_text_pos_emb if _rope_text_pos_emb is not None else None
            _call_vis = None if _call_kv is not None else vision_states
            _call_mask = None if _call_kv is not None else ~image_mask
            if getattr(sidecar_module, "visual_transform_mode", "none") == "latent_compressor":
                _call_mask = None
            query_override = None
            if args.sidecar_query_source == "qwen_native":
                query_override = qwen_native_sidecar_query(language_model, layer_idx, h, text_position_ids)
            elif args.sidecar_query_source == "qwen_trainable_native":
                query_override = qwen_trainable_native_sidecar_query(
                    sidecar_module,
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                )
            if args.state_tokens > 0:
                pred_delta, sidecar_state, coeff = sidecar(
                    h,
                    _call_vis,
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=_call_kv,
                    vision_padding_mask=_call_mask,
                    text_attention=text_attention,
                    visual_mass=visual_mass_override,
                    output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                    text_attention_heads=text_attention_heads,
                    position_embeddings=_call_pos,
                    visual_position_embeddings=_dynamic_visual_pos_emb,
                    query_states=query_override,
                    initial_hidden_states=initial_text_hidden,
                    return_state=True,
                    return_coefficients=True,
                )
            else:
                pred_delta, coeff = sidecar(
                    h,
                    _call_vis,
                    layer_tensor,
                    visual_kv=_call_kv,
                    vision_padding_mask=_call_mask,
                    text_attention=text_attention,
                    visual_mass=visual_mass_override,
                    output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                    text_attention_heads=text_attention_heads,
                    position_embeddings=_call_pos,
                    visual_position_embeddings=_dynamic_visual_pos_emb,
                    query_states=query_override,
                    initial_hidden_states=initial_text_hidden,
                    return_coefficients=True,
                )
            cached_anchor_coeff = coeff
        mark("sidecar_s", op_start)
        if layer_idx in effect_layers:
            if args.effect_target == "teacher_visual":
                full_effect_state = scatter_batched_positions(
                    teacher_states[layer_idx].to(dtype=dtype),
                    text_positions,
                    h,
                    text_mask,
                )
            else:
                full_effect_state = build_full_from_text_and_memory(
                    teacher_states[layer_idx].to(dtype=dtype),
                    text_positions,
                    h,
                    text_mask,
                    image_positions,
                    vision_states,
                    image_mask,
                )
            with torch.no_grad():
                op_start = mark_start()
                target_delta = compute_qwen3vl_attention_effect_batched(
                    language_model,
                    layer_idx,
                    full_effect_state,
                    h,
                    full_position_ids,
                    text_position_ids,
                    text_positions,
                    full_mask,
                    text_mask,
                ).detach()
                mark("effect_target_s", op_start)
                target_mass = None
                if (
                    args.lambda_mass > 0.0
                    and sidecar_module.output_mode
                    in {
                        "factorized_native_head_o",
                        "factorized_native_head_o_pure",
                        "factorized_native_head_o_residual",
                    }
                    and args.factorized_mass_mode == "learned"
                ):
                    target_mass = compute_qwen3vl_visual_attention_mass_batched(
                        language_model,
                        layer_idx,
                        full_effect_state,
                        full_position_ids,
                        text_positions,
                        image_positions,
                        full_mask,
                        text_mask,
                        image_mask,
                        reduce_heads="none",
                    ).detach()
            layer_loss_weight = late_loss_weight if layer_idx in late_loss_layers else 1.0
            layer_effect = masked_nmse(pred_delta, target_delta, sidecar_token_mask)
            layer_effect_cos = masked_cos(pred_delta, target_delta, sidecar_token_mask)
            layer_effect_rms = masked_rms_abs(pred_delta, target_delta, sidecar_token_mask)
            effect_terms.append(layer_effect)
            effect_loss_terms.append(layer_effect)
            effect_loss_weights.append(layer_loss_weight)
            effect_cos_terms.append(layer_effect_cos)
            effect_cos_loss_terms.append(1.0 - layer_effect_cos)
            effect_cos_loss_weights.append(layer_loss_weight)
            effect_rms_terms.append(layer_effect_rms)
            effect_rms_loss_terms.append(layer_effect_rms)
            effect_rms_loss_weights.append(layer_loss_weight)
            valid = sidecar_token_mask.bool()
            pred_effect_rms_terms.append(pred_delta.float()[valid].pow(2).mean().sqrt())
            target_effect_rms_terms.append(target_delta.float()[valid].pow(2).mean().sqrt())
            if target_mass is not None:
                pred_mass = sidecar_module.last_visual_mass
                if pred_mass is None:
                    raise RuntimeError("mass supervision requested but Sidecar did not expose visual mass")
                if pred_mass.shape != target_mass.shape:
                    raise ValueError(
                        f"pred_mass shape {tuple(pred_mass.shape)} != target_mass shape {tuple(target_mass.shape)}"
                    )
                layer_mass = masked_mass_bce(
                    pred_mass,
                    target_mass,
                    sidecar_token_mask,
                    args.mass_positive_threshold,
                    args.mass_positive_weight,
                )
                mass_terms.append(layer_mass)
                mass_loss_terms.append(layer_mass)
                mass_loss_weights.append(layer_loss_weight)
                mass_mse_terms.append(masked_mass_mse(pred_mass, target_mass, sidecar_token_mask))
                mass_pred_mean_terms.append(pred_mass.float()[valid].mean())
                mass_target_mean_terms.append(target_mass.float()[valid].mean())
                target_valid = target_mass.float()[valid]
                positive = target_valid > float(args.mass_positive_threshold)
                mass_target_pos_rate_terms.append(positive.float().mean())
                if bool(positive.any()):
                    mass_target_pos_mean_terms.append(target_valid[positive].mean())
                else:
                    mass_target_pos_mean_terms.append(target_valid.new_zeros(()))
                mass_target_max_terms.append(target_valid.max())
        if text_attention is None:
            op_start = mark_start()
            h = run_qwen3vl_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                pred_delta.masked_fill(~sidecar_token_mask.unsqueeze(-1), 0.0),
                padding_mask=text_padding_mask,
            )
        else:
            op_start = mark_start()
            h = run_qwen3vl_layer_text_from_attention_output(
                language_model,
                layer_idx,
                h,
                text_attention,
                pred_delta.masked_fill(~sidecar_token_mask.unsqueeze(-1), 0.0),
            )
        mark("block_tail_s", op_start)
        state_idx = layer_idx + 1
        if state_idx in trajectory_layers:
            target_h = teacher_text_states[state_idx].to(dtype=dtype)
            pred_h = language_model.norm(h) if state_idx == num_layers else h
            layer_loss_weight = late_loss_weight if layer_idx in late_loss_layers else 1.0
            layer_traj = masked_directional_mse(pred_h, target_h, text_mask)
            layer_traj_rms = masked_rms_abs(pred_h, target_h, text_mask)
            traj_terms.append(layer_traj)
            traj_loss_terms.append(layer_traj)
            traj_loss_weights.append(layer_loss_weight)
            traj_rms_terms.append(layer_traj_rms)
            traj_rms_loss_terms.append(layer_traj_rms)
            traj_rms_loss_weights.append(layer_loss_weight)

    student_logits = teacher_model.lm_head(language_model.norm(h))
    effect_loss = torch.stack(effect_terms).mean() if effect_terms else student_logits.new_zeros(())
    effect_loss_weighted = weighted_mean(effect_loss_terms, effect_loss_weights, student_logits)
    effect_cos = torch.stack(effect_cos_terms).mean() if effect_cos_terms else student_logits.new_zeros(())
    effect_cos_loss = weighted_mean(effect_cos_loss_terms, effect_cos_loss_weights, student_logits)
    pred_effect_rms = torch.stack(pred_effect_rms_terms).mean() if pred_effect_rms_terms else student_logits.new_zeros(())
    target_effect_rms = (
        torch.stack(target_effect_rms_terms).mean() if target_effect_rms_terms else student_logits.new_zeros(())
    )
    effect_rms = torch.stack(effect_rms_terms).mean() if effect_rms_terms else student_logits.new_zeros(())
    effect_rms_weighted = weighted_mean(effect_rms_loss_terms, effect_rms_loss_weights, student_logits)
    mass_loss = torch.stack(mass_terms).mean() if mass_terms else student_logits.new_zeros(())
    mass_loss_weighted = weighted_mean(mass_loss_terms, mass_loss_weights, student_logits)
    mass_mse = torch.stack(mass_mse_terms).mean() if mass_mse_terms else student_logits.new_zeros(())
    mass_pred_mean = torch.stack(mass_pred_mean_terms).mean() if mass_pred_mean_terms else student_logits.new_zeros(())
    mass_target_mean = torch.stack(mass_target_mean_terms).mean() if mass_target_mean_terms else student_logits.new_zeros(())
    mass_target_pos_rate = (
        torch.stack(mass_target_pos_rate_terms).mean() if mass_target_pos_rate_terms else student_logits.new_zeros(())
    )
    mass_target_pos_mean = (
        torch.stack(mass_target_pos_mean_terms).mean() if mass_target_pos_mean_terms else student_logits.new_zeros(())
    )
    mass_target_max = torch.stack(mass_target_max_terms).mean() if mass_target_max_terms else student_logits.new_zeros(())
    traj_loss = torch.stack(traj_terms).mean() if traj_terms else student_logits.new_zeros(())
    traj_loss_weighted = weighted_mean(traj_loss_terms, traj_loss_weights, student_logits)
    traj_rms = torch.stack(traj_rms_terms).mean() if traj_rms_terms else student_logits.new_zeros(())
    traj_rms_weighted = weighted_mean(traj_rms_loss_terms, traj_rms_loss_weights, student_logits)
    logit_kl = masked_topk_kl(student_logits, teacher_logits, text_ids, answer_mask, args.temperature, 1024)
    ce_loss = answer_token_ce(student_logits, text_ids, answer_mask)
    main_loss_scale = 0.0 if int(global_step) < int(args.effect_start_step) else 1.0
    mass_weight = (
        args.lambda_mass
        if main_loss_scale == 0.0 or args.lambda_mass_after_effect_start is None
        else args.lambda_mass_after_effect_start
    )
    loss = (
        main_loss_scale * args.lambda_effect * effect_loss_weighted
        + main_loss_scale * args.lambda_effect_cos * effect_cos_loss
        + main_loss_scale * args.lambda_effect_rms * effect_rms_weighted
        + mass_weight * mass_loss_weighted
        + main_loss_scale * args.lambda_trajectory * traj_loss_weighted
        + main_loss_scale * args.lambda_trajectory_rms * traj_rms_weighted
        + main_loss_scale * args.lambda_logit * logit_kl
        + main_loss_scale * args.lambda_ce * ce_loss
    )
    metrics: dict[str, float | int | str] = {
        "loss": float(loss.detach()),
        "effect": float(effect_loss.detach()),
        "effect_weighted": float(effect_loss_weighted.detach()),
        "effect_cos": float(effect_cos.detach()),
        "effect_cos_loss_weighted": float(effect_cos_loss.detach()),
        "effect_rms": float(effect_rms.detach()),
        "effect_rms_weighted": float(effect_rms_weighted.detach()),
        "mass": float(mass_loss.detach()),
        "mass_weighted": float(mass_loss_weighted.detach()),
        "mass_mse": float(mass_mse.detach()),
        "mass_pred_mean": float(mass_pred_mean.detach()),
        "mass_target_mean": float(mass_target_mean.detach()),
        "mass_target_pos_rate": float(mass_target_pos_rate.detach()),
        "mass_target_pos_mean": float(mass_target_pos_mean.detach()),
        "mass_target_max": float(mass_target_max.detach()),
        "mass_weight": float(mass_weight),
        "main_loss_scale": float(main_loss_scale),
        "trajectory": float(traj_loss.detach()),
        "trajectory_weighted": float(traj_loss_weighted.detach()),
        "trajectory_rms": float(traj_rms.detach()),
        "trajectory_rms_weighted": float(traj_rms_weighted.detach()),
        "logit_kl": float(logit_kl.detach()),
        "ce": float(ce_loss.detach()),
        "pred_effect_rms": float(pred_effect_rms.detach()),
        "target_effect_rms": float(target_effect_rms.detach()),
        "teacher_forced": int(teacher_mix >= 1.0),
        "teacher_mix": float(teacher_mix),
        "effect_layers_count": int(len(effect_layers)),
        "trajectory_layers_count": int(len(trajectory_layers)),
        "late_loss_layers_count": int(len(late_loss_layers)),
        "late_loss_weight": float(late_loss_weight),
        "text_tokens": float(text_mask.sum().item()) / max(1, len(rows)),
        "image_tokens": float(image_mask.sum().item()) / max(1, len(rows)),
        "batch_size": int(len(rows)),
        "image": image_paths[0],
    }
    metrics.update(timings)
    return loss, metrics


def main() -> None:
    args = parse_args()
    debug_print("parsed args")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        if args.distributed_engine == "deepspeed":
            import deepspeed

            deepspeed.init_distributed(dist_backend=args.dist_backend)
        else:
            dist.init_process_group(args.dist_backend)
        debug_print("distributed initialized")
        if args.required_world_size > 1 and dist.get_world_size() != args.required_world_size:
            raise RuntimeError(f"expected {args.required_world_size} ranks, got {dist.get_world_size()}")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    rank_id = dist.get_rank() if distributed_is_initialized() else 0
    world_size = dist.get_world_size() if distributed_is_initialized() else 1
    torch.manual_seed(args.seed + rank_id)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if distributed_is_initialized():
        debug_print("before output barrier")
        distributed_barrier(device)
        debug_print("after output barrier")
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "train_metrics.jsonl"
    if is_rank0():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    debug_print("before load_frozen_qwen3vl")
    processor, teacher_model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    debug_print("after load_frozen_qwen3vl")
    language_model = get_language_model(teacher_model)
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    debug_print("before sidecar init")
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        state_tokens=args.state_tokens,
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=args.output_mode in {"residual", "factorized_lowrank"},
        reader_mlp_ratio=args.reader_mlp_ratio,
        reader_activation=args.reader_activation,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=args.shared_basis,
        output_mode=args.output_mode,
        corrector_layers=args.corrector_layers,
        corrector_dim=args.corrector_dim,
        block_corrector_groups=args.block_corrector_groups,
        block_corrector_dim=args.block_corrector_dim,
        use_rope=args.use_rope,
        layer_condition_mode=args.layer_condition_mode,
        visual_transform_mode=args.visual_transform_mode,
        reader_mode=args.reader_mode,
        latent_tokens=args.latent_tokens,
        visual_transform_rank=args.visual_transform_rank,
        visual_transform_activation=args.visual_transform_activation,
    ).to(device=device, dtype=dtype)
    debug_print("after sidecar init")
    if args.sidecar_query_source == "qwen_trainable_native" or args.sidecar_visual_kv_source == "qwen_trainable_native":
        attach_qwen_trainable_native_qkv(
            sidecar,
            language_model,
            train_query=args.sidecar_query_source == "qwen_trainable_native",
            train_visual_kv=args.sidecar_visual_kv_source == "qwen_trainable_native",
        )
    if args.sidecar_visual_kv_source == "qwen_anchor_trainable":
        attach_qwen_anchor_trainable_visual_kv(sidecar, language_model, args.qwen_anchor_layers)
    if args.sidecar_visual_kv_source == "qwen_first_layer_film":
        attach_qwen_first_layer_film_visual_kv(sidecar, language_model)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = sidecar.load_state_dict(state_dict, strict=False)
        if is_rank0():
            print(
                f"initialized sidecar from {args.init_checkpoint} "
                f"missing={list(missing)} unexpected={list(unexpected)}",
                flush=True,
            )
    elif args.native_qkv_init != "none":
        initialize_sidecar_from_qwen_native_qkv(sidecar, language_model, args.native_qkv_init)
        if is_rank0():
            print(
                f"initialized sidecar q/k/v from Qwen native projections mode={args.native_qkv_init}",
                flush=True,
            )
    if args.output_init_std > 0.0 and not args.init_checkpoint:
        if sidecar.coeff_head is not None:
            nn.init.normal_(sidecar.coeff_head.weight, mean=0.0, std=float(args.output_init_std))
        if sidecar.visual_full_head is not None:
            nn.init.normal_(sidecar.visual_full_head.weight, mean=0.0, std=float(args.output_init_std))
    if sidecar.layer_adapter_up is not None:
        for up in sidecar.layer_adapter_up:
            nn.init.normal_(up.weight, mean=0.0, std=float(args.output_init_std))
    if args.sidecar_query_source == "qwen_native":
        for param in sidecar.q_proj.parameters():
            param.requires_grad_(False)
    if args.sidecar_visual_kv_source in {"qwen_native", "qwen_first_layer", "qwen_first_layer_film", "qwen_anchor_trainable"}:
        for param in sidecar.k_proj.parameters():
            param.requires_grad_(False)
        for param in sidecar.v_proj.parameters():
            param.requires_grad_(False)
    sync_module_state(sidecar)
    sidecar.train()
    basis_param_ids = {id(sidecar.basis)} if isinstance(sidecar.basis, nn.Parameter) else set()
    main_params = [p for p in sidecar.parameters() if p.requires_grad and id(p) not in basis_param_ids]
    trainable: list[dict[str, object]] = [{"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if isinstance(sidecar.basis, nn.Parameter):
        trainable.append(
            {
                "params": [sidecar.basis],
                "lr": args.lr * args.basis_lr_mult,
                "weight_decay": args.weight_decay,
            }
        )

    ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.grad_clip
    ds_config.setdefault("optimizer", {"type": "AdamW", "params": {}})
    ds_config["optimizer"].setdefault("params", {})
    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
    debug_print("before optimizer engine init")
    if distributed and args.distributed_engine == "deepspeed":
        import deepspeed

        sidecar_engine, _, _, _ = deepspeed.initialize(model=sidecar, model_parameters=trainable, config=ds_config)
    elif distributed:
        optimizer = torch.optim.AdamW(trainable)
        sidecar_engine = TorchGradSyncEngine(sidecar, optimizer)
    else:
        optimizer = torch.optim.AdamW(trainable)
        sidecar_engine = SimpleEngine(sidecar, optimizer)
    sidecar_engine.train()
    debug_print("after optimizer engine init")
    base_lrs = [float(group.get("lr", args.lr)) for group in optimizer_param_groups(sidecar_engine)]

    if args.effect_target == "native_vprefix" and args.visual_memory_mode != "vprefix":
        raise ValueError("--effect-target native_vprefix requires --visual-memory-mode vprefix")

    dataset = JsonlDataset(args.data, max_samples=args.max_samples, start_index=args.start_index, decode_images=False)
    if len(dataset) == 0:
        raise RuntimeError("empty training dataset")
    order_start = time.perf_counter()
    sample_order = build_training_order(args, dataset, world_size)
    order_s = time.perf_counter() - order_start

    wandb_run = None
    if args.wandb and is_rank0() and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            mode=args.wandb_mode,
            config={**vars(args), "dataset_size": len(dataset), "world_size": world_size},
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    if is_rank0():
        print(
            f"qwen deepspeed train world_size={world_size} grad_accum={args.gradient_accumulation_steps} "
            f"micro_batch={args.micro_batch_size_per_gpu} "
            f"effective_global_batch={world_size * args.gradient_accumulation_steps * args.micro_batch_size_per_gpu} "
            f"max_steps={args.max_steps} visual_memory_mode={args.visual_memory_mode} "
            f"effect_target={args.effect_target} output_mode={args.output_mode} "
            f"batch_sampling={args.batch_sampling} order_s={order_s:.2f} "
            f"deepspeed_config={args.deepspeed_config}",
            flush=True,
        )
        (output_dir / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2), encoding="utf-8")

    global_step = 0
    if args.resume_deepspeed_dir:
        resume_tag = args.resume_deepspeed_tag or None
        loaded_path, _client_state = sidecar_engine.load_checkpoint(args.resume_deepspeed_dir, tag=resume_tag)
        if loaded_path is None:
            raise RuntimeError(f"failed to resume DeepSpeed checkpoint from {args.resume_deepspeed_dir} tag={resume_tag}")
        tag_for_step = args.resume_deepspeed_tag
        if not tag_for_step:
            latest_path = Path(args.resume_deepspeed_dir) / "latest"
            if latest_path.exists():
                tag_for_step = latest_path.read_text(encoding="utf-8").strip()
        if tag_for_step.startswith("step") and tag_for_step[4:].isdigit():
            global_step = int(tag_for_step[4:])
        if is_rank0():
            print(
                f"resumed DeepSpeed checkpoint path={loaded_path} tag={tag_for_step or resume_tag} "
                f"global_step={global_step}",
                flush=True,
            )
    while global_step < args.max_steps:
        step_start = time.perf_counter()
        accum_metrics: dict[str, float] = {}
        last_extra: dict[str, int | str] = {}
        current_lr = set_engine_lr(sidecar_engine, base_lrs, lr_multiplier(args, global_step))
        for micro_idx in range(args.gradient_accumulation_steps):
            sample_base = (
                args.start_index
                + global_step * args.gradient_accumulation_steps * world_size * args.micro_batch_size_per_gpu
                + micro_idx * world_size * args.micro_batch_size_per_gpu
                + rank_id * args.micro_batch_size_per_gpu
            )
            rows = [
                dataset[sample_order[(sample_base + row_offset) % len(sample_order)]]
                for row_offset in range(args.micro_batch_size_per_gpu)
            ]
            op_start = time.perf_counter()
            loss, metrics = compute_loss_for_row(
                args,
                processor,
                teacher_model,
                language_model,
                sidecar_engine.module,
                rows,
                global_step,
                device,
                dtype,
                num_layers,
            )
            if args.profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            loss_forward_s = time.perf_counter() - op_start
            op_start = time.perf_counter()
            sidecar_engine.backward(loss)
            if args.profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_s = time.perf_counter() - op_start
            op_start = time.perf_counter()
            sidecar_engine.step()
            if args.profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            optimizer_s = time.perf_counter() - op_start
            metrics["loss_forward_s"] = loss_forward_s
            metrics["backward_s"] = backward_s
            metrics["optimizer_s"] = optimizer_s
            metrics["lr"] = current_lr
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + float(value) / float(args.gradient_accumulation_steps)
                else:
                    last_extra[key] = value

        global_step += 1
        if global_step % args.log_every == 0:
            reduced = reduce_metric_dict(accum_metrics, device)
            metrics: dict[str, float | int | str] = {
                "step": int(global_step),
                **reduced,
                "lambda_effect": float(args.lambda_effect),
                "lambda_effect_cos": float(args.lambda_effect_cos),
                "lambda_effect_rms": float(args.lambda_effect_rms),
                "lambda_trajectory": float(args.lambda_trajectory),
                "lambda_trajectory_rms": float(args.lambda_trajectory_rms),
                "lambda_logit": float(args.lambda_logit),
                "lambda_ce": float(args.lambda_ce),
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": float(args.warmup_ratio),
                "min_lr_ratio": float(args.min_lr_ratio),
                "global_batch": int(world_size * args.gradient_accumulation_steps * args.micro_batch_size_per_gpu),
                "sec_per_step": float(time.perf_counter() - step_start),
            }
            if is_rank0():
                metrics.update(last_extra)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                print(
                    f"step={global_step} loss={float(metrics['loss']):.6f} effect={float(metrics['effect']):.6f} "
                    f"effect_cos={float(metrics['effect_cos']):.6f} effect_rms={float(metrics['effect_rms']):.6f} "
                    f"mass={float(metrics['mass']):.6f} mass_mse={float(metrics['mass_mse']):.6f} "
                    f"mass_pred={float(metrics['mass_pred_mean']):.6f} mass_tgt={float(metrics['mass_target_mean']):.6f} "
                    f"mass_pos={float(metrics['mass_target_pos_rate']):.4f} "
                    f"mass_pos_mean={float(metrics['mass_target_pos_mean']):.6f} "
                    f"mass_max={float(metrics['mass_target_max']):.6f} "
                    f"trajectory={float(metrics['trajectory']):.6f} "
                    f"traj_rms={float(metrics['trajectory_rms']):.6f} logit_kl={float(metrics['logit_kl']):.6f} "
                    f"pred_effect_rms={float(metrics['pred_effect_rms']):.6f} "
                    f"target_effect_rms={float(metrics['target_effect_rms']):.6f} "
                    f"teacher_mix={float(metrics['teacher_mix']):.3f} "
                    f"effect_layers={int(float(metrics['effect_layers_count']))} "
                    f"trajectory_layers={int(float(metrics['trajectory_layers_count']))} "
                    f"global_batch={metrics['global_batch']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"train/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
        if is_rank0() and global_step % args.save_every == 0:
            sidecar_engine.save_checkpoint(str(output_dir / "deepspeed"), tag=f"step{global_step}")
            save_sidecar_checkpoint(sidecar_engine.module, output_dir / f"attention_sidecar_step{global_step}.pt", args, global_step)
        elif global_step % args.save_every == 0:
            sidecar_engine.save_checkpoint(str(output_dir / "deepspeed"), tag=f"step{global_step}")

    sidecar_engine.save_checkpoint(str(output_dir / "deepspeed"), tag="final")
    if is_rank0():
        save_sidecar_checkpoint(sidecar_engine.module, output_dir / f"attention_sidecar_step{global_step}.pt", args, global_step)
        save_sidecar_checkpoint(sidecar_engine.module, output_dir / "attention_sidecar_final.pt", args, global_step)
    if distributed_is_initialized():
        distributed_barrier(device)
        dist.destroy_process_group()
    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
