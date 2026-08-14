"""Unified trainer for LLaVA KV adapters and Qwen3-VL visual-delta adapters."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler

from src.data import OPDDataset, VQADataset, collate_fn
from src.model import (
    PerLayerKVAdapter,
    QWEN_VISUAL_DELTA_MODES,
    QwenVisualDeltaAdapter,
    dtype_from_name,
    extract_vision_kv,
    gather_batched_positions,
    get_qwen_text_image_positions,
    load_frozen_llava,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    qwen_position_ids,
    qwen_visual_delta_logits,
    resolve_row_image_path,
    student_forward_with_visual_kv,
    teacher_forward,
)


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 1024,
) -> torch.Tensor:
    """KL divergence on top-K teacher logits."""
    k = min(topk, teacher_logits.shape[-1])
    _, indices = teacher_logits.topk(k, dim=-1)
    t_topk = teacher_logits.gather(-1, indices)
    s_topk = student_logits.gather(-1, indices)
    t_prob = F.softmax(t_topk, dim=-1)
    s_logprob = F.log_softmax(s_topk, dim=-1)
    return F.kl_div(s_logprob, t_prob, reduction="batchmean")




class JsonlDataset:
    def __init__(self, path: str | Path, max_samples: int | None = None, start_index: int = 0) -> None:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if start_index:
            rows = rows[int(start_index) :]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified trainer for LLaVA KV adapters and Qwen3-VL visual-delta adapters.")
    parser.add_argument("--model-kind", choices=("llava", "qwen"), default="llava")

    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/pixmo_ama_train.jsonl")
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving LLaVA image paths in JSONL")
    parser.add_argument("--image-root", default="", help="Root for resolving Qwen image paths in JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--warmup-start-lr-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")

    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--distributed-engine", choices=("deepspeed", "torch_grad_sync"), default="torch_grad_sync")
    parser.add_argument("--required-world-size", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-answer-tokens", type=int, default=9999)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-layers", default="22,23", help="Comma-separated ViT layer indices")
    parser.add_argument("--bottleneck-dim", type=int, default=0)
    parser.add_argument("--concat-source", action="store_true")
    parser.add_argument("--dataset-type", default="vqa", choices=("vqa", "opd"))

    parser.add_argument("--output-mode", choices=QWEN_VISUAL_DELTA_MODES, default="native_visual_kv_split")
    parser.add_argument("--visual-adapter-rank", type=int, default=128)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--reader-activation", choices=("gelu", "silu", "swiglu", "situ_glu"), default="situ_glu")
    parser.add_argument("--lambda-logit", type=float, default=4.0)
    parser.add_argument("--lambda-trajectory", type=float, default=0.5)
    parser.add_argument("--lambda-kv-mse", type=float, default=0.0)
    parser.add_argument("--loss-normalization", choices=("token", "sample"), default="sample")
    parser.add_argument("--trajectory-layers", default="4,8,12,16,20,24,28,32,36")
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--batch-sampling", choices=("sequential", "pixel_bucket"), default="pixel_bucket")
    parser.add_argument("--pixel-bucket-size", type=int, default=512)
    parser.add_argument("--pixel-area-cache", default="")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


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


def reduce_metric_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed_is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, values)}


class SimpleEngine:
    def __init__(self, module: nn.Module, optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
        self.module = module
        self.optimizer = optimizer
        self.grad_clip = float(grad_clip)

    def train(self) -> None:
        self.module.train()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in self.module.parameters() if p.requires_grad], self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, output_dir: str, tag: str) -> None:
        if distributed_is_initialized() and dist.get_rank() != 0:
            return
        path = Path(output_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"module": self.module.state_dict()}, path / "model_states.pt")


class TorchGradSyncEngine(SimpleEngine):
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


def optimizer_param_groups(engine: object) -> list[dict[str, Any]]:
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
    total_steps = max(1, int(args.max_steps))
    warmup_steps = max(0, min(int(math.ceil(float(args.warmup_ratio) * total_steps)), total_steps))
    start_ratio = max(0.0, float(args.warmup_start_lr_ratio))
    min_ratio = max(0.0, float(args.min_lr_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        progress = float(step + 1) / float(warmup_steps)
        return start_ratio + (1.0 - start_ratio) * progress
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_engine_lr(engine: object, base_lrs: list[float], multiplier: float) -> float:
    groups = optimizer_param_groups(engine)
    if not groups:
        return 0.0
    for group, base_lr in zip(groups, base_lrs, strict=False):
        group["lr"] = float(base_lr) * float(multiplier)
    return float(groups[0].get("lr", 0.0))


def parse_trajectory_layers(spec: str, num_layers: int) -> set[int]:
    if spec.strip() == "all":
        return set(range(1, num_layers + 1))
    out: set[int] = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if 1 <= value <= num_layers:
            out.add(value)
    return out


def masked_directional_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalization: str = "token",
) -> torch.Tensor:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    pred_norm = pred.float() / pred.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    target_norm = target.float() / target.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    per_token = (pred_norm - target_norm).pow(2).mean(dim=-1)
    valid_2d = valid.squeeze(-1)
    if normalization == "sample":
        per_sample = (per_token * valid_2d).sum(dim=1) / valid_2d.sum(dim=1).clamp_min(1.0)
        has_valid = valid_2d.sum(dim=1) > 0
        if int(has_valid.sum().item()) == 0:
            return pred.new_zeros(())
        return per_sample[has_valid].mean()
    return (per_token * valid_2d).sum() / valid_2d.sum().clamp_min(1.0)


def masked_topk_kl_stats(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_ids: Tensor,
    answer_mask: Tensor,
    temperature: float,
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = student_logits.shape[0]
    if k <= 0:
        return student_logits.new_zeros(()), student_logits.new_zeros((batch,)), student_logits.new_zeros((batch,))
    shift_mask = answer_mask[:, 1:].bool()
    answer_counts = shift_mask.sum(dim=1).to(device=student_logits.device, dtype=torch.float32)
    valid_count = answer_counts.sum()
    if int(valid_count.item()) == 0:
        return student_logits.new_zeros(()), student_logits.new_zeros((batch,)), answer_counts
    shift_student = student_logits[:, :-1][shift_mask].float()
    shift_teacher = teacher_logits[:, :-1][shift_mask].float()
    shift_targets = target_ids[:, 1:][shift_mask]
    k_eff = min(k, shift_teacher.shape[-1])
    topk = torch.topk(shift_teacher, k=k_eff, dim=-1).indices
    target_idx = shift_targets.unsqueeze(-1)
    if k_eff == shift_teacher.shape[-1]:
        gather_idx = topk
    else:
        target_in_topk = topk.eq(target_idx).any(dim=-1, keepdim=True)
        gather_idx = torch.where(target_in_topk, topk, torch.cat([topk[..., :-1], target_idx], dim=-1))
    gathered_teacher = torch.gather(shift_teacher, dim=-1, index=gather_idx) / temperature
    gathered_student = torch.gather(shift_student, dim=-1, index=gather_idx) / temperature
    kl = F.kl_div(
        F.log_softmax(gathered_student, dim=-1),
        F.softmax(gathered_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    kl = kl * (temperature * temperature)
    batch_ids = (
        torch.arange(batch, device=student_logits.device)
        .unsqueeze(1)
        .expand_as(shift_mask)[shift_mask]
    )
    per_sample_sum = student_logits.new_zeros((batch,), dtype=torch.float32)
    per_sample_sum.scatter_add_(0, batch_ids, kl.to(device=student_logits.device, dtype=torch.float32))
    per_sample = per_sample_sum / answer_counts.clamp_min(1.0)
    token_mean = kl.sum() / valid_count.float().clamp_min(1.0)
    return token_mean.to(dtype=student_logits.dtype), per_sample.to(dtype=student_logits.dtype), answer_counts


def masked_topk_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_ids: Tensor,
    answer_mask: Tensor,
    temperature: float,
    k: int,
    *,
    normalization: str = "token",
) -> tuple[Tensor, Tensor, Tensor]:
    token_mean, per_sample, answer_counts = masked_topk_kl_stats(
        student_logits,
        teacher_logits,
        target_ids,
        answer_mask,
        temperature,
        k,
    )
    if normalization == "sample":
        has_answer = answer_counts > 0
        if int(has_answer.sum().item()) == 0:
            return token_mean, per_sample, answer_counts
        return per_sample[has_answer].mean(), per_sample, answer_counts
    return token_mean, per_sample, answer_counts


def image_pixel_area(row: dict[str, Any], image_root: Path | None) -> int:
    path = resolve_row_image_path(row, image_root)
    try:
        with Image.open(path) as image:
            width, height = image.size
        return max(1, int(width) * int(height))
    except Exception:
        return 0


def load_or_build_pixel_areas(args: argparse.Namespace, dataset: JsonlDataset) -> list[int]:
    cache_path = Path(args.pixel_area_cache) if args.pixel_area_cache else Path(str(args.data) + ".pixel_areas.json")

    def read_cache() -> list[int] | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            areas = payload.get("areas") if isinstance(payload, dict) else payload
            if isinstance(areas, list) and len(areas) == len(dataset):
                return [int(x) for x in areas]
        except Exception:
            pass
        return None

    cached = read_cache()
    if cached is not None:
        return cached

    if distributed_is_initialized() and not is_rank0():
        distributed_barrier()
        cached = read_cache()
        if cached is None:
            raise RuntimeError(f"rank0 did not create a valid pixel-area cache: {cache_path}")
        return cached

    root = Path(args.image_root) if str(args.image_root).strip() else None
    areas = [image_pixel_area(row, root) for row in dataset.rows]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps({"data": str(args.data), "count": len(dataset), "areas": areas}), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception:
        pass
    if distributed_is_initialized():
        distributed_barrier()
    return areas


def build_training_order(args: argparse.Namespace, dataset: JsonlDataset, world_size: int) -> list[int]:
    if args.batch_sampling == "sequential":
        return list(range(len(dataset)))
    min_bucket = max(1, int(args.micro_batch_size_per_gpu) * max(1, int(world_size)))
    bucket_size = max(min_bucket, int(args.pixel_bucket_size))
    areas = load_or_build_pixel_areas(args, dataset)
    sized = sorted((area, idx) for idx, area in enumerate(areas))
    buckets = [[idx for _, idx in sized[start : start + bucket_size]] for start in range(0, len(sized), bucket_size)]
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    order: list[int] = []
    for bucket_idx in torch.randperm(len(buckets), generator=generator).tolist():
        bucket = buckets[int(bucket_idx)]
        if len(bucket) > 1:
            perm = torch.randperm(len(bucket), generator=generator).tolist()
            order.extend(bucket[int(i)] for i in perm)
        else:
            order.extend(bucket)
    return order


def save_checkpoint(adapter: QwenVisualDeltaAdapter, output_path: Path, args: argparse.Namespace, global_step: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "args": vars(args),
            "global_step": int(global_step),
            "adapter_config": {
                "type": "qwen_visual_delta",
                "output_mode": args.output_mode,
                "visual_adapter_rank": args.visual_adapter_rank,
            },
        },
        output_path,
    )


def trainable_parameters_for_mode(adapter: QwenVisualDeltaAdapter) -> list[nn.Parameter]:
    for param in adapter.parameters():
        param.requires_grad_(False)
    for name, param in adapter.named_parameters():
        if name.startswith("visual_adapter_"):
            param.requires_grad_(True)
        if adapter.mode == "native_visual_kv_split" and (
            name == "gate"
            or name.startswith("reader_")
            or name.startswith("mass_head")
        ):
            param.requires_grad_(True)
    return [param for param in adapter.parameters() if param.requires_grad]


def compute_loss_for_rows(
    args: argparse.Namespace,
    processor: Any,
    model: torch.nn.Module,
    adapter: QwenVisualDeltaAdapter,
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    inputs, text_ids, answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
        processor,
        rows,
        Path(args.image_root) if str(args.image_root).strip() else None,
        device,
        include_answers=True,
    )
    assert text_ids is not None and answer_mask is not None
    trajectory_layers = parse_trajectory_layers(args.trajectory_layers, num_layers)
    with torch.no_grad():
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        full_position_ids = qwen_position_ids(model, inputs)
        text_positions, _, _, text_mask, _, _ = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_text_states = {
            state_idx: gather_batched_positions(teacher.hidden_states[state_idx].detach(), text_positions, text_mask).detach()
            for state_idx in sorted(trajectory_layers)
            if state_idx < len(teacher.hidden_states)
        }
        teacher_logits = gather_batched_positions(teacher.logits.detach(), text_positions, text_mask)

    student_logits, student_text_mask, student_states = qwen_visual_delta_logits(
        model,
        adapter,
        inputs,
        initial_hidden=teacher.hidden_states[0].detach(),
        position_ids=full_position_ids,
        collect_states=True,
        collect_state_indices=trajectory_layers,
    )
    if student_states is None:
        raise RuntimeError("student states were not collected")
    if student_text_mask.shape != text_mask.shape:
        raise RuntimeError("student/teacher text masks differ")

    traj_terms = []
    for state_idx in sorted(trajectory_layers):
        if state_idx not in teacher_text_states or state_idx >= len(student_states):
            continue
        pred_h = student_states[state_idx]
        if pred_h.numel() == 0:
            continue
        if state_idx == num_layers:
            pred_h = model.model.language_model.norm(pred_h)
        traj_terms.append(
            masked_directional_mse(
                pred_h,
                teacher_text_states[state_idx].to(dtype=pred_h.dtype),
                text_mask,
                normalization=args.loss_normalization,
            )
        )
    trajectory = torch.stack(traj_terms).mean() if traj_terms else student_logits.new_zeros(())
    logit_kl, per_sample_kl, answer_counts = masked_topk_kl(
        student_logits,
        teacher_logits,
        text_ids,
        answer_mask,
        args.temperature,
        args.kl_topk,
        normalization=args.loss_normalization,
    )
    kv_mse = student_logits.new_zeros(())
    loss = args.lambda_logit * logit_kl + args.lambda_trajectory * trajectory + args.lambda_kv_mse * kv_mse

    mass_mean = student_logits.new_zeros(())
    if adapter.last_visual_mass is not None:
        valid = text_mask.to(device=adapter.last_visual_mass.device).bool()
        mass_mean = adapter.last_visual_mass.float()[valid].mean()
    sources = [str(row.get("source", "")) for row in rows]
    ocrvqa_mask = torch.tensor(
        [source == "ocrvqa" for source in sources],
        device=student_logits.device,
        dtype=torch.bool,
    )
    pixmo_mask = torch.tensor(
        [source == "pixmo_clean" for source in sources],
        device=student_logits.device,
        dtype=torch.bool,
    )

    def masked_mean(values: Tensor, valid_mask: Tensor) -> Tensor:
        valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
        if int(valid_mask.sum().item()) == 0:
            return values.new_zeros(())
        return values[valid_mask].float().mean()

    answer_total = answer_counts.float().sum().clamp_min(1.0)
    ocr_answer_tokens = answer_counts.float()[ocrvqa_mask].sum() if int(ocrvqa_mask.sum().item()) else answer_counts.new_zeros(())
    pixmo_answer_tokens = answer_counts.float()[pixmo_mask].sum() if int(pixmo_mask.sum().item()) else answer_counts.new_zeros(())
    metrics: dict[str, float | int | str] = {
        "loss": float(loss.detach()),
        "logit_kl": float(logit_kl.detach()),
        "logit_kl_ocrvqa": float(masked_mean(per_sample_kl.detach(), ocrvqa_mask).detach()),
        "logit_kl_pixmo_clean": float(masked_mean(per_sample_kl.detach(), pixmo_mask).detach()),
        "trajectory": float(trajectory.detach()),
        "kv_mse": float(kv_mse.detach()),
        "visual_mass": float(mass_mean.detach()),
        "text_tokens": float(text_mask.sum().item()) / max(1, len(rows)),
        "answer_tokens": float(answer_counts.sum().item()) / max(1, len(rows)),
        "ocrvqa_frac": float(ocrvqa_mask.float().mean().item()),
        "pixmo_clean_frac": float(pixmo_mask.float().mean().item()),
        "ocrvqa_answer_token_frac": float((ocr_answer_tokens / answer_total).item()),
        "pixmo_clean_answer_token_frac": float((pixmo_answer_tokens / answer_total).item()),
        "image": image_paths[0],
        "batch_size": int(len(rows)),
    }
    return loss, metrics



def run_llava(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device=str(device))
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    source_layers = [int(x) for x in args.source_layers.split(",")]
    language_model = model.model.language_model
    vision_config = getattr(model.config, "vision_config", None)
    source_dim = int(getattr(vision_config, "hidden_size", 1024))
    num_llm_layers = len(language_model.layers)
    num_heads = getattr(language_model.config, "num_key_value_heads", language_model.config.num_attention_heads)
    head_dim = language_model.config.hidden_size // language_model.config.num_attention_heads
    if is_main:
        print(f"LLM: {num_llm_layers} layers, {num_heads} heads, head_dim={head_dim}")
    adapter_config = {
        "num_llm_layers": num_llm_layers,
        "num_source_layers": len(source_layers),
        "source_dim": source_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "concat_source": args.concat_source,
        "use_activation": False,
        "source_layers": source_layers,
    }
    adapter = PerLayerKVAdapter(
        num_llm_layers=num_llm_layers,
        num_source_layers=len(source_layers),
        source_dim=source_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        bottleneck_dim=args.bottleneck_dim,
        concat_source=args.concat_source,
    )

    trainable_params = sum(p.numel() for p in adapter.parameters())
    if is_main:
        print(f"Adapter trainable params: {trainable_params / 1e6:.2f}M")

    wandb_run = None
    if is_main and args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.output_dir).name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "adapter_trainable_params": trainable_params,
                "adapter_trainable_millions": trainable_params / 1e6,
                "world_size": world_size,
            },
        )

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt["state_dict"])
        if is_main:
            print(f"Loaded init checkpoint: {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    with open(args.deepspeed_config, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    engine, optimizer, _, _ = deepspeed.initialize(
        model=adapter,
        optimizer=optimizer,
        config=ds_config,
    )

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    min_lr = args.lr * args.min_lr_ratio

    def get_lr(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return args.lr * step_idx / max(warmup_steps, 1)
        progress = (step_idx - warmup_steps) / max(args.max_steps - warmup_steps, 1)
        return min_lr + (args.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    DatasetCls = OPDDataset if args.dataset_type == "opd" else VQADataset
    dataset = DatasetCls(
        args.data,
        processor,
        data_root=args.data_root,
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
    step = 0

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            B = input_ids.shape[0]
            pixel_values = batch["pixel_values"].to(device) if torch.is_tensor(batch["pixel_values"]) else batch["pixel_values"]
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            with torch.no_grad():
                image_sizes = batch.get("image_sizes")
                if isinstance(pixel_values, list):
                    # Variable crops: process per-sample
                    source_k_list, source_v_list, teacher_logits_list = [], [], []
                    for i in range(B):
                        pv_i = pixel_values[i].unsqueeze(0).to(device)
                        sk, sv = extract_vision_kv(model, pv_i, source_layer_indices=source_layers)
                        source_k_list.append(sk)
                        source_v_list.append(sv)
                        isz = image_sizes[i:i+1].to(device) if image_sizes is not None and torch.is_tensor(image_sizes) else None
                        tl = teacher_forward(
                            model,
                            input_ids[i:i+1],
                            pv_i,
                            attention_mask=attention_mask[i:i+1],
                            image_sizes=isz,
                        )
                        teacher_logits_list.append(tl)
                    source_k = source_v = teacher_logits = None
                else:
                    if image_sizes is not None and torch.is_tensor(image_sizes):
                        image_sizes = image_sizes.to(device)
                    source_k, source_v = extract_vision_kv(model, pixel_values, source_layer_indices=source_layers)
                    teacher_logits = teacher_forward(model, input_ids, pixel_values, attention_mask, image_sizes=image_sizes)
                    source_k_list = source_v_list = teacher_logits_list = None

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for i in range(B):
                single_ids = input_ids[i:i+1]
                if source_k_list is not None:
                    single_sk = source_k_list[i]
                    single_sv = source_v_list[i]
                else:
                    single_sk = source_k[i:i+1]
                    single_sv = source_v[i:i+1]

                student_logits = student_forward_with_visual_kv(
                    model,
                    single_ids,
                    engine.module,
                    single_sk,
                    single_sv,
                    image_token_id,
                    attention_mask=attention_mask[i:i+1],
                )

                # prompt_len is text-only (image tokens excluded)
                text_prompt_len = int(prompt_lens[i].item())
                num_text = student_logits.shape[1]

                # Student: text-only, answer starts at text_prompt_len-1 (causal: predict next)
                s_start = max(0, text_prompt_len - 1)
                # Exclude pad tokens from answer range
                n_image_tokens = (single_ids[0] == image_token_id).sum().item()
                actual_len = int(attention_mask[i].sum().item()) - n_image_tokens
                s_end = min(s_start + args.max_answer_tokens, actual_len)
                s_answer = student_logits[0, s_start:s_end]

                # Teacher: full sequence with image tokens expanded
                n_image_tokens = (single_ids[0] == image_token_id).sum().item()
                t_start = max(0, n_image_tokens + text_prompt_len - 1)
                t_actual_len = int(attention_mask[i].sum().item())
                if teacher_logits_list is not None:
                    t_logits_i = teacher_logits_list[i][0]
                    t_end = min(t_start + args.max_answer_tokens, t_actual_len, t_logits_i.shape[0])
                    t_answer = t_logits_i[t_start:t_end]
                else:
                    t_end = min(t_start + args.max_answer_tokens, t_actual_len, teacher_logits.shape[1])
                    t_answer = teacher_logits[i, t_start:t_end]

                if s_answer.shape[0] > 0 and t_answer.shape[0] > 0:
                    min_len = min(s_answer.shape[0], t_answer.shape[0])
                    kl = topk_kl_loss(
                        s_answer[:min_len].float(),
                        t_answer[:min_len].float(),
                        topk=args.kl_topk,
                    )
                    total_loss = total_loss + kl

            loss = total_loss / B
            engine.backward(loss)
            current_lr = get_lr(step + 1)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            engine.step()

            if is_main and step % args.log_every == 0:
                item = {"step": step, "loss": float(loss.item()), "lr": current_lr}
                print(json.dumps(item), flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(item) + "\n")
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": item["loss"],
                        "train/kl_loss": item["loss"],
                        "train/lr": current_lr,
                        "train/step": step,
                    }, step=step)

            if is_main and step > 0 and step % args.save_every == 0:
                ckpt_path = Path(args.output_dir) / f"step_{step}.pt"
                torch.save({
                    "state_dict": engine.module.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "adapter_config": adapter_config,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

            step += 1

        if step >= args.max_steps:
            break

    if is_main:
        final_path = Path(args.output_dir) / "final.pt"
        torch.save({
            "state_dict": engine.module.state_dict(),
            "step": step,
            "args": vars(args),
            "adapter_config": adapter_config,
        }, final_path)
        print(f"Training complete. Final checkpoint: {final_path}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


def run_qwen(args: argparse.Namespace) -> None:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        if args.distributed_engine == "deepspeed":
            import deepspeed

            deepspeed.init_distributed(dist_backend=args.dist_backend)
        else:
            dist.init_process_group(args.dist_backend)
        if args.required_world_size > 1 and dist.get_world_size() != args.required_world_size:
            raise RuntimeError(f"expected {args.required_world_size} ranks, got {dist.get_world_size()}")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    rank_id = dist.get_rank() if distributed_is_initialized() else 0
    world_size = dist.get_world_size() if distributed_is_initialized() else 1
    configure_torch_runtime()
    random.seed(args.seed + rank_id)
    torch.manual_seed(args.seed + rank_id)
    dtype = dtype_from_name(args.dtype)

    output_dir = Path(args.output_dir)
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "train_metrics.jsonl"
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    distributed_barrier(device)

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = model.model.language_model
    num_layers = len(language_model.layers)
    adapter = QwenVisualDeltaAdapter.from_language_model(
        language_model,
        mode=args.output_mode,
        reader_mlp_ratio=args.reader_mlp_ratio,
        reader_activation=args.reader_activation,
        visual_adapter_rank=args.visual_adapter_rank,
    ).to(device=device, dtype=dtype)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
        if is_rank0():
            print(f"loaded init checkpoint {args.init_checkpoint} missing={list(missing)} unexpected={list(unexpected)}", flush=True)

    trainable_params = trainable_parameters_for_mode(adapter)
    if not trainable_params:
        raise RuntimeError(f"no trainable parameters for output_mode={args.output_mode}")
    sync_module_state(adapter)
    adapter.train()
    trainable_count = sum(param.numel() for param in trainable_params)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if distributed and args.distributed_engine == "deepspeed":
        import deepspeed

        ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
        ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size_per_gpu
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["gradient_clipping"] = args.grad_clip
        engine, _, _, _ = deepspeed.initialize(model=adapter, model_parameters=trainable_params, optimizer=optimizer, config=ds_config)
    elif distributed:
        engine = TorchGradSyncEngine(adapter, optimizer, args.grad_clip)
    else:
        engine = SimpleEngine(adapter, optimizer, args.grad_clip)
    engine.train()
    base_lrs = [float(group.get("lr", args.lr)) for group in optimizer_param_groups(engine)]

    dataset = JsonlDataset(args.data, max_samples=args.max_samples, start_index=args.start_index)
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
            name=args.wandb_run_name or output_dir.parent.name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            mode=args.wandb_mode,
            config={**vars(args), "dataset_size": len(dataset), "world_size": world_size, "trainable_params": trainable_count},
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    if is_rank0():
        print(
            f"qwen visual-delta train mode={args.output_mode} world_size={world_size} "
            f"micro_batch={args.micro_batch_size_per_gpu} grad_accum={args.gradient_accumulation_steps} "
            f"max_steps={args.max_steps} trainable={trainable_count/1e6:.2f}M "
            f"lr={args.lr} scheduler={args.lr_scheduler} warmup={args.warmup_ratio} "
            f"batch_sampling={args.batch_sampling} order_s={order_s:.2f}",
            flush=True,
        )

    global_step = 0
    while global_step < args.max_steps:
        step_start = time.perf_counter()
        current_lr = set_engine_lr(engine, base_lrs, lr_multiplier(args, global_step))
        accum_metrics: dict[str, float] = {}
        last_extra: dict[str, str | int] = {}
        for micro_idx in range(args.gradient_accumulation_steps):
            sample_base = (
                global_step * args.gradient_accumulation_steps * world_size * args.micro_batch_size_per_gpu
                + micro_idx * world_size * args.micro_batch_size_per_gpu
                + rank_id * args.micro_batch_size_per_gpu
            )
            rows = [
                dataset[sample_order[(sample_base + offset) % len(sample_order)]]
                for offset in range(args.micro_batch_size_per_gpu)
            ]
            forward_start = time.perf_counter()
            loss, metrics = compute_loss_for_rows(args, processor, model, engine.module, rows, device, dtype, num_layers)
            loss = loss / float(args.gradient_accumulation_steps)
            engine.backward(loss)
            metrics["loss_forward_s"] = time.perf_counter() - forward_start
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + float(value) / float(args.gradient_accumulation_steps)
                else:
                    last_extra[key] = value
        engine.step()
        global_step += 1

        if global_step % args.log_every == 0:
            accum_metrics["lr"] = current_lr
            accum_metrics["sec_per_step"] = time.perf_counter() - step_start
            reduced = reduce_metric_dict(accum_metrics, device)
            payload: dict[str, float | int | str] = {
                "step": int(global_step),
                **reduced,
                "lambda_logit": float(args.lambda_logit),
                "lambda_trajectory": float(args.lambda_trajectory),
                "lambda_kv_mse": float(args.lambda_kv_mse),
                "global_batch": int(world_size * args.gradient_accumulation_steps * args.micro_batch_size_per_gpu),
            }
            if is_rank0():
                payload.update(last_extra)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(
                    f"step={global_step} loss={float(payload['loss']):.6f} "
                    f"logit_kl={float(payload['logit_kl']):.6f} trajectory={float(payload['trajectory']):.6f} "
                    f"kv_mse={float(payload['kv_mse']):.6f} visual_mass={float(payload['visual_mass']):.6f} "
                    f"lr={float(payload['lr']):.3e} global_batch={payload['global_batch']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"train/{key}": value for key, value in payload.items() if isinstance(value, (int, float))})

        if global_step % args.save_every == 0:
            engine.save_checkpoint(str(output_dir / "optimizer"), tag=f"step{global_step}")
            if is_rank0():
                save_checkpoint(engine.module, output_dir / f"qwen_visual_delta_step{global_step}.pt", args, global_step)

    engine.save_checkpoint(str(output_dir / "optimizer"), tag="final")
    if is_rank0():
        save_checkpoint(engine.module, output_dir / f"qwen_visual_delta_step{global_step}.pt", args, global_step)
        save_checkpoint(engine.module, output_dir / "qwen_visual_delta_final.pt", args, global_step)
    distributed_barrier(device)
    if distributed_is_initialized():
        dist.destroy_process_group()
    if wandb_run is not None:
        import wandb

        wandb.finish()


def main() -> None:
    args = parse_args()
    if args.model_kind == "qwen":
        if args.lr_scheduler is None:
            args.lr_scheduler = "constant"
        if args.warmup_ratio is None:
            args.warmup_ratio = 0.0
        run_qwen(args)
        return
    if args.lr_scheduler is None:
        args.lr_scheduler = "cosine"
    if args.warmup_ratio is None:
        args.warmup_ratio = 0.2
    if not args.init_checkpoint:
        args.init_checkpoint = None
    run_llava(args)


if __name__ == "__main__":
    main()
