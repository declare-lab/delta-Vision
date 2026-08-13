from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value.detach().float()
    reduced = value.detach().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def normalized_mse(pred: Tensor, target: Tensor) -> Tensor:
    pred = pred.float()
    target = target.float()
    return (pred - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-6)


def rms_normalize(hidden: Tensor) -> Tensor:
    hidden = hidden.float()
    rms = hidden.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return hidden / rms


def directional_mse(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(rms_normalize(pred), rms_normalize(target))


def residual_cosine(pred: Tensor, target: Tensor) -> Tensor:
    return F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1, eps=1e-6).mean()


def rms_scale_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_rms = pred.float().pow(2).mean(dim=-1).sqrt()
    target_rms = target.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return (torch.log(pred_rms.clamp_min(1e-6) / target_rms).pow(2)).mean()


def rms_scale_abs_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_rms = pred.float().pow(2).mean(dim=-1).sqrt()
    target_rms = target.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return torch.log(pred_rms.clamp_min(1e-6) / target_rms).abs().mean()


def hidden_cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    return 1.0 - residual_cosine(pred, target)


def scheduled_sidecar_scale(args: Namespace, global_step: int) -> float:
    scale = float(args.sidecar_scale)
    if args.sidecar_scale_warmup_steps > 0:
        scale *= min(1.0, max(0.0, float(global_step) / float(args.sidecar_scale_warmup_steps)))
    return scale


def scheduled_rollout_teacher_mix(args: Namespace, global_step: int) -> float:
    if args.rollout_teacher_forcing_steps > 0 and global_step < args.rollout_teacher_forcing_steps:
        return float(args.rollout_teacher_mix_start)
    mixed_steps = int(args.rollout_mixed_steps)
    if mixed_steps <= 0:
        return 0.0
    mixed_progress = global_step - int(args.rollout_teacher_forcing_steps)
    if mixed_progress < 0:
        return float(args.rollout_teacher_mix_start)
    if mixed_progress >= mixed_steps:
        return 0.0
    start = float(args.rollout_teacher_mix_start)
    return start * (1.0 - float(mixed_progress) / float(mixed_steps))


def trajectory_weight(layer_state_idx: int, late_start: int, late_weight: float) -> float:
    if late_weight == 1.0:
        return 1.0
    return float(late_weight) if int(layer_state_idx) >= int(late_start) else 1.0


def sample_stratified_layers(num_layers: int, count: int, device: torch.device) -> list[int]:
    if count <= 0:
        return []
    if count > num_layers:
        raise ValueError("effect-layers-per-sample cannot exceed num_layers")
    band_count = 4
    band_size = max(1, num_layers // band_count)
    bands = [(i * band_size, min(num_layers, (i + 1) * band_size)) for i in range(band_count)]
    chosen: list[int] = []
    per_band = max(1, count // len(bands))
    for start, end in bands:
        if len(chosen) >= count or start >= end:
            break
        take = min(per_band, count - len(chosen), end - start)
        perm = torch.randperm(end - start, device=device)[:take].tolist()
        chosen.extend(start + int(i) for i in perm)
    while len(chosen) < count:
        candidate = int(torch.randint(0, num_layers, (1,), device=device).item())
        if candidate not in chosen:
            chosen.append(candidate)
    return sorted(chosen)


def sample_stratified_layers_from_candidates(candidates: list[int], count: int, device: torch.device) -> list[int]:
    if count <= 0 or not candidates:
        return []
    unique_candidates = sorted(set(int(x) for x in candidates))
    if count >= len(unique_candidates):
        return unique_candidates
    candidate_tensor = torch.tensor(unique_candidates, device=device, dtype=torch.long)
    positions = sample_stratified_layers(len(unique_candidates), count, device)
    return sorted(int(candidate_tensor[pos].item()) for pos in positions)


def save_plain_checkpoint(model_engine: Any, output_dir: Path, tag: str, args: Namespace) -> None:
    if not is_rank0():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model_engine.module.state_dict().items()}
    torch.save({"state_dict": state_dict, "args": vars(args)}, output_dir / f"attention_sidecar_{tag}.pt")


def mixed_full_hidden_with_student_text(
    teacher_full_hidden: Tensor,
    text_positions: Tensor,
    student_text_hidden: Tensor,
) -> Tensor:
    mixed = teacher_full_hidden.clone()
    mixed[:, text_positions.to(device=mixed.device), :] = student_text_hidden.detach().to(
        device=mixed.device,
        dtype=mixed.dtype,
    )
    return mixed


def init_wandb(args: Namespace, dataset_size: int) -> Any | None:
    if not args.wandb or not is_rank0() or args.wandb_mode == "disabled":
        return None
    import wandb

    config = vars(args).copy()
    config["dataset_size"] = dataset_size
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    return run


def validate_training_paths(args: Namespace) -> None:
    for name in ("data", "model_path", "basis", "deepspeed_config"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    if args.init_checkpoint is not None and not Path(args.init_checkpoint).exists():
        raise FileNotFoundError(f"missing init_checkpoint: {args.init_checkpoint}")
    if "mmstar" in str(args.basis).lower() and not args.allow_mmstar_basis:
        raise ValueError("refusing to train PixMo Sidecar with an MMStar attention basis")
