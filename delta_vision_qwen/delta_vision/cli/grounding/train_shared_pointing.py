#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from delta_vision.data import JsonlDataset, collate_rows
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.modeling import build_rollout_model, image_token_id, load_frozen_llava, load_rollout_checkpoint
from delta_vision.grounding.pointing import (
    PointHead,
    parse_points,
    point_distance,
    point_target_loss,
    save_point_checkpoint,
    sidecar_point_hidden_batch,
)
from delta_vision.training.utils import is_rank0, reduce_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fine-tune shared Sidecar for PixMo-Points grounding.")
    parser.add_argument("--data", default="data/pixmo_points/train_50000.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2_coeff.json")
    parser.add_argument("--init-sidecar-checkpoint", required=True)
    parser.add_argument("--init-point-checkpoint", default="")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/shared_pointing")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--point-head-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--basis-lr-mult", type=float, default=0.2)
    parser.add_argument("--point-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device, distributed


class PointingTrainModel(nn.Module):
    def __init__(self, rollout_model: torch.nn.Module, point_head: torch.nn.Module) -> None:
        super().__init__()
        self.rollout_model = rollout_model
        self.point_head = point_head


def save_checkpoint(
    output_dir: Path,
    tag: str,
    engine: Any,
    args: argparse.Namespace,
    global_step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(output_dir / "deepspeed"), tag=tag)
    if not is_rank0():
        return
    sidecar_model = engine.module.rollout_model
    head = engine.module.point_head
    torch.save(
        {
            "state_dict": sidecar_model.state_dict(),
            "point_head": head.state_dict(),
            "global_step": global_step,
            "args": vars(args),
        },
        output_dir / f"shared_pointing_{tag}.pt",
    )
    save_point_checkpoint(
        output_dir,
        tag,
        head,
        {"global_step": global_step, "args": vars(args)},
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    device, distributed = setup_distributed(args)
    torch.manual_seed(args.seed + (dist.get_rank() if distributed else 0))
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)

    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    img_token = image_token_id(model, processor)
    image_seq_length = int(getattr(model.config, "image_seq_length", 0))
    if image_seq_length <= 0:
        raise RuntimeError("LLaVA config must define image_seq_length")
    rollout_model = build_rollout_model(args, dtype, device)
    load_rollout_checkpoint(rollout_model, args.init_sidecar_checkpoint)
    point_head = PointHead(args.hidden_size, args.point_head_dim).to(device=device)
    if args.init_point_checkpoint:
        checkpoint = torch.load(args.init_point_checkpoint, map_location="cpu")
        point_head.load_state_dict(checkpoint["point_head"], strict=True)

    dataset = JsonlDataset(args.data)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
        pin_memory=True,
    )
    effective_max_steps = int(args.max_steps) if int(args.max_steps) > 0 else len(dataloader)

    train_model = PointingTrainModel(rollout_model, point_head).to(device=device)
    train_model.train()

    basis_param_ids = {id(train_model.rollout_model.sidecar.basis)}
    sidecar_params = [
        param
        for param in train_model.rollout_model.parameters()
        if param.requires_grad and id(param) not in basis_param_ids
    ]
    basis_params = [
        param
        for param in train_model.rollout_model.parameters()
        if param.requires_grad and id(param) in basis_param_ids
    ]
    trainable = [
        {"params": sidecar_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": train_model.point_head.parameters(), "lr": args.point_lr, "weight_decay": args.weight_decay},
    ]
    if basis_params:
        trainable.append(
            {
                "params": basis_params,
                "lr": args.lr * args.basis_lr_mult,
                "weight_decay": args.weight_decay,
            }
        )
    ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.grad_clip
    engine, _, _, _ = deepspeed.initialize(model=train_model, model_parameters=trainable, config=ds_config)

    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (output_dir / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2), encoding="utf-8")
        metrics_path = output_dir / "train_metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        print(
            f"pointing train: samples={len(dataset)} batch={args.batch_size} max_steps={effective_max_steps} "
            f"init={args.init_sidecar_checkpoint}",
            flush=True,
        )

    global_step = 0
    running_loss = 0.0
    running_dist = 0.0
    while global_step < effective_max_steps:
        if distributed and sampler is not None:
            sampler.set_epoch(global_step)
        for rows in dataloader:
            if global_step >= effective_max_steps:
                break
            hidden = sidecar_point_hidden_batch(
                processor,
                model,
                language_model,
                engine.module.rollout_model,
                img_token,
                image_seq_length,
                rows,
                device,
                dtype,
            )
            preds = engine.module.point_head(hidden)
            batch_loss_terms: list[torch.Tensor] = []
            batch_dist_terms: list[torch.Tensor] = []
            for pred, row in zip(preds, rows, strict=True):
                points = parse_points(row["points"]).to(device=device)
                batch_loss_terms.append(point_target_loss(pred, points))
                with torch.no_grad():
                    batch_dist_terms.append(point_distance(pred.detach(), points))
            loss = torch.stack(batch_loss_terms).mean()
            engine.backward(loss)
            engine.step()

            step_loss = torch.stack(batch_loss_terms).mean().detach()
            step_dist = torch.stack(batch_dist_terms).mean().detach()
            if distributed:
                step_loss = reduce_mean(step_loss)
                step_dist = reduce_mean(step_dist)
            running_loss += float(step_loss.cpu())
            running_dist += float(step_dist.cpu())
            global_step += 1

            if global_step % args.log_every == 0 and is_rank0():
                denom = float(args.log_every)
                metrics = {
                    "step": global_step,
                    "point_loss": running_loss / denom,
                    "point_distance_xy100": running_dist / denom,
                    "lr": args.lr,
                    "point_lr": args.point_lr,
                }
                print(
                    f"step={global_step} point_loss={metrics['point_loss']:.6f} "
                    f"point_dist={metrics['point_distance_xy100']:.3f}",
                    flush=True,
                )
                with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                running_loss = 0.0
                running_dist = 0.0
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(output_dir, f"step{global_step}", engine, args, global_step)

    save_checkpoint(output_dir, "final", engine, args, global_step)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
