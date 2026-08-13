#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from collections import UserDict
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from delta_vision.data import JsonlDataset
from delta_vision.evaluation.metrics import masked_kl
from delta_vision.models.gemma4 import (
    build_gemma4_initial_context,
    compute_gemma4_attention_effect_batched,
    gather_batched_positions,
    gemma4_attention_masks,
    gemma4_attention_output,
    gemma4_lm_logits,
    gemma4_text_attention_masks,
    get_gemma4_language_model,
    get_gemma4_text_image_positions,
    load_frozen_gemma4,
    prepare_gemma4_batch_inputs,
    run_gemma4_layer_text_from_attention_output,
    scatter_batched_positions,
)
from delta_vision.models.llava import dtype_from_name
from delta_vision.models.sidecar import DeltaVisionModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Gemma4 delta-vision trainer.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--shared-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--effect-layers-per-sample", type=int, default=8)
    parser.add_argument("--trajectory-layers", default="4,8,12,16,20,24,28,32,36,40,42")
    parser.add_argument("--teacher-force-steps", type=int, default=500)
    parser.add_argument("--teacher-force-mix", type=float, default=0.3)
    parser.add_argument("--teacher-mix-end-step", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--basis-lr-mult", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-effect", type=float, default=0.5)
    parser.add_argument("--lambda-effect-cos", type=float, default=0.0)
    parser.add_argument("--lambda-trajectory", type=float, default=4.0)
    parser.add_argument("--lambda-trajectory-rms", type=float, default=0.0)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--output-init-std", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2_coeff.json")
    parser.add_argument("--required-world-size", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-vision")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


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


def parse_trajectory_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(1, num_layers + 1))
    return {int(item) for item in spec.split(",") if item.strip() and 0 < int(item) <= num_layers}


def sample_effect_layers(num_layers: int, count: int, device: torch.device) -> set[int]:
    count = min(int(count), int(num_layers))
    if count <= 0:
        return set()
    return {int(x.item()) for x in torch.randperm(num_layers, device=device)[:count]}


def teacher_mix_ratio(args: argparse.Namespace, global_step: int) -> float:
    force_steps = int(args.teacher_force_steps)
    mix_end = int(args.teacher_mix_end_step)
    force_mix = max(0.0, min(1.0, float(args.teacher_force_mix)))
    if global_step < force_steps:
        return force_mix
    if mix_end <= force_steps or global_step >= mix_end:
        return 0.0
    progress = float(global_step - force_steps) / float(max(1, mix_end - force_steps))
    return max(0.0, min(1.0, force_mix * (1.0 - progress)))


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


def compute_loss_for_rows(
    args: argparse.Namespace,
    processor: object,
    teacher_model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    rows: list[dict[str, object]],
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    inputs, answer_lens, image_paths = prepare_gemma4_batch_inputs(
        processor,
        rows,
        "image",
        "question",
        "answer",
        None,
        device,
    )
    with torch.no_grad():
        teacher = teacher_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden0, full_position_ids, per_layer_inputs, image_mask_raw, full_attention_masks = build_gemma4_initial_context(
            teacher_model,
            inputs,
        )
    text_positions, image_positions, text_position_ids, text_mask, image_mask, answer_mask = get_gemma4_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        image_mask_raw,
        full_position_ids,
        answer_lens,
    )
    del image_positions
    teacher_states = [state.detach() for state in teacher.hidden_states]
    teacher_text_states = [gather_batched_positions(state, text_positions, text_mask).detach() for state in teacher_states]
    teacher_logits = gather_batched_positions(teacher.logits, text_positions, text_mask).detach()

    # Re-gather visual memory from the raw image mask positions.
    image_pos = torch.zeros_like(image_mask, dtype=torch.long)
    for batch_idx in range(inputs["input_ids"].shape[0]):
        pos = torch.nonzero(image_mask_raw[batch_idx].bool(), as_tuple=False).flatten()
        image_pos[batch_idx, : pos.numel()] = pos.to(device=image_pos.device)
    vision_states = gather_batched_positions(hidden0, image_pos, image_mask).to(dtype=dtype)

    text_attention_masks = gemma4_text_attention_masks(language_model, teacher_text_states[0].to(dtype=dtype), text_mask, text_position_ids)
    full_shared_kv: UserDict = UserDict()
    text_shared_kv: UserDict = UserDict()
    visual_kv = sidecar.prepare_visual_kv(vision_states, ~image_mask)
    h = teacher_text_states[0].to(dtype=dtype).masked_fill(~text_mask.unsqueeze(-1), 0.0)
    effect_layers = sample_effect_layers(num_layers, args.effect_layers_per_sample, device)
    trajectory_layers = parse_trajectory_layers(args.trajectory_layers, num_layers)
    effect_terms: list[torch.Tensor] = []
    effect_cos_terms: list[torch.Tensor] = []
    traj_terms: list[torch.Tensor] = []
    traj_rms_terms: list[torch.Tensor] = []
    teacher_mix = teacher_mix_ratio(args, global_step)
    for layer_idx in range(num_layers):
        if teacher_mix > 0.0:
            teacher_h = teacher_text_states[layer_idx].to(dtype=dtype).masked_fill(~text_mask.unsqueeze(-1), 0.0)
            h = teacher_h if teacher_mix >= 1.0 else h * (1.0 - teacher_mix) + teacher_h * teacher_mix
            h = h.masked_fill(~text_mask.unsqueeze(-1), 0.0)

        with torch.no_grad():
            full_attn = gemma4_attention_output(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                full_attention_masks,
                full_position_ids,
                full_shared_kv,
            )
        text_attention = gemma4_attention_output(
            language_model,
            layer_idx,
            h,
            text_attention_masks,
            text_position_ids,
            text_shared_kv,
        )
        layer_tensor = torch.full((h.shape[0],), layer_idx, device=device, dtype=torch.long)
        pred_delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv)
        if layer_idx in effect_layers:
            with torch.no_grad():
                full_effect_state = scatter_batched_positions(
                    teacher_states[layer_idx].to(dtype=dtype),
                    text_positions,
                    h.detach(),
                    text_mask,
                )
                target_shared = UserDict(dict(full_shared_kv))
                target_joint = gemma4_attention_output(
                    language_model,
                    layer_idx,
                    full_effect_state,
                    full_attention_masks,
                    full_position_ids,
                    target_shared,
                )
                target_delta = gather_batched_positions(target_joint, text_positions, text_mask).detach() - text_attention.detach()
            effect_terms.append(masked_nmse(pred_delta, target_delta, text_mask))
            effect_cos_terms.append(masked_cos(pred_delta, target_delta, text_mask))

        per_layer_input = None
        if per_layer_inputs is not None:
            per_layer_input_full = per_layer_inputs[:, :, layer_idx, :]
            per_layer_input = gather_batched_positions(per_layer_input_full, text_positions, text_mask).to(dtype=dtype)
        h = run_gemma4_layer_text_from_attention_output(
            language_model,
            layer_idx,
            h,
            text_attention,
            pred_delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
            per_layer_input,
        ).masked_fill(~text_mask.unsqueeze(-1), 0.0)
        state_idx = layer_idx + 1
        if state_idx in trajectory_layers:
            target_h = teacher_text_states[state_idx].to(dtype=dtype)
            pred_h = language_model.norm(h) if state_idx == num_layers else h
            traj_terms.append(masked_directional_mse(pred_h, target_h, text_mask))
            traj_rms_terms.append(masked_rms_abs(pred_h, target_h, text_mask))

    student_logits = gemma4_lm_logits(teacher_model, language_model.norm(h))
    effect_loss = torch.stack(effect_terms).mean() if effect_terms else student_logits.new_zeros(())
    effect_cos = torch.stack(effect_cos_terms).mean() if effect_cos_terms else student_logits.new_zeros(())
    traj_loss = torch.stack(traj_terms).mean() if traj_terms else student_logits.new_zeros(())
    traj_rms = torch.stack(traj_rms_terms).mean() if traj_rms_terms else student_logits.new_zeros(())
    logit_kl = masked_kl(student_logits, teacher_logits, answer_mask, args.temperature)
    loss = (
        args.lambda_effect * effect_loss
        + args.lambda_effect_cos * (1.0 - effect_cos)
        + args.lambda_trajectory * traj_loss
        + args.lambda_trajectory_rms * traj_rms
        + args.lambda_logit * logit_kl
    )
    return loss, {
        "loss": float(loss.detach()),
        "effect": float(effect_loss.detach()),
        "effect_cos": float(effect_cos.detach()),
        "trajectory": float(traj_loss.detach()),
        "trajectory_rms": float(traj_rms.detach()),
        "logit_kl": float(logit_kl.detach()),
        "teacher_mix": float(teacher_mix),
        "effect_layers_count": int(len(effect_layers)),
        "trajectory_layers_count": int(len(trajectory_layers)),
        "text_tokens": float(text_mask.sum().item()) / max(1, len(rows)),
        "image_tokens": float(image_mask.sum().item()) / max(1, len(rows)),
        "batch_size": int(len(rows)),
        "image": image_paths[0],
    }


def main() -> None:
    args = parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend=args.dist_backend)
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
        dist.barrier()
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "train_metrics.jsonl"

    processor, teacher_model = load_frozen_gemma4(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_gemma4_language_model(teacher_model)
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=True,
        reader_mlp_ratio=args.reader_mlp_ratio,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=args.shared_basis,
        output_mode="residual",
    ).to(device=device, dtype=dtype)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        sidecar.load_state_dict(state_dict, strict=False)
    elif args.output_init_std > 0:
        nn.init.normal_(sidecar.coeff_head.weight, mean=0.0, std=float(args.output_init_std))
        if sidecar.layer_adapter_up is not None:
            for up in sidecar.layer_adapter_up:
                nn.init.normal_(up.weight, mean=0.0, std=float(args.output_init_std))
    sidecar.train()

    basis_param_ids = {id(sidecar.basis)} if isinstance(sidecar.basis, nn.Parameter) else set()
    main_params = [p for p in sidecar.parameters() if p.requires_grad and id(p) not in basis_param_ids]
    trainable: list[dict[str, object]] = [{"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if isinstance(sidecar.basis, nn.Parameter):
        trainable.append({"params": [sidecar.basis], "lr": args.lr * args.basis_lr_mult, "weight_decay": args.weight_decay})
    ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.grad_clip
    ds_config.setdefault("optimizer", {"type": "AdamW", "params": {}})
    ds_config["optimizer"].setdefault("params", {})
    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
    sidecar_engine, _, _, _ = deepspeed.initialize(model=sidecar, model_parameters=trainable, config=ds_config)
    sidecar_engine.train()

    dataset = JsonlDataset(args.data, max_samples=args.max_samples, start_index=args.start_index, decode_images=False)
    if len(dataset) == 0:
        raise RuntimeError("empty training dataset")
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
            f"gemma4 deepspeed train world_size={world_size} micro_batch={args.micro_batch_size_per_gpu} "
            f"grad_accum={args.gradient_accumulation_steps} global_batch={world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps} "
            f"hidden={hidden_size} layers={num_layers} max_steps={args.max_steps}",
            flush=True,
        )

    global_step = 0
    while global_step < args.max_steps:
        step_start = time.perf_counter()
        accum: dict[str, float] = {}
        last_extra: dict[str, int | str] = {}
        for micro_idx in range(args.gradient_accumulation_steps):
            sample_base = (
                global_step * args.gradient_accumulation_steps * world_size * args.micro_batch_size_per_gpu
                + micro_idx * world_size * args.micro_batch_size_per_gpu
                + rank_id * args.micro_batch_size_per_gpu
            )
            rows = [dataset[(sample_base + offset) % len(dataset)] for offset in range(args.micro_batch_size_per_gpu)]
            loss, metrics = compute_loss_for_rows(
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
            sidecar_engine.backward(loss)
            sidecar_engine.step()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    accum[key] = accum.get(key, 0.0) + float(value) / float(args.gradient_accumulation_steps)
                else:
                    last_extra[key] = value
        global_step += 1
        if global_step % args.log_every == 0:
            reduced = reduce_metric_dict(accum, device)
            metrics_out: dict[str, float | int | str] = {
                "step": global_step,
                **reduced,
                "lambda_effect": float(args.lambda_effect),
                "lambda_effect_cos": float(args.lambda_effect_cos),
                "lambda_trajectory": float(args.lambda_trajectory),
                "lambda_trajectory_rms": float(args.lambda_trajectory_rms),
                "lambda_logit": float(args.lambda_logit),
                "global_batch": int(world_size * args.gradient_accumulation_steps * args.micro_batch_size_per_gpu),
                "sec_per_step": float(time.perf_counter() - step_start),
            }
            if is_rank0():
                metrics_out.update(last_extra)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_out, ensure_ascii=False) + "\n")
                print(
                    f"step={global_step} loss={float(metrics_out['loss']):.6f} "
                    f"effect={float(metrics_out['effect']):.6f} effect_cos={float(metrics_out['effect_cos']):.6f} "
                    f"trajectory={float(metrics_out['trajectory']):.6f} logit_kl={float(metrics_out['logit_kl']):.6f} "
                    f"teacher_mix={float(metrics_out['teacher_mix']):.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"train/{key}": value for key, value in metrics_out.items() if isinstance(value, (int, float))})
        if global_step % args.save_every == 0:
            sidecar_engine.save_checkpoint(str(output_dir / "deepspeed"), tag=f"step{global_step}")
            if is_rank0():
                save_sidecar_checkpoint(sidecar_engine.module, output_dir / f"attention_sidecar_step{global_step}.pt", args, global_step)

    sidecar_engine.save_checkpoint(str(output_dir / "deepspeed"), tag="final")
    if is_rank0():
        save_sidecar_checkpoint(sidecar_engine.module, output_dir / f"attention_sidecar_step{global_step}.pt", args, global_step)
        save_sidecar_checkpoint(sidecar_engine.module, output_dir / "attention_sidecar_final.pt", args, global_step)
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
