from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from delta_vision.data import JsonlDataset, collate_rows, parse_state_layers, parse_zero_based_layers
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.modeling import (
    assert_llava_layer_count,
    build_rollout_model,
    image_token_id as resolve_image_token_id,
    load_frozen_llava,
    load_rollout_checkpoint,
)
from delta_vision.runtime.rollout import prepare_batch_inputs, prepare_sample_inputs
from delta_vision.training.rollout_kd import rollout_losses_for_batch, rollout_losses_for_sample
from delta_vision.training.args import parse_args
from delta_vision.training.utils import (
    init_wandb,
    is_rank0,
    reduce_mean,
    sample_stratified_layers_from_candidates,
    save_plain_checkpoint,
    scheduled_rollout_teacher_mix,
    scheduled_sidecar_scale,
    validate_training_paths,
)


def sync_if_enabled(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    validate_training_paths(args)
    torch.set_float32_matmul_precision("high")
    local_rank = args.local_rank
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != args.required_world_size:
        raise RuntimeError(f"expected {args.required_world_size} distributed ranks, got {dist.get_world_size()}")

    torch.manual_seed(args.seed + dist.get_rank())
    device = torch.device("cuda", local_rank)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)

    processor, teacher_model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(teacher_model)
    assert_llava_layer_count(teacher_model, args.num_layers)
    image_token_id = resolve_image_token_id(teacher_model, processor)
    image_seq_length = int(getattr(teacher_model.config, "image_seq_length", 0))
    if image_seq_length <= 0:
        raise RuntimeError("teacher config must define a positive image_seq_length for Sidecar rollout")
    if args.batched_rollout and args.lambda_image_negative > 0.0:
        raise RuntimeError("batched rollout does not support image-negative loss; disable it or use single-sample rollout")
    if args.effect_loss_mode == "coeff":
        raise RuntimeError("shared mainline uses a trainable basis; use --effect-loss-mode residual")

    rollout_model = build_rollout_model(args, dtype)
    if args.init_checkpoint is not None:
        missing, unexpected, skipped, sliced = load_rollout_checkpoint(
            rollout_model,
            args.init_checkpoint,
            ignore_checkpoint_basis=args.ignore_checkpoint_basis,
            ignore_mismatched_checkpoint_shapes=args.ignore_mismatched_checkpoint_shapes,
            slice_mismatched_checkpoint_prefix=args.slice_mismatched_checkpoint_prefix,
        )
        if is_rank0():
            print(f"loaded init checkpoint: {args.init_checkpoint}", flush=True)
            if missing or unexpected:
                print(f"checkpoint load missing={missing} unexpected={unexpected}", flush=True)
            if args.slice_mismatched_checkpoint_prefix and sliced:
                print(f"checkpoint load sliced_mismatched={sliced}", flush=True)
            if args.ignore_mismatched_checkpoint_shapes and skipped:
                print(f"checkpoint load skipped_mismatched={skipped}", flush=True)
    elif is_rank0():
        print("training Sidecar from random initialization", flush=True)

    ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.grad_clip
    ds_config.setdefault("optimizer", {"type": "AdamW", "params": {}})
    ds_config["optimizer"].setdefault("params", {})
    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
    basis_param_ids = {id(rollout_model.sidecar.basis)} if isinstance(rollout_model.sidecar.basis, torch.nn.Parameter) else set()
    sidecar_params = [
        param
        for param in rollout_model.sidecar.parameters()
        if param.requires_grad and id(param) not in basis_param_ids
    ]
    basis_params = [
        param
        for param in rollout_model.sidecar.parameters()
        if param.requires_grad and id(param) in basis_param_ids
    ]
    trainable: list[dict[str, Any]] = []
    if sidecar_params:
        trainable.append({"params": sidecar_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if basis_params:
        trainable.append(
            {
                "params": basis_params,
                "lr": args.lr * args.basis_lr_mult,
                "weight_decay": args.weight_decay,
            }
        )
    engine, _, _, _ = deepspeed.initialize(model=rollout_model, model_parameters=trainable, config=ds_config)
    engine.train()
    if args.resume_deepspeed_dir:
        load_path, _ = engine.load_checkpoint(
            args.resume_deepspeed_dir,
            tag=args.resume_deepspeed_tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(
                f"failed to load DeepSpeed checkpoint dir={args.resume_deepspeed_dir} "
                f"tag={args.resume_deepspeed_tag}"
            )
        if is_rank0():
            print(
                f"loaded DeepSpeed checkpoint: {load_path} "
                f"resume_global_step={args.resume_global_step}",
                flush=True,
            )

    dataset = JsonlDataset(
        args.data,
        max_samples=args.max_samples,
        start_index=args.start_index,
        decode_images=args.decode_images_in_workers,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    trajectory_layers = set(parse_state_layers(args.trajectory_layers, args.num_layers))
    if args.active_layers == "all":
        active_layers = set(range(args.num_layers))
    else:
        active_layers = set(parse_zero_based_layers(args.active_layers, args.num_layers))
    if args.effect_layers_per_sample > len(active_layers):
        raise RuntimeError(
            f"effect_layers_per_sample={args.effect_layers_per_sample} exceeds active_layers={len(active_layers)}"
        )
    supervise_effect_metrics = args.lambda_effect != 0.0 or args.lambda_effect_cos != 0.0
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (output_dir / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2), encoding="utf-8")
        print(
            f"attention rollout KD: samples={len(dataset)} world_size={dist.get_world_size()} "
            f"per_gpu_batch={args.batch_size} trajectory_layers={args.trajectory_layers} "
            f"active_layers={sorted(active_layers)} "
            f"effect_layers_per_sample={args.effect_layers_per_sample} batched_rollout={args.batched_rollout}",
            flush=True,
        )
    wandb_run = init_wandb(args, len(dataset))

    global_step = args.resume_global_step
    steps_per_epoch = len(dataloader)
    start_epoch = 0
    resume_skip_batches = 0
    if args.resume_global_step > 0:
        if steps_per_epoch <= 0:
            raise RuntimeError("cannot resume with an empty dataloader")
        start_epoch = args.resume_global_step // steps_per_epoch
        resume_skip_batches = args.resume_global_step % steps_per_epoch
        if start_epoch >= args.epochs and (args.max_steps is None or args.resume_global_step < args.max_steps):
            raise RuntimeError(
                f"resume_global_step={args.resume_global_step} maps past epochs={args.epochs}; "
                f"steps_per_epoch={steps_per_epoch}"
            )
        if is_rank0():
            print(
                f"resuming dataloader at epoch={start_epoch} skip_batches={resume_skip_batches} "
                f"steps_per_epoch={steps_per_epoch}",
                flush=True,
            )
    running = {
        "loss": 0.0,
        "trajectory": 0.0,
        "trajectory_rms": 0.0,
        "logit_kl": 0.0,
        "effect": 0.0,
        "effect_cos": 0.0,
        "sidecar_scale": 0.0,
        "teacher_mix": 0.0,
        "prepare_s": 0.0,
        "loss_forward_s": 0.0,
        "backward_s": 0.0,
        "optimizer_s": 0.0,
    }
    log_window_start = time.perf_counter()
    log_window_samples = 0
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, rows in enumerate(dataloader):
            if epoch == start_epoch and batch_idx < resume_skip_batches:
                continue
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            batch_count = max(len(rows), 1)
            scalar_logs = {key: 0.0 for key in running}
            sidecar_scale = scheduled_sidecar_scale(args, global_step)
            rollout_teacher_mix = scheduled_rollout_teacher_mix(args, global_step)
            sync_if_enabled(device, args.sync_perf_timing)
            prepare_start = time.perf_counter()
            if args.batched_rollout:
                inputs, text_ids, answer_mask, _, _ = prepare_batch_inputs(
                    processor,
                    rows,
                    "image",
                    "question",
                    "answer",
                    None,
                    image_token_id,
                    device,
                )
                sync_if_enabled(device, args.sync_perf_timing)
                loss_start = time.perf_counter()
                effect_layers = (
                    sample_stratified_layers_from_candidates(
                        sorted(active_layers),
                        args.effect_layers_per_sample,
                        device,
                    )
                    if supervise_effect_metrics
                    else []
                )
                (
                    trajectory_loss,
                    _trajectory_cos_loss,
                    trajectory_rms_loss,
                    logit_loss,
                    _topk_logit_loss,
                    _answer_margin_loss,
                    _image_negative_loss,
                    _task_loss,
                    effect_loss,
                    effect_cos,
                    _,
                ) = rollout_losses_for_batch(
                    teacher_model,
                    language_model,
                    engine.module,
                    inputs,
                    text_ids,
                    answer_mask,
                    image_token_id,
                    image_seq_length,
                    trajectory_layers,
                    active_layers,
                    effect_layers,
                    args.trajectory_loss_mode,
                    args.effect_loss_mode,
                    args.effect_input_state,
                    args.effect_target_state,
                    sidecar_scale,
                    args.temperature,
                    dtype,
                    args.trajectory_late_start,
                    args.trajectory_late_weight,
                    0,
                    0,
                    False,
                    args.image_negative_margin,
                    args.image_negative_mode,
                    rollout_teacher_mix,
                    args.sidecar_token_mode,
                )
                loss = (
                    args.lambda_trajectory * trajectory_loss
                    + args.lambda_trajectory_rms * trajectory_rms_loss
                    + args.lambda_logit * logit_loss
                    + args.lambda_effect * effect_loss
                    + args.lambda_effect_cos * (1.0 - effect_cos)
                )
                sync_if_enabled(device, args.sync_perf_timing)
                backward_start = time.perf_counter()
                engine.backward(loss)
                sync_if_enabled(device, args.sync_perf_timing)
                optimizer_start = time.perf_counter()
                scalar_logs["loss"] = float(loss.detach())
                scalar_logs["trajectory"] = float(trajectory_loss.detach())
                scalar_logs["trajectory_rms"] = float(trajectory_rms_loss.detach())
                scalar_logs["logit_kl"] = float(logit_loss.detach())
                scalar_logs["effect"] = float(effect_loss.detach())
                scalar_logs["effect_cos"] = float(effect_cos.detach())
                scalar_logs["sidecar_scale"] = sidecar_scale
                scalar_logs["teacher_mix"] = rollout_teacher_mix
                scalar_logs["prepare_s"] = loss_start - prepare_start
                scalar_logs["loss_forward_s"] = backward_start - loss_start
                scalar_logs["backward_s"] = optimizer_start - backward_start
            else:
                sync_if_enabled(device, args.sync_perf_timing)
                loss_start = time.perf_counter()
                for row in rows:
                    inputs, text_ids, answer_mask, _ = prepare_sample_inputs(
                        processor,
                        row,
                        "image",
                        "question",
                        "answer",
                        None,
                        image_token_id,
                        device,
                    )
                    effect_layers = (
                        sample_stratified_layers_from_candidates(
                            sorted(active_layers),
                            args.effect_layers_per_sample,
                            device,
                        )
                        if supervise_effect_metrics
                        else []
                    )
                    (
                        trajectory_loss,
                        _trajectory_cos_loss,
                        trajectory_rms_loss,
                        logit_loss,
                        _topk_logit_loss,
                        _answer_margin_loss,
                        _image_negative_loss,
                        _task_loss,
                        effect_loss,
                        effect_cos,
                        _,
                    ) = rollout_losses_for_sample(
                        teacher_model,
                        language_model,
                        engine.module,
                        row,
                        inputs,
                        text_ids,
                        answer_mask,
                        image_token_id,
                        trajectory_layers,
                        active_layers,
                        effect_layers,
                        args.trajectory_loss_mode,
                        args.effect_loss_mode,
                        args.effect_input_state,
                        args.effect_target_state,
                        sidecar_scale,
                        args.temperature,
                        dtype,
                        args.trajectory_late_start,
                        args.trajectory_late_weight,
                        0,
                        0,
                        False,
                        args.image_negative_margin,
                        args.image_negative_mode,
                        rollout_teacher_mix,
                        args.sidecar_token_mode,
                    )
                    loss = (
                        args.lambda_trajectory * trajectory_loss
                        + args.lambda_trajectory_rms * trajectory_rms_loss
                        + args.lambda_logit * logit_loss
                        + args.lambda_effect * effect_loss
                        + args.lambda_effect_cos * (1.0 - effect_cos)
                    )
                    sync_if_enabled(device, args.sync_perf_timing)
                    backward_start = time.perf_counter()
                    engine.backward(loss / batch_count)
                    sync_if_enabled(device, args.sync_perf_timing)
                    after_backward = time.perf_counter()
                    scalar_logs["loss"] += float(loss.detach()) / batch_count
                    scalar_logs["trajectory"] += float(trajectory_loss.detach()) / batch_count
                    scalar_logs["trajectory_rms"] += float(trajectory_rms_loss.detach()) / batch_count
                    scalar_logs["logit_kl"] += float(logit_loss.detach()) / batch_count
                    scalar_logs["effect"] += float(effect_loss.detach()) / batch_count
                    scalar_logs["effect_cos"] += float(effect_cos.detach()) / batch_count
                    scalar_logs["sidecar_scale"] += sidecar_scale / batch_count
                    scalar_logs["teacher_mix"] += rollout_teacher_mix / batch_count
                    scalar_logs["loss_forward_s"] += (backward_start - loss_start) / batch_count
                    scalar_logs["backward_s"] += (after_backward - backward_start) / batch_count
                    loss_start = time.perf_counter()
                scalar_logs["prepare_s"] = 0.0
            sync_if_enabled(device, args.sync_perf_timing)
            optimizer_start = time.perf_counter()
            engine.step()
            sync_if_enabled(device, args.sync_perf_timing)
            scalar_logs["optimizer_s"] = time.perf_counter() - optimizer_start

            global_step += 1
            log_window_samples += batch_count * dist.get_world_size()
            for key in running:
                running[key] += scalar_logs[key]

            if global_step % args.log_every == 0:
                elapsed = max(time.perf_counter() - log_window_start, 1e-6)
                sec_per_step = elapsed / float(args.log_every)
                samples_per_s = float(log_window_samples) / elapsed
                reduced = {
                    key: reduce_mean(torch.tensor(value / args.log_every, device=device)).item()
                    for key, value in running.items()
                }
                if is_rank0():
                    print(
                        "step={step} epoch={epoch} loss={loss:.6f} trajectory={trajectory:.6f} "
                        "traj_rms={trajectory_rms:.6f} logit_kl={logit_kl:.6f} "
                        "effect={effect:.6f} effect_cos={effect_cos:.6f} "
                        "sidecar_scale={sidecar_scale:.4f} teacher_mix={teacher_mix:.4f} "
                        "sec_per_step={sec_per_step:.3f} samples_per_s={samples_per_s:.3f} "
                        "prepare_s={prepare_s:.3f} loss_forward_s={loss_forward_s:.3f} "
                        "backward_s={backward_s:.3f} optimizer_s={optimizer_s:.3f}".format(
                            step=global_step,
                            epoch=epoch,
                            sec_per_step=sec_per_step,
                            samples_per_s=samples_per_s,
                            **reduced,
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        import wandb

                        wandb.log(
                            {
                                "train/step": global_step,
                                "train/epoch": epoch,
                                "train/loss": reduced["loss"],
                                "train/trajectory": reduced["trajectory"],
                                "train/trajectory_rms": reduced["trajectory_rms"],
                                "train/logit_kl": reduced["logit_kl"],
                                "train/effect": reduced["effect"],
                                "train/effect_cos": reduced["effect_cos"],
                                "train/sidecar_scale": reduced["sidecar_scale"],
                                "train/rollout_teacher_mix": rollout_teacher_mix,
                                "train/lr": args.lr,
                                "train/basis_lr": args.lr * args.basis_lr_mult,
                                "train/sec_per_step": sec_per_step,
                                "train/samples_per_s": samples_per_s,
                                "train/prepare_s": reduced["prepare_s"],
                                "train/loss_forward_s": reduced["loss_forward_s"],
                                "train/backward_s": reduced["backward_s"],
                                "train/optimizer_s": reduced["optimizer_s"],
                            },
                            step=global_step,
                        )
                running = {key: 0.0 for key in running}
                log_window_start = time.perf_counter()
                log_window_samples = 0

            if args.save_every > 0 and global_step % args.save_every == 0:
                engine.save_checkpoint(str(output_dir / "deepspeed"), tag=f"step{global_step}")
                save_plain_checkpoint(engine, output_dir, f"step{global_step}", args)
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    engine.save_checkpoint(str(output_dir / "deepspeed"), tag="final")
    save_plain_checkpoint(engine, output_dir, "final", args)
    if wandb_run is not None:
        import wandb

        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
