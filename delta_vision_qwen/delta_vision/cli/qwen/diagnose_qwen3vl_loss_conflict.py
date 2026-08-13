#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from delta_vision.cli.qwen.train_qwen3vl_sidecar import compute_loss_for_row
from delta_vision.data import JsonlDataset
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import load_frozen_qwen3vl
from delta_vision.models.sidecar import DeltaVisionModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose whether Qwen3-VL Sidecar effect loss conflicts with logit KL.")
    parser.add_argument("--data", default="artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def training_args_from_checkpoint(checkpoint: dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    values = dict(checkpoint.get("args", {}))
    defaults = {
        "profile_timing": False,
        "effect_layers_per_sample": 36,
        "effect_layer_sampling": "all",
        "hard_effect_layers": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35",
        "trajectory_layers": "4,8,12,16,20,24,28,32,36",
        "visual_memory_mode": "v0",
        "effect_target": "teacher_visual",
        "teacher_force_steps": 0,
        "teacher_force_mix": 0.0,
        "teacher_mix_end_step": 0,
        "factorized_mass_mode": "learned",
        "fixed_visual_mass": 0.12,
        "sidecar_query_source": "qwen_native",
        "sidecar_visual_kv_source": "qwen_native",
        "state_tokens": 0,
        "use_rope": False,
        "temperature": 2.0,
        "lambda_effect": 1.0,
        "lambda_effect_cos": 0.0,
        "lambda_effect_rms": 0.0,
        "lambda_trajectory": 4.0,
        "lambda_trajectory_rms": 0.0,
        "lambda_logit": 1.0,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    values["data"] = cli_args.data
    values["model_path"] = cli_args.model_path
    values["device"] = cli_args.device
    values["dtype"] = cli_args.dtype
    values["attn_implementation"] = cli_args.attn_implementation
    return argparse.Namespace(**values)


def load_trainable_sidecar(
    checkpoint: dict,
    train_args: argparse.Namespace,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DeltaVisionModule:
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=int(train_args.rank),
        sidecar_dim=int(train_args.sidecar_dim),
        num_heads=int(train_args.num_heads),
        state_tokens=int(getattr(train_args, "state_tokens", 0)),
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=train_args.output_mode in {"residual", "factorized_lowrank"},
        reader_mlp_ratio=float(train_args.reader_mlp_ratio),
        reader_activation=str(getattr(train_args, "reader_activation", "situ_glu")),
        layer_adapter_rank=int(train_args.layer_adapter_rank),
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=bool(train_args.shared_basis),
        output_mode=str(train_args.output_mode),
        corrector_layers=str(getattr(train_args, "corrector_layers", "")),
        corrector_dim=int(getattr(train_args, "corrector_dim", 0)),
        block_corrector_groups=str(getattr(train_args, "block_corrector_groups", "")),
        block_corrector_dim=int(getattr(train_args, "block_corrector_dim", 0)),
        use_rope=bool(getattr(train_args, "use_rope", False)),
    ).to(device=device, dtype=dtype)
    sidecar.load_state_dict(checkpoint["state_dict"], strict=True)
    sidecar.train()
    return sidecar


def loss_args(base: argparse.Namespace, *, effect: bool, logit: bool, trajectory: bool = False) -> argparse.Namespace:
    out = copy.deepcopy(base)
    out.lambda_effect = 1.0 if effect else 0.0
    out.lambda_effect_cos = 0.0
    out.lambda_effect_rms = 0.0
    out.lambda_trajectory = 1.0 if trajectory else 0.0
    out.lambda_trajectory_rms = 0.0
    out.lambda_logit = 1.0 if logit else 0.0
    return out


def grad_snapshot(sidecar: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    grads: dict[str, torch.Tensor] = {}
    group_norm2: dict[str, float] = {}
    for name, param in sidecar.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float().cpu()
        grads[name] = grad
        group = name.split(".", 1)[0]
        group_norm2[group] = group_norm2.get(group, 0.0) + float(grad.pow(2).sum().item())
    return grads, group_norm2


def compare_grads(effect_grads: dict[str, torch.Tensor], kl_grads: dict[str, torch.Tensor]) -> dict[str, object]:
    dot = 0.0
    effect_norm2 = 0.0
    kl_norm2 = 0.0
    by_group: dict[str, dict[str, float]] = {}
    for name in sorted(set(effect_grads) | set(kl_grads)):
        eg = effect_grads.get(name)
        kg = kl_grads.get(name)
        if eg is None and kg is None:
            continue
        if eg is None:
            kg = kg.float()
            local_kl = float(kg.pow(2).sum().item())
            local_effect = 0.0
            local_dot = 0.0
        elif kg is None:
            eg = eg.float()
            local_effect = float(eg.pow(2).sum().item())
            local_kl = 0.0
            local_dot = 0.0
        else:
            eg = eg.float()
            kg = kg.float()
            local_effect = float(eg.pow(2).sum().item())
            local_kl = float(kg.pow(2).sum().item())
            local_dot = float((eg * kg).sum().item())
        dot += local_dot
        effect_norm2 += local_effect
        kl_norm2 += local_kl
        group = name.split(".", 1)[0]
        row = by_group.setdefault(group, {"dot": 0.0, "effect_norm2": 0.0, "kl_norm2": 0.0})
        row["dot"] += local_dot
        row["effect_norm2"] += local_effect
        row["kl_norm2"] += local_kl
    eps = 1e-12
    groups_out = {}
    for group, row in by_group.items():
        denom = (row["effect_norm2"] ** 0.5) * (row["kl_norm2"] ** 0.5)
        groups_out[group] = {
            "grad_cos": row["dot"] / max(denom, eps),
            "effect_grad_norm": row["effect_norm2"] ** 0.5,
            "kl_grad_norm": row["kl_norm2"] ** 0.5,
        }
    denom = (effect_norm2**0.5) * (kl_norm2**0.5)
    return {
        "grad_cos": dot / max(denom, eps),
        "effect_grad_norm": effect_norm2**0.5,
        "kl_grad_norm": kl_norm2**0.5,
        "groups": groups_out,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = training_args_from_checkpoint(checkpoint, args)

    processor, teacher_model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(teacher_model)
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    dataset = JsonlDataset(args.data, decode_images=False)
    sidecar = load_trainable_sidecar(checkpoint, train_args, hidden_size, num_layers, device, dtype)

    rows_out = []
    for batch_idx in range(args.max_batches):
        start = args.start_index + batch_idx * args.micro_batch_size
        rows = [dataset[(start + offset) % len(dataset)] for offset in range(args.micro_batch_size)]

        sidecar.zero_grad(set_to_none=True)
        effect_loss, effect_metrics = compute_loss_for_row(
            loss_args(train_args, effect=True, logit=False),
            processor,
            teacher_model,
            language_model,
            sidecar,
            rows,
            int(checkpoint.get("global_step", 0)),
            device,
            dtype,
            num_layers,
        )
        effect_loss.backward()
        effect_grads, _ = grad_snapshot(sidecar)

        sidecar.zero_grad(set_to_none=True)
        kl_loss, kl_metrics = compute_loss_for_row(
            loss_args(train_args, effect=False, logit=True),
            processor,
            teacher_model,
            language_model,
            sidecar,
            rows,
            int(checkpoint.get("global_step", 0)),
            device,
            dtype,
            num_layers,
        )
        kl_loss.backward()
        kl_grads, _ = grad_snapshot(sidecar)

        cmp = compare_grads(effect_grads, kl_grads)
        row = {
            "batch": batch_idx,
            "start_index": start,
            "effect_loss": float(effect_loss.detach().item()),
            "logit_kl": float(kl_loss.detach().item()),
            "effect_cos_metric": float(effect_metrics.get("effect_cos", 0.0)),
            "trajectory_metric": float(effect_metrics.get("trajectory", 0.0)),
            **cmp,
        }
        rows_out.append(row)
        print(
            f"batch={batch_idx} grad_cos={row['grad_cos']:.4f} "
            f"effect={row['effect_loss']:.4f} kl={row['logit_kl']:.4f} "
            f"effect_cos_metric={row['effect_cos_metric']:.4f}",
            flush=True,
        )
        sidecar.zero_grad(set_to_none=True)

    avg = {
        "grad_cos": sum(float(row["grad_cos"]) for row in rows_out) / max(1, len(rows_out)),
        "effect_loss": sum(float(row["effect_loss"]) for row in rows_out) / max(1, len(rows_out)),
        "logit_kl": sum(float(row["logit_kl"]) for row in rows_out) / max(1, len(rows_out)),
        "effect_cos_metric": sum(float(row["effect_cos_metric"]) for row in rows_out) / max(1, len(rows_out)),
    }
    payload = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "start_index": args.start_index,
        "max_batches": args.max_batches,
        "micro_batch_size": args.micro_batch_size,
        "average": avg,
        "batches": rows_out,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"average": avg, "output_json": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
