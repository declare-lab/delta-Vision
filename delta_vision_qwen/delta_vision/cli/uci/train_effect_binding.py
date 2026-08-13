#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from delta_vision.uci.model import (
    ModelBinder,
    UniversalContextEncoder,
    UniversalContextKVEncoder,
    UniversalKVBinder,
    masked_effect_losses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train/evaluate UCI effect binding.")
    parser.add_argument("--mode", choices=("shared", "bind", "independent", "eval"), required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--architecture", choices=("bank", "kv"), default="bank")
    parser.add_argument("--train-samples", type=int, default=0, help="0 means use all but eval-samples.")
    parser.add_argument("--eval-samples", type=int, default=0, help="0 means evaluate on all samples.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-slots", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--cache-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-context-eval", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def load_trace(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "samples" not in payload:
        raise ValueError(f"{path} is not a UCI trace file")
    return payload


def split_samples(samples: list[dict[str, Any]], train_samples: int, eval_samples: int) -> tuple[list[int], list[int]]:
    n = len(samples)
    if eval_samples <= 0:
        return list(range(n)), list(range(n))
    train_end = n - eval_samples if train_samples <= 0 else min(train_samples, n - eval_samples)
    train_idx = list(range(train_end))
    eval_idx = list(range(max(train_end, 0), n))
    return train_idx, eval_idx


def build_modules(
    traces: list[dict[str, Any]],
    latent_slots: int,
    latent_dim: int,
    encoder_layers: int,
    encoder_heads: int,
    device: torch.device,
    architecture: str = "bank",
    cache_dim: int = 256,
) -> tuple[nn.Module, nn.ModuleDict]:
    context_dim = int(traces[0]["context_dim"])
    if architecture == "bank":
        encoder: nn.Module = UniversalContextEncoder(
            input_dim=context_dim,
            latent_slots=latent_slots,
            latent_dim=latent_dim,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
        ).to(device)
    else:
        encoder = UniversalContextKVEncoder(
            input_dim=context_dim,
            cache_slots=latent_slots,
            cache_dim=cache_dim,
            latent_dim=latent_dim,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
        ).to(device)
    binders = nn.ModuleDict()
    for trace in traces:
        if architecture == "bank":
            binder = ModelBinder(
                num_layers=max(int(x) for x in trace["layers"]) + 1,
                model_dim=int(trace["hidden_size"]),
                latent_slots=latent_slots,
                latent_dim=latent_dim,
                layer_ids=[int(x) for x in trace["layers"]],
            )
        else:
            binder = UniversalKVBinder(
                num_layers=max(int(x) for x in trace["layers"]) + 1,
                model_dim=int(trace["hidden_size"]),
                cache_dim=cache_dim,
                layer_ids=[int(x) for x in trace["layers"]],
            )
        binders[str(trace["teacher"])] = binder.to(device)
    return encoder, binders


def load_checkpoint(path: str, encoder: nn.Module, binders: nn.ModuleDict, strict: bool = False) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"], strict=strict)
    loaded = ckpt.get("binders", {})
    for name, state in loaded.items():
        if name in binders:
            binders[name].load_state_dict(state, strict=strict)


def save_checkpoint(path: Path, encoder: nn.Module, binders: nn.ModuleDict, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "binders": {name: module.state_dict() for name, module in binders.items()},
            "meta": meta,
        },
        path,
    )


def make_batch(
    trace: dict[str, Any],
    sample_indices: list[int],
    device: torch.device,
    *,
    random_token: bool = True,
    shuffle_context: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [trace["samples"][idx] for idx in sample_indices]
    context_items = [s["context_tokens"] for s in samples]
    if shuffle_context and len(context_items) > 1:
        context_items = context_items[1:] + context_items[:1]
    context = torch.stack(context_items, dim=0).to(device)
    hidden_items = []
    delta_items = []
    for sample in samples:
        h = sample["hiddens"]
        d = sample["deltas"]
        if h.ndim == 2:
            hidden_items.append(h)
            delta_items.append(d)
        elif h.ndim == 3:
            token_count = int(h.shape[1])
            if random_token:
                token_rms = d.float().pow(2).mean(dim=(0, 2)).sqrt()
                valid = torch.nonzero(token_rms > 1e-3, as_tuple=False).flatten().tolist()
                token_idx = int(random.choice(valid)) if valid else random.randrange(token_count)
            else:
                token_idx = token_count - 1
            hidden_items.append(h[:, token_idx])
            delta_items.append(d[:, token_idx])
        else:
            raise ValueError(f"unsupported hidden tensor rank: {h.ndim}")
    hiddens = torch.stack(hidden_items, dim=0).to(device)
    deltas = torch.stack(delta_items, dim=0).to(device)
    return context, hiddens, deltas


def forward_trace(
    encoder: nn.Module,
    binder: ModelBinder,
    trace: dict[str, Any],
    sample_indices: list[int],
    device: torch.device,
    random_token: bool = True,
    shuffle_context: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    context, hiddens, deltas = make_batch(
        trace,
        sample_indices,
        device,
        random_token=random_token,
        shuffle_context=shuffle_context,
    )
    encoded = encoder(context)
    losses = []
    cos_terms = []
    nmse_terms = []
    for local_layer_idx, layer_idx in enumerate(trace["layers"]):
        pred = binder(encoded, hiddens[:, local_layer_idx], int(layer_idx))
        target = deltas[:, local_layer_idx]
        loss_dict = masked_effect_losses(pred, target)
        losses.append(loss_dict["nmse"] + 0.05 * (1.0 - loss_dict["cos"]))
        cos_terms.append(loss_dict["cos"].detach())
        nmse_terms.append(loss_dict["nmse"].detach())
    loss = torch.stack(losses).mean()
    metrics = {
        "effect_nmse": float(torch.stack(nmse_terms).mean().cpu()),
        "effect_cos": float(torch.stack(cos_terms).mean().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    binders: nn.ModuleDict,
    traces: list[dict[str, Any]],
    eval_indices: dict[str, list[int]],
    device: torch.device,
    batch_size: int,
    shuffle_context: bool = False,
) -> dict[str, Any]:
    encoder.eval()
    for binder in binders.values():
        binder.eval()
    out: dict[str, Any] = {}
    for trace in traces:
        name = str(trace["teacher"])
        indices = eval_indices[name]
        losses = []
        coss = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            loss, metrics = forward_trace(
                encoder,
                binders[name],
                trace,
                batch,
                device,
                random_token=False,
                shuffle_context=shuffle_context,
            )
            losses.append(float(loss.cpu()))
            coss.append(float(metrics["effect_cos"]))
        out[name] = {
            "samples": len(indices),
            "loss": sum(losses) / max(1, len(losses)),
            "effect_cos": sum(coss) / max(1, len(coss)),
        }
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    traces = [load_trace(path) for path in args.traces]
    encoder, binders = build_modules(
        traces,
        args.latent_slots,
        args.latent_dim,
        args.encoder_layers,
        args.encoder_heads,
        device,
        args.architecture,
        args.cache_dim,
    )
    if args.checkpoint:
        load_checkpoint(args.checkpoint, encoder, binders, strict=False)
    if args.mode == "bind":
        for param in encoder.parameters():
            param.requires_grad_(False)
    if args.mode == "eval":
        for param in encoder.parameters():
            param.requires_grad_(False)
        for param in binders.parameters():
            param.requires_grad_(False)

    train_indices: dict[str, list[int]] = {}
    eval_indices: dict[str, list[int]] = {}
    for trace in traces:
        train_idx, eval_idx = split_samples(trace["samples"], args.train_samples, args.eval_samples)
        train_indices[str(trace["teacher"])] = train_idx
        eval_indices[str(trace["teacher"])] = eval_idx

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": args.mode,
        "traces": args.traces,
        "latent_slots": args.latent_slots,
        "latent_dim": args.latent_dim,
        "architecture": args.architecture,
        "cache_dim": args.cache_dim,
        "train_indices": {k: [min(v) if v else None, max(v) if v else None, len(v)] for k, v in train_indices.items()},
        "eval_indices": {k: [min(v) if v else None, max(v) if v else None, len(v)] for k, v in eval_indices.items()},
    }

    if args.mode != "eval":
        params = [p for p in list(encoder.parameters()) + list(binders.parameters()) if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        teacher_names = [str(trace["teacher"]) for trace in traces]
        trace_by_name = {str(trace["teacher"]): trace for trace in traces}
        for step in range(1, args.max_steps + 1):
            name = teacher_names[(step - 1) % len(teacher_names)]
            indices = train_indices[name]
            batch = random.choices(indices, k=args.batch_size)
            encoder.train()
            binders[name].train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = forward_trace(encoder, binders[name], trace_by_name[name], batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if step % args.log_every == 0 or step == 1:
                print(
                    f"step={step} teacher={name} loss={float(loss.detach().cpu()):.4f} "
                    f"nmse={metrics['effect_nmse']:.4f} cos={metrics['effect_cos']:.4f}",
                    flush=True,
                )
        save_checkpoint(out_dir / "checkpoint.pt", encoder, binders, meta)

    eval_result = evaluate(
        encoder,
        binders,
        traces,
        eval_indices,
        device,
        args.batch_size,
        shuffle_context=args.shuffle_context_eval,
    )
    result = {"meta": meta, "eval": eval_result}
    (out_dir / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
