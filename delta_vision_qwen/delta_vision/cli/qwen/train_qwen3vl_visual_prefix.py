#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from delta_vision.data import JsonlDataset
from delta_vision.evaluation.metrics import masked_topk_kl
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    run_qwen3vl_full_layer_with_text_delta,
)
from delta_vision.models.visual_prefix import VisualPrefixCompressor, grid_position_ids
from delta_vision.cli.qwen.train_qwen3vl_sidecar import answer_token_ce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Qwen3-VL small visual-prefix baseline.")
    parser.add_argument("--data", default="artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-prefix-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-logit", type=float, default=4.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def save_checkpoint(module: nn.Module, path: Path, args: argparse.Namespace, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
            "args": vars(args),
            "step": int(step),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    processor, teacher_model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(teacher_model)
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    compressor = VisualPrefixCompressor(hidden_size, args.num_prefix_tokens, args.num_heads).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = JsonlDataset(args.data, max_samples=args.max_samples, decode_images=False)
    if len(dataset) == 0:
        raise RuntimeError("empty training dataset")

    metrics_path = output_dir / "train_metrics.jsonl"
    for step in range(1, args.max_steps + 1):
        row = dataset[(step - 1) % len(dataset)]
        with torch.no_grad():
            inputs, text_ids, answer_mask, _ = prepare_qwen3vl_batch_inputs(
                processor,
                [row],
                "image",
                "question",
                "answer",
                None,
                device,
            )
            teacher = teacher_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(teacher_model, inputs)
            text_positions, image_positions, text_position_ids, text_mask, image_mask, _ = get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                full_position_ids,
            )
            text_h0 = gather_batched_positions(hidden0.to(dtype=dtype), text_positions, text_mask)
            v0 = gather_batched_positions(hidden0.to(dtype=dtype), image_positions, image_mask)
            teacher_logits = gather_batched_positions(teacher.logits, text_positions, text_mask).detach()

        prefix = compressor(v0, image_mask)
        batch, prefix_len = prefix.shape[:2]
        full_h = torch.cat([prefix, text_h0], dim=1)
        prefix_pos = grid_position_ids(batch, prefix_len, device).to(dtype=text_position_ids.dtype)
        full_pos = torch.cat([prefix_pos, text_position_ids], dim=2).contiguous()
        full_mask = torch.cat(
            [
                torch.ones(batch, prefix_len, device=device, dtype=torch.long),
                text_mask.to(dtype=torch.long),
            ],
            dim=1,
        )
        full_text_positions = torch.arange(text_h0.shape[1], device=device).view(1, -1).expand(batch, -1) + prefix_len
        h = full_h
        for layer_idx in range(num_layers):
            h = run_qwen3vl_full_layer_with_text_delta(
                language_model,
                layer_idx,
                h,
                full_pos,
                full_mask,
                full_text_positions,
                text_delta=None,
            )
        student_text_h = h[:, prefix_len : prefix_len + text_h0.shape[1]]
        student_logits = teacher_model.lm_head(language_model.norm(student_text_h))
        logit_kl = masked_topk_kl(student_logits, teacher_logits, text_ids, answer_mask, args.temperature, 1024)
        ce = answer_token_ce(student_logits, text_ids, answer_mask)
        loss = args.lambda_logit * logit_kl + args.lambda_ce * ce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
        optimizer.step()

        metrics = {
            "step": step,
            "loss": float(loss.detach()),
            "logit_kl": float(logit_kl.detach()),
            "ce": float(ce.detach()),
            "image_tokens": int(image_mask.sum().item()),
            "text_tokens": int(text_mask.sum().item()),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if step % args.log_every == 0:
            print(
                f"step={step} loss={metrics['loss']:.6f} logit_kl={metrics['logit_kl']:.6f} "
                f"ce={metrics['ce']:.6f} image_tokens={metrics['image_tokens']}",
                flush=True,
            )
        if step % args.save_every == 0:
            save_checkpoint(compressor, output_dir / f"visual_prefix_step{step}.pt", args, step)
    save_checkpoint(compressor, output_dir / "visual_prefix_final.pt", args, args.max_steps)


if __name__ == "__main__":
    main()
