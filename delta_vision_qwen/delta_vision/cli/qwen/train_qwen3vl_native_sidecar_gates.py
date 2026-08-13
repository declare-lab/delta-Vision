#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from delta_vision.data import JsonlDataset
from delta_vision.evaluation.metrics import masked_kl
from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_sample_inputs,
)
from delta_vision.runtime.qwen_analytic_sidecar import (
    QwenNativeAttentionSidecar,
    run_analytic_qwen3vl_sidecar_only_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Qwen native-attention sidecar layer gates.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-gate-reg", type=float, default=1.0)
    parser.add_argument("--gate-init", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def save_checkpoint(sidecar: QwenNativeAttentionSidecar, path: Path, args: argparse.Namespace, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in sidecar.state_dict().items()},
            "args": {
                **vars(args),
                "sidecar_backend": "native_attention",
                "visual_memory_mode": "vprefix",
                "global_step": int(step),
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    metrics_path = output_dir / "train_metrics.jsonl"

    processor, teacher_model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(teacher_model)
    sidecar = QwenNativeAttentionSidecar(
        num_layers=len(language_model.layers),
        gate_init=args.gate_init,
        train_gates=True,
    ).to(device=device, dtype=dtype)
    optim = torch.optim.AdamW(sidecar.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = JsonlDataset(args.data, max_samples=args.max_samples, start_index=args.start_index, decode_images=False)
    if len(dataset) == 0:
        raise RuntimeError("empty training dataset")

    for step in range(1, args.max_steps + 1):
        row = dataset[(step - 1) % len(dataset)]
        inputs, _, answer_mask, image_path = prepare_qwen3vl_sample_inputs(
            processor,
            row,
            "image",
            "question",
            "answer",
            None,
            device,
        )
        with torch.no_grad():
            teacher = teacher_model(
                **inputs,
                return_dict=True,
                use_cache=False,
            )
            hidden0, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(
                teacher_model,
                inputs,
            )
            text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                position_ids,
            )
            teacher_text_logits = torch.gather(
                teacher.logits,
                dim=1,
                index=text_pos.unsqueeze(-1).expand(-1, -1, teacher.logits.shape[-1]),
            ).detach()

        student_h = run_analytic_qwen3vl_sidecar_only_rollout(
            language_model,
            hidden0.detach(),
            position_ids,
            inputs["attention_mask"],
            text_pos,
            image_pos,
            text_position_ids,
            text_mask,
            image_mask,
            full_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
            "vprefix",
            dtype,
            native_sidecar=sidecar,
        )
        student_logits = teacher_model.lm_head(language_model.norm(student_h))
        logit_kl = masked_kl(student_logits, teacher_text_logits, answer_mask, args.temperature)
        gate_reg = (sidecar.gate.float() - 1.0).pow(2).mean()
        loss = args.lambda_logit * logit_kl + args.lambda_gate_reg * gate_reg

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
        optim.step()

        if step % args.log_every == 0 or step == 1:
            metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "logit_kl": float(logit_kl.detach()),
                "gate_reg": float(gate_reg.detach()),
                "gate_mean": float(sidecar.gate.detach().float().mean()),
                "gate_min": float(sidecar.gate.detach().float().min()),
                "gate_max": float(sidecar.gate.detach().float().max()),
                "text_tokens": int(text_mask.sum().item()),
                "image_tokens": int(image_mask.sum().item()),
                "image": image_path,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(sidecar, output_dir / f"native_sidecar_step{step}.pt", args, step)

    save_checkpoint(sidecar, output_dir / "native_sidecar_final.pt", args, args.max_steps)


if __name__ == "__main__":
    main()
