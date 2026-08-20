"""Run test-only recurrent Qwen embedding_adapter training.

This wrapper monkeypatches QwenEmbeddingAdapter inside the current Python process so
diagnostic recurrent training can run without changing src/ or scripts/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.train as train_module  # noqa: E402
from src.model import (  # noqa: E402
    QwenEmbeddingAdapter,
    gather_batched_positions,
    get_qwen_text_image_positions,
    qwen_lm_head_logits,
    qwen_text_attention_output,
    qwen_text_attention_output_with_visual_kv,
    prepare_qwen_embedding_adapter_inputs,
    qwen_position_ids,
    run_qwen_layer_from_attention_output,
)


ATTENTION_OUTPUT_WEIGHT = float(os.environ.get("ATTENTION_OUTPUT_WEIGHT", "2.0"))
ATTENTION_LOSS_MODE = os.environ.get("ATTENTION_LOSS_MODE", "effect_delta").strip().lower()
if ATTENTION_LOSS_MODE not in {"effect_delta", "joint_attention"}:
    raise ValueError("ATTENTION_LOSS_MODE must be effect_delta or joint_attention")


def recurrent_all_visual_memories_batched(self: QwenEmbeddingAdapter, visual_memory: torch.Tensor) -> torch.Tensor:
    memories = []
    current = visual_memory
    for layer_idx in range(self.num_layers):
        current = self.visual_memory_for_layer(current, layer_idx)
        memories.append(current)
    return torch.stack(memories, dim=0)


QwenEmbeddingAdapter.all_visual_memories_batched = recurrent_all_visual_memories_batched  # type: ignore[method-assign]


def _masked_nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    numerator = (pred.float() - target.float()).pow(2).mul(valid).sum()
    denominator = target.float().pow(2).mul(valid).sum().clamp_min(1e-6)
    return numerator / denominator


def _qwen_adapter_logits_and_effects(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    prepared: dict[str, torch.Tensor],
    teacher_hidden_states: tuple[torch.Tensor, ...],
    image_positions: torch.Tensor,
    image_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    language_model = model.model.language_model
    h = prepared["h"]
    all_visual_memories = adapter.all_visual_memories_batched(prepared["visual_memory"])
    student_effects = []
    target_effects = []
    text_padding_mask = ~prepared["text_mask"]
    for layer_idx in range(int(adapter.num_layers)):
        text_only_attention = None
        if ATTENTION_LOSS_MODE == "effect_delta":
            text_only_attention = qwen_text_attention_output(
                language_model,
                layer_idx,
                h,
                prepared["text_position_ids"],
                padding_mask=text_padding_mask,
            )
        joint_attention = qwen_text_attention_output_with_visual_kv(
            language_model,
            layer_idx,
            h,
            prepared["text_position_ids"],
            all_visual_memories[layer_idx],
            prepared["visual_position_ids"],
            prefix_attention_mask=prepared["prefix_attention_mask"],
        )
        if ATTENTION_LOSS_MODE == "effect_delta":
            if text_only_attention is None:
                raise RuntimeError("text_only_attention was not computed")
            student_effects.append(joint_attention - text_only_attention)
        else:
            student_effects.append(joint_attention)

        with torch.no_grad():
            target_text_h = h.detach()
            target_text_only = None
            if ATTENTION_LOSS_MODE == "effect_delta":
                target_text_only = qwen_text_attention_output(
                    language_model,
                    layer_idx,
                    target_text_h,
                    prepared["text_position_ids"],
                    padding_mask=text_padding_mask,
                )
            target_visual_h = gather_batched_positions(
                teacher_hidden_states[layer_idx].detach(),
                image_positions,
                image_mask,
            ).to(dtype=h.dtype)
            target_joint = qwen_text_attention_output_with_visual_kv(
                language_model,
                layer_idx,
                target_text_h,
                prepared["text_position_ids"],
                target_visual_h,
                prepared["visual_position_ids"],
                prefix_attention_mask=prepared["prefix_attention_mask"],
            )
            if ATTENTION_LOSS_MODE == "effect_delta":
                if target_text_only is None:
                    raise RuntimeError("target_text_only was not computed")
                target_effects.append((target_joint - target_text_only).detach())
            else:
                target_effects.append(target_joint.detach())

        h = run_qwen_layer_from_attention_output(language_model, layer_idx, h, joint_attention, None)
    logits = qwen_lm_head_logits(model, language_model, h, prepared["text_mask"])
    return logits, prepared["text_mask"], student_effects, target_effects


def compute_qwen_loss_with_visual_memory_alignment(
    args,
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    inputs: dict[str, torch.Tensor],
    text_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    num_layers: int,
):
    loss_mode = str(args.supervision_loss)
    if loss_mode != "distill":
        return train_module._ORIGINAL_COMPUTE_QWEN_LOSS(
            args,
            model,
            adapter,
            inputs,
            text_ids,
            answer_mask,
            num_layers,
        )

    with torch.no_grad():
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        full_position_ids = qwen_position_ids(model, inputs)
        text_positions, image_pos, _, text_mask, image_mask, _ = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_logits = gather_batched_positions(teacher.logits.detach(), text_positions, text_mask)
        initial_hidden = teacher.hidden_states[0].detach()

    prepared = prepare_qwen_embedding_adapter_inputs(
        model,
        adapter,
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        initial_hidden,
        full_position_ids,
        reuse_position_embeddings=True,
    )
    student_logits, student_text_mask, student_effects, target_effects = _qwen_adapter_logits_and_effects(
        model,
        adapter,
        prepared,
        teacher.hidden_states,
        image_pos,
        image_mask,
    )

    if student_text_mask.shape != text_mask.shape:
        raise RuntimeError("student/teacher text masks differ")

    effect_terms = [
        _masked_nmse(student_effects[layer_idx], target_effects[layer_idx], text_mask)
        for layer_idx in range(min(len(student_effects), len(target_effects)))
    ]
    effect = torch.stack(effect_terms).mean() if effect_terms else student_logits.new_zeros(())
    logit_kl, per_sample_supervision, answer_counts = train_module.masked_topk_kl(
        student_logits,
        teacher_logits,
        text_ids,
        answer_mask,
        args.temperature,
        args.kl_topk,
        normalization=args.loss_normalization,
    )
    kv_mse = student_logits.new_zeros(())
    trajectory = student_logits.new_zeros(())
    loss = args.lambda_logit * logit_kl + ATTENTION_OUTPUT_WEIGHT * effect

    metrics = {
        "loss": float(loss.detach()),
        "ce": 0.0,
        "logit_kl": float(logit_kl.detach()),
        "trajectory": float(trajectory.detach()),
        "kv_mse": float(kv_mse.detach()),
        "effect": float(effect.detach()),
        "attention_output": float(effect.detach()),
        "lambda_attention_output": float(ATTENTION_OUTPUT_WEIGHT),
        "attention_loss_mode": ATTENTION_LOSS_MODE,
        "visual_memory_mse": 0.0,
        "visual_memory_log_norm_mse": 0.0,
        "visual_memory_cosine": 0.0,
        "lambda_visual_memory_mse": 0.0,
        "lambda_visual_memory_cosine": 0.0,
        "visual_mass": 0.0,
        "text_tokens": float(student_text_mask.sum().item()) / max(1, student_text_mask.shape[0]),
        "answer_tokens": float(answer_counts.sum().item()) / max(1, answer_counts.shape[0]),
    }
    return loss, metrics, per_sample_supervision, answer_counts, student_text_mask


train_module._ORIGINAL_COMPUTE_QWEN_LOSS = train_module.compute_qwen_loss_for_prepared_inputs
train_module.compute_qwen_loss_for_prepared_inputs = compute_qwen_loss_with_visual_memory_alignment


if __name__ == "__main__":
    train_module.main()
