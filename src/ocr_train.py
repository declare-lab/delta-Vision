"""Train Qwen3-VL embedding adapter for rendered-text OCR distillation.

This is an experimental entrypoint kept outside the main trainer on purpose.
It expects paired rendered-text rows with:
  - images or image: rendered page image path(s)
  - image_root: optional image root
  - text_context: raw textual context for the teacher
  - question or rendered_question: question for the student
  - answer: short answer suffix supervised for both teacher/student
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from src.model import (
    QwenEmbeddingAdapter,
    _compile_exact_qwen_apply_rotary_pos_emb,
    build_qwen_initial_context,
    dtype_from_name,
    load_frozen_qwen3vl,
    prepare_qwen_embedding_adapter_inputs,
    prepare_qwen3vl_batch_inputs,
    qwen_embedding_adapter_prefill_cache_prepared,
    qwen_embedding_adapter_logits,
)
from src.train import lr_multiplier, optimizer_param_groups, save_checkpoint, set_engine_lr, trainable_parameters_for_mode


class JsonlRows:
    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        require_answer_visible: bool = False,
        shuffle: bool = True,
        seed: int = 49,
    ) -> None:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if require_answer_visible and row.get("answer_visible") is False:
                    continue
                if not str(row.get("text_context") or "").strip():
                    continue
                if not str(row.get("answer") or "").strip():
                    continue
                rows.append(row)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if not rows:
            raise RuntimeError(f"no usable rows found in {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, start: int, batch_size: int) -> list[dict[str, Any]]:
        return [self.rows[(start + offset) % len(self.rows)] for offset in range(batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OCR/rendered-text teacher -> rendered-image adapter student training.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/rendered_text_copy_2048/paired_train.jsonl")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="artifacts/experiments/test_rendered_text_teacher")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--micro-batch-size-per-gpu", "--batch-size", dest="micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--required-world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--require-answer-visible", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=0, help="0 means no teacher text truncation.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-start-lr-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--teacher-mode", choices=("text", "image"), default="text")
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0, help="Optional student CE on all answer tokens; 0 keeps KL-only behavior.")
    parser.add_argument("--lambda-effect", type=float, default=0.0, help="Optional attention-output effect MSE alignment loss.")
    parser.add_argument("--effect-mask", choices=("answer", "all_text"), default="answer")
    parser.add_argument("--effect-layers", choices=("last", "all"), default="last")
    parser.add_argument("--lambda-prefill-kv", type=float, default=0.0, help="Optional prefill K/V MSE alignment loss.")
    parser.add_argument("--prefill-kv-layers", choices=("last", "all"), default="all")
    parser.add_argument("--prefill-kv-eps", type=float, default=1e-6)
    parser.add_argument("--visual-adapter-rank", type=int, default=128)
    parser.add_argument("--output-mode", default="embedding_adapter")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed_is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(value.item()) for key, value in zip(keys, values)}


RENDERED_PAGE_INSTRUCTION = "Read the ordered page images and answer using only their text."
QA_IMAGE_INSTRUCTION = "Use the image text to answer the question."
QA_FINAL_ANSWER_INSTRUCTION = "Answer directly with a short phrase."
COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."


def is_copy_transcription(row: dict[str, Any]) -> bool:
    return str(row.get("task_type") or "").strip() in {"copy", "copy_transcription"}


def cleaned_rendered_question(row: dict[str, Any]) -> str:
    question = str(row.get("raw_question") or row.get("question") or row.get("rendered_question") or "").strip()
    prefixes = (
        RENDERED_PAGE_INSTRUCTION,
        "The attached page images are consecutive pages in order.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix) :].lstrip("\n ").strip()
                changed = True
    return question


def text_teacher_prompt(processor: Any, row: dict[str, Any], max_context_chars: int) -> str:
    context = str(row["text_context"]).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]
    question = cleaned_rendered_question(row)
    if is_copy_transcription(row):
        instruction = question or COPY_TRANSCRIPTION_INSTRUCTION
        text = f"Text:\n{context}\n\n{instruction}"
    else:
        text = f"Context:\n{context}\n\nQuestion:\n{question}\n\n{QA_FINAL_ANSWER_INSTRUCTION}"
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_text_teacher_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    max_context_chars: int,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    texts: list[str] = []
    answer_lens: list[int] = []
    eos = processor.tokenizer.eos_token or ""
    for row in rows:
        prompt = text_teacher_prompt(processor, row, max_context_chars)
        answer = str(row.get("answer", "")).strip()
        suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
        texts.append(f"{prompt}{suffix}")
        answer_lens.append(len(processor.tokenizer(suffix, add_special_tokens=False).input_ids))

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    try:
        encoded = processor.tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch_idx, answer_len in enumerate(answer_lens):
        valid_len = int(attention_mask[batch_idx].sum().item())
        start = max(0, valid_len - int(answer_len))
        answer_mask[batch_idx, start:valid_len] = True
    return inputs, input_ids, answer_mask


def student_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        question = cleaned_rendered_question(copy)
        if is_copy_transcription(copy):
            copy["question"] = question or COPY_TRANSCRIPTION_INSTRUCTION
        else:
            copy["question"] = f"{QA_IMAGE_INSTRUCTION}\n{QA_FINAL_ANSWER_INSTRUCTION}\n\nQuestion: {question}"
        result.append(copy)
    return result


def masked_unaligned_topk_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_ids: Tensor,
    student_answer_mask: Tensor,
    teacher_answer_mask: Tensor,
    *,
    temperature: float,
    topk: int,
) -> tuple[Tensor, Tensor]:
    losses: list[Tensor] = []
    counts: list[int] = []
    for batch_idx in range(student_logits.shape[0]):
        s_mask = student_answer_mask[batch_idx, 1:].bool()
        t_mask = teacher_answer_mask[batch_idx, 1:].bool()
        s = student_logits[batch_idx, :-1][s_mask]
        t = teacher_logits[batch_idx, :-1][t_mask]
        targets = student_ids[batch_idx, 1:][s_mask]
        count = min(int(s.shape[0]), int(t.shape[0]), int(targets.shape[0]))
        counts.append(count)
        if count <= 0:
            continue
        s = s[-count:].float()
        t = t[-count:].float()
        targets = targets[-count:].long()
        k_eff = min(int(topk), int(t.shape[-1]))
        topk_idx = torch.topk(t, k=k_eff, dim=-1).indices
        target_idx = targets.unsqueeze(-1)
        if k_eff < int(t.shape[-1]):
            target_in_topk = topk_idx.eq(target_idx).any(dim=-1, keepdim=True)
            gather_idx = torch.where(target_in_topk, topk_idx, torch.cat([topk_idx[..., :-1], target_idx], dim=-1))
        else:
            gather_idx = topk_idx
        s_gathered = torch.gather(s, dim=-1, index=gather_idx) / float(temperature)
        t_gathered = torch.gather(t, dim=-1, index=gather_idx) / float(temperature)
        kl = F.kl_div(
            F.log_softmax(s_gathered, dim=-1),
            F.softmax(t_gathered, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        token_loss = kl * (float(temperature) * float(temperature))
        losses.append(token_loss)
    answer_counts = torch.tensor(counts, device=student_logits.device, dtype=torch.float32)
    if not losses:
        return student_logits.new_zeros(()), answer_counts
    flat_losses = torch.cat(losses)
    return flat_losses.float().mean().to(dtype=student_logits.dtype), answer_counts


def masked_unaligned_student_answer_ce(
    student_logits: Tensor,
    student_ids: Tensor,
    student_answer_mask: Tensor,
    teacher_answer_mask: Tensor | None,
) -> Tensor:
    losses: list[Tensor] = []
    for batch_idx in range(student_logits.shape[0]):
        s_mask = student_answer_mask[batch_idx, 1:].bool()
        s = student_logits[batch_idx, :-1][s_mask]
        targets = student_ids[batch_idx, 1:][s_mask]
        teacher_count = int(teacher_answer_mask[batch_idx, 1:].bool().sum().item()) if teacher_answer_mask is not None else int(targets.shape[0])
        count = min(int(s.shape[0]), teacher_count, int(targets.shape[0]))
        if count <= 0:
            continue
        s = s[-count:].float()
        targets = targets[-count:].long()
        losses.append(F.cross_entropy(s, targets, reduction="none"))
    if not losses:
        return student_logits.new_zeros((), dtype=torch.float32)
    return torch.cat(losses).mean()


def masked_unaligned_answer_diagnostics(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_ids: Tensor,
    student_answer_mask: Tensor,
    teacher_answer_mask: Tensor,
) -> dict[str, Tensor]:
    teacher_ce_losses: list[Tensor] = []
    student_correct: list[Tensor] = []
    teacher_correct: list[Tensor] = []
    top1_agree: list[Tensor] = []
    for batch_idx in range(student_logits.shape[0]):
        s_mask = student_answer_mask[batch_idx, 1:].bool()
        t_mask = teacher_answer_mask[batch_idx, 1:].bool()
        s = student_logits[batch_idx, :-1][s_mask]
        t = teacher_logits[batch_idx, :-1][t_mask]
        targets = student_ids[batch_idx, 1:][s_mask]
        count = min(int(s.shape[0]), int(t.shape[0]), int(targets.shape[0]))
        if count <= 0:
            continue
        s = s[-count:].float()
        t = t[-count:].float()
        targets = targets[-count:].long()
        s_top1 = s.argmax(dim=-1)
        t_top1 = t.argmax(dim=-1)
        teacher_ce_losses.append(F.cross_entropy(t, targets, reduction="none"))
        student_correct.append(s_top1.eq(targets).float())
        teacher_correct.append(t_top1.eq(targets).float())
        top1_agree.append(s_top1.eq(t_top1).float())

    if not teacher_ce_losses:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        return {
            "teacher_ce": zero,
            "student_target_acc": zero,
            "teacher_target_acc": zero,
            "top1_agreement": zero,
        }
    return {
        "teacher_ce": torch.cat(teacher_ce_losses).mean(),
        "student_target_acc": torch.cat(student_correct).mean(),
        "teacher_target_acc": torch.cat(teacher_correct).mean(),
        "top1_agreement": torch.cat(top1_agree).mean(),
    }


def gather_text_logits(logits: Tensor, text_positions: Tensor) -> Tensor:
    rows = []
    for batch_idx in range(logits.shape[0]):
        rows.append(logits[batch_idx, text_positions[batch_idx].long()])
    return torch.stack(rows, dim=0)


def gather_text_hidden(hidden: Tensor, text_positions: Tensor) -> Tensor:
    rows = []
    for batch_idx in range(hidden.shape[0]):
        rows.append(hidden[batch_idx, text_positions[batch_idx].long()])
    return torch.stack(rows, dim=0)


def text_alignment_mask(answer_mask: Tensor, text_mask: Tensor, mode: str) -> Tensor:
    if mode == "answer":
        return answer_mask.bool()
    if mode == "all_text":
        return text_mask.bool()
    raise ValueError(f"unknown text alignment mask mode: {mode}")


def selected_layer_indices(num_layers: int, mode: str) -> set[int]:
    if mode == "last":
        return {int(num_layers) - 1}
    if mode == "all":
        return set(range(int(num_layers)))
    raise ValueError(f"unknown layer selection mode: {mode}")


def capture_attention_effects(model: torch.nn.Module, layer_indices: set[int]) -> tuple[list[Tensor], list[Any]]:
    effects: list[Tensor] = []
    handles: list[Any] = []
    layers = model.model.language_model.layers
    for layer_idx in sorted(layer_indices):
        module = layers[layer_idx].self_attn.o_proj

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Tensor, *, _layer_idx: int = layer_idx) -> None:
            effects.append(output)

        handles.append(module.register_forward_hook(hook))
    return effects, handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def masked_effect_mse_alignment(
    student_effects: list[Tensor],
    teacher_effects: list[Tensor],
    text_positions: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if len(student_effects) != len(teacher_effects):
        raise RuntimeError(f"effect count mismatch: student={len(student_effects)} teacher={len(teacher_effects)}")
    losses: list[Tensor] = []
    cosines: list[Tensor] = []
    token_counts: list[Tensor] = []
    valid = mask.bool()
    for student_effect, teacher_effect in zip(student_effects, teacher_effects):
        teacher_text = gather_text_hidden(teacher_effect.detach(), text_positions)
        if student_effect.shape[:2] != teacher_text.shape[:2]:
            raise RuntimeError(f"effect shape mismatch: student={tuple(student_effect.shape)} teacher={tuple(teacher_text.shape)}")
        if not bool(valid.any().item()):
            continue
        student = student_effect[valid].float()
        teacher = teacher_text[valid].float()
        losses.append(F.mse_loss(student, teacher, reduction="mean"))
        cosines.append(F.cosine_similarity(student, teacher, dim=-1))
        token_counts.append(torch.tensor(float(student.shape[0]), device=student_effect.device))
    if not losses:
        zero = student_effects[0].new_zeros((), dtype=torch.float32) if student_effects else torch.tensor(0.0)
        return zero, zero, zero
    loss = torch.stack(losses).mean()
    cosine = torch.cat(cosines).mean()
    tokens = torch.stack(token_counts).sum()
    return loss.to(dtype=student_effects[0].dtype), cosine, tokens


def capture_native_prefill_kv(model: torch.nn.Module, layer_indices: set[int]) -> tuple[dict[int, dict[str, Tensor]], list[Any]]:
    captured: dict[int, dict[str, Tensor]] = {}
    handles: list[Any] = []
    layers = model.model.language_model.layers
    for layer_idx in sorted(layer_indices):
        attn = layers[layer_idx].self_attn

        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            _layer_idx: int = layer_idx,
        ) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("unable to capture native Qwen prefill K/V: missing hidden_states or position_embeddings")
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            _, key = _compile_exact_qwen_apply_rotary_pos_emb(key, key, position_embeddings)
            captured[_layer_idx] = {"key": key.detach(), "value": value.detach()}

        handles.append(attn.register_forward_pre_hook(hook, with_kwargs=True))
    return captured, handles


def _gather_kv_positions(kv: Tensor, positions: Tensor) -> Tensor:
    index = positions.long().clamp_min(0)[:, None, :, None].expand(-1, kv.shape[1], -1, kv.shape[-1])
    return torch.gather(kv, dim=2, index=index)


def _masked_normalized_mse(student: Tensor, teacher: Tensor, mask: Tensor, eps: float) -> Tensor:
    if not bool(mask.any().item()):
        return student.new_zeros((), dtype=torch.float32)
    valid = mask[:, None, :, None].to(device=student.device, dtype=torch.bool)
    diff = (student.float() - teacher.float()).masked_fill(~valid, 0.0)
    ref = teacher.float().masked_fill(~valid, 0.0)
    return diff.square().sum() / ref.square().sum().clamp_min(float(eps))


def _paired_masked_normalized_mse(
    first_student: Tensor,
    first_teacher: Tensor,
    first_mask: Tensor,
    second_student: Tensor,
    second_teacher: Tensor,
    second_mask: Tensor,
    eps: float,
) -> Tensor:
    first_valid = first_mask[:, None, :, None].to(device=first_student.device, dtype=torch.bool)
    second_valid = second_mask[:, None, :, None].to(device=second_student.device, dtype=torch.bool)
    numerator = (first_student.float() - first_teacher.float()).masked_fill(~first_valid, 0.0).square().sum()
    numerator = numerator + (second_student.float() - second_teacher.float()).masked_fill(~second_valid, 0.0).square().sum()
    denominator = first_teacher.float().masked_fill(~first_valid, 0.0).square().sum()
    denominator = denominator + second_teacher.float().masked_fill(~second_valid, 0.0).square().sum()
    return numerator / denominator.clamp_min(float(eps))


def masked_prefill_kv_alignment(
    student_cache: dict[str, Any],
    teacher_kv: dict[int, dict[str, Tensor]],
    *,
    layer_indices: set[int],
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    text_positions = student_cache["text_positions"]
    image_positions = student_cache["image_positions"]
    text_mask = student_cache["text_mask"].bool()
    image_mask = student_cache["image_mask"].bool()
    image_end = image_positions.masked_fill(~image_mask, -1).amax(dim=1, keepdim=True)
    post_image_text_mask = text_mask & (text_positions > image_end)
    valid_tokens = image_mask.sum() + post_image_text_mask.sum()
    losses: list[Tensor] = []
    cosines: list[Tensor] = []
    for layer_idx in sorted(layer_indices):
        student_layer = student_cache["layers"][layer_idx]
        teacher_layer = teacher_kv[layer_idx]
        teacher_image_key = _gather_kv_positions(teacher_layer["key"], image_positions)
        teacher_image_value = _gather_kv_positions(teacher_layer["value"], image_positions)
        teacher_text_key = _gather_kv_positions(teacher_layer["key"], text_positions)
        teacher_text_value = _gather_kv_positions(teacher_layer["value"], text_positions)
        key_loss = _paired_masked_normalized_mse(
            student_layer["visual_key"],
            teacher_image_key,
            image_mask,
            student_layer["text_key"],
            teacher_text_key,
            post_image_text_mask,
            eps,
        )
        value_loss = _paired_masked_normalized_mse(
            student_layer["visual_value"],
            teacher_image_value,
            image_mask,
            student_layer["text_value"],
            teacher_text_value,
            post_image_text_mask,
            eps,
        )
        losses.append(0.5 * (key_loss + value_loss))
        if bool(image_mask.any().item()):
            cosines.append(
                F.cosine_similarity(
                    student_layer["visual_value"].transpose(1, 2)[image_mask].float(),
                    teacher_image_value.transpose(1, 2)[image_mask].float(),
                    dim=-1,
                )
            )
        if bool(post_image_text_mask.any().item()):
            cosines.append(
                F.cosine_similarity(
                    student_layer["text_value"].transpose(1, 2)[post_image_text_mask].float(),
                    teacher_text_value.transpose(1, 2)[post_image_text_mask].float(),
                    dim=-1,
                )
            )
    if not losses:
        zero = next(iter(student_cache["layers"][0].values())).new_zeros((), dtype=torch.float32)
        return zero, zero, zero
    loss = torch.stack(losses).mean()
    cosine = torch.cat(cosines).mean() if cosines else loss.new_zeros((), dtype=torch.float32)
    return loss.to(dtype=next(iter(student_cache["layers"][0].values())).dtype), cosine, valid_tokens.float()


def main() -> None:
    args = parse_args()
    if float(args.lambda_prefill_kv) != 0.0 and args.teacher_mode != "image":
        raise ValueError("--lambda-prefill-kv requires --teacher-mode image")
    needs_teacher_logits = float(args.lambda_logit) != 0.0 or float(args.lambda_effect) != 0.0
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        if args.required_world_size > 1 and dist.get_world_size() != args.required_world_size:
            raise RuntimeError(f"expected {args.required_world_size} ranks, got {dist.get_world_size()}")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    rank = dist.get_rank() if distributed_is_initialized() else 0
    world_size = dist.get_world_size() if distributed_is_initialized() else 1

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if distributed_is_initialized():
        dist.barrier()

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = model.model.language_model
    adapter = QwenEmbeddingAdapter.from_language_model(
        language_model,
        mode=args.output_mode,
        visual_adapter_rank=args.visual_adapter_rank,
    ).to(device=device, dtype=dtype)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
        print(f"loaded init checkpoint {args.init_checkpoint} missing={list(missing)} unexpected={list(unexpected)}", flush=True)

    trainable = trainable_parameters_for_mode(adapter)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if distributed:
        ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
        ds_config["train_micro_batch_size_per_gpu"] = int(args.micro_batch_size_per_gpu)
        ds_config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
        ds_config["gradient_clipping"] = float(args.grad_clip)
        engine, _, _, _ = deepspeed.initialize(
            model=adapter,
            model_parameters=trainable,
            optimizer=optimizer,
            config=ds_config,
        )
    else:
        from src.train import SimpleEngine

        engine = SimpleEngine(adapter, optimizer, args.grad_clip)
    engine.train()
    base_lrs = [float(group.get("lr", args.lr)) for group in optimizer_param_groups(engine)]

    data = JsonlRows(
        args.data,
        max_samples=args.max_samples,
        require_answer_visible=args.require_answer_visible,
        shuffle=True,
        seed=args.seed,
    )
    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    metrics_path = Path(args.metrics_jsonl) if str(args.metrics_jsonl).strip() else output_dir / "train_metrics.jsonl"
    if is_rank0():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"rendered text-teacher train rows={len(data)} world_size={world_size} "
            f"micro_batch={args.micro_batch_size_per_gpu} grad_accum={args.gradient_accumulation_steps} "
            f"global_batch={world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps} "
            f"steps={args.max_steps} trainable={sum(p.numel() for p in trainable)/1e6:.2f}M",
            flush=True,
        )

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled" and is_rank0():
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.parent.name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "dataset_size": len(data),
                "world_size": world_size,
                "global_batch": world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps,
            },
        )

    for step in range(1, int(args.max_steps) + 1):
        start_s = time.perf_counter()
        current_lr = set_engine_lr(engine, base_lrs, lr_multiplier(args, step - 1))
        accum: dict[str, float] = {
            "logit_kl": 0.0,
            "effect_loss": 0.0,
            "effect_cos": 0.0,
            "effect_tokens": 0.0,
            "prefill_kv_loss": 0.0,
            "prefill_kv_cos": 0.0,
            "prefill_kv_tokens": 0.0,
            "student_ce": 0.0,
            "teacher_ce": 0.0,
            "student_target_acc": 0.0,
            "teacher_target_acc": 0.0,
            "top1_agreement": 0.0,
            "loss": 0.0,
            "answer_tokens": 0.0,
        }
        image_paths: list[str] = []
        for micro_idx in range(int(args.gradient_accumulation_steps)):
            sample_base = (
                (step - 1) * int(args.gradient_accumulation_steps) * world_size * int(args.micro_batch_size_per_gpu)
                + micro_idx * world_size * int(args.micro_batch_size_per_gpu)
                + rank * int(args.micro_batch_size_per_gpu)
            )
            rows = data.batch(sample_base, int(args.micro_batch_size_per_gpu))
            student_inputs, student_ids, student_answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
                processor,
                student_rows(rows),
                image_root,
                device,
                include_answers=True,
            )
            assert student_ids is not None and student_answer_mask is not None

            teacher_logits: Tensor | None = None
            teacher_answer_mask: Tensor | None = None
            teacher_effects: list[Tensor] | None = None
            student_effects: list[Tensor] | None = None
            effect_mask = None
            if not needs_teacher_logits:
                student_logits, _, _ = qwen_embedding_adapter_logits(model, engine.module, student_inputs, collect_states=False)
            elif args.teacher_mode == "text":
                    if float(args.lambda_effect) != 0.0:
                        raise ValueError("--lambda-effect requires --teacher-mode image")
                    teacher_inputs, _, teacher_answer_mask = prepare_text_teacher_inputs(
                        processor,
                        rows,
                        device,
                        max_context_chars=args.max_context_chars,
                    )
                    with torch.no_grad():
                        teacher = model(**teacher_inputs, return_dict=True, use_cache=False)
                        teacher_logits = teacher.logits.detach()
                    student_logits, _, _ = qwen_embedding_adapter_logits(model, engine.module, student_inputs, collect_states=False)
            elif args.teacher_mode == "image":
                initial_hidden, position_ids = build_qwen_initial_context(model, student_inputs)
                prepared = prepare_qwen_embedding_adapter_inputs(
                    model,
                    engine.module,
                    student_inputs["input_ids"],
                    student_inputs["attention_mask"],
                    student_inputs["mm_token_type_ids"],
                    initial_hidden,
                    position_ids,
                )
                effect_indices = selected_layer_indices(len(model.model.language_model.layers), args.effect_layers)
                with torch.no_grad():
                    teacher_handles: list[Any] = []
                    if float(args.lambda_effect) != 0.0:
                        teacher_effects, teacher_handles = capture_attention_effects(model, effect_indices)
                    teacher = model(
                        **student_inputs,
                        return_dict=True,
                        use_cache=False,
                        output_hidden_states=False,
                    )
                    remove_hooks(teacher_handles)
                    teacher_logits = gather_text_logits(teacher.logits.detach(), prepared["text_positions"])
                student_handles = []
                if float(args.lambda_effect) != 0.0:
                    student_effects, student_handles = capture_attention_effects(model, effect_indices)
                try:
                    student_logits, _, _ = qwen_embedding_adapter_logits(
                        model,
                        engine.module,
                        student_inputs,
                        initial_hidden=initial_hidden,
                        position_ids=position_ids,
                        collect_states=False,
                    )
                finally:
                    remove_hooks(student_handles)
                teacher_answer_mask = student_answer_mask
                if float(args.lambda_effect) != 0.0:
                    effect_mask = text_alignment_mask(student_answer_mask, prepared["text_mask"], args.effect_mask)
            else:
                raise ValueError(f"unknown teacher mode: {args.teacher_mode}")

            if teacher_logits is not None and teacher_answer_mask is not None:
                kl, answer_counts = masked_unaligned_topk_kl(
                    student_logits,
                    teacher_logits,
                    student_ids,
                    student_answer_mask,
                    teacher_answer_mask,
                    temperature=args.temperature,
                    topk=args.kl_topk,
                )
                diagnostics = masked_unaligned_answer_diagnostics(
                    student_logits,
                    teacher_logits,
                    student_ids,
                    student_answer_mask,
                    teacher_answer_mask,
                )
            else:
                kl = student_logits.new_zeros(())
                answer_counts = student_answer_mask.sum(dim=1).to(device=student_logits.device, dtype=torch.float32)
                zero = student_logits.new_zeros((), dtype=torch.float32)
                diagnostics = {
                    "teacher_ce": zero,
                    "student_target_acc": zero,
                    "teacher_target_acc": zero,
                    "top1_agreement": zero,
                }
            ce = masked_unaligned_student_answer_ce(
                student_logits,
                student_ids,
                student_answer_mask,
                teacher_answer_mask,
            ).to(dtype=student_logits.dtype)
            effect_loss = student_logits.new_zeros(())
            effect_cos = torch.zeros((), device=device, dtype=torch.float32)
            effect_tokens = torch.zeros((), device=device, dtype=torch.float32)
            if float(args.lambda_effect) != 0.0:
                assert student_effects is not None and teacher_effects is not None and effect_mask is not None
                effect_loss, effect_cos, effect_tokens = masked_effect_mse_alignment(
                    student_effects,
                    teacher_effects,
                    prepared["text_positions"],
                    effect_mask,
                )
            prefill_kv_loss = student_logits.new_zeros(())
            prefill_kv_cos = torch.zeros((), device=device, dtype=torch.float32)
            prefill_kv_tokens = torch.zeros((), device=device, dtype=torch.float32)
            if float(args.lambda_prefill_kv) != 0.0:
                prefill_inputs, _, _, _ = prepare_qwen3vl_batch_inputs(
                    processor,
                    student_rows(rows),
                    image_root,
                    device,
                    include_answers=False,
                )
                prefill_initial_hidden, prefill_position_ids = build_qwen_initial_context(model, prefill_inputs)
                prefill_prepared = prepare_qwen_embedding_adapter_inputs(
                    model,
                    engine.module,
                    prefill_inputs["input_ids"],
                    prefill_inputs["attention_mask"],
                    prefill_inputs["mm_token_type_ids"],
                    prefill_initial_hidden,
                    prefill_position_ids,
                )
                prefill_layer_indices = selected_layer_indices(len(model.model.language_model.layers), args.prefill_kv_layers)
                with torch.no_grad():
                    teacher_kv, teacher_kv_handles = capture_native_prefill_kv(model, prefill_layer_indices)
                    try:
                        model(
                            **prefill_inputs,
                            return_dict=True,
                            use_cache=False,
                            output_hidden_states=False,
                        )
                    finally:
                        remove_hooks(teacher_kv_handles)
                _, _, student_cache = qwen_embedding_adapter_prefill_cache_prepared(
                    model,
                    engine.module,
                    h=prefill_prepared["h"],
                    visual_memory=prefill_prepared["visual_memory"],
                    text_mask=prefill_prepared["text_mask"],
                    image_mask=prefill_prepared["image_mask"],
                    text_positions=prefill_prepared["text_positions"],
                    image_positions=prefill_prepared["image_positions"],
                    text_position_ids=prefill_prepared["text_position_ids"],
                    visual_position_ids=prefill_prepared["visual_position_ids"],
                    prefix_attention_mask=prefill_prepared["prefix_attention_mask"],
                    text_position_embeddings=prefill_prepared["text_position_embeddings"],
                    visual_position_embeddings=prefill_prepared["visual_position_embeddings"],
                    logits_to_keep=1,
                )
                prefill_kv_loss, prefill_kv_cos, prefill_kv_tokens = masked_prefill_kv_alignment(
                    student_cache,
                    teacher_kv,
                    layer_indices=prefill_layer_indices,
                    eps=float(args.prefill_kv_eps),
                )
            unscaled_loss = (
                float(args.lambda_logit) * kl
                + float(args.lambda_ce) * ce
                + float(args.lambda_effect) * effect_loss
                + float(args.lambda_prefill_kv) * prefill_kv_loss
            )
            loss = unscaled_loss / float(args.gradient_accumulation_steps)
            engine.backward(loss)
            accum["logit_kl"] += float(kl.detach()) / float(args.gradient_accumulation_steps)
            accum["effect_loss"] += float(effect_loss.detach()) / float(args.gradient_accumulation_steps)
            accum["effect_cos"] += float(effect_cos.detach()) / float(args.gradient_accumulation_steps)
            accum["effect_tokens"] += float(effect_tokens.detach()) / float(args.gradient_accumulation_steps)
            accum["prefill_kv_loss"] += float(prefill_kv_loss.detach()) / float(args.gradient_accumulation_steps)
            accum["prefill_kv_cos"] += float(prefill_kv_cos.detach()) / float(args.gradient_accumulation_steps)
            accum["prefill_kv_tokens"] += float(prefill_kv_tokens.detach()) / float(args.gradient_accumulation_steps)
            accum["student_ce"] += float(ce.detach()) / float(args.gradient_accumulation_steps)
            accum["teacher_ce"] += float(diagnostics["teacher_ce"].detach()) / float(args.gradient_accumulation_steps)
            accum["student_target_acc"] += float(diagnostics["student_target_acc"].detach()) / float(args.gradient_accumulation_steps)
            accum["teacher_target_acc"] += float(diagnostics["teacher_target_acc"].detach()) / float(args.gradient_accumulation_steps)
            accum["top1_agreement"] += float(diagnostics["top1_agreement"].detach()) / float(args.gradient_accumulation_steps)
            accum["loss"] += float(unscaled_loss.detach()) / float(args.gradient_accumulation_steps)
            accum["answer_tokens"] += float(answer_counts.mean().item()) / float(args.gradient_accumulation_steps)
        engine.step()

        if step % int(args.log_every) == 0 or step == 1:
            accum["lr"] = float(current_lr)
            accum["sec_per_step"] = time.perf_counter() - start_s
            reduced = reduce_metrics(accum, device)
            payload = {
                "step": step,
                **reduced,
                "global_batch": int(world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps),
                "image": image_paths[0] if image_paths else "",
            }
            if is_rank0():
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(
                    f"step={step} loss={payload['loss']:.6f} kl={payload['logit_kl']:.6f} "
                    f"effect={payload['effect_loss']:.6f} effect_cos={payload['effect_cos']:.3f} "
                    f"prefill_kv={payload['prefill_kv_loss']:.6f} prefill_kv_cos={payload['prefill_kv_cos']:.3f} "
                    f"student_ce={payload['student_ce']:.6f} teacher_ce={payload['teacher_ce']:.6f} "
                    f"student_acc={payload['student_target_acc']:.3f} teacher_acc={payload['teacher_target_acc']:.3f} "
                    f"agree={payload['top1_agreement']:.3f} "
                    f"answer_tokens={payload['answer_tokens']:.1f} prefill_kv_tokens={payload['prefill_kv_tokens']:.1f} lr={payload['lr']:.3e} "
                    f"global_batch={payload['global_batch']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"train/{k}": v for k, v in payload.items() if isinstance(v, (int, float))}, step=step)

        if step % int(args.save_every) == 0:
            if hasattr(engine, "save_checkpoint"):
                engine.save_checkpoint(str(output_dir / "optimizer"), tag=f"step{step}")
            if is_rank0():
                save_checkpoint(engine.module, output_dir / f"qwen_rendered_text_teacher_step{step}.pt", args, step)
                save_checkpoint(engine.module, output_dir / f"qwen_embedding_adapter_step{step}.pt", args, step)

    if hasattr(engine, "save_checkpoint"):
        engine.save_checkpoint(str(output_dir / "optimizer"), tag="final")
    if is_rank0():
        save_checkpoint(engine.module, output_dir / "qwen_rendered_text_teacher_final.pt", args, int(args.max_steps))
        save_checkpoint(engine.module, output_dir / "qwen_embedding_adapter_final.pt", args, int(args.max_steps))
    if wandb_run is not None:
        import wandb

        wandb.finish()
    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
