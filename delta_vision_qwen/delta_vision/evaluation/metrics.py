from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

OPTIONS = ("A", "B", "C", "D")


def option_token_ids(tokenizer: Any, device: torch.device | None = None) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for option in OPTIONS:
        ids: set[int] = set()
        for text in (option, f" {option}", f"{option}.", f" {option}.", f"{option}:", f" {option}:"):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if encoded:
                ids.add(int(encoded[-1]))
        if not ids:
            raise ValueError(f"could not encode option {option}")
        out[option] = torch.tensor(sorted(ids), device=device, dtype=torch.long)
    return out


def option_token_id_lists(tokenizer: Any) -> dict[str, list[int]]:
    return {key: value.cpu().tolist() for key, value in option_token_ids(tokenizer, None).items()}


def option_scores(logits: Tensor, option_ids: dict[str, Tensor | list[int]]) -> Tensor:
    scores = []
    for option in OPTIONS:
        ids = option_ids[option]
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, device=logits.device, dtype=torch.long)
        scores.append(logits.index_select(0, ids.to(device=logits.device)).max())
    return torch.stack(scores, dim=0)


def predict_option(logits: Tensor, option_ids: dict[str, Tensor | list[int]]) -> str:
    scores = option_scores(logits, option_ids)
    return OPTIONS[int(scores.argmax().item())]


def option_distribution(logits: Tensor, option_ids: dict[str, Tensor | list[int]]) -> Tensor:
    return F.softmax(option_scores(logits, option_ids).float(), dim=0)


def has_choice_fields(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("choices", "options", "candidates", "A", "B", "C", "D"))


def first_answer_prediction_index(answer_mask: Tensor) -> int | None:
    answer_positions = torch.nonzero(answer_mask[0], as_tuple=False).flatten()
    if answer_positions.numel() == 0:
        return None
    first_answer = int(answer_positions[0].item())
    if first_answer <= 0:
        return None
    return first_answer - 1


def masked_ce(student_logits: Tensor, target_ids: Tensor, answer_mask: Tensor) -> Tensor:
    shift_logits = student_logits[:, :-1].float()
    shift_targets = target_ids[:, 1:]
    shift_mask = answer_mask[:, 1:]
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_targets.reshape(-1),
        reduction="none",
    ).view_as(shift_targets)
    denom = shift_mask.float().sum().clamp_min(1.0)
    return (losses * shift_mask.float()).sum() / denom


def masked_kl(student_logits: Tensor, teacher_logits: Tensor, answer_mask: Tensor, temperature: float) -> Tensor:
    shift_student = student_logits[:, :-1].float() / temperature
    shift_teacher = teacher_logits[:, :-1].float() / temperature
    shift_mask = answer_mask[:, 1:]
    kl = F.kl_div(
        F.log_softmax(shift_student, dim=-1),
        F.softmax(shift_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    denom = shift_mask.float().sum().clamp_min(1.0)
    return (kl * shift_mask.float()).sum() * (temperature * temperature) / denom


def masked_topk_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_ids: Tensor,
    answer_mask: Tensor,
    temperature: float,
    k: int,
) -> Tensor:
    if k <= 0:
        return student_logits.new_zeros(())
    shift_student = student_logits[:, :-1].float()
    shift_teacher = teacher_logits[:, :-1].float()
    shift_targets = target_ids[:, 1:]
    shift_mask = answer_mask[:, 1:]
    if shift_mask.float().sum() == 0:
        return student_logits.new_zeros(())
    k_eff = min(k, shift_teacher.shape[-1])
    topk = torch.topk(shift_teacher, k=k_eff, dim=-1).indices
    target_idx = shift_targets.unsqueeze(-1)
    target_in_topk = topk.eq(target_idx).any(dim=-1, keepdim=True)
    if k_eff == shift_teacher.shape[-1]:
        gather_idx = topk
    else:
        gather_idx = torch.where(target_in_topk, topk, torch.cat([topk[..., :-1], target_idx], dim=-1))
    gathered_teacher = torch.gather(shift_teacher, dim=-1, index=gather_idx) / temperature
    gathered_student = torch.gather(shift_student, dim=-1, index=gather_idx) / temperature
    kl = F.kl_div(
        F.log_softmax(gathered_student, dim=-1),
        F.softmax(gathered_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    denom = shift_mask.float().sum().clamp_min(1.0)
    return (kl * shift_mask.float()).sum() * (temperature * temperature) / denom


def masked_answer_margin_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_ids: Tensor,
    answer_mask: Tensor,
    k: int,
) -> Tensor:
    if k <= 0:
        return student_logits.new_zeros(())
    shift_student = student_logits[:, :-1].float()
    shift_teacher = teacher_logits[:, :-1].float()
    shift_targets = target_ids[:, 1:]
    shift_mask = answer_mask[:, 1:]
    if shift_mask.float().sum() == 0:
        return student_logits.new_zeros(())
    k_eff = min(k, shift_teacher.shape[-1])
    topk = torch.topk(shift_teacher, k=k_eff, dim=-1).indices
    teacher_top = torch.gather(shift_teacher, dim=-1, index=topk)
    student_top = torch.gather(shift_student, dim=-1, index=topk)
    is_target = topk.eq(shift_targets.unsqueeze(-1))
    teacher_other = teacher_top.masked_fill(is_target, float("-inf")).amax(dim=-1)
    student_other = student_top.masked_fill(is_target, float("-inf")).amax(dim=-1)
    all_target = torch.isinf(teacher_other)
    teacher_other = torch.where(all_target, teacher_top.amin(dim=-1), teacher_other)
    student_other = torch.where(all_target, student_top.amin(dim=-1), student_other)
    teacher_target = torch.gather(shift_teacher, dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
    student_target = torch.gather(shift_student, dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
    teacher_margin = teacher_target - teacher_other
    student_margin = student_target - student_other
    margin_loss = (student_margin - teacher_margin).pow(2)
    denom = shift_mask.float().sum().clamp_min(1.0)
    return (margin_loss * shift_mask.float()).sum() / denom
