from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from delta_vision.models.llava import (
    compute_llama_attention_effect,
    compute_llama_attention_effect_batched,
    compute_llama_text_attention_output,
    gather_batched_positions,
    get_batched_text_and_image_positions,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    run_llama_layer_text_with_attention_delta,
    scatter_batched_positions,
)
from delta_vision.models.modeling import DeltaVisionModel
from delta_vision.runtime.basis import project_delta_to_coefficients, reconstruct_delta
from delta_vision.training.utils import (
    directional_mse,
    hidden_cosine_loss,
    mixed_full_hidden_with_student_text,
    normalized_mse,
    residual_cosine,
    rms_scale_abs_loss,
    rms_scale_loss,
    trajectory_weight,
)


def masked_normalized_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=pred.device, dtype=pred.float().dtype).unsqueeze(-1)
    numerator = (pred.float() - target.float()).pow(2) * valid
    denominator = target.float().pow(2) * valid
    return numerator.sum() / denominator.sum().clamp_min(1e-6)


def masked_directional_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())

    pred_tokens = pred[valid]
    target_tokens = target[valid]
    hidden = pred.shape[-1]
    total = pred.new_zeros((), dtype=torch.float32)
    chunk_size = 2048
    for start in range(0, pred_tokens.shape[0], chunk_size):
        pred_chunk = pred_tokens[start : start + chunk_size].float()
        target_chunk = target_tokens[start : start + chunk_size].float()
        pred_norm = pred_chunk / pred_chunk.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        target_norm = target_chunk / target_chunk.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        total = total + (pred_norm - target_norm).pow(2).sum()
    denom = pred.new_tensor(float(pred_tokens.shape[0] * hidden), dtype=torch.float32).clamp_min(1.0)
    return total / denom


def masked_residual_cosine(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    return F.cosine_similarity(pred.float()[valid], target.float()[valid], dim=-1, eps=1e-6).mean()


def masked_rms_scale_abs_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_rms = pred.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    target_rms = target.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return torch.log(pred_rms[valid] / target_rms[valid]).abs().mean()


def masked_rms_scale_loss_batch(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=pred.device).bool()
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_rms = pred.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    target_rms = target.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return torch.log(pred_rms[valid] / target_rms[valid]).pow(2).mean()


def answer_position_logits(
    lm_head: torch.nn.Module,
    norm: torch.nn.Module,
    h: Tensor,
    teacher_final_h: Tensor,
    text_ids: Tensor,
    answer_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    prediction_mask = answer_mask[:, 1:].to(device=h.device).bool()
    if not bool(prediction_mask.any()):
        vocab = int(lm_head.weight.shape[0])
        return (
            h.new_empty((0, vocab)),
            teacher_final_h.new_empty((0, vocab)),
            text_ids.new_empty((0,), dtype=text_ids.dtype),
        )
    selected_h = h[:, :-1][prediction_mask]
    student_logits = lm_head(norm(selected_h))
    with torch.no_grad():
        selected_teacher_logits = lm_head(teacher_final_h[:, :-1][prediction_mask]).detach()
    selected_targets = text_ids[:, 1:].to(device=h.device)[prediction_mask]
    return student_logits, selected_teacher_logits, selected_targets


def selected_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
    if student_logits.numel() == 0:
        return student_logits.new_zeros(())
    total = student_logits.new_zeros((), dtype=torch.float32)
    chunk_size = 8
    for start in range(0, student_logits.shape[0], chunk_size):
        student_chunk = student_logits[start : start + chunk_size].float() / temperature
        teacher_chunk = teacher_logits[start : start + chunk_size].float() / temperature
        total = total + F.kl_div(
            F.log_softmax(student_chunk, dim=-1),
            F.softmax(teacher_chunk, dim=-1),
            reduction="sum",
        )
    return total * (temperature * temperature) / student_logits.shape[0]


def rollout_losses_for_batch(
    teacher_model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    inputs: dict[str, Tensor],
    text_ids: Tensor,
    answer_mask: Tensor,
    image_token_id: int,
    image_seq_length: int,
    trajectory_layers: set[int],
    active_layers: set[int],
    effect_layers: list[int],
    trajectory_loss_mode: str,
    effect_loss_mode: str,
    effect_input_state: str,
    effect_target_state: str,
    sidecar_scale: float,
    temperature: float,
    dtype: torch.dtype,
    trajectory_late_start: int,
    trajectory_late_weight: float,
    topk_logit_k: int,
    answer_margin_topk: int,
    enable_image_negative: bool,
    image_negative_margin: float,
    image_negative_mode: str,
    rollout_teacher_mix: float,
    sidecar_token_mode: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    del image_negative_margin, image_negative_mode
    if enable_image_negative:
        raise ValueError("batched rollout does not support image-negative loss; disable it or use single-sample rollout")
    if effect_input_state == "teacher" and rollout_model.sidecar.state_tokens > 0:
        raise ValueError("teacher effect_input_state is only implemented for stateless Sidecar")

    device = inputs["input_ids"].device
    batch = inputs["input_ids"].shape[0]
    layers = get_lm_layers(language_model)
    num_layers = len(layers)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    effect_layer_set = set(effect_layers)
    with torch.no_grad():
        teacher = teacher_model.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    teacher_full_states = {idx: state.detach() for idx, state in enumerate(teacher.hidden_states)}
    merged_len = teacher_full_states[0].shape[1]
    positions = get_batched_text_and_image_positions(
        inputs["input_ids"],
        inputs.get("attention_mask"),
        merged_len,
        image_token_id,
        image_seq_length,
    )
    text_mask = positions.text_mask.to(device=device)
    text_padding_mask = ~text_mask
    if sidecar_token_mode == "all":
        sidecar_token_mask = text_mask
    elif sidecar_token_mode == "prompt_only":
        sidecar_token_mask = text_mask & ~answer_mask.to(device=device).bool()
    else:
        raise ValueError(f"unsupported sidecar_token_mode: {sidecar_token_mode}")
    teacher_text_states = {
        state_idx: gather_batched_positions(state, positions.text_positions, text_mask).detach()
        for state_idx, state in teacher_full_states.items()
    }
    vision_states = gather_batched_positions(
        teacher_full_states[0],
        positions.image_positions,
        positions.image_mask,
    ).to(dtype=dtype)

    sidecar = rollout_model.sidecar
    visual_kv = sidecar.prepare_visual_kv(vision_states, ~positions.image_mask.to(device=device))
    sidecar_state = sidecar.initial_state(vision_states, ~positions.image_mask.to(device=device)) if sidecar.state_tokens > 0 else None
    h = teacher_text_states[0].to(dtype=dtype)
    h = h.masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
    position_ids = positions.text_position_ids.to(device=device)

    trajectory_terms: list[Tensor] = []
    trajectory_cos_terms: list[Tensor] = []
    trajectory_rms_terms: list[Tensor] = []
    effect_terms: list[Tensor] = []
    effect_cosines: list[Tensor] = []

    for layer_idx in range(num_layers):
        layer_is_active = layer_idx in active_layers
        supervise_effect = layer_is_active and layer_idx in effect_layer_set
        layer_tensor = torch.full((batch,), layer_idx, device=device, dtype=torch.long)
        pred_effect_delta: Tensor | None = None
        pred_coeff: Tensor | None = None
        text_attention = None
        needs_text_attention = bool(getattr(sidecar, "output_mode", "residual") != "residual") and layer_is_active
        if needs_text_attention:
            text_attention = compute_llama_text_attention_output(
                language_model,
                layer_idx,
                h,
                position_ids,
                padding_mask=text_padding_mask,
                layer=layers[layer_idx],
                rotary_owner=rotary_owner,
            )

        if not layer_is_active:
            attn_delta = None
        elif sidecar.state_tokens > 0:
            if supervise_effect and effect_input_state == "student":
                pred_effect_delta, sidecar_state, pred_coeff = sidecar(
                    h,
                    None,
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    return_coefficients=True,
                    text_attention=text_attention,
                )
                attn_delta = pred_effect_delta
            else:
                attn_delta, sidecar_state = sidecar(
                    h,
                    None,
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    text_attention=text_attention,
                )
        else:
            attn_delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv, text_attention=text_attention)

        if supervise_effect:
            if pred_effect_delta is not None and pred_coeff is not None:
                pass
            elif effect_input_state == "student":
                pred_effect_delta, pred_coeff = sidecar(
                    h,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                    text_attention=text_attention,
                )
            elif effect_input_state == "teacher":
                teacher_effect_input = teacher_text_states[layer_idx].to(dtype=dtype)
                teacher_text_attention = None
                if getattr(sidecar, "output_mode", "residual") != "residual":
                    teacher_text_attention = compute_llama_text_attention_output(
                        language_model,
                        layer_idx,
                        teacher_effect_input,
                        position_ids,
                        padding_mask=text_padding_mask,
                        layer=layers[layer_idx],
                        rotary_owner=rotary_owner,
                    )
                pred_effect_delta, pred_coeff = sidecar(
                    teacher_effect_input,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                    text_attention=teacher_text_attention,
                )
            else:
                raise ValueError(f"unsupported effect input state: {effect_input_state}")

            if effect_target_state == "teacher":
                full_effect_state = teacher_full_states[layer_idx].to(dtype=dtype)
            elif effect_target_state == "student":
                full_effect_state = scatter_batched_positions(
                    teacher_full_states[layer_idx].to(dtype=dtype),
                    positions.text_positions,
                    h,
                    text_mask,
                )
            else:
                raise ValueError(f"unsupported effect target state: {effect_target_state}")
            with torch.no_grad():
                attn_target = compute_llama_attention_effect_batched(
                    language_model,
                    layer_idx,
                    full_effect_state,
                    h if effect_target_state == "student" else teacher_text_states[layer_idx].to(dtype=dtype),
                    positions.text_positions,
                    position_ids,
                    positions.full_mask,
                    text_mask,
                ).detach()
            layer_basis = sidecar.layer_basis(layer_tensor, device, dtype)
            basis_is_trainable = bool(sidecar.basis.requires_grad)
            if basis_is_trainable:
                target_coeff = pred_coeff.detach().new_zeros(pred_coeff.shape)
                residual_target = attn_target
            else:
                with torch.no_grad():
                    target_coeff = project_delta_to_coefficients(attn_target, layer_basis.detach())
                    target_lowrank = reconstruct_delta(target_coeff, layer_basis.detach())
                residual_target = target_lowrank
            pred_lowrank = pred_effect_delta if basis_is_trainable else (
                reconstruct_delta(pred_coeff, layer_basis) * sidecar.gate[layer_tensor].view(batch, 1, 1)
            )
            effect_mask = sidecar_token_mask
            coeff_loss = pred_coeff.new_zeros(()) if basis_is_trainable else masked_normalized_mse(pred_coeff, target_coeff, effect_mask)
            residual_loss = masked_normalized_mse(pred_lowrank, residual_target, effect_mask)
            cosine = masked_residual_cosine(pred_lowrank, residual_target, effect_mask)
            if effect_loss_mode == "coeff":
                effect_terms.append(coeff_loss)
            elif effect_loss_mode == "residual":
                effect_terms.append(residual_loss)
            elif effect_loss_mode == "mixed":
                effect_terms.append(0.2 * coeff_loss + residual_loss)
            else:
                raise ValueError(f"unsupported effect_loss_mode: {effect_loss_mode}")
            effect_cosines.append(cosine)

        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=(
                attn_delta.masked_fill(~sidecar_token_mask.unsqueeze(-1), 0.0) * sidecar_scale
                if attn_delta is not None
                else None
            ),
            padding_mask=text_padding_mask,
            layer=layers[layer_idx],
            rotary_owner=rotary_owner,
        )
        state_idx = layer_idx + 1
        if state_idx in trajectory_layers:
            weight = trajectory_weight(state_idx, trajectory_late_start, trajectory_late_weight)
            target_h = teacher_text_states[state_idx].to(dtype=dtype)
            if trajectory_loss_mode == "direction":
                trajectory_terms.append(weight * masked_directional_mse(h, target_h, text_mask))
                trajectory_rms_terms.append(weight * masked_rms_scale_abs_loss(h, target_h, text_mask))
            elif trajectory_loss_mode == "nmse":
                trajectory_terms.append(weight * masked_normalized_mse(h, target_h, text_mask))
                trajectory_rms_terms.append(weight * masked_rms_scale_loss_batch(h, target_h, text_mask))
            else:
                raise ValueError(f"unsupported trajectory_loss_mode: {trajectory_loss_mode}")
            trajectory_cos_terms.append(weight * (1.0 - masked_residual_cosine(h, target_h, text_mask)))
        if rollout_teacher_mix > 0.0 and state_idx < num_layers:
            teacher_next = teacher_text_states[state_idx].to(device=h.device, dtype=h.dtype)
            h = torch.lerp(h, teacher_next, float(rollout_teacher_mix)).masked_fill(text_padding_mask.unsqueeze(-1), 0.0)

    norm = get_lm_norm(language_model)
    student_answer_logits, teacher_answer_logits, answer_targets = answer_position_logits(
        teacher_model.lm_head,
        norm,
        h,
        teacher_text_states[num_layers].to(dtype=dtype),
        text_ids,
        answer_mask,
    )
    trajectory_loss = torch.stack(trajectory_terms).mean() if trajectory_terms else h.new_zeros(())
    trajectory_cos_loss = torch.stack(trajectory_cos_terms).mean() if trajectory_cos_terms else h.new_zeros(())
    trajectory_rms_loss = torch.stack(trajectory_rms_terms).mean() if trajectory_rms_terms else h.new_zeros(())
    logit_loss = selected_kl(student_answer_logits, teacher_answer_logits, temperature)
    topk_logit_loss = student_answer_logits.new_zeros(())
    answer_margin_loss = student_answer_logits.new_zeros(())
    task_loss = student_answer_logits.new_zeros(())
    image_negative_loss = student_answer_logits.new_zeros(())
    effect_loss = torch.stack(effect_terms).mean() if effect_terms else h.new_zeros(())
    effect_cos = torch.stack(effect_cosines).mean() if effect_cosines else h.new_zeros(())
    metrics = {
        "trajectory": trajectory_loss.detach(),
        "trajectory_cos": trajectory_cos_loss.detach(),
        "trajectory_rms": trajectory_rms_loss.detach(),
        "logit_kl": logit_loss.detach(),
        "topk_logit_kl": topk_logit_loss.detach(),
        "answer_margin": answer_margin_loss.detach(),
        "image_negative": image_negative_loss.detach(),
        "task": task_loss.detach(),
        "effect": effect_loss.detach(),
        "effect_cos": effect_cos.detach(),
        "sidecar_scale": h.new_tensor(sidecar_scale),
    }
    return (
        trajectory_loss,
        trajectory_cos_loss,
        trajectory_rms_loss,
        logit_loss,
        topk_logit_loss,
        answer_margin_loss,
        image_negative_loss,
        task_loss,
        effect_loss,
        effect_cos,
        metrics,
    )


def rollout_losses_for_sample(
    teacher_model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    row: dict[str, Any],
    inputs: dict[str, Tensor],
    text_ids: Tensor,
    answer_mask: Tensor,
    image_token_id: int,
    trajectory_layers: set[int],
    active_layers: set[int],
    effect_layers: list[int],
    trajectory_loss_mode: str,
    effect_loss_mode: str,
    effect_input_state: str,
    effect_target_state: str,
    sidecar_scale: float,
    temperature: float,
    dtype: torch.dtype,
    trajectory_late_start: int,
    trajectory_late_weight: float,
    topk_logit_k: int,
    answer_margin_topk: int,
    enable_image_negative: bool,
    image_negative_margin: float,
    image_negative_mode: str,
    rollout_teacher_mix: float,
    sidecar_token_mode: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    device = inputs["input_ids"].device
    layers = get_lm_layers(language_model)
    num_layers = len(layers)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    effect_layer_set = set(effect_layers)
    with torch.no_grad():
        teacher = teacher_model.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    teacher_full_states = {idx: state.detach() for idx, state in enumerate(teacher.hidden_states)}
    merged_len = teacher_full_states[0].shape[1]
    text_pos, image_pos, text_position_ids = get_text_and_image_positions(
        inputs["input_ids"],
        merged_len,
        image_token_id,
    )
    text_pos = text_pos.to(device)
    image_pos = image_pos.to(device)
    position_ids = text_position_ids.to(device).unsqueeze(0)
    teacher_text_states = {
        state_idx: state.index_select(1, text_pos).detach()
        for state_idx, state in teacher_full_states.items()
    }
    vision_states = teacher_full_states[0].index_select(1, image_pos).to(dtype=dtype)
    token_mask = torch.ones_like(answer_mask, dtype=torch.bool, device=device)
    if sidecar_token_mode == "all":
        sidecar_token_mask = token_mask
    elif sidecar_token_mode == "prompt_only":
        sidecar_token_mask = token_mask & ~answer_mask.to(device=device).bool()
    else:
        raise ValueError(f"unsupported sidecar_token_mode: {sidecar_token_mode}")

    sidecar = rollout_model.sidecar
    visual_kv = sidecar.prepare_visual_kv(vision_states, None)
    sidecar_state = sidecar.initial_state(vision_states, None) if sidecar.state_tokens > 0 else None
    h = teacher_text_states[0].to(dtype=dtype)
    trajectory_terms: list[Tensor] = []
    trajectory_cos_terms: list[Tensor] = []
    trajectory_rms_terms: list[Tensor] = []
    effect_terms: list[Tensor] = []
    effect_cosines: list[Tensor] = []

    for layer_idx in range(num_layers):
        layer_is_active = layer_idx in active_layers
        supervise_effect = layer_is_active and layer_idx in effect_layer_set
        layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
        pred_effect_delta: Tensor | None = None
        pred_coeff: Tensor | None = None
        text_attention = None
        needs_text_attention = bool(getattr(sidecar, "output_mode", "residual") != "residual") and layer_is_active
        if needs_text_attention:
            text_attention = compute_llama_text_attention_output(
                language_model,
                layer_idx,
                h,
                position_ids,
                layer=layers[layer_idx],
                rotary_owner=rotary_owner,
            )
        if not layer_is_active:
            attn_delta = None
        elif sidecar.state_tokens > 0:
            if supervise_effect and effect_input_state == "teacher":
                raise ValueError("teacher effect_input_state is only implemented for stateless Sidecar")
            if supervise_effect and effect_input_state == "student":
                attn_delta, sidecar_state, pred_coeff = sidecar(
                    h,
                    None,
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    return_coefficients=True,
                    text_attention=text_attention,
                )
                pred_effect_delta = attn_delta
            else:
                attn_delta, sidecar_state = sidecar(
                    h,
                    None,
                    layer_tensor,
                    sidecar_state=sidecar_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    text_attention=text_attention,
                )
        else:
            attn_delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv, text_attention=text_attention)

        if supervise_effect:
            if pred_effect_delta is not None and pred_coeff is not None:
                pass
            elif effect_input_state == "student":
                pred_effect_delta, pred_coeff = sidecar(
                    h,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                    text_attention=text_attention,
                )
            elif effect_input_state == "teacher":
                teacher_effect_input = teacher_text_states[layer_idx].to(dtype=dtype)
                teacher_text_attention = None
                if getattr(sidecar, "output_mode", "residual") != "residual":
                    teacher_text_attention = compute_llama_text_attention_output(
                        language_model,
                        layer_idx,
                        teacher_effect_input,
                        position_ids,
                        layer=layers[layer_idx],
                        rotary_owner=rotary_owner,
                    )
                pred_effect_delta, pred_coeff = sidecar(
                    teacher_effect_input,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                    text_attention=teacher_text_attention,
                )
            else:
                raise ValueError(f"unsupported effect input state: {effect_input_state}")
            if effect_target_state == "teacher":
                full_effect_state = teacher_full_states[layer_idx].to(dtype=dtype)
            elif effect_target_state == "student":
                full_effect_state = mixed_full_hidden_with_student_text(
                    teacher_full_states[layer_idx].to(dtype=dtype),
                    text_pos,
                    h,
                )
            else:
                raise ValueError(f"unsupported effect target state: {effect_target_state}")
            with torch.no_grad():
                attn_target = compute_llama_attention_effect(
                    language_model,
                    layer_idx,
                    full_effect_state,
                    text_pos,
                ).detach()
            layer_basis = sidecar.layer_basis(layer_tensor, device, dtype)
            basis_is_trainable = bool(sidecar.basis.requires_grad)
            if basis_is_trainable:
                target_coeff = pred_coeff.detach().new_zeros(pred_coeff.shape)
                residual_target = attn_target
            else:
                with torch.no_grad():
                    target_coeff = project_delta_to_coefficients(attn_target, layer_basis.detach())
                    target_lowrank = reconstruct_delta(target_coeff, layer_basis.detach())
                residual_target = target_lowrank
            pred_lowrank = pred_effect_delta if basis_is_trainable else (
                reconstruct_delta(pred_coeff, layer_basis) * sidecar.gate[layer_tensor].view(1, 1, 1)
            )
            coeff_loss = (
                pred_coeff.new_zeros(())
                if basis_is_trainable
                else masked_normalized_mse(pred_coeff, target_coeff, sidecar_token_mask)
            )
            residual_loss = masked_normalized_mse(pred_lowrank, residual_target, sidecar_token_mask)
            cosine = masked_residual_cosine(pred_lowrank, residual_target, sidecar_token_mask)
            if effect_loss_mode == "coeff":
                effect_terms.append(coeff_loss)
            elif effect_loss_mode == "residual":
                effect_terms.append(residual_loss)
            elif effect_loss_mode == "mixed":
                effect_terms.append(0.2 * coeff_loss + residual_loss)
            else:
                raise ValueError(f"unsupported effect_loss_mode: {effect_loss_mode}")
            effect_cosines.append(cosine)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=(
                attn_delta.masked_fill(~sidecar_token_mask.unsqueeze(-1), 0.0) * sidecar_scale
                if attn_delta is not None
                else None
            ),
            layer=layers[layer_idx],
            rotary_owner=rotary_owner,
        )
        state_idx = layer_idx + 1
        if state_idx in trajectory_layers:
            weight = trajectory_weight(state_idx, trajectory_late_start, trajectory_late_weight)
            target_h = teacher_text_states[state_idx].to(dtype=dtype)
            if trajectory_loss_mode == "direction":
                trajectory_terms.append(weight * directional_mse(h, target_h))
                trajectory_rms_terms.append(weight * rms_scale_abs_loss(h, target_h))
            elif trajectory_loss_mode == "nmse":
                trajectory_terms.append(weight * normalized_mse(h, target_h))
                trajectory_rms_terms.append(weight * rms_scale_loss(h, target_h))
            else:
                raise ValueError(f"unsupported trajectory_loss_mode: {trajectory_loss_mode}")
            trajectory_cos_terms.append(weight * hidden_cosine_loss(h, target_h))
        if rollout_teacher_mix > 0.0 and state_idx < num_layers:
            teacher_next = teacher_text_states[state_idx].to(device=h.device, dtype=h.dtype)
            h = torch.lerp(h, teacher_next, float(rollout_teacher_mix))

    norm = get_lm_norm(language_model)
    student_answer_logits, teacher_answer_logits, answer_targets = answer_position_logits(
        teacher_model.lm_head,
        norm,
        h,
        teacher_text_states[num_layers].to(dtype=dtype),
        text_ids,
        answer_mask,
    )
    trajectory_loss = torch.stack(trajectory_terms).mean() if trajectory_terms else h.new_zeros(())
    trajectory_cos_loss = torch.stack(trajectory_cos_terms).mean() if trajectory_cos_terms else h.new_zeros(())
    trajectory_rms_loss = torch.stack(trajectory_rms_terms).mean() if trajectory_rms_terms else h.new_zeros(())
    logit_loss = selected_kl(student_answer_logits, teacher_answer_logits, temperature)
    topk_logit_loss = student_answer_logits.new_zeros(())
    answer_margin_loss = student_answer_logits.new_zeros(())
    task_loss = student_answer_logits.new_zeros(())
    image_negative_loss = student_answer_logits.new_zeros(())
    if enable_image_negative and image_negative_mode == "zero":
        zero_vision = torch.zeros_like(vision_states)
        neg_visual_kv = sidecar.prepare_visual_kv(zero_vision, None)
        neg_state = sidecar.initial_state(zero_vision, None) if sidecar.state_tokens > 0 else None
        neg_h = teacher_text_states[0].to(dtype=dtype)
        for layer_idx in range(num_layers):
            layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
            if layer_idx not in active_layers:
                neg_attn_delta = None
            elif sidecar.state_tokens > 0:
                neg_attn_delta, neg_state = sidecar(
                    neg_h,
                    None,
                    layer_tensor,
                    sidecar_state=neg_state,
                    visual_kv=neg_visual_kv,
                    return_state=True,
                )
            else:
                neg_attn_delta = sidecar(neg_h, None, layer_tensor, visual_kv=neg_visual_kv)
            neg_h = run_llama_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                neg_h,
                position_ids,
                attention_delta=(
                    neg_attn_delta.masked_fill(~sidecar_token_mask.unsqueeze(-1), 0.0) * sidecar_scale
                    if neg_attn_delta is not None
                    else None
                ),
                layer=layers[layer_idx],
                rotary_owner=rotary_owner,
            )
        neg_answer_logits, _, _ = answer_position_logits(
            teacher_model.lm_head,
            norm,
            neg_h,
            teacher_text_states[num_layers].to(dtype=dtype),
            text_ids,
            answer_mask,
        )
        with torch.no_grad():
            positive_kl = selected_kl(student_answer_logits.detach(), teacher_answer_logits, temperature)
        negative_kl = selected_kl(neg_answer_logits, teacher_answer_logits, temperature)
        image_negative_loss = F.relu(positive_kl + image_negative_margin - negative_kl)
    elif enable_image_negative:
        raise ValueError(f"unsupported image_negative_mode: {image_negative_mode}")

    effect_loss = torch.stack(effect_terms).mean() if effect_terms else h.new_zeros(())
    effect_cos = torch.stack(effect_cosines).mean() if effect_cosines else h.new_zeros(())
    metrics = {
        "trajectory": trajectory_loss.detach(),
        "trajectory_cos": trajectory_cos_loss.detach(),
        "trajectory_rms": trajectory_rms_loss.detach(),
        "logit_kl": logit_loss.detach(),
        "topk_logit_kl": topk_logit_loss.detach(),
        "answer_margin": answer_margin_loss.detach(),
        "image_negative": image_negative_loss.detach(),
        "task": task_loss.detach(),
        "effect": effect_loss.detach(),
        "effect_cos": effect_cos.detach(),
        "sidecar_scale": h.new_tensor(sidecar_scale),
    }
    return (
        trajectory_loss,
        trajectory_cos_loss,
        trajectory_rms_loss,
        logit_loss,
        topk_logit_loss,
        answer_margin_loss,
        image_negative_loss,
        task_loss,
        effect_loss,
        effect_cos,
        metrics,
    )
