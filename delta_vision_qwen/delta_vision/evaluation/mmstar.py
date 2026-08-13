from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from delta_vision.models.llava import (
    compute_llama_text_attention_output,
    get_lm_layers,
    get_lm_norm,
    run_llama_layer_text_only,
    run_llama_layer_text_with_attention_delta,
)
from delta_vision.models.modeling import DeltaVisionModel
from delta_vision.data import parse_int_set, parse_zero_based_layers
from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option


@dataclass
class TraceEvalConfig:
    effects_dir: Path
    output_json: Path
    start_sample: int = 0
    max_samples: int | None = None
    batch_size: int = 1
    hidden_distance_layers: str = "0,8,16,24,32"
    active_layers: str = "all"
    sidecar_scale: float = 1.0


def _pad_sequence_tensors(tensors: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("empty tensor list")
    max_len = max(int(t.shape[0]) for t in tensors)
    hidden = int(tensors[0].shape[-1])
    out = torch.zeros((len(tensors), max_len, hidden), device=device, dtype=dtype)
    mask = torch.zeros((len(tensors), max_len), device=device, dtype=torch.bool)
    for idx, tensor in enumerate(tensors):
        seq_len = int(tensor.shape[0])
        out[idx, :seq_len] = tensor.to(device=device, dtype=dtype)
        mask[idx, :seq_len] = True
    return out, mask


def _pad_vector_tensors(tensors: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("empty tensor list")
    max_len = max(int(t.shape[0]) for t in tensors)
    out = torch.zeros((len(tensors), max_len), device=device, dtype=tensors[0].dtype)
    mask = torch.zeros((len(tensors), max_len), device=device, dtype=torch.bool)
    for idx, tensor in enumerate(tensors):
        seq_len = int(tensor.shape[0])
        out[idx, :seq_len] = tensor.to(device=device)
        mask[idx, :seq_len] = True
    return out, mask


@torch.inference_mode()
def teacher_final_hidden(language_model: torch.nn.Module, item: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    num_layers = len(get_lm_layers(language_model))
    h31 = item["teacher_hiddens"][num_layers - 1].unsqueeze(0).to(device=device, dtype=dtype)
    position_ids = item["text_positions"].to(device=device).unsqueeze(0)
    h32_text = run_llama_layer_text_only(language_model, num_layers - 1, h31, position_ids)
    return h32_text + item["deltas"][num_layers - 1].unsqueeze(0).to(device=device, dtype=dtype)


@torch.inference_mode()
def teacher_final_hidden_batch(
    language_model: torch.nn.Module,
    items: list[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers = len(get_lm_layers(language_model))
    h31, text_mask = _pad_sequence_tensors([item["teacher_hiddens"][num_layers - 1] for item in items], device, dtype)
    delta31, _ = _pad_sequence_tensors([item["deltas"][num_layers - 1] for item in items], device, dtype)
    position_ids, _ = _pad_vector_tensors([item["text_positions"] for item in items], device)
    h32 = run_llama_layer_text_with_attention_delta(
        language_model,
        num_layers - 1,
        h31,
        position_ids,
        attention_delta=delta31,
        padding_mask=~text_mask,
    )
    return h32, text_mask


@torch.inference_mode()
def rollout_student(
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    item: dict,
    hidden_distance_layers: set[int],
    active_layers: set[int],
    sidecar_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[int, float]]:
    sidecar = rollout_model.sidecar
    h = item["teacher_hiddens"][0].unsqueeze(0).to(device=device, dtype=dtype)
    vision = item["vision_tokens"].unsqueeze(0).to(device=device, dtype=dtype)
    visual_kv = sidecar.prepare_visual_kv(vision, None)
    position_ids = item["text_positions"].to(device=device).unsqueeze(0)
    num_layers = len(get_lm_layers(language_model))
    hidden_mse = {}
    state = sidecar.initial_state(vision, None) if sidecar.state_tokens > 0 else None
    for layer_idx in range(num_layers):
        if layer_idx in hidden_distance_layers:
            teacher_h = item["teacher_hiddens"][layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            hidden_mse[layer_idx] = float((h.float() - teacher_h.float()).pow(2).mean().item())
        layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
        text_attention = None
        if getattr(sidecar, "output_mode", "residual") != "residual" and layer_idx in active_layers:
            text_attention = compute_llama_text_attention_output(language_model, layer_idx, h, position_ids)
        if layer_idx not in active_layers:
            attn_delta = None
        elif sidecar.state_tokens > 0:
            attn_delta, state = sidecar(
                h,
                None,
                layer_tensor,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
                text_attention=text_attention,
            )
        else:
            attn_delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv, text_attention=text_attention)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=attn_delta * sidecar_scale if attn_delta is not None else None,
        )
    if num_layers in hidden_distance_layers:
        teacher_h = teacher_final_hidden(language_model, item, device, dtype)
        hidden_mse[num_layers] = float((h.float() - teacher_h.float()).pow(2).mean().item())
    return h, hidden_mse


@torch.inference_mode()
def rollout_student_batch(
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    items: list[dict],
    hidden_distance_layers: set[int],
    active_layers: set[int],
    sidecar_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[float]]]:
    sidecar = rollout_model.sidecar
    h, text_mask = _pad_sequence_tensors([item["teacher_hiddens"][0] for item in items], device, dtype)
    vision, vision_mask = _pad_sequence_tensors([item["vision_tokens"] for item in items], device, dtype)
    position_ids, _ = _pad_vector_tensors([item["text_positions"] for item in items], device)
    visual_kv = sidecar.prepare_visual_kv(vision, ~vision_mask)
    state = sidecar.initial_state(vision, ~vision_mask) if sidecar.state_tokens > 0 else None
    num_layers = len(get_lm_layers(language_model))
    hidden_mse: dict[int, list[float]] = {layer: [] for layer in hidden_distance_layers}
    for layer_idx in range(num_layers):
        if layer_idx in hidden_distance_layers:
            teacher_h, _ = _pad_sequence_tensors([item["teacher_hiddens"][layer_idx] for item in items], device, dtype)
            per_token = (h.float() - teacher_h.float()).pow(2).mean(dim=-1)
            per_sample = (per_token * text_mask.float()).sum(dim=1) / text_mask.float().sum(dim=1).clamp_min(1.0)
            hidden_mse[layer_idx].extend(float(x) for x in per_sample.cpu())
        layer_tensor = torch.full((len(items),), layer_idx, device=device, dtype=torch.long)
        text_attention = None
        if getattr(sidecar, "output_mode", "residual") != "residual" and layer_idx in active_layers:
            text_attention = compute_llama_text_attention_output(
                language_model,
                layer_idx,
                h,
                position_ids,
                padding_mask=~text_mask,
            )
        if layer_idx not in active_layers:
            attn_delta = None
        elif sidecar.state_tokens > 0:
            attn_delta, state = sidecar(
                h,
                None,
                layer_tensor,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
                text_attention=text_attention,
            )
        else:
            attn_delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv, text_attention=text_attention)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=attn_delta * sidecar_scale if attn_delta is not None else None,
            padding_mask=~text_mask,
        )
    if num_layers in hidden_distance_layers:
        teacher_h, _ = teacher_final_hidden_batch(language_model, items, device, dtype)
        per_token = (h.float() - teacher_h.float()).pow(2).mean(dim=-1)
        per_sample = (per_token * text_mask.float()).sum(dim=1) / text_mask.float().sum(dim=1).clamp_min(1.0)
        hidden_mse[num_layers].extend(float(x) for x in per_sample.cpu())
    return h, text_mask, hidden_mse


@torch.inference_mode()
def evaluate_mmstar_traces(
    processor,
    model,
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    config: TraceEvalConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    manifest = json.loads((config.effects_dir / "manifest.json").read_text(encoding="utf-8"))
    sample_names = manifest["samples"][config.start_sample :]
    if config.max_samples is not None:
        sample_names = sample_names[: config.max_samples]

    norm = get_lm_norm(language_model)
    option_ids = option_token_id_lists(processor.tokenizer)
    hidden_layers = parse_int_set(config.hidden_distance_layers)
    if config.active_layers == "all":
        active_layers = set(range(len(get_lm_layers(language_model))))
    else:
        active_layers = set(parse_zero_based_layers(config.active_layers, len(get_lm_layers(language_model))))
    correct = 0
    agree = 0
    teacher_correct = 0
    teacher_correct_agree = 0
    kl_sum = 0.0
    hidden_sums = {layer: 0.0 for layer in hidden_layers}
    hidden_counts = {layer: 0 for layer in hidden_layers}

    batch_size = max(1, int(config.batch_size))
    for batch_start in range(0, len(sample_names), batch_size):
        names = sample_names[batch_start : batch_start + batch_size]
        items = [torch.load(config.effects_dir / name, map_location="cpu") for name in names]
        teacher_h, text_mask = teacher_final_hidden_batch(language_model, items, device, dtype)
        student_h, _, hidden_mse = rollout_student_batch(
            language_model,
            rollout_model,
            items,
            hidden_layers,
            active_layers,
            config.sidecar_scale,
            device,
            dtype,
        )
        teacher_all_logits = model.lm_head(norm(teacher_h))
        student_all_logits = model.lm_head(norm(student_h))
        last_indices = text_mask.long().sum(dim=1).sub(1).clamp_min(0)
        for item_idx, item in enumerate(items):
            gold = str(item["sample"]["answer"]).strip().upper()[:1]
            idx = int(last_indices[item_idx].item())
            teacher_logits = teacher_all_logits[item_idx, idx]
            student_logits = student_all_logits[item_idx, idx]
            teacher_pred = predict_option(teacher_logits, option_ids)
            teacher_dist = option_distribution(teacher_logits, option_ids)
            student_pred = predict_option(student_logits, option_ids)
            student_dist = option_distribution(student_logits, option_ids)

            correct += int(student_pred == gold)
            agree += int(student_pred == teacher_pred)
            is_teacher_correct = teacher_pred == gold
            teacher_correct += int(is_teacher_correct)
            teacher_correct_agree += int(is_teacher_correct and student_pred == teacher_pred)
            kl_sum += float(F.kl_div(student_dist.log(), teacher_dist, reduction="sum").item())
        for layer, values in hidden_mse.items():
            hidden_sums[layer] += sum(values)
            hidden_counts[layer] += len(values)
        done = min(batch_start + len(names), len(sample_names))
        if done % 50 == 0 or done == len(sample_names):
            print(f"evaluated {done}/{len(sample_names)}", flush=True)

    n = len(sample_names)
    metrics = {
        "num_samples": n,
        "correct": correct,
        "agree": agree,
        "accuracy": correct / max(n, 1),
        "teacher_agreement": agree / max(n, 1),
        "teacher_correct_and_agree": teacher_correct_agree,
        "teacher_correct_retention": teacher_correct_agree / max(teacher_correct, 1),
        "output_kl_sum": kl_sum,
        "output_kl": kl_sum / max(n, 1),
        "teacher_correct": teacher_correct,
        "active_layers": sorted(active_layers),
        "mean_hidden_mse": {
            str(layer): hidden_sums[layer] / max(hidden_counts[layer], 1)
            for layer in sorted(hidden_sums)
        },
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
