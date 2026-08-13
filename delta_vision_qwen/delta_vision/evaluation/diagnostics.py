from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from delta_vision.models.llava import (
    get_lm_layers,
    run_llama_layer_text_with_attention_delta,
    run_llama_text_attention_residual,
)
from delta_vision.models.modeling import DeltaVisionModel
from delta_vision.runtime.basis import project_delta_to_coefficients, reconstruct_delta
from delta_vision.training.utils import normalized_mse, residual_cosine


@dataclass
class DiagnosticConfig:
    effects_dir: Path
    output_json: Path
    max_samples: int | None = None
    start_sample: int = 0
    layers: str = "0-31"
    sidecar_scale: float = 1.0


def _coeff_cosine(pred: Tensor, target: Tensor) -> Tensor:
    return F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1, eps=1e-6).mean()


def _norm_ratio(pred: Tensor, target: Tensor) -> Tensor:
    pred_norm = pred.float().pow(2).mean().sqrt()
    target_norm = target.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return pred_norm / target_norm


def _parse_layers(spec: str, num_layers: int) -> set[int]:
    spec = spec.strip()
    if not spec or spec == "all":
        return set(range(num_layers))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    if any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"diagnostic layers must be in [0, {num_layers - 1}]")
    return layers


def _add_metric(sums: dict[str, dict[int, float]], counts: dict[str, dict[int, int]], name: str, layer: int, value: Tensor | float) -> None:
    if torch.is_tensor(value):
        scalar = float(value.detach().float().cpu().item())
    else:
        scalar = float(value)
    sums[name][layer] += scalar
    counts[name][layer] += 1


def _finalize(sums: dict[str, dict[int, float]], counts: dict[str, dict[int, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, layer_sums in sums.items():
        out[name] = {
            str(layer): layer_sums[layer] / max(counts[name][layer], 1)
            for layer in sorted(layer_sums)
        }
    return out


@torch.inference_mode()
def diagnose_sidecar_traces(
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    config: DiagnosticConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    manifest = json.loads((config.effects_dir / "manifest.json").read_text(encoding="utf-8"))
    sample_names = manifest["samples"][config.start_sample :]
    if config.max_samples is not None:
        sample_names = sample_names[: config.max_samples]

    sidecar = rollout_model.sidecar
    num_layers = len(get_lm_layers(language_model))
    selected_layers = _parse_layers(config.layers, num_layers)

    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for sample_idx, name in enumerate(sample_names):
        item = torch.load(config.effects_dir / name, map_location="cpu")
        vision = item["vision_tokens"].unsqueeze(0).to(device=device, dtype=dtype)
        visual_kv = sidecar.prepare_visual_kv(vision, None)
        position_ids = item["text_positions"].to(device=device).unsqueeze(0)

        teacher_state = sidecar.initial_state(vision, None) if sidecar.state_tokens > 0 else None
        rollout_state = sidecar.initial_state(vision, None) if sidecar.state_tokens > 0 else None
        h_student = item["teacher_hiddens"][0].unsqueeze(0).to(device=device, dtype=dtype)

        for layer_idx in range(num_layers):
            layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
            teacher_h = item["teacher_hiddens"][layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            target_delta = item["deltas"][layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            layer_basis = sidecar.layer_basis(layer_tensor, device, dtype)
            target_coeff = project_delta_to_coefficients(target_delta, layer_basis)
            target_lowrank = reconstruct_delta(target_coeff, layer_basis)

            if sidecar.state_tokens > 0:
                tf_delta, teacher_state, tf_coeff = sidecar(
                    teacher_h,
                    None,
                    layer_tensor,
                    sidecar_state=teacher_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    return_coefficients=True,
                )
                ro_delta, rollout_state, ro_coeff = sidecar(
                    h_student,
                    None,
                    layer_tensor,
                    sidecar_state=rollout_state,
                    visual_kv=visual_kv,
                    return_state=True,
                    return_coefficients=True,
                )
            else:
                tf_delta, tf_coeff = sidecar(
                    teacher_h,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                )
                ro_delta, ro_coeff = sidecar(
                    h_student,
                    None,
                    layer_tensor,
                    visual_kv=visual_kv,
                    return_coefficients=True,
                )

            if layer_idx in selected_layers:
                _add_metric(sums, counts, "hidden_mse_pre", layer_idx, normalized_mse(h_student, teacher_h))
                _add_metric(sums, counts, "teacher_forced_coeff_cos", layer_idx, _coeff_cosine(tf_coeff, target_coeff))
                _add_metric(sums, counts, "teacher_forced_coeff_nmse", layer_idx, normalized_mse(tf_coeff, target_coeff))
                _add_metric(sums, counts, "teacher_forced_residual_cos", layer_idx, residual_cosine(tf_delta, target_lowrank))
                _add_metric(sums, counts, "teacher_forced_residual_nmse", layer_idx, normalized_mse(tf_delta, target_lowrank))
                _add_metric(sums, counts, "teacher_forced_norm_ratio", layer_idx, _norm_ratio(tf_delta, target_lowrank))
                _add_metric(sums, counts, "rollout_coeff_cos", layer_idx, _coeff_cosine(ro_coeff, target_coeff))
                _add_metric(sums, counts, "rollout_coeff_nmse", layer_idx, normalized_mse(ro_coeff, target_coeff))
                _add_metric(sums, counts, "rollout_residual_cos", layer_idx, residual_cosine(ro_delta, target_lowrank))
                _add_metric(sums, counts, "rollout_residual_nmse", layer_idx, normalized_mse(ro_delta, target_lowrank))
                _add_metric(sums, counts, "rollout_norm_ratio", layer_idx, _norm_ratio(ro_delta, target_lowrank))

                teacher_attn_state = run_llama_text_attention_residual(
                    language_model,
                    layer_idx,
                    teacher_h,
                    position_ids,
                    attention_delta=target_delta,
                )
                rollout_attn_state = run_llama_text_attention_residual(
                    language_model,
                    layer_idx,
                    h_student,
                    position_ids,
                    attention_delta=ro_delta * config.sidecar_scale,
                )
                _add_metric(sums, counts, "attention_state_nmse", layer_idx, normalized_mse(rollout_attn_state, teacher_attn_state))
                _add_metric(sums, counts, "attention_state_cos", layer_idx, residual_cosine(rollout_attn_state, teacher_attn_state))

            h_student = run_llama_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h_student,
                position_ids,
                attention_delta=ro_delta * config.sidecar_scale,
            )
        if (sample_idx + 1) % 25 == 0:
            print(f"diagnosed {sample_idx + 1}/{len(sample_names)}", flush=True)

    metrics = {
        "num_samples": len(sample_names),
        "start_sample": config.start_sample,
        "layers": sorted(selected_layers),
        "metrics": _finalize(sums, counts),
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
