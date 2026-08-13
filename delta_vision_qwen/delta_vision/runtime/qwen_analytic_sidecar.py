from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from delta_vision.models.qwen3vl import (
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    qwen3vl_prefix_visual_memory_by_layer,
    qwen3vl_visual_memory_by_layer,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)


class QwenNativeAttentionSidecar(nn.Module):
    """Native Qwen attention-effect oracle with optional per-layer gates.

    This is not a learned effect predictor. Visual readout is computed with the
    frozen Qwen attention operator; trainable gates can only calibrate the
    per-layer effect. With gate=1 it is the exact analytic vprefix oracle.
    """

    def __init__(self, num_layers: int, gate_init: float = 1.0, train_gates: bool = False) -> None:
        super().__init__()
        gates = torch.full((int(num_layers),), float(gate_init))
        if train_gates:
            self.gate = nn.Parameter(gates)
        else:
            self.register_buffer("gate", gates, persistent=True)

    def layer_scale(self, layer_idx: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.gate[int(layer_idx)].to(device=device, dtype=dtype).view(1, 1, 1)


def build_full_from_text_and_memory(
    reference_full: Tensor,
    text_positions: Tensor,
    text_hidden: Tensor,
    text_mask: Tensor,
    image_positions: Tensor,
    image_memory: Tensor,
    image_mask: Tensor,
) -> Tensor:
    """Build a full Qwen hidden sequence from text rollout state and visual memory."""
    full = scatter_batched_positions(reference_full, text_positions, text_hidden, text_mask)
    return scatter_batched_positions(full, image_positions, image_memory, image_mask)


def qwen3vl_visual_memories_for_mode(
    language_model: torch.nn.Module,
    hidden0: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    image_positions: Tensor,
    image_mask: Tensor,
    visual_pos_masks: Tensor,
    deepstack_visual_embeds: list[Tensor] | None,
    mode: str,
) -> list[Tensor]:
    """Return the visual memory used by the Qwen native-effect oracle."""
    if mode == "vprefix":
        return qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            hidden0,
            position_ids,
            attention_mask,
            image_positions,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
    return qwen3vl_visual_memory_by_layer(
        hidden0,
        image_positions,
        image_mask,
        deepstack_visual_embeds,
        mode,
        len(language_model.layers),
    )


def analytic_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    reference_full: Tensor,
    text_hidden: Tensor,
    image_memory: Tensor,
    full_position_ids: Tensor,
    text_position_ids: Tensor,
    text_positions: Tensor,
    image_positions: Tensor,
    full_mask: Tensor,
    text_mask: Tensor,
    image_mask: Tensor,
) -> Tensor:
    """Compute the exact attention-level visual effect for the current text state.

    This is an oracle/teacher target, not prediction: visual tokens are not in
    the main text rollout, but frozen Qwen attention is still used to compute
    the effect that joint Qwen attention would have added to text tokens.
    """
    full = build_full_from_text_and_memory(
        reference_full,
        text_positions,
        text_hidden,
        text_mask,
        image_positions,
        image_memory,
        image_mask,
    )
    return compute_qwen3vl_attention_effect_batched(
        language_model,
        layer_idx,
        full,
        text_hidden,
        full_position_ids,
        text_position_ids,
        text_positions,
        full_mask,
        text_mask,
    )


def run_analytic_qwen3vl_sidecar_only_rollout(
    language_model: torch.nn.Module,
    hidden0: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    text_positions: Tensor,
    image_positions: Tensor,
    text_position_ids: Tensor,
    text_mask: Tensor,
    image_mask: Tensor,
    full_mask: Tensor,
    visual_pos_masks: Tensor,
    deepstack_visual_embeds: list[Tensor] | None,
    visual_memory_mode: str,
    dtype: torch.dtype,
    native_sidecar: QwenNativeAttentionSidecar | None = None,
) -> Tensor:
    """Run Qwen text-only backbone with exact native visual effects.

    The returned hidden states contain only text positions. This function is the
    canonical Qwen native-effect oracle path. It is not a learned sidecar
    predictor because `delta` is computed by frozen Qwen attention at every
    layer.
    """
    h = gather_batched_positions(hidden0, text_positions, text_mask).to(dtype=dtype)
    visual_memories = qwen3vl_visual_memories_for_mode(
        language_model,
        hidden0.to(dtype=dtype),
        position_ids,
        attention_mask,
        image_positions,
        image_mask,
        visual_pos_masks,
        deepstack_visual_embeds,
        visual_memory_mode,
    )
    text_padding_mask = ~text_mask
    for layer_idx in range(len(language_model.layers)):
        delta = analytic_attention_delta(
            language_model,
            layer_idx,
            hidden0.to(dtype=dtype),
            h,
            visual_memories[layer_idx].to(dtype=dtype),
            position_ids,
            text_position_ids,
            text_positions,
            image_positions,
            full_mask,
            text_mask,
            image_mask,
        )
        if native_sidecar is not None:
            delta = delta * native_sidecar.layer_scale(layer_idx, delta.dtype, delta.device)
        h = run_qwen3vl_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            text_position_ids,
            delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
            padding_mask=text_padding_mask,
        )
    return h
