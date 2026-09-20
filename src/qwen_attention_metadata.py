"""Keep single-request FlashAttention metadata separate from sparse RoPE positions.

HF's FA2 heuristic sends pruned positions with gaps to its varlen kernel. Keep that
kernel for identical BF16 numerics, but derive its single-sequence metadata once
per pruned position tensor instead of rebuilding it in every decoder layer.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class AttentionMetadataOptimization:
    handles: list = field(default_factory=list)
    enabled: bool = True
    unpacked: bool = False
    metadata: dict = field(default_factory=dict)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.metadata.clear()


def optimize_qwen_attention_metadata(model) -> AttentionMetadataOptimization:
    """Cache packed-position inference for verified increasing text positions.

Preserve genuine position resets, supplied cu_seqlens, padding masks, all cache
tensors and position_embeddings. Evaluate the sequence condition once before any
layer prunes tokens. This also avoids redundant packed checks in one-token decode.
"""
    language = model.model.language_model
    state = AttentionMetadataOptimization()
    if language.config._attn_implementation != "flash_attention_2":
        state.enabled = False
        return state

    def before_language(module, args, kwargs):
        state.unpacked = False
        state.metadata.clear()
        if not state.enabled:
            return
        positions = kwargs.get("position_ids")
        if positions is None:
            return
        if positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[0]
        elif positions.ndim != 2:
            return
        # A strictly increasing sequence may have pruned gaps, but has no resets.
        state.unpacked = positions.shape[-1] <= 1 or bool(torch.all(positions.diff(dim=-1) > 0))

    def before_attention(module, args, kwargs):
        if state.enabled and state.unpacked and "position_ids" in kwargs:
            positions = kwargs["position_ids"]
            if positions is None:
                return
            metadata_keys = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            supplied = [kwargs.get(key) is not None for key in metadata_keys]
            if any(supplied) and not all(supplied):
                return
            kwargs = dict(kwargs)
            kwargs.pop("position_ids")
            if not all(supplied):
                key = (positions.data_ptr(), tuple(positions.shape), tuple(positions.stride()))
                if key not in state.metadata:
                    from transformers.modeling_flash_attention_utils import _is_packed_sequence
                    metadata = {}
                    # A single position is always equal to its own minimum, so
                    # the FA heuristic is false without any device scalar read.
                    if positions.shape[-1] > 1 and bool(_is_packed_sequence(positions, batch_size=positions.shape[0])):
                        length = positions.shape[-1]
                        cumulative = torch.tensor([0, length], device=positions.device, dtype=torch.int32)
                        metadata = dict(cu_seq_lens_q=cumulative, cu_seq_lens_k=cumulative,
                                        max_length_q=length, max_length_k=length)
                    # Retain the source tensor so a later pruning allocation cannot reuse its pointer.
                    state.metadata[key] = (positions, metadata)
                kwargs.update(state.metadata[key][1])
            return args, kwargs

    state.handles.append(language.register_forward_pre_hook(before_language, with_kwargs=True))
    for layer in language.layers:
        state.handles.append(layer.self_attn.register_forward_pre_hook(before_attention, with_kwargs=True))
    return state
