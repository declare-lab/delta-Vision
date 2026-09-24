"""Remove text-to-visual access at selected Qwen3.5 FA layers, including decode.

Visual queries retain native causal attention. Keys/values remain in the full
cache; only the readout for text queries uses a temporary text-only KV view.
No model-global attention registry or linear-attention layer is modified.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def text_indices(visual_mask, query_length, key_length):
    """Single unpadded prompt, then single-token cached decoding only."""
    assert visual_mask.ndim == 2 and visual_mask.shape[0] == 1
    n = visual_mask.shape[1]
    assert key_length >= n
    if query_length == key_length:
        assert key_length == n, 'Only the initial full prompt is supported'
    else:
        assert query_length == 1 and key_length > n, 'Expected cached one-token decode'
    kv_visual = torch.cat((visual_mask[0], visual_mask.new_zeros(key_length - n)))
    q_visual = kv_visual[-query_length:]
    return (~q_visual).nonzero().flatten(), (~kv_visual).nonzero().flatten()


def block_visual_readout(module, query, key, value, visual_mask, interface, **kwargs):
    """Q/K/V already include native normalization and original rotary positions."""
    qi, ki = text_indices(visual_mask, query.shape[2], key.shape[2])
    # Preserve the native result on visual queries exactly. Compressed text order
    # preserves original causal ordering, including the pre-image text prefix.
    full, weights = interface(module, query, key, value, None,
                              dropout=0.0, scaling=module.scaling, **kwargs)
    text, _ = interface(module, query.index_select(2, qi), key.index_select(2, ki),
                        value.index_select(2, ki), None, dropout=0.0,
                        scaling=module.scaling)
    return full.index_copy(1, qi, text), weights


@contextmanager
def remove_full_attention_visual_effect(model, visual_mask, layers, *, block=True):
    """With block=False, exercise the same forward copy as a native parity gate."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
    )
    originals = []
    audit = {'layers': list(layers), 'block': block, 'calls': []}

    def forward(module, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        assert not module.training
        assert hidden_states.shape[0] == 1
        assert module.config._attn_implementation == 'flash_attention_2'
        assert attention_mask is None or bool(attention_mask.eq(1).all()), 'Padding unsupported'
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        q, gate = torch.chunk(module.q_proj(hidden_states).view(
            *input_shape, -1, module.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        q = module.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, module.layer_idx)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            module.config._attn_implementation, eager_attention_forward)
        if block:
            out, weights = block_visual_readout(module, q, k, v, visual_mask, interface, **kwargs)
        else:
            out, weights = interface(module, q, k, v, attention_mask,
                                     dropout=0.0, scaling=module.scaling, **kwargs)
        audit['calls'].append(dict(layer=module.layer_idx, query_length=q.shape[2],
                                   cache_length=k.shape[2]))
        out = out.reshape(*input_shape, -1).contiguous()
        return module.o_proj(out * torch.sigmoid(gate)), weights

    try:
        assert len(layers) == len(set(layers))
        for i in layers:
            layer = model.model.language_model.layers[i]
            assert layer.block_type == 'full_attention', i
            module = layer.self_attn
            originals.append((module, module.forward))
            module.forward = MethodType(forward, module)
        yield audit
    finally:
        for module, original in originals:
            module.forward = original
