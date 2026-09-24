"""Independent full-sequence reference for suppressing visual KV at chosen layers."""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS


@contextmanager
def native_blocked_visual_attention(visual_positions, full_length, blocked_layers):
    """Native HF keeps all KV; remove visual rows at attention time, including decode.

This is an audit reference only. The production path instead skips the blocked
adapter modules/projections and stores empty visual KV at those layers.
Native visual-query outputs can be discarded because every layer resets visual
inputs independently, while text continues to propagate.
"""
    original = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
    blocked = set(blocked_layers)
    def attention(module, query, key, value, attention_mask, scaling, **kwargs):
        if module.layer_idx not in blocked:
            return original(module,query,key,value,attention_mask,scaling=scaling,**kwargs)
        from flash_attn.flash_attn_interface import flash_attn_func
        assert attention_mask is None
        valid = torch.ones(key.shape[2],dtype=torch.bool,device=key.device)
        valid[visual_positions] = False
        kt,vt = key[:,:,valid],value[:,:,valid]
        if query.shape[2] == 1:
            qt = query
            query_valid = None
        else:
            assert query.shape[2] == full_length
            query_valid = valid[:full_length]
            qt = query[:,:,query_valid]
        heads = flash_attn_func(qt.transpose(1,2),kt.transpose(1,2),vt.transpose(1,2),
            dropout_p=0.,softmax_scale=scaling,causal=True)
        if query_valid is None:
            return heads,None
        output = query.new_zeros((query.shape[0],query.shape[2],query.shape[1],query.shape[3]))
        output[:,query_valid] = heads
        return output,None
    with patch.dict(ALL_ATTENTION_FUNCTIONS,{'flash_attention_2':attention}):
        yield
