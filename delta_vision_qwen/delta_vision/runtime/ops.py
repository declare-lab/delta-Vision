from __future__ import annotations

from dataclasses import dataclass
import os

import torch
from torch import Tensor
from torch.nn import functional as F

from delta_vision.kernels.triton_cross_attention import triton_decode_cross_attention

USE_TRITON_CROSS_ATTENTION = os.environ.get("VISUAL_SIDECAR_USE_TRITON_CROSS_ATTN", "0") == "1"


@dataclass
class VisualKVCache:
    """Projected static visual memory for sidecar cross-attention."""

    # Stored in scaled_dot_product_attention layout: [batch, heads, vision_len, head_dim].
    # For GQA, heads can be fewer than the query heads.
    key: Tensor
    value: Tensor
    padding_mask: Tensor | None = None


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    if x.ndim != 3:
        raise ValueError("expected [batch, seq, dim]")
    batch, seq_len, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError("hidden dim must be divisible by num_heads")
    return x.view(batch, seq_len, num_heads, dim // num_heads)


def merge_heads(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError("expected [batch, seq, heads, head_dim]")
    batch, seq_len, num_heads, head_dim = x.shape
    return x.reshape(batch, seq_len, num_heads * head_dim)


def cross_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Low-level sidecar cross-attention.

    Args:
        query: [batch, query_len, heads, head_dim].
        key: [batch, heads, vision_len, head_dim].
        value: [batch, heads, vision_len, head_dim].
        padding_mask: optional bool tensor [batch, vision_len], True for padding.

    Returns:
        [batch, query_len, heads, head_dim].
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes differ")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query and key batch/head dimensions are incompatible")
    query_heads = int(query.shape[2])
    key_heads = int(key.shape[1])
    if query_heads != key_heads and query_heads % key_heads != 0:
        raise ValueError("query heads must equal or be a multiple of key/value heads")
    if USE_TRITON_CROSS_ATTENTION and padding_mask is None and not training and dropout_p == 0.0:
        fast_out = triton_decode_cross_attention(query, key, value)
        if fast_out is not None:
            return fast_out.contiguous()

    query_t = query.transpose(1, 2)
    attn_mask = None
    if padding_mask is not None:
        if padding_mask.shape != (key.shape[0], key.shape[2]):
            raise ValueError("padding_mask must have shape [batch, vision_len]")
        attn_mask = torch.zeros(
            (query.shape[0], 1, 1, key.shape[2]),
            device=query.device,
            dtype=query.dtype,
        )
        attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], torch.finfo(query.dtype).min)

    try:
        out = F.scaled_dot_product_attention(
            query_t,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
            enable_gqa=query_heads != key_heads,
        )
    except TypeError:
        if query_heads != key_heads:
            repeat = query_heads // key_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            query_t,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
        )
    return out.transpose(1, 2).contiguous()
