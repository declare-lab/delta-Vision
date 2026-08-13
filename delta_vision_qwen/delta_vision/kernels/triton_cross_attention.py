from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on CUDA runtime
    triton = None
    tl = None


def _enabled() -> bool:
    return triton is not None and os.environ.get("VISUAL_SIDECAR_USE_TRITON_CROSS_ATTN", "0") == "1"


if triton is not None:

    @triton.jit
    def _decode_cross_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        vision_len: tl.constexpr,
        head_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        query_heads: tl.constexpr,
        key_heads: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        scale: tl.constexpr,
    ) -> None:
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        kv_group = query_heads // key_heads
        kv_head_id = head_id // kv_group
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        mask_n = offs_n < vision_len
        mask_d = offs_d < head_dim

        q = tl.load(
            q_ptr + batch_id * q_stride_b + head_id * q_stride_h + offs_d * q_stride_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + batch_id * k_stride_b
            + kv_head_id * k_stride_h
            + offs_n[:, None] * k_stride_n
            + offs_d[None, :] * k_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask_n, scores, -float("inf"))
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=0)

        value = tl.load(
            v_ptr
            + batch_id * v_stride_b
            + kv_head_id * v_stride_h
            + offs_n[:, None] * v_stride_n
            + offs_d[None, :] * v_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(value * probs[:, None], axis=0)
        tl.store(
            out_ptr + batch_id * out_stride_b + head_id * out_stride_h + offs_d * out_stride_d,
            out,
            mask=mask_d,
        )


def triton_decode_cross_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor | None:
    """Decode-only cross-attention for query [B,1,H,D], key/value [B,H,N,D].

    Returns [B,1,H,D] or None when the Triton fast path is not applicable.
    """
    if not _enabled() or not query.is_cuda:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    if query.shape[1] != 1 or key.shape != value.shape:
        return None
    batch, _, heads, head_dim = query.shape
    key_heads = key.shape[1]
    if key.shape[0] != batch or key.shape[3] != head_dim:
        return None
    if heads != key_heads and heads % key_heads != 0:
        return None
    vision_len = key.shape[2]
    if head_dim > 256 or vision_len > 2048:
        return None
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        return None

    out = torch.empty((batch, heads, head_dim), device=query.device, dtype=query.dtype)
    block_n = triton.next_power_of_2(vision_len)
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch, heads)
    _decode_cross_attn_kernel[grid](
        query,
        key,
        value,
        out,
        vision_len,
        head_dim,
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        heads,
        key_heads,
        block_n,
        block_d,
        head_dim ** -0.5,
        num_warps=8,
    )
    return out.unsqueeze(1)
