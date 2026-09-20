"""Exact inference-only copies and BF16-rounded RoPE for the FA2 adapter.

Keep both BF16 multiplications before the RoPE sum; ordinary compiler fusion
would change rounding. Attention still uses the same FA2 varlen kernel/plan.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X, C, S, Y, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, COUNT: tl.constexpr,
          X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
          Y0: tl.constexpr, Y1: tl.constexpr, Y2: tl.constexpr, Y3: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    d, n, h, b = i % D, i // D % N, i // (D * N) % H, i // (D * N * H)
    offset = b * X0 + h * X1 + n * X2
    x = tl.load(X + offset + d * X3, valid, other=0).to(tl.float32)
    rotated = tl.load(X + offset + ((d + D // 2) % D) * X3, valid, other=0).to(tl.float32)
    rotated = tl.where(d < D // 2, -rotated, rotated)
    cos = tl.load(C + b * C0 + n * C1 + d * C2, valid, other=0).to(tl.float32)
    sin = tl.load(S + b * S0 + n * S1 + d * S2, valid, other=0).to(tl.float32)
    first = (x * cos).to(Y.dtype.element_ty).to(tl.float32)
    second = (rotated * sin).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + b * Y0 + h * Y1 + n * Y2 + d * Y3, first + second, valid)


def exact_rope(states, embeddings):
    cos, sin = embeddings
    if (states.ndim != 4 or cos.ndim != 3 or sin.ndim != 3 or states.shape[-1] % 2
        or states.dtype != torch.bfloat16 or cos.dtype != states.dtype or sin.dtype != states.dtype
        or not states.is_cuda or torch.is_grad_enabled()):
        raise ValueError('Exact adapter RoPE requires inference with CUDA BF16 tensors')
    output = torch.empty_like(states)
    _, heads, length, dim = states.shape
    _rope[(triton.cdiv(states.numel(), 256),)](states, cos, sin, output, heads, length, dim,
        states.numel(), *states.stride(), *output.stride(), *cos.stride(), *sin.stride(), 256,
        enable_fp_fusion=False)
    return output


@triton.jit
def _pack_split(VK, VV, TK, TV, IDX, OUT, H: tl.constexpr, D: tl.constexpr,
                VLEN: tl.constexpr, COUNT: tl.constexpr,
                VKH: tl.constexpr, VKN: tl.constexpr, VKD: tl.constexpr,
                VVH: tl.constexpr, VVN: tl.constexpr, VVD: tl.constexpr,
                TKH: tl.constexpr, TKN: tl.constexpr, TKD: tl.constexpr,
                TVH: tl.constexpr, TVN: tl.constexpr, TVD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    d, h, n = i % D, i // D % H, i // (D * H)
    source = tl.load(IDX + n, valid, other=0)
    visual = source < VLEN
    if tl.program_id(1) == 0:
        v = tl.load(VK + h * VKH + source * VKN + d * VKD, valid & visual, other=0)
        t = tl.load(TK + h * TKH + (source - VLEN) * TKN + d * TKD, valid & ~visual, other=0)
    else:
        v = tl.load(VV + h * VVH + source * VVN + d * VVD, valid & visual, other=0)
        t = tl.load(TV + h * TVH + (source - VLEN) * TVN + d * TVD, valid & ~visual, other=0)
    tl.store(OUT + tl.program_id(1) * COUNT + i, tl.where(visual, v, t), valid)


def split_attention_heads(query, visual_key, visual_value, text_key, text_value, *, scaling, plan):
    batch, heads, length, dim = query.shape
    if batch != 1 or torch.is_grad_enabled():
        raise ValueError('Split KV packing requires batch-one inference')
    kv_heads = text_key.shape[1]
    size = plan['key_indices'].numel()
    packed = torch.empty((2, size, kv_heads, dim), device=query.device, dtype=query.dtype)
    count = size * kv_heads * dim
    _pack_split[(triton.cdiv(count, 256), 2)](visual_key, visual_value, text_key, text_value,
        plan['key_indices'], packed, kv_heads, dim, visual_key.shape[2], count,
        *visual_key.stride()[1:], *visual_value.stride()[1:],
        *text_key.stride()[1:], *text_value.stride()[1:], 256)
    q = query.transpose(1, 2).reshape(-1, heads, dim)
    from src.qwen_adapter_fa2 import varlen_attention
    result = varlen_attention(q, packed[0], packed[1], scaling=scaling, plan=plan)
    return result.reshape(batch, length, heads, dim).contiguous()


@triton.jit
def _pack_native(VK, VV, TK, TV, OUT, D: tl.constexpr, N: tl.constexpr,
                 VLEN: tl.constexpr, COUNT: tl.constexpr, OUT_KV: tl.constexpr,
                 VKH: tl.constexpr, VKN: tl.constexpr, VKD: tl.constexpr,
                 VVH: tl.constexpr, VVN: tl.constexpr, VVD: tl.constexpr,
                 TKH: tl.constexpr, TKN: tl.constexpr, TKD: tl.constexpr,
                 TVH: tl.constexpr, TVN: tl.constexpr, TVD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    d, n, h = i % D, i // D % N, i // (D * N)
    visual = n < VLEN
    if tl.program_id(1) == 0:
        v = tl.load(VK + h * VKH + n * VKN + d * VKD, valid & visual, other=0)
        t = tl.load(TK + h * TKH + (n - VLEN) * TKN + d * TKD, valid & ~visual, other=0)
    else:
        v = tl.load(VV + h * VVH + n * VVN + d * VVD, valid & visual, other=0)
        t = tl.load(TV + h * TVH + (n - VLEN) * TVN + d * TVD, valid & ~visual, other=0)
    tl.store(OUT + tl.program_id(1) * OUT_KV + i, tl.where(visual, v, t), valid)


def pack_native_layer(visual_key, visual_value, text_key, text_value, output):
    """Write split, potentially strided KV directly to [2, heads, V+T, dim]."""
    if visual_key.shape[0] != 1 or torch.is_grad_enabled():
        raise ValueError('Native KV packing requires batch-one inference')
    _, heads, length, dim = output.shape
    assert length == visual_key.shape[2] + text_key.shape[2]
    assert output.stride()[1:] == (length * dim, dim, 1)
    count = heads * length * dim
    _pack_native[(triton.cdiv(count, 256), 2)](visual_key, visual_value, text_key, text_value,
        output, dim, length, visual_key.shape[2], count, output.stride(0),
        *visual_key.stride()[1:], *visual_value.stride()[1:],
        *text_key.stride()[1:], *text_value.stride()[1:], 256)
