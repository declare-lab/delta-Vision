"""Qwen3-VL inference kernels: KV packing, RoPE, RMSNorm and projection fusion."""


# Exact inference-only copies and BF16-rounded RoPE for the FA2 adapter.
import torch
import triton
import triton.language as tl


@triton.jit
def _vllm_qk_norm_rope(Q, K, QW, KW, POS, CACHE, QIDX, OQ, OK,
                       NQ: tl.constexpr, NK: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                       D: tl.constexpr, QS: tl.constexpr, KS: tl.constexpr,
                       PS: tl.constexpr, CS: tl.constexpr, EPSQ: tl.constexpr, EPSK: tl.constexpr,
                       INDEXED: tl.constexpr, MULTI: tl.constexpr, INTERLEAVED: tl.constexpr,
                       MT: tl.constexpr, MH: tl.constexpr, MW: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    if row < NQ * HQ:
        token, head = row // HQ, row % HQ
        pos_token = tl.load(QIDX + token).to(tl.int32) if INDEXED else token
        x = tl.load(Q + token * QS + head * D + d).to(tl.float32)
        w = tl.load(QW + d).to(tl.float32)
        eps = EPSQ
    else:
        kr = row - NQ * HQ
        token, head = kr // HK, kr % HK
        pos_token = token
        x = tl.load(K + token * KS + head * D + d).to(tl.float32)
        w = tl.load(KW + d).to(tl.float32)
        eps = EPSK
    scale = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    # Preserve the native Qwen/vLLM BF16 normalization boundaries.
    normalized = (x * scale).to(Q.dtype.element_ty).to(tl.float32)
    normalized = (normalized * w).to(Q.dtype.element_ty).to(tl.float32)
    freq = d % (D // 2)
    axis = tl.full((D,), 0, tl.int32)
    if MULTI:
        if INTERLEAVED:
            axis = tl.where((freq % 3 == 1) & (freq < 3 * MH), 1, axis)
            axis = tl.where((freq % 3 == 2) & (freq < 3 * MW), 2, axis)
        else:
            axis = tl.where(freq < MT, 0, tl.where(freq < MT + MH, 1, 2))
    position = tl.load(POS + axis * PS + pos_token)
    cos = tl.load(CACHE + position * CS + freq).to(tl.float32)
    sin = tl.load(CACHE + position * CS + D // 2 + freq).to(tl.float32)
    rotated = tl.gather(normalized, (d + D // 2) % D, 0)
    rotated = tl.where(d < D // 2, -rotated, rotated)
    if INDEXED:
        # Compact adapter prefill uses FlashAttention's FP32 rotary arithmetic.
        result = tl.where(d < D // 2, tl.fma(normalized, cos, rotated * sin),
                          tl.fma(rotated, sin, normalized * cos))
    else:
        # Native MRoPE contracts one product into the add, retaining the other
        # product's BF16 boundary (different operands in the two rotary halves).
        first = (normalized * cos).to(Q.dtype.element_ty).to(tl.float32)
        second = (rotated * sin).to(Q.dtype.element_ty).to(tl.float32)
        result = tl.where(d < D // 2, tl.fma(normalized, cos, second),
                          tl.fma(rotated, sin, first))
    if row < NQ * HQ:
        tl.store(OQ + row * D + d, result)
    else:
        tl.store(OK + (row - NQ * HQ) * D + d, result)


def vllm_qk_norm_rope(attn, query, key, positions, query_indices=None):
    """One kernel for both Q/K head norms and MRoPE; native KV ownership stays intact."""
    q = query.view(-1, attn.num_heads, attn.head_dim)
    k = key.view(-1, attn.num_kv_heads, attn.head_dim)
    rope = attn.rotary_emb
    assert q.dtype == k.dtype == torch.bfloat16 and attn.head_dim == rope.rotary_dim == 128
    assert q.stride(-1) == k.stride(-1) == 1 and q.stride(1) == k.stride(1) == 128
    assert positions.stride(-1) == 1
    oq, ok = torch.empty(q.shape, device=q.device, dtype=q.dtype), torch.empty(k.shape, device=k.device, dtype=k.dtype)
    cache = rope._match_cos_sin_cache_dtype(q)
    sections = rope.mrope_section or (64, 0, 0)
    _vllm_qk_norm_rope[(q.shape[0] * attn.num_heads + k.shape[0] * attn.num_kv_heads,)](
        q, k, attn.q_norm.weight, attn.k_norm.weight, positions, cache,
        positions if query_indices is None else query_indices, oq, ok,
        q.shape[0], k.shape[0], attn.num_heads, attn.num_kv_heads, 128, q.stride(0), k.stride(0),
        positions.stride(0) if positions.ndim == 2 else 0, cache.stride(0),
        attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
        query_indices is not None, positions.ndim == 2, rope.mrope_interleaved, *sections,
        num_warps=4, enable_fp_fusion=False)
    return oq, ok


@triton.jit
def _layerwise_rmsnorm(X, W, Y, N: tl.constexpr, H: tl.constexpr,
                       EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + d, d < H, other=0).to(tl.float32)
    w = tl.load(W + (row // N) * H + d, d < H, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    normalized = (x * scale).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + d, normalized * w, d < H)


def layerwise_rmsnorm(memories, weights, eps):
    layers, tokens, hidden = memories.shape
    assert memories.is_contiguous() and weights.is_contiguous()
    output = torch.empty_like(memories)
    _layerwise_rmsnorm[(layers * tokens,)](memories, weights, output, tokens, hidden,
                                          eps, triton.next_power_of_2(hidden), enable_fp_fusion=False)
    return output


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
    from src.attention import varlen_attention
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


# Single-kernel BF16 RMSNorm with the installed PyTorch mean summation order.
@triton.jit
def _native_order_norm(X, W, Y, N: tl.constexpr, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                       GROUP: tl.constexpr, OUT_BLOCK: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    lane = tl.arange(0, WIDTH)
    a = tl.full((GROUP, WIDTH), 0, tl.float32)
    b = tl.full((GROUP, WIDTH), 0, tl.float32)
    c = tl.full((GROUP, WIDTH), 0, tl.float32)
    d = tl.full((GROUP, WIDTH), 0, tl.float32)
    for iteration in range(triton.cdiv(N, WIDTH * 4)):
        col = lane * 4 + iteration * WIDTH * 4
        valid = (row[:, None] < ROWS) & (col[None, :] < N)
        x0 = tl.load(X + row[:, None] * N + col[None, :], valid, other=0).to(tl.float32)
        x1 = tl.load(X + row[:, None] * N + col[None, :] + 1, valid, other=0).to(tl.float32)
        x2 = tl.load(X + row[:, None] * N + col[None, :] + 2, valid, other=0).to(tl.float32)
        x3 = tl.load(X + row[:, None] * N + col[None, :] + 3, valid, other=0).to(tl.float32)
        a = a + x0 * x0
        b = b + x1 * x1
        c = c + x2 * x2
        d = d + x3 * x3
    value = ((a + b) + c) + d
    for shift in tl.static_range(0, 9):
        offset = WIDTH // (2 ** (shift + 1))
        if offset >= 1:
            indices = tl.broadcast_to(((lane + offset) % WIDTH)[None, :], (GROUP, WIDTH))
            value = value + tl.gather(value, indices, axis=1)
    # Do not use tl.sum for the arithmetic tree: layout-dependent per-thread
    # aggregation can choose a different order even on a 32-element reduction.
    variance = tl.sum(tl.where(lane[None, :] == 0, value, 0), 1) * (1.0 / N)
    scale = tl.rsqrt(variance + EPS)
    col = tl.arange(0, OUT_BLOCK)
    valid = (row[:, None] < ROWS) & (col[None, :] < N)
    x = tl.load(X + row[:, None] * N + col[None, :], valid, other=0).to(tl.float32)
    weight = tl.load(W + col, col < N, other=0).to(tl.float32)
    normalized = (x * scale[:, None]).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row[:, None] * N + col[None, :], normalized * weight[None, :], valid)


def native_order_rmsnorm(x, weight, eps):
    n, rows = x.shape[-1], x.numel() // x.shape[-1]
    if (n not in (128, 2560) or not x.is_contiguous() or not x.is_cuda
        or x.dtype != torch.bfloat16 or weight.dtype != x.dtype or torch.is_grad_enabled()):
        raise ValueError('Native-order RMSNorm supports contiguous CUDA BF16 Qwen inference')
    row_pow2 = 1 << (rows.bit_length()-1)
    dim0_pow2 = min(512, 1 << ((n//4).bit_length()-1))
    height = min(row_pow2, 16)
    width = min(dim0_pow2, 512//height)
    group = 1 if rows < 4 else 4
    output = torch.empty_like(x)
    _native_order_norm[(triton.cdiv(rows, group),)](x, weight, output, n, rows, width, group,
        triton.next_power_of_2(n), eps, num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _norm_rope(X, W, C, S, Y, HEADS: tl.constexpr, LENGTH: tl.constexpr, ROWS: tl.constexpr,
               X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
               Y0: tl.constexpr, Y1: tl.constexpr, Y2: tl.constexpr,
               C0: tl.constexpr, C1: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
               EPS: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    h, n, b = row % HEADS, row // HEADS % LENGTH, row // (HEADS * LENGTH)
    base = b * X0 + h * X1 + n * X2
    lane = tl.arange(0, 32)
    valid = row[:, None] < ROWS
    x0 = tl.load(X + base[:, None] + lane[None, :] * 4, valid, other=0).to(tl.float32)
    x1 = tl.load(X + base[:, None] + lane[None, :] * 4 + 1, valid, other=0).to(tl.float32)
    x2 = tl.load(X + base[:, None] + lane[None, :] * 4 + 2, valid, other=0).to(tl.float32)
    x3 = tl.load(X + base[:, None] + lane[None, :] * 4 + 3, valid, other=0).to(tl.float32)
    value = ((x0*x0 + x1*x1) + x2*x2) + x3*x3
    for shift in tl.static_range(0, 5):
        offset = 16 // (2 ** shift)
        indices = tl.broadcast_to(((lane+offset)%32)[None, :], (GROUP, 32))
        value = value + tl.gather(value, indices, axis=1)
    variance = tl.sum(tl.where(lane[None, :] == 0, value, 0), 1) * (1.0/128)
    scale = tl.rsqrt(variance + EPS)
    col = tl.arange(0, 128)
    x = tl.load(X + base[:, None] + col[None, :], valid, other=0).to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    normalized = (x * scale[:, None]).to(Y.dtype.element_ty).to(tl.float32)
    weighted = (normalized * weight[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    rotated = tl.gather(weighted, tl.broadcast_to(((col+64)%128)[None, :], (GROUP, 128)), axis=1)
    rotated = tl.where(col[None, :] < 64, -rotated, rotated)
    cos = tl.load(C + b[:, None] * C0 + n[:, None] * C1 + col[None, :], valid, other=0).to(tl.float32)
    sin = tl.load(S + b[:, None] * S0 + n[:, None] * S1 + col[None, :], valid, other=0).to(tl.float32)
    first = (weighted * cos).to(Y.dtype.element_ty).to(tl.float32)
    second = (rotated * sin).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + b[:, None] * Y0 + h[:, None] * Y1 + n[:, None] * Y2 + col[None, :], first+second, valid)


def native_order_norm_rope(states, weight, eps, embeddings):
    cos, sin = embeddings
    if (states.shape[-1] != 128 or states.stride(-1) != 1 or cos.stride(-1) != 1 or sin.stride(-1) != 1
        or states.dtype != torch.bfloat16 or cos.dtype != states.dtype or sin.dtype != states.dtype
        or weight.dtype != states.dtype or not states.is_cuda or torch.is_grad_enabled()):
        raise ValueError('Fused norm/RoPE requires CUDA BF16 inference with head dimension 128')
    output = torch.empty_like(states)
    batch, heads, length, dim = states.shape
    rows = batch * heads * length
    _norm_rope[(triton.cdiv(rows, 4),)](states, weight, cos, sin, output, heads, length, rows,
        *states.stride()[:3], *output.stride()[:3], *cos.stride()[:2], *sin.stride()[:2], eps, 4,
        num_warps=4, enable_fp_fusion=False)
    return output


# Inference-only Qwen RMSNorm fusion, retaining the intermediate BF16 rounding.
@triton.jit
def _qwen_rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    variance = tl.sum(x * x, 0) / N
    normalized = x * tl.rsqrt(variance + EPS)
    # Qwen casts the normalized vector BEFORE multiplying the norm weight.
    rounded = normalized.to(Y.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + col, col < N, other=0).to(tl.float32)
    tl.store(Y + row * N + col, rounded * weight, col < N)


@triton.jit
def _squared_fp32(X, S, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < COUNT, other=0).to(tl.float32)
    tl.store(S + offsets, x * x, offsets < COUNT)


@triton.jit
def _normalize_from_variance(X, V, W, Y, N: tl.constexpr, COUNT: tl.constexpr,
                             EPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < COUNT
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    variance = tl.load(V + offsets // N, valid, other=0)
    normalized = (x * tl.rsqrt(variance + EPS)).to(Y.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + offsets % N, valid, other=0).to(tl.float32)
    tl.store(Y + offsets, normalized * weight, valid)


def fused_qwen_rmsnorm(x, weight, eps, *, exact_reduction=True):
    n = x.shape[-1]
    output = torch.empty_like(x)
    if exact_reduction:
        # Reuse the native mean reduction order. A different reduction tree can
        # cross BF16 rounding boundaries and compound through 36 decoder layers.
        squared = torch.empty_like(x, dtype=torch.float32)
        grid = (triton.cdiv(x.numel(), 256),)
        _squared_fp32[grid](x, squared, x.numel(), 256, enable_fp_fusion=False)
        variance = squared.mean(-1, keepdim=True)
        _normalize_from_variance[grid](x, variance, weight, output, n, x.numel(), eps, 256,
            enable_fp_fusion=False)
    else:
        _qwen_rmsnorm[(x.numel() // n,)](x, weight, output, n, eps,
            triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return output


class FusedQwenNorms:
    """Opt-in benchmark intervention; does not modify any other model instance."""
    def __init__(self, model):
        self.enabled = True
        self.native_order = False
        self.originals = []
        for module in model.model.language_model.modules():
            if 'RMSNorm' not in type(module).__name__ or not hasattr(module, 'variance_epsilon'):
                continue
            original = module.forward
            self.originals.append((module, original))
            def forward(hidden_states, _module=module, _original=original):
                x = hidden_states
                if (not self.enabled or torch.is_grad_enabled() or not x.is_cuda
                    or x.dtype != torch.bfloat16 or _module.weight.dtype != x.dtype
                    or not x.is_contiguous() or x.shape[-1] > 8192):
                    return _original(x)
                if self.native_order and x.shape[-1] in (128, 2560):
                    from src.kernels import native_order_rmsnorm
                    return native_order_rmsnorm(x, _module.weight, _module.variance_epsilon)
                return fused_qwen_rmsnorm(x, _module.weight, _module.variance_epsilon)
            module.forward = forward

    def remove(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()


# Use exact BF16 RoPE kernels in one native Qwen model instance.
from types import FunctionType, MethodType


class QwenExactRoPE:
    def __init__(self, model):
        self.enabled = True
        self.fuse_norm = False
        self.originals = []
        self.norm_originals = []
        for layer in model.model.language_model.layers:
            attention = layer.self_attn
            original = attention.forward
            function = original.__func__
            original_rope = function.__globals__['apply_rotary_pos_emb']
            state = dict(active=False)
            def rope(query, key, cos, sin, unsqueeze_dim=1, _original=original_rope, _state=state, _attention=attention):
                if _state['active']:
                    from src.kernels import native_order_norm_rope
                    return (native_order_norm_rope(query, _attention.q_norm.weight, _attention.q_norm.variance_epsilon, (cos,sin)),
                            native_order_norm_rope(key, _attention.k_norm.weight, _attention.k_norm.variance_epsilon, (cos,sin)))
                if (self.enabled and not torch.is_grad_enabled() and query.is_cuda
                    and query.dtype == key.dtype == cos.dtype == sin.dtype == torch.bfloat16
                    and unsqueeze_dim == 1):
                    return exact_rope(query, (cos, sin)), exact_rope(key, (cos, sin))
                return _original(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
            # Bind the unchanged native attention forward with a private globals
            # dictionary so other models and adapter/reference paths are untouched.
            local = FunctionType(function.__code__, dict(function.__globals__, apply_rotary_pos_emb=rope),
                function.__name__, function.__defaults__, function.__closure__)
            local.__kwdefaults__ = function.__kwdefaults__
            self.originals.append((attention, original))
            bound = MethodType(local, attention)
            for norm in (attention.q_norm, attention.k_norm):
                norm_forward = norm.forward
                self.norm_originals.append((norm, norm_forward))
                def maybe_defer(x, _state=state, _original=norm_forward):
                    return x if _state['active'] else _original(x)
                norm.forward = maybe_defer
            def forward(*args, _bound=bound, _state=state, _attention=attention, **kwargs):
                x = args[0] if args else kwargs.get('hidden_states')
                previous = _state['active']
                _state['active'] = (self.enabled and self.fuse_norm and not torch.is_grad_enabled()
                    and x is not None and x.is_cuda and x.dtype == torch.bfloat16
                    and _attention.head_dim == 128
                    and not any(n._forward_hooks or n._forward_pre_hooks for n in (_attention.q_norm, _attention.k_norm)))
                try:
                    return _bound(*args, **kwargs)
                finally:
                    _state['active'] = previous
            attention.forward = forward

    def remove(self):
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()
        for norm, original in self.norm_originals:
            norm.forward = original
        self.norm_originals.clear()


# Combine native Q/K/V and gate/up GEMMs for single-token Qwen inference.
import torch.nn.functional as F


class QwenFusedProjections:
    def __init__(self, model):
        self.enabled = True
        self.groups = []
        self.originals = []
        for layer in model.model.language_model.layers:
            self._group([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            self._group([layer.mlp.gate_proj, layer.mlp.up_proj])

    def _group(self, modules):
        if any(m.bias is not None for m in modules):
            raise ValueError('Qwen projection fusion currently requires bias-free layers')
        sizes = [m.weight.shape[0] for m in modules]
        with torch.no_grad():
            weight = torch.cat([m.weight for m in modules], dim=0)
            for module, view in zip(modules, weight.split(sizes, dim=0)):
                module.weight.data = view
        state = dict(weight=weight, sizes=sizes, source=None, outputs=None)
        self.groups.append(state)
        for index, module in enumerate(modules):
            original = module.forward
            self.originals.append((module, original))
            def forward(x, _index=index, _original=original):
                if not self.enabled or torch.is_grad_enabled() or x.ndim != 3 or x.shape[:2] != (1, 1):
                    return _original(x)
                if _index == 0:
                    state['source'] = x
                    state['outputs'] = F.linear(x, state['weight']).split(state['sizes'], dim=-1)
                if state['source'] is not x or state['outputs'] is None:
                    return _original(x)
                output = state['outputs'][_index]
                if _index + 1 == len(modules):
                    state['source'] = state['outputs'] = None
                return output
            module.forward = forward

    def remove(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.groups.clear()
