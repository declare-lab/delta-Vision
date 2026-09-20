"""Single-kernel BF16 RMSNorm with the installed PyTorch mean summation order.

ATen Reduce.cuh vectorizes contiguous FP32 means by four: four sequential
accumulators per thread, a left fold across those four, then a descending binary
tree across threads. Ordinary sum(x*x) changes that ordering.
"""
import torch
import triton
import triton.language as tl


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
