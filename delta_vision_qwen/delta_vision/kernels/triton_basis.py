from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import depends on CUDA runtime
    triton = None
    tl = None


def _triton_enabled() -> bool:
    return triton is not None and os.environ.get("VISUAL_SIDECAR_USE_TRITON_BASIS", "0") == "1"


if triton is not None:

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        acc = tl.zeros((block_m, block_n), tl.float32)
        for k_start in range(0, k, block_k):
            k_idxs = k_start + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a, b)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


def _triton_matmul_2d(a: Tensor, b: Tensor) -> Tensor:
    if not _triton_enabled() or not a.is_cuda or not b.is_cuda:
        return a @ b
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected [M, K] @ [K, N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype not in (torch.float16, torch.bfloat16):
        return a @ b

    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m = 16 if m <= 64 else 32
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m,
        block_n,
        block_k,
        num_warps=4,
    )
    return c


class _FrozenBasisReconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients: Tensor, basis: Tensor) -> Tensor:
        if coefficients.ndim != 3 or basis.ndim != 3:
            raise ValueError("expected coefficients [B, T, R] and basis [B, R, H]")
        if coefficients.shape[0] != basis.shape[0] or coefficients.shape[2] != basis.shape[1]:
            raise ValueError("coefficient and basis shapes are incompatible")
        if basis.shape[0] != 1:
            out = torch.matmul(coefficients, basis)
            ctx.save_for_backward(basis)
            ctx.used_triton = False
            return out

        batch, text_len, rank = coefficients.shape
        hidden = basis.shape[2]
        coeff_2d = coefficients.reshape(batch * text_len, rank)
        basis_2d = basis[0]
        out = _triton_matmul_2d(coeff_2d, basis_2d).reshape(batch, text_len, hidden)
        ctx.save_for_backward(basis)
        ctx.used_triton = True
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None]:
        (basis,) = ctx.saved_tensors
        if basis.shape[0] != 1 or not getattr(ctx, "used_triton", False):
            grad_coeff = torch.matmul(grad_output, basis.transpose(1, 2))
            return grad_coeff, None

        batch, text_len, hidden = grad_output.shape
        rank = basis.shape[1]
        grad_2d = grad_output.reshape(batch * text_len, hidden)
        basis_t = basis[0].transpose(0, 1)
        grad_coeff = _triton_matmul_2d(grad_2d, basis_t).reshape(batch, text_len, rank)
        return grad_coeff, None


def reconstruct_delta_frozen_basis(coefficients: Tensor, basis: Tensor) -> Tensor:
    """Reconstruct delta with a frozen basis, using Triton for batch-1 fast path."""
    if not _triton_enabled() or basis.requires_grad:
        return torch.matmul(coefficients, basis)
    return _FrozenBasisReconstruct.apply(coefficients, basis)
