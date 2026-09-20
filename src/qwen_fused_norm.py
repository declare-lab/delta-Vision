"""Inference-only Qwen RMSNorm fusion, retaining the intermediate BF16 rounding."""
import torch
import triton
import triton.language as tl


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
                    from src.qwen_native_order_norm import native_order_rmsnorm
                    return native_order_rmsnorm(x, _module.weight, _module.variance_epsilon)
                return fused_qwen_rmsnorm(x, _module.weight, _module.variance_epsilon)
            module.forward = forward

    def remove(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
