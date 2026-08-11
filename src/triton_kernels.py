"""Triton fused kernels for vision KV adapter."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_adapter_kv_kernel(
    # Source inputs (mixed from source layers)
    mixed_ptr,         # [B, N_vis, source_dim]
    # Weight pointers
    weight_ptr,        # [target_dim, source_dim] - k or v projection weight
    bias_ptr,          # [target_dim] - k or v projection bias
    # Output
    out_ptr,           # [B, N_vis, target_dim]
    # Gate
    gate_val,          # scalar sigmoid(gate)
    # Dims
    B: tl.constexpr,
    N_vis: tl.constexpr,
    source_dim: tl.constexpr,
    target_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: linear projection + gate scaling for one adapter layer."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Offset into mixed: [pid_b, pid_n * BLOCK_N : (pid_n+1) * BLOCK_N, :]
    n_start = pid_n * BLOCK_N
    d_start = pid_d * BLOCK_D

    n_offsets = n_start + tl.arange(0, BLOCK_N)
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    # Accumulate matmul: out[b, n, d] = sum_k(mixed[b, n, k] * weight[d, k]) + bias[d]
    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, source_dim, BLOCK_D):
        k_offsets = k_start + tl.arange(0, BLOCK_D)

        # Load mixed[b, n, k]
        mixed_offsets = pid_b * N_vis * source_dim + n_offsets[:, None] * source_dim + k_offsets[None, :]
        mixed_mask = (n_offsets[:, None] < N_vis) & (k_offsets[None, :] < source_dim)
        mixed_vals = tl.load(mixed_ptr + mixed_offsets, mask=mixed_mask, other=0.0)

        # Load weight[d, k]
        weight_offsets = d_offsets[:, None] * source_dim + k_offsets[None, :]
        weight_mask = (d_offsets[:, None] < target_dim) & (k_offsets[None, :] < source_dim)
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        # Accumulate
        acc += tl.dot(mixed_vals.to(tl.float32), tl.trans(weight_vals.to(tl.float32)))

    # Add bias
    bias_vals = tl.load(bias_ptr + d_offsets, mask=d_offsets < target_dim, other=0.0)
    acc += bias_vals[None, :]

    # Apply gate
    acc = acc * gate_val

    # Store
    out_offsets = pid_b * N_vis * target_dim + n_offsets[:, None] * target_dim + d_offsets[None, :]
    out_mask = (n_offsets[:, None] < N_vis) & (d_offsets[None, :] < target_dim)
    tl.store(out_ptr + out_offsets, acc.to(tl.bfloat16), mask=out_mask)


def fused_adapter_layer_forward(
    mixed_k: torch.Tensor,   # [B, N_vis, source_dim]
    mixed_v: torch.Tensor,
    k_weight: torch.Tensor,  # [target_dim, source_dim]
    k_bias: torch.Tensor,    # [target_dim]
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    gate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused adapter forward for one layer using Triton."""
    B, N_vis, source_dim = mixed_k.shape
    target_dim = k_weight.shape[0]

    out_k = torch.empty(B, N_vis, target_dim, device=mixed_k.device, dtype=torch.bfloat16)
    out_v = torch.empty(B, N_vis, target_dim, device=mixed_v.device, dtype=torch.bfloat16)

    BLOCK_N = min(64, N_vis)
    BLOCK_D = min(128, target_dim)

    grid = (B, triton.cdiv(N_vis, BLOCK_N), triton.cdiv(target_dim, BLOCK_D))

    _fused_adapter_kv_kernel[grid](
        mixed_k, k_weight, k_bias, out_k, gate,
        B, N_vis, source_dim, target_dim, BLOCK_N, BLOCK_D,
    )
    _fused_adapter_kv_kernel[grid](
        mixed_v, v_weight, v_bias, out_v, gate,
        B, N_vis, source_dim, target_dim, BLOCK_N, BLOCK_D,
    )

    return out_k, out_v


@triton.jit
def _fused_source_mix_kernel(
    # source: [B, num_source, N_vis, source_dim]
    source_ptr,
    # weights: [num_source] (after softmax)
    weights_ptr,
    # output: [B, N_vis, source_dim]
    out_ptr,
    B: tl.constexpr,
    num_source: tl.constexpr,
    N_vis: tl.constexpr,
    source_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused weighted sum over source layers."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    d_start = pid_d * BLOCK_D
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for s in range(num_source):
        w = tl.load(weights_ptr + s)
        src_offsets = pid_b * num_source * N_vis * source_dim + s * N_vis * source_dim + n_offsets[:, None] * source_dim + d_offsets[None, :]
        mask = (n_offsets[:, None] < N_vis) & (d_offsets[None, :] < source_dim)
        vals = tl.load(source_ptr + src_offsets, mask=mask, other=0.0)
        acc += w.to(tl.float32) * vals.to(tl.float32)

    out_offsets = pid_b * N_vis * source_dim + n_offsets[:, None] * source_dim + d_offsets[None, :]
    out_mask = (n_offsets[:, None] < N_vis) & (d_offsets[None, :] < source_dim)
    tl.store(out_ptr + out_offsets, acc.to(tl.bfloat16), mask=out_mask)


def fused_source_mix(
    source: torch.Tensor,  # [B, num_source, N_vis, source_dim]
    weights: torch.Tensor,  # [num_source] softmaxed
) -> torch.Tensor:
    """Fused weighted sum over source layers."""
    B, num_source, N_vis, source_dim = source.shape
    out = torch.empty(B, N_vis, source_dim, device=source.device, dtype=torch.bfloat16)

    BLOCK_N = min(64, N_vis)
    BLOCK_D = min(128, source_dim)
    grid = (B, triton.cdiv(N_vis, BLOCK_N), triton.cdiv(source_dim, BLOCK_D))

    _fused_source_mix_kernel[grid](
        source, weights, out,
        B, num_source, N_vis, source_dim, BLOCK_N, BLOCK_D,
    )
    return out


def fused_adapter_forward_all_layers(
    adapter,
    source_k: torch.Tensor,  # [B, num_source, N_vis, source_dim]
    source_v: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run all adapter layers with Triton fused kernels.

    Returns per-layer K and V: list of [B, N_vis, num_heads, head_dim]
    """
    import torch.nn.functional as F

    all_k = []
    all_v = []

    for layer_idx in range(adapter.num_llm_layers):
        # Source mixing
        weights = F.softmax(adapter.source_mix[layer_idx].float(), dim=-1).to(torch.bfloat16)
        mixed_k = fused_source_mix(source_k, weights)
        mixed_v = fused_source_mix(source_v, weights)

        # Gate
        gate = torch.sigmoid(adapter.gates[layer_idx]).item()

        # Projection
        if adapter.k_projs is not None:
            k_w = adapter.k_projs[layer_idx].weight  # [target_dim, source_dim]
            k_b = adapter.k_projs[layer_idx].bias
            v_w = adapter.v_projs[layer_idx].weight
            v_b = adapter.v_projs[layer_idx].bias
        else:
            # bottleneck not fused yet, fall back
            vis_k, vis_v = adapter.forward_layer(source_k.to(torch.bfloat16), source_v.to(torch.bfloat16), layer_idx)
            all_k.append(vis_k)
            all_v.append(vis_v)
            continue

        out_k, out_v = fused_adapter_layer_forward(mixed_k, mixed_v, k_w, k_b, v_w, v_b, gate)

        B, N_vis, target_dim = out_k.shape
        out_k = out_k.view(B, N_vis, adapter.num_heads, adapter.head_dim)
        out_v = out_v.view(B, N_vis, adapter.num_heads, adapter.head_dim)
        all_k.append(out_k)
        all_v.append(out_v)

    return all_k, all_v
