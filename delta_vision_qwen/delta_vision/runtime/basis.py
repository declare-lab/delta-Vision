from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from delta_vision.kernels import reconstruct_delta_frozen_basis


def load_layer_basis(path: str | Path, rank: int, num_layers: int, hidden_size: int) -> Tensor:
    loaded = torch.load(path, map_location="cpu")
    raw_basis = loaded["basis"]
    layers = []
    for layer_idx in range(num_layers):
        basis = raw_basis[layer_idx].float()
        if rank > basis.shape[0]:
            raise ValueError(f"rank {rank} exceeds stored basis rank {basis.shape[0]} for layer {layer_idx}")
        basis = basis[:rank]
        if basis.shape != (rank, hidden_size):
            raise ValueError(f"basis for layer {layer_idx} has shape {tuple(basis.shape)}")
        layers.append(basis)
    return torch.stack(layers, dim=0)


def project_delta_to_coefficients(delta: Tensor, basis: Tensor) -> Tensor:
    """Project residuals onto an orthonormal basis.

    Args:
        delta: [batch, text_len, hidden]
        basis: [batch, rank, hidden]
    """
    basis = basis.to(device=delta.device, dtype=delta.dtype)
    return torch.matmul(delta, basis.transpose(1, 2))


def reconstruct_delta(coefficients: Tensor, basis: Tensor) -> Tensor:
    """Reconstruct residuals from basis coefficients."""
    basis = basis.to(device=coefficients.device, dtype=coefficients.dtype)
    if basis.requires_grad:
        return torch.matmul(coefficients, basis)
    return reconstruct_delta_frozen_basis(coefficients, basis)
