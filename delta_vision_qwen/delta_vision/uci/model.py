from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype=x.dtype)


class UniversalContextEncoder(nn.Module):
    """Compile context tokens into a fixed set of shared latent slots U(c)."""

    def __init__(
        self,
        input_dim: int = 1024,
        latent_slots: int = 64,
        latent_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_slots = latent_slots
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.latents = nn.Parameter(torch.empty(latent_slots, latent_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.cross_norm = RMSNorm(latent_dim)
        self.blocks = nn.ModuleList()
        hidden = int(latent_dim * mlp_ratio)
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn_norm": RMSNorm(latent_dim),
                        "attn": nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                        "mlp_norm": RMSNorm(latent_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(latent_dim, hidden, bias=False),
                            nn.SiLU(),
                            nn.Linear(hidden, latent_dim, bias=False),
                        ),
                    }
                )
            )

    def forward(self, context_tokens: Tensor, context_mask: Tensor | None = None) -> Tensor:
        batch = context_tokens.shape[0]
        x = self.input_proj(context_tokens)
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        q = self.cross_norm(latents)
        update, _ = self.cross_attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + update
        for block in self.blocks:
            q = block["attn_norm"](latents)
            update, _ = block["attn"](q, q, q, need_weights=False)
            latents = latents + update
            latents = latents + block["mlp"](block["mlp_norm"](latents))
        return latents


class ModelBinder(nn.Module):
    """Small model-specific adapter B_m/G_m binding U(c) to one frozen LLM."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        latent_slots: int = 64,
        latent_dim: int = 512,
        layer_ids: list[int] | None = None,
        layer_mix_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.latent_slots = latent_slots
        self.layer_ids = layer_ids if layer_ids is not None else list(range(num_layers))
        self.layer_to_index = {int(layer): idx for idx, layer in enumerate(self.layer_ids)}
        self.align = nn.Linear(latent_dim, model_dim, bias=False)
        mix = torch.eye(latent_slots).unsqueeze(0).repeat(len(self.layer_ids), 1, 1)
        self.layer_mix = nn.Parameter(mix + 0.01 * torch.randn_like(mix))
        self.hidden_norm = RMSNorm(model_dim)
        self.gate = nn.Linear(model_dim, latent_slots, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01 / math.sqrt(model_dim))
        self.output_scale = nn.Parameter(torch.full((len(self.layer_ids),), 0.1))
        self.layer_mix_rank = layer_mix_rank

    def layer_bank(self, u: Tensor, layer_idx: int) -> Tensor:
        if int(layer_idx) not in self.layer_to_index:
            raise KeyError(f"layer {layer_idx} is not configured for this binder")
        idx = self.layer_to_index[int(layer_idx)]
        aligned = self.align(u)
        return torch.einsum("rs,bsd->brd", self.layer_mix[idx].to(dtype=aligned.dtype), aligned)

    def forward(self, u: Tensor, hidden: Tensor, layer_idx: int) -> Tensor:
        idx = self.layer_to_index[int(layer_idx)]
        bank = self.layer_bank(u, layer_idx)
        gate = self.gate(self.hidden_norm(hidden))
        delta = torch.einsum("br,brd->bd", gate.to(dtype=bank.dtype), bank)
        return delta * self.output_scale[idx].to(dtype=delta.dtype)


class UniversalContextInterface(nn.Module):
    def __init__(self, encoder: UniversalContextEncoder, binders: dict[str, ModelBinder]) -> None:
        super().__init__()
        self.encoder = encoder
        self.binders = nn.ModuleDict(binders)

    def forward(self, teacher: str, context_tokens: Tensor, hidden: Tensor, layer_idx: int) -> Tensor:
        u = self.encoder(context_tokens)
        return self.binders[teacher](u, hidden, layer_idx)


def masked_effect_losses(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    pred_f = pred.float()
    target_f = target.float()
    nmse = (pred_f - target_f).pow(2).mean() / target_f.pow(2).mean().clamp_min(1e-4)
    cos = F.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-6).mean()
    mse = (pred_f - target_f).pow(2).mean()
    return {"nmse": nmse, "cos": cos, "mse": mse}


class UniversalContextKVEncoder(nn.Module):
    """Compile context tokens into one model-independent external KV cache."""

    def __init__(
        self,
        input_dim: int = 1024,
        cache_slots: int = 64,
        cache_dim: int = 256,
        latent_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.cache_slots = cache_slots
        self.cache_dim = cache_dim
        self.slot_encoder = UniversalContextEncoder(
            input_dim=input_dim,
            latent_slots=cache_slots,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.k_proj = nn.Linear(latent_dim, cache_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, cache_dim, bias=False)

    def forward(self, context_tokens: Tensor, context_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        u = self.slot_encoder(context_tokens, context_mask)
        k = F.normalize(self.k_proj(u).float(), p=2, dim=-1, eps=1e-6).to(dtype=u.dtype)
        v = self.v_proj(u)
        return k, v


class UniversalKVBinder(nn.Module):
    """Model-specific binder reading the same K^u/V^u cache."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        cache_dim: int = 256,
        layer_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.cache_dim = cache_dim
        self.layer_ids = layer_ids if layer_ids is not None else list(range(num_layers))
        self.layer_to_index = {int(layer): idx for idx, layer in enumerate(self.layer_ids)}
        self.hidden_norm = RMSNorm(model_dim)
        self.query = nn.ModuleList([nn.Linear(model_dim, cache_dim, bias=False) for _ in self.layer_ids])
        self.output = nn.ModuleList([nn.Linear(cache_dim, model_dim, bias=False) for _ in self.layer_ids])
        self.gate = nn.Parameter(torch.full((len(self.layer_ids),), 0.1))
        for module in self.query:
            nn.init.normal_(module.weight, std=0.01 / math.sqrt(model_dim))
        for module in self.output:
            nn.init.normal_(module.weight, std=0.01 / math.sqrt(cache_dim))

    def forward(self, kv_cache: tuple[Tensor, Tensor], hidden: Tensor, layer_idx: int) -> Tensor:
        if int(layer_idx) not in self.layer_to_index:
            raise KeyError(f"layer {layer_idx} is not configured for this binder")
        idx = self.layer_to_index[int(layer_idx)]
        k_u, v_u = kv_cache
        q = self.query[idx](self.hidden_norm(hidden))
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(dtype=k_u.dtype)
        scores = torch.einsum("bd,bnd->bn", q, k_u) / math.sqrt(float(self.cache_dim))
        probs = F.softmax(scores.float(), dim=-1).to(dtype=v_u.dtype)
        readout = torch.einsum("bn,bnd->bd", probs, v_u)
        return self.output[idx](readout).to(dtype=hidden.dtype) * self.gate[idx].to(dtype=hidden.dtype)
