from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerwiseVisionKVAdapter(nn.Module):
    """Map frozen vision-backbone states to per-LLM-layer visual K/V content.

    The adapter predicts pre-RoPE key content and values. RoPE is applied later
    by the target LLM attention using the original visual token positions.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        source_layers: list[int],
        source_dim: int,
        num_key_value_heads: int,
        head_dim: int,
        bottleneck_dim: int = 32,
        shared_down: bool = True,
        identity_down: bool = False,
    ) -> None:
        super().__init__()
        if not source_layers:
            raise ValueError("source_layers must be non-empty")
        self.num_layers = int(num_layers)
        self.source_layers = [int(x) for x in source_layers]
        self.source_dim = int(source_dim)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.shared_down = bool(shared_down)
        self.identity_down = bool(identity_down)

        self.source_mix = nn.Parameter(torch.zeros(self.num_layers, len(self.source_layers)))
        if self.identity_down:
            if self.bottleneck_dim != self.source_dim:
                raise ValueError("identity_down requires bottleneck_dim == source_dim")
            self.down = nn.Identity()
        elif self.shared_down:
            self.down = nn.Linear(self.source_dim, self.bottleneck_dim, bias=False)
        else:
            self.down = nn.ModuleList(
                [nn.Linear(self.source_dim, self.bottleneck_dim, bias=False) for _ in range(self.num_layers)]
            )
        self.up = nn.ModuleList(
            [
                nn.Linear(
                    self.bottleneck_dim,
                    2 * self.num_key_value_heads * self.head_dim,
                    bias=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_scale = nn.Parameter(torch.ones(self.num_layers))

        if self.identity_down:
            pass
        elif self.shared_down:
            nn.init.normal_(self.down.weight, std=0.02)
        else:
            for module in self.down:
                nn.init.normal_(module.weight, std=0.02)
        for module in self.up:
            nn.init.normal_(module.weight, std=0.01)

    def forward_layer(self, source_states: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return predicted visual K/V for one LLM layer.

        Args:
            source_states: [B, S, Nv, source_dim], one entry per source layer.
            layer_idx: target LLM layer index.

        Returns:
            key_content, value: both [B, Nv, H_kv, D_head].
        """
        if source_states.ndim != 4:
            raise ValueError("source_states must have shape [B, S, Nv, D]")
        idx = int(layer_idx)
        weights = F.softmax(self.source_mix[idx].float(), dim=-1).to(dtype=source_states.dtype)
        mixed = torch.einsum("s,bsnd->bnd", weights, source_states)
        if self.identity_down:
            hidden = mixed
        else:
            down = self.down if self.shared_down else self.down[idx]
            hidden = F.silu(down(mixed))
        out = self.up[idx](hidden) * self.output_scale[idx].to(dtype=hidden.dtype)
        batch, tokens, _ = out.shape
        out = out.view(batch, tokens, 2, self.num_key_value_heads, self.head_dim)
        return out[:, :, 0].contiguous(), out[:, :, 1].contiguous()

    def forward(self, source_states: Tensor) -> tuple[Tensor, Tensor]:
        keys = []
        values = []
        for layer_idx in range(self.num_layers):
            key, value = self.forward_layer(source_states, layer_idx)
            keys.append(key)
            values.append(value)
        return torch.stack(keys, dim=1), torch.stack(values, dim=1)


class SplitVisionKVAdapter(nn.Module):
    """Map source-model visual K and V caches to target per-layer visual K/V.

    The K and V paths are intentionally separate: source visual K only predicts
    target visual K content, and source visual V only predicts target visual V.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        source_layers: list[int],
        source_dim: int,
        num_key_value_heads: int,
        head_dim: int,
        bottleneck_dim: int = 32,
        shared_down: bool = True,
        identity_down: bool = False,
    ) -> None:
        super().__init__()
        if not source_layers:
            raise ValueError("source_layers must be non-empty")
        self.num_layers = int(num_layers)
        self.source_layers = [int(x) for x in source_layers]
        self.source_dim = int(source_dim)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.shared_down = bool(shared_down)
        self.identity_down = bool(identity_down)

        self.source_mix_k = nn.Parameter(torch.zeros(self.num_layers, len(self.source_layers)))
        self.source_mix_v = nn.Parameter(torch.zeros(self.num_layers, len(self.source_layers)))
        if self.identity_down:
            if self.bottleneck_dim != self.source_dim:
                raise ValueError("identity_down requires bottleneck_dim == source_dim")
            self.down_k = nn.Identity()
            self.down_v = nn.Identity()
        elif self.shared_down:
            self.down_k = nn.Linear(self.source_dim, self.bottleneck_dim, bias=False)
            self.down_v = nn.Linear(self.source_dim, self.bottleneck_dim, bias=False)
        else:
            self.down_k = nn.ModuleList(
                [nn.Linear(self.source_dim, self.bottleneck_dim, bias=False) for _ in range(self.num_layers)]
            )
            self.down_v = nn.ModuleList(
                [nn.Linear(self.source_dim, self.bottleneck_dim, bias=False) for _ in range(self.num_layers)]
            )
        self.up_k = nn.ModuleList(
            [nn.Linear(self.bottleneck_dim, self.num_key_value_heads * self.head_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.up_v = nn.ModuleList(
            [nn.Linear(self.bottleneck_dim, self.num_key_value_heads * self.head_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.output_scale_k = nn.Parameter(torch.ones(self.num_layers))
        self.output_scale_v = nn.Parameter(torch.ones(self.num_layers))

        modules: list[nn.Module] = []
        if not self.identity_down:
            if self.shared_down:
                modules.extend([self.down_k, self.down_v])
            else:
                modules.extend(list(self.down_k))
                modules.extend(list(self.down_v))
        modules.extend(list(self.up_k))
        modules.extend(list(self.up_v))
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02 if module.out_features == self.bottleneck_dim else 0.01)

    def _mix(self, source: Tensor, layer_idx: int, mix: Tensor) -> Tensor:
        weights = F.softmax(mix[int(layer_idx)].float(), dim=-1).to(dtype=source.dtype)
        return torch.einsum("s,bsnd->bnd", weights, source)

    def forward_layer(self, source_k: Tensor, source_v: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        if source_k.ndim != 4 or source_v.ndim != 4:
            raise ValueError("source_k/source_v must have shape [B, S, Nv, D]")
        idx = int(layer_idx)
        mixed_k = self._mix(source_k, idx, self.source_mix_k)
        mixed_v = self._mix(source_v, idx, self.source_mix_v)
        if self.identity_down:
            hidden_k = mixed_k
            hidden_v = mixed_v
        else:
            down_k = self.down_k if self.shared_down else self.down_k[idx]
            down_v = self.down_v if self.shared_down else self.down_v[idx]
            hidden_k = F.silu(down_k(mixed_k))
            hidden_v = F.silu(down_v(mixed_v))
        key = self.up_k[idx](hidden_k) * self.output_scale_k[idx].to(dtype=hidden_k.dtype)
        value = self.up_v[idx](hidden_v) * self.output_scale_v[idx].to(dtype=hidden_v.dtype)
        batch, tokens, _ = key.shape
        key = key.view(batch, tokens, self.num_key_value_heads, self.head_dim)
        value = value.view(batch, tokens, self.num_key_value_heads, self.head_dim)
        return key.contiguous(), value.contiguous()

    def forward(self, source_k: Tensor, source_v: Tensor) -> tuple[Tensor, Tensor]:
        keys = []
        values = []
        for layer_idx in range(self.num_layers):
            key, value = self.forward_layer(source_k, source_v, layer_idx)
            keys.append(key)
            values.append(value)
        return torch.stack(keys, dim=1), torch.stack(values, dim=1)


class HeadwiseSplitVisionKVAdapter(nn.Module):
    """Head-aware source visual K/V -> target visual K/V mapper.

    Source CLIP K/V is treated as [source_heads, source_head_dim] instead of a
    flat vector. Each target LLM KV head first mixes source heads, then applies
    a per-target-layer/head projection from source_head_dim to target head_dim.
    K and V use separate parameters.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        source_layers: list[int],
        source_dim: int,
        source_num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if not source_layers:
            raise ValueError("source_layers must be non-empty")
        if int(source_dim) % int(source_num_heads) != 0:
            raise ValueError("source_dim must be divisible by source_num_heads")
        self.num_layers = int(num_layers)
        self.source_layers = [int(x) for x in source_layers]
        self.source_dim = int(source_dim)
        self.source_num_heads = int(source_num_heads)
        self.source_head_dim = self.source_dim // self.source_num_heads
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.bottleneck_dim = self.source_head_dim
        self.shared_down = False
        self.identity_down = False

        self.source_mix_k = nn.Parameter(torch.zeros(self.num_layers, len(self.source_layers)))
        self.source_mix_v = nn.Parameter(torch.zeros(self.num_layers, len(self.source_layers)))
        self.head_mix_k = nn.Parameter(torch.zeros(self.num_layers, self.num_key_value_heads, self.source_num_heads))
        self.head_mix_v = nn.Parameter(torch.zeros(self.num_layers, self.num_key_value_heads, self.source_num_heads))
        self.proj_k = nn.Parameter(
            torch.empty(self.num_layers, self.num_key_value_heads, self.source_head_dim, self.head_dim)
        )
        self.proj_v = nn.Parameter(
            torch.empty(self.num_layers, self.num_key_value_heads, self.source_head_dim, self.head_dim)
        )
        self.output_scale_k = nn.Parameter(torch.ones(self.num_layers))
        self.output_scale_v = nn.Parameter(torch.ones(self.num_layers))
        nn.init.normal_(self.proj_k, std=0.01)
        nn.init.normal_(self.proj_v, std=0.01)

    def _layer_mix(self, source: Tensor, layer_idx: int, mix: Tensor) -> Tensor:
        if source.ndim != 4:
            raise ValueError("source must have shape [B, S, Nv, D]")
        if source.shape[-1] != self.source_dim:
            raise ValueError(f"source dim mismatch: got {source.shape[-1]} expected {self.source_dim}")
        weights = F.softmax(mix[int(layer_idx)].float(), dim=-1).to(dtype=source.dtype)
        mixed = torch.einsum("s,bsnd->bnd", weights, source)
        return mixed.view(mixed.shape[0], mixed.shape[1], self.source_num_heads, self.source_head_dim)

    def forward_layer(self, source_k: Tensor, source_v: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        idx = int(layer_idx)
        mixed_k = self._layer_mix(source_k, idx, self.source_mix_k)
        mixed_v = self._layer_mix(source_v, idx, self.source_mix_v)
        head_w_k = F.softmax(self.head_mix_k[idx].float(), dim=-1).to(dtype=mixed_k.dtype)
        head_w_v = F.softmax(self.head_mix_v[idx].float(), dim=-1).to(dtype=mixed_v.dtype)
        per_target_k = torch.einsum("ts,bnsd->bntd", head_w_k, mixed_k)
        per_target_v = torch.einsum("ts,bnsd->bntd", head_w_v, mixed_v)
        key = torch.einsum("bntd,tdm->bntm", per_target_k, self.proj_k[idx].to(dtype=per_target_k.dtype))
        value = torch.einsum("bntd,tdm->bntm", per_target_v, self.proj_v[idx].to(dtype=per_target_v.dtype))
        key = key * self.output_scale_k[idx].to(dtype=key.dtype)
        value = value * self.output_scale_v[idx].to(dtype=value.dtype)
        return key.contiguous(), value.contiguous()

    def forward(self, source_k: Tensor, source_v: Tensor) -> tuple[Tensor, Tensor]:
        keys = []
        values = []
        for layer_idx in range(self.num_layers):
            key, value = self.forward_layer(source_k, source_v, layer_idx)
            keys.append(key)
            values.append(value)
        return torch.stack(keys, dim=1), torch.stack(values, dim=1)
