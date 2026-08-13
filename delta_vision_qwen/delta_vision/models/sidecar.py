from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from delta_vision.runtime.basis import reconstruct_delta
from delta_vision.runtime.ops import VisualKVCache, cross_attention, merge_heads, split_heads


SUPPORTED_OUTPUT_MODES = {
    "residual",
    "residual_full",
    "factorized_lowrank",
    "factorized_full",
    "factorized_native_o",
    "factorized_native_head_o",
    "factorized_native_head_o_pure",
    "factorized_native_head_o_residual",
    "native_cross_attention",
    "native_visual_kv_injection",
    "native_visual_kv_split",
}
LOWRANK_OUTPUT_MODES = {
    "residual",
    "factorized_lowrank",
    "factorized_native_head_o",
    "factorized_native_head_o_residual",
}
HEADWISE_NATIVE_DELTA_MODES = {
    "factorized_native_head_o",
    "factorized_native_head_o_pure",
    "factorized_native_head_o_residual",
    "native_visual_kv_split",
}
NATIVE_OUTPUT_PROJECTION_MODES = HEADWISE_NATIVE_DELTA_MODES | {
    "factorized_native_o",
    "native_cross_attention",
}
NO_LAYER_CONDITION_MODES = {
    "native_visual_kv_injection",
    "native_visual_kv_split",
}


class ReaderMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        if activation not in {"gelu", "silu", "swiglu", "situ_glu"}:
            raise ValueError(f"unsupported reader activation: {activation}")
        self.activation = activation
        self.norm = nn.LayerNorm(hidden_size)
        if activation in {"swiglu", "situ_glu"}:
            # Keep GLU parameter count close to the dense MLP:
            # dense has 2*d*mlp_dim, GLU has 3*d*inner.
            inner_dim = max(hidden_size, int(round(mlp_dim * 2.0 / 3.0)))
            self.gate_proj = nn.Linear(hidden_size, inner_dim)
            self.up_proj = nn.Linear(hidden_size, inner_dim)
            self.down_proj = nn.Linear(inner_dim, hidden_size)
            self.fc1 = None
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(hidden_size, mlp_dim)
            self.fc2 = nn.Linear(mlp_dim, hidden_size)
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None

    @staticmethod
    def _situ(x: Tensor) -> Tensor:
        cap = 100.0
        return cap * torch.tanh(F.silu(x) / cap)

    @staticmethod
    def _soft_cap(x: Tensor) -> Tensor:
        cap = 100.0
        return cap * torch.tanh(x / cap)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        if self.activation == "gelu":
            assert self.fc1 is not None and self.fc2 is not None
            return self.fc2(F.gelu(self.fc1(x)))
        if self.activation == "silu":
            assert self.fc1 is not None and self.fc2 is not None
            return self.fc2(F.silu(self.fc1(x)))
        assert self.gate_proj is not None and self.up_proj is not None and self.down_proj is not None
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.activation == "situ_glu":
            hidden = self._situ(gate) * self._soft_cap(up)
        else:
            hidden = F.silu(gate) * up
        return self.down_proj(hidden)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_to_tensor(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding to a tensor.

    t: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim] from Qwen rotary_emb
    """
    if t.ndim == 4 and t.shape[1] != cos.shape[1] and t.shape[2] == cos.shape[1]:
        # t is [batch, heads, seq, head_dim], cos is [batch, seq, head_dim]
        cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
        sin = sin.unsqueeze(1)
    elif t.ndim == 4 and t.shape[1] == cos.shape[1]:
        # t is [batch, seq, heads, head_dim], cos is [batch, seq, head_dim]
        cos = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
        sin = sin.unsqueeze(2)
    return (t * cos) + (_rotate_half(t) * sin)




class DeltaVisionModule(nn.Module):
    """Shared delta-vision module that predicts low-rank visual-effect coefficients.

    Each layer owns a frozen or trainable effect basis B_l with shape
    [rank, hidden_size]. The shared reader predicts coefficients Z_l in that
    basis and reconstructs the residual as:

        Delta_hat_l = alpha_l * Z_l @ B_l
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        rank: int,
        sidecar_dim: int = 512,
        num_heads: int = 8,
        state_tokens: int = 0,
        dropout: float = 0.0,
        gate_init: float = 1.0,
        basis: Tensor | None = None,
        train_basis: bool = False,
        reader_mlp_ratio: float = 0.0,
        reader_activation: str = "situ_glu",
        layer_adapter_rank: int = 0,
        reader_fuse_query: bool = False,
        reader_concat_query: bool = False,
        normalize_basis_rows: bool = False,
        shared_basis: bool = False,
        output_mode: str = "residual",
        corrector_layers: str | Iterable[int] | None = None,
        corrector_dim: int = 0,
        block_corrector_groups: str | None = None,
        block_corrector_dim: int = 0,
        use_rope: bool = False,
        layer_condition_mode: str = "query",
        reader_mode: str = "cross_attention",
        latent_tokens: int = 64,
        visual_transform_mode: str = "none",
        visual_transform_rank: int = 128,
        visual_transform_activation: str = "gelu",
    ) -> None:
        super().__init__()
        if sidecar_dim % num_heads != 0:
            raise ValueError("sidecar_dim must be divisible by num_heads")
        if output_mode not in SUPPORTED_OUTPUT_MODES:
            raise ValueError(f"unsupported Sidecar output_mode: {output_mode}")
        if output_mode in NO_LAYER_CONDITION_MODES:
            layer_condition_mode = "none"
        expected_basis_shape = (1, rank, hidden_size) if shared_basis else (num_layers, rank, hidden_size)
        if basis is not None and basis.shape != expected_basis_shape:
            raise ValueError(f"basis must have shape {expected_basis_shape}")
        if layer_adapter_rank < 0:
            raise ValueError("layer_adapter_rank must be non-negative")
        if reader_mode not in {"cross_attention", "pooled"}:
            raise ValueError(f"unsupported reader_mode: {reader_mode}")
        if visual_transform_mode not in {
            "none",
            "layer_kv_adapter",
            "stage_kv_adapter",
            "stage_kv_film",
            "depth_pos_mixing",
            "recurrent_adapter",
            "full_cascade_adapter",
            "latent_compressor",
            "context_shift",
        }:
            raise ValueError(f"unsupported visual_transform_mode: {visual_transform_mode}")
        if latent_tokens <= 0:
            raise ValueError("latent_tokens must be positive")
        if layer_condition_mode not in {"query", "none", "post_film", "hdelta_film"}:
            raise ValueError(f"unsupported layer_condition_mode: {layer_condition_mode}")
        if visual_transform_rank < 0:
            raise ValueError("visual_transform_rank must be non-negative")
        if visual_transform_activation not in {"gelu", "silu"}:
            raise ValueError(f"unsupported visual_transform_activation: {visual_transform_activation}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rank = rank
        self.sidecar_dim = sidecar_dim
        self.state_tokens = state_tokens
        self.num_heads = num_heads
        self.reader_mlp_ratio = float(reader_mlp_ratio)
        self.reader_activation = reader_activation
        self.layer_adapter_rank = int(layer_adapter_rank)
        self.reader_fuse_query = bool(reader_fuse_query)
        self.reader_concat_query = bool(reader_concat_query)
        self.normalize_basis_rows = bool(normalize_basis_rows)
        self.shared_basis = bool(shared_basis)
        self.output_mode = output_mode
        self.uses_lowrank_output = output_mode in LOWRANK_OUTPUT_MODES
        self.corrector_layers = self._parse_corrector_layers(corrector_layers, num_layers)
        self.corrector_dim = int(corrector_dim)
        self.block_corrector_groups = self._parse_block_corrector_groups(block_corrector_groups, num_layers)
        self.block_corrector_dim = int(block_corrector_dim)
        self._block_corrector_layer_to_group = {
            layer: group_idx for group_idx, layers in enumerate(self.block_corrector_groups) for layer in layers
        }
        self.runtime_use_state = True
        self.use_rope = bool(use_rope)
        self.layer_condition_mode = str(layer_condition_mode)
        self.reader_mode = str(reader_mode)
        self.latent_tokens = int(latent_tokens)
        self.visual_transform_mode = str(visual_transform_mode)
        self.visual_transform_rank = int(visual_transform_rank)
        self.visual_transform_activation = str(visual_transform_activation)
        self._cached_inference_basis: Tensor | None = None
        self._cached_inference_basis_device: torch.device | None = None
        self._cached_inference_basis_dtype: torch.dtype | None = None
        self._cached_adapter_down: Tensor | None = None
        self._cached_adapter_up: Tensor | None = None
        self._cached_coeff_basis: Tensor | None = None
        self._cached_adapter_basis: Tensor | None = None
        self._cached_kv_proj_weight: Tensor | None = None
        self.runtime_fold_output_basis = False
        self.last_visual_mass: Tensor | None = None

        self.q_proj = nn.Linear(hidden_size, sidecar_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, sidecar_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, sidecar_dim, bias=False)
        self.visual_pool_norm = nn.LayerNorm(hidden_size) if self.reader_mode == "pooled" else None
        self.visual_pool_proj = nn.Linear(hidden_size, sidecar_dim, bias=False) if self.reader_mode == "pooled" else None
        self.visual_latent_query = None
        self.visual_latent_norm = None
        self.visual_latent_k = None
        self.visual_latent_v = None
        if self.layer_condition_mode == "none":
            self.layer_embed = None
            self.layer_condition_proj = None
            self.post_layer_film = None
            self.hdelta_film = None
        elif self.layer_condition_mode == "hdelta_film":
            self.layer_embed = None
            self.layer_condition_proj = None
            self.post_layer_film = None
            self.hdelta_film = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, sidecar_dim * 2, bias=False),
            )
        else:
            self.layer_embed = nn.Embedding(num_layers, sidecar_dim)
            self.layer_condition_proj = nn.Linear(sidecar_dim, sidecar_dim, bias=False)
            self.post_layer_film = (
                nn.Linear(sidecar_dim, sidecar_dim * 2, bias=False)
                if self.layer_condition_mode == "post_film"
                else None
            )
            self.hdelta_film = None
        self.reader_norm = nn.LayerNorm(sidecar_dim)
        self.reader_fusion_proj = (
            nn.Linear(sidecar_dim * 2, sidecar_dim, bias=False) if self.reader_concat_query else None
        )
        if reader_mlp_ratio > 0:
            mlp_dim = max(sidecar_dim, int(round(sidecar_dim * reader_mlp_ratio)))
            self.reader_mlp = ReaderMLP(sidecar_dim, mlp_dim, reader_activation)
        else:
            self.reader_mlp = None
        self.coeff_head = nn.Linear(sidecar_dim, rank, bias=False) if self.uses_lowrank_output else None
        if output_mode in HEADWISE_NATIVE_DELTA_MODES:
            self.mass_head = nn.Linear(sidecar_dim, num_heads, bias=True)
        elif output_mode.startswith("factorized"):
            self.mass_head = nn.Linear(sidecar_dim, 1, bias=True)
        else:
            self.mass_head = None
        self.visual_full_head = (
            nn.Linear(sidecar_dim, hidden_size, bias=False)
            if output_mode in {"residual_full", "factorized_full"}
            else None
        )
        if self.uses_lowrank_output and layer_adapter_rank > 0:
            self.layer_adapter_down = nn.ModuleList(
                [nn.Linear(sidecar_dim, layer_adapter_rank, bias=False) for _ in range(num_layers)]
            )
            self.layer_adapter_up = nn.ModuleList(
                [nn.Linear(layer_adapter_rank, rank, bias=False) for _ in range(num_layers)]
            )
        else:
            self.layer_adapter_down = None
            self.layer_adapter_up = None
        if self.uses_lowrank_output and self.corrector_layers and self.corrector_dim > 0:
            self.corrector_down = nn.ModuleDict(
                {str(layer): nn.Linear(sidecar_dim, self.corrector_dim, bias=False) for layer in self.corrector_layers}
            )
            self.corrector_up = nn.ModuleDict(
                {str(layer): nn.Linear(self.corrector_dim, rank, bias=False) for layer in self.corrector_layers}
            )
            self.corrector_gate = nn.Parameter(torch.zeros(num_layers))
        else:
            self.corrector_down = None
            self.corrector_up = None
            self.register_parameter("corrector_gate", None)
        if self.block_corrector_groups and self.block_corrector_dim > 0:
            self.block_corrector_down = nn.ModuleDict(
                {
                    str(group_idx): nn.Linear(sidecar_dim, self.block_corrector_dim, bias=False)
                    for group_idx in range(len(self.block_corrector_groups))
                }
            )
            self.block_corrector_up = nn.ModuleDict(
                {
                    str(group_idx): nn.Linear(self.block_corrector_dim, hidden_size, bias=False)
                    for group_idx in range(len(self.block_corrector_groups))
                }
            )
            self.block_corrector_gate = nn.Parameter(torch.zeros(num_layers))
        else:
            self.block_corrector_down = None
            self.block_corrector_up = None
            self.register_parameter("block_corrector_gate", None)
        self.visual_context_norm = None
        self.visual_context_current = None
        self.visual_context_delta = None
        self.register_parameter("visual_context_gate", None)
        if self.visual_transform_mode == "depth_pos_mixing":
            self.visual_depth_embed = nn.Embedding(num_layers, hidden_size)
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.register_parameter("visual_cascade_gate", None)
            self.register_parameter("visual_recurrent_gate", None)
        elif self.visual_transform_mode == "layer_kv_adapter":
            adapter_rank = max(1, self.visual_transform_rank)
            self.visual_depth_embed = None
            self.visual_adapter_down = nn.ModuleList(
                [nn.Linear(hidden_size, adapter_rank, bias=False) for _ in range(num_layers)]
            )
            self.visual_adapter_up = nn.ModuleList(
                [nn.Linear(adapter_rank, hidden_size, bias=False) for _ in range(num_layers)]
            )
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.register_parameter("visual_cascade_gate", None)
            self.register_parameter("visual_recurrent_gate", None)
        elif self.visual_transform_mode in {"stage_kv_adapter", "stage_kv_film"}:
            adapter_rank = max(1, self.visual_transform_rank)
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = nn.ModuleList(
                [nn.Linear(hidden_size, adapter_rank, bias=False) for _ in range(4)]
            )
            self.visual_stage_up = nn.ModuleList(
                [nn.Linear(adapter_rank, hidden_size, bias=False) for _ in range(4)]
            )
            self.visual_stage_film = (
                nn.Parameter(torch.zeros(num_layers, 2, hidden_size))
                if self.visual_transform_mode == "stage_kv_film"
                else None
            )
            self.register_parameter("visual_cascade_gate", None)
            self.register_parameter("visual_recurrent_gate", None)
        elif self.visual_transform_mode == "recurrent_adapter":
            adapter_rank = max(1, self.visual_transform_rank)
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = nn.LayerNorm(hidden_size)
            self.visual_recurrent_text = nn.Linear(hidden_size, hidden_size, bias=False)
            self.visual_recurrent_down = nn.Linear(hidden_size, adapter_rank, bias=False)
            self.visual_recurrent_up = nn.Linear(adapter_rank, hidden_size, bias=False)
            self.visual_recurrent_gate = nn.Parameter(torch.zeros(num_layers))
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.register_parameter("visual_cascade_gate", None)
        elif self.visual_transform_mode == "full_cascade_adapter":
            adapter_rank = max(1, self.visual_transform_rank)
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.register_parameter("visual_recurrent_gate", None)
            self.visual_cascade_norm = nn.LayerNorm(hidden_size)
            self.visual_cascade_text = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)]
            )
            self.visual_cascade_down = nn.ModuleList(
                [nn.Linear(hidden_size, adapter_rank, bias=False) for _ in range(3)]
            )
            self.visual_cascade_up = nn.ModuleList(
                [nn.Linear(adapter_rank, hidden_size, bias=False) for _ in range(3)]
            )
            self.visual_cascade_gate = nn.Parameter(torch.full((3,), 0.01))
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.visual_latent_query = None
            self.visual_latent_norm = None
            self.visual_latent_k = None
            self.visual_latent_v = None
        elif self.visual_transform_mode == "latent_compressor":
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.register_parameter("visual_recurrent_gate", None)
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.register_parameter("visual_cascade_gate", None)
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.visual_latent_query = nn.Parameter(torch.empty(self.latent_tokens, sidecar_dim))
            self.visual_latent_norm = nn.LayerNorm(hidden_size)
            self.visual_latent_k = nn.Linear(hidden_size, sidecar_dim, bias=False)
            self.visual_latent_v = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.visual_transform_mode == "context_shift":
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.visual_latent_query = None
            self.visual_latent_norm = None
            self.visual_latent_k = None
            self.visual_latent_v = None
            self.visual_context_norm = nn.LayerNorm(hidden_size)
            self.visual_context_current = nn.Linear(hidden_size, hidden_size, bias=False)
            self.visual_context_delta = nn.Linear(hidden_size, hidden_size, bias=False)
            self.visual_context_gate = nn.Parameter(torch.zeros(num_layers))
        else:
            self.visual_depth_embed = None
            self.visual_adapter_down = None
            self.visual_adapter_up = None
            self.visual_recurrent_norm = None
            self.visual_recurrent_text = None
            self.visual_recurrent_down = None
            self.visual_recurrent_up = None
            self.visual_cascade_norm = None
            self.visual_cascade_text = None
            self.visual_cascade_down = None
            self.visual_cascade_up = None
            self.visual_stage_down = None
            self.visual_stage_up = None
            self.visual_stage_film = None
            self.visual_cascade_up = None
            self.visual_latent_query = None
            self.visual_latent_norm = None
            self.visual_latent_k = None
            self.visual_latent_v = None
            self.register_parameter("visual_cascade_gate", None)
            self.register_parameter("visual_recurrent_gate", None)
        self.dropout = dropout
        self.gate = nn.Parameter(torch.full((num_layers,), float(gate_init)))

        if self.uses_lowrank_output and basis is None:
            basis = torch.empty(*expected_basis_shape)
            for layer_idx in range(expected_basis_shape[0]):
                nn.init.orthogonal_(basis[layer_idx])
        if self.uses_lowrank_output and train_basis:
            if basis is None:
                raise RuntimeError("low-rank output basis unexpectedly missing")
            self.basis = nn.Parameter(basis.float())
        elif self.uses_lowrank_output:
            if basis is None:
                raise RuntimeError("low-rank output basis unexpectedly missing")
            self.register_buffer("basis", basis.float(), persistent=True)
        else:
            self.register_parameter("basis", None)

        if state_tokens > 0:
            self.state_seed = nn.Parameter(torch.empty(state_tokens, sidecar_dim))
            self.state_init_proj = nn.Linear(sidecar_dim, sidecar_dim)
            self.state_update = nn.Sequential(
                nn.LayerNorm(sidecar_dim),
                nn.Linear(sidecar_dim, sidecar_dim * 2),
                nn.GELU(),
                nn.Linear(sidecar_dim * 2, sidecar_dim),
            )
            self.state_norm = nn.LayerNorm(sidecar_dim)
        else:
            self.register_parameter("state_seed", None)
            self.state_init_proj = None
            self.state_update = None
            self.state_norm = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.layer_embed is not None:
            nn.init.normal_(self.layer_embed.weight, mean=0.0, std=0.02)
        if self.layer_condition_proj is not None:
            nn.init.eye_(self.layer_condition_proj.weight)
        if self.post_layer_film is not None:
            nn.init.zeros_(self.post_layer_film.weight)
        if self.hdelta_film is not None:
            nn.init.zeros_(self.hdelta_film[-1].weight)
        if self.coeff_head is not None:
            nn.init.zeros_(self.coeff_head.weight)
        if self.mass_head is not None:
            nn.init.zeros_(self.mass_head.weight)
            nn.init.constant_(self.mass_head.bias, -2.0)
        if self.visual_full_head is not None:
            nn.init.zeros_(self.visual_full_head.weight)
        if self.visual_context_current is not None:
            nn.init.zeros_(self.visual_context_current.weight)
        if self.visual_context_delta is not None:
            nn.init.zeros_(self.visual_context_delta.weight)
        if self.layer_adapter_up is not None:
            for up in self.layer_adapter_up:
                nn.init.zeros_(up.weight)
        if self.state_seed is not None:
            nn.init.normal_(self.state_seed, mean=0.0, std=0.02)
        if self.corrector_up is not None:
            for up in self.corrector_up.values():
                nn.init.zeros_(up.weight)
        if self.block_corrector_up is not None:
            for up in self.block_corrector_up.values():
                nn.init.xavier_uniform_(up.weight)
        if self.visual_depth_embed is not None:
            nn.init.zeros_(self.visual_depth_embed.weight)
        if self.visual_adapter_up is not None:
            for up in self.visual_adapter_up:
                nn.init.zeros_(up.weight)
        if self.visual_recurrent_up is not None:
            nn.init.zeros_(self.visual_recurrent_up.weight)
        if self.visual_latent_query is not None:
            nn.init.normal_(self.visual_latent_query, mean=0.0, std=0.02)

    def _pooled_visual_features(self, vision_states: Tensor, vision_padding_mask: Tensor | None) -> Tensor:
        if self.visual_pool_norm is None or self.visual_pool_proj is None:
            raise RuntimeError("pooled reader is not initialized")
        if vision_padding_mask is None:
            pooled = vision_states.mean(dim=1)
        else:
            keep = (~vision_padding_mask).to(dtype=vision_states.dtype).unsqueeze(-1)
            pooled = (vision_states * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        pooled = self.visual_pool_norm(pooled)
        return self.visual_pool_proj(pooled).unsqueeze(1)

    def _compress_visual_latents(
        self,
        vision_states: Tensor,
        vision_padding_mask: Tensor | None,
    ) -> Tensor:
        if (
            self.visual_latent_query is None
            or self.visual_latent_norm is None
            or self.visual_latent_k is None
            or self.visual_latent_v is None
        ):
            raise RuntimeError("latent visual compressor is not initialized")
        memory = self.visual_latent_norm(vision_states)
        key = self.visual_latent_k(memory)
        value = self.visual_latent_v(memory)
        query = self.visual_latent_query.to(device=vision_states.device, dtype=key.dtype)
        scores = torch.einsum("ld,bnd->bln", query, key) / (float(self.sidecar_dim) ** 0.5)
        if vision_padding_mask is not None:
            scores = scores.masked_fill(vision_padding_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        return torch.einsum("bln,bnh->blh", weights, value)

    @staticmethod
    def _parse_corrector_layers(layers: str | Iterable[int] | None, num_layers: int) -> tuple[int, ...]:
        if layers is None:
            return ()
        if isinstance(layers, str):
            if not layers.strip():
                return ()
            parsed = {int(item.strip()) for item in layers.split(",") if item.strip()}
        else:
            parsed = {int(item) for item in layers}
        invalid = [layer for layer in parsed if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(f"corrector layer out of range: {invalid}")
        return tuple(sorted(parsed))

    @staticmethod
    def _parse_block_corrector_groups(groups: str | None, num_layers: int) -> tuple[tuple[int, ...], ...]:
        if groups is None or not str(groups).strip():
            return ()
        parsed_groups: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for raw_group in str(groups).split(";"):
            raw_group = raw_group.strip()
            if not raw_group:
                continue
            layers: set[int] = set()
            for item in raw_group.split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    start_s, end_s = item.split("-", 1)
                    start, end = int(start_s), int(end_s)
                    if end < start:
                        raise ValueError(f"invalid block corrector layer range: {item}")
                    layers.update(range(start, end + 1))
                else:
                    layers.add(int(item))
            invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
            if invalid:
                raise ValueError(f"block corrector layer out of range: {invalid}")
            overlap = sorted(seen.intersection(layers))
            if overlap:
                raise ValueError(f"block corrector layers appear in multiple groups: {overlap}")
            seen.update(layers)
            if layers:
                parsed_groups.append(tuple(sorted(layers)))
        return tuple(parsed_groups)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        optional_keys = (
            "layer_embed.weight",
            "layer_condition_proj.weight",
            "post_layer_film.weight",
            "hdelta_film.0.weight",
            "hdelta_film.0.bias",
            "hdelta_film.1.weight",
        )
        if self.layer_condition_mode in {"none", "hdelta_film"}:
            for key in optional_keys:
                state_dict.pop(prefix + key, None)
        else:
            layer_condition_key = prefix + "layer_condition_proj.weight"
            if self.layer_condition_proj is not None and layer_condition_key not in state_dict:
                state_dict[layer_condition_key] = torch.eye(self.sidecar_dim)
            post_film_key = prefix + "post_layer_film.weight"
            if self.post_layer_film is not None and post_film_key not in state_dict:
                state_dict[post_film_key] = torch.zeros(self.sidecar_dim * 2, self.sidecar_dim)
        if not self.uses_lowrank_output:
            unused_prefixes = (
                prefix + "basis",
                prefix + "coeff_head.",
                prefix + "layer_adapter_down.",
                prefix + "layer_adapter_up.",
                prefix + "corrector_down.",
                prefix + "corrector_up.",
                prefix + "corrector_gate",
            )
            for key in list(state_dict.keys()):
                if any(key.startswith(unused) for unused in unused_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _query_with_layer_condition(self, q_content: Tensor, layer_condition: Tensor | None) -> Tensor:
        if self.layer_condition_mode == "query" and layer_condition is not None:
            return q_content + layer_condition
        return q_content

    def _apply_post_layer_condition(self, reader_features: Tensor, layer_condition: Tensor | None) -> Tensor:
        if self.layer_condition_mode != "post_film":
            return reader_features
        if self.post_layer_film is None or layer_condition is None:
            raise RuntimeError("post_film layer conditioning is not initialized")
        scale_shift = self.post_layer_film(layer_condition.to(dtype=reader_features.dtype))
        scale, shift = scale_shift.chunk(2, dim=-1)
        return reader_features * (1.0 + scale) + shift

    def _apply_hdelta_condition(
        self,
        reader_features: Tensor,
        hidden_states: Tensor,
        initial_hidden_states: Tensor | None,
    ) -> Tensor:
        if self.layer_condition_mode != "hdelta_film":
            return reader_features
        if self.hdelta_film is None:
            raise RuntimeError("hdelta_film conditioning is not initialized")
        if initial_hidden_states is None:
            delta_hidden = torch.zeros_like(hidden_states)
        else:
            if initial_hidden_states.shape != hidden_states.shape:
                raise ValueError(
                    "initial_hidden_states must match hidden_states for hdelta_film: "
                    f"{tuple(initial_hidden_states.shape)} != {tuple(hidden_states.shape)}"
                )
            delta_hidden = hidden_states - initial_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
        cond = torch.cat([hidden_states, delta_hidden], dim=-1)
        scale_shift = self.hdelta_film(cond.to(dtype=reader_features.dtype))
        scale, shift = scale_shift.chunk(2, dim=-1)
        return reader_features * (1.0 + scale) + shift

    def _layer_condition(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        single_layer_id: int | None = None,
        layer_idx_tensor: Tensor | None = None,
    ) -> Tensor | None:
        if self.layer_condition_mode in {"none", "hdelta_film"}:
            return None
        if self.layer_embed is None or self.layer_condition_proj is None:
            raise RuntimeError("layer conditioning is not initialized")
        if single_layer_id is not None:
            layer_embed = self.layer_embed.weight[single_layer_id].to(device=device, dtype=dtype).view(1, 1, -1)
            layer_embed = layer_embed.expand(batch, 1, -1)
        else:
            if layer_idx_tensor is None:
                raise ValueError("layer_idx_tensor is required for batched layer conditioning")
            layer_embed = self.layer_embed(layer_idx_tensor).unsqueeze(1).to(dtype=dtype)
        return self.layer_condition_proj(layer_embed)

    def _apply_coeff_corrector(
        self,
        coeff: Tensor,
        reader_features: Tensor,
        single_layer_id: int | None,
        layer_idx_tensor: Tensor | None,
    ) -> Tensor:
        if self.corrector_down is None or self.corrector_up is None or self.corrector_gate is None:
            return coeff
        if single_layer_id is not None:
            key = str(single_layer_id)
            if key not in self.corrector_down:
                return coeff
            corr = self.corrector_up[key](F.gelu(self.corrector_down[key](reader_features)))
            gate = self.corrector_gate[single_layer_id].to(device=coeff.device, dtype=coeff.dtype).view(1, 1, 1)
            return coeff + gate * corr
        if layer_idx_tensor is None:
            return coeff
        corr_coeff = torch.zeros_like(coeff)
        for layer_id in self.corrector_layers:
            mask = layer_idx_tensor == int(layer_id)
            if not bool(mask.any()):
                continue
            key = str(layer_id)
            corr = self.corrector_up[key](F.gelu(self.corrector_down[key](reader_features[mask])))
            gate = self.corrector_gate[layer_id].to(device=coeff.device, dtype=coeff.dtype).view(1, 1, 1)
            corr_coeff[mask] = gate * corr
        return coeff + corr_coeff

    def _apply_block_corrector(
        self,
        residual: Tensor,
        reader_features: Tensor,
        single_layer_id: int | None,
        layer_idx_tensor: Tensor | None,
    ) -> Tensor:
        if (
            self.block_corrector_down is None
            or self.block_corrector_up is None
            or self.block_corrector_gate is None
        ):
            return residual
        if single_layer_id is not None:
            group_idx = self._block_corrector_layer_to_group.get(int(single_layer_id))
            if group_idx is None:
                return residual
            key = str(group_idx)
            corr = self.block_corrector_up[key](F.silu(self.block_corrector_down[key](reader_features)))
            gate = self.block_corrector_gate[single_layer_id].to(device=residual.device, dtype=residual.dtype).view(1, 1, 1)
            return residual + gate * corr.to(dtype=residual.dtype)
        if layer_idx_tensor is None:
            return residual
        corr_residual = torch.zeros_like(residual)
        for group_idx, layers in enumerate(self.block_corrector_groups):
            layer_mask = torch.zeros_like(layer_idx_tensor, dtype=torch.bool)
            for layer_id in layers:
                layer_mask = layer_mask | (layer_idx_tensor == int(layer_id))
            if not bool(layer_mask.any()):
                continue
            key = str(group_idx)
            corr = self.block_corrector_up[key](F.silu(self.block_corrector_down[key](reader_features[layer_mask])))
            gates = self.block_corrector_gate[layer_idx_tensor[layer_mask]].to(
                device=residual.device,
                dtype=residual.dtype,
            ).view(-1, 1, 1)
            corr_residual[layer_mask] = gates * corr.to(dtype=residual.dtype)
        return residual + corr_residual

    def prepare_visual_kv(
        self,
        vision_states: Tensor,
        vision_padding_mask: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
    ) -> VisualKVCache:
        if vision_states.ndim != 3:
            raise ValueError("vision_states must be a rank-3 tensor")
        if vision_padding_mask is not None and not bool(vision_padding_mask.any().item()):
            vision_padding_mask = None
        if (
            self._cached_kv_proj_weight is not None
            and not self.training
            and not self.k_proj.weight.requires_grad
            and not self.v_proj.weight.requires_grad
        ):
            kv = F.linear(vision_states, self._cached_kv_proj_weight.to(device=vision_states.device, dtype=vision_states.dtype))
            key_states, value_states = kv.split(self.sidecar_dim, dim=-1)
        else:
            key_states = self.k_proj(vision_states)
            value_states = self.v_proj(vision_states)
        key = split_heads(key_states, self.num_heads).transpose(1, 2).contiguous()
        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            key = _apply_rope_to_tensor(key, cos, sin)
        value = split_heads(value_states, self.num_heads).transpose(1, 2).contiguous()
        return VisualKVCache(
            key=key,
            value=value,
            padding_mask=vision_padding_mask,
        )

    def initial_state(self, vision_states: Tensor, vision_padding_mask: Tensor | None = None) -> Tensor | None:
        if self.state_tokens == 0:
            return None
        if vision_states.ndim != 3:
            raise ValueError("vision_states must be a rank-3 tensor")
        kv = self.v_proj(vision_states)
        if vision_padding_mask is None:
            summary = kv.mean(dim=1, keepdim=True)
        else:
            keep = (~vision_padding_mask).to(dtype=kv.dtype).unsqueeze(-1)
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
            summary = (kv * keep).sum(dim=1, keepdim=True) / denom
        assert self.state_init_proj is not None
        seed = self.state_seed.unsqueeze(0).expand(vision_states.shape[0], -1, -1)
        return seed + self.state_init_proj(summary)

    def initial_visual_memory_state(self, vision_states: Tensor) -> Tensor | None:
        if self.visual_transform_mode == "full_cascade_adapter":
            return vision_states
        if self.visual_transform_mode == "latent_compressor":
            return None
        if self.visual_transform_mode != "recurrent_adapter":
            return None
        return vision_states.mean(dim=1, keepdim=True)

    def _visual_transform_act(self, x: Tensor) -> Tensor:
        if self.visual_transform_activation == "silu":
            return F.silu(x)
        return F.gelu(x)

    def visual_memory_for_layer(
        self,
        vision_states: Tensor,
        layer_idx: int,
        hidden_states: Tensor | None = None,
        initial_hidden_states: Tensor | None = None,
        current_visual_memory: Tensor | None = None,
        vision_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return the layer-conditioned visual memory and next recurrent memory.

        This transforms the hidden-space visual memory before any Sidecar or
        native Qwen K/V projection. It lets the same V0 expose different
        memories across layers without saving Qwen's full visual KV cache.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        if self.visual_transform_mode == "none":
            return vision_states, current_visual_memory
        if self.visual_transform_mode == "latent_compressor":
            if current_visual_memory is None:
                current_visual_memory = self._compress_visual_latents(vision_states, vision_padding_mask)
            return current_visual_memory, current_visual_memory
        if self.visual_transform_mode == "context_shift":
            if self.visual_context_norm is None or self.visual_context_current is None or self.visual_context_delta is None:
                raise RuntimeError("visual context shift is not initialized")
            if hidden_states is None:
                current_summary = torch.zeros(
                    vision_states.shape[0],
                    1,
                    vision_states.shape[-1],
                    device=vision_states.device,
                    dtype=vision_states.dtype,
                )
                delta_summary = current_summary
            else:
                current_summary = hidden_states.mean(dim=1, keepdim=True)
                if initial_hidden_states is None:
                    delta_summary = torch.zeros_like(current_summary)
                else:
                    delta_summary = (hidden_states - initial_hidden_states.to(dtype=hidden_states.dtype)).mean(
                        dim=1,
                        keepdim=True,
                    )
            context = self.visual_context_current(current_summary) + self.visual_context_delta(delta_summary)
            context = self.visual_context_norm(context.to(dtype=vision_states.dtype))
            gate = torch.tanh(self.visual_context_gate[layer_idx]).to(
                device=vision_states.device,
                dtype=vision_states.dtype,
            ).view(1, 1, 1)
            return vision_states + gate * context, current_visual_memory
        if self.visual_transform_mode == "depth_pos_mixing":
            if self.visual_depth_embed is None:
                raise RuntimeError("visual_depth_embed is not initialized")
            depth = self.visual_depth_embed.weight[layer_idx].to(device=vision_states.device, dtype=vision_states.dtype)
            return vision_states + depth.view(1, 1, -1), current_visual_memory
        if self.visual_transform_mode == "layer_kv_adapter":
            if self.visual_adapter_down is None or self.visual_adapter_up is None:
                raise RuntimeError("visual layer adapters are not initialized")
            adapted = self.visual_adapter_down[layer_idx](vision_states)
            adapted = self.visual_adapter_up[layer_idx](self._visual_transform_act(adapted))
            return vision_states + adapted.to(dtype=vision_states.dtype), current_visual_memory
        if self.visual_transform_mode in {"stage_kv_adapter", "stage_kv_film"}:
            if self.visual_stage_down is None or self.visual_stage_up is None:
                raise RuntimeError("visual stage adapters are not initialized")
            num_stages = len(self.visual_stage_down)
            stage_idx = min(num_stages - 1, int(layer_idx * num_stages // max(1, self.num_layers)))
            stage_start = int(stage_idx * self.num_layers // num_stages)
            if current_visual_memory is None or layer_idx == stage_start:
                adapted = self.visual_stage_down[stage_idx](vision_states)
                adapted = self.visual_stage_up[stage_idx](self._visual_transform_act(adapted))
                current_visual_memory = vision_states + adapted.to(dtype=vision_states.dtype)
            memory = current_visual_memory
            if self.visual_transform_mode == "stage_kv_film":
                if self.visual_stage_film is None:
                    raise RuntimeError("visual stage FiLM is not initialized")
                film = self.visual_stage_film[layer_idx].to(device=memory.device, dtype=memory.dtype)
                gamma = film[0].view(1, 1, -1)
                beta = film[1].view(1, 1, -1)
                memory = memory * (1.0 + gamma) + beta
            return memory, current_visual_memory
        if self.visual_transform_mode == "full_cascade_adapter":
            if (
                self.visual_cascade_norm is None
                or self.visual_cascade_text is None
                or self.visual_cascade_down is None
                or self.visual_cascade_up is None
                or self.visual_cascade_gate is None
            ):
                raise RuntimeError("visual cascade adapter is not initialized")
            stage_idx = min(2, int(layer_idx * 3 // max(1, self.num_layers)))
            memory = current_visual_memory if current_visual_memory is not None else vision_states
            if hidden_states is None:
                text_context = torch.zeros(
                    memory.shape[0],
                    1,
                    memory.shape[-1],
                    device=memory.device,
                    dtype=memory.dtype,
                )
            else:
                text_context = hidden_states.mean(dim=1, keepdim=True)
                text_context = self.visual_cascade_text[stage_idx](text_context)
            update_input = self.visual_cascade_norm(memory + text_context.to(dtype=memory.dtype))
            update = self.visual_cascade_up[stage_idx](
                self._visual_transform_act(self.visual_cascade_down[stage_idx](update_input))
            )
            gate = self.visual_cascade_gate[stage_idx].to(device=memory.device, dtype=memory.dtype).view(1, 1, 1)
            next_memory = memory + gate * update.to(dtype=memory.dtype)
            return memory, next_memory
        if self.visual_transform_mode != "recurrent_adapter":
            raise RuntimeError(f"unsupported visual_transform_mode: {self.visual_transform_mode}")
        if (
            self.visual_recurrent_norm is None
            or self.visual_recurrent_text is None
            or self.visual_recurrent_down is None
            or self.visual_recurrent_up is None
            or self.visual_recurrent_gate is None
        ):
            raise RuntimeError("visual recurrent adapter is not initialized")
        state = current_visual_memory if current_visual_memory is not None else self.initial_visual_memory_state(vision_states)
        if state is None:
            raise RuntimeError("recurrent visual state unexpectedly missing")
        if hidden_states is None:
            text_summary = torch.zeros(
                vision_states.shape[0],
                1,
                vision_states.shape[-1],
                device=vision_states.device,
                dtype=vision_states.dtype,
            )
        else:
            text_summary = hidden_states.mean(dim=1, keepdim=True)
            text_summary = self.visual_recurrent_text(text_summary)
        gate = torch.tanh(self.visual_recurrent_gate[layer_idx]).to(
            device=vision_states.device,
            dtype=vision_states.dtype,
        ).view(1, 1, 1)
        memory_input = self.visual_recurrent_norm(vision_states + state.to(dtype=vision_states.dtype))
        memory_update = self.visual_recurrent_up(F.gelu(self.visual_recurrent_down(memory_input)))
        transformed = vision_states + gate * memory_update.to(dtype=vision_states.dtype)
        state_input = self.visual_recurrent_norm(state + text_summary.to(dtype=state.dtype))
        state_update = self.visual_recurrent_up(F.gelu(self.visual_recurrent_down(state_input)))
        next_state = state + gate * state_update.to(dtype=state.dtype)
        return transformed, next_state

    def layer_basis(self, layer_idx_tensor: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.basis is None:
            raise RuntimeError("low-rank basis is not initialized for this output mode")
        basis_source = self._inference_basis(device, dtype)
        if basis_source is None:
            basis_source = self.basis
        if self.shared_basis:
            basis = basis_source
            if layer_idx_tensor.ndim > 0 and layer_idx_tensor.shape[0] != 1:
                basis = basis.expand(layer_idx_tensor.shape[0], -1, -1)
        else:
            basis = basis_source[layer_idx_tensor]
        if basis.device != device or basis.dtype != dtype:
            basis = basis.to(device=device, dtype=dtype)
        if self.normalize_basis_rows and basis_source is self.basis:
            basis = F.normalize(basis.float(), p=2, dim=-1, eps=1e-6).to(dtype=dtype)
        return basis

    def clear_inference_cache(self) -> None:
        self._cached_inference_basis = None
        self._cached_inference_basis_device = None
        self._cached_inference_basis_dtype = None
        self._cached_adapter_down = None
        self._cached_adapter_up = None
        self._cached_coeff_basis = None
        self._cached_adapter_basis = None
        self._cached_kv_proj_weight = None

    def prepare_inference_cache(self, device: torch.device, dtype: torch.dtype) -> None:
        trainable_adapters = False
        if self.layer_adapter_down is not None and self.layer_adapter_up is not None:
            trainable_adapters = any(param.requires_grad for param in self.layer_adapter_down.parameters())
            trainable_adapters = trainable_adapters or any(param.requires_grad for param in self.layer_adapter_up.parameters())
        basis_trainable = self.basis is not None and self.basis.requires_grad
        if self.training or basis_trainable or trainable_adapters:
            self.clear_inference_cache()
            return
        if self.basis is None:
            if not self.k_proj.weight.requires_grad and not self.v_proj.weight.requires_grad:
                self._cached_kv_proj_weight = torch.cat(
                    [self.k_proj.weight.detach(), self.v_proj.weight.detach()],
                    dim=0,
                ).to(device=device, dtype=dtype).contiguous()
            return
        basis = self.basis
        if self.normalize_basis_rows:
            basis = F.normalize(basis.float(), p=2, dim=-1, eps=1e-6)
        self._cached_inference_basis = basis.to(device=device, dtype=dtype).contiguous()
        self._cached_inference_basis_device = device
        self._cached_inference_basis_dtype = dtype
        self._cached_kv_proj_weight = torch.cat(
            [self.k_proj.weight.detach(), self.v_proj.weight.detach()],
            dim=0,
        ).to(device=device, dtype=dtype).contiguous()
        if self.layer_adapter_down is not None and self.layer_adapter_up is not None:
            self._cached_adapter_down = torch.stack(
                [module.weight.detach() for module in self.layer_adapter_down],
                dim=0,
            ).to(device=device, dtype=dtype).contiguous()
            self._cached_adapter_up = torch.stack(
                [module.weight.detach() for module in self.layer_adapter_up],
                dim=0,
            ).to(device=device, dtype=dtype).contiguous()
        if self.shared_basis:
            basis_2d = self._cached_inference_basis[0]
            self._cached_coeff_basis = torch.matmul(
                self.coeff_head.weight.detach().to(device=device, dtype=dtype).transpose(0, 1),
                basis_2d,
            ).contiguous()
            if self._cached_adapter_up is not None:
                self._cached_adapter_basis = torch.matmul(
                    self._cached_adapter_up.transpose(1, 2),
                    basis_2d.unsqueeze(0),
                ).contiguous()

    def _inference_basis(self, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        if self.basis is None:
            return None
        if self.training or self.basis.requires_grad:
            return None
        if (
            self._cached_inference_basis is None
            or self._cached_inference_basis_device != device
            or self._cached_inference_basis_dtype != dtype
        ):
            self.prepare_inference_cache(device, dtype)
        return self._cached_inference_basis

    def decode_no_state(self, hidden_states: Tensor, layer_idx_tensor: Tensor, visual_kv: VisualKVCache) -> Tensor:
        """Inference-only no-state Sidecar path for autoregressive decode.

        This is mathematically the same as forward(..., sidecar_state=None,
        return_state=False) when runtime_use_state is disabled. It intentionally
        avoids the generic forward path's Python branches so torch.compile can
        produce a smaller decode graph.
        """
        if not self.uses_lowrank_output:
            raise RuntimeError("decode_no_state is only available for low-rank residual output modes")
        batch = hidden_states.shape[0]
        device = hidden_states.device
        layer_idx_tensor = layer_idx_tensor.to(device=device, dtype=torch.long)
        layer_condition = self._layer_condition(
            batch=batch,
            device=device,
            dtype=hidden_states.dtype,
            layer_idx_tensor=layer_idx_tensor,
        )

        q = self.q_proj(hidden_states)
        q = self._query_with_layer_condition(q, layer_condition)
        attn_out = merge_heads(
            cross_attention(
                query=split_heads(q, self.num_heads),
                key=visual_kv.key,
                value=visual_kv.value,
                padding_mask=visual_kv.padding_mask,
                dropout_p=self.dropout,
                training=self.training,
            )
        )
        if self.reader_concat_query:
            assert self.reader_fusion_proj is not None
            reader_input = self.reader_fusion_proj(torch.cat([attn_out, q], dim=-1))
        elif self.reader_fuse_query:
            reader_input = attn_out + q
        else:
            reader_input = attn_out

        reader_features = self.reader_norm(reader_input)
        if self.reader_mlp is not None:
            reader_features = reader_features + self.reader_mlp(reader_features)
        reader_features = self._apply_post_layer_condition(reader_features, layer_condition)

        if (
            self.output_mode == "residual"
            and
            self.runtime_fold_output_basis
            and self.corrector_down is None
            and self._cached_coeff_basis is not None
            and self._cached_adapter_down is not None
            and self._cached_adapter_basis is not None
        ):
            residual = torch.matmul(reader_features, self._cached_coeff_basis)
            down_weight = self._cached_adapter_down[layer_idx_tensor]
            adapter_basis = self._cached_adapter_basis[layer_idx_tensor]
            adapted = torch.einsum("btd,bad->bta", reader_features, down_weight)
            adapted = torch.nn.functional.gelu(adapted)
            residual = residual + torch.einsum("bta,bah->bth", adapted, adapter_basis)
            gates = self.gate[layer_idx_tensor].view(batch, 1, 1).to(device=device, dtype=residual.dtype)
            return residual * gates

        if self.coeff_head is None:
            raise RuntimeError("coeff_head is not initialized")
        coeff = self.coeff_head(reader_features)
        if self.layer_adapter_rank > 0:
            assert self._cached_adapter_down is not None and self._cached_adapter_up is not None
            down_weight = self._cached_adapter_down[layer_idx_tensor]
            up_weight = self._cached_adapter_up[layer_idx_tensor]
            adapted = torch.einsum("btd,bad->bta", reader_features, down_weight)
            adapted = torch.nn.functional.gelu(adapted)
            coeff = coeff + torch.einsum("bta,bra->btr", adapted, up_weight)
        coeff = self._apply_coeff_corrector(coeff, reader_features, None, layer_idx_tensor)
        basis = self.layer_basis(layer_idx_tensor, device, coeff.dtype)
        residual = reconstruct_delta(coeff, basis)
        gates = self.gate[layer_idx_tensor].view(batch, 1, 1)
        return residual * gates

    def decode_no_state_layer(self, hidden_states: Tensor, layer_idx: int, visual_kv: VisualKVCache) -> Tensor:
        """Inference-only no-state path specialized to a fixed layer id.

        This avoids dynamic layer-index gathers in autoregressive decode. It is
        mathematically equivalent to decode_no_state(..., layer_idx_tensor=[l])
        for a homogeneous batch at layer l.
        """
        if not self.uses_lowrank_output:
            raise RuntimeError("decode_no_state_layer is only available for low-rank residual output modes")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        batch = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        layer_condition = self._layer_condition(
            batch=batch,
            device=device,
            dtype=dtype,
            single_layer_id=layer_idx,
        )

        q = self.q_proj(hidden_states)
        q = self._query_with_layer_condition(q, layer_condition)
        attn_out = merge_heads(
            cross_attention(
                query=split_heads(q, self.num_heads),
                key=visual_kv.key,
                value=visual_kv.value,
                padding_mask=visual_kv.padding_mask,
                dropout_p=self.dropout,
                training=self.training,
            )
        )
        if self.reader_concat_query:
            assert self.reader_fusion_proj is not None
            reader_input = self.reader_fusion_proj(torch.cat([attn_out, q], dim=-1))
        elif self.reader_fuse_query:
            reader_input = attn_out + q
        else:
            reader_input = attn_out

        reader_features = self.reader_norm(reader_input)
        if self.reader_mlp is not None:
            reader_features = reader_features + self.reader_mlp(reader_features)
        reader_features = self._apply_post_layer_condition(reader_features, layer_condition)

        if (
            self.runtime_fold_output_basis
            and self.corrector_down is None
            and self._cached_coeff_basis is not None
            and self._cached_adapter_down is not None
            and self._cached_adapter_basis is not None
        ):
            residual = torch.matmul(reader_features, self._cached_coeff_basis)
            adapted = F.linear(reader_features, self._cached_adapter_down[layer_idx])
            adapted = torch.nn.functional.gelu(adapted)
            residual = residual + F.linear(adapted, self._cached_adapter_basis[layer_idx].transpose(0, 1))
            gate = self.gate[layer_idx].to(device=device, dtype=residual.dtype).view(1, 1, 1)
            return residual * gate

        if self.coeff_head is None:
            raise RuntimeError("coeff_head is not initialized")
        coeff = self.coeff_head(reader_features)
        if self.layer_adapter_rank > 0:
            if self._cached_adapter_down is not None and self._cached_adapter_up is not None:
                adapted = F.linear(reader_features, self._cached_adapter_down[layer_idx])
                adapted = torch.nn.functional.gelu(adapted)
                coeff = coeff + F.linear(adapted, self._cached_adapter_up[layer_idx])
            else:
                assert self.layer_adapter_down is not None and self.layer_adapter_up is not None
                adapted = self.layer_adapter_down[layer_idx](reader_features)
                coeff = coeff + self.layer_adapter_up[layer_idx](torch.nn.functional.gelu(adapted))
        coeff = self._apply_coeff_corrector(coeff, reader_features, layer_idx, None)

        basis_source = self._inference_basis(device, coeff.dtype)
        if basis_source is None:
            basis_source = self.basis
        if self.shared_basis:
            basis = basis_source
        else:
            basis = basis_source[layer_idx].unsqueeze(0)
        if basis.device != device or basis.dtype != coeff.dtype:
            basis = basis.to(device=device, dtype=coeff.dtype)
        if self.normalize_basis_rows and basis_source is self.basis:
            basis = F.normalize(basis.float(), p=2, dim=-1, eps=1e-6).to(dtype=coeff.dtype)
        residual = reconstruct_delta(coeff, basis)
        gate = self.gate[layer_idx].to(device=device, dtype=residual.dtype).view(1, 1, 1)
        return residual * gate

    def forward(
        self,
        hidden_states: Tensor,
        vision_states: Tensor | None,
        layer_idx: int | Tensor,
        sidecar_state: Tensor | None = None,
        visual_kv: VisualKVCache | None = None,
        vision_padding_mask: Tensor | None = None,
        return_state: bool = False,
        return_coefficients: bool = False,
        text_attention: Tensor | None = None,
        visual_mass: Tensor | None = None,
        output_projection: nn.Module | None = None,
        text_attention_heads: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        visual_position_embeddings: tuple[Tensor, Tensor] | None = None,
        query_states: Tensor | None = None,
        initial_hidden_states: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor | None, Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be a rank-3 tensor")
        self.last_visual_mass = None
        if query_states is not None and query_states.shape != (hidden_states.shape[0], hidden_states.shape[1], self.sidecar_dim):
            raise ValueError(
                "query_states must have shape "
                f"{(hidden_states.shape[0], hidden_states.shape[1], self.sidecar_dim)}, "
                f"got {tuple(query_states.shape)}"
            )
        if self.output_mode.startswith("factorized") and text_attention is None:
            raise ValueError(f"text_attention is required for Sidecar output_mode={self.output_mode}")
        if self.output_mode in NATIVE_OUTPUT_PROJECTION_MODES and output_projection is None:
            raise ValueError(f"output_projection is required for {self.output_mode}")
        if self.output_mode in HEADWISE_NATIVE_DELTA_MODES and text_attention_heads is None:
            raise ValueError(f"text_attention_heads is required for {self.output_mode}")
        if self.reader_mode != "pooled" and visual_kv is None:
            if vision_states is None:
                raise ValueError("either vision_states or visual_kv must be provided")
            visual_kv = self.prepare_visual_kv(
                vision_states,
                vision_padding_mask,
                position_embeddings=visual_position_embeddings,
            )
        elif visual_kv is not None and vision_padding_mask is not None:
            raise ValueError("pass padding_mask through visual_kv when visual_kv is provided")

        batch = hidden_states.shape[0]
        device = hidden_states.device
        single_layer_id: int | None = None
        if not torch.is_tensor(layer_idx):
            single_layer_id = int(layer_idx)
            layer_idx_tensor = None
        else:
            layer_idx_tensor = layer_idx.to(device=device, dtype=torch.long)
            if layer_idx_tensor.ndim == 0:
                single_layer_id = int(layer_idx_tensor.item())
                layer_idx_tensor = None
            elif layer_idx_tensor.shape != (batch,):
                raise ValueError("layer_idx must be scalar or shape [batch]")
        if single_layer_id is not None:
            if single_layer_id < 0 or single_layer_id >= self.num_layers:
                raise ValueError(f"layer_idx out of range: {single_layer_id}")
        else:
            assert layer_idx_tensor is not None
            if layer_idx_tensor.shape != (batch,):
                raise ValueError("layer_idx must be scalar or shape [batch]")
        layer_condition = self._layer_condition(
            batch=batch,
            device=device,
            dtype=hidden_states.dtype,
            single_layer_id=single_layer_id,
            layer_idx_tensor=layer_idx_tensor,
        )

        if query_states is not None:
            position_embeddings = None
        q_content = query_states if query_states is not None else self.q_proj(hidden_states)
        if query_states is None and self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            q_content_heads = split_heads(q_content, self.num_heads)
            q_content_heads = _apply_rope_to_tensor(q_content_heads, cos, sin)
            q_content = merge_heads(q_content_heads)
        reader_q = self._query_with_layer_condition(q_content, layer_condition)
        # When a native model query is supplied, it already includes that
        # layer's q_proj/q_norm/RoPE semantics. Adding a learned layer vector to
        # the attention query changes the visual attention scores, so keep the
        # native query pure and use the layer condition only in the reader.
        attn_q = q_content if query_states is not None else reader_q

        next_state = None
        use_state = self.state_tokens > 0 and self.runtime_use_state
        if use_state:
            if sidecar_state is None:
                if vision_states is None:
                    raise ValueError("vision_states is required to initialize state")
                sidecar_state = self.initial_state(vision_states, vision_padding_mask)
            if sidecar_state is None:
                raise RuntimeError("initial_state unexpectedly returned None")
            assert self.state_norm is not None
            state_q_input = sidecar_state if layer_condition is None else sidecar_state + layer_condition
            state_q = self.state_norm(state_q_input)
            q_all = torch.cat([attn_q, state_q], dim=1)
        else:
            if sidecar_state is not None and self.state_tokens == 0:
                raise ValueError("sidecar_state was provided but state_tokens == 0")
            q_all = attn_q

        if self.reader_mode == "pooled":
            if vision_states is None:
                raise ValueError("vision_states is required for pooled reader_mode")
            pooled = self._pooled_visual_features(vision_states, vision_padding_mask)
            text_attn_out = pooled.expand(-1, hidden_states.shape[1], -1)
            text_attn_heads_out = split_heads(text_attn_out, self.num_heads)
        else:
            assert visual_kv is not None
            q_heads = split_heads(q_all, self.num_heads)
            attn_heads_out = cross_attention(
                query=q_heads,
                key=visual_kv.key,
                value=visual_kv.value,
                padding_mask=visual_kv.padding_mask,
                dropout_p=self.dropout,
                training=self.training,
            )
            text_attn_heads_out = attn_heads_out[:, : hidden_states.shape[1]]
            attn_out = merge_heads(attn_heads_out)
            if use_state:
                text_attn_out = attn_out[:, : hidden_states.shape[1]]
                state_attn_out = attn_out[:, hidden_states.shape[1] :]
                assert self.state_update is not None and self.state_norm is not None
                text_context = reader_q.mean(dim=1, keepdim=True).expand_as(state_attn_out)
                next_state = self.state_norm(sidecar_state + self.state_update(state_attn_out + text_context))
            else:
                text_attn_out = attn_out

        if self.output_mode == "native_cross_attention":
            if output_projection is None:
                raise RuntimeError("output_projection unexpectedly missing")
            residual = output_projection(text_attn_out)
            if single_layer_id is not None:
                gates = self.gate[single_layer_id].to(device=device, dtype=residual.dtype).view(1, 1, 1)
            else:
                assert layer_idx_tensor is not None
                gates = self.gate[layer_idx_tensor].view(batch, 1, 1).to(device=device, dtype=residual.dtype)
            residual = residual * gates
            empty_coeff = residual.new_empty((*residual.shape[:2], 0))
            if return_state and return_coefficients:
                return residual, next_state, empty_coeff
            if return_state:
                return residual, next_state
            if return_coefficients:
                return residual, empty_coeff
            return residual

        if self.reader_concat_query:
            if self.reader_fusion_proj is None:
                raise RuntimeError("reader_fusion_proj is not initialized")
            reader_input = self.reader_fusion_proj(torch.cat([text_attn_out, reader_q], dim=-1))
        elif self.reader_fuse_query:
            reader_input = text_attn_out + reader_q
        else:
            reader_input = text_attn_out
        reader_features = self.reader_norm(reader_input)
        if self.reader_mlp is not None:
            reader_features = reader_features + self.reader_mlp(reader_features)
        reader_features = self._apply_post_layer_condition(reader_features, layer_condition)
        reader_features = self._apply_hdelta_condition(reader_features, hidden_states, initial_hidden_states)
        if self.output_mode == "residual_full":
            if self.visual_full_head is None:
                raise RuntimeError("visual_full_head is not initialized")
            residual = self.visual_full_head(reader_features)
            if single_layer_id is not None:
                gates = self.gate[single_layer_id].to(device=device, dtype=residual.dtype).view(1, 1, 1)
            else:
                assert layer_idx_tensor is not None
                gates = self.gate[layer_idx_tensor].view(batch, 1, 1).to(dtype=residual.dtype)
            residual = residual * gates
            if return_state and return_coefficients:
                return residual, next_state, residual.new_empty((*residual.shape[:2], 0))
            if return_state:
                return residual, next_state
            if return_coefficients:
                return residual, residual.new_empty((*residual.shape[:2], 0))
            return residual
        coeff: Tensor | None = None
        basis: Tensor | None = None
        if single_layer_id is not None:
            gates = self.gate[single_layer_id].to(device=device, dtype=reader_features.dtype).view(1, 1, 1)
        else:
            assert layer_idx_tensor is not None
            gates = self.gate[layer_idx_tensor].view(batch, 1, 1).to(device=device, dtype=reader_features.dtype)
        if self.uses_lowrank_output:
            if self.coeff_head is None:
                raise RuntimeError("coeff_head is not initialized")
            coeff = self.coeff_head(reader_features)
            if self.layer_adapter_rank > 0:
                if self.layer_adapter_down is None or self.layer_adapter_up is None:
                    raise RuntimeError("layer adapters are not initialized")
                if (
                    self.output_mode == "residual"
                    and self.runtime_fold_output_basis
                    and self.corrector_down is None
                    and not return_coefficients
                    and single_layer_id is None
                    and self._cached_coeff_basis is not None
                    and self._cached_adapter_down is not None
                    and self._cached_adapter_basis is not None
                ):
                    assert layer_idx_tensor is not None
                    residual = torch.matmul(reader_features, self._cached_coeff_basis)
                    down_weight = self._cached_adapter_down[layer_idx_tensor]
                    adapter_basis = self._cached_adapter_basis[layer_idx_tensor]
                    adapted = torch.einsum("btd,bad->bta", reader_features, down_weight)
                    adapted = torch.nn.functional.gelu(adapted)
                    residual = residual + torch.einsum("bta,bah->bth", adapted, adapter_basis)
                    gates = self.gate[layer_idx_tensor].view(batch, 1, 1).to(device=device, dtype=residual.dtype)
                    residual = residual * gates
                    if return_state:
                        return residual, next_state
                    return residual
                if single_layer_id is None and self._cached_adapter_down is not None and self._cached_adapter_up is not None:
                    assert layer_idx_tensor is not None
                    down_weight = self._cached_adapter_down[layer_idx_tensor]
                    up_weight = self._cached_adapter_up[layer_idx_tensor]
                    adapted = torch.einsum("btd,bad->bta", reader_features, down_weight)
                    adapted = torch.nn.functional.gelu(adapted)
                    coeff = coeff + torch.einsum("bta,bra->btr", adapted, up_weight)
                elif single_layer_id is not None:
                    adapted = self.layer_adapter_down[single_layer_id](reader_features)
                    coeff = coeff + self.layer_adapter_up[single_layer_id](torch.nn.functional.gelu(adapted))
                elif bool((layer_idx_tensor == layer_idx_tensor[0]).all()):
                    layer_id = int(layer_idx_tensor[0].item())
                    adapted = self.layer_adapter_down[layer_id](reader_features)
                    coeff = coeff + self.layer_adapter_up[layer_id](torch.nn.functional.gelu(adapted))
                else:
                    adapter_coeff = torch.zeros_like(coeff)
                    for idx in layer_idx_tensor.unique().tolist():
                        layer_id = int(idx)
                        mask = layer_idx_tensor == layer_id
                        adapted = self.layer_adapter_down[layer_id](reader_features[mask])
                        adapted = torch.nn.functional.gelu(adapted)
                        adapter_coeff[mask] = self.layer_adapter_up[layer_id](adapted)
                    coeff = coeff + adapter_coeff
            coeff = self._apply_coeff_corrector(coeff, reader_features, single_layer_id, layer_idx_tensor)
            if single_layer_id is not None:
                basis_source = self._inference_basis(device, coeff.dtype)
                if basis_source is None:
                    basis_source = self.basis
                if basis_source is None:
                    raise RuntimeError("low-rank basis is not initialized")
                if self.shared_basis:
                    basis = basis_source
                else:
                    basis = basis_source[single_layer_id].unsqueeze(0)
                if basis.shape[0] == 1 and batch != 1 and basis.requires_grad:
                    basis = basis.expand(batch, -1, -1)
                if basis.device != device or basis.dtype != coeff.dtype:
                    basis = basis.to(device=device, dtype=coeff.dtype)
                if self.normalize_basis_rows and basis_source is self.basis:
                    basis = F.normalize(basis.float(), p=2, dim=-1, eps=1e-6).to(dtype=coeff.dtype)
            else:
                assert layer_idx_tensor is not None
                basis = self.layer_basis(layer_idx_tensor, device, coeff.dtype)
        if self.output_mode == "factorized_full":
            if self.visual_full_head is None:
                raise RuntimeError("visual_full_head is not initialized")
            residual = self.visual_full_head(reader_features)
        elif self.output_mode == "factorized_native_o":
            residual = output_projection(text_attn_out)
        elif self.output_mode in HEADWISE_NATIVE_DELTA_MODES:
            if text_attention_heads is None:
                raise RuntimeError("text_attention_heads unexpectedly missing")
            if text_attention_heads.shape != text_attn_heads_out.shape:
                raise ValueError(
                    f"text_attention_heads shape {tuple(text_attention_heads.shape)} != "
                    f"visual attention heads shape {tuple(text_attn_heads_out.shape)}"
                )
            if visual_mass is None:
                if self.mass_head is None:
                    raise RuntimeError("mass_head is not initialized")
                visual_mass_heads = torch.sigmoid(self.mass_head(reader_features)).to(dtype=text_attn_heads_out.dtype).unsqueeze(-1)
            else:
                visual_mass_heads = visual_mass.to(device=text_attn_heads_out.device, dtype=text_attn_heads_out.dtype)
                if visual_mass_heads.ndim == 0:
                    visual_mass_heads = visual_mass_heads.view(1, 1, 1, 1)
                elif visual_mass_heads.ndim == 3:
                    visual_mass_heads = visual_mass_heads.unsqueeze(-1)
                if visual_mass_heads.ndim != 4:
                    raise ValueError("headwise visual_mass must be scalar, [batch, text_len, heads], or [batch, text_len, heads, 1]")
                if (
                    visual_mass_heads.shape[0] not in (1, text_attn_heads_out.shape[0])
                    or visual_mass_heads.shape[1] not in (1, text_attn_heads_out.shape[1])
                    or visual_mass_heads.shape[2] not in (1, text_attn_heads_out.shape[2])
                ):
                    raise ValueError(
                        f"visual_mass shape {tuple(visual_mass.shape)} is not broadcastable to {tuple(text_attn_heads_out.shape)}"
                    )
            self.last_visual_mass = visual_mass_heads.squeeze(-1)
            residual_heads = visual_mass_heads * (
                text_attn_heads_out - text_attention_heads.to(dtype=text_attn_heads_out.dtype)
            )
            residual = output_projection(merge_heads(residual_heads))
            if self.output_mode == "factorized_native_head_o_residual":
                if coeff is None or basis is None:
                    raise RuntimeError("low-rank residual path is not initialized")
                residual = residual + reconstruct_delta(coeff, basis).to(dtype=residual.dtype)
        else:
            if coeff is None or basis is None:
                raise RuntimeError("low-rank residual path is not initialized")
            residual = reconstruct_delta(coeff, basis)
        if self.output_mode in HEADWISE_NATIVE_DELTA_MODES:
            pass
        elif self.output_mode != "residual":
            if text_attention is None:
                raise RuntimeError("text_attention unexpectedly missing")
            if text_attention.shape != residual.shape:
                raise ValueError(f"text_attention shape {tuple(text_attention.shape)} != residual shape {tuple(residual.shape)}")
            if visual_mass is None:
                if self.mass_head is None:
                    raise RuntimeError("mass_head is not initialized")
                visual_mass = torch.sigmoid(self.mass_head(reader_features)).to(dtype=residual.dtype)
            else:
                visual_mass = visual_mass.to(device=residual.device, dtype=residual.dtype)
                if visual_mass.ndim == 0:
                    visual_mass = visual_mass.view(1, 1, 1)
                if visual_mass.ndim != 3:
                    raise ValueError("visual_mass must be scalar or have shape [batch, text_len, 1]")
                if visual_mass.shape[0] not in (1, residual.shape[0]) or visual_mass.shape[1] not in (1, residual.shape[1]):
                    raise ValueError(
                        f"visual_mass shape {tuple(visual_mass.shape)} is not broadcastable to {tuple(residual.shape)}"
                    )
            residual = visual_mass * (residual - text_attention.to(dtype=residual.dtype))
        residual = self._apply_block_corrector(residual, reader_features, single_layer_id, layer_idx_tensor)
        residual = residual * gates

        if return_state and return_coefficients:
            if coeff is None:
                coeff = residual.new_empty((*residual.shape[:2], 0))
            return residual, next_state, coeff
        if return_state:
            return residual, next_state
        if return_coefficients:
            if coeff is None:
                coeff = residual.new_empty((*residual.shape[:2], 0))
            return residual, coeff
        return residual


# Backward-compatible name used by older scripts/checkpoints.
BasisCoefficientSidecar = DeltaVisionModule
