"""Vision KV Adapter and embedding adapter implementations."""
from __future__ import annotations

import hashlib
import gc
import json
import io
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers import initialization as hf_initialization
from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import load_state_dict, local_torch_dtype, set_zero3_state
from transformers.monkey_patching import apply_patches, patch_output_recorders
from transformers.utils import ContextManagers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb, repeat_kv as llama_repeat_kv
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as qwen_apply_rotary_pos_emb, repeat_kv as qwen_repeat_kv
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


class PerLayerKVAdapter(nn.Module):
    """Per-LLM-layer adapter that maps vision encoder KV to LLM KV space.

    For each of the 32 LLM layers:
      - Learns a soft mixture over source_layers (last 2 ViT layers)
      - Projects mixed source K and V to LLM dim via independent linear layers
      - gate scalar controls adapter strength (init sigmoid(-5) for gradual warmup)
    """

    def __init__(
        self,
        num_llm_layers: int = 32,
        num_source_layers: int = 2,
        source_dim: int = 1024,
        num_heads: int = 32,
        head_dim: int = 128,
        bottleneck_dim: int = 0,
        concat_source: bool = False,
        use_activation: bool = False,
        expansion_dim: int = 0,
    ):
        super().__init__()
        self.num_llm_layers = num_llm_layers
        self.num_source_layers = num_source_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bottleneck_dim = bottleneck_dim
        self.expansion_dim = expansion_dim
        self.concat_source = concat_source
        self.use_activation = use_activation
        target_dim = num_heads * head_dim

        # Input dim depends on concat vs mix mode
        input_dim = source_dim * num_source_layers if concat_source else source_dim

        self.source_mix = nn.Parameter(torch.zeros(num_llm_layers, num_source_layers))

        if expansion_dim > 0:
            # GLU MLP: Linear -> gate * up -> Linear (SwiGLU style)
            self.k_gate = nn.ModuleList([nn.Linear(input_dim, expansion_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_up_mlp = nn.ModuleList([nn.Linear(input_dim, expansion_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_down_mlp = nn.ModuleList([nn.Linear(expansion_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_gate = nn.ModuleList([nn.Linear(input_dim, expansion_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_up_mlp = nn.ModuleList([nn.Linear(input_dim, expansion_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_down_mlp = nn.ModuleList([nn.Linear(expansion_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_projs = None
            self.v_projs = None
            self.k_down = None
            self.k_up = None
            self.v_down = None
            self.v_up = None
        elif bottleneck_dim > 0:
            self.k_down = nn.ModuleList([nn.Linear(input_dim, bottleneck_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_up = nn.ModuleList([nn.Linear(bottleneck_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_down = nn.ModuleList([nn.Linear(input_dim, bottleneck_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_up = nn.ModuleList([nn.Linear(bottleneck_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_projs = None
            self.v_projs = None
        else:
            self.k_projs = nn.ModuleList([nn.Linear(input_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_projs = nn.ModuleList([nn.Linear(input_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_down = None
            self.k_up = None
            self.v_down = None
            self.v_up = None

        self.gates = nn.Parameter(torch.full((num_llm_layers,), -5.0))
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, "k_gate") and self.k_gate is not None:
            for gate, up in zip(self.k_gate, self.k_up_mlp):
                nn.init.xavier_normal_(gate.weight, gain=1.0)
                nn.init.zeros_(gate.bias)
                nn.init.xavier_normal_(up.weight, gain=1.0)
                nn.init.zeros_(up.bias)
            for gate, up in zip(self.v_gate, self.v_up_mlp):
                nn.init.xavier_normal_(gate.weight, gain=1.0)
                nn.init.zeros_(gate.bias)
                nn.init.xavier_normal_(up.weight, gain=1.0)
                nn.init.zeros_(up.bias)
            for down in list(self.k_down_mlp) + list(self.v_down_mlp):
                nn.init.xavier_normal_(down.weight, gain=0.1)
                nn.init.zeros_(down.bias)
        elif self.k_projs is not None:
            for proj in list(self.k_projs) + list(self.v_projs):
                nn.init.xavier_normal_(proj.weight, gain=0.1)
                nn.init.zeros_(proj.bias)
        else:
            for proj in list(self.k_down) + list(self.v_down):
                nn.init.xavier_normal_(proj.weight, gain=1.0)
                nn.init.zeros_(proj.bias)
            for proj in list(self.k_up) + list(self.v_up):
                nn.init.xavier_normal_(proj.weight, gain=0.1)
                nn.init.zeros_(proj.bias)

    def forward_layer(
        self,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce visual K,V for one LLM layer.

        Args:
            source_k: [B, num_source_layers, N_vis, source_dim]
            source_v: [B, num_source_layers, N_vis, source_dim]
            layer_idx: which LLM layer

        Returns:
            key: [B, N_vis, num_heads, head_dim]
            value: [B, N_vis, num_heads, head_dim]
        """
        return self.forward_layer_from_modules(
            source_k,
            source_v,
            source_mix=self.source_mix[layer_idx],
            gate_logit=self.gates[layer_idx],
            k_proj=None if self.k_projs is None else self.k_projs[layer_idx],
            v_proj=None if self.v_projs is None else self.v_projs[layer_idx],
            k_down=None if self.k_down is None else self.k_down[layer_idx],
            k_up=None if self.k_up is None else self.k_up[layer_idx],
            v_down=None if self.v_down is None else self.v_down[layer_idx],
            v_up=None if self.v_up is None else self.v_up[layer_idx],
            k_gate=None if getattr(self, "k_gate", None) is None else self.k_gate[layer_idx],
            k_up_mlp=None if getattr(self, "k_up_mlp", None) is None else self.k_up_mlp[layer_idx],
            k_down_mlp=None if getattr(self, "k_down_mlp", None) is None else self.k_down_mlp[layer_idx],
            v_gate=None if getattr(self, "v_gate", None) is None else self.v_gate[layer_idx],
            v_up_mlp=None if getattr(self, "v_up_mlp", None) is None else self.v_up_mlp[layer_idx],
            v_down_mlp=None if getattr(self, "v_down_mlp", None) is None else self.v_down_mlp[layer_idx],
        )

    def forward_layer_from_modules(
        self,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        *,
        source_mix: torch.Tensor,
        gate_logit: torch.Tensor,
        k_proj: nn.Module | None,
        v_proj: nn.Module | None,
        k_down: nn.Module | None,
        k_up: nn.Module | None,
        v_down: nn.Module | None,
        v_up: nn.Module | None,
        k_gate: nn.Module | None,
        k_up_mlp: nn.Module | None,
        k_down_mlp: nn.Module | None,
        v_gate: nn.Module | None,
        v_up_mlp: nn.Module | None,
        v_down_mlp: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.concat_source:
            # Concat mode: [B, num_source, N_vis, D] -> [B, N_vis, num_source*D]
            B, S, N, D = source_k.shape
            mixed_k = source_k.permute(0, 2, 1, 3).reshape(B, N, S * D)
            mixed_v = source_v.permute(0, 2, 1, 3).reshape(B, N, S * D)
        else:
            # Weighted sum mode
            weights = F.softmax(source_mix.float(), dim=-1)
            weights = weights.to(source_k.dtype)
            mixed_k = torch.einsum("s,bsnd->bnd", weights, source_k)
            mixed_v = torch.einsum("s,bsnd->bnd", weights, source_v)

        B, N, _ = mixed_k.shape
        gate = torch.sigmoid(gate_logit)

        if k_proj is not None:
            if v_proj is None:
                raise RuntimeError("v_proj is required when k_proj is set")
            key = k_proj(mixed_k)
            value = v_proj(mixed_v)
            if self.use_activation:
                key = F.silu(key)
                value = F.silu(value)
        elif k_down_mlp is not None:
            # SwiGLU: silu(gate(x)) * up(x), then down proj
            if k_gate is None or k_up_mlp is None or v_gate is None or v_up_mlp is None or v_down_mlp is None:
                raise RuntimeError("SwiGLU adapter modules are incomplete")
            key = k_down_mlp(F.silu(k_gate(mixed_k)) * k_up_mlp(mixed_k))
            value = v_down_mlp(F.silu(v_gate(mixed_v)) * v_up_mlp(mixed_v))
        else:
            if k_down is None or k_up is None or v_down is None or v_up is None:
                raise RuntimeError("bottleneck adapter modules are incomplete")
            key = k_up(F.silu(k_down(mixed_k)))
            value = v_up(F.silu(v_down(mixed_v)))

        key = key.view(B, N, self.num_heads, self.head_dim) * gate
        value = value.view(B, N, self.num_heads, self.head_dim) * gate

        return key, value


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def infer_adapter_config_from_checkpoint(
    ckpt: dict,
    language_model: nn.Module | None = None,
) -> tuple[dict, list[int]]:
    """Infer adapter constructor kwargs from an old or new checkpoint."""
    state_dict = ckpt["state_dict"]
    ckpt_args = ckpt.get("args", {})
    saved_config = ckpt.get("adapter_config", {})

    if "source_mix" not in state_dict:
        raise ValueError("Adapter checkpoint is missing source_mix; cannot infer layer count")
    num_llm_layers, num_source_layers = state_dict["source_mix"].shape

    source_layers_raw = saved_config.get("source_layers", ckpt_args.get("source_layers"))
    if source_layers_raw is None:
        source_layers = [22, 23] if num_source_layers == 2 else list(range(num_source_layers))
    elif isinstance(source_layers_raw, str):
        source_layers = [int(x) for x in source_layers_raw.split(",") if x]
    else:
        source_layers = [int(x) for x in source_layers_raw]
    if len(source_layers) != num_source_layers:
        source_layers = [22, 23] if num_source_layers == 2 else list(range(num_source_layers))

    bottleneck_dim = int(saved_config.get("bottleneck_dim", ckpt_args.get("bottleneck_dim", 0)))
    concat_source = _as_bool(saved_config.get("concat_source", ckpt_args.get("concat_source", False)))
    use_activation = _as_bool(saved_config.get("use_activation", ckpt_args.get("use_activation", False)))

    expansion_dim = int(saved_config.get("expansion_dim", ckpt_args.get("expansion_dim", 0)))
    if "k_gate.0.weight" in state_dict and "k_up_mlp.0.weight" in state_dict:
        expansion_dim = state_dict["k_gate.0.weight"].shape[0]
        input_dim = state_dict["k_gate.0.weight"].shape[1]
        target_dim = state_dict["k_down_mlp.0.weight"].shape[0]
        bottleneck_dim = 0
    elif "k_projs.0.weight" in state_dict:
        target_dim, input_dim = state_dict["k_projs.0.weight"].shape
        bottleneck_dim = 0
    elif "k_down.0.weight" in state_dict and "k_up.0.weight" in state_dict:
        input_dim = state_dict["k_down.0.weight"].shape[1]
        bottleneck_dim = state_dict["k_down.0.weight"].shape[0]
        target_dim = state_dict["k_up.0.weight"].shape[0]
    else:
        raise ValueError("Adapter checkpoint has no recognizable projection weights")

    source_dim = saved_config.get("source_dim", ckpt_args.get("source_dim"))
    if source_dim is None:
        source_dim = input_dim // num_source_layers if concat_source else input_dim
    source_dim = int(source_dim)

    head_dim = saved_config.get("head_dim", ckpt_args.get("head_dim"))
    num_heads = saved_config.get("num_heads", ckpt_args.get("num_heads"))
    if language_model is not None:
        first_attn = language_model.layers[0].self_attn
        cfg_head_dim = int(
            getattr(
                first_attn,
                "head_dim",
                language_model.config.hidden_size // language_model.config.num_attention_heads,
            )
        )
        if target_dim % cfg_head_dim == 0:
            head_dim = cfg_head_dim
            num_heads = target_dim // cfg_head_dim

    if head_dim is None:
        head_dim = 128 if target_dim % 128 == 0 else target_dim
    head_dim = int(head_dim)
    if num_heads is None:
        if target_dim % head_dim != 0:
            raise ValueError(f"Cannot infer num_heads: target_dim={target_dim}, head_dim={head_dim}")
        num_heads = target_dim // head_dim
    num_heads = int(num_heads)

    config = {
        "num_llm_layers": int(num_llm_layers),
        "num_source_layers": int(num_source_layers),
        "source_dim": source_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "bottleneck_dim": int(bottleneck_dim),
        "expansion_dim": int(expansion_dim),
        "concat_source": concat_source,
        "use_activation": use_activation,
    }
    return config, source_layers


def load_adapter_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    language_model: nn.Module | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, list[int], dict]:
    from src.model_setup import create_qwen_adapter, create_llava_kv_adapter
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    saved_config = ckpt.get("adapter_config", {}) if isinstance(ckpt, dict) else {}
    output_mode = canonical_adapter_mode(saved_config.get("output_mode") or ckpt_args.get("output_mode") or "kv_adapter")
    if is_embedding_adapter_mode(output_mode):
        if language_model is None:
            raise ValueError("language_model is required to load an embedding_adapter checkpoint")
        state_dict = ckpt["state_dict"]
        first_down = state_dict.get("visual_adapter_down.0.weight")
        visual_adapter_rank = int(
            saved_config.get(
                "visual_adapter_rank",
                ckpt_args.get("visual_adapter_rank", first_down.shape[0] if first_down is not None else 128),
            )
        )
        adapter = create_qwen_adapter(
            language_model, mode=output_mode, rank=visual_adapter_rank,
        )
        missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
        adapter.to(device=device, dtype=dtype)
        adapter.eval()
        metadata = {
            "output_mode": output_mode,
            "visual_adapter_rank": visual_adapter_rank,
            "source_layers": saved_config.get("source_layers", [22, 23]),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
        return adapter, list(metadata["source_layers"]), metadata

    config, source_layers = infer_adapter_config_from_checkpoint(ckpt, language_model=language_model)
    adapter = create_llava_kv_adapter(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    metadata = {
        **config,
        **{
            key: value
            for key, value in ckpt.get("adapter_config", {}).items()
            if key not in config
        },
    }
    metadata["output_mode"] = output_mode
    adapter.output_mode = metadata["output_mode"]
    return adapter, source_layers, metadata


def _get_vision_tower(model: LlavaForConditionalGeneration):
    return model.model.vision_tower


def _get_language_model(model: LlavaForConditionalGeneration):
    return model.model.language_model


def _normalize_layer_indices(layer_indices: list[int], num_layers: int) -> list[int]:
    """Normalize negative layer indices while preserving user-specified order."""
    normalized = []
    seen = set()
    for layer_idx in layer_indices:
        idx = (layer_idx % num_layers) if layer_idx < 0 else layer_idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} resolves to {idx}, outside [0, {num_layers})")
        if idx in seen:
            raise ValueError(f"Duplicate source layer {layer_idx} resolves to {idx}")
        normalized.append(idx)
        seen.add(idx)
    return normalized


@torch.no_grad()
def extract_vision_kv(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
    source_layer_indices: list[int] = [22, 23],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract K,V projections from the last N layers of CLIP ViT.

    Returns:
        source_k: [B, num_source_layers, N_vis, hidden_dim]
        source_v: [B, num_source_layers, N_vis, hidden_dim]
    """
    vision_tower = _get_vision_tower(model)
    vision_model = vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower

    # Handle multi-crop (LLaVA-1.6): [B, num_crops, C, H, W] -> [B*num_crops, C, H, W]
    if pixel_values.ndim == 5:
        B_orig, num_crops = pixel_values.shape[:2]
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
    else:
        B_orig, num_crops = pixel_values.shape[0], 1

    hidden = vision_model.embeddings(pixel_values)
    hidden = vision_model.pre_layrnorm(hidden)

    layers = vision_model.encoder.layers
    num_layers = len(layers)
    wanted = _normalize_layer_indices(source_layer_indices, num_layers)
    wanted_set = set(wanted)

    collected_k = {}
    collected_v = {}

    for idx, layer in enumerate(layers):
        normed = layer.layer_norm1(hidden)
        if idx in wanted_set:
            collected_k[idx] = layer.self_attn.k_proj(normed)[:, 1:].float()
            collected_v[idx] = layer.self_attn.v_proj(normed)[:, 1:].float()
        layer_out = layer(hidden, attention_mask=None)
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    source_k = torch.stack([collected_k[i] for i in wanted], dim=1)
    source_v = torch.stack([collected_v[i] for i in wanted], dim=1)

    # For multi-crop: merge crops into token dimension [B, S, num_crops*N_vis, D]
    if num_crops > 1:
        B_total, S, N_vis, D = source_k.shape
        source_k = source_k.view(B_orig, num_crops, S, N_vis, D).permute(0, 2, 1, 3, 4).reshape(B_orig, S, num_crops * N_vis, D)
        source_v = source_v.view(B_orig, num_crops, S, N_vis, D).permute(0, 2, 1, 3, 4).reshape(B_orig, S, num_crops * N_vis, D)

    return source_k, source_v


def _apply_rope(
    rotary_emb: nn.Module,
    states: torch.Tensor,
    position_ids: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to states."""
    cos, sin = rotary_emb(reference, position_ids)
    rotated, _ = llama_apply_rotary_pos_emb(states, states, cos, sin)
    return rotated


@torch.compiler.disable
def _eager_llama_apply_rope_pair_from_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return llama_apply_rotary_pos_emb(query, key, *position_embeddings)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compiler.disable
def _eager_apply_rope_one_from_embeddings(
    states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> Tensor:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated = states * cos
    rotated.add_(_rotate_half(states) * sin)
    return rotated


def _apply_rope_one_from_embeddings(
    states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> Tensor:
    if torch.compiler.is_compiling():
        return _eager_apply_rope_one_from_embeddings(states, position_embeddings)
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated = states * cos
    rotated.add_(_rotate_half(states) * sin)
    return rotated


def _apply_rope_pair_from_embeddings(
    query: Tensor,
    key: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    if torch.compiler.is_compiling():
        return _eager_llama_apply_rope_pair_from_embeddings(query, key, position_embeddings)
    return llama_apply_rotary_pos_emb(query, key, *position_embeddings)


@torch.compiler.disable
def _eager_qwen_apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    return qwen_apply_rotary_pos_emb(query, key, cos, sin)


def _compile_exact_qwen_apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    if torch.compiler.is_compiling():
        return _eager_qwen_apply_rotary_pos_emb(query, key, *position_embeddings)
    return qwen_apply_rotary_pos_emb(query, key, *position_embeddings)


@torch.compiler.disable
def _eager_module_call(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module(*args, **kwargs)


def _compile_exact_module_call(module: Any, *args: Any, **kwargs: Any) -> Any:
    if torch.compiler.is_compiling():
        return _eager_module_call(module, *args, **kwargs)
    return module(*args, **kwargs)


@torch.compiler.disable
def _eager_prefix_causal_attention_heads(
    query: Tensor,
    visual_key: Tensor,
    visual_value: Tensor,
    text_key: Tensor,
    text_value: Tensor,
    *,
    scaling: float | None,
    attention_mask: Tensor,
) -> Tensor:
    key = torch.cat([visual_key, text_key], dim=2)
    value = torch.cat([visual_value, text_value], dim=2)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    ).transpose(1, 2).contiguous()


def _prefix_causal_attention_heads(
    query: Tensor,
    visual_key: Tensor,
    visual_value: Tensor,
    text_key: Tensor,
    text_value: Tensor,
    *,
    scaling: float | None,
    attention_mask: Tensor,
    attention_plan: dict[str, Any] | None = None,
) -> Tensor:
    if attention_plan is not None:
        from src.attention import attention_heads
        return attention_heads(query, torch.cat([visual_key, text_key], dim=2),
            torch.cat([visual_value, text_value], dim=2), scaling=scaling, plan=attention_plan)
    if torch.compiler.is_compiling():
        return _eager_prefix_causal_attention_heads(
            query,
            visual_key,
            visual_value,
            text_key,
            text_value,
            scaling=scaling,
            attention_mask=attention_mask,
        )
    key = torch.cat([visual_key, text_key], dim=2)
    value = torch.cat([visual_value, text_value], dim=2)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    ).transpose(1, 2).contiguous()


def _hf_sdpa_prefix_causal_attention_heads(
    module: nn.Module,
    query: Tensor,
    visual_key: Tensor,
    visual_value: Tensor,
    text_key: Tensor,
    text_value: Tensor,
    *,
    scaling: float | None,
    attention_mask: Tensor,
) -> Tensor:
    key = torch.cat([visual_key, text_key], dim=2)
    value = torch.cat([visual_value, text_value], dim=2)
    attn_output, _ = hf_sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=scaling,
        is_causal=False,
    )
    return attn_output.contiguous()


def prepare_llava_kv_adapter_inputs(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    valid_mask = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(input_ids[0], dtype=torch.bool)
    text_mask = (input_ids[0] != image_token_id) & valid_mask
    image_mask = (input_ids[0] == image_token_id) & valid_mask
    text_ids = input_ids[:, text_mask]

    language_model = _get_language_model(model)
    text_embeds = language_model.embed_tokens(text_ids)

    layers = language_model.layers
    norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    B, T, _ = text_embeds.shape
    N_vis = source_k.shape[2]
    device = text_embeds.device
    dtype = text_embeds.dtype
    source_k = source_k.to(device=device, dtype=dtype)
    source_v = source_v.to(device=device, dtype=dtype)

    text_positions = torch.where(text_mask)[0].to(device)
    image_positions = torch.where(image_mask)[0].to(device)
    text_position_ids = text_positions.unsqueeze(0).expand(B, -1)
    if image_positions.numel() == N_vis:
        image_position_ids = image_positions.unsqueeze(0).expand(B, -1)
    elif image_positions.numel() > 0:
        start_pos = int(image_positions[0].item())
        image_position_ids = torch.arange(start_pos, start_pos + N_vis, device=device).unsqueeze(0).expand(B, -1)
    else:
        image_position_ids = torch.arange(N_vis, device=device).unsqueeze(0).expand(B, -1)

    text_position_embeddings = rotary_emb(text_embeds, text_position_ids)
    image_position_embeddings = rotary_emb(text_embeds, image_position_ids)
    attn_mask = _llava_prefix_attention_mask(text_position_ids, image_position_ids, dtype=dtype)
    return {
        "hidden": text_embeds,
        "source_k": source_k,
        "source_v": source_v,
        "text_position_ids": text_position_ids,
        "image_position_ids": image_position_ids,
        "text_position_embeddings": text_position_embeddings,
        "image_position_embeddings": image_position_embeddings,
        "attn_mask": attn_mask,
    }


def student_forward_with_visual_kv_prepared(
    model: LlavaForConditionalGeneration,
    adapter: PerLayerKVAdapter,
    *,
    hidden: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    text_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    image_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_mask: torch.Tensor,
    text_position_ids: torch.Tensor | None = None,
    image_position_ids: torch.Tensor | None = None,
    use_hf_attention: bool = False,
) -> torch.Tensor:
    language_model = _get_language_model(model)
    layers = language_model.layers
    norm = language_model.norm
    adapter_forward_layer = adapter.forward_layer_from_modules
    compile_exact = torch.compiler.is_compiling()

    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = _eager_module_call(layer.input_layernorm, hidden) if compile_exact else layer.input_layernorm(hidden)
        attn = layer.self_attn

        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_k = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        q, text_k = _apply_rope_pair_from_embeddings(q, text_k, text_position_embeddings)

        vis_k, vis_v = adapter_forward_layer(
            source_k,
            source_v,
            source_mix=adapter.source_mix[layer_idx],
            gate_logit=adapter.gates[layer_idx],
            k_proj=None if adapter.k_projs is None else adapter.k_projs[layer_idx],
            v_proj=None if adapter.v_projs is None else adapter.v_projs[layer_idx],
            k_down=None if adapter.k_down is None else adapter.k_down[layer_idx],
            k_up=None if adapter.k_up is None else adapter.k_up[layer_idx],
            v_down=None if adapter.v_down is None else adapter.v_down[layer_idx],
            v_up=None if adapter.v_up is None else adapter.v_up[layer_idx],
            k_gate=None if getattr(adapter, "k_gate", None) is None else adapter.k_gate[layer_idx],
            k_up_mlp=None if getattr(adapter, "k_up_mlp", None) is None else adapter.k_up_mlp[layer_idx],
            k_down_mlp=None if getattr(adapter, "k_down_mlp", None) is None else adapter.k_down_mlp[layer_idx],
            v_gate=None if getattr(adapter, "v_gate", None) is None else adapter.v_gate[layer_idx],
            v_up_mlp=None if getattr(adapter, "v_up_mlp", None) is None else adapter.v_up_mlp[layer_idx],
            v_down_mlp=None if getattr(adapter, "v_down_mlp", None) is None else adapter.v_down_mlp[layer_idx],
        )
        vis_k = vis_k.transpose(1, 2)
        vis_v = vis_v.transpose(1, 2)
        vis_k = _apply_rope_one_from_embeddings(vis_k, image_position_embeddings)

        attn_out = (
            _hf_sdpa_prefix_causal_attention_heads(
                attn,
                q,
                vis_k,
                vis_v,
                text_k,
                text_v,
                attention_mask=attn_mask,
                scaling=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
            )
            if use_hf_attention
            else _prefix_causal_attention_heads(
                q,
                vis_k,
                vis_v,
                text_k,
                text_v,
                attention_mask=attn_mask,
                scaling=None,
            )
        )
        attn_out = attn.o_proj(attn_out.reshape(*input_shape, -1))

        hidden = residual + attn_out
        residual = hidden
        post_normed = _eager_module_call(layer.post_attention_layernorm, hidden) if compile_exact else layer.post_attention_layernorm(hidden)
        hidden = residual + (_eager_module_call(layer.mlp, post_normed) if compile_exact else layer.mlp(post_normed))

    hidden = _eager_module_call(norm, hidden) if compile_exact else norm(hidden)
    logits = model.lm_head(hidden)
    return logits


def student_forward_with_visual_kv_prepared_hf_attention(
    model: LlavaForConditionalGeneration,
    adapter: PerLayerKVAdapter,
    **prepared: Any,
) -> torch.Tensor:
    return student_forward_with_visual_kv_prepared(model, adapter, **prepared, use_hf_attention=True)


def _llava_prefix_attention_mask(
    text_position_ids: torch.Tensor,
    image_position_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    img_allowed = text_position_ids.unsqueeze(2) >= image_position_ids.unsqueeze(1)
    text_allowed = text_position_ids.unsqueeze(2) >= text_position_ids.unsqueeze(1)
    prefix_mask = torch.cat([img_allowed, text_allowed], dim=-1)
    return prefix_mask.unsqueeze(1).contiguous()


def llava_kv_adapter_prefill_cache_prepared(
    model: LlavaForConditionalGeneration,
    adapter: PerLayerKVAdapter,
    *,
    hidden: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_position_ids: torch.Tensor,
    text_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    image_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    language_model = _get_language_model(model)
    layers = language_model.layers
    layer_caches: list[dict[str, torch.Tensor]] = []
    layer_inputs: list[torch.Tensor] = []
    layer_after_attention: list[torch.Tensor] = []

    for layer_idx, layer in enumerate(layers):
        layer_inputs.append(hidden)
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        query, text_key = _apply_rope_pair_from_embeddings(query, text_key, text_position_embeddings)

        visual_key, visual_value = adapter.forward_layer(source_k, source_v, layer_idx)
        visual_key = visual_key.transpose(1, 2)
        visual_value = visual_value.transpose(1, 2)
        visual_key = _apply_rope_one_from_embeddings(visual_key, image_position_embeddings)
        heads = _prefix_causal_attention_heads(
            query,
            visual_key,
            visual_value,
            text_key,
            text_value,
            attention_mask=attn_mask,
            scaling=None,
        )
        layer_caches.append({"visual_key": visual_key.contiguous(), "visual_value": visual_value.contiguous()})
        attention_output = attn.o_proj(heads.reshape(*input_shape, -1).contiguous())
        hidden = residual + attention_output
        layer_after_attention.append(hidden)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    logits = model.lm_head(language_model.norm(hidden))
    cache = {
        "layers": layer_caches,
        "layer_inputs": layer_inputs,
        "layer_after_attention": layer_after_attention,
        "text_position_ids": text_position_ids.clone(),
        "image_position_ids": image_position_ids.clone(),
        "next_position_ids": text_position_ids[:, -1:].clone() + 1,
        "mode": "kv_adapter",
    }
    return logits, cache


def llava_kv_adapter_prefill_cache(
    model: LlavaForConditionalGeneration,
    adapter: PerLayerKVAdapter,
    input_ids: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prepared = prepare_llava_kv_adapter_inputs(
        model,
        input_ids,
        source_k,
        source_v,
        image_token_id,
        attention_mask=attention_mask,
    )
    return llava_kv_adapter_prefill_cache_prepared(model, adapter, **prepared)


def llava_kv_adapter_decode_step_shape_exact(
    model: LlavaForConditionalGeneration,
    adapter: PerLayerKVAdapter,
    token_ids: torch.Tensor,
    cache: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    language_model = _get_language_model(model)
    h = language_model.embed_tokens(token_ids)
    next_position_ids = cache["next_position_ids"]
    full_text_position_ids = torch.cat([cache["text_position_ids"], next_position_ids], dim=1)
    attention_mask = _llava_prefix_attention_mask(
        full_text_position_ids,
        cache["image_position_ids"],
        dtype=h.dtype,
    )

    for layer_idx, layer in enumerate(language_model.layers):
        full_layer_input = torch.cat([cache["layer_inputs"][layer_idx], h], dim=1)
        residual = full_layer_input
        normed = layer.input_layernorm(full_layer_input)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        position_embeddings = language_model.rotary_emb(normed, full_text_position_ids)
        query, text_key = _apply_rope_pair_from_embeddings(query, text_key, position_embeddings)
        layer_cache = cache["layers"][layer_idx]
        heads = _prefix_causal_attention_heads(
            query,
            layer_cache["visual_key"],
            layer_cache["visual_value"],
            text_key,
            text_value,
            attention_mask=attention_mask,
            scaling=None,
        )
        full_attention = attn.o_proj(heads.reshape(*input_shape, -1).contiguous())
        full_after_attention = residual + full_attention
        full_output = full_after_attention + layer.mlp(layer.post_attention_layernorm(full_after_attention))
        h = full_output[:, -1:]
        cache["layer_inputs"][layer_idx] = full_layer_input
        cache["layer_after_attention"][layer_idx] = full_after_attention

    cache["text_position_ids"] = full_text_position_ids
    cache["next_position_ids"] = next_position_ids + 1
    logits = model.lm_head(language_model.norm(h))
    return logits, cache


def student_forward_with_visual_kv(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    adapter: PerLayerKVAdapter,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run LLM forward with adapter-predicted visual KV instead of image token embeddings.

    Text tokens only go through the LLM; visual information enters via
    concatenated KV in each attention layer.

    Returns:
        logits: [B, text_seq_len, vocab_size]
    """
    prepared = prepare_llava_kv_adapter_inputs(
        model,
        input_ids,
        source_k,
        source_v,
        image_token_id,
        attention_mask=attention_mask,
    )
    return student_forward_with_visual_kv_prepared(model, adapter, **prepared)


@torch.no_grad()
def teacher_forward(
    model,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    image_sizes: list | None = None,
) -> torch.Tensor:
    """Standard LLaVA/LlavaNext forward to get teacher logits."""
    kwargs = dict(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
    if image_sizes is not None:
        kwargs["image_sizes"] = image_sizes
    outputs = model(**kwargs)
    return outputs.logits


def load_frozen_llava(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_implementation: str = "eager",
) -> tuple:
    """Load LLaVA model with all parameters frozen."""
    processor = AutoProcessor.from_pretrained(model_path)
    if attn_implementation == "auto":
        attn_implementation = "eager"
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **kwargs,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def llava_projected_image_features(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(model.model, "image_newline"):
        if image_sizes is None:
            raise ValueError("LLaVA-Next embedding_adapter requires image_sizes")
        image_outputs = model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
        features = image_outputs.pooler_output
        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise NotImplementedError("LLaVA-Next embedding_adapter currently expects one image per forward")
            return features[0].unsqueeze(0)
        return features.unsqueeze(0) if features.ndim == 2 else features
    if pixel_values.ndim != 4:
        raise NotImplementedError("LLaVA embedding_adapter only supports fixed-grid LLaVA-style image features")
    image_outputs = model.model.vision_tower(pixel_values, output_hidden_states=True)
    image_features = image_outputs.hidden_states[model.config.vision_feature_layer]
    image_features = image_features[:, 1:]
    return model.model.multi_modal_projector(image_features)


def prepare_llava_embedding_adapter_inputs(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
    visual_memory: torch.Tensor | None = None,
    image_sizes: torch.Tensor | None = None,
) -> dict[str, Any]:
    if visual_memory is None:
        visual_memory = llava_projected_image_features(model, pixel_values, image_sizes=image_sizes)

    language_model = _get_language_model(model)
    valid_mask = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(input_ids[0], dtype=torch.bool)
    text_mask = (input_ids[0] != image_token_id) & valid_mask
    image_mask = (input_ids[0] == image_token_id) & valid_mask
    text_ids = input_ids[:, text_mask]
    hidden = language_model.embed_tokens(text_ids)

    layers = language_model.layers
    rotary_emb = language_model.rotary_emb
    batch, text_len, _ = hidden.shape
    image_len = visual_memory.shape[1]
    device = hidden.device
    dtype = hidden.dtype
    visual_memory = visual_memory.to(device=device, dtype=dtype)

    text_positions = torch.where(text_mask)[0].to(device)
    image_positions = torch.where(image_mask)[0].to(device)
    text_position_ids = text_positions.unsqueeze(0).expand(batch, -1)
    if image_positions.numel() == image_len:
        image_position_ids = image_positions.unsqueeze(0).expand(batch, -1)
    elif image_positions.numel() > 0:
        start_pos = int(image_positions[0].item())
        image_position_ids = torch.arange(start_pos, start_pos + image_len, device=device).unsqueeze(0).expand(batch, -1)
    else:
        image_position_ids = torch.arange(image_len, device=device).unsqueeze(0).expand(batch, -1)

    attn_mask = _llava_prefix_attention_mask(text_position_ids, image_position_ids, dtype=dtype)
    text_position_embeddings = rotary_emb(hidden, text_position_ids)
    image_position_embeddings = rotary_emb(hidden, image_position_ids)
    return {
        "hidden": hidden,
        "visual_memory": visual_memory,
        "text_position_ids": text_position_ids,
        "image_position_ids": image_position_ids,
        "text_position_embeddings": text_position_embeddings,
        "image_position_embeddings": image_position_embeddings,
        "attn_mask": attn_mask,
    }


def student_forward_llava_embedding_adapter_prepared(
    model: LlavaForConditionalGeneration,
    adapter: "QwenEmbeddingAdapter",
    *,
    hidden: torch.Tensor,
    visual_memory: torch.Tensor,
    text_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    image_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_mask: torch.Tensor,
    text_position_ids: torch.Tensor | None = None,
    image_position_ids: torch.Tensor | None = None,
    use_hf_attention: bool = False,
) -> torch.Tensor:
    language_model = _get_language_model(model)
    layers = language_model.layers
    compile_exact = torch.compiler.is_compiling()
    all_vis_memories = adapter.all_visual_memories_batched(visual_memory)  # [L, B, N, H]

    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = _eager_module_call(layer.input_layernorm, hidden) if compile_exact else layer.input_layernorm(hidden)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        query, text_key = _apply_rope_pair_from_embeddings(query, text_key, text_position_embeddings)

        vision_states = all_vis_memories[layer_idx]
        normed_vision = _eager_module_call(layer.input_layernorm, vision_states) if compile_exact else layer.input_layernorm(vision_states)
        vision_shape = normed_vision.shape[:-1]
        vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
        visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_key = _apply_rope_one_from_embeddings(visual_key, image_position_embeddings)

        attn_out = (
            _hf_sdpa_prefix_causal_attention_heads(
                attn,
                query,
                visual_key,
                visual_value,
                text_key,
                text_value,
                attention_mask=attn_mask,
                scaling=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
            )
            if use_hf_attention
            else _prefix_causal_attention_heads(
                query,
                visual_key,
                visual_value,
                text_key,
                text_value,
                attention_mask=attn_mask,
                scaling=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
            )
        )
        hidden = residual + attn.o_proj(attn_out.reshape(*input_shape, -1))
        residual = hidden
        post_normed = _eager_module_call(layer.post_attention_layernorm, hidden) if compile_exact else layer.post_attention_layernorm(hidden)
        hidden = residual + (_eager_module_call(layer.mlp, post_normed) if compile_exact else layer.mlp(post_normed))

    hidden = _eager_module_call(language_model.norm, hidden) if compile_exact else language_model.norm(hidden)
    return model.lm_head(hidden)


def student_forward_llava_embedding_adapter_prepared_hf_attention(
    model: LlavaForConditionalGeneration,
    adapter: "QwenEmbeddingAdapter",
    **prepared: Any,
) -> torch.Tensor:
    return student_forward_llava_embedding_adapter_prepared(model, adapter, **prepared, use_hf_attention=True)


def llava_embedding_adapter_prefill_cache_prepared(
    model: LlavaForConditionalGeneration,
    adapter: "QwenEmbeddingAdapter",
    *,
    hidden: torch.Tensor,
    visual_memory: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_position_ids: torch.Tensor,
    text_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    image_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    language_model = _get_language_model(model)
    layer_caches: list[dict[str, torch.Tensor]] = []
    layer_inputs: list[torch.Tensor] = []
    layer_after_attention: list[torch.Tensor] = []
    layer_visual_memory = visual_memory

    for layer_idx, layer in enumerate(language_model.layers):
        layer_inputs.append(hidden)
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        query, text_key = _apply_rope_pair_from_embeddings(query, text_key, text_position_embeddings)

        vision_states = adapter.visual_memory_for_layer(layer_visual_memory, layer_idx)
        if adapter.mode == RECURRENT_EMBEDDING_ADAPTER_MODE:
            layer_visual_memory = vision_states
        normed_vision = layer.input_layernorm(vision_states)
        vision_shape = normed_vision.shape[:-1]
        vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
        visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_key = _apply_rope_one_from_embeddings(visual_key, image_position_embeddings)
        heads = _prefix_causal_attention_heads(
            query,
            visual_key,
            visual_value,
            text_key,
            text_value,
            attention_mask=attn_mask,
            scaling=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
        )
        layer_caches.append({"visual_key": visual_key.contiguous(), "visual_value": visual_value.contiguous()})
        attention_output = attn.o_proj(heads.reshape(*input_shape, -1).contiguous())
        hidden = residual + attention_output
        layer_after_attention.append(hidden)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    logits = model.lm_head(language_model.norm(hidden))
    cache = {
        "layers": layer_caches,
        "layer_inputs": layer_inputs,
        "layer_after_attention": layer_after_attention,
        "text_position_ids": text_position_ids.clone(),
        "image_position_ids": image_position_ids.clone(),
        "next_position_ids": text_position_ids[:, -1:].clone() + 1,
        "mode": adapter.mode,
    }
    return logits, cache


def llava_embedding_adapter_prefill_cache(
    model: LlavaForConditionalGeneration,
    adapter: "QwenEmbeddingAdapter",
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
    visual_memory: torch.Tensor | None = None,
    image_sizes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prepared = prepare_llava_embedding_adapter_inputs(
        model,
        input_ids,
        pixel_values,
        image_token_id,
        attention_mask=attention_mask,
        visual_memory=visual_memory,
        image_sizes=image_sizes,
    )
    return llava_embedding_adapter_prefill_cache_prepared(model, adapter, **prepared)


def llava_embedding_adapter_decode_step_shape_exact(
    model: LlavaForConditionalGeneration,
    adapter: "QwenEmbeddingAdapter",
    token_ids: torch.Tensor,
    cache: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    language_model = _get_language_model(model)
    h = language_model.embed_tokens(token_ids)
    next_position_ids = cache["next_position_ids"]
    full_text_position_ids = torch.cat([cache["text_position_ids"], next_position_ids], dim=1)
    attention_mask = _llava_prefix_attention_mask(
        full_text_position_ids,
        cache["image_position_ids"],
        dtype=h.dtype,
    )

    for layer_idx, layer in enumerate(language_model.layers):
        full_layer_input = torch.cat([cache["layer_inputs"][layer_idx], h], dim=1)
        residual = full_layer_input
        normed = layer.input_layernorm(full_layer_input)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        position_embeddings = language_model.rotary_emb(normed, full_text_position_ids)
        query, text_key = _apply_rope_pair_from_embeddings(query, text_key, position_embeddings)
        layer_cache = cache["layers"][layer_idx]
        heads = _prefix_causal_attention_heads(
            query,
            layer_cache["visual_key"],
            layer_cache["visual_value"],
            text_key,
            text_value,
            attention_mask=attention_mask,
            scaling=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
        )
        full_attention = attn.o_proj(heads.reshape(*input_shape, -1).contiguous())
        full_after_attention = residual + full_attention
        full_output = full_after_attention + layer.mlp(layer.post_attention_layernorm(full_after_attention))
        h = full_output[:, -1:]
        cache["layer_inputs"][layer_idx] = full_layer_input
        cache["layer_after_attention"][layer_idx] = full_after_attention

    cache["text_position_ids"] = full_text_position_ids
    cache["next_position_ids"] = next_position_ids + 1
    logits = model.lm_head(language_model.norm(h))
    return logits, cache


def student_forward_llava_embedding_adapter(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    adapter: "QwenEmbeddingAdapter",
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
    visual_memory: torch.Tensor | None = None,
    image_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run LLaVA text-only LLM with native projected image states as per-layer KV prefix.

    This mirrors the Qwen embedding adapter path: image features are projected to LLM
    hidden size once, adapted by a per-layer low-rank residual, then converted
    to K/V by the frozen language layer's native k_proj/v_proj. Image tokens do
    not pass through the LLM FFN.
    """
    prepared = prepare_llava_embedding_adapter_inputs(
        model,
        input_ids,
        pixel_values,
        image_token_id,
        attention_mask=attention_mask,
        visual_memory=visual_memory,
        image_sizes=image_sizes,
    )
    return student_forward_llava_embedding_adapter_prepared(model, adapter, **prepared)


# Adapter mode names.
KV_ADAPTER_MODE = "kv_adapter"
EMBEDDING_ADAPTER_MODE = "embedding_adapter"
RECURRENT_EMBEDDING_ADAPTER_MODE = "recurrent_embedding_adapter"
_LEGACY_KV_ADAPTER_MODE = "adapter" + "_only"
_LEGACY_EMBEDDING_ADAPTER_MODE = "native" + "_visual" + "_kv" + "_" + "inject" + "ion"

QWEN_EMBEDDING_ADAPTER_MODES = (EMBEDDING_ADAPTER_MODE, RECURRENT_EMBEDDING_ADAPTER_MODE)
LLAVA_OUTPUT_MODES = (KV_ADAPTER_MODE, EMBEDDING_ADAPTER_MODE, RECURRENT_EMBEDDING_ADAPTER_MODE)


def canonical_adapter_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    value = str(mode).strip()
    aliases = {
        KV_ADAPTER_MODE: KV_ADAPTER_MODE,
        "kv-adapter": KV_ADAPTER_MODE,
        "kv adapter": KV_ADAPTER_MODE,
        _LEGACY_KV_ADAPTER_MODE: KV_ADAPTER_MODE,
        EMBEDDING_ADAPTER_MODE: EMBEDDING_ADAPTER_MODE,
        "embedding-adapter": EMBEDDING_ADAPTER_MODE,
        "embedding adapter": EMBEDDING_ADAPTER_MODE,
        _LEGACY_EMBEDDING_ADAPTER_MODE: EMBEDDING_ADAPTER_MODE,
        RECURRENT_EMBEDDING_ADAPTER_MODE: RECURRENT_EMBEDDING_ADAPTER_MODE,
        "recurrent-embedding-adapter": RECURRENT_EMBEDDING_ADAPTER_MODE,
        "recurrent embedding adapter": RECURRENT_EMBEDDING_ADAPTER_MODE,
    }
    return aliases.get(value, value)


def is_embedding_adapter_mode(mode: str | None) -> bool:
    return canonical_adapter_mode(mode) in QWEN_EMBEDDING_ADAPTER_MODES


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def qwen_prompt(processor: Any, question: str, num_images: int = 1) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(max(1, int(num_images)))]
            + [{"type": "text", "text": question.strip()}],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_row_image_paths(row: dict[str, Any], image_root: Path | None = None) -> list[Path]:
    raw_paths = row.get("images")
    if raw_paths is None:
        raw_paths = [row["image"]]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("row must contain image or non-empty images")

    root = image_root
    row_root = str(row.get("image_root") or "").strip()
    if row_root:
        root = Path(row_root)

    paths: list[Path] = []
    for raw_path in raw_paths:
        image_path = Path(str(raw_path))
        if image_path.is_absolute():
            paths.append(image_path)
        elif root is not None:
            paths.append(root / image_path)
        else:
            paths.append(image_path)
    return paths


def resolve_row_image_path(row: dict[str, Any], image_root: Path | None = None) -> Path:
    return resolve_row_image_paths(row, image_root)[0]


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _rank0_print(message: str) -> None:
    if _dist_rank() == 0:
        print(message, flush=True)


def _checkpoint_shards(model_path: str | Path) -> list[Path]:
    model_dir = Path(model_path)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(str(name) for name in index.get("weight_map", {}).values()))
        return [model_dir / name for name in filenames]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"no safetensors checkpoint shards found under {model_dir}")


def _load_qwen3vl_zero3_sharded(
    model_path: str,
    model_cls: type[Qwen3VLForConditionalGeneration] | type[Qwen3VLMoeForConditionalGeneration],
    dtype: torch.dtype,
    attn_implementation: str,
    deepspeed_config: dict[str, Any],
) -> Qwen3VLForConditionalGeneration | Qwen3VLMoeForConditionalGeneration:
    import deepspeed

    model_dir = Path(model_path)
    shards = _checkpoint_shards(model_dir)
    config = AutoConfig.from_pretrained(model_path)
    config.dtype = dtype
    for sub_config_key in getattr(config, "sub_configs", []):
        sub_config = getattr(config, sub_config_key, None)
        if sub_config is not None:
            sub_config.dtype = dtype
    if attn_implementation:
        config._attn_implementation = attn_implementation
        for sub_config_key in getattr(config, "sub_configs", []):
            sub_config = getattr(config, sub_config_key, None)
            if sub_config is not None:
                sub_config._attn_implementation = attn_implementation

    _rank0_print(
        f"zero3 sharded load: build empty {model_cls.__name__} dtype={dtype} "
        f"attn={attn_implementation} shards={len(shards)}"
    )
    init_contexts = [
        apply_patches(),
        local_torch_dtype(dtype, model_cls.__name__),
        hf_initialization.no_init_weights(),
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config),
        set_zero3_state(),
    ]
    with ContextManagers(init_contexts):
        model = model_cls(config)
        patch_output_recorders(model)

    loaded_keys: set[str] = set()
    for idx, shard_path in enumerate(shards, start=1):
        wall_start = time.perf_counter()
        _rank0_print(f"zero3 sharded load: shard {idx}/{len(shards)} {shard_path.name}")
        state_dict = load_state_dict(str(shard_path), map_location="cpu")
        loaded_keys.update(str(key) for key in state_dict.keys())
        _load_state_dict_into_zero3_model(model, state_dict)
        del state_dict
        gc.collect()
        elapsed_s = time.perf_counter() - wall_start
        _rank0_print(f"zero3 sharded load: shard {idx}/{len(shards)} done in {elapsed_s:.2f}s")

    expected_keys = {str(name) for name, _ in model.named_parameters()}
    expected_keys.update(str(name) for name, _ in model.named_buffers())
    missing = sorted(expected_keys - loaded_keys)
    if missing:
        _rank0_print(f"zero3 sharded load: missing {len(missing)} keys, first={missing[:8]}")
    model.tie_weights()
    _rank0_print(f"zero3 sharded load: complete loaded_keys={len(loaded_keys)}")
    return model


def load_frozen_qwen3vl(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
    device_map: str | dict[str, Any] | None = None,
    max_memory: dict[Any, str] | None = None,
    move_to_device: bool = True,
    zero3_sharded_load: bool = False,
    deepspeed_config: dict[str, Any] | None = None,
    deepstack: bool = False,
) -> tuple[Any, Qwen3VLForConditionalGeneration | Qwen3VLMoeForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(model_path)
    if attn_implementation == "auto":
        attn_implementation = "flash_attention_2"
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type") or "")
    model_cls = Qwen3VLMoeForConditionalGeneration if model_type == "qwen3_vl_moe" else Qwen3VLForConditionalGeneration
    if zero3_sharded_load:
        if device_map is not None:
            raise ValueError("zero3_sharded_load cannot be combined with device_map")
        if deepspeed_config is None:
            raise ValueError("zero3_sharded_load requires deepspeed_config")
        model = _load_qwen3vl_zero3_sharded(
            model_path,
            model_cls,
            dtype,
            attn_implementation,
            deepspeed_config,
        )
    else:
        model = model_cls.from_pretrained(model_path, **kwargs)
    from .model_setup import disable_qwen_deepstack

    if not deepstack:
        disable_qwen_deepstack(model)
    if device_map is None and move_to_device:
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


def module_device(module: torch.nn.Module, fallback: torch.device | None = None) -> torch.device:
    for param in module.parameters(recurse=True):
        if param.device.type != "meta":
            return param.device
    for buffer in module.buffers(recurse=True):
        if buffer.device.type != "meta":
            return buffer.device
    if fallback is not None:
        return fallback
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def qwen_input_device(model: torch.nn.Module) -> torch.device:
    qwen_model = model.model
    return module_device(qwen_model.get_input_embeddings(), module_device(qwen_model))


def qwen_visual_device(model: torch.nn.Module) -> torch.device:
    qwen_model = model.model
    return module_device(qwen_model.visual, qwen_input_device(model))


def move_qwen_inputs_to_device(inputs: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _move_position_embeddings(
    embeddings: tuple[Tensor, Tensor] | None,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    if embeddings is None:
        return None
    return tuple(tensor.to(device) for tensor in embeddings)  # type: ignore[return-value]


def qwen3vl_text_ids_and_answer_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    answer_token_lens: list[int],
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    rows_ids: list[list[int]] = []
    max_text = 0
    for batch_idx in range(input_ids.shape[0]):
        valid = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        ids = [
            int(input_ids[batch_idx, int(src_idx)].item())
            for src_idx in valid
            if int(mm_token_type_ids[batch_idx, int(src_idx)].item()) == 0
        ]
        if not ids:
            raise ValueError("empty Qwen text sequence after removing image tokens")
        rows_ids.append(ids)
        max_text = max(max_text, len(ids))

    text_ids = torch.full(
        (input_ids.shape[0], max_text),
        int(pad_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    text_mask = torch.zeros((input_ids.shape[0], max_text), device=input_ids.device, dtype=torch.bool)
    answer_mask = torch.zeros_like(text_mask)
    for batch_idx, ids in enumerate(rows_ids):
        text_len = len(ids)
        text_ids[batch_idx, :text_len] = torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)
        text_mask[batch_idx, :text_len] = True
        answer_len = max(0, min(int(answer_token_lens[batch_idx]), text_len))
        if answer_len > 0:
            answer_mask[batch_idx, text_len - answer_len : text_len] = True
    return text_ids, answer_mask, text_mask


def prepare_qwen3vl_batch_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: Path | None,
    device: torch.device,
    *,
    include_answers: bool,
    answer_instruction: str = "",
) -> tuple[dict[str, Tensor], Tensor | None, Tensor | None, list[str]]:
    texts: list[str] = []
    images: list[Image.Image] = []
    image_paths: list[str] = []
    answer_token_lens: list[int] = []
    eos = processor.tokenizer.eos_token or ""
    for row in rows:
        row_image_paths = resolve_row_image_paths(row, image_root)
        question = str(row["question"]).strip()
        if answer_instruction:
            question = f"{question}\n{answer_instruction.strip()}"
        prompt = qwen_prompt(processor, question, num_images=len(row_image_paths))
        if include_answers:
            answer = str(row.get("answer", "")).strip()
            suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
            texts.append(f"{prompt}{suffix}")
            answer_token_lens.append(len(processor.tokenizer(suffix, add_special_tokens=False).input_ids))
        else:
            texts.append(prompt)
            answer_token_lens.append(0)
        for image_path in row_image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
            image_paths.append(str(image_path))

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    try:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    if "mm_token_type_ids" not in inputs:
        raise ValueError("Qwen processor did not return mm_token_type_ids; M-RoPE would be invalid")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    if not include_answers:
        return inputs, None, None, image_paths

    pad_id = int(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
    text_ids, answer_mask, _ = qwen3vl_text_ids_and_answer_mask(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        answer_token_lens,
        pad_id,
    )
    return inputs, text_ids, answer_mask, image_paths


def qwen_position_ids(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
    *,
    inputs_embeds: Tensor | None = None,
) -> Tensor:
    qwen_model = model.model
    device = inputs_embeds.device if inputs_embeds is not None else qwen_input_device(model)
    input_ids = inputs["input_ids"].to(device)
    if inputs_embeds is None:
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    position_ids = qwen_model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=inputs.get("image_grid_thw").to(device) if torch.is_tensor(inputs.get("image_grid_thw")) else inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw").to(device) if torch.is_tensor(inputs.get("video_grid_thw")) else inputs.get("video_grid_thw"),
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.get("attention_mask").to(device) if torch.is_tensor(inputs.get("attention_mask")) else inputs.get("attention_mask"),
        past_key_values=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids").to(device) if torch.is_tensor(inputs.get("mm_token_type_ids")) else inputs.get("mm_token_type_ids"),
    )
    if position_ids is None:
        raise RuntimeError("Qwen3-VL position_ids could not be computed")
    return position_ids


@torch.no_grad()
def qwen_visual_grid_metadata(model: torch.nn.Module, image_grid_thw: Tensor) -> dict[str, Any]:
    visual = model.model.visual
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        image_grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    position_ids = get_vision_position_ids(image_grid_thw, visual.spatial_merge_size)
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(image_grid_thw, visual.config)
    return {
        "interp_indices": interp_indices,
        "interp_weights": interp_weights,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }


@torch.no_grad()
def build_qwen_initial_context(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
    *,
    position_ids: Tensor | None = None,
    visual_grid_metadata: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    qwen_model = model.model
    input_device = qwen_input_device(model)
    visual_device = qwen_visual_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    position_inputs_embeds = inputs_embeds
    metadata = visual_grid_metadata or {}
    for pixels_key, grid_key, token_id, kind in (
        ("pixel_values", "image_grid_thw", qwen_model.config.image_token_id, "image"),
        ("pixel_values_videos", "video_grid_thw", qwen_model.config.video_token_id, "video"),
    ):
        if not torch.is_tensor(inputs.get(pixels_key)):
            continue
        # Keep the original flat metadata format for image-only callers.
        media_kwargs = metadata.get(kind, {}) if "video" in metadata else (metadata if kind == "image" else {})
        media_kwargs = {key: value.to(visual_device) if torch.is_tensor(value) else value
                        for key, value in media_kwargs.items()}
        outputs = qwen_model.visual(
            inputs[pixels_key].to(visual_device).type(qwen_model.visual.dtype),
            grid_thw=inputs[grid_key].to(visual_device), return_dict=True, **media_kwargs,
        )
        embeds = outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = (input_ids == int(token_id)).unsqueeze(-1).to(device=inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(mask, embeds)
    if position_ids is None:
        position_ids = qwen_position_ids(model, inputs, inputs_embeds=position_inputs_embeds)
    return inputs_embeds, position_ids


def _hash_tensor(hasher: Any, tensor: Tensor) -> None:
    cpu = tensor.detach().to("cpu").contiguous()
    hasher.update(str(tuple(cpu.shape)).encode("utf-8"))
    hasher.update(str(cpu.dtype).encode("utf-8"))
    buffer = io.BytesIO()
    torch.save(cpu, buffer)
    hasher.update(buffer.getbuffer())


def qwen_initial_context_cache_key(model: torch.nn.Module, inputs: dict[str, Tensor], dtype: torch.dtype) -> str:
    hasher = hashlib.sha1()
    config = getattr(model, "config", None)
    hasher.update(str(getattr(config, "name_or_path", "")).encode("utf-8"))
    hasher.update(str(dtype).encode("utf-8"))
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw", "pixel_values"):
        value = inputs.get(key)
        if torch.is_tensor(value):
            hasher.update(key.encode("utf-8"))
            _hash_tensor(hasher, value)
    return hasher.hexdigest()


@torch.no_grad()
def load_or_build_qwen_initial_context(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
    *,
    cache_dir: str | Path | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if cache_dir is None:
        return build_qwen_initial_context(model, inputs)
    if dtype is None:
        dtype = inputs["pixel_values"].dtype if torch.is_tensor(inputs.get("pixel_values")) else next(model.parameters()).dtype
    cache_root = Path(cache_dir)
    cache_path = cache_root / f"{qwen_initial_context_cache_key(model, inputs, dtype)}.pt"
    device = inputs["input_ids"].device
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["initial_hidden"].to(device=device, dtype=dtype), cached["position_ids"].to(device=device)

    initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "initial_hidden": initial_hidden.detach().to("cpu"),
            "position_ids": position_ids.detach().to("cpu"),
        },
        tmp_path,
    )
    os.replace(tmp_path, cache_path)
    return initial_hidden, position_ids


def get_qwen_text_image_positions(
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = input_ids.device
    valid = attention_mask.to(device=device, dtype=torch.bool)
    token_types = mm_token_type_ids.to(device=device)
    text_valid = valid & (token_types == 0)
    image_valid = valid & (token_types != 0)
    batch = input_ids.shape[0]
    if batch == 1:
        text_src = torch.where(text_valid[0])[0]
        image_src = torch.where(image_valid[0])[0]
        if text_src.numel() == 0 or image_src.numel() == 0:
            raise ValueError("Qwen sample must contain both text and image positions")
        text_positions = text_src.unsqueeze(0)
        image_positions = image_src.unsqueeze(0)
        text_mask = torch.ones((1, text_src.numel()), device=device, dtype=torch.bool)
        image_mask = torch.ones((1, image_src.numel()), device=device, dtype=torch.bool)
        text_position_ids = position_ids[:, :1, text_src]
        return text_positions, image_positions, text_position_ids, text_mask, image_mask, valid

    text_counts = text_valid.sum(dim=1)
    image_counts = image_valid.sum(dim=1)
    if not bool(((text_counts > 0) & (image_counts > 0)).all().item()):
        raise ValueError("Qwen sample must contain both text and image positions")

    max_text = int(text_counts.max().item())
    max_image = int(image_counts.max().item())
    text_positions = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    image_positions = torch.zeros((batch, max_image), device=device, dtype=torch.long)
    text_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    image_mask = torch.zeros((batch, max_image), device=device, dtype=torch.bool)
    text_position_ids = torch.zeros((3, batch, max_text), device=device, dtype=position_ids.dtype)

    text_batch, text_src = torch.nonzero(text_valid, as_tuple=True)
    image_batch, image_src = torch.nonzero(image_valid, as_tuple=True)
    text_rank = text_valid.to(dtype=torch.long).cumsum(dim=1).sub(1)[text_batch, text_src]
    image_rank = image_valid.to(dtype=torch.long).cumsum(dim=1).sub(1)[image_batch, image_src]

    text_positions[text_batch, text_rank] = text_src
    image_positions[image_batch, image_rank] = image_src
    text_mask[text_batch, text_rank] = True
    image_mask[image_batch, image_rank] = True

    dim_idx = torch.arange(3, device=device).view(3, 1).expand(-1, text_src.numel())
    text_batch_idx = text_batch.view(1, -1).expand(3, -1)
    text_rank_idx = text_rank.view(1, -1).expand(3, -1)
    text_position_ids[dim_idx, text_batch_idx, text_rank_idx] = position_ids[:, text_batch, text_src]
    return text_positions, image_positions, text_position_ids, text_mask, image_mask, valid


def gather_batched_positions(hidden_states: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
    idx = positions.to(device=hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    gathered = torch.gather(hidden_states, dim=1, index=idx)
    return gathered * mask.to(device=hidden_states.device, dtype=gathered.dtype).unsqueeze(-1)


def qwen_visual_position_ids(full_position_ids: Tensor, image_positions: Tensor, image_mask: Tensor) -> Tensor:
    if image_positions.shape[0] == 1:
        return full_position_ids[:, :1, image_positions[0].long()]
    visual_position_ids = torch.zeros(
        3,
        image_positions.shape[0],
        image_positions.shape[1],
        device=image_positions.device,
        dtype=full_position_ids.dtype,
    )
    valid = image_mask.bool()
    batch_idx = torch.arange(image_positions.shape[0], device=image_positions.device).unsqueeze(1).expand_as(image_positions)
    for dim_idx in range(3):
        dim_positions = full_position_ids[dim_idx]
        visual_position_ids[dim_idx][valid] = dim_positions[batch_idx[valid], image_positions[valid].long()]
    return visual_position_ids


def qwen_prefix_causal_attention_mask(
    text_mask: Tensor,
    image_mask: Tensor,
    device: torch.device,
    *,
    text_positions: Tensor | None = None,
    image_positions: Tensor | None = None,
) -> Tensor:
    batch, text_len = text_mask.shape
    visual_len = image_mask.shape[1]
    if batch == 1 and text_positions is not None and image_positions is not None:
        text_order = text_positions[0].to(device=device)
        image_order = image_positions[0].to(device=device)
        text_allowed = text_order.view(1, text_len) <= text_order.view(text_len, 1)
        visual_allowed = image_order.view(1, visual_len) <= text_order.view(text_len, 1)
        return torch.cat([visual_allowed, text_allowed], dim=-1).view(1, 1, text_len, visual_len + text_len)

    valid_text = text_mask.to(device=device, dtype=torch.bool)
    valid_visual = image_mask.to(device=device, dtype=torch.bool)
    if text_positions is not None and image_positions is not None:
        text_positions = text_positions.to(device=device)
        image_positions = image_positions.to(device=device)
        text_allowed = (text_positions[:, None, :] <= text_positions[:, :, None]) & valid_text[:, None, :]
        visual_allowed = (image_positions[:, None, :] <= text_positions[:, :, None]) & valid_visual[:, None, :]
    else:
        causal = torch.ones((text_len, text_len), device=device, dtype=torch.bool).tril()
        text_allowed = causal.view(1, text_len, text_len) & valid_text.view(batch, 1, text_len)
        visual_allowed = valid_visual.view(batch, 1, visual_len).expand(batch, text_len, visual_len)
    return torch.cat([visual_allowed, text_allowed], dim=-1).unsqueeze(1)


class QwenEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mode: str,
        visual_adapter_rank: int = 128,
    ) -> None:
        super().__init__()
        mode = canonical_adapter_mode(mode)
        if mode not in QWEN_EMBEDDING_ADAPTER_MODES:
            raise ValueError(f"unsupported Qwen embedding adapter mode: {mode}")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attn_dim = int(num_heads) * int(head_dim)
        self.mode = mode
        self.visual_adapter_rank = int(visual_adapter_rank)
        rank = max(1, int(visual_adapter_rank))
        self.visual_adapter_down = nn.ModuleList([nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers)])
        self.visual_adapter_up = nn.ModuleList([nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers)])
        self.reset_parameters()

    def precompute_stacked_weights(self) -> None:
        """Stack per-layer adapter weights for batched forward. Call after loading checkpoint."""
        self._down_stacked = torch.stack([m.weight for m in self.visual_adapter_down])  # [L, R, H]
        self._up_stacked = torch.stack([m.weight for m in self.visual_adapter_up])      # [L, H, R]

    def reset_parameters(self) -> None:
        for up in self.visual_adapter_up:
            nn.init.zeros_(up.weight)

    @classmethod
    def from_language_model(
        cls,
        language_model: torch.nn.Module,
        *,
        mode: str,
        visual_adapter_rank: int = 128,
    ) -> "QwenEmbeddingAdapter":
        cfg = language_model.config
        return cls(
            hidden_size=int(cfg.hidden_size),
            num_layers=len(language_model.layers),
            num_heads=int(cfg.num_attention_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            mode=mode,
            visual_adapter_rank=visual_adapter_rank,
        )

    def all_visual_memories_batched(self, visual_memory: Tensor) -> Tensor:
        """Compute all L adapted visual memories in one batched BMM.
        
        Returns [L, B, N, H] - stack of adapted visual memories for each layer.
        Static embedding_adapter is batched; recurrent_embedding_adapter feeds each
        layer's visual memory into the next layer's adapter block.
        """
        if self.mode == RECURRENT_EMBEDDING_ADAPTER_MODE:
            memories = []
            current = visual_memory
            for layer_idx in range(self.num_layers):
                current = self.visual_memory_for_layer(current, layer_idx)
                memories.append(current)
            return torch.stack(memories, dim=0)

        if self.training and any(hasattr(param, "ds_id") for param in self.parameters()):
            memories = [self.visual_memory_for_layer(visual_memory, layer_idx) for layer_idx in range(self.num_layers)]
            return torch.stack(memories, dim=0)

        L = self.num_layers
        B, N, H = visual_memory.shape
        if self.training:
            down_stacked = torch.stack([m.weight for m in self.visual_adapter_down])
            up_stacked = torch.stack([m.weight for m in self.visual_adapter_up])
        else:
            if not hasattr(self, "_down_stacked"):
                self.precompute_stacked_weights()
            down_stacked = self._down_stacked
            up_stacked = self._up_stacked
        vm_flat = visual_memory.unsqueeze(0).expand(L, -1, -1, -1).reshape(L, B * N, H)
        adapted = torch.bmm(vm_flat, down_stacked.to(visual_memory.dtype).transpose(1, 2))  # [L, B*N, R]
        adapted = F.silu(adapted)
        adapted = torch.bmm(adapted, up_stacked.to(visual_memory.dtype).transpose(1, 2))    # [L, B*N, H]
        adapted = adapted.reshape(L, B, N, H)
        return visual_memory.unsqueeze(0) + adapted.to(visual_memory.dtype)  # [L, B, N, H]

    def visual_memory_for_layer(self, visual_memory: Tensor, layer_idx: int) -> Tensor:
        return self.visual_memory_from_modules(
            visual_memory,
            self.visual_adapter_down[layer_idx],
            self.visual_adapter_up[layer_idx],
        )

    def visual_memories_for_layers_batched(self, visual_memory: Tensor, layer_indices) -> Tensor:
        """Inference BMM over active layers only; disabled adapter MLPs do no work."""
        if self.training or self.mode == RECURRENT_EMBEDDING_ADAPTER_MODE:
            raise ValueError('Layer-subset batching requires a static inference adapter')
        indices = tuple(layer_indices)
        if not indices:
            return visual_memory.new_empty((0, *visual_memory.shape))
        if not hasattr(self, '_subset_stacked'):
            self._subset_stacked = {}
        if indices not in self._subset_stacked:
            self._subset_stacked[indices] = (
                torch.stack([self.visual_adapter_down[i].weight for i in indices]),
                torch.stack([self.visual_adapter_up[i].weight for i in indices]))
        down, up = self._subset_stacked[indices]
        layers, (batch, tokens, hidden) = len(indices), visual_memory.shape
        memory = visual_memory.unsqueeze(0).expand(layers,-1,-1,-1).reshape(layers,batch*tokens,hidden)
        delta = torch.bmm(memory,down.to(memory.dtype).transpose(1,2))
        delta = torch.bmm(F.silu(delta),up.to(memory.dtype).transpose(1,2))
        return visual_memory.unsqueeze(0)+delta.reshape(layers,batch,tokens,hidden)

    def base_visual_memory_for_layer(self, visual_memory: Tensor, layer_idx: int) -> Tensor:
        if self.mode != RECURRENT_EMBEDDING_ADAPTER_MODE:
            return visual_memory
        current = visual_memory
        for idx in range(layer_idx):
            current = self.visual_memory_for_layer(current, idx)
        return current

    def visual_memory_from_modules(
        self,
        visual_memory: Tensor,
        down: nn.Module,
        up: nn.Module,
    ) -> Tensor:
        adapted = down(visual_memory)
        adapted = up(F.silu(adapted))
        adapted = adapted.to(dtype=visual_memory.dtype)
        adapted.add_(visual_memory)
        return adapted


def qwen_project_visual_kv_for_layer(
    language_model: torch.nn.Module,
    layer: torch.nn.Module,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    padding_mask: Tensor | None,
    *,
    repeat_kv: bool = True,
    position_embeddings: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    key = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        position_embeddings = language_model.rotary_emb(normed, visual_position_ids)
    key = _apply_rope_one_from_embeddings(key, position_embeddings)
    if repeat_kv:
        key = qwen_repeat_kv(key, int(attn.num_key_value_groups))
        value = qwen_repeat_kv(value, int(attn.num_key_value_groups))
    key = key.contiguous()
    value = value.contiguous()
    return key, value, padding_mask


def qwen_prefix_causal_attention_heads(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scaling: float,
    attention_mask: Tensor | None = None,
    attention_plan: dict[str, Any] | None = None,
) -> Tensor:
    if attention_plan is not None:
        from src.attention import attention_heads
        return attention_heads(query, key, value, scaling=scaling, plan=attention_plan)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(scaling),
        enable_gqa=query.shape[1] != key.shape[1],
    ).transpose(1, 2).contiguous()


def qwen_lm_head_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    hidden_states: Tensor,
    text_mask: Tensor | None = None,
    *,
    logits_to_keep: int = 0,
) -> Tensor:
    if text_mask is not None:
        text_mask = text_mask.to(hidden_states.device)
    if logits_to_keep > 0:
        if logits_to_keep == 1 and text_mask is not None:
            if text_mask.shape[0] == 1:
                hidden_states = hidden_states[:, -1:]
            else:
                last_idx = text_mask.long().sum(dim=1).sub(1).clamp_min(0)
                batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                hidden_states = hidden_states[batch_idx, last_idx].unsqueeze(1)
        else:
            hidden_states = hidden_states[:, -int(logits_to_keep) :]
    norm_device = module_device(language_model.norm, hidden_states.device)
    hidden_states = hidden_states.to(norm_device)
    if torch.compiler.is_compiling():
        hidden_states = _eager_module_call(language_model.norm, hidden_states)
    else:
        hidden_states = language_model.norm(hidden_states)
    head_device = module_device(model.lm_head, hidden_states.device)
    hidden_states = hidden_states.to(head_device)
    return model.lm_head(hidden_states)


def prepare_qwen_embedding_adapter_inputs(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    initial_hidden: Tensor,
    position_ids: Tensor,
    *,
    reuse_position_embeddings: bool = True,
) -> dict[str, Any]:
    language_model = model.model.language_model
    adapter_dtype = next(adapter.parameters()).dtype
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_qwen_text_image_positions(
        input_ids,
        attention_mask,
        mm_token_type_ids,
        position_ids,
    )
    visual_position_ids = qwen_visual_position_ids(position_ids, image_pos, image_mask)
    visual_memory = gather_batched_positions(initial_hidden, image_pos, image_mask).to(dtype=adapter_dtype)
    h = gather_batched_positions(initial_hidden, text_pos, text_mask).to(dtype=adapter_dtype)
    prefix_attention_mask = qwen_prefix_causal_attention_mask(
        text_mask,
        image_mask,
        h.device,
        text_positions=text_pos,
        image_positions=image_pos,
    )
    text_position_embeddings = None
    visual_position_embeddings = None
    if reuse_position_embeddings:
        rotary_emb = language_model.rotary_emb
        text_position_embeddings = rotary_emb(h, text_position_ids)
        visual_position_embeddings = rotary_emb(visual_memory, visual_position_ids)
    attention_plan = None
    if getattr(model, "_adapter_attention_implementation", None) == "flash_attention_2":
        from src.attention import prefix_plan
        attention_plan = prefix_plan(text_pos, image_pos, text_mask, image_mask)
    return {
        "attention_plan": attention_plan,
        "h": h,
        "visual_memory": visual_memory,
        "text_mask": text_mask,
        "image_mask": image_mask,
        "text_positions": text_pos,
        "image_positions": image_pos,
        "text_position_ids": text_position_ids,
        "visual_position_ids": visual_position_ids,
        "prefix_attention_mask": prefix_attention_mask,
        "text_position_embeddings": text_position_embeddings,
        "visual_position_embeddings": visual_position_embeddings,
    }


def qwen_embedding_adapter_logits_prepared(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    *,
    h: Tensor,
    visual_memory: Tensor,
    text_mask: Tensor,
    text_position_ids: Tensor,
    visual_position_ids: Tensor,
    prefix_attention_mask: Tensor,
    attention_plan: dict[str, Any] | None = None,
    text_position_embeddings: tuple[Tensor, Tensor] | None = None,
    visual_position_embeddings: tuple[Tensor, Tensor] | None = None,
    logits_to_keep: int = 0,
    use_hf_attention: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor] | None]:
    language_model = model.model.language_model
    layers = language_model.layers
    rotary_emb = language_model.rotary_emb
    compile_exact = torch.compiler.is_compiling()
    # Pre-compute all visual memories in one batched BMM
    all_vis_memories = adapter.all_visual_memories_batched(visual_memory)  # [L, B, N, H]
    for layer_idx, layer in enumerate(layers):
        layer_device = module_device(layer, h.device)
        h = h.to(layer_device)
        text_position_ids_layer = text_position_ids.to(layer_device)
        visual_position_ids_layer = visual_position_ids.to(layer_device)
        prefix_attention_mask_layer = prefix_attention_mask.to(layer_device)
        layer_text_position_embeddings = _move_position_embeddings(text_position_embeddings, layer_device)
        layer_visual_position_embeddings = _move_position_embeddings(visual_position_embeddings, layer_device)
        attn = layer.self_attn
        normed_text = _eager_module_call(layer.input_layernorm, h) if compile_exact else layer.input_layernorm(h)
        text_shape = normed_text.shape[:-1]
        hidden_shape = (*text_shape, -1, attn.head_dim)
        raw_query = attn.q_proj(normed_text).view(hidden_shape)
        raw_text_key = attn.k_proj(normed_text).view(hidden_shape)
        query = (_eager_module_call(attn.q_norm, raw_query) if compile_exact else attn.q_norm(raw_query)).transpose(1, 2)
        text_key = (_eager_module_call(attn.k_norm, raw_text_key) if compile_exact else attn.k_norm(raw_text_key)).transpose(1, 2)
        text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
        if layer_text_position_embeddings is None:
            layer_text_position_embeddings = rotary_emb(normed_text, text_position_ids_layer)
        query, text_key = _compile_exact_qwen_apply_rotary_pos_emb(query, text_key, layer_text_position_embeddings)

        vision_states = all_vis_memories[layer_idx].to(layer_device)
        normed_vision = _eager_module_call(layer.input_layernorm, vision_states) if compile_exact else layer.input_layernorm(vision_states)
        vision_shape = normed_vision.shape[:-1]
        vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
        raw_visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape)
        visual_key = (_eager_module_call(attn.k_norm, raw_visual_key) if compile_exact else attn.k_norm(raw_visual_key)).transpose(1, 2)
        visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        if layer_visual_position_embeddings is None:
            layer_visual_position_embeddings = rotary_emb(normed_vision, visual_position_ids_layer)
        visual_key = _apply_rope_one_from_embeddings(visual_key, layer_visual_position_embeddings)

        heads = (
            _hf_sdpa_prefix_causal_attention_heads(
                attn,
                query,
                visual_key,
                visual_value,
                text_key,
                text_value,
                attention_mask=prefix_attention_mask_layer,
                scaling=float(attn.scaling),
            )
            if use_hf_attention
            else _prefix_causal_attention_heads(
                query,
                visual_key,
                visual_value,
                text_key,
                text_value,
                attention_mask=prefix_attention_mask_layer,
                scaling=float(attn.scaling),
                attention_plan=attention_plan,
            )
        )
        text_attention = attn.o_proj(heads.reshape(*text_shape, -1).contiguous())
        h = h + text_attention.to(dtype=h.dtype)
        residual = h
        h = _eager_module_call(layer.post_attention_layernorm, h) if compile_exact else layer.post_attention_layernorm(h)
        h = _eager_module_call(layer.mlp, h) if compile_exact else layer.mlp(h)
        h = residual + h
    logits = qwen_lm_head_logits(model, language_model, h, text_mask, logits_to_keep=logits_to_keep)
    return logits, text_mask, None


def qwen_embedding_adapter_logits_prepared_hf_attention(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    **prepared: Any,
) -> tuple[Tensor, Tensor, list[Tensor] | None]:
    return qwen_embedding_adapter_logits_prepared(model, adapter, **prepared, use_hf_attention=True)


def qwen_embedding_adapter_prefill_cache_prepared(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    *,
    h: Tensor,
    visual_memory: Tensor,
    text_mask: Tensor,
    image_mask: Tensor,
    text_positions: Tensor,
    image_positions: Tensor,
    text_position_ids: Tensor,
    visual_position_ids: Tensor,
    prefix_attention_mask: Tensor,
    attention_plan: dict[str, Any] | None = None,
    text_position_embeddings: tuple[Tensor, Tensor] | None = None,
    visual_position_embeddings: tuple[Tensor, Tensor] | None = None,
    logits_to_keep: int = 1,
    retain_prefix_states: bool = True,
    batch_visual_memories: bool = False,
    exact_kernels: bool = False,
    pack_native_cache: bool = False,
    fused_norm_rope: bool = False,
    start_layer: int = 0,
    prefix_layer_caches: list[dict[str, Tensor]] | None = None,
    blocked_visual_layers: list[int] | tuple[int, ...] | None = None,
    text_only_attention_plan: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    language_model = model.model.language_model
    layers = language_model.layers
    rotary_emb = language_model.rotary_emb
    if not 0 <= start_layer < len(layers):
        raise ValueError('Invalid adapter start layer')
    if len(prefix_layer_caches or []) != start_layer:
        raise ValueError('One native KV cache is required for every prefix layer')
    if start_layer and (retain_prefix_states or pack_native_cache or adapter.mode == RECURRENT_EMBEDDING_ADAPTER_MODE):
        raise ValueError('Partial prefill requires static adapter, fast decode, and unpacked caches')
    if start_layer and not (attention_plan and attention_plan.get('dense_decode_ready', False)):
        raise ValueError('Mixed-length prefix caches require unpadded FA2 dense decode')
    blocked_layers = set(blocked_visual_layers or ())
    if any(i < start_layer or i >= len(layers) for i in blocked_layers):
        raise ValueError('Blocked visual layers must belong to the adapter suffix')
    if blocked_layers and (retain_prefix_states
                           or adapter.mode == RECURRENT_EMBEDDING_ADAPTER_MODE
                           or not (attention_plan and attention_plan.get('dense_decode_ready', False))):
        raise ValueError('Visual blocking requires static adapter memories and FA2 fast decode')
    text_only_plan = text_only_attention_plan
    if blocked_layers and text_only_plan is None:
        from src.attention import prefix_plan
        text_only_plan = prefix_plan(text_positions, image_positions[:, :0], text_mask, image_mask[:, :0])
    layer_caches: list[dict[str, Tensor]] = [dict(c) for c in (prefix_layer_caches or [])]
    layer_inputs: list[Tensor] = []
    layer_after_attention: list[Tensor] = []
    layer_visual_memory = visual_memory
    active_layers = tuple(i for i in range(start_layer,len(layers)) if i not in blocked_layers)
    memory_offsets = {i:j for j,i in enumerate(active_layers)} if blocked_layers else {i:i for i in range(len(layers))}
    all_visual_memories = None
    if batch_visual_memories:
        all_visual_memories = (adapter.visual_memories_for_layers_batched(visual_memory,active_layers)
            if blocked_layers else adapter.all_visual_memories_batched(visual_memory))
    if exact_kernels:
        if attention_plan is None or h.shape[0] != 1 or torch.is_grad_enabled():
            raise ValueError('Exact adapter kernels require batch-one FA2 inference')
        from src.kernels import exact_rope, split_attention_heads
    if fused_norm_rope:
        if not exact_kernels:
            raise ValueError('Fused norm/RoPE requires exact adapter kernels')
        from src.kernels import native_order_norm_rope
    packed_native_kv = None
    packed_native_groups = []
    packed_slots = {}
    if pack_native_cache:
        if not exact_kernels or retain_prefix_states:
            raise ValueError('Packed native KV requires the exact fast adapter path')
        from src.kernels import pack_native_layer
        if blocked_layers:
            for indices, length in [(tuple(sorted(blocked_layers)),h.shape[1]),
                                     (active_layers,visual_memory.shape[1]+h.shape[1])]:
                if not indices:continue
                buffer=h.new_empty((2,len(indices),1,language_model.config.num_key_value_heads,length,layers[0].self_attn.head_dim))
                packed_native_groups.append((indices,buffer))
                for offset,index in enumerate(indices):packed_slots[index]=(buffer,offset)
        else:
            packed_native_kv = h.new_empty((2, len(layers), 1,
                language_model.config.num_key_value_heads, visual_memory.shape[1] + h.shape[1],
                layers[0].self_attn.head_dim))

    for layer_idx, layer in enumerate(layers):
        if layer_idx < start_layer:
            continue
        if retain_prefix_states:
            layer_inputs.append(h)
        attn = layer.self_attn
        normed_text = _compile_exact_module_call(layer.input_layernorm, h)
        text_shape = normed_text.shape[:-1]
        hidden_shape = (*text_shape, -1, attn.head_dim)
        raw_query = attn.q_proj(normed_text).view(hidden_shape)
        raw_text_key = attn.k_proj(normed_text).view(hidden_shape)
        query = raw_query.transpose(1, 2) if fused_norm_rope else _compile_exact_module_call(attn.q_norm, raw_query).transpose(1, 2)
        text_key = raw_text_key.transpose(1, 2) if fused_norm_rope else _compile_exact_module_call(attn.k_norm, raw_text_key).transpose(1, 2)
        text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
        layer_text_position_embeddings = text_position_embeddings
        if layer_text_position_embeddings is None:
            layer_text_position_embeddings = rotary_emb(normed_text, text_position_ids)
        if fused_norm_rope:
            query = native_order_norm_rope(query, attn.q_norm.weight, attn.q_norm.variance_epsilon, layer_text_position_embeddings)
            text_key = native_order_norm_rope(text_key, attn.k_norm.weight, attn.k_norm.variance_epsilon, layer_text_position_embeddings)
        elif exact_kernels:
            query = exact_rope(query, layer_text_position_embeddings)
            text_key = exact_rope(text_key, layer_text_position_embeddings)
        else:
            query, text_key = _compile_exact_qwen_apply_rotary_pos_emb(query, text_key, layer_text_position_embeddings)

        layer_attention_plan = attention_plan
        layer_attention_mask = prefix_attention_mask
        if layer_idx in blocked_layers:
            # Empty visual caches remove keys from the softmax denominator too.
            # Do not execute this layer's adapter, visual norm, or visual K/V projections.
            visual_key, visual_value = text_key[:, :, :0], text_value[:, :, :0]
            layer_attention_plan = text_only_plan
            layer_attention_mask = prefix_attention_mask[..., visual_memory.shape[1]:]
        else:
            vision_states = (all_visual_memories[memory_offsets[layer_idx]] if all_visual_memories is not None
                             else adapter.visual_memory_for_layer(layer_visual_memory, layer_idx))
            if adapter.mode == RECURRENT_EMBEDDING_ADAPTER_MODE:
                layer_visual_memory = vision_states
            normed_vision = _compile_exact_module_call(layer.input_layernorm, vision_states)
            vision_shape = normed_vision.shape[:-1]
            vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
            raw_visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape)
            visual_key = raw_visual_key.transpose(1, 2) if fused_norm_rope else _compile_exact_module_call(attn.k_norm, raw_visual_key).transpose(1, 2)
            visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
            layer_visual_position_embeddings = visual_position_embeddings
            if layer_visual_position_embeddings is None:
                layer_visual_position_embeddings = rotary_emb(normed_vision, visual_position_ids)
            if fused_norm_rope:
                visual_key = native_order_norm_rope(visual_key, attn.k_norm.weight, attn.k_norm.variance_epsilon, layer_visual_position_embeddings)
            else:
                visual_key = (exact_rope(visual_key, layer_visual_position_embeddings) if exact_kernels
                              else _apply_rope_one_from_embeddings(visual_key, layer_visual_position_embeddings))

        heads = split_attention_heads(query, visual_key, visual_value, text_key, text_value,
            scaling=float(attn.scaling), plan=layer_attention_plan) if exact_kernels else _prefix_causal_attention_heads(
            query,
            visual_key,
            visual_value,
            text_key,
            text_value,
            attention_mask=layer_attention_mask,
            scaling=float(attn.scaling),
            attention_plan=layer_attention_plan,
        )
        if packed_native_kv is not None or packed_native_groups:
            if packed_native_groups:
                buffer,offset=packed_slots[layer_idx]
                packed=buffer[:,offset,0]
            else:
                packed = packed_native_kv[:, layer_idx, 0]
            pack_native_layer(visual_key, visual_value, text_key, text_value, packed)
            visual_length = visual_key.shape[2]
            layer_caches.append(dict(
                visual_key=packed[0, :, :visual_length].unsqueeze(0),
                text_key=packed[0, :, visual_length:].unsqueeze(0),
                visual_value=packed[1, :, :visual_length].unsqueeze(0),
                text_value=packed[1, :, visual_length:].unsqueeze(0)))
        else:
            layer_caches.append({
                "text_key": text_key.contiguous(),
                "text_value": text_value.contiguous(),
                "visual_key": visual_key.contiguous(),
                "visual_value": visual_value.contiguous(),
            })
        text_attention = attn.o_proj(heads.reshape(*text_shape, -1).contiguous())
        h = h + text_attention.to(dtype=h.dtype)
        if retain_prefix_states:
            layer_after_attention.append(h)
        residual = h
        h = _compile_exact_module_call(layer.post_attention_layernorm, h)
        h = _compile_exact_module_call(layer.mlp, h)
        h = residual + h

    logits = qwen_lm_head_logits(model, language_model, h, text_mask, logits_to_keep=logits_to_keep)
    last_idx = text_mask.long().sum(dim=1).sub(1).clamp_min(0).view(1, -1, 1).expand(text_position_ids.shape[0], -1, 1)
    next_position_ids = text_position_ids.gather(2, last_idx) + 1
    next_text_positions = text_positions.gather(1, text_mask.long().sum(dim=1).sub(1).clamp_min(0).view(-1, 1)) + 1
    cache = {
        "attention_implementation": "flash_attention_2" if attention_plan is not None else "sdpa",
        "dense_decode_ready": bool(attention_plan and attention_plan.get("dense_decode_ready", False)),
        "layers": layer_caches,
        "text_mask": text_mask.clone(),
        "image_mask": image_mask.clone(),
        "text_positions": text_positions.clone(),
        "image_positions": image_positions.clone(),
        "text_position_ids": text_position_ids.clone(),
        "layer_inputs": layer_inputs,
        "layer_after_attention": layer_after_attention,
        "next_position_ids": next_position_ids,
        "next_text_positions": next_text_positions,
    }
    if packed_native_kv is not None:
        cache['_packed_native_kv'] = packed_native_kv
    if packed_native_groups:
        cache['_packed_native_groups'] = packed_native_groups
    if blocked_layers:
        cache['blocked_visual_layers'] = sorted(blocked_layers)
    return logits, text_mask, cache


def qwen_embedding_adapter_prefill_cache(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    initial_hidden: Tensor,
    position_ids: Tensor,
    *,
    logits_to_keep: int = 1,
    reuse_position_embeddings: bool = True,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    prepared = prepare_qwen_embedding_adapter_inputs(
        model,
        adapter,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        initial_hidden,
        position_ids,
        reuse_position_embeddings=reuse_position_embeddings,
    )
    return qwen_embedding_adapter_prefill_cache_prepared(
        model,
        adapter,
        h=prepared["h"],
        visual_memory=prepared["visual_memory"],
        text_mask=prepared["text_mask"],
        image_mask=prepared["image_mask"],
        text_positions=prepared["text_positions"],
        image_positions=prepared["image_positions"],
        text_position_ids=prepared["text_position_ids"],
        visual_position_ids=prepared["visual_position_ids"],
        prefix_attention_mask=prepared["prefix_attention_mask"],
        attention_plan=prepared.get("attention_plan"),
        text_position_embeddings=prepared["text_position_embeddings"],
        visual_position_embeddings=prepared["visual_position_embeddings"],
        logits_to_keep=logits_to_keep,
    )


def _qwen_decode_attention_mask(
    cache: dict[str, Any],
    token_position_ids: Tensor,
    current_text_mask: Tensor | None = None,
) -> Tensor | None:
    text_mask = cache["text_mask"].to(dtype=torch.bool)
    image_mask = cache["image_mask"].to(dtype=torch.bool)
    image_positions = cache["image_positions"].to(device=token_position_ids.device)
    # M-RoPE coordinates compress image grids and are not sequence offsets.
    # Compare positions in the original sequence on both sides of causality.
    current_pos = cache["next_text_positions"].to(device=token_position_ids.device).view(-1, 1)
    visual_allowed = image_mask & (image_positions <= current_pos)
    if current_text_mask is None:
        current_text = torch.ones((text_mask.shape[0], 1), device=text_mask.device, dtype=torch.bool)
    else:
        current_text = current_text_mask.to(device=text_mask.device, dtype=torch.bool).view(-1, 1)
    allowed = torch.cat([visual_allowed, text_mask, current_text], dim=1)
    return allowed[:, None, None, :]


def qwen_embedding_adapter_decode_step(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    token_ids: Tensor,
    cache: dict[str, Any],
    *,
    logits_to_keep: int = 1,
    token_active_mask: Tensor | None = None,
    attention_plan: dict[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    language_model = model.model.language_model
    h = model.model.get_input_embeddings()(token_ids)
    token_position_ids = cache["next_position_ids"]
    token_position_embeddings = language_model.rotary_emb(h, token_position_ids)
    dense_decode = cache.get("dense_decode_ready", False) and token_active_mask is None
    if cache.get('blocked_visual_layers') and not dense_decode:
        raise ValueError('Layer-wise visual blocking requires unpadded single-request fast decode')
    attention_mask = None if dense_decode else _qwen_decode_attention_mask(cache, token_position_ids, token_active_mask)

    if attention_plan is None and cache.get("attention_implementation") == "flash_attention_2":
        from src.attention import decode_plan
        attention_plan = {"dense_decode": True} if dense_decode else decode_plan(attention_mask)
    for layer_idx, layer in enumerate(language_model.layers):
        attn = layer.self_attn
        normed_text = _compile_exact_module_call(layer.input_layernorm, h)
        text_shape = normed_text.shape[:-1]
        hidden_shape = (*text_shape, -1, attn.head_dim)
        raw_query = attn.q_proj(normed_text).view(hidden_shape)
        raw_text_key = attn.k_proj(normed_text).view(hidden_shape)
        query = _compile_exact_module_call(attn.q_norm, raw_query).transpose(1, 2)
        text_key = _compile_exact_module_call(attn.k_norm, raw_text_key).transpose(1, 2)
        text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
        query, text_key = _compile_exact_qwen_apply_rotary_pos_emb(query, text_key, token_position_embeddings)

        layer_cache = cache["layers"][layer_idx]
        key = torch.cat([layer_cache["visual_key"], layer_cache["text_key"], text_key], dim=2)
        value = torch.cat([layer_cache["visual_value"], layer_cache["text_value"], text_value], dim=2)
        heads = qwen_prefix_causal_attention_heads(
            query,
            key,
            value,
            attention_mask=attention_mask,
            attention_plan=attention_plan,
            scaling=float(attn.scaling),
        )
        # Attention already assembled the growing KV. Retain disjoint views
        # of that storage instead of concatenating the text cache a second time.
        visual_length = layer_cache["visual_key"].shape[2]
        layer_cache["visual_key"], layer_cache["text_key"] = key[:, :, :visual_length], key[:, :, visual_length:]
        layer_cache["visual_value"], layer_cache["text_value"] = value[:, :, :visual_length], value[:, :, visual_length:]
        text_attention = attn.o_proj(heads.reshape(*text_shape, -1).contiguous())
        h = h + text_attention.to(dtype=h.dtype)
        residual = h
        h = _compile_exact_module_call(layer.post_attention_layernorm, h)
        h = _compile_exact_module_call(layer.mlp, h)
        h = residual + h

    if token_active_mask is None:
        active_column = torch.ones((cache["text_mask"].shape[0], 1), device=cache["text_mask"].device, dtype=torch.bool)
    else:
        active_column = token_active_mask.to(device=cache["text_mask"].device, dtype=torch.bool).view(-1, 1)
    cache["text_mask"] = torch.cat([cache["text_mask"], active_column], dim=1)
    cache["dense_decode_ready"] = dense_decode
    cache["next_position_ids"] = token_position_ids + 1
    cache["next_text_positions"] = cache["next_text_positions"] + 1
    logits = qwen_lm_head_logits(model, language_model, h, None, logits_to_keep=logits_to_keep)
    return logits, cache


def qwen_embedding_adapter_decode_step_shape_exact(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    token_ids: Tensor,
    cache: dict[str, Any],
    *,
    logits_to_keep: int = 1,
    token_active_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    language_model = model.model.language_model
    h = model.model.get_input_embeddings()(token_ids)
    token_position_ids = cache["next_position_ids"]
    text_position_ids = cache["text_position_ids"]
    token_text_positions = cache["next_text_positions"]
    full_text_position_ids = torch.cat([text_position_ids, token_position_ids], dim=2)
    if token_active_mask is None:
        active_column = torch.ones((cache["text_mask"].shape[0], 1), device=cache["text_mask"].device, dtype=torch.bool)
    else:
        active_column = token_active_mask.to(device=cache["text_mask"].device, dtype=torch.bool).view(-1, 1)
    full_text_mask = torch.cat([cache["text_mask"], active_column], dim=1)
    full_text_positions = torch.cat([cache["text_positions"], token_text_positions], dim=1)
    attention_mask = qwen_prefix_causal_attention_mask(
        full_text_mask,
        cache["image_mask"],
        h.device,
        text_positions=full_text_positions,
        image_positions=cache["image_positions"],
    )

    attention_plan = None
    if cache.get("attention_implementation") == "flash_attention_2":
        from src.attention import prefix_plan
        attention_plan = prefix_plan(full_text_positions, cache["image_positions"], full_text_mask, cache["image_mask"])
    for layer_idx, layer in enumerate(language_model.layers):
        attn = layer.self_attn
        prompt_layer_input = cache["layer_inputs"][layer_idx]
        full_layer_input = torch.cat([prompt_layer_input, h], dim=1)
        normed_text = _compile_exact_module_call(layer.input_layernorm, full_layer_input)
        text_shape = normed_text.shape[:-1]
        hidden_shape = (*text_shape, -1, attn.head_dim)
        raw_query = attn.q_proj(normed_text).view(hidden_shape)
        raw_text_key = attn.k_proj(normed_text).view(hidden_shape)
        query = _compile_exact_module_call(attn.q_norm, raw_query).transpose(1, 2)
        text_key = _compile_exact_module_call(attn.k_norm, raw_text_key).transpose(1, 2)
        text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
        text_position_embeddings = language_model.rotary_emb(normed_text, full_text_position_ids)
        query, text_key = _compile_exact_qwen_apply_rotary_pos_emb(query, text_key, text_position_embeddings)

        layer_cache = cache["layers"][layer_idx]
        heads = _prefix_causal_attention_heads(
            query,
            layer_cache["visual_key"],
            layer_cache["visual_value"],
            text_key,
            text_value,
            attention_mask=attention_mask,
            attention_plan=attention_plan,
            scaling=float(attn.scaling),
        )
        full_attention = attn.o_proj(heads.reshape(*text_shape, -1).contiguous())
        full_after_attention = full_layer_input + full_attention.to(dtype=full_layer_input.dtype)
        residual = full_after_attention
        full_post_normed = _compile_exact_module_call(layer.post_attention_layernorm, full_after_attention)
        full_output = residual + _compile_exact_module_call(layer.mlp, full_post_normed)
        h = full_output[:, -1:]
        cache["layer_inputs"][layer_idx] = full_layer_input
        cache["layer_after_attention"][layer_idx] = full_after_attention
        cache["layers"][layer_idx]["text_key"] = text_key.contiguous()
        cache["layers"][layer_idx]["text_value"] = text_value.contiguous()

    cache["text_mask"] = full_text_mask
    cache["text_positions"] = full_text_positions
    cache["text_position_ids"] = full_text_position_ids
    cache["next_position_ids"] = token_position_ids + 1
    cache["next_text_positions"] = token_text_positions + 1
    logits = qwen_lm_head_logits(model, language_model, h, None, logits_to_keep=logits_to_keep)
    return logits, cache


def qwen_embedding_adapter_logits_from_tensors(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    initial_hidden: Tensor,
    position_ids: Tensor,
    *,
    compact_no_padding: bool = False,
    logits_to_keep: int = 0,
    reuse_position_embeddings: bool = True,
) -> tuple[Tensor, Tensor, list[Tensor] | None]:
    prepared = prepare_qwen_embedding_adapter_inputs(
        model,
        adapter,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        initial_hidden,
        position_ids,
        reuse_position_embeddings=reuse_position_embeddings,
    )
    return qwen_embedding_adapter_logits_prepared(
        model,
        adapter,
        h=prepared["h"],
        visual_memory=prepared["visual_memory"],
        text_mask=prepared["text_mask"],
        text_position_ids=prepared["text_position_ids"],
        visual_position_ids=prepared["visual_position_ids"],
        prefix_attention_mask=prepared["prefix_attention_mask"],
        attention_plan=prepared.get("attention_plan"),
        text_position_embeddings=prepared["text_position_embeddings"],
        visual_position_embeddings=prepared["visual_position_embeddings"],
        logits_to_keep=logits_to_keep,
    )


def qwen_embedding_adapter_logits(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    inputs: dict[str, Tensor],
    *,
    initial_hidden: Tensor | None = None,
    position_ids: Tensor | None = None,
    compact_no_padding: bool = False,
    logits_to_keep: int = 0,
    reuse_position_embeddings: bool = True,
) -> tuple[Tensor, Tensor, list[Tensor] | None]:
    if initial_hidden is None or position_ids is None:
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    return qwen_embedding_adapter_logits_from_tensors(
        model,
        adapter,
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        initial_hidden,
        position_ids,
        compact_no_padding=compact_no_padding,
        logits_to_keep=logits_to_keep,
        reuse_position_embeddings=reuse_position_embeddings,
    )


def load_qwen_embedding_adapter_checkpoint(
    checkpoint_path: str | Path,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[QwenEmbeddingAdapter, dict[str, Any]]:
    # Compatibility API; training and evaluation share model_setup.
    from src.model_setup import load_qwen_embedding_adapter_checkpoint as load
    return load(checkpoint_path, language_model, device, dtype)
