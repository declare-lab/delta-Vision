"""Vision KV Adapter: inject vision encoder KV cache into LLM layers."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb, repeat_kv as llama_repeat_kv
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as qwen_apply_rotary_pos_emb, repeat_kv as qwen_repeat_kv


class PerLayerKVAdapter(nn.Module):
    """Per-LLM-layer adapter that maps vision encoder KV to LLM KV space.

    For each of the 32 LLM layers:
      - Learns a soft mixture over source_layers (last 2 ViT layers)
      - Projects mixed source K and V to LLM dim via independent linear layers
      - gate scalar controls injection strength (init sigmoid(-5) for gradual warmup)
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
        if self.concat_source:
            # Concat mode: [B, num_source, N_vis, D] -> [B, N_vis, num_source*D]
            B, S, N, D = source_k.shape
            mixed_k = source_k.permute(0, 2, 1, 3).reshape(B, N, S * D)
            mixed_v = source_v.permute(0, 2, 1, 3).reshape(B, N, S * D)
        else:
            # Weighted sum mode
            weights = F.softmax(self.source_mix[layer_idx].float(), dim=-1)
            weights = weights.to(source_k.dtype)
            mixed_k = torch.einsum("s,bsnd->bnd", weights, source_k)
            mixed_v = torch.einsum("s,bsnd->bnd", weights, source_v)

        B, N, _ = mixed_k.shape
        gate = torch.sigmoid(self.gates[layer_idx])

        if self.k_projs is not None:
            key = self.k_projs[layer_idx](mixed_k)
            value = self.v_projs[layer_idx](mixed_v)
            if self.use_activation:
                key = F.silu(key)
                value = F.silu(value)
        elif self.expansion_dim > 0:
            # SwiGLU: silu(gate(x)) * up(x), then down proj
            key = self.k_down_mlp[layer_idx](F.silu(self.k_gate[layer_idx](mixed_k)) * self.k_up_mlp[layer_idx](mixed_k))
            value = self.v_down_mlp[layer_idx](F.silu(self.v_gate[layer_idx](mixed_v)) * self.v_up_mlp[layer_idx](mixed_v))
        else:
            key = self.k_up[layer_idx](F.silu(self.k_down[layer_idx](mixed_k)))
            value = self.v_up[layer_idx](F.silu(self.v_down[layer_idx](mixed_v)))

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
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    saved_config = ckpt.get("adapter_config", {}) if isinstance(ckpt, dict) else {}
    output_mode = saved_config.get("output_mode") or ckpt_args.get("output_mode") or "adapter_only"
    if output_mode == "native_visual_kv_injection":
        if language_model is None:
            raise ValueError("language_model is required to load native_visual_kv_injection adapters")
        state_dict = ckpt["state_dict"]
        first_down = state_dict.get("visual_adapter_down.0.weight")
        visual_adapter_rank = int(
            saved_config.get(
                "visual_adapter_rank",
                ckpt_args.get("visual_adapter_rank", first_down.shape[0] if first_down is not None else 128),
            )
        )
        adapter = QwenVisualDeltaAdapter.from_language_model(
            language_model,
            mode="native_visual_kv_injection",
            visual_adapter_rank=visual_adapter_rank,
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
        adapter.output_mode = output_mode
        return adapter, list(metadata["source_layers"]), metadata

    config, source_layers = infer_adapter_config_from_checkpoint(ckpt, language_model=language_model)
    adapter = PerLayerKVAdapter(**config)
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

    hidden = text_embeds

    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn

        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_k = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        q = _apply_rope(rotary_emb, q, text_position_ids, normed)
        text_k = _apply_rope(rotary_emb, text_k, text_position_ids, normed)

        vis_k, vis_v = adapter.forward_layer(source_k.to(dtype=dtype), source_v.to(dtype=dtype), layer_idx)
        vis_k = vis_k.transpose(1, 2)
        vis_v = vis_v.transpose(1, 2)
        vis_k = _apply_rope(rotary_emb, vis_k, image_position_ids, normed)

        k = torch.cat([vis_k, text_k], dim=2)
        v = torch.cat([vis_v, text_v], dim=2)

        # GQA: repeat KV heads to match Q heads
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            k = llama_repeat_kv(k, num_q_heads // num_kv_heads)
            v = llama_repeat_kv(v, num_q_heads // num_kv_heads)

        # Build a position-based causal mask in the original prompt order.
        text_pos = text_position_ids[0]
        img_pos = image_position_ids[0]
        img_allowed = text_pos.unsqueeze(1) >= img_pos.unsqueeze(0)
        text_allowed = text_pos.unsqueeze(1) >= text_pos.unsqueeze(0)
        causal_mask = torch.cat([img_allowed, text_allowed], dim=1)
        attn_mask = torch.zeros(1, 1, T, N_vis + T, device=device, dtype=dtype)
        attn_mask.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_out = attn.o_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits


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
) -> tuple:
    """Load LLaVA model with all parameters frozen."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def student_forward_mixed(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    adapter: PerLayerKVAdapter,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Mixed forward: original LLaVA path + adapter KV added to image token KV positions.

    Standard LLaVA embeds image as 576 tokens. In each attention layer, the KV
    at image token positions gets the adapter-predicted KV added on top.
    """
    if pixel_values.ndim != 4 or hasattr(model.model, "image_newline"):
        raise NotImplementedError("student_forward_mixed only supports fixed-grid LLaVA-style image features")

    # Standard LLaVA: merge image features into embeddings
    image_outputs = model.model.vision_tower(pixel_values, output_hidden_states=True)
    image_features = image_outputs.hidden_states[model.config.vision_feature_layer]
    image_features = image_features[:, 1:]  # remove CLS
    image_features = model.model.multi_modal_projector(image_features)

    # Build merged input embeddings
    embed_tokens = model.model.language_model.embed_tokens
    text_embeds = embed_tokens(input_ids)
    B, seq_len, _ = text_embeds.shape
    n_image = image_features.shape[1]  # 576

    # Find image token positions and replace with image features
    image_mask = input_ids == image_token_id
    final_embeds = text_embeds.clone()
    for i in range(B):
        img_positions = torch.where(image_mask[i])[0]
        if img_positions.numel() > 0:
            n = min(img_positions.numel(), n_image)
            final_embeds[i, img_positions[:n]] = image_features[i, :n].to(final_embeds.dtype)

    language_model = _get_language_model(model)
    layers = language_model.layers
    norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    B, T, _ = final_embeds.shape
    device = final_embeds.device
    dtype = final_embeds.dtype
    position_ids = torch.arange(T, device=device).unsqueeze(0)

    # Find image positions for adding adapter KV
    img_pos_list = []
    for i in range(B):
        img_pos_list.append(torch.where(image_mask[i])[0][:n_image])

    hidden = final_embeds

    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn

        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        k = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        q = _apply_rope(rotary_emb, q, position_ids, normed)
        k = _apply_rope(rotary_emb, k, position_ids, normed)

        # Add adapter KV to image token positions
        vis_k, vis_v = adapter.forward_layer(source_k.to(dtype=dtype), source_v.to(dtype=dtype), layer_idx)
        # vis_k, vis_v: [B, N_vis, num_heads, head_dim]
        vis_k_t = vis_k.transpose(1, 2)  # [B, heads, N_vis, head_dim]
        vis_v_t = vis_v.transpose(1, 2)

        # Apply RoPE to adapter K using image positions
        for i in range(B):
            img_pos = img_pos_list[i]
            n = min(img_pos.numel(), vis_k_t.shape[2])
            if n == 0:
                continue
            img_pos = img_pos[:n]
            img_position_ids = img_pos.unsqueeze(0)
            adapter_k_roped = _apply_rope(rotary_emb, vis_k_t[i:i+1, :, :n], img_position_ids, normed[i:i+1])
            k[i:i+1, :, img_pos] = k[i:i+1, :, img_pos] + adapter_k_roped
            v[i:i+1, :, img_pos] = v[i:i+1, :, img_pos] + vis_v_t[i:i+1, :, :n]

        # Standard causal attention
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        attn_mask = torch.zeros(1, 1, T, T, device=device, dtype=dtype)
        attn_mask.masked_fill_(~causal_mask, torch.finfo(dtype).min)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_out = attn.o_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits


@torch.no_grad()
def llava_projected_image_features(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    if pixel_values.ndim != 4 or hasattr(model.model, "image_newline"):
        raise NotImplementedError("native LLaVA visual KV injection only supports fixed-grid LLaVA-style image features")
    image_outputs = model.model.vision_tower(pixel_values, output_hidden_states=True)
    image_features = image_outputs.hidden_states[model.config.vision_feature_layer]
    image_features = image_features[:, 1:]
    return model.model.multi_modal_projector(image_features)


def student_forward_llava_injection(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    adapter: "QwenVisualDeltaAdapter",
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
    visual_memory: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run LLaVA text-only LLM with native projected image states as per-layer KV prefix.

    This mirrors the Qwen visual-delta path: image features are projected to LLM
    hidden size once, adapted by a per-layer low-rank residual, then converted
    to K/V by the frozen language layer's native k_proj/v_proj. Image tokens do
    not pass through the LLM FFN.
    """
    if visual_memory is None:
        visual_memory = llava_projected_image_features(model, pixel_values)

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

    img_allowed = text_position_ids.unsqueeze(2) >= image_position_ids.unsqueeze(1)
    text_allowed = text_position_ids.unsqueeze(2) >= text_position_ids.unsqueeze(1)
    prefix_mask = torch.cat([img_allowed, text_allowed], dim=-1)
    attn_mask = torch.zeros((batch, 1, text_len, image_len + text_len), device=device, dtype=dtype)
    attn_mask.masked_fill_(~prefix_mask.unsqueeze(1), torch.finfo(dtype).min)

    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        text_key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        text_value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        query = _apply_rope(rotary_emb, query, text_position_ids, normed)
        text_key = _apply_rope(rotary_emb, text_key, text_position_ids, normed)

        vision_states = adapter.visual_memory_from_modules(
            visual_memory,
            adapter.visual_adapter_down[layer_idx],
            adapter.visual_adapter_up[layer_idx],
        )
        normed_vision = layer.input_layernorm(vision_states)
        vision_shape = normed_vision.shape[:-1]
        vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
        visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        visual_key = _apply_rope(rotary_emb, visual_key, image_position_ids, normed_vision)

        key = torch.cat([visual_key, text_key], dim=2)
        value = torch.cat([visual_value, text_value], dim=2)
        if query.shape[1] != key.shape[1]:
            key = llama_repeat_kv(key, query.shape[1] // key.shape[1])
            value = llama_repeat_kv(value, query.shape[1] // value.shape[1])
        attn_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=float(getattr(attn, "scaling", attn.head_dim ** -0.5)),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        hidden = residual + attn.o_proj(attn_out)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = language_model.norm(hidden)
    return model.lm_head(hidden)


def student_forward_optimized(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    adapter: PerLayerKVAdapter,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for the older flash-attention experiment.

    The old Q-padding trick only matched a synthetic image-prefix layout. Route
    through the canonical path so benchmarks cannot silently use different
    sequence semantics.
    """
    return student_forward_with_visual_kv(
        model, input_ids, adapter, source_k, source_v, image_token_id, attention_mask=attention_mask
    )


def student_forward_flex(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    adapter: PerLayerKVAdapter,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for the older flex-attention experiment."""
    return student_forward_with_visual_kv(
        model, input_ids, adapter, source_k, source_v, image_token_id, attention_mask=attention_mask
    )


# Qwen3-VL visual-delta adapters.
QWEN_VISUAL_DELTA_MODES = ("native_visual_kv_injection",)
LLAVA_OUTPUT_MODES = ("adapter_only", "native_visual_kv_injection")


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


def load_frozen_qwen3vl(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
) -> tuple[Any, Qwen3VLForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(model_path)
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


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


def qwen_position_ids(model: torch.nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    qwen_model = model.model
    position_ids = qwen_model.compute_3d_position_ids(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        inputs_embeds=qwen_model.get_input_embeddings()(inputs["input_ids"]),
        attention_mask=inputs.get("attention_mask"),
        past_key_values=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    if position_ids is None:
        raise RuntimeError("Qwen3-VL position_ids could not be computed")
    return position_ids


@torch.no_grad()
def build_qwen_initial_context(model: torch.nn.Module, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    qwen_model = model.model
    input_ids = inputs["input_ids"]
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_outputs = qwen_model.get_image_features(
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        return_dict=True,
    )
    image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds, qwen_position_ids(model, inputs)


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
    batch = input_ids.shape[0]
    rows_text: list[list[int]] = []
    rows_image: list[list[int]] = []
    max_text = 0
    max_image = 0
    for batch_idx in range(batch):
        valid = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        text = [int(i) for i in valid if int(mm_token_type_ids[batch_idx, int(i)].item()) == 0]
        image = [int(i) for i in valid if int(mm_token_type_ids[batch_idx, int(i)].item()) == 1]
        if not text or not image:
            raise ValueError("Qwen sample must contain both text and image positions")
        rows_text.append(text)
        rows_image.append(image)
        max_text = max(max_text, len(text))
        max_image = max(max_image, len(image))

    device = input_ids.device
    text_positions = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    image_positions = torch.zeros((batch, max_image), device=device, dtype=torch.long)
    text_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    image_mask = torch.zeros((batch, max_image), device=device, dtype=torch.bool)
    text_position_ids = torch.zeros((3, batch, max_text), device=device, dtype=position_ids.dtype)
    for batch_idx in range(batch):
        t = torch.tensor(rows_text[batch_idx], device=device, dtype=torch.long)
        v = torch.tensor(rows_image[batch_idx], device=device, dtype=torch.long)
        text_positions[batch_idx, : t.numel()] = t
        image_positions[batch_idx, : v.numel()] = v
        text_mask[batch_idx, : t.numel()] = True
        image_mask[batch_idx, : v.numel()] = True
        text_position_ids[:, batch_idx, : t.numel()] = position_ids[:, batch_idx].index_select(1, t)
    return text_positions, image_positions, text_position_ids, text_mask, image_mask, attention_mask.bool()


def gather_batched_positions(hidden_states: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
    idx = positions.to(device=hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    gathered = torch.gather(hidden_states, dim=1, index=idx)
    return gathered * mask.to(device=hidden_states.device, dtype=gathered.dtype).unsqueeze(-1)


def qwen_visual_position_ids(full_position_ids: Tensor, image_positions: Tensor, image_mask: Tensor) -> Tensor:
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


def qwen_can_skip_padding_masks(text_mask: Tensor, image_mask: Tensor, compact_no_padding: bool) -> bool:
    if not compact_no_padding:
        return False
    if text_mask.shape[0] <= 1:
        return True
    return bool(text_mask.all().item() and image_mask.all().item())


def qwen_prefix_causal_attention_mask(text_mask: Tensor, image_mask: Tensor, device: torch.device) -> Tensor:
    batch, text_len = text_mask.shape
    visual_len = image_mask.shape[1]
    valid_text = text_mask.to(device=device, dtype=torch.bool)
    valid_visual = image_mask.to(device=device, dtype=torch.bool)
    causal = torch.ones((text_len, text_len), device=device, dtype=torch.bool).tril()
    text_allowed = causal.view(1, text_len, text_len) & valid_text.view(batch, 1, text_len)
    visual_allowed = valid_visual.view(batch, 1, visual_len).expand(batch, text_len, visual_len)
    return torch.cat([visual_allowed, text_allowed], dim=-1).unsqueeze(1)


class QwenVisualDeltaAdapter(nn.Module):
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
        if mode not in QWEN_VISUAL_DELTA_MODES:
            raise ValueError(f"unsupported Qwen visual-delta mode: {mode}")
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
    ) -> "QwenVisualDeltaAdapter":
        cfg = language_model.config
        return cls(
            hidden_size=int(cfg.hidden_size),
            num_layers=len(language_model.layers),
            num_heads=int(cfg.num_attention_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            mode=mode,
            visual_adapter_rank=visual_adapter_rank,
        )

    def visual_memory_for_layer(self, visual_memory: Tensor, layer_idx: int) -> Tensor:
        return self.visual_memory_from_modules(
            visual_memory,
            self.visual_adapter_down[layer_idx],
            self.visual_adapter_up[layer_idx],
        )

    def visual_memory_from_modules(
        self,
        visual_memory: Tensor,
        down: nn.Module,
        up: nn.Module,
    ) -> Tensor:
        adapted = down(visual_memory)
        adapted = up(F.silu(adapted))
        return visual_memory + adapted.to(dtype=visual_memory.dtype)


def qwen_text_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    return qwen_text_attention_output_for_layer(
        language_model,
        language_model.layers[layer_idx],
        hidden_states,
        position_ids,
        padding_mask,
    )


def qwen_text_attention_output_for_layer(
    language_model: torch.nn.Module,
    layer: torch.nn.Module,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    attention_mask_2d = None if padding_mask is None else (~padding_mask).to(dtype=torch.long)
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    normed = layer.input_layernorm(hidden_states)
    attn_output, _ = layer.self_attn(
        hidden_states=normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    return attn_output


def qwen_native_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    padding_mask: Tensor | None,
    *,
    repeat_kv: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    return qwen_native_visual_kv_for_layer(
        language_model,
        language_model.layers[layer_idx],
        vision_states,
        visual_position_ids,
        padding_mask,
        repeat_kv=repeat_kv,
    )


def qwen_native_visual_kv_for_layer(
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
    _, key = qwen_apply_rotary_pos_emb(key, key, *position_embeddings)
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
) -> Tensor:
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
    if logits_to_keep > 0:
        if logits_to_keep == 1 and text_mask is not None:
            last_idx = text_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            hidden_states = hidden_states[batch_idx, last_idx].unsqueeze(1)
        else:
            hidden_states = hidden_states[:, -int(logits_to_keep) :]
    return model.lm_head(language_model.norm(hidden_states))


def qwen_text_attention_output_with_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    text_padding_mask: Tensor | None = None,
    vision_padding_mask: Tensor | None = None,
    prefix_attention_mask: Tensor | None = None,
) -> Tensor:
    return qwen_text_attention_output_with_visual_kv_for_layer(
        language_model,
        language_model.layers[layer_idx],
        hidden_states,
        position_ids,
        vision_states,
        visual_position_ids,
        text_padding_mask=text_padding_mask,
        vision_padding_mask=vision_padding_mask,
        prefix_attention_mask=prefix_attention_mask,
    )


def qwen_text_attention_output_with_visual_kv_for_layer(
    language_model: torch.nn.Module,
    layer: torch.nn.Module,
    hidden_states: Tensor,
    position_ids: Tensor,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    text_padding_mask: Tensor | None = None,
    vision_padding_mask: Tensor | None = None,
    prefix_attention_mask: Tensor | None = None,
    text_position_embeddings: tuple[Tensor, Tensor] | None = None,
    visual_position_embeddings: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    attn = layer.self_attn
    normed_text = layer.input_layernorm(hidden_states)
    text_shape = normed_text.shape[:-1]
    hidden_shape = (*text_shape, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_key = attn.k_norm(attn.k_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
    if text_position_embeddings is None:
        text_position_embeddings = language_model.rotary_emb(normed_text, position_ids)
    query, text_key = qwen_apply_rotary_pos_emb(query, text_key, *text_position_embeddings)

    visual_key, visual_value, _ = qwen_native_visual_kv_for_layer(
        language_model,
        layer,
        vision_states,
        visual_position_ids,
        vision_padding_mask,
        repeat_kv=False,
        position_embeddings=visual_position_embeddings,
    )

    batch, text_len = hidden_states.shape[:2]
    visual_len = vision_states.shape[1]
    device = hidden_states.device
    if prefix_attention_mask is None:
        valid_text = torch.ones((batch, text_len), device=device, dtype=torch.bool)
        if text_padding_mask is not None:
            valid_text = ~text_padding_mask.to(device=device, dtype=torch.bool)
        valid_visual = torch.ones((batch, visual_len), device=device, dtype=torch.bool)
        if vision_padding_mask is not None:
            valid_visual = ~vision_padding_mask.to(device=device, dtype=torch.bool)
        attention_mask = qwen_prefix_causal_attention_mask(valid_text, valid_visual, device)
    else:
        attention_mask = prefix_attention_mask
    key = torch.cat([visual_key, text_key], dim=2)
    value = torch.cat([visual_value, text_value], dim=2)
    heads = qwen_prefix_causal_attention_heads(
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=float(attn.scaling),
    )
    return attn.o_proj(heads.reshape(*text_shape, -1).contiguous())


def run_qwen_layer_from_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_output: Tensor,
    attention_delta: Tensor | None,
) -> Tensor:
    return run_qwen_layer_from_attention_output_for_layer(
        language_model.layers[layer_idx],
        hidden_states,
        attention_output,
        attention_delta,
    )


def run_qwen_layer_from_attention_output_for_layer(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    attention_output: Tensor,
    attention_delta: Tensor | None,
) -> Tensor:
    residual = hidden_states
    hidden_states = residual + attention_output.to(dtype=hidden_states.dtype)
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def qwen_visual_delta_logits(
    model: torch.nn.Module,
    adapter: QwenVisualDeltaAdapter,
    inputs: dict[str, Tensor],
    *,
    initial_hidden: Tensor | None = None,
    position_ids: Tensor | None = None,
    collect_states: bool = False,
    compact_no_padding: bool = False,
    collect_state_indices: set[int] | None = None,
    logits_to_keep: int = 0,
    reuse_position_embeddings: bool = True,
) -> tuple[Tensor, Tensor, list[Tensor] | None]:
    language_model = model.model.language_model
    if initial_hidden is None or position_ids is None:
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_qwen_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    visual_position_ids = qwen_visual_position_ids(position_ids, image_pos, image_mask)
    compact_no_padding = qwen_can_skip_padding_masks(text_mask, image_mask, compact_no_padding)
    text_padding_mask = None if compact_no_padding else ~text_mask
    vision_padding_mask = None if compact_no_padding else ~image_mask
    visual_memory = gather_batched_positions(initial_hidden, image_pos, image_mask).to(
        dtype=next(adapter.parameters()).dtype
    )
    h = gather_batched_positions(initial_hidden, text_pos, text_mask).to(dtype=next(adapter.parameters()).dtype)
    prefix_attention_mask = qwen_prefix_causal_attention_mask(text_mask, image_mask, h.device)
    text_position_embeddings = None
    visual_position_embeddings = None
    if reuse_position_embeddings:
        text_position_embeddings = language_model.rotary_emb(h, text_position_ids)
        visual_position_embeddings = language_model.rotary_emb(visual_memory, visual_position_ids)
    states = None
    if collect_states:
        if collect_state_indices is None:
            states = [h]
        else:
            states = [h.new_empty(0) for _ in range(len(language_model.layers) + 1)]
            if 0 in collect_state_indices:
                states[0] = h
    for layer_idx, layer in enumerate(language_model.layers):
        vision_states = adapter.visual_memory_from_modules(
            visual_memory,
            adapter.visual_adapter_down[layer_idx],
            adapter.visual_adapter_up[layer_idx],
        )
        text_attention = qwen_text_attention_output_with_visual_kv_for_layer(
            language_model,
            layer,
            h,
            text_position_ids,
            vision_states,
            visual_position_ids,
            text_padding_mask=text_padding_mask,
            vision_padding_mask=vision_padding_mask,
            prefix_attention_mask=prefix_attention_mask,
            text_position_embeddings=text_position_embeddings,
            visual_position_embeddings=visual_position_embeddings,
        )
        h = run_qwen_layer_from_attention_output_for_layer(layer, h, text_attention, None)
        if states is not None:
            state_idx = layer_idx + 1
            if collect_state_indices is None:
                states.append(h)
            elif state_idx in collect_state_indices:
                states[state_idx] = h
    logits = qwen_lm_head_logits(model, language_model, h, text_mask, logits_to_keep=logits_to_keep)
    return logits, text_mask, states


def load_qwen_visual_delta_checkpoint(
    checkpoint_path: str | Path,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[QwenVisualDeltaAdapter, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    mode = str(checkpoint_args.get("output_mode", "native_visual_kv_injection"))
    if mode not in QWEN_VISUAL_DELTA_MODES:
        raise ValueError(f"checkpoint output_mode={mode!r} is not a Qwen visual-delta mode")
    adapter = QwenVisualDeltaAdapter.from_language_model(
        language_model,
        mode=mode,
        visual_adapter_rank=int(
            checkpoint_args.get(
                "visual_adapter_rank",
                checkpoint_args.get("visual_transform_rank", 128),
            )
        ),
    ).to(device=device, dtype=dtype)
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    meta = {
        "args": checkpoint_args,
        "global_step": checkpoint.get("global_step", checkpoint.get("step", None)) if isinstance(checkpoint, dict) else None,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
    return adapter, meta
