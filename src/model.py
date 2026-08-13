"""Vision KV Adapter: inject vision encoder KV cache into LLM layers."""
from __future__ import annotations

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
) -> tuple[PerLayerKVAdapter, list[int], dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
QWEN_VISUAL_DELTA_MODES = ("native_visual_kv_injection", "native_visual_kv_split")


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def qwen_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
        image_path = Path(str(row["image"]))
        if image_root is not None and not image_path.is_absolute():
            image_path = image_root / image_path
        question = str(row["question"]).strip()
        if answer_instruction:
            question = f"{question}\n{answer_instruction.strip()}"
        prompt = qwen_prompt(processor, question)
        if include_answers:
            answer = str(row.get("answer", "")).strip()
            suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
            texts.append(f"{prompt}{suffix}")
            answer_token_lens.append(len(processor.tokenizer(suffix, add_special_tokens=False).input_ids))
        else:
            texts.append(prompt)
            answer_token_lens.append(0)
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


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch, seq_len, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError(f"hidden dim {dim} is not divisible by num_heads={num_heads}")
    return x.view(batch, seq_len, num_heads, dim // num_heads)


def merge_heads(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError("expected [batch, seq, heads, head_dim]")
    return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])


class ReaderMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, activation: str = "situ_glu") -> None:
        super().__init__()
        if activation not in {"gelu", "silu", "swiglu", "situ_glu"}:
            raise ValueError(f"unsupported reader activation: {activation}")
        self.activation = activation
        self.norm = nn.LayerNorm(hidden_size)
        if activation in {"swiglu", "situ_glu"}:
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
        return 100.0 * torch.tanh(F.silu(x) / 100.0)

    @staticmethod
    def _soft_cap(x: Tensor) -> Tensor:
        return 100.0 * torch.tanh(x / 100.0)

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
        hidden = self._situ(gate) * self._soft_cap(up) if self.activation == "situ_glu" else F.silu(gate) * up
        return self.down_proj(hidden)


class QwenVisualDeltaAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mode: str,
        reader_mlp_ratio: float = 4.0,
        reader_activation: str = "situ_glu",
        visual_adapter_rank: int = 128,
        gate_init: float = 1.0,
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
        self.reader_mlp_ratio = float(reader_mlp_ratio)
        self.reader_activation = reader_activation
        self.visual_adapter_rank = int(visual_adapter_rank)
        self.reader_norm = nn.LayerNorm(self.attn_dim)
        self.reader_fusion_proj = nn.Linear(self.attn_dim * 2, self.attn_dim, bias=False)
        mlp_dim = max(self.attn_dim, int(round(self.attn_dim * float(reader_mlp_ratio))))
        self.reader_mlp = ReaderMLP(self.attn_dim, mlp_dim, reader_activation) if reader_mlp_ratio > 0 else None
        self.mass_head = nn.Linear(self.attn_dim, num_heads, bias=True)
        rank = max(1, int(visual_adapter_rank))
        self.visual_adapter_down = nn.ModuleList([nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers)])
        self.visual_adapter_up = nn.ModuleList([nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers)])
        self.gate = nn.Parameter(torch.full((num_layers,), float(gate_init)))
        self.last_visual_mass: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.mass_head.weight)
        nn.init.constant_(self.mass_head.bias, -2.0)
        for up in self.visual_adapter_up:
            nn.init.zeros_(up.weight)

    @classmethod
    def from_language_model(
        cls,
        language_model: torch.nn.Module,
        *,
        mode: str,
        reader_mlp_ratio: float = 4.0,
        reader_activation: str = "situ_glu",
        visual_adapter_rank: int = 128,
    ) -> "QwenVisualDeltaAdapter":
        cfg = language_model.config
        return cls(
            hidden_size=int(cfg.hidden_size),
            num_layers=len(language_model.layers),
            num_heads=int(cfg.num_attention_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            mode=mode,
            reader_mlp_ratio=reader_mlp_ratio,
            reader_activation=reader_activation,
            visual_adapter_rank=visual_adapter_rank,
        )

    def visual_memory_for_layer(self, visual_memory: Tensor, layer_idx: int) -> Tensor:
        adapted = self.visual_adapter_down[layer_idx](visual_memory)
        adapted = self.visual_adapter_up[layer_idx](F.silu(adapted))
        return visual_memory + adapted.to(dtype=visual_memory.dtype)

    def split_delta(
        self,
        *,
        query_heads: Tensor,
        visual_heads: Tensor,
        text_heads: Tensor,
        output_projection: nn.Module,
        layer_idx: int,
    ) -> Tensor:
        self.last_visual_mass = None
        visual_merged = merge_heads(visual_heads)
        query_merged = merge_heads(query_heads.transpose(1, 2).contiguous())
        reader_input = self.reader_fusion_proj(torch.cat([visual_merged, query_merged], dim=-1))
        reader_features = self.reader_norm(reader_input)
        if self.reader_mlp is not None:
            reader_features = reader_features + self.reader_mlp(reader_features)
        visual_mass = torch.sigmoid(self.mass_head(reader_features)).to(dtype=visual_heads.dtype).unsqueeze(-1)
        self.last_visual_mass = visual_mass.squeeze(-1)
        residual_heads = visual_mass * (visual_heads - text_heads.to(dtype=visual_heads.dtype))
        residual = output_projection(merge_heads(residual_heads))
        gate = self.gate[layer_idx].to(device=residual.device, dtype=residual.dtype).view(1, 1, 1)
        return residual * gate


def qwen_text_query_heads(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    query, _ = qwen_apply_rotary_pos_emb(query, query, *language_model.rotary_emb(normed, position_ids))
    return query.contiguous()


def qwen_text_attention_heads(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    if padding_mask is None:
        valid_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
    else:
        valid_mask = ~padding_mask.to(device=hidden_states.device, dtype=torch.bool)
    seq_len = hidden_states.shape[1]
    causal = torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool).tril()
    attention_mask = causal.view(1, 1, seq_len, seq_len) & valid_mask.view(valid_mask.shape[0], 1, 1, seq_len)
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    query, key = qwen_apply_rotary_pos_emb(query, key, *language_model.rotary_emb(normed, position_ids))
    key = qwen_repeat_kv(key, int(attn.num_key_value_groups))
    value = qwen_repeat_kv(value, int(attn.num_key_value_groups))
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(attn.scaling),
    )
    return out.transpose(1, 2).contiguous()


def qwen_text_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
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
) -> tuple[Tensor, Tensor, Tensor | None]:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    key = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    _, key = qwen_apply_rotary_pos_emb(key, key, *language_model.rotary_emb(normed, visual_position_ids))
    key = qwen_repeat_kv(key, int(attn.num_key_value_groups)).contiguous()
    value = qwen_repeat_kv(value, int(attn.num_key_value_groups)).contiguous()
    return key, value, padding_mask


def qwen_visual_attention_heads(
    query_heads: Tensor,
    visual_key: Tensor,
    visual_value: Tensor,
    visual_padding_mask: Tensor | None,
    *,
    scaling: float,
) -> Tensor:
    attn_mask = None
    if visual_padding_mask is not None:
        attn_mask = torch.zeros(
            (query_heads.shape[0], 1, 1, visual_key.shape[2]),
            device=query_heads.device,
            dtype=query_heads.dtype,
        )
        attn_mask = attn_mask.masked_fill(
            visual_padding_mask[:, None, None, :],
            torch.finfo(query_heads.dtype).min,
        )
    out = F.scaled_dot_product_attention(
        query_heads,
        visual_key,
        visual_value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(scaling),
    )
    return out.transpose(1, 2).contiguous()


def qwen_text_attention_output_with_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    text_padding_mask: Tensor | None = None,
    vision_padding_mask: Tensor | None = None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    normed_text = layer.input_layernorm(hidden_states)
    text_shape = normed_text.shape[:-1]
    hidden_shape = (*text_shape, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_key = attn.k_norm(attn.k_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
    query, text_key = qwen_apply_rotary_pos_emb(query, text_key, *language_model.rotary_emb(normed_text, position_ids))

    visual_key, visual_value, _ = qwen_native_visual_kv(
        language_model,
        layer_idx,
        vision_states,
        visual_position_ids,
        vision_padding_mask,
    )
    text_key = qwen_repeat_kv(text_key, int(attn.num_key_value_groups))
    text_value = qwen_repeat_kv(text_value, int(attn.num_key_value_groups))
    key = torch.cat([visual_key, text_key], dim=2)
    value = torch.cat([visual_value, text_value], dim=2)

    batch, text_len = hidden_states.shape[:2]
    visual_len = vision_states.shape[1]
    device = hidden_states.device
    valid_text = torch.ones((batch, text_len), device=device, dtype=torch.bool)
    if text_padding_mask is not None:
        valid_text = ~text_padding_mask.to(device=device, dtype=torch.bool)
    valid_visual = torch.ones((batch, visual_len), device=device, dtype=torch.bool)
    if vision_padding_mask is not None:
        valid_visual = ~vision_padding_mask.to(device=device, dtype=torch.bool)
    causal = torch.ones((text_len, text_len), device=device, dtype=torch.bool).tril()
    text_allowed = causal.view(1, text_len, text_len) & valid_text.view(batch, 1, text_len)
    visual_allowed = valid_visual.view(batch, 1, visual_len).expand(batch, text_len, visual_len)
    attention_mask = torch.cat([visual_allowed, text_allowed], dim=-1).unsqueeze(1)
    heads = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(attn.scaling),
    ).transpose(1, 2).contiguous()
    return attn.o_proj(heads.reshape(*text_shape, -1).contiguous())


def run_qwen_layer_from_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_output: Tensor,
    attention_delta: Tensor | None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
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
    text_padding_mask = ~text_mask
    visual_memory = gather_batched_positions(initial_hidden, image_pos, image_mask).to(
        dtype=next(adapter.parameters()).dtype
    )
    h = gather_batched_positions(initial_hidden, text_pos, text_mask).to(dtype=next(adapter.parameters()).dtype)
    states = [h] if collect_states else None
    for layer_idx, layer in enumerate(language_model.layers):
        vision_states = adapter.visual_memory_for_layer(visual_memory, layer_idx)
        if adapter.mode == "native_visual_kv_injection":
            text_attention = qwen_text_attention_output_with_visual_kv(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                vision_states,
                visual_position_ids,
                text_padding_mask=text_padding_mask,
                vision_padding_mask=~image_mask,
            )
            delta = None
        else:
            text_attention = qwen_text_attention_output(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
            text_heads = qwen_text_attention_heads(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
            query_heads = qwen_text_query_heads(language_model, layer_idx, h, text_position_ids)
            visual_key, visual_value, visual_padding = qwen_native_visual_kv(
                language_model,
                layer_idx,
                vision_states,
                visual_position_ids,
                ~image_mask,
            )
            visual_heads = qwen_visual_attention_heads(
                query_heads,
                visual_key,
                visual_value,
                visual_padding,
                scaling=float(layer.self_attn.scaling),
            )
            delta = adapter.split_delta(
                query_heads=query_heads,
                visual_heads=visual_heads,
                text_heads=text_heads,
                output_projection=layer.self_attn.o_proj,
                layer_idx=layer_idx,
            )
            delta = delta.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        h = run_qwen_layer_from_attention_output(language_model, layer_idx, h, text_attention, delta)
        if states is not None:
            states.append(h)
    logits = model.lm_head(language_model.norm(h))
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
    mode = str(checkpoint_args.get("output_mode", "native_visual_kv_split"))
    if mode not in QWEN_VISUAL_DELTA_MODES:
        raise ValueError(f"checkpoint output_mode={mode!r} is not a Qwen visual-delta mode")
    adapter = QwenVisualDeltaAdapter.from_language_model(
        language_model,
        mode=mode,
        reader_mlp_ratio=float(checkpoint_args.get("reader_mlp_ratio", 4.0)),
        reader_activation=str(checkpoint_args.get("reader_activation", "situ_glu")),
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
