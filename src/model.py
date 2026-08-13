"""Vision KV Adapter: inject vision encoder KV cache into LLM layers."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from torch.nn.attention.bias import causal_lower_right
except Exception:  # pragma: no cover - older torch fallback
    causal_lower_right = None

QWEN_SOURCE_RAW_SPATIAL_CONCAT = "raw_spatial_concat"


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
    rotated, _ = apply_rotary_pos_emb(states, states, cos, sin)
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
            k = repeat_kv(k, num_q_heads // num_kv_heads)
            v = repeat_kv(v, num_q_heads // num_kv_heads)

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


# === Qwen3-VL Support ===

def _get_qwen_vision(model):
    return model.model.visual


def qwen_source_dim(model, source_mode: str) -> int:
    visual = model.model.visual
    if source_mode == QWEN_SOURCE_RAW_SPATIAL_CONCAT:
        hidden_size = int(getattr(visual.config, "hidden_size"))
        merge_unit = int(getattr(visual, "spatial_merge_unit", getattr(visual, "spatial_merge_size", 2) ** 2))
        return hidden_size * merge_unit
    raise ValueError(f"Unknown Qwen source_mode={source_mode!r}; expected {QWEN_SOURCE_RAW_SPATIAL_CONCAT!r}")


@torch.no_grad()
def extract_vision_kv_qwen(
    model,
    pixel_values,
    grid_thw,
    source_layer_indices=[22, 23],
    source_mode: str = QWEN_SOURCE_RAW_SPATIAL_CONCAT,
):
    """Extract Qwen3-VL visual source features for the adapter.

    source_mode="raw_spatial_concat" hooks raw ViT attention K/V projections
    and concatenates each spatial_merge_size**2 patch group. This matches the
    LLM image-token count without passing source features through Qwen's
    learned PatchMerger.
    """
    visual = model.model.visual
    num_blocks = len(visual.blocks)
    wanted = _normalize_layer_indices(source_layer_indices, num_blocks)
    if source_mode != QWEN_SOURCE_RAW_SPATIAL_CONCAT:
        raise ValueError(f"Unknown Qwen source_mode={source_mode!r}; expected {QWEN_SOURCE_RAW_SPATIAL_CONCAT!r}")

    qkv_outputs = {}

    def make_qkv_hook(layer_idx):
        def hook_fn(module, input, output):
            _, k, v = output.chunk(3, dim=-1)
            qkv_outputs[layer_idx] = (k.float().detach(), v.float().detach())
        return hook_fn

    hooks = []
    for idx in wanted:
        hooks.append(visual.blocks[idx].attn.qkv.register_forward_hook(make_qkv_hook(idx)))

    try:
        visual(pixel_values, grid_thw=grid_thw)
    finally:
        for h in hooks:
            h.remove()

    for idx in wanted:
        if idx not in qkv_outputs:
            raise RuntimeError(f"Qwen vision layer {idx} was not captured")
    source_k = torch.stack([qkv_outputs[i][0] for i in wanted], dim=0).unsqueeze(0)
    source_v = torch.stack([qkv_outputs[i][1] for i in wanted], dim=0).unsqueeze(0)

    S = source_k.shape[1]
    N = source_k.shape[2]
    D = source_k.shape[3]
    merge_unit = int(getattr(visual, "spatial_merge_unit", getattr(visual, "spatial_merge_size", 2) ** 2))
    if N % merge_unit != 0:
        raise ValueError(f"Cannot spatial-merge {N} Qwen tokens by merge_unit={merge_unit}")
    source_k = source_k.view(1, S, N // merge_unit, merge_unit * D)
    source_v = source_v.view(1, S, N // merge_unit, merge_unit * D)
    expected_tokens = int((grid_thw.prod(dim=-1) // merge_unit).sum().item())
    if source_k.shape[2] != expected_tokens:
        raise ValueError(f"Qwen source tokens {source_k.shape[2]} != grid-derived image tokens {expected_tokens}")

    return source_k, source_v


def student_forward_qwen(
    model,
    input_ids,
    adapter,
    source_k,
    source_v,
    image_grid_thw=None,
    spatial_merge_size=1,
    full_input_ids=None,
    mm_token_type_ids=None,
    attention_mask=None,
    use_lower_right_causal=False,
):
    language_model = model.model.language_model
    layers = language_model.layers
    norm = language_model.norm
    rotary_emb = language_model.rotary_emb
    text_embeds = language_model.embed_tokens(input_ids)
    B, T, _ = text_embeds.shape
    N_vis = source_k.shape[2]
    device = text_embeds.device
    dtype = text_embeds.dtype
    if image_grid_thw is not None:
        t_dim = int(image_grid_thw[0][0])
        h_dim = int(image_grid_thw[0][1]) // spatial_merge_size
        w_dim = int(image_grid_thw[0][2]) // spatial_merge_size
        # Image positions: temporal=0+start, height=start..start+h-1, width=start..start+w-1
        # start_position = 0 (image comes first, no text before it)
        start_pos = 0
        temporal_ids = torch.zeros(N_vis, dtype=torch.long, device=device) + start_pos
        height_ids = (torch.arange(h_dim, device=device) + start_pos).repeat_interleave(w_dim).repeat(t_dim)[:N_vis]
        width_ids = (torch.arange(w_dim, device=device) + start_pos).repeat(h_dim * t_dim)[:N_vis]
        img_pos_3d = torch.stack([temporal_ids, height_ids, width_ids], dim=0).unsqueeze(1)
        img_causal_pos = torch.arange(N_vis, device=device)
        # Text positions: start at max(h_dim, w_dim) after image, all 3 dims same
        text_start = start_pos + max(h_dim, w_dim)
    else:
        img_pos_3d = torch.zeros(3, 1, N_vis, dtype=torch.long, device=device)
        img_causal_pos = torch.arange(N_vis, device=device)
        text_start = N_vis
    text_pos_1d = torch.arange(text_start, text_start + T, device=device)
    text_pos_3d = text_pos_1d.unsqueeze(0).expand(3, -1).unsqueeze(1)
    text_causal_pos = torch.arange(N_vis, N_vis + T, device=device)
    rope_dim = rotary_emb.inv_freq.shape[0] * 2
    if full_input_ids is not None and mm_token_type_ids is not None and image_grid_thw is not None:
        position_ids, _ = model.model.get_rope_index(
            full_input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        img_token_id = int(getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", 151655)))
        valid_full = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(full_input_ids[0], dtype=torch.bool)
        img_mask_full = (full_input_ids[0] == img_token_id) & valid_full
        txt_mask_full = (~img_mask_full) & valid_full
        img_positions_full = position_ids[:, 0, img_mask_full]
        # source_k already post-merge, no further downsample
        img_pos_3d = img_positions_full.unsqueeze(1)
        text_pos_3d = position_ids[:, 0, txt_mask_full].unsqueeze(1)
        full_order = torch.arange(full_input_ids.shape[1], device=device)
        img_causal_pos = full_order[img_mask_full]
        text_causal_pos = full_order[txt_mask_full]
    if img_pos_3d.shape[-1] != N_vis:
        raise ValueError(f"Qwen image RoPE position count {img_pos_3d.shape[-1]} != source tokens {N_vis}")
    if text_pos_3d.shape[-1] != T:
        raise ValueError(f"Qwen text RoPE position count {text_pos_3d.shape[-1]} != text tokens {T}")
    split_causal = False
    pre_text_len = 0
    post_text_len = T
    if use_lower_right_causal and causal_lower_right is not None and img_causal_pos.numel() > 0:
        img_start = int(img_causal_pos[0].item())
        img_end = int(img_causal_pos[-1].item())
        image_is_contiguous = bool(torch.equal(img_causal_pos, torch.arange(img_start, img_end + 1, device=device)))
        text_outside_image = bool(((text_causal_pos < img_start) | (text_causal_pos > img_end)).all().item())
        if image_is_contiguous and text_outside_image:
            pre_text_len = int((text_causal_pos < img_start).sum().item())
            post_text_len = T - pre_text_len
            split_causal = True
    cos_img, sin_img = rotary_emb(torch.zeros(1, N_vis, rope_dim, device=device), img_pos_3d)
    cos_txt, sin_txt = rotary_emb(torch.zeros(1, T, rope_dim, device=device), text_pos_3d)
    hidden = text_embeds
    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(normed)
        text_k = attn.k_proj(normed)
        text_v = attn.v_proj(normed)
        head_dim = attn.head_dim
        num_q_heads = q.shape[-1] // head_dim
        num_kv_heads = text_k.shape[-1] // head_dim
        q = q.view(B, T, num_q_heads, head_dim).transpose(1, 2)
        text_k = text_k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
        text_v = text_v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            text_k = attn.k_norm(text_k)
        q, text_k = apply_rotary_pos_emb(q, text_k, cos_txt, sin_txt)
        vis_k, vis_v = adapter.forward_layer(source_k.to(dtype), source_v.to(dtype), layer_idx)
        vis_k = vis_k.transpose(1, 2)
        vis_v = vis_v.transpose(1, 2)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            vis_k = attn.k_norm(vis_k)
        vis_k, _ = apply_rotary_pos_emb(vis_k, vis_k, cos_img, sin_img)
        if split_causal:
            attn_parts = []
            if pre_text_len > 0:
                pre_q = q[:, :, :pre_text_len].to(dtype)
                pre_k = text_k[:, :, :pre_text_len].to(dtype)
                pre_v = text_v[:, :, :pre_text_len].to(dtype)
                if num_q_heads != num_kv_heads:
                    pre_k = repeat_kv(pre_k, num_q_heads // num_kv_heads)
                    pre_v = repeat_kv(pre_v, num_q_heads // num_kv_heads)
                attn_parts.append(
                    F.scaled_dot_product_attention(pre_q, pre_k, pre_v, dropout_p=0.0, is_causal=True)
                )
            if post_text_len > 0:
                post_q = q[:, :, pre_text_len:].to(dtype)
                post_k = torch.cat(
                    [text_k[:, :, :pre_text_len], vis_k, text_k[:, :, pre_text_len:]],
                    dim=2,
                ).to(dtype)
                post_v = torch.cat(
                    [text_v[:, :, :pre_text_len], vis_v, text_v[:, :, pre_text_len:]],
                    dim=2,
                ).to(dtype)
                if num_q_heads != num_kv_heads:
                    post_k = repeat_kv(post_k, num_q_heads // num_kv_heads)
                    post_v = repeat_kv(post_v, num_q_heads // num_kv_heads)
                attn_parts.append(
                    F.scaled_dot_product_attention(
                        post_q,
                        post_k,
                        post_v,
                        attn_mask=causal_lower_right(post_text_len, pre_text_len + N_vis + post_text_len),
                        dropout_p=0.0,
                    )
                )
            attn_out = torch.cat(attn_parts, dim=2)
        else:
            k = torch.cat([vis_k, text_k], dim=2)
            v = torch.cat([vis_v, text_v], dim=2)
            if num_q_heads != num_kv_heads:
                k = repeat_kv(k, num_q_heads // num_kv_heads)
                v = repeat_kv(v, num_q_heads // num_kv_heads)
            # Preserve original prompt causality even though visual KV is prepended.
            img_allowed = text_causal_pos.unsqueeze(1) >= img_causal_pos.unsqueeze(0)
            text_allowed = text_causal_pos.unsqueeze(1) >= text_causal_pos.unsqueeze(0)
            causal_mask = torch.cat([img_allowed, text_allowed], dim=1)
            attn_mask = torch.zeros(1, 1, T, N_vis + T, device=device, dtype=dtype)
            attn_mask.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
            attn_out = F.scaled_dot_product_attention(
                q.to(dtype), k.to(dtype), v.to(dtype), attn_mask=attn_mask, dropout_p=0.0
            )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, -1)
        attn_out = attn.o_proj(attn_out)
        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits
