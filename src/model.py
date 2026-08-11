"""Vision KV Adapter: inject vision encoder KV cache into LLM layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class PerLayerKVAdapter(nn.Module):
    """Per-LLM-layer adapter that maps vision encoder KV to LLM KV space.

    For each of the 32 LLM layers:
      - Learns a soft mixture over source_layers (last 2 ViT layers)
      - Projects mixed source K and V to LLM dim via independent linear layers
      - gate scalar controls injection strength (init=0 for gradual warmup)
    """

    def __init__(
        self,
        num_llm_layers: int = 32,
        num_source_layers: int = 2,
        source_dim: int = 1024,
        num_heads: int = 32,
        head_dim: int = 128,
        bottleneck_dim: int = 0,
    ):
        super().__init__()
        self.num_llm_layers = num_llm_layers
        self.num_source_layers = num_source_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bottleneck_dim = bottleneck_dim
        target_dim = num_heads * head_dim

        self.source_mix = nn.Parameter(torch.zeros(num_llm_layers, num_source_layers))

        if bottleneck_dim > 0:
            self.k_down = nn.ModuleList([nn.Linear(source_dim, bottleneck_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_up = nn.ModuleList([nn.Linear(bottleneck_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_down = nn.ModuleList([nn.Linear(source_dim, bottleneck_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_up = nn.ModuleList([nn.Linear(bottleneck_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_projs = None
            self.v_projs = None
        else:
            self.k_projs = nn.ModuleList([nn.Linear(source_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.v_projs = nn.ModuleList([nn.Linear(source_dim, target_dim, bias=True) for _ in range(num_llm_layers)])
            self.k_down = None
            self.k_up = None
            self.v_down = None
            self.v_up = None

        self.gates = nn.Parameter(torch.zeros(num_llm_layers))
        self._init_weights()

    def _init_weights(self):
        if self.k_projs is not None:
            for proj in list(self.k_projs) + list(self.v_projs):
                nn.init.xavier_normal_(proj.weight, gain=0.01)
                nn.init.zeros_(proj.bias)
        else:
            for proj in list(self.k_down) + list(self.v_down):
                nn.init.xavier_normal_(proj.weight, gain=0.02)
                nn.init.zeros_(proj.bias)
            for proj in list(self.k_up) + list(self.v_up):
                nn.init.xavier_normal_(proj.weight, gain=0.01)
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
        weights = F.softmax(self.source_mix[layer_idx].float(), dim=-1)
        weights = weights.to(source_k.dtype)

        mixed_k = torch.einsum("s,bsnd->bnd", weights, source_k)
        mixed_v = torch.einsum("s,bsnd->bnd", weights, source_v)

        B, N, _ = mixed_k.shape
        gate = torch.sigmoid(self.gates[layer_idx])

        if self.k_projs is not None:
            key = self.k_projs[layer_idx](mixed_k)
            value = self.v_projs[layer_idx](mixed_v)
        else:
            key = self.k_up[layer_idx](F.silu(self.k_down[layer_idx](mixed_k)))
            value = self.v_up[layer_idx](F.silu(self.v_down[layer_idx](mixed_v)))

        key = key.view(B, N, self.num_heads, self.head_dim) * gate
        value = value.view(B, N, self.num_heads, self.head_dim) * gate

        return key, value


def _get_vision_tower(model: LlavaForConditionalGeneration):
    return model.model.vision_tower


def _get_language_model(model: LlavaForConditionalGeneration):
    return model.model.language_model


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

    hidden = vision_model.embeddings(pixel_values)
    hidden = vision_model.pre_layrnorm(hidden)

    layers = vision_model.encoder.layers
    num_layers = len(layers)
    wanted = {(idx % num_layers) if idx < 0 else idx for idx in source_layer_indices}

    collected_k = {}
    collected_v = {}

    for idx, layer in enumerate(layers):
        normed = layer.layer_norm1(hidden)
        if idx in wanted:
            collected_k[idx] = layer.self_attn.k_proj(normed)[:, 1:].float()
            collected_v[idx] = layer.self_attn.v_proj(normed)[:, 1:].float()
        layer_out = layer(hidden, attention_mask=None, causal_attention_mask=None)
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    ordered_indices = sorted(wanted)
    source_k = torch.stack([collected_k[i] for i in ordered_indices], dim=1)
    source_v = torch.stack([collected_v[i] for i in ordered_indices], dim=1)
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
) -> torch.Tensor:
    """Run LLM forward with adapter-predicted visual KV instead of image token embeddings.

    Text tokens only go through the LLM; visual information enters via
    concatenated KV in each attention layer.

    Returns:
        logits: [B, text_seq_len, vocab_size]
    """
    text_mask = input_ids[0] != image_token_id
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

    text_position_ids = torch.arange(N_vis, N_vis + T, device=device).unsqueeze(0)
    image_position_ids = torch.arange(N_vis, device=device).unsqueeze(0)

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

        # Build causal mask: text can attend to all image tokens + causally to text
        text_pos = text_position_ids[0]
        img_pos = image_position_ids[0]
        # image tokens are always visible to all text (they precede text)
        img_allowed = torch.ones(T, N_vis, device=device, dtype=torch.bool)
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
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard LLaVA forward to get teacher logits."""
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    return outputs.logits


def load_frozen_llava(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple:
    """Load LLaVA model with all parameters frozen."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
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
            final_embeds[i, img_positions[:n_image]] = image_features[i, :img_positions.numel()].to(final_embeds.dtype)

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
            img_position_ids = img_pos.unsqueeze(0)
            adapter_k_roped = _apply_rope(rotary_emb, vis_k_t[i:i+1], img_position_ids, normed[i:i+1])
            k[i:i+1, :, img_pos] = k[i:i+1, :, img_pos] + adapter_k_roped
            v[i:i+1, :, img_pos] = v[i:i+1, :, img_pos] + vis_v_t[i:i+1]

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
) -> torch.Tensor:
    """Optimized forward using flash attention via Q-padding trick.

    Pads Q with zeros to match KV length so is_causal=True (flash attn) works.
    The padded output rows are discarded.
    """
    text_mask = input_ids[0] != image_token_id
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

    text_position_ids = torch.arange(N_vis, N_vis + T, device=device).unsqueeze(0)
    image_position_ids = torch.arange(N_vis, device=device).unsqueeze(0)
    full_position_ids = torch.arange(N_vis + T, device=device).unsqueeze(0)

    # Pre-compute adapter KV for all layers
    src_k_cast = source_k.to(dtype=dtype)
    src_v_cast = source_v.to(dtype=dtype)

    all_vis_k = []
    all_vis_v = []
    dummy_ref = text_embeds[:, :1, :]
    cos_img, sin_img = rotary_emb(dummy_ref, image_position_ids)
    for layer_idx in range(len(layers)):
        vis_k, vis_v = adapter.forward_layer(src_k_cast, src_v_cast, layer_idx)
        vis_k = vis_k.transpose(1, 2)  # [B, heads, N_vis, head_dim]
        vis_v = vis_v.transpose(1, 2)
        vis_k, _ = apply_rotary_pos_emb(vis_k, vis_k, cos_img, sin_img)
        all_vis_k.append(vis_k)
        all_vis_v.append(vis_v)

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

        cos_txt, sin_txt = rotary_emb(normed, text_position_ids)
        q, text_k = apply_rotary_pos_emb(q, text_k, cos_txt, sin_txt)

        # K, V: [vis | text] = N_vis + T columns
        k = torch.cat([all_vis_k[layer_idx], text_k], dim=2)
        v = torch.cat([all_vis_v[layer_idx], text_v], dim=2)

        # Pad Q with zeros to length N_vis + T so we can use is_causal=True (flash attn)
        # Padded Q rows (0..N_vis-1) produce garbage output that we discard
        q_pad = torch.zeros(B, q.shape[1], N_vis, q.shape[3], device=device, dtype=dtype)
        q_full = torch.cat([q_pad, q], dim=2)  # [B, heads, N_vis+T, head_dim]

        attn_out_full = F.scaled_dot_product_attention(
            q_full, k, v,
            dropout_p=0.0,
            is_causal=True,
        )
        # Slice out only the text rows
        attn_out = attn_out_full[:, :, N_vis:, :]

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_out = attn.o_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits


def student_forward_flex(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    adapter: PerLayerKVAdapter,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Optimized forward using flex_attention (no Q-padding waste).

    flex_attention compiles a custom prefix-causal mask into an efficient kernel,
    computing attention only for the 67 actual Q rows instead of padding to 643.
    """
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    text_mask = input_ids[0] != image_token_id
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

    text_position_ids = torch.arange(N_vis, N_vis + T, device=device).unsqueeze(0)
    image_position_ids = torch.arange(N_vis, device=device).unsqueeze(0)

    # Create prefix-causal block mask once
    def prefix_causal(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + N_vis

    block_mask = create_block_mask(prefix_causal, B=1, H=1, Q_LEN=T, KV_LEN=N_vis + T, device=device)

    # Pre-compute adapter KV
    src_k_cast = source_k.to(dtype=dtype)
    src_v_cast = source_v.to(dtype=dtype)

    all_vis_k = []
    all_vis_v = []
    dummy_ref = text_embeds[:, :1, :]
    cos_img, sin_img = rotary_emb(dummy_ref, image_position_ids)
    for layer_idx in range(len(layers)):
        vis_k, vis_v = adapter.forward_layer(src_k_cast, src_v_cast, layer_idx)
        vis_k = vis_k.transpose(1, 2)
        vis_v = vis_v.transpose(1, 2)
        vis_k, _ = apply_rotary_pos_emb(vis_k, vis_k, cos_img, sin_img)
        all_vis_k.append(vis_k)
        all_vis_v.append(vis_v)

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

        cos_txt, sin_txt = rotary_emb(normed, text_position_ids)
        q, text_k = apply_rotary_pos_emb(q, text_k, cos_txt, sin_txt)

        k = torch.cat([all_vis_k[layer_idx], text_k], dim=2)
        v = torch.cat([all_vis_v[layer_idx], text_v], dim=2)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_out = attn.o_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits
