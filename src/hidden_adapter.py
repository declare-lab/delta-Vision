"""HiddenStateAdapter: map ViT V to LLM hidden space, let LLM k_proj/v_proj create KV."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class HiddenStateAdapter(nn.Module):
    def __init__(self, source_dim=1024, hidden_dim=2560, num_source_layers=2, num_llm_layers=36):
        super().__init__()
        self.num_llm_layers = num_llm_layers
        self.source_mix = nn.Parameter(torch.zeros(num_llm_layers, num_source_layers))
        self.projs = nn.ModuleList([nn.Linear(source_dim, hidden_dim, bias=True) for _ in range(num_llm_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_llm_layers)])
        for proj in self.projs:
            nn.init.xavier_normal_(proj.weight, gain=1.0)
            nn.init.zeros_(proj.bias)

    def forward_layer(self, source, layer_idx):
        weights = F.softmax(self.source_mix[layer_idx].float(), dim=-1).to(source.dtype)
        mixed = torch.einsum("s,bsnd->bnd", weights, source)
        return self.norms[layer_idx](self.projs[layer_idx](mixed))


def student_forward_hidden_adapter(model, input_ids, adapter, source, image_token_id, image_grid_thw=None):
    language_model = model.model.language_model
    layers = language_model.layers
    norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    text_mask = input_ids[0] != image_token_id
    text_ids = input_ids[:, text_mask]
    text_embeds = language_model.embed_tokens(text_ids)
    B, T, _ = text_embeds.shape
    device = text_embeds.device
    dtype = text_embeds.dtype

    visual_hidden_base = source.to(dtype)  # defer per-layer
    N_vis = source.shape[2]

    if image_grid_thw is not None:
        t_dim = int(image_grid_thw[0][0])
        h_dim = int(image_grid_thw[0][1])
        w_dim = int(image_grid_thw[0][2])
        temporal_ids = torch.zeros(N_vis, dtype=torch.long, device=device)
        height_ids = torch.arange(h_dim, device=device).repeat_interleave(w_dim).repeat(t_dim)[:N_vis]
        width_ids = torch.arange(w_dim, device=device).repeat(h_dim * t_dim)[:N_vis]
        img_pos_3d = torch.stack([temporal_ids, height_ids, width_ids], dim=0).unsqueeze(1)
    else:
        img_pos_3d = torch.zeros(3, 1, N_vis, dtype=torch.long, device=device)

    text_pos_1d = torch.arange(N_vis, N_vis + T, device=device)
    text_pos_3d = text_pos_1d.unsqueeze(0).expand(3, -1).unsqueeze(1)
    rope_dim = rotary_emb.inv_freq.shape[0] * 2
    cos_img, sin_img = rotary_emb(torch.zeros(1, N_vis, rope_dim, device=device), img_pos_3d)
    cos_txt, sin_txt = rotary_emb(torch.zeros(1, T, rope_dim, device=device), text_pos_3d)

    hidden = text_embeds
    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        head_dim = attn.head_dim

        q = attn.q_proj(normed)
        text_k = attn.k_proj(normed)
        text_v = attn.v_proj(normed)
        num_q_heads = q.shape[-1] // head_dim
        num_kv_heads = text_k.shape[-1] // head_dim

        q = q.view(B, T, num_q_heads, head_dim).transpose(1, 2)
        text_k = text_k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
        text_v = text_v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)

        visual_hidden = adapter.forward_layer(visual_hidden_base, layer_idx)
        vis_k = attn.k_proj(visual_hidden)
        vis_v = attn.v_proj(visual_hidden)
        vis_k = vis_k.view(B, N_vis, num_kv_heads, head_dim).transpose(1, 2)
        vis_v = vis_v.view(B, N_vis, num_kv_heads, head_dim).transpose(1, 2)

        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            text_k = attn.k_norm(text_k)
            vis_k = attn.k_norm(vis_k)

        q, text_k = apply_rotary_pos_emb(q, text_k, cos_txt, sin_txt)
        vis_k, _ = apply_rotary_pos_emb(vis_k, vis_k, cos_img, sin_img)

        k = torch.cat([vis_k, text_k], dim=2)
        v = torch.cat([vis_v, text_v], dim=2)
        if num_q_heads != num_kv_heads:
            k = repeat_kv(k, num_q_heads // num_kv_heads)
            v = repeat_kv(v, num_q_heads // num_kv_heads)

        img_allowed = torch.ones(T, N_vis, device=device, dtype=torch.bool)
        text_allowed = text_pos_1d.unsqueeze(1) >= text_pos_1d.unsqueeze(0)
        causal_mask = torch.cat([img_allowed, text_allowed], dim=1)
        attn_mask = torch.zeros(1, 1, T, N_vis + T, device=device, dtype=dtype)
        attn_mask.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

        attn_out = F.scaled_dot_product_attention(q.to(dtype), k.to(dtype), v.to(dtype), attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, -1)
        attn_out = attn.o_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    hidden = norm(hidden)
    logits = model.lm_head(hidden)
    return logits
