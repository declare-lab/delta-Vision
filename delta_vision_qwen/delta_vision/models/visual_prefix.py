from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VisualPrefixCompressor(nn.Module):
    """Compress variable-length visual tokens into a fixed small visual prefix."""

    def __init__(self, hidden_size: int, num_tokens: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_size) * 0.02)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, visual_tokens: Tensor, visual_mask: Tensor | None = None) -> Tensor:
        batch = visual_tokens.shape[0]
        queries = self.queries.to(device=visual_tokens.device, dtype=visual_tokens.dtype).expand(batch, -1, -1)
        q = self.q_proj(queries).view(batch, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual_tokens).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_tokens).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if visual_mask is not None:
            invalid = ~visual_mask.to(device=visual_tokens.device).bool()
            attn_mask = invalid.unsqueeze(1).unsqueeze(2)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(batch, self.num_tokens, self.hidden_size)
        return self.norm(self.out_proj(attn_out) + queries)


def grid_position_ids(batch: int, num_tokens: int, device: torch.device) -> Tensor:
    width = int(math.ceil(math.sqrt(num_tokens)))
    height = int(math.ceil(num_tokens / width))
    h_pos = torch.arange(height, device=device).unsqueeze(1).expand(height, width).reshape(-1)[:num_tokens]
    w_pos = torch.arange(width, device=device).unsqueeze(0).expand(height, width).reshape(-1)[:num_tokens]
    t_pos = torch.zeros(num_tokens, device=device, dtype=torch.long)
    return torch.stack([t_pos, h_pos.long(), w_pos.long()], dim=0).unsqueeze(1).expand(3, batch, num_tokens)
