"""Differentiable FA2 bridge for the archived document adapter trainer (batch1)."""
import torch
from src.qwen_adapter_fa2 import prefix_plan, attention_heads


class DocumentTrainingAttention:
    def __init__(self):
        self.plan = None
        self.calls = 0

    def prepare(self, inputs):
        mask = inputs['attention_mask'].bool()
        assert mask.shape[0] == 1 and bool(mask.all())
        visual = inputs['mm_token_type_ids'].ne(0)
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        text_positions = positions[~visual].unsqueeze(0)
        image_positions = positions[visual].unsqueeze(0)
        if self.calls == 0:
            self.expected_mask = torch.cat([image_positions, text_positions], dim=1)[:, None, None, :] <= text_positions[:, None, :, None]
        self.plan = prefix_plan(text_positions, image_positions,
            torch.ones_like(text_positions, dtype=torch.bool), torch.ones_like(image_positions, dtype=torch.bool))

    def __call__(self, query, visual_key, visual_value, text_key, text_value, *, scaling, attention_mask):
        assert self.plan is not None and query.shape[0] == 1
        if self.calls == 0:
            assert torch.equal(attention_mask, self.expected_mask), 'FA2 plan differs from original causal mask'
            del self.expected_mask
        self.calls += 1
        return attention_heads(query, torch.cat([visual_key, text_key], dim=2),
            torch.cat([visual_value, text_value], dim=2), scaling=scaling, plan=self.plan)
