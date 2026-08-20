"""Test-only recurrent Qwen embedding-adapter eval entrypoint."""
from __future__ import annotations

import torch

from src.model import QwenEmbeddingAdapter


def recurrent_all_visual_memories_batched(self: QwenEmbeddingAdapter, visual_memory: torch.Tensor) -> torch.Tensor:
    memories = []
    current = visual_memory
    for layer_idx in range(self.num_layers):
        current = self.visual_memory_for_layer(current, layer_idx)
        memories.append(current)
    return torch.stack(memories, dim=0)


QwenEmbeddingAdapter.all_visual_memories_batched = recurrent_all_visual_memories_batched


if __name__ == "__main__":
    from src.eval_benchmarks import main

    main()
