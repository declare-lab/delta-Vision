"""Run test-only recurrent Qwen embedding_adapter training.

This wrapper monkeypatches QwenEmbeddingAdapter inside the current Python process so
diagnostic recurrent training can run without changing src/ or scripts/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import QwenEmbeddingAdapter  # noqa: E402
from src.train import main  # noqa: E402


def recurrent_all_visual_memories_batched(self: QwenEmbeddingAdapter, visual_memory: torch.Tensor) -> torch.Tensor:
    memories = []
    current = visual_memory
    for layer_idx in range(self.num_layers):
        current = self.visual_memory_for_layer(current, layer_idx)
        memories.append(current)
    return torch.stack(memories, dim=0)


QwenEmbeddingAdapter.all_visual_memories_batched = recurrent_all_visual_memories_batched  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
