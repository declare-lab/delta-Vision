from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


def parse_layers(spec: str, available_layers: list[int]) -> list[int]:
    if spec == "all":
        return available_layers
    requested = [int(x) for x in spec.split(",") if x.strip()]
    missing = sorted(set(requested) - set(available_layers))
    if missing:
        raise ValueError(f"requested layers are missing from effects: {missing}")
    return requested


class EffectDataset(Dataset[dict[str, Any]]):
    """Flattened layer-wise teacher-effect dataset.

    Each item corresponds to one `(sample, layer)` pair and contains:

    - teacher text hidden state `H_l^T`
    - static visual tokens `V_0`
    - teacher effect residual `Delta_l^T`
    """

    def __init__(
        self,
        effects_dir: str | Path,
        layers: str = "all",
        max_items: int | None = None,
        start_item: int = 0,
        mmap: bool = False,
    ) -> None:
        self.effects_dir = Path(effects_dir)
        manifest_path = self.effects_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        sample_names = self.manifest.get("samples", [])
        if not sample_names:
            raise ValueError(f"manifest has no samples: {manifest_path}")

        available_layers = [int(x) for x in self.manifest["layers"]]
        selected_layers = parse_layers(layers, available_layers)
        self.sample_files = [self.effects_dir / name for name in sample_names]
        self.items = [
            (sample_idx, layer_idx)
            for sample_idx in range(len(self.sample_files))
            for layer_idx in selected_layers
        ]
        if start_item < 0:
            raise ValueError("start_item must be non-negative")
        self.items = self.items[start_item:]
        if max_items is not None:
            self.items = self.items[:max_items]
        self.mmap = mmap
        self._cache_sample_idx: int | None = None
        self._cache_item: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx, layer_idx = self.items[index]
        if self._cache_sample_idx == sample_idx and self._cache_item is not None:
            item = self._cache_item
        else:
            load_kwargs = {"map_location": "cpu"}
            if self.mmap:
                load_kwargs["mmap"] = True
            item = torch.load(self.sample_files[sample_idx], **load_kwargs)
            self._cache_sample_idx = sample_idx
            self._cache_item = item
        teacher_hiddens = item.get("teacher_hiddens", {})
        if layer_idx not in teacher_hiddens:
            raise KeyError(
                f"sample {self.sample_files[sample_idx]} has no teacher hidden for layer {layer_idx}; "
                "run extraction with --store-hidden"
            )
        return {
            "hidden_states": teacher_hiddens[layer_idx].float(),
            "vision_states": item["vision_tokens"].float(),
            "target_delta": item["deltas"][layer_idx].float(),
            "layer_idx": layer_idx,
            "sample_idx": sample_idx,
        }


def pad_2d(sequences: list[Tensor]) -> tuple[Tensor, Tensor]:
    if not sequences:
        raise ValueError("empty sequence list")
    max_len = max(x.shape[0] for x in sequences)
    hidden = sequences[0].shape[1]
    out = sequences[0].new_zeros((len(sequences), max_len, hidden))
    padding_mask = torch.ones((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        out[i, : seq.shape[0]] = seq
        padding_mask[i, : seq.shape[0]] = False
    return out, padding_mask


def collate_effects(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    hidden_states, text_padding_mask = pad_2d([x["hidden_states"] for x in batch])
    target_delta, _ = pad_2d([x["target_delta"] for x in batch])
    vision_states, vision_padding_mask = pad_2d([x["vision_states"] for x in batch])
    return {
        "hidden_states": hidden_states,
        "target_delta": target_delta,
        "vision_states": vision_states,
        "text_padding_mask": text_padding_mask,
        "vision_padding_mask": vision_padding_mask,
        "layer_idx": torch.tensor([x["layer_idx"] for x in batch], dtype=torch.long),
        "sample_idx": torch.tensor([x["sample_idx"] for x in batch], dtype=torch.long),
    }
