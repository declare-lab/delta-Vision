from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LCDConfig:
    """Paper Sec. 3.4 settings. Ratios mean *fraction pruned*, not retained."""

    min_compression_ratio: float = 0.20
    max_compression_ratio: float = 0.95
    teacher_gap_start: float = 0.10
    teacher_gap_end: float = 0.30
    shallowest_layer: int = 2
    temperature: float = 2.0
    distill_weight: float = 0.70
    pruning_method: str = "dart"
    min_tokens_to_keep: int = 1

    def validate(self, num_layers: int | None = None) -> None:
        for name in ("min_compression_ratio", "max_compression_ratio", "teacher_gap_start",
                     "teacher_gap_end", "distill_weight"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.min_compression_ratio > self.max_compression_ratio:
            raise ValueError("min_compression_ratio cannot exceed max_compression_ratio")
        if self.pruning_method not in {"dart", "random"}:
            raise ValueError("pruning_method must be 'dart' or 'random'")
        if num_layers is not None and not 0 <= self.shallowest_layer < num_layers:
            raise ValueError(f"shallowest_layer must be in [0, {num_layers - 1}]")

    @property
    def final_keep_ratio(self) -> float:
        return 1.0 - self.max_compression_ratio


class LCDSchedule:
    def __init__(self, config: LCDConfig, num_layers: int, seed: int = 42):
        config.validate(num_layers)
        self.config = config
        self.num_layers = num_layers
        self.rng = random.Random(seed)

    def at(self, step: int, max_steps: int) -> dict[str, float | int]:
        beta = min(max(step / max(max_steps, 1), 0.0), 1.0)
        # Paper Eq. (9), converted to zero-based layer indices. Python round uses
        # bankers rounding, so floor(x + .5) implements mathematical Round.
        deepest = self.num_layers - 1
        layer = math.floor(deepest - beta * (deepest - self.config.shallowest_layer) + 0.5)
        student = self.rng.uniform(
            self.config.min_compression_ratio, self.config.max_compression_ratio
        )
        gap = self.config.teacher_gap_start + beta * (
            self.config.teacher_gap_end - self.config.teacher_gap_start
        )
        return {
            "progress": beta,
            "layer": layer,
            "student_ratio": student,
            "teacher_ratio": max(0.0, student - gap),
            "teacher_gap": gap,
        }


def select_visual_tokens(hidden: torch.Tensor, visual_mask: torch.Tensor, compression_ratio: float,
                         method: str = "dart", min_keep: int = 1) -> torch.Tensor:
    """Return sorted sequence indices; currently requires per-device batch size one.

    DART is redundancy-based: greedy farthest-point selection in normalized feature
    space. It retains diverse tokens and adds no trainable model component.
    """
    if hidden.shape[0] != 1:
        raise ValueError("LCD pruning requires per_device_train_batch_size=1; use gradient accumulation")
    visual_idx = visual_mask[0].nonzero(as_tuple=False).flatten()
    n_visual = visual_idx.numel()
    keep_n = min(n_visual, max(min_keep, int(round(n_visual * (1.0 - compression_ratio)))))
    if keep_n >= n_visual:
        return torch.arange(hidden.shape[1], device=hidden.device)
    if method == "random":
        selected = visual_idx[torch.randperm(n_visual, device=hidden.device)[:keep_n]]
    else:
        features = F.normalize(hidden[0, visual_idx].float(), dim=-1)
        # Start at the token least similar to the global centroid, then greedily
        # maximize distance to the selected set. This is a memory-bounded DART-style
        # diversity selector rather than materializing an NxN similarity matrix.
        centroid = F.normalize(features.mean(0, keepdim=True), dim=-1)
        first = (features @ centroid.T).squeeze(-1).argmin()
        chosen = [first]
        best_similarity = features @ features[first]
        for _ in range(1, keep_n):
            nxt = best_similarity.argmin()
            chosen.append(nxt)
            best_similarity = torch.maximum(best_similarity, features @ features[nxt])
        selected = visual_idx[torch.stack(chosen)]
    non_visual = (~visual_mask[0]).nonzero(as_tuple=False).flatten()
    return torch.cat((non_visual, selected)).sort().values


def masked_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
              temperature: float) -> torch.Tensor:
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    t = float(temperature)
    teacher_probs = F.softmax(teacher_logits.detach().float() / t, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / t, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)

