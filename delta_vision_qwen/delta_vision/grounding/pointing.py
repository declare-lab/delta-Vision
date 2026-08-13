from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    gather_batched_positions,
    get_batched_text_and_image_positions,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    run_llama_layer_text_with_attention_delta,
)
from delta_vision.models.modeling import DeltaVisionModel


def point_prompt(label: str) -> str:
    return llava15_prompt(f"Point to the {str(label).strip()}.")


def parse_points(points: Any) -> Tensor:
    parsed: list[list[float]] = []
    for point in points:
        if isinstance(point, dict):
            x = float(point["x"])
            y = float(point["y"])
        else:
            x = float(point[0])
            y = float(point[1])
        if math.isfinite(x) and math.isfinite(y):
            parsed.append([max(0.0, min(100.0, x)), max(0.0, min(100.0, y))])
    if not parsed:
        raise ValueError("points must contain at least one finite point")
    return torch.tensor(parsed, dtype=torch.float32)


class PointHead(nn.Module):
    def __init__(self, hidden_size: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return torch.sigmoid(self.net(hidden.float())) * 100.0


@torch.inference_mode()
def load_point_checkpoint(
    point_head: PointHead,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("point_head", checkpoint.get("state_dict", checkpoint))
    point_head.load_state_dict(state, strict=True)
    point_head.to(device=device)
    return checkpoint


def save_point_checkpoint(
    output_dir: str | Path,
    tag: str,
    point_head: PointHead,
    metadata: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_head": point_head.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, out / f"point_head_{tag}.pt")


def point_target_loss(pred: Tensor, points: Tensor) -> Tensor:
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [num_points, 2]")
    per_point = F.smooth_l1_loss(
        pred.float().unsqueeze(0).expand(points.shape[0], -1),
        points.to(device=pred.device, dtype=pred.dtype),
        reduction="none",
        beta=2.0,
    ).mean(dim=-1)
    return per_point.min()


def point_distance(pred: Tensor, points: Tensor) -> Tensor:
    return torch.cdist(pred.float().view(1, 2), points.to(device=pred.device).float()).min()


def point_in_masks_xy100(point_xy100: tuple[float, float], mask_path: str | Path) -> bool:
    data = np.load(mask_path)
    masks = data["masks"].astype(bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.shape[0] == 0:
        return False
    height, width = int(masks.shape[-2]), int(masks.shape[-1])
    x = int(round(max(0.0, min(100.0, point_xy100[0])) / 100.0 * max(width - 1, 1)))
    y = int(round(max(0.0, min(100.0, point_xy100[1])) / 100.0 * max(height - 1, 1)))
    return bool(masks[:, y, x].any())


def read_point_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    return rows


def sidecar_point_hidden(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    image_token_id: int,
    image_path: str | Path,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=point_prompt(label), images=image, return_tensors="pt")
    image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    text_pos, image_pos, prompt_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden0.shape[1],
        image_token_id,
    )
    text_pos = text_pos.to(device)
    image_pos = image_pos.to(device)
    h = hidden0.index_select(1, text_pos).to(dtype=dtype)
    vision = hidden0.index_select(1, image_pos).to(dtype=dtype)
    position_ids = prompt_positions.to(device).unsqueeze(0)

    sidecar = rollout_model.sidecar
    visual_kv = sidecar.prepare_visual_kv(vision, None)
    state = sidecar.initial_state(vision, None) if sidecar.state_tokens > 0 else None
    for layer_idx in range(len(get_lm_layers(language_model))):
        layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
        if sidecar.state_tokens > 0:
            delta, state = sidecar(
                h,
                None,
                layer_tensor,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
            )
        else:
            delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=delta,
        )
    h = get_lm_norm(language_model)(h)
    return h[0, -1]


def sidecar_point_hidden_batch(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: DeltaVisionModel,
    image_token_id: int,
    image_seq_length: int,
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    from PIL import Image

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    images = []
    prompts = []
    try:
        for row in rows:
            images.append(Image.open(row["image"]).convert("RGB"))
            prompts.append(point_prompt(str(row["label"])))
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    finally:
        processor.tokenizer.padding_side = old_padding_side
        for image in images:
            image.close()

    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    positions = get_batched_text_and_image_positions(
        inputs["input_ids"],
        inputs.get("attention_mask"),
        hidden0.shape[1],
        image_token_id,
        image_seq_length,
    )
    text_mask = positions.text_mask.to(device=device)
    text_padding_mask = ~text_mask
    h = gather_batched_positions(hidden0, positions.text_positions, text_mask).to(dtype=dtype)
    vision = gather_batched_positions(hidden0, positions.image_positions, positions.image_mask).to(dtype=dtype)
    h = h.masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
    position_ids = positions.text_position_ids.to(device=device)

    sidecar = rollout_model.sidecar
    visual_kv = sidecar.prepare_visual_kv(vision, ~positions.image_mask.to(device=device))
    state = sidecar.initial_state(vision, ~positions.image_mask.to(device=device)) if sidecar.state_tokens > 0 else None
    batch = h.shape[0]
    for layer_idx in range(len(get_lm_layers(language_model))):
        layer_tensor = torch.full((batch,), layer_idx, device=device, dtype=torch.long)
        if sidecar.state_tokens > 0:
            delta, state = sidecar(
                h,
                None,
                layer_tensor,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
            )
        else:
            delta = sidecar(h, None, layer_tensor, visual_kv=visual_kv)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=delta,
            padding_mask=text_padding_mask,
        )
    h = get_lm_norm(language_model)(h)
    last_indices = text_mask.long().sum(dim=1).sub(1).clamp_min(0)
    return h[torch.arange(batch, device=device), last_indices]
