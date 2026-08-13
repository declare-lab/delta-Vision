from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from delta_vision.models.llava import get_language_model, get_lm_layers
from delta_vision.models.sidecar import DeltaVisionModule
from delta_vision.runtime.basis import load_layer_basis


class DeltaVisionModel(nn.Module):
    def __init__(self, delta_vision: DeltaVisionModule) -> None:
        super().__init__()
        # Keep the historical module name so existing checkpoints with
        # "sidecar.*" state_dict keys continue to load.
        self.sidecar = delta_vision

    @property
    def delta_vision(self) -> DeltaVisionModule:
        return self.sidecar


# Backward-compatible alias used by older scripts.
SidecarRolloutModel = DeltaVisionModel


def image_token_id(model: torch.nn.Module, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_index", None)
    if token_id is None:
        token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    return int(token_id)


def load_frozen_llava(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "eager",
) -> tuple[Any, LlavaForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


def build_rollout_model(args: Namespace, dtype: torch.dtype, device: torch.device | None = None) -> DeltaVisionModel:
    basis = load_layer_basis(args.basis, args.rank, args.num_layers, args.hidden_size)[:1].contiguous()
    delta_vision = DeltaVisionModule(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        state_tokens=args.state_tokens,
        dropout=getattr(args, "dropout", 0.0),
        gate_init=getattr(args, "gate_init", 1.0),
        basis=basis,
        train_basis=getattr(args, "sidecar_output_mode", "residual") != "factorized_full",
        reader_mlp_ratio=args.reader_mlp_ratio,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_fuse_query=getattr(args, "reader_fuse_query", False),
        reader_concat_query=getattr(args, "reader_concat_query", False),
        normalize_basis_rows=True,
        shared_basis=True,
        output_mode=getattr(args, "sidecar_output_mode", "residual"),
    ).to(dtype=dtype)
    model = DeltaVisionModel(delta_vision=delta_vision)
    if device is not None:
        model = model.to(device=device)
    return model


def load_rollout_checkpoint(
    rollout_model: DeltaVisionModel,
    checkpoint_path: str | Path,
    ignore_checkpoint_basis: bool = False,
    ignore_mismatched_checkpoint_shapes: bool = False,
    slice_mismatched_checkpoint_prefix: bool = False,
) -> tuple[list[str], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if ignore_checkpoint_basis:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("basis", "sidecar.basis") and not key.endswith(".basis")
        }
    skipped = []
    sliced = []
    if ignore_mismatched_checkpoint_shapes or slice_mismatched_checkpoint_prefix:
        reference = rollout_model.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if key in reference and tuple(reference[key].shape) != tuple(value.shape):
                target_shape = tuple(reference[key].shape)
                source_shape = tuple(value.shape)
                can_slice = (
                    slice_mismatched_checkpoint_prefix
                    and len(source_shape) == len(target_shape)
                    and all(target_dim <= source_dim for target_dim, source_dim in zip(target_shape, source_shape))
                )
                if can_slice:
                    slices = tuple(slice(0, dim) for dim in target_shape)
                    filtered[key] = value[slices].contiguous()
                    sliced.append((key, source_shape, target_shape))
                    continue
                if ignore_mismatched_checkpoint_shapes:
                    skipped.append((key, source_shape, target_shape))
                    continue
                raise RuntimeError(
                    f"checkpoint tensor shape mismatch for {key}: source={source_shape} target={target_shape}"
                )
            filtered[key] = value
        state_dict = filtered
    if any(key.startswith("sidecar.") for key in state_dict):
        missing, unexpected = rollout_model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = rollout_model.sidecar.load_state_dict(
            state_dict,
            strict=not ignore_checkpoint_basis,
        )
    return list(missing), list(unexpected), skipped, sliced


def assert_llava_layer_count(model: torch.nn.Module, expected_layers: int) -> None:
    language_model = get_language_model(model)
    actual_layers = len(get_lm_layers(language_model))
    if actual_layers != expected_layers:
        raise RuntimeError(f"expected {expected_layers} layers, got {actual_layers}")
