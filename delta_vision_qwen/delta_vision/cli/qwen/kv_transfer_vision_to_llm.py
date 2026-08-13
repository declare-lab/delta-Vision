#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt,
    candidate_ids,
    candidate_kind,
    distribution,
    normalize_gold,
    predict,
    qwen_logits,
)
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_attention_output,
)


def parse_int_list(spec: str) -> list[int]:
    out = [int(x) for x in spec.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.float(), torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)], dim=-1)


def stats_to_device(stats: dict[str, torch.Tensor | int], device: torch.device) -> dict[str, torch.Tensor | int]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in stats.items()}


def stats_to_cpu(stats: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor | int]:
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in stats.items()}


def qwen3vl_visual_position_ids(
    full_position_ids: torch.Tensor,
    image_positions: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    visual_position_ids = torch.zeros(
        3,
        image_positions.shape[0],
        image_positions.shape[1],
        device=image_positions.device,
        dtype=full_position_ids.dtype,
    )
    valid = image_mask.bool()
    batch_idx = torch.arange(image_positions.shape[0], device=image_positions.device).unsqueeze(1).expand_as(image_positions)
    for dim_idx in range(3):
        dim_positions = full_position_ids[dim_idx]
        visual_position_ids[dim_idx][valid] = dim_positions[batch_idx[valid], image_positions[valid].long()]
    return visual_position_ids


def valid_length(attention_mask: torch.Tensor) -> int:
    return int(attention_mask[0].bool().sum().item())


def image_prefix_span(image_positions: torch.Tensor, image_mask: torch.Tensor) -> tuple[int, int, torch.Tensor]:
    valid_image_positions = image_positions[0, image_mask[0].bool()].long()
    if valid_image_positions.numel() == 0:
        raise ValueError("empty image positions")
    image_start = int(valid_image_positions.min().item())
    image_end = int(valid_image_positions.max().item()) + 1
    expected = torch.arange(image_start, image_end, device=valid_image_positions.device, dtype=valid_image_positions.dtype)
    if valid_image_positions.numel() != expected.numel() or not bool(torch.equal(valid_image_positions, expected)):
        raise ValueError("the first KV-transfer implementation expects a single contiguous image block")
    return image_start, image_end, valid_image_positions


@torch.inference_mode()
def build_inputs(processor: Any, row: dict[str, Any], benchmark: str, device: torch.device) -> dict[str, torch.Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


@torch.inference_mode()
def vision_source(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
) -> torch.Tensor:
    out = model.model.visual(
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        output_hidden_states=True,
        return_dict=True,
    )
    merged = [model.model.visual.merger(out.hidden_states[idx]) for idx in source_layers]
    return torch.cat(merged, dim=-1)


@torch.inference_mode()
def vision_outputs_by_layer(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
) -> dict[int, torch.Tensor]:
    out = model.model.visual(
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        output_hidden_states=True,
        return_dict=True,
    )
    return {int(idx): model.model.visual.merger(out.hidden_states[int(idx)]) for idx in source_layers}


@torch.inference_mode()
def all_vision_sources(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
) -> torch.Tensor:
    out = model.model.visual(
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        output_hidden_states=True,
        return_dict=True,
    )
    return torch.stack([add_bias(model.model.visual.merger(out.hidden_states[idx])) for idx in source_layers], dim=0)


def gather_topk_source(
    layer_sources: dict[int, torch.Tensor],
    selected_layers: list[int],
) -> torch.Tensor:
    return add_bias(torch.cat([layer_sources[int(idx)] for idx in selected_layers], dim=-1))


@torch.inference_mode()
def collect_stats_for_row(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
    dtype: torch.dtype,
    stats: dict[str, torch.Tensor | int],
) -> None:
    source = add_bias(vision_source(model, inputs, source_layers))
    x = source
    stats_device = stats["xtx"].device
    xtx = x.transpose(0, 1).matmul(x).double().to(device=stats_device)
    stats["xtx"] += xtx
    stats["num_tokens"] += int(x.shape[0])
    stats["num_samples"] += 1

    hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, _text_position_ids, _text_mask, image_mask, _full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    _image_start, image_end, image_indices = image_prefix_span(image_pos, image_mask)
    h = hidden0[:, :image_end].to(dtype=dtype).contiguous()
    prefix_position_ids = full_position_ids[:, :, :image_end].contiguous()
    prefix_attention_mask = inputs["attention_mask"][:, :image_end].contiguous()

    for layer_idx, layer in enumerate(language_model.layers):
        attn = layer.self_attn
        normed = layer.input_layernorm(h)
        image_normed = normed.index_select(1, image_indices)
        input_shape = normed.shape[:-1]
        image_shape = image_normed.shape[:-1]
        image_hidden_shape = (*image_shape, -1, attn.head_dim)
        key_content = attn.k_norm(attn.k_proj(image_normed).view(image_hidden_shape)).squeeze(0).float()
        value = attn.v_proj(image_normed).view(image_hidden_shape).squeeze(0).float()

        # Per target layer, per KV head, exactly as the KV-transfer paper.
        stats["xty_k"][layer_idx] += torch.einsum("nd,nhm->dhm", x, key_content).double().to(device=stats_device)
        stats["xty_v"][layer_idx] += torch.einsum("nd,nhm->dhm", x, value).double().to(device=stats_device)
        stats["y_sum_k"][layer_idx] += key_content.double().sum(dim=0).to(device=stats_device)
        stats["y_sum_v"][layer_idx] += value.double().sum(dim=0).to(device=stats_device)
        stats["y_sq_k"][layer_idx] += key_content.double().square().sum(dim=0).to(device=stats_device)
        stats["y_sq_v"][layer_idx] += value.double().square().sum(dim=0).to(device=stats_device)

        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=h,
            attention_mask=prefix_attention_mask,
            past_key_values=None,
            position_ids=None,
        )
        position_embeddings = language_model.rotary_emb(h, prefix_position_ids)
        residual = h
        attn_out = qwen3vl_attention_output(attn, normed, position_embeddings, attention_mask)
        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h


def init_stats(num_layers: int, num_kv_heads: int, head_dim: int, source_dim: int) -> dict[str, torch.Tensor | int]:
    d = source_dim + 1
    return {
        "xtx": torch.zeros((d, d), dtype=torch.float64),
        "xty_k": torch.zeros((num_layers, d, num_kv_heads, head_dim), dtype=torch.float64),
        "xty_v": torch.zeros((num_layers, d, num_kv_heads, head_dim), dtype=torch.float64),
        "y_sum_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float64),
        "y_sum_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float64),
        "y_sq_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float64),
        "y_sq_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float64),
        "num_tokens": 0,
        "num_samples": 0,
    }


def init_all_source_stats(
    num_sources: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    source_dim: int,
) -> dict[str, torch.Tensor | int]:
    d = source_dim + 1
    return {
        "xtx": torch.zeros((num_sources, d, d), dtype=torch.float32),
        "xty_k": torch.zeros((num_sources, num_layers, d, num_kv_heads, head_dim), dtype=torch.float32),
        "xty_v": torch.zeros((num_sources, num_layers, d, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sum_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sum_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sq_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sq_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "num_tokens": 0,
        "num_samples": 0,
    }


def init_selected_stats(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    source_dim: int,
) -> dict[str, torch.Tensor | int]:
    d = source_dim + 1
    return {
        "xtx": torch.zeros((num_layers, d, d), dtype=torch.float32),
        "xty_k": torch.zeros((num_layers, d, num_kv_heads, head_dim), dtype=torch.float32),
        "xty_v": torch.zeros((num_layers, d, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sum_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sum_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sq_k": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "y_sq_v": torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float32),
        "num_tokens": 0,
        "num_samples": 0,
    }


@torch.inference_mode()
def collect_all_source_stats_for_row(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
    dtype: torch.dtype,
    stats: dict[str, torch.Tensor | int],
) -> None:
    sources = all_vision_sources(model, inputs, source_layers)
    stats_device = stats["xtx"].device
    stats["xtx"] += torch.einsum("snd,sne->sde", sources, sources).to(device=stats_device)
    stats["num_tokens"] += int(sources.shape[1])
    stats["num_samples"] += 1

    hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    _text_pos, image_pos, _text_position_ids, _text_mask, image_mask, _full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    _image_start, image_end, image_indices = image_prefix_span(image_pos, image_mask)
    h = hidden0[:, :image_end].to(dtype=dtype).contiguous()
    prefix_position_ids = full_position_ids[:, :, :image_end].contiguous()
    prefix_attention_mask = inputs["attention_mask"][:, :image_end].contiguous()

    for layer_idx, layer in enumerate(language_model.layers):
        attn = layer.self_attn
        normed = layer.input_layernorm(h)
        image_normed = normed.index_select(1, image_indices)
        image_shape = image_normed.shape[:-1]
        image_hidden_shape = (*image_shape, -1, attn.head_dim)
        key_content = attn.k_norm(attn.k_proj(image_normed).view(image_hidden_shape)).squeeze(0).float()
        value = attn.v_proj(image_normed).view(image_hidden_shape).squeeze(0).float()

        stats["xty_k"][:, layer_idx] += torch.einsum("snd,nhm->sdhm", sources, key_content).to(device=stats_device)
        stats["xty_v"][:, layer_idx] += torch.einsum("snd,nhm->sdhm", sources, value).to(device=stats_device)
        stats["y_sum_k"][layer_idx] += key_content.sum(dim=0).to(device=stats_device)
        stats["y_sum_v"][layer_idx] += value.sum(dim=0).to(device=stats_device)
        stats["y_sq_k"][layer_idx] += key_content.square().sum(dim=0).to(device=stats_device)
        stats["y_sq_v"][layer_idx] += value.square().sum(dim=0).to(device=stats_device)

        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=h,
            attention_mask=prefix_attention_mask,
            past_key_values=None,
            position_ids=None,
        )
        position_embeddings = language_model.rotary_emb(h, prefix_position_ids)
        residual = h
        attn_out = qwen3vl_attention_output(attn, normed, position_embeddings, attention_mask)
        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h


@torch.inference_mode()
def collect_selected_stats_for_row(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    selected_source_layers: list[list[int]],
    all_source_layers: list[int],
    dtype: torch.dtype,
    stats: dict[str, torch.Tensor | int],
) -> None:
    layer_sources = vision_outputs_by_layer(model, inputs, all_source_layers)
    stats_device = stats["xtx"].device
    hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    _text_pos, image_pos, _text_position_ids, _text_mask, image_mask, _full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    _image_start, image_end, image_indices = image_prefix_span(image_pos, image_mask)
    h = hidden0[:, :image_end].to(dtype=dtype).contiguous()
    prefix_position_ids = full_position_ids[:, :, :image_end].contiguous()
    prefix_attention_mask = inputs["attention_mask"][:, :image_end].contiguous()
    stats["num_tokens"] += int(image_indices.numel())
    stats["num_samples"] += 1

    for layer_idx, layer in enumerate(language_model.layers):
        source = gather_topk_source(layer_sources, selected_source_layers[layer_idx])
        stats["xtx"][layer_idx] += source.transpose(0, 1).matmul(source).to(device=stats_device)
        attn = layer.self_attn
        normed = layer.input_layernorm(h)
        image_normed = normed.index_select(1, image_indices)
        image_shape = image_normed.shape[:-1]
        image_hidden_shape = (*image_shape, -1, attn.head_dim)
        key_content = attn.k_norm(attn.k_proj(image_normed).view(image_hidden_shape)).squeeze(0).float()
        value = attn.v_proj(image_normed).view(image_hidden_shape).squeeze(0).float()

        stats["xty_k"][layer_idx] += torch.einsum("nd,nhm->dhm", source, key_content).to(device=stats_device)
        stats["xty_v"][layer_idx] += torch.einsum("nd,nhm->dhm", source, value).to(device=stats_device)
        stats["y_sum_k"][layer_idx] += key_content.sum(dim=0).to(device=stats_device)
        stats["y_sum_v"][layer_idx] += value.sum(dim=0).to(device=stats_device)
        stats["y_sq_k"][layer_idx] += key_content.square().sum(dim=0).to(device=stats_device)
        stats["y_sq_v"][layer_idx] += value.square().sum(dim=0).to(device=stats_device)

        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=h,
            attention_mask=prefix_attention_mask,
            past_key_values=None,
            position_ids=None,
        )
        position_embeddings = language_model.rotary_emb(h, prefix_position_ids)
        residual = h
        attn_out = qwen3vl_attention_output(attn, normed, position_embeddings, attention_mask)
        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h


def cmd_collect(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    source_layers = parse_int_list(args.source_layers)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    cfg = language_model.config
    source_dim = int(model.config.vision_config.out_hidden_size) * len(source_layers)
    stats = init_stats(
        num_layers=len(language_model.layers),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        source_dim=source_dim,
    )
    if args.stats_device != "cpu":
        stats = stats_to_device(stats, device)
    for idx, row in enumerate(rows):
        inputs = build_inputs(processor, row, args.benchmark, device)
        collect_stats_for_row(model, language_model, inputs, source_layers, dtype, stats)
        if (idx + 1) % args.log_every == 0:
            print(f"collected {idx + 1}/{len(rows)} samples tokens={stats['num_tokens']}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": stats_to_cpu(stats), "args": vars(args), "source_layers": source_layers}, out)
    print(f"wrote {out}", flush=True)


def cmd_collect_all(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    source_layers = parse_int_list(args.source_layers)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    cfg = language_model.config
    source_dim = int(model.config.vision_config.out_hidden_size)
    stats = init_all_source_stats(
        num_sources=len(source_layers),
        num_layers=len(language_model.layers),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        source_dim=source_dim,
    )
    if args.stats_device != "cpu":
        stats = stats_to_device(stats, device)
    for idx, row in enumerate(rows):
        inputs = build_inputs(processor, row, args.benchmark, device)
        collect_all_source_stats_for_row(model, language_model, inputs, source_layers, dtype, stats)
        if (idx + 1) % args.log_every == 0:
            print(f"collected {idx + 1}/{len(rows)} samples tokens={stats['num_tokens']}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": stats_to_cpu(stats), "args": vars(args), "source_layers": source_layers}, out)
    print(f"wrote {out}", flush=True)


def load_selection(path: str | Path) -> list[list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [[int(x) for x in row["source_layers"]] for row in payload["selection"]]


def cmd_collect_selected(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    selected_source_layers = load_selection(args.selection_json)
    all_source_layers = sorted({layer for row in selected_source_layers for layer in row})
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    cfg = language_model.config
    source_dim = int(model.config.vision_config.out_hidden_size) * len(selected_source_layers[0])
    stats = init_selected_stats(
        num_layers=len(language_model.layers),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        source_dim=source_dim,
    )
    if args.stats_device != "cpu":
        stats = stats_to_device(stats, device)
    for idx, row in enumerate(rows):
        inputs = build_inputs(processor, row, args.benchmark, device)
        collect_selected_stats_for_row(
            model,
            language_model,
            inputs,
            selected_source_layers,
            all_source_layers,
            dtype,
            stats,
        )
        if (idx + 1) % args.log_every == 0:
            print(f"collected {idx + 1}/{len(rows)} samples tokens={stats['num_tokens']}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stats": stats_to_cpu(stats),
            "args": vars(args),
            "selected_source_layers": selected_source_layers,
            "all_source_layers": all_source_layers,
        },
        out,
    )
    print(f"wrote {out}", flush=True)


def cmd_fit(args: argparse.Namespace) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.stats]
    stats = payloads[0]["stats"]
    for payload in payloads[1:]:
        other = payload["stats"]
        for key, value in stats.items():
            stats[key] = value + other[key]
    xtx = stats["xtx"].clone()
    eye = torch.eye(xtx.shape[0], dtype=torch.float64)
    xtx = xtx + float(args.ridge) * eye
    chol = torch.linalg.cholesky(xtx)

    weights: dict[str, torch.Tensor] = {}
    r2: dict[str, torch.Tensor] = {}
    for name in ("k", "v"):
        xty = stats[f"xty_{name}"]
        layers, d, heads, head_dim = xty.shape
        rhs = xty.permute(1, 0, 2, 3).reshape(d, -1)
        sol = torch.cholesky_solve(rhs, chol).reshape(d, layers, heads, head_dim).permute(1, 2, 0, 3)
        weights[name] = sol.float().contiguous()
        pred_cross = (sol * xty.permute(0, 2, 1, 3)).sum(dim=(2, 3))
        pred_quad = torch.einsum("lhdf,de,lhef->lh", sol, xtx, sol)
        sse = stats[f"y_sq_{name}"].sum(dim=-1) - 2.0 * pred_cross + pred_quad
        n = max(int(stats["num_tokens"]), 1)
        sst = stats[f"y_sq_{name}"].sum(dim=-1) - stats[f"y_sum_{name}"].square().sum(dim=-1) / n
        r2[name] = (1.0 - sse / sst.clamp_min(1e-12)).float().clamp(max=1.0).contiguous()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": weights,
            "stats": {
                "num_tokens": int(stats["num_tokens"]),
                "num_samples": int(stats["num_samples"]),
                "r2_k": r2["k"],
                "r2_v": r2["v"],
                "y_sum_k": stats["y_sum_k"],
                "y_sum_v": stats["y_sum_v"],
                "y_sq_k": stats["y_sq_k"],
                "y_sq_v": stats["y_sq_v"],
            },
            "source_layers": payloads[0]["source_layers"],
            "ridge": float(args.ridge),
        },
        out,
    )
    print(f"wrote {out}", flush=True)


def cmd_select(args: argparse.Namespace) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.mappers]
    if not payloads:
        raise ValueError("no mapper files")
    source_layers = [int(payload["source_layers"][0]) for payload in payloads]
    scores = []
    for payload in payloads:
        r2_k = payload["stats"]["r2_k"].float().mean(dim=1)
        r2_v = payload["stats"]["r2_v"].float().mean(dim=1)
        scores.append(0.5 * (r2_k + r2_v))
    score_matrix = torch.stack(scores, dim=0)
    best_source_idx = score_matrix.argmax(dim=0)
    best_source_layers = torch.tensor([source_layers[int(i)] for i in best_source_idx], dtype=torch.long)

    weights = {}
    for name in ("k", "v"):
        per_source = torch.stack([payload["weights"][name] for payload in payloads], dim=0)
        selected = torch.empty_like(per_source[0])
        for layer_idx, source_idx in enumerate(best_source_idx.tolist()):
            selected[layer_idx] = per_source[int(source_idx), layer_idx]
        weights[name] = selected.contiguous()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": weights,
            "source_layer_for_target": best_source_layers,
            "source_layers": sorted(set(source_layers)),
            "source_scores": score_matrix,
            "selection": [
                {
                    "target_layer": int(layer_idx),
                    "source_layer": int(best_source_layers[layer_idx].item()),
                    "score": float(score_matrix[int(best_source_idx[layer_idx]), layer_idx].item()),
                }
                for layer_idx in range(best_source_layers.numel())
            ],
            "ridge": payloads[0].get("ridge"),
            "selection_rule": "top1_head_averaged_r2_over_all_vision_encoder_layers",
        },
        out,
    )
    summary = {
        "output": str(out),
        "source_layers": source_layers,
        "selection": [
            {
                "target_layer": int(layer_idx),
                "source_layer": int(best_source_layers[layer_idx].item()),
                "score": float(score_matrix[int(best_source_idx[layer_idx]), layer_idx].item()),
            }
            for layer_idx in range(best_source_layers.numel())
        ],
    }
    print(json.dumps(summary, indent=2), flush=True)


def cmd_fit_select_all(args: argparse.Namespace) -> None:
    payload = torch.load(args.stats[0], map_location="cpu", weights_only=False)
    stats = payload["stats"]
    for path in args.stats[1:]:
        other = torch.load(path, map_location="cpu", weights_only=False)["stats"]
        for key, value in stats.items():
            stats[key] = value + other[key]
    source_layers = [int(x) for x in payload["source_layers"]]
    num_sources, num_layers, d, num_heads, head_dim = stats["xty_k"].shape
    selected_score = torch.full((num_layers,), -float("inf"), dtype=torch.float32)
    selected_source_indices = torch.zeros((num_layers,), dtype=torch.long)
    selected_weights = {
        "k": torch.empty((num_layers, num_heads, d, head_dim), dtype=torch.float32),
        "v": torch.empty((num_layers, num_heads, d, head_dim), dtype=torch.float32),
    }
    score_matrix = torch.empty((num_sources, num_layers), dtype=torch.float32)
    n = max(int(stats["num_tokens"]), 1)

    for source_idx in range(num_sources):
        xtx = stats["xtx"][source_idx].double()
        xtx = xtx + float(args.ridge) * torch.eye(d, dtype=torch.float64)
        chol = torch.linalg.cholesky(xtx)
        source_weights: dict[str, torch.Tensor] = {}
        source_r2: dict[str, torch.Tensor] = {}
        for name in ("k", "v"):
            xty = stats[f"xty_{name}"][source_idx].double()
            rhs = xty.permute(1, 0, 2, 3).reshape(d, -1)
            sol = torch.cholesky_solve(rhs, chol).reshape(d, num_layers, num_heads, head_dim).permute(1, 2, 0, 3)
            pred_cross = (sol * xty.permute(0, 2, 1, 3)).sum(dim=(2, 3))
            pred_quad = torch.einsum("lhdf,de,lhef->lh", sol, xtx, sol)
            sse = stats[f"y_sq_{name}"].double().sum(dim=-1) - 2.0 * pred_cross + pred_quad
            sst = (
                stats[f"y_sq_{name}"].double().sum(dim=-1)
                - stats[f"y_sum_{name}"].double().square().sum(dim=-1) / n
            )
            source_weights[name] = sol.float().contiguous()
            source_r2[name] = (1.0 - sse / sst.clamp_min(1e-12)).float().clamp(max=1.0)
        score = 0.5 * (source_r2["k"].mean(dim=1) + source_r2["v"].mean(dim=1))
        score_matrix[source_idx] = score
        better = score > selected_score
        selected_score[better] = score[better]
        selected_source_indices[better] = source_idx
        for layer_idx in torch.nonzero(better, as_tuple=False).flatten().tolist():
            selected_weights["k"][layer_idx] = source_weights["k"][layer_idx]
            selected_weights["v"][layer_idx] = source_weights["v"][layer_idx]

    best_source_layers = torch.tensor(
        [source_layers[int(idx)] for idx in selected_source_indices.tolist()],
        dtype=torch.long,
    )


def cmd_select_topk(args: argparse.Namespace) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.stats]
    score_chunks: list[torch.Tensor] = []
    source_layers: list[int] = []
    for payload in payloads:
        stats = payload["stats"]
        chunk_layers = [int(x) for x in payload["source_layers"]]
        num_sources, num_layers, d, _num_heads, _head_dim = stats["xty_k"].shape
        chunk_scores = torch.empty((num_sources, num_layers), dtype=torch.float32)
        n = max(int(stats["num_tokens"]), 1)
        for source_idx in range(num_sources):
            xtx = stats["xtx"][source_idx].double()
            xtx = xtx + float(args.ridge) * torch.eye(d, dtype=torch.float64)
            chol = torch.linalg.cholesky(xtx)
            layer_scores = []
            for name in ("k", "v"):
                xty = stats[f"xty_{name}"][source_idx].double()
                rhs = xty.permute(1, 0, 2, 3).reshape(d, -1)
                sol = torch.cholesky_solve(rhs, chol).reshape(d, num_layers, _num_heads, _head_dim).permute(1, 2, 0, 3)
                pred_cross = (sol * xty.permute(0, 2, 1, 3)).sum(dim=(2, 3))
                pred_quad = torch.einsum("lhdf,de,lhef->lh", sol, xtx, sol)
                sse = stats[f"y_sq_{name}"].double().sum(dim=-1) - 2.0 * pred_cross + pred_quad
                sst = (
                    stats[f"y_sq_{name}"].double().sum(dim=-1)
                    - stats[f"y_sum_{name}"].double().square().sum(dim=-1) / n
                )
                layer_scores.append((1.0 - sse / sst.clamp_min(1e-12)).float().mean(dim=1))
            chunk_scores[source_idx] = 0.5 * (layer_scores[0] + layer_scores[1])
        score_chunks.append(chunk_scores)
        source_layers.extend(chunk_layers)
    score_matrix = torch.cat(score_chunks, dim=0)
    num_sources = score_matrix.shape[0]
    topk = min(int(args.topk), num_sources)
    top_scores, top_indices = torch.topk(score_matrix, k=topk, dim=0)
    selection = []
    for layer_idx in range(num_layers):
        selection.append(
            {
                "target_layer": int(layer_idx),
                "source_layers": [int(source_layers[int(i)]) for i in top_indices[:, layer_idx].tolist()],
                "scores": [float(x) for x in top_scores[:, layer_idx].tolist()],
            }
        )
    result = {
        "source_layers": source_layers,
        "topk": topk,
        "selection": selection,
        "score_matrix": score_matrix.tolist(),
        "selection_rule": "topk_head_averaged_r2_over_all_vision_encoder_layers",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "topk": topk, "selection": selection}, indent=2), flush=True)


def cmd_fit_selected(args: argparse.Namespace) -> None:
    payload = torch.load(args.stats[0], map_location="cpu", weights_only=False)
    stats = payload["stats"]
    for path in args.stats[1:]:
        other = torch.load(path, map_location="cpu", weights_only=False)["stats"]
        for key, value in stats.items():
            stats[key] = value + other[key]
    selected_source_layers = payload["selected_source_layers"]
    num_layers, d, num_heads, head_dim = stats["xty_k"].shape
    weights: dict[str, torch.Tensor] = {}
    xtx_all = stats["xtx"].double()
    eye = torch.eye(d, dtype=torch.float64)
    for name in ("k", "v"):
        selected = torch.empty((num_layers, num_heads, d, head_dim), dtype=torch.float32)
        for layer_idx in range(num_layers):
            xtx = xtx_all[layer_idx] + float(args.ridge) * eye
            chol = torch.linalg.cholesky(xtx)
            xty = stats[f"xty_{name}"][layer_idx].double()
            rhs = xty.permute(0, 1, 2).reshape(d, -1)
            sol = torch.cholesky_solve(rhs, chol).reshape(d, num_heads, head_dim).permute(1, 0, 2)
            selected[layer_idx] = sol.float()
        weights[name] = selected.contiguous()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": weights,
            "source_layers_per_target": selected_source_layers,
            "source_layers": payload["all_source_layers"],
            "ridge": float(args.ridge),
            "selection_rule": "topk_concat_refit_per_target_layer_per_head",
        },
        out,
    )
    print(f"wrote {out}", flush=True)


def _fit_single_source_from_stats(
    stats: dict[str, torch.Tensor | int],
    source_idx: int,
    ridge: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    num_sources, num_layers, d, num_heads, head_dim = stats["xty_k"].shape
    del num_sources
    xtx = stats["xtx"][source_idx].double()
    xtx = xtx + float(ridge) * torch.eye(d, dtype=torch.float64)
    chol = torch.linalg.cholesky(xtx)
    n = max(int(stats["num_tokens"]), 1)
    weights: dict[str, torch.Tensor] = {}
    scores = []
    for name in ("k", "v"):
        xty = stats[f"xty_{name}"][source_idx].double()
        rhs = xty.permute(1, 0, 2, 3).reshape(d, -1)
        sol = torch.cholesky_solve(rhs, chol).reshape(d, num_layers, num_heads, head_dim).permute(1, 2, 0, 3)
        pred_cross = (sol * xty.permute(0, 2, 1, 3)).sum(dim=(2, 3))
        pred_quad = torch.einsum("lhdf,de,lhef->lh", sol, xtx, sol)
        sse = stats[f"y_sq_{name}"].double().sum(dim=-1) - 2.0 * pred_cross + pred_quad
        sst = (
            stats[f"y_sq_{name}"].double().sum(dim=-1)
            - stats[f"y_sum_{name}"].double().square().sum(dim=-1) / n
        )
        r2 = (1.0 - sse / sst.clamp_min(1e-12)).float().clamp(max=1.0)
        scores.append(r2.mean(dim=1))
        weights[name] = sol.float().contiguous()
    return weights, 0.5 * (scores[0] + scores[1])


def cmd_fit_ensemble_topk(args: argparse.Namespace) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.stats]
    source_layers: list[int] = []
    score_chunks: list[torch.Tensor] = []
    num_layers = None
    num_heads = None
    d = None
    head_dim = None
    for payload in payloads:
        stats = payload["stats"]
        chunk_layers = [int(x) for x in payload["source_layers"]]
        chunk_scores = []
        for source_idx in range(len(chunk_layers)):
            _weights, score = _fit_single_source_from_stats(stats, source_idx, args.ridge)
            chunk_scores.append(score)
        score_chunks.append(torch.stack(chunk_scores, dim=0))
        source_layers.extend(chunk_layers)
        if num_layers is None:
            _num_sources, num_layers, d, num_heads, head_dim = stats["xty_k"].shape
    score_matrix = torch.cat(score_chunks, dim=0)
    topk = min(int(args.topk), score_matrix.shape[0])
    top_scores, top_indices = torch.topk(score_matrix, k=topk, dim=0)
    mix = top_scores.clamp_min(0.0)
    mix_sum = mix.sum(dim=0, keepdim=True)
    mix = torch.where(mix_sum > 0, mix / mix_sum.clamp_min(1e-12), torch.full_like(mix, 1.0 / topk))
    source_to_chunk: dict[int, tuple[int, int]] = {}
    offset = 0
    for chunk_idx, payload in enumerate(payloads):
        chunk_layers = [int(x) for x in payload["source_layers"]]
        for local_idx, _layer_id in enumerate(chunk_layers):
            source_to_chunk[offset + local_idx] = (chunk_idx, local_idx)
        offset += len(chunk_layers)
    selected_weights = {
        "k": torch.empty((num_layers, topk, num_heads, d, head_dim), dtype=torch.float32),
        "v": torch.empty((num_layers, topk, num_heads, d, head_dim), dtype=torch.float32),
    }
    selected_source_layers: list[list[int]] = []
    selected_scores: list[list[float]] = []
    for layer_idx in range(num_layers):
        selected_source_layers.append([int(source_layers[int(i)]) for i in top_indices[:, layer_idx].tolist()])
        selected_scores.append([float(x) for x in top_scores[:, layer_idx].tolist()])
    needed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for layer_idx in range(num_layers):
        for rank_idx, global_source_idx in enumerate(top_indices[:, layer_idx].tolist()):
            needed.setdefault(source_to_chunk[int(global_source_idx)], []).append((layer_idx, rank_idx))
    for (chunk_idx, local_idx), destinations in needed.items():
        stats = payloads[chunk_idx]["stats"]
        weights, _score = _fit_single_source_from_stats(stats, local_idx, args.ridge)
        for layer_idx, rank_idx in destinations:
            for name in ("k", "v"):
                selected_weights[name][layer_idx, rank_idx] = weights[name][layer_idx]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": selected_weights,
            "source_layers_per_target": selected_source_layers,
            "source_mix_weights_per_target": mix.transpose(0, 1).contiguous(),
            "source_scores_per_target": selected_scores,
            "source_layers": source_layers,
            "ridge": float(args.ridge),
            "selection_rule": "topk_weighted_ensemble_single_source_per_head_ridge",
            "topk": topk,
        },
        out,
    )
    summary = {
        "output": str(out),
        "topk": topk,
        "selection": [
            {
                "target_layer": int(layer_idx),
                "source_layers": selected_source_layers[layer_idx],
                "scores": selected_scores[layer_idx],
                "mix": [float(x) for x in mix[:, layer_idx].tolist()],
            }
            for layer_idx in range(num_layers)
        ],
    }
    print(json.dumps(summary, indent=2), flush=True)


def prefix_cache(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> tuple[DynamicCache, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    _text_pos, image_pos, _text_position_ids, _text_mask, image_mask, _full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    _image_start, image_end, image_indices = image_prefix_span(image_pos, image_mask)
    prefix_embeds = hidden0[:, :image_end].to(dtype=dtype).contiguous()
    prefix_position_ids = full_position_ids[:, :, :image_end].contiguous()
    prefix_attention_mask = inputs["attention_mask"][:, :image_end].contiguous()
    cache_out = language_model(
        inputs_embeds=prefix_embeds,
        attention_mask=prefix_attention_mask,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=True,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    )
    cache = cache_out.past_key_values
    return cache, image_end, image_indices, hidden0, full_position_ids, image_pos, image_mask, valid_length(inputs["attention_mask"])


@torch.inference_mode()
def mapped_visual_cache(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    mapper: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[DynamicCache, int, torch.Tensor, torch.Tensor, int]:
    cache, image_end, image_indices, hidden0, full_position_ids, image_pos, image_mask, valid_len = prefix_cache(
        model,
        language_model,
        inputs,
        dtype,
    )
    if "source_layers_per_target" in mapper:
        source_layers = sorted({int(layer) for row in mapper["source_layers_per_target"] for layer in row})
        layer_sources = vision_outputs_by_layer(model, inputs, source_layers)
    elif "source_layer_for_target" in mapper:
        source_layers = sorted({int(x) for x in mapper["source_layer_for_target"].tolist()})
        source_by_layer = {
            layer_idx: add_bias(vision_source(model, inputs, [layer_idx]))
            for layer_idx in source_layers
        }
    else:
        source_by_layer = {-1: add_bias(vision_source(model, inputs, mapper["source_layers"]))}
    visual_position_ids = qwen3vl_visual_position_ids(full_position_ids, image_pos, image_mask)
    weights = mapper["weights"]
    for layer_idx, layer in enumerate(language_model.layers):
        if "source_mix_weights_per_target" in mapper:
            source_list = mapper["source_layers_per_target"][layer_idx]
            mix = mapper["source_mix_weights_per_target"][layer_idx].to(device=full_position_ids.device)
            k_parts = []
            v_parts = []
            for rank_idx, source_layer in enumerate(source_list):
                source = add_bias(layer_sources[int(source_layer)])
                wk = weights["k"][layer_idx, rank_idx].to(device=source.device)
                wv = weights["v"][layer_idx, rank_idx].to(device=source.device)
                k_parts.append(torch.einsum("nd,hdm->nhm", source, wk))
                v_parts.append(torch.einsum("nd,hdm->nhm", source, wv))
            k_content = torch.einsum("r,rnhm->nhm", mix, torch.stack(k_parts, dim=0)).unsqueeze(0).to(dtype=dtype)
            value = torch.einsum("r,rnhm->nhm", mix, torch.stack(v_parts, dim=0)).unsqueeze(0).to(dtype=dtype)
        elif "source_layers_per_target" in mapper:
            source = gather_topk_source(layer_sources, mapper["source_layers_per_target"][layer_idx])
            wk = weights["k"][layer_idx].to(device=source.device)
            wv = weights["v"][layer_idx].to(device=source.device)
            k_content = torch.einsum("nd,hdm->nhm", source, wk).unsqueeze(0).to(dtype=dtype)
            value = torch.einsum("nd,hdm->nhm", source, wv).unsqueeze(0).to(dtype=dtype)
        else:
            if "source_layer_for_target" in mapper:
                source = source_by_layer[int(mapper["source_layer_for_target"][layer_idx].item())]
            else:
                source = source_by_layer[-1]
            wk = weights["k"][layer_idx].to(device=source.device)
            wv = weights["v"][layer_idx].to(device=source.device)
            k_content = torch.einsum("nd,hdm->nhm", source, wk).unsqueeze(0).to(dtype=dtype)
            value = torch.einsum("nd,hdm->nhm", source, wv).unsqueeze(0).to(dtype=dtype)
        key_states = k_content.transpose(1, 2).contiguous()
        value_states = value.transpose(1, 2).contiguous()
        _, key_states = apply_rotary_pos_emb(
            key_states,
            key_states,
            *language_model.rotary_emb(k_content, visual_position_ids),
        )
        cache.layers[layer_idx].keys.index_copy_(2, image_indices, key_states)
        cache.layers[layer_idx].values.index_copy_(2, image_indices, value_states)
    return cache, image_end, hidden0, full_position_ids, valid_len


@torch.inference_mode()
def true_visual_cache(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> tuple[DynamicCache, int, torch.Tensor, torch.Tensor, int]:
    cache, image_end, _image_indices, hidden0, full_position_ids, _image_pos, _image_mask, valid_len = prefix_cache(
        model,
        language_model,
        inputs,
        dtype,
    )
    return cache, image_end, hidden0, full_position_ids, valid_len


@torch.inference_mode()
def qwen_without_extra_vision_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    inputs = build_inputs(processor, row, benchmark, device)
    hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    text_pos, _image_pos, _text_position_ids, text_mask, _image_mask, _full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    valid_len = valid_length(inputs["attention_mask"])
    out = language_model(
        inputs_embeds=hidden0[:, :valid_len].to(dtype=dtype).contiguous(),
        attention_mask=inputs["attention_mask"][:, :valid_len].contiguous(),
        position_ids=full_position_ids[:, :, :valid_len].contiguous(),
        past_key_values=None,
        use_cache=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    )
    logits = model.lm_head(out.last_hidden_state)
    last_text = int(text_pos[0, int(text_mask[0].sum().item()) - 1].item())
    return logits[0, last_text]


@torch.inference_mode()
def cache_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    cache: DynamicCache,
    prefix_len: int,
    hidden0: torch.Tensor,
    full_position_ids: torch.Tensor,
    valid_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    suffix_embeds = hidden0[:, prefix_len:valid_len].to(dtype=dtype).contiguous()
    suffix_position_ids = full_position_ids[:, :, prefix_len:valid_len].contiguous()
    attention_mask = torch.ones(
        (suffix_embeds.shape[0], valid_len),
        device=device,
        dtype=torch.long,
    )
    out = language_model(
        inputs_embeds=suffix_embeds,
        attention_mask=attention_mask,
        position_ids=suffix_position_ids,
        past_key_values=cache,
        use_cache=True,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    )
    logits = model.lm_head(out.last_hidden_state)
    return logits[0, -1]


def cmd_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    mapper = torch.load(args.mapper, map_location="cpu", weights_only=False)
    mapper["weights"] = {k: v.to(device=device) for k, v in mapper["weights"].items()}
    option_stats = {
        "qwen": {"scored": 0, "correct": 0},
        "qwen_without_extra_vision": {"scored": 0, "correct": 0},
        "mapped_kv": {"scored": 0, "correct": 0, "agree": 0, "ret": 0, "kl": 0.0},
        "true_visual_kv": {"scored": 0, "correct": 0, "agree": 0, "ret": 0, "kl": 0.0},
    }
    predictions = []
    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        inputs = build_inputs(processor, row, args.benchmark, device)
        qwen_full_logits = qwen_logits(processor, model, row, args.benchmark, device)
        teacher_logits = qwen_without_extra_vision_logits(
            processor,
            model,
            language_model,
            row,
            args.benchmark,
            device,
            dtype,
        )
        mapped_cache, mapped_prefix_len, mapped_hidden0, mapped_position_ids, mapped_valid_len = mapped_visual_cache(
            model,
            language_model,
            inputs,
            mapper,
            dtype,
        )
        pred_logits = cache_logits(
            processor,
            model,
            language_model,
            mapped_cache,
            mapped_prefix_len,
            mapped_hidden0,
            mapped_position_ids,
            mapped_valid_len,
            device,
            dtype,
        )
        true_cache, true_prefix_len, true_hidden0, true_position_ids, true_valid_len = true_visual_cache(
            model,
            language_model,
            inputs,
            dtype,
        )
        true_logits = cache_logits(
            processor,
            model,
            language_model,
            true_cache,
            true_prefix_len,
            true_hidden0,
            true_position_ids,
            true_valid_len,
            device,
            dtype,
        )
        qwen_full_pred = predict(qwen_full_logits, ids)
        teacher_pred = predict(teacher_logits, ids)
        pred = predict(pred_logits, ids)
        true_pred = predict(true_logits, ids)
        teacher_correct = teacher_pred == gold
        teacher_dist = distribution(teacher_logits, ids)
        pred_dist = distribution(pred_logits, ids)
        true_dist = distribution(true_logits, ids)
        kl = F.kl_div(pred_dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
        true_kl = F.kl_div(true_dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
        option_stats["qwen"]["scored"] += 1
        option_stats["qwen"]["correct"] += int(qwen_full_pred == gold)
        option_stats["qwen_without_extra_vision"]["scored"] += 1
        option_stats["qwen_without_extra_vision"]["correct"] += int(teacher_correct)
        option_stats["mapped_kv"]["scored"] += 1
        option_stats["mapped_kv"]["correct"] += int(pred == gold)
        option_stats["mapped_kv"]["agree"] += int(pred == teacher_pred)
        option_stats["mapped_kv"]["ret"] += int(teacher_correct and pred == gold)
        option_stats["mapped_kv"]["kl"] += float(kl)
        option_stats["true_visual_kv"]["scored"] += 1
        option_stats["true_visual_kv"]["correct"] += int(true_pred == gold)
        option_stats["true_visual_kv"]["agree"] += int(true_pred == teacher_pred)
        option_stats["true_visual_kv"]["ret"] += int(teacher_correct and true_pred == gold)
        option_stats["true_visual_kv"]["kl"] += float(true_kl)
        predictions.append(
            {
                "index": row.get("index", args.start_index + idx),
                "gold": gold,
                "qwen": qwen_full_pred,
                "qwen_without_extra_vision": teacher_pred,
                "mapped_kv": pred,
                "true_visual_kv": true_pred,
            }
        )
        if (idx + 1) % args.log_every == 0:
            print(f"evaluated {idx + 1}/{len(rows)}", flush=True)
    q_scored = max(1, option_stats["qwen"]["scored"])
    nd_scored = max(1, option_stats["qwen_without_extra_vision"]["scored"])
    nd_correct = max(1, option_stats["qwen_without_extra_vision"]["correct"])
    m_scored = max(1, option_stats["mapped_kv"]["scored"])
    t_scored = max(1, option_stats["true_visual_kv"]["scored"])
    result = {
        "benchmark": args.benchmark,
        "data": args.data,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "mapper": args.mapper,
        "results": [
            {
                "setting": "qwen",
                "scored": option_stats["qwen"]["scored"],
                "correct": option_stats["qwen"]["correct"],
                "accuracy": option_stats["qwen"]["correct"] / q_scored,
            },
            {
                "setting": "qwen_without_extra_vision",
                "scored": option_stats["qwen_without_extra_vision"]["scored"],
                "correct": option_stats["qwen_without_extra_vision"]["correct"],
                "accuracy": option_stats["qwen_without_extra_vision"]["correct"] / nd_scored,
            },
            {
                "setting": "mapped_kv",
                "scored": option_stats["mapped_kv"]["scored"],
                "correct": option_stats["mapped_kv"]["correct"],
                "accuracy": option_stats["mapped_kv"]["correct"] / m_scored,
                "qwen_agreement": option_stats["mapped_kv"]["agree"] / m_scored,
                "qwen_correct_retention": option_stats["mapped_kv"]["ret"] / nd_correct,
                "output_kl_to_qwen": option_stats["mapped_kv"]["kl"] / m_scored,
            },
            {
                "setting": "true_visual_kv",
                "scored": option_stats["true_visual_kv"]["scored"],
                "correct": option_stats["true_visual_kv"]["correct"],
                "accuracy": option_stats["true_visual_kv"]["correct"] / t_scored,
                "qwen_agreement": option_stats["true_visual_kv"]["agree"] / t_scored,
                "qwen_correct_retention": option_stats["true_visual_kv"]["ret"] / nd_correct,
                "output_kl_to_qwen": option_stats["true_visual_kv"]["kl"] / t_scored,
            },
        ],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions_jsonl:
        pred_out = Path(args.predictions_jsonl)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL vision-encoder to LLM visual-KV transfer diagnostics.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "collect-all", "collect-selected", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--benchmark", choices=("mmstar", "realworldqa"), default="mmstar")
        p.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
        p.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
        p.add_argument("--start-index", type=int, default=0)
        p.add_argument("--max-samples", type=int, default=None)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
        p.add_argument("--attn-implementation", default="flash_attention_2")
        p.add_argument("--log-every", type=int, default=25)
        p.add_argument("--stats-device", choices=("cpu", "cuda"), default="cuda")
    sub.choices["collect"].add_argument("--source-layers", default="24")
    sub.choices["collect"].add_argument("--output", required=True)
    sub.choices["collect-all"].add_argument("--source-layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    sub.choices["collect-all"].add_argument("--output", required=True)
    sub.choices["collect-selected"].add_argument("--selection-json", required=True)
    sub.choices["collect-selected"].add_argument("--output", required=True)
    pfit = sub.add_parser("fit")
    pfit.add_argument("--stats", nargs="+", required=True)
    pfit.add_argument("--output", required=True)
    pfit.add_argument("--ridge", type=float, default=1e-2)
    pselect = sub.add_parser("select")
    pselect.add_argument("--mappers", nargs="+", required=True)
    pselect.add_argument("--output", required=True)
    pfitall = sub.add_parser("fit-select-all")
    pfitall.add_argument("--stats", nargs="+", required=True)
    pfitall.add_argument("--output", required=True)
    pfitall.add_argument("--ridge", type=float, default=1e-2)
    ptopk = sub.add_parser("select-topk")
    ptopk.add_argument("--stats", nargs="+", required=True)
    ptopk.add_argument("--output-json", required=True)
    ptopk.add_argument("--topk", type=int, default=4)
    ptopk.add_argument("--ridge", type=float, default=1e-2)
    pfitsel = sub.add_parser("fit-selected")
    pfitsel.add_argument("--stats", nargs="+", required=True)
    pfitsel.add_argument("--output", required=True)
    pfitsel.add_argument("--ridge", type=float, default=1e-2)
    pensemble = sub.add_parser("fit-ensemble-topk")
    pensemble.add_argument("--stats", nargs="+", required=True)
    pensemble.add_argument("--output", required=True)
    pensemble.add_argument("--topk", type=int, default=8)
    pensemble.add_argument("--ridge", type=float, default=1e-2)
    sub.choices["eval"].add_argument("--mapper", required=True)
    sub.choices["eval"].add_argument("--output-json", required=True)
    sub.choices["eval"].add_argument("--predictions-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "collect-all":
        cmd_collect_all(args)
    elif args.cmd == "collect-selected":
        cmd_collect_selected(args)
    elif args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "fit-select-all":
        cmd_fit_select_all(args)
    elif args.cmd == "select-topk":
        cmd_select_topk(args)
    elif args.cmd == "fit-selected":
        cmd_fit_selected(args)
    elif args.cmd == "fit-ensemble-topk":
        cmd_fit_ensemble_topk(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
