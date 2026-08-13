#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_text_and_image_positions,
    llama_attention_output,
    llava15_prompt,
    make_causal_mask,
    read_jsonl,
)
from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option
from delta_vision.models.modeling import image_token_id, load_frozen_llava


def parse_int_list(spec: str) -> list[int]:
    out = [int(x) for x in spec.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def stats_to_device(stats: dict[str, torch.Tensor | int], device: torch.device) -> dict[str, torch.Tensor | int]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in stats.items()}


def stats_to_cpu(stats: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor | int]:
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in stats.items()}


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [x.float(), torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)],
        dim=-1,
    )


def _llava_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _vision_tower(model: torch.nn.Module) -> torch.nn.Module:
    llava_model = _llava_model(model)
    if hasattr(llava_model, "vision_tower"):
        return llava_model.vision_tower
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    raise AttributeError("could not locate LLaVA vision tower")


@torch.inference_mode()
def build_inputs(processor: Any, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    question = str(row["question"]).strip()
    if "Answer directly with only the letter" not in question:
        question = f"{question}\nAnswer directly with only the letter of the correct option."
    with Image.open(row["image"]) as image:
        inputs = processor(text=llava15_prompt(question), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


@torch.inference_mode()
def all_vision_sources(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
) -> torch.Tensor:
    out = _vision_tower(model)(
        inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True,
    )
    sources = []
    for idx in source_layers:
        hidden = out.hidden_states[int(idx)]
        if hidden.shape[0] != 1:
            raise ValueError("LLaVA KV transfer currently expects batch size 1")
        if hidden.shape[1] <= 1:
            raise ValueError("vision hidden state does not contain a CLS token plus patch tokens")
        sources.append(add_bias(hidden[0, 1:]))
    return torch.stack(sources, dim=0)


@torch.inference_mode()
def vision_outputs_by_layer(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    source_layers: list[int],
) -> dict[int, torch.Tensor]:
    out = _vision_tower(model)(
        inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True,
    )
    return {int(idx): out.hidden_states[int(idx)][0, 1:].float() for idx in source_layers}


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


@torch.inference_mode()
def collect_all_source_stats_for_row(
    model: torch.nn.Module,
    processor: Any,
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

    hidden = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
        vision_feature_layer=getattr(model.config, "vision_feature_layer", None),
        vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", None),
    ).to(dtype=dtype)
    image_id = image_token_id(model, processor)
    _text_positions, image_positions, _teacher_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden.shape[1],
        image_id,
    )
    image_positions = image_positions.to(device=hidden.device)
    if image_positions.numel() != sources.shape[1]:
        raise ValueError(f"source/target visual token mismatch: source={sources.shape[1]} target={image_positions.numel()}")

    layers = get_lm_layers(language_model)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention_mask = make_causal_mask(1, hidden.shape[1], hidden.device, hidden.dtype)

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        image_normed = normed.index_select(1, image_positions)
        image_shape = image_normed.shape[:-1]
        image_hidden_shape = (*image_shape, -1, attn.head_dim)
        key_content = attn.k_proj(image_normed).view(image_hidden_shape).squeeze(0).float()
        value = attn.v_proj(image_normed).view(image_hidden_shape).squeeze(0).float()

        stats["xty_k"][:, layer_idx] += torch.einsum("snd,nhm->sdhm", sources, key_content).to(device=stats_device)
        stats["xty_v"][:, layer_idx] += torch.einsum("snd,nhm->sdhm", sources, value).to(device=stats_device)
        stats["y_sum_k"][layer_idx] += key_content.sum(dim=0).to(device=stats_device)
        stats["y_sum_v"][layer_idx] += value.sum(dim=0).to(device=stats_device)
        stats["y_sq_k"][layer_idx] += key_content.square().sum(dim=0).to(device=stats_device)
        stats["y_sq_v"][layer_idx] += value.square().sum(dim=0).to(device=stats_device)

        residual = hidden
        position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
        attn_out = llama_attention_output(layer.self_attn, normed, position_embeddings, attention_mask, is_causal=False)
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden


def cmd_collect_all(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    source_layers = parse_int_list(args.source_layers)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    layers = get_lm_layers(language_model)
    cfg = language_model.config
    source_dim = int(model.config.vision_config.hidden_size)
    stats = init_all_source_stats(
        num_sources=len(source_layers),
        num_layers=len(layers),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        source_dim=source_dim,
    )
    if args.stats_device != "cpu":
        stats = stats_to_device(stats, device)
    for idx, row in enumerate(rows):
        inputs = build_inputs(processor, row, device)
        collect_all_source_stats_for_row(model, processor, language_model, inputs, source_layers, dtype, stats)
        if (idx + 1) % args.log_every == 0:
            print(f"collected {idx + 1}/{len(rows)} samples tokens={stats['num_tokens']}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": stats_to_cpu(stats), "args": vars(args), "source_layers": source_layers}, out)
    print(f"wrote {out}", flush=True)


def _source_scores_from_stats(stats: dict[str, torch.Tensor | int], ridge: float) -> torch.Tensor:
    num_sources, num_layers, d, num_heads, head_dim = stats["xty_k"].shape
    score_matrix = torch.empty((num_sources, num_layers), dtype=torch.float32)
    n = max(int(stats["num_tokens"]), 1)
    for source_idx in range(num_sources):
        xtx = stats["xtx"][source_idx].double()
        xtx = xtx + float(ridge) * torch.eye(d, dtype=torch.float64)
        chol = torch.linalg.cholesky(xtx)
        layer_scores = []
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
            layer_scores.append((1.0 - sse / sst.clamp_min(1e-12)).float().mean(dim=1))
        score_matrix[source_idx] = 0.5 * (layer_scores[0] + layer_scores[1])
    return score_matrix


def _fit_single_source_from_stats(
    stats: dict[str, torch.Tensor | int],
    source_idx: int,
    ridge: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    _num_sources, num_layers, d, num_heads, head_dim = stats["xty_k"].shape
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
        scores.append((1.0 - sse / sst.clamp_min(1e-12)).float().clamp(max=1.0).mean(dim=1))
        weights[name] = sol.float().contiguous()
    return weights, 0.5 * (scores[0] + scores[1])


def cmd_select_topk(args: argparse.Namespace) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.stats]
    score_chunks: list[torch.Tensor] = []
    source_layers: list[int] = []
    for payload in payloads:
        stats = payload["stats"]
        score_chunks.append(_source_scores_from_stats(stats, args.ridge))
        source_layers.extend([int(x) for x in payload["source_layers"]])
    score_matrix = torch.cat(score_chunks, dim=0)
    topk = min(int(args.topk), score_matrix.shape[0])
    top_scores, top_indices = torch.topk(score_matrix, k=topk, dim=0)
    selection = []
    for layer_idx in range(score_matrix.shape[1]):
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
        "score_mean": float(score_matrix.mean().item()),
        "score_max": float(score_matrix.max().item()),
        "score_min": float(score_matrix.min().item()),
        "top1_mean": float(top_scores[0].mean().item()),
        "topk_mean": float(top_scores.mean().item()),
        "selection_rule": "topk_head_averaged_r2_over_all_clip_vision_encoder_layers",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "topk": topk, "summary": result, "selection": selection}, indent=2), flush=True)


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
    if num_layers is None or num_heads is None or d is None or head_dim is None:
        raise ValueError("empty stats")
    score_matrix = torch.cat(score_chunks, dim=0)
    topk = min(int(args.topk), score_matrix.shape[0])
    top_scores, top_indices = torch.topk(score_matrix, k=topk, dim=0)
    mix = top_scores.clamp_min(0.0)
    mix_sum = mix.sum(dim=0, keepdim=True)
    mix = torch.where(mix_sum > 0, mix / mix_sum.clamp_min(1e-12), torch.full_like(mix, 1.0 / topk))
    source_to_chunk: dict[int, tuple[int, int]] = {}
    offset = 0
    for chunk_idx, payload in enumerate(payloads):
        for local_idx, _layer_id in enumerate([int(x) for x in payload["source_layers"]]):
            source_to_chunk[offset + local_idx] = (chunk_idx, local_idx)
        offset += len(payload["source_layers"])
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
    print(json.dumps({"output": str(out), "topk": topk}, indent=2), flush=True)


def _mapped_image_kv(
    layer_sources: dict[int, torch.Tensor],
    mapper: dict[str, Any],
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mapper["weights"]
    source_list = mapper["source_layers_per_target"][layer_idx]
    mix = mapper["source_mix_weights_per_target"][layer_idx].to(device=device)
    k_parts = []
    v_parts = []
    for rank_idx, source_layer in enumerate(source_list):
        source = add_bias(layer_sources[int(source_layer)].to(device=device))
        wk = weights["k"][layer_idx, rank_idx].to(device=device)
        wv = weights["v"][layer_idx, rank_idx].to(device=device)
        k_parts.append(torch.einsum("nd,hdm->nhm", source, wk))
        v_parts.append(torch.einsum("nd,hdm->nhm", source, wv))
    k_content = torch.einsum("r,rnhm->nhm", mix, torch.stack(k_parts, dim=0)).unsqueeze(0).to(dtype=dtype)
    value = torch.einsum("r,rnhm->nhm", mix, torch.stack(v_parts, dim=0)).unsqueeze(0).to(dtype=dtype)
    return k_content, value


def _attention_with_mapped_image_kv(
    self_attn: torch.nn.Module,
    normed: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    image_positions: torch.Tensor,
    image_k_content: torch.Tensor,
    image_value: torch.Tensor,
) -> torch.Tensor:
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
    key_content = self_attn.k_proj(normed).view(hidden_shape)
    value = self_attn.v_proj(normed).view(hidden_shape)
    key_content[:, image_positions] = image_k_content.to(device=normed.device, dtype=key_content.dtype)
    value[:, image_positions] = image_value.to(device=normed.device, dtype=value.dtype)
    key_states = key_content.transpose(1, 2)
    value_states = value.transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    num_key_value_groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask[:, :, :, : key_states.shape[-2]],
        dropout_p=0.0,
        is_causal=False,
        scale=self_attn.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self_attn.o_proj(attn_output)


@torch.inference_mode()
def mapped_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    mapper: dict[str, Any],
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
        vision_feature_layer=getattr(model.config, "vision_feature_layer", None),
        vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", None),
    ).to(dtype=dtype)
    image_id = image_token_id(model, processor)
    _text_positions, image_positions, _teacher_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden.shape[1],
        image_id,
    )
    image_positions = image_positions.to(device=hidden.device)
    source_layers = sorted({int(layer) for row in mapper["source_layers_per_target"] for layer in row})
    layer_sources = vision_outputs_by_layer(model, inputs, source_layers)
    layers = get_lm_layers(language_model)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention_mask = make_causal_mask(1, hidden.shape[1], hidden.device, hidden.dtype)
    for layer_idx, layer in enumerate(layers):
        normed = layer.input_layernorm(hidden)
        image_k, image_v = _mapped_image_kv(layer_sources, mapper, layer_idx, hidden.device, dtype)
        position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
        residual = hidden
        attn_out = _attention_with_mapped_image_kv(
            layer.self_attn,
            normed,
            position_embeddings,
            attention_mask,
            image_positions,
            image_k,
            image_v,
        )
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden
    norm = language_model.model.norm if hasattr(language_model, "model") else language_model.norm
    logits = model.lm_head(norm(hidden))
    text_positions, _image_positions, _teacher_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden.shape[1],
        image_token_id(model, processor),
    )
    return logits[0, int(text_positions[-1].item())]


@torch.inference_mode()
def native_logits(
    model: torch.nn.Module,
    processor: Any,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = model(**inputs, use_cache=False)
    text_positions, _image_positions, _teacher_positions = get_text_and_image_positions(
        inputs["input_ids"],
        out.logits.shape[1],
        image_token_id(model, processor),
    )
    return out.logits[0, int(text_positions[-1].item())]


def cmd_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    mapper = torch.load(args.mapper, map_location="cpu", weights_only=False)
    mapper["weights"] = {key: value.to(device=device) for key, value in mapper["weights"].items()}
    option_ids = option_token_id_lists(processor.tokenizer)
    stats = {
        "llava": {"scored": 0, "correct": 0},
        "mapped_kv": {"scored": 0, "correct": 0, "agree": 0, "ret": 0, "kl": 0.0},
    }
    predictions = []
    for idx, row in enumerate(rows):
        gold = str(row["answer"]).strip().upper()[:1]
        inputs = build_inputs(processor, row, device)
        teacher_logits = native_logits(model, processor, inputs)
        pred_logits = mapped_logits(processor, model, language_model, inputs, mapper, dtype)
        teacher_pred = predict_option(teacher_logits, option_ids)
        pred = predict_option(pred_logits, option_ids)
        teacher_correct = teacher_pred == gold
        teacher_dist = option_distribution(teacher_logits, option_ids)
        pred_dist = option_distribution(pred_logits, option_ids)
        stats["llava"]["scored"] += 1
        stats["llava"]["correct"] += int(teacher_correct)
        stats["mapped_kv"]["scored"] += 1
        stats["mapped_kv"]["correct"] += int(pred == gold)
        stats["mapped_kv"]["agree"] += int(pred == teacher_pred)
        stats["mapped_kv"]["ret"] += int(teacher_correct and pred == gold)
        stats["mapped_kv"]["kl"] += float(F.kl_div(pred_dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item())
        predictions.append(
            {
                "index": row.get("index", args.start_index + idx),
                "gold": gold,
                "llava": teacher_pred,
                "mapped_kv": pred,
            }
        )
        if (idx + 1) % args.log_every == 0:
            print(f"evaluated {idx + 1}/{len(rows)}", flush=True)
    llava_n = max(1, stats["llava"]["scored"])
    mapped_n = max(1, stats["mapped_kv"]["scored"])
    llava_correct = max(1, stats["llava"]["correct"])
    result = {
        "data": args.data,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "mapper": args.mapper,
        "results": [
            {
                "setting": "llava",
                "scored": stats["llava"]["scored"],
                "correct": stats["llava"]["correct"],
                "accuracy": stats["llava"]["correct"] / llava_n,
            },
            {
                "setting": "mapped_kv",
                "scored": stats["mapped_kv"]["scored"],
                "correct": stats["mapped_kv"]["correct"],
                "accuracy": stats["mapped_kv"]["correct"] / mapped_n,
                "llava_agreement": stats["mapped_kv"]["agree"] / mapped_n,
                "llava_correct_retention": stats["mapped_kv"]["ret"] / llava_correct,
                "output_kl_to_llava": stats["mapped_kv"]["kl"] / mapped_n,
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
    parser = argparse.ArgumentParser("LLaVA vision-encoder source -> LLM visual-KV transfer diagnostic.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect-all")
    collect.add_argument("--data", required=True)
    collect.add_argument("--model-path", required=True)
    collect.add_argument("--start-index", type=int, default=0)
    collect.add_argument("--max-samples", type=int, default=None)
    collect.add_argument("--source-layers", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    collect.add_argument("--attn-implementation", default="eager")
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--stats-device", default="cpu", choices=["cpu", "cuda"])
    collect.add_argument("--log-every", type=int, default=10)

    select = sub.add_parser("select-topk")
    select.add_argument("--stats", nargs="+", required=True)
    select.add_argument("--output-json", required=True)
    select.add_argument("--topk", type=int, default=8)
    select.add_argument("--ridge", type=float, default=1e-2)
    fit_ensemble = sub.add_parser("fit-ensemble-topk")
    fit_ensemble.add_argument("--stats", nargs="+", required=True)
    fit_ensemble.add_argument("--output", required=True)
    fit_ensemble.add_argument("--topk", type=int, default=8)
    fit_ensemble.add_argument("--ridge", type=float, default=1.0)
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--mapper", required=True)
    eval_parser.add_argument("--output-json", required=True)
    eval_parser.add_argument("--predictions-jsonl", default="")
    eval_parser.add_argument("--start-index", type=int, default=1000)
    eval_parser.add_argument("--max-samples", type=int, default=500)
    eval_parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    eval_parser.add_argument("--attn-implementation", default="eager")
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "collect-all":
        cmd_collect_all(args)
    elif args.cmd == "select-topk":
        cmd_select_topk(args)
    elif args.cmd == "fit-ensemble-topk":
        cmd_fit_ensemble_topk(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
