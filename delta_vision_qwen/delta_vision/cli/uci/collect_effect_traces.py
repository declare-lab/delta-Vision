#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.masking_utils import create_causal_mask

from delta_vision.cli.diagnostics.cross_teacher_context_factors import mmstar_question
from delta_vision.models.llava import (
    build_llava_initial_hidden,
    compute_llama_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_text_and_image_positions,
    llama_attention_output,
    llava15_prompt,
    make_causal_mask,
    read_jsonl,
)
from delta_vision.models.modeling import image_token_id, load_frozen_llava
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_attention_output,
    qwen3vl_prompt,
)


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "stride4":
        return list(range(0, num_layers, 4))
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect UCI effect traces.")
    parser.add_argument("--teacher", choices=("llava", "qwen3vl", "qwen3vl_thinking"), required=True)
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--context-cache", default="", help="Optional .pt cache with precomputed context tokens.")
    parser.add_argument("--write-context-cache", default="", help="Optional .pt path to save context tokens and rows metadata.")
    parser.add_argument("--teacher-model-path", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--layers", default="stride4")
    parser.add_argument("--token-mode", choices=("last", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--log-every", type=int, default=4)
    return parser.parse_args()


def _vision_tower(model: torch.nn.Module) -> torch.nn.Module:
    llava_model = model.model if hasattr(model, "model") else model
    if hasattr(llava_model, "vision_tower"):
        return llava_model.vision_tower
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    raise AttributeError("could not locate LLaVA vision tower")


@torch.inference_mode()
def collect_context_tokens(
    rows: list[dict[str, Any]],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    log_every: int,
) -> list[torch.Tensor]:
    processor, model = load_frozen_llava(model_path, dtype, device, attn_implementation)
    tower = _vision_tower(model)
    context_tokens: list[torch.Tensor] = []
    for idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=llava15_prompt(mmstar_question(row)), images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = tower(pixel_values, output_hidden_states=True, return_dict=True)
        layer = int(getattr(model.config, "vision_feature_layer", -2))
        hidden = out.hidden_states[layer][0, 1:].float().cpu()
        context_tokens.append(hidden)
        if (idx + 1) % log_every == 0:
            print(f"context tokens {idx + 1}/{len(rows)}", flush=True)
    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return context_tokens


def _llava_full_layer(language_model: torch.nn.Module, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    layer = get_lm_layers(language_model)[layer_idx]
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention_mask = make_causal_mask(1, hidden.shape[1], hidden.device, hidden.dtype)
    residual = hidden
    normed = layer.input_layernorm(hidden)
    position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
    attn_out = llama_attention_output(layer.self_attn, normed, position_embeddings, attention_mask, is_causal=False)
    hidden = residual + attn_out
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    return residual + hidden


@torch.inference_mode()
def collect_llava_targets(
    rows: list[dict[str, Any]],
    context_tokens: list[torch.Tensor],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    layer_spec: str,
    token_mode: str,
    log_every: int,
) -> dict[str, Any]:
    processor, model = load_frozen_llava(model_path, dtype, device, attn_implementation)
    language_model = get_language_model(model)
    layers = get_lm_layers(language_model)
    selected_layers = parse_layers(layer_spec, len(layers))
    img_id = image_token_id(model, processor)
    samples = []
    for idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=llava15_prompt(mmstar_question(row)), images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        hidden = build_llava_initial_hidden(
            model,
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_sizes=inputs.get("image_sizes"),
            vision_feature_layer=getattr(model.config, "vision_feature_layer", None),
            vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", None),
        ).to(dtype=dtype)
        text_pos, _image_pos, _teacher_pos = get_text_and_image_positions(inputs["input_ids"], hidden.shape[1], img_id)
        hiddens = []
        deltas = []
        for layer_idx in range(len(layers)):
            if layer_idx in selected_layers:
                delta = compute_llama_attention_effect(language_model, layer_idx, hidden, text_pos)
                if token_mode == "last":
                    hiddens.append(hidden[0, int(text_pos[-1].item())].float().cpu())
                    deltas.append(delta[0, -1].float().cpu())
                else:
                    hiddens.append(hidden[0].index_select(0, text_pos.to(device=hidden.device)).float().cpu())
                    deltas.append(delta[0].float().cpu())
            hidden = _llava_full_layer(language_model, layer_idx, hidden)
        samples.append(
            {
                "index": row.get("index", idx),
                "context_tokens": context_tokens[idx],
                "hiddens": torch.stack(hiddens, dim=0),
                "deltas": torch.stack(deltas, dim=0),
            }
        )
        if (idx + 1) % log_every == 0:
            print(f"llava targets {idx + 1}/{len(rows)}", flush=True)
    return {
        "teacher": "llava",
        "model_path": model_path,
        "layers": selected_layers,
        "hidden_size": int(language_model.config.hidden_size),
        "context_dim": int(context_tokens[0].shape[-1]),
        "token_mode": token_mode,
        "samples": samples,
    }


def _qwen_full_layer(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> torch.Tensor:
    layer = language_model.layers[layer_idx]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden, position_ids)
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attn_out = qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
    hidden = residual + attn_out
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    return residual + hidden


@torch.inference_mode()
def collect_qwen_targets(
    rows: list[dict[str, Any]],
    context_tokens: list[torch.Tensor],
    teacher_name: str,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    layer_spec: str,
    token_mode: str,
    log_every: int,
) -> dict[str, Any]:
    processor, model = load_frozen_qwen3vl(model_path, dtype, device, attn_implementation)
    language_model = get_language_model(model)
    selected_layers = parse_layers(layer_spec, len(language_model.layers))
    samples = []
    for idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=qwen3vl_prompt(processor, mmstar_question(row)), images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        hidden, full_position_ids, _visual_pos_masks, _deepstack = build_qwen3vl_initial_context(model, inputs)
        hidden = hidden.to(dtype=dtype)
        text_pos, _image_pos, text_position_ids, text_mask, _image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        valid_text_indices = torch.nonzero(text_mask[0].bool(), as_tuple=False).flatten()
        last_idx = int(valid_text_indices[-1].item())
        hiddens = []
        deltas = []
        for layer_idx in range(len(language_model.layers)):
            if layer_idx in selected_layers:
                text_hidden = gather_batched_positions(hidden, text_pos, text_mask)
                delta = compute_qwen3vl_attention_effect_batched(
                    language_model,
                    layer_idx,
                    hidden,
                    text_hidden,
                    full_position_ids,
                    text_position_ids,
                    text_pos,
                    full_mask,
                    text_mask,
                )
                if token_mode == "last":
                    hiddens.append(text_hidden[0, last_idx].float().cpu())
                    deltas.append(delta[0, last_idx].float().cpu())
                else:
                    hiddens.append(text_hidden[0, valid_text_indices].float().cpu())
                    deltas.append(delta[0, valid_text_indices].float().cpu())
            hidden = _qwen_full_layer(language_model, layer_idx, hidden, full_position_ids, inputs["attention_mask"])
        samples.append(
            {
                "index": row.get("index", idx),
                "context_tokens": context_tokens[idx],
                "hiddens": torch.stack(hiddens, dim=0),
                "deltas": torch.stack(deltas, dim=0),
            }
        )
        if (idx + 1) % log_every == 0:
            print(f"{teacher_name} targets {idx + 1}/{len(rows)}", flush=True)
    return {
        "teacher": teacher_name,
        "model_path": model_path,
        "layers": selected_layers,
        "hidden_size": int(language_model.config.hidden_size),
        "context_dim": int(context_tokens[0].shape[-1]),
        "token_mode": token_mode,
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    rows = rows[args.start_index : args.start_index + args.max_samples]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    teacher_path = args.teacher_model_path
    if not teacher_path:
        if args.teacher == "llava":
            teacher_path = args.context_model_path
        elif args.teacher == "qwen3vl":
            teacher_path = "models/Qwen3-VL-4B-Instruct"
        elif args.teacher == "qwen3vl_thinking":
            teacher_path = "models/Qwen3-VL-4B-Thinking"
    if args.context_cache:
        cache = torch.load(args.context_cache, map_location="cpu", weights_only=False)
        context_tokens = cache["context_tokens"]
        if len(context_tokens) != len(rows):
            raise ValueError(
                f"context cache length mismatch: cache={len(context_tokens)} rows={len(rows)}"
            )
        print(f"loaded context cache {args.context_cache} with {len(context_tokens)} samples", flush=True)
    else:
        context_tokens = collect_context_tokens(
            rows,
            args.context_model_path,
            device,
            dtype,
            args.attn_implementation,
            args.log_every,
        )
        if args.write_context_cache:
            cache_out = Path(args.write_context_cache)
            cache_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "context_tokens": context_tokens,
                    "data": args.data,
                    "start_index": args.start_index,
                    "max_samples": len(rows),
                    "context_model_path": args.context_model_path,
                },
                cache_out,
            )
            print(f"wrote context cache {cache_out}", flush=True)
    if args.teacher == "llava":
        payload = collect_llava_targets(
            rows,
            context_tokens,
            teacher_path,
            device,
            dtype,
            args.attn_implementation,
            args.layers,
            args.token_mode,
            args.log_every,
        )
    else:
        payload = collect_qwen_targets(
            rows,
            context_tokens,
            args.teacher,
            teacher_path,
            device,
            dtype,
            args.attn_implementation,
            args.layers,
            args.token_mode,
            args.log_every,
        )
    payload["data"] = args.data
    payload["start_index"] = args.start_index
    payload["max_samples"] = len(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({k: v for k, v in payload.items() if k != "samples"} | {"output": str(out), "num_samples": len(payload["samples"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
