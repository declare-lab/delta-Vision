#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_embed_tokens,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    read_jsonl,
    run_llama_layer_text_with_attention_delta_cache,
)
from delta_vision.models.modeling import build_rollout_model, image_token_id, load_frozen_llava, load_rollout_checkpoint


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mark_compiled_sidecar_step() -> None:
    if torch.cuda.is_available():
        torch.compiler.cudagraph_mark_step_begin()


def layer_id_tensor(layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.full((hidden_states.shape[0],), layer_idx, device=hidden_states.device, dtype=torch.long)


def make_layer_id_tensors(num_layers: int, hidden_states: torch.Tensor) -> list[torch.Tensor]:
    return [layer_id_tensor(layer_idx, hidden_states) for layer_idx in range(num_layers)]


def build_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question") or row.get("label") or "Describe the image.").strip()
    if "ASSISTANT:" in question:
        return question
    return llava15_prompt(question)


def parse_active_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(num_layers))
    values = {int(x) for x in spec.split(",") if x.strip()}
    if any(x < 0 or x >= num_layers for x in values):
        raise ValueError(f"--active-layers must contain zero-based layer ids in [0, {num_layers - 1}]")
    return values


@torch.inference_mode()
def prepare_sidecar_inputs(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    image_token: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = Image.open(row["image"]).convert("RGB")
    try:
        inputs = processor(text=build_prompt(row), images=image, return_tensors="pt")
    finally:
        image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    text_pos, image_pos, prompt_positions = get_text_and_image_positions(inputs["input_ids"], hidden0.shape[1], image_token)
    text_hidden = hidden0.index_select(1, text_pos.to(device)).to(dtype=dtype)
    vision_tokens = hidden0.index_select(1, image_pos.to(device)).to(dtype=dtype)
    position_ids = prompt_positions.to(device).unsqueeze(0)
    return text_hidden, vision_tokens, position_ids


@torch.inference_mode()
def sidecar_prefill(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
    position_ids: torch.Tensor,
    active_layers: set[int],
) -> tuple[torch.Tensor, DynamicCache, Any, Any]:
    sidecar = rollout_model.sidecar
    sidecar_dtype = next(sidecar.parameters()).dtype
    sidecar_vision = vision_tokens.to(dtype=sidecar_dtype) if vision_tokens.dtype != sidecar_dtype else vision_tokens
    visual_kv = sidecar.prepare_visual_kv(sidecar_vision, None)
    sidecar_state = (
        sidecar.initial_state(sidecar_vision, None)
        if sidecar.state_tokens > 0 and sidecar.runtime_use_state
        else None
    )
    cache = DynamicCache(config=language_model.config)
    h = text_hidden
    cache_position = torch.arange(h.shape[1], device=h.device)
    lm_layers = list(get_lm_layers(language_model))
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    lm_norm = get_lm_norm(language_model)
    num_layers = len(lm_layers)
    layer_tensors = make_layer_id_tensors(num_layers, h)
    position_embeddings = rotary_owner.rotary_emb(h, position_ids)
    for layer_idx in range(num_layers):
        if layer_idx not in active_layers:
            delta = None
        elif sidecar.state_tokens > 0 and sidecar.runtime_use_state:
            mark_compiled_sidecar_step()
            sidecar_h = h.to(dtype=sidecar_dtype) if h.dtype != sidecar_dtype else h
            layer_arg = layer_tensors[layer_idx]
            delta, sidecar_state = sidecar(
                sidecar_h,
                None,
                layer_arg,
                sidecar_state=sidecar_state,
                visual_kv=visual_kv,
                return_state=True,
            )
            sidecar_state = sidecar_state.clone()
        else:
            mark_compiled_sidecar_step()
            sidecar_h = h.to(dtype=sidecar_dtype) if h.dtype != sidecar_dtype else h
            delta = sidecar(sidecar_h, None, layer_tensors[layer_idx], visual_kv=visual_kv)
        if delta is not None and delta.dtype != h.dtype:
            delta = delta.to(dtype=h.dtype)
        h = run_llama_layer_text_with_attention_delta_cache(
            language_model,
            layer_idx,
            h,
            position_ids,
            cache_position,
            cache,
            attention_delta=delta,
            layer=lm_layers[layer_idx],
            rotary_owner=rotary_owner,
            position_embeddings=position_embeddings,
        )
    logits = model.lm_head(lm_norm(h))
    return logits[:, -1], cache, visual_kv, sidecar_state


@torch.inference_mode()
def sidecar_decode_step(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    token_id: int,
    position_id: int,
    cache: DynamicCache,
    visual_kv: Any,
    sidecar_state: Any,
    active_layers: set[int],
    layer_tensors: list[torch.Tensor] | None = None,
    lm_layers: list[torch.nn.Module] | None = None,
    rotary_owner: torch.nn.Module | None = None,
    lm_embed: torch.nn.Module | None = None,
    lm_norm: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, Any]:
    base_sidecar = rollout_model.sidecar
    sidecar = getattr(rollout_model, "sidecar_decode", base_sidecar)
    static_sidecar_layers = getattr(rollout_model, "sidecar_decode_layers", None)
    sidecar_dtype = next(base_sidecar.parameters()).dtype
    token = torch.tensor([[token_id]], device=base_sidecar.gate.device, dtype=torch.long)
    if lm_layers is None:
        lm_layers = list(get_lm_layers(language_model))
    if rotary_owner is None:
        rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    if lm_embed is None:
        lm_embed = get_lm_embed_tokens(language_model)
    if lm_norm is None:
        lm_norm = get_lm_norm(language_model)
    h = lm_embed(token)
    position_ids = torch.tensor([[position_id]], device=h.device, dtype=torch.long)
    cache_position = torch.tensor([position_id], device=h.device, dtype=torch.long)
    position_embeddings = rotary_owner.rotary_emb(h, position_ids)
    num_layers = len(lm_layers)
    if layer_tensors is None:
        layer_tensors = make_layer_id_tensors(num_layers, h)
    for layer_idx in range(num_layers):
        if layer_idx not in active_layers:
            delta = None
        elif base_sidecar.state_tokens > 0 and base_sidecar.runtime_use_state:
            mark_compiled_sidecar_step()
            sidecar_h = h.to(dtype=sidecar_dtype) if h.dtype != sidecar_dtype else h
            layer_arg = layer_tensors[layer_idx]
            delta, sidecar_state = sidecar(
                sidecar_h,
                None,
                layer_arg,
                sidecar_state=sidecar_state,
                visual_kv=visual_kv,
                return_state=True,
            )
            sidecar_state = sidecar_state.clone()
        else:
            mark_compiled_sidecar_step()
            sidecar_h = h.to(dtype=sidecar_dtype) if h.dtype != sidecar_dtype else h
            if static_sidecar_layers is not None:
                delta = static_sidecar_layers[layer_idx](sidecar_h, visual_kv)
            elif getattr(rollout_model, "sidecar_decode_no_state", False):
                delta = sidecar(sidecar_h, layer_tensors[layer_idx], visual_kv)
            else:
                delta = sidecar(sidecar_h, None, layer_tensors[layer_idx], visual_kv=visual_kv)
        if delta is not None and delta.dtype != h.dtype:
            delta = delta.to(dtype=h.dtype)
        h = run_llama_layer_text_with_attention_delta_cache(
            language_model,
            layer_idx,
            h,
            position_ids,
            cache_position,
            cache,
            attention_delta=delta,
            layer=lm_layers[layer_idx],
            rotary_owner=rotary_owner,
            position_embeddings=position_embeddings,
        )
    logits = model.lm_head(lm_norm(h))
    return logits[:, -1], sidecar_state


@torch.inference_mode()
def text_decode_step(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    token_id: int,
    position_id: int,
    cache: DynamicCache,
    device: torch.device,
    lm_layers: list[torch.nn.Module] | None = None,
    rotary_owner: torch.nn.Module | None = None,
    lm_embed: torch.nn.Module | None = None,
    lm_norm: torch.nn.Module | None = None,
) -> torch.Tensor:
    token = torch.tensor([[token_id]], device=device, dtype=torch.long)
    if lm_layers is None:
        lm_layers = list(get_lm_layers(language_model))
    if rotary_owner is None:
        rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    if lm_embed is None:
        lm_embed = get_lm_embed_tokens(language_model)
    if lm_norm is None:
        lm_norm = get_lm_norm(language_model)
    h = lm_embed(token)
    position_ids = torch.tensor([[position_id]], device=device, dtype=torch.long)
    cache_position = torch.tensor([position_id], device=device, dtype=torch.long)
    position_embeddings = rotary_owner.rotary_emb(h, position_ids)
    for layer_idx in range(len(lm_layers)):
        h = run_llama_layer_text_with_attention_delta_cache(
            language_model,
            layer_idx,
            h,
            position_ids,
            cache_position,
            cache,
            attention_delta=None,
            layer=lm_layers[layer_idx],
            rotary_owner=rotary_owner,
            position_embeddings=position_embeddings,
        )
    return model.lm_head(lm_norm(h))[:, -1]


@torch.inference_mode()
def hf_text_decode_step(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    token_id: int,
    position_id: int,
    cache: DynamicCache,
    device: torch.device,
) -> torch.Tensor:
    token = torch.tensor([[token_id]], device=device, dtype=torch.long)
    position_ids = torch.tensor([[position_id]], device=device, dtype=torch.long)
    cache_position = torch.tensor([position_id], device=device, dtype=torch.long)
    outputs = language_model(
        input_ids=token,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )
    return model.lm_head(outputs.last_hidden_state)[:, -1]


@torch.inference_mode()
def text_only_generate_cached(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    image_token: int,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    ignore_eos: bool = False,
    hf_decode: bool = False,
) -> dict[str, Any]:
    cuda_sync(device)
    t_visual = time.perf_counter()
    text_hidden, _vision_tokens, position_ids = prepare_sidecar_inputs(
        processor,
        model,
        row,
        image_token,
        device,
        dtype,
    )
    cuda_sync(device)
    visual_s = time.perf_counter() - t_visual

    cache = DynamicCache(config=language_model.config)
    h = text_hidden
    cache_position = torch.arange(h.shape[1], device=h.device)
    cuda_sync(device)
    t0 = time.perf_counter()
    lm_layers = list(get_lm_layers(language_model))
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    lm_embed = get_lm_embed_tokens(language_model)
    lm_norm = get_lm_norm(language_model)
    position_embeddings = rotary_owner.rotary_emb(h, position_ids)
    for layer_idx in range(len(lm_layers)):
        h = run_llama_layer_text_with_attention_delta_cache(
            language_model,
            layer_idx,
            h,
            position_ids,
            cache_position,
            cache,
            attention_delta=None,
            layer=lm_layers[layer_idx],
            rotary_owner=rotary_owner,
            position_embeddings=position_embeddings,
        )
    logits = model.lm_head(lm_norm(h))
    cuda_sync(device)
    prefill_s = time.perf_counter() - t0

    generated: list[int] = []
    eos = processor.tokenizer.eos_token_id
    next_id = int(logits[:, -1].float().argmax(dim=-1).item())
    start_pos = int(position_ids[0, -1].item()) + 1
    cuda_sync(device)
    t1 = time.perf_counter()
    for step in range(max_new_tokens):
        if next_id == eos and not ignore_eos:
            break
        generated.append(next_id)
        if step == max_new_tokens - 1:
            break
        if hf_decode:
            logits = hf_text_decode_step(model, language_model, next_id, start_pos + step, cache, device)
        else:
            logits = text_decode_step(
                model,
                language_model,
                next_id,
                start_pos + step,
                cache,
                device,
                lm_layers=lm_layers,
                rotary_owner=rotary_owner,
                lm_embed=lm_embed,
                lm_norm=lm_norm,
            )
        next_id = int(logits.float().argmax(dim=-1).item())
    cuda_sync(device)
    decode_s = time.perf_counter() - t1
    return {
        "text": processor.tokenizer.decode(generated, skip_special_tokens=True),
        "tokens": len(generated),
        "visual_s": visual_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": visual_s + prefill_s + decode_s,
        "tokens_per_s": len(generated) / max(decode_s, 1e-9),
        "total_tokens_per_s": len(generated) / max(visual_s + prefill_s + decode_s, 1e-9),
        "prompt_text_tokens": int(text_hidden.shape[1]),
    }


@torch.inference_mode()
def sidecar_generate_cached(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    row: dict[str, Any],
    image_token: int,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    prefill_active_layers: set[int],
    decode_active_layers: set[int],
    ignore_eos: bool = False,
    hf_decode_when_no_sidecar: bool = False,
) -> dict[str, Any]:
    cuda_sync(device)
    t_visual = time.perf_counter()
    text_hidden, vision_tokens, position_ids = prepare_sidecar_inputs(
        processor,
        model,
        row,
        image_token,
        device,
        dtype,
    )
    cuda_sync(device)
    visual_s = time.perf_counter() - t_visual
    t0 = time.perf_counter()
    logits, cache, visual_kv, sidecar_state = sidecar_prefill(
        model,
        language_model,
        rollout_model,
        text_hidden,
        vision_tokens,
        position_ids,
        prefill_active_layers,
    )
    cuda_sync(device)
    prefill_s = time.perf_counter() - t0
    generated: list[int] = []
    eos = processor.tokenizer.eos_token_id
    next_id = int(logits.float().argmax(dim=-1).item())
    start_pos = int(position_ids[0, -1].item()) + 1
    lm_layers = list(get_lm_layers(language_model))
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    lm_embed = get_lm_embed_tokens(language_model)
    lm_norm = get_lm_norm(language_model)
    decode_layer_tensors = make_layer_id_tensors(len(lm_layers), text_hidden[:, :1])
    cuda_sync(device)
    t1 = time.perf_counter()
    for step in range(max_new_tokens):
        if next_id == eos and not ignore_eos:
            break
        generated.append(next_id)
        if step == max_new_tokens - 1:
            break
        if decode_active_layers:
            logits, sidecar_state = sidecar_decode_step(
                model,
                language_model,
                rollout_model,
                next_id,
                start_pos + step,
                cache,
                visual_kv,
                sidecar_state,
                decode_active_layers,
                decode_layer_tensors,
                lm_layers=lm_layers,
                rotary_owner=rotary_owner,
                lm_embed=lm_embed,
                lm_norm=lm_norm,
            )
        else:
            if hf_decode_when_no_sidecar:
                logits = hf_text_decode_step(model, language_model, next_id, start_pos + step, cache, device)
            else:
                logits = text_decode_step(
                    model,
                    language_model,
                    next_id,
                    start_pos + step,
                    cache,
                    device,
                    lm_layers=lm_layers,
                    rotary_owner=rotary_owner,
                    lm_embed=lm_embed,
                    lm_norm=lm_norm,
                )
        next_id = int(logits.float().argmax(dim=-1).item())
    cuda_sync(device)
    decode_s = time.perf_counter() - t1
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "text": text,
        "tokens": len(generated),
        "visual_s": visual_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": visual_s + prefill_s + decode_s,
        "tokens_per_s": len(generated) / max(decode_s, 1e-9),
        "total_tokens_per_s": len(generated) / max(visual_s + prefill_s + decode_s, 1e-9),
        "prompt_text_tokens": int(text_hidden.shape[1]),
        "vision_tokens": int(vision_tokens.shape[1]),
    }


@torch.inference_mode()
def llava_generate_timed(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    max_new_tokens: int,
    device: torch.device,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    image = Image.open(row["image"]).convert("RGB")
    try:
        inputs = processor(text=build_prompt(row), images=image, return_tensors="pt")
    finally:
        image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    cuda_sync(device)
    t0 = time.perf_counter()
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if ignore_eos:
        generate_kwargs["min_new_tokens"] = max_new_tokens
    output = model.generate(**inputs, **generate_kwargs)
    cuda_sync(device)
    total_s = time.perf_counter() - t0
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return {
        "text": processor.tokenizer.decode(new_tokens, skip_special_tokens=True),
        "tokens": int(new_tokens.numel()),
        "total_s": total_s,
        "tokens_per_s": int(new_tokens.numel()) / max(total_s, 1e-9),
        "source_tokens": int(inputs["input_ids"].shape[1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark cached Shared Sidecar generation against LLaVA generation.")
    parser.add_argument("--data", default="data/pixmo_points/eval_test.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output-json", default="artifacts/eval/speed/cached_sidecar_vs_llava.json")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--sidecar-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--active-layers", default="all")
    parser.add_argument("--prefill-active-layers", default=None)
    parser.add_argument("--decode-active-layers", default=None)
    parser.add_argument("--ignore-mismatched-checkpoint-shapes", action="store_true")
    parser.add_argument("--slice-mismatched-checkpoint-prefix", action="store_true")
    parser.add_argument("--disable-sidecar-state", action="store_true")
    parser.add_argument("--use-sidecar-state", action="store_true")
    parser.add_argument("--no-freeze-basis-for-inference", action="store_true")
    parser.add_argument("--no-compile-sidecar", action="store_true")
    parser.add_argument("--static-sidecar-layers", action="store_true")
    parser.add_argument("--fold-sidecar-output-basis", action="store_true")
    parser.add_argument("--include-text-only", action="store_true")
    parser.add_argument("--hf-decode-when-no-sidecar", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.max_samples + args.warmup)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    sidecar_dtype = dtype_from_name(args.sidecar_dtype) if args.sidecar_dtype else dtype
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    image_token = image_token_id(model, processor)
    rollout_model = build_rollout_model(args, sidecar_dtype, device)
    load_rollout_checkpoint(
        rollout_model,
        args.checkpoint,
        ignore_mismatched_checkpoint_shapes=args.ignore_mismatched_checkpoint_shapes,
        slice_mismatched_checkpoint_prefix=args.slice_mismatched_checkpoint_prefix,
    )
    rollout_model.eval()
    for param in rollout_model.sidecar.parameters():
        param.requires_grad_(False)
    if not args.no_freeze_basis_for_inference:
        rollout_model.sidecar.basis.requires_grad_(False)
    rollout_model.sidecar.prepare_inference_cache(device, sidecar_dtype)
    rollout_model.sidecar.runtime_fold_output_basis = bool(args.fold_sidecar_output_basis)
    rollout_model.sidecar.runtime_use_state = bool(args.use_sidecar_state and not args.disable_sidecar_state)
    if not args.no_compile_sidecar:
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
        if rollout_model.sidecar.runtime_use_state:
            rollout_model.sidecar_decode = torch.compile(rollout_model.sidecar, mode="reduce-overhead")
            rollout_model.sidecar_decode_no_state = False
        else:
            rollout_model.sidecar_decode = torch.compile(rollout_model.sidecar.decode_no_state, mode="reduce-overhead")
            rollout_model.sidecar_decode_no_state = True
    if args.static_sidecar_layers and not rollout_model.sidecar.runtime_use_state:
        def make_static_sidecar_layer(layer_idx: int):
            def static_sidecar_layer(hidden_states: torch.Tensor, visual_kv: Any) -> torch.Tensor:
                return rollout_model.sidecar.decode_no_state_layer(hidden_states, layer_idx, visual_kv)

            return static_sidecar_layer

        rollout_model.sidecar_decode_layers = [
            make_static_sidecar_layer(layer_idx)
            for layer_idx in range(len(get_lm_layers(language_model)))
        ]
    active_layers = parse_active_layers(args.active_layers, len(get_lm_layers(language_model)))
    prefill_active_layers = (
        parse_active_layers(args.prefill_active_layers, len(get_lm_layers(language_model)))
        if args.prefill_active_layers is not None
        else active_layers
    )
    decode_active_layers = (
        parse_active_layers(args.decode_active_layers, len(get_lm_layers(language_model)))
        if args.decode_active_layers is not None
        else active_layers
    )

    results = []
    for idx, row in enumerate(rows):
        llava = llava_generate_timed(processor, model, row, args.max_new_tokens, device, args.ignore_eos)
        text_only = (
            text_only_generate_cached(
                processor,
                model,
                language_model,
                row,
                image_token,
                args.max_new_tokens,
                device,
                dtype,
                args.ignore_eos,
                args.hf_decode_when_no_sidecar,
            )
            if args.include_text_only
            else None
        )
        sidecar = sidecar_generate_cached(
            processor,
            model,
            language_model,
            rollout_model,
            row,
            image_token,
            args.max_new_tokens,
            device,
            dtype,
            prefill_active_layers,
            decode_active_layers,
            args.ignore_eos,
            args.hf_decode_when_no_sidecar,
        )
        record = {
            "index": idx,
            "label": row.get("label"),
            "llava": llava,
            "text_only_cached": text_only,
            "sidecar_cached": sidecar,
            "warmup": idx < args.warmup,
        }
        results.append(record)
        print(
            f"{idx+1}/{len(rows)} warmup={record['warmup']} "
            f"llava={llava['tokens_per_s']:.2f} tok/s "
            f"text_only={(text_only['tokens_per_s'] if text_only is not None else 0.0):.2f} tok/s "
            f"sidecar_decode={sidecar['tokens_per_s']:.2f} tok/s "
            f"sidecar_total={sidecar['total_tokens_per_s']:.2f} tok/s "
            f"sidecar_visual={sidecar['visual_s']:.3f}s "
            f"sidecar_prefill={sidecar['prefill_s']:.3f}s",
            flush=True,
        )

    measured = [item for item in results if not item["warmup"]]
    summary = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "active_layers": sorted(active_layers),
        "prefill_active_layers": sorted(prefill_active_layers),
        "decode_active_layers": sorted(decode_active_layers),
        "num_samples": len(measured),
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": bool(args.ignore_eos),
        "llava_total_tokens_per_s": mean([x["llava"]["tokens_per_s"] for x in measured]),
        "llava_total_s": mean([x["llava"]["total_s"] for x in measured]),
        "text_only_decode_tokens_per_s": mean(
            [x["text_only_cached"]["tokens_per_s"] for x in measured if x["text_only_cached"] is not None]
        )
        if args.include_text_only
        else None,
        "text_only_total_tokens_per_s": mean(
            [x["text_only_cached"]["total_tokens_per_s"] for x in measured if x["text_only_cached"] is not None]
        )
        if args.include_text_only
        else None,
        "sidecar_decode_tokens_per_s": mean([x["sidecar_cached"]["tokens_per_s"] for x in measured]),
        "sidecar_total_tokens_per_s": mean([x["sidecar_cached"]["total_tokens_per_s"] for x in measured]),
        "sidecar_visual_s": mean([x["sidecar_cached"]["visual_s"] for x in measured]),
        "sidecar_prefill_s": mean([x["sidecar_cached"]["prefill_s"] for x in measured]),
        "sidecar_decode_s": mean([x["sidecar_cached"]["decode_s"] for x in measured]),
        "sidecar_total_s": mean([x["sidecar_cached"]["total_s"] for x in measured]),
        "speedup_decode_vs_llava_total": mean([x["sidecar_cached"]["tokens_per_s"] for x in measured])
        / max(mean([x["llava"]["tokens_per_s"] for x in measured]), 1e-9),
        "speedup_total_vs_llava_total": mean([x["sidecar_cached"]["total_tokens_per_s"] for x in measured])
        / max(mean([x["llava"]["tokens_per_s"] for x in measured]), 1e-9),
        "records": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
