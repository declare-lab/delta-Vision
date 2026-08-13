#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from delta_vision.cli.qwen.eval_qwen3vl_sidecar import (
    build_prompt as build_scoring_prompt,
    load_sidecar,
    qwen3vl_visual_position_ids,
    sidecar_rope_visual_kv,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from delta_vision.cli.qwen.train_qwen3vl_sidecar import qwen_native_sidecar_query, qwen_native_visual_kv
from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_text_attention_outputs_cache,
    qwen3vl_visual_memory_by_layer,
    qwen3vl_prefix_visual_memory_by_layer,
    run_qwen3vl_full_layer_with_text_delta_cache,
    run_qwen3vl_layer_text_from_attention_output,
    run_qwen3vl_layer_text_with_attention_delta_cache,
)
from delta_vision.models.sidecar import DeltaVisionModule
from delta_vision.runtime.ops import VisualKVCache


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype for accounting: {dtype}")


def bytes_to_mib(value: float | int) -> float:
    return float(value) / 1024.0 / 1024.0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.empty_cache()
    sync(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def peak(device: torch.device, baseline: int) -> tuple[float, float, float]:
    if device.type != "cuda":
        return 0.0, 0.0, 0.0
    sync(device)
    max_alloc = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    return bytes_to_mib(max_alloc), bytes_to_mib(max_reserved), bytes_to_mib(max(0, max_alloc - baseline))


def profiled_call(
    enabled: bool,
    timing: dict[str, float] | None,
    key: str,
    device: torch.device,
    fn: Any,
) -> Any:
    if not enabled:
        return fn()
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - start
    return result


def build_generation_prompt(processor: Any, row: dict[str, Any], benchmark: str, include_image: bool) -> str:
    question = str(row["question"]).strip()
    if benchmark == "mmstar":
        question = f"{question}\nAnswer directly with only the letter of the correct option."
    elif benchmark == "realworldqa":
        question = f"{question}\nAnswer directly with the final answer only."
    content: list[dict[str, str]] = []
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_inputs(
    processor: Any,
    row: dict[str, Any],
    benchmark: str,
    include_image: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt = build_generation_prompt(processor, row, benchmark, include_image=include_image)
    if include_image:
        with Image.open(row["image"]) as image:
            inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def generation_stop_ids(tokenizer: Any, ignore_eos: bool) -> list[int] | None:
    if ignore_eos:
        return None
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(int(x) for x in tokenizer.eos_token_id)
        else:
            ids.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(ids) if ids else None


def qwen_visual_kv_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_layers = len(language_model.layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(num_layers * image_tokens * 2 * num_kv_heads * head_dim * dtype_bytes(dtype))


def one_layer_visual_hidden_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    return int(image_tokens * int(language_model.config.hidden_size) * dtype_bytes(dtype))


def sidecar_external_qwen_kv_temp_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_heads = int(getattr(config, "num_attention_heads"))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(image_tokens * 2 * num_heads * head_dim * dtype_bytes(dtype))


def sidecar_external_qwen_compact_kv_temp_bytes(language_model: torch.nn.Module, image_tokens: int, dtype: torch.dtype) -> int:
    config = language_model.config
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(image_tokens * 2 * num_kv_heads * head_dim * dtype_bytes(dtype))


def sidecar_param_bytes(sidecar: DeltaVisionModule | None) -> int:
    if sidecar is None:
        return 0
    return int(sum(p.numel() * p.element_size() for p in sidecar.parameters()))


def output_token_count(output: torch.Tensor, prompt_tokens: int) -> int:
    return max(0, int(output.shape[-1]) - int(prompt_tokens))


def decode_generated_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def parse_mmstar_answer(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    match = re.search(r"\b([A-E])\b", cleaned.upper())
    if match is not None:
        return match.group(1)
    first = cleaned[:1].upper()
    return first if first in {"A", "B", "C", "D", "E"} else None


CHOICE_LINE_RE = re.compile(r"(?m)^\s*\(?([A-D])\)?[.)：:]\s*(.+?)\s*$")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def row_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices") or []
    if choices:
        return [str(choice) for choice in choices]
    found: dict[str, str] = {}
    for letter, text in CHOICE_LINE_RE.findall(str(row.get("question", ""))):
        found[letter.upper()] = text.strip()
    return [found[letter] for letter in ("A", "B", "C", "D") if letter in found]


def first_option(text: str) -> str | None:
    stripped = text.strip().upper()
    if stripped[:1] in {"A", "B", "C", "D"}:
        return stripped[:1]
    match = re.search(r"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-D])[\)\].]?", stripped)
    if match is not None:
        return match.group(1)
    matches = re.findall(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", stripped)
    return matches[-1] if matches else None


def parse_realworldqa_answer(row: dict[str, Any], text: str) -> tuple[str | None, bool]:
    gold = str(row.get("answer", "")).strip()
    pred = text.strip()
    choices = row_choices(row)
    gold_letter = first_option(gold)
    if gold_letter is not None:
        pred_letter = first_option(pred)
        return pred_letter, bool(pred_letter == gold_letter)
    if choices:
        pred_letter = first_option(pred)
        if pred_letter is not None:
            idx = ord(pred_letter) - ord("A")
            if 0 <= idx < len(choices):
                pred_choice = str(choices[idx])
                return pred_letter, normalize_text(pred_choice) == normalize_text(gold)
        pred_norm = normalize_text(pred)
        gold_norm = normalize_text(gold)
        return pred if pred else None, bool(pred_norm == gold_norm or (gold_norm and gold_norm in pred_norm))
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(pred)
    if gold.lower() in {"yes", "no"}:
        if pred.lower().startswith("yes"):
            return "Yes", gold.lower() == "yes"
        if pred.lower().startswith("no"):
            return "No", gold.lower() == "no"
    return pred if pred else None, bool(pred_norm == gold_norm or (gold_norm and gold_norm in pred_norm))


@torch.inference_mode()
def generate_hf(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    mode: str,
    max_new_tokens: int,
    ignore_eos: bool,
    device: torch.device,
) -> dict[str, Any]:
    include_image = mode == "qwen"
    inputs = prepare_inputs(processor, row, benchmark, include_image=include_image, device=device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    baseline = reset_peak(device)
    start = time.perf_counter()
    generate_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    eos_ids = generation_stop_ids(processor.tokenizer, ignore_eos=ignore_eos)
    if eos_ids is None:
        generate_kwargs["eos_token_id"] = None
    else:
        generate_kwargs["eos_token_id"] = eos_ids
    output = model.generate(**inputs, **generate_kwargs)
    sync(device)
    elapsed = time.perf_counter() - start
    max_alloc, max_reserved, incremental = peak(device, baseline)
    generated_tokens = output_token_count(output, prompt_tokens)
    generated_ids = output[0, prompt_tokens:].detach().cpu().tolist()
    return {
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_ids": generated_ids,
        "generated_text": decode_generated_text(processor.tokenizer, generated_ids),
        "prefill_sec": None,
        "decode_sec": None,
        "total_sec": elapsed,
        "tokens_per_sec": generated_tokens / max(elapsed, 1e-9),
        "max_allocated_mib": max_alloc,
        "max_reserved_mib": max_reserved,
        "incremental_peak_mib": incremental,
    }


def decode_position_ids(last_position_ids: torch.Tensor, step: int) -> torch.Tensor:
    return last_position_ids[:, :, -1:] + int(step + 1)


def full_attention_mask_for_cache(cache: DynamicCache, device: torch.device) -> torch.Tensor:
    return torch.ones((1, int(cache.get_seq_length()) + 1), device=device, dtype=torch.long)


def sidecar_delta_for_layer(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    visual_kv: Any,
    text_attention: torch.Tensor | None,
    text_attention_heads: torch.Tensor | None,
    query_source: str,
    scale: float,
    query_states: torch.Tensor | None = None,
) -> torch.Tensor:
    layer_tensor = torch.full((hidden_states.shape[0],), layer_idx, device=hidden_states.device, dtype=torch.long)
    query_override = query_states
    if query_source == "qwen_native":
        if query_override is None:
            query_override = qwen_native_sidecar_query(language_model, layer_idx, hidden_states, position_ids)
    elif query_source != "sidecar":
        raise ValueError(f"unsupported sidecar query source: {query_source}")
    delta = sidecar(
        hidden_states,
        None,
        layer_tensor,
        visual_kv=visual_kv,
        text_attention=text_attention,
        output_projection=language_model.layers[layer_idx].self_attn.o_proj,
        text_attention_heads=text_attention_heads,
        query_states=query_override,
    )
    return delta * float(scale)


def qwen_native_visual_kv_compact(
    language_model: torch.nn.Module,
    layer_idx: int,
    vision_states: torch.Tensor,
    visual_position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
    repeat_heads: bool,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> VisualKVCache:
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(vision_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        position_embeddings = language_model.rotary_emb(normed, visual_position_ids)
    _, key_states = apply_rotary_pos_emb(key_states, key_states, *position_embeddings)
    if repeat_heads:
        key_states = repeat_kv(key_states, int(self_attn.num_key_value_groups))
        value_states = repeat_kv(value_states, int(self_attn.num_key_value_groups))
    return VisualKVCache(key=key_states.contiguous(), value=value_states.contiguous(), padding_mask=padding_mask)


def layer_visual_kv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    visual_memory: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_positions: torch.Tensor,
    image_mask: torch.Tensor,
    visual_position_ids: torch.Tensor,
    dtype: torch.dtype,
    source: str,
    compact_qwen_kv: bool,
    visual_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Any:
    if source == "qwen_native":
        return qwen_native_visual_kv_compact(
            language_model,
            layer_idx,
            visual_memory.to(dtype=dtype),
            visual_position_ids,
            None,
            repeat_heads=not compact_qwen_kv,
            position_embeddings=visual_position_embeddings,
        )
    if source == "qwen_first_layer":
        return qwen_native_visual_kv_compact(
            language_model,
            0,
            visual_memory.to(dtype=dtype),
            visual_position_ids,
            None,
            repeat_heads=not compact_qwen_kv,
            position_embeddings=visual_position_embeddings,
        )
    if source == "sidecar":
        visual_kv, _ = sidecar_rope_visual_kv(
            sidecar,
            language_model,
            visual_memory.to(dtype=dtype),
            full_position_ids,
            text_position_ids,
            image_positions,
            image_mask,
            dtype,
        )
        return visual_kv
    raise ValueError(f"unsupported sidecar visual KV source: {source}")


def ensure_static_visual_cache_supported(sidecar: DeltaVisionModule, cache_mode: str) -> None:
    if cache_mode == "stream":
        return
    mode = str(getattr(sidecar, "visual_transform_mode", "none"))
    if mode in {"recurrent_adapter", "full_cascade_adapter"}:
        raise ValueError(
            f"--sidecar-visual-cache-mode={cache_mode} is only valid for visual transforms "
            f"that do not depend on the current decode hidden state; got {mode}"
        )


def precompute_sidecar_visual_caches(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    visual_memories: list[torch.Tensor],
    position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_pos: torch.Tensor,
    image_mask: torch.Tensor,
    visual_position_ids: torch.Tensor,
    dtype: torch.dtype,
    visual_kv_source: str,
    compact_qwen_visual_kv: bool,
    cache_mode: str,
    visual_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[list[torch.Tensor] | None, list[Any] | None, torch.Tensor | None]:
    ensure_static_visual_cache_supported(sidecar, cache_mode)
    if cache_mode == "stream":
        return None, None, sidecar.initial_visual_memory_state(visual_memories[0].to(dtype=dtype))

    visual_cache: list[torch.Tensor] = []
    visual_kv_cache: list[Any] | None = [] if cache_mode == "cache_kv" else None
    visual_memory_state = sidecar.initial_visual_memory_state(visual_memories[0].to(dtype=dtype))
    for layer_idx in range(len(language_model.layers)):
        visual_memory, visual_memory_state = sidecar.visual_memory_for_layer(
            visual_memories[layer_idx].to(dtype=dtype),
            layer_idx,
            hidden_states=None,
            current_visual_memory=visual_memory_state,
            vision_padding_mask=~image_mask,
        )
        visual_memory = visual_memory.contiguous()
        visual_cache.append(visual_memory)
        if visual_kv_cache is not None:
            visual_kv_cache.append(
                layer_visual_kv(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memory,
                    position_ids,
                    text_position_ids,
                    image_pos,
                    image_mask,
                    visual_position_ids,
                    dtype,
                    visual_kv_source,
                    compact_qwen_visual_kv,
                    visual_position_embeddings,
                )
            )
    return visual_cache, visual_kv_cache, visual_memory_state


def get_sidecar_visual_for_layer(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    visual_memories: list[torch.Tensor],
    visual_memory_state: torch.Tensor | None,
    visual_memory_cache: list[torch.Tensor] | None,
    visual_kv_cache: list[Any] | None,
    h: torch.Tensor,
    position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_pos: torch.Tensor,
    image_mask: torch.Tensor,
    visual_position_ids: torch.Tensor,
    dtype: torch.dtype,
    visual_kv_source: str,
    compact_qwen_visual_kv: bool,
    visual_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
    if visual_kv_cache is not None:
        visual_memory = (
            visual_memory_cache[layer_idx]
            if visual_memory_cache is not None
            else visual_memories[layer_idx].to(dtype=dtype)
        )
        return visual_memory, visual_kv_cache[layer_idx], visual_memory_state
    if visual_memory_cache is not None:
        visual_memory = visual_memory_cache[layer_idx]
    else:
        visual_memory, visual_memory_state = sidecar.visual_memory_for_layer(
            visual_memories[layer_idx].to(dtype=dtype),
            layer_idx,
            hidden_states=h,
            current_visual_memory=visual_memory_state,
            vision_padding_mask=~image_mask,
        )
    visual_kv = layer_visual_kv(
        sidecar,
        language_model,
        layer_idx,
        visual_memory,
        position_ids,
        text_position_ids,
        image_pos,
        image_mask,
        visual_position_ids,
        dtype,
        visual_kv_source,
        compact_qwen_visual_kv,
        visual_position_embeddings,
    )
    return visual_memory, visual_kv, visual_memory_state


@torch.inference_mode()
def custom_prefill(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    row: dict[str, Any],
    benchmark: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    sidecar_scale: float,
    query_source: str,
    visual_kv_source: str,
    compact_qwen_visual_kv: bool,
    sidecar_visual_cache_mode: str,
    profile_sidecar_timing: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = prepare_inputs(processor, row, benchmark, include_image=True, device=device)
    h, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    if visual_memory_mode == "vprefix":
        visual_memories = qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            h.to(dtype=dtype),
            position_ids,
            inputs["attention_mask"],
            image_pos,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
    else:
        visual_memories = qwen3vl_visual_memory_by_layer(
            h.to(dtype=dtype),
            image_pos,
            image_mask,
            deepstack_visual_embeds,
            visual_memory_mode,
            len(language_model.layers),
        )
    visual_position_ids = qwen3vl_visual_position_ids(position_ids, image_pos, image_mask)
    visual_position_embeddings = language_model.rotary_emb(visual_memories[0].to(dtype=dtype), visual_position_ids)
    text_cache = DynamicCache(config=language_model.config)
    full_cache = DynamicCache(config=language_model.config) if mode == "hybrid" else None
    timing: dict[str, float] = {}
    visual_memory_cache, visual_kv_cache, visual_memory_state = profiled_call(
        profile_sidecar_timing,
        timing,
        "prefill_visual_cache",
        device,
        lambda: precompute_sidecar_visual_caches(
            sidecar,
            language_model,
            visual_memories,
            position_ids,
            text_position_ids,
            image_pos,
            image_mask,
            visual_position_ids,
            dtype,
            visual_kv_source,
            compact_qwen_visual_kv,
            sidecar_visual_cache_mode,
            visual_position_embeddings,
        ),
    )

    if mode in {"sidecar_only", "prefill_only", "prefill_hf"}:
        h_text = gather_batched_positions(h, text_pos, text_mask).to(dtype=dtype)
        text_position_embeddings = language_model.rotary_emb(h_text, text_position_ids)
        text_attention_mask = text_mask.to(dtype=torch.long)
        for layer_idx in range(len(language_model.layers)):
            _visual_memory, visual_kv, visual_memory_state = profiled_call(
                profile_sidecar_timing,
                timing,
                "prefill_visual_kv",
                device,
                lambda layer_idx=layer_idx, h_text=h_text, visual_memory_state=visual_memory_state: get_sidecar_visual_for_layer(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memories,
                    visual_memory_state,
                    visual_memory_cache,
                    visual_kv_cache,
                    h_text,
                    position_ids,
                    text_position_ids,
                    image_pos,
                    image_mask,
                    visual_position_ids,
                    dtype,
                    visual_kv_source,
                    compact_qwen_visual_kv,
                    visual_position_embeddings,
                ),
            )
            text_attention, text_heads, query_states = profiled_call(
                profile_sidecar_timing,
                timing,
                "prefill_text_attention",
                device,
                lambda layer_idx=layer_idx, h_text=h_text: qwen3vl_text_attention_outputs_cache(
                    language_model,
                    layer_idx,
                    h_text,
                    text_position_ids,
                    past_key_values=text_cache,
                    attention_mask_2d=text_attention_mask,
                    position_embeddings=text_position_embeddings,
                    mask_past_key_values=None,
                    return_query_states=True,
                ),
            )
            delta = profiled_call(
                profile_sidecar_timing,
                timing,
                "prefill_sidecar_delta",
                device,
                lambda layer_idx=layer_idx, h_text=h_text, visual_kv=visual_kv, text_attention=text_attention, text_heads=text_heads, query_states=query_states: sidecar_delta_for_layer(
                    sidecar,
                    language_model,
                    layer_idx,
                    h_text,
                    text_position_ids,
                    visual_kv,
                    text_attention if sidecar.output_mode.startswith("factorized") else None,
                    text_heads if sidecar.output_mode in {"factorized_native_head_o", "factorized_native_head_o_residual"} else None,
                    query_source,
                    sidecar_scale,
                    query_states if query_source == "qwen_native" else None,
                ),
            )
            h_text = profiled_call(
                profile_sidecar_timing,
                timing,
                "prefill_block_tail",
                device,
                lambda layer_idx=layer_idx, h_text=h_text, text_attention=text_attention, delta=delta: run_qwen3vl_layer_text_from_attention_output(
                    language_model,
                    layer_idx,
                    h_text,
                    text_attention,
                    delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                ),
            )
        valid_text = int(text_mask[0].sum().item())
        logits = model.lm_head(language_model.norm(h_text))[0, valid_text - 1]
        return logits, {
            "text_cache": text_cache,
            "full_cache": None,
            "last_position_ids": text_position_ids[:, :, valid_text - 1 : valid_text],
            "full_position_ids": position_ids,
            "visual_memories": visual_memories,
            "visual_memory_cache": visual_memory_cache,
            "visual_kv_cache": visual_kv_cache,
            "visual_memory_state": visual_memory_state,
            "visual_position_ids": visual_position_ids,
            "visual_position_embeddings": visual_position_embeddings,
            "image_positions": image_pos,
            "image_mask": image_mask,
            "timing": timing,
            "profile_sidecar_timing": profile_sidecar_timing,
            "text_tokens": int(text_mask.sum().item()),
            "image_tokens": int(image_mask.sum().item()),
        }

    if mode != "hybrid":
        raise ValueError(f"unsupported custom mode: {mode}")
    attention_mask_2d = inputs["attention_mask"].to(dtype=torch.long)
    text_position_embeddings = language_model.rotary_emb(gather_batched_positions(h, text_pos, text_mask).to(dtype=dtype), text_position_ids)
    for layer_idx in range(len(language_model.layers)):
        text_hidden = gather_batched_positions(h, text_pos, text_mask)
        _visual_memory, visual_kv, visual_memory_state = profiled_call(
            profile_sidecar_timing,
            timing,
            "prefill_visual_kv",
            device,
            lambda layer_idx=layer_idx, text_hidden=text_hidden, visual_memory_state=visual_memory_state: get_sidecar_visual_for_layer(
                sidecar,
                language_model,
                layer_idx,
                visual_memories,
                visual_memory_state,
                visual_memory_cache,
                visual_kv_cache,
                text_hidden,
                position_ids,
                text_position_ids,
                image_pos,
                image_mask,
                visual_position_ids,
                dtype,
                visual_kv_source,
                compact_qwen_visual_kv,
                visual_position_embeddings,
            ),
        )
        text_attention, text_heads, query_states = profiled_call(
            profile_sidecar_timing,
            timing,
            "prefill_text_attention",
            device,
            lambda layer_idx=layer_idx, text_hidden=text_hidden: qwen3vl_text_attention_outputs_cache(
                language_model,
                layer_idx,
                text_hidden,
                text_position_ids,
                past_key_values=text_cache,
                attention_mask_2d=text_mask.to(dtype=torch.long),
                position_embeddings=text_position_embeddings,
                mask_past_key_values=None,
                return_query_states=True,
            ),
        )
        delta = profiled_call(
            profile_sidecar_timing,
            timing,
            "prefill_sidecar_delta",
            device,
            lambda layer_idx=layer_idx, text_hidden=text_hidden, visual_kv=visual_kv, text_attention=text_attention, text_heads=text_heads, query_states=query_states: sidecar_delta_for_layer(
                sidecar,
                language_model,
                layer_idx,
                text_hidden,
                text_position_ids,
                visual_kv,
                text_attention if sidecar.output_mode.startswith("factorized") else None,
                text_heads if sidecar.output_mode in {"factorized_native_head_o", "factorized_native_head_o_residual"} else None,
                query_source,
                sidecar_scale,
                query_states if query_source == "qwen_native" else None,
            ),
        )
        h = profiled_call(
            profile_sidecar_timing,
            timing,
            "prefill_block_tail",
            device,
            lambda layer_idx=layer_idx, h=h, delta=delta: run_qwen3vl_full_layer_with_text_delta_cache(
                language_model,
                layer_idx,
                h,
                position_ids,
                attention_mask_2d,
                text_pos,
                delta,
                past_key_values=full_cache,
                mask_past_key_values=None,
            ),
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            h = language_model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
    valid_text = int(text_mask[0].sum().item())
    last_text_pos = int(text_pos[0, valid_text - 1].item())
    logits = model.lm_head(language_model.norm(h))[0, last_text_pos]
    return logits, {
        "text_cache": text_cache,
        "full_cache": full_cache,
        "last_position_ids": position_ids[:, :, last_text_pos : last_text_pos + 1],
        "full_position_ids": position_ids,
        "visual_memories": visual_memories,
        "visual_memory_cache": visual_memory_cache,
        "visual_kv_cache": visual_kv_cache,
        "visual_memory_state": visual_memory_state,
        "visual_position_ids": visual_position_ids,
        "visual_position_embeddings": visual_position_embeddings,
        "image_positions": image_pos,
        "image_mask": image_mask,
        "timing": timing,
        "profile_sidecar_timing": profile_sidecar_timing,
        "text_tokens": int(text_mask.sum().item()),
        "image_tokens": int(image_mask.sum().item()),
    }


@torch.inference_mode()
def custom_decode_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    token_id: int,
    step: int,
    context: dict[str, Any],
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    sidecar_scale: float,
    query_source: str,
    visual_kv_source: str,
    compact_qwen_visual_kv: bool,
    sidecar_visual_cache_mode: str,
    profile_sidecar_timing: bool,
) -> torch.Tensor:
    token = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
    h = language_model.embed_tokens(token).to(dtype=dtype)
    position_ids = decode_position_ids(context["last_position_ids"], step)
    position_embeddings = language_model.rotary_emb(h, position_ids)
    image_mask = context["image_mask"]
    image_pos = context["image_positions"]
    visual_position_ids = context["visual_position_ids"]
    visual_position_embeddings = context.get("visual_position_embeddings")
    text_cache: DynamicCache = context["text_cache"]
    full_cache: DynamicCache | None = context["full_cache"]
    visual_memory_state = context["visual_memory_state"]
    visual_memories = context["visual_memories"]
    visual_memory_cache = context.get("visual_memory_cache")
    visual_kv_cache = context.get("visual_kv_cache")
    timing = context.get("timing")

    for layer_idx in range(len(language_model.layers)):
        if mode == "prefill_only":
            h = profiled_call(
                profile_sidecar_timing,
                timing,
                "decode_text_layer",
                device,
                lambda layer_idx=layer_idx, h=h: run_qwen3vl_layer_text_with_attention_delta_cache(
                    language_model,
                    layer_idx,
                    h,
                    position_ids,
                    attention_delta=None,
                    past_key_values=text_cache,
                    attention_mask_2d=None,
                    position_embeddings=position_embeddings,
                    mask_past_key_values=text_cache,
                ),
            )
            continue
        if mode == "prefill_hf":
            outputs = profiled_call(
                profile_sidecar_timing,
                timing,
                "decode_hf_text_model",
                device,
                lambda h=h: language_model(
                    input_ids=token,
                    position_ids=position_ids,
                    past_key_values=text_cache,
                    use_cache=True,
                ),
            )
            h = outputs.last_hidden_state[:, -1:, :].to(dtype=dtype)
            continue
        _visual_memory, visual_kv, visual_memory_state = profiled_call(
            profile_sidecar_timing,
            timing,
            "decode_visual_kv",
            device,
            lambda layer_idx=layer_idx, h=h, visual_memory_state=visual_memory_state: get_sidecar_visual_for_layer(
                sidecar,
                language_model,
                layer_idx,
                visual_memories,
                visual_memory_state,
                visual_memory_cache,
                visual_kv_cache,
                h,
                context["full_position_ids"],
                position_ids,
                image_pos,
                image_mask,
                visual_position_ids,
                dtype,
                visual_kv_source,
                compact_qwen_visual_kv,
                visual_position_embeddings,
            ),
        )
        text_attention, text_heads, query_states = profiled_call(
            profile_sidecar_timing,
            timing,
            "decode_text_attention",
            device,
                lambda layer_idx=layer_idx, h=h: qwen3vl_text_attention_outputs_cache(
                    language_model,
                    layer_idx,
                    h,
                    position_ids,
                    past_key_values=text_cache,
                    attention_mask_2d=None,
                    position_embeddings=position_embeddings,
                    mask_past_key_values=text_cache,
                    return_query_states=True,
                ),
        )
        delta = profiled_call(
            profile_sidecar_timing,
            timing,
            "decode_sidecar_delta",
            device,
            lambda layer_idx=layer_idx, h=h, visual_kv=visual_kv, text_attention=text_attention, text_heads=text_heads, query_states=query_states: sidecar_delta_for_layer(
                sidecar,
                language_model,
                layer_idx,
                h,
                position_ids,
                visual_kv,
                text_attention if sidecar.output_mode.startswith("factorized") else None,
                text_heads if sidecar.output_mode in {"factorized_native_head_o", "factorized_native_head_o_residual"} else None,
                query_source,
                sidecar_scale,
                query_states if query_source == "qwen_native" else None,
            ),
        )
        if mode == "sidecar_only":
            h = profiled_call(
                profile_sidecar_timing,
                timing,
                "decode_block_tail",
                device,
                lambda layer_idx=layer_idx, h=h, text_attention=text_attention, delta=delta: run_qwen3vl_layer_text_from_attention_output(
                    language_model,
                    layer_idx,
                    h,
                    text_attention,
                    delta,
                ),
            )
        elif mode == "hybrid":
            if full_cache is None:
                raise RuntimeError("hybrid decode requires full_cache")
            h = profiled_call(
                profile_sidecar_timing,
                timing,
                "decode_block_tail",
                device,
                lambda layer_idx=layer_idx, h=h, delta=delta: run_qwen3vl_full_layer_with_text_delta_cache(
                    language_model,
                    layer_idx,
                    h,
                    position_ids,
                    full_attention_mask_for_cache(full_cache, device),
                    None,
                    delta,
                    past_key_values=full_cache,
                    mask_past_key_values=full_cache,
                ),
            )
        else:
            raise ValueError(f"unsupported custom mode: {mode}")
    context["visual_memory_state"] = visual_memory_state
    return model.lm_head(language_model.norm(h))[0, -1]


@torch.inference_mode()
def generate_custom(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    row: dict[str, Any],
    benchmark: str,
    mode: str,
    max_new_tokens: int,
    ignore_eos: bool,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    sidecar_scale: float,
    query_source: str,
    visual_kv_source: str,
    compact_qwen_visual_kv: bool,
    sidecar_visual_cache_mode: str,
    profile_sidecar_timing: bool,
) -> dict[str, Any]:
    baseline = reset_peak(device)
    sync(device)
    prefill_start = time.perf_counter()
    logits, context = custom_prefill(
        processor,
        model,
        language_model,
        sidecar,
        row,
        benchmark,
        mode,
        device,
        dtype,
        visual_memory_mode,
        sidecar_scale,
        query_source,
        visual_kv_source,
        compact_qwen_visual_kv,
        sidecar_visual_cache_mode,
        profile_sidecar_timing,
    )
    sync(device)
    prefill_sec = time.perf_counter() - prefill_start
    generated_ids: list[int] = []
    stop_ids = set(generation_stop_ids(processor.tokenizer, ignore_eos=ignore_eos) or [])
    decode_start = time.perf_counter()
    for step in range(max_new_tokens):
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        if stop_ids and next_id in stop_ids:
            break
        logits = custom_decode_logits(
            model,
            language_model,
            sidecar,
            next_id,
            step,
            context,
            mode,
            device,
            dtype,
            sidecar_scale,
            query_source,
            visual_kv_source,
            compact_qwen_visual_kv,
            sidecar_visual_cache_mode,
            profile_sidecar_timing,
        )
    sync(device)
    decode_sec = time.perf_counter() - decode_start
    max_alloc, max_reserved, incremental = peak(device, baseline)
    total_sec = prefill_sec + decode_sec
    return {
        "mode": mode,
        "prompt_tokens": context["text_tokens"] if mode in {"sidecar_only", "prefill_only", "prefill_hf"} else context["text_tokens"] + context["image_tokens"],
        "text_tokens": context["text_tokens"],
        "image_tokens": context["image_tokens"],
        "generated_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_text": decode_generated_text(processor.tokenizer, generated_ids),
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "total_sec": total_sec,
        "tokens_per_sec": len(generated_ids) / max(total_sec, 1e-9),
        "decode_tokens_per_sec": len(generated_ids) / max(decode_sec, 1e-9),
        "max_allocated_mib": max_alloc,
        "max_reserved_mib": max_reserved,
        "incremental_peak_mib": incremental,
        "timing": context.get("timing", {}) if profile_sidecar_timing else {},
    }


def sample_accounting(
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule | None,
    image_tokens: int,
    dtype: torch.dtype,
    compact_qwen_visual_kv: bool,
) -> dict[str, float]:
    native_visual_kv = qwen_visual_kv_bytes(language_model, image_tokens, dtype)
    v0_hidden = one_layer_visual_hidden_bytes(language_model, image_tokens, dtype)
    external_temp = (
        sidecar_external_qwen_compact_kv_temp_bytes(language_model, image_tokens, dtype)
        if compact_qwen_visual_kv
        else sidecar_external_qwen_kv_temp_bytes(language_model, image_tokens, dtype)
    )
    return {
        "qwen_persistent_visual_kv_mib": bytes_to_mib(native_visual_kv),
        "sidecar_persistent_v0_mib": bytes_to_mib(v0_hidden),
        "sidecar_cache_all_visual_memory_v_l_mib_if_materialized": bytes_to_mib(v0_hidden * len(language_model.layers)),
        "sidecar_one_layer_external_visual_kv_temp_mib": bytes_to_mib(external_temp),
        "sidecar_cache_all_external_visual_kv_mib_if_materialized": bytes_to_mib(external_temp * len(language_model.layers)),
        "sidecar_param_mib": bytes_to_mib(sidecar_param_bytes(sidecar)),
        "persistent_visual_kv_saved_mib_sidecar_only": bytes_to_mib(native_visual_kv - v0_hidden),
        "qwen_visual_kv_over_sidecar_v0": float(native_visual_kv / max(v0_hidden, 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark Qwen3-VL native vs Delta-Vision cached decode.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), default="mmstar")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", default="artifacts/bench/qwen_decode_cache.json")
    parser.add_argument("--modes", default="qwen,text_only,sidecar_only,hybrid")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--visual-memory-mode", default="v0")
    parser.add_argument("--compact-qwen-visual-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sidecar-visual-cache-mode",
        choices=("stream", "cache_v", "cache_kv"),
        default="stream",
        help=(
            "stream recomputes per-layer sidecar visual memory and K/V on every decode token; "
            "cache_v materializes all per-layer V_l once; cache_kv materializes all per-layer external visual K/V once."
        ),
    )
    parser.add_argument(
        "--profile-sidecar-timing",
        action="store_true",
        help="Synchronize and record sidecar decode timing sections. Diagnostic only; it slows generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, None)[args.start_index : args.start_index + args.max_samples]
    if not rows:
        raise ValueError(f"no rows loaded from {args.data}")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    sidecar = None
    sidecar_args = None
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if any(mode in {"sidecar_only", "prefill_only", "prefill_hf", "hybrid"} for mode in modes):
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for sidecar_only/hybrid modes")
        sidecar_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            rank=512,
            sidecar_dim=1024,
            num_heads=8,
            state_tokens=0,
            reader_mlp_ratio=4.0,
            reader_activation="gelu",
            layer_adapter_rank=128,
            block_corrector_groups="",
            block_corrector_dim=0,
            shared_basis=True,
            output_mode="residual",
            visual_memory_mode=args.visual_memory_mode,
            sidecar_scale=args.sidecar_scale,
            use_rope=False,
            sidecar_query_source="sidecar",
            sidecar_visual_kv_source="sidecar",
            factorized_mass_mode="learned",
            fixed_visual_mass=0.12,
        )
        sidecar = load_sidecar(sidecar_args, int(language_model.config.hidden_size), len(language_model.layers), device, dtype)
        args.visual_memory_mode = str(getattr(sidecar_args, "visual_memory_mode", args.visual_memory_mode))

    per_sample: list[dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        row_record: dict[str, Any] = {"row_index": args.start_index + sample_idx, "image": row.get("image"), "modes": {}}
        row_record["answer"] = str(row.get("answer", "")).strip()
        image_tokens_for_accounting = None
        for mode in modes:
            if mode in {"qwen", "text_only"}:
                result = generate_hf(
                    processor,
                    model,
                    row,
                    args.benchmark,
                    mode,
                    args.max_new_tokens,
                    args.ignore_eos,
                    device,
                )
            elif mode in {"sidecar_only", "prefill_only", "prefill_hf", "hybrid"}:
                assert sidecar is not None and sidecar_args is not None
                result = generate_custom(
                    processor,
                    model,
                    language_model,
                    sidecar,
                    row,
                    args.benchmark,
                    mode,
                    args.max_new_tokens,
                    args.ignore_eos,
                    device,
                    dtype,
                    args.visual_memory_mode,
                    args.sidecar_scale,
                    str(getattr(sidecar_args, "sidecar_query_source", "sidecar")),
                    str(getattr(sidecar_args, "sidecar_visual_kv_source", "sidecar")),
                    args.compact_qwen_visual_kv,
                    args.sidecar_visual_cache_mode,
                    args.profile_sidecar_timing,
                )
                image_tokens_for_accounting = int(result["image_tokens"])
            else:
                raise ValueError(f"unknown mode: {mode}")
            if args.benchmark == "mmstar":
                prediction = parse_mmstar_answer(str(result.get("generated_text", "")))
                answer = str(row_record["answer"]).upper()[:1]
                result["prediction"] = prediction
                result["correct"] = bool(prediction == answer) if prediction is not None else False
            elif args.benchmark == "realworldqa":
                prediction, correct = parse_realworldqa_answer(row, str(result.get("generated_text", "")))
                result["prediction"] = prediction
                result["correct"] = correct
            row_record["modes"][mode] = result
            if image_tokens_for_accounting is None and "image_tokens" in result:
                image_tokens_for_accounting = int(result["image_tokens"])
        if image_tokens_for_accounting is None:
            inputs = prepare_inputs(processor, row, args.benchmark, include_image=True, device=device)
            hidden0, position_ids, _visual_pos_masks, _deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
            _, _, _, _, image_mask, _ = get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                position_ids,
            )
            image_tokens_for_accounting = int(image_mask.sum().item())
            del hidden0, position_ids
        row_record["accounting"] = sample_accounting(
            language_model,
            sidecar,
            image_tokens_for_accounting,
            dtype,
            args.compact_qwen_visual_kv,
        )
        per_sample.append(row_record)
        print(json.dumps(row_record, ensure_ascii=False), flush=True)

    aggregate: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_rows = [row["modes"][mode] for row in per_sample if mode in row["modes"]]
        if not mode_rows:
            continue
        numeric_keys = [key for key, value in mode_rows[0].items() if isinstance(value, (int, float))]
        aggregate[mode] = {
            key: sum(float(row[key]) for row in mode_rows) / len(mode_rows)
            for key in numeric_keys
        }
        if args.benchmark in {"mmstar", "realworldqa"}:
            aggregate[mode]["accuracy"] = sum(float(row.get("correct", False)) for row in mode_rows) / len(mode_rows)
            aggregate[mode]["parse_rate"] = sum(float(row.get("prediction") is not None) for row in mode_rows) / len(mode_rows)
    accounting_keys = list(per_sample[0]["accounting"].keys())
    aggregate_accounting = {
        key: sum(float(row["accounting"][key]) for row in per_sample) / len(per_sample)
        for key in accounting_keys
    }
    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "sidecar_scale": args.sidecar_scale,
        "visual_memory_mode": args.visual_memory_mode,
        "compact_qwen_visual_kv": args.compact_qwen_visual_kv,
        "sidecar_visual_cache_mode": args.sidecar_visual_cache_mode,
        "profile_sidecar_timing": args.profile_sidecar_timing,
        "sidecar_config": vars(sidecar_args) if sidecar_args is not None else None,
        "aggregate": aggregate,
        "aggregate_accounting": aggregate_accounting,
        "samples": per_sample,
        "notes": [
            "qwen/text_only use HuggingFace generate with use_cache=True.",
            "sidecar_only keeps only text/generated tokens in the main LLM cache.",
            "sidecar_visual_cache_mode=stream recomputes/discards per-layer external visual V/KV; cache_v materializes all V_l once; cache_kv materializes all external visual K/V once.",
            "compact_qwen_visual_kv stores external Qwen-native visual K/V in GQA KV-head layout instead of repeated query-head layout.",
            "hybrid keeps native full visual K/V in the main LLM cache and adds sidecar deltas, so it is an enhancement baseline rather than a memory-saving mode.",
            "For factorized_native_head_o checkpoints, custom decode computes cached text-only attention heads and updates the text KV cache exactly once per layer.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(output), "aggregate": aggregate, "aggregate_accounting": aggregate_accounting}, indent=2), flush=True)


if __name__ == "__main__":
    main()
