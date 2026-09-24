"""Generation-based visual attention-effect SVD benchmark sweep.

This evaluates the delta-vision style oracle

    DeltaA_l = A_joint_l(text positions) - A_text_l

by generating answers, not by scoring answer-option logits directly. The
generated text is scored with the same benchmark metric helpers used by the
repo eval path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    repeat_kv as llama_repeat_kv,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmarks import build_benchmark_prompt, get_benchmark_spec, score_prediction, summarize_metric
from src.evaluate import _eos_token_ids, _structured_answer_ready, parse_qwen_device_map, parse_qwen_max_memory
from src.model import (
    dtype_from_name,
    gather_batched_positions,
    get_qwen_text_image_positions,
    load_frozen_llava,
    load_frozen_qwen3vl,
    module_device,
    qwen_prompt,
    qwen_input_device,
    qwen_position_ids,
    resolve_row_image_path,
    resolve_row_image_paths,
)
from analysis.fig03_visual_effect.native_effect_core import (
    load_layer_basis,
    logits_from_text_hidden,
    project_reconstruct_delta,
    qwen_layer_text_with_attention_delta,
    qwen_visual_attention_effect,
    read_jsonl,
    save_basis_atomic,
    select_contiguous_shard,
)


DEFAULT_QWEN4B = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_LLAVA7B = "/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf"
DEFAULT_RANKS = "16,32,64,128,256,512,1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generation-based visual-effect SVD benchmark sweep.")
    parser.add_argument("--model-kind", choices=("qwen", "llava"), required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--benchmark", choices=("sqa", "realworldqa", "mmstar"), required=True)
    parser.add_argument("--data", default="", help="Defaults to the benchmark registry path.")
    parser.add_argument("--image-root", default="", help="Defaults to the JSONL parent directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basis", default="")
    parser.add_argument("--reuse-basis", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("build-and-eval", "build-basis", "eval"),
        default="build-and-eval",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="0 means all rows.")
    parser.add_argument("--basis-max-samples", type=int, default=0, help="0 means use --max-samples.")
    parser.add_argument("--max-rank", type=int, default=1024)
    parser.add_argument("--ranks", default=DEFAULT_RANKS)
    parser.add_argument(
        "--rank-layer-scopes",
        default="all",
        help="Comma-separated layer scopes for rank modes: all,first5,last5. "
        "For first/last scopes, non-selected layers keep the full effect.",
    )
    parser.add_argument(
        "--zero-layer-scopes",
        default="",
        help="Comma-separated layer scopes to zero out visual attention effect: firstN,lastN,all. "
        "Non-selected layers keep the full effect.",
    )
    parser.add_argument(
        "--causal-zero-layer-scopes",
        default="",
        help="Comma-separated layer scopes for causal Qwen text->visual attention blocking: firstN,lastN,all. "
        "Unlike --zero-layer-scopes, this intervenes inside the full image+text forward and cascades naturally.",
    )
    parser.add_argument("--max-tokens-per-layer", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 means benchmark default.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="", help="Defaults to flash_attention_2 for Qwen, eager for LLaVA.")
    parser.add_argument("--qwen-device-map", default="")
    parser.add_argument("--qwen-max-memory", default="")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Qwen-only batch size for basis/eval.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def parse_ranks(spec: str, max_rank: int) -> list[int]:
    ranks = sorted({int(item) for item in str(spec).split(",") if item.strip()})
    if not ranks:
        return []
    if min(ranks) <= 0:
        raise ValueError("--ranks must be positive")
    if max(ranks) > int(max_rank):
        raise ValueError(f"requested rank {max(ranks)} exceeds --max-rank {max_rank}")
    return ranks


def parse_rank_layer_scopes(spec: str) -> tuple[str, ...]:
    scopes = tuple(dict.fromkeys(item.strip().lower() for item in str(spec).split(",") if item.strip()))
    if not scopes:
        return ()
    return scopes


def layer_scope_indices(scope: str, num_layers: int) -> set[int] | None:
    if scope == "all":
        return None
    match = re.fullmatch(r"first(\d+)", scope)
    if match:
        count = min(int(match.group(1)), int(num_layers))
        return set(range(count))
    match = re.fullmatch(r"last(\d+)", scope)
    if match:
        count = min(int(match.group(1)), int(num_layers))
        return set(range(int(num_layers) - count, int(num_layers)))
    raise ValueError(f"unsupported layer scope: {scope}; expected all, firstN, or lastN")


def rank_mode_name(rank: int, scope: str) -> str:
    return f"rank_{int(rank)}" if scope == "all" else f"rank_{int(rank)}_{scope}"


def zero_mode_name(scope: str) -> str:
    return "zero_visual" if scope == "all" else f"zero_{scope}"


def causal_zero_mode_name(scope: str) -> str:
    return "causal_zero_visual" if scope == "all" else f"causal_zero_{scope}"


def parse_rank_rollout(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"rank_(\d+)(?:_(.+))?", name)
    if not match:
        raise ValueError(f"unsupported rank rollout mode: {name}")
    return int(match.group(1)), (match.group(2) or "all")


def parse_zero_rollout(name: str) -> str:
    if name == "zero_visual":
        return "all"
    match = re.fullmatch(r"zero_(.+)", name)
    if not match:
        raise ValueError(f"unsupported zero rollout mode: {name}")
    return match.group(1)


def parse_causal_zero_rollout(name: str) -> str:
    if name == "causal_zero_visual":
        return "all"
    match = re.fullmatch(r"causal_zero_(.+)", name)
    if not match:
        raise ValueError(f"unsupported causal zero rollout mode: {name}")
    return match.group(1)


def move_tensors(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def append_generated_to_inputs(inputs: dict[str, Tensor], generated_ids: list[int]) -> dict[str, Tensor]:
    if not generated_ids:
        return inputs
    out = dict(inputs)
    device = inputs["input_ids"].device
    gen = torch.tensor([generated_ids], device=device, dtype=inputs["input_ids"].dtype)
    out["input_ids"] = torch.cat([inputs["input_ids"], gen], dim=1)
    out["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(gen)], dim=1)
    if "mm_token_type_ids" in inputs:
        out["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(gen)], dim=1)
    return out


def append_generated_to_batch_inputs(
    inputs: dict[str, Tensor],
    generated_ids: list[list[int]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    batch = int(inputs["input_ids"].shape[0])
    if len(generated_ids) != batch:
        raise ValueError(f"generated batch size mismatch: got {len(generated_ids)} expected {batch}")
    device = inputs["input_ids"].device
    rows_ids: list[Tensor] = []
    rows_mm: list[Tensor] = []
    max_len = 0
    for batch_idx in range(batch):
        valid = inputs["attention_mask"][batch_idx].to(device=device, dtype=torch.bool)
        src_ids = inputs["input_ids"][batch_idx][valid]
        if "mm_token_type_ids" in inputs:
            src_mm = inputs["mm_token_type_ids"][batch_idx][valid]
        else:
            src_mm = torch.zeros_like(src_ids)
        gen = torch.tensor(generated_ids[batch_idx], device=device, dtype=src_ids.dtype)
        if gen.numel() > 0:
            src_ids = torch.cat([src_ids, gen], dim=0)
            src_mm = torch.cat([src_mm, torch.zeros_like(gen)], dim=0)
        rows_ids.append(src_ids)
        rows_mm.append(src_mm)
        max_len = max(max_len, int(src_ids.numel()))

    out = dict(inputs)
    out_ids = torch.full((batch, max_len), int(pad_token_id), device=device, dtype=inputs["input_ids"].dtype)
    out_mask = torch.zeros((batch, max_len), device=device, dtype=inputs["attention_mask"].dtype)
    out_mm = torch.zeros((batch, max_len), device=device, dtype=inputs["mm_token_type_ids"].dtype)
    for batch_idx, row_ids in enumerate(rows_ids):
        row_len = int(row_ids.numel())
        out_ids[batch_idx, :row_len] = row_ids
        out_mask[batch_idx, :row_len] = 1
        out_mm[batch_idx, :row_len] = rows_mm[batch_idx]
    out["input_ids"] = out_ids
    out["attention_mask"] = out_mask
    out["mm_token_type_ids"] = out_mm
    return out


def generated_batch_arg(inputs: dict[str, Tensor], generated_ids: list[int] | list[list[int]]) -> list[list[int]]:
    batch = int(inputs["input_ids"].shape[0])
    if batch == 1 and (not generated_ids or isinstance(generated_ids[0], int)):
        return [list(generated_ids)]  # type: ignore[list-item]
    if len(generated_ids) != batch:
        raise ValueError(f"generated batch size mismatch: got {len(generated_ids)} expected {batch}")
    return [list(row_ids) for row_ids in generated_ids]  # type: ignore[union-attr]


def row_batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def generated_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def should_stop_generation(tokenizer: Any, token_id: int, text: str, metric: str, choices: list[Any] | None) -> bool:
    if int(token_id) in _eos_token_ids(tokenizer):
        return True
    return _structured_answer_ready(metric, text, choices)


def qwen_text_to_visual_block_mask(
    attention_mask: Tensor | None,
    *,
    mm_token_type_ids: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if attention_mask is None:
        raise ValueError("causal visual blocking requires materialized attention_mask; use eager attention")
    if attention_mask.ndim != 4:
        raise ValueError(f"causal visual blocking expects a 4D attention mask, got shape={tuple(attention_mask.shape)}")

    device = attention_mask.device
    token_types = mm_token_type_ids.to(device=device)
    valid = valid_mask.to(device=device, dtype=torch.bool)
    query_len = int(attention_mask.shape[-2])
    key_len = int(attention_mask.shape[-1])
    if token_types.shape[1] < max(query_len, key_len):
        raise ValueError(
            f"mm_token_type_ids length {token_types.shape[1]} is shorter than attention mask "
            f"query/key lengths {query_len}/{key_len}"
        )

    query_types = token_types[:, -query_len:]
    query_valid = valid[:, -query_len:]
    key_types = token_types[:, -key_len:]
    key_valid = valid[:, -key_len:]
    text_queries = query_valid & (query_types == 0)
    visual_keys = key_valid & (key_types == 1)
    block = text_queries[:, None, :, None] & visual_keys[:, None, None, :]
    if not bool(block.any().item()):
        return attention_mask

    patched = attention_mask.clone()
    if patched.dtype == torch.bool:
        return patched.masked_fill(block, False)
    if not torch.is_floating_point(patched):
        raise ValueError(f"unsupported attention mask dtype for causal visual blocking: {patched.dtype}")
    return patched.masked_fill(block, torch.finfo(patched.dtype).min)


@contextmanager
def qwen_causal_visual_block_context(
    language_model: torch.nn.Module,
    *,
    blocked_layers: set[int] | None,
    mm_token_type_ids: Tensor,
    valid_mask: Tensor,
):
    if blocked_layers is None:
        selected = set(range(len(language_model.layers)))
    else:
        selected = set(blocked_layers)
    originals: list[tuple[torch.nn.Module, Any]] = []

    try:
        for layer_idx in sorted(selected):
            layer = language_model.layers[layer_idx]
            attn = layer.self_attn
            original_forward = attn.forward

            def wrapped_forward(*args: Any, _original_forward=original_forward, **kwargs: Any):
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = qwen_text_to_visual_block_mask(
                    kwargs.get("attention_mask"),
                    mm_token_type_ids=mm_token_type_ids,
                    valid_mask=valid_mask,
                )
                return _original_forward(*args, **kwargs)

            originals.append((attn, original_forward))
            attn.forward = wrapped_forward  # type: ignore[method-assign]
        yield
    finally:
        for attn, original_forward in originals:
            attn.forward = original_forward  # type: ignore[method-assign]


def score_text_prediction(spec: Any, row: dict[str, Any], prediction_text: str) -> dict[str, Any]:
    return score_prediction(
        metric=spec.metric,
        prediction_text=prediction_text,
        answer=row.get("answer"),
        answers=row.get("answers"),
        choices=row.get("choices"),
        question=row.get("question"),
    )


def llava15_prompt(question: str) -> str:
    return f"USER: <image>\n{question.strip()}\nASSISTANT:"


def llava_image_token_id(model: Any, processor: Any) -> int:
    token_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if isinstance(token_id, int):
        return token_id
    token_id = getattr(processor, "image_token_id", None)
    if isinstance(token_id, int):
        return token_id
    converted = processor.tokenizer.convert_tokens_to_ids("<image>")
    if not isinstance(converted, int) or converted < 0:
        raise ValueError("could not resolve LLaVA image token id")
    return int(converted)


def llava_text_and_image_positions(
    input_ids: Tensor,
    merged_len: int,
    image_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("LLaVA SVD generation currently expects batch size 1")
    ids = input_ids[0].tolist()
    image_indices = [idx for idx, token_id in enumerate(ids) if int(token_id) == int(image_token_id)]
    if len(image_indices) > 1:
        text_positions = [idx for idx, token_id in enumerate(ids) if int(token_id) != int(image_token_id)]
        image_positions = image_indices
        return (
            torch.tensor(text_positions, dtype=torch.long),
            torch.tensor(image_positions, dtype=torch.long),
            torch.tensor(text_positions, dtype=torch.long),
        )
    if len(image_indices) != 1:
        raise ValueError(f"expected one <image> token, found {len(image_indices)}")

    image_idx = image_indices[0]
    num_image_tokens = int(merged_len) - (int(input_ids.shape[1]) - 1)
    if num_image_tokens <= 0:
        raise ValueError("could not infer a positive number of merged image tokens")

    text_positions: list[int] = []
    prompt_positions: list[int] = []
    for src_idx in range(input_ids.shape[1]):
        if src_idx == image_idx:
            continue
        merged_idx = src_idx if src_idx < image_idx else src_idx + num_image_tokens - 1
        text_positions.append(int(merged_idx))
        prompt_positions.append(int(merged_idx))
    image_positions = list(range(image_idx, image_idx + num_image_tokens))
    return (
        torch.tensor(text_positions, dtype=torch.long),
        torch.tensor(image_positions, dtype=torch.long),
        torch.tensor(prompt_positions, dtype=torch.long),
    )


def llama_attention_output(
    self_attn: torch.nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    *,
    is_causal: bool,
) -> Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, *position_embeddings)

    groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key_states = llama_repeat_kv(key_states, groups)
    value_states = llama_repeat_kv(value_states, groups)
    attn_mask = attention_mask
    if attn_mask is not None and not is_causal:
        attn_mask = attn_mask[:, :, :, : key_states.shape[-2]]
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=float(getattr(self_attn, "scaling", self_attn.head_dim ** -0.5)),
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self_attn.o_proj(attn_output)


def llava_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: Tensor,
    text_positions: Tensor,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    device = module_device(layer, full_hidden_states.device)
    full_hidden_states = full_hidden_states.to(device=device)
    text_positions = text_positions.to(device=device)
    text_hidden_states = full_hidden_states.index_select(1, text_positions)
    full_position_ids = torch.arange(full_hidden_states.shape[1], device=device).unsqueeze(0)
    text_position_ids = text_positions.unsqueeze(0)

    full_normed = layer.input_layernorm(full_hidden_states)
    text_normed = layer.input_layernorm(text_hidden_states)
    full_pos_emb = language_model.rotary_emb(full_normed, full_position_ids)
    text_pos_emb = language_model.rotary_emb(text_normed, text_position_ids)
    joint = llama_attention_output(layer.self_attn, full_normed, full_pos_emb, None, is_causal=True)
    text = llama_attention_output(layer.self_attn, text_normed, text_pos_emb, None, is_causal=True)
    return joint.index_select(1, text_positions) - text


def llava_layer_text_with_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_delta: Tensor | None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    device = module_device(layer, hidden_states.device)
    hidden_states = hidden_states.to(device=device)
    position_ids = position_ids.to(device=device)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    pos_emb = language_model.rotary_emb(normed, position_ids)
    attn_out = llama_attention_output(layer.self_attn, normed, pos_emb, None, is_causal=True)
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(device=hidden_states.device, dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


class GenerationSvdEngine:
    def __init__(
        self,
        *,
        model_kind: str,
        model_path: str,
        benchmark: str,
        image_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        attn_implementation: str,
        qwen_device_map: str | dict[str, Any] | None,
        qwen_max_memory: dict[Any, str] | None,
    ) -> None:
        self.model_kind = model_kind
        self.model_path = model_path
        self.spec = get_benchmark_spec(benchmark)
        self.image_root = image_root
        self.device = device
        self.dtype = dtype
        if model_kind == "qwen":
            self.processor, self.model = load_frozen_qwen3vl(
                model_path,
                dtype,
                device,
                attn_implementation or "flash_attention_2",
                device_map=qwen_device_map,
                max_memory=qwen_max_memory,
            )
            if qwen_device_map is not None:
                self.device = qwen_input_device(self.model)
            self.language_model = self.model.model.language_model
            self.image_token_id = None
        elif model_kind == "llava":
            self.processor, self.model = load_frozen_llava(
                model_path,
                dtype,
                str(device),
                attn_implementation or "eager",
            )
            self.language_model = self.model.model.language_model
            self.image_token_id = llava_image_token_id(self.model, self.processor)
        else:
            raise ValueError(f"unsupported model kind: {model_kind}")
        self.tokenizer = self.processor.tokenizer
        self.num_layers = len(self.language_model.layers)
        self.hidden_size = int(self.language_model.config.hidden_size)

    def benchmark_question(self, row: dict[str, Any]) -> str:
        return build_benchmark_prompt(row, self.spec)

    def prepare_inputs(self, row: dict[str, Any]) -> dict[str, Tensor]:
        question = self.benchmark_question(row)
        if self.model_kind == "qwen":
            image_path = resolve_row_image_path(row, self.image_root)
            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB").copy()
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image_rgb}, {"type": "text", "text": question}],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image_rgb], return_tensors="pt", padding=True)
            image_rgb.close()
            if "mm_token_type_ids" not in inputs:
                raise ValueError("Qwen processor did not return mm_token_type_ids")
            return move_tensors(inputs, self.device)

        image_path = resolve_row_image_path(row, self.image_root)
        prompt = llava15_prompt(question)
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            inputs = self.processor(text=prompt, images=image_rgb, return_tensors="pt")
        return move_tensors(inputs, self.device)

    def prepare_inputs_batch(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        if self.model_kind != "qwen":
            if len(rows) != 1:
                raise ValueError("LLaVA SVD generation currently supports batch size 1")
            return self.prepare_inputs(rows[0])

        texts: list[str] = []
        images: list[Image.Image] = []
        try:
            for row in rows:
                image_paths = resolve_row_image_paths(row, self.image_root)
                texts.append(qwen_prompt(self.processor, self.benchmark_question(row), num_images=len(image_paths)))
                for image_path in image_paths:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB").copy())
            old_padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
            self.processor.tokenizer.padding_side = "left"
            try:
                inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            finally:
                self.processor.tokenizer.padding_side = old_padding_side
        finally:
            for image in images:
                image.close()
        if "mm_token_type_ids" not in inputs:
            raise ValueError("Qwen processor did not return mm_token_type_ids")
        return move_tensors(inputs, self.device)

    @torch.inference_mode()
    def teacher_generate(self, base_inputs: dict[str, Tensor], max_new_tokens: int) -> str:
        return self.teacher_generate_batch(base_inputs, max_new_tokens)[0]

    @torch.inference_mode()
    def teacher_generate_batch(self, base_inputs: dict[str, Tensor], max_new_tokens: int) -> list[str]:
        if self.model_kind == "qwen" and hasattr(self.model.model, "rope_deltas"):
            self.model.model.rope_deltas = None
        kwargs = dict(base_inputs)
        kwargs.update({"do_sample": False, "max_new_tokens": int(max_new_tokens), "use_cache": True})
        eos_ids = sorted(_eos_token_ids(self.tokenizer))
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        generated = self.model.generate(**kwargs)
        prompt_len = int(base_inputs["input_ids"].shape[1])
        return [
            self.tokenizer.decode(generated[batch_idx, prompt_len:], skip_special_tokens=True).strip()
            for batch_idx in range(int(generated.shape[0]))
        ]

    @torch.inference_mode()
    def causal_visual_block_generate_batch(
        self,
        base_inputs: dict[str, Tensor],
        *,
        blocked_layers: set[int] | None,
        max_new_tokens: int,
        choices_batch: list[list[Any] | None],
    ) -> list[str]:
        if self.model_kind != "qwen":
            raise ValueError("causal visual blocking is currently implemented only for Qwen")
        batch = int(base_inputs["input_ids"].shape[0])
        if len(choices_batch) != batch:
            raise ValueError(f"choices batch size mismatch: got {len(choices_batch)} expected {batch}")
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        generated: list[list[int]] = [[] for _ in range(batch)]
        done = [False for _ in range(batch)]

        for _ in range(int(max_new_tokens)):
            inputs = append_generated_to_batch_inputs(base_inputs, generated, int(pad_id or 0))
            if hasattr(self.model.model, "rope_deltas"):
                self.model.model.rope_deltas = None
            with qwen_causal_visual_block_context(
                self.language_model,
                blocked_layers=blocked_layers,
                mm_token_type_ids=inputs["mm_token_type_ids"],
                valid_mask=inputs["attention_mask"],
            ):
                outputs = self.model(**inputs, return_dict=True, use_cache=False)
            logits = outputs.logits
            valid_lengths = inputs["attention_mask"].to(device=logits.device, dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[batch_idx, valid_lengths]
            next_ids = torch.argmax(next_logits.float(), dim=-1).detach().to(device="cpu").tolist()
            for batch_idx_int, next_id_raw in enumerate(next_ids):
                if done[batch_idx_int]:
                    continue
                next_id = int(next_id_raw)
                generated[batch_idx_int].append(next_id)
                text = generated_text(self.tokenizer, generated[batch_idx_int])
                if should_stop_generation(self.tokenizer, next_id, text, self.spec.metric, choices_batch[batch_idx_int]):
                    done[batch_idx_int] = True
            if all(done):
                break

        return [generated_text(self.tokenizer, token_ids) for token_ids in generated]

    @torch.inference_mode()
    def build_basis(
        self,
        rows: list[dict[str, Any]],
        max_rank: int,
        max_tokens_per_layer: int,
        log_every: int,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        if self.model_kind != "qwen":
            batch_size = 1
        batch_size = max(1, int(batch_size))
        banks: dict[int, list[Tensor]] = {layer_idx: [] for layer_idx in range(self.num_layers)}
        counts = {layer_idx: 0 for layer_idx in range(self.num_layers)}
        processed = 0
        for batch_rows in row_batches(rows, batch_size):
            base_inputs = self.prepare_inputs_batch(batch_rows)
            trace_generated: list[int] | list[list[int]] = [[] for _ in batch_rows] if self.model_kind == "qwen" else []
            trace = self.build_trace(base_inputs, trace_generated)
            valid = trace["text_mask"].to(device=trace["text_hidden0"].device).bool()
            for layer_idx in range(self.num_layers):
                if counts[layer_idx] >= int(max_tokens_per_layer):
                    continue
                delta = trace["attention_deltas"][layer_idx].to(device=valid.device)
                tokens = delta[valid]
                if tokens.numel() == 0:
                    continue
                remaining = int(max_tokens_per_layer) - counts[layer_idx]
                if tokens.shape[0] > remaining:
                    perm = torch.randperm(tokens.shape[0], device=tokens.device)[:remaining]
                    tokens = tokens.index_select(0, perm)
                banks[layer_idx].append(tokens.detach().to(device="cpu", dtype=torch.float16))
                counts[layer_idx] += int(tokens.shape[0])
            processed += len(batch_rows)
            if processed % int(log_every) == 0 or processed == len(rows):
                print(
                    f"basis collect {self.model_kind}/{self.spec.name} {processed}/{len(rows)} "
                    f"min_tokens={min(counts.values())} max_tokens={max(counts.values())}",
                    flush=True,
                )
            if all(counts[layer_idx] >= int(max_tokens_per_layer) for layer_idx in range(self.num_layers)):
                print(f"basis token banks full after {processed}/{len(rows)} rows", flush=True)
                break

        bases: dict[int, Tensor] = {}
        energies: dict[int, Tensor] = {}
        svd_device = self.device
        for layer_idx in range(self.num_layers):
            if not banks[layer_idx]:
                raise RuntimeError(f"empty SVD token bank for layer {layer_idx}")
            matrix = torch.cat(banks[layer_idx], dim=0).float()
            if matrix.shape[1] != self.hidden_size:
                raise RuntimeError(f"hidden size mismatch layer={layer_idx}: got {matrix.shape[1]} expected {self.hidden_size}")
            matrix = matrix - matrix.mean(dim=0, keepdim=True)
            matrix = matrix.to(device=svd_device)
            _, svals, vh = torch.linalg.svd(matrix, full_matrices=False)
            rank = min(int(max_rank), int(vh.shape[0]))
            bases[layer_idx] = vh[:rank].detach().cpu().to(torch.float16)
            energies[layer_idx] = svals.detach().cpu().float().pow(2)
            print(f"basis svd layer={layer_idx} tokens={matrix.shape[0]} rank={rank}", flush=True)
        return {
            "format_version": 3,
            "task": "visual_attention_effect_generation_svd_basis",
            "model_kind": self.model_kind,
            "benchmark": self.spec.name,
            "model_path": self.model_path,
            "basis": bases,
            "singular_energy": energies,
            "counts": counts,
            "max_rank": int(max_rank),
            "max_tokens_per_layer": int(max_tokens_per_layer),
            "num_basis_rows": len(rows),
        }

    @torch.inference_mode()
    def build_trace(self, base_inputs: dict[str, Tensor], generated_ids: list[int] | list[list[int]]) -> dict[str, Any]:
        if self.model_kind == "qwen":
            generated_batch = generated_batch_arg(base_inputs, generated_ids)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0)
            inputs = append_generated_to_batch_inputs(base_inputs, generated_batch, int(pad_id or 0))
            if hasattr(self.model.model, "rope_deltas"):
                self.model.model.rope_deltas = None
            full_position_ids = qwen_position_ids(self.model, inputs)
            teacher = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                full_position_ids,
            )
            teacher_states = [state.detach().to(dtype=self.dtype) for state in teacher.hidden_states[:-1]]
            teacher_text_states = [
                gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=self.dtype)
                for state in teacher_states
            ]
            deltas = []
            for layer_idx in range(self.num_layers):
                delta = qwen_visual_attention_effect(
                    self.language_model,
                    layer_idx,
                    teacher_states[layer_idx],
                    teacher_text_states[layer_idx],
                    full_position_ids,
                    text_position_ids,
                    text_positions,
                    full_mask,
                    text_mask,
                )
                delta = delta.masked_fill(~text_mask.to(device=delta.device).unsqueeze(-1), 0.0)
                deltas.append(delta.detach().cpu().to(torch.float16))
            return {
                "text_hidden0": teacher_text_states[0].detach().cpu().to(torch.float16),
                "text_position_ids": text_position_ids.detach().cpu(),
                "text_mask": text_mask.detach().cpu(),
                "attention_deltas": deltas,
            }

        inputs = append_generated_to_inputs(base_inputs, generated_ids)  # type: ignore[arg-type]
        teacher = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        teacher_states = [state.detach().to(dtype=self.dtype) for state in teacher.hidden_states[:-1]]
        text_positions, _, prompt_positions = llava_text_and_image_positions(
            inputs["input_ids"],
            teacher_states[0].shape[1],
            int(self.image_token_id),
        )
        text_positions = text_positions.to(device=teacher_states[0].device)
        prompt_positions = prompt_positions.to(device=teacher_states[0].device)
        text_hidden0 = teacher_states[0].index_select(1, text_positions)
        deltas = []
        for layer_idx in range(self.num_layers):
            delta = llava_attention_effect(self.language_model, layer_idx, teacher_states[layer_idx], text_positions)
            deltas.append(delta.detach().cpu().to(torch.float16))
        text_mask = torch.ones((1, int(text_positions.numel())), dtype=torch.bool)
        return {
            "text_hidden0": text_hidden0.detach().cpu().to(torch.float16),
            "text_position_ids": prompt_positions.unsqueeze(0).detach().cpu(),
            "text_mask": text_mask,
            "attention_deltas": deltas,
        }

    @torch.inference_mode()
    def rollout_next_logits(
        self,
        trace: dict[str, Any],
        *,
        mode: str,
        basis: Tensor | None,
        rank: int | None,
        compress_layers: set[int] | None,
    ) -> Tensor:
        if mode not in {"no_visual", "full_effect", "rank", "zero"}:
            raise ValueError(f"unsupported rollout mode: {mode}")
        if mode == "rank" and (basis is None or rank is None):
            raise ValueError("rank rollout requires basis and rank")
        h = trace["text_hidden0"].to(device=self.device, dtype=self.dtype)
        text_position_ids = trace["text_position_ids"]
        text_mask = trace["text_mask"]
        attention_deltas = trace["attention_deltas"]
        if self.model_kind == "qwen":
            for layer_idx in range(self.num_layers):
                delta = None
                if mode != "no_visual":
                    delta = attention_deltas[layer_idx].to(device=h.device, dtype=self.dtype)
                    if mode == "zero" and (compress_layers is None or layer_idx in compress_layers):
                        delta = None
                    if mode == "rank":
                        assert basis is not None and rank is not None
                        if compress_layers is None or layer_idx in compress_layers:
                            delta = project_reconstruct_delta(
                                delta,
                                basis[layer_idx].to(device=delta.device, dtype=self.dtype),
                                rank,
                            )
                h = qwen_layer_text_with_attention_delta(
                    self.language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    text_mask,
                )
            logits = logits_from_text_hidden(self.model, self.language_model, h)
            valid_lengths = text_mask.to(device=logits.device, dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(logits.shape[0], device=logits.device)
            return logits[batch_idx, valid_lengths]

        for layer_idx in range(self.num_layers):
            delta = None
            if mode != "no_visual":
                delta = attention_deltas[layer_idx].to(device=h.device, dtype=self.dtype)
                if mode == "zero" and (compress_layers is None or layer_idx in compress_layers):
                    delta = None
                if mode == "rank":
                    assert basis is not None and rank is not None
                    if compress_layers is None or layer_idx in compress_layers:
                        delta = project_reconstruct_delta(
                            delta,
                            basis[layer_idx].to(device=delta.device, dtype=self.dtype),
                            rank,
                        )
            h = llava_layer_text_with_attention_delta(
                self.language_model,
                layer_idx,
                h,
                text_position_ids,
                delta,
            )
        norm_device = module_device(self.language_model.norm, h.device)
        logits = self.model.lm_head(self.language_model.norm(h.to(device=norm_device)))
        return logits[0, -1]

    @torch.inference_mode()
    def oracle_generate(
        self,
        base_inputs: dict[str, Tensor],
        *,
        mode: str,
        basis: Tensor | None,
        rank: int | None,
        compress_layers: set[int] | None,
        max_new_tokens: int,
        choices: list[Any] | None,
    ) -> str:
        if self.model_kind == "qwen":
            return self.oracle_generate_batch(
                base_inputs,
                mode=mode,
                basis=basis,
                rank=rank,
                compress_layers=compress_layers,
                max_new_tokens=max_new_tokens,
                choices_batch=[choices],
            )[0]

        generated: list[int] = []
        rollout_mode = "rank" if mode.startswith("rank_") else "zero" if mode.startswith("zero_") else mode
        rollout_rank = rank
        if mode.startswith("rank_") and rollout_rank is None:
            rollout_rank, _scope = parse_rank_rollout(mode)
        for _ in range(int(max_new_tokens)):
            trace = self.build_trace(base_inputs, generated)
            logits = self.rollout_next_logits(
                trace,
                mode=rollout_mode,
                basis=basis,
                rank=rollout_rank,
                compress_layers=compress_layers,
            )
            next_id = int(torch.argmax(logits.float()).item())
            generated.append(next_id)
            text = generated_text(self.tokenizer, generated)
            if should_stop_generation(self.tokenizer, next_id, text, self.spec.metric, choices):
                break
        return generated_text(self.tokenizer, generated)

    @torch.inference_mode()
    def oracle_generate_batch(
        self,
        base_inputs: dict[str, Tensor],
        *,
        mode: str,
        basis: Tensor | None,
        rank: int | None,
        compress_layers: set[int] | None,
        max_new_tokens: int,
        choices_batch: list[list[Any] | None],
    ) -> list[str]:
        if self.model_kind != "qwen":
            if int(base_inputs["input_ids"].shape[0]) != 1:
                raise ValueError("LLaVA SVD generation currently supports batch size 1")
            return [
                self.oracle_generate(
                    base_inputs,
                    mode=mode,
                    basis=basis,
                    rank=rank,
                    compress_layers=compress_layers,
                    max_new_tokens=max_new_tokens,
                    choices=choices_batch[0] if choices_batch else None,
                )
            ]

        batch = int(base_inputs["input_ids"].shape[0])
        if len(choices_batch) != batch:
            raise ValueError(f"choices batch size mismatch: got {len(choices_batch)} expected {batch}")
        generated: list[list[int]] = [[] for _ in range(batch)]
        done = [False for _ in range(batch)]
        rollout_mode = "rank" if mode.startswith("rank_") else "zero" if mode.startswith("zero_") else mode
        rollout_rank = rank
        if mode.startswith("rank_") and rollout_rank is None:
            rollout_rank, _scope = parse_rank_rollout(mode)

        for _ in range(int(max_new_tokens)):
            trace = self.build_trace(base_inputs, generated)
            logits = self.rollout_next_logits(
                trace,
                mode=rollout_mode,
                basis=basis,
                rank=rollout_rank,
                compress_layers=compress_layers,
            )
            next_ids = torch.argmax(logits.float(), dim=-1).detach().to(device="cpu").tolist()
            for batch_idx, next_id_raw in enumerate(next_ids):
                if done[batch_idx]:
                    continue
                next_id = int(next_id_raw)
                generated[batch_idx].append(next_id)
                text = generated_text(self.tokenizer, generated[batch_idx])
                if should_stop_generation(self.tokenizer, next_id, text, self.spec.metric, choices_batch[batch_idx]):
                    done[batch_idx] = True
            if all(done):
                break

        return [generated_text(self.tokenizer, token_ids) for token_ids in generated]


def json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if key in {"basis", "singular_energy"}:
            continue
        if isinstance(value, dict):
            safe[key] = {str(k): int(v) if isinstance(v, int) else v for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def rank_at_energy(cumsum: Tensor, total: float, threshold: float) -> int:
    if total <= 0.0:
        return 0
    hits = torch.nonzero((cumsum / total) >= float(threshold), as_tuple=False)
    return int(hits[0].item() + 1) if hits.numel() else int(cumsum.numel())


def energy_rows_from_meta(meta: dict[str, Any], ranks: list[int], num_layers: int) -> list[dict[str, Any]]:
    raw = meta.get("singular_energy") or {}
    counts = meta.get("counts") or {}
    rows: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        key: Any = layer_idx if layer_idx in raw else str(layer_idx)
        energy = torch.as_tensor(raw[key]).float()
        total = float(energy.sum().item())
        cumsum = energy.cumsum(dim=0)
        row: dict[str, Any] = {
            "layer": layer_idx,
            "tokens": int(counts.get(layer_idx, counts.get(str(layer_idx), 0))),
            "available_rank": int(energy.numel()),
            "rank_90": rank_at_energy(cumsum, total, 0.90),
            "rank_95": rank_at_energy(cumsum, total, 0.95),
            "rank_99": rank_at_energy(cumsum, total, 0.99),
        }
        for rank in ranks:
            idx = min(int(rank), int(cumsum.numel())) - 1
            row[f"energy_rank_{rank}"] = float(cumsum[idx].item() / total) if total > 0.0 and idx >= 0 else 0.0
        rows.append(row)
    return rows


def write_energy_csv(path: Path, rows: list[dict[str, Any]], ranks: list[int]) -> None:
    fieldnames = ["layer", "tokens", "available_rank", "rank_90", "rank_95", "rank_99"] + [
        f"energy_rank_{rank}" for rank in ranks
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    model_path = args.model_path or (DEFAULT_QWEN4B if args.model_kind == "qwen" else DEFAULT_LLAVA7B)
    spec = get_benchmark_spec(args.benchmark)
    data_path = Path(args.data or spec.default_data)
    image_root = Path(args.image_root) if args.image_root else data_path.parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = parse_ranks(args.ranks, int(args.max_rank))
    rank_layer_scopes = parse_rank_layer_scopes(args.rank_layer_scopes)
    zero_layer_scopes = parse_rank_layer_scopes(args.zero_layer_scopes)
    causal_zero_layer_scopes = parse_rank_layer_scopes(args.causal_zero_layer_scopes)
    if causal_zero_layer_scopes and args.model_kind != "qwen":
        raise ValueError("--causal-zero-layer-scopes is currently implemented only for --model-kind qwen")
    max_new_tokens = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else int(spec.max_new_tokens)
    batch_size = max(1, int(args.batch_size))
    if args.model_kind != "qwen" and batch_size != 1:
        raise ValueError("--batch-size > 1 is currently supported only for Qwen")
    basis_path = Path(args.basis) if args.basis else out_dir / f"{args.model_kind}_{spec.name}_visual_effect_basis_rank{args.max_rank}.pt"
    attn_implementation = args.attn_implementation
    if causal_zero_layer_scopes and args.model_kind == "qwen":
        if attn_implementation and attn_implementation != "eager":
            print(
                f"override --attn-implementation {attn_implementation!r} -> 'eager' "
                "for causal text->visual attention blocking",
                flush=True,
            )
        attn_implementation = "eager"

    eval_limit = None if int(args.max_samples) == 0 else int(args.max_samples)
    eval_source_rows = read_jsonl(data_path, eval_limit)
    eval_rows, sample_start, sample_end = select_contiguous_shard(eval_source_rows, int(args.num_shards), int(args.shard_id))
    basis_limit = int(args.basis_max_samples) if int(args.basis_max_samples) > 0 else int(args.max_samples)
    basis_limit = None if basis_limit == 0 else basis_limit
    basis_rows = read_jsonl(data_path, basis_limit)

    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    engine = GenerationSvdEngine(
        model_kind=args.model_kind,
        model_path=model_path,
        benchmark=spec.name,
        image_root=image_root,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        qwen_device_map=parse_qwen_device_map(args.qwen_device_map),
        qwen_max_memory=parse_qwen_max_memory(args.qwen_max_memory),
    )
    scope_layers = {
        scope: layer_scope_indices(scope, engine.num_layers)
        for scope in tuple(dict.fromkeys((*rank_layer_scopes, *zero_layer_scopes, *causal_zero_layer_scopes)))
    }

    basis_meta: dict[str, Any] = {}
    if args.mode in {"build-and-eval", "build-basis"}:
        if args.reuse_basis and basis_path.exists():
            loaded = torch.load(basis_path, map_location="cpu", weights_only=False)
            basis_meta = {key: value for key, value in loaded.items() if key != "basis"}
            print(f"reuse basis: {basis_path}", flush=True)
        else:
            print(
                f"build basis model={args.model_kind} benchmark={spec.name} rows={len(basis_rows)} "
                f"max_rank={args.max_rank} max_tokens_per_layer={args.max_tokens_per_layer}",
                flush=True,
            )
            payload = engine.build_basis(
                basis_rows,
                int(args.max_rank),
                int(args.max_tokens_per_layer),
                int(args.log_every),
                batch_size,
            )
            payload.update({"data": str(data_path), "image_root": str(image_root), "seed": int(args.seed)})
            save_basis_atomic(payload, basis_path)
            basis_meta = {key: value for key, value in payload.items() if key != "basis"}
            print(f"wrote basis: {basis_path}", flush=True)

    if args.mode == "build-basis":
        energy_rows = energy_rows_from_meta(basis_meta, ranks, engine.num_layers)
        write_energy_csv(out_dir / "layer_energy.csv", energy_rows, ranks)
        result = {
            "task": "visual_effect_svd_generate_build_basis",
            "model_kind": args.model_kind,
            "benchmark": spec.name,
            "basis": str(basis_path),
            "basis_meta": json_safe_meta(basis_meta),
            "layer_energy": energy_rows,
        }
        (out_dir / "layer_energy.json").write_text(json.dumps({"layers": energy_rows}, indent=2), encoding="utf-8")
        (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return

    basis: Tensor | None = None
    needs_basis = bool(ranks)
    if needs_basis and not basis_path.exists():
        raise FileNotFoundError(f"basis not found: {basis_path}")
    if not basis_meta and basis_path.exists():
        loaded_meta = torch.load(basis_path, map_location="cpu", weights_only=False)
        basis_meta = {key: value for key, value in loaded_meta.items() if key != "basis"}
    if needs_basis:
        basis = load_layer_basis(
            basis_path,
            max_rank=int(args.max_rank),
            num_layers=engine.num_layers,
            hidden_size=engine.hidden_size,
            dtype=dtype,
        )
        for rank in ranks:
            if rank > int(basis.shape[1]):
                raise ValueError(f"rank {rank} exceeds available basis rank {basis.shape[1]}")

    mode_names = ["teacher", "no_visual", "full_effect"] + [
        rank_mode_name(rank, scope) for rank in ranks for scope in rank_layer_scopes
    ] + [zero_mode_name(scope) for scope in zero_layer_scopes] + [
        causal_zero_mode_name(scope) for scope in causal_zero_layer_scopes
    ]
    scored: dict[str, list[dict[str, Any]]] = {name: [] for name in mode_names}
    timings: dict[str, float] = {name: 0.0 for name in mode_names}
    predictions: list[dict[str, Any]] = []

    print(
        f"eval generate model={args.model_kind} benchmark={spec.name} rows={len(eval_rows)} "
        f"shard={args.shard_id}/{args.num_shards} samples=[{sample_start},{sample_end}) "
        f"max_new_tokens={max_new_tokens} ranks={','.join(str(rank) for rank in ranks)} batch_size={batch_size}",
        flush=True,
    )
    total_start = time.perf_counter()
    if args.model_kind == "qwen":
        processed = 0
        for batch_rows in row_batches(eval_rows, batch_size):
            base_inputs = engine.prepare_inputs_batch(batch_rows)
            batch_records: list[dict[str, Any]] = []
            for batch_offset, row in enumerate(batch_rows):
                batch_records.append(
                    {
                        "index": row.get("index", sample_start + processed + batch_offset),
                        "answer": row.get("answer"),
                        "benchmark": spec.name,
                    }
                )

            start = time.perf_counter()
            teacher_texts = engine.teacher_generate_batch(base_inputs, max_new_tokens)
            timings["teacher"] += time.perf_counter() - start
            for row, record, teacher_text in zip(batch_rows, batch_records, teacher_texts):
                teacher_eval = score_text_prediction(spec, row, teacher_text)
                scored["teacher"].append(teacher_eval)
                record["teacher_text"] = teacher_text
                record["teacher_eval"] = teacher_eval

            choices_batch = [row.get("choices") for row in batch_rows]
            for mode_name in mode_names[1:]:
                rollout_rank = None
                compress_layers = None
                if mode_name.startswith("rank_"):
                    rollout_rank, scope = parse_rank_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                elif mode_name.startswith("zero_"):
                    scope = parse_zero_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                elif mode_name.startswith("causal_zero_"):
                    scope = parse_causal_zero_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                start = time.perf_counter()
                if mode_name.startswith("causal_zero_"):
                    oracle_texts = engine.causal_visual_block_generate_batch(
                        base_inputs,
                        blocked_layers=compress_layers,
                        max_new_tokens=max_new_tokens,
                        choices_batch=choices_batch,
                    )
                else:
                    oracle_texts = engine.oracle_generate_batch(
                        base_inputs,
                        mode=mode_name,
                        basis=basis,
                        rank=rollout_rank,
                        compress_layers=compress_layers,
                        max_new_tokens=max_new_tokens,
                        choices_batch=choices_batch,
                    )
                timings[mode_name] += time.perf_counter() - start
                for row, record, oracle_text in zip(batch_rows, batch_records, oracle_texts):
                    oracle_eval = score_text_prediction(spec, row, oracle_text)
                    scored[mode_name].append(oracle_eval)
                    record[f"{mode_name}_text"] = oracle_text
                    record[f"{mode_name}_eval"] = oracle_eval

            predictions.extend(batch_records)
            processed += len(batch_rows)
            if processed % int(args.log_every) == 0 or processed == len(eval_rows):
                parts = []
                progress_names = ["teacher", "full_effect"]
                if ranks:
                    progress_names.extend(
                        name for name in mode_names if name.startswith(f"rank_{ranks[-1]}") and name in scored
                    )
                progress_names.extend(name for name in mode_names if name.startswith("zero_"))
                progress_names.extend(name for name in mode_names if name.startswith("causal_zero_"))
                for name in progress_names:
                    summary = summarize_metric(spec.metric, scored[name], eval_rows[: len(scored[name])])
                    parts.append(f"{name}={summary['score']:.4f}")
                print(f"[{processed}/{len(eval_rows)}] " + " ".join(parts), flush=True)
    else:
        for local_idx, row in enumerate(eval_rows, start=1):
            base_inputs = engine.prepare_inputs(row)
            record = {
                "index": row.get("index", sample_start + local_idx - 1),
                "answer": row.get("answer"),
                "benchmark": spec.name,
            }

            start = time.perf_counter()
            teacher_text = engine.teacher_generate(base_inputs, max_new_tokens)
            timings["teacher"] += time.perf_counter() - start
            teacher_eval = score_text_prediction(spec, row, teacher_text)
            scored["teacher"].append(teacher_eval)
            record["teacher_text"] = teacher_text
            record["teacher_eval"] = teacher_eval

            for mode_name in mode_names[1:]:
                rollout_rank = None
                compress_layers = None
                if mode_name.startswith("rank_"):
                    rollout_rank, scope = parse_rank_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                elif mode_name.startswith("zero_"):
                    scope = parse_zero_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                elif mode_name.startswith("causal_zero_"):
                    scope = parse_causal_zero_rollout(mode_name)
                    compress_layers = scope_layers[scope]
                start = time.perf_counter()
                if mode_name.startswith("causal_zero_"):
                    oracle_text = engine.causal_visual_block_generate_batch(
                        base_inputs,
                        blocked_layers=compress_layers,
                        max_new_tokens=max_new_tokens,
                        choices_batch=[row.get("choices")],
                    )[0]
                else:
                    oracle_text = engine.oracle_generate(
                        base_inputs,
                        mode=mode_name,
                        basis=basis,
                        rank=rollout_rank,
                        compress_layers=compress_layers,
                        max_new_tokens=max_new_tokens,
                        choices=row.get("choices"),
                    )
                timings[mode_name] += time.perf_counter() - start
                oracle_eval = score_text_prediction(spec, row, oracle_text)
                scored[mode_name].append(oracle_eval)
                record[f"{mode_name}_text"] = oracle_text
                record[f"{mode_name}_eval"] = oracle_eval

            predictions.append(record)
            if local_idx % int(args.log_every) == 0 or local_idx == len(eval_rows):
                parts = []
                progress_names = ["teacher", "full_effect"]
                if ranks:
                    progress_names.extend(
                        name for name in mode_names if name.startswith(f"rank_{ranks[-1]}") and name in scored
                    )
                progress_names.extend(name for name in mode_names if name.startswith("zero_"))
                progress_names.extend(name for name in mode_names if name.startswith("causal_zero_"))
                for name in progress_names:
                    summary = summarize_metric(spec.metric, scored[name], eval_rows[: len(scored[name])])
                    parts.append(f"{name}={summary['score']:.4f}")
                print(f"[{local_idx}/{len(eval_rows)}] " + " ".join(parts), flush=True)

    elapsed = time.perf_counter() - total_start
    metrics = {name: summarize_metric(spec.metric, scored[name], eval_rows[: len(scored[name])]) for name in mode_names}
    teacher_score = float(metrics["teacher"].get("score", 0.0))
    full_score = float(metrics["full_effect"].get("score", 0.0))
    for name in mode_names:
        score = float(metrics[name].get("score", 0.0))
        metrics[name]["drop_vs_teacher"] = teacher_score - score
        metrics[name]["drop_vs_full_effect"] = full_score - score

    result = {
        "task": "visual_effect_svd_generation_benchmark",
        "oracle": "DeltaA_l = A_joint_l(text_positions) - A_text_l, generated autoregressively and scored with src.benchmarks",
        "model_kind": args.model_kind,
        "model_path": model_path,
        "benchmark": spec.name,
        "metric": spec.metric,
        "data": str(data_path),
        "image_root": str(image_root),
        "basis": str(basis_path),
        "basis_meta": json_safe_meta(basis_meta),
        "max_rank": int(args.max_rank),
        "ranks": ranks,
        "rank_layer_scopes": list(rank_layer_scopes),
        "zero_layer_scopes": list(zero_layer_scopes),
        "causal_zero_layer_scopes": list(causal_zero_layer_scopes),
        "rank_layer_indices": {
            scope: (None if indices is None else sorted(indices)) for scope, indices in scope_layers.items()
        },
        "max_new_tokens": max_new_tokens,
        "total_samples": len(predictions),
        "source_total_samples": len(eval_source_rows),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "batch_size": batch_size,
        "metrics": metrics,
        "timing": {
            "total_s": elapsed,
            "avg_s": elapsed / max(1, len(predictions)),
            "by_mode_s": timings,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
