#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VISUAL_SIDECAR_USE_TRITON_BASIS", "1")
os.environ.setdefault("VISUAL_SIDECAR_USE_TRITON_CROSS_ATTN", "1")

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_visual_memory_by_layer,
    run_qwen3vl_full_layer_with_text_delta_cache,
    run_qwen3vl_layer_text_with_attention_delta_cache,
    run_qwen3vl_full_layer_with_text_delta,
    run_qwen3vl_layer_text_with_attention_delta,
)
from delta_vision.models.sidecar import DeltaVisionModule

OPTIONS = ("A", "B", "C", "D")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
CHOICE_LINE_RE = re.compile(r"(?m)^\s*\(?([A-D])\)?[.)：:]\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generation-based Qwen3-VL-Thinking evaluation.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mode", choices=("qwen", "text_only", "sidecar_only", "hybrid"), default="qwen")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=128)
    parser.add_argument("--visual-memory-mode", choices=("v0", "vdeep", "vcum"), default="v0")
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-time-per-sample",
        type=float,
        default=0.0,
        help="Deprecated compatibility flag. Generation is not time-truncated.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--thinking-mode",
        choices=("natural", "force_final"),
        default="force_final",
        help="natural starts from <think>; force_final closes an empty thinking block and evaluates final-answer generation.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


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
    return [found[letter] for letter in OPTIONS if letter in found]


def first_option(text: str) -> str:
    stripped = text.strip().upper()
    if stripped[:1] in OPTIONS:
        return stripped[:1]
    final_match = re.search(r"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-D])[\)\].]?", stripped)
    if final_match:
        return final_match.group(1)
    correct_match = re.search(r"(?:OPTION\s*)?([A-D])\s+(?:IS\s+)?CORRECT", stripped)
    if correct_match:
        return correct_match.group(1)
    option_match = re.findall(r"OPTION\s+([A-D])", stripped)
    if option_match:
        return option_match[-1]
    matches = re.findall(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", stripped)
    return matches[-1] if matches else ""


def last_number(text: str) -> float | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def final_answer_text(generation: str) -> str:
    text = generation.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def score_prediction(row: dict[str, Any], prediction: str) -> tuple[bool, str]:
    gold = str(row.get("answer", "")).strip()
    pred = final_answer_text(prediction)
    choices = row_choices(row)

    gold_letter = first_option(gold)
    if gold_letter in OPTIONS:
        pred_letter = first_option(pred)
        return pred_letter == gold_letter, pred_letter or pred

    if choices:
        pred_letter = first_option(pred)
        if pred_letter:
            idx = ord(pred_letter) - ord("A")
            if 0 <= idx < len(choices):
                return normalize_text(str(choices[idx])) == normalize_text(gold), pred_letter
        pred_norm = normalize_text(pred)
        return pred_norm == normalize_text(gold) or normalize_text(gold) in pred_norm, pred

    gold_head = gold.upper()[:1]
    pred_clean = pred.strip()
    if gold_head in {"A", "B", "C", "D", "Y", "N"} and len(gold) <= 3:
        pred_head = pred_clean.upper()[:1]
        if pred_clean.lower().startswith("yes"):
            pred_head = "Y"
        elif pred_clean.lower().startswith("no"):
            pred_head = "N"
        return pred_head == gold_head, pred_head

    answer_type = str(row.get("answer_type", "")).lower()
    if answer_type in {"integer", "float"} or NUMBER_RE.fullmatch(gold.replace(",", "")):
        gold_num = last_number(gold)
        pred_num = last_number(pred)
        if gold_num is None or pred_num is None:
            return False, pred
        if answer_type == "integer":
            return int(round(pred_num)) == int(round(gold_num)), str(pred_num)
        precision = row.get("precision")
        places = int(precision) if isinstance(precision, (int, float)) and math.isfinite(float(precision)) else 2
        return abs(pred_num - gold_num) <= max(10 ** (-places), 1e-3), str(pred_num)

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    return pred_norm == gold_norm or gold_norm in pred_norm, pred


def generation_stop_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for token_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids


def build_messages(row: dict[str, Any], benchmark: str, include_image: bool = True) -> list[dict[str, Any]]:
    question = str(row["question"]).strip()
    if benchmark == "mmstar":
        suffix = (
            "Think briefly. After </think>, give the final answer as only one letter: A, B, C, or D."
        )
    elif row_choices(row):
        suffix = "Think briefly. After </think>, give the final answer as only one option letter."
    else:
        suffix = "Think briefly. After </think>, give the final answer only."
    content: list[dict[str, str]] = []
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"{question}\n{suffix}"})
    return [{"role": "user", "content": content}]


def build_prompt(
    processor: Any,
    row: dict[str, Any],
    benchmark: str,
    thinking_mode: str,
    include_image: bool = True,
) -> tuple[str, int]:
    prompt = processor.apply_chat_template(
        build_messages(row, benchmark, include_image=include_image),
        tokenize=False,
        add_generation_prompt=True,
    )
    if thinking_mode == "force_final":
        prompt = f"{prompt}\n</think>\n\n"
    return prompt


def load_sidecar(
    args: argparse.Namespace,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DeltaVisionModule:
    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required for mode={args.mode}")
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        state_tokens=0,
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=True,
        reader_mlp_ratio=args.reader_mlp_ratio,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=True,
    ).to(device=device, dtype=dtype)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = sidecar.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"checkpoint load missing={missing} unexpected={unexpected}", flush=True)
    sidecar.eval()
    for param in sidecar.parameters():
        param.requires_grad_(False)
    sidecar.runtime_fold_output_basis = True
    sidecar.prepare_inference_cache(device, dtype)
    return sidecar


def _decode_position_ids(base_position_ids: torch.Tensor, step: int) -> torch.Tensor:
    return base_position_ids[:, :, -1:] + int(step + 1)


def _decode_attention_mask(cache: DynamicCache, device: torch.device) -> torch.Tensor:
    past_len = int(cache.get_seq_length())
    return torch.ones((1, past_len + 1), device=device, dtype=torch.long)


def _sidecar_delta_decode(
    sidecar: DeltaVisionModule,
    hidden_states: torch.Tensor,
    layer_idx: int,
    visual_kv: Any,
    sidecar_scale: float,
) -> torch.Tensor:
    sidecar_dtype = next(sidecar.parameters()).dtype
    sidecar_h = hidden_states.to(dtype=sidecar_dtype) if hidden_states.dtype != sidecar_dtype else hidden_states
    return sidecar.decode_no_state_layer(sidecar_h, layer_idx, visual_kv) * float(sidecar_scale)


@torch.inference_mode()
def sidecar_or_hybrid_prefill(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    prompt: str,
    image_path: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    sidecar_scale: float,
) -> tuple[torch.Tensor, DynamicCache, dict[str, Any]]:
    with Image.open(image_path) as image:
        inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    h, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    visual_memories = qwen3vl_visual_memory_by_layer(
        h.to(dtype=dtype),
        image_pos,
        image_mask,
        deepstack_visual_embeds,
        visual_memory_mode,
        len(language_model.layers),
    )
    if visual_memory_mode == "vcum":
        visual_kvs = [
            sidecar.prepare_visual_kv(memory.to(dtype=dtype), ~image_mask)
            for memory in visual_memories
        ]
    else:
        visual_kvs = [sidecar.prepare_visual_kv(visual_memories[0].to(dtype=dtype), ~image_mask)]

    cache = DynamicCache(config=language_model.config)
    if mode == "sidecar_only":
        h_text = gather_batched_positions(h, text_pos, text_mask).to(dtype=dtype)
        text_attention_mask = text_mask.to(dtype=torch.long)
        position_embeddings = language_model.rotary_emb(h_text, text_position_ids)
        for layer_idx in range(len(language_model.layers)):
            visual_kv = visual_kvs[layer_idx] if visual_memory_mode == "vcum" else visual_kvs[0]
            delta = _sidecar_delta_decode(sidecar, h_text, layer_idx, visual_kv, sidecar_scale)
            delta = delta.masked_fill(~text_mask.unsqueeze(-1), 0.0)
            h_text = run_qwen3vl_layer_text_with_attention_delta_cache(
                language_model,
                layer_idx,
                h_text,
                text_position_ids,
                delta,
                past_key_values=cache,
                attention_mask_2d=text_attention_mask,
                position_embeddings=position_embeddings,
            )
        valid_text = int(text_mask[0].sum().item())
        logits = model.lm_head(language_model.norm(h_text))[0, valid_text - 1]
        context = {
            "mode": mode,
            "cache": cache,
            "visual_kvs": visual_kvs,
            "last_position_ids": text_position_ids[:, :, valid_text - 1 : valid_text],
        }
        return logits, cache, context

    if mode != "hybrid":
        raise ValueError(f"unsupported custom generation mode: {mode}")
    attention_mask_2d = inputs.get("attention_mask")
    position_embeddings = language_model.rotary_emb(h, position_ids)
    for layer_idx in range(len(language_model.layers)):
        text_hidden = gather_batched_positions(h, text_pos, text_mask)
        visual_kv = visual_kvs[layer_idx] if visual_memory_mode == "vcum" else visual_kvs[0]
        delta = _sidecar_delta_decode(sidecar, text_hidden, layer_idx, visual_kv, sidecar_scale)
        h = run_qwen3vl_full_layer_with_text_delta_cache(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_mask_2d,
            text_pos,
            delta,
            past_key_values=cache,
            position_embeddings=position_embeddings,
        )
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            h = language_model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
    valid_text = int(text_mask[0].sum().item())
    last_text_pos = int(text_pos[0, valid_text - 1].item())
    logits = model.lm_head(language_model.norm(h))[0, last_text_pos]
    context = {
        "mode": mode,
        "cache": cache,
        "visual_kvs": visual_kvs,
        "last_position_ids": position_ids[:, :, last_text_pos : last_text_pos + 1],
    }
    return logits, cache, context


@torch.inference_mode()
def sidecar_or_hybrid_decode_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    token_id: int,
    step: int,
    cache: DynamicCache,
    context: dict[str, Any],
    device: torch.device,
    sidecar_scale: float,
    visual_memory_mode: str,
) -> torch.Tensor:
    token = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
    h = language_model.embed_tokens(token)
    position_ids = _decode_position_ids(context["last_position_ids"], step)
    attention_mask_2d = _decode_attention_mask(cache, device)
    position_embeddings = language_model.rotary_emb(h, position_ids)
    visual_kvs = context["visual_kvs"]
    for layer_idx in range(len(language_model.layers)):
        visual_kv = visual_kvs[layer_idx] if visual_memory_mode == "vcum" else visual_kvs[0]
        delta = _sidecar_delta_decode(sidecar, h, layer_idx, visual_kv, sidecar_scale)
        if context["mode"] == "sidecar_only":
            h = run_qwen3vl_layer_text_with_attention_delta_cache(
                language_model,
                layer_idx,
                h,
                position_ids,
                delta,
                past_key_values=cache,
                attention_mask_2d=attention_mask_2d,
                position_embeddings=position_embeddings,
                mask_past_key_values=cache,
            )
        else:
            h = run_qwen3vl_full_layer_with_text_delta_cache(
                language_model,
                layer_idx,
                h,
                position_ids,
                attention_mask_2d,
                None,
                delta,
                past_key_values=cache,
                position_embeddings=position_embeddings,
                mask_past_key_values=cache,
            )
    return model.lm_head(language_model.norm(h))[0, -1]


@torch.inference_mode()
def sidecar_or_hybrid_next_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    prompt: str,
    image_path: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    sidecar_scale: float,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    h, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    visual_memories = qwen3vl_visual_memory_by_layer(
        h.to(dtype=dtype),
        image_pos,
        image_mask,
        deepstack_visual_embeds,
        visual_memory_mode,
        len(language_model.layers),
    )
    shared_visual_kv = None
    if visual_memory_mode != "vcum":
        shared_visual_kv = sidecar.prepare_visual_kv(visual_memories[0].to(dtype=dtype), ~image_mask)

    if mode == "sidecar_only":
        h_text = gather_batched_positions(h, text_pos, text_mask).to(dtype=dtype)
        text_padding_mask = ~text_mask
        for layer_idx in range(len(language_model.layers)):
            layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
            visual_kv = shared_visual_kv
            if visual_kv is None:
                visual_kv = sidecar.prepare_visual_kv(visual_memories[layer_idx].to(dtype=dtype), ~image_mask)
            delta = sidecar(h_text, None, layer_tensor, visual_kv=visual_kv) * float(sidecar_scale)
            h_text = run_qwen3vl_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h_text,
                text_position_ids,
                delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                padding_mask=text_padding_mask,
            )
        logits = model.lm_head(language_model.norm(h_text))
        return logits[0, int(text_mask[0].sum().item()) - 1]

    if mode != "hybrid":
        raise ValueError(f"unsupported custom generation mode: {mode}")
    attention_mask_2d = inputs.get("attention_mask")
    for layer_idx in range(len(language_model.layers)):
        layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
        text_hidden = gather_batched_positions(h, text_pos, text_mask)
        visual_kv = shared_visual_kv
        if visual_kv is None:
            visual_kv = sidecar.prepare_visual_kv(visual_memories[layer_idx].to(dtype=dtype), ~image_mask)
        delta = sidecar(text_hidden, None, layer_tensor, visual_kv=visual_kv) * float(sidecar_scale)
        h = run_qwen3vl_full_layer_with_text_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_mask_2d,
            text_pos,
            delta,
        )
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            h = language_model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
    logits = model.lm_head(language_model.norm(h))
    return logits[0, int(text_pos[0, int(text_mask[0].sum().item()) - 1].item())]


@torch.inference_mode()
def generate_one(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module | None,
    sidecar: DeltaVisionModule | None,
    row: dict[str, Any],
    benchmark: str,
    mode: str,
    max_new_tokens: int,
    thinking_mode: str,
    max_time_per_sample: float,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    sidecar_scale: float,
) -> str:
    include_image = mode != "text_only"
    prompt = build_prompt(processor, row, benchmark, thinking_mode, include_image=include_image)
    if mode in {"qwen", "text_only"}:
        if include_image:
            with Image.open(row["image"]) as image:
                inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": generation_stop_ids(processor.tokenizer),
        }
        output = model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=False).strip(), int(new_tokens.numel())

    if sidecar is None or language_model is None:
        raise ValueError(f"sidecar and language_model are required for mode={mode}")
    generated_ids: list[int] = []
    stop_ids = set(generation_stop_ids(processor.tokenizer))
    logits, cache, custom_context = sidecar_or_hybrid_prefill(
        processor,
        model,
        language_model,
        sidecar,
        prompt,
        str(row["image"]),
        mode,
        device,
        dtype,
        visual_memory_mode,
        sidecar_scale,
    )
    for _ in range(max_new_tokens):
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        if next_id in stop_ids:
            break
        logits = sidecar_or_hybrid_decode_logits(
            model,
            language_model,
            sidecar,
            next_id,
            len(generated_ids) - 1,
            cache,
            custom_context,
            device,
            sidecar_scale,
            visual_memory_mode,
        )
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=False).strip(), len(generated_ids)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.max_samples)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    sidecar = None
    if args.mode in {"sidecar_only", "hybrid"}:
        sidecar = load_sidecar(args, int(language_model.config.hidden_size), len(language_model.layers), device, dtype)

    correct = 0
    predictions: list[dict[str, Any]] = []
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text("", encoding="utf-8")
    for idx, row in enumerate(rows):
        start = time.time()
        print(f"start {idx + 1}/{len(rows)} index={row.get('index', idx)}", flush=True)
        generation, generated_tokens = generate_one(
            processor,
            model,
            language_model,
            sidecar,
            row,
            args.benchmark,
            args.mode,
            args.max_new_tokens,
            args.thinking_mode,
            args.max_time_per_sample,
            device,
            dtype,
            args.visual_memory_mode,
            args.sidecar_scale,
        )
        ok, parsed = score_prediction(row, generation)
        elapsed = time.time() - start
        correct += int(ok)
        predictions.append(
            pred_record := {
                "index": row.get("index", idx),
                "gold": row.get("answer", ""),
                "parsed": parsed,
                "correct": ok,
                "final_answer_text": final_answer_text(generation),
                "generation": generation,
                "generated_tokens": generated_tokens,
                "elapsed_sec": elapsed,
                "tokens_per_sec": generated_tokens / max(elapsed, 1e-6),
            }
        )
        with pred_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
        if (idx + 1) % max(int(args.print_every), 1) == 0 or idx + 1 == len(rows):
            print(
                f"evaluated {idx + 1}/{len(rows)} acc={correct / (idx + 1):.4f} "
                f"last_ok={int(ok)} parsed={parsed!r} tokens={generated_tokens} "
                f"tok_s={generated_tokens / max(elapsed, 1e-6):.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    total_elapsed = sum(float(item["elapsed_sec"]) for item in predictions)
    total_tokens = sum(int(item["generated_tokens"]) for item in predictions)
    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "model_path": args.model_path,
        "eval_mode": "thinking_generation",
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "visual_memory_mode": args.visual_memory_mode,
        "sidecar_scale": args.sidecar_scale,
        "thinking_mode": args.thinking_mode,
        "max_new_tokens": args.max_new_tokens,
        "max_time_per_sample": args.max_time_per_sample,
        "scored": len(rows),
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "generated_tokens": total_tokens,
        "generation_elapsed_sec": total_elapsed,
        "generation_tokens_per_sec": total_tokens / max(total_elapsed, 1e-6),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
