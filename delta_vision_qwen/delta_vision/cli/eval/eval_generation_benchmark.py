#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import string
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_embed_tokens,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    run_llama_layer_text_with_attention_delta,
)
from delta_vision.evaluation.metrics import OPTIONS, option_scores, option_token_id_lists
from delta_vision.models.modeling import build_rollout_model, image_token_id, load_frozen_llava, load_rollout_checkpoint

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
CHOICE_LINE_RE = re.compile(r"(?m)^\s*\(?([A-D])\)?[.)：:]\s*(.+?)\s*$")


def parse_active_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(num_layers))
    if spec.strip() == "":
        return set()
    layers = {int(x) for x in spec.split(",") if x.strip() != ""}
    if any(x < 0 or x >= num_layers for x in layers):
        raise ValueError(f"--active-layers must contain zero-based layer ids in [0, {num_layers - 1}]")
    return layers


def read_jsonl(path: str | Path, max_samples: int | None = None, start_sample: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_sample:
                continue
            if line.strip():
                rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


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


def gold_choice(row: dict[str, Any]) -> str:
    gold = str(row.get("answer", "")).strip()
    if gold.upper()[:1] in OPTIONS and len(gold) <= 3:
        return gold.upper()[:1]
    gold_norm = normalize_text(gold)
    for idx, choice in enumerate(row_choices(row)[: len(OPTIONS)]):
        choice_norm = normalize_text(choice)
        if choice_norm == gold_norm or gold_norm in choice_norm or choice_norm in gold_norm:
            return OPTIONS[idx]
    return ""


def is_choice_row(row: dict[str, Any]) -> bool:
    return bool(row_choices(row) and gold_choice(row))


def first_option(text: str) -> str:
    stripped = text.strip().upper()
    if stripped[:1] in OPTIONS:
        return stripped[:1]
    match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", stripped)
    return match.group(1) if match else ""


def last_number(text: str) -> float | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def score_prediction(row: dict[str, Any], prediction: str) -> bool:
    gold = str(row.get("answer", "")).strip()
    pred = prediction.strip()
    choices = row_choices(row)
    answer_type = str(row.get("answer_type", "")).lower()

    if choices:
        pred_letter = first_option(pred)
        gold_letter = first_option(gold)
        if pred_letter and gold_letter:
            return pred_letter == gold_letter
        if pred_letter:
            idx = ord(pred_letter) - ord("A")
            if 0 <= idx < len(choices):
                return normalize_text(str(choices[idx])) == normalize_text(gold)
        pred_norm = normalize_text(pred)
        return pred_norm == normalize_text(gold) or normalize_text(gold) in pred_norm

    gold_head = gold.upper()[:1]
    if gold_head in {"A", "B", "C", "D", "Y", "N"} and len(gold) <= 3:
        pred_head = pred.upper()[:1]
        if gold_head in {"Y", "N"}:
            if pred.lower().startswith("yes"):
                pred_head = "Y"
            elif pred.lower().startswith("no"):
                pred_head = "N"
        return pred_head == gold_head

    if answer_type in {"integer", "float"} or NUMBER_RE.fullmatch(gold.replace(",", "")):
        gold_num = last_number(gold)
        pred_num = last_number(pred)
        if gold_num is None or pred_num is None:
            return False
        if answer_type == "integer":
            return int(round(pred_num)) == int(round(gold_num))
        precision = row.get("precision")
        places = int(precision) if isinstance(precision, (int, float)) and math.isfinite(float(precision)) else 2
        return abs(pred_num - gold_num) <= max(10 ** (-places), 1e-3)

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    return pred_norm == gold_norm or gold_norm in pred_norm


def build_prompt(row: dict[str, Any], benchmark: str) -> str:
    question = str(row["question"]).strip()
    if benchmark == "mathvista":
        if row_choices(row):
            suffix = "Answer directly with only the letter of the correct option."
        else:
            suffix = "Answer directly with the final answer only."
    else:
        suffix = "Answer directly with the final answer only."
    return llava15_prompt(f"{question}\n{suffix}")


@torch.inference_mode()
def llava_generate(processor: Any, model: torch.nn.Module, row: dict[str, Any], benchmark: str, max_new_tokens: int, device: torch.device) -> str:
    image = Image.open(row["image"]).convert("RGB")
    prompt = build_prompt(row, benchmark)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def llava_prompt_logits(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    image_token: int,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(row["image"]).convert("RGB")
    prompt = build_prompt(row, benchmark)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    out = model(**inputs, return_dict=True, use_cache=False)
    text_pos, _, _ = get_text_and_image_positions(inputs["input_ids"], out.logits.shape[1], image_token)
    return out.logits[0, int(text_pos[-1].item())]


def sidecar_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    initial_text_hidden: torch.Tensor,
    generated_ids: list[int],
    prompt_position_ids: torch.Tensor,
    vision_tokens: torch.Tensor,
    active_layers: set[int],
) -> torch.Tensor:
    sidecar = rollout_model.sidecar
    if generated_ids:
        token_ids = torch.tensor([generated_ids], device=initial_text_hidden.device, dtype=torch.long)
        gen_hidden = get_lm_embed_tokens(language_model)(token_ids).to(dtype=initial_text_hidden.dtype)
        h = torch.cat([initial_text_hidden, gen_hidden], dim=1)
        start = int(prompt_position_ids[0, -1].item()) + 1
        gen_pos = torch.arange(start, start + len(generated_ids), device=prompt_position_ids.device).unsqueeze(0)
        position_ids = torch.cat([prompt_position_ids, gen_pos], dim=1)
    else:
        h = initial_text_hidden
        position_ids = prompt_position_ids

    visual_kv = sidecar.prepare_visual_kv(vision_tokens, None)
    state = sidecar.initial_state(vision_tokens, None) if sidecar.state_tokens > 0 else None
    num_layers = len(get_lm_layers(language_model))
    for layer_idx in range(num_layers):
        if layer_idx not in active_layers:
            delta = None
        elif sidecar.state_tokens > 0:
            delta, state = sidecar(
                h,
                None,
                layer_idx,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
            )
        else:
            delta = sidecar(h, None, layer_idx, visual_kv=visual_kv)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=delta,
        )
    return model.lm_head(get_lm_norm(language_model)(h))


def text_only_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    h = get_lm_embed_tokens(language_model)(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    for layer_idx in range(len(get_lm_layers(language_model))):
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=None,
        )
    return model.lm_head(get_lm_norm(language_model)(h))


@torch.inference_mode()
def text_only_prompt_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
) -> torch.Tensor:
    prompt = build_prompt(row, benchmark).replace("<image>\n", "")
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = text_only_logits(model, language_model, input_ids)
    return logits[0, -1]


@torch.inference_mode()
def text_only_generate(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    prompt = build_prompt(row, benchmark).replace("<image>\n", "")
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated: list[int] = []
    eos = processor.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if generated:
            gen_ids = torch.tensor([generated], device=device, dtype=torch.long)
            step_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            step_ids = input_ids
        logits = text_only_logits(model, language_model, step_ids)
        next_id = int(logits[0, -1].float().argmax().item())
        if next_id == eos:
            break
        generated.append(next_id)
        token = processor.tokenizer.decode([next_id], skip_special_tokens=True)
        if "\n" in token and generated:
            break
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def sidecar_generate(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    max_new_tokens: int,
    image_token: int,
    device: torch.device,
    dtype: torch.dtype,
    active_layers: set[int],
) -> str:
    image = Image.open(row["image"]).convert("RGB")
    prompt = build_prompt(row, benchmark)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    text_pos, image_pos, prompt_positions = get_text_and_image_positions(inputs["input_ids"], hidden0.shape[1], image_token)
    text_pos = text_pos.to(device)
    image_pos = image_pos.to(device)
    initial_text_hidden = hidden0.index_select(1, text_pos).to(dtype=dtype)
    vision_tokens = hidden0.index_select(1, image_pos).to(dtype=dtype)
    prompt_position_ids = prompt_positions.to(device).unsqueeze(0)

    generated: list[int] = []
    eos = processor.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        logits = sidecar_logits(
            model,
            language_model,
            rollout_model,
            initial_text_hidden,
            generated,
            prompt_position_ids,
            vision_tokens,
            active_layers,
        )
        next_id = int(logits[0, -1].float().argmax().item())
        if next_id == eos:
            break
        generated.append(next_id)
        token = processor.tokenizer.decode([next_id], skip_special_tokens=True)
        if "\n" in token and generated:
            break
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def sidecar_prompt_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    image_token: int,
    device: torch.device,
    dtype: torch.dtype,
    active_layers: set[int],
) -> torch.Tensor:
    image = Image.open(row["image"]).convert("RGB")
    prompt = build_prompt(row, benchmark)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    text_pos, image_pos, prompt_positions = get_text_and_image_positions(inputs["input_ids"], hidden0.shape[1], image_token)
    initial_text_hidden = hidden0.index_select(1, text_pos.to(device)).to(dtype=dtype)
    vision_tokens = hidden0.index_select(1, image_pos.to(device)).to(dtype=dtype)
    logits = sidecar_logits(
        model,
        language_model,
        rollout_model,
        initial_text_hidden,
        [],
        prompt_positions.to(device).unsqueeze(0),
        vision_tokens,
        active_layers,
    )
    return logits[0, -1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate LLaVA or Sidecar generation on local JSONL benchmarks.")
    parser.add_argument("--benchmark", choices=("mathvista", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-kind", choices=("llava", "sidecar", "text-only"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--active-layers", default="all")
    parser.add_argument("--ignore-mismatched-checkpoint-shapes", action="store_true")
    parser.add_argument("--slice-mismatched-checkpoint-prefix", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.max_samples, args.start_sample)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    image_token = image_token_id(model, processor)

    rollout_model = None
    if args.model_kind == "sidecar":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --model-kind sidecar")
        rollout_model = build_rollout_model(args, dtype, device)
        load_rollout_checkpoint(
            rollout_model,
            args.checkpoint,
            ignore_mismatched_checkpoint_shapes=args.ignore_mismatched_checkpoint_shapes,
            slice_mismatched_checkpoint_prefix=args.slice_mismatched_checkpoint_prefix,
        )
        rollout_model.eval()
    active_layers = parse_active_layers(args.active_layers, args.num_layers)

    correct = 0
    predictions = []
    by_type: dict[str, list[int]] = {}
    option_ids = option_token_id_lists(processor.tokenizer)
    choice_scored = 0
    generated_scored = 0
    for idx, row in enumerate(rows):
        if is_choice_row(row):
            choice_scored += 1
            gold = gold_choice(row)
            if args.model_kind == "llava":
                logits = llava_prompt_logits(processor, model, row, args.benchmark, image_token, device)
            elif args.model_kind == "sidecar":
                assert rollout_model is not None
                logits = sidecar_prompt_logits(
                    processor,
                    model,
                    language_model,
                    rollout_model,
                    row,
                    args.benchmark,
                    image_token,
                    device,
                    dtype,
                    active_layers,
                )
            else:
                logits = text_only_prompt_logits(processor, model, language_model, row, args.benchmark, device)
            scores = option_scores(logits.float(), option_ids)
            pred = OPTIONS[int(scores.argmax().item())]
            ok = pred == gold
        elif args.model_kind == "llava":
            generated_scored += 1
            pred = llava_generate(processor, model, row, args.benchmark, args.max_new_tokens, device)
            ok = score_prediction(row, pred)
        elif args.model_kind == "sidecar":
            generated_scored += 1
            assert rollout_model is not None
            pred = sidecar_generate(
                processor,
                model,
                language_model,
                rollout_model,
                row,
                args.benchmark,
                args.max_new_tokens,
                image_token,
                device,
                dtype,
                active_layers,
            )
            ok = score_prediction(row, pred)
        else:
            generated_scored += 1
            pred = text_only_generate(
                processor,
                model,
                language_model,
                row,
                args.benchmark,
                args.max_new_tokens,
                device,
            )
            ok = score_prediction(row, pred)
        correct += int(ok)
        key = str(row.get("question_type", "all"))
        by_type.setdefault(key, [0, 0])
        by_type[key][0] += int(ok)
        by_type[key][1] += 1
        predictions.append(
            {
                "index": row.get("index", idx),
                "answer": row.get("answer", ""),
                "prediction": pred,
                "correct": ok,
                "question_type": row.get("question_type", ""),
                "answer_type": row.get("answer_type", ""),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"evaluated {idx + 1}/{len(rows)} correct={correct}", flush=True)

    metrics = {
        "benchmark": args.benchmark,
        "model_kind": args.model_kind,
        "num_samples": len(rows),
        "choice_logit_scored": choice_scored,
        "generation_scored": generated_scored,
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "by_question_type": {
            key: {"correct": val[0], "total": val[1], "accuracy": val[0] / max(val[1], 1)}
            for key, val in sorted(by_type.items())
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
