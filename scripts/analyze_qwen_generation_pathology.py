#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_benchmarks import configure_torch_runtime
from src.model import (
    build_qwen_initial_context,
    dtype_from_name,
    load_frozen_qwen3vl,
    load_qwen_visual_delta_checkpoint,
    qwen_visual_delta_logits,
)


DEFAULT_MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_ROOT = "/lustre-data/leijingdi/code/delta-vision"
DEFAULT_PROMPTS = ROOT_DIR / "artifacts/diagnostics/open_ended_prompts_100.jsonl"
DEFAULT_PROMPT_SOURCE = ROOT_DIR / "data/pixmo_clean_llava_instruct_ocr_256k_v1/train.jsonl"
OCR_SUCCESS_DIR = (
    ROOT_DIR
    / "artifacts/experiments/qwen_topk1024_freezeqkv"
    / "qwen_opd_rollout2k_injection_2000step_20260816_043953/checkpoints"
)
PIXMO_OCRVQA_12K_DIR = (
    ROOT_DIR
    / "artifacts/experiments/qwen_topk1024_freezeqkv"
    / "qwen_mixed_pixmo_ocrvqa_injection_12k_cosine_warmup02_20260813_162014/checkpoints"
)

OPEN_ENDED_PROMPTS = [
    "Describe the image in detail, including any visible text.",
    "What is happening in this image? Explain the important visual details.",
    "Look carefully at the image. What objects, text, and layout can you see?",
    "Please explain what the image shows and mention any readable words.",
    "用中文详细描述这张图片，包括能读到的文字。",
]


@dataclass(frozen=True)
class CheckpointSpec:
    family: str
    label: str
    step: int
    path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose Qwen adapter entropy collapse and repetition across checkpoints.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-jsonl", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--prompt-source-jsonl", default=str(DEFAULT_PROMPT_SOURCE))
    parser.add_argument("--max-prompts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=None,
        help="Optional diagnostic-only filter; prompts with larger source images are skipped before batching.",
    )
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "artifacts/diagnostics/qwen_generation_pathology"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-default-ckpts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="family:label:step:/path/to/ckpt.pt")
    parser.add_argument("--families", default="ocr_success,pixmo_ocrvqa_12k", help="Comma-separated default families.")
    parser.add_argument("--kl-stride", type=int, default=1, help="Compute KL(student||base) every N decode steps; 1 means every step.")
    parser.add_argument("--limit-checkpoints", type=int, default=None, help="Debug option: keep only first N non-base checkpoints.")
    parser.add_argument("--force-prompts", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def strip_quotes(text: str) -> str:
    text = str(text).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def resolve_path(path: str | Path, *, base: Path = ROOT_DIR) -> Path:
    text = strip_quotes(str(path))
    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw
    return (base / raw).resolve()


def resolve_image_path(row: dict[str, Any], default_root: str | Path) -> Path:
    image = Path(strip_quotes(str(row["image"]))).expanduser()
    if image.is_absolute():
        return image
    roots = []
    image_root = str(row.get("image_root", "")).strip()
    if image_root:
        roots.append(Path(image_root).expanduser())
    roots.extend([ROOT_DIR, Path(default_root).expanduser()])
    for root in roots:
        candidate = root / image
        if candidate.exists():
            return candidate
    return roots[0] / image


def make_prompt_jsonl(path: Path, source_jsonl: Path, *, count: int, data_root: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    with source_jsonl.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if len(rows) >= count:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if "image" not in row:
                continue
            image_path = resolve_image_path(row, data_root)
            image_key = str(image_path)
            if image_key in seen_images or not image_path.exists():
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                continue
            seen_images.add(image_key)
            prompt = OPEN_ENDED_PROMPTS[len(rows) % len(OPEN_ENDED_PROMPTS)]
            rows.append(
                {
                    "index": len(rows),
                    "image": str(image_path),
                    "prompt": prompt,
                    "source": row.get("source", ""),
                    "source_row_index": row_index,
                }
            )
    if len(rows) < count:
        raise RuntimeError(f"only built {len(rows)} prompts from {source_jsonl}, requested {count}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt_path = resolve_path(args.prompt_jsonl)
    source_path = resolve_path(args.prompt_source_jsonl)
    if args.force_prompts or not prompt_path.exists():
        make_prompt_jsonl(prompt_path, source_path, count=max(args.max_prompts, 100), data_root=args.data_root)
    rows: list[dict[str, Any]] = []
    skipped_large = 0
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if args.max_image_pixels is not None:
                    image_path = resolve_image_path(row, args.data_root)
                    try:
                        with Image.open(image_path) as image:
                            width, height = image.size
                    except Exception:
                        continue
                    if width * height > int(args.max_image_pixels):
                        skipped_large += 1
                        continue
                rows.append(row)
            if len(rows) >= int(args.max_prompts):
                break
    if not rows:
        raise RuntimeError(f"no prompts loaded from {prompt_path}")
    if skipped_large:
        print(f"skipped {skipped_large} prompts above max_image_pixels={args.max_image_pixels}", flush=True)
    return rows


def default_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    families = {item.strip() for item in str(args.families).split(",") if item.strip()}
    if "ocr_success" in families:
        for path in sorted(OCR_SUCCESS_DIR.glob("qwen_visual_delta_step*.pt"), key=checkpoint_step):
            specs.append(CheckpointSpec("ocr_success", path.stem.replace("qwen_visual_delta_", ""), checkpoint_step(path), path))
    if "pixmo_ocrvqa_12k" in families:
        for path in sorted(PIXMO_OCRVQA_12K_DIR.glob("qwen_visual_delta_step*.pt"), key=checkpoint_step):
            specs.append(CheckpointSpec("pixmo_ocrvqa_12k", path.stem.replace("qwen_visual_delta_", ""), checkpoint_step(path), path))
    return specs


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    if "step" not in stem:
        return -1
    try:
        return int(stem.rsplit("step", 1)[1])
    except ValueError:
        return -1


def parse_checkpoint_arg(spec: str) -> CheckpointSpec:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError("--checkpoint must be family:label:step:/path/to/ckpt.pt")
    family, label, step, path = parts
    return CheckpointSpec(family, label, int(step), resolve_path(path))


def collect_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    if args.include_base:
        specs.append(CheckpointSpec("base", "base", 0, None))
    non_base: list[CheckpointSpec] = []
    if args.include_default_ckpts:
        non_base.extend(default_checkpoints(args))
    non_base.extend(parse_checkpoint_arg(item) for item in args.checkpoint)
    if args.limit_checkpoints is not None:
        non_base = non_base[: max(0, int(args.limit_checkpoints))]
    return specs + non_base


def qwen_inputs(processor: Any, image: Image.Image, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}


def qwen_batch_inputs(processor: Any, prompt_rows: list[dict[str, Any]], device: torch.device, data_root: str) -> dict[str, torch.Tensor]:
    images: list[Image.Image] = []
    texts: list[str] = []
    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "left"
    try:
        for row in prompt_rows:
            image_path = resolve_image_path(row, data_root)
            with Image.open(image_path) as image:
                image = image.convert("RGB").copy()
            images.append(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": str(row["prompt"]).strip()},
                    ],
                }
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}


def eos_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for raw_id in [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)]:
        if isinstance(raw_id, int):
            ids.add(int(raw_id))
        elif isinstance(raw_id, list):
            ids.update(int(item) for item in raw_id if item is not None)
    for token in ("<|endoftext|>", "<|im_end|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return ids


def distribution_metrics(logits: torch.Tensor, eos_ids: set[int]) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()
    entropy = float((-(probs * log_probs).sum()).detach().cpu())
    top2 = torch.topk(logits_f, k=2)
    margin = float((top2.values[0] - top2.values[1]).detach().cpu())
    eos_prob = 0.0
    valid_eos = [idx for idx in eos_ids if 0 <= idx < probs.shape[-1]]
    if valid_eos:
        eos_prob = float(probs[torch.tensor(valid_eos, device=probs.device)].sum().detach().cpu())
    return entropy, eos_prob, margin, probs, log_probs


def kl_student_base(student_probs: torch.Tensor, student_log_probs: torch.Tensor, base_log_probs: torch.Tensor) -> float:
    kl = (student_probs * (student_log_probs - base_log_probs)).sum()
    return float(kl.detach().cpu())


def repeated_ngram_rate(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    return float((len(grams) - len(set(grams))) / max(1, len(grams)))


def unique_token_ratio(tokens: list[int]) -> float:
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


@torch.inference_mode()
def base_next_logits(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    return model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]


def append_token_to_inputs(inputs: dict[str, torch.Tensor], token: int) -> None:
    token_tensor = torch.tensor([[int(token)]], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], token_tensor], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(token_tensor)], dim=1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(token_tensor)], dim=1)


def append_tokens_to_inputs(inputs: dict[str, torch.Tensor], tokens: torch.Tensor) -> None:
    token_tensor = tokens.to(device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype).view(-1, 1)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], token_tensor], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(token_tensor)], dim=1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(token_tensor)], dim=1)


def distribution_metrics_batch(
    logits: torch.Tensor,
    eos_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(logits_f, k=2, dim=-1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    eos_prob = torch.zeros((logits.shape[0],), device=logits.device, dtype=torch.float32)
    valid_eos = [idx for idx in eos_ids if 0 <= idx < probs.shape[-1]]
    if valid_eos:
        eos_prob = probs[:, torch.tensor(valid_eos, device=probs.device)].sum(dim=-1)
    return entropy, eos_prob, margin, probs, log_probs


@torch.inference_mode()
def base_next_logits_batch(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    return model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]


@torch.inference_mode()
def generate_base_batch_metrics(
    model: Any,
    processor: Any,
    initial_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[list[list[int]], list[list[dict[str, float]]]]:
    eos_ids = eos_token_ids(processor.tokenizer)
    eos_fallback = int(next(iter(eos_ids))) if eos_ids else int(processor.tokenizer.eos_token_id or 0)
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    batch = int(inputs["input_ids"].shape[0])
    tokens: list[list[int]] = [[] for _ in range(batch)]
    per_step: list[list[dict[str, float]]] = [[] for _ in range(batch)]
    active = torch.ones((batch,), device=inputs["input_ids"].device, dtype=torch.bool)
    for step_idx in range(max_new_tokens):
        logits = base_next_logits_batch(model, inputs)
        entropy, eos_prob, margin, _, _ = distribution_metrics_batch(logits, eos_ids)
        next_tokens = torch.argmax(logits, dim=-1)
        append_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_fallback))
        for row_idx in range(batch):
            if not bool(active[row_idx].item()):
                continue
            token = int(next_tokens[row_idx].item())
            tokens[row_idx].append(token)
            per_step[row_idx].append(
                {
                    "decode_step": float(step_idx + 1),
                    "entropy": float(entropy[row_idx].detach().cpu()),
                    "eos_prob": float(eos_prob[row_idx].detach().cpu()),
                    "max_logit_margin": float(margin[row_idx].detach().cpu()),
                    "kl_student_base": 0.0,
                }
            )
        active = active & ~torch.tensor([int(t.item()) in eos_ids for t in next_tokens], device=active.device, dtype=torch.bool)
        append_tokens_to_inputs(inputs, append_tokens)
        if not bool(active.any().item()):
            break
    return tokens, per_step


@torch.inference_mode()
def generate_adapter_batch_metrics(
    model: Any,
    processor: Any,
    adapter: Any,
    initial_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    kl_stride: int,
) -> tuple[list[list[int]], list[list[dict[str, float]]]]:
    eos_ids = eos_token_ids(processor.tokenizer)
    eos_fallback = int(next(iter(eos_ids))) if eos_ids else int(processor.tokenizer.eos_token_id or 0)
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    base_inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    full_mask = inputs["attention_mask"]
    last_pos_idx = full_mask.long().sum(dim=1).sub(1).view(1, -1, 1).expand(position_ids.shape[0], -1, 1)
    token_position_ids = position_ids.gather(2, last_pos_idx)
    token_embeddings = model.model.get_input_embeddings()
    batch = int(inputs["input_ids"].shape[0])
    tokens: list[list[int]] = [[] for _ in range(batch)]
    per_step: list[list[dict[str, float]]] = [[] for _ in range(batch)]
    active = torch.ones((batch,), device=inputs["input_ids"].device, dtype=torch.bool)
    kl_stride = max(1, int(kl_stride))
    for step_idx in range(max_new_tokens):
        logits, _, _ = qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            collect_states=False,
            compact_no_padding=True,
            logits_to_keep=1,
        )
        student_logits = logits[:, -1, :]
        entropy, eos_prob, margin, student_probs, student_log_probs = distribution_metrics_batch(student_logits, eos_ids)
        kl_values = torch.full((batch,), float("nan"), device=student_logits.device, dtype=torch.float32)
        if step_idx % kl_stride == 0:
            base_logits = base_next_logits_batch(model, base_inputs)
            _, _, _, _, base_log_probs = distribution_metrics_batch(base_logits, eos_ids)
            kl_values = (student_probs * (student_log_probs - base_log_probs)).sum(dim=-1)
        next_tokens = torch.argmax(student_logits, dim=-1)
        append_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_fallback))
        for row_idx in range(batch):
            if not bool(active[row_idx].item()):
                continue
            token = int(next_tokens[row_idx].item())
            tokens[row_idx].append(token)
            per_step[row_idx].append(
                {
                    "decode_step": float(step_idx + 1),
                    "entropy": float(entropy[row_idx].detach().cpu()),
                    "eos_prob": float(eos_prob[row_idx].detach().cpu()),
                    "max_logit_margin": float(margin[row_idx].detach().cpu()),
                    "kl_student_base": float(kl_values[row_idx].detach().cpu()),
                }
            )
        active = active & ~torch.tensor([int(t.item()) in eos_ids for t in next_tokens], device=active.device, dtype=torch.bool)
        token_tensor = append_tokens.to(dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device).view(-1, 1)
        append_tokens_to_inputs(inputs, append_tokens)
        append_tokens_to_inputs(base_inputs, append_tokens)
        initial_hidden = torch.cat(
            [initial_hidden, token_embeddings(token_tensor).to(device=initial_hidden.device, dtype=initial_hidden.dtype)],
            dim=1,
        )
        token_position_ids = token_position_ids + 1
        position_ids = torch.cat([position_ids, token_position_ids], dim=2)
        if not bool(active.any().item()):
            break
    return tokens, per_step


@torch.inference_mode()
def generate_base_metrics(
    model: Any,
    processor: Any,
    initial_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[list[int], list[dict[str, float]]]:
    eos_ids = eos_token_ids(processor.tokenizer)
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    tokens: list[int] = []
    per_step: list[dict[str, float]] = []
    for step_idx in range(max_new_tokens):
        logits = base_next_logits(model, inputs)
        entropy, eos_prob, margin, _, _ = distribution_metrics(logits, eos_ids)
        next_token = int(torch.argmax(logits).item())
        tokens.append(next_token)
        per_step.append(
            {
                "decode_step": float(step_idx + 1),
                "entropy": entropy,
                "eos_prob": eos_prob,
                "max_logit_margin": margin,
                "kl_student_base": 0.0,
            }
        )
        append_token_to_inputs(inputs, next_token)
        if next_token in eos_ids:
            break
    return tokens, per_step


@torch.inference_mode()
def generate_adapter_metrics(
    model: Any,
    processor: Any,
    adapter: Any,
    initial_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    kl_stride: int,
) -> tuple[list[int], list[dict[str, float]]]:
    eos_ids = eos_token_ids(processor.tokenizer)
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    base_inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_inputs.items()}
    initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    full_mask = inputs["attention_mask"]
    last_pos_idx = full_mask.long().sum(dim=1).sub(1).view(1, -1, 1).expand(position_ids.shape[0], -1, 1)
    token_position_ids = position_ids.gather(2, last_pos_idx)
    token_embeddings = model.model.get_input_embeddings()
    tokens: list[int] = []
    per_step: list[dict[str, float]] = []
    kl_stride = max(1, int(kl_stride))
    for step_idx in range(max_new_tokens):
        logits, text_mask, _ = qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            collect_states=False,
            compact_no_padding=True,
            logits_to_keep=1,
        )
        student_logits = logits[0, -1]
        entropy, eos_prob, margin, student_probs, student_log_probs = distribution_metrics(student_logits, eos_ids)
        kl_value = math.nan
        if step_idx % kl_stride == 0:
            base_logits = base_next_logits(model, base_inputs)
            _, _, _, _, base_log_probs = distribution_metrics(base_logits, eos_ids)
            kl_value = kl_student_base(student_probs, student_log_probs, base_log_probs)
        next_token = int(torch.argmax(student_logits).item())
        tokens.append(next_token)
        per_step.append(
            {
                "decode_step": float(step_idx + 1),
                "entropy": entropy,
                "eos_prob": eos_prob,
                "max_logit_margin": margin,
                "kl_student_base": kl_value,
            }
        )
        token_tensor = torch.tensor([[next_token]], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
        append_token_to_inputs(inputs, next_token)
        append_token_to_inputs(base_inputs, next_token)
        initial_hidden = torch.cat(
            [initial_hidden, token_embeddings(token_tensor).to(device=initial_hidden.device, dtype=initial_hidden.dtype)],
            dim=1,
        )
        token_position_ids = token_position_ids + 1
        position_ids = torch.cat([position_ids, token_position_ids], dim=2)
        if next_token in eos_ids:
            break
    return tokens, per_step


def finite_mean(values: list[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return math.nan
    return float(sum(filtered) / len(filtered))


def summarize_prompt(
    spec: CheckpointSpec,
    prompt_row: dict[str, Any],
    tokens: list[int],
    per_step: list[dict[str, float]],
    tokenizer: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    eos_ids = eos_token_ids(tokenizer)
    ended_eos = bool(tokens and tokens[-1] in eos_ids)
    return {
        "family": spec.family,
        "label": spec.label,
        "checkpoint_step": spec.step,
        "checkpoint": str(spec.path) if spec.path else "base",
        "prompt_index": prompt_row.get("index"),
        "source": prompt_row.get("source", ""),
        "completion_length": len(tokens),
        "ended_eos": int(ended_eos),
        "hit_max_tokens": int(len(tokens) >= max_new_tokens and not ended_eos),
        "mean_entropy": finite_mean([row["entropy"] for row in per_step]),
        "mean_eos_prob": finite_mean([row["eos_prob"] for row in per_step]),
        "mean_max_logit_margin": finite_mean([row["max_logit_margin"] for row in per_step]),
        "mean_kl_student_base": finite_mean([row["kl_student_base"] for row in per_step]),
        "repeat_4gram_rate": repeated_ngram_rate(tokens, n=4),
        "unique_token_ratio": unique_token_ratio(tokens),
        "prompt": prompt_row.get("prompt", ""),
        "image": prompt_row.get("image", ""),
        "completion": decoded.replace("\n", "\\n")[:1000],
    }


def aggregate_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["label"], int(row["checkpoint_step"])), []).append(row)
    out: list[dict[str, Any]] = []
    numeric_keys = [
        "completion_length",
        "ended_eos",
        "hit_max_tokens",
        "mean_entropy",
        "mean_eos_prob",
        "mean_max_logit_margin",
        "mean_kl_student_base",
        "repeat_4gram_rate",
        "unique_token_ratio",
    ]
    for (family, label, step), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        row = {"family": family, "label": label, "checkpoint_step": step, "num_prompts": len(items)}
        for key in numeric_keys:
            row[key] = finite_mean([float(item[key]) for item in items])
        out.append(row)
    return out


def aggregate_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["family"], row["label"], int(row["checkpoint_step"]), int(row["decode_step"])),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (family, label, ckpt_step, decode_step), items in sorted(groups.items(), key=lambda item: item[0]):
        out.append(
            {
                "family": family,
                "label": label,
                "checkpoint_step": ckpt_step,
                "decode_step": decode_step,
                "num_prompts_alive": len(items),
                "entropy": finite_mean([float(item["entropy"]) for item in items]),
                "eos_prob": finite_mean([float(item["eos_prob"]) for item in items]),
                "max_logit_margin": finite_mean([float(item["max_logit_margin"]) for item in items]),
                "kl_student_base": finite_mean([float(item["kl_student_base"]) for item in items]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plots: {exc}", flush=True)
        return
    metrics = [
        ("mean_entropy", "Mean Token Entropy"),
        ("repeat_4gram_rate", "4-gram Repetition Rate"),
        ("unique_token_ratio", "Unique-token Ratio"),
        ("completion_length", "Completion Length"),
        ("mean_eos_prob", "Mean EOS Probability"),
        ("mean_max_logit_margin", "Mean Max-logit Margin"),
        ("mean_kl_student_base", "Mean KL(student || base)"),
    ]
    families = sorted({row["family"] for row in summary_rows})
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()
    for ax, (key, title) in zip(axes_flat, metrics):
        for family in families:
            rows = sorted([row for row in summary_rows if row["family"] == family], key=lambda row: int(row["checkpoint_step"]))
            if not rows:
                continue
            ax.plot(
                [int(row["checkpoint_step"]) for row in rows],
                [float(row[key]) for row in rows],
                marker="o",
                label=family,
            )
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "checkpoint_pathology_summary.png", dpi=180)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = load_prompts(args)
    checkpoint_specs = collect_checkpoints(args)
    if len(checkpoint_specs) <= int(args.include_base):
        raise RuntimeError("no checkpoints selected")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    configure_torch_runtime()
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    sample_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    run_meta = {
        "model_path": args.model_path,
        "prompt_jsonl": str(resolve_path(args.prompt_jsonl)),
        "num_prompts": len(prompt_rows),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "max_image_pixels": args.max_image_pixels,
        "kl_stride": args.kl_stride,
        "checkpoints": [
            {"family": spec.family, "label": spec.label, "step": spec.step, "path": str(spec.path) if spec.path else "base"}
            for spec in checkpoint_specs
        ],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for ckpt_idx, spec in enumerate(checkpoint_specs, start=1):
        print(f"=== [{ckpt_idx}/{len(checkpoint_specs)}] {spec.family}:{spec.label} step={spec.step} ===", flush=True)
        adapter = None
        if spec.path is not None:
            adapter, meta = load_qwen_visual_delta_checkpoint(spec.path, model.model.language_model, device, dtype)
            if meta["missing"] or meta["unexpected"]:
                print(f"checkpoint load missing={meta['missing']} unexpected={meta['unexpected']}", flush=True)

        batch_size = max(1, int(args.batch_size))
        completed = 0
        for batch_start in range(0, len(prompt_rows), batch_size):
            batch_t0 = time.perf_counter()
            batch_rows = prompt_rows[batch_start : batch_start + batch_size]
            inputs = qwen_batch_inputs(processor, batch_rows, device, args.data_root)
            if adapter is None:
                batch_tokens, batch_steps = generate_base_batch_metrics(
                    model,
                    processor,
                    inputs,
                    max_new_tokens=int(args.max_new_tokens),
                )
            else:
                batch_tokens, batch_steps = generate_adapter_batch_metrics(
                    model,
                    processor,
                    adapter,
                    inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    kl_stride=int(args.kl_stride),
                )

            for local_idx, prompt_row in enumerate(batch_rows):
                tokens = batch_tokens[local_idx]
                per_step = batch_steps[local_idx]
                sample_rows.append(
                    summarize_prompt(spec, prompt_row, tokens, per_step, processor.tokenizer, int(args.max_new_tokens))
                )
                for row in per_step:
                    step_rows.append(
                        {
                            "family": spec.family,
                            "label": spec.label,
                            "checkpoint_step": spec.step,
                            "prompt_index": prompt_row.get("index"),
                            **row,
                        }
                    )

            completed += len(batch_rows)
            batch_sec = time.perf_counter() - batch_t0
            print(f"  prompts {completed}/{len(prompt_rows)} batch_sec={batch_sec:.1f}", flush=True)
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = aggregate_sample_rows(sample_rows)
    decode_rows = aggregate_step_rows(step_rows)
    write_csv(out_dir / "per_prompt_metrics.csv", sample_rows)
    write_csv(out_dir / "checkpoint_summary.csv", summary_rows)
    write_csv(out_dir / "decode_step_metrics.csv", decode_rows)
    if not args.no_plot:
        plot_summary(summary_rows, out_dir)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    run()
