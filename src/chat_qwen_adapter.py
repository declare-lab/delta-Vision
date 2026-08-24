#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]

from src.eval_benchmarks import configure_torch_runtime, generate_adapter_qwen, generate_teacher_qwen
from src.model import (
    dtype_from_name,
    load_frozen_qwen3vl,
    load_qwen_embedding_adapter_checkpoint,
)


DEFAULT_MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_CKPT = os.environ.get(
    "QWEN_EMBEDDING_ADAPTER_CKPT",
    "/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/qwen_embedding_adapter_step500.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Interactive Qwen3-VL embedding_adapter chat.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Qwen embedding_adapter checkpoint path. Can also set QWEN_EMBEDDING_ADAPTER_CKPT.")
    parser.add_argument("--image", default=None, help="Initial image path. You can also set it with /image in chat.")
    parser.add_argument("--mode", choices=("adapter", "teacher", "both"), default="both")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.checkpoint:
        raise SystemExit("set --checkpoint or QWEN_EMBEDDING_ADAPTER_CKPT")
    return args


def resolve_path(path: str | Path) -> Path:
    text = str(path).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    for quote in ("'", '"'):
        marker = f"{quote}/"
        if marker in text:
            text = "/" + text.split(marker, 1)[1]
            if text.endswith(quote):
                text = text[:-1]
            text = text.strip()
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (ROOT_DIR / candidate).resolve()


def load_image(path: str | Path | None) -> tuple[Image.Image | None, Path | None]:
    if path is None or not str(path).strip():
        return None, None
    image_path = resolve_path(path)
    image = Image.open(image_path).convert("RGB")
    return image, image_path


def build_inputs(processor: Any, image: Image.Image | None, messages: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [image] if image is not None else None
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    required = ("input_ids", "attention_mask", "mm_token_type_ids")
    missing = [key for key in required if key not in inputs]
    if missing:
        raise ValueError(f"processor output is missing {missing}; adapter chat requires Qwen multimodal inputs")
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}


def build_messages(
    question: str,
    image: Image.Image | None,
    history: list[tuple[str, str]],
    keep_history: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    prior = history if keep_history else []
    for idx, (user_text, assistant_text) in enumerate(prior):
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if idx == 0 and image is not None:
            content.insert(0, {"type": "image", "image": image})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

    content = [{"type": "text", "text": question}]
    if image is not None and not prior:
        content.insert(0, {"type": "image", "image": image})
    messages.append({"role": "user", "content": content})
    return messages


@torch.inference_mode()
def generate_once(
    *,
    mode: str,
    model: Any,
    processor: Any,
    adapter: Any,
    image: Image.Image | None,
    messages: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, str]:
    inputs = build_inputs(processor, image, messages, device)
    common = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "mm_token_type_ids": inputs["mm_token_type_ids"],
        "max_new_tokens": max_new_tokens,
    }
    if common["pixel_values"] is None or common["image_grid_thw"] is None:
        if mode in {"adapter", "both"}:
            raise ValueError("adapter mode needs an image. Use --image PATH or /image PATH.")

    outputs: dict[str, str] = {}
    if mode in {"teacher", "both"}:
        _, text = generate_teacher_qwen(model, processor, **common)
        outputs["teacher"] = text.strip()
    if mode in {"adapter", "both"}:
        _, text = generate_adapter_qwen(model, processor, adapter, **common, early_stop_metric=None)
        outputs["adapter"] = text.strip()
    return outputs


def print_help() -> None:
    print(
        "Commands:\n"
        "  /image PATH        set or replace the image and clear history\n"
        "  /mode adapter|teacher|both\n"
        "  /reset             clear chat history\n"
        "  /history on|off    enable or disable multi-turn history\n"
        "  /exit              quit\n",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    model_path = str(resolve_path(args.model_path)) if not Path(args.model_path).is_absolute() else args.model_path
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    configure_torch_runtime()
    print(f"Loading model: {model_path}", flush=True)
    processor, model = load_frozen_qwen3vl(model_path, dtype, device, args.attn_implementation)
    print(f"Loading adapter: {checkpoint}", flush=True)
    adapter, meta = load_qwen_embedding_adapter_checkpoint(checkpoint, model.model.language_model, device, dtype)
    print(
        f"Loaded adapter mode={adapter.mode} step={meta.get('global_step')} missing={len(meta['missing'])} unexpected={len(meta['unexpected'])}",
        flush=True,
    )

    image, image_path = load_image(args.image)
    mode = args.mode
    keep_history = bool(args.history)
    history: list[tuple[str, str]] = []
    print(f"mode={mode} image={image_path or 'none'} history={'on' if keep_history else 'off'}", flush=True)
    print_help()

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            break
        if question == "/reset":
            history.clear()
            print("history cleared", flush=True)
            continue
        if question.startswith("/history"):
            parts = question.split()
            if len(parts) != 2 or parts[1] not in {"on", "off"}:
                print("usage: /history on|off", flush=True)
                continue
            keep_history = parts[1] == "on"
            print(f"history={'on' if keep_history else 'off'}", flush=True)
            continue
        if question.startswith("/mode"):
            parts = question.split()
            if len(parts) != 2 or parts[1] not in {"adapter", "teacher", "both"}:
                print("usage: /mode adapter|teacher|both", flush=True)
                continue
            mode = parts[1]
            print(f"mode={mode}", flush=True)
            continue
        if question.startswith("/image"):
            path = question[len("/image") :].strip()
            if not path:
                print("usage: /image PATH", flush=True)
                continue
            image, image_path = load_image(path)
            history.clear()
            print(f"image={image_path}; history cleared", flush=True)
            continue

        messages = build_messages(question, image, history, keep_history)
        try:
            outputs = generate_once(
                mode=mode,
                model=model,
                processor=processor,
                adapter=adapter,
                image=image,
                messages=messages,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"error: {exc}", flush=True)
            continue

        assistant_for_history = ""
        if "teacher" in outputs:
            print(f"teacher> {outputs['teacher']}", flush=True)
            assistant_for_history = outputs["teacher"]
        if "adapter" in outputs:
            print(f"adapter> {outputs['adapter']}", flush=True)
            assistant_for_history = outputs["adapter"]
        if keep_history and assistant_for_history:
            history.append((question, assistant_for_history))


if __name__ == "__main__":
    main()
