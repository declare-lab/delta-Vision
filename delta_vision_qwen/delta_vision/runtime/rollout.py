from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from delta_vision.models.llava import (
    llava15_prompt,
)


def build_full_text(question: str, answer: str, eos_token: str | None) -> tuple[str, str]:
    prompt = llava15_prompt(question.strip())
    clean_answer = answer.strip()
    if eos_token and not clean_answer.endswith(eos_token):
        clean_answer = clean_answer + eos_token
    return prompt, f"{prompt} {clean_answer}"


def source_text_ids_and_answer_mask(
    input_ids: Tensor,
    prompt_len: int,
    image_token_id: int,
) -> tuple[Tensor, Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("rollout currently expects batch size 1")
    image_count = int((input_ids[0] == image_token_id).sum().item())
    effective_prompt_len = prompt_len + max(0, image_count - 1)
    text_ids: list[int] = []
    answer_mask: list[bool] = []
    for src_idx, token_id in enumerate(input_ids[0].tolist()):
        if token_id == image_token_id:
            continue
        text_ids.append(token_id)
        answer_mask.append(src_idx >= effective_prompt_len)
    return (
        torch.tensor(text_ids, dtype=torch.long, device=input_ids.device).unsqueeze(0),
        torch.tensor(answer_mask, dtype=torch.bool, device=input_ids.device).unsqueeze(0),
    )


def source_text_ids_and_answer_mask_batch(
    input_ids: Tensor,
    attention_mask: Tensor,
    prompt_lens: list[int],
    image_token_id: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
    if len(prompt_lens) != input_ids.shape[0]:
        raise ValueError("prompt_lens length must match batch size")

    rows_ids: list[list[int]] = []
    rows_answer_mask: list[list[bool]] = []
    max_text = 0
    for batch_idx in range(input_ids.shape[0]):
        text_ids: list[int] = []
        answer_mask: list[bool] = []
        valid_src = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        image_count = int((input_ids[batch_idx] == image_token_id).logical_and(attention_mask[batch_idx].bool()).sum().item())
        effective_prompt_len = int(prompt_lens[batch_idx]) + max(0, image_count - 1)
        for src_idx in valid_src:
            src_idx = int(src_idx)
            token_id = int(input_ids[batch_idx, src_idx].item())
            if token_id == image_token_id:
                continue
            text_ids.append(token_id)
            answer_mask.append(src_idx >= effective_prompt_len)
        rows_ids.append(text_ids)
        rows_answer_mask.append(answer_mask)
        max_text = max(max_text, len(text_ids))

    text_ids_tensor = torch.full(
        (input_ids.shape[0], max_text),
        int(pad_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    answer_mask_tensor = torch.zeros((input_ids.shape[0], max_text), device=input_ids.device, dtype=torch.bool)
    text_mask_tensor = torch.zeros((input_ids.shape[0], max_text), device=input_ids.device, dtype=torch.bool)
    for batch_idx, ids in enumerate(rows_ids):
        text_len = len(ids)
        if text_len == 0:
            raise ValueError("empty text sequence after removing image token")
        text_ids_tensor[batch_idx, :text_len] = torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)
        answer_mask_tensor[batch_idx, :text_len] = torch.tensor(
            rows_answer_mask[batch_idx],
            device=input_ids.device,
            dtype=torch.bool,
        )
        text_mask_tensor[batch_idx, :text_len] = True
    return text_ids_tensor, answer_mask_tensor, text_mask_tensor


def prepare_sample_inputs(
    processor: Any,
    row: dict[str, Any],
    image_key: str,
    question_key: str,
    answer_key: str,
    image_root: Path | None,
    image_token_id: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor, str]:
    image_path = Path(row[image_key])
    if image_root is not None and not image_path.is_absolute():
        image_path = image_root / image_path
    question = str(row[question_key])
    answer = str(row[answer_key])
    prompt, full_text = build_full_text(question, answer, processor.tokenizer.eos_token)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=full_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device, non_blocking=True)
    text_ids, answer_mask = source_text_ids_and_answer_mask(
        inputs["input_ids"],
        prompt_ids.shape[1],
        image_token_id,
    )
    return inputs, text_ids, answer_mask, str(image_path)


def prepare_batch_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    image_key: str,
    question_key: str,
    answer_key: str,
    image_root: Path | None,
    image_token_id: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, list[str]]:
    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    prompts: list[str] = []
    full_texts: list[str] = []
    images: list[Image.Image] = []
    image_paths: list[str] = []
    prompt_lens: list[int] = []
    try:
        for row in rows:
            image_path = Path(row[image_key])
            if image_root is not None and not image_path.is_absolute():
                image_path = image_root / image_path
            prompt, full_text = build_full_text(
                str(row[question_key]),
                str(row[answer_key]),
                processor.tokenizer.eos_token,
            )
            prompts.append(prompt)
            full_texts.append(full_text)
            image_paths.append(str(image_path))
            decoded_image = row.get("_decoded_image")
            if decoded_image is not None:
                images.append(decoded_image)
            else:
                images.append(Image.open(image_path).convert("RGB"))
        prompt_lens = [
            len(ids)
            for ids in processor.tokenizer(prompts, add_special_tokens=True, padding=False).input_ids
        ]

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
            inputs["attention_mask"] = attention_mask
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        text_ids, answer_mask, text_mask = source_text_ids_and_answer_mask_batch(
            inputs["input_ids"],
            attention_mask,
            prompt_lens,
            image_token_id,
            int(pad_token_id),
        )
    finally:
        processor.tokenizer.padding_side = old_padding_side
        for image in images:
            image.close()
    return inputs, text_ids, answer_mask, text_mask, image_paths
