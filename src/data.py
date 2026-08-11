"""Dataset for VQA training with image + question + answer."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """Load pixmo_ama style JSONL: {image, question, answer}.

    image paths in the JSONL are relative to data_root.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        data_root: str | None = None,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.processor = processor
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.rows)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = self.data_root / row["image"]
        question = str(row["question"]).strip()
        answer = str(row.get("answer", "")).strip()

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        full_text = f"{prompt} {answer}" if answer else prompt

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")

        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        # prompt_len counts all tokens including image tokens
        # We want the text-only prompt length (excluding image tokens)
        image_token_id = getattr(self.processor, "image_token_id", 32000)
        if image_token_id is None:
            image_token_id = 32000
        num_image_in_prompt = (prompt_inputs["input_ids"] == image_token_id).sum().item()
        prompt_len = prompt_inputs["input_ids"].shape[1] - num_image_in_prompt

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }
        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"]
        return result


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to max length in batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id = 0

    input_ids = []
    pixel_values = []
    attention_masks = []
    prompt_lens = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        ids = F.pad(item["input_ids"], (0, pad_len), value=pad_id)
        mask = F.pad(item["attention_mask"], (0, pad_len), value=0)

        input_ids.append(ids)
        pixel_values.append(item["pixel_values"])
        attention_masks.append(mask)
        prompt_lens.append(item["prompt_len"])

    result = {
        "input_ids": torch.stack(input_ids),
        "pixel_values": torch.stack(pixel_values),
        "attention_mask": torch.stack(attention_masks),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }
    if "image_sizes" in batch[0]:
        result["image_sizes"] = [item["image_sizes"] for item in batch]
    return result


class MMStarDataset(Dataset):
    """Load MMStar eval JSONL: {index, image, question, answer, category}.

    image paths in the JSONL are relative to data_root.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        data_root: str | None = None,
        max_samples: int | None = None,
    ):
        self.processor = processor
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = self.data_root / row["image"]
        question = str(row["question"]).strip()
        gold = str(row["answer"]).strip().upper()[:1]

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "gold": gold,
            "index": row.get("index", idx),
        }


class OPDDataset(Dataset):
    """Vision-OPD-6K dataset."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        data_root: str | None = None,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.processor = processor
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.rows)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = self.data_root / row["images"][0]
        problem = row["problem"]
        answer = str(row["answer"]).strip()

        prompt = f"USER: {problem}\nASSISTANT:"
        full_text = f"{prompt} {answer}"

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        image_token_id = 32000
        num_image_in_prompt = (prompt_inputs["input_ids"] == image_token_id).sum().item()
        prompt_len = prompt_inputs["input_ids"].shape[1] - num_image_in_prompt

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }
