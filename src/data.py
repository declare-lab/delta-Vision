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
        "pixel_values": torch.stack(pixel_values) if all(p.shape == pixel_values[0].shape for p in pixel_values) else pixel_values,
        "attention_mask": torch.stack(attention_masks),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }
    if "image_sizes" in batch[0]:
        sizes = [item["image_sizes"] for item in batch]
        if torch.is_tensor(sizes[0]):
            result["image_sizes"] = torch.cat(sizes, dim=0)
        else:
            result["image_sizes"] = sizes
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
        answer_instruction: str = "",
    ):
        self.processor = processor
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent
        self.answer_instruction = answer_instruction.strip()

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
        if self.answer_instruction:
            question = f"{question}\n{self.answer_instruction}"
        gold = str(row["answer"]).strip().upper()[:1]

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "gold": gold,
            "index": row.get("index", idx),
        }
        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"].squeeze(0)
        return result


class QwenMMStarDataset(Dataset):
    """MMStar eval dataset formatted with Qwen3-VL chat template."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        data_root: str | None = None,
        max_samples: int | None = None,
        answer_instruction: str = "Answer directly with only the letter of the correct option.",
    ):
        self.processor = processor
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent
        self.answer_instruction = answer_instruction.strip()

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
        if self.answer_instruction:
            question = f"{question}\n{self.answer_instruction}"
        gold = str(row["answer"]).strip().upper()[:1]

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        if "mm_token_type_ids" not in inputs:
            raise ValueError("Qwen processor did not return mm_token_type_ids; M-RoPE positions would be invalid")

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0),
            "gold": gold,
            "index": row.get("index", idx),
        }
        return result


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

        prompt = f"USER: <image>\n{problem}\nASSISTANT:"
        full_text = f"{prompt} {answer}"

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        image_token_id = getattr(self.processor, "image_token_id", 32000)
        if image_token_id is None:
            image_token_id = 32000
        num_image_in_prompt = (prompt_inputs["input_ids"] == image_token_id).sum().item()
        prompt_len = prompt_inputs["input_ids"].shape[1] - num_image_in_prompt

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }
