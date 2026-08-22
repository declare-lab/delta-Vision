"""Dataset for VQA training with image + question + answer."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.benchmarks import build_benchmark_prompt, get_benchmark_spec


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


class LlavaBenchmarkDataset(Dataset):
    """Generic LLaVA benchmark dataset using the unified benchmark JSONL schema."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        benchmark: str,
        data_root: str | None = None,
        max_samples: int | None = None,
        answer_instruction: str | None = None,
    ):
        self.processor = processor
        self.spec = get_benchmark_spec(benchmark)
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent
        self.answer_instruction = answer_instruction

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = self._image_path(row)
        question = build_benchmark_prompt(row, self.spec, self.answer_instruction)
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        image = Image.open(image_path).convert("RGB")
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        finally:
            image.close()

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "answer": row.get("answer"),
            "answers": row.get("answers"),
            "choices": row.get("choices"),
            "row": row,
            "index": row.get("index", idx),
        }
        if "image_sizes" in inputs:
            item["image_sizes"] = inputs["image_sizes"].squeeze(0)
        return item

    def _image_path(self, row: dict) -> Path:
        raw_paths = row.get("images")
        if raw_paths is None:
            raw_paths = [row["image"]]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("LLaVA benchmark row must contain image or non-empty images")
        if len(raw_paths) != 1:
            raise ValueError("LLaVA benchmark evaluation currently supports exactly one image per sample")

        root = self.data_root
        row_root = str(row.get("image_root") or "").strip()
        if row_root:
            root = Path(row_root)
        image_path = Path(str(raw_paths[0]))
        if not image_path.is_absolute():
            image_path = root / image_path
        return image_path


class QwenBenchmarkDataset(Dataset):
    """Generic Qwen3-VL benchmark dataset using the unified benchmark JSONL schema."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        benchmark: str,
        data_root: str | None = None,
        max_samples: int | None = None,
        answer_instruction: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.processor = processor
        self.spec = get_benchmark_spec(benchmark)
        self.data_root = Path(data_root) if data_root else Path(jsonl_path).parent
        self.answer_instruction = answer_instruction
        self.cache_dir = Path(cache_dir) if cache_dir else None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_paths = self._image_paths(row)

        question = build_benchmark_prompt(row, self.spec, self.answer_instruction)
        cache_question = str(row.get("problem") or question)
        cache_path = self._cache_path(row, image_paths, cache_question)
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            item = cached["item"]
            item["row"] = row
            item["answer"] = row.get("answer")
            item["answers"] = row.get("answers")
            item["choices"] = row.get("choices")
            item["index"] = row.get("index", idx)
            return item

        images = [Image.open(path).convert("RGB") for path in image_paths]
        content = self._qwen_message_content(row, question, images)
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        for image in images:
            image.close()
        if "mm_token_type_ids" not in inputs:
            raise ValueError("Qwen processor did not return mm_token_type_ids; M-RoPE positions would be invalid")

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0),
            "answer": row.get("answer"),
            "answers": row.get("answers"),
            "choices": row.get("choices"),
            "row": row,
            "index": row.get("index", idx),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
            torch.save({"item": item}, tmp_path)
            os.replace(tmp_path, cache_path)
        return item

    def _qwen_message_content(self, row: dict, question: str, images: list[Image.Image]) -> list[dict]:
        problem = str(row.get("problem") or "")
        if not problem or "<|image_" not in problem:
            return [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": question}]

        content: list[dict] = []
        last = 0
        used: set[int] = set()
        for match in re.finditer(r"<\|image_(\d+)\|>", problem):
            image_idx = int(match.group(1)) - 1
            segment = problem[last : match.start()]
            if segment:
                content.append({"type": "text", "text": segment})
            if 0 <= image_idx < len(images):
                content.append({"type": "image", "image": images[image_idx]})
                used.add(image_idx)
            last = match.end()
        tail = problem[last:]
        if tail:
            content.append({"type": "text", "text": tail})
        for image_idx, image in enumerate(images):
            if image_idx not in used:
                content.append({"type": "image", "image": image})
        return content or ([{"type": "image", "image": image} for image in images] + [{"type": "text", "text": question}])

    def _image_paths(self, row: dict) -> list[Path]:
        raw_paths = row.get("images")
        if raw_paths is None:
            raw_paths = [row["image"]]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("Qwen benchmark row must contain image or non-empty images")
        root = self.data_root
        row_root = str(row.get("image_root") or "").strip()
        if row_root:
            root = Path(row_root)
        image_paths = []
        for raw_path in raw_paths:
            image_path = Path(str(raw_path))
            if not image_path.is_absolute():
                image_path = root / image_path
            image_paths.append(image_path)
        return image_paths

    def _cache_path(self, row: dict, image_paths: list[Path], question: str) -> Path | None:
        if self.cache_dir is None:
            return None
        image_stats = []
        for image_path in image_paths:
            stat = image_path.stat()
            image_stats.append(
                {
                    "image": str(image_path),
                    "image_size": int(stat.st_size),
                    "image_mtime_ns": int(stat.st_mtime_ns),
                }
            )
        key = {
            "benchmark": self.spec.name,
            "processor": str(getattr(self.processor, "name_or_path", "")),
            "images": image_stats,
            "question": question,
            "answer_instruction": self.answer_instruction,
            "index": row.get("index"),
        }
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / self.spec.name / f"{digest}.pt"
