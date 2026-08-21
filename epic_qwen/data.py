from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class JsonlVQADataset(Dataset):
    def __init__(self, path: str, max_samples: int | None = None):
        with open(path, encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"image": row["image"], "question": row["question"], "answer": row["answer"]}


class QwenCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def _render(self, question: str, answer: str | None):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question},
        ]}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("Use per_device_train_batch_size=1 and gradient accumulation")
        row = rows[0]
        with Image.open(Path(row["image"])) as image:
            image = image.convert("RGB")
            prompt = self._render(row["question"], None)
            full = self._render(row["question"], row["answer"])
            prompt_batch = self.processor(text=[prompt], images=[image], return_tensors="pt")
            # Let the multimodal processor finish expanding all image placeholders
            # before truncating. Truncating inside Qwen's processor can cut an image
            # span and makes the image-token count disagree with image_grid_thw.
            batch = self.processor(text=[full], images=[image], return_tensors="pt")
        if batch["input_ids"].shape[1] > self.max_length:
            image_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            original_images = int(batch["input_ids"].eq(image_id).sum())
            batch["input_ids"] = batch["input_ids"][:, :self.max_length]
            batch["attention_mask"] = batch["attention_mask"][:, :self.max_length]
            if int(batch["input_ids"].eq(image_id).sum()) != original_images:
                raise ValueError(
                    "Visual tokens alone exceed max_length; lower data.max_pixels"
                )
        prompt_len = min(prompt_batch["input_ids"].shape[1], batch["input_ids"].shape[1])
        labels = batch["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not labels.ne(-100).any():
            raise ValueError(
                "Answer was fully truncated; increase data.max_length or reduce the image resolution"
            )
        batch["labels"] = labels
        return batch
