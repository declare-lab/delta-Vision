from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..modeling import get_keep_indices, install_lcd_forward, set_pruning


class QwenEPICRunner:
    def __init__(self, model_path: str, adapter_path: str | None, *, keep_ratio: float = 0.05,
                 pruning_layer: int = 2, pruning_method: str = "dart",
                 max_pixels: int = 1048576, attn_implementation: str = "sdpa"):
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.size["longest_edge"] = max_pixels
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
            attn_implementation=attn_implementation, trust_remote_code=True,
        )
        install_lcd_forward(model)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.keep_ratio, self.pruning_layer = keep_ratio, pruning_layer
        self.pruning_method = pruning_method

    @staticmethod
    def _load_images(images):
        loaded = []
        for image in images:
            if isinstance(image, Image.Image):
                loaded.append(image.convert("RGB"))
            else:
                with Image.open(Path(image)) as handle:
                    loaded.append(handle.convert("RGB"))
        return loaded

    @staticmethod
    def _content(question: str, image_count: int):
        return [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": question}]

    def _encode(self, question, images, answer=None):
        messages = [{"role": "user", "content": self._content(question, len(images))}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )
        batch = self.processor(text=[text], images=images, return_tensors="pt")
        return {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _set_budget(self, batch):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        mask = batch["input_ids"].eq(base.config.image_token_id) | batch["input_ids"].eq(
            base.config.video_token_id
        )
        set_pruning(self.model, layer=self.pruning_layer, ratio=1.0 - self.keep_ratio,
                    method=self.pruning_method, min_keep=1, full_visual_mask=mask)

    @torch.inference_mode()
    def score_choices(self, question: str, images, choices: list[str]):
        images = self._load_images(images)
        scores = []
        for index, choice in enumerate(choices):
            letter = chr(ord("A") + index)
            candidate = letter
            prompt = f"{question}\n" + "\n".join(
                f"{chr(ord('A') + i)}. {text}" for i, text in enumerate(choices)
            ) + "\nAnswer with only the option letter."
            prompt_batch = self._encode(prompt, images)
            batch = self._encode(prompt, images, candidate)
            prompt_len = min(prompt_batch["input_ids"].shape[1], batch["input_ids"].shape[1])
            labels = batch["input_ids"].clone()
            labels[:, :prompt_len] = -100
            self._set_budget(batch)
            outputs = self.model(**batch, use_cache=False, return_dict=True)
            keep = get_keep_indices(self.model)
            labels = labels[:, keep]
            shift_logits, shift_labels = outputs.logits[:, :-1], labels[:, 1:]
            mask = shift_labels.ne(-100)
            loss = F.cross_entropy(shift_logits[mask].float(), shift_labels[mask], reduction="mean")
            scores.append(-float(loss))
        best = max(range(len(scores)), key=scores.__getitem__)
        return chr(ord("A") + best), scores

    @torch.inference_mode()
    def generate(self, question: str, images, max_new_tokens: int = 64):
        """Correct cache-free greedy decoding for layer-varying LCD KV lengths."""
        images = self._load_images(images)
        batch = self._encode(question, images)
        prompt_len = batch["input_ids"].shape[1]
        self._set_budget(batch)
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        for _ in range(max_new_tokens):
            outputs = self.model(**batch, use_cache=False, return_dict=True)
            token = outputs.logits[:, -1].argmax(-1, keepdim=True)
            batch["input_ids"] = torch.cat((batch["input_ids"], token), dim=1)
            batch["attention_mask"] = torch.cat(
                (batch["attention_mask"], torch.ones_like(token)), dim=1
            )
            if int(token) in eos_ids:
                break
        return self.processor.tokenizer.decode(
            batch["input_ids"][0, prompt_len:], skip_special_tokens=True
        ).strip()


def normalize_answer(answer):
    return re.sub(r"\s+", " ", str(answer or "").strip().lower())


def choice_correct(prediction: str, answer: str | None, choices: list[str]):
    if answer is None:
        return None
    pred = prediction.strip().upper()[:1]
    gold = str(answer).strip()
    if len(gold) == 1 and gold.upper() in "ABCDE":
        return pred == gold.upper()
    try:
        return normalize_answer(choices[ord(pred) - ord("A")]) == normalize_answer(gold)
    except (IndexError, TypeError):
        return False

