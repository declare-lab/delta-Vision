#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
import string
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.eval_benchmarks import configure_torch_runtime, generate_adapter_qwen, generate_teacher_qwen
from src.model import dtype_from_name, load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HotpotQA rendered-text evaluation with official EM/F1 answer metrics.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default="", help="Qwen embedding adapter checkpoint. Required for adapter_image.")
    parser.add_argument("--validation", default="data/benchmarks/hotpotqa/validation.jsonl")
    parser.add_argument("--render-metadata", default="data/benchmarks/hotpotqa/render/metadata.jsonl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--modes",
        default="base_image,base_text,adapter_image",
        help="Comma/space separated subset of base_image, base_text, adapter_image.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    modes = [item for item in re.split(r"[\s,]+", args.modes.strip()) if item]
    valid_modes = {"base_image", "base_text", "adapter_image"}
    bad = [mode for mode in modes if mode not in valid_modes]
    if bad:
        raise ValueError(f"unsupported modes={bad}; choose from {sorted(valid_modes)}")
    if "adapter_image" in modes and not args.checkpoint:
        raise ValueError("--checkpoint is required when modes includes adapter_image")
    args.modes = modes
    return args


def resolve_path(path: str | Path, root: str | Path = ROOT_DIR) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(root) / candidate).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_context_text(row: dict[str, Any]) -> str:
    context_texts = row.get("context_texts")
    if isinstance(context_texts, list) and context_texts:
        return "\n\n".join(str(item) for item in context_texts)
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    blocks: list[str] = []
    for title, sent_list in zip(titles, sentences):
        text = " ".join(str(sent).strip() for sent in sent_list if str(sent).strip())
        blocks.append(f"{title}\n{text}".strip())
    return "\n\n".join(blocks)


def load_render_pages(path: Path) -> dict[int, list[dict[str, Any]]]:
    pages_by_row: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("skipped"):
                continue
            pages_by_row[int(item["row_index"])].append(item)
    for pages in pages_by_row.values():
        pages.sort(key=lambda item: int(item.get("page_index", 0)))
    return dict(pages_by_row)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens or num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_metric_over_gold_answers(prediction: str, answers: list[str], metric_fn) -> float:
    return max(metric_fn(prediction, answer) for answer in answers)


def gold_answers(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer", "")
    if isinstance(answer, list):
        answers = [str(item) for item in answer]
    else:
        answers = [str(answer)]
    return [item for item in answers if item.strip()] or [""]


def answer_prompt(question: str) -> str:
    return (
        "Answer the HotpotQA question using the provided context. "
        "Return only the answer span, yes, or no. Do not explain.\n\n"
        f"Question: {question}"
    )


def build_text_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        "Context:\n"
        f"{build_context_text(row)}\n\n"
        f"{answer_prompt(str(row.get('question', '')))}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def build_image_messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": answer_prompt(question)})
    return [{"role": "user", "content": content}]


def processor_inputs(processor: Any, messages: list[dict[str, Any]], images: list[Image.Image] | None, device: torch.device) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=images, return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}


@torch.inference_mode()
def generate_base_text(model: Any, processor: Any, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> str:
    eos_ids = []
    tokenizer = processor.tokenizer
    for token_id in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)):
        if token_id is not None and int(token_id) not in eos_ids:
            eos_ids.append(int(token_id))
    kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
        kwargs["pad_token_id"] = eos_ids[0]
    generated = model.generate(**kwargs)
    return processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def load_images(pages: list[dict[str, Any]], data_root: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    for page in pages:
        image_path = resolve_path(page["image"], data_root)
        images.append(Image.open(image_path).convert("RGB"))
    return images


def select_rows(rows: list[dict[str, Any]], pages_by_row: dict[int, list[dict[str, Any]]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any], list[dict[str, Any]]]]:
    selected: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for row_index, row in enumerate(rows):
        if row_index < args.start_index:
            continue
        pages = pages_by_row.get(row_index)
        if not pages:
            continue
        selected.append((row_index, row, pages))
    if args.num_shards > 1:
        selected = [item for idx, item in enumerate(selected) if idx % args.num_shards == args.shard_id]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    return selected


def summarize(predictions: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(predictions), "modes": {}}
    for mode in modes:
        mode_items = [item for item in predictions if item.get(mode) is not None]
        em = sum(float(item[mode]["em"]) for item in mode_items) / max(len(mode_items), 1)
        f1 = sum(float(item[mode]["f1"]) for item in mode_items) / max(len(mode_items), 1)
        summary["modes"][mode] = {"samples": len(mode_items), "exact_match": em, "f1": f1}
    return summary


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_path(args.data_root)
    validation_path = resolve_path(args.validation)
    metadata_path = resolve_path(args.render_metadata)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = read_jsonl(validation_path)
    pages_by_row = load_render_pages(metadata_path)
    selected = select_rows(rows, pages_by_row, args)
    if not selected:
        raise RuntimeError("no HotpotQA rows with rendered pages were selected")

    configure_torch_runtime()
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter = None
    if "adapter_image" in args.modes:
        checkpoint = resolve_path(args.checkpoint)
        adapter, _meta = load_qwen_embedding_adapter_checkpoint(checkpoint, model.model.language_model, device, dtype)

    predictions_path = output_dir / f"predictions_shard{args.shard_id:05d}.jsonl"
    predictions: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as out:
        for idx, (row_index, row, pages) in enumerate(selected, start=1):
            answers = gold_answers(row)
            record: dict[str, Any] = {
                "id": row.get("id"),
                "row_index": row_index,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "num_pages": len(pages),
                "page_indices": [int(page.get("page_index", 0)) for page in pages],
            }

            images: list[Image.Image] | None = None
            if "base_image" in args.modes or "adapter_image" in args.modes:
                images = load_images(pages, data_root)
                messages = build_image_messages(str(row.get("question", "")), images)
                inputs = processor_inputs(processor, messages, images, device)
                if "base_image" in args.modes:
                    _option, text = generate_teacher_qwen(
                        model,
                        processor,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"],
                        mm_token_type_ids=inputs["mm_token_type_ids"],
                        max_new_tokens=args.max_new_tokens,
                    )
                    record["base_image"] = {
                        "prediction": text.strip(),
                        "em": best_metric_over_gold_answers(text, answers, exact_match_score),
                        "f1": best_metric_over_gold_answers(text, answers, f1_score),
                    }
                if "adapter_image" in args.modes:
                    if adapter is None:
                        raise RuntimeError("adapter was not loaded")
                    _option, text = generate_adapter_qwen(
                        model,
                        processor,
                        adapter,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"],
                        mm_token_type_ids=inputs["mm_token_type_ids"],
                        max_new_tokens=args.max_new_tokens,
                        early_stop_metric=None,
                    )
                    record["adapter_image"] = {
                        "prediction": text.strip(),
                        "em": best_metric_over_gold_answers(text, answers, exact_match_score),
                        "f1": best_metric_over_gold_answers(text, answers, f1_score),
                    }

            if "base_text" in args.modes:
                messages = build_text_messages(row)
                inputs = processor_inputs(processor, messages, None, device)
                text = generate_base_text(model, processor, inputs, args.max_new_tokens)
                record["base_text"] = {
                    "prediction": text.strip(),
                    "em": best_metric_over_gold_answers(text, answers, exact_match_score),
                    "f1": best_metric_over_gold_answers(text, answers, f1_score),
                }

            predictions.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if idx % args.log_every == 0 or idx == len(selected):
                interim = summarize(predictions, args.modes)
                print(json.dumps({"done": idx, "total": len(selected), **interim}, ensure_ascii=False), flush=True)

    summary = summarize(predictions, args.modes)
    summary.update(
        {
            "validation": str(validation_path),
            "render_metadata": str(metadata_path),
            "model_path": args.model_path,
            "checkpoint": args.checkpoint or None,
            "max_new_tokens": args.max_new_tokens,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "predictions": str(predictions_path),
        }
    )
    results_path = output_dir / f"results_shard{args.shard_id:05d}.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
