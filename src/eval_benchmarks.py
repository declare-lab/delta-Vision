"""Sharded benchmark evaluation for vision KV adapters."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "artifacts/torch_compile_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import torch

from src.benchmarks import (
    BENCHMARK_SPECS,
    canonical_benchmark_name,
    estimate_qwen_kv_cache_mb,
    estimate_qwen_prefill_flops,
    extract_choice,
    extract_yes_no,
    format_seconds_minsec,
    get_benchmark_spec,
    safe_mean,
    score_prediction,
    summarize_metric,
)
from src.model import (
    extract_vision_kv,
    load_frozen_llava,
    load_adapter_checkpoint,
    PerLayerKVAdapter,
    student_forward_llava_injection,
    student_forward_with_visual_kv,
    teacher_forward,
    dtype_from_name,
    build_qwen_initial_context,
    load_or_build_qwen_initial_context,
    load_frozen_qwen3vl,
    load_qwen_visual_delta_checkpoint,
    qwen_visual_delta_logits,
)
from src.data import MMStarDataset, QwenBenchmarkDataset, QwenMMStarDataset


OPTION_LETTERS = ["A", "B", "C", "D"]


def extract_option_from_text(text: str) -> str | None:
    """Extract an A/B/C/D answer from generated text."""
    clean = text.strip().upper()
    if not clean:
        return None

    patterns = [
        r"(?:ANSWER|OPTION|CHOICE|答案|选项)\s*(?:IS|是|:|：)?\s*[\(\[]?\s*([ABCD])(?:\b|[\)\]\.。,:：])",
        r"^[\s\(\[]*([ABCD])(?:[\)\]\.。,:：\s]|$)",
        r"(?<![A-Z])([ABCD])(?![A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    return None


def get_option_token_ids(tokenizer) -> dict[str, list[int]]:
    """Get single-token IDs for common option-letter renderings."""
    result = {}
    for letter in OPTION_LETTERS:
        ids = []
        for text in (letter, f" {letter}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
        if not ids:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            ids.append(encoded[-1])
        result[letter] = sorted(set(ids))
    return result


def predict_option(logits: torch.Tensor, option_ids: dict[str, list[int]]) -> str:
    """Predict the most likely option letter from logits."""
    best_letter = "A"
    best_score = float("-inf")
    for letter, ids in option_ids.items():
        score = max(logits[tid].item() for tid in ids)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def _eos_token_ids(tokenizer) -> set[int]:
    eos_ids = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        eos_ids.update(int(x) for x in eos_token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("<|im_end|>", "</s>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                eos_ids.add(token_id)
    return eos_ids


def _structured_answer_ready(metric: str | None, text: str, choices: list[Any] | None = None) -> bool:
    if metric == "multi_choice":
        return extract_choice(text, choices) is not None
    if metric in {"mme", "pope_f1"}:
        return extract_yes_no(text) is not None
    return False


def _next_token_logits(logits: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return logits[0, -1]
    return logits[0, int(text_mask[0].sum().item()) - 1]


@torch.inference_mode()
def generate_teacher_llava(
    model,
    processor,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    image_sizes=None,
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if image_sizes is not None:
        kwargs["image_sizes"] = image_sizes
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id

    generated = model.generate(**kwargs)
    new_tokens = generated[0, input_ids.shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def generate_adapter_llava(
    model,
    processor,
    adapter: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    source_k: torch.Tensor | None,
    source_v: torch.Tensor | None,
    image_token_id: int,
    attention_mask: torch.Tensor,
    output_mode: str = "adapter_only",
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    generated = []
    eos_ids = _eos_token_ids(processor.tokenizer)

    for _ in range(max_new_tokens):
        if output_mode == "native_visual_kv_injection":
            logits = student_forward_llava_injection(
                model,
                full_ids,
                pixel_values,
                adapter,
                image_token_id,
                attention_mask=full_mask,
            )
            next_logits = logits[0, -1]
        else:
            assert source_k is not None and source_v is not None
            logits = student_forward_with_visual_kv(
                model,
                full_ids,
                adapter,
                source_k,
                source_v,
                image_token_id,
                attention_mask=full_mask,
            )
            next_logits = logits[0, -1]
        next_token = int(torch.argmax(next_logits).item())
        generated.append(next_token)
        token_tensor = torch.tensor([[next_token]], dtype=full_ids.dtype, device=full_ids.device)
        full_ids = torch.cat([full_ids, token_tensor], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(token_tensor)], dim=1)
        if next_token in eos_ids:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return extract_option_from_text(text), text


def load_adapter(checkpoint_path: str, model, device: torch.device) -> tuple[torch.nn.Module, list[int], str]:
    """Load trained adapter from checkpoint."""
    adapter, source_layers, metadata = load_adapter_checkpoint(
        checkpoint_path,
        device=device,
        language_model=model.model.language_model,
        dtype=torch.bfloat16,
    )
    return adapter, source_layers, str(metadata.get("output_mode") or "adapter_only")


@torch.inference_mode()
def evaluate_llava_shard(
    model,
    processor,
    adapter: torch.nn.Module,
    dataset: MMStarDataset,
    device: torch.device,
    image_token_id: int,
    source_layers: list[int],
    log_every: int = 25,
    max_new_tokens: int = 8,
    eval_mode: str = "generate",
    output_mode: str = "adapter_only",
) -> dict:
    """Evaluate adapter on a shard of MMStar."""
    option_ids = get_option_token_ids(processor.tokenizer)

    stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "adapter_correct_when_teacher_correct": 0,
        "agree": 0,
        "teacher_invalid": 0,
        "adapter_invalid": 0,
    }
    predictions = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        image_sizes = item.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.unsqueeze(0).to(device) if torch.is_tensor(image_sizes) else image_sizes
        gold = item["gold"]

        if output_mode == "native_visual_kv_injection":
            source_k = source_v = None
        else:
            source_k, source_v = extract_vision_kv(model, pixel_values, source_layer_indices=source_layers)
        if eval_mode == "logits":
            teacher_logits = teacher_forward(model, input_ids, pixel_values, attention_mask, image_sizes=image_sizes)
            full_last_idx = int(attention_mask[0].sum().item()) - 1
            teacher_last = teacher_logits[0, full_last_idx]

            if output_mode == "native_visual_kv_injection":
                student_logits = student_forward_llava_injection(
                    model,
                    input_ids,
                    pixel_values,
                    adapter,
                    image_token_id,
                    attention_mask=attention_mask,
                )
                student_last = student_logits[0, -1]
            else:
                assert source_k is not None and source_v is not None
                student_logits = student_forward_with_visual_kv(
                    model, input_ids, adapter, source_k, source_v, image_token_id, attention_mask=attention_mask
                )
                student_last = student_logits[0, -1]

            teacher_pred = predict_option(teacher_last, option_ids)
            adapter_pred = predict_option(student_last, option_ids)
            teacher_text = ""
            adapter_text = ""
        elif eval_mode == "generate":
            teacher_pred, teacher_text = generate_teacher_llava(
                model,
                processor,
                input_ids,
                pixel_values,
                attention_mask,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
            )
            adapter_pred, adapter_text = generate_adapter_llava(
                model,
                processor,
                adapter,
                input_ids,
                pixel_values,
                source_k,
                source_v,
                image_token_id,
                attention_mask,
                output_mode=output_mode,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown eval_mode={eval_mode!r}")

        stats["scored"] += 1
        stats["teacher_correct"] += int(teacher_pred == gold)
        stats["adapter_correct"] += int(adapter_pred == gold)
        stats["agree"] += int(adapter_pred == teacher_pred)
        stats["teacher_invalid"] += int(teacher_pred is None)
        stats["adapter_invalid"] += int(adapter_pred is None)
        stats["adapter_correct_when_teacher_correct"] += int(teacher_pred == gold and adapter_pred == gold)

        predictions.append({
            "index": item["index"],
            "gold": gold,
            "teacher": teacher_pred,
            "adapter": adapter_pred,
            "teacher_text": teacher_text,
            "adapter_text": adapter_text,
        })

        if (idx + 1) % log_every == 0:
            acc = stats["adapter_correct"] / stats["scored"]
            print(f"[{idx+1}/{len(dataset)}] adapter_acc={acc:.4f}", flush=True)

    return {"stats": stats, "predictions": predictions, "output_mode": output_mode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified benchmark evaluation for LLaVA KV adapters and Qwen3-VL visual-delta adapters.")
    parser.add_argument("--model-kind", choices=("llava", "qwen"), default="llava")
    parser.add_argument("--benchmark", default="mmstar", help=f"Benchmark name. Choices: {', '.join(sorted(BENCHMARK_SPECS))}")
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving image paths")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None, help="If set, only run this shard")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--eval-mode", choices=("generate", "logits"), default="generate")
    parser.add_argument("--output-mode", choices=("adapter_only", "native_visual_kv_injection"), default=None)
    parser.add_argument("--answer-instruction", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--measure-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured-answer-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-cache", action=argparse.BooleanOptionalAction, default=True, help="Cache deterministic Qwen teacher generations across adapter checkpoints.")
    parser.add_argument("--teacher-cache-dir", default=None)
    parser.add_argument("--require-teacher-cache", action="store_true", help="Fail instead of generating teacher outputs when the teacher cache is missing or stale.")
    parser.add_argument("--last-logits-only", action=argparse.BooleanOptionalAction, default=True, help="Only compute logits for the next-token position during generation/eval prefill.")
    parser.add_argument("--input-cache-dir", default=None, help="Optional cache directory for preprocessed Qwen benchmark tensors.")
    parser.add_argument("--context-cache-dir", default=None, help="Optional cache directory for Qwen initial_hidden/position_ids tensors.")
    args = parser.parse_args()
    args.benchmark = canonical_benchmark_name(args.benchmark)
    return args


def run_llava_single_shard(args, shard_id: int, num_shards: int):
    """Run evaluation on a single shard (one GPU)."""
    device = torch.device("cuda:0")
    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device="cuda:0")
    adapter, source_layers, checkpoint_mode = load_adapter(args.checkpoint, model, device)
    output_mode = args.output_mode or checkpoint_mode
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    full_dataset = MMStarDataset(
        args.data,
        processor,
        data_root=args.data_root,
        max_samples=args.max_samples,
        answer_instruction=args.answer_instruction or get_benchmark_spec("mmstar").answer_instruction,
    )
    total = len(full_dataset)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)

    full_dataset.rows = full_dataset.rows[start:end]
    print(f"Shard {shard_id}: samples [{start}, {end}) = {len(full_dataset)} items; output_mode={output_mode}", flush=True)

    result = evaluate_llava_shard(
        model,
        processor,
        adapter,
        full_dataset,
        device,
        image_token_id,
        source_layers,
        args.log_every,
        args.max_new_tokens,
        args.eval_mode,
        output_mode,
    )

    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
    return result



@torch.inference_mode()
def generate_teacher_qwen(
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str | None, str]:
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    eos_ids = sorted(_eos_token_ids(processor.tokenizer))
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    generated = model.generate(**kwargs)
    text = processor.tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
    return extract_option_from_text(text), text


@torch.inference_mode()
def generate_adapter_qwen(
    model,
    processor,
    adapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    max_new_tokens: int,
    adapter_logits_fn=None,
    initial_hidden: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    prefill_logits: torch.Tensor | None = None,
    prefill_text_mask: torch.Tensor | None = None,
    last_logits_only: bool = True,
    early_stop_metric: str | None = None,
    choices: list[Any] | None = None,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    full_mm_ids = mm_token_type_ids.clone()
    generated: list[int] = []
    eos_ids = _eos_token_ids(processor.tokenizer)
    token_embeddings = model.model.get_input_embeddings()
    inputs = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": full_mm_ids,
    }
    if initial_hidden is None or position_ids is None:
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    last_pos_idx = full_mask.long().sum(dim=1).sub(1).view(1, -1, 1).expand(position_ids.shape[0], -1, 1)
    token_position_ids = position_ids.gather(2, last_pos_idx)
    text = ""
    for _ in range(max_new_tokens):
        inputs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": full_mm_ids,
        }
        if prefill_logits is not None and prefill_text_mask is not None:
            logits = prefill_logits
            text_mask = prefill_text_mask
            prefill_logits = None
            prefill_text_mask = None
        elif adapter_logits_fn is None:
            logits, text_mask, _ = qwen_visual_delta_logits(
                model,
                adapter,
                inputs,
                initial_hidden=initial_hidden,
                position_ids=position_ids,
                collect_states=False,
                compact_no_padding=True,
                logits_to_keep=1 if last_logits_only else 0,
            )
        else:
            logits, text_mask = adapter_logits_fn(inputs, initial_hidden, position_ids)
        next_token = int(torch.argmax(_next_token_logits(logits, text_mask)).item())
        generated.append(next_token)
        token = torch.tensor([[next_token]], dtype=full_ids.dtype, device=full_ids.device)
        full_ids = torch.cat([full_ids, token], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(token)], dim=1)
        full_mm_ids = torch.cat([full_mm_ids, torch.zeros_like(token)], dim=1)
        initial_hidden = torch.cat(
            [initial_hidden, token_embeddings(token).to(device=initial_hidden.device, dtype=initial_hidden.dtype)],
            dim=1,
        )
        token_position_ids = token_position_ids + 1
        position_ids = torch.cat([position_ids, token_position_ids], dim=2)
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if next_token in eos_ids or _structured_answer_ready(early_stop_metric, text, choices):
            break
    return extract_option_from_text(text), text


@torch.inference_mode()
def evaluate_qwen_shard(
    model,
    processor,
    adapter,
    dataset: QwenMMStarDataset,
    device: torch.device,
    log_every: int,
    max_new_tokens: int,
) -> dict:
    stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "adapter_correct_when_teacher_correct": 0,
        "agree": 0,
        "teacher_invalid": 0,
        "adapter_invalid": 0,
    }
    predictions = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].to(device)
        image_grid_thw = item["image_grid_thw"].to(device)
        mm_token_type_ids = item["mm_token_type_ids"].unsqueeze(0).to(device)
        gold = item["gold"]

        teacher_pred, teacher_text = generate_teacher_qwen(
            model,
            processor,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            max_new_tokens,
            early_stop_metric="multi_choice",
        )
        adapter_pred, adapter_text = generate_adapter_qwen(
            model,
            processor,
            adapter,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            max_new_tokens,
        )

        stats["scored"] += 1
        stats["teacher_correct"] += int(teacher_pred == gold)
        stats["adapter_correct"] += int(adapter_pred == gold)
        stats["adapter_correct_when_teacher_correct"] += int(teacher_pred == gold and adapter_pred == gold)
        stats["agree"] += int(teacher_pred == adapter_pred)
        stats["teacher_invalid"] += int(teacher_pred is None)
        stats["adapter_invalid"] += int(adapter_pred is None)
        predictions.append(
            {
                "index": item["index"],
                "gold": gold,
                "teacher": teacher_pred,
                "adapter": adapter_pred,
                "teacher_text": teacher_text,
                "adapter_text": adapter_text,
            }
        )
        if (idx + 1) % log_every == 0:
            n = max(stats["scored"], 1)
            print(
                f"[{idx+1}/{len(dataset)}] teacher={stats['teacher_correct']/n:.4f} "
                f"adapter={stats['adapter_correct']/n:.4f} agreement={stats['agree']/n:.4f}",
                flush=True,
            )
    return {"stats": stats, "predictions": predictions, "output_mode": adapter.mode}


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def _timed_call(fn, enabled: bool = True):
    if not enabled:
        return 0.0, None
    _sync_cuda()
    start = time.perf_counter()
    value = fn()
    _sync_cuda()
    return time.perf_counter() - start, value


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def build_qwen_adapter_logits_fn(
    model,
    adapter,
    *,
    compile_adapter: bool,
    compile_mode: str,
    compile_dynamic: bool,
    last_logits_only: bool,
):
    logits_to_keep = 1 if last_logits_only else 0

    def direct(inputs, initial_hidden=None, position_ids=None):
        logits, text_mask, _ = qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            collect_states=False,
            compact_no_padding=True,
            logits_to_keep=logits_to_keep,
        )
        return logits, text_mask

    if not compile_adapter:
        return direct

    def cached_forward(
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids,
        initial_hidden,
        position_ids,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        logits, text_mask, _ = qwen_visual_delta_logits(
            model,
            adapter,
            inputs,
            initial_hidden=initial_hidden,
            position_ids=position_ids,
            collect_states=False,
            compact_no_padding=True,
            logits_to_keep=logits_to_keep,
        )
        return logits, text_mask

    compiled_cached = torch.compile(cached_forward, mode=compile_mode, dynamic=compile_dynamic)

    def compiled(inputs, initial_hidden=None, position_ids=None):
        if initial_hidden is None or position_ids is None:
            initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
        return compiled_cached(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
        )

    return compiled


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.bfloat16, torch.int16}:
        return 2
    return 1


def _qwen_token_counts(attention_mask: torch.Tensor, mm_token_type_ids: torch.Tensor) -> tuple[int, int]:
    valid = attention_mask.bool()
    visual = int(((mm_token_type_ids == 1) & valid).sum().item())
    text = int(((mm_token_type_ids == 0) & valid).sum().item())
    return text, visual


def _visual_adapter_rank(adapter) -> int:
    rank = int(getattr(adapter, "visual_adapter_rank", 0))
    if rank > 0:
        return rank
    down = getattr(adapter, "visual_adapter_down", None)
    if down:
        return int(down[0].out_features)
    return 0


def _teacher_cache_row_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": row.get("index", idx),
            "image": row.get("image"),
            "images": row.get("images"),
            "question": row.get("question"),
            "answer": row.get("answer"),
        }
        for idx, row in enumerate(rows)
    ]


def _load_teacher_cache(path: Path | None, expected_meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Ignoring unreadable teacher cache {path}: {exc}", flush=True)
        return None
    cached_meta = payload.get("meta")
    output_keys = (
        "benchmark",
        "data",
        "data_root",
        "model_path",
        "max_samples",
        "max_new_tokens",
        "answer_instruction",
        "num_shards",
        "shard_id",
        "start",
        "end",
        "rows",
    )
    if not isinstance(cached_meta, dict) or any(cached_meta.get(key) != expected_meta.get(key) for key in output_keys):
        print(f"Ignoring stale teacher cache {path}", flush=True)
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_meta.get("rows", [])):
        print(f"Ignoring incomplete teacher cache {path}", flush=True)
        return None
    print(f"Loaded teacher cache {path}", flush=True)
    return entries


def _save_teacher_cache(path: Path | None, meta: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps({"meta": meta, "entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)
    print(f"Saved teacher cache {path}", flush=True)


def summarize_qwen_benchmark_predictions(
    *,
    benchmark: str,
    predictions: list[dict[str, Any]],
    output_modes: list[str] | None = None,
) -> dict[str, Any]:
    spec = get_benchmark_spec(benchmark)
    rows = [item.get("row", {}) for item in predictions]
    teacher_items = [item["teacher_eval"] for item in predictions]
    adapter_items = [item["adapter_eval"] for item in predictions if item.get("adapter_eval") is not None]
    teacher = summarize_metric(spec.metric, teacher_items, rows)
    adapter = summarize_metric(spec.metric, adapter_items, rows[: len(adapter_items)]) if adapter_items else None

    teacher_correct = sum(1 for item in teacher_items if float(item.get("score", 0.0)) > 0.0)
    adapter_correct = sum(1 for item in adapter_items if float(item.get("score", 0.0)) > 0.0)
    both_correct = sum(
        1
        for item in predictions
        if item.get("adapter_eval") is not None
        and float(item["teacher_eval"].get("score", 0.0)) > 0.0
        and float(item["adapter_eval"].get("score", 0.0)) > 0.0
    )
    agree = sum(
        1
        for item in predictions
        if item.get("adapter_eval") is not None
        and item["teacher_eval"].get("prediction") == item["adapter_eval"].get("prediction")
    )
    adapter_total = len(adapter_items)

    teacher_total_s = sum(float(item.get("teacher_total_s", 0.0)) for item in predictions)
    teacher_prefill_s = sum(float(item.get("teacher_prefill_s", 0.0)) for item in predictions)
    adapter_total_s = sum(float(item.get("adapter_total_s", 0.0)) for item in predictions)
    adapter_prefill_s = sum(float(item.get("adapter_prefill_s", 0.0)) for item in predictions)

    timing = {
        "teacher_total_s": teacher_total_s,
        "teacher_total_minsec": format_seconds_minsec(teacher_total_s),
        "teacher_prefill_s": teacher_prefill_s,
        "teacher_prefill_minsec": format_seconds_minsec(teacher_prefill_s),
        "adapter_total_s": adapter_total_s,
        "adapter_total_minsec": format_seconds_minsec(adapter_total_s),
        "adapter_prefill_s": adapter_prefill_s,
        "adapter_prefill_minsec": format_seconds_minsec(adapter_prefill_s),
        "speedup_total": (teacher_total_s / adapter_total_s) if adapter_total_s > 0 else None,
        "speedup_prefill": (teacher_prefill_s / adapter_prefill_s) if adapter_prefill_s > 0 else None,
    }
    resources = {
        "teacher_kv_cache_mb_avg": safe_mean([item.get("teacher_kv_cache_mb", 0.0) for item in predictions]),
        "adapter_kv_cache_mb_avg": safe_mean([item.get("adapter_kv_cache_mb", 0.0) for item in predictions]),
        "teacher_prefill_flops_avg": safe_mean([item.get("teacher_prefill_flops", 0.0) for item in predictions]),
        "adapter_prefill_flops_avg": safe_mean([item.get("adapter_prefill_flops", 0.0) for item in predictions]),
    }

    summary: dict[str, Any] = {
        "benchmark": spec.name,
        "display_name": spec.display_name,
        "metric": spec.metric,
        "total_samples": len(predictions),
        "teacher": teacher,
        "adapter": adapter,
        "agreement": (agree / max(adapter_total, 1)) if adapter_total else None,
        "retention": (both_correct / teacher_correct) if teacher_correct else None,
        "teacher_correct": teacher_correct,
        "adapter_correct": adapter_correct,
        "timing": timing,
        "resources": resources,
        "total_time_minsec": timing["adapter_total_minsec"],
        "prefilling_time_minsec": timing["adapter_prefill_minsec"],
        "flops": resources["adapter_prefill_flops_avg"],
        "kv_cache_mb": resources["adapter_kv_cache_mb_avg"],
        "speedup_total": timing["speedup_total"],
        "speedup_prefilling": timing["speedup_prefill"],
    }
    if output_modes:
        summary["output_modes"] = sorted(set(output_modes))
    if spec.metric == "pope_f1" and adapter:
        summary["pope_f1"] = adapter.get("f1", 0.0)
    return summary


@torch.inference_mode()
def evaluate_qwen_benchmark_shard(
    model,
    processor,
    adapter,
    dataset: QwenBenchmarkDataset,
    device: torch.device,
    log_every: int,
    max_new_tokens: int,
    benchmark: str,
    measure_prefill: bool,
    dtype: torch.dtype,
    adapter_logits_fn=None,
    compile_warmup: bool = False,
    context_cache_dir: str | None = None,
    teacher_cache_path: Path | None = None,
    teacher_cache_meta: dict[str, Any] | None = None,
    require_teacher_cache: bool = False,
    structured_answer_early_stop: bool = True,
    last_logits_only: bool = True,
) -> dict:
    spec = get_benchmark_spec(benchmark)
    language_config = model.model.language_model.config
    predictions: list[dict[str, Any]] = []
    teacher_cache_entries = _load_teacher_cache(teacher_cache_path, teacher_cache_meta) if teacher_cache_meta else None
    if require_teacher_cache and teacher_cache_entries is None:
        raise FileNotFoundError(f"required teacher cache is missing or stale: {teacher_cache_path}")
    teacher_cache_to_write: list[dict[str, Any]] = []
    if compile_warmup and adapter_logits_fn is not None and len(dataset) > 0:
        item = dataset[0]
        warm_inputs = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": item["pixel_values"].to(device),
            "image_grid_thw": item["image_grid_thw"].to(device),
            "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0).to(device),
        }
        initial_hidden, position_ids = build_qwen_initial_context(model, warm_inputs)
        adapter_logits_fn(warm_inputs, initial_hidden, position_ids)
        _sync_cuda()
    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].to(device)
        image_grid_thw = item["image_grid_thw"].to(device)
        mm_token_type_ids = item["mm_token_type_ids"].unsqueeze(0).to(device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

        text_tokens, image_tokens = _qwen_token_counts(attention_mask, mm_token_type_ids)
        cached_teacher = teacher_cache_entries[idx] if teacher_cache_entries is not None else None
        if cached_teacher is None:
            if require_teacher_cache:
                raise FileNotFoundError(f"required teacher cache entry {idx} is missing: {teacher_cache_path}")
            teacher_prefill_s, _ = _timed_call(
                lambda: model(**inputs, logits_to_keep=1 if last_logits_only else 0).logits,
                enabled=measure_prefill,
            )
            teacher_total_s, (_, teacher_text) = _timed_call(
                lambda: generate_teacher_qwen(
                    model,
                    processor,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    mm_token_type_ids,
                    max_new_tokens,
                )
            )
            teacher_cache_to_write.append(
                {
                    "teacher_text": teacher_text,
                    "teacher_prefill_s": teacher_prefill_s,
                    "teacher_total_s": teacher_total_s,
                }
            )
        else:
            teacher_text = str(cached_teacher.get("teacher_text", ""))
            teacher_prefill_s = float(cached_teacher.get("teacher_prefill_s", 0.0))
            teacher_total_s = float(cached_teacher.get("teacher_total_s", 0.0))

        def adapter_prefill():
            initial_hidden, position_ids = load_or_build_qwen_initial_context(
                model,
                inputs,
                cache_dir=context_cache_dir,
                dtype=dtype,
            )
            if adapter_logits_fn is None:
                logits, text_mask, _ = qwen_visual_delta_logits(
                    model,
                    adapter,
                    dict(inputs),
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    collect_states=False,
                    compact_no_padding=True,
                    logits_to_keep=1 if last_logits_only else 0,
                )
            else:
                logits, text_mask = adapter_logits_fn(dict(inputs), initial_hidden, position_ids)
            return logits, text_mask, initial_hidden, position_ids

        adapter_prefill_s, adapter_prefill_payload = _timed_call(
            adapter_prefill,
            enabled=measure_prefill,
        )
        if adapter_prefill_payload is None:
            adapter_total_s, (_, adapter_text) = _timed_call(
                lambda: generate_adapter_qwen(
                    model,
                    processor,
                    adapter,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    mm_token_type_ids,
                    max_new_tokens,
                    adapter_logits_fn=adapter_logits_fn,
                    last_logits_only=last_logits_only,
                    early_stop_metric=spec.metric if structured_answer_early_stop else None,
                    choices=item.get("choices"),
                )
            )
        else:
            prefill_logits, prefill_text_mask, initial_hidden, position_ids = adapter_prefill_payload
            adapter_continuation_s, (_, adapter_text) = _timed_call(
                lambda: generate_adapter_qwen(
                    model,
                    processor,
                    adapter,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    mm_token_type_ids,
                    max_new_tokens,
                    adapter_logits_fn=adapter_logits_fn,
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    prefill_logits=prefill_logits,
                    prefill_text_mask=prefill_text_mask,
                    last_logits_only=last_logits_only,
                    early_stop_metric=spec.metric if structured_answer_early_stop else None,
                    choices=item.get("choices"),
                )
            )
            adapter_total_s = adapter_prefill_s + adapter_continuation_s

        teacher_eval = score_prediction(
            metric=spec.metric,
            prediction_text=teacher_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=item.get("choices"),
        )
        adapter_eval = score_prediction(
            metric=spec.metric,
            prediction_text=adapter_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=item.get("choices"),
        )
        teacher_kv = estimate_qwen_kv_cache_mb(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            dtype_bytes=_dtype_bytes(dtype),
            adapter=False,
        )
        adapter_kv = estimate_qwen_kv_cache_mb(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            dtype_bytes=_dtype_bytes(dtype),
            adapter=True,
        )
        teacher_flops = estimate_qwen_prefill_flops(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
        )
        adapter_flops = estimate_qwen_prefill_flops(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            adapter_mode=str(getattr(adapter, "mode", "")),
            visual_adapter_rank=_visual_adapter_rank(adapter),
        )

        predictions.append(
            {
                "index": item["index"],
                "row": item["row"],
                "teacher_text": teacher_text,
                "adapter_text": adapter_text,
                "teacher_eval": teacher_eval,
                "adapter_eval": adapter_eval,
                "teacher_total_s": teacher_total_s,
                "adapter_total_s": adapter_total_s,
                "teacher_prefill_s": teacher_prefill_s,
                "adapter_prefill_s": adapter_prefill_s,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "teacher_kv_cache_mb": teacher_kv,
                "adapter_kv_cache_mb": adapter_kv,
                "teacher_prefill_flops": teacher_flops,
                "adapter_prefill_flops": adapter_flops,
            }
        )
        if (idx + 1) % log_every == 0:
            summary = summarize_qwen_benchmark_predictions(
                benchmark=benchmark,
                predictions=predictions,
                output_modes=[adapter.mode],
            )
            teacher_score = summary["teacher"]["score"]
            adapter_score = summary["adapter"]["score"] if summary["adapter"] else 0.0
            print(
                f"[{idx+1}/{len(dataset)}] {spec.display_name} teacher={teacher_score:.4f} "
                f"adapter={adapter_score:.4f}",
                flush=True,
            )

    summary = summarize_qwen_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=[adapter.mode])
    if teacher_cache_entries is None and teacher_cache_meta is not None and len(teacher_cache_to_write) == len(dataset):
        _save_teacher_cache(teacher_cache_path, teacher_cache_meta, teacher_cache_to_write)
    return {"summary": summary, "predictions": predictions, "output_mode": adapter.mode}


def run_qwen_single_shard(args: argparse.Namespace, shard_id: int, num_shards: int) -> dict:
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    configure_torch_runtime()
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter, meta = load_qwen_visual_delta_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    if meta["missing"] or meta["unexpected"]:
        print(f"checkpoint load missing={meta['missing']} unexpected={meta['unexpected']}", flush=True)
    adapter_logits_fn = build_qwen_adapter_logits_fn(
        model,
        adapter,
        compile_adapter=bool(args.compile_adapter),
        compile_mode=args.compile_mode,
        compile_dynamic=bool(args.compile_dynamic),
        last_logits_only=bool(args.last_logits_only),
    )

    dataset = QwenBenchmarkDataset(
        args.data,
        processor,
        args.benchmark,
        data_root=args.data_root,
        max_samples=args.max_samples,
        answer_instruction=args.answer_instruction,
        cache_dir=args.input_cache_dir,
    )
    total = len(dataset)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)
    dataset.rows = dataset.rows[start:end]
    print(f"Shard {shard_id}: samples [{start}, {end}) = {len(dataset)} items; mode={adapter.mode}", flush=True)
    teacher_cache_path = None
    teacher_cache_meta = None
    if args.teacher_cache and args.teacher_cache_dir:
        teacher_cache_path = (
            Path(args.teacher_cache_dir)
            / args.benchmark
            / f"{Path(args.data).name}.max{args.max_samples or 'all'}.new{args.max_new_tokens}.shard{shard_id}of{num_shards}.json"
        )
        teacher_cache_meta = {
            "benchmark": args.benchmark,
            "data": str(Path(args.data)),
            "data_root": str(Path(args.data_root)),
            "model_path": str(Path(args.model_path)),
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "measure_prefill": bool(args.measure_prefill),
            "last_logits_only": bool(args.last_logits_only),
            "answer_instruction": args.answer_instruction,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "start": start,
            "end": end,
            "rows": _teacher_cache_row_keys(dataset.rows),
        }
    if args.require_teacher_cache and teacher_cache_meta is None:
        raise ValueError("--require-teacher-cache requires --teacher-cache-dir")
    result = evaluate_qwen_benchmark_shard(
        model,
        processor,
        adapter,
        dataset,
        device,
        args.log_every,
        args.max_new_tokens,
        args.benchmark,
        args.measure_prefill,
        dtype,
        adapter_logits_fn=adapter_logits_fn,
        compile_warmup=bool(args.compile_adapter and args.compile_warmup),
        context_cache_dir=args.context_cache_dir,
        teacher_cache_path=teacher_cache_path,
        teacher_cache_meta=teacher_cache_meta,
        require_teacher_cache=bool(args.require_teacher_cache),
        structured_answer_early_stop=bool(args.structured_answer_early_stop),
        last_logits_only=bool(args.last_logits_only),
    )
    out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
    return result



def merge_shards(output_dir: str, num_shards: int) -> dict:
    all_stats = {
        "scored": 0,
        "teacher_correct": 0,
        "adapter_correct": 0,
        "adapter_correct_when_teacher_correct": 0,
        "agree": 0,
        "teacher_invalid": 0,
        "adapter_invalid": 0,
    }
    all_predictions = []
    output_modes = set()

    for shard_id in range(num_shards):
        path = Path(output_dir) / f"shard_{shard_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard result: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in all_stats:
            all_stats[key] += int(data["stats"].get(key, 0))
        all_predictions.extend(data.get("predictions", []))
        if data.get("output_mode"):
            output_modes.add(data["output_mode"])

    n = max(all_stats["scored"], 1)
    tc = all_stats["teacher_correct"]
    merged = {
        "total_samples": all_stats["scored"],
        "teacher_accuracy": all_stats["teacher_correct"] / n,
        "adapter_accuracy": all_stats["adapter_correct"] / n,
        "agreement": all_stats["agree"] / n,
        "retention": (all_stats["adapter_correct_when_teacher_correct"] / tc) if tc else 0.0,
        "teacher_invalid_rate": all_stats["teacher_invalid"] / n,
        "adapter_invalid_rate": all_stats["adapter_invalid"] / n,
    }
    if output_modes:
        merged["output_modes"] = sorted(output_modes)

    out_dir = Path(output_dir)
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(all_predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2), flush=True)
    return merged


def merge_benchmark_shards(output_dir: str, num_shards: int, benchmark: str) -> dict:
    all_predictions: list[dict[str, Any]] = []
    output_modes: list[str] = []
    for shard_id in range(num_shards):
        path = Path(output_dir) / f"shard_{shard_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        all_predictions.extend(data.get("predictions", []))
        if data.get("output_mode"):
            output_modes.append(str(data["output_mode"]))

    merged = summarize_qwen_benchmark_predictions(
        benchmark=benchmark,
        predictions=all_predictions,
        output_modes=output_modes,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(all_predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "Benchmark",
            "Metric",
            "Teacher Score",
            "Adapter Score",
            "Total Time (Min:Sec)",
            "Prefilling Time (Min:Sec)",
            "FLOPs",
            "KV Cache (MB)",
            "POPE F1",
            "Speedup Total",
            "Speedup Prefilling",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        adapter = merged.get("adapter") or {}
        teacher = merged.get("teacher") or {}
        writer.writerow(
            {
                "Benchmark": merged.get("display_name", benchmark),
                "Metric": merged.get("metric", ""),
                "Teacher Score": teacher.get("score", ""),
                "Adapter Score": adapter.get("score", ""),
                "Total Time (Min:Sec)": merged.get("total_time_minsec", ""),
                "Prefilling Time (Min:Sec)": merged.get("prefilling_time_minsec", ""),
                "FLOPs": merged.get("flops", ""),
                "KV Cache (MB)": merged.get("kv_cache_mb", ""),
                "POPE F1": merged.get("pope_f1", ""),
                "Speedup Total": merged.get("speedup_total", ""),
                "Speedup Prefilling": merged.get("speedup_prefilling", ""),
            }
        )
    print(json.dumps(merged, indent=2), flush=True)
    return merged


def main() -> None:
    args = parse_args()
    if args.shard_id is not None:
        if args.model_kind == "qwen":
            run_qwen_single_shard(args, args.shard_id, args.num_shards)
        else:
            run_llava_single_shard(args, args.shard_id, args.num_shards)
    else:
        if args.model_kind == "qwen":
            merge_benchmark_shards(args.output_dir, args.num_shards, args.benchmark)
        elif args.benchmark == "mmstar":
            merge_shards(args.output_dir, args.num_shards)
        else:
            raise ValueError(f"Generic benchmark merge is only implemented for Qwen, got model_kind={args.model_kind}")


if __name__ == "__main__":
    main()
