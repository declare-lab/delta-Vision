"""Sharded benchmark evaluation for vision KV adapters."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

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
    score_prediction,
    summarize_metric,
)
from src.model import (
    extract_vision_kv,
    llava_kv_adapter_decode_step_shape_exact,
    llava_kv_adapter_prefill_cache,
    llava_embedding_adapter_decode_step_shape_exact,
    llava_embedding_adapter_prefill_cache,
    PerLayerKVAdapter,
    student_forward_llava_embedding_adapter,
    student_forward_with_visual_kv,
    teacher_forward,
    canonical_adapter_mode,
    dtype_from_name,
    is_embedding_adapter_mode,
    build_qwen_initial_context,
    load_or_build_qwen_initial_context,
    prepare_qwen_embedding_adapter_inputs,
    qwen_input_device,
    qwen_embedding_adapter_decode_step,
    qwen_embedding_adapter_decode_step_shape_exact,
    qwen_embedding_adapter_prefill_cache,
    qwen_embedding_adapter_logits,
    qwen_embedding_adapter_logits_from_tensors,
    qwen_embedding_adapter_logits_prepared,
)
from src.model_setup import load_frozen_llava, load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint, load_adapter_checkpoint
from src.data import LlavaBenchmarkDataset, QwenBenchmarkDataset


OPTION_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

PERCEPTIONBENCH_JUDGE_TEMPLATE = """Please act as a professional teacher and grade the student's answer. Below are the question, the student's answer, and the reference answer. Based on the question and the reference answer, analyze the student's answer and judge whether it correctly answers the question.

Return your judgment in the following format. The first field [reason] gives the reason for your judgment; the second field [judge] gives your verdict as a single boolean, i.e., True or False. Make sure your output ends with True or False.
[reason]
<a brief justification of no more than 100 tokens>
[judge]
False

Notice:
0. Compare the reference answer and student's answer, focusing especially on the content after summarizing phrases such as "Final answer:".
1. If the question contains multiple sub-questions, judge correct only when all sub-questions are consistent.
2. For multiple-answer questions, the student's answer must contain all correct answers without extra ones.
3. If equivalent after simplification, provide the simplification process.
4. For numerical answers, integers or values with at most 4 significant figures must match exactly; values with more than 4 significant figures must match within 4 significant figures. Convert units when needed.
5. For English writing questions, if the student does not answer in English, judge it incorrect.
6. For physics, chemistry, and biology questions, if the reference answer contains a technical term, the student's answer must contain that exact term; synonyms are not accepted.
7. When the question asks to explain a term, redundant explanation is not penalized, but missing key points are incorrect.
8. For multiple-choice questions, judge incorrect whenever the selected option differs from the reference answer.

Now you may begin grading.
========== [Question] ==========
{problem}
========== [Student Answer] ==========
{assistant_answer}
========== [Reference Answer] ==========
{reference_answer}
========== [Your Judgment] ==========
"""


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


def get_option_token_ids(tokenizer, num_options: int = 4) -> dict[str, list[int]]:
    """Get single-token IDs for common option-letter renderings."""
    result = {}
    for letter in OPTION_LETTERS[: max(1, min(int(num_options), len(OPTION_LETTERS)))]:
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
    best_letter = next(iter(option_ids), "A")
    best_score = float("-inf")
    for letter, ids in option_ids.items():
        score = max(logits[tid].item() for tid in ids)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def _openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv("LLM_JUDGE_TIMEOUT", "900"))) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"judge request failed after {max_retries + 1} attempts: {last_error}") from last_error


def _decode_llm_judge(text: str) -> tuple[bool, str]:
    clean = str(text).strip()
    reason = clean
    if "[reason]" in clean and "[judge]" in clean:
        reason = clean.split("[judge]")[0].split("[reason]")[-1].strip()
        verdict = clean.split("[judge]")[-1].strip()
    else:
        verdict = clean
    return ("true" in verdict.lower()), reason[:500]


def maybe_llm_judge_eval(
    *,
    metric: str,
    eval_item: dict[str, Any],
    row: dict[str, Any],
    prediction_text: str,
    answer: Any,
    force: bool = False,
    base_url_override: str | None = None,
) -> dict[str, Any]:
    if metric != "llm_judge" or not eval_item.get("needs_judge"):
        return eval_item
    if not force and os.getenv("LLM_JUDGE_DURING_GENERATION", "0") != "1":
        return eval_item
    base_url = base_url_override or os.getenv("LLM_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    judge_model = os.getenv("LLM_JUDGE_MODEL") or os.getenv("JUDGE_MODEL")
    api_key = os.getenv("LLM_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    if not base_url or not judge_model:
        eval_item = dict(eval_item)
        eval_item.update({"score": 0.0, "invalid": True, "judge_error": "LLM_JUDGE_BASE_URL and LLM_JUDGE_MODEL are required"})
        return eval_item
    question = str(row.get("problem") or row.get("question") or "")
    reference = str(answer if answer is not None else row.get("answer", "")).strip()
    prompt = PERCEPTIONBENCH_JUDGE_TEMPLATE.format(
        problem=question,
        assistant_answer=str(prediction_text).strip(),
        reference_answer=reference,
    )
    try:
        response = _openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_JUDGE_MAX_TOKENS", "512")),
            max_retries=int(os.getenv("LLM_JUDGE_MAX_RETRIES", "3")),
        )
        ok, reason = _decode_llm_judge(response)
        eval_item = dict(eval_item)
        eval_item.update({"score": float(ok), "invalid": False, "judge_reason": reason, "judge_model": judge_model})
        return eval_item
    except Exception as exc:
        eval_item = dict(eval_item)
        eval_item.update({"score": 0.0, "invalid": True, "judge_error": str(exc), "judge_model": judge_model})
        return eval_item


def apply_deferred_llm_judge(
    *,
    benchmark: str,
    predictions: list[dict[str, Any]],
    log_every: int = 25,
) -> None:
    spec = get_benchmark_spec(benchmark)
    if spec.metric != "llm_judge":
        return

    pending: list[tuple[dict[str, Any], str, str]] = []
    for item in predictions:
        teacher_eval = item.get("teacher_eval") or {}
        adapter_eval = item.get("adapter_eval") or {}
        if teacher_eval.get("needs_judge"):
            pending.append((item, "teacher_eval", "teacher_text"))
        if adapter_eval.get("needs_judge"):
            pending.append((item, "adapter_eval", "adapter_text"))

    total = len(pending)
    if not total:
        return

    base_urls_text = os.getenv("LLM_JUDGE_BASE_URLS", "")
    base_urls = [url.strip() for url in base_urls_text.split(",") if url.strip()]
    if not base_urls:
        single_url = os.getenv("LLM_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        base_urls = [single_url] if single_url else []
    workers = int(os.getenv("LLM_JUDGE_WORKERS", str(max(1, len(base_urls)))))
    workers = max(1, workers)
    print(f"Running deferred LLM judge for {total} {spec.display_name} predictions with {workers} workers", flush=True)

    def judge_one(idx_and_pending: tuple[int, tuple[dict[str, Any], str, str]]) -> tuple[dict[str, Any], str, dict[str, Any]]:
        idx, (item, eval_key, text_key) = idx_and_pending
        base_url = base_urls[(idx - 1) % len(base_urls)] if base_urls else None
        result = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=item[eval_key],
            row=item.get("row", {}),
            prediction_text=str(item.get(text_key) or ""),
            answer=item.get("row", {}).get("answer") or item[eval_key].get("gold"),
            force=True,
            base_url_override=base_url,
        )
        return item, eval_key, result

    completed = 0
    if workers == 1:
        for idx_and_pending in enumerate(pending, start=1):
            item, eval_key, result = judge_one(idx_and_pending)
            item[eval_key] = result
            completed += 1
            if completed % max(1, int(log_every)) == 0 or completed == total:
                print(f"[judge {completed}/{total}] {spec.display_name}", flush=True)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(judge_one, item) for item in enumerate(pending, start=1)]
        for future in concurrent.futures.as_completed(futures):
            item, eval_key, result = future.result()
            item[eval_key] = result
            completed += 1
            if completed % max(1, int(log_every)) == 0 or completed == total:
                print(f"[judge {completed}/{total}] {spec.display_name}", flush=True)


def _prediction_from_text(metric: str | None, text: str, choices: list[Any] | None = None) -> str | None:
    if metric == "multi_choice" or metric is None:
        return extract_choice(text, choices)
    if metric in {"mme", "pope_f1"}:
        return extract_yes_no(text)
    return None


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


def _next_token_logits_batch(logits: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return logits[:, -1]
    last_idx = text_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, last_idx]


def _pad_qwen_1d(tensors: list[torch.Tensor], *, pad_value: int, device: torch.device) -> torch.Tensor:
    max_len = max(int(t.numel()) for t in tensors)
    output = torch.full((len(tensors), max_len), int(pad_value), dtype=tensors[0].dtype, device=device)
    for idx, tensor in enumerate(tensors):
        tensor = tensor.to(device)
        output[idx, : tensor.numel()] = tensor
    return output


def stack_qwen_benchmark_items(
    items: list[dict[str, Any]],
    *,
    device: torch.device,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": _pad_qwen_1d([item["input_ids"] for item in items], pad_value=pad_token_id, device=device),
        "attention_mask": _pad_qwen_1d([item["attention_mask"] for item in items], pad_value=0, device=device),
        "mm_token_type_ids": _pad_qwen_1d([item["mm_token_type_ids"] for item in items], pad_value=0, device=device),
        "pixel_values": torch.cat([item["pixel_values"].to(device) for item in items], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"].to(device) for item in items], dim=0),
    }


def _qwen_eval_index_batches(
    dataset: QwenBenchmarkDataset,
    *,
    batch_size: int,
    max_batch_tokens: int,
    bucket_by_length: bool,
) -> list[list[int]]:
    batch_size = max(1, int(batch_size))
    max_batch_tokens = int(max_batch_tokens)
    keyed_indices = []
    for idx in range(len(dataset)):
        row = dataset.rows[idx]
        rough_len = len(str(row.get("question", ""))) + len(str(row.get("image", ""))) // 8
        keyed_indices.append((max(1, rough_len), idx))
    if bucket_by_length and batch_size > 1:
        keyed_indices.sort(key=lambda item: item[0])

    batches: list[list[int]] = []
    current: list[int] = []
    current_max_len = 0
    for item_len, idx in keyed_indices:
        next_max_len = max(current_max_len, item_len)
        next_size = len(current) + 1
        overflow_size = next_size > batch_size
        overflow_tokens = current and max_batch_tokens > 0 and next_max_len * next_size > max_batch_tokens
        if overflow_size or overflow_tokens:
            batches.append(current)
            current = []
            current_max_len = 0
        current.append(idx)
        current_max_len = max(current_max_len, item_len)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def generate_teacher_llava(
    model,
    processor,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    image_sizes=None,
    max_new_tokens: int = 8,
    early_stop_metric: str | None = None,
    choices: list[Any] | None = None,
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
    return _prediction_from_text(early_stop_metric, text, choices), text


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
    output_mode: str = "kv_adapter",
    max_new_tokens: int = 8,
    adapter_decode_cache: bool = True,
    early_stop_metric: str | None = None,
    choices: list[Any] | None = None,
    image_sizes=None,
) -> tuple[str | None, str]:
    full_ids = input_ids.clone()
    full_mask = attention_mask.clone()
    generated = []
    eos_ids = _eos_token_ids(processor.tokenizer)
    logits = None
    decode_cache = None
    if adapter_decode_cache:
        if is_embedding_adapter_mode(output_mode):
            logits, decode_cache = llava_embedding_adapter_prefill_cache(
                model,
                adapter,
                full_ids,
                pixel_values,
                image_token_id,
                attention_mask=full_mask,
                image_sizes=image_sizes,
            )
        else:
            assert source_k is not None and source_v is not None
            logits, decode_cache = llava_kv_adapter_prefill_cache(
                model,
                adapter,
                full_ids,
                source_k,
                source_v,
                image_token_id,
                attention_mask=full_mask,
            )

    for _ in range(max_new_tokens):
        if logits is None:
            if is_embedding_adapter_mode(output_mode):
                logits = student_forward_llava_embedding_adapter(
                    model,
                    full_ids,
                    pixel_values,
                    adapter,
                    image_token_id,
                    attention_mask=full_mask,
                    image_sizes=image_sizes,
                )
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
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if next_token in eos_ids or _structured_answer_ready(early_stop_metric, text, choices):
            break
        if adapter_decode_cache:
            assert decode_cache is not None
            if is_embedding_adapter_mode(output_mode):
                logits, decode_cache = llava_embedding_adapter_decode_step_shape_exact(model, adapter, token_tensor, decode_cache)
            else:
                logits, decode_cache = llava_kv_adapter_decode_step_shape_exact(model, adapter, token_tensor, decode_cache)
        else:
            logits = None

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return _prediction_from_text(early_stop_metric, text, choices), text


def load_adapter(checkpoint_path: str, model, device: torch.device) -> tuple[torch.nn.Module, list[int], str]:
    """Load trained adapter from checkpoint."""
    adapter, source_layers, metadata = load_adapter_checkpoint(
        checkpoint_path,
        device=device,
        language_model=model.model.language_model,
        dtype=torch.bfloat16,
    )
    return adapter, source_layers, str(metadata.get("output_mode") or "kv_adapter")


@torch.inference_mode()
def evaluate_llava_shard(
    model,
    processor,
    adapter: torch.nn.Module,
    dataset: LlavaBenchmarkDataset,
    device: torch.device,
    image_token_id: int,
    source_layers: list[int],
    log_every: int = 25,
    max_new_tokens: int = 8,
    benchmark: str = "mmstar",
    measure_prefill: bool = True,
    output_mode: str = "kv_adapter",
    structured_answer_early_stop: bool = True,
    adapter_decode_cache: bool = True,
    verify_decode_cache_generation: int = 0,
) -> dict:
    """Evaluate a LLaVA adapter shard with the shared benchmark registry."""
    spec = get_benchmark_spec(benchmark)
    predictions: list[dict[str, Any]] = []

    def timed_value(fn, *, enabled: bool = True):
        if not enabled:
            return 0.0, fn()
        _sync_cuda()
        start = time.perf_counter()
        value = fn()
        _sync_cuda()
        return time.perf_counter() - start, value

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        image_sizes = item.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.unsqueeze(0).to(device) if torch.is_tensor(image_sizes) else image_sizes
        choices = item.get("choices")
        text_tokens, image_tokens = _llava_token_counts(input_ids, attention_mask, image_token_id)

        if is_embedding_adapter_mode(output_mode):
            source_k = source_v = None
            source_s = 0.0
        else:
            source_s, source_pair = timed_value(
                lambda: extract_vision_kv(model, pixel_values, source_layer_indices=source_layers),
                enabled=measure_prefill,
            )
            source_k, source_v = source_pair

        teacher_prefill_s, _ = _timed_call(
            lambda: teacher_forward(model, input_ids, pixel_values, attention_mask, image_sizes=image_sizes),
            enabled=measure_prefill,
        )
        teacher_total_s, (_, teacher_text) = timed_value(
            lambda: generate_teacher_llava(
                model,
                processor,
                input_ids,
                pixel_values,
                attention_mask,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                early_stop_metric=spec.metric,
                choices=choices,
            )
        )

        def adapter_prefill():
            if is_embedding_adapter_mode(output_mode):
                if adapter_decode_cache:
                    return llava_embedding_adapter_prefill_cache(
                        model,
                        adapter,
                        input_ids,
                        pixel_values,
                        image_token_id,
                        attention_mask=attention_mask,
                        image_sizes=image_sizes,
                    )
                return student_forward_llava_embedding_adapter(
                    model,
                    input_ids,
                    pixel_values,
                    adapter,
                    image_token_id,
                    attention_mask=attention_mask,
                    image_sizes=image_sizes,
                )
            assert source_k is not None and source_v is not None
            if adapter_decode_cache:
                return llava_kv_adapter_prefill_cache(
                    model,
                    adapter,
                    input_ids,
                    source_k,
                    source_v,
                    image_token_id,
                    attention_mask=attention_mask,
                )
            return student_forward_with_visual_kv(
                model,
                input_ids,
                adapter,
                source_k,
                source_v,
                image_token_id,
                attention_mask=attention_mask,
            )

        adapter_prefill_s, _ = _timed_call(adapter_prefill, enabled=measure_prefill)
        adapter_prefill_s += source_s
        adapter_total_s, (_, adapter_text) = timed_value(
            lambda: generate_adapter_llava(
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
                adapter_decode_cache=adapter_decode_cache,
                early_stop_metric=spec.metric if structured_answer_early_stop else None,
                choices=choices,
                image_sizes=image_sizes,
            )
        )
        adapter_total_s += source_s
        if adapter_decode_cache and idx < int(verify_decode_cache_generation):
            _, recompute_text = generate_adapter_llava(
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
                adapter_decode_cache=False,
                early_stop_metric=spec.metric if structured_answer_early_stop else None,
                choices=choices,
                image_sizes=image_sizes,
            )
            if recompute_text != adapter_text:
                raise RuntimeError(
                    "LLaVA decode-cache generation changed decoded text: "
                    f"dataset_index={idx} recompute={recompute_text!r} cache={adapter_text!r}"
                )

        teacher_eval = score_prediction(
            metric=spec.metric,
            prediction_text=teacher_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=choices,
            question=item.get("question") or item.get("row", {}).get("question"),
        )
        teacher_eval = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=teacher_eval,
            row=item.get("row", {}),
            prediction_text=teacher_text,
            answer=item.get("answer"),
        )
        adapter_eval = score_prediction(
            metric=spec.metric,
            prediction_text=adapter_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=choices,
            question=item.get("question") or item.get("row", {}).get("question"),
        )
        adapter_eval = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=adapter_eval,
            row=item.get("row", {}),
            prediction_text=adapter_text,
            answer=item.get("answer"),
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
                "teacher_kv_cache_mb": None,
                "adapter_kv_cache_mb": None,
                "teacher_prefill_flops": None,
                "adapter_prefill_flops": None,
            }
        )

        if (idx + 1) % log_every == 0:
            summary = summarize_benchmark_predictions(
                benchmark=benchmark,
                predictions=predictions,
                output_modes=[output_mode],
            )
            teacher_score = summary["teacher"]["score"]
            adapter_score = summary["adapter"]["score"] if summary["adapter"] else 0.0
            print(
                f"[{idx+1}/{len(dataset)}] {spec.display_name} teacher={teacher_score:.4f} "
                f"adapter={adapter_score:.4f}",
                flush=True,
            )

    summary = summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=[output_mode])
    return {"summary": summary, "predictions": predictions, "output_mode": output_mode}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified benchmark evaluation for LLaVA kv_adapter and Qwen3-VL embedding_adapter.")
    parser.add_argument("--model-kind", choices=("llava", "qwen"), default="llava")
    parser.add_argument("--benchmark", default="mmstar", help=f"Benchmark name. Choices: {', '.join(sorted(BENCHMARK_SPECS))}")
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default=None)
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving image paths")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None, help="If set, only run this shard")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-mode", choices=("kv_adapter", "embedding_adapter", "recurrent_embedding_adapter"), default=None)
    parser.add_argument("--answer-instruction", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-device-map", default="", help="Optional Qwen HF device_map, e.g. auto, for single-process multi-GPU loading.")
    parser.add_argument("--qwen-max-memory", default="", help="Optional HF max_memory, JSON or comma list like 0=120GiB,1=120GiB,cpu=200GiB.")
    parser.add_argument("--measure-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-only", action="store_true", help="Evaluate only the frozen/base model; does not require --checkpoint.")
    parser.add_argument("--compile-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-verify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-max-diff", type=float, default=0.0)
    parser.add_argument("--compile-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured-answer-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-cache", action=argparse.BooleanOptionalAction, default=True, help="Cache deterministic Qwen teacher generations across adapter checkpoints.")
    parser.add_argument("--teacher-cache-dir", default=None)
    parser.add_argument("--require-teacher-cache", action="store_true", help="Fail instead of generating teacher outputs when the teacher cache is missing or stale.")
    parser.add_argument("--last-logits-only", action=argparse.BooleanOptionalAction, default=True, help="Only compute logits for the next-token position during generation/eval prefill.")
    parser.add_argument("--input-cache-dir", default=None, help="Optional cache directory for preprocessed Qwen benchmark tensors.")
    parser.add_argument("--context-cache-dir", default=None, help="Optional cache directory for Qwen initial_hidden/position_ids tensors.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Batch Qwen adapter generation. Teacher generation remains single-sample/cache-safe.")
    parser.add_argument(
        "--eval-max-batch-tokens",
        type=int,
        default=0,
        help="Optional rough token budget for Qwen eval batches. 0 disables.",
    )
    parser.add_argument(
        "--eval-bucket-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort Qwen batched eval samples by rough length to reduce padding. Only affects eval_batch_size > 1.",
    )
    parser.add_argument(
        "--verify-batched-generation",
        type=int,
        default=0,
        help="Compare this many Qwen batched adapter generations against single-sample generation and fail on any decoded-text mismatch.",
    )
    parser.add_argument(
        "--adapter-decode-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adapter text KV cache during Qwen generation. Disable with --no-adapter-decode-cache for old recompute generation.",
    )
    parser.add_argument(
        "--adapter-decode-cache-mode",
        choices=("shape_exact", "fast"),
        default="shape_exact",
        help="Qwen decode cache mode. shape_exact preserves full-seq linear/MLP shapes; fast uses q_len=1 kernels and is diagnostic.",
    )
    parser.add_argument(
        "--verify-decode-cache-generation",
        type=int,
        default=0,
        help="Compare this many Qwen decode-cache generations against old full-recompute generation.",
    )
    args = parser.parse_args(argv)
    args.benchmark = canonical_benchmark_name(args.benchmark)
    spec = get_benchmark_spec(args.benchmark)
    if args.data is None:
        args.data = spec.default_data
    if args.max_new_tokens is None:
        args.max_new_tokens = spec.max_new_tokens
    if args.output_mode is not None:
        args.output_mode = canonical_adapter_mode(args.output_mode)
    return args


def parse_qwen_device_map(value: str) -> str | dict[str, Any] | None:
    value = str(value or "").strip()
    if not value or value.lower() in {"none", "replicated"}:
        return None
    if value.startswith("{"):
        return json.loads(value)
    return value


def parse_qwen_max_memory(value: str) -> dict[Any, str] | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value.startswith("{"):
        parsed = json.loads(value)
        return {int(key) if str(key).isdigit() else key: str(mem) for key, mem in parsed.items()}
    max_memory: dict[Any, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid --qwen-max-memory item {item!r}; expected key=value")
        key, mem = item.split("=", 1)
        key = key.strip()
        max_memory[int(key) if key.isdigit() else key] = mem.strip()
    return max_memory


def run_llava_single_shard(args, shard_id: int, num_shards: int):
    """Run evaluation on a single shard (one GPU)."""
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    configure_torch_runtime()
    processor, model = load_frozen_llava(
        args.model_path,
        dtype=dtype,
        device="cuda:0",
        attn_implementation=args.attn_implementation,
    )
    if args.teacher_only:
        raise ValueError("--teacher-only is currently implemented for model-kind=qwen")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --teacher-only is set")
    adapter, source_layers, checkpoint_mode = load_adapter(args.checkpoint, model, device)
    output_mode = args.output_mode or checkpoint_mode
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    full_dataset = LlavaBenchmarkDataset(
        args.data,
        processor,
        args.benchmark,
        data_root=args.data_root,
        max_samples=args.max_samples,
        answer_instruction=args.answer_instruction,
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
        args.benchmark,
        args.measure_prefill,
        output_mode,
        bool(args.structured_answer_early_stop),
        bool(args.adapter_decode_cache),
        int(args.verify_decode_cache_generation),
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
    early_stop_metric: str | None = None,
    choices: list[Any] | None = None,
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
            logits, text_mask, _ = qwen_embedding_adapter_logits(
                model,
                adapter,
                inputs,
                initial_hidden=initial_hidden,
                position_ids=position_ids,
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
def generate_adapter_qwen_batch(
    model,
    processor,
    adapter,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    adapter_logits_fn: Callable | None = None,
    initial_hidden: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    prefill_logits: torch.Tensor | None = None,
    prefill_text_mask: torch.Tensor | None = None,
    last_logits_only: bool = True,
    early_stop_metric: str | None = None,
    choices: list[list[Any] | None] | None = None,
) -> tuple[list[str | None], list[str]]:
    full_ids = inputs["input_ids"].clone()
    full_mask = inputs["attention_mask"].clone()
    full_mm_ids = inputs["mm_token_type_ids"].clone()
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    batch_size = int(full_ids.shape[0])
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    texts = ["" for _ in range(batch_size)]
    active = torch.ones(batch_size, dtype=torch.bool, device=full_ids.device)
    eos_ids = _eos_token_ids(processor.tokenizer)
    eos_or_pad = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_or_pad is None:
        eos_or_pad = getattr(processor.tokenizer, "pad_token_id", 0)
    eos_or_pad = int(eos_or_pad if eos_or_pad is not None else 0)
    token_embeddings = model.model.get_input_embeddings()
    if choices is None:
        choices = [None for _ in range(batch_size)]

    if initial_hidden is None or position_ids is None:
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    last_pos_idx = full_mask.long().sum(dim=1).sub(1).view(1, -1, 1).expand(position_ids.shape[0], -1, 1)
    next_position_ids = position_ids.gather(2, last_pos_idx)

    for _ in range(max_new_tokens):
        step_inputs = {
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
            logits, text_mask, _ = qwen_embedding_adapter_logits(
                model,
                adapter,
                step_inputs,
                initial_hidden=initial_hidden,
                position_ids=position_ids,
                compact_no_padding=True,
                logits_to_keep=1 if last_logits_only else 0,
            )
        else:
            logits, text_mask = adapter_logits_fn(step_inputs, initial_hidden, position_ids)
        next_tokens = torch.argmax(_next_token_logits_batch(logits, text_mask), dim=-1)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_or_pad))
        token = next_tokens.view(batch_size, 1).to(dtype=full_ids.dtype)

        for row_idx, token_id in enumerate(next_tokens.detach().tolist()):
            if not bool(active[row_idx].item()):
                continue
            token_int = int(token_id)
            generated[row_idx].append(token_int)
            texts[row_idx] = processor.tokenizer.decode(generated[row_idx], skip_special_tokens=True)
            if token_int in eos_ids or _structured_answer_ready(early_stop_metric, texts[row_idx], choices[row_idx]):
                active[row_idx] = False

        full_ids = torch.cat([full_ids, token], dim=1)
        active_column = active.view(batch_size, 1).to(dtype=full_mask.dtype)
        full_mask = torch.cat([full_mask, active_column], dim=1)
        full_mm_ids = torch.cat([full_mm_ids, torch.zeros_like(token)], dim=1)
        initial_hidden = torch.cat(
            [initial_hidden, token_embeddings(token).to(device=initial_hidden.device, dtype=initial_hidden.dtype)],
            dim=1,
        )
        next_position_ids = next_position_ids + 1
        position_ids = torch.cat([position_ids, next_position_ids], dim=2)
        if not bool(active.any().item()):
            break
    return [extract_option_from_text(text) for text in texts], texts


@torch.inference_mode()
def generate_adapter_qwen_decode_cache(
    model,
    processor,
    adapter,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    initial_hidden: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    prefill_logits: torch.Tensor | None = None,
    prefill_text_mask: torch.Tensor | None = None,
    decode_cache: dict[str, Any] | None = None,
    decode_cache_mode: str = "shape_exact",
    last_logits_only: bool = True,
    early_stop_metric: str | None = None,
    choices: list[list[Any] | None] | list[Any] | None = None,
    decode_step_metrics: dict[str, Any] | None = None,
) -> tuple[list[str | None], list[str]]:
    if max_new_tokens <= 0:
        batch_size = int(inputs["input_ids"].shape[0])
        return [None for _ in range(batch_size)], ["" for _ in range(batch_size)]
    if initial_hidden is None or position_ids is None:
        initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
    if decode_cache is None or prefill_logits is None or prefill_text_mask is None:
        prefill_logits, prefill_text_mask, decode_cache = qwen_embedding_adapter_prefill_cache(
            model,
            adapter,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
            logits_to_keep=1 if last_logits_only else 0,
        )

    batch_size = int(inputs["input_ids"].shape[0])
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    texts = ["" for _ in range(batch_size)]
    active = torch.ones(batch_size, dtype=torch.bool, device=inputs["input_ids"].device)
    eos_ids = _eos_token_ids(processor.tokenizer)
    eos_or_pad = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_or_pad is None:
        eos_or_pad = getattr(processor.tokenizer, "pad_token_id", 0)
    eos_or_pad = int(eos_or_pad if eos_or_pad is not None else 0)
    if choices is None or (batch_size == 1 and (not isinstance(choices, list) or (choices and not isinstance(choices[0], list)))):
        batch_choices: list[Any] = [choices]
    else:
        batch_choices = list(choices)

    logits = prefill_logits
    text_mask = prefill_text_mask
    for step_idx in range(max_new_tokens):
        next_tokens = torch.argmax(_next_token_logits_batch(logits, text_mask), dim=-1)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_or_pad))
        was_active = active.clone()
        for row_idx, token_id in enumerate(next_tokens.detach().tolist()):
            if not bool(was_active[row_idx].item()):
                continue
            token_int = int(token_id)
            generated[row_idx].append(token_int)
            texts[row_idx] = processor.tokenizer.decode(generated[row_idx], skip_special_tokens=True)
            row_choices = batch_choices[row_idx] if row_idx < len(batch_choices) else None
            if token_int in eos_ids or _structured_answer_ready(early_stop_metric, texts[row_idx], row_choices):
                active[row_idx] = False
        if step_idx == max_new_tokens - 1 or not bool(active.any().item()):
            break
        decode_step = qwen_embedding_adapter_decode_step_shape_exact if decode_cache_mode == "shape_exact" else qwen_embedding_adapter_decode_step
        if decode_cache_mode == "fast" and getattr(model, "_adapter_decode_graph_runner", None) is not None:
            decode_step = model._adapter_decode_graph_runner
        def advance_decode():
            next_logits, updated_cache = decode_step(
                model, adapter,
                next_tokens.view(batch_size, 1).to(dtype=inputs["input_ids"].dtype),
                # The loop only advances a single-sample request while active.
                decode_cache, logits_to_keep=1, token_active_mask=None if batch_size == 1 else was_active,
            )
            # Keep the caller's cache handle current for final KV accounting.
            # Graph replay returns owned tensors in a fresh container.
            if updated_cache is not decode_cache:
                decode_cache.clear()
                decode_cache.update(updated_cache)
            return next_logits, decode_cache
        if decode_step_metrics is None:
            logits, decode_cache = advance_decode()
        else:
            elapsed, (logits, decode_cache) = _timed_call(advance_decode)
            decode_step_metrics.setdefault("step_times_s", []).append(elapsed)
        text_mask = torch.ones((batch_size, 1), device=logits.device, dtype=torch.bool)
    if decode_step_metrics is not None:
        decode_step_metrics["generated_token_ids"] = generated
    return [extract_option_from_text(text) for text in texts], texts


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tensor_tree_nbytes(value: Any, seen: set[tuple[str, int]] | None = None) -> int:
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        try:
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()))
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        except Exception:
            return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(item, seen) for item in value)
    from transformers.cache_utils import Cache
    if isinstance(value, Cache):
        return _tensor_tree_nbytes([(layer.keys, layer.values) for layer in value.layers], seen)
    return 0


def _tensor_tree_mb(value: Any) -> float:
    return _tensor_tree_nbytes(value) / (1024.0**2)


@torch.inference_mode()
def _timed_call(fn, enabled: bool = True):
    if not enabled:
        return 0.0, None
    _sync_cuda()
    start = time.perf_counter()
    value = fn()
    _sync_cuda()
    return time.perf_counter() - start, value


@torch.inference_mode()
def _timed_cuda_peak_call(fn, enabled: bool = True):
    if not enabled:
        return 0.0, None, {}
    _sync_cuda()
    stats: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base_allocated = torch.cuda.memory_allocated()
        base_reserved = torch.cuda.memory_reserved()
    else:
        base_allocated = 0
        base_reserved = 0
    start = time.perf_counter()
    value = fn()
    _sync_cuda()
    elapsed = time.perf_counter() - start
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        stats = {
            "peak_allocated_mb": peak_allocated / (1024.0**2),
            "peak_reserved_mb": peak_reserved / (1024.0**2),
            "peak_allocated_delta_mb": max(0, peak_allocated - base_allocated) / (1024.0**2),
            "peak_reserved_delta_mb": max(0, peak_reserved - base_reserved) / (1024.0**2),
        }
    return elapsed, value, stats


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_same_logits_and_mask(
    name: str,
    eager: tuple[torch.Tensor, torch.Tensor],
    compiled: tuple[torch.Tensor, torch.Tensor],
    max_diff: float,
) -> None:
    eager_logits, eager_mask = eager
    compiled_logits, compiled_mask = compiled
    if eager_logits.shape != compiled_logits.shape:
        raise RuntimeError(
            f"{name} compile changed logits shape: eager={tuple(eager_logits.shape)} compiled={tuple(compiled_logits.shape)}"
        )
    if not torch.equal(eager_mask, compiled_mask):
        raise RuntimeError(f"{name} compile changed text_mask")
    diff = (eager_logits.float() - compiled_logits.float()).abs()
    observed = float(diff.max().item()) if diff.numel() else 0.0
    if observed > float(max_diff):
        raise RuntimeError(f"{name} compile changed logits: max_diff={observed:.8g} allowed={float(max_diff):.8g}")


def build_qwen_adapter_logits_fn(
    model,
    adapter,
    *,
    compile_adapter: bool,
    compile_mode: str,
    compile_dynamic: bool,
    last_logits_only: bool,
    compile_verify: bool = True,
    compile_max_diff: float = 0.0,
):
    logits_to_keep = 1 if last_logits_only else 0

    def direct(inputs, initial_hidden=None, position_ids=None):
        if initial_hidden is None or position_ids is None:
            logits, text_mask, _ = qwen_embedding_adapter_logits(
                model,
                adapter,
                inputs,
                initial_hidden=initial_hidden,
                position_ids=position_ids,
                compact_no_padding=True,
                logits_to_keep=logits_to_keep,
            )
        else:
            logits, text_mask, _ = qwen_embedding_adapter_logits_from_tensors(
                model,
                adapter,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                initial_hidden,
                position_ids,
                compact_no_padding=True,
                logits_to_keep=logits_to_keep,
            )
        return logits, text_mask

    if not compile_adapter:
        return direct

    def prepared_forward(prepared):
        logits, text_mask, _ = qwen_embedding_adapter_logits_prepared(
            model,
            adapter,
            h=prepared["h"],
            visual_memory=prepared["visual_memory"],
            text_mask=prepared["text_mask"],
            text_position_ids=prepared["text_position_ids"],
            visual_position_ids=prepared["visual_position_ids"],
            prefix_attention_mask=prepared["prefix_attention_mask"],
            text_position_embeddings=prepared["text_position_embeddings"],
            visual_position_embeddings=prepared["visual_position_embeddings"],
            logits_to_keep=logits_to_keep,
        )
        return logits, text_mask

    def cached_forward_prepared(
        h,
        visual_memory,
        text_mask,
        text_position_ids,
        visual_position_ids,
        prefix_attention_mask,
        text_cos,
        text_sin,
        visual_cos,
        visual_sin,
    ):
        logits, text_mask_out, _ = qwen_embedding_adapter_logits_prepared(
            model,
            adapter,
            h=h,
            visual_memory=visual_memory,
            text_mask=text_mask,
            text_position_ids=text_position_ids,
            visual_position_ids=visual_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            text_position_embeddings=(text_cos, text_sin),
            visual_position_embeddings=(visual_cos, visual_sin),
            logits_to_keep=logits_to_keep,
        )
        return logits, text_mask_out

    compiled_cached = torch.compile(cached_forward_prepared, mode=compile_mode, dynamic=compile_dynamic)
    verified = False

    def compiled(inputs, initial_hidden=None, position_ids=None):
        nonlocal verified
        if initial_hidden is None or position_ids is None:
            initial_hidden, position_ids = build_qwen_initial_context(model, inputs)
        prepared = prepare_qwen_embedding_adapter_inputs(
            model,
            adapter,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
            reuse_position_embeddings=True,
        )
        text_cos, text_sin = prepared["text_position_embeddings"]
        visual_cos, visual_sin = prepared["visual_position_embeddings"]
        tensor_args = (
            prepared["h"],
            prepared["visual_memory"],
            prepared["text_mask"],
            prepared["text_position_ids"],
            prepared["visual_position_ids"],
            prepared["prefix_attention_mask"],
            text_cos,
            text_sin,
            visual_cos,
            visual_sin,
        )
        if compile_verify and not verified:
            eager_out = prepared_forward(prepared)
            compiled_out = compiled_cached(*tensor_args)
            assert_same_logits_and_mask("qwen_eval_cached_adapter", eager_out, compiled_out, compile_max_diff)
            verified = True
            return compiled_out
        return compiled_cached(*tensor_args)

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


def _llava_token_counts(input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_id: int) -> tuple[int, int]:
    valid = attention_mask.bool()
    visual = int(((input_ids == int(image_token_id)) & valid).sum().item())
    text = int(valid.sum().item()) - visual
    return max(text, 0), visual


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
        "measure_prefill",
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


def summarize_benchmark_predictions(
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
    adapter_decode_s = sum(float(item.get("adapter_decode_s", 0.0)) for item in predictions)

    timing = {
        "teacher_total_s": teacher_total_s,
        "teacher_total_minsec": format_seconds_minsec(teacher_total_s),
        "teacher_prefill_s": teacher_prefill_s,
        "teacher_prefill_minsec": format_seconds_minsec(teacher_prefill_s),
        "adapter_total_s": adapter_total_s,
        "adapter_total_minsec": format_seconds_minsec(adapter_total_s),
        "adapter_prefill_s": adapter_prefill_s,
        "adapter_prefill_minsec": format_seconds_minsec(adapter_prefill_s),
        "adapter_decode_s": adapter_decode_s,
        "adapter_decode_minsec": format_seconds_minsec(adapter_decode_s),
        "speedup_total": (teacher_total_s / adapter_total_s) if adapter_total_s > 0 else None,
        "speedup_prefill": (teacher_prefill_s / adapter_prefill_s) if adapter_prefill_s > 0 else None,
    }

    def resource_mean(key: str) -> float | None:
        values = [item.get(key) for item in predictions]
        numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
        if not numeric:
            return None
        return sum(numeric) / len(numeric)

    resources = {
        "teacher_kv_cache_mb_avg": resource_mean("teacher_kv_cache_mb"),
        "adapter_kv_cache_mb_avg": resource_mean("adapter_kv_cache_mb"),
        "teacher_prefill_flops_avg": resource_mean("teacher_prefill_flops"),
        "adapter_prefill_flops_avg": resource_mean("adapter_prefill_flops"),
        "teacher_prefill_peak_allocated_mb_avg": resource_mean("teacher_prefill_peak_allocated_mb"),
        "teacher_prefill_peak_reserved_mb_avg": resource_mean("teacher_prefill_peak_reserved_mb"),
        "teacher_prefill_peak_allocated_delta_mb_avg": resource_mean("teacher_prefill_peak_allocated_delta_mb"),
        "teacher_prefill_peak_reserved_delta_mb_avg": resource_mean("teacher_prefill_peak_reserved_delta_mb"),
        "adapter_prefill_peak_allocated_mb_avg": resource_mean("adapter_prefill_peak_allocated_mb"),
        "adapter_prefill_peak_reserved_mb_avg": resource_mean("adapter_prefill_peak_reserved_mb"),
        "adapter_decode_peak_allocated_mb_avg": resource_mean("adapter_decode_peak_allocated_mb"),
        "adapter_decode_peak_reserved_mb_avg": resource_mean("adapter_decode_peak_reserved_mb"),
        "adapter_prefill_peak_allocated_delta_mb_avg": resource_mean("adapter_prefill_peak_allocated_delta_mb"),
        "adapter_prefill_peak_reserved_delta_mb_avg": resource_mean("adapter_prefill_peak_reserved_delta_mb"),
        "adapter_decode_peak_allocated_delta_mb_avg": resource_mean("adapter_decode_peak_allocated_delta_mb"),
        "adapter_decode_peak_reserved_delta_mb_avg": resource_mean("adapter_decode_peak_reserved_delta_mb"),
        "adapter_prefill_decode_cache_mb_avg": resource_mean("adapter_prefill_decode_cache_mb"),
        "adapter_prefill_kv_cache_mb_avg": resource_mean("adapter_prefill_kv_cache_mb"),
        "adapter_final_decode_cache_mb_avg": resource_mean("adapter_final_decode_cache_mb"),
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


def summarize_qwen_benchmark_predictions(
    *,
    benchmark: str,
    predictions: list[dict[str, Any]],
    output_modes: list[str] | None = None,
) -> dict[str, Any]:
    return summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=output_modes)


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
    eval_batch_size: int = 1,
    eval_max_batch_tokens: int = 0,
    eval_bucket_by_length: bool = True,
    verify_batched_generation: int = 0,
    adapter_decode_cache: bool = True,
    adapter_decode_cache_mode: str = "shape_exact",
    verify_decode_cache_generation: int = 0,
    adapter_prefill_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] | None = None,
    prefill_warmup: int = 0,
    measure_decode_steps: bool = False,
) -> dict:
    if measure_decode_steps and (int(eval_batch_size) != 1 or not measure_prefill or not adapter_decode_cache):
        raise ValueError("Decode-step timing requires single-sample cached prefill evaluation")
    if int(eval_batch_size) > 1:
        return evaluate_qwen_benchmark_shard_batched(
            model,
            processor,
            adapter,
            dataset,
            device,
            log_every,
            max_new_tokens,
            benchmark,
            measure_prefill,
            dtype,
            adapter_logits_fn=adapter_logits_fn,
            compile_warmup=compile_warmup,
            context_cache_dir=context_cache_dir,
            teacher_cache_path=teacher_cache_path,
            teacher_cache_meta=teacher_cache_meta,
            require_teacher_cache=require_teacher_cache,
            structured_answer_early_stop=structured_answer_early_stop,
            last_logits_only=last_logits_only,
            eval_batch_size=eval_batch_size,
            eval_max_batch_tokens=eval_max_batch_tokens,
            eval_bucket_by_length=eval_bucket_by_length,
            verify_batched_generation=verify_batched_generation,
            adapter_decode_cache=adapter_decode_cache,
            adapter_decode_cache_mode=adapter_decode_cache_mode,
            verify_decode_cache_generation=verify_decode_cache_generation,
            adapter_prefill_fn=adapter_prefill_fn,
            prefill_warmup=prefill_warmup,
        )

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
        teacher_prefill_stats: dict[str, float] = {}
        if cached_teacher is None:
            if require_teacher_cache:
                raise FileNotFoundError(f"required teacher cache entry {idx} is missing: {teacher_cache_path}")
            if measure_prefill and int(prefill_warmup) > 0:
                for _ in range(int(prefill_warmup)):
                    model(**inputs, logits_to_keep=1 if last_logits_only else 0).logits
                _sync_cuda()
            teacher_prefill_s, _, teacher_prefill_stats = _timed_cuda_peak_call(
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
            if adapter_prefill_fn is not None:
                if not adapter_decode_cache:
                    raise ValueError("adapter_prefill_fn requires adapter_decode_cache=True")
                return adapter_prefill_fn(inputs)
            initial_hidden, position_ids = load_or_build_qwen_initial_context(
                model,
                inputs,
                cache_dir=context_cache_dir,
                dtype=dtype,
            )
            if adapter_decode_cache:
                logits, text_mask, decode_cache = qwen_embedding_adapter_prefill_cache(
                    model,
                    adapter,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["mm_token_type_ids"],
                    initial_hidden,
                    position_ids,
                    logits_to_keep=1 if last_logits_only else 0,
                )
                return logits, text_mask, initial_hidden, position_ids, decode_cache
            if adapter_logits_fn is None:
                logits, text_mask, _ = qwen_embedding_adapter_logits(
                    model,
                    adapter,
                    dict(inputs),
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    compact_no_padding=True,
                    logits_to_keep=1 if last_logits_only else 0,
                )
            else:
                logits, text_mask = adapter_logits_fn(dict(inputs), initial_hidden, position_ids)
            return logits, text_mask, initial_hidden, position_ids

        decode_graph_runner = getattr(model, "_adapter_decode_graph_runner", None)
        if measure_prefill and int(prefill_warmup) > 0:
            for _ in range(int(prefill_warmup)):
                warm_payload = adapter_prefill()
                if decode_graph_runner is not None:
                    decode_graph_runner.allow_capture = True
                    try:
                        generate_adapter_qwen_decode_cache(model, processor, adapter, inputs, max_new_tokens,
                            initial_hidden=warm_payload[2], position_ids=warm_payload[3],
                            prefill_logits=warm_payload[0], prefill_text_mask=warm_payload[1], decode_cache=warm_payload[4],
                            decode_cache_mode="fast", last_logits_only=last_logits_only,
                            early_stop_metric=spec.metric if structured_answer_early_stop else None, choices=item.get("choices"))
                    finally:
                        decode_graph_runner.allow_capture = False
            _sync_cuda()
        decode_graph_before = decode_graph_runner.stats() if decode_graph_runner is not None else None
        adapter_prefill_s, adapter_prefill_payload, adapter_prefill_stats = _timed_cuda_peak_call(
            adapter_prefill,
            enabled=measure_prefill,
        )
        adapter_decode_s = 0.0
        adapter_decode_stats: dict[str, float] = {}
        adapter_prefill_decode_cache_mb: float | None = None
        adapter_prefill_kv_cache_mb: float | None = None
        adapter_final_decode_cache_mb: float | None = None
        decode_step_metrics = {} if measure_decode_steps else None
        if adapter_prefill_payload is None:
            if adapter_decode_cache:
                adapter_total_s, (_, adapter_texts), adapter_decode_stats = _timed_cuda_peak_call(
                    lambda: generate_adapter_qwen_decode_cache(
                        model,
                        processor,
                        adapter,
                        inputs,
                        max_new_tokens,
                        decode_cache_mode=adapter_decode_cache_mode,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=item.get("choices"),
                    )
                )
                adapter_text = adapter_texts[0]
            else:
                adapter_total_s, (_, adapter_text), adapter_decode_stats = _timed_cuda_peak_call(
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
            adapter_decode_s = adapter_total_s
        else:
            if adapter_decode_cache:
                prefill_logits, prefill_text_mask, initial_hidden, position_ids, decode_cache = adapter_prefill_payload
                adapter_prefill_decode_cache_mb = _tensor_tree_mb(decode_cache)
                # Inspect outside both timed regions; these layers hold text/visual K/V only.
                adapter_prefill_kv_cache_mb = _tensor_tree_mb(decode_cache.get("_native_cache", decode_cache["layers"]))
                adapter_continuation_s, (_, adapter_texts), adapter_decode_stats = _timed_cuda_peak_call(
                    lambda: generate_adapter_qwen_decode_cache(
                        model,
                        processor,
                        adapter,
                        inputs,
                        max_new_tokens,
                        initial_hidden=initial_hidden,
                        position_ids=position_ids,
                        prefill_logits=prefill_logits,
                        prefill_text_mask=prefill_text_mask,
                        decode_cache=decode_cache,
                        decode_cache_mode=adapter_decode_cache_mode,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=item.get("choices"),
                        decode_step_metrics=decode_step_metrics,
                    )
                )
                adapter_text = adapter_texts[0]
                adapter_final_decode_cache_mb = _tensor_tree_mb(decode_cache)
            else:
                prefill_logits, prefill_text_mask, initial_hidden, position_ids = adapter_prefill_payload
                adapter_continuation_s, (_, adapter_text), adapter_decode_stats = _timed_cuda_peak_call(
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
            adapter_decode_s = adapter_continuation_s
            adapter_total_s = adapter_prefill_s + adapter_continuation_s

        if decode_graph_runner is not None:
            graph_after = decode_graph_runner.stats()
            if graph_after["captures"] != decode_graph_before["captures"] or graph_after["cold_fallbacks"] != decode_graph_before["cold_fallbacks"]:
                raise RuntimeError(f"Adapter decode graph capture/miss during timed sample {idx}; run with per-sample warmup")
        if adapter_decode_cache and idx < int(verify_decode_cache_generation):
            _, recompute_text = generate_adapter_qwen(
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
            if recompute_text != adapter_text:
                raise RuntimeError(
                    "Qwen decode-cache generation changed decoded text: "
                    f"dataset_index={idx} recompute={recompute_text!r} cache={adapter_text!r}"
                )

        teacher_eval = score_prediction(
            metric=spec.metric,
            prediction_text=teacher_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=item.get("choices"),
            question=item.get("question") or item.get("row", {}).get("question"),
        )
        teacher_eval = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=teacher_eval,
            row=item.get("row", {}),
            prediction_text=teacher_text,
            answer=item.get("answer"),
        )
        adapter_eval = score_prediction(
            metric=spec.metric,
            prediction_text=adapter_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=item.get("choices"),
            question=item.get("question") or item.get("row", {}).get("question"),
        )
        adapter_eval = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=adapter_eval,
            row=item.get("row", {}),
            prediction_text=adapter_text,
            answer=item.get("answer"),
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
                "adapter_decode_s": adapter_decode_s,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "teacher_kv_cache_mb": teacher_kv,
                "adapter_kv_cache_mb": adapter_kv,
                "teacher_prefill_flops": teacher_flops,
                "adapter_prefill_flops": adapter_flops,
                "teacher_prefill_peak_allocated_mb": teacher_prefill_stats.get("peak_allocated_mb"),
                "teacher_prefill_peak_reserved_mb": teacher_prefill_stats.get("peak_reserved_mb"),
                "teacher_prefill_peak_allocated_delta_mb": teacher_prefill_stats.get("peak_allocated_delta_mb"),
                "teacher_prefill_peak_reserved_delta_mb": teacher_prefill_stats.get("peak_reserved_delta_mb"),
                "adapter_prefill_peak_allocated_mb": adapter_prefill_stats.get("peak_allocated_mb"),
                "adapter_prefill_peak_reserved_mb": adapter_prefill_stats.get("peak_reserved_mb"),
                "adapter_prefill_peak_allocated_delta_mb": adapter_prefill_stats.get("peak_allocated_delta_mb"),
                "adapter_prefill_peak_reserved_delta_mb": adapter_prefill_stats.get("peak_reserved_delta_mb"),
                "adapter_decode_peak_allocated_mb": adapter_decode_stats.get("peak_allocated_mb"),
                "adapter_decode_peak_reserved_mb": adapter_decode_stats.get("peak_reserved_mb"),
                "adapter_decode_peak_allocated_delta_mb": adapter_decode_stats.get("peak_allocated_delta_mb"),
                "adapter_decode_peak_reserved_delta_mb": adapter_decode_stats.get("peak_reserved_delta_mb"),
                "adapter_prefill_decode_cache_mb": adapter_prefill_decode_cache_mb,
                "adapter_prefill_kv_cache_mb": adapter_prefill_kv_cache_mb,
                "adapter_final_decode_cache_mb": adapter_final_decode_cache_mb,
                **({
                    "adapter_decode_step_times_s": decode_step_metrics.get("step_times_s", []),
                    "adapter_decode_steps": len(decode_step_metrics.get("step_times_s", [])),
                    "adapter_decode_forward_s": sum(decode_step_metrics.get("step_times_s", [])),
                    "adapter_generated_token_ids": decode_step_metrics.get("generated_token_ids", []),
                    "adapter_generation_overhead_s": adapter_decode_s - sum(decode_step_metrics.get("step_times_s", [])),
                } if decode_step_metrics is not None else {}),
            }
        )
        if (idx + 1) % log_every == 0:
            summary = summarize_benchmark_predictions(
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

    summary = summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=[adapter.mode])
    if teacher_cache_entries is None and teacher_cache_meta is not None and len(teacher_cache_to_write) == len(dataset):
        _save_teacher_cache(teacher_cache_path, teacher_cache_meta, teacher_cache_to_write)
    if measure_decode_steps:
        for key in ("adapter_decode_steps", "adapter_decode_forward_s", "adapter_generation_overhead_s"):
            summary["timing"][key] = sum(p[key] for p in predictions)
    return {"summary": summary, "predictions": predictions, "output_mode": adapter.mode}


@torch.inference_mode()
def evaluate_qwen_benchmark_shard_batched(
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
    eval_batch_size: int = 1,
    eval_max_batch_tokens: int = 0,
    eval_bucket_by_length: bool = True,
    verify_batched_generation: int = 0,
    adapter_decode_cache: bool = True,
    adapter_decode_cache_mode: str = "shape_exact",
    verify_decode_cache_generation: int = 0,
    adapter_prefill_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] | None = None,
    prefill_warmup: int = 0,
) -> dict:
    spec = get_benchmark_spec(benchmark)
    language_config = model.model.language_model.config
    predictions: list[dict[str, Any]] = []
    teacher_cache_entries = _load_teacher_cache(teacher_cache_path, teacher_cache_meta) if teacher_cache_meta else None
    if require_teacher_cache and teacher_cache_entries is None:
        raise FileNotFoundError(f"required teacher cache is missing or stale: {teacher_cache_path}")
    teacher_cache_to_write: list[dict[str, Any] | None] = (
        [None for _ in range(len(dataset))] if teacher_cache_entries is None and teacher_cache_meta is not None else []
    )
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = int(pad_token_id if pad_token_id is not None else 0)

    if compile_warmup and adapter_logits_fn is not None and len(dataset) > 0:
        item = dataset[0]
        warm_inputs = stack_qwen_benchmark_items([item], device=device, pad_token_id=pad_token_id)
        initial_hidden, position_ids = build_qwen_initial_context(model, warm_inputs)
        adapter_logits_fn(warm_inputs, initial_hidden, position_ids)
        _sync_cuda()

    processed = 0
    verified = 0
    index_batches = _qwen_eval_index_batches(
        dataset,
        batch_size=int(eval_batch_size),
        max_batch_tokens=int(eval_max_batch_tokens),
        bucket_by_length=bool(eval_bucket_by_length),
    )
    for index_batch in index_batches:
        items = [dataset[idx] for idx in index_batch]
        batch_inputs = stack_qwen_benchmark_items(items, device=device, pad_token_id=pad_token_id)

        teacher_records: list[tuple[str, float, float]] = []
        for local_idx, idx in enumerate(index_batch):
            item = items[local_idx]
            single_inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
                "pixel_values": item["pixel_values"].to(device),
                "image_grid_thw": item["image_grid_thw"].to(device),
                "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0).to(device),
            }
            cached_teacher = teacher_cache_entries[idx] if teacher_cache_entries is not None else None
            if cached_teacher is None:
                if require_teacher_cache:
                    raise FileNotFoundError(f"required teacher cache entry {idx} is missing: {teacher_cache_path}")
                teacher_prefill_s, _ = _timed_call(
                    lambda inputs=single_inputs: model(**inputs, logits_to_keep=1 if last_logits_only else 0).logits,
                    enabled=measure_prefill,
                )
                teacher_total_s, (_, teacher_text) = _timed_call(
                    lambda inputs=single_inputs: generate_teacher_qwen(
                        model,
                        processor,
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["pixel_values"],
                        inputs["image_grid_thw"],
                        inputs["mm_token_type_ids"],
                        max_new_tokens,
                    )
                )
                if teacher_cache_to_write:
                    teacher_cache_to_write[idx] = {
                        "teacher_text": teacher_text,
                        "teacher_prefill_s": teacher_prefill_s,
                        "teacher_total_s": teacher_total_s,
                    }
            else:
                teacher_text = str(cached_teacher.get("teacher_text", ""))
                teacher_prefill_s = float(cached_teacher.get("teacher_prefill_s", 0.0))
                teacher_total_s = float(cached_teacher.get("teacher_total_s", 0.0))
            teacher_records.append((teacher_text, teacher_total_s, teacher_prefill_s))

        def adapter_prefill_batch():
            initial_hidden, position_ids = load_or_build_qwen_initial_context(
                model,
                batch_inputs,
                cache_dir=context_cache_dir,
                dtype=dtype,
            )
            if adapter_decode_cache:
                logits, text_mask, decode_cache = qwen_embedding_adapter_prefill_cache(
                    model,
                    adapter,
                    batch_inputs["input_ids"],
                    batch_inputs["attention_mask"],
                    batch_inputs["mm_token_type_ids"],
                    initial_hidden,
                    position_ids,
                    logits_to_keep=1 if last_logits_only else 0,
                )
                return logits, text_mask, initial_hidden, position_ids, decode_cache
            if adapter_logits_fn is None:
                logits, text_mask, _ = qwen_embedding_adapter_logits(
                    model,
                    adapter,
                    dict(batch_inputs),
                    initial_hidden=initial_hidden,
                    position_ids=position_ids,
                    compact_no_padding=True,
                    logits_to_keep=1 if last_logits_only else 0,
                )
            else:
                logits, text_mask = adapter_logits_fn(dict(batch_inputs), initial_hidden, position_ids)
            return logits, text_mask, initial_hidden, position_ids

        adapter_prefill_s, adapter_prefill_payload = _timed_call(adapter_prefill_batch, enabled=measure_prefill)
        if adapter_prefill_payload is None:
            if adapter_decode_cache:
                adapter_total_s, (_, adapter_texts) = _timed_call(
                    lambda: generate_adapter_qwen_decode_cache(
                        model,
                        processor,
                        adapter,
                        batch_inputs,
                        max_new_tokens,
                        decode_cache_mode=adapter_decode_cache_mode,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=[item.get("choices") for item in items],
                    )
                )
            else:
                adapter_total_s, (_, adapter_texts) = _timed_call(
                    lambda: generate_adapter_qwen_batch(
                        model,
                        processor,
                        adapter,
                        batch_inputs,
                        max_new_tokens,
                        adapter_logits_fn=adapter_logits_fn,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=[item.get("choices") for item in items],
                    )
                )
            adapter_prefill_share = 0.0
            adapter_total_share = adapter_total_s / max(1, len(items))
        else:
            if adapter_decode_cache:
                prefill_logits, prefill_text_mask, initial_hidden, position_ids, decode_cache = adapter_prefill_payload
                adapter_continuation_s, (_, adapter_texts) = _timed_call(
                    lambda: generate_adapter_qwen_decode_cache(
                        model,
                        processor,
                        adapter,
                        batch_inputs,
                        max_new_tokens,
                        initial_hidden=initial_hidden,
                        position_ids=position_ids,
                        prefill_logits=prefill_logits,
                        prefill_text_mask=prefill_text_mask,
                        decode_cache=decode_cache,
                        decode_cache_mode=adapter_decode_cache_mode,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=[item.get("choices") for item in items],
                    )
                )
            else:
                prefill_logits, prefill_text_mask, initial_hidden, position_ids = adapter_prefill_payload
                adapter_continuation_s, (_, adapter_texts) = _timed_call(
                    lambda: generate_adapter_qwen_batch(
                        model,
                        processor,
                        adapter,
                        batch_inputs,
                        max_new_tokens,
                        adapter_logits_fn=adapter_logits_fn,
                        initial_hidden=initial_hidden,
                        position_ids=position_ids,
                        prefill_logits=prefill_logits,
                        prefill_text_mask=prefill_text_mask,
                        last_logits_only=last_logits_only,
                        early_stop_metric=spec.metric if structured_answer_early_stop else None,
                        choices=[item.get("choices") for item in items],
                    )
                )
            adapter_prefill_share = adapter_prefill_s / max(1, len(items))
            adapter_total_share = (adapter_prefill_s + adapter_continuation_s) / max(1, len(items))

        verify_limit = max(int(verify_batched_generation), int(verify_decode_cache_generation) if adapter_decode_cache else 0)
        if verify_limit > 0 and verified < verify_limit:
            for local_idx, item in enumerate(items):
                if verified >= verify_limit:
                    break
                single_inputs = {
                    "input_ids": item["input_ids"].unsqueeze(0).to(device),
                    "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
                    "pixel_values": item["pixel_values"].to(device),
                    "image_grid_thw": item["image_grid_thw"].to(device),
                    "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0).to(device),
                }
                _, single_text = generate_adapter_qwen(
                    model,
                    processor,
                    adapter,
                    single_inputs["input_ids"],
                    single_inputs["attention_mask"],
                    single_inputs["pixel_values"],
                    single_inputs["image_grid_thw"],
                    single_inputs["mm_token_type_ids"],
                    max_new_tokens,
                    adapter_logits_fn=adapter_logits_fn,
                    last_logits_only=last_logits_only,
                    early_stop_metric=spec.metric if structured_answer_early_stop else None,
                    choices=item.get("choices"),
                )
                if single_text != adapter_texts[local_idx]:
                    raise RuntimeError(
                        "Qwen batched adapter generation changed decoded text: "
                        f"dataset_index={index_batch[local_idx]} single={single_text!r} batch={adapter_texts[local_idx]!r}"
                    )
                verified += 1

        for local_idx, idx in enumerate(index_batch):
            item = items[local_idx]
            teacher_text, teacher_total_s, teacher_prefill_s = teacher_records[local_idx]
            adapter_text = adapter_texts[local_idx]
            text_tokens, image_tokens = _qwen_token_counts(
                item["attention_mask"].unsqueeze(0),
                item["mm_token_type_ids"].unsqueeze(0),
            )
            teacher_eval = score_prediction(
                metric=spec.metric,
                prediction_text=teacher_text,
                answer=item.get("answer"),
                answers=item.get("answers"),
                choices=item.get("choices"),
                question=item.get("question") or item.get("row", {}).get("question"),
            )
            teacher_eval = maybe_llm_judge_eval(
                metric=spec.metric,
                eval_item=teacher_eval,
                row=item.get("row", {}),
                prediction_text=teacher_text,
                answer=item.get("answer"),
            )
            adapter_eval = score_prediction(
                metric=spec.metric,
                prediction_text=adapter_text,
                answer=item.get("answer"),
                answers=item.get("answers"),
                choices=item.get("choices"),
                question=item.get("question") or item.get("row", {}).get("question"),
            )
            adapter_eval = maybe_llm_judge_eval(
                metric=spec.metric,
                eval_item=adapter_eval,
                row=item.get("row", {}),
                prediction_text=adapter_text,
                answer=item.get("answer"),
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
                    "adapter_total_s": adapter_total_share,
                    "teacher_prefill_s": teacher_prefill_s,
                    "adapter_prefill_s": adapter_prefill_share,
                    "text_tokens": text_tokens,
                    "image_tokens": image_tokens,
                    "teacher_kv_cache_mb": teacher_kv,
                    "adapter_kv_cache_mb": adapter_kv,
                    "teacher_prefill_flops": teacher_flops,
                    "adapter_prefill_flops": adapter_flops,
                    "eval_batch_size": len(items),
                }
            )
            processed += 1
        if processed % log_every == 0 or processed == len(dataset):
            summary = summarize_benchmark_predictions(
                benchmark=benchmark,
                predictions=predictions,
                output_modes=[adapter.mode],
            )
            teacher_score = summary["teacher"]["score"]
            adapter_score = summary["adapter"]["score"] if summary["adapter"] else 0.0
            print(
                f"[{processed}/{len(dataset)}] {spec.display_name} teacher={teacher_score:.4f} "
                f"adapter={adapter_score:.4f} eval_batch_size={int(eval_batch_size)}",
                flush=True,
            )

    summary = summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=[adapter.mode])
    if teacher_cache_to_write and all(entry is not None for entry in teacher_cache_to_write):
        _save_teacher_cache(teacher_cache_path, teacher_cache_meta, [entry for entry in teacher_cache_to_write if entry is not None])
    return {"summary": summary, "predictions": predictions, "output_mode": adapter.mode}


@torch.inference_mode()
def evaluate_qwen_teacher_shard(
    model,
    processor,
    dataset: QwenBenchmarkDataset,
    device: torch.device,
    log_every: int,
    max_new_tokens: int,
    benchmark: str,
    measure_prefill: bool,
    dtype: torch.dtype,
    last_logits_only: bool = True,
) -> dict:
    spec = get_benchmark_spec(benchmark)
    language_config = model.model.language_model.config
    predictions: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        inputs = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": item["pixel_values"].to(device),
            "image_grid_thw": item["image_grid_thw"].to(device),
            "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0).to(device),
        }
        text_tokens, image_tokens = _qwen_token_counts(inputs["attention_mask"], inputs["mm_token_type_ids"])
        teacher_prefill_s, _ = _timed_call(
            lambda: model(**inputs, logits_to_keep=1 if last_logits_only else 0).logits,
            enabled=measure_prefill,
        )
        teacher_total_s, (_, teacher_text) = _timed_call(
            lambda: generate_teacher_qwen(
                model,
                processor,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                inputs["mm_token_type_ids"],
                max_new_tokens,
            )
        )
        teacher_eval = score_prediction(
            metric=spec.metric,
            prediction_text=teacher_text,
            answer=item.get("answer"),
            answers=item.get("answers"),
            choices=item.get("choices"),
            question=item.get("question") or item.get("row", {}).get("question"),
        )
        teacher_eval = maybe_llm_judge_eval(
            metric=spec.metric,
            eval_item=teacher_eval,
            row=item.get("row", {}),
            prediction_text=teacher_text,
            answer=item.get("answer"),
        )
        teacher_kv = estimate_qwen_kv_cache_mb(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            dtype_bytes=_dtype_bytes(dtype),
            adapter=False,
        )
        teacher_flops = estimate_qwen_prefill_flops(language_config, text_tokens=text_tokens, image_tokens=image_tokens)
        predictions.append(
            {
                "index": item["index"],
                "row": item["row"],
                "teacher_text": teacher_text,
                "adapter_text": None,
                "teacher_eval": teacher_eval,
                "adapter_eval": None,
                "teacher_total_s": teacher_total_s,
                "adapter_total_s": 0.0,
                "teacher_prefill_s": teacher_prefill_s,
                "adapter_prefill_s": 0.0,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "teacher_kv_cache_mb": teacher_kv,
                "adapter_kv_cache_mb": None,
                "teacher_prefill_flops": teacher_flops,
                "adapter_prefill_flops": None,
            }
        )
        if (idx + 1) % log_every == 0:
            summary = summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=["teacher_only"])
            print(f"[{idx+1}/{len(dataset)}] {spec.display_name} teacher={summary['teacher']['score']:.4f}", flush=True)
    summary = summarize_benchmark_predictions(benchmark=benchmark, predictions=predictions, output_modes=["teacher_only"])
    return {"summary": summary, "predictions": predictions, "output_mode": "teacher_only"}


def run_qwen_single_shard(args: argparse.Namespace, shard_id: int, num_shards: int) -> dict:
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    configure_torch_runtime()
    qwen_device_map = parse_qwen_device_map(args.qwen_device_map)
    qwen_max_memory = parse_qwen_max_memory(args.qwen_max_memory)
    processor, model = load_frozen_qwen3vl(
        args.model_path,
        dtype,
        device,
        args.attn_implementation,
        device_map=qwen_device_map,
        max_memory=qwen_max_memory,
    )
    if qwen_device_map is not None:
        device = qwen_input_device(model)
        if shard_id == 0:
            print(f"qwen_device_map={qwen_device_map} input_device={device} max_memory={qwen_max_memory or 'auto'}", flush=True)
    if args.teacher_only:
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
        print(f"Shard {shard_id}: samples [{start}, {end}) = {len(dataset)} items; mode=teacher_only", flush=True)
        result = evaluate_qwen_teacher_shard(
            model,
            processor,
            dataset,
            device,
            args.log_every,
            args.max_new_tokens,
            args.benchmark,
            args.measure_prefill,
            dtype,
            last_logits_only=bool(args.last_logits_only),
        )
        out_path = Path(args.output_dir) / f"shard_{shard_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Shard {shard_id} done. Saved to {out_path}", flush=True)
        return result
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --teacher-only is set")
    adapter, meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    if args.output_mode is not None:
        if not is_embedding_adapter_mode(args.output_mode):
            raise ValueError(f"Qwen only supports embedding adapter output modes, got {args.output_mode!r}")
        adapter.mode = args.output_mode
    if meta["missing"] or meta["unexpected"]:
        print(f"checkpoint load missing={meta['missing']} unexpected={meta['unexpected']}", flush=True)
    adapter_logits_fn = build_qwen_adapter_logits_fn(
        model,
        adapter,
        compile_adapter=bool(args.compile_adapter),
        compile_mode=args.compile_mode,
        compile_dynamic=bool(args.compile_dynamic),
        last_logits_only=bool(args.last_logits_only),
        compile_verify=bool(args.compile_verify),
        compile_max_diff=float(args.compile_max_diff),
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
        eval_batch_size=int(args.eval_batch_size),
        eval_max_batch_tokens=int(args.eval_max_batch_tokens),
        eval_bucket_by_length=bool(args.eval_bucket_by_length),
        verify_batched_generation=int(args.verify_batched_generation),
        adapter_decode_cache=bool(args.adapter_decode_cache),
        adapter_decode_cache_mode=str(args.adapter_decode_cache_mode),
        verify_decode_cache_generation=int(args.verify_decode_cache_generation),
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

    apply_deferred_llm_judge(benchmark=benchmark, predictions=all_predictions, log_every=25)
    merged = summarize_benchmark_predictions(
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


def run(args: argparse.Namespace) -> None:
    set_global_seed(int(args.seed))
    if args.shard_id is not None:
        if args.model_kind == "qwen":
            run_qwen_single_shard(args, args.shard_id, args.num_shards)
        else:
            run_llava_single_shard(args, args.shard_id, args.num_shards)
    else:
        merge_benchmark_shards(args.output_dir, args.num_shards, args.benchmark)


def main(argv=None) -> None:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
