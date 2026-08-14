"""Benchmark registry, prompt formatting, and lightweight metrics."""
from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    metric: str
    default_data: str
    answer_instruction: str
    max_new_tokens: int = 16


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "mmstar": BenchmarkSpec(
        name="mmstar",
        display_name="MMStar",
        metric="multi_choice",
        default_data="data/mmstar/mmstar_val.jsonl",
        answer_instruction="Answer directly with only the letter of the correct option.",
        max_new_tokens=8,
    ),
    "gqa": BenchmarkSpec(
        name="gqa",
        display_name="GQA",
        metric="exact",
        default_data="data/benchmarks/gqa/testdev_balanced.jsonl",
        answer_instruction="Answer directly with a short phrase.",
    ),
    "mmb": BenchmarkSpec(
        name="mmb",
        display_name="MMB",
        metric="multi_choice",
        default_data="data/benchmarks/mmb/dev.jsonl",
        answer_instruction="Answer directly with only the letter of the correct option.",
        max_new_tokens=8,
    ),
    "mmb-cn": BenchmarkSpec(
        name="mmb-cn",
        display_name="MMB-CN",
        metric="multi_choice",
        default_data="data/benchmarks/mmb-cn/dev.jsonl",
        answer_instruction="请直接回答正确选项的字母。",
        max_new_tokens=8,
    ),
    "mme": BenchmarkSpec(
        name="mme",
        display_name="MME",
        metric="mme",
        default_data="data/benchmarks/mme/test.jsonl",
        answer_instruction="Answer directly with yes or no.",
        max_new_tokens=8,
    ),
    "pope": BenchmarkSpec(
        name="pope",
        display_name="POPE",
        metric="pope_f1",
        default_data="data/benchmarks/pope/test.jsonl",
        answer_instruction="Answer directly with yes or no.",
        max_new_tokens=8,
    ),
    "sqa": BenchmarkSpec(
        name="sqa",
        display_name="SQA",
        metric="multi_choice",
        default_data="data/benchmarks/sqa/test.jsonl",
        answer_instruction="Answer directly with only the letter of the correct option.",
        max_new_tokens=8,
    ),
    "vqav2": BenchmarkSpec(
        name="vqav2",
        display_name="VQA-v2",
        metric="vqa",
        default_data="data/benchmarks/vqav2/validation.jsonl",
        answer_instruction="Answer directly with a short phrase.",
    ),
    "textvqa": BenchmarkSpec(
        name="textvqa",
        display_name="TextVQA",
        metric="vqa",
        default_data="data/benchmarks/textvqa/validation.jsonl",
        answer_instruction="Answer directly with a short phrase.",
    ),
    "vizwiz": BenchmarkSpec(
        name="vizwiz",
        display_name="VizWiz",
        metric="vqa",
        default_data="data/benchmarks/vizwiz/val.jsonl",
        answer_instruction="Answer directly with a short phrase.",
    ),
    "ocrbench": BenchmarkSpec(
        name="ocrbench",
        display_name="OCRBench",
        metric="relaxed_exact",
        default_data="data/benchmarks/ocrbench/test.jsonl",
        answer_instruction="Answer directly with a short phrase.",
    ),
}


def canonical_benchmark_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    aliases = {
        "mmbench": "mmb",
        "mmb-en": "mmb",
        "mmbench-en": "mmb",
        "mmbench-cn": "mmb-cn",
        "sciqa": "sqa",
        "scienceqa": "sqa",
        "vqa": "vqav2",
        "vqa-v2": "vqav2",
        "vqa^v2": "vqav2",
        "vqa-text": "textvqa",
        "vqa^text": "textvqa",
        "text-vqa": "textvqa",
        "ocr": "ocrbench",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARK_SPECS:
        raise ValueError(f"Unsupported benchmark={name!r}; choose from {sorted(BENCHMARK_SPECS)}")
    return key


def all_benchmark_names() -> list[str]:
    return list(BENCHMARK_SPECS)


def parse_benchmark_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return all_benchmark_names()
    if isinstance(value, str):
        raw = re.split(r"[\s,]+", value.strip())
    else:
        raw = []
        for item in value:
            raw.extend(re.split(r"[\s,]+", str(item).strip()))
    names = [item for item in raw if item]
    if not names or any(item.lower() == "all" for item in names):
        return all_benchmark_names()
    parsed: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = canonical_benchmark_name(name)
        if canonical not in seen:
            parsed.append(canonical)
            seen.add(canonical)
    return parsed


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    return BENCHMARK_SPECS[canonical_benchmark_name(name)]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_benchmark_prompt(row: dict[str, Any], spec: BenchmarkSpec, answer_instruction: str | None = None) -> str:
    question = _stringify(row.get("question"))
    hint = _stringify(row.get("hint"))
    if hint:
        question = f"{hint}\n{question}" if question else hint

    choices = row.get("choices") or []
    if isinstance(choices, dict):
        choices = [choices[key] for key in sorted(choices) if _stringify(choices[key])]
    if choices and not _question_has_lettered_choices(question, len(choices)):
        lines = [question]
        for idx, choice in enumerate(choices):
            lines.append(f"{string.ascii_uppercase[idx]}. {_stringify(choice)}")
        question = "\n".join(lines)

    instruction = spec.answer_instruction if answer_instruction is None else answer_instruction.strip()
    if instruction:
        question = f"{question}\n{instruction}"
    return question


def _question_has_lettered_choices(question: str, num_choices: int) -> bool:
    upper = question.upper()
    for idx in range(min(num_choices, 6)):
        letter = string.ascii_uppercase[idx]
        if re.search(rf"(?:^|\n)\s*{letter}\s*[\.\):：、]", upper):
            return True
    return False


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: Any) -> str:
    text = _stringify(text).lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.translate(_PUNCT_TABLE)
    words = [word for word in text.split() if word not in _ARTICLES]
    return " ".join(words)


def extract_yes_no(text: str) -> str | None:
    clean = text.strip().lower()
    match = re.search(r"\b(yes|no)\b", clean)
    if match:
        return match.group(1)
    if clean.startswith(("是", "对", "有")):
        return "yes"
    if clean.startswith(("否", "不", "没有")):
        return "no"
    return None


def _choice_letters(num_choices: int) -> list[str]:
    return list(string.ascii_uppercase[: max(0, min(num_choices, 26))])


def canonical_choice(value: Any, choices: list[Any] | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return string.ascii_uppercase[value] if 0 <= value < 26 else None
    text = _stringify(value)
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 1 and upper in string.ascii_uppercase:
        return upper
    match = re.match(r"^\s*([A-Z])\s*[\.\):：、]?", upper)
    if match:
        return match.group(1)
    if choices:
        norm = normalize_answer(text)
        for idx, choice in enumerate(choices):
            if norm and norm == normalize_answer(choice):
                return string.ascii_uppercase[idx]
    return None


def extract_choice(text: str, choices: list[Any] | None = None) -> str | None:
    num_choices = len(choices or []) or 6
    letters = _choice_letters(num_choices)
    clean = text.strip()
    upper = clean.upper()
    patterns = [
        r"(?:ANSWER|OPTION|CHOICE|答案|选项)\s*(?:IS|是|:|：)?\s*[\(\[]?\s*([A-Z])(?:\b|[\)\]\.。,:：])",
        r"^[\s\(\[]*([A-Z])(?:[\)\]\.。,:：\s]|$)",
        r"(?<![A-Z])([A-Z])(?![A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match and match.group(1) in letters:
            return match.group(1)
    if choices:
        norm = normalize_answer(clean)
        for idx, choice in enumerate(choices):
            choice_norm = normalize_answer(choice)
            if choice_norm and (norm == choice_norm or choice_norm in norm):
                return string.ascii_uppercase[idx]
    return None


def _answer_list(answer: Any, answers: Any = None) -> list[Any]:
    values: list[Any] = []
    if answers is not None:
        if isinstance(answers, list):
            for item in answers:
                if isinstance(item, dict):
                    values.append(item.get("answer", item.get("text", item.get("label"))))
                else:
                    values.append(item)
        else:
            values.append(answers)
    if answer is not None:
        if isinstance(answer, list):
            values.extend(answer)
        else:
            values.append(answer)
    return [value for value in values if _stringify(value)]


def score_prediction(
    *,
    metric: str,
    prediction_text: str,
    answer: Any,
    answers: Any = None,
    choices: list[Any] | None = None,
) -> dict[str, Any]:
    if metric == "multi_choice":
        pred = extract_choice(prediction_text, choices)
        gold = canonical_choice(answer, choices)
        return {
            "prediction": pred,
            "gold": gold,
            "score": float(pred is not None and gold is not None and pred == gold),
            "invalid": pred is None,
        }

    if metric in {"pope_f1", "mme"}:
        pred = extract_yes_no(prediction_text)
        gold = extract_yes_no(_stringify(answer))
        return {
            "prediction": pred,
            "gold": gold,
            "score": float(pred is not None and gold is not None and pred == gold),
            "invalid": pred is None,
        }

    pred_norm = normalize_answer(prediction_text)
    if metric == "vqa":
        gold_values = _answer_list(None, answers) if answers is not None else _answer_list(answer)
        counts = Counter(normalize_answer(value) for value in gold_values)
        score = min(1.0, counts.get(pred_norm, 0) / 3.0) if counts else 0.0
        gold = normalize_answer(answer if answer is not None else (gold_values[0] if gold_values else ""))
        return {"prediction": pred_norm, "gold": gold, "score": score, "invalid": not bool(pred_norm)}

    gold_values = _answer_list(answer, answers)
    gold_norms = [normalize_answer(value) for value in gold_values]
    if metric == "relaxed_exact":
        score = 0.0
        for gold in gold_norms:
            if gold and (pred_norm == gold or gold in pred_norm):
                score = 1.0
                break
    else:
        score = float(bool(pred_norm) and pred_norm in set(gold_norms))
    return {
        "prediction": pred_norm,
        "gold": gold_norms[0] if gold_norms else "",
        "score": score,
        "invalid": not bool(pred_norm),
    }


def summarize_metric(metric: str, scored: list[dict[str, Any]], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    total = len(scored)
    invalid = sum(1 for item in scored if item.get("invalid"))
    score = sum(float(item.get("score", 0.0)) for item in scored) / max(total, 1)
    summary: dict[str, Any] = {
        "samples": total,
        "score": score,
        "accuracy": score,
        "invalid_rate": invalid / max(total, 1),
    }
    if metric == "pope_f1":
        tp = fp = fn = tn = 0
        for item in scored:
            pred = item.get("prediction")
            gold = item.get("gold")
            if pred == "yes" and gold == "yes":
                tp += 1
            elif pred == "yes" and gold == "no":
                fp += 1
            elif pred == "no" and gold == "yes":
                fn += 1
            elif pred == "no" and gold == "no":
                tn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        summary.update({"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    if metric == "mme" and rows is not None:
        summary.update(_summarize_mme(scored, rows))
    return summary


def _summarize_mme(scored: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[int]] = defaultdict(list)
    pair_hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item, row in zip(scored, rows):
        category = _stringify(row.get("category")) or "unknown"
        hit = int(float(item.get("score", 0.0)) > 0.0)
        by_category[category].append(hit)
        pair_key = _mme_pair_key(row)
        pair_hits[(category, pair_key)].append(hit)

    category_scores: dict[str, dict[str, float]] = {}
    total_score = 0.0
    for category, hits in by_category.items():
        acc = 100.0 * sum(hits) / max(len(hits), 1)
        pairs = [values for (cat, _), values in pair_hits.items() if cat == category]
        acc_plus = 100.0 * sum(1 for values in pairs if values and all(values)) / max(len(pairs), 1)
        category_score = acc + acc_plus
        total_score += category_score
        category_scores[category] = {"acc": acc, "acc_plus": acc_plus, "score": category_score}
    return {
        "mme_score": total_score,
        "mme_category_scores": category_scores,
    }


def _mme_pair_key(row: dict[str, Any]) -> str:
    for key in ("pair_id", "image_id", "image", "question_id", "index"):
        value = _stringify(row.get(key))
        if value:
            if key == "question_id":
                value = re.sub(r"[_-]?\d+$", "", value)
            return value
    return "unknown"


def estimate_qwen_kv_cache_mb(
    language_config: Any,
    *,
    text_tokens: int,
    image_tokens: int,
    dtype_bytes: int,
    adapter: bool,
) -> float:
    layers = int(getattr(language_config, "num_hidden_layers", 0))
    kv_heads = int(getattr(language_config, "num_key_value_heads", getattr(language_config, "num_attention_heads", 0)))
    head_dim = int(getattr(language_config, "head_dim", language_config.hidden_size // language_config.num_attention_heads))
    hidden_size = int(getattr(language_config, "hidden_size"))
    seq_tokens = int(text_tokens) if adapter else int(text_tokens) + int(image_tokens)
    language_kv = layers * seq_tokens * kv_heads * head_dim * 2 * dtype_bytes
    visual_memory = int(image_tokens) * hidden_size * dtype_bytes if adapter else 0
    return (language_kv + visual_memory) / (1024.0**2)


def estimate_qwen_prefill_flops(
    language_config: Any,
    *,
    text_tokens: int,
    image_tokens: int,
    adapter_mode: str | None = None,
    visual_adapter_rank: int = 0,
) -> float:
    layers = int(getattr(language_config, "num_hidden_layers", 0))
    hidden = int(getattr(language_config, "hidden_size"))
    heads = int(getattr(language_config, "num_attention_heads"))
    kv_heads = int(getattr(language_config, "num_key_value_heads", heads))
    head_dim = int(getattr(language_config, "head_dim", hidden // heads))
    intermediate = int(getattr(language_config, "intermediate_size", hidden * 4))

    def llm(seq_len: int) -> float:
        qkv_o_macs = hidden * (heads * head_dim + 2 * kv_heads * head_dim + heads * head_dim)
        mlp_macs = 3 * hidden * intermediate
        linear = 2.0 * seq_len * layers * (qkv_o_macs + mlp_macs)
        attention = 4.0 * layers * heads * (seq_len**2) * head_dim
        return linear + attention

    if not adapter_mode:
        return llm(text_tokens + image_tokens)

    flops = llm(text_tokens)
    visual_kv = 4.0 * layers * image_tokens * hidden * kv_heads * head_dim
    cross_attention = 4.0 * layers * heads * text_tokens * image_tokens * head_dim
    flops += visual_kv + cross_attention
    if "split" in adapter_mode:
        flops += llm(text_tokens) * 0.45
    if visual_adapter_rank > 0:
        flops += 4.0 * layers * image_tokens * hidden * visual_adapter_rank
    return flops


def format_seconds_minsec(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}:{remain:05.2f}"


def safe_mean(values: list[float]) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(values) / max(len(values), 1)
