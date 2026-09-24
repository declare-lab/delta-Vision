"""Original text-reconstruction metrics required by the Figure 3 oracle helper."""

from __future__ import annotations

COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."

def normalized(text: str) -> str:
    return " ".join(str(text).strip().split())

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]

def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalized(prediction).split()
    ref_tokens = normalized(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in ref_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)

def score_text(prediction: str, reference: str) -> dict[str, float]:
    ref = str(reference)
    pred = str(prediction)
    cer = levenshtein(pred, ref) / max(1, len(ref))
    return {
        "cer": float(cer),
        "char_acc": float(max(0.0, 1.0 - cer)),
        "token_f1": float(token_f1(pred, ref)),
        "exact": float(normalized(pred) == normalized(ref)),
        "pred_chars": float(len(pred)),
        "ref_chars": float(len(ref)),
    }

def mean_metric(items: list[dict[str, float]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item[key]) for item in items) / len(items))

def summarize(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "samples": float(len(items)),
        "cer": mean_metric(items, "cer"),
        "char_acc": mean_metric(items, "char_acc"),
        "token_f1": mean_metric(items, "token_f1"),
        "exact": mean_metric(items, "exact"),
        "pred_chars": mean_metric(items, "pred_chars"),
        "ref_chars": mean_metric(items, "ref_chars"),
        "pred_tokens": mean_metric(items, "pred_tokens") if "pred_tokens" in items[0] else 0.0,
        "hit_max_new_tokens": mean_metric(items, "hit_max_new_tokens") if "hit_max_new_tokens" in items[0] else 0.0,
    }
