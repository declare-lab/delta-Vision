"""ChartQA relaxed accuracy and the DocVQA/InfographicVQA ANLS reference metric.

References:
https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/chartqa/utils.py
https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py
ANLS preserves the reference's raw-string length denominator and inclusive 0.5
similarity threshold. Do not apply generic VQA punctuation/article normalization.
"""
from __future__ import annotations


def relaxed_correctness(prediction: str, target: str) -> float:
    def number(text):
        try:
            return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
        except ValueError:
            return None
    pred_number, target_number = number(prediction), number(target)
    if pred_number is not None and target_number:
        return float(abs(pred_number-target_number) / abs(target_number) <= .05)
    return float(prediction.lower() == target.lower())


def edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a)+1))
    for j, right in enumerate(b, 1):
        current = [j]
        for i, left in enumerate(a, 1):
            current.append(previous[i-1] if left == right else 1+min(previous[i-1], previous[i], current[-1]))
        previous = current
    return previous[-1]


def anls(prediction: str, targets: list[str]) -> float:
    if not targets:
        raise ValueError('ANLS requires reference answers')
    pred = ' '.join(prediction.strip().lower().split())
    values = []
    for target in targets:
        gt = ' '.join(target.strip().lower().split())
        length = max(len(target.upper()), len(prediction.upper()))
        values.append(0.0 if length == 0 else edit_distance(gt, pred) / length)
    score = 1-min(values)
    return score if score >= .5 else 0.0
