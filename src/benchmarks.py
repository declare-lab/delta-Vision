"""Benchmark definitions, prompts, answer extraction and scoring metrics."""
from __future__ import annotations


# ChartQA relaxed accuracy and the DocVQA/InfographicVQA ANLS reference metric.
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


# Benchmark registry, prompt formatting, and lightweight metrics.
import math
import re
import string
from collections import defaultdict
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
    "chartqa": BenchmarkSpec(
        name="chartqa", display_name="ChartQA", metric="chartqa_relaxed",
        default_data="data/benchmarks/chartqa/eval1000_seed42.jsonl",
        answer_instruction="Answer the question using a single word or phrase.", max_new_tokens=128,
    ),
    "docvqa": BenchmarkSpec(
        name="docvqa", display_name="DocVQA", metric="anls",
        default_data="data/benchmarks/docvqa/eval1000_seed42.jsonl",
        answer_instruction="Answer the question using a single word or phrase.", max_new_tokens=128,
    ),
    "infographicvqa": BenchmarkSpec(
        name="infographicvqa", display_name="InfographicVQA", metric="anls",
        default_data="data/benchmarks/infographicvqa/eval1000_seed42.jsonl",
        answer_instruction="Answer the question using a single word or phrase.", max_new_tokens=128,
    ),
    "videomme": BenchmarkSpec(
        name="videomme", display_name="Video-MME", metric="multi_choice",
        default_data="data/benchmarks/videomme/test.jsonl",
        answer_instruction="Answer with the option's letter from the given choices directly.",
        max_new_tokens=16,
    ),
    "mmstar": BenchmarkSpec(
        name="mmstar",
        display_name="MMStar",
        metric="multi_choice",
        default_data="data/benchmarks/mmstar/mmstar_val.jsonl",
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
        answer_instruction="Answer with only one word or a short phrase. Do not explain your answer.",
    ),
    "realworldqa": BenchmarkSpec(
        name="realworldqa",
        display_name="RealWorldQA",
        metric="realworldqa",
        default_data="data/benchmarks/realworldqa/test.jsonl",
        answer_instruction="Answer directly with the final answer only.",
        max_new_tokens=8,
    ),
}


DEFAULT_BENCHMARK_NAMES: tuple[str, ...] = (
    "mmstar",
    "gqa",
    "mmb",
    "mmb-cn",
    "mme",
    "pope",
    "sqa",
    "vqav2",
    "realworldqa",
)


def canonical_benchmark_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    aliases = {
        "chart-qa": "chartqa",
        "doc-vqa": "docvqa",
        "infovqa": "infographicvqa",
        "infographic-vqa": "infographicvqa",
        "video-mme": "videomme",
        "mmbench": "mmb",
        "mmb-en": "mmb",
        "mmbench-en": "mmb",
        "mmbench-cn": "mmb-cn",
        "sciqa": "sqa",
        "scienceqa": "sqa",
        "vqa": "vqav2",
        "vqa-v2": "vqav2",
        "vqa^v2": "vqav2",
        "realworld": "realworldqa",
        "real-world-qa": "realworldqa",
        "real-worldqa": "realworldqa",
        "rwqa": "realworldqa",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARK_SPECS:
        raise ValueError(f"Unsupported benchmark={name!r}; choose from {sorted(BENCHMARK_SPECS)}")
    return key


def all_benchmark_names() -> list[str]:
    return list(DEFAULT_BENCHMARK_NAMES)


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


def _choice_list(choices: Any) -> list[Any] | None:
    if choices is None:
        return None
    if isinstance(choices, dict):
        return [choices[key] for key in sorted(choices) if _stringify(choices[key])]
    if isinstance(choices, (list, tuple)):
        return list(choices)
    return None


def build_benchmark_prompt(row: dict[str, Any], spec: BenchmarkSpec, answer_instruction: str | None = None) -> str:
    question = _stringify(row.get("question"))
    hint = _stringify(row.get("hint"))
    if hint:
        question = f"{hint}\n{question}" if question else hint

    choices = _choice_list(row.get("choices")) or []
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
_VQA_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VQA_COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
_VQA_PUNCTUATIONS = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_answer(text: Any) -> str:
    text = _stringify(text).lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.translate(_PUNCT_TABLE)
    words = [word for word in text.split() if word not in _ARTICLES]
    return " ".join(words)


def normalize_vqa_answer(text: Any) -> str:
    """EvalAI-style answer normalization used by VQAv2."""
    out = _stringify(text).lower().replace(",", "").replace("?", "").replace("'s", " 's")
    out = out.replace("\n", " ").replace("\t", " ").strip()
    for punct in _VQA_PUNCTUATIONS:
        if (punct + " " in out or " " + punct in out) or re.search(_VQA_COMMA_STRIP, out):
            out = out.replace(punct, "")
        else:
            out = out.replace(punct, " ")
    out = _VQA_PERIOD_STRIP.sub("", out)
    words = []
    for word in out.lower().split():
        word = _VQA_NUMBER_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_VQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_consensus_score(prediction: Any, answers: list[Any]) -> float:
    pred_norm = normalize_vqa_answer(prediction)
    gold_norms = [normalize_vqa_answer(value) for value in answers if _stringify(value)]
    if not gold_norms:
        return 0.0
    scores = []
    for idx, _ in enumerate(gold_norms):
        other_answers = gold_norms[:idx] + gold_norms[idx + 1 :]
        matches = sum(1 for value in other_answers if value == pred_norm)
        scores.append(min(1.0, float(matches) / 3.0))
    return sum(scores) / max(len(scores), 1)


def _answer_surface(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].replace('**', '').replace('__', '').replace('`', '').strip()


def _final_answer_segment(text: str) -> str:
    """Select the last explicit answer/correction without consulting the gold."""
    clean = _answer_surface(text)
    markers = list(re.finditer(
        r"(?i:\b(?:final\s+|correct\s+)?answer\b|\bcorrection\b|正确答案|最终答案|答案|更正)"
        r"\s*(?:(?i:is)\b\s*[:：]?|(?:是|为)\s*[:：]?|[:：])\s*", clean))
    return clean[markers[-1].end():].strip() if markers else clean


def extract_yes_no(text: str) -> str | None:
    clean = _final_answer_segment(text).lower()
    found = set(re.findall(r"\b(yes|no)\b", clean))
    if found:
        return next(iter(found)) if len(found) == 1 else None
    if clean.startswith(("是", "对", "有")):
        return "yes"
    if clean.startswith(("否", "不", "没有")):
        return "no"
    return None


def _choice_letters(num_choices: int) -> list[str]:
    return list(string.ascii_uppercase[: max(0, min(num_choices, 26))])


def canonical_choice(value: Any, choices: list[Any] | None = None) -> str | None:
    choices = _choice_list(choices)
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
    # A textual answer such as "blue" must not be interpreted as option B.
    match = re.fullmatch(r"\s*[\(\[]?([A-Z])[\)\]]?\s*[\.\):：、]?\s*", upper)
    if match:
        return match.group(1)
    if choices:
        norm = normalize_answer(text)
        for idx, choice in enumerate(choices):
            if norm and norm == normalize_answer(choice):
                return string.ascii_uppercase[idx]
    return None


def extract_choice(text: str, choices: list[Any] | None = None) -> str | None:
    """Read an explicit answer label or an unambiguous whole option text.

    Never uppercase prose and search it for isolated letters: the article 'a'
    is not an answer A. Merely mentioning an option inside reasoning also does
    not constitute selecting that option.
    """
    choices = _choice_list(choices)
    num_choices = len(choices or []) or 6
    letters = _choice_letters(num_choices)
    clean = _answer_surface(text)
    # Permit normal Markdown answer formatting, e.g. **B** or `B`.
    clean = clean.replace('**', '').replace('__', '').replace('`', '').strip()
    explicit = list(re.finditer(
        r"(?i:\b(?:final\s+)?answer\b|\b(?:correct|final)\s+(?:option|choice)\b|正确答案|最终答案|正确选项|答案)"
        r"\s*(?:(?i:is)|是|为)?\s*[:：]?\s*[\(\[]?\s*([A-Z])(?:\b|[\)\]\.。,:：、])", clean))
    valid = [(match.start(1), match.group(1)) for match in explicit if match.group(1) in letters]
    # Final standalone answer labels can follow a paragraph of reasoning.
    # Never scan arbitrary intermediate sentences for single letters.
    last_line = next((line.strip() for line in reversed(clean.splitlines()) if line.strip()), '')
    labeled_lines = re.findall(r"(?m)^\s*[\(\[]?([A-Za-z])[\)\]\.。,:：、]", clean)
    for candidate, offset in ((last_line, clean.rfind(last_line)), (clean, 0)):
        match = re.match(r"^\s*[\(\[]?([A-Za-z])(?:[\)\]\.。,:：、]|\s*$|\s*\n)", candidate)
        if match and match.group(1).upper() in letters:
            # A reproduced list of alternatives is not a chosen answer.
            bare_label = re.fullmatch(r"[\(\[]?[A-Za-z][\)\]\.。]?", candidate.strip())
            if len(set(labeled_lines)) <= 1 or bare_label:
                valid.append((offset + match.start(1), match.group(1).upper()))
    if valid:
        return max(valid, key=lambda item: item[0])[1]
    if choices:
        norm = normalize_answer(clean)
        matches = [letters[idx] for idx, choice in enumerate(choices[:len(letters)])
                   if normalize_answer(choice) and norm == normalize_answer(choice)]
        if len(matches) == 1:
            return matches[0]
    return None


def _realworldqa_choices_from_question(question: Any) -> list[str]:
    found: dict[str, str] = {}
    for letter, text in re.findall(r"(?m)^\s*([A-D])(?:[\.\):：、]\s*|[ \t]+)(.+?)\s*$", _stringify(question)):
        found[letter.upper()] = text.strip()
    return [found[letter] for letter in "ABCD" if letter in found]


def score_realworldqa_prediction(
    prediction_text: str,
    answer: Any,
    choices: list[Any] | None = None,
    question: Any = None,
) -> dict[str, Any]:
    """RealWorldQA: explicit option labels or unambiguous whole-answer matches.

    Never match a gold option letter inside prose or a number inside another
    number. This parser is local, not a claim of an official scoring protocol.
    """
    choices = _choice_list(choices) or _realworldqa_choices_from_question(question)
    pred_text = _stringify(prediction_text).strip()
    gold_text = _stringify(answer).strip()
    clean = _final_answer_segment(pred_text)
    pred_norm = normalize_answer(clean)
    gold_norm = normalize_answer(gold_text)

    def contains_phrase(text, phrase):
        return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))

    if choices:
        letters = _choice_letters(len(choices))
        gold_letter = gold_text.upper() if gold_text.upper() in letters else None
        if gold_letter is None:
            matches = [letters[i] for i, c in enumerate(choices)
                       if normalize_answer(c) == gold_norm and gold_norm]
            gold_letter = matches[0] if len(matches) == 1 else None
        pred_letter = extract_choice(clean, choices)
        return {"prediction": pred_letter, "gold": gold_letter or gold_text,
                "score": float(pred_letter is not None and pred_letter == gold_letter),
                "invalid": pred_letter is None}

    if gold_text.lower() in {"yes", "no"}:
        prediction = extract_yes_no(clean)
        return {"prediction": prediction, "gold": gold_text.lower(),
                "score": float(prediction == gold_text.lower()), "invalid": prediction is None}
    # Preserve short explanations, but do not award points for a negated answer
    # or an unresolved enumeration of multiple different numeric answers.
    negated = bool(gold_norm and re.search(
        r"\b(?:not|no|isnt|arent)\s+(?:(?:a|an|the)\s+)?" + re.escape(gold_norm) + r"\b", pred_norm))
    conflicting_numbers = bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", gold_text) and
                               len(set(re.findall(r"[+-]?\d+(?:\.\d+)?", clean))) > 1)
    return {"prediction": pred_norm, "gold": gold_norm,
            "score": float(not negated and not conflicting_numbers and contains_phrase(pred_norm, gold_norm)),
            "invalid": not bool(pred_norm) or negated or conflicting_numbers}


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
    question: Any = None,
) -> dict[str, Any]:
    choices = _choice_list(choices)
    if metric in {"chartqa_relaxed", "anls"}:
        from src.benchmarks import anls, relaxed_correctness
        targets = [str(value) for value in _answer_list(answer, answers)]
        if not targets:
            raise ValueError(f"{metric} requires public reference answers")
        score = (anls(prediction_text, targets) if metric == "anls" else
                 max(relaxed_correctness(prediction_text, target) for target in targets))
        return {"prediction": prediction_text, "gold": targets, "score": score,
                "invalid": not bool(prediction_text.strip())}
    if metric == "realworldqa":
        return score_realworldqa_prediction(
            prediction_text=prediction_text,
            answer=answer,
            choices=choices,
            question=question,
        )

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

    answer_segment = _final_answer_segment(prediction_text)
    pred_norm = normalize_answer(answer_segment)
    if metric == "vqa":
        gold_values = _answer_list(None, answers) if answers is not None else _answer_list(answer)
        pred_vqa = normalize_vqa_answer(answer_segment)
        score = vqa_consensus_score(answer_segment, gold_values)
        gold = normalize_vqa_answer(answer if answer is not None else (gold_values[0] if gold_values else ""))
        return {"prediction": pred_vqa, "gold": gold, "score": score, "invalid": not bool(pred_vqa)}

    if metric == "llm_judge":
        pred = _stringify(prediction_text)
        gold = _stringify(answer if answer is not None else (answers[0] if isinstance(answers, list) and answers else ""))
        return {"prediction": pred, "gold": gold, "score": 0.0, "invalid": not bool(pred), "needs_judge": True}

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
    question_accuracy = sum(float(item.get("score", 0.0)) for item in scored) / max(total, 1)
    summary: dict[str, Any] = {
        "samples": total,
        "score": question_accuracy,
        "accuracy": question_accuracy,
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
        mme_summary = _summarize_mme(scored, rows)
        summary.update(mme_summary)
        summary["question_accuracy"] = question_accuracy
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
    max_score = 200.0 * max(len(category_scores), 1)
    return {
        "mme_score": total_score,
        "mme_score_max": max_score,
        "mme_score_normalized": total_score / max_score,
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
