from __future__ import annotations

import base64
import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass
class EvalSample:
    uid: str
    question: str
    images: list[Any]
    choices: list[str] = field(default_factory=list)
    answer: str | None = None
    category: str = "default"
    meta: dict[str, Any] = field(default_factory=dict)


class EvalAdapter:
    name = "base"

    def __init__(self, data: str, split: str = "test", image_root: str | None = None):
        self.data, self.split = data, split
        self.image_root = Path(image_root) if image_root else None

    def __iter__(self) -> Iterable[EvalSample]:
        raise NotImplementedError

    def summarize(self, records: list[dict]) -> dict:
        valid = [r for r in records if r.get("correct") is not None]
        return {
            "samples": len(records), "scored": len(valid),
            "accuracy": sum(bool(r["correct"]) for r in valid) / len(valid) if valid else None,
        }


class GenericJsonlAdapter(EvalAdapter):
    name = "jsonl"

    def __iter__(self):
        root = self.image_root or Path(self.data).parent
        with open(self.data, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                images = row.get("images") or ([row["image"]] if row.get("image") else [])
                images = [str(root / x) if not Path(x).is_absolute() else x for x in images]
                yield EvalSample(
                    uid=str(row.get("id", index)), question=row["question"], images=images,
                    choices=list(row.get("choices", [])), answer=row.get("answer"),
                    category=str(row.get("category", "default")), meta=row,
                )


class MMBenchAdapter(EvalAdapter):
    name = "mmbench"

    @staticmethod
    def _sample(row, index):
        def present(value):
            return value is not None and str(value).strip().lower() not in ("", "nan", "none")

        choices = [str(row[k]) for k in ("A", "B", "C", "D", "E") if k in row and present(row[k])]
        hint = row.get("hint")
        question = f"{hint}\n{row['question']}" if present(hint) else row["question"]
        raw_image = row["image"]
        if isinstance(raw_image, dict):
            if raw_image.get("bytes") is not None:
                image = Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
            elif raw_image.get("path"):
                image = Image.open(raw_image["path"]).convert("RGB")
            else:
                raise ValueError(f"MMBench sample {index} has an empty image struct")
        elif isinstance(raw_image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(raw_image)).convert("RGB")
        else:
            image = Image.open(io.BytesIO(base64.b64decode(raw_image))).convert("RGB")
        meta = dict(row)
        meta.pop("image", None)
        if "L2-category" in meta:
            meta["l2_category"] = meta["L2-category"]
        return EvalSample(
            str(row.get("index", index)), question, [image], choices,
            row.get("answer") if present(row.get("answer")) else None,
            str(row.get("category", "default")), meta,
        )

    def _iter_tsv(self, path: Path):
        with open(self.data, encoding="utf-8", newline="") as handle:
            csv.field_size_limit(sys.maxsize)
            rows = csv.DictReader(handle, delimiter="\t")
            for index, row in enumerate(rows):
                yield self._sample(row, index)

    def _parquet_files(self, path: Path):
        if path.is_file():
            return [path]
        if self.split == "chinese_culture":
            pattern = path / "chinese_culture" / "test-*.parquet"
        elif self.split in ("dev", "test"):
            pattern = path / "data" / f"{self.split}-*.parquet"
        else:
            raise ValueError("MMBench directory split must be dev, test, or chinese_culture")
        files = sorted(pattern.parent.glob(pattern.name))
        if not files:
            raise FileNotFoundError(f"No MMBench parquet files matched {pattern}")
        return files

    def _iter_parquet(self, path: Path):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet MMBench requires pyarrow") from exc
        index = 0
        for parquet_path in self._parquet_files(path):
            for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    yield self._sample(row, index)
                    index += 1

    def __iter__(self):
        path = Path(self.data)
        if path.is_dir() or path.suffix.lower() == ".parquet":
            yield from self._iter_parquet(path)
        else:
            yield from self._iter_tsv(path)


class MMStarAdapter(EvalAdapter):
    name = "mmstar"

    @staticmethod
    def _split_question_choices(text: str) -> tuple[str, list[str]]:
        """Parse both MMStar's inline Options and multiline Choices layouts."""
        if "Options:" in text:
            question, raw_choices = text.rsplit("Options:", 1)
            matches = list(re.finditer(r"(?:^|,\s+)([A-E]):\s+", raw_choices.strip()))
        else:
            if "Choices:" in text:
                question, raw_choices = text.rsplit("Choices:", 1)
            else:
                first_choice = re.search(r"(?m)^\s*\([A-E]\)\s*", text)
                if first_choice is None:
                    raise ValueError("MMStar question has no recognizable choices")
                question, raw_choices = text[:first_choice.start()], text[first_choice.start():]
            matches = list(re.finditer(r"(?m)^\s*\(([A-E])\)\s*", raw_choices))
        if len(matches) < 2:
            raise ValueError(f"Could not parse MMStar choices from: {text[:200]!r}")
        choices = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_choices)
            choices.append(raw_choices[start:end].strip().rstrip(","))
        return question.strip(), choices

    def __iter__(self):
        csv.field_size_limit(sys.maxsize)
        with open(self.data, encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            required = {"index", "question", "answer", "category", "l2_category", "image"}
            missing = required.difference(rows.fieldnames or ())
            if missing:
                raise ValueError(f"MMStar TSV is missing required columns: {sorted(missing)}")
            for row_number, row in enumerate(rows, 2):
                try:
                    question, choices = self._split_question_choices(row["question"])
                    image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
                except Exception as exc:
                    raise ValueError(f"Invalid MMStar sample at TSV row {row_number}") from exc
                yield EvalSample(
                    uid=str(row["index"]), question=question, images=[image], choices=choices,
                    answer=row["answer"], category=row["category"], meta=row,
                )

    def summarize(self, records):
        summary = super().summarize(records)

        def grouped_accuracy(key):
            groups = {}
            for row in records:
                if row.get("correct") is not None:
                    groups.setdefault(row["meta"].get(key, "unknown"), []).append(bool(row["correct"]))
            return {name: sum(values) / len(values) for name, values in sorted(groups.items())}

        summary["mmstar_score"] = 100 * summary["accuracy"] if summary["accuracy"] is not None else None
        summary["accuracy_by_category"] = grouped_accuracy("category")
        summary["accuracy_by_l2_category"] = grouped_accuracy("l2_category")
        return summary


class VizWizAdapter(EvalAdapter):
    name = "vizwiz"

    SHORT_ANSWER_PROMPT = (
        "Answer the visual question with only one word or a short phrase. "
        "Do not explain your answer and do not use a full sentence. "
        "If the question cannot be answered from the image, answer exactly: unanswerable.\n\n"
        "Question: {question}\nShort answer:"
    )

    NUMBER_WORDS = {
        "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10",
    }

    @classmethod
    def normalize_vqa_answer(cls, answer: str) -> str:
        answer = str(answer or "").replace("\n", " ").replace("\t", " ").strip().lower()
        answer = re.sub(r"(?<!\d)[.,](?!\d)", " ", answer)
        answer = re.sub(r"[;!?\"{}()\[\]]", " ", answer)
        words = []
        for word in answer.split():
            word = cls.NUMBER_WORDS.get(word, word)
            if word not in ("a", "an", "the"):
                words.append(word)
        return " ".join(words)

    @staticmethod
    def _image(raw_image, uid):
        if isinstance(raw_image, dict) and raw_image.get("bytes") is not None:
            return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
        if isinstance(raw_image, dict) and raw_image.get("path"):
            return Image.open(raw_image["path"]).convert("RGB")
        if isinstance(raw_image, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw_image)).convert("RGB")
        raise ValueError(f"VizWiz sample {uid} has no readable image")

    def _files(self):
        root = Path(self.data)
        if root.is_file():
            return [root]
        candidates = sorted((root / "data").glob(f"{self.split}-*.parquet"))
        if not candidates:
            candidates = sorted(root.glob(f"{self.split}-*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No VizWiz {self.split!r} parquet files found under {root}")
        shard_match = re.search(r"-of-(\d+)-", candidates[0].name)
        if shard_match and len(candidates) != int(shard_match.group(1)):
            raise FileNotFoundError(
                f"VizWiz {self.split} download is incomplete: found {len(candidates)} of "
                f"{int(shard_match.group(1))} parquet shards under {candidates[0].parent}"
            )
        return candidates

    def __iter__(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("VizWiz parquet loading requires pyarrow") from exc
        for parquet_path in self._files():
            for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    uid = str(row["question_id"])
                    answers = list(row.get("answers") or [])
                    meta = {"answers": answers, "category": row.get("category", "default")}
                    yield EvalSample(
                        uid=uid, question=self.SHORT_ANSWER_PROMPT.format(question=row["question"]),
                        images=[self._image(row["image"], uid)],
                        answer=None, category=str(row.get("category", "default")), meta=meta,
                    )

    def score_prediction(self, prediction: str, sample: EvalSample):
        answers = [self.normalize_vqa_answer(x) for x in sample.meta.get("answers", [])]
        if not answers:
            return None
        prediction = self.normalize_vqa_answer(prediction)
        scores = []
        for held_out in range(len(answers)):
            matches = sum(prediction == answer for index, answer in enumerate(answers)
                          if index != held_out)
            scores.append(min(1.0, matches / 3.0))
        return sum(scores) / len(scores)

    def summarize(self, records):
        valid = [row for row in records if row.get("correct") is not None]
        summary = {
            "samples": len(records), "scored": len(valid),
            "accuracy": sum(float(row["correct"]) for row in valid) / len(valid) if valid else None,
        }
        categories = {}
        for category in sorted({row["category"] for row in valid}):
            rows = [row for row in valid if row["category"] == category]
            categories[category] = sum(float(row["correct"]) for row in rows) / len(rows)
        summary["vizwiz_score"] = 100 * summary["accuracy"] if summary["accuracy"] is not None else None
        summary["accuracy_by_category"] = categories
        return summary


class ScienceQAAdapter(EvalAdapter):
    name = "scienceqa"

    def _files(self):
        path = Path(self.data)
        if path.is_file():
            return [path]
        files = sorted(path.glob(f"{self.split}-*.parquet"))
        if not files:
            files = sorted((path / "ScienceQA-IMG").glob(f"{self.split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No ScienceQA {self.split!r} parquet files found under {path}")
        return files

    @staticmethod
    def _image(raw_image, uid):
        if isinstance(raw_image, dict) and raw_image.get("bytes") is not None:
            return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
        if isinstance(raw_image, dict) and raw_image.get("path"):
            return Image.open(raw_image["path"]).convert("RGB")
        if isinstance(raw_image, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw_image)).convert("RGB")
        raise ValueError(f"ScienceQA sample {uid} has no readable image")

    def __iter__(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("ScienceQA parquet loading requires pyarrow") from exc
        uid = 0
        for parquet_path in self._files():
            for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    choices = [str(choice) for choice in row["choices"]]
                    answer_index = int(row["answer"])
                    if not 0 <= answer_index < len(choices):
                        raise ValueError(
                            f"ScienceQA sample {uid} answer index {answer_index} is invalid for "
                            f"{len(choices)} choices"
                        )
                    hint = str(row.get("hint") or "").strip()
                    question = f"{hint}\n{row['question']}" if hint else row["question"]
                    meta = {key: row.get(key) for key in (
                        "task", "grade", "subject", "topic", "category", "skill",
                    )}
                    yield EvalSample(
                        uid=str(uid), question=question,
                        images=[self._image(row["image"], uid)], choices=choices,
                        answer=chr(ord("A") + answer_index),
                        category=str(row.get("subject", "default")), meta=meta,
                    )
                    uid += 1

    def summarize(self, records):
        summary = super().summarize(records)

        def grouped_accuracy(key):
            groups = {}
            for row in records:
                if row.get("correct") is not None:
                    groups.setdefault(str(row["meta"].get(key, "unknown")), []).append(
                        bool(row["correct"])
                    )
            return {name: sum(values) / len(values) for name, values in sorted(groups.items())}

        summary["scienceqa_score"] = 100 * summary["accuracy"] if summary["accuracy"] is not None else None
        summary["accuracy_by_subject"] = grouped_accuracy("subject")
        summary["accuracy_by_grade"] = grouped_accuracy("grade")
        summary["accuracy_by_category"] = grouped_accuracy("category")
        return summary


class MMEAdapter(EvalAdapter):
    name = "mme"

    TASKS = (
        "existence", "count", "position", "color", "posters", "celebrity",
        "scene", "landmark", "artwork", "OCR", "commonsense_reasoning",
        "numerical_calculation", "text_translation", "code_reasoning",
    )

    @staticmethod
    def _find_image(directory: Path, stem: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"MME image for annotation {directory / (stem + '.txt')} was not found")

    def _iter_per_image_files(self, root: Path):
        found_tasks = [task for task in self.TASKS if (root / task).is_dir()]
        if not found_tasks:
            raise FileNotFoundError(
                f"No MME task directories found under {root}; expected directories such as existence/ and OCR/"
            )
        for category in found_tasks:
            category_root = root / category
            question_index = 0
            for question_file in sorted(category_root.glob("*.txt")):
                image_path = self._find_image(category_root, question_file.stem)
                questions = []
                with question_file.open(encoding="utf-8-sig") as handle:
                    for line_number, line in enumerate(handle, 1):
                        line = line.rstrip("\r\n")
                        if not line:
                            continue
                        parts = line.rsplit("\t", 1)
                        if len(parts) != 2:
                            raise ValueError(
                                f"Invalid MME annotation at {question_file}:{line_number}; "
                                "expected question<TAB>answer"
                            )
                        questions.append((parts[0], parts[1]))
                if len(questions) != 2:
                    raise ValueError(
                        f"Official MME expects exactly two questions per image, but "
                        f"{question_file} contains {len(questions)}"
                    )
                for question, answer in questions:
                    yield EvalSample(
                        f"{category}:{question_index}", question, [str(image_path)],
                        ["Yes", "No"], answer, category,
                        {"image_name": image_path.name, "question": question,
                         "source": str(question_file)},
                    )
                    question_index += 1

    def _iter_aggregated_files(self, question_root: Path):
        dataset_root = question_root.parent
        image_index: dict[str, list[Path]] = {}
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for path in dataset_root.rglob(suffix):
                image_index.setdefault(path.name, []).append(path)
        for question_file in sorted(question_root.glob("*.txt")):
            category = question_file.stem
            with question_file.open(encoding="utf-8-sig") as handle:
                for index, line in enumerate(handle):
                    parts = line.rstrip("\r\n").split("\t", 2)
                    if len(parts) != 3:
                        continue
                    image_name, question, answer = parts
                    matches = image_index.get(Path(image_name).name, [])
                    image_path = next((p for p in matches if category in p.parts), None)
                    image_path = image_path or (matches[0] if matches else None)
                    if image_path is None:
                        raise FileNotFoundError(
                            f"MME image {image_name!r} referenced by {question_file} was not found"
                        )
                    yield EvalSample(
                        f"{category}:{index}", question, [str(image_path)], ["Yes", "No"],
                        answer, category, {"image_name": image_name, "question": question,
                                           "source": str(question_file)},
                    )

    def __iter__(self):
        root = Path(self.data)
        question_root = root if root.name == "questions_answers_YN" else root / "questions_answers_YN"
        if question_root.is_dir():
            yield from self._iter_aggregated_files(question_root)
        else:
            yield from self._iter_per_image_files(root)

    def summarize(self, records):
        summary = super().summarize(records)
        categories = {}
        for category in sorted({r["category"] for r in records}):
            rows = [r for r in records if r["category"] == category and r.get("correct") is not None]
            accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
            by_image = {}
            for row in rows:
                by_image.setdefault(row["meta"].get("image_name", row["id"]), []).append(row["correct"])
            accuracy_plus = (sum(len(x) == 2 and all(x) for x in by_image.values()) / len(by_image)
                             if by_image else 0.0)
            categories[category] = {"accuracy": accuracy, "accuracy_plus": accuracy_plus,
                                    "mme_score": 100 * (accuracy + accuracy_plus)}
        summary["categories"] = categories
        summary["mme_total"] = sum(x["mme_score"] for x in categories.values())
        return summary


class MMMUAdapter(EvalAdapter):
    name = "mmmu"

    def __iter__(self):
        try:
            from datasets import get_dataset_config_names, load_dataset, load_from_disk
        except ImportError as exc:
            raise RuntimeError("MMMU adapter requires: pip install datasets") from exc
        path = Path(self.data)
        datasets = []
        if path.exists() and (path / "dataset_dict.json").exists():
            datasets.append(("default", load_from_disk(str(path))[self.split]))
        elif path.exists() and (path / "state.json").exists():
            datasets.append(("default", load_from_disk(str(path))))
        else:
            for config in get_dataset_config_names(self.data):
                datasets.append((config, load_dataset(self.data, config, split=self.split)))
        for category, dataset in datasets:
            for index, row in enumerate(dataset):
                images = [row[k] for k in sorted(row) if k.startswith("image_") and row[k] is not None]
                choices = row.get("options") or row.get("choices") or []
                if isinstance(choices, str):
                    try: choices = json.loads(choices.replace("'", '"'))
                    except Exception: choices = []
                yield EvalSample(str(row.get("id", f"{category}:{index}")), row["question"], images,
                                 list(choices), row.get("answer"), category, dict(row))


ADAPTERS = {cls.name: cls for cls in (
    GenericJsonlAdapter, MMBenchAdapter, MMStarAdapter, VizWizAdapter, ScienceQAAdapter,
    MMEAdapter, MMMUAdapter,
)}


def get_adapter(name: str, **kwargs) -> EvalAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset {name!r}; available: {', '.join(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)
