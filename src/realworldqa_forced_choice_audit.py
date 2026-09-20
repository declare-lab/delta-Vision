"""Forced-choice audit for RealWorldQA visual-self results.

This diagnostic scores only samples with explicit A/B/C/D options or yes/no
answers. It avoids judging by generated free-form text so we can separate
format-following failures from visual information loss.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import types

import torch

from src.benchmarks import _realworldqa_choices_from_question
from src.data import QwenBenchmarkDataset
from src.initial_token_mlp_probe import runtime, teacher
from src.native_visual_self_attention_eval import Intervention
from src.visual_cross_token_ablation import prepare


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/diagnostics/channel_native_cache_20260916/realworldqa_eval.jsonl"
OUT = ROOT / "artifacts/diagnostics" / os.environ.get("RQA_OUT", "realworldqa_forced_choice_audit_20260917")
TEXT_FIRST = os.environ.get("RQA_TEXT_FIRST", "0") == "1"
NO_EXTRA_INSTRUCTION = os.environ.get("RQA_NO_EXTRA_INSTRUCTION", "0") == "1"
LIMIT = int(os.environ.get("RQA_LIMIT", "0") or "0")


def realworldqa_open_labels():
    labels = []
    seen = set()
    with (ROOT / "data/benchmarks/realworldqa/test.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            answer = str(row.get("answer", "")).strip()
            if _realworldqa_choices_from_question(row.get("question")):
                continue
            if answer.lower() in {"yes", "no"}:
                continue
            if answer and answer not in seen:
                seen.add(answer)
                labels.append(answer)
    return labels


OPEN_LABELS = realworldqa_open_labels()


def single_token_ids(tokenizer, texts):
    ids = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


def answer_variants(text):
    variants = [text]
    lower = text.lower()
    title = text[:1].upper() + text[1:].lower() if text else text
    for value in (lower, title):
        if value and value not in variants:
            variants.append(value)
    expanded = []
    for value in variants:
        expanded.extend((value, f" {value}", f"\n{value}"))
    return expanded


def candidate_token_ids(tokenizer, kind, labels):
    result = {}
    if kind == "mcq":
        for label in labels:
            result[label] = single_token_ids(
                tokenizer,
                (label, f" {label}", f"\n{label}", f"{label}.", f" {label}."),
            )
    elif kind == "yesno":
        result["yes"] = single_token_ids(tokenizer, ("Yes", " yes", "yes", "\nYes"))
        result["no"] = single_token_ids(tokenizer, ("No", " no", "no", "\nNo"))
    else:
        for label in labels:
            ids = single_token_ids(tokenizer, answer_variants(label))
            if not ids:
                # A few answers split into digit/subword pieces. This audit uses
                # the first token only; these labels are rare and are reported as
                # an answer-vocabulary diagnostic, not an official metric.
                ids = sorted(set(
                    tokenizer.encode(value, add_special_tokens=False)[0]
                    for value in answer_variants(label)
                    if tokenizer.encode(value, add_special_tokens=False)
                ))
            result[label] = ids
    missing = [label for label, ids in result.items() if not ids]
    if missing:
        raise RuntimeError(f"Could not build single-token candidates for {missing}")
    return result


def sample_kind(row):
    choices = _realworldqa_choices_from_question(row.get("question"))
    answer = str(row.get("answer", "")).strip()
    if choices:
        labels = [chr(65 + i) for i in range(len(choices))]
        return "mcq", labels, answer.upper()
    if answer.lower() in {"yes", "no"}:
        return "yesno", ["yes", "no"], answer.lower()
    if answer:
        return "open", OPEN_LABELS, answer
    return None, [], None


@torch.inference_mode()
def score_next_token(model, tokenizer, inputs, candidates):
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    logits = model(**inputs).logits[0, -1].float()
    scores = {}
    for label, ids in candidates.items():
        idx = torch.tensor(ids, device=logits.device)
        scores[label] = float(logits.index_select(0, idx).max())
    return max(scores, key=scores.get), scores


def worker(shard, world):
    runtime()
    processor, model = teacher()
    tokenizer = processor.tokenizer
    control = Intervention(model)
    ds = QwenBenchmarkDataset(
        str(DATA),
        processor,
        "realworldqa",
        answer_instruction="" if NO_EXTRA_INSTRUCTION else None,
    )
    if TEXT_FIRST:
        def text_first_content(self, row, question, images):
            return [{"type": "text", "text": question}] + [{"type": "image", "image": image} for image in images]
        ds._qwen_message_content = types.MethodType(text_first_content, ds)
    total = min(len(ds), LIMIT) if LIMIT else len(ds)
    out_path = OUT / f"rows{shard}.jsonl"
    with out_path.open("w", buffering=1) as out:
        for idx in range(shard, total, world):
            item = ds[idx]
            kind, labels, gold = sample_kind(item["row"])
            if kind is None:
                continue
            inputs = prepare(item, model.device)
            pos = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
            assert len(pos) > 0
            control.pos = pos
            candidates = candidate_token_ids(tokenizer, kind, labels)
            for mode in ("native", "visual_self_only"):
                control.enabled = mode != "native"
                control.changed = []
                control.errors = []
                control.validate = False
                pred, scores = score_next_token(model, tokenizer, inputs, candidates)
                expected = [] if mode == "native" else list(range(36))
                assert control.changed == expected, (idx, mode, control.changed[:3], control.changed[-3:])
                assert not control.pending
                out.write(json.dumps({
                    "sample": idx,
                    "kind": kind,
                    "mode": mode,
                    "gold": gold,
                    "prediction": pred,
                    "score": float(pred == gold),
                    "candidate_scores": scores,
                }) + "\n")
            if idx // world % 25 == 0:
                print("realworldqa", shard, idx, flush=True)
    (OUT / f"done{shard}.json").write_text(json.dumps({"complete": True}))


def launch(world):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROTOCOL.md").write_text(
        "Forced-choice RealWorldQA audit for Qwen3-VL-4B, FA2, DeepStack off. "
        "Uses the same native_visual_self_attention_eval Intervention as the "
        "generation run. Scores explicit MCQ, yes/no, and open-answer-vocabulary "
        "samples by next-token candidate logits. Open labels with multi-token "
        "tokenization use their first token only, so this is a diagnostic metric. "
        f"TEXT_FIRST={TEXT_FIRST}; NO_EXTRA_INSTRUCTION={NO_EXTRA_INSTRUCTION}; "
        f"LIMIT={LIMIT or 'all'}.\n"
    )
    jobs = []
    status = {"state": "running", "started": time.time(), "world": world}
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    try:
        for rank in range(world):
            log = (OUT / f"gpu{rank}.log").open("w")
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.realworldqa_forced_choice_audit", "worker", str(rank), str(world)],
                cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="4"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((proc, log))
        while any(proc.poll() is None for proc, _ in jobs):
            if any(proc.poll() not in (None, 0) for proc, _ in jobs):
                raise RuntimeError("worker failed")
            time.sleep(3)
        rows = []
        for rank in range(world):
            rows.extend(json.loads(line) for line in (OUT / f"rows{rank}.jsonl").read_text().splitlines())
        summary = {}
        for mode in ("native", "visual_self_only"):
            summary[mode] = {}
            for kind in ("mcq", "yesno", "open"):
                subset = [row for row in rows if row["mode"] == mode and row["kind"] == kind]
                summary[mode][kind] = {
                    "samples": len(subset),
                    "correct": sum(row["score"] for row in subset),
                    "accuracy_pct": 100.0 * sum(row["score"] for row in subset) / max(1, len(subset)),
                }
            subset = [row for row in rows if row["mode"] == mode and row["kind"] in {"mcq", "yesno"}]
            summary[mode]["mcq_plus_yesno"] = {
                "samples": len(subset),
                "correct": sum(row["score"] for row in subset),
                "accuracy_pct": 100.0 * sum(row["score"] for row in subset) / max(1, len(subset)),
            }
            subset = [row for row in rows if row["mode"] == mode]
            summary[mode]["all_answer_vocab"] = {
                "samples": len(subset),
                "correct": sum(row["score"] for row in subset),
                "accuracy_pct": 100.0 * sum(row["score"] for row in subset) / max(1, len(subset)),
            }
        (OUT / "results.json").write_text(json.dumps(summary, indent=2))
        status.update(state="complete", finished=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        (OUT / "status.json").write_text(json.dumps(status, indent=2))
        for proc, log in jobs:
            if proc.poll() is None:
                proc.terminate()
            log.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        launch(8)
