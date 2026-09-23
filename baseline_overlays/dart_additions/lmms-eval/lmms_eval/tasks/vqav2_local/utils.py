from pathlib import Path
import statistics

from PIL import Image

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


DATA_ROOT = Path("/lustre-data/leijingdi/code/vision-kv-inject/data/benchmarks/vqav2")


def vqav2_local_doc_to_visual(doc):
    image_path = Path(doc["image"])
    if not image_path.is_absolute():
        image_path = DATA_ROOT / image_path
    return [Image.open(image_path).convert("RGB")]


def vqav2_local_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def vqav2_local_process_results(doc, result):
    processor = EvalAIAnswerProcessor()
    pred = processor(result[0])
    accuracy = 0.0

    answers = doc.get("answers") or []
    if answers:
        for ans in answers:
            ans["answer"] = ans["answer"].replace("\n", " ").replace("\t", " ").strip()
        gt_answers = [ans["answer"] for ans in answers]

        if len(set(gt_answers)) > 1:
            for ans in answers:
                ans["answer"] = processor.process_punctuation(ans["answer"])
                ans["answer"] = processor.process_digit_article(ans["answer"])
            pred = processor.process_punctuation(pred)
            pred = processor.process_digit_article(pred)

        scores = []
        for gt in answers:
            other = [item for item in answers if item != gt]
            matches = [item for item in other if item["answer"] == pred]
            scores.append(min(1.0, float(len(matches)) / 3.0))
        accuracy = statistics.mean(scores)

    return {"exact_match": accuracy}
