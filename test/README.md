# Rendered Text Teacher Experiment

This directory keeps the text-context teacher experiment separate from the main trainer.

Teacher input:

```text
Context: raw text_context
Question: question
Answer: answer
```

Student input:

```text
rendered page image(s) + rendered_question/question + answer
```

Only the answer-token suffix is aligned for supervision. This avoids requiring the
teacher's long text prompt and the student's rendered-image prompt to have matching
token positions.

Run a small smoke train:

```bash
MAX_STEPS=10 REQUIRE_ANSWER_VISIBLE=1 test/train_rendered_text_teacher.sh
```

Useful environment variables:

```text
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
DATA=data/rendered_context_qa_eval_v1/paired.jsonl
OUTPUT_DIR=artifacts/experiments/test_rendered_text_teacher
BATCH_SIZE=1
MAX_STEPS=100
MAX_CONTEXT_CHARS=60000
LAMBDA_LOGIT=2.0
LAMBDA_CE=0.0
REQUIRE_ANSWER_VISIBLE=1
```
