#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

"$PY" test/build_ocr_mixed_data.py \
  --copy-train "${COPY_TRAIN:-$ROOT_DIR/data/train/rendered_text_copy_2048/paired_train.jsonl}" \
  --copy-eval "${COPY_EVAL:-$ROOT_DIR/data/train/rendered_text_copy_2048/paired_eval.jsonl}" \
  --qa-train "${QA_TRAIN:-$ROOT_DIR/data/train/rendered_text/paired_train.jsonl}" \
  --qa-eval "${QA_EVAL:-$ROOT_DIR/data/train/rendered_text/paired_eval.jsonl}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/data/train/ocr_mixed_copy70_qa30_2048}" \
  --model-path "${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}" \
  --train-size "${TRAIN_SIZE:-100000}" \
  --eval-size "${EVAL_SIZE:-1000}" \
  --qa-ratio "${QA_RATIO:-0.3}" \
  --max-qa-answer-tokens "${MAX_QA_ANSWER_TOKENS:-32}" \
  --max-copy-answer-tokens "${MAX_COPY_ANSWER_TOKENS:-2048}" \
  --min-copy-answer-tokens "${MIN_COPY_ANSWER_TOKENS:-1024}" \
  --seed "${SEED:-49}"
