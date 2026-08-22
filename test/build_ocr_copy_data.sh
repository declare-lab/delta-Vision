#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-2048}
DATASET_NAME=${DATASET_NAME:-rendered_text_copy_${MAX_ANSWER_TOKENS}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/data/train/$DATASET_NAME}

"$PY" test/build_ocr_copy_data.py \
  --source-train "${SOURCE_TRAIN:-$ROOT_DIR/data/train/rendered_text/paired_train.jsonl}" \
  --source-eval "${SOURCE_EVAL:-$ROOT_DIR/data/train/rendered_text/paired_eval.jsonl}" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --model-path "${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}" \
  --train-size "${TRAIN_SIZE:-100000}" \
  --eval-size "${EVAL_SIZE:-1000}" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --min-answer-tokens "${MIN_ANSWER_TOKENS:-1024}" \
  --page-width "${PAGE_WIDTH:-1344}" \
  --page-height "${PAGE_HEIGHT:-2304}" \
  --font-size "${FONT_SIZE:-13}" \
  --jpeg-quality "${JPEG_QUALITY:-94}" \
  --preview "${PREVIEW:-32}" \
  --workers "${WORKERS:-128}" \
  --tokenize-batch-size "${TOKENIZE_BATCH_SIZE:-1024}" \
  --seed "${SEED:-49}" \
  --sample-with-replacement \
  --max-build-attempt-factor "${MAX_BUILD_ATTEMPT_FACTOR:-8}"
