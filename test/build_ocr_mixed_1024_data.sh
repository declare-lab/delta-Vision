#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/data/train/ocr_overlap_copy70_qa30_1024}
MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}

"$PY" test/build_ocr_overlap_data.py \
  --qa-train "${QA_TRAIN:-$ROOT_DIR/data/train/rendered_text/paired_train.jsonl}" \
  --qa-eval "${QA_EVAL:-$ROOT_DIR/data/train/rendered_text/paired_eval.jsonl}" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --train-size "${TRAIN_SIZE:-100000}" \
  --eval-size "${EVAL_SIZE:-1000}" \
  --qa-ratio "${QA_RATIO:-0.3}" \
  --max-context-tokens "${MAX_CONTEXT_TOKENS:-1024}" \
  --min-context-tokens "${MIN_CONTEXT_TOKENS:-1}" \
  --max-qa-answer-tokens "${MAX_QA_ANSWER_TOKENS:-32}" \
  --tokenize-batch-size "${TOKENIZE_BATCH_SIZE:-512}" \
  --workers "${WORKERS:-128}" \
  --seed "${SEED:-49}" \
  ${USE_ALL_COPY_CONTEXTS:+--use-all-copy-contexts}
