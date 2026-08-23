#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
RUN_NAME=${RUN_NAME:-rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442}
STEP=${STEP:-500}

"$PY" test/diagnose_ocr_transcribe_then_qa.py \
  --checkpoint "${CHECKPOINT:-$ROOT_DIR/artifacts/experiments/$RUN_NAME/qwen_embedding_adapter_step${STEP}.pt}" \
  --data "${DATA:-$ROOT_DIR/data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_span_100.jsonl}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/artifacts/eval/qwen/$RUN_NAME/eager/diagnose_transcribe_then_qa_step${STEP}}" \
  --max-samples "${MAX_SAMPLES:-20}" \
  --max-transcribe-tokens "${MAX_TRANSCRIBE_TOKENS:-512}" \
  --max-answer-tokens "${MAX_ANSWER_TOKENS:-64}" \
  "$@"
