#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
OCR_CONTEXT_TOKENS=${OCR_CONTEXT_TOKENS:-2048}
DATA=${DATA:-$ROOT_DIR/data/train/rendered_text_copy_${OCR_CONTEXT_TOKENS}/paired_eval.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT_DIR/data/train/rendered_text_copy_${OCR_CONTEXT_TOKENS}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/eval/ocr_eval}

CMD=(
  "$PY" -m src.ocr_eval
  --model-path "$MODEL_PATH"
  --data "$DATA"
  --image-root "$IMAGE_ROOT"
  --output-dir "$OUTPUT_DIR"
  --dtype "${DTYPE:-bfloat16}"
  --attn-implementation "${ATTN_IMPL:-flash_attention_2}"
  --max-samples "${MAX_SAMPLES:-1000}"
  --max-new-tokens "${MAX_NEW_TOKENS:-360}"
  --log-every "${LOG_EVERY:-25}"
  --adapter-decode-cache-mode "${ADAPTER_DECODE_CACHE_MODE:-shape_exact}"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
else
  CMD+=(--no-eval-adapter)
fi
if [[ "${EVAL_TEXT_TEACHER:-1}" == "0" ]]; then
  CMD+=(--no-eval-text-teacher)
fi
if [[ "${EVAL_IMAGE_TEACHER:-1}" == "0" ]]; then
  CMD+=(--no-eval-image-teacher)
fi

"${CMD[@]}"
