#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
DATA=${DATA:-$ROOT_DIR/data/rendered_context_qa_eval_v1/paired.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/experiments/test_rendered_text_teacher}

CMD=(
  "$PY" test/rendered_text_teacher_train.py
  --model-path "$MODEL_PATH"
  --data "$DATA"
  --output-dir "$OUTPUT_DIR"
  --device "${DEVICE:-cuda:0}"
  --dtype "${DTYPE:-bfloat16}"
  --attn-implementation "${ATTN_IMPL:-flash_attention_2}"
  --batch-size "${BATCH_SIZE:-1}"
  --max-steps "${MAX_STEPS:-100}"
  --save-every "${SAVE_EVERY:-100}"
  --log-every "${LOG_EVERY:-5}"
  --lr "${LR:-5e-5}"
  --kl-topk "${KL_TOPK:-1024}"
  --temperature "${TEMPERATURE:-2.0}"
  --lambda-logit "${LAMBDA_LOGIT:-2.0}"
  --lambda-ce "${LAMBDA_CE:-0.0}"
  --visual-adapter-rank "${VISUAL_ADAPTER_RANK:-128}"
  --max-context-chars "${MAX_CONTEXT_CHARS:-60000}"
)

if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${REQUIRE_ANSWER_VISIBLE:-}" ]]; then
  CMD+=(--require-answer-visible)
fi

"${CMD[@]}"
