#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}"
CHECKPOINT="${CHECKPOINT:-}"
VALIDATION="${VALIDATION:-${ROOT_DIR}/data/benchmarks/hotpotqa/validation.jsonl}"
RENDER_METADATA="${RENDER_METADATA:-${ROOT_DIR}/data/benchmarks/hotpotqa/render/metadata.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/eval/hotpotqa_rendered}"
MODES="${MODES:-base_image,base_text,adapter_image}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE="${DEVICE:-cuda:0}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
START_INDEX="${START_INDEX:-0}"

ARGS=(
  "${ROOT_DIR}/src/eval_hotpotqa_rendered.py"
  --model-path "${MODEL_PATH}"
  --validation "${VALIDATION}"
  --render-metadata "${RENDER_METADATA}"
  --data-root "${ROOT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --modes "${MODES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --device "${DEVICE}"
  --log-every "${LOG_EVERY}"
  --num-shards "${NUM_SHARDS}"
  --shard-id "${SHARD_ID}"
  --start-index "${START_INDEX}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

exec "${PY}" "${ARGS[@]}"
