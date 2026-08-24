#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
CHECKPOINT=${CHECKPOINT:-$ROOT_DIR/artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/qwen_embedding_adapter_step500.pt}
IMAGE=${IMAGE:-$ROOT_DIR/data/train/rendered_text_copy_300/pages/train_000000_0a145f6f89202452.jpg}
MODE=${MODE:-adapter}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
DTYPE=${DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}

exec "$PY" src/chat_qwen_adapter.py \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --image "$IMAGE" \
  --mode "$MODE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL"
