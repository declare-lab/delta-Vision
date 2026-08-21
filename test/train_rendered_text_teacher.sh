#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
DATA=${DATA:-$ROOT_DIR/data/train/rendered_text/paired_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/experiments/test_rendered_text_teacher}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29551}
REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}

CMD=(
  "$PY" -m torch.distributed.run
  --nproc_per_node "$NPROC_PER_NODE"
  --master_port "$MASTER_PORT"
  test/rendered_text_teacher_train.py
  --model-path "$MODEL_PATH"
  --data "$DATA"
  --output-dir "$OUTPUT_DIR"
  --dtype "${DTYPE:-bfloat16}"
  --attn-implementation "${ATTN_IMPL:-flash_attention_2}"
  --micro-batch-size-per-gpu "${MICRO_BATCH_SIZE_PER_GPU:-4}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --required-world-size "$REQUIRED_WORLD_SIZE"
  --max-steps "${MAX_STEPS:-3125}"
  --save-every "${SAVE_EVERY:-500}"
  --log-every "${LOG_EVERY:-5}"
  --lr "${LR:-5e-5}"
  --lr-scheduler "${LR_SCHEDULER:-cosine}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --min-lr-ratio "${MIN_LR_RATIO:-0.1}"
  --kl-topk "${KL_TOPK:-1024}"
  --reverse-kl-weight "${REVERSE_KL_WEIGHT:-0.0}"
  --first-token-weight "${FIRST_TOKEN_WEIGHT:-1.0}"
  --temperature "${TEMPERATURE:-2.0}"
  --lambda-logit "${LAMBDA_LOGIT:-1.0}"
  --visual-adapter-rank "${VISUAL_ADAPTER_RANK:-128}"
  --max-context-chars "${MAX_CONTEXT_CHARS:-0}"
  --deepspeed-config "${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2_coeff.json}"
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
if [[ "${WANDB:-1}" != "0" ]]; then
  CMD+=(--wandb --wandb-project "${WANDB_PROJECT:-vision-kv-inject}" --wandb-run-name "${WANDB_RUN_NAME:-$(basename "$OUTPUT_DIR")}" --wandb-mode "${WANDB_MODE:-online}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  CMD+=(--wandb-mode disabled)
fi

"${CMD[@]}"
