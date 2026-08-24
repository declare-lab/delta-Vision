#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${KEEP_NCCL_ENV:-0}" != "1" ]]; then
  unset NCCL_NET
  unset NCCL_IB_DISABLE
  unset NCCL_SOCKET_IFNAME
  unset GLOO_SOCKET_IFNAME
  unset TORCH_NCCL_ASYNC_ERROR_HANDLING
fi

MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
OCR_DATASET=${OCR_DATASET:-rendered_text_copy_300}
DATA=${DATA:-$ROOT_DIR/data/train/$OCR_DATASET/paired_train.jsonl}
RUN_NAME=${RUN_NAME:-${OCR_DATASET}_kl_ds8_mb4_wandb_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/experiments/$RUN_NAME}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29551}
REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}

CMD=(
  "$PY" -m torch.distributed.run
  --nproc_per_node "$NPROC_PER_NODE"
  --master_port "$MASTER_PORT"
  -m src.ocr_train
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
  --temperature "${TEMPERATURE:-2.0}"
  --teacher-mode "${TEACHER_MODE:-text}"
  --lambda-logit "${LAMBDA_LOGIT:-1.0}"
  --lambda-ce "${LAMBDA_CE:-0.0}"
  --lambda-effect "${LAMBDA_EFFECT:-0.0}"
  --effect-mask "${EFFECT_MASK:-answer}"
  --effect-layers "${EFFECT_LAYERS:-last}"
  --lambda-prefill-kv "${LAMBDA_PREFILL_KV:-0.0}"
  --prefill-kv-layers "${PREFILL_KV_LAYERS:-all}"
  --prefill-kv-eps "${PREFILL_KV_EPS:-1e-6}"
  --visual-adapter-rank "${VISUAL_ADAPTER_RANK:-128}"
  --max-context-chars "${MAX_CONTEXT_CHARS:-0}"
  --deepspeed-config "${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2.json}"
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
