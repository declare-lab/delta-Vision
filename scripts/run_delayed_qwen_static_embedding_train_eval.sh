#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DELAY_SECONDS="${DELAY_SECONDS:-7200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen_static_embedding_2000_ds_zero2_warmup005_mb1_ga4_delayed_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/artifacts/logs}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/${RUN_NAME}.train.log}"
DRIVER_LOG="${DRIVER_LOG:-$LOG_DIR/${RUN_NAME}.driver.log}"
BENCHMARKS="${BENCHMARKS:-mmstar,gqa,mmb,mmb-cn,mme,pope,sqa,vqav2,realworldqa}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

{
  echo "run_name=$RUN_NAME"
  echo "delay_seconds=$DELAY_SECONDS"
  echo "run_root=$RUN_ROOT"
  echo "ckpt_dir=$CKPT_DIR"
  echo "train_log=$TRAIN_LOG"
  echo "benchmarks=$BENCHMARKS"
  echo "scheduled_at=$(date -Is)"
  echo "start_after=$(date -Is -d "+${DELAY_SECONDS} seconds" 2>/dev/null || true)"
} | tee -a "$DRIVER_LOG"

sleep "$DELAY_SECONDS"

echo "train_start=$(date -Is)" | tee -a "$DRIVER_LOG"

MODEL_KIND=qwen \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$CKPT_DIR" \
LOG_FILE="$TRAIN_LOG" \
METRICS_JSONL="$CKPT_DIR/train_metrics.jsonl" \
OUTPUT_MODE=embedding_adapter \
DATA=data/train/pixmo/pixmo_ama_full_valid.clean.jsonl \
IMAGE_ROOT="$ROOT_DIR/data/train/pixmo" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
NPROC_PER_NODE=8 \
REQUIRED_WORLD_SIZE=8 \
MAX_STEPS=2000 \
SAVE_EVERY=500 \
MICRO_BATCH_SIZE_PER_GPU=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
LR=5e-5 \
LR_SCHEDULER=cosine \
WARMUP_RATIO=0.05 \
SUPERVISION_LOSS=distill \
LAMBDA_LOGIT=2.0 \
LAMBDA_KV_MSE=0.0 \
WEIGHT_DECAY=0.0 \
GRAD_CLIP=1.0 \
DISTRIBUTED_ENGINE=deepspeed \
DS_CONFIG="$ROOT_DIR/configs/ds_zero2.json" \
WANDB=1 \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_PROJECT="${WANDB_PROJECT:-vision-kv-inject}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}" \
bash scripts/train.sh

echo "train_done=$(date -Is)" | tee -a "$DRIVER_LOG"

echo "eval_start=$(date -Is)" | tee -a "$DRIVER_LOG"
MODEL_KIND=qwen \
OUTPUT_MODE=embedding_adapter \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
NUM_SHARDS=8 \
MAX_SAMPLES="${MAX_SAMPLES:-1000}" \
COMPILE_ADAPTER="${COMPILE_ADAPTER:-1}" \
TEACHER_CACHE="${TEACHER_CACHE:-1}" \
INPUT_CACHE="${INPUT_CACHE:-1}" \
RUNTIME_TAG="${RUNTIME_TAG:-compiled}" \
FORCE_EVAL="${FORCE_EVAL:-0}" \
bash scripts/eval_benchmark.sh \
  --benchmarks "$BENCHMARKS" \
  --run-dir "$CKPT_DIR" \
  --steps "2000"

echo "eval_done=$(date -Is)" | tee -a "$DRIVER_LOG"
