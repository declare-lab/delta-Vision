#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP_NAME=${GROUP_NAME:-five_model_kl_only_${STAMP}}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT_DIR/data/train/pixmo}
DATA=${DATA:-$ROOT_DIR/data/train/pixmo/pixmo_ama_full_valid.clean.jsonl}

MAX_STEPS=${MAX_STEPS:-2000}
SAVE_EVERY=${SAVE_EVERY:-$MAX_STEPS}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
BENCHMARKS=${BENCHMARKS:-all}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-vision-kv-inject}
WANDB_MODE=${WANDB_MODE:-online}

if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_STEPS=${SMOKE_MAX_STEPS:-1}
  SAVE_EVERY=${SMOKE_SAVE_EVERY:-$MAX_STEPS}
  MAX_SAMPLES=${SMOKE_MAX_SAMPLES:-2}
  BENCHMARKS=${SMOKE_BENCHMARKS:-mmstar}
  WANDB=${SMOKE_WANDB:-0}
  NUM_SHARDS=${SMOKE_NUM_SHARDS:-1}
  NPROC_PER_NODE=${SMOKE_NPROC_PER_NODE:-1}
  MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
  LLAVA_DIST_BACKEND=${LLAVA_DIST_BACKEND:-gloo}
fi

COMMON_TRAIN_ENV=(
  DATA="$DATA"
  DATA_ROOT="${TRAIN_DATA_ROOT:-$IMAGE_ROOT}"
  IMAGE_ROOT="$IMAGE_ROOT"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  NPROC_PER_NODE="$NPROC_PER_NODE"
  REQUIRED_WORLD_SIZE="$NPROC_PER_NODE"
  MAX_STEPS="$MAX_STEPS"
  SAVE_EVERY="$SAVE_EVERY"
  LOG_EVERY="${LOG_EVERY:-5}"
  LR="${LR:-5e-5}"
  LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
  WARMUP_START_LR_RATIO="${WARMUP_START_LR_RATIO:-0.0}"
  MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
  TEMPERATURE="${TEMPERATURE:-2.0}"
  KL_TOPK="${KL_TOPK:-1024}"
  LOSS_NORMALIZATION="${LOSS_NORMALIZATION:-token}"
  SUPERVISION_LOSS="${SUPERVISION_LOSS:-distill}"
  LAMBDA_LOGIT="${LAMBDA_LOGIT:-1.0}"
  VISUAL_ADAPTER_RANK="${VISUAL_ADAPTER_RANK:-128}"
  SEED="${SEED:-44}"
  WANDB="$WANDB"
  WANDB_PROJECT="$WANDB_PROJECT"
  WANDB_MODE="$WANDB_MODE"
)

QWEN_TRAIN_ENV=(
  MODEL_KIND=qwen
  OUTPUT_MODE="${QWEN_OUTPUT_MODE:-embedding_adapter}"
  MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
  DS_CONFIG="${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2.json}"
  DISTRIBUTED_ENGINE="${DISTRIBUTED_ENGINE:-deepspeed}"
  PIXEL_AREA_CACHE="${PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/pixmo_ama_full_valid.clean.pixel_areas.json}"
  MASTER_PORT="${QWEN_MASTER_PORT:-29540}"
)

LLAVA_TRAIN_ENV=(
  MODEL_KIND=llava
  OUTPUT_MODE="${LLAVA_OUTPUT_MODE:-embedding_adapter}"
  BATCH_SIZE="${LLAVA_BATCH_SIZE:-${MICRO_BATCH_SIZE_PER_GPU:-4}}"
  DS_CONFIG="${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2.json}"
  DISTRIBUTED_ENGINE="${DISTRIBUTED_ENGINE:-deepspeed}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
  DIST_BACKEND="${LLAVA_DIST_BACKEND:-${DIST_BACKEND:-nccl}}"
  MASTER_PORT="${LLAVA_MASTER_PORT:-29550}"
)

MODELS=(
  "llava|llava-1.5-7b-hf|$DATA_ROOT/models/llava-1.5-7b-hf|embedding_adapter"
  "llava|llava-1.5-13b-hf|$ROOT_DIR/model/llava-1.5-13b-hf|embedding_adapter"
  "llava|llava-v1.6-mistral-7b-hf|$ROOT_DIR/model/llava-v1.6-mistral-7b-hf|embedding_adapter"
  "qwen|qwen3-vl-8b|$ROOT_DIR/model/Qwen3-VL-8B-Instruct|embedding_adapter"
  "qwen|qwen3-vl-30b-a3b|$ROOT_DIR/model/Qwen3-VL-30B-A3B-Instruct|embedding_adapter"
)

run_eval() {
  local model_kind="$1"
  local model_label="$2"
  local model_path="$3"
  local run_root="$4"
  local eval_root="$ROOT_DIR/artifacts/eval/${model_kind}/${GROUP_NAME}/${model_label}"

  echo "=== Eval final checkpoint: ${model_label} benchmarks=${BENCHMARKS} max_samples=${MAX_SAMPLES} ==="
  env \
    MODEL_KIND="$model_kind" \
    MODEL_PATH="$model_path" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NUM_SHARDS="$NUM_SHARDS" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    RUN_DIR="$run_root" \
    RUN_NAME="${GROUP_NAME}_${model_label}" \
    OUT_ROOT="$eval_root" \
    TEACHER_CACHE="${TEACHER_CACHE:-1}" \
    COMPILE_ADAPTER="${COMPILE_ADAPTER:-1}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}" \
    bash scripts/eval_benchmark.sh --benchmarks "$BENCHMARKS" --run-dir "$run_root" --step final
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_kind model_label model_path model_output_mode <<< "$entry"
  if [[ -n "${MODEL_FILTER:-}" && ",${MODEL_FILTER}," != *",${model_label},"* ]]; then
    continue
  fi
  run_name="${GROUP_NAME}_${model_label}"
  if [[ "$model_kind" == "qwen" ]]; then
    output_dir="$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/${run_name}/checkpoints"
    train_env=("${COMMON_TRAIN_ENV[@]}" "${QWEN_TRAIN_ENV[@]}")
  else
    output_dir="$ROOT_DIR/artifacts/experiments/llava_kl_only/${run_name}"
    train_env=("${COMMON_TRAIN_ENV[@]}" "${LLAVA_TRAIN_ENV[@]}")
  fi
  log_file="$ROOT_DIR/artifacts/logs/${run_name}.train.log"

  echo "=== Train ${model_label} kind=${model_kind} model=${model_path} ==="
  env "${train_env[@]}" \
    MODEL_PATH="$model_path" \
    OUTPUT_MODE="$model_output_mode" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    LOG_FILE="$log_file" \
    WANDB_RUN_NAME="$run_name" \
    bash scripts/train.sh

  run_eval "$model_kind" "$model_label" "$model_path" "$output_dir"
done

echo "=== Done: ${GROUP_NAME} ==="
