#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${IN_TMUX_PAYLOAD:-0}" != "1" && "${NO_TMUX:-0}" != "1" ]]; then
  SESSION_NAME=${SESSION_NAME:-qwen_256k_6k_train_eval}
  if command -v tmux >/dev/null 2>&1; then
    TMUX_ENV_VARS=(
      PY DATA_ROOT DATA RUN_NAME RUN_DIR OUTPUT_DIR LOG_FILE METRICS_JSONL
      MAX_STEPS SAVE_EVERY LR_SCHEDULER WARMUP_RATIO MIN_LR_RATIO
      MAX_SAMPLES NUM_SHARDS CUDA_VISIBLE_DEVICES BENCHMARKS RUNTIME_TAG
      FORCE_EVAL MEASURE_PREFILL COMPILE_ADAPTER COMPILE_MODE COMPILE_DYNAMIC COMPILE_WARMUP
      WANDB WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_RUN_ID WANDB_MODE
      NPROC_PER_NODE PIXEL_AREA_CACHE
    )
    TMUX_ENV_ARGS=()
    for var_name in "${TMUX_ENV_VARS[@]}"; do
      if [[ -v "$var_name" ]]; then
        TMUX_ENV_ARGS+=("$var_name=${!var_name}")
      fi
    done
    TMUX_ENV_PREFIX="$(printf ' %q' "${TMUX_ENV_ARGS[@]}")"
    tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT_DIR' && env IN_TMUX_PAYLOAD=1$TMUX_ENV_PREFIX bash '$0'"
    echo "started tmux session: $SESSION_NAME"
    echo "attach: tmux attach -t $SESSION_NAME"
    exit 0
  fi
  echo "tmux not found; running in the foreground. Set NO_TMUX=1 to silence this message."
fi

PY=${PY:-$ROOT_DIR/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}
DATA=${DATA:-$ROOT_DIR/data/pixmo_clean_llava_instruct_ocr_256k_v1/train.jsonl}

RUN_NAME=${RUN_NAME:-qwen_256k_pixmo_llava_ocr_opd6k_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/checkpoints}
LOG_FILE=${LOG_FILE:-$ROOT_DIR/artifacts/logs/${RUN_NAME}.train_then_eval.log}
METRICS_JSONL=${METRICS_JSONL:-$OUTPUT_DIR/train_metrics.jsonl}

MAX_STEPS=${MAX_STEPS:-6000}
SAVE_EVERY=${SAVE_EVERY:-500}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_STEPS_DISPLAY="$(awk -v warmup_ratio="$WARMUP_RATIO" -v total="$MAX_STEPS" 'BEGIN { printf "%d", warmup_ratio * total }')"
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
NUM_SHARDS=${NUM_SHARDS:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
BENCHMARKS=${BENCHMARKS:-all}
RUNTIME_TAG=${RUNTIME_TAG:-compiled_dense}

mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

{
  echo "=== one-shot qwen train then benchmark ==="
  echo "root=$ROOT_DIR"
  echo "data=$DATA"
  echo "run_name=$RUN_NAME"
  echo "run_dir=$RUN_DIR"
  echo "output_dir=$OUTPUT_DIR"
  echo "log=$LOG_FILE"
  echo "max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
  echo "lr_scheduler=$LR_SCHEDULER warmup_ratio=$WARMUP_RATIO warmup_steps~=$WARMUP_STEPS_DISPLAY min_lr_ratio=$MIN_LR_RATIO"
  echo "benchmarks=$BENCHMARKS max_samples=$MAX_SAMPLES num_shards=$NUM_SHARDS cuda=$CUDA_VISIBLE_DEVICES"
  echo "start=$(date)"
} | tee -a "$LOG_FILE"

DATA_ROOT="$DATA_ROOT" \
DATA="$DATA" \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_FILE="$LOG_FILE" \
METRICS_JSONL="$METRICS_JSONL" \
MAX_STEPS="$MAX_STEPS" \
SAVE_EVERY="$SAVE_EVERY" \
SUPERVISION_LOSS=opd \
KL_TOPK=1024 \
LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-token} \
LAMBDA_TRAJECTORY=${LAMBDA_TRAJECTORY:-1.0} \
LR_SCHEDULER="$LR_SCHEDULER" \
WARMUP_RATIO="$WARMUP_RATIO" \
MIN_LR_RATIO="$MIN_LR_RATIO" \
OUTPUT_MODE=native_visual_kv_injection \
WANDB=${WANDB:-1} \
WANDB_PROJECT=${WANDB_PROJECT:-vision-kv-inject} \
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME} \
NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
NUM_SHARDS="$NUM_SHARDS" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PIXEL_AREA_CACHE=${PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/${RUN_NAME}.pixel_areas.json} \
bash scripts/train_qwen_delta.sh

STEP_LIST=$(seq "$MAX_STEPS" "-$SAVE_EVERY" "$SAVE_EVERY")
{
  echo "=== training done $(date) ==="
  echo "eval_steps=$STEP_LIST"
} | tee -a "$LOG_FILE"

for step in $STEP_LIST; do
  ckpt="$OUTPUT_DIR/qwen_visual_delta_step${step}.pt"
  if [[ ! -s "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" | tee -a "$LOG_FILE" >&2
    exit 1
  fi

  echo "=== benchmark step=$step $(date) ===" | tee -a "$LOG_FILE"
  RUN_DIR="$RUN_DIR" \
  STEP="$step" \
  BENCHMARKS="$BENCHMARKS" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  NUM_SHARDS="$NUM_SHARDS" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  RUNTIME_TAG="$RUNTIME_TAG" \
  FORCE_EVAL=${FORCE_EVAL:-0} \
  MEASURE_PREFILL=${MEASURE_PREFILL:-1} \
  COMPILE_ADAPTER=${COMPILE_ADAPTER:-1} \
  COMPILE_MODE=${COMPILE_MODE:-reduce-overhead} \
  COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-1} \
  COMPILE_WARMUP=${COMPILE_WARMUP:-1} \
  bash scripts/run_qwen_benchmark_1k.sh --run-dir "$RUN_DIR" --step "$step" --benchmarks "$BENCHMARKS" --max-samples "$MAX_SAMPLES" --num-shards "$NUM_SHARDS" 2>&1 | tee -a "$LOG_FILE"
done

echo "=== one-shot done $(date) ===" | tee -a "$LOG_FILE"
