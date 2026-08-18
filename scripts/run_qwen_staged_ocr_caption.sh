#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_qwen_staged_ocr_caption.sh

Default plan:
  Stage1: PixMo + long-caption distillation, 500 steps.
  Stage2: OCR-targeted distillation, 500 steps.
  Stage3: PixMo + long-caption OPD, 1000 steps.

Defaults:
  loss_normalization=token
  lambda_trajectory=1.0
  lr=5e-5, constant LR, no warmup

Useful environment variables:
  PREPARE_ONLY=1     Check data/config only, then exit before training.
  FORCE_RETRAIN=1    Run stages even when their final checkpoint exists.
  RUN_EVAL=1         Evaluate the final Stage3 checkpoint after training.
  EVAL_BENCHMARKS=   Comma-separated list, default textvqa,ocrbench,mmstar.
EOF
  exit 0
fi

RUN_NAME=${RUN_NAME:-qwen_caption500_ocr500_captionopd1000_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME}
MASTER_LOG=${MASTER_LOG:-$ROOT_DIR/artifacts/logs/${RUN_NAME}.staged_train.log}

STAGE1_NAME=${STAGE1_NAME:-stage1_caption_distill500}
STAGE1_DATA=${STAGE1_DATA:-$ROOT_DIR/data/pixmo_caption_16k_tf500_v1/train.jsonl}
STAGE1_PIXEL_AREA_CACHE=${STAGE1_PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/pixmo_caption_16k_tf500_v1.pixel_areas.json}
STAGE1_STEPS=${STAGE1_STEPS:-500}
STAGE1_SUPERVISION_LOSS=${STAGE1_SUPERVISION_LOSS:-distill}

STAGE2_NAME=${STAGE2_NAME:-stage2_ocr_distill500}
STAGE2_DATA=${STAGE2_DATA:-$ROOT_DIR/data/ocrbench_target_mix_32k_stage3_v2_clean/train.jsonl}
STAGE2_PIXEL_AREA_CACHE=${STAGE2_PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/ocrbench_target_mix_32k_stage3_v2_clean.pixel_areas.json}
STAGE2_STEPS=${STAGE2_STEPS:-500}
STAGE2_SUPERVISION_LOSS=${STAGE2_SUPERVISION_LOSS:-distill}

STAGE3_NAME=${STAGE3_NAME:-stage3_caption_opd1000}
STAGE3_DATA=${STAGE3_DATA:-$ROOT_DIR/data/pixmo_caption_16k_opd500_v1/train.jsonl}
STAGE3_PIXEL_AREA_CACHE=${STAGE3_PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/pixmo_caption_16k_opd500_v1.pixel_areas.json}
STAGE3_STEPS=${STAGE3_STEPS:-1000}
STAGE3_SUPERVISION_LOSS=${STAGE3_SUPERVISION_LOSS:-opd}

LR=${LR:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-token}
LAMBDA_TRAJECTORY=${LAMBDA_TRAJECTORY:-1.0}
LAMBDA_LOGIT=${LAMBDA_LOGIT:-4.0}
LAMBDA_KV_MSE=${LAMBDA_KV_MSE:-0.0}
KL_TOPK=${KL_TOPK:-1024}
OPD_ROLLOUT_MAX_NEW_TOKENS=${OPD_ROLLOUT_MAX_NEW_TOKENS:-32}
OUTPUT_MODE=${OUTPUT_MODE:-native_visual_kv_injection}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-5}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-vision-kv-inject}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}
RUN_EVAL=${RUN_EVAL:-0}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MASTER_PORT_STAGE1=${MASTER_PORT_STAGE1:-29540}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-29541}
MASTER_PORT_STAGE3=${MASTER_PORT_STAGE3:-29542}

EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-textvqa,ocrbench,mmstar}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1000}
EVAL_NUM_SHARDS=${EVAL_NUM_SHARDS:-$NPROC_PER_NODE}
EVAL_RUNTIME_TAG=${EVAL_RUNTIME_TAG:-eager_score}
EVAL_MEASURE_PREFILL=${EVAL_MEASURE_PREFILL:-0}
EVAL_COMPILE_ADAPTER=${EVAL_COMPILE_ADAPTER:-0}
EVAL_COMPILE_WARMUP=${EVAL_COMPILE_WARMUP:-0}
EVAL_COMPILE_DYNAMIC=${EVAL_COMPILE_DYNAMIC:-1}

mkdir -p "$RUN_DIR" "$(dirname "$MASTER_LOG")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

line_count() {
  wc -l < "$1" | tr -d ' '
}

run_stage() {
  local stage_name="$1"
  local data="$2"
  local pixel_cache="$3"
  local steps="$4"
  local supervision_loss="$5"
  local init_checkpoint="$6"
  local master_port="$7"

  local output_dir="$RUN_DIR/$stage_name/checkpoints"
  local metrics_jsonl="$output_dir/train_metrics.jsonl"
  local log_file="$ROOT_DIR/artifacts/logs/${RUN_NAME}.${stage_name}.log"
  local final_ckpt="$output_dir/qwen_visual_delta_step${steps}.pt"

  if [[ "$FORCE_RETRAIN" != "1" && -s "$final_ckpt" ]]; then
    log "skip $stage_name; found $final_ckpt"
    return
  fi

  if [[ -n "$init_checkpoint" ]]; then
    require_file "$init_checkpoint"
  fi

  log "=== $stage_name: $supervision_loss steps=$steps data=$data rows=$(line_count "$data") init=${init_checkpoint:-none} ==="
  DATA_ROOT="$DATA_ROOT" \
  DATA="$data" \
  PIXEL_AREA_CACHE="$pixel_cache" \
  RUN_NAME="${RUN_NAME}_${stage_name}" \
  OUTPUT_DIR="$output_dir" \
  LOG_FILE="$log_file" \
  METRICS_JSONL="$metrics_jsonl" \
  INIT_CHECKPOINT="$init_checkpoint" \
  MAX_STEPS="$steps" \
  SAVE_EVERY="$SAVE_EVERY" \
  LOG_EVERY="$LOG_EVERY" \
  LR="$LR" \
  LR_SCHEDULER="$LR_SCHEDULER" \
  WARMUP_RATIO="$WARMUP_RATIO" \
  MIN_LR_RATIO="$MIN_LR_RATIO" \
  SUPERVISION_LOSS="$supervision_loss" \
  LOSS_NORMALIZATION="$LOSS_NORMALIZATION" \
  LAMBDA_TRAJECTORY="$LAMBDA_TRAJECTORY" \
  LAMBDA_LOGIT="$LAMBDA_LOGIT" \
  LAMBDA_KV_MSE="$LAMBDA_KV_MSE" \
  KL_TOPK="$KL_TOPK" \
  OPD_ROLLOUT_MAX_NEW_TOKENS="$OPD_ROLLOUT_MAX_NEW_TOKENS" \
  OUTPUT_MODE="$OUTPUT_MODE" \
  WANDB="$WANDB" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}_${stage_name}}" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  REQUIRED_WORLD_SIZE="$REQUIRED_WORLD_SIZE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  MASTER_PORT="$master_port" \
  bash scripts/train_qwen_delta.sh

  require_file "$final_ckpt"
}

require_file "$STAGE1_DATA"
require_file "$STAGE1_PIXEL_AREA_CACHE"
require_file "$STAGE2_DATA"
require_file "$STAGE2_PIXEL_AREA_CACHE"
require_file "$STAGE3_DATA"
require_file "$STAGE3_PIXEL_AREA_CACHE"

STAGE1_CKPT="$RUN_DIR/$STAGE1_NAME/checkpoints/qwen_visual_delta_step${STAGE1_STEPS}.pt"
STAGE2_CKPT="$RUN_DIR/$STAGE2_NAME/checkpoints/qwen_visual_delta_step${STAGE2_STEPS}.pt"
STAGE3_CKPT="$RUN_DIR/$STAGE3_NAME/checkpoints/qwen_visual_delta_step${STAGE3_STEPS}.pt"
STAGE3_RUN_DIR="$RUN_DIR/$STAGE3_NAME"

log "run_name=$RUN_NAME"
log "run_dir=$RUN_DIR"
log "stage1=$STAGE1_NAME loss=$STAGE1_SUPERVISION_LOSS rows=$(line_count "$STAGE1_DATA") data=$STAGE1_DATA"
log "stage2=$STAGE2_NAME loss=$STAGE2_SUPERVISION_LOSS rows=$(line_count "$STAGE2_DATA") data=$STAGE2_DATA"
log "stage3=$STAGE3_NAME loss=$STAGE3_SUPERVISION_LOSS rows=$(line_count "$STAGE3_DATA") data=$STAGE3_DATA"
log "loss_normalization=$LOSS_NORMALIZATION lambda_trajectory=$LAMBDA_TRAJECTORY lambda_logit=$LAMBDA_LOGIT lr=$LR scheduler=$LR_SCHEDULER warmup=$WARMUP_RATIO"
log "force_retrain=$FORCE_RETRAIN run_eval=$RUN_EVAL"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  log "prepare_only=1; stop before training"
  exit 0
fi

run_stage "$STAGE1_NAME" "$STAGE1_DATA" "$STAGE1_PIXEL_AREA_CACHE" "$STAGE1_STEPS" "$STAGE1_SUPERVISION_LOSS" "" "$MASTER_PORT_STAGE1"
run_stage "$STAGE2_NAME" "$STAGE2_DATA" "$STAGE2_PIXEL_AREA_CACHE" "$STAGE2_STEPS" "$STAGE2_SUPERVISION_LOSS" "$STAGE1_CKPT" "$MASTER_PORT_STAGE2"
run_stage "$STAGE3_NAME" "$STAGE3_DATA" "$STAGE3_PIXEL_AREA_CACHE" "$STAGE3_STEPS" "$STAGE3_SUPERVISION_LOSS" "$STAGE2_CKPT" "$MASTER_PORT_STAGE3"

if [[ "$RUN_EVAL" == "1" ]]; then
  EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-$ROOT_DIR/artifacts/eval/qwen_topk1024_freezeqkv/$RUN_NAME/$EVAL_RUNTIME_TAG}
  log "=== eval final stage step=$STAGE3_STEPS benchmarks=$EVAL_BENCHMARKS ==="
  env -u DATA \
    DATA_ROOT="$DATA_ROOT" \
    RUN_DIR="$STAGE3_RUN_DIR" \
    STEP="$STAGE3_STEPS" \
    BENCHMARKS="$EVAL_BENCHMARKS" \
    MAX_SAMPLES="$EVAL_MAX_SAMPLES" \
    NUM_SHARDS="$EVAL_NUM_SHARDS" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    RUNTIME_TAG="$EVAL_RUNTIME_TAG" \
    OUT_ROOT="$EVAL_OUT_ROOT" \
    FORCE_EVAL=1 \
    MEASURE_PREFILL="$EVAL_MEASURE_PREFILL" \
    COMPILE_ADAPTER="$EVAL_COMPILE_ADAPTER" \
    COMPILE_WARMUP="$EVAL_COMPILE_WARMUP" \
    COMPILE_DYNAMIC="$EVAL_COMPILE_DYNAMIC" \
    bash scripts/run_qwen_benchmark_1k.sh \
      --run-dir "$STAGE3_RUN_DIR" \
      --step "$STAGE3_STEPS" \
      --benchmarks "$EVAL_BENCHMARKS" \
      --max-samples "$EVAL_MAX_SAMPLES" \
      --num-shards "$EVAL_NUM_SHARDS" \
      --out-root "$EVAL_OUT_ROOT" \
      --force 2>&1 | tee -a "$MASTER_LOG"
fi

log "done"
log "stage1_ckpt=$STAGE1_CKPT"
log "stage2_ckpt=$STAGE2_CKPT"
log "stage3_ckpt=$STAGE3_CKPT"
