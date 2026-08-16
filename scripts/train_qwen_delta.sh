#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME=${RUN_NAME:-qwen_visual_delta_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME/checkpoints}
LOG_FILE=${LOG_FILE:-$ROOT_DIR/artifacts/logs/${RUN_NAME}.train.log}
METRICS_JSONL=${METRICS_JSONL:-$OUTPUT_DIR/train_metrics.jsonl}

MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-4B-Instruct}
DATA=${DATA:-artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl}
DS_CONFIG=${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2_coeff.json}
PIXEL_AREA_CACHE=${PIXEL_AREA_CACHE:-$ROOT_DIR/artifacts/cache/pixmo_ama_full_valid.clean.pixel_areas.json}

if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$DATA_ROOT/$MODEL_PATH"
fi
if [[ "$DATA" != /* ]]; then
  if [[ -f "$ROOT_DIR/$DATA" ]]; then
    DATA="$ROOT_DIR/$DATA"
  else
    DATA="$DATA_ROOT/$DATA"
  fi
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29540}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
DTYPE=${DTYPE:-bfloat16}
MAX_STEPS=${MAX_STEPS:-500}
SAVE_EVERY=${SAVE_EVERY:-$MAX_STEPS}
LOG_EVERY=${LOG_EVERY:-5}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}
DISTRIBUTED_ENGINE=${DISTRIBUTED_ENGINE:-torch_grad_sync}

LR=${LR:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
WARMUP_START_LR_RATIO=${WARMUP_START_LR_RATIO:-0.0}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-sample}
SUPERVISION_LOSS=${SUPERVISION_LOSS:-distill}
LAMBDA_LOGIT=${LAMBDA_LOGIT:-4.0}
if [[ "$SUPERVISION_LOSS" == "opd" ]]; then
  LAMBDA_TRAJECTORY=${LAMBDA_TRAJECTORY:-0.0}
else
  LAMBDA_TRAJECTORY=${LAMBDA_TRAJECTORY:-0.5}
fi
LAMBDA_KV_MSE=${LAMBDA_KV_MSE:-0.0}
OPD_ROLLOUT_MAX_NEW_TOKENS=${OPD_ROLLOUT_MAX_NEW_TOKENS:-32}
OUTPUT_MODE=${OUTPUT_MODE:-native_visual_kv_split}
VISUAL_ADAPTER_RANK=${VISUAL_ADAPTER_RANK:-128}
READER_MLP_RATIO=${READER_MLP_RATIO:-4.0}
READER_ACTIVATION=${READER_ACTIVATION:-situ_glu}

export CUDA_VISIBLE_DEVICES
if [[ "${KEEP_NCCL_ENV:-0}" != "1" ]]; then
  unset NCCL_NET
  unset NCCL_IB_DISABLE
  unset NCCL_SOCKET_IFNAME
  unset GLOO_SOCKET_IFNAME
  unset TORCH_NCCL_ASYNC_ERROR_HANDLING
fi
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
  export NCCL_IB_DISABLE
fi
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
fi
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME
fi
if [[ -n "${NCCL_NET:-}" ]]; then
  export NCCL_NET
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export DELTA_VISION_IMAGE_ROOT="$DATA_ROOT"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$PIXEL_AREA_CACHE")"

echo "=== Qwen3-VL visual-delta train ==="
echo "root=$ROOT_DIR"
echo "data_root=$DATA_ROOT"
echo "run_name=$RUN_NAME"
echo "output_dir=$OUTPUT_DIR"
echo "metrics=$METRICS_JSONL"
echo "pixel_area_cache=$PIXEL_AREA_CACHE"
echo "log=$LOG_FILE"
echo "nproc=$NPROC_PER_NODE cuda=$CUDA_VISIBLE_DEVICES"
echo "max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
echo "lr=$LR scheduler=$LR_SCHEDULER warmup_ratio=$WARMUP_RATIO loss_normalization=$LOSS_NORMALIZATION supervision_loss=$SUPERVISION_LOSS output_mode=$OUTPUT_MODE attn=$ATTN_IMPL distributed_engine=$DISTRIBUTED_ENGINE"
echo "lambda_logit=$LAMBDA_LOGIT lambda_trajectory=$LAMBDA_TRAJECTORY lambda_kv_mse=$LAMBDA_KV_MSE opd_rollout_max_new_tokens=$OPD_ROLLOUT_MAX_NEW_TOKENS"
echo "visual_adapter_rank=$VISUAL_ADAPTER_RANK reader_mlp_ratio=$READER_MLP_RATIO reader_activation=$READER_ACTIVATION"

CMD=(
  "$PY" -m torch.distributed.run
  --nproc_per_node "$NPROC_PER_NODE"
  --master_port "$MASTER_PORT"
  -m src.train \
  --model-kind qwen \
  --data "$DATA" \
  --image-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-jsonl "$METRICS_JSONL" \
  --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY" \
  --required-world-size "$REQUIRED_WORLD_SIZE" \
  --micro-batch-size-per-gpu "$MICRO_BATCH_SIZE_PER_GPU" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --lr "$LR" --lr-scheduler "$LR_SCHEDULER" \
  --warmup-ratio "$WARMUP_RATIO" \
  --warmup-start-lr-ratio "$WARMUP_START_LR_RATIO" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --weight-decay 0.01 --temperature 2.0 \
  --lambda-trajectory "$LAMBDA_TRAJECTORY" --lambda-logit "$LAMBDA_LOGIT" --lambda-kv-mse "$LAMBDA_KV_MSE" \
  --loss-normalization "$LOSS_NORMALIZATION" \
  --supervision-loss "$SUPERVISION_LOSS" \
  --opd-rollout-max-new-tokens "$OPD_ROLLOUT_MAX_NEW_TOKENS" \
  --output-mode "$OUTPUT_MODE" \
  --visual-adapter-rank "$VISUAL_ADAPTER_RANK" \
  --reader-mlp-ratio "$READER_MLP_RATIO" --reader-activation "$READER_ACTIVATION" \
  --batch-sampling pixel_bucket \
  --pixel-bucket-size 512 \
  --pixel-area-cache "$PIXEL_AREA_CACHE" \
  --trajectory-layers 4,8,12,16,20,24,28,32,36 \
  --deepspeed-config "$DS_CONFIG" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --dist-backend nccl \
  --distributed-engine "$DISTRIBUTED_ENGINE" \
  --grad-clip 1.0 \
  --log-every "$LOG_EVERY" \
  --seed 44
)

if [[ "${WANDB:-0}" == "1" ]]; then
  CMD+=(
    --wandb
    --wandb-project "${WANDB_PROJECT:-vision-kv-inject}"
    --wandb-run-name "${WANDB_RUN_NAME:-$RUN_NAME}"
    --wandb-mode "${WANDB_MODE:-online}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    CMD+=(--wandb-run-id "$WANDB_RUN_ID")
  fi
else
  CMD+=(--wandb-mode disabled)
fi

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
