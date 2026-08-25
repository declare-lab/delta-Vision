#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL_KIND=${MODEL_KIND:-qwen}
MODEL_KIND="$(printf '%s' "$MODEL_KIND" | tr '[:upper:]' '[:lower:]')"
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT_DIR/data/train/pixmo}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/train.sh

Configure with environment variables.

Common:
  MODEL_KIND=qwen|llava
  RUN_NAME=NAME
  MODEL_PATH=PATH
  DATA=JSONL
  DATA_ROOT=PATH
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MAX_STEPS=12000
  SAVE_EVERY=1000

Qwen defaults train embedding_adapter with token-mean KL distillation.
Set OUTPUT_MODE=recurrent_embedding_adapter for the recurrent embedding adapter.
LLaVA defaults train kv_adapter.
EOF
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export DELTA_VISION_IMAGE_ROOT="$IMAGE_ROOT"

if [[ "$MODEL_KIND" == "qwen" ]]; then
  OUTPUT_MODE=${OUTPUT_MODE:-embedding_adapter}
  if [[ "$OUTPUT_MODE" == "recurrent_embedding_adapter" ]]; then
    RUN_NAME=${RUN_NAME:-qwen_recurrent_embedding_adapter_$(date +%Y%m%d_%H%M%S)}
  else
    RUN_NAME=${RUN_NAME:-qwen_embedding_adapter_$(date +%Y%m%d_%H%M%S)}
  fi
  OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME/checkpoints}
  LOG_FILE=${LOG_FILE:-$ROOT_DIR/artifacts/logs/${RUN_NAME}.train.log}
  METRICS_JSONL=${METRICS_JSONL:-$OUTPUT_DIR/train_metrics.jsonl}
  INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

  MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-4B-Instruct}
  DATA=${DATA:-data/train/pixmo/pixmo_ama_full_valid.clean.jsonl}
  DS_CONFIG=${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2.json}
  PIXEL_AREA_CACHE=${PIXEL_AREA_CACHE:-$ROOT_DIR/data/train/pixmo/pixmo_ama_full_valid.clean.jsonl.pixel_areas.json}

  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_PORT=${MASTER_PORT:-29540}
  ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
  DTYPE=${DTYPE:-bfloat16}
  QWEN_DEVICE_MAP=${QWEN_DEVICE_MAP:-}
  QWEN_MAX_MEMORY=${QWEN_MAX_MEMORY:-}
  MAX_STEPS=${MAX_STEPS:-500}
  SAVE_EVERY=${SAVE_EVERY:-$MAX_STEPS}
  LOG_EVERY=${LOG_EVERY:-5}
  MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
  REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}
  DISTRIBUTED_ENGINE=${DISTRIBUTED_ENGINE:-torch_grad_sync}
  if [[ -n "$QWEN_DEVICE_MAP" && "$QWEN_DEVICE_MAP" != "none" && "$QWEN_DEVICE_MAP" != "replicated" ]]; then
    if [[ "$NPROC_PER_NODE" != "1" ]]; then
      echo "QWEN_DEVICE_MAP=$QWEN_DEVICE_MAP requires NPROC_PER_NODE=1 so one process can see all visible GPUs" >&2
      exit 1
    fi
    REQUIRED_WORLD_SIZE=1
  fi

  LR=${LR:-5e-5}
  LR_SCHEDULER=${LR_SCHEDULER:-constant}
  WARMUP_RATIO=${WARMUP_RATIO:-0.0}
  WARMUP_START_LR_RATIO=${WARMUP_START_LR_RATIO:-0.0}
  MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
  LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-token}
  SUPERVISION_LOSS=${SUPERVISION_LOSS:-distill}
  LAMBDA_LOGIT=${LAMBDA_LOGIT:-2.0}
  KL_TOPK=${KL_TOPK:-1024}
  VISUAL_ADAPTER_RANK=${VISUAL_ADAPTER_RANK:-128}

  if [[ "$MODEL_PATH" != /* ]]; then
    if [[ -e "$ROOT_DIR/$MODEL_PATH" ]]; then
      MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
    elif [[ -e "$DATA_ROOT/$MODEL_PATH" ]]; then
      MODEL_PATH="$DATA_ROOT/$MODEL_PATH"
    elif [[ "$MODEL_PATH" != */*/* && "$MODEL_PATH" == */* ]]; then
      MODEL_PATH="$MODEL_PATH"
    else
      MODEL_PATH="$DATA_ROOT/$MODEL_PATH"
    fi
  fi
  if [[ "$DATA" != /* ]]; then
    if [[ -f "$ROOT_DIR/$DATA" ]]; then
      DATA="$ROOT_DIR/$DATA"
    else
      DATA="$DATA_ROOT/$DATA"
    fi
  fi

  if [[ "${KEEP_NCCL_ENV:-0}" != "1" ]]; then
    unset NCCL_NET
    unset NCCL_IB_DISABLE
    unset NCCL_SOCKET_IFNAME
    unset GLOO_SOCKET_IFNAME
    unset TORCH_NCCL_ASYNC_ERROR_HANDLING
  fi
  for name in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET; do
    if [[ -n "${!name:-}" ]]; then
      export "$name"
    fi
  done

  mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$PIXEL_AREA_CACHE")"

  echo "=== Qwen3-VL embedding_adapter train ==="
  echo "root=$ROOT_DIR"
  echo "data_root=$DATA_ROOT"
  echo "image_root=$IMAGE_ROOT"
  echo "run_name=$RUN_NAME"
  echo "output_dir=$OUTPUT_DIR"
  echo "metrics=$METRICS_JSONL"
  echo "init_checkpoint=${INIT_CHECKPOINT:-none}"
  echo "pixel_area_cache=$PIXEL_AREA_CACHE"
  echo "log=$LOG_FILE"
  echo "nproc=$NPROC_PER_NODE cuda=$CUDA_VISIBLE_DEVICES"
  echo "max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
  echo "lr=$LR scheduler=$LR_SCHEDULER warmup_ratio=$WARMUP_RATIO loss_normalization=$LOSS_NORMALIZATION supervision_loss=$SUPERVISION_LOSS kl_topk=$KL_TOPK output_mode=$OUTPUT_MODE attn=$ATTN_IMPL distributed_engine=$DISTRIBUTED_ENGINE"
  echo "lambda_logit=$LAMBDA_LOGIT"
  echo "visual_adapter_rank=$VISUAL_ADAPTER_RANK"
  echo "qwen_device_map=${QWEN_DEVICE_MAP:-none} qwen_max_memory=${QWEN_MAX_MEMORY:-auto}"

  CMD=(
    "$PY" -m torch.distributed.run
    --nproc_per_node "$NPROC_PER_NODE"
    --master_port "$MASTER_PORT"
    -m src.train
    --model-kind qwen
    --data "$DATA"
    --image-root "$IMAGE_ROOT"
    --model-path "$MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --metrics-jsonl "$METRICS_JSONL"
    --init-checkpoint "$INIT_CHECKPOINT"
    --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY"
    --required-world-size "$REQUIRED_WORLD_SIZE"
    --micro-batch-size-per-gpu "$MICRO_BATCH_SIZE_PER_GPU" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --lr "$LR" --lr-scheduler "$LR_SCHEDULER"
    --warmup-ratio "$WARMUP_RATIO"
    --warmup-start-lr-ratio "$WARMUP_START_LR_RATIO"
    --min-lr-ratio "$MIN_LR_RATIO"
    --weight-decay "${WEIGHT_DECAY:-0.0}" --temperature "${TEMPERATURE:-2.0}"
    --kl-topk "$KL_TOPK"
    --lambda-logit "$LAMBDA_LOGIT"
    --loss-normalization "$LOSS_NORMALIZATION"
    --supervision-loss "$SUPERVISION_LOSS"
    --output-mode "$OUTPUT_MODE"
    --visual-adapter-rank "$VISUAL_ADAPTER_RANK"
    --batch-sampling "${BATCH_SAMPLING:-pixel_bucket}"
    --pixel-bucket-size "${PIXEL_BUCKET_SIZE:-512}"
    --pixel-area-cache "$PIXEL_AREA_CACHE"
    --deepspeed-config "$DS_CONFIG"
    --dtype "$DTYPE"
    --attn-implementation "$ATTN_IMPL"
    --qwen-device-map "$QWEN_DEVICE_MAP"
    --qwen-max-memory "$QWEN_MAX_MEMORY"
    --dist-backend "${DIST_BACKEND:-nccl}"
    --distributed-engine "$DISTRIBUTED_ENGINE"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --log-every "$LOG_EVERY"
    --seed "${SEED:-44}"
  )

elif [[ "$MODEL_KIND" == "llava" ]]; then
  OUTPUT_MODE=${OUTPUT_MODE:-kv_adapter}
  if [[ "$OUTPUT_MODE" == "recurrent_embedding_adapter" ]]; then
    RUN_NAME=${RUN_NAME:-llava_recurrent_embedding_adapter_$(date +%Y%m%d_%H%M%S)}
  elif [[ "$OUTPUT_MODE" == "embedding_adapter" ]]; then
    RUN_NAME=${RUN_NAME:-llava_embedding_adapter_$(date +%Y%m%d_%H%M%S)}
  else
    RUN_NAME=${RUN_NAME:-llava_kv_adapter_$(date +%Y%m%d_%H%M%S)}
  fi
  OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/artifacts/$RUN_NAME}
  LOG_FILE=${LOG_FILE:-$ROOT_DIR/artifacts/logs/${RUN_NAME}.train.log}
  METRICS_JSONL=${METRICS_JSONL:-$OUTPUT_DIR/train_metrics.jsonl}
  INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
  MODEL_PATH=${MODEL_PATH:-models/llava-1.5-7b-hf}
  DATA=${DATA:-data/pixmo_ama_train.jsonl}
  DS_CONFIG=${DS_CONFIG:-$ROOT_DIR/configs/ds_zero2.json}
  NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_GPUS:-8}}
  MASTER_PORT=${MASTER_PORT:-29500}
  ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
  DTYPE=${DTYPE:-bfloat16}
  MAX_STEPS=${MAX_STEPS:-500}
  SAVE_EVERY=${SAVE_EVERY:-$MAX_STEPS}
  LOG_EVERY=${LOG_EVERY:-5}
  BATCH_SIZE=${BATCH_SIZE:-4}
  LR=${LR:-5e-5}
  LR_SCHEDULER=${LR_SCHEDULER:-constant}
  WARMUP_RATIO=${WARMUP_RATIO:-0.0}
  WARMUP_START_LR_RATIO=${WARMUP_START_LR_RATIO:-0.0}
  MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
  WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
  TEMPERATURE=${TEMPERATURE:-2.0}
  KL_TOPK=${KL_TOPK:-1024}
  LOSS_NORMALIZATION=${LOSS_NORMALIZATION:-token}
  SUPERVISION_LOSS=${SUPERVISION_LOSS:-distill}
  LAMBDA_LOGIT=${LAMBDA_LOGIT:-2.0}
  VISUAL_ADAPTER_RANK=${VISUAL_ADAPTER_RANK:-128}
  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
  GRAD_CLIP=${GRAD_CLIP:-1.0}
  SEED=${SEED:-44}
  REQUIRED_WORLD_SIZE=${REQUIRED_WORLD_SIZE:-$NPROC_PER_NODE}
  DISTRIBUTED_ENGINE=${DISTRIBUTED_ENGINE:-deepspeed}

  if [[ "$MODEL_PATH" != /* ]]; then
    if [[ -e "$ROOT_DIR/$MODEL_PATH" ]]; then
      MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
    else
      MODEL_PATH="$DATA_ROOT/$MODEL_PATH"
    fi
  fi
  if [[ "$DATA" != /* ]]; then
    if [[ -f "$ROOT_DIR/$DATA" ]]; then
      DATA="$ROOT_DIR/$DATA"
    else
      DATA="$DATA_ROOT/$DATA"
    fi
  fi

  if [[ "${KEEP_NCCL_ENV:-0}" != "1" ]]; then
    unset NCCL_NET
    unset NCCL_IB_DISABLE
    unset NCCL_SOCKET_IFNAME
    unset GLOO_SOCKET_IFNAME
    unset TORCH_NCCL_ASYNC_ERROR_HANDLING
  fi
  for name in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET NCCL_NET_GDR_LEVEL NCCL_TUNER_CONFIG_PATH; do
    if [[ -n "${!name:-}" ]]; then
      export "$name"
    fi
  done

  mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

  echo "=== LLaVA kv_adapter train ==="
  echo "root=$ROOT_DIR"
  echo "run_name=$RUN_NAME"
  echo "output_dir=$OUTPUT_DIR"
  echo "model_path=$MODEL_PATH"
  echo "data=$DATA"
  echo "nproc=$NPROC_PER_NODE cuda=$CUDA_VISIBLE_DEVICES"
  echo "max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
  echo "lr=$LR scheduler=$LR_SCHEDULER warmup_ratio=$WARMUP_RATIO loss_normalization=$LOSS_NORMALIZATION supervision_loss=$SUPERVISION_LOSS kl_topk=$KL_TOPK output_mode=$OUTPUT_MODE attn=$ATTN_IMPL distributed_engine=$DISTRIBUTED_ENGINE"
  echo "lambda_logit=$LAMBDA_LOGIT"
  echo "visual_adapter_rank=$VISUAL_ADAPTER_RANK"

  CMD=(
    "$PY" -m torch.distributed.run
    --nproc_per_node "$NPROC_PER_NODE"
    --master_port "$MASTER_PORT"
    -m src.train
    --model-kind llava
    --model-path "$MODEL_PATH"
    --data "$DATA"
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --metrics-jsonl "$METRICS_JSONL"
    --init-checkpoint "$INIT_CHECKPOINT"
    --max-steps "$MAX_STEPS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --lr-scheduler "$LR_SCHEDULER"
    --warmup-ratio "$WARMUP_RATIO"
    --warmup-start-lr-ratio "$WARMUP_START_LR_RATIO"
    --min-lr-ratio "$MIN_LR_RATIO"
    --weight-decay "$WEIGHT_DECAY"
    --temperature "$TEMPERATURE"
    --kl-topk "$KL_TOPK"
    --lambda-logit "$LAMBDA_LOGIT"
    --loss-normalization "$LOSS_NORMALIZATION"
    --supervision-loss "$SUPERVISION_LOSS"
    --log-every "$LOG_EVERY"
    --save-every "$SAVE_EVERY"
    --output-mode "$OUTPUT_MODE"
    --visual-adapter-rank "$VISUAL_ADAPTER_RANK"
    --deepspeed-config "$DS_CONFIG"
    --dist-backend "${DIST_BACKEND:-nccl}"
    --distributed-engine "$DISTRIBUTED_ENGINE"
    --required-world-size "$REQUIRED_WORLD_SIZE"
    --grad-clip "$GRAD_CLIP"
    --dtype "$DTYPE"
    --attn-implementation "$ATTN_IMPL"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
  )
else
  echo "MODEL_KIND must be qwen or llava, got $MODEL_KIND" >&2
  exit 1
fi

if [[ "${WANDB:-1}" == "1" ]]; then
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

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE:-/dev/stdout}"
