#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="$ROOT_DIR/.venv/bin/python"
export PYTHONPATH="$ROOT_DIR"


# Common settings: edit values here, rather than inheriting environment defaults.
WEIGHT_DECAY=0.01
TEMPERATURE=2.0
BATCH_SAMPLING="pixel_bucket"
PIXEL_BUCKET_SIZE=512
DIST_BACKEND="nccl"
GRAD_CLIP=1.0
SEED=44
KEEP_NCCL_ENV=0
WANDB=1
WANDB_PROJECT="vision-kv-inject"
WANDB_MODE="online"
WANDB_ENTITY=""
WANDB_RUN_ID=""

MODEL_KIND="qwen"
MODEL_KIND="$(printf '%s' "$MODEL_KIND" | tr '[:upper:]' '[:lower:]')"
DATA_ROOT="/lustre-data/leijingdi/code/delta-vision"
IMAGE_ROOT="$ROOT_DIR/data/train/pixmo"

if (( $# > 0 )) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/train.sh

Edit the assignments in this script. Extra CLI arguments override training options.

Common:
  MODEL_KIND=qwen|llava
  RUN_NAME=NAME
  MODEL_PATH=PATH
  DATA=JSONL
  DATA_ROOT=PATH
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MAX_STEPS=2000
  SAVE_EVERY=500

Qwen defaults train embedding_adapter with token-mean KL distillation.
Set OUTPUT_MODE=recurrent_embedding_adapter for the recurrent embedding adapter.
LLaVA defaults train kv_adapter.
EOF
  exit 0
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export DELTA_VISION_IMAGE_ROOT="$IMAGE_ROOT"

if [[ "$MODEL_KIND" == "qwen" ]]; then
  OUTPUT_MODE="embedding_adapter"
  if [[ "$OUTPUT_MODE" == "recurrent_embedding_adapter" ]]; then
    RUN_NAME="qwen_recurrent_embedding_adapter_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_NAME="qwen_embedding_adapter_$(date +%Y%m%d_%H%M%S)"
  fi
  OUTPUT_DIR="$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/$RUN_NAME/checkpoints"
  LOG_FILE="$ROOT_DIR/artifacts/logs/${RUN_NAME}.train.log"
  METRICS_JSONL="$OUTPUT_DIR/train_metrics.jsonl"
  INIT_CHECKPOINT=""

  MODEL_PATH="models/Qwen3-VL-4B-Instruct"
  DATA="data/train/pixmo/pixmo_ama_full_valid.clean.jsonl"
  DS_CONFIG="$ROOT_DIR/configs/ds_zero2.json"
  PIXEL_AREA_CACHE="$ROOT_DIR/data/train/pixmo/pixmo_ama_full_valid.clean.jsonl.pixel_areas.json"

  NPROC_PER_NODE="8"
  MASTER_PORT="29540"
  ATTN_IMPL="flash_attention_2"
  DTYPE="bfloat16"
  QWEN_DEVICE_MAP=""
  QWEN_MAX_MEMORY=""
  MAX_STEPS="2000"
  SAVE_EVERY="500"
  LOG_EVERY="5"
  MICRO_BATCH_SIZE_PER_GPU="4"
  GRADIENT_ACCUMULATION_STEPS="1"
  REQUIRED_WORLD_SIZE="$NPROC_PER_NODE"
  DISTRIBUTED_ENGINE="deepspeed"
  if [[ -n "$QWEN_DEVICE_MAP" && "$QWEN_DEVICE_MAP" != "none" && "$QWEN_DEVICE_MAP" != "replicated" ]]; then
    if [[ "$NPROC_PER_NODE" != "1" ]]; then
      echo "QWEN_DEVICE_MAP=$QWEN_DEVICE_MAP requires NPROC_PER_NODE=1 so one process can see all visible GPUs" >&2
      exit 1
    fi
    REQUIRED_WORLD_SIZE=1
  fi

  LR="5e-5"
  LR_SCHEDULER="cosine"
  WARMUP_RATIO="0.03"
  WARMUP_START_LR_RATIO="0.0"
  MIN_LR_RATIO="0.1"
  LOSS_NORMALIZATION="token"
  SUPERVISION_LOSS="distill"
  LAMBDA_LOGIT="1.0"
  KL_TOPK="1024"
  VISUAL_ADAPTER_RANK="128"

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

  if [[ "${KEEP_NCCL_ENV}" != "1" ]]; then
    unset NCCL_NET
    unset NCCL_IB_DISABLE
    unset NCCL_SOCKET_IFNAME
    unset GLOO_SOCKET_IFNAME
    unset TORCH_NCCL_ASYNC_ERROR_HANDLING
  fi
  for name in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET; do
    if [[ -v "$name" && -n "${!name}" ]]; then
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
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "pixel_area_cache=$PIXEL_AREA_CACHE"
  echo "log=$LOG_FILE"
  echo "nproc=$NPROC_PER_NODE cuda=$CUDA_VISIBLE_DEVICES"
  echo "max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
  echo "lr=$LR scheduler=$LR_SCHEDULER warmup_ratio=$WARMUP_RATIO loss_normalization=$LOSS_NORMALIZATION supervision_loss=$SUPERVISION_LOSS kl_topk=$KL_TOPK output_mode=$OUTPUT_MODE attn=$ATTN_IMPL distributed_engine=$DISTRIBUTED_ENGINE"
  echo "lambda_logit=$LAMBDA_LOGIT"
  echo "visual_adapter_rank=$VISUAL_ADAPTER_RANK"
  echo "qwen_device_map=${QWEN_DEVICE_MAP} qwen_max_memory=${QWEN_MAX_MEMORY}"

  CMD=(
    "$PY" -m torch.distributed.run
    --nproc_per_node "$NPROC_PER_NODE"
    --master_port "$MASTER_PORT"
    -m src.run train --family "$MODEL_KIND" --
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
    --weight-decay "${WEIGHT_DECAY}" --temperature "${TEMPERATURE}"
    --kl-topk "$KL_TOPK"
    --lambda-logit "$LAMBDA_LOGIT"
    --loss-normalization "$LOSS_NORMALIZATION"
    --supervision-loss "$SUPERVISION_LOSS"
    --output-mode "$OUTPUT_MODE"
    --visual-adapter-rank "$VISUAL_ADAPTER_RANK"
    --batch-sampling "${BATCH_SAMPLING}"
    --pixel-bucket-size "${PIXEL_BUCKET_SIZE}"
    --pixel-area-cache "$PIXEL_AREA_CACHE"
    --deepspeed-config "$DS_CONFIG"
    --dtype "$DTYPE"
    --attn-implementation "$ATTN_IMPL"
    --teacher-deepstack
    --qwen-device-map "$QWEN_DEVICE_MAP"
    --qwen-max-memory "$QWEN_MAX_MEMORY"
    --dist-backend "${DIST_BACKEND}"
    --distributed-engine "$DISTRIBUTED_ENGINE"
    --grad-clip "${GRAD_CLIP}"
    --log-every "$LOG_EVERY"
    --seed "${SEED}"
  )

elif [[ "$MODEL_KIND" == "llava" ]]; then
  OUTPUT_MODE="kv_adapter"
  if [[ "$OUTPUT_MODE" == "recurrent_embedding_adapter" ]]; then
    RUN_NAME="llava_recurrent_embedding_adapter_$(date +%Y%m%d_%H%M%S)"
  elif [[ "$OUTPUT_MODE" == "embedding_adapter" ]]; then
    RUN_NAME="llava_embedding_adapter_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_NAME="llava_kv_adapter_$(date +%Y%m%d_%H%M%S)"
  fi
  OUTPUT_DIR="$ROOT_DIR/artifacts/$RUN_NAME"
  LOG_FILE="$ROOT_DIR/artifacts/logs/${RUN_NAME}.train.log"
  METRICS_JSONL="$OUTPUT_DIR/train_metrics.jsonl"
  INIT_CHECKPOINT=""
  MODEL_PATH="models/llava-1.5-7b-hf"
  DATA="data/pixmo_ama_train.jsonl"
  DS_CONFIG="$ROOT_DIR/configs/ds_zero2.json"
  NPROC_PER_NODE=8
  MASTER_PORT="29500"
  ATTN_IMPL="flash_attention_2"
  DTYPE="bfloat16"
  MAX_STEPS="500"
  SAVE_EVERY="$MAX_STEPS"
  LOG_EVERY="5"
  BATCH_SIZE="4"
  LR="5e-5"
  LR_SCHEDULER="constant"
  WARMUP_RATIO="0.0"
  WARMUP_START_LR_RATIO="0.0"
  MIN_LR_RATIO="0.1"
  WEIGHT_DECAY="0.0"
  TEMPERATURE="2.0"
  KL_TOPK="1024"
  LOSS_NORMALIZATION="token"
  SUPERVISION_LOSS="distill"
  LAMBDA_LOGIT="2.0"
  VISUAL_ADAPTER_RANK="128"
  GRADIENT_ACCUMULATION_STEPS="1"
  GRAD_CLIP="1.0"
  SEED="44"
  REQUIRED_WORLD_SIZE="$NPROC_PER_NODE"
  DISTRIBUTED_ENGINE="deepspeed"

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

  if [[ "${KEEP_NCCL_ENV}" != "1" ]]; then
    unset NCCL_NET
    unset NCCL_IB_DISABLE
    unset NCCL_SOCKET_IFNAME
    unset GLOO_SOCKET_IFNAME
    unset TORCH_NCCL_ASYNC_ERROR_HANDLING
  fi
  for name in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET NCCL_NET_GDR_LEVEL NCCL_TUNER_CONFIG_PATH; do
    if [[ -v "$name" && -n "${!name}" ]]; then
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
    -m src.run train --family "$MODEL_KIND" --
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
    --dist-backend "${DIST_BACKEND}"
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

if [[ "${WANDB}" == "1" ]]; then
  CMD+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "$RUN_NAME"
    --wandb-mode "${WANDB_MODE}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_RUN_ID}" ]]; then
    CMD+=(--wandb-run-id "$WANDB_RUN_ID")
  fi
else
  CMD+=(--no-wandb --wandb-mode disabled)
fi

CMD+=("$@")
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
