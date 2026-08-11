#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-../delta-vision/models/llava-1.5-7b-hf}
DATA=${DATA:-../delta-vision/data/pixmo_ama_train.jsonl}
DATA_ROOT=${DATA_ROOT:-../delta-vision}
RUN_NAME=${RUN_NAME:-vkv_inject_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-artifacts/$RUN_NAME}

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_NET=Socket
unset NCCL_NET_PLUGIN 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WANDB_ARGS=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT:-vision-kv-inject} --wandb-run-name ${RUN_NAME} --wandb-mode ${WANDB_MODE:-online}"
fi

echo "=== Vision KV Inject Training ==="
echo "run=$RUN_NAME"
echo "output=$OUT_DIR"
echo "data=$DATA"
echo "gpus=$NUM_GPUS"

$PY -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  -m src.train \
  --model-path "$MODEL_PATH" \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --max-steps "${MAX_STEPS:-4000}" \
  --lr "${LR:-1e-4}" \
  --kl-topk "${KL_TOPK:-1024}" \
  --log-every "${LOG_EVERY:-10}" \
  --save-every "${SAVE_EVERY:-500}" \
  --deepspeed-config configs/ds_zero2.json \
  --seed 42 \
  $WANDB_ARGS
