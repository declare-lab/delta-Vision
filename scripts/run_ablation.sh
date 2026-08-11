#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=.venv/bin/python
MODEL_PATH=../delta-vision/models/llava-1.5-7b-hf
DATA=../delta-vision/data/pixmo_ama_train.jsonl
DATA_ROOT=../delta-vision
MAX_STEPS=${MAX_STEPS:-500}
NUM_GPUS=8
MASTER_PORT=29500

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_NET=Socket
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_experiment() {
    local name=$1
    local source_layers=$2
    local bottleneck=$3
    local out_dir="artifacts/ablation_${name}"

    echo "========================================"
    echo "Experiment: $name"
    echo "  source_layers=$source_layers bottleneck=$bottleneck"
    echo "  output=$out_dir"
    echo "========================================"

    $PY -m torch.distributed.run \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      -m src.train \
      --model-path "$MODEL_PATH" \
      --data "$DATA" \
      --data-root "$DATA_ROOT" \
      --output-dir "$out_dir" \
      --source-layers "$source_layers" \
      --bottleneck-dim "$bottleneck" \
      --max-steps "$MAX_STEPS" \
      --lr 1e-4 \
      --kl-topk 1024 \
      --log-every 10 \
      --save-every 500 \
      --deepspeed-config configs/ds_zero2.json \
      --seed 42

    echo "=== Evaluating $name ==="
    mkdir -p "${out_dir}/eval"
    for shard in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$shard $PY src/eval_three_modes.py \
          --model-path "$MODEL_PATH" \
          --data ../delta-vision/data/mmstar/mmstar_val.jsonl \
          --data-root ../delta-vision \
          --checkpoint "${out_dir}/step_${MAX_STEPS}.pt" \
          --output-dir "${out_dir}/eval" \
          --num-shards 8 \
          --max-samples 1000 \
          --shard-id $shard \
          > "${out_dir}/eval/shard_${shard}.log" 2>&1 &
    done
    wait
    $PY src/eval_three_modes.py \
      --model-path "$MODEL_PATH" \
      --data ../delta-vision/data/mmstar/mmstar_val.jsonl \
      --data-root ../delta-vision \
      --checkpoint "${out_dir}/step_${MAX_STEPS}.pt" \
      --output-dir "${out_dir}/eval" \
      --num-shards 8
    echo ""
}

# Experiment 1: full answer, 2 layers (22,23), no bottleneck - baseline with unlimited tokens
run_experiment "2layer_22_23_full" "22,23" "0"

# Experiment 2: full answer, single layer 23 only
run_experiment "1layer_23_full" "23" "0"

# Experiment 3: full answer, single layer 22 only
run_experiment "1layer_22_full" "22" "0"

# Experiment 4: full answer, 2 layers (22,23), bottleneck 256
run_experiment "2layer_22_23_bn256" "22,23" "256"

echo "========================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE"
echo "========================================"
