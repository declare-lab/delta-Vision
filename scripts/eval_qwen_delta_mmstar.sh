#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BENCHMARK=${BENCHMARK:-mmstar}
MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-4B-Instruct}
DATA=${DATA:-data/mmstar/mmstar_val.jsonl}
STEP=${STEP:-500}

if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$DATA_ROOT/$MODEL_PATH"
fi
if [[ "$DATA" != /* ]]; then
  DATA="$DATA_ROOT/$DATA"
fi

if [[ -z "${CKPT:-}" && -n "${CHECKPOINT:-}" ]]; then
  CKPT="$CHECKPOINT"
fi
if [[ -z "${CKPT:-}" && -n "${RUN_DIR:-}" ]]; then
  if [[ -f "$RUN_DIR/qwen_visual_delta_step${STEP}.pt" ]]; then
    CKPT="$RUN_DIR/qwen_visual_delta_step${STEP}.pt"
  elif [[ -f "$RUN_DIR/checkpoints/qwen_visual_delta_step${STEP}.pt" ]]; then
    CKPT="$RUN_DIR/checkpoints/qwen_visual_delta_step${STEP}.pt"
  fi
fi
if [[ -z "${CKPT:-}" ]]; then
  echo "set CKPT to qwen_visual_delta_step${STEP}.pt, or set RUN_DIR" >&2
  exit 1
fi

RUN_NAME=${RUN_NAME:-$(basename "$(dirname "$(dirname "$CKPT")")")}
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/artifacts/eval/qwen_topk1024_freezeqkv/$RUN_NAME/step${STEP}_mmstar_${NUM_SHARDS}gpu}

DTYPE=${DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DELTA_VISION_IMAGE_ROOT="$DATA_ROOT"

SHARD_DIR="$OUT_DIR/shards"
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.json "$SHARD_DIR"/shard_*.log

echo "=== Qwen3-VL visual-delta MMStar eval ==="
echo "root=$ROOT_DIR"
echo "data_root=$DATA_ROOT"
echo "checkpoint=$CKPT"
echo "output=$OUT_DIR"
echo "benchmark=$BENCHMARK max_samples=$MAX_SAMPLES shards=$NUM_SHARDS"
echo "attn=$ATTN_IMPL dtype=$DTYPE"

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ "${#DEVICES[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "need at least $NUM_SHARDS CUDA devices, got ${#DEVICES[@]} from CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_CSV" >&2
  exit 1
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m src.eval_mmstar \
    --model-kind qwen \
    --data "$DATA" \
    --data-root "$DATA_ROOT" \
    --model-path "$MODEL_PATH" \
    --checkpoint "$CKPT" \
    --output-dir "$SHARD_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard" \
    --max-samples "$MAX_SAMPLES" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    > "$SHARD_DIR/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" -m src.eval_mmstar \
  --model-kind qwen \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CKPT" \
  --output-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS" \
  --max-samples "$MAX_SAMPLES" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL"

cp "$SHARD_DIR/results.json" "$OUT_DIR/results.json"
cp "$SHARD_DIR/predictions.json" "$OUT_DIR/predictions.json"
