#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-../delta-vision/models/llava-1.5-7b-hf}
DATA=${DATA:-../delta-vision/data/mmstar/mmstar_val.jsonl}
DATA_ROOT=${DATA_ROOT:-../delta-vision}
CHECKPOINT=${CHECKPOINT:?Error: set CHECKPOINT to adapter .pt file}
OUT_DIR=${OUT_DIR:-artifacts/eval_benchmark_$(date +%Y%m%d_%H%M%S)}

NUM_SHARDS=${NUM_SHARDS:-8}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== Vision KV Inject - Benchmark Evaluation (default: MMStar) ==="
echo "checkpoint=$CHECKPOINT"
echo "output=$OUT_DIR"
echo "shards=$NUM_SHARDS"

mkdir -p "$OUT_DIR"

IFS="," read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ ${#DEVICES[@]} -lt $NUM_SHARDS ]]; then
  echo "Need at least $NUM_SHARDS GPUs, got ${#DEVICES[@]}" >&2
  exit 1
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m src.eval_benchmarks \
    --model-path "$MODEL_PATH" \
    --data "$DATA" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard" \
    > "$OUT_DIR/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

echo "Waiting for all shards..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== Merging results ==="
$PY -m src.eval_benchmarks \
  --model-path "$MODEL_PATH" \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --num-shards "$NUM_SHARDS"

echo "Done. Results at $OUT_DIR/results.json"
