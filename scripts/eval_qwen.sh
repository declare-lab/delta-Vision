#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
DATA=${DATA:-../delta-vision/data/mmstar/mmstar_val.jsonl}
DATA_ROOT=${DATA_ROOT:-../delta-vision}
CHECKPOINT=${CHECKPOINT:?Error: set CHECKPOINT to adapter .pt file}
OUT_DIR=${OUT_DIR:-artifacts/eval_qwen_mmstar_$(date +%Y%m%d_%H%M%S)}

NUM_SHARDS=${NUM_SHARDS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== Vision KV Inject - Qwen Benchmark Generation Evaluation (default: MMStar) ==="
echo "checkpoint=$CHECKPOINT"
echo "output=$OUT_DIR"
echo "shards=$NUM_SHARDS"
if [[ -n "$MAX_SAMPLES" ]]; then
  echo "max_samples=$MAX_SAMPLES"
fi

mkdir -p "$OUT_DIR"

IFS="," read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ ${#DEVICES[@]} -lt $NUM_SHARDS ]]; then
  echo "Need at least $NUM_SHARDS GPUs, got ${#DEVICES[@]}" >&2
  exit 1
fi

pids=()
MAX_SAMPLE_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m src.eval_benchmarks \
    --model-kind qwen \
    --model-path "$MODEL_PATH" \
    --data "$DATA" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${MAX_SAMPLE_ARGS[@]}" \
    > "$OUT_DIR/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

echo "Waiting for all shards..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== Merging results ==="
$PY -m src.eval_benchmarks \
  --model-kind qwen \
  --model-path "$MODEL_PATH" \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --num-shards "$NUM_SHARDS" \
  "${MAX_SAMPLE_ARGS[@]}"

echo "Done. Results at $OUT_DIR/results.json"
