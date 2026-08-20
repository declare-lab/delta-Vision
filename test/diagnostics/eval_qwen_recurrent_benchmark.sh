#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}
MODEL_PATH=${MODEL_PATH:-/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct}
CKPT=${CKPT:?set CKPT to qwen_embedding_adapter checkpoint}
BENCHMARK=${BENCHMARK:-mmstar}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/test/results/eval_qwen_recurrent/step500_${BENCHMARK}_1000samples_8gpu}
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
DTYPE=${DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
MEASURE_PREFILL=${MEASURE_PREFILL:-1}
COMPILE_ADAPTER=${COMPILE_ADAPTER:-0}
COMPILE_MODE=${COMPILE_MODE:-reduce-overhead}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-1}
COMPILE_VERIFY=${COMPILE_VERIFY:-0}
COMPILE_WARMUP=${COMPILE_WARMUP:-1}
LAST_LOGITS_ONLY=${LAST_LOGITS_ONLY:-1}
STRUCTURED_ANSWER_EARLY_STOP=${STRUCTURED_ANSWER_EARLY_STOP:-1}
TEACHER_CACHE=${TEACHER_CACHE:-1}
TEACHER_CACHE_DIR=${TEACHER_CACHE_DIR:-$ROOT_DIR/test/results/cache/qwen_recurrent_teacher_cache}
INPUT_CACHE_DIR=${INPUT_CACHE_DIR:-$ROOT_DIR/test/results/cache/qwen_benchmark_inputs}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/test/results/cache/torch_compile_cache}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export DELTA_VISION_IMAGE_ROOT="$DATA_ROOT"

if [[ -z "$MAX_NEW_TOKENS" ]]; then
  MAX_NEW_TOKENS="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import get_benchmark_spec
print(get_benchmark_spec(sys.argv[1]).max_new_tokens)
PY
)"
fi

DATA="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import get_benchmark_spec
print(get_benchmark_spec(sys.argv[1]).default_data)
PY
)"
if [[ "$DATA" != /* ]]; then
  DATA="$DATA_ROOT/$DATA"
fi

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ "${#DEVICES[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "need at least $NUM_SHARDS CUDA devices, got ${#DEVICES[@]} from CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_CSV" >&2
  exit 1
fi

SHARD_DIR="$OUT_DIR/shards"
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.json "$SHARD_DIR"/shard_*.log

MAX_SAMPLE_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi
PREFILL_ARGS=()
if [[ "$MEASURE_PREFILL" == "1" ]]; then
  PREFILL_ARGS=(--measure-prefill)
else
  PREFILL_ARGS=(--no-measure-prefill)
fi
COMPILE_ARGS=()
if [[ "$COMPILE_ADAPTER" == "1" ]]; then
  COMPILE_ARGS=(--compile-adapter)
else
  COMPILE_ARGS=(--no-compile-adapter)
fi
COMPILE_ARGS+=(--compile-mode "$COMPILE_MODE")
if [[ "$COMPILE_DYNAMIC" == "1" ]]; then
  COMPILE_ARGS+=(--compile-dynamic)
else
  COMPILE_ARGS+=(--no-compile-dynamic)
fi
if [[ "$COMPILE_VERIFY" == "1" ]]; then
  COMPILE_ARGS+=(--compile-verify)
else
  COMPILE_ARGS+=(--no-compile-verify)
fi
if [[ "$COMPILE_WARMUP" == "1" ]]; then
  COMPILE_ARGS+=(--compile-warmup)
else
  COMPILE_ARGS+=(--no-compile-warmup)
fi
EARLY_STOP_ARGS=()
if [[ "$STRUCTURED_ANSWER_EARLY_STOP" == "1" ]]; then
  EARLY_STOP_ARGS=(--structured-answer-early-stop)
else
  EARLY_STOP_ARGS=(--no-structured-answer-early-stop)
fi
TEACHER_CACHE_ARGS=()
if [[ "$TEACHER_CACHE" == "1" ]]; then
  TEACHER_CACHE_ARGS=(--teacher-cache --teacher-cache-dir "$TEACHER_CACHE_DIR")
else
  TEACHER_CACHE_ARGS=(--no-teacher-cache)
fi
LAST_LOGITS_ARGS=()
if [[ "$LAST_LOGITS_ONLY" == "1" ]]; then
  LAST_LOGITS_ARGS=(--last-logits-only)
else
  LAST_LOGITS_ARGS=(--no-last-logits-only)
fi

echo "=== Test-only recurrent Qwen benchmark eval ==="
echo "benchmark=$BENCHMARK checkpoint=$CKPT out=$OUT_DIR shards=$NUM_SHARDS compile=$COMPILE_ADAPTER"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" test/diagnostics/eval_qwen_recurrent_embedding_adapter.py \
    --model-kind qwen \
    --benchmark "$BENCHMARK" \
    --data "$DATA" \
    --data-root "$DATA_ROOT" \
    --model-path "$MODEL_PATH" \
    --checkpoint "$CKPT" \
    --output-dir "$SHARD_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${MAX_SAMPLE_ARGS[@]}" \
    "${PREFILL_ARGS[@]}" \
    "${COMPILE_ARGS[@]}" \
    "${EARLY_STOP_ARGS[@]}" \
    "${TEACHER_CACHE_ARGS[@]}" \
    "${LAST_LOGITS_ARGS[@]}" \
    --input-cache-dir "$INPUT_CACHE_DIR" \
    --no-adapter-decode-cache \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    > "$SHARD_DIR/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" test/diagnostics/eval_qwen_recurrent_embedding_adapter.py \
  --model-kind qwen \
  --benchmark "$BENCHMARK" \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CKPT" \
  --output-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${PREFILL_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "${EARLY_STOP_ARGS[@]}" \
  "${TEACHER_CACHE_ARGS[@]}" \
  "${LAST_LOGITS_ARGS[@]}" \
  --input-cache-dir "$INPUT_CACHE_DIR" \
  --no-adapter-decode-cache \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL"

cp "$SHARD_DIR/results.json" "$OUT_DIR/results.json"
cp "$SHARD_DIR/predictions.json" "$OUT_DIR/predictions.json"
if [[ -f "$SHARD_DIR/summary.csv" ]]; then
  cp "$SHARD_DIR/summary.csv" "$OUT_DIR/summary.csv"
fi

echo "Done. Results at $OUT_DIR/results.json"
