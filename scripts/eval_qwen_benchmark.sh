#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BENCHMARK=${BENCHMARK:-mmstar}
MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-4B-Instruct}
STEP=${STEP:-500}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --data)
      DATA="$2"
      shift 2
      ;;
    --checkpoint|--ckpt)
      CKPT="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/eval_qwen_benchmark.sh [benchmark] [options]

Options:
  --benchmark NAME       Benchmark name, e.g. mmstar, ocrbench, textvqa.
  --data PATH            Override benchmark JSONL path.
  --checkpoint PATH      Adapter checkpoint path.
  --run-dir PATH         Run directory containing checkpoints.
  --step N               Checkpoint step.
  --out-dir PATH         Output directory.
  --max-samples N        Evaluation subset size.
  --num-shards N         Number of GPU shards.
EOF
      exit 0
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 1
      ;;
    *)
      BENCHMARK="$1"
      shift
      ;;
  esac
done

BENCHMARK="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import canonical_benchmark_name
print(canonical_benchmark_name(sys.argv[1]))
PY
)"

if [[ -z "${DATA:-}" ]]; then
  DATA="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import get_benchmark_spec
print(get_benchmark_spec(sys.argv[1]).default_data)
PY
)"
fi

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
if [[ -z "${MAX_SAMPLES+x}" ]]; then
  if [[ "$BENCHMARK" == "mmstar" ]]; then
    MAX_SAMPLES=1000
  else
    MAX_SAMPLES=
  fi
fi
OUT_DIR=${OUT_DIR:-$ROOT_DIR/artifacts/eval/qwen_topk1024_freezeqkv/$RUN_NAME/step${STEP}_${BENCHMARK}_${NUM_SHARDS}gpu}

DTYPE=${DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [[ -z "${MAX_NEW_TOKENS+x}" ]]; then
  MAX_NEW_TOKENS="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import get_benchmark_spec
print(get_benchmark_spec(sys.argv[1]).max_new_tokens)
PY
)"
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/artifacts/torch_compile_cache}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export DELTA_VISION_IMAGE_ROOT="$DATA_ROOT"

SHARD_DIR="$OUT_DIR/shards"
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.json "$SHARD_DIR"/shard_*.log

echo "=== Qwen3-VL visual-delta benchmark eval ==="
echo "root=$ROOT_DIR"
echo "data_root=$DATA_ROOT"
echo "checkpoint=$CKPT"
echo "output=$OUT_DIR"
echo "benchmark=$BENCHMARK max_samples=$MAX_SAMPLES shards=$NUM_SHARDS"
echo "attn=$ATTN_IMPL dtype=$DTYPE"
echo "compile_adapter=${COMPILE_ADAPTER:-1} compile_mode=${COMPILE_MODE:-reduce-overhead} compile_dynamic=${COMPILE_DYNAMIC:-1} compile_warmup=${COMPILE_WARMUP:-1}"
echo "torchinductor_cache=$TORCHINDUCTOR_CACHE_DIR compile_threads=$TORCHINDUCTOR_COMPILE_THREADS"
echo "structured_answer_early_stop=${STRUCTURED_ANSWER_EARLY_STOP:-1}"
echo "teacher_cache=${TEACHER_CACHE:-1} teacher_cache_dir=${TEACHER_CACHE_DIR:-}"
echo "require_teacher_cache=${REQUIRE_TEACHER_CACHE:-0}"
echo "last_logits_only=${LAST_LOGITS_ONLY:-1}"
echo "input_cache=${INPUT_CACHE:-1} input_cache_dir=${INPUT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_benchmark_inputs}"
echo "context_cache=${CONTEXT_CACHE:-0} context_cache_dir=${CONTEXT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_initial_contexts}"

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ "${#DEVICES[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "need at least $NUM_SHARDS CUDA devices, got ${#DEVICES[@]} from CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_CSV" >&2
  exit 1
fi

pids=()
MAX_SAMPLE_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi
ANSWER_ARGS=()
if [[ -n "${ANSWER_INSTRUCTION:-}" ]]; then
  ANSWER_ARGS=(--answer-instruction "$ANSWER_INSTRUCTION")
fi
PREFILL_ARGS=()
if [[ "${MEASURE_PREFILL:-1}" == "0" ]]; then
  PREFILL_ARGS=(--no-measure-prefill)
fi
EARLY_STOP_ARGS=()
if [[ "${STRUCTURED_ANSWER_EARLY_STOP:-1}" == "1" ]]; then
  EARLY_STOP_ARGS=(--structured-answer-early-stop)
else
  EARLY_STOP_ARGS=(--no-structured-answer-early-stop)
fi
COMPILE_ARGS=()
if [[ "${COMPILE_ADAPTER:-1}" == "1" ]]; then
  COMPILE_ARGS=(--compile-adapter)
else
  COMPILE_ARGS=(--no-compile-adapter)
fi
COMPILE_ARGS+=(--compile-mode "${COMPILE_MODE:-reduce-overhead}")
if [[ "${COMPILE_DYNAMIC:-1}" == "1" ]]; then
  COMPILE_ARGS+=(--compile-dynamic)
else
  COMPILE_ARGS+=(--no-compile-dynamic)
fi
if [[ "${COMPILE_WARMUP:-1}" == "1" ]]; then
  COMPILE_ARGS+=(--compile-warmup)
else
  COMPILE_ARGS+=(--no-compile-warmup)
fi
TEACHER_CACHE_ARGS=()
if [[ "${TEACHER_CACHE:-1}" == "1" ]]; then
  TEACHER_CACHE_ARGS=(--teacher-cache)
  if [[ -n "${TEACHER_CACHE_DIR:-}" ]]; then
    TEACHER_CACHE_ARGS+=(--teacher-cache-dir "$TEACHER_CACHE_DIR")
  fi
else
  TEACHER_CACHE_ARGS=(--no-teacher-cache)
fi
if [[ "${REQUIRE_TEACHER_CACHE:-0}" == "1" ]]; then
  TEACHER_CACHE_ARGS+=(--require-teacher-cache)
fi
LAST_LOGITS_ARGS=()
if [[ "${LAST_LOGITS_ONLY:-1}" == "1" ]]; then
  LAST_LOGITS_ARGS=(--last-logits-only)
else
  LAST_LOGITS_ARGS=(--no-last-logits-only)
fi
INPUT_CACHE_ARGS=()
if [[ "${INPUT_CACHE:-1}" == "1" ]]; then
  INPUT_CACHE_ARGS=(--input-cache-dir "${INPUT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_benchmark_inputs}")
fi
CONTEXT_CACHE_ARGS=()
if [[ "${CONTEXT_CACHE:-0}" == "1" ]]; then
  CONTEXT_CACHE_ARGS=(--context-cache-dir "${CONTEXT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_initial_contexts}")
fi
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m src.eval_benchmarks \
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
    "${ANSWER_ARGS[@]}" \
    "${PREFILL_ARGS[@]}" \
    "${EARLY_STOP_ARGS[@]}" \
    "${COMPILE_ARGS[@]}" \
    "${TEACHER_CACHE_ARGS[@]}" \
    "${LAST_LOGITS_ARGS[@]}" \
    "${INPUT_CACHE_ARGS[@]}" \
    "${CONTEXT_CACHE_ARGS[@]}" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    > "$SHARD_DIR/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" -m src.eval_benchmarks \
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
  "${ANSWER_ARGS[@]}" \
  "${PREFILL_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "${TEACHER_CACHE_ARGS[@]}" \
  "${LAST_LOGITS_ARGS[@]}" \
  "${INPUT_CACHE_ARGS[@]}" \
  "${CONTEXT_CACHE_ARGS[@]}" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL"

cp "$SHARD_DIR/results.json" "$OUT_DIR/results.json"
cp "$SHARD_DIR/predictions.json" "$OUT_DIR/predictions.json"
if [[ -f "$SHARD_DIR/summary.csv" ]]; then
  cp "$SHARD_DIR/summary.csv" "$OUT_DIR/summary.csv"
fi
