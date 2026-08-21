#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${RUN_DIR:-}" && -z "${CHECKPOINT:-}" ]]; then
  echo "set RUN_DIR or CHECKPOINT" >&2
  exit 1
fi

MODEL_KIND=${MODEL_KIND:-qwen}
BENCHMARK=${BENCHMARK:-rendered-context-qa}
DATA=${DATA:-$ROOT_DIR/data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_span_100.jsonl}
STEP=${STEP:-500}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_SHARDS=${NUM_SHARDS:-8}
COMPILE_ADAPTER=${COMPILE_ADAPTER:-0}
FORCE_EVAL=${FORCE_EVAL:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export MODEL_KIND BENCHMARK DATA STEP MAX_SAMPLES MAX_NEW_TOKENS NUM_SHARDS COMPILE_ADAPTER FORCE_EVAL CUDA_VISIBLE_DEVICES

CMD=(
  "$ROOT_DIR/scripts/eval_benchmark.sh"
  "$BENCHMARK"
  --model-kind "$MODEL_KIND"
  --data "$DATA"
  --step "$STEP"
  --max-samples "$MAX_SAMPLES"
  --num-shards "$NUM_SHARDS"
  --force
)

if [[ -n "${RUN_DIR:-}" ]]; then
  CMD+=(--run-dir "$RUN_DIR")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  CMD+=(--out-dir "$OUT_DIR")
fi

"${CMD[@]}"
