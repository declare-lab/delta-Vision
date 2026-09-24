#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT_EXPLICIT=${DATA_ROOT+x}
DATA_ROOT=${DATA_ROOT:-$ROOT_DIR}
MODEL_ROOT=${MODEL_ROOT:-/lustre-data/leijingdi/code/delta-vision}
PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL_KIND=${MODEL_KIND:-qwen}
BENCHMARK=${BENCHMARK:-mmstar}
BENCHMARKS=${BENCHMARKS:-}
OUTPUT_MODE=${OUTPUT_MODE:-}
TEACHER_ONLY=${TEACHER_ONLY:-0}
STEP=${STEP:-500}
STEPS=${STEPS:-}
EVAL_ALL_CKPTS=${EVAL_ALL_CKPTS:-0}
FORCE_EVAL=${FORCE_EVAL:-0}
NUM_SHARDS_EXPLICIT=${NUM_SHARDS+x}
POSITIONAL_BENCHMARKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-kind)
      MODEL_KIND="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
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
    --output-mode)
      OUTPUT_MODE="$2"
      shift 2
      ;;
    --teacher-only)
      TEACHER_ONLY=1
      shift
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --all-ckpts)
      EVAL_ALL_CKPTS=1
      shift
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
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
    --force)
      FORCE_EVAL=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/eval_benchmark.sh [benchmark ...] [options]

Single eval:
  scripts/eval_benchmark.sh sqa --model-kind qwen --run-dir RUN --step 12000

Batch eval:
  scripts/eval_benchmark.sh --benchmarks sqa,vqav2,realworldqa --run-dir RUN --step 12000
  scripts/eval_benchmark.sh --benchmarks all --run-dir RUN --steps "1000 2000 3000"  # default benchmark set
  scripts/eval_benchmark.sh --benchmarks all --run-dir RUN --all-ckpts              # default benchmark set

Options:
  --model-kind llava|qwen  Model family. Can also set MODEL_KIND.
  --benchmark NAME         Single benchmark name.
  --benchmarks LIST        Comma/space-separated benchmark names, or all.
  --model-path PATH        Base model path. Relative paths resolve under MODEL_ROOT.
  --data PATH              Override benchmark JSONL path. Use only for single eval.
  --checkpoint PATH        Adapter checkpoint path. Use only for single eval.
  --output-mode MODE       Override checkpoint output mode, e.g. recurrent_embedding_adapter.
  --teacher-only           Evaluate only the base model; no checkpoint required.
  --run-dir PATH           Run directory containing checkpoints.
  --step N                 Single checkpoint step.
  --steps LIST             Space-separated checkpoint steps.
  --all-ckpts              Evaluate every supported step checkpoint under RUN_DIR.
  --out-dir PATH           Single-eval output directory.
  --out-root PATH          Batch output root.
  --max-samples N          Evaluation subset size.
  --num-shards N           Number of GPU shards.
  --force                  Re-run existing batch results.

Runtime env:
  COMPILE_ADAPTER=0        Use eager adapter path.
  MEASURE_PREFILL=0        Skip timing-only prefill measurement.
  TEACHER_CACHE=1          Cache deterministic Qwen teacher generations.
EOF
      exit 0
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 1
      ;;
    *)
      POSITIONAL_BENCHMARKS+=("$1")
      shift
      ;;
  esac
done

MODEL_KIND="$(printf '%s' "$MODEL_KIND" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL_KIND" != "qwen" && "$MODEL_KIND" != "llava" ]]; then
  echo "MODEL_KIND must be qwen or llava, got $MODEL_KIND" >&2
  exit 1
fi
QWEN_DEVICE_MAP=${QWEN_DEVICE_MAP:-}
QWEN_MAX_MEMORY=${QWEN_MAX_MEMORY:-}
SHARDED_QWEN=0
if [[ "$MODEL_KIND" == "qwen" && -n "$QWEN_DEVICE_MAP" && "$QWEN_DEVICE_MAP" != "none" && "$QWEN_DEVICE_MAP" != "replicated" ]]; then
  SHARDED_QWEN=1
fi

if [[ "${#POSITIONAL_BENCHMARKS[@]}" -gt 1 ]]; then
  BENCHMARKS="${POSITIONAL_BENCHMARKS[*]}"
elif [[ "${#POSITIONAL_BENCHMARKS[@]}" -eq 1 ]]; then
  BENCHMARK="${POSITIONAL_BENCHMARKS[0]}"
fi

resolve_benchmarks() {
  "$PY" - "$1" <<'PY'
import sys
from src.benchmarks import parse_benchmark_names
print(" ".join(parse_benchmark_names(sys.argv[1])))
PY
}

checkpoint_dir_for_run() {
  local run_dir="$1"
  if [[ -d "$run_dir/checkpoints" ]]; then
    printf '%s\n' "$run_dir/checkpoints"
  else
    printf '%s\n' "$run_dir"
  fi
}

step_list_for_run() {
  local run_dir="$1"
  local ckpt_dir
  ckpt_dir="$(checkpoint_dir_for_run "$run_dir")"
  if [[ "$MODEL_KIND" == "qwen" ]]; then
    local legacy_qwen_checkpoint_prefix="qwen_visual""_""del""ta"
    find "$ckpt_dir" -maxdepth 1 -type f \( -name 'qwen_embedding_adapter_step*.pt' -o -name 'qwen_recurrent_embedding_adapter_step*.pt' -o -name "${legacy_qwen_checkpoint_prefix}_step*.pt" \) -printf '%f\n' \
      | sed -E "s/^(qwen_embedding_adapter|qwen_recurrent_embedding_adapter|${legacy_qwen_checkpoint_prefix})_step([0-9]+)\.pt$/\2/" \
      | sort -n -u
  else
    find "$ckpt_dir" -maxdepth 1 -type f -name 'step_*.pt' -printf '%f\n' \
      | sed -E 's/^step_([0-9]+)\.pt$/\1/' \
      | sort -n -u
  fi
}

if [[ "${SINGLE_EVAL:-0}" != "1" && ( -n "$BENCHMARKS" || -n "$STEPS" || "$EVAL_ALL_CKPTS" == "1" ) ]]; then
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "batch eval requires --run-dir" >&2
    exit 1
  fi
  BENCHMARK_LIST="$(resolve_benchmarks "${BENCHMARKS:-$BENCHMARK}")"
  if [[ -z "$STEPS" && "$EVAL_ALL_CKPTS" == "1" ]]; then
    STEPS="$(step_list_for_run "$RUN_DIR" | xargs)"
    if [[ -z "$STEPS" ]]; then
      echo "no supported step checkpoints found under $(checkpoint_dir_for_run "$RUN_DIR")" >&2
      exit 1
    fi
  else
    STEPS=${STEPS:-$STEP}
  fi

  RUN_LABEL="$(basename "$RUN_DIR")"
  if [[ "$RUN_LABEL" == "checkpoints" ]]; then
    RUN_LABEL="$(basename "$(dirname "$RUN_DIR")")"
  fi
  if [[ "$SHARDED_QWEN" == "1" && -z "$NUM_SHARDS_EXPLICIT" ]]; then
    NUM_SHARDS=1
  else
    NUM_SHARDS=${NUM_SHARDS:-8}
  fi
  if [[ "$SHARDED_QWEN" == "1" && "$NUM_SHARDS" != "1" ]]; then
    echo "QWEN_DEVICE_MAP=$QWEN_DEVICE_MAP requires NUM_SHARDS=1 so one process can see all visible GPUs" >&2
    exit 1
  fi
  MAX_SAMPLES=${MAX_SAMPLES:-1000}
  if [[ -z "${RUNTIME_TAG:-}" ]]; then
    if [[ "${COMPILE_ADAPTER:-1}" == "1" ]]; then
      RUNTIME_TAG=compiled
    else
      RUNTIME_TAG=eager
    fi
  fi
  OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/artifacts/eval/$MODEL_KIND/$RUN_LABEL/$RUNTIME_TAG}
  TEACHER_CACHE_DIR=${TEACHER_CACHE_DIR:-$OUT_ROOT/teacher_cache}

  echo "=== Unified VLM adapter benchmark batch eval ==="
  echo "root=$ROOT_DIR"
  echo "model_kind=$MODEL_KIND"
  echo "run_dir=$RUN_DIR"
  echo "steps=$STEPS"
  echo "benchmarks=$BENCHMARK_LIST"
  echo "max_samples=$MAX_SAMPLES"
  echo "num_shards=$NUM_SHARDS cuda=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  echo "compile_adapter=${COMPILE_ADAPTER:-1} compile_verify=${COMPILE_VERIFY:-0}"
  echo "out_root=$OUT_ROOT"
  echo "force_eval=$FORCE_EVAL"

  for step in $STEPS; do
    for benchmark in $BENCHMARK_LIST; do
      out_dir="$OUT_ROOT/step${step}_${benchmark}_${MAX_SAMPLES}samples_${NUM_SHARDS}gpu"
      if [[ "$FORCE_EVAL" != "1" && -s "$out_dir/results.json" ]]; then
        echo "=== SKIP existing step=$step benchmark=$benchmark $out_dir/results.json ==="
        continue
      fi
      echo "=== START step=$step benchmark=$benchmark $(date) ==="
      MODEL_KIND="$MODEL_KIND" \
      BENCHMARKS= \
      STEPS= \
      EVAL_ALL_CKPTS=0 \
      RUN_DIR="$RUN_DIR" \
      STEP="$step" \
      OUT_DIR="$out_dir" \
      OUTPUT_MODE="$OUTPUT_MODE" \
      MAX_SAMPLES="$MAX_SAMPLES" \
      NUM_SHARDS="$NUM_SHARDS" \
      QWEN_DEVICE_MAP="$QWEN_DEVICE_MAP" \
      QWEN_MAX_MEMORY="$QWEN_MAX_MEMORY" \
      TEACHER_CACHE_DIR="$TEACHER_CACHE_DIR" \
      SINGLE_EVAL=1 \
      bash "$0" "$benchmark"
      echo "=== DONE step=$step benchmark=$benchmark $(date) ==="
    done
  done

  "$PY" - "$OUT_ROOT" "$MAX_SAMPLES" "$NUM_SHARDS" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
max_samples = sys.argv[2]
num_shards = sys.argv[3]
rows = []
for result_path in sorted(out_root.glob(f"step*_*_{max_samples}samples_{num_shards}gpu/results.json")):
    name = result_path.parent.name
    if not name.startswith("step"):
        continue
    try:
        step_text, rest = name[4:].split("_", 1)
        benchmark = rest.rsplit(f"_{max_samples}samples_{num_shards}gpu", 1)[0]
        step = int(step_text) if step_text.isdigit() else step_text
    except ValueError:
        continue
    data = json.loads(result_path.read_text(encoding="utf-8"))
    teacher = data.get("teacher") or {}
    adapter = data.get("adapter") or {}
    timing = data.get("timing", {})
    resources = data.get("resources", {})
    teacher_score = teacher.get("score")
    adapter_score = adapter.get("score")
    rows.append({
        "step": step,
        "benchmark": data.get("benchmark", benchmark),
        "display_name": data.get("display_name", benchmark),
        "metric": data.get("metric", ""),
        "total_samples": data.get("total_samples"),
        "teacher_score": teacher_score,
        "adapter_score": adapter_score,
        "gap": (teacher_score - adapter_score) if teacher_score is not None and adapter_score is not None else None,
        "retention": data.get("retention"),
        "agreement": data.get("agreement"),
        "teacher_invalid_rate": teacher.get("invalid_rate"),
        "adapter_invalid_rate": adapter.get("invalid_rate"),
        "teacher_total_s": timing.get("teacher_total_s"),
        "adapter_total_s": timing.get("adapter_total_s"),
        "teacher_prefill_s": timing.get("teacher_prefill_s"),
        "adapter_prefill_s": timing.get("adapter_prefill_s"),
        "speedup_total": timing.get("speedup_total"),
        "speedup_prefill": timing.get("speedup_prefill"),
        "teacher_kv_cache_mb": resources.get("teacher_kv_cache_mb_avg"),
        "adapter_kv_cache_mb": resources.get("adapter_kv_cache_mb_avg"),
        "teacher_prefill_flops": resources.get("teacher_prefill_flops_avg"),
        "adapter_prefill_flops": resources.get("adapter_prefill_flops_avg"),
        "pope_f1": data.get("pope_f1"),
        "result_dir": str(result_path.parent),
    })

def step_sort_key(step):
    if isinstance(step, int):
        return (0, step)
    return (1, str(step))

rows.sort(key=lambda item: (*step_sort_key(item["step"]), str(item["benchmark"])))
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "all_ckpt_benchmark_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
fields = list(rows[0].keys()) if rows else [
    "step", "benchmark", "display_name", "metric", "total_samples", "teacher_score", "adapter_score", "gap",
    "retention", "agreement", "teacher_invalid_rate", "adapter_invalid_rate", "teacher_total_s", "adapter_total_s",
    "teacher_prefill_s", "adapter_prefill_s", "speedup_total", "speedup_prefill", "teacher_kv_cache_mb",
    "adapter_kv_cache_mb", "teacher_prefill_flops", "adapter_prefill_flops", "pope_f1", "result_dir",
]
with (out_root / "all_ckpt_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

benchmarks = sorted({row["benchmark"] for row in rows})
by_step = {}
for row in rows:
    by_step.setdefault(row["step"], {})[row["benchmark"]] = row.get("adapter_score")
with (out_root / "all_ckpt_scores_wide.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", *benchmarks])
    for step in sorted(by_step, key=step_sort_key):
        writer.writerow([step, *[by_step[step].get(benchmark) for benchmark in benchmarks]])
print(f"Wrote aggregate summaries under {out_root}")
PY
  echo "=== ALL DONE $(date) ==="
  exit 0
fi

BENCHMARK="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import canonical_benchmark_name
print(canonical_benchmark_name(sys.argv[1]))
PY
)"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ "$MODEL_KIND" == "qwen" ]]; then
    MODEL_PATH="models/Qwen3-VL-4B-Instruct"
  else
    MODEL_PATH="models/llava-1.5-7b-hf"
  fi
fi
if [[ -z "${DATA:-}" ]]; then
  DATA="$("$PY" - "$BENCHMARK" <<'PY'
import sys
from src.benchmarks import get_benchmark_spec
print(get_benchmark_spec(sys.argv[1]).default_data)
PY
)"
fi

if [[ "$MODEL_PATH" != /* ]]; then
  if [[ -e "$ROOT_DIR/$MODEL_PATH" ]]; then
    MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
  elif [[ -e "$MODEL_ROOT/$MODEL_PATH" ]]; then
    MODEL_PATH="$MODEL_ROOT/$MODEL_PATH"
  elif [[ "$MODEL_PATH" != */*/* && "$MODEL_PATH" == */* ]]; then
    MODEL_PATH="$MODEL_PATH"
  else
    MODEL_PATH="$MODEL_ROOT/$MODEL_PATH"
  fi
fi
if [[ "$DATA" != /* ]]; then
  if [[ -f "$ROOT_DIR/$DATA" ]]; then
    DATA="$ROOT_DIR/$DATA"
  else
    DATA="$DATA_ROOT/$DATA"
  fi
fi
if [[ -z "$DATA_ROOT_EXPLICIT" && "$DATA" == "$ROOT_DIR"/data/benchmarks/* ]]; then
  DATA_ROOT="$(dirname "$DATA")"
fi

if [[ -z "${CKPT:-}" && -n "${CHECKPOINT:-}" ]]; then
  CKPT="$CHECKPOINT"
fi
if [[ -z "${CKPT:-}" && -n "${RUN_DIR:-}" ]]; then
  if [[ "$MODEL_KIND" == "qwen" ]]; then
    legacy_qwen_checkpoint_prefix="qwen_visual""_""del""ta"
    for candidate in \
      "$RUN_DIR/qwen_embedding_adapter_step${STEP}.pt" \
      "$RUN_DIR/checkpoints/qwen_embedding_adapter_step${STEP}.pt" \
      "$RUN_DIR/qwen_embedding_adapter_final.pt" \
      "$RUN_DIR/checkpoints/qwen_embedding_adapter_final.pt" \
      "$RUN_DIR/qwen_recurrent_embedding_adapter_step${STEP}.pt" \
      "$RUN_DIR/checkpoints/qwen_recurrent_embedding_adapter_step${STEP}.pt" \
      "$RUN_DIR/qwen_recurrent_embedding_adapter_final.pt" \
      "$RUN_DIR/checkpoints/qwen_recurrent_embedding_adapter_final.pt" \
      "$RUN_DIR/${legacy_qwen_checkpoint_prefix}_step${STEP}.pt" \
      "$RUN_DIR/checkpoints/${legacy_qwen_checkpoint_prefix}_step${STEP}.pt" \
      "$RUN_DIR/${legacy_qwen_checkpoint_prefix}_final.pt" \
      "$RUN_DIR/checkpoints/${legacy_qwen_checkpoint_prefix}_final.pt"; do
      if [[ -f "$candidate" ]]; then
        CKPT="$candidate"
        break
      fi
    done
  else
    for candidate in \
      "$RUN_DIR/step_${STEP}.pt" \
      "$RUN_DIR/checkpoints/step_${STEP}.pt" \
      "$RUN_DIR/final.pt" \
      "$RUN_DIR/checkpoints/final.pt"; do
      if [[ -f "$candidate" ]]; then
        CKPT="$candidate"
        break
      fi
    done
  fi
fi
if [[ -z "${CKPT:-}" && "$TEACHER_ONLY" != "1" ]]; then
  echo "set CKPT/CHECKPOINT, or set RUN_DIR with a supported checkpoint name" >&2
  exit 1
fi

if [[ "$TEACHER_ONLY" == "1" ]]; then
  RUN_NAME=${RUN_NAME:-$(basename "$MODEL_PATH")_teacher_only}
else
  RUN_NAME=${RUN_NAME:-$(basename "$(dirname "$CKPT")")}
  if [[ "$RUN_NAME" == "checkpoints" ]]; then
    RUN_NAME="$(basename "$(dirname "$(dirname "$CKPT")")")"
  fi
fi
if [[ "$SHARDED_QWEN" == "1" && -z "$NUM_SHARDS_EXPLICIT" ]]; then
  NUM_SHARDS=1
else
  NUM_SHARDS=${NUM_SHARDS:-8}
fi
if [[ "$SHARDED_QWEN" == "1" && "$NUM_SHARDS" != "1" ]]; then
  echo "QWEN_DEVICE_MAP=$QWEN_DEVICE_MAP requires NUM_SHARDS=1 so one process can see all visible GPUs" >&2
  exit 1
fi
if [[ -z "${MAX_SAMPLES+x}" ]]; then
  if [[ "$BENCHMARK" == "mmstar" ]]; then
    MAX_SAMPLES=1000
  else
    MAX_SAMPLES=
  fi
fi
OUT_DIR=${OUT_DIR:-$ROOT_DIR/artifacts/eval/$MODEL_KIND/$RUN_NAME/step${STEP}_${BENCHMARK}_${NUM_SHARDS}gpu}

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

echo "=== Unified VLM adapter benchmark eval ==="
echo "root=$ROOT_DIR"
echo "model_kind=$MODEL_KIND"
echo "data_root=$DATA_ROOT"
echo "model_root=$MODEL_ROOT"
echo "model_path=$MODEL_PATH"
echo "checkpoint=${CKPT:-teacher_only}"
echo "teacher_only=$TEACHER_ONLY"
echo "output=$OUT_DIR"
echo "benchmark=$BENCHMARK max_samples=$MAX_SAMPLES max_new_tokens=$MAX_NEW_TOKENS shards=$NUM_SHARDS"
echo "attn=$ATTN_IMPL dtype=$DTYPE"
echo "qwen_device_map=${QWEN_DEVICE_MAP:-none} qwen_max_memory=${QWEN_MAX_MEMORY:-auto}"
echo "measure_prefill=${MEASURE_PREFILL:-1}"
echo "compile_adapter=${COMPILE_ADAPTER:-1} compile_verify=${COMPILE_VERIFY:-0}"
echo "adapter_decode_cache=${ADAPTER_DECODE_CACHE:-1} adapter_decode_cache_mode=${ADAPTER_DECODE_CACHE_MODE:-shape_exact}"
echo "structured_answer_early_stop=${STRUCTURED_ANSWER_EARLY_STOP:-1}"
echo "eval_batch_size=${EVAL_BATCH_SIZE:-128} eval_max_batch_tokens=${EVAL_MAX_BATCH_TOKENS:-0}"

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ "${#DEVICES[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "need at least $NUM_SHARDS CUDA devices, got ${#DEVICES[@]} from CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_CSV" >&2
  exit 1
fi

MAX_SAMPLE_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi
ANSWER_ARGS=()
if [[ -n "${ANSWER_INSTRUCTION:-}" ]]; then
  ANSWER_ARGS=(--answer-instruction "$ANSWER_INSTRUCTION")
fi
OUTPUT_MODE_ARGS=()
if [[ -n "$OUTPUT_MODE" ]]; then
  OUTPUT_MODE_ARGS=(--output-mode "$OUTPUT_MODE")
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
if [[ "${COMPILE_VERIFY:-0}" == "1" ]]; then
  COMPILE_ARGS+=(--compile-verify)
else
  COMPILE_ARGS+=(--no-compile-verify)
fi
if [[ -n "${COMPILE_MAX_DIFF:-}" ]]; then
  COMPILE_ARGS+=(--compile-max-diff "$COMPILE_MAX_DIFF")
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
ADAPTER_DECODE_ARGS=()
if [[ "${ADAPTER_DECODE_CACHE:-1}" == "1" ]]; then
  ADAPTER_DECODE_ARGS=(--adapter-decode-cache)
else
  ADAPTER_DECODE_ARGS=(--no-adapter-decode-cache)
fi
ADAPTER_DECODE_ARGS+=(--adapter-decode-cache-mode "${ADAPTER_DECODE_CACHE_MODE:-shape_exact}")
if [[ -n "${VERIFY_DECODE_CACHE_GENERATION:-}" ]]; then
  ADAPTER_DECODE_ARGS+=(--verify-decode-cache-generation "$VERIFY_DECODE_CACHE_GENERATION")
fi
INPUT_CACHE_ARGS=()
if [[ "$MODEL_KIND" == "qwen" && "${INPUT_CACHE:-1}" == "1" ]]; then
  INPUT_CACHE_ARGS=(--input-cache-dir "${INPUT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_benchmark_inputs}")
fi
CONTEXT_CACHE_ARGS=()
if [[ "$MODEL_KIND" == "qwen" && "${CONTEXT_CACHE:-0}" == "1" ]]; then
  CONTEXT_CACHE_ARGS=(--context-cache-dir "${CONTEXT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_initial_contexts}")
fi
EVAL_BATCH_ARGS=()
if [[ "$MODEL_KIND" == "qwen" ]]; then
  EVAL_BATCH_ARGS=(--eval-batch-size "${EVAL_BATCH_SIZE:-128}" --eval-max-batch-tokens "${EVAL_MAX_BATCH_TOKENS:-0}")
fi
QWEN_DEVICE_MAP_ARGS=()
if [[ "$MODEL_KIND" == "qwen" ]]; then
  QWEN_DEVICE_MAP_ARGS=(--qwen-device-map "$QWEN_DEVICE_MAP" --qwen-max-memory "$QWEN_MAX_MEMORY")
fi
TEACHER_ONLY_ARGS=()
CHECKPOINT_ARGS=()
if [[ "$TEACHER_ONLY" == "1" ]]; then
  TEACHER_ONLY_ARGS=(--teacher-only)
else
  CHECKPOINT_ARGS=(--checkpoint "$CKPT")
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  if [[ "$SHARDED_QWEN" == "1" ]]; then
    shard_cuda="$CUDA_DEVICES_CSV"
  else
    shard_cuda="$gpu"
  fi
  echo "Launching shard $shard on CUDA_VISIBLE_DEVICES=$shard_cuda"
  CUDA_VISIBLE_DEVICES="$shard_cuda" "$PY" -m src.run eval --family "$MODEL_KIND" -- \
    --model-kind "$MODEL_KIND" \
    --benchmark "$BENCHMARK" \
    --data "$DATA" \
    --data-root "$DATA_ROOT" \
    --model-path "$MODEL_PATH" \
    "${CHECKPOINT_ARGS[@]}" \
    --output-dir "$SHARD_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${MAX_SAMPLE_ARGS[@]}" \
    "${ANSWER_ARGS[@]}" \
    "${OUTPUT_MODE_ARGS[@]}" \
    "${PREFILL_ARGS[@]}" \
    "${EARLY_STOP_ARGS[@]}" \
    "${COMPILE_ARGS[@]}" \
    "${TEACHER_CACHE_ARGS[@]}" \
    "${LAST_LOGITS_ARGS[@]}" \
    "${ADAPTER_DECODE_ARGS[@]}" \
    "${INPUT_CACHE_ARGS[@]}" \
    "${CONTEXT_CACHE_ARGS[@]}" \
    "${EVAL_BATCH_ARGS[@]}" \
    "${QWEN_DEVICE_MAP_ARGS[@]}" \
    "${TEACHER_ONLY_ARGS[@]}" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    > "$SHARD_DIR/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" -m src.run eval --family "$MODEL_KIND" -- \
  --model-kind "$MODEL_KIND" \
  --benchmark "$BENCHMARK" \
  --data "$DATA" \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  "${CHECKPOINT_ARGS[@]}" \
  --output-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${ANSWER_ARGS[@]}" \
  "${OUTPUT_MODE_ARGS[@]}" \
  "${PREFILL_ARGS[@]}" \
  "${EARLY_STOP_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "${TEACHER_CACHE_ARGS[@]}" \
  "${LAST_LOGITS_ARGS[@]}" \
  "${ADAPTER_DECODE_ARGS[@]}" \
  "${INPUT_CACHE_ARGS[@]}" \
  "${CONTEXT_CACHE_ARGS[@]}" \
  "${EVAL_BATCH_ARGS[@]}" \
  "${QWEN_DEVICE_MAP_ARGS[@]}" \
  "${TEACHER_ONLY_ARGS[@]}" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL"

cp "$SHARD_DIR/results.json" "$OUT_DIR/results.json"
cp "$SHARD_DIR/predictions.json" "$OUT_DIR/predictions.json"
if [[ -f "$SHARD_DIR/summary.csv" ]]; then
  cp "$SHARD_DIR/summary.csv" "$OUT_DIR/summary.csv"
fi
echo "Done. Results at $OUT_DIR/results.json"
