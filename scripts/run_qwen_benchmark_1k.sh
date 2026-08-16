#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
RUN_DIR=${RUN_DIR:-$ROOT_DIR/artifacts/experiments/qwen_topk1024_freezeqkv/qwen_visual_delta_injection_500_20260813_103031}
STEP=${STEP:-500}
EVAL_ALL_CKPTS=${EVAL_ALL_CKPTS:-0}
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MEASURE_PREFILL=${MEASURE_PREFILL:-1}
COMPILE_ADAPTER=${COMPILE_ADAPTER:-1}
COMPILE_MODE=${COMPILE_MODE:-reduce-overhead}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-1}
COMPILE_WARMUP=${COMPILE_WARMUP:-1}
ADAPTER_DECODE_CACHE=${ADAPTER_DECODE_CACHE:-0}
BENCHMARKS=${BENCHMARKS:-all}
FORCE_EVAL=${FORCE_EVAL:-0}

POSITIONAL_BENCHMARKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks)
      BENCHMARKS="$2"
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
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --all-ckpts)
      EVAL_ALL_CKPTS=1
      shift
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE_EVAL=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/run_qwen_benchmark_1k.sh [benchmark ...] [options]

Examples:
  scripts/run_qwen_benchmark_1k.sh mmstar ocrbench textvqa --run-dir RUN --step 1000
  scripts/run_qwen_benchmark_1k.sh --benchmarks mmstar,ocrbench,textvqa --run-dir RUN --step 1000
  scripts/run_qwen_benchmark_1k.sh --benchmarks all --run-dir RUN --step 1000

Options:
  --benchmarks LIST      Comma/space-separated benchmark names, or all.
  --run-dir PATH         Run directory or checkpoints directory.
  --step N               Single checkpoint step.
  --steps LIST           Space-separated checkpoint steps.
  --all-ckpts            Evaluate every qwen_visual_delta_step*.pt under RUN_DIR.
  --max-samples N        Per-benchmark subset size; default 1000.
  --num-shards N         Number of GPU shards; default 8.
  --out-root PATH        Aggregate output root.
  --force                Re-run even if results.json exists.
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

if [[ "${#POSITIONAL_BENCHMARKS[@]}" -gt 0 ]]; then
  BENCHMARKS="${POSITIONAL_BENCHMARKS[*]}"
fi
BENCHMARKS="$("$PY" - "$BENCHMARKS" <<'PY'
import sys
from src.benchmarks import parse_benchmark_names
print(" ".join(parse_benchmark_names(sys.argv[1])))
PY
)"
if [[ -z "${RUNTIME_TAG:-}" ]]; then
  if [[ "$ADAPTER_DECODE_CACHE" == "1" ]]; then
    RUNTIME_TAG=decode_cache
  elif [[ "$COMPILE_ADAPTER" == "1" ]]; then
    RUNTIME_TAG=compiled_dense
  else
    RUNTIME_TAG=eager_dense
  fi
fi
RUN_LABEL="$(basename "$RUN_DIR")"
if [[ "$RUN_LABEL" == "checkpoints" ]]; then
  RUN_LABEL="$(basename "$(dirname "$RUN_DIR")")"
fi
OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/artifacts/eval/qwen_topk1024_freezeqkv/$RUN_LABEL/$RUNTIME_TAG}
TEACHER_CACHE=${TEACHER_CACHE:-1}
TEACHER_CACHE_DIR=${TEACHER_CACHE_DIR:-$OUT_ROOT/teacher_cache}
LAST_LOGITS_ONLY=${LAST_LOGITS_ONLY:-1}
INPUT_CACHE=${INPUT_CACHE:-1}
INPUT_CACHE_DIR=${INPUT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_benchmark_inputs}

if [[ -z "${STEPS:-}" && "$EVAL_ALL_CKPTS" == "1" ]]; then
  CKPT_DIR="$RUN_DIR"
  if [[ -d "$RUN_DIR/checkpoints" ]]; then
    CKPT_DIR="$RUN_DIR/checkpoints"
  fi
  mapfile -t STEP_LIST < <(
    find "$CKPT_DIR" -maxdepth 1 -type f -name 'qwen_visual_delta_step*.pt' -printf '%f\n' \
      | sed -E 's/^qwen_visual_delta_step([0-9]+)\.pt$/\1/' \
      | sort -n
  )
  if [[ "${#STEP_LIST[@]}" -eq 0 ]]; then
    echo "no qwen_visual_delta_step*.pt found under $CKPT_DIR" >&2
    exit 1
  fi
  STEPS="${STEP_LIST[*]}"
else
  STEPS=${STEPS:-$STEP}
fi

echo "=== Qwen base teacher + injection adapter benchmark ==="
echo "root=$ROOT_DIR"
echo "run_dir=$RUN_DIR"
echo "steps=$STEPS"
echo "benchmarks=$BENCHMARKS"
echo "max_samples=$MAX_SAMPLES"
echo "num_shards=$NUM_SHARDS cuda=$CUDA_VISIBLE_DEVICES"
echo "compile_adapter=$COMPILE_ADAPTER compile_mode=$COMPILE_MODE compile_dynamic=$COMPILE_DYNAMIC compile_warmup=$COMPILE_WARMUP"
echo "adapter_decode_cache=$ADAPTER_DECODE_CACHE"
echo "runtime_tag=$RUNTIME_TAG out_root=$OUT_ROOT"
echo "teacher_cache=$TEACHER_CACHE teacher_cache_dir=$TEACHER_CACHE_DIR"
echo "last_logits_only=$LAST_LOGITS_ONLY input_cache=$INPUT_CACHE input_cache_dir=$INPUT_CACHE_DIR"
echo "context_cache=${CONTEXT_CACHE:-0} context_cache_dir=${CONTEXT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_initial_contexts}"
echo "force_eval=$FORCE_EVAL"
echo "start=$(date)"

for step in $STEPS; do
  for benchmark in $BENCHMARKS; do
    out_dir="$OUT_ROOT/step${step}_${benchmark}_${MAX_SAMPLES}samples_${NUM_SHARDS}gpu"
    if [[ "$FORCE_EVAL" != "1" && -s "$out_dir/results.json" ]]; then
      echo "=== SKIP existing step=$step benchmark=$benchmark $out_dir/results.json ==="
      continue
    fi
    echo "=== START step=$step benchmark=$benchmark $(date) ==="
    BENCHMARK="$benchmark" \
    RUN_DIR="$RUN_DIR" \
    STEP="$step" \
    OUT_DIR="$out_dir" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    NUM_SHARDS="$NUM_SHARDS" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MEASURE_PREFILL="$MEASURE_PREFILL" \
    COMPILE_ADAPTER="$COMPILE_ADAPTER" \
    COMPILE_MODE="$COMPILE_MODE" \
    COMPILE_DYNAMIC="$COMPILE_DYNAMIC" \
    COMPILE_WARMUP="$COMPILE_WARMUP" \
    ADAPTER_DECODE_CACHE="$ADAPTER_DECODE_CACHE" \
    TEACHER_CACHE="$TEACHER_CACHE" \
    TEACHER_CACHE_DIR="$TEACHER_CACHE_DIR" \
    LAST_LOGITS_ONLY="$LAST_LOGITS_ONLY" \
    INPUT_CACHE="$INPUT_CACHE" \
    INPUT_CACHE_DIR="$INPUT_CACHE_DIR" \
    CONTEXT_CACHE="${CONTEXT_CACHE:-0}" \
    CONTEXT_CACHE_DIR="${CONTEXT_CACHE_DIR:-$ROOT_DIR/artifacts/cache/qwen_initial_contexts}" \
    bash scripts/eval_qwen_delta_mmstar.sh "$benchmark"
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
        step = int(step_text)
    except ValueError:
        continue
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if "teacher" in data:
        teacher_score = data.get("teacher", {}).get("score")
        adapter_score = (data.get("adapter") or {}).get("score")
        metric = data.get("metric")
        teacher_invalid = data.get("teacher", {}).get("invalid_rate")
        adapter_invalid = (data.get("adapter") or {}).get("invalid_rate")
        timing = data.get("timing", {})
        resources = data.get("resources", {})
        row = {
            "step": step,
            "benchmark": data.get("benchmark", benchmark),
            "display_name": data.get("display_name", benchmark),
            "metric": metric,
            "total_samples": data.get("total_samples"),
            "teacher_score": teacher_score,
            "adapter_score": adapter_score,
            "gap": (teacher_score - adapter_score) if teacher_score is not None and adapter_score is not None else None,
            "retention": data.get("retention"),
            "agreement": data.get("agreement"),
            "teacher_invalid_rate": teacher_invalid,
            "adapter_invalid_rate": adapter_invalid,
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
        }
    else:
        teacher_score = data.get("teacher_accuracy")
        adapter_score = data.get("adapter_accuracy")
        row = {
            "step": step,
            "benchmark": benchmark,
            "display_name": benchmark,
            "metric": "multi_choice",
            "total_samples": data.get("total_samples"),
            "teacher_score": teacher_score,
            "adapter_score": adapter_score,
            "gap": (teacher_score - adapter_score) if teacher_score is not None and adapter_score is not None else None,
            "retention": data.get("retention"),
            "agreement": data.get("agreement"),
            "teacher_invalid_rate": data.get("teacher_invalid_rate"),
            "adapter_invalid_rate": data.get("adapter_invalid_rate"),
            "teacher_total_s": None,
            "adapter_total_s": None,
            "teacher_prefill_s": None,
            "adapter_prefill_s": None,
            "speedup_total": None,
            "speedup_prefill": None,
            "teacher_kv_cache_mb": None,
            "adapter_kv_cache_mb": None,
            "teacher_prefill_flops": None,
            "adapter_prefill_flops": None,
            "pope_f1": None,
            "result_dir": str(result_path.parent),
        }
    rows.append(row)

rows.sort(key=lambda item: (int(item["step"]), str(item["benchmark"])))
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "all_ckpt_benchmark_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

fields = [
    "step",
    "benchmark",
    "display_name",
    "metric",
    "total_samples",
    "teacher_score",
    "adapter_score",
    "gap",
    "retention",
    "agreement",
    "teacher_invalid_rate",
    "adapter_invalid_rate",
    "teacher_total_s",
    "adapter_total_s",
    "teacher_prefill_s",
    "adapter_prefill_s",
    "speedup_total",
    "speedup_prefill",
    "teacher_kv_cache_mb",
    "adapter_kv_cache_mb",
    "teacher_prefill_flops",
    "adapter_prefill_flops",
    "pope_f1",
    "result_dir",
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
    for step in sorted(by_step):
        writer.writerow([step, *[by_step[step].get(benchmark) for benchmark in benchmarks]])

print(f"Wrote aggregate summaries under {out_root}")
PY

echo "=== ALL DONE $(date) ==="
