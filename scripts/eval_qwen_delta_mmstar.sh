#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/lustre-data/leijingdi/code/delta-vision}

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/delta_vision_qwen${PYTHONPATH:+:$PYTHONPATH}"

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
  if [[ -f "$RUN_DIR/attention_sidecar_step${STEP}.pt" ]]; then
    CKPT="$RUN_DIR/attention_sidecar_step${STEP}.pt"
  elif [[ -f "$RUN_DIR/checkpoints/attention_sidecar_step${STEP}.pt" ]]; then
    CKPT="$RUN_DIR/checkpoints/attention_sidecar_step${STEP}.pt"
  fi
fi
if [[ -z "${CKPT:-}" ]]; then
  echo "set CKPT to attention_sidecar_step${STEP}.pt, or set RUN_DIR" >&2
  exit 1
fi

RUN_NAME=${RUN_NAME:-$(basename "$(dirname "$(dirname "$CKPT")")")}
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/artifacts/eval/qwen_topk1024_freezeqkv/$RUN_NAME/step${STEP}_mmstar_${NUM_SHARDS}gpu}

DTYPE=${DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
VISUAL_MEMORY_MODE=${VISUAL_MEMORY_MODE:-v0}
SIDECAR_SCALE=${SIDECAR_SCALE:-1}
SIDECAR_BACKEND=${SIDECAR_BACKEND:-learned}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DELTA_VISION_IMAGE_ROOT="$DATA_ROOT"

SHARD_DIR="$OUT_DIR/shards"
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.jsonl "$SHARD_DIR"/result_*.json "$SHARD_DIR"/shard_*.log

echo "=== Qwen3-VL sidecar MMStar exact-repro eval ==="
echo "root=$ROOT_DIR"
echo "data_root=$DATA_ROOT"
echo "checkpoint=$CKPT"
echo "output=$OUT_DIR"
echo "benchmark=$BENCHMARK max_samples=$MAX_SAMPLES shards=$NUM_SHARDS"
echo "attn=$ATTN_IMPL dtype=$DTYPE visual_memory=$VISUAL_MEMORY_MODE"

"$PY" - <<'PY' "$DATA" "$SHARD_DIR" "$MAX_SAMPLES" "$NUM_SHARDS"
import sys
from pathlib import Path

data = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
max_samples = int(sys.argv[3])
num_shards = int(sys.argv[4])
handles = [
    (out_dir / f"shard_{idx:02d}.jsonl").open("w", encoding="utf-8")
    for idx in range(num_shards)
]
try:
    with data.open("r", encoding="utf-8") as src:
        for idx, line in enumerate(src):
            if idx >= max_samples:
                break
            handles[idx % num_shards].write(line)
finally:
    for handle in handles:
        handle.close()
PY

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_CSV"
if [[ "${#DEVICES[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "need at least $NUM_SHARDS CUDA devices, got ${#DEVICES[@]} from CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_CSV" >&2
  exit 1
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICES[$shard]}"
  shard_file="$SHARD_DIR/shard_$(printf '%02d' "$shard").jsonl"
  out_json="$SHARD_DIR/result_$(printf '%02d' "$shard").json"
  echo "Launching shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m delta_vision.cli.qwen.eval_qwen3vl_sidecar \
    --benchmark "$BENCHMARK" \
    --data "$shard_file" \
    --model-path "$MODEL_PATH" \
    --checkpoint "$CKPT" \
    --sidecar-backend "$SIDECAR_BACKEND" \
    --output-json "$out_json" \
    --max-samples 1000000 \
    --visual-memory-mode "$VISUAL_MEMORY_MODE" \
    --sidecar-scale "$SIDECAR_SCALE" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    --device cuda:0 \
    > "$SHARD_DIR/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" - <<'PY' "$OUT_DIR" "$DATA" "$MAX_SAMPLES" "$CKPT" "$NUM_SHARDS" "$BENCHMARK"
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
data = sys.argv[2]
max_samples = int(sys.argv[3])
ckpt = sys.argv[4]
num_shards = int(sys.argv[5])
benchmark = sys.argv[6]
payloads = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir / "shards").glob("result_*.json"))]
if len(payloads) != num_shards:
    raise SystemExit(f"expected {num_shards} shard results, got {len(payloads)}")

by_setting = {}
qwen_correct_by_shard = []
for payload in payloads:
    qwen_row = next(row for row in payload["results"] if row["setting"] in {"qwen", "teacher"})
    qwen_correct_by_shard.append(int(qwen_row["correct"]))

for payload, shard_qwen_correct in zip(payloads, qwen_correct_by_shard):
    for row in payload["results"]:
        setting = "qwen" if row["setting"] == "teacher" else row["setting"]
        acc = by_setting.setdefault(
            setting,
            {
                "setting": setting,
                "scored": 0,
                "correct": 0,
                "_agree": 0.0,
                "_ret_num": 0.0,
                "_ret_den": 0,
                "_kl": 0.0,
            },
        )
        scored = int(row.get("scored", 0))
        correct = int(row.get("correct", 0))
        acc["scored"] += scored
        acc["correct"] += correct
        if setting != "qwen":
            acc["_agree"] += float(row.get("qwen_agreement", row.get("teacher_agreement", 0.0))) * scored
            acc["_ret_num"] += (
                float(row.get("qwen_correct_retention", row.get("teacher_correct_retention", 0.0)))
                * shard_qwen_correct
            )
            acc["_ret_den"] += shard_qwen_correct
            acc["_kl"] += float(row.get("output_kl_to_qwen", row.get("output_kl_to_teacher", 0.0))) * scored

ordered = []
for setting in ("qwen", "no_visual", "sidecar_only", "hybrid"):
    if setting not in by_setting:
        continue
    row = by_setting[setting]
    out = {
        "setting": setting,
        "scored": row["scored"],
        "correct": row["correct"],
        "accuracy": row["correct"] / max(1, row["scored"]),
    }
    if setting != "qwen":
        out["qwen_agreement"] = row["_agree"] / max(1, row["scored"])
        out["qwen_correct_retention"] = row["_ret_num"] / max(1, row["_ret_den"])
        out["output_kl_to_qwen"] = row["_kl"] / max(1, row["scored"])
    ordered.append(out)

merged = {
    "benchmark": benchmark,
    "data": data,
    "max_samples": max_samples,
    "checkpoint": ckpt,
    "num_shards": num_shards,
    "results": ordered,
}
out_path = out_dir / f"{benchmark}_merged_qwen_format.json"
out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"wrote {out_path}")
PY
