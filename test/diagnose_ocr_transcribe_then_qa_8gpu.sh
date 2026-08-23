#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NUM_SHARDS=${NUM_SHARDS:-8}
CUDA_DEVICES_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_DEVICES_CSV"
if (( ${#CUDA_DEVICES[@]} < NUM_SHARDS )); then
  echo "need at least NUM_SHARDS CUDA devices in CUDA_VISIBLE_DEVICES; got ${CUDA_DEVICES_CSV}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT:-}" || -z "${DATA:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "set CHECKPOINT, DATA, and OUTPUT_DIR" >&2
  exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/shards"

pids=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  shard_dir="$OUTPUT_DIR/shards/shard_${shard}"
  mkdir -p "$shard_dir"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$shard]}"
    "$PY" test/diagnose_ocr_transcribe_then_qa.py \
      --checkpoint "$CHECKPOINT" \
      --data "$DATA" \
      --output-dir "$shard_dir" \
      --max-samples "${MAX_SAMPLES:-20}" \
      --max-transcribe-tokens "${MAX_TRANSCRIBE_TOKENS:-512}" \
      --max-answer-tokens "${MAX_ANSWER_TOKENS:-64}" \
      --metric "${METRIC:-token_f1}" \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$shard" \
      "$@"
  ) >"$OUTPUT_DIR/shards/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more shards failed; logs are in $OUTPUT_DIR/shards" >&2
  exit 1
fi

"$PY" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
predictions = []
results = []
for path in sorted((root / "shards").glob("shard_*/results.json")):
    results.append(json.loads(path.read_text()))
    pred_path = path.with_name("predictions.json")
    predictions.extend(json.loads(pred_path.read_text()))

metric_names = sorted({name for result in results for name in result.get("metrics", {})})
merged_metrics = {}
for name in metric_names:
    total = 0
    score_sum = 0.0
    invalid_sum = 0.0
    for result in results:
        item = result.get("metrics", {}).get(name)
        if not item:
            continue
        samples = int(item.get("samples", 0))
        total += samples
        score_sum += float(item.get("score", item.get("accuracy", 0.0))) * samples
        invalid_sum += float(item.get("invalid_rate", 0.0)) * samples
    merged_metrics[name] = {
        "samples": total,
        "score": score_sum / max(1, total),
        "accuracy": score_sum / max(1, total),
        "invalid_rate": invalid_sum / max(1, total),
    }

merged = {
    "task": "diagnose_ocr_transcribe_then_qa_8gpu",
    "num_shards": len(results),
    "total_samples": len(predictions),
    "metric": results[0].get("metric") if results else None,
    "data": results[0].get("data") if results else None,
    "checkpoint": results[0].get("checkpoint") if results else None,
    "metrics": merged_metrics,
}
predictions.sort(key=lambda row: str(row.get("id", row.get("index", ""))))
(root / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
(root / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(merged, indent=2, ensure_ascii=False), flush=True)
PY
