#!/usr/bin/env bash
set -euo pipefail

# Run all no-train baselines across all retentions and benchmarks
# Uses 8 GPUs in parallel, one method-retention combo per GPU

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
EVAL_SCRIPT="${ROOT}/baselines/eval_baselines.py"

MODEL_LABELS="${MODEL_LABELS:-all}"
METHODS="${METHODS:-fastv dart visionzip sparsevlm divprune zoo}"
RETENTIONS="${RETENTIONS:-0.05 0.10 0.15 0.20}"
BENCHMARKS="${BENCHMARKS:-mmstar,gqa,mmb,mmb-cn,mme,pope,sqa,vqav2,realworldqa}"
MAX_SAMPLES=${MAX_SAMPLES:-1000}

echo "=== Baseline Evaluation ==="
echo "Models: $MODEL_LABELS"
echo "Methods: $METHODS"
echo "Retentions: $RETENTIONS"
echo "Benchmarks: $BENCHMARKS"
echo "Max samples: $MAX_SAMPLES"
echo ""

NUM_GPUS=8
gpu=0
pids=()
labels=()

for method in $METHODS; do
  for ret in $RETENTIONS; do
    echo "Launching: $method ret=$ret on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY $EVAL_SCRIPT \
      --method "$method" \
      --model-label "$MODEL_LABELS" \
      --retention "$ret" \
      --benchmark "$BENCHMARKS" \
      --max-samples "$MAX_SAMPLES" \
      > "${ROOT}/artifacts/eval/baselines/${method}_ret$(printf '%02d' $(echo "$ret * 100" | python3 -c 'import sys; print(int(float(sys.stdin.read())))'))_eval.log" 2>&1 &
    pids+=($!)
    labels+=("$method/ret$ret")
    gpu=$(( (gpu + 1) % NUM_GPUS ))

    # If all GPUs busy, wait for one to finish
    if [ ${#pids[@]} -ge $NUM_GPUS ]; then
      wait "${pids[0]}"
      echo "  Done: ${labels[0]}"
      pids=("${pids[@]:1}")
      labels=("${labels[@]:1}")
    fi
  done
done

# Wait for remaining
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  echo "  Done: ${labels[$i]}"
done

echo ""
echo "=== All evaluations complete ==="

# Print summary
$PY -c "
import json
from pathlib import Path
base = Path('${ROOT}/artifacts/eval/baselines')
models_raw = '$MODEL_LABELS'
if models_raw.strip().lower() == 'all':
    models = ['llava-1.5-7b-hf', 'llava-1.5-13b-hf', 'llava-v1.6-mistral-7b-hf', 'qwen3-vl-8b', 'qwen3-vl-30b-a3b']
else:
    models = [item for item in models_raw.replace(',', ' ').split() if item]
methods = '$METHODS'.split()
retentions = '$RETENTIONS'.split()
benchmarks = '$BENCHMARKS'.split(',')
print(f\"{'Model':<26} {'Method':<10} {'Ret':>4} | \" + ' | '.join(f'{b[:6]:>6}' for b in benchmarks) + ' | AVG')
print('-' * 130)
for model in models:
    for m in methods:
        for r in retentions:
            ret_str = f'ret{int(float(r)*100):02d}'
            scores = []
            cells = []
            for b in benchmarks:
                f = base / model / m / ret_str / b / 'results.json'
                if f.exists():
                    d = json.loads(f.read_text())
                    s = d.get('score', 0)
                    scores.append(s)
                    cells.append(f'{s:.3f}')
                else:
                    cells.append('  --- ')
            avg = sum(scores)/len(scores) if scores else 0
            print(f'{model:<26} {m:<10} {int(float(r)*100):>3}% | ' + ' | '.join(f'{c:>6}' for c in cells) + f' | {avg:.3f}')
        print()
"
