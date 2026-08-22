#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
EVAL_SCRIPT="${ROOT}/baselines/eval_baselines.py"

METHODS="${METHODS:-divprune zoo}"
RETENTIONS="${RETENTIONS:-0.05 0.10 0.15 0.20}"
BENCHMARKS="${BENCHMARKS:-mmstar,gqa,mmb,mmb-cn,mme,pope,sqa,vqav2,realworldqa,perceptionbench}"
FULL_BENCHMARKS="${FULL_BENCHMARKS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_ID="${RUN_ID:-divprune_zoo_ret_grid_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/artifacts/eval/baselines/logs"

mkdir -p "${LOG_DIR}"

echo "run_id=${RUN_ID}"
echo "methods=${METHODS}"
echo "retentions=${RETENTIONS}"
echo "benchmarks=${BENCHMARKS}"
echo "full_benchmarks=${FULL_BENCHMARKS}"
echo "max_samples=${MAX_SAMPLES}"
echo "num_gpus=${NUM_GPUS}"
echo "started=$(date -Is)"

gpu=0
pids=()
labels=()

for method in ${METHODS}; do
  for ret in ${RETENTIONS}; do
    ret_pct="$("${PY}" -c "print(f'{int(float(\"${ret}\") * 100):02d}')")"
    log="${LOG_DIR}/${RUN_ID}_${method}_ret${ret_pct}_gpu${gpu}.log"
    echo "launch method=${method} ret=${ret} gpu=${gpu} log=${log}"
    (
      set -u
      echo "phase=limited benchmark=${BENCHMARKS} max_samples=${MAX_SAMPLES}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EVAL_SCRIPT}" \
        --method "${method}" \
        --retention "${ret}" \
        --benchmark "${BENCHMARKS}" \
        --data-root "${ROOT}" \
        --max-samples "${MAX_SAMPLES}" \
        --log-every 50
      if [[ -n "${FULL_BENCHMARKS}" ]]; then
        echo "phase=full benchmark=${FULL_BENCHMARKS} max_samples=all"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EVAL_SCRIPT}" \
          --method "${method}" \
          --retention "${ret}" \
          --benchmark "${FULL_BENCHMARKS}" \
          --data-root "${ROOT}" \
          --max-samples 0 \
          --log-every 50
      fi
    ) > "${log}" 2>&1 &
    pids+=("$!")
    labels+=("${method}/ret${ret}/gpu${gpu}")
    gpu=$(( (gpu + 1) % NUM_GPUS ))
  done
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "done ${labels[$i]}"
  else
    code=$?
    echo "failed ${labels[$i]} exit=${code}"
    status="${code}"
  fi
done

echo "finished=$(date -Is) status=${status}"
exit "${status}"
