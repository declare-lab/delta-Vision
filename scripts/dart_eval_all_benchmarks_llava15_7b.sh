#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DART_DIR="${ROOT_DIR}/baselines/dart"

export PYTHONPATH="${DART_DIR}/.pydeps:${DART_DIR}:${DART_DIR}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/lustre-data/leijingdi/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ "${KEEP_NCCL_ENV:-0}" != "1" ]]; then
  unset NCCL_NET
  unset NCCL_IB_DISABLE
  unset NCCL_SOCKET_IFNAME
  unset GLOO_SOCKET_IFNAME
  unset TORCH_NCCL_ASYNC_ERROR_HANDLING
fi
for name in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET NCCL_NET_GDR_LEVEL NCCL_TUNER_CONFIG_PATH; do
  if [[ -n "${!name:-}" ]]; then
    export "$name"
  fi
done

CKPT="${CKPT:-liuhaotian/llava-v1.5-7b}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/eval/dart/all_benchmarks_llava15_7b_8gpu_32tok_1000}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi

TASKS="${TASKS:-mmstar,gqa,mmbench_en_dev,mmbench_cn_dev,mme,pope,scienceqa_img,vqav2_local,realworldqa}"
LIMIT="${LIMIT:-1000}"
REDUCTION_RATIO="${REDUCTION_RATIO:-0.778}"
MAX_NUM_TRUNCTION="${MAX_NUM_TRUNCTION:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29674}"

mkdir -p "${OUT_DIR}"

ARGS=(
  -m lmms_eval
  --model llava
  --model_args "pretrained=${CKPT},conv_template=vicuna_v1,attn_implementation=sdpa,Sparse=True,reduction_ratio=${REDUCTION_RATIO},max_num_trunction=${MAX_NUM_TRUNCTION},pruned_layer=2,image_token_start_index=35,image_token_length=576,pivot_image_token=4,pivot_text_token=4"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --limit "${LIMIT}"
  --log_samples
  --log_samples_suffix "dart_all_1000"
  --output_path "${OUT_DIR}"
)

echo "=== DART LLaVA-1.5-7B all benchmarks ==="
echo "root=${ROOT_DIR}"
echo "tasks=${TASKS}"
echo "limit=${LIMIT}"
echo "num_processes=${NUM_PROCESSES} cuda=${CUDA_VISIBLE_DEVICES}"
echo "max_num_trunction=${MAX_NUM_TRUNCTION}"
echo "out_dir=${OUT_DIR}"

cd "${DART_DIR}/lmms-eval"
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  "${ROOT_DIR}/.venv/bin/python" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${ARGS[@]}"
else
  "${ROOT_DIR}/.venv/bin/python" "${ARGS[@]}"
fi
