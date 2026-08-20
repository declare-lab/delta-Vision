# Vision KV Inject

## Diagnostic Boundary

All diagnostic experiments must live under `test/`.

- Put diagnostic scripts in `test/diagnostics/`.
- Put diagnostic configs in `test/configs/`.
- Put diagnostic outputs in `test/results/`.
- Do not put diagnostic scripts, temporary configs, logs, plots, JSON/CSV outputs, or scratch
  experiment results under `src/`, `scripts/`, or `artifacts/`.

This repository trains and evaluates lightweight visual adapters for VLM prefill acceleration.
The current maintained scope is intentionally small:

- LLaVA `kv_adapter`
- LLaVA `embedding_adapter`
- Qwen3-VL `embedding_adapter`

Training is unified through `scripts/train.sh`. Benchmark evaluation is unified through
`scripts/eval_benchmark.sh`. Prefill speed and small metric tables are handled by
`src/benchmark_prefill.py`.

## Architecture Scope

| Model family | Adapter mode | Status | Main idea |
| --- | --- | --- | --- |
| LLaVA | `kv_adapter` | maintained | Project vision-encoder K/V into LLM-layer visual K/V. |
| LLaVA | `embedding_adapter` | maintained | Use projected image embeddings as visual memory and inject them through the adapter path. |
| Qwen3-VL | `embedding_adapter` | maintained | Use Qwen visual embeddings as adapter memory with frozen Qwen native projections. |

### LLaVA `kv_adapter`

This is the LLaVA-specific KV adapter path. The vision encoder provides source K/V tensors,
and each language-model layer uses a small adapter to map source visual K/V into that layer's
KV space.

Properties:

- Input source: selected CLIP vision transformer K/V layers.
- Adapter output: per-layer visual K/V for the LLM attention blocks.
- Text path: original LLM text tokens remain unchanged.
- Visual-token MLP work is skipped, which is the main prefill speed win.
- Checkpoints are saved as `step_N.pt` or `final.pt`.

Default training mode:

```bash
MODEL_KIND=llava OUTPUT_MODE=kv_adapter scripts/train.sh
```

### LLaVA `embedding_adapter`

This is the shared embedding-adapter style applied to LLaVA. Instead of using raw vision
K/V as the source, it uses the LLaVA-projected image embeddings as visual memory.

Properties:

- Input source: LLaVA projected image embeddings.
- Adapter output: per-layer visual memory injected through the adapter attention path.
- This mode shares more structure with the Qwen `embedding_adapter` path.
- Checkpoints are loaded through the same LLaVA checkpoint loader, with `output_mode`
  recorded in checkpoint metadata.

Training:

```bash
MODEL_KIND=llava OUTPUT_MODE=embedding_adapter scripts/train.sh
```

### Qwen3-VL `embedding_adapter`

This is the only maintained Qwen training architecture. It uses Qwen's own visual embedding
stream as the adapter memory and keeps the base Qwen3-VL model frozen.

Properties:

- Input source: Qwen3-VL image-token hidden states after the native visual path.
- Adapter output: trainable adapter visual memory for language-model layers.
- Base model: frozen.
- Default loss normalization: token mean.
- Default joint-attention loss weight: `lambda_joint_attention=1.0`, supervised on every language-model layer.
- Checkpoints are saved as `qwen_embedding_adapter_stepN.pt` and
  `qwen_embedding_adapter_final.pt`.

Default training:

```bash
MODEL_KIND=qwen scripts/train.sh
```

Useful Qwen defaults:

```text
OUTPUT_MODE=embedding_adapter
LOSS_NORMALIZATION=token
LAMBDA_LOGIT=2.0
LAMBDA_JOINT_ATTENTION=1.0
LAMBDA_KV_MSE=0.0
KL_TOPK=1024
MAX_STEPS=500
SAVE_EVERY=500
```

## Qwen Mask Semantics

The old `022e580` training code used an all-visual-visible prefix mask: every text token could
attend to every visual token. That reproduces the old 12k checkpoint behavior, but it is not the
mask semantics used by the current code.

The current Qwen `embedding_adapter` uses position-aware masking:

- text tokens attend causally to previous text tokens;
- text tokens attend only to visual tokens whose original sequence positions are not in the future;
- this is the intended training/evaluation path going forward.

## Training

The supported training entry is:

```bash
scripts/train.sh
```

Common environment variables:

```text
MODEL_KIND=qwen|llava
MODEL_PATH=/path/to/base/model
DATA=/path/to/train.jsonl
DATA_ROOT=/path/to/data/root
RUN_NAME=my_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAX_STEPS=12000
SAVE_EVERY=1000
WANDB=1
```

Qwen 500-step Pixmo-clean smoke training:

```bash
MODEL_KIND=qwen \
RUN_NAME=qwen_pixmo_clean_500 \
MAX_STEPS=500 \
SAVE_EVERY=500 \
WANDB=1 \
scripts/train.sh
```

Qwen 12k-style mixed run:

```bash
MODEL_KIND=qwen \
RUN_NAME=qwen_mixed_pixmo_ocrvqa_embedding_adapter_12k \
DATA=data/pixmo_clean_ocrvqa/train.jsonl \
MAX_STEPS=12000 \
SAVE_EVERY=1000 \
LR=5e-5 \
LR_SCHEDULER=cosine \
WARMUP_RATIO=0.2 \
MIN_LR_RATIO=0.1 \
LOSS_NORMALIZATION=token \
LAMBDA_TRAJECTORY=0.5 \
scripts/train.sh
```

LLaVA KV adapter training:

```bash
MODEL_KIND=llava \
OUTPUT_MODE=kv_adapter \
RUN_NAME=llava_kv_adapter \
scripts/train.sh
```

LLaVA embedding adapter training:

```bash
MODEL_KIND=llava \
OUTPUT_MODE=embedding_adapter \
RUN_NAME=llava_embedding_adapter \
scripts/train.sh
```

## Benchmark Evaluation

The supported benchmark evaluation entry is:

```bash
scripts/eval_benchmark.sh
```

Supported benchmark names:

```text
mmstar, gqa, mmb, mmb-cn, mme, pope, sqa, vqav2, textvqa, vizwiz, ocrbench
```

Single Qwen benchmark:

```bash
MODEL_KIND=qwen \
COMPILE_ADAPTER=0 \
scripts/eval_benchmark.sh sqa \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 12000 \
  --max-samples 1000 \
  --num-shards 8
```

Three-benchmark Qwen eval:

```bash
MODEL_KIND=qwen \
COMPILE_ADAPTER=0 \
scripts/eval_benchmark.sh --benchmarks mmstar,sqa,vqav2 \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 500 \
  --max-samples 1000 \
  --num-shards 8
```

Compiled adapter eval:

```bash
MODEL_KIND=qwen \
COMPILE_ADAPTER=1 \
COMPILE_VERIFY=0 \
scripts/eval_benchmark.sh --benchmarks mmstar,sqa,vqav2 \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 500 \
  --max-samples 1000 \
  --num-shards 8
```

LLaVA benchmark eval uses the same wrapper:

```bash
MODEL_KIND=llava \
COMPILE_ADAPTER=0 \
scripts/eval_benchmark.sh mmstar \
  --run-dir artifacts/RUN_NAME \
  --step 4000 \
  --num-shards 8
```

Outputs include per-shard files plus a merged `results.json`. For batch runs, the wrapper also
writes aggregate CSV/JSON summaries with:

- teacher score or F1;
- adapter score or F1;
- total generation time;
- prefill time;
- prefill FLOPs;
- KV cache size;
- agreement and retention metrics when available.

## Prefill Speed / Metric Table

`src/benchmark_prefill.py` is the unified prefill benchmark utility. The old
`src/qwen_benchmark_utils.py` path has been removed.

Small Qwen metric table:

```bash
.venv/bin/python -m src.benchmark_prefill \
  --model-kind qwen \
  --model-path /path/to/Qwen3-VL-4B-Instruct \
  --checkpoint /path/to/qwen_embedding_adapter_step500.pt \
  --metric-table \
  --benchmark pope \
  --metric-samples 100 \
  --data-root /path/to/data/root
```

Single-sample prefill timing:

```bash
.venv/bin/python -m src.benchmark_prefill \
  --model-kind qwen \
  --model-path /path/to/Qwen3-VL-4B-Instruct \
  --checkpoint /path/to/qwen_embedding_adapter_step500.pt \
  --sample-jsonl /path/to/eval.jsonl \
  --sample-index 0 \
  --data-root /path/to/data/root \
  --n-runs 20 \
  --warmup 5
```

For LLaVA:

```bash
.venv/bin/python -m src.benchmark_prefill \
  --model-kind llava \
  --model-path /path/to/llava-1.5-7b-hf \
  --checkpoint /path/to/step_4000.pt \
  --output-mode kv_adapter \
  --sample-jsonl /path/to/eval.jsonl \
  --sample-index 0 \
  --data-root /path/to/data/root
```

## Source Layout

```text
src/train.py              Unified trainer.
src/eval_benchmarks.py    Unified benchmark generation/evaluation.
src/benchmark_prefill.py  Prefill speed benchmark and small metric tables.
src/model.py              Frozen model loaders, adapter modules, adapter forward paths.
src/data.py               Training and benchmark datasets.
src/benchmarks.py         Benchmark registry and scoring configuration.
src/chat_qwen_adapter.py  Qwen adapter chat/debug utility.
```

Shell entries:

```text
scripts/train.sh          Unified training wrapper.
scripts/eval_benchmark.sh Unified benchmark evaluation wrapper.
```

Diagnostic experiments:

```text
test/diagnostics/         Diagnostic experiment scripts.
test/configs/             Diagnostic experiment configs.
test/results/             Diagnostic outputs, ignored by git.
```

Keep exploratory diagnostics under `test/`. Do not mix temporary experiment code or results into
`src/`, `scripts/`, or `artifacts/`.

## Current Compatibility Policy

- New code should use `kv_adapter` for the LLaVA KV adapter.
- New code should use `embedding_adapter` for the shared embedding-memory adapter path.
- Legacy checkpoint names with `qwen_visual_delta_*` are still resolved by the eval wrapper for
  compatibility, but new checkpoints should use `qwen_embedding_adapter_*`.
- `eval-mode=logits` has been removed.
