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

Qwen3-VL-235B train/eval:

```bash
scripts/train_eval_qwen235b_kl_only.sh
```

This script downloads `Qwen/Qwen3-VL-235B-A22B-Instruct` to
`model/Qwen3-VL-235B-A22B-Instruct`, then trains the Qwen `embedding_adapter` with the same
KL settings as `scripts/train_eval_five_models_kl_only.sh`. Because the 235B backbone cannot
be replicated per rank, it uses `QWEN_DEVICE_MAP=auto`, `NPROC_PER_NODE=1`, `NUM_SHARDS=1`,
`MICRO_BATCH_SIZE_PER_GPU=1`, and `GRADIENT_ACCUMULATION_STEPS=32` to keep the effective
global batch at 32 while one process shards the frozen backbone over all visible GPUs.

Continue download or reuse the same run paths:

```bash
DOWNLOAD=0 STAMP=<same_stamp> scripts/train_eval_qwen235b_kl_only.sh
```

The Hugging Face download resumes automatically. Training restarts from step 0 unless an
`INIT_CHECKPOINT` is supplied; use `SKIP_TRAIN=1` or `SKIP_EVAL=1` to run only one stage.
For 235B adapter evaluation, `COMPILE_ADAPTER=0`, `ADAPTER_DECODE_CACHE=0`, and
`EVAL_BATCH_SIZE=1` are the safe defaults.

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

Default benchmark names:

```text
mmstar, gqa, mmb, mmb-cn, mme, pope, sqa, vqav2, realworldqa
```

Additional benchmark names can be run explicitly:

```text
perceptionbench, rendered-context-qa
```

PerceptionBench expects a converted JSONL file at
`data/benchmarks/perceptionbench/test.jsonl`.
Each item should follow the shared VQA schema with `image` or `images`, `question`,
and `answer`, `answers`, or `choices`.

PerceptionBench uses an OpenAI-compatible LLM judge. Run it with the same
benchmark entrypoint:

```bash
LLM_JUDGE_BASE_URL=http://127.0.0.1:8001/v1 \
LLM_JUDGE_MODEL=qwen3.5-4b-judge \
MODEL_KIND=qwen \
scripts/eval_benchmark.sh perceptionbench --teacher-only --model-path /path/to/qwen3-vl-model
```

PerceptionBench local evaluation note:

- Official judge configuration uses `MAX_TOKENS=65536` with `gpt-oss-120b`.
- Local runs use `/lustre-data/leijingdi/models/Qwen3.5-4B` as the judge.
- For `Qwen3-VL-4B-Instruct`, full PerceptionBench with `MAX_NEW_TOKENS=128`
  took about 30-31 minutes wall time on 8 GPUs: about 23 minutes for answer
  generation plus about 7.5 minutes for batched judging.
- The local `Qwen3.5-4B` judge score from that run was `0.156` on 3000 samples
  (`468/3000`, invalid rate `0.0`). This is not directly comparable to the
  official `gpt-oss-120b` judge score.
- PerceptionBench uses the same benchmark entrypoint as the other benchmarks:
  `scripts/eval_benchmark.sh perceptionbench`.

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

Qwen adapter and pruning speed comparison using the original evaluation entry points:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m src.benchmark_prefill \
  --model-path /lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct \
  --checkpoint artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt \
  --compare-methods all --comparison-protocol original \
  --comparison-deepstack off \
  --cuda-graph --cuda-graph-context --attn-implementation flash_attention_2 \
  --native-cuda-graphs --optimize-attention-metadata --measure-decode-steps \
  --retentions 0.05 0.2 --benchmark mmstar \
  --sample-jsonl data/benchmarks/mmstar/mmstar_speedtest_200.jsonl \
  --data-root data/benchmarks/mmstar --metric-samples 200 \
  --metric-prefill-warmup 1 --max-new-tokens 8 \
  --adapter-decode-cache-mode fast --log-every 25 \
  --output-json test/results/all_fa2_mmstar200/table.json
```

`--compare-methods` defaults to `--comparison-protocol original`. It runs
`benchmark_prefill --metric-table` for base + adapter, then `baselines.eval_baselines`
for base + FastV, DART, DivPrune, ZooPrune, SparseVLM and VisionZip at each retention.
These are the repository's Qwen ports. Methods can also be selected individually.

- Adapter prefill uses the original vision-context and prefill-cache CUDA Graphs.
  Graph capture and warmup occur outside timing; vision runs for every request.
  In this fast path, `--attn-implementation flash_attention_2` selects FA2 for
  adapter text attention as well as native vision attention. Text queries are
  grouped by their original sequence positions so text before an image cannot
  attend that image. Both cached decode modes also use FA2. This FA2 adapter
  benchmark currently requires unpadded batch-size-one inputs.
  `fast` is the decode default: each step processes one new token with growing KV,
  using a warmed whole-decode CUDA Graph. It omits saved prefix layer activations.
  `shape_exact` retains the older full-text-prefix recomputation for BF16 shape
  comparisons. It is not the fast decode path. Unpadded cached queries that see
  the entire prefix use native dense FA2 without gathering K/V or rereading the
  mask on the CPU each step; other masks retain the varlen FA2 path.
  For this unpadded FA2 path, prefill constructs an owned native Qwen cache once;
  subsequent steps call the same native Qwen decoder used by base. Cache preparation
  is included in prefill timing. Explicit sequence and M-RoPE positions preserve
  the adapter's original logits. Masked prefixes retain the manual FA2 decoder,
  whose graphs share immutable visual K/V across steps. Returned caches retain
  their own data across later requests.
- `--native-cuda-graphs` also graphs the native vision and decoder layers for base
  and all six pruning methods, plus the whole native forward during cached decode.
  In FA2 `--compare-methods --comparison-protocol original` runs, this now defaults
  on whenever adapter CUDA Graphs are enabled; attention metadata reuse also defaults
  on. This avoids comparing the adapter's graph replay against native Python dispatch.
  Use `--no-native-cuda-graphs --no-optimize-attention-metadata` only when explicitly
  measuring that older execution path.
  DivPrune/ZooPrune selection and DART neighbor tensor work use warmed graphs;
  pruning audit CPU copies are disabled during benchmark execution. DART retains
  its original candidate order and top-k tie behavior. Every sample checks exact token, logit and KV
  equality with eager FA2 before timing. Graph preparation is separately reported;
  timed capture or a missing warmed shape fails the run. ZooPrune random state is
  restored around preparation so warmup does not change its selections.
- DeepStack is disabled project-wide for training, evaluation, and timing: auxiliary vision mergers are
  skipped and language injection is forbidden, including on the adapter's vision
  path. Both adapter teachers and students follow this policy. Only `--comparison-deepstack off` is accepted. Historical frozen runs and results retain their original settings.
  Base and pruning use original HF `generate`, stopping
  at EOS or the generation limit. Adapter retains the original metric-table
  structured-answer early stop. MMStar defaults to an 8-token limit. These original
  stop policies can produce different output lengths; total speedup includes that effect.
  For equal-work prefill/decode comparisons, run
  `test/diagnostics/paired_runtime_execution.py --tokens 8`: it alternates each
  method with base on the same GPU, suppresses EOS for all methods, and checks all
  eight logits and the final growing KV against eager execution before timing.
  Its totals are a separate fixed-output protocol, not the historical stopping run.
- Times are sums across the selected samples. Image loading, preprocessing,
  CPU-to-GPU copies and warmup are excluded. Prefill includes vision and the
  method's processing through first-token logits. Adapter total adds continuation
  to its measured prefill; native total measures the original `generate` call.
- Each retention group and adapter run records its own measured base denominator.
  Total speedup is paired base total / method total; prefill speedup is paired base
  prefill / method prefill. The output retains all base rows and reference values.
  With native graphs enabled, the adapter uses the optimized native base reference
  from the baseline run; the legacy eager teacher row remains in raw adapter output.
- Adapter `kv_cache_mb` uses measured prefill K/V storage when available; the old
  estimate is retained as `analytic_kv_cache_mb`. Adapter
  `actual_prefill_kv_cache_mb` counts retained text and visual K/V tensors;
  `actual_prefill_decode_cache_mb` includes additional decoding state. The old
  analytic text-KV-plus-one-visual-memory estimate is not actual runtime cache usage.
- FLOPs preserve the original analytical decoder prefill formula (2 FLOPs/MAC),
  excluding vision, selection, LM head, normalization and softmax. They are mean
  per-sample prefill FLOPs, not whole-model or complete-generation FLOPs.
- `Peak Memory` / `peak_memory_mb` is the maximum per-request CUDA allocated
  memory over the dataset, in MiB. It includes the evaluation process's weights,
  retained graph pools, KV and activations. Capture itself is excluded; other
  processes' memory is not counted. Per-stage maxima, mean per-request peaks and
  reserved memory are also saved. Native evaluation enables this with
  `--measure-peak-memory`; the comparison orchestrator enables it automatically.

Outputs include JSON/CSV/Markdown tables, a protocol JSON with commands and hashes,
raw entry-point logs, paired references, and per-sample results. `--metric-table`
also writes a `.details.json` sidecar when `--output-json` is provided.

`--comparison-protocol fixed-work` explicitly selects the earlier fixed-token
benchmark with a custom cached decode loop. Its `--comparison-runs`,
`--comparison-decode-mode` and `--comparison-deepstack` options belong to that
separate protocol. It does not reproduce the historical MMStar speed table.

### Corrected prefill / decode accounting

Add `--optimize-attention-metadata --measure-decode-steps` to the comparison command
above to enable the FA2 metadata fix and direct decode measurements. The original
baseline entry supports the equivalent flags `--optimize-attention-metadata
--measure-decode`.

The metadata fix preserves original RoPE, padding masks, caches, **and the original
varlen/dense kernel choice**. It caches sequence metadata once per pruned position
tensor instead of re-inferring it at every layer. Simply switching the varlen kernel
to a dense kernel can change BF16 near-tie answers and is not the implemented fix.

For baselines, `generation_prefill_time_s` measures the first forward inside native
`generate`; `decode_time_s` measures subsequent cached one-token forwards directly.
`generation_overhead_s` accounts for the remaining generation work. Thus total equals
generation prefill + decode forwards + overhead. `prefilling_time_s` still records
the separately measured standalone prefill and must not be subtracted from total to
claim a decode time. Reports also include `decode_steps`, `decode_ms_per_step`,
`generated_tokens`, and actual retained K/V storage.

For the adapter, direct decode-step timing is optional and separate from its existing
continuation timer. Structured-answer early stopping can finish at the first token:
zero decode steps means decode ms/token is **not applicable**, not zero-cost decoding.
Shared-GPU measurements include contention and must be labelled separately from
isolated speed measurements.

The focused **screenshot reproduction: adapter and FastV 5%** is in
[`test/results/screenshot_adapter_fastv_20260915/README.zh.md`](test/results/screenshot_adapter_fastv_20260915/README.zh.md).
It contains recovered historical commands/CSV/forward code, the measured FA2 and
DeepStack-off results, Peak Memory, and the remaining protocol differences.

The earlier **FA2, DeepStack-off** MMStar 200 comparison is in
[`test/results/deepstack_off_20260915/README.zh.md`](test/results/deepstack_off_20260915/README.zh.md).
It includes Peak Memory, actual KV, the native adapter decoder, and same-input
5%/20% interleaving. The JSON/CSV retain the distinct measured base denominators.

The earlier MMStar 200 stage report is in
[`test/results/prefill_decode_corrected_20260915/README.zh.md`](test/results/prefill_decode_corrected_20260915/README.zh.md).
It includes actual KV storage, direct decode steps, all requested resource/speed
columns, and output parity against the original run. The accompanying
[`FA2 diagnosis`](test/results/prefill_slowdown_investigation_20260915/README.zh.md)
records the preserved kernels and removed repeated metadata operations.
The [decode operator breakdown](test/results/decode_operator_breakdown_20260915/README.zh.md)
also measures actual attention shapes, attributes GPU kernels, and uses an exact
fixed-step CUDA Graph diagnostic to distinguish attention savings from native
dispatch cost. The diagnostic graph timings are not complete-generation results.

Legacy base-versus-adapter quality/metric table:

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

## Current Compatibility Policy

- New code should use `kv_adapter` for the LLaVA KV adapter.
- New code should use `embedding_adapter` for the shared embedding-memory adapter path.
- Legacy checkpoint names with `qwen_visual_delta_*` are still resolved by the eval wrapper for
  compatibility, but new checkpoints should use `qwen_embedding_adapter_*`.
- `eval-mode=logits` has been removed.

## OCR Training And Evaluation

OCR adapter training uses the Qwen3-VL path and DeepSpeed ZeRO-2 by default:

```bash
scripts/ocr_train.sh
```

Default training inputs and hyperparameters:

```text
MODEL_PATH=/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct
OCR_DATASET=ocr_overlap_allcopy_only_1024
DATA=data/train/$OCR_DATASET/paired_train.jsonl
NPROC_PER_NODE=8
MICRO_BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=1824
SAVE_EVERY=500
LR=5e-5
LR_SCHEDULER=cosine
WARMUP_RATIO=0.05
DS_CONFIG=configs/ds_zero2.json
WANDB_MODE=online
```

To resume or initialize from an adapter checkpoint:

```bash
INIT_CHECKPOINT=/path/to/qwen_embedding_adapter_stepXXXX.pt scripts/ocr_train.sh
```

OCR copy-style evaluation uses `scripts/ocr_eval.sh`:

```bash
CHECKPOINT=/path/to/qwen_embedding_adapter_stepXXXX.pt scripts/ocr_eval.sh
```

Default OCR eval inputs:

```text
OCR_DATASET=ocr_overlap_allcopy_only_1024
DATA=data/train/$OCR_DATASET/paired_eval.jsonl
IMAGE_ROOT=data/train/rendered_text
MAX_SAMPLES=1000
MAX_NEW_TOKENS=1024
```

Without `CHECKPOINT`, `ocr_eval.sh` runs teacher-only paths by passing
`--no-eval-adapter`.

Rendered-context QA evaluation uses the normal benchmark wrapper through
`scripts/ocr_qa_eval.sh`:

```bash
RUN_DIR=/path/to/run/checkpoints scripts/ocr_qa_eval.sh
```

Defaults:

```text
BENCHMARK=rendered-context-qa
DATA=data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_span_100.jsonl
STEP=500
MAX_SAMPLES=100
MAX_NEW_TOKENS=512
NUM_SHARDS=8
COMPILE_ADAPTER=0
```
## Qwen3.5 hybrid-attention adapter workflow

The Qwen3.5-4B integration is in `src/qwen35_embedding.py` and
`src/qwen35_experiment.py`. Each of the 32 layers has an independent
2560 → 128 → 2560 residual MLP taking the initial visual embedding. The
24 GatedDeltaNet layers preserve original-order causal convolution and all
visual/text recurrent-state updates. Visual mixer outputs are discarded;
the FFN processes text only. This implementation still computes visual
query/readout work and is not a maximum-speed implementation.

Training freezes the native model and uses the existing answer-token top-1024
KL objective at temperature 2. No hidden-state, attention or recurrent-state
loss is added. Microbatch-one accumulation preserves the original PixMo
microbatch-four answer-token weighting. DeepStack and thinking are disabled.

- `scripts/queue_qwen35_pixmo.py`: the local experiment queue, with a preceding
  run dependency, correctness checks, native nine-benchmark evaluation,
  2000-step PixMo-AMA training and adapter evaluation. Inspect its local paths
  and predecessor before launching it on another machine.
- `scripts/qwen35_worker.py`: validation, distributed training and sharded
  evaluation entrypoints, configured by the prepared run's `config.json`.
- `configs/qwen35_adapter_requirements.txt`: isolated FLA/causal-conv1d/Triton
  dependencies. Install with `python -m pip install --no-deps --no-build-isolation
  --target artifacts/dependencies/qwen35_python -r configs/qwen35_adapter_requirements.txt`.
  Triton 3.7.1 is required here to avoid the older gated-backward issue on H200.
- `test/diagnostics/test_qwen35_embedding.py`: CPU checks for hybrid-cache
  equivalence and KL value/gradient/accumulation parity with the existing loss.

The document continuation workflows are `scripts/train_document_embedding128.py`
and `scripts/train_document_recurrent128.py`, with data preparation in
`scripts/prepare_document_training.py`. They use the training splits of ChartQA,
DocVQA and InfographicVQA, excluding exact encoded-image or decoded-RGB matches
to the held-out data. The evaluation uses ChartQA relaxed accuracy and
DocVQA/InfographicVQA ANLS, implemented in `src/document_metrics.py`.
