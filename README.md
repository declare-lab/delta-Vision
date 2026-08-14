# Vision KV Inject

Inject vision encoder KV cache into LLM attention layers via per-layer adapters, replacing visual token embeddings for faster prefill.

## Method



Each LLM layer has an independent adapter that maps ViT KV to LLM KV space:
- source_mix: learnable softmax weights over source ViT layers
- k_proj / v_proj: Linear(source_dim -> num_kv_heads * head_dim, bias=True)
- gate: sigmoid scalar (initialized at -5.0 before sigmoid)

Text tokens go through the LLM normally. Visual info enters only via KV injection in attention.
No MLP computation on visual tokens at any layer -- the main source of speedup.

## Results

### LLaVA Family (shared CLIP ViT-L/14@336)

| Model | LLM | Teacher | Adapter-only | Params | E2E Speedup | KV Cached |
|-------|-----|---------|-------------|--------|-------------|-----------|
| LLaVA-1.5-7B | Vicuna-7B (MHA) | 38.2% | 36.4% (95%) | 269M | 1.87x | 2.50x |
| LLaVA-1.5-13B | Vicuna-13B (MHA) | 38.7% | 34.6% (89%) | 420M | 1.96x | 2.89x |
| LLaVA-1.6 Mistral | Mistral-7B (GQA-8) | 48.0% | 43.5% (91%) | 67M | 4.08x | 6.50x |

### Qwen3-VL-4B (Qwen3 ViT, M-RoPE)

| Config | Teacher | Adapter-only (best) | Params | Notes |
|--------|---------|---------------------|--------|-------|
| ViT KV source, 4000 steps | 57.8% | 44.7% (77%) | 75.6M | older raw ViT KV path |
| Legacy delta-vision factorized native head, 500 steps | 65.3% | 54.9% adapter-only / 65.0% hybrid reference | lightweight adapter | `qwen_topk1024_freezeqkv_no_layer_20260807_100141` |

The Qwen path now uses Qwen's own V0 image-token embeddings and frozen native Q/K/V projections.
Supported training modes are `native_visual_kv_injection` and `native_visual_kv_split`; the split
mode predicts `mass * (A_visual - A_text)` before the native Qwen output projection.

The 500-step unified-entry Qwen reproduction runs use Pixmo-clean, 8x H200, global batch 32,
constant LR `5e-5`, no warmup, `lambda_logit=4.0`, `lambda_trajectory=0.5`, and MMStar
1k short-generation eval.

| Run | Mode / reader | Trainable | Step-500 loss | Step-500 KL | Mass | Teacher | Adapter | Agreement | Retention |
|-----|---------------|-----------|---------------|-------------|------|---------|---------|-----------|-----------|
| `qwen_visual_delta_injection_500_20260813_103031` | `native_visual_kv_injection` | 23.59M | 2.0758 | 0.5106 | -- | 64.9% | 53.3% | 69.0% | 72.1% |
| `qwen_visual_delta_split_500_20260813_103828` | `native_visual_kv_split`, reader MLP 4x | 191.54M | 2.0036 | 0.4949 | 0.0023 | 64.9% | 53.2% | 67.8% | 71.5% |
| `qwen_visual_delta_split_nomlp_500_20260813_110508` | `native_visual_kv_split`, no reader MLP | 57.29M | 2.0551 | 0.5077 | 0.0091 | 64.9% | 52.8% | 66.6% | 70.3% |

No-reader split is the direct concat ablation:
`concat(A_visual, Q_text) -> Linear(8192, 4096) -> LayerNorm -> Linear(4096, 32) -> sigmoid`.
It cuts trainable parameters from 191.54M to 57.29M, but the 500-step MMStar score is slightly
lower than the reader-MLP split and injection runs. The mass stays small, so the model mostly learns
a conservative visual delta.

Training entry:

```bash
scripts/train_qwen_delta.sh
```

Useful overrides:

```bash
RUN_NAME=qwen_topk1024_freezeqkv_no_layer_repro \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train_qwen_delta.sh
```

No-reader-MLP concat ablation:

```bash
RUN_NAME=qwen_visual_delta_split_nomlp_500 \
OUTPUT_MODE=native_visual_kv_split \
READER_MLP_RATIO=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train_qwen_delta.sh
```

The default keeps the reproduced schedule: constant LR and no warmup. For a new non-exact experiment,
set `LR_SCHEDULER=cosine` and `WARMUP_RATIO=0.2`.

MMStar short-generation eval:

```bash
RUN_DIR=artifacts/experiments/qwen_topk1024_freezeqkv/qwen_topk1024_freezeqkv_no_layer_repro \
scripts/eval_qwen_delta_mmstar.sh
```

Generic Qwen benchmark eval uses the shared benchmark registry. Supported names are:
`mmstar`, `gqa`, `mmb`, `mmb-cn`, `mme`, `pope`, `sqa`, `vqav2`, `textvqa`, `vizwiz`, `ocrbench`.

Run one benchmark:

```bash
scripts/eval_qwen_delta_mmstar.sh ocrbench \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000 \
  --max-samples 1000
```

Run an explicit benchmark list and write aggregate CSV/JSON summaries:

```bash
scripts/run_qwen_benchmark_1k.sh mmstar ocrbench textvqa \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000
```

Use all registered benchmarks:

```bash
scripts/run_qwen_benchmark_1k.sh --benchmarks all \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000
```

Qwen 1k benchmark for `qwen_visual_delta_injection_500_20260813_103031`
is saved under `artifacts/eval/qwen_topk1024_freezeqkv/qwen_visual_delta_injection_500_20260813_103031/`.

| Model | GQA | MMB | MMB-CN | MME | POPE | SQA | VQA-v2 | TextVQA | VizWiz | OCRBench |
|-------|-----|-----|--------|-----|------|-----|--------|---------|--------|----------|
| Base Teacher | 61.6 | 87.5 | 87.7 | 84.7 | 89.3 | 93.3 | 80.9 | 82.3 | 25.9 | 80.7 |
| Injection | 54.1 | 82.4 | 82.0 | 79.4 | 86.3 | 81.4 | 75.1 | 63.2 | 18.5 | 49.5 |
| Gap | -7.5 | -5.1 | -5.7 | -5.3 | -3.0 | -11.9 | -5.8 | -19.1 | -7.3 | -31.2 |

| Model | Total Time | Prefilling Time | FLOPs | KV Cache | POPE F1 | Speedup Total | Speedup Prefilling |
|-------|------------|-----------------|-------|----------|---------|---------------|--------------------|
| Base Teacher | 20:31.89 | 9:49.44 | 4.589e16 | 76.39 MB | 88.7 | 1.00x | 1.00x |
| Injection | 58:04.41 | 17:08.16 | 5.413e15 | 8.58 MB | 86.1 | 0.35x | 0.57x |

Qwen mixed Pixmo-clean + OCRvQA 12k run:

- Run: `qwen_mixed_pixmo_ocrvqa_injection_12k_cosine_warmup02_20260813_162014`
- Checkpoints: `artifacts/experiments/qwen_topk1024_freezeqkv/qwen_mixed_pixmo_ocrvqa_injection_12k_cosine_warmup02_20260813_162014/checkpoints/`
- Eval: `artifacts/eval/qwen_topk1024_freezeqkv/qwen_mixed_pixmo_ocrvqa_injection_12k_cosine_warmup02_20260813_162014/`
- Training: 8x H200, mixed `data/pixmo_clean_ocrvqa/train.jsonl`, 12k steps, save every 1000, LR `5e-5`, cosine schedule, `WARMUP_RATIO=0.2`, `MIN_LR_RATIO=0.1`, `lambda_logit=4.0`, `lambda_trajectory=0.5`, wandb off.
- Full metrics: `all_ckpt_benchmark_summary.csv`, `all_ckpt_benchmark_summary.json`, `all_ckpt_scores_wide.csv`.
- Validation: 132/132 checkpoint-benchmark results completed. `all_ckpt_benchmark_summary.json` has 132 rows.

Adapter scores, 1000-sample subsets:

| Step | MMStar | GQA | MMB | MMB-CN | MME | POPE | SQA | VQA-v2 | TextVQA | VizWiz | OCRBench |
|------|--------|-----|-----|--------|-----|------|-----|--------|---------|--------|----------|
| 1000 | 0.479 | 0.511 | 0.799 | 0.800 | 0.770 | 0.823 | 0.810 | 0.685 | 0.530 | 0.091 | 0.395 |
| 2000 | 0.525 | 0.540 | 0.835 | 0.823 | 0.784 | 0.857 | 0.814 | 0.744 | 0.635 | 0.188 | 0.495 |
| 3000 | 0.537 | 0.545 | 0.834 | 0.827 | 0.795 | 0.872 | 0.823 | 0.760 | 0.662 | 0.204 | 0.532 |
| 4000 | 0.522 | 0.553 | 0.833 | 0.824 | 0.796 | 0.865 | 0.822 | 0.763 | 0.670 | 0.201 | 0.527 |
| 5000 | 0.530 | 0.545 | 0.837 | 0.825 | 0.800 | 0.867 | 0.818 | 0.767 | 0.674 | 0.213 | 0.545 |
| 6000 | 0.538 | 0.554 | 0.845 | 0.820 | 0.803 | 0.872 | 0.821 | 0.771 | 0.678 | 0.215 | 0.541 |
| 7000 | 0.538 | 0.562 | 0.842 | 0.828 | 0.804 | 0.875 | 0.818 | 0.773 | 0.679 | 0.215 | 0.547 |
| 8000 | 0.540 | 0.562 | 0.845 | 0.822 | 0.808 | 0.874 | 0.822 | 0.776 | 0.680 | 0.214 | 0.543 |
| 9000 | 0.541 | 0.563 | 0.845 | 0.821 | 0.804 | 0.876 | 0.816 | 0.780 | 0.688 | 0.222 | 0.541 |
| 10000 | 0.544 | 0.567 | 0.845 | 0.821 | 0.804 | 0.874 | 0.817 | 0.776 | 0.686 | 0.222 | 0.543 |
| 11000 | 0.542 | 0.564 | 0.844 | 0.822 | 0.805 | 0.873 | 0.817 | 0.776 | 0.687 | 0.219 | 0.543 |
| 12000 | 0.539 | 0.564 | 0.845 | 0.823 | 0.806 | 0.875 | 0.817 | 0.772 | 0.686 | 0.220 | 0.543 |

Best observed checkpoints:

| Benchmark | Best step | Score |
|-----------|-----------|-------|
| MMStar | 10000 | 0.544 |
| GQA | 10000 | 0.567 |
| MMB | 6000/8000/9000/10000/12000 | 0.845 |
| MMB-CN | 7000 | 0.828 |
| MME | 8000 | 0.808 |
| POPE | 9000 | 0.876 |
| SQA | 3000 | 0.823 |
| VQA-v2 | 9000 | 0.780 |
| TextVQA | 9000 | 0.688 |
| VizWiz | 9000/10000 | 0.222 |
| OCRBench | 7000 | 0.547 |

Current safe speed defaults:
- `ADAPTER_DECODE_CACHE=1` for `native_visual_kv_injection`. It reuses the measured adapter prefill
  logits and KV cache for generation instead of recomputing the full prefix at every decoded token.
  After the cache update, a 5-benchmark smoke (`mmstar`, `vqav2`, `textvqa`, `vizwiz`, `ocrbench`;
  10 samples each) matched no-cache generation text and parsed predictions exactly.
- `FAST_SPLIT_TEXT=0` and `FAST_INJECTION_PREFIX=0`. These fused mask paths reduce prefill time, but direct logits checks showed drift, so they must remain explicit experiments.
- `COMPILE_ADAPTER=1` in the benchmark wrapper compiles the adapter prefill path only when decode
  cache is not used. Compiling the full Qwen teacher is opt-in because HF Qwen graph breaks and
  compilation overhead made it slower in smoke tests.

### Prefill Speed

| Model | Visual Tokens | Teacher | Ours (compiled) | Speedup |
|-------|-------------|---------|-----------------|---------|
| LLaVA-1.5-7B | 576 | 37.0 ms | 15.4 ms | 2.50x |
| LLaVA-1.5-13B | 576 | 35.3 ms | 12.2 ms | 2.89x |
| LLaVA-1.6 Mistral | 2880 | 391.5 ms | 60.2 ms | 6.50x |
| Qwen3-VL-4B injection smoke | 144 | 38.9 ms | 25.0 ms | 1.55x |

### Ablation (7B, 500 steps)

| Experiment | Adapter-only | Mixed (original + adapter) |
|------------|-------------|---------------------------|
| 2-layer weighted sum | 34.5% | 29.5% |
| 1-layer 23 only | 33.6% | 37.4% |
| 2-layer + bottleneck 256 | 25.9% | 39.0% (exceeds teacher) |
| ViT hidden states as source | 30.1% | -- |
| ViT K,V projections (default) | 34.5% | -- |

## Key Findings

1. Speed scales with visual tokens: more crops/patches = larger speedup (6.5x for LLaVA-1.6)
2. ViT K,V projections > hidden states as source
3. GQA models need tiny adapters: 67M for Mistral (8 KV heads)
4. Same vision encoder serves multiple LLMs via per-LLM adapters
5. Bottleneck + Mixed mode can exceed teacher accuracy

## Supported Models
- LLaVA-1.5-7B/13B (CLIP ViT + Vicuna)
- LLaVA-1.6 Mistral (CLIP ViT + Mistral, multi-crop)
- Qwen3-VL-4B (Qwen3 ViT + M-RoPE)

## Data Preparation

Mix Pixmo-clean with FineVision LLaVA-Instruct-150K:

```bash
.venv/bin/python scripts/prepare_mixed_data_v2.py
```

Default output:

```text
/lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k/train.jsonl
```

Use that file with `--data-root /lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k`.

## Project Structure


## Data

### Mixed Training Data (pixmo_clean + FineVision LLaVA-150K)

- Path: `/lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k/train.jsonl`
- Data root: `/lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k`
- Samples: 293,705 (pixmo_clean ~135K + LLaVA-Instruct-150K ~158K)
- Format: `{image, question, answer, source}`
- Each sample has `source` field: `pixmo_clean` or `finevision_llava150k`
- Images: absolute paths, pre-extracted
- Shuffled

### Mixed Training Data (pixmo_clean + OCRvQA)

- Path: `/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_ocrvqa/train.jsonl`
- Samples: 301,741 (`pixmo_clean` 135,995 + `ocrvqa` 165,746)
- OCRvQA images: `/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_ocrvqa/ocrvqa_images`
- Format: `{image, image_root, question, answer, source}`
- The Qwen training path supports per-row `image_root`; this is required because Pixmo images live under `/lustre-data/leijingdi/code/delta-vision`, while OCRvQA images live in this repo.
- Pixel-area cache: `/lustre-data/leijingdi/code/vision-kv-inject/artifacts/cache/pixmo_clean_ocrvqa.pixel_areas.json`

### Mixed Training Data (pixmo_clean + OCR Mix 300K)

- Path: `/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_ocrmix_300k/train.jsonl`
- Manifest: `/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_ocrmix_300k/manifest.json`
- Samples: 300,000
- Mix: `pixmo_clean` 135,000; `docvqa` 22,500; `pdfvqa` 11,250; `ureader_qa_processed` 11,250; `textvqa` 24,750; `st_vqa` 20,250; `infographic_vqa` 11,250; `chartqa` 20,250; `plotqa` 13,500; `sroie` 9,000; `invoices_receipts` 3,750; `funsd` 2,250; `ocrvqa` 15,000
- Format: `{image, image_root, question, answer, source}`
- Token stats: `/lustre-data/leijingdi/code/vision-kv-inject/artifacts/data_stats/ocrmix300k_token_stats_exact.json`
- Use sample-normalized training loss for this mix: `--loss-normalization sample`

### Other Data

- Pixmo clean only: `/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl` (135K)
- Pixmo clean image root: `/lustre-data/leijingdi/code/delta-vision`
- MMStar eval: `/lustre-data/leijingdi/code/delta-vision/data/mmstar/mmstar_val.jsonl` (1500 samples)

## TODO
- [x] Replace hardcoded Qwen 2x2 concat with spatial_merge_size-aware raw ViT QKV concat
- [x] Add Qwen3-VL MMStar generation eval for this adapter
- [x] Add Pixmo-clean + FineVision LLaVA-150K mixed data preparation
- [ ] Add DeepStack-aware Qwen source features
- [x] Add optional KV MSE loss for per-layer Qwen supervision
- [x] Cosine LR schedule with warmup for training
- [ ] More training data / longer training
