# Vision KV Adapter

Train and evaluate the adapter paths used in this repo:
- LLaVA `kv_adapter`: maps vision-encoder K/V into LLM attention layers and skips image-token MLP work.
- LLaVA/Qwen `embedding_adapter`: uses projected image embeddings as adapter memory.

## Method



The LLaVA `kv_adapter` path gives each LLM layer an independent adapter that maps ViT KV to LLM KV space:
- source_mix: learnable softmax weights over source ViT layers
- k_proj / v_proj: Linear(source_dim -> num_kv_heads * head_dim, bias=True)
- gate: sigmoid scalar (initialized at -5.0 before sigmoid)

Text tokens go through the LLM normally. Visual info enters through the selected adapter path.
No MLP computation on visual tokens at any layer -- the main source of speedup.

## Results

### Latest Speed Snapshot (latest)

| Model | Teacher (ms) | Ours (e2e ms) | Speedup |
|---|---:|---:|---:|
| Qwen3-VL-4B embedding_adapter e2e | 37.03 | 18.61 | 1.99x |
| LLaVA kv_adapter e2e | 36.59 | 20.77 | 1.76x |

这两条是当前最新一次短任务 benchmark 的 e2e 对比，数值会随输入长度和 batch 略有波动。

### LLaVA Family (shared CLIP ViT-L/14@336)

| Model | LLM | Teacher | KV Adapter | Params | E2E Speedup | KV Cached |
|-------|-----|---------|-------------|--------|-------------|-----------|
| LLaVA-1.5-7B | Vicuna-7B (MHA) | 38.2% | 36.4% (95%) | 269M | 1.87x | 2.50x |
| LLaVA-1.5-13B | Vicuna-13B (MHA) | 38.7% | 34.6% (89%) | 420M | 1.96x | 2.89x |
| LLaVA-1.6 Mistral | Mistral-7B (GQA-8) | 48.0% | 43.5% (91%) | 67M | 4.08x | 6.50x |

### Qwen3-VL-4B (Qwen3 ViT, M-RoPE)

| Config | Teacher | embedding_adapter | Params | Notes |
|--------|---------|---------------------|--------|-------|
| ViT KV source, 4000 steps | 57.8% | 44.7% (77%) | 75.6M | older raw ViT KV path |
| Legacy factorized native head, 500 steps | 65.3% | 54.9% embedding_adapter / 65.0% hybrid reference | lightweight adapter | `qwen_topk1024_freezeqkv_no_layer_20260807_100141` |

The Qwen path now uses Qwen's own V0 image-token embeddings and frozen native Q/K/V projections.
The supported Qwen training mode is `embedding_adapter`.

Masking note: legacy `022e580` `native_visual_kv_injection` trained with visual memory as a
full prefix, so every text token could attend to every visual token. That reproduces the old
12k checkpoint exactly, but it is not the causal semantics we want going forward. Current
`embedding_adapter` uses position-aware masking: a text token can attend only to visual tokens
whose original sequence positions are not in the future, plus causal previous text tokens.

The 500-step unified-entry Qwen reproduction runs use Pixmo-clean, 8x H200, global batch 32,
constant LR `5e-5`, no warmup, `lambda_logit=4.0`, `lambda_trajectory=0.5`, and MMStar
1k short-generation eval.

| Run | Mode / reader | Trainable | Step-500 loss | Step-500 KL | Mass | Teacher | Adapter | Agreement | Retention |
|-----|---------------|-----------|---------------|-------------|------|---------|---------|-----------|-----------|
| `qwen_embedding_adapter_500` | `embedding_adapter` | 23.59M | 2.0758 | 0.5106 | -- | 64.9% | 53.3% | 69.0% | 72.1% |

Training entry:

```bash
MODEL_KIND=qwen scripts/train.sh
```

Useful overrides:

```bash
RUN_NAME=qwen_topk1024_freezeqkv_no_layer_repro \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_KIND=qwen scripts/train.sh
```

The default keeps the reproduced schedule: constant LR and no warmup. For a new non-exact experiment,
set `LR_SCHEDULER=cosine` and `WARMUP_RATIO=0.2`.

Qwen benchmark eval, defaulting to MMStar:

```bash
RUN_DIR=artifacts/experiments/qwen_topk1024_freezeqkv/qwen_topk1024_freezeqkv_no_layer_repro \
MODEL_KIND=qwen scripts/eval_benchmark.sh
```

Generic Qwen benchmark eval uses the shared benchmark registry. Supported names are:
`mmstar`, `gqa`, `mmb`, `mmb-cn`, `mme`, `pope`, `sqa`, `vqav2`, `textvqa`, `vizwiz`, `ocrbench`.

Run one benchmark:

```bash
MODEL_KIND=qwen scripts/eval_benchmark.sh ocrbench \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000 \
  --max-samples 1000
```

Run an explicit benchmark list and write aggregate CSV/JSON summaries:

```bash
MODEL_KIND=qwen scripts/eval_benchmark.sh mmstar ocrbench textvqa \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000
```

Use all registered benchmarks:

```bash
MODEL_KIND=qwen scripts/eval_benchmark.sh --benchmarks all \
  --run-dir artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME \
  --step 1000
```

Qwen 1k benchmark output for `RUN_NAME` is saved under
`artifacts/eval/qwen_topk1024_freezeqkv/RUN_NAME/`.

| Model | GQA | MMB | MMB-CN | MME | POPE | SQA | VQA-v2 | TextVQA | VizWiz | OCRBench |
|-------|-----|-----|--------|-----|------|-----|--------|---------|--------|----------|
| Base Teacher | 61.6 | 87.5 | 87.7 | 84.7 | 89.3 | 93.3 | 80.9 | 82.3 | 25.9 | 80.7 |
| embedding_adapter | 54.1 | 82.4 | 82.0 | 79.4 | 86.3 | 81.4 | 75.1 | 63.2 | 18.5 | 49.5 |
| Gap | -7.5 | -5.1 | -5.7 | -5.3 | -3.0 | -11.9 | -5.8 | -19.1 | -7.3 | -31.2 |

| Model | Total Time | Prefilling Time | FLOPs | KV Cache | POPE F1 | Speedup Total | Speedup Prefilling |
|-------|------------|-----------------|-------|----------|---------|---------------|--------------------|
| Base Teacher | 20:31.89 | 9:49.44 | 4.589e16 | 76.39 MB | 88.7 | 1.00x | 1.00x |
| embedding_adapter | 58:04.41 | 17:08.16 | 5.413e15 | 8.58 MB | 86.1 | 0.35x | 0.57x |

Qwen mixed Pixmo-clean + OCRvQA 12k run:

- Run: `qwen_mixed_pixmo_ocrvqa_embedding_adapter_12k_cosine_warmup02_RUN_ID`
- Checkpoints: `artifacts/experiments/qwen_topk1024_freezeqkv/qwen_mixed_pixmo_ocrvqa_embedding_adapter_12k_cosine_warmup02_RUN_ID/checkpoints/`
- Eval: `artifacts/eval/qwen_topk1024_freezeqkv/qwen_mixed_pixmo_ocrvqa_embedding_adapter_12k_cosine_warmup02_RUN_ID/`
- Training: 8x H200, mixed `data/pixmo_clean_ocrvqa/train.jsonl`, 12k steps, save every 1000, LR `5e-5`, cosine schedule, `WARMUP_RATIO=0.2`, `MIN_LR_RATIO=0.1`, `lambda_logit=4.0`, `lambda_trajectory=0.5`, wandb off.
- Full metrics: `all_ckpt_benchmark_summary.csv`, `all_ckpt_benchmark_summary.json`, `all_ckpt_scores_wide.csv`.
- Validation: 132/132 checkpoint-benchmark results completed. `all_ckpt_benchmark_summary.json` has 132 rows.

OCR-aligned training data:

- Pixmo-clean + OCR-heavy 300k source mix: `data/pixmo_clean_ocrmix_300k/train.jsonl`
- No-Pixmo short-answer aligned OCR mix: `data/ocr_aligned_no_pixmo_filtered/train.jsonl` (148,667 rows)
- OCRBench-targeted v1 mix: `data/ocrbench_target_mix_v1/train.jsonl` (205,242 rows)

The OCRBench-targeted v1 mix keeps the no-Pixmo aligned rows and appends 50k HME100K formula
recognition samples plus 6,575 CORD receipt/KIE field-QA samples. It preserves existing prompts
for base rows and avoids full-page OCR dumps, binary yes/no targets, unanswerable labels, and long
generic answers.

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
- Qwen train/eval/benchmark use the same dense reference adapter forward. The previous decode-cache
  and fused prefix/split switches were removed after direct logits checks showed drift.
- `COMPILE_ADAPTER=1` in the benchmark wrapper compiles the adapter prefill path. Compiling the full
  Qwen teacher is opt-in because HF Qwen graph breaks and compilation overhead made it slower in
  smoke tests.

### Prefill Speed

| Model | Visual Tokens | Teacher | Ours (compiled) | Speedup |
|-------|-------------|---------|-----------------|---------|
| LLaVA-1.5-7B | 576 | 37.0 ms | 15.4 ms | 2.50x |
| LLaVA-1.5-13B | 576 | 35.3 ms | 12.2 ms | 2.89x |
| LLaVA-1.6 Mistral | 2880 | 391.5 ms | 60.2 ms | 6.50x |
| Qwen3-VL-4B embedding_adapter smoke | 144 | 38.9 ms | 25.0 ms | 1.55x |

### Ablation (7B, 500 steps)

| Experiment | KV Adapter | Mixed (original + adapter) |
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
.venv/bin/python scripts/data/prepare_mixed_data_v2.py
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
