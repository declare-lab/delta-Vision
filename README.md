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

| Config | Teacher | Adapter-only (best) | Params |
|--------|---------|--------------------|----|
| ViT KV source, 4000 steps | 57.8% | 44.7% (77%) | 75.6M |

### Prefill Speed (torch.compile, both sides, H200)

| Model | Visual Tokens | Teacher | Ours (compiled) | Speedup |
|-------|-------------|---------|-----------------|---------|
| LLaVA-1.5-7B | 576 | 37.0 ms | 15.4 ms | 2.50x |
| LLaVA-1.5-13B | 576 | 35.3 ms | 12.2 ms | 2.89x |
| LLaVA-1.6 Mistral | 2880 | 391.5 ms | 60.2 ms | 6.50x |
| Qwen3-VL-4B | 768 | 36.9 ms | 7.1 ms | 5.19x |

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

### Other Data

- Pixmo clean only: `/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl` (135K)
- MMStar eval: `/lustre-data/leijingdi/code/delta-vision/data/mmstar/mmstar_val.jsonl` (1500 samples)

## TODO
- [x] Replace hardcoded Qwen 2x2 concat with spatial_merge_size-aware raw ViT QKV concat
- [x] Add Qwen3-VL MMStar generation eval for this adapter
- [x] Add Pixmo-clean + FineVision LLaVA-150K mixed data preparation
- [ ] Add DeepStack-aware Qwen source features
- [x] Add optional KV MSE loss for per-layer Qwen supervision
- [x] Cosine LR schedule with warmup for training
- [ ] More training data / longer training
