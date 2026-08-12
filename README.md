# Vision KV Inject

Inject vision encoder KV cache into LLM attention layers via lightweight adapters, replacing visual token embeddings for faster prefill.

## Two Methods

### Method 1: Direct KV Prediction

- Adapter directly predicts each layer's visual K,V
- Simple architecture, fast training
- Best for speed (no dependency on frozen LLM weights)

### Method 2: Hidden State Adapter (experimental)

- Adapter maps to LLM hidden space, LLM's own k_proj/v_proj create KV
- Leverages LLM's native projection weights (already trained for visual tokens)
- Shows phase-transition learning (sudden loss drop after ~400 steps)

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
| Method 1 (4000 steps) | 57.8% | 44.7% (77%) | 75.6M |
| Method 1 (8000 steps) | 57.8% | 44.7% (77%) | 75.6M |
| Method 2 (500 steps, phase transition) | 69.5%* | 25.0%* | 94.6M |

*200 samples evaluation

### Prefill Speed (torch.compile, H200)

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

1. **Speed scales with visual tokens**: More crops/patches = larger speedup (6.5x for LLaVA-1.6)
2. **ViT K,V projections > hidden states** as source: attention-ready features transfer better
3. **GQA models need tiny adapters**: 67M for Mistral (8 KV heads), vs 269M for Vicuna (32 heads)
4. **M-RoPE matters**: Correct 3D position encoding for Qwen3-VL visual tokens
5. **Phase transition in Method 2**: Sudden loss collapse after ~400 steps when adapter finds the right mapping
6. **Same vision encoder, multiple LLMs**: One ViT serves different backends via per-LLM adapters

## Architecture Details

### Adapter Design (Method 1, per LLM layer)
- : learnable softmax weights over 2 ViT source layers
- : Linear(source_dim -> num_kv_heads * head_dim, bias=True)
- : sigmoid scalar (init=0, gradual injection)

### Training
- Loss: top-1024 KL divergence (student vs teacher logits)
- Optimizer: AdamW, lr=1e-4
- Infrastructure: DeepSpeed ZeRO-2, 8x H200 GPUs
- Data: PixMo-AMA (VQA)

### Supported Models
- LLaVA-1.5-7B/13B (CLIP ViT + Vicuna)
- LLaVA-1.6 Mistral (CLIP ViT + Mistral, multi-crop)
- Qwen3-VL-4B (Qwen3 ViT + M-RoPE)

## Project Structure


## TODO
- [ ] Train Method 2 to 4000 steps (currently shows phase transition at step 400)
- [ ] Add trajectory loss (intermediate hidden state matching)
- [ ] Cosine LR schedule with warmup
- [ ] Qwen3.5-VL validation
- [ ] Visual token pooling (compress 768 -> 64 tokens)
