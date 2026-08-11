# Vision KV Inject

**Idea**: Replace visual token embeddings in LLaVA with direct KV injection from the vision encoder into LLM attention layers, enabling faster prefill while maintaining accuracy.

## Architecture

```
Image -> CLIP ViT (frozen) -> Extract K,V from last 2 layers (22, 23)
                                    |
                         Per-layer KV Adapter (trainable)
                                    |
              LLM Attention: concat(visual_KV, text_KV) -> output
```

### Standard LLaVA
- Image -> ViT -> Projector -> 576 visual tokens as LLM input embeddings
- LLM processes 576 + T tokens through all layers (Q, K, V, MLP on every token)

### Ours (Vision KV Inject)
- Image -> ViT -> Extract K,V projections from layers 22,23 -> Adapter -> Visual KV
- LLM only processes T text tokens; visual info enters via KV concatenation in attention
- **No MLP computation on visual tokens** -- the main source of speedup

### Adapter Design
Each LLM layer has an independent adapter:
- `source_mix`: learnable softmax weights over 2 source ViT layers
- `k_proj`: Linear(1024 -> num_heads * head_dim, bias=True)
- `v_proj`: Linear(1024 -> num_heads * head_dim, bias=True)
- `gate`: sigmoid scalar controlling injection strength (init=0)

### Training
- **Frozen**: Vision encoder + LLM (all parameters)
- **Trainable**: Only adapter (~269M for 7B, ~420M for 13B)
- **Loss**: Top-1024 KL divergence between teacher (original LLaVA) and student logits on answer tokens
- **Infrastructure**: DeepSpeed ZeRO-2, 8x H200 GPUs

## Results

### LLaVA-1.5-7B (MMStar 1000 samples)

| Setting | Adapter-only | Mixed (original + adapter KV) |
|---------|-------------|-------------------------------|
| Teacher (original LLaVA) | -- | 38.2% |
| Step 500 | 35.3% | 28.7% |
| Step 1500 (peak) | **36.4%** (95.3% retention) | 29.8% |
| Bottleneck 256 + Mixed | 25.9% | **39.0%** (exceeds teacher!) |
| Single layer 23 + Mixed | 33.6% | **37.4%** |

### LLaVA-1.5-13B (MMStar 1000 samples)

| Setting | Adapter-only | Mixed |
|---------|-------------|-------|
| Teacher (original LLaVA-13B) | -- | 38.7% |
| Step 2000 (peak) | **34.5%** (89% retention) | 28.0% |
| Step 4000 | 34.6% | 28.9% |

### Prefill Speed (torch.compile, both sides, H200)

| Model | Method | Latency | Speedup |
|-------|--------|---------|---------|
| **7B** | LLaVA compiled | 37.0 ms | 1.00x |
| | Ours e2e (no cache) | 20.7 ms | **1.87x** |
| | Ours (KV cached) | 15.4 ms | **2.50x** |
| **13B** | LLaVA compiled | 35.3 ms | 1.00x |
| | Ours e2e (no cache) | 18.0 ms | **1.96x** |
| | Ours (KV cached) | 12.2 ms | **2.89x** |

### Ablation (7B, 500 steps, MMStar)

| Experiment | Params | Adapter-only | Mixed |
|------------|--------|-------------|-------|
| 2-layer (22,23) weighted sum | 269M | 34.5% | 29.5% |
| 2-layer (22,23) concat | 537M | 35.2% | 31.5% |
| 1-layer 22 only | 269M | 34.9% | 28.6% |
| 1-layer 23 only | 269M | 33.6% | 37.4% |
| 2-layer + bottleneck 256 | ~34M | 25.9% | **39.0%** |

## Key Findings

1. **Speed**: 1.87-1.96x e2e prefill speedup (2.5-2.9x with KV caching)
2. **Accuracy**: Adapter-only recovers 89-95% of teacher performance; Mixed + bottleneck can exceed teacher
3. **Bottleneck is better for Mixed mode**: Constraining adapter capacity forces complementary residuals rather than conflicting replacements
4. **Same vision encoder across models**: 7B and 13B share identical CLIP ViT -- adapter is the only per-LLM component
5. **Practical use case**: Multi-turn conversations benefit most (visual KV computed once, reused across turns)

## Usage

### Training
```bash
# 7B, 8 GPUs
bash scripts/train.sh

# 13B
.venv/bin/python -m torch.distributed.run --nproc_per_node 8 -m src.train \
  --model-path model/llava-1.5-13b-hf \
  --data ../delta-vision/data/pixmo_ama_train.jsonl \
  --data-root ../delta-vision \
  --output-dir artifacts/run_name \
  --batch-size 2 --max-steps 4000 --lr 1e-4 \
  --wandb --wandb-project vision-kv-inject
```

### Evaluation (8-GPU sharded)
```bash
CHECKPOINT=artifacts/run/final.pt bash scripts/eval.sh
```

## Project Structure
```
configs/ds_zero2.json        # DeepSpeed ZeRO-2 config
src/
  model.py                   # Adapter, KV extraction, forward variants
  data.py                    # VQA/OPD/MMStar datasets
  train.py                   # DeepSpeed training loop
  eval_mmstar.py             # 8-GPU sharded evaluation
  eval_three_modes.py        # Teacher/adapter/mixed comparison
  triton_kernels.py          # Triton fused ops (experimental)
scripts/
  train.sh                   # Training launcher
  eval.sh                    # Eval launcher
  run_ablation.sh            # Ablation experiments
```
