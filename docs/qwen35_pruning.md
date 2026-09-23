# Qwen3.5-4B DART / DivPrune

## Retention definition

The requested ratios are 5% and 20% of **visual tokens**, excluding text and
image boundary markers. Keep counts use the existing decimal round-to-nearest
budget, with at least one visual token per image.

| Method | Pruning location (zero-based) | Layers 0–3 | Layers 4–31 |
|---|---|---|---|
| DART | Before layer 4, after the first full attention at layer 3 | 100% | Requested 5% or 20% |
| DivPrune | Before layer 0 | Requested 5% or 20% | Requested 5% or 20% |

The user explicitly chose **post-pruning-layer retention for DART**. Its ideal
all-layer visual-token retention is therefore 16.875% / 30%, before integer
rounding. DivPrune's all-layer retention is 5% / 20%. Both ratios and actual
per-layer counts are saved per question; they must not be mislabeled as equal
all-layer budgets.

## Implementation

`src/qwen35_pruning.py` copies the existing Qwen3-VL DART and DivPrune selectors
without changing their ranking, pivot counts, Python set order or tie behavior.
DART uses layer 3's native normalized, post-RoPE keys and the layer output
normalized by the decoder's final norm. Its pivots remain 5 image / 3 text
tokens. DivPrune selects from the initial projected visual embeddings.

Selected visual tokens and all text tokens remain in their original order, with
original M-RoPE coordinates. Native FA2 full attention and native GatedDeltaNet
via FLA / causal-conv1d remain in use. No DeepStack or thinking is enabled.
This is batch-one, unpadded, single-image inference.

DART's layers 0–3 keep their full caches; later layers build compressed caches.
Recurrent states are not manually truncated or reset. Generation uses the native
cached greedy decoder with a shared 4096-token maximum and EOS-aware scoring.
Raw generated token IDs and the stopping reason are saved. Unfinished outputs at
the cap are marked invalid and scored zero for every method.

## Validation and evaluation

CPU FP32 tests check unchanged selectors, full-retention native identity, exact
token/cache lengths, and three-step cached versus full-prefix equivalence for
both methods at both ratios. Real BF16 validation on MMStar and RealWorldQA
checks full-retention logits and every hybrid-cache tensor against native,
followed by cached-versus-full-prefix checks with the native model's own rounding
error as a control. DART's initial selection is held fixed in full-prefix cache
tests so generated answer tokens cannot change its text pivots.

The suite requires a corrected Qwen3.5 native/adapter reference using the same
scorer and uniform seed=44 question sampling without replacement. The historical
first-1000 reference is rejected. All methods share the frozen manifests:
MMStar, GQA, MMB, MMB-CN, MME, POPE, SQA, VQAv2 (1000 each), and RealWorldQA (765).
Four configurations produce 35,060 predictions. Each receives two GPU shards.
Saved answers are rescored, and input hashes and image grids are checked against
the native reference before writing `RESULTS.md` and `summary.json`.
MME/POPE use per-question accuracy; VQAv2 uses the existing soft score. AVG is the
unweighted mean of nine unrounded benchmark scores.

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/queue_qwen35_pruning.py \
  --reference-run /absolute/path/to/corrected_seed44_native_adapter_run
```

Each run snapshots its source and manifests. The supervisor fails on a failed
validation or worker; it never substitutes a slower attention implementation or
silently omits failed questions.
