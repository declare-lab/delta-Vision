# Qwen3.5 recurrent-memory mechanism experiments

## Fixed inputs and checkpoint

- Qwen3.5-4B, 32 decoder layers, hidden size 2560; DeepStack OFF.
- 24 GatedDeltaNet linear layers and 8 native FA2 layers (zero-based 3,7,...,31).
- Each linear layer has 32 value-head recurrent matrices, each key128 × value128, accumulated in FP32.
- Frozen PixMo-AMA static embedding adapter, rank128, step2000:
  `artifacts/experiments/qwen35_pixmo/qwen35_4b_embedding128_pixmo2000_20260920_064827/checkpoints/qwen35_embedding_adapter_step2000.pt`.
- Existing seed44 manifests: SQA1000, RealWorldQA765, MMStar1000, copied by reference with SHA256 validation from the final paired Qwen3.5 evaluation. No resampling, changed prompts, reduced image resolution or changed generation caps.
- Generation/answer extraction remains the paired Qwen3.5 path: greedy, thinking disabled, cap8 for these three benchmarks; save generated IDs, EOS status and invalid flags. An unfinished response is scored using that path's existing invalid-zero rule. This experiment does not modify scoring code.
- No new training. Store per-sample/head metrics, predictions and plots; no full hidden/state caches.

## Part 1: independent per-head ranks and causal rank intervention

For each **sample n, linear layer l, value head h**:

1. Use that layer's current normalized token inputs, native projections, causal convolution, gates and FLA recurrence.
2. Record Sin immediately before the contiguous visual span and Sout immediately after the last visual token (before vision_end and the question suffix).
3. In the counterfactual, visual positions make an **identity recurrent transition**: disable both forgetting (`g=0`) and the delta-rule write (`beta=0`). Thus S_no-vis = Sin over this single contiguous image span. Setting beta=0 alone would still forget and is a different intervention, recorded separately.
4. Delta[n,l,h] = Sout - S_no-vis. Perform an uncentered full SVD of each 128×128 matrix independently (CPU LAPACK in FP64; factors restored to FP32 for state interventions); use strict FP32, not TF32, for reconstruction and subspace products. This exact small-matrix CPU decomposition is faster than the validated cuSOLVER QR path; no randomized approximation or rank truncation is used in computing the spectrum. No concatenation of heads, no shared dataset basis, no learned projection.
5. `r95 = min r with sum(sigma[:r]^2)/sum(sigma^2) >= .95`, similarly r90. Effective rank = exp(entropy(sigma/sum(sigma))). A numerically zero matrix is rank0 and has undefined cosine.
6. Preserve the 32 individual head ranks. Heatmaps average samples at each (l,h); layer curves then average heads. Dataset macro-curves give the three datasets equal weight, irrespective of sample counts. Also report rank/128.

For task intervention, restore `S_no-vis + rank_r(Delta)` at the image boundary, then run the native suffix recurrence and cached decode. Ranks: 0,8,16,32,64,128. Intervene at all24 linear layers in the same forward; **recompute from the current trajectory**, never inject a saved native trajectory. Full-attention layers remain native. Run on both native and adapter models, plus each model's unintervened reference.

Rank128 is full rank, not compression. Directly restarting the native chunk recurrence at the image boundary caused nonzero full-rank KL in validation. The implemented intervention therefore uses the affine dependence on boundary state: for the same suffix Q/K/V/g/beta, evaluate F(target) and F(Sout), and add their difference to the original unsplit native core output and final recurrent cache. This preserves native chunk rounding at the full-restoration endpoint. The readout difference is applied BEFORE gated normalization/output projection. Sin and Sout are measured by native prefix runs starting at token0 to keep prefix chunk alignment.

Require decomposition-vs-native kernel parity, strict FP32 full-SVD relative error <2e-5, bounded rank0-vs-direct-identity-transition differences, small first-token KL, and identical full-rank/native generated sequences in the diagnostic gate. Full-set rank128 accuracy/KL remains an explicit control rather than assumed identical. The paired floating-point correction is a controlled implementation of an affine recurrent-state intervention; its rank0 agreement with direct gate masking is reported.

### Scope of “no-vis”

This is **no visual recurrent transition at the selected layer**, not a no-image model. It removes visual-position forgetting as well as writing. Native visual hidden propagation and full-attention layers remain present. Causal convolution is deliberately held fixed to isolate the recurrent-state path; a separate local diagnostic zeros visual convolution inputs as well and records the remaining convolution-mediated effect. Do not call state-only rank0 “all visual access blocked.”

## Part 2: representation versus functional state effect

Primary paired comparison holds the **same native layer input on text tokens, same prefix recurrent state, same prefix convolution context**, replacing only V_l with M_l=E+MLP_l(E). Both pass through the same frozen input RMSNorm and mixer weights.

- Representation: token-mean cosine and normalized MSE before RMSNorm; after RMSNorm also reported.
- Functional: head-mean cosine, normalized Frobenius error, normalized MSE of DeltaS; left/key and right/value singular-subspace overlap at ranks8,16,32,64. Overlap is `||U_teacher^T U_adapter||_F^2/r`; right subspaces use V. Record boundary spectral gaps, because bases at nearly degenerate singular values are unstable.
- Also report actual adapter-trajectory state similarity separately; that includes upstream text-state changes and is not the controlled representation-only comparison.
- Normalized MSE is `||prediction-reference||_F^2 / ||reference||_F^2`; normalized Frobenius error is its square root. Token/head means are explicit.

Compare teacher SVD oracle reconstruction with adapter reconstruction at each rank. Task comparison uses an additional **state-only adapter swap**: compute the adapter-produced boundary state from the current native prefix and substitute it into the otherwise native suffix recurrence, holding suffix Q/K/V/gates/convolution fixed. Compare with rank-r SVD interventions at the same boundaries. The actual adapter's accuracy is also reported, but it is a different, whole-model intervention.

Next-token KL uses the complete vocabulary, teacher-to-variant direction, temperature1, at the first answer-token position. It is not answer accuracy or a claim about the entire generated distribution.

A per-example SVD is an oracle compression diagnostic. It does not establish one shared low-dimensional subspace across images, nor can a task-accuracy advantage alone establish which directions KD learned.

## Part 3: common text-side effects and perturbation sensitivity

Readout: post-visual prompt text tokens only, mixer output **after its native output gate/norm/projection and before the residual**, shape Nt×2560. Prefix text cannot read future visual tokens and is excluded. The set of text positions is identical for LA and FA in each example.

- FA no-vis: native FA2 on text-only Q/K/V, retaining original rotary positions; equivalent to removing visual keys for the text queries. Keep normal visual-query behavior outside this local diagnostic.
- LA no-vis: freeze the visual recurrent transitions as in Part1; preserve convolution. Also report state+convolution ablation ranks separately.
- SVD of the text-output difference: report absolute r95 and r95/min(Nt,2560), plus signal norm. Rank cannot exceed the number of measured text tokens. Do not interpret small absolute rank without this upper bound.

Sensitivity:

- Single-layer interventions, both native and adapter.
- Isotropic Gaussian direction shared across layers and methods within each sample; perturb the prenorm visual hidden with target norm ratios 0,.01,.05,.1,.2. Record realized BF16 ratios.
- Recompute that layer and propagate downstream normally. For adapter, subsequent layers still overwrite visual inputs with their own M_l, matching its real implementation.
- Measure local text mixer-output error, post-FFN text hidden error and full-vocabulary first-answer-token KL after downstream propagation.
- Batch suffix replays for efficiency and include epsilon0 in the same batch; use its logits as the local perturbation reference, and record epsilon0-vs-original KL to expose BF16 GEMM shape rounding.

Plots: dataset-specific layer curves and 24×32 head-rank heatmaps, paired representation/state cosine curves, rank-recovery accuracy/KL, FA/LA normalized effect ranks, and perturbation-response curves. Add equal-weight dataset macro curves. Different layer types occupy different depths and have different weights; FA/LA comparisons are observational within this trained model, not a controlled architecture replacement.

## Execution and validation

`src/qwen35_memory_probe.py` implements the controlled operations. `scripts/qwen35_memory_probe_worker.py` exposes validate, analysis, accuracy and sensitivity stages. Unit tests cover squared-energy ranks, independent heads, zero states, rank endpoints and KL direction. Three actual benchmark examples gate native kernel reconstruction and rank128 generation before launching full manifests. Diagnostic samples are never substituted for full benchmark results.
