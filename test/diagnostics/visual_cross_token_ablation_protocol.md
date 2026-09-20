# RealWorldQA visual cross-token ablation

## Fixed protocol

- Native frozen Qwen3-VL-4B-Instruct, BF16, FlashAttention 2; native DeepStack retained.
- Existing RealWorldQA `data/benchmarks/realworldqa/test.jsonl`, all 765 examples, unchanged processor and prompt.
- Existing native greedy-generation evaluator and scorer, maximum 8 answer tokens. No answer text enters the prompt.
- Language-model layers 0–35: exactly ONE layer intervened per variant, never cumulative deletion.
- Visual rows are expanded image-token positions, not template or question tokens.
- For a visual query, split native causal attention into visual self, visual cross, and visible nonvisual/context terms.
- Preserve the original softmax denominator. Self-only removes visual cross; Cross-only removes visual self. Both retain nonvisual context. No normalization of the remaining weights.
- Native residual and FFN remain. Text query rows are untouched at the intervened layer. Later native layers may respond to the changed visual states; that is the measured effect.
- Decode is native and uses each variant's own prefill cache. Only prefill visual output rows are intervened.

## Controls and implementation

Full native is an independent no-intervention generation. Each layer also has a Full reconstructed control using the same FP32 decomposition as the two deletions, cast to BF16 before native W_O. This detects rounding effects rather than attributing them to edge removal. Thus there are 109 variants per example: one native + 36 × three variants. Total: 83,385 scored generations.

Visual feature/DeepStack outputs are cached per image without changing them. Native head outputs and replacement components are cached for each layer in the native pass. Because only ONE layer is changed, its upstream computation is native. Every intervention checks the full input to W_O is bitwise identical to the native cached input. This optimization must NOT be reused for cumulative layer-group interventions.

The native Full observation leaves W_O input unchanged. Eight-worker smoke tests compare unobserved/observed generated answers, and check the final-layer intervention leaves answers identical. Last-layer visual-only changes cannot reach text through a later attention block, so zero drop there is an implementation check, not independent evidence of redundancy.

The initial smoke test found `yes` vs `Yes` after numerical reconstruction, with unchanged score; this motivated the full per-layer, per-example reconstructed Full control. Failed preliminary smoke artifacts are retained separately; they do not enter formal results.

## Mixing statistics

- M_visual = Frobenius norm(cross) / Frobenius norm(self + cross), before W_O.
- M_total = Frobenius norm(cross) / Frobenius norm(self + cross + nonvisual), before W_O.
- Also calculate the corresponding ratios after native W_O (linear contributions, no projection bias).
- Record head/query ratio mean and P95, near-zero denominators, head-specific norm ratios, and self/cross/nonvisual attention masses.
- Ratios are computed per example, then averaged equally across examples. Norm ratios can exceed one through cancellation; they are not probabilities or information fractions.

## Outputs and interpretation

`artifacts/diagnostics/realworldqa_visual_cross_token_20260912/` contains the protocol, raw `shard*.jsonl`, self-check records, worker completion records, full tables, and the final plot. Raw output retains every answer and individual score for paired comparison.

Report both native-relative and numerically matched accuracy drops. Use paired sample bootstrap CIs (10,000 resamples, seed 44), with harmed/helped counts so cancellation in net accuracy is visible. These are per-layer, not simultaneous multiple-testing-adjusted intervals.

Small single-layer loss does not imply multi-layer removal is harmless. The visual encoder and all unmodified layers still perform their own contextual computation. Large mixing magnitude with small accuracy loss is evidence about this intervention's marginal task effect, not proof that all mixed information is redundant.

## Commands

```bash
.venv/bin/python -m unittest discover -s test/diagnostics -p test_visual_cross_token_ablation.py
.venv/bin/python -u -m src.visual_cross_token_ablation launch
# Only if restarting the exact same interrupted protocol:
.venv/bin/python -u -m src.visual_cross_token_ablation launch --resume
```

The launcher runs eight foreground-supervised GPU workers and automatically merges only after complete coverage. It terminates only its own workers if one fails. No training process is started or stopped.
