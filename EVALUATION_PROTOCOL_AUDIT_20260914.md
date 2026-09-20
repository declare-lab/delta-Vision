# Multi-image / video evaluation protocol audit — 2026-09-14

This audit changes no trained weights or production dataset manifests. The previous tables are not a clean comparison of adapter versus pruning baselines. A high expected adapter score is not itself evidence of a bug; the concrete issues below are independently reproducible.

## 1. Confirmed MMIU question-loss bug

Conversion source: `../vision-kv-inject-attention-sink/scripts/prepare_multimodal_benchmarks.py`, function `prepare_mmiu`:

```python
question = str(doc.get("context") or doc.get("question") or "").strip()
```

The Hugging Face source has **separate context and question fields**. Context is not a reliable substitute for the question. All 11,698 local rows match this context-preferred conversion. In the evaluated first 1,000 rows, these 600 inputs demonstrably omit the actual question:

| Task | Evaluated rows | Actual local question content |
|---|---:|---|
| visual_quality_assessment_q_bench+ | 200 | Only `Candidates: ...` |
| visual_quality_assessment_ve_lol_l | 200 | Only `Candidates: ...` |
| casuality_reasoning_next_qa | 200 | Generic instruction to answer the given question, plus options; no actual question |

For example, source row 400 asks whether the first image is sharper than the second, but the local prompt contains only `Candidates: A. No B. Yes`. Source row 800 asks why the boy's arm was constantly moving, but the local prompt contains a generic 16-image instruction and answer choices.

The other 400 first-1,000 rows contain rephrased questions in context; exact string mismatch alone does **not** mean a question is missing. Both adapters and pruning baselines used the broken manifest. Therefore this is a validity failure affecting both sides, not proof that correction specifically favors adapters.

## 2. Confirmed MuirBench cross-table prompt mismatch

Pruning run: `artifacts/diagnostics/multimodal_baselines_nodeepstack_20260913` uses default `QwenBenchmarkDataset` interleaved layout.

Adapter run: `artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913` explicitly uses `media_first_v1`: images first, numbered end labels, references converted to text, question/options after images.

Both runs disable DeepStack, but the input sequences differ. Their native base scores are **53.50% versus 51.20%**. These tables cannot be directly joined as a controlled method comparison.

Official prompt and parsing references:

- https://github.com/muirbench/MuirBench/blob/main/eval/utils/preprocess.py
- https://github.com/muirbench/MuirBench/blob/main/eval/utils/postprocess.py

The official parser can randomly choose an answer when extraction fails; ours scores invalid extraction as incorrect. Random guessing must not be introduced merely to inflate accuracy. Report extraction failures explicitly. Neither our interleaving convention nor the media-first rewrite should be called an exact reproduction of the official format without documenting the model-specific image insertion.

## 3. Video input budget is not the Qwen reported protocol

Both local video evaluations use eight full-window frames and real timestamps. Qwen3-VL reports 2 fps for these benchmarks, at most 2,048 frames and 224K total video tokens, with 640 tokens per frame.

Source: https://arxiv.org/html/2511.21631v1, section 5.9.

Eight frames can miss temporal evidence, but the size and direction of the resulting accuracy change are not measured by this audit. This budget is shared across local methods and alone does not establish an adapter-specific disadvantage. DeepStack-off is also an explicit local ablation, not the unmodified official model.

## 4. The first 1,000 examples are not a representative benchmark sample

| Dataset | Local total | First-1,000 coverage |
|---|---:|---|
| MuirBench | 2,600 | 10 of 12 tasks |
| MMIU | 11,698 | Six source-task groups, with the sixth partial |
| Video-MME | 2,700 | 900 short, 100 medium, zero long |
| MVBench | 3,800 | Five tasks, 200 examples each |

MVBench annotations contain 20 tasks, but local media are absent for `fine_grained_pose`; conversion skips its 200 examples. The resulting 19-task local collection must not be labeled the complete 20-task benchmark.

Coverage bias does not necessarily depress scores: short-heavy video sampling may make a subset easier. All comparisons need identical, explicitly listed sample IDs; aggregate scores here describe these subsets only.

## 5. Output budget diagnostic (secondary to the fairness issues)

The latest run caps generation at eight tokens. MuirBench SFT has 159 capped outputs, including 76 invalid parsed answers. Base has three capped/invalid outputs; KL one; recurrent KL two; OPD none. MMIU and Video-MME have no capped outputs in this run. MVBench has many capped outputs, but all already yield a parsed option.

`src/muir_length_audit.py` replays **every** capped MuirBench output, regardless of correctness, with a 128-token cap, identical weights/layout/DeepStack-off policy, and exact first-eight-token text assertions. Results go to `artifacts/diagnostics/muir_length_audit_20260914/`; originals are never overwritten. This is a generation-budget check, not an explanation for all adapter-versus-pruning gaps.

Completed: all 165 capped cases replayed successfully; every first-eight-token text matched. Increasing the budget changes the full-1,000 SFT score from 39.10% to 40.10% (10 additional net correct); native, KL and recurrent KL scores do not change. OPD had no capped cases. Nine outputs still hit 128 tokens, so this is not a claim of unlimited-generation convergence. The measured generation cap effect does not explain the large KL-adapter gap.

## Required interpretation

The MMIU omission and MuirBench input mismatch prevent accepting the current comparison as final. Correcting them is necessary whether scores rise or fall. They do not prove that the adapter must outperform a baseline. Video temporal coverage and subset composition must be separately labeled and tested; do not combine these confounds into an unsupported claim of a single root cause.
