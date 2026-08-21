# Rendered Text Context Adapter Notes

This document summarizes the current architecture, data, training runs, and experiments for the rendered-text context adapter work.

## Goal

We want to test whether textual context can be rendered as an image and then used through a visual adapter as a compressed/visual substitute for the original text context.

The intended division of labor is:

- The frozen LLM already has the QA ability.
- The adapter should help collect visual text information from rendered context images.
- The adapter output should be usable by the frozen LLM as context, not merely cause the model to copy or continue the visible text.

## Architecture

Base model:

- Qwen3-VL-4B-Instruct
- Path: `/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct`
- LLM and vision backbone are frozen.

Trainable module:

- `QwenEmbeddingAdapter`
- About 23.59M trainable parameters.
- It adapts Qwen visual memory before injection into the language model layers.
- Current mode: `embedding_adapter`.

Teacher/student setup:

- Teacher sees raw text context.
- Student sees rendered context image through the adapter.
- Original training used KL on answer/copy target logits.
- Later diagnostics added student CE, teacher CE, token accuracy, and top-1 agreement.

Relevant files:

- Training script: `test/rendered_text_teacher_train.py`
- Copy training wrapper: `test/train_rendered_text_copy.sh`
- Copy eval script: `test/eval_rendered_text_copy.py`
- Copy eval wrapper: `test/eval_rendered_text_copy.sh`
- Low-rank KV eval: `test/eval_text_context_lowrank_kv.py`

## Data

### Original Rendered QA Data

Path:

- `data/train/rendered_text/paired_train.jsonl`
- `data/train/rendered_text/paired_eval.jsonl`

This data mixes several sources:

- MSMARCO
- MSMARCO v2
- HotpotQA
- TATQA
- LongBench variants
- others in small amounts

Problem found:

- Some evaluation rows have very long contexts.
- TATQA introduces table/numeric reasoning, which is not the clean target when we only want to test ordinary textual context.

### Copy/Transcription Data

New data directory:

- `data/train/rendered_text_copy_300`

Construction:

- Source rows came from existing rendered text paired data.
- Answer is the visible rendered text span itself.
- Question is:
  `Transcribe all visible text in the image exactly. Preserve line breaks.`
- Intended max answer length was about 300 Qwen tokens.

Generated size:

- Train: 100,000 rows
- Eval: 1,000 rows
- Images: 101,000 jpg files

Measured token stats:

- Train: min 69, max 302, mean 297.92, median 300
- Eval: min 121, max 301, mean 296.24, median 300

Note: a few rows are 301-302 tokens because trimming was approximate.

### MSMARCO Paragraph-Only QA Benchmark

To avoid table confounds, a paragraph-only QA benchmark was created from MSMARCO.

Path:

- `data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_100.jsonl`

Selection:

- Source: `data/train/rendered_text/paired_train.jsonl`
- Source dataset: `msmarco`
- Qwen context tokens: 200-400
- Sample count: 100

Stats:

- Available MSMARCO rows in range: 2,655
- Selected rows: 100
- Min tokens: 206
- Max tokens: 399
- Mean tokens: 334.62
- Median tokens: 343.5

This is the current clean benchmark for ordinary paragraph context QA.

## Main Training Runs

### KL-Only Copy Training

Run:

- `artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442`

W&B:

- `https://wandb.ai/3252043862-beijing-institute-of-technology/vision-kv-inject/runs/y1n07wvb`

Config:

- 8 GPUs
- Per GPU batch: 4
- Global batch: 32
- KL-only: `lambda_logit=1.0`, `lambda_ce=0.0`
- Max steps: 3125
- Saved step500 checkpoint:
  `artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/qwen_embedding_adapter_step500.pt`

Training behavior:

- KL dropped rapidly from about 9 to about 0.3 by step500.
- Student target token accuracy reached about 0.97-0.98 on copy targets.
- This fixed the earlier issue where KL plateaued around 3 on the older dataset.

### CE-Heavy Continuation From Step500

Run:

- `artifacts/experiments/rendered_text_copy_300_kl1_ce2_from_step500_ds8_mb4_20260821_091822`

W&B:

- `https://wandb.ai/3252043862-beijing-institute-of-technology/vision-kv-inject/runs/o1d83o8y`

Config:

- Init checkpoint: KL-only step500
- 8 GPUs
- Per GPU batch: 4
- Global batch: 32
- `lambda_logit=1.0`
- `lambda_ce=2.0`
- LR: `2e-5`
- Saved checkpoints: step250 and step500

Step500 checkpoint:

- `artifacts/experiments/rendered_text_copy_300_kl1_ce2_from_step500_ds8_mb4_20260821_091822/qwen_embedding_adapter_step500.pt`

Training behavior:

- Student CE quickly dropped to about 0.04-0.08.
- Student token accuracy stayed around 0.984-0.988.
- Training remained stable.

## Experiments And Results

### Text Context Low-Rank KV Compression

Script:

- `test/eval_text_context_lowrank_kv.py`

Output:

- `artifacts/experiments/text_context_lowrank_kv_eval1000_20260821_050503/results.json`

Eval:

- 1,000 rendered-text QA eval rows
- Full text-context teacher score: 0.520
- Rank 32: 0.375
- Rank 64: 0.461
- Rank 128: 0.505

Interpretation:

- Text context influence on attention/KV has substantial low-rank structure.
- Rank 128 nearly preserves full teacher score on that eval.

### Original KL-Only Rendered QA Run Analysis

Run:

- `artifacts/experiments/rendered_text_teacher_v0_kl_only_ds8_mb2acc2_gb32_4000_20260820_144827`

Findings:

- Training log reached only about step1330, although max steps was 4000.
- Eval at step1000:
  - text-only teacher: 0.52
  - native image teacher: 0.509
  - adapter: 0.203
- KL plateau after step1000 was real:
  - step1001-step1330 slope about -0.012 KL per 100 steps
  - mean KL about 3.71

Overfit checks:

- Overfit 8 rows:
  - KL 8.4375 -> 0.0042
  - adapter eval 0.625 vs teacher 0.5
- Overfit 32 rows:
  - KL 7.3535 -> 0.0499
  - adapter eval 0.46875 vs teacher 0.40625

Interpretation:

- Adapter can overfit small data.
- The old plateau was not because the model cannot learn at all.
- Dataset/task construction was likely a major issue.

### Copy/Transcription Eval On 300-Token QA Images

Dataset:

- `data/benchmarks/rendered_qa_300/all_200_400_transcribe.jsonl`
- 76 rows
- Average context length: about 304 Qwen tokens

Checkpoint:

- KL-only step500

Results:

- Adapter text recovery:
  - CER: 0.300
  - character accuracy: 0.703
  - token F1: 0.806
  - exact: 0.0

Interpretation:

- The adapter can recover a large amount of visible text.
- It is not a pure OCR failure.
- Exact transcription is still imperfect.

### 300-Token QA, Mixed MSMARCO + TATQA

Dataset:

- `data/benchmarks/rendered_qa_300/all_200_400.jsonl`
- 76 rows
- Sources:
  - MSMARCO: 16
  - TATQA: 60

Checkpoint:

- KL-only step500

QA result:

- image teacher: 52/76 = 0.684
- adapter: 15/76 = 0.197

By source:

- MSMARCO:
  - teacher: 0.4375
  - adapter: 0.375
- TATQA:
  - teacher: 0.75
  - adapter: 0.15

Interpretation:

- The mixed benchmark is misleading for the current goal because TATQA introduces table/numeric reasoning.
- The adapter performs much better on ordinary paragraph context than on tables.
- TATQA should be excluded from the clean context benchmark for now.

### Adapter Transcription Text -> Text-Only QA

Question:

If adapter can transcribe, can the frozen LLM answer using the adapter-generated transcription as text?

Dataset:

- same 76 mixed rows

Results:

- direct adapter QA: 0.197
- adapter transcription -> text-only LLM QA: 0.408
- image teacher QA: 0.684

Interpretation:

- LLM can answer better when given the adapter's generated text as real tokens.
- Therefore direct visual adapter states are not equivalent to normal text context.
- However, transcription errors also reduce QA score.

### Evidence-Line Localization

Dataset:

- `data/benchmarks/rendered_qa_300/all_200_400_answer_line.jsonl`
- 72 rows
- Gold target is the visible line containing the answer.
- 4 rows were filtered because the answer was cross-line or computed.

Checkpoint:

- KL-only step500

Results:

Overall:

- image teacher token F1: 0.313
- adapter token F1: 0.262
- adapter exact: 0.014
- adapter CER: 3.410

By source:

- MSMARCO:
  - image teacher F1: 0.385
  - adapter F1: 0.427
  - adapter CER: 0.675
- TATQA:
  - image teacher F1: 0.292
  - adapter F1: 0.215
  - adapter CER: 4.191

Interpretation:

- The evidence-line prompt is not ideal because even the image teacher often outputs short answers instead of whole lines.
- MSMARCO evidence localization is usable.
- TATQA/table rows remain poor.

### MSMARCO Paragraph-Only QA, KL-Only Step500

Dataset:

- `data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_100.jsonl`

Checkpoint:

- KL-only step500

Results:

- image teacher: 49/100
- adapter: 25/100

Answer length:

- teacher mean: 7.14 words
- teacher median: 4 words
- adapter mean: 30.63 words
- adapter median: 34.5 words
- adapter outputs over 20 words: 82/100

Interpretation:

- The adapter often outputs long passage-like text instead of short answers.
- It does contain useful context information, but answer style is strongly biased toward copy/continuation.
- The main issue on paragraph context is answer quality/style, not table reasoning.

### Max-New-Tokens And Prompt Ablation

Dataset:

- MSMARCO paragraph-only 100 rows

Checkpoint:

- KL-only step500

Compared settings:

- Original: max_new_tokens=48
- Default prompt + max_new_tokens=8
- Shortest-answer prompt + max_new_tokens=8
- Default prompt + max_new_tokens=12

Results:

| Setting | Teacher | Adapter |
| --- | ---: | ---: |
| Original max_new=48 | 49% | 25% |
| Default prompt, max_new=8 | 32% | 8% |
| Shortest prompt, max_new=8 | 32% | 8% |
| Default prompt, max_new=12 | 40% | 10% |

Shortest prompt:

`Answer with only the shortest answer span. Do not repeat the context.`

Interpretation:

- Hard truncation hurts.
- Many correct relaxed-exact matches require generating enough text to include the gold span.
- Prompt strengthening did not help under max_new=8.
- Inference-side max token reduction is not a solution.

### CE-Heavy Continuation Eval

Dataset:

- MSMARCO paragraph-only 100 rows

Checkpoint:

- CE-heavy continuation step500

Results:

- image teacher: 49/100
- adapter: 27/100

Compared to KL-only step500:

| Checkpoint | Adapter QA |
| --- | ---: |
| KL-only step500 | 25% |
| CE-heavy continuation step500 | 27% |

Answer length:

| Checkpoint | Mean words | Median words | Outputs >20 words |
| --- | ---: | ---: | ---: |
| KL-only step500 | 30.63 | 34.5 | 82/100 |
| CE-heavy continuation step500 | 32.61 | 35.0 | 89/100 |

Interpretation:

- Larger CE gives only a small +2 point improvement.
- It does not solve long passage-like outputs.
- The copy/continuation behavior remains.

## Current Understanding

The adapter has learned useful visual text information. Evidence:

- Copy/transcription token F1 is about 0.8 on 300-token context images.
- On MSMARCO, some adapter outputs contain the correct answer span.
- Adapter can beat teacher on a few individual MSMARCO examples.

However, direct QA is weak because:

- The adapter tends to continue or summarize visible context rather than output a short answer.
- The LLM's QA ability is not fully activated by the adapter's continuous visual context state.
- Direct adapter state is not equivalent to feeding real text tokens.

Important distinction:

- `image -> adapter -> long evidence-ish text` works partially.
- `image -> adapter -> short QA answer` is unstable.
- `adapter transcription -> text-only LLM QA` improves over direct adapter QA, which shows the frozen LLM can use the information when it is presented as normal text tokens.

## Practical Conclusion

For now, the adapter is usable as a visual text collector, but not yet reliable as a direct short-answer QA system.

The most promising near-term use is a two-stage pipeline:

1. `image + question -> adapter output` as evidence-like text.
2. `question + adapter output -> text-only answer extractor`.

Training-only fixes tried so far:

- KL-only copy training: learns transcription/copy well, QA style is poor.
- CE-heavy continuation: small QA gain, no style fix.

Training ideas not yet implemented:

- Train question-conditioned evidence span generation.
- Train an answer extractor on adapter outputs.
- Investigate intermediate-state/KV alignment, but naive pooling or suffix hidden alignment may be problematic.

## Key Artifacts

Datasets:

- Copy train/eval: `data/train/rendered_text_copy_300`
- MSMARCO QA benchmark: `data/benchmarks/rendered_qa_300_msmarco/msmarco_200_400_100.jsonl`
- Mixed QA benchmark: `data/benchmarks/rendered_qa_300/all_200_400.jsonl`
- Mixed transcription benchmark: `data/benchmarks/rendered_qa_300/all_200_400_transcribe.jsonl`
- Evidence-line benchmark: `data/benchmarks/rendered_qa_300/all_200_400_answer_line.jsonl`

Checkpoints:

- KL-only step500:
  `artifacts/experiments/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/qwen_embedding_adapter_step500.pt`
- CE-heavy step500:
  `artifacts/experiments/rendered_text_copy_300_kl1_ce2_from_step500_ds8_mb4_20260821_091822/qwen_embedding_adapter_step500.pt`

Eval results:

- KL-only MSMARCO QA:
  `artifacts/eval/qwen/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/eager/step500_rendered_qa_300_msmarco_100samples_8gpu/results.json`
- CE-heavy MSMARCO QA:
  `artifacts/eval/qwen/rendered_text_copy_300_kl1_ce2_from_step500_ds8_mb4_20260821_091822/eager/step500_rendered_qa_300_msmarco_100samples_8gpu/results.json`
- Mixed 76-row QA:
  `artifacts/eval/qwen/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/eager/step500_rendered_qa_300_all_76samples_1gpu_v2/results.json`
- Mixed 76-row transcription:
  `artifacts/eval/qwen/rendered_text_copy_300_kl_ds8_mb4_wandb_20260821_072442/eager/step500_rendered_qa_300_transcribe_76samples_adapter/results.json`

