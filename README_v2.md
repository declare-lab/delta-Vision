# Pixmo + OCR + LLaVA Instruct + Rendered Text Data Mix v1

## Goal

Build one Qwen visual-delta training set that covers both current general benchmarks and OCR-heavy benchmarks.

Current eval set is not only OCR:

| Benchmark | Main task type | Data need |
| --- | --- | --- |
| MMStar | multi-choice perception/reasoning/math/science | general MC + reasoning |
| GQA | short-answer object/attribute/relation QA | general short answer |
| MMB / MMB-CN | multi-choice general VQA, OCR, reasoning, Chinese | general MC + some OCR + Chinese |
| MME | yes/no perception, OCR, reasoning | yes/no + object/category/OCR |
| POPE | yes/no object existence | object existence yes/no |
| SQA | science multi-choice | science/diagram/common-sense MC |
| VQA-v2 | general short-answer VQA | general short answer |
| VizWiz | real user VQA, often OCR/unanswerable | short answer + realistic OCR |
| OCRBench | text recognition/doc/KIE/formula/structured OCR | OCR, document, KIE, formula, rendered text |

The target is to improve general retention without losing OCRBench gains.

There is also a longer-term target: storing long textual context as images and letting the visual-delta path retrieve useful information from that visual memory. Rendered text is therefore not only an OCR augmentation block; it is the first controlled training source for long-context-as-image behavior.

## Previous OCR-Focused Result

Baseline run:

```text
artifacts/experiments/qwen_topk1024_freezeqkv/qwen_opd_rollout2k_injection_2000step_20260816_043953
```

Eval directory:

```text
artifacts/eval/qwen_topk1024_freezeqkv/qwen_opd_rollout2k_injection_2000step_20260816_043953/compiled_dense
```

Setting:

```text
step1500, OPD top1024, 1000 samples per benchmark, 8 GPUs
```

| Benchmark | Teacher | Adapter | Drop | Retention | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| OCRBench | 80.7 | 78.3 | -2.4 | 95.4% | 86.9% |
| MMStar | 64.9 | 49.8 | -15.1 | 65.6% | 62.0% |
| GQA | 61.6 | 51.9 | -9.7 | 73.7% | 65.0% |
| MMB | 87.5 | 80.6 | -6.9 | 88.1% | 84.3% |
| MMB-CN | 87.7 | 78.0 | -9.7 | 85.5% | 82.6% |
| MME acc | 84.7 | 75.0 | -9.7 | 82.4% | 79.9% |
| POPE | 89.3 | 84.1 | -5.2 | 90.8% | 88.8% |
| SQA | 93.3 | 80.1 | -13.2 | 84.2% | 82.8% |
| VQA-v2 | 80.9 | 71.3 | -9.6 | 84.5% | 67.6% |
| VizWiz | 25.9 | 18.9 | -7.0 | 59.9% | 24.3% |

Interpretation:

- OCR retention is strong.
- General VQA and reasoning benchmarks still drop too much, especially MMStar, GQA, SQA, VQA-v2, and VizWiz.
- The next 6k-step training mix should add more general visual/instruction data while keeping OCR at 30%.

## Current 6k-Step Training Mix

Rendered text is not part of the current 6k-step training mix. It is kept as a separate source cache for later long-context-as-image validation experiments.

Use a 100% mixture with OCR capped at about 30%.

| Bucket | Ratio | Target role |
| --- | ---: | --- |
| Pixmo-clean | 35% | broad visual coverage, object/count/color/spatial/general QA |
| FineVision LLaVA-Instruct-150K | 35% | instruction-following general VQA and richer reasoning |
| Current clean OCR mix | 30% | OCRBench/doc/chart/KIE/formula coverage |

For a 6k-step run with global batch 32, the model sees exactly 192k training examples:

```text
6,000 steps * 32 samples/step = 192,000 samples
```

The raw sample consumption is 192k rows, so the training pool should be larger than that to avoid an exact one-epoch boundary.

Using exactly 192k rows is too tight: a 6k-step run would consume the whole pool once. The current main training dataset is a shuffled 256k pool, so 6k steps consume about 75% of it.

The current main training dataset is therefore 256k rows:

| Bucket | Rows |
| --- | ---: |
| Pixmo-clean | 96,000 |
| FineVision LLaVA-Instruct-150K | 96,000 |
| Current clean OCR mix | 64,000 |
| Total | 256,000 |

At 6k steps:

```text
192,000 / 256,000 = 75%
```

For a smaller 80k-row ablation:

| Bucket | Rows |
| --- | ---: |
| Pixmo-clean | 28,000 |
| FineVision LLaVA-Instruct-150K | 28,000 |
| Current clean OCR mix | 24,000 |
| Total | 80,000 |

## Source Paths

Pixmo-clean:

```text
/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl
```

FineVision LLaVA-Instruct-150K source:

```text
/lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k/train.jsonl
```

Only use rows with:

```text
source == "finevision_llava_150k"
```

Current clean OCR mix:

```text
/lustre-data/leijingdi/code/vision-kv-inject/data/ocrbench_target_mix_v2_80k/train.jsonl
```

Current 256k training dataset:

```text
/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_llava_instruct_ocr_256k_v1/train.jsonl
/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_llava_instruct_ocr_256k_v1/manifest.json
```

Previous exact-one-epoch 192k version, kept for reference:

```text
/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_llava_instruct_ocr_192k_v1/train.jsonl
/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_llava_instruct_ocr_192k_v1/manifest.json
```

Rendered real-text source cache for later validation:

```text
/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/
/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/download_manifest_real_text.json
```

## Why Not Raw Pixmo-Heavy

Pixmo-clean has good visual diversity but its supervision style is mismatched to the current evals.

Measured on Pixmo-clean:

| Stat | Value |
| --- | ---: |
| Rows | 135,995 |
| Answer words avg | 75.9 |
| Answer words p50 | 67 |
| Answer words p90 | 140 |
| Answer length <= 5 words | 2,667 rows, 1.96% |

Most current benchmark answers are one word, yes/no, or one option letter. Therefore Pixmo should not dominate the dataset in raw long-answer form.

## Filtering Policy

Apply light filtering only. Do not rewrite all prompts into one template.

For Pixmo-clean and LLaVA-Instruct:

- Keep original question text.
- Keep original answer text for now, but cap very long answers.
- Drop rows with empty image/question/answer.
- Drop rows whose answer exceeds 160 words.
- Prefer sampling rows with answer length <= 120 words.
- Keep yes/no, count, color, spatial, object, and OCR-like questions.
- Do not force all answers to short answers in v1, because this would require a teacher rewrite pass and may introduce another failure mode.

For current OCR mix:

- Use `ocrbench_target_mix_v2_80k` as the base clean OCR source.
- Preserve original question/answer.
- Keep all major sources: docvqa, pdfvqa, ureader_qa_processed, st_vqa, chartqa, plotqa, infographic_vqa, sroie, funsd, ocrvqa, hme100k, cord_receipt_kie.
- Drop only obviously bad rows: empty answer, unanswerable labels, full-page OCR dumps, binary yes/no if they appear again.

For rendered real-text validation sources:

- Do not mix into the current 6k-step training dataset.
- Keep as raw text/context sources for later rendered-image validation experiments.
- Use short, directly checkable answers.
- Do not use benchmark validation/test images or rows.

## Rendered Text Source Search

Rendered data should be built from real text datasets first. Manual templates are only for formatting/rendering, not for inventing toy content.
These sources are downloaded and saved for later validation experiments only. They are not included in the current 256k training JSONL.

The sources below were checked for availability and sample fields.

| Rendered bucket | Candidate data | Available fields | Use in v1 |
| --- | --- | --- | --- |
| Long-context QA | `zai-org/LongBench` | `context`, `input`, `answers`, `dataset`, `language`, `length` | Yes, small curated subset; good for rendered page QA templates |
| Long-context MC | `zai-org/LongBench-v2` | `context`, `question`, `choice_A-D`, `answer`, `domain`, `sub_domain`, `difficulty` | Yes, small hard subset; do not oversample |
| Synthetic needle retrieval | `tonychenxyz/ruler-full` | `prompt`, `category`, `extra_info.ground_truth`, `context_length` | No; too artificial for this data mix |
| Long reports | `ccdv/govreport-summarization` | `report`, `summary` | Maybe; useful for long pages, but answers are summaries and too long unless converted to retrieval QA |
| Table + text QA | `next-tat/TAT-QA` | `table`, `paragraphs`, `questions` with answers | Yes, strong fit for table+paragraph rendering |
| Financial table QA | `wandb/finqa-data-processed` | `pre_text`, `post_text`, `table`, `query`, `output`, `program` | Yes, strong fit for table/logical/numeric lookup |
| Wikipedia table QA | `DongfuJiang/FeTaQA` | `table_array`, `question`, `answer`, highlighted cells | Yes, but answers can be sentence-length; filter short answers |
| Multi-hop wiki QA | `hotpotqa/hotpot_qa` | `context`, `question`, `answer`, `supporting_facts` | Yes, good for paragraph retrieval and distractors |
| Web/passages QA | `microsoft/ms_marco` | `query`, `passages`, `answers` | Yes, render selected passages; answers usually short enough |
| Wiki document short QA | `abacusai/WikiQA-Free_Form_QA` | conversation containing document/question/short answer | Maybe; useful if parsing is clean |
| Code pages | `Nan-Do/code-search-net-python` | `code`, `docstring`, `summary`, `repo`, `path`, `func_name` | Yes, render code and ask function/path/docstring questions |
| Code long context | `LongBench/repobench-p` | `context`, `input`, `answers`, `language` | Yes, small subset for code retrieval/completion style |
| Real logs | `logfit-project/HDFS_v1` | date/time/level/component/content/block_id/anomaly | Yes, render log windows and ask block/component/error lookup |
| Real log corpus | `bolu61/loghub_2` | raw Apache/HDFS/Hadoop/OpenStack/Spark/etc. logs | Yes, sample small windows; no need to download all systems |
| Dialogue | `knkarthick/samsum` | `dialogue`, `summary` | Maybe; summary too long, use extracted speaker/status/deadline questions if generated carefully |
| Emails | `corbt/enron-emails` | from/to/date/subject/body | Yes, but sanitize/filter personal-looking content and keep small |
| Support tickets | `Tobi-Bueck/customer-support-tickets` | subject/body/answer/type/queue/priority/language/tags | Maybe; multilingual, answers long; use metadata lookup or English-only subset |

LongBench v1 was downloaded only for inspection. It contains relevant files including `qasper`, `gov_report`, `hotpotqa`, `2wikimqa`, `musique`, `qmsum`, `samsum`, `repobench-p`, `passage_retrieval`, `passage_count`, `multifieldqa_en`, and `multifieldqa_zh`.

Downloaded source cache:

| Source | Local path | Size |
| --- | --- | ---: |
| LongBench | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/zai-org__LongBench` | 0.106 GB |
| LongBench-v2 | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/zai-org__LongBench-v2` | 0.434 GB |
| GovReport | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/ccdv__govreport-summarization` | 0.472 GB |
| TAT-QA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/next-tat__TAT-QA` | 0.016 GB |
| FinQA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/wandb__finqa-data-processed` | 0.030 GB |
| FeTaQA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/DongfuJiang__FeTaQA` | 0.018 GB |
| HybridQA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/wenhu__hybrid_qa` | 0.180 GB |
| HotpotQA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/hotpotqa__hotpot_qa` | 0.695 GB |
| MS MARCO | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/microsoft__ms_marco` | 2.164 GB |
| WikiQA | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/microsoft__wiki_qa` | 0.003 GB |
| Natural Questions pairs | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/sentence-transformers__natural-questions` | 0.041 GB |
| WikiQA-Free Form | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/abacusai__WikiQA-Free_Form_QA` | 0.010 GB |
| CodeSearchNet Python | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/Nan-Do__code-search-net-python` | 0.558 GB |
| HDFS_v1 logs | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/logfit-project__HDFS_v1` | 0.405 GB |
| LogHub raw logs | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/bolu61__loghub_2` | 4.706 GB |
| SAMSum | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/knkarthick__samsum` | 0.010 GB |
| DialogSum | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/knkarthick__dialogsum` | 0.012 GB |
| MultiWOZ 2.2 | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/tuetschek__multi_woz_v22` | 0.077 GB |
| Enron emails | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/corbt__enron-emails` | 0.460 GB |
| Support tickets | `/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/Tobi-Bueck__customer-support-tickets` | 0.049 GB |

Download manifest:

```text
/lustre-data/leijingdi/code/vision-kv-inject/data/rendered_real_text_sources/download_manifest_real_text.json
```

Explicitly excluded:

```text
tonychenxyz/ruler-full
```

Reason: RULER/needle retrieval is too artificial for the current rendered validation direction.

## Later Rendered Validation Buckets

Rendered data is for a later validation experiment, not for the current 6k-step training mix.
If building a 19.2k rendered validation/training block later, use data-source-driven buckets:

| Bucket | Rows | Primary sources | What the image looks like | Question style |
| --- | ---: | --- | --- | --- |
| Long QA pages | 3,600 | LongBench, HotpotQA, MS MARCO | one or more dense text pages with distractor paragraphs | short answer lookup / multi-hop |
| Table + paragraph pages | 3,600 | TAT-QA, FinQA, FeTaQA | financial/table/wiki table plus surrounding text | cell lookup, comparison, numeric answer |
| Code/config pages | 2,400 | CodeSearchNet, RepoBench-P | code file snippets, imports, comments, config-like blocks | function/path/variable/next-line lookup |
| Log pages | 2,400 | HDFS_v1, LogHub | dense log windows with timestamp, level, component | block id, component, level, anomaly lookup |
| Email/ticket/dialogue pages | 2,400 | Enron, SAMSum, support tickets | email thread, ticket, chat transcript | sender/date/priority/owner/status lookup |
| Real record/config pages | 2,400 | GovReport, FinQA, TAT-QA, CodeSearchNet, logs, support tickets | real dense records, configs, tables, and structured text pages | exact lookup, count, verification |
| Chinese/multilingual pages | 1,200 | LongBench zh, support-ticket multilingual subset | Chinese/mixed text document pages | short answer lookup |
| OCR primitives | 1,200 | generated strings only | short labels/IDs/numbers | exact read; kept small |

For an 8k rendered validation ablation, halve these counts.

This should stay intentionally small. Rendered pages should be high-quality and targeted rather than huge.

## Rendering Rules

- Render one realistic page/window from the source text, then ask a question whose answer is visible in that rendered image.
- Prefer answers of 1-8 words.
- Avoid full-page OCR dump targets except a small diagnostic subset.
- Use multiple layouts, but the layout should follow the source type: report page, table page, code page, log window, email thread, ticket page, record/config page.
- For dense pages, allow up to 3 QA rows per rendered image, but keep most images one QA each to avoid memorizing the visual context.
- Add light visual variation: font, margins, line spacing, page width, mild JPEG/noise/blur, but keep the text readable.
- Do not use our multimodal benchmark validation/test rows as training data.
- Track rendered source separately in `source`, e.g. `rendered_longbench`, `rendered_govreport`, `rendered_tatqa`, `rendered_finqa`, `rendered_codesearchnet`, `rendered_loghub`.

Excluded source:

- `tonychenxyz/ruler-full` is not used in v1 because the needle/KV prompts are too artificial. Real long-context pages from LongBench, government reports, financial documents, code, logs, emails, tickets, and web QA remain in scope.

## Rendered Acceptance Check

Before training, manually inspect a preview grid with at least 5 examples per rendered bucket.

Each example should show:

- rendered image
- source dataset
- source row id if available
- question
- answer
- visible evidence span or table cell if available

Reject a rendered bucket if:

- the answer is not visible in the image
- the image is unreadable
- the prompt is a repeated toy template
- answers are mostly long summaries
- one visual format dominates
- train/dev examples reuse near-identical source text

## JSONL Schema

Every row should use the current Qwen training schema:

```json
{
  "image": "relative/or/absolute/path.jpg",
  "image_root": "/optional/root/for/relative/path",
  "question": "question text",
  "answer": "answer text",
  "source": "pixmo_clean|finevision_llava_150k|...|rendered_text",
  "conversation_turn": 0
}
```

Rules:

- If `image` is absolute, `image_root` is optional.
- If `image` is relative and data is outside the final dataset directory, set `image_root`.
- For rendered images, use paths relative to the final dataset directory and set `image_root` to the final dataset directory.
- Keep `source` accurate because training logs source-specific loss groups.

## Sampling Details

Use deterministic sampling with a fixed seed, initially `49`.

Pixmo-clean:

- Sample 96k rows for the current 256k version.
- Normalize rows by adding `source: pixmo_clean`.
- Prefer shorter answers under the cap, but keep some long answers for general instruction ability.

FineVision LLaVA-Instruct:

- Sample 96k rows from `source == finevision_llava_150k` for the current 256k version.
- Add `image_root: /lustre-data/leijingdi/code/delta-vision/data/pixmo_clean_finevision_llava150k` when image paths are relative.

Current OCR:

- Sample 64k rows from `ocrbench_target_mix_v2_80k` for the current 256k version.
- Preserve existing `image_root`.
- Keep source distribution close to the existing clean OCR mix unless a specific source is too small.

Rendered text:

- Not included in the current 256k training dataset.
- For later rendered validation experiments, use source-specific names such as `rendered_longbench`, `rendered_tatqa`, `rendered_finqa`, and `rendered_loghub`.
- Use task-specific questions, not one unified prompt.

Shuffle the final combined JSONL after source sampling.

## Expected Effect

Compared with `ocrbench_target_mix_v2_80k` alone:

- Better MMStar/MMB/SQA/GQA/VQA-v2 retention because 75% of the data pool is general visual/instruction data.
- OCRBench should remain strong because 25% of the data pool is still clean OCR-heavy data.
- Less risk of the model becoming an OCR-only adapter.

Compared with `pixmo_clean_ocrmix_300k`:

- Less long-answer pressure.
- Less raw Pixmo dominance.
- Cleaner OCR supervision.
- More direct coverage of the current eval tasks.

## Validation Before Training

Before launching training, check:

- Total rows and exact source counts.
- Image path validity on at least 1,000 random rows.
- Answer word length stats per source.
- Question type distribution: yes/no, count, color, spatial, OCR-like, short QA.
- A 16-sample Qwen dataloader smoke test.
- A 1-step train smoke test with OPD top1024.

## First Training Run

Use the current mainline training setup unless explicitly changing recipe:

```bash
DATA=/lustre-data/leijingdi/code/vision-kv-inject/data/pixmo_clean_llava_instruct_ocr_256k_v1/train.jsonl \
DATA_ROOT=/lustre-data/leijingdi/code/delta-vision \
SUPERVISION_LOSS=opd \
KL_TOPK=1024 \
LOSS_NORMALIZATION=sample \
LR_SCHEDULER=constant \
WARMUP_RATIO=0.0 \
MAX_STEPS=6000 \
SAVE_EVERY=500 \
WANDB=1 \
bash scripts/train_qwen_delta.sh
```

Evaluate every 500 steps on:

```text
mmstar gqa mmb mmb-cn mme pope sqa vqav2 vizwiz ocrbench
```

Primary regression checks:

- OCRBench should stay close to the current 78.3 adapter result.
- MMStar/SQA/GQA/MMB should improve relative to the OCR-focused run.
