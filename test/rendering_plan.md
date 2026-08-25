# Text Rendering Plan

## Goal

Render text-only memory/context data into compact Pillow images for Qwen3-VL-4B.

Primary target:

```text
text_tokens / visual_tokens ~= 1.5
```

Use the Qwen3-VL-4B tokenizer for `text_tokens` and the Qwen3-VL processor output
(`mm_token_type_ids`) for measured `visual_tokens`.

## Data Units

### Qasper

Input:

```text
data/train/qasper-data/agent_memory_qasper_ctx8192_episode_safe_seed42.sectioned.jsonl
```

Each row already has user messages regrouped by original paper sections:

- `Abstract` is one user block.
- Each `Section: ...` block is one user block.
- The final `Question:` user message is not a context block.

Rendering rule:

- Group context user blocks into token-balanced pages.
- Preserve original order.
- Balance pages by Qwen text token count, not by number of blocks.
- Do not render the final `Question:` into context images.
- Keep the question/answer text in metadata for later QA construction.
- Do not use a fixed 4-image or 8-image split.

Final Qasper training plan:

- Work from `data/train/qasper-data/agent_memory_qasper_ctx8192_episode_safe_seed42.sectioned.jsonl`.
- Within each row, use sectioned context user blocks:

  ```text
  Abstract
  Section: ...
  Section: ...
  ...
  ```

- Exclude the final `Question:` user message from rendered context pages.
- Group section blocks into pages with:

  ```text
  target_page_tokens = 1024
  soft_max_page_tokens = 1280 or 1536
  min_page_tokens = 512
  ```

- Preserve section order.
- Merge short sections/pages with adjacent pages.
- If a single section is much longer than the soft max, allow it to become its own page
  first; paragraph-level splitting can be added later if needed.
- Each grouped page becomes one image.
- Keep original row-level QA fields in metadata.

### HotpotQA

Input:

```text
data/train/hotpotqa/train.jsonl
data/benchmarks/hotpotqa/validation.jsonl
```

Rendering rule:

- Initial rule: each `context_texts` element is one rendered image.
- A `context_texts` element is one title plus all sentences under that title.
- Preserve original `question`, `answer`, `supporting_facts`, `support_facts`,
  and `context` fields in metadata.

Diagnostic note:

- The initial one-`context_texts`-per-image rule is too fragmented for HotpotQA.
- Many HotpotQA context units are short, so forcing the `text_tokens / visual_tokens ~= 1.5`
  target creates very small images.
- Very small images are compact but lower quality for OCR/visual reading and do not look like
  normal document pages.
- A better HotpotQA rule is to group multiple `context_texts` units from the same row into
  2 to 4 rendered pages, or into token-balanced pages of roughly `<=512` text tokens.

Final HotpotQA benchmark plan:

- Work from `data/benchmarks/hotpotqa/validation.jsonl`.
- Within each row, preserve the order of `context_texts`.
- Group `context_texts` into token-balanced pages:

  ```text
  target_page_tokens = 512
  soft_max_page_tokens = 768
  min_page_tokens = 256
  ```

- Do not allow very short context units to become standalone pages when there is an
  adjacent page to merge with.
- If an entire row has less than `min_page_tokens`, allow a smaller dynamic canvas rather
  than forcing a large page.
- Each grouped page becomes one image.
- Keep all original row-level QA fields in the per-image metadata.

## Layout Requirements

- Images must be compact.
- Width and height are both controlled automatically.
- Avoid long thin images.
- Avoid large blank space at the bottom.
- Prefer near-square or mildly rectangular pages.
- Formulas are rendered as plain text source, not LaTeX math.
- Readability is more important than exactly hitting the token compression target.
- Do not make tiny images just to satisfy the ratio.
- Use minimum dimensions for normal page-like rendering, especially for short HotpotQA contexts.

Default layout constraints:

```text
target_ratio = 1.5
max_qasper_context_images = 4 or 8
min_width = 512 or 640 px
min_height = 320 px for normal pages
short_page_min_width = 320 px
short_page_min_height = 160 px
min_font_size = 14 or 15
padding = 24..36 px
line_spacing = 2..4 px
paragraph_spacing = 0.5..1.0 line
font = /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

For the current compact HotpotQA grouped style:

```text
min_width = 640
min_height = 320
short_page_min_width = 320
short_page_min_height = 160
min_font_size = 14
padding candidates = 10, 12, 16
line_spacing candidates = 1, 2, 3
```

## Layout Search

For each render unit:

1. Count text tokens with Qwen3-VL-4B tokenizer.
2. Compute target visual tokens:

   ```text
   target_visual_tokens = text_tokens / 1.5
   ```

3. Convert target visual tokens to approximate target pixel area:

   ```text
   target_area ~= target_visual_tokens * 1024
   ```

   This matches measured Qwen3-VL behavior where `512x512 -> 256` image tokens
   and `1024x1024 -> 1024` image tokens.

4. Search candidate layouts:

   - widths around the target square size;
   - font sizes within a readable range;
   - small padding and line spacing;
   - dynamic height from actual wrapped text.

5. Render candidate with Pillow.
6. Measure actual visual tokens with the Qwen3-VL processor.
7. Select the candidate with the best score:

   ```text
   score =
     abs(text_tokens / visual_tokens - 1.5)
     + aspect_ratio_penalty
     + blank_space_penalty
   ```

For production rendering, the score should treat the token ratio as a soft constraint:

```text
score =
  readability_penalty
  + page_shape_penalty
  + blank_space_penalty
  + soft_ratio_penalty
```

Avoid layouts with tiny width/height or unreadably small font even when they match the
`1.5` ratio.

## Output

Rendered images should go under:

```text
test/results/rendered_text/
```

For final HotpotQA benchmark rendering, write under:

```text
data/benchmarks/hotpotqa/rendered_validation_grouped/
  images/
    row_000000_page_00.png
    row_000000_page_01.png
    ...
  metadata.jsonl
  summary.json
  args.json
```

Do not mix diagnostic partial outputs with final outputs. If a run is interrupted, delete
or archive that output directory before restarting.

Suggested layout:

```text
test/results/rendered_text/
  qasper/
    row_000000/
      context_00.png
      context_01.png
      ...
      metadata.json
  hotpotqa/
    train/
      row_000000/
        context_00.png
        ...
        metadata.json
    validation/
      row_000000/
        context_00.png
        ...
        metadata.json
```

For final Qasper training rendering, write under:

```text
data/train/render-data/qasper/
  images/
    row_000000_page_00.png
    row_000000_page_01.png
    ...
  metadata.jsonl
  summary.json
  args.json
```

For final HotpotQA training rendering, write under:

```text
data/train/render-data/hotpotqa/
  images/
    row_000000_page_00.png
    row_000000_page_01.png
    ...
  metadata.jsonl
  summary.json
  args.json
```

Each metadata item should include:

```json
{
  "image": "context_00.png",
  "source": "qasper|hotpotqa",
  "row_index": 0,
  "unit_index": 0,
  "text": "...",
  "text_tokens": 512,
  "visual_tokens": 341,
  "ratio": 1.50,
  "width": 640,
  "height": 640,
  "font_size": 18,
  "line_count": 32,
  "padding": 28
}
```

For HotpotQA grouped rendering, each metadata line corresponds to exactly one rendered image
and must include enough information to map the image back to the source JSONL:

```json
{
  "image": "data/benchmarks/hotpotqa/rendered_validation_grouped/images/row_000705_page_01.png",
  "source": "hotpotqa",
  "split": "validation",
  "row_index": 705,
  "page_index": 1,
  "num_pages": 4,
  "context_indices": [3, 4],
  "titles": [
    "Boise State–Nevada football rivalry",
    "Leon Rice (basketball)"
  ],
  "id": "5add10ca5542994ed6169c59",
  "question": "Were Illinois Institute of Technology and Boise State University both bounded before 1950?",
  "answer": "yes",
  "supporting_facts": {
    "title": ["Illinois Institute of Technology", "Boise State University"],
    "sent_id": [1, 1]
  },
  "support_facts": [
    {
      "title": "Illinois Institute of Technology",
      "sent_id": 1,
      "sentence": "It traces its history to several 19th century engineering and professional education institutions in the United States.",
      "text": "Illinois Institute of Technology\nIt traces its history to several 19th century engineering and professional education institutions in the United States."
    }
  ],
  "text": "the exact text rendered into this image",
  "text_tokens": 464,
  "visual_tokens": 308,
  "ratio": 1.506
}
```

Mapping rule:

```python
source_row = validation_jsonl[row_index]
rendered_contexts = [source_row["context_texts"][i] for i in context_indices]
rendered_titles = [source_row["context"]["title"][i] for i in context_indices]
```

For Qasper grouped rendering, each metadata line corresponds to one rendered context image:

```json
{
  "image": "data/train/render-data/qasper/images/row_000000_page_00.png",
  "source": "qasper",
  "split": "train",
  "row_index": 0,
  "page_index": 0,
  "num_pages": 6,
  "block_indices": [0, 1, 2],
  "block_token_counts": [261, 793, 622],
  "paper_id": "1909.06937",
  "question_id": "b4f5bf3b7b37e2f22d13b724ca8fe7d0888e04a2",
  "question": "Question:\n...",
  "answer": "speaker systems in the real world",
  "text": "the exact section text rendered into this image",
  "text_tokens": 1024,
  "visual_tokens": 683,
  "ratio": 1.499
}
```

Mapping rule:

```python
source_row = qasper_sectioned_jsonl[row_index]
user_messages = [m for m in source_row["messages"] if m["role"] == "user"]
context_blocks = user_messages[:-1]  # final user is Question
rendered_blocks = [context_blocks[i] for i in block_indices]
```

## First Test Run

Initial diagnostic run:

- Qasper: first 100 rows with 1024-token grouped pages.
- HotpotQA: use a small subset first. Validate whether grouped pages are needed before
  running the full validation split.

The diagnostic should print aggregate stats:

- number of images;
- text token min/median/max;
- visual token min/median/max;
- ratio min/median/max;
- width/height min/median/max;
- count outside target ratio tolerance.

## Current Diagnostic Outcome

The first HotpotQA benchmark rendering attempt used one `context_texts` item per image and
started a full validation render under:

```text
data/benchmarks/hotpotqa/rendered_validation
```

That run was stopped because the output quality was not good enough:

- single-title HotpotQA units are often too short;
- images become too small when optimized for `text_tokens / visual_tokens ~= 1.5`;
- the resulting pages are compact but not visually robust;
- the partial output should be treated as diagnostic scratch, not final data.

Next HotpotQA rendering should group context units within each row into fewer, denser pages.

Follow-up grouped diagnostic:

- Grouping reduced HotpotQA validation from `73700` individual context units to about
  `21863` rendered pages.
- Main ratio distribution was good:

  ```text
  p50 ~= 1.50
  p90 ~= 1.53
  p95 ~= 1.54
  mean ~= 1.49
  ```

- The remaining bad cases were very short whole-row pages, where `min_width/min_height`
  forced too many visual tokens.
- The code should handle these by allowing a smaller canvas for rows/pages below
  `min_page_tokens`.

## Resume Instructions

Rendering scripts are resumable. A rendered item is skipped when all of these match:

- image file exists;
- `item_key` exists in metadata;
- `text_hash` matches;
- `layout_version` matches;
- `layout_config_hash` matches.

Metadata is written through a temporary file and atomically replaced at the end of a run.
For sharded runs, each completed shard writes:

```text
OUTPUT_DIR/shards/shard_00000_of_00016.jsonl
OUTPUT_DIR/shards/shard_00000_of_00016.done
```

If a run is interrupted, completed shard files are safe to keep. Restarting the same command
with the same layout args will reuse completed images/metadata.

### Qasper Train

Output:

```text
data/train/render-data/qasper
```

Resume command:

```bash
.venv/bin/python test/diagnostics/render_qasper_grouped.py \
  --workers 224 \
  --output-dir data/train/render-data/qasper
```

Force a full rerender only when intentionally changing layout/data:

```bash
.venv/bin/python test/diagnostics/render_qasper_grouped.py \
  --workers 224 \
  --output-dir data/train/render-data/qasper \
  --overwrite
```

### HotpotQA Train

Output:

```text
data/train/render-data/hotpot
```

The train split is large. Run it in 16 shards so interruption does not require a full rerun.

Resume all shards in order:

```bash
for shard in $(seq 0 15); do
  .venv/bin/python test/diagnostics/render_hotpotqa_benchmark_grouped.py \
    --data data/train/hotpotqa/train.jsonl \
    --output-dir data/train/render-data/hotpot \
    --split train \
    --workers 224 \
    --num-shards 16 \
    --shard-id "$shard"
done
```

Resume from a known shard, for example shard 5:

```bash
for shard in $(seq 5 15); do
  .venv/bin/python test/diagnostics/render_hotpotqa_benchmark_grouped.py \
    --data data/train/hotpotqa/train.jsonl \
    --output-dir data/train/render-data/hotpot \
    --split train \
    --workers 224 \
    --num-shards 16 \
    --shard-id "$shard"
done
```

Completed shard metadata files are under:

```text
data/train/render-data/hotpot/shards/
```

### HotpotQA Benchmark

Output:

```text
data/benchmarks/hotpotqa/render
```

Resume command:

```bash
.venv/bin/python test/diagnostics/render_hotpotqa_benchmark_grouped.py \
  --data data/benchmarks/hotpotqa/validation.jsonl \
  --output-dir data/benchmarks/hotpotqa/render \
  --split validation \
  --workers 224
```

Use shards if desired:

```bash
for shard in $(seq 0 3); do
  .venv/bin/python test/diagnostics/render_hotpotqa_benchmark_grouped.py \
    --data data/benchmarks/hotpotqa/validation.jsonl \
    --output-dir data/benchmarks/hotpotqa/render \
    --split validation \
    --workers 224 \
    --num-shards 4 \
    --shard-id "$shard"
done
```

### Checking Progress

Count rendered images and metadata:

```bash
find data/train/render-data/qasper/images -type f -name '*.png' | wc -l
wc -l data/train/render-data/qasper/metadata.jsonl

find data/train/render-data/hotpot/images -type f -name '*.png' | wc -l
find data/train/render-data/hotpot/shards -type f -name 'shard_*_of_*.jsonl' -print

find data/benchmarks/hotpotqa/render/images -type f -name '*.png' | wc -l
wc -l data/benchmarks/hotpotqa/render/metadata.jsonl
```
