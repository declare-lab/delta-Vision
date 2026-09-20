"""Audit immutable evaluation prompts, gold-field isolation, and answer parsing.

Reads existing processed inputs without regenerating images or changing scores.
The full cached prompt must equal a fresh question/options-only chat template,
after collapsing visual placeholders and the processor's video timestamps.
This tests evaluation-pipeline leakage, not pretraining contamination.
"""
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/baseline_answer_leakage_20260914'
CASES = {
    'muirbench': ('muir_random1000_seed42_matched_20260914', 'muirbench_random1000.jsonl'),
    'mmiu': ('mmiu_random1000_seed42_all_methods_20260914', 'mmiu_random1000.jsonl'),
    'videomme': ('mmiu_video_all_methods_matched_20260914', '../video_balanced_base_adapter_20260914/videomme_selected.jsonl'),
    'mvbench': ('mmiu_video_all_methods_matched_20260914', '../video_balanced_base_adapter_20260914/mvbench_selected.jsonl'),
}
SENTINEL = 'AUDIT_GOLD_MUST_NOT_ENTER_MODEL_729461'


def run():
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from transformers import AutoProcessor
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import build_benchmark_prompt
    from src.audit_mmiu_random_results import extract_answer
    from src.multimodal_baseline_suite import MODEL
    from baselines.eval_baselines import _qwen_inputs_from_item

    torch.set_num_threads(4)
    os.environ['QWEN_VIDEO_SAMPLING'] = 'full_timestamp_v1'
    os.environ['QWEN_VIDEO_NUM_FRAMES'] = '8'
    processor = AutoProcessor.from_pretrained(MODEL)
    OUT.mkdir(parents=True, exist_ok=False)
    summary = {}
    parser_changes = []
    suspicious = []
    failures = []
    allowed = {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values',
               'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'}

    def render(ds, row):
        question = build_benchmark_prompt(row, ds.spec, ds.answer_instruction)
        content = ds._qwen_message_content(row, question, [None] * len(ds._image_paths(row)),
                                          [None] * len(ds._video_paths(row)))
        return processor.apply_chat_template([{'role': 'user', 'content': content}],
                                             tokenize=False, add_generation_prompt=True)

    def collapse(text):
        # Native processor expands each video into timestamped frame groups,
        # enclosed by the original vision-start/end pair. Only remove that
        # specific media expansion; never strip arbitrary prompt text.
        text = re.sub(r'(?:<[0-9]+(?:\.[0-9]+)? seconds><\|vision_start\|>'
                      r'(?:<\|video_pad\|>)+<\|vision_end\|>)+', '<|video_pad|>', text)
        return re.sub(r'(?:<\|image_pad\|>)+', '<|image_pad|>', text)

    with (OUT / 'inputs.jsonl').open('w', buffering=1) as log:
        for bench, (folder, relative_manifest) in CASES.items():
            root = ROOT / 'artifacts/diagnostics' / folder
            manifest = (root / relative_manifest).resolve()
            ds = QwenBenchmarkDataset(str(manifest), processor, bench,
                data_root=str(ROOT / 'data/benchmarks' / bench), max_samples=1000,
                cache_dir=root / 'processed' / bench, prompt_layout='media_first_v1')
            counts = collections.Counter()
            for i, row in enumerate(ds.rows):
                expected = render(ds, row)
                changed = dict(row)
                for field in ('answer', 'answer_text', 'answers', 'label', 'labels', 'gold',
                              'correct_answer', 'teacher_answer'):
                    changed[field] = SENTINEL
                mutated = render(ds, changed)
                assert expected == mutated and SENTINEL not in mutated, (bench, i, 'gold enters rendering')
                q = build_benchmark_prompt(row, ds.spec, ds.answer_instruction)
                paths = ds._image_paths(row); videos = ds._video_paths(row)
                cache = ds._cache_path(row, paths, videos, str(row.get('problem') or q))
                assert cache.is_file(), (bench, i, 'missing actual input cache', str(cache))
                item = torch.load(cache, map_location='cpu', weights_only=False, mmap=True)['item']
                actual = processor.tokenizer.decode(item['input_ids'], skip_special_tokens=False)
                exact = collapse(actual) == expected
                if not exact:
                    failures.append(dict(benchmark=bench,index=i,expected=expected,actual=collapse(actual)))
                assert actual.endswith('<|im_start|>assistant\n'), (bench,i,'assistant answer in input')
                assert re.findall(r'<\|im_start\|>(\w+)', actual) == ['user', 'assistant'], (bench,i,'extra chat turns')
                before = _qwen_inputs_from_item(item, torch.device('cpu'))
                altered = dict(item, row=changed, **{k:SENTINEL for k in
                    ('answer','answer_text','answers','label','labels','gold','correct_answer','teacher_answer')})
                after = _qwen_inputs_from_item(altered, torch.device('cpu'))
                assert set(before) == set(after) <= allowed
                for key in before:
                    assert torch.is_tensor(before[key]) and torch.is_tensor(after[key])
                    # The same unmodified tensor storage, not just a matching
                    # shape: no pixels or input tokens depend on gold metadata.
                    assert before[key].data_ptr() == after[key].data_ptr()
                    assert before[key].shape == after[key].shape
                if re.search(r'(?:correct answer|ground truth|答案|answer)\s*(?:is|:|=|是)\s*[([]?[A-F]\b', q, re.I):
                    suspicious.append(dict(benchmark=bench,index=i,question=q,answer=row['answer']))
                counts['samples'] += 1
                counts['cached_prompt_exact'] += exact
                counts['gold_mutation_no_input_change'] += 1
                counts['empty_assistant_prefix'] += 1
                log.write(json.dumps(dict(benchmark=bench,index=i,source_index=row['index'],
                    cached_prompt_exact=exact,gold_mutation_no_input_change=True,
                    model_keys=sorted(before),prompt_sha256=hashlib.sha256(actual.encode()).hexdigest()))+'\n')
                if i % 200 == 0: print(bench, i, 'prompt/gold audit', flush=True)
            per_method = collections.defaultdict(lambda: collections.Counter())
            input_hashes = {}
            for path in sorted(root.glob('*_shard*.jsonl')):
                for line in path.open():
                    r = json.loads(line)
                    if r['benchmark'] != bench: continue
                    sample = ds.rows[r['index']]
                    assert r['source_index'] == sample['index'] and r['gold'] == sample['answer']
                    old = input_hashes.setdefault(r['index'], r['input_sha256'])
                    assert old == r['input_sha256'], (bench,r['index'],'method-dependent input')
                    pred = extract_answer(r['text'], sample['choices'])
                    strict_score = int(pred is not None and pred == sample['answer'])
                    key = f"{r['method']}@{r['retention']}"
                    stats = per_method[key]
                    stats['n'] += 1; stats['old_correct'] += r['score']; stats['strict_correct'] += strict_score
                    stats['prediction_changes'] += pred != r['prediction']
                    if pred != r['prediction'] or strict_score != r['score']:
                        parser_changes.append(dict(benchmark=bench,index=r['index'],method=r['method'],
                            retention=r['retention'],text=r['text'],old_prediction=r['prediction'],
                            strict_prediction=pred,gold=r['gold'],old_score=r['score'],strict_score=strict_score))
            summary[bench] = dict(counts, recorded_predictions=sum(s['n'] for s in per_method.values()),
                                  method_input_hashes_equal=True, per_method=dict(per_method))
            print(bench, dict(counts), flush=True)
    payload = dict(scope='Evaluation pipeline only; does not establish absence of training contamination',
                   benchmarks=summary, cached_prompt_failures=len(failures),
                   parser_changes=len(parser_changes), suspicious_prompt_count=len(suspicious))
    for name, value in [('summary.json', payload), ('prompt_failures.json', failures),
                        ('parser_changes.json',parser_changes),('suspicious_prompts.json',suspicious)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    run()
