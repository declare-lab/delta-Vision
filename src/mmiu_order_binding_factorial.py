"""Paired image-order x answer-option mapping audit, not a benchmark rewrite."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/mmiu_order_binding_factorial_20260914_v2'
TASKS = {'forensic_detection_blink', 'visual_quality_assessment_ve_lol_l'}
MODES = {'original': (0, 0), 'images_rotated': (1, 0),
         'options_rotated': (0, 1), 'both_rotated': (1, 1)}


def remap_question(question, choices, shift):
    # Rewrite ordinal meanings, not letters. This also handles inline and
    # duplicate option blocks without deleting any of the original formatting.
    result = question
    counts = [question.count(choice) for choice in choices]
    assert all(counts) and len(set(choices)) == len(choices)
    for j, choice in enumerate(choices):
        result = result.replace(choice, f'__AUDIT_ORDINAL_{j}__')
    for j in range(len(choices)):
        result = result.replace(f'__AUDIT_ORDINAL_{j}__', choices[(j + shift) % len(choices)])
    if not shift:
        assert result == question
    return result


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import build_benchmark_prompt, score_prediction
    from src.mmiu_binding_protocol_audit import MODEL, CHECKPOINT, MANIFEST
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='sdpa').eval().requires_grad_(False)
    from src.qwen_deepstack import disable_qwen_deepstack
    disable_qwen_deepstack(model)
    processor = AutoProcessor.from_pretrained(MODEL)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,
        model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.model.language_model.register_forward_pre_hook(
        lambda m, a, k: (a, dict(k, deepstack_visual_embeds=None)), with_kwargs=True)
    def reject(*args, **kwargs):
        raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process = reject
    ds = QwenBenchmarkDataset(str(MANIFEST), processor, 'mmiu', max_samples=1000,
        data_root=str(ROOT / 'data/benchmarks/mmiu'), prompt_layout='media_first_v1')
    indices = [i for i, row in enumerate(ds.rows) if row['task'] in TASKS]
    assert len(indices) == 332
    original_path = ROOT / 'artifacts/diagnostics/mmiu_address_position_audit_20260914'
    previous = {(r['index'], r['method']): r for p in original_path.glob('rows_*.jsonl')
        for line in p.open() if (r := json.loads(line))['mode'] == 'plain'}
    output = OUT / f'rows_{shard}.jsonl'
    assert not output.exists()
    with torch.inference_mode(), output.open('w', buffering=1) as out:
        for index in indices[shard::8]:
            row = ds.rows[index]
            choices = row['choices']
            n = len(choices)
            assert n == len(row['images']) and n in (2, 4)
            assert choices == ['the first image', 'the second image', 'the third image', 'the fourth image'][:n]
            question = build_benchmark_prompt(row, ds.spec)
            images = [Image.open(p).convert('RGB') for p in ds._image_paths(row)]
            item = ds[index]
            original_tensors = None
            for mode, (image_shift, option_shift) in MODES.items():
                order = [(j + image_shift) % n for j in range(n)]
                reordered = [images[j] for j in order]
                q = remap_question(question, choices, option_shift)
                new_choices = [choices[(j + option_shift) % n] for j in range(n)]
                gold = chr(65 + (ord(row['answer']) - 65 - image_shift - option_shift) % n)
                content = [{'type': 'image', 'image': im} for im in reordered] + [{'type': 'text', 'text': q}]
                prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=False, add_generation_prompt=True)
                cpu_inputs = dict(processor(text=[prompt], images=reordered, return_tensors='pt', padding=True))
                if mode == 'original':
                    for k, value in cpu_inputs.items():
                        expected = item[k].unsqueeze(0) if k in ('input_ids', 'attention_mask', 'mm_token_type_ids') else item[k]
                        assert torch.equal(value, expected), (index, k)
                    original_tensors = cpu_inputs
                # Explicitly validate flattened vision patch order against grids.
                sizes = original_tensors['image_grid_thw'].prod(-1).tolist()
                chunks = original_tensors['pixel_values'].split(sizes)
                assert torch.equal(cpu_inputs['pixel_values'], torch.cat([chunks[j] for j in order]))
                assert torch.equal(cpu_inputs['image_grid_thw'], original_tensors['image_grid_thw'][order])
                initial = {k: v.cuda() for k, v in cpu_inputs.items()}
                for method in ('base', 'adapter'):
                    inputs = dict(initial)
                    generated = []
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    first_probs = None
                    for step in range(128):
                        model.model.rope_deltas = None
                        logits = (model(**inputs, use_cache=False, logits_to_keep=1).logits
                            if method == 'base' else ref.qwen_embedding_adapter_logits(
                                model, adapter, inputs, logits_to_keep=1)[0])
                        if step == 0:
                            ids = [processor.tokenizer.encode(chr(65 + j), add_special_tokens=False) for j in range(n)]
                            assert all(len(x) == 1 for x in ids)
                            first_probs = logits[0, -1, [x[0] for x in ids]].float().softmax(-1).tolist()
                        token = int(logits[0, -1].argmax())
                        generated.append(token)
                        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+j) for j in range(n)]:
                            break
                        new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                        inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
                        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
                    scored = score_prediction(metric=ds.spec.metric, prediction_text=text,
                        answer=gold, choices=new_choices, question=q)
                    pred = scored['prediction']
                    chosen = ord(pred)-65 if pred in list('ABCD')[:n] else None
                    physical_slot = (chosen + option_shift) % n if chosen is not None else None
                    identity = order[physical_slot] if chosen is not None else None
                    if mode == 'original':
                        assert pred == previous[index, method]['prediction'], (index, method, 'Original did not reproduce')
                    out.write(json.dumps(dict(index=index, task=row['task'], mode=mode, method=method,
                        image_order=order, choices=new_choices, expected_gold=gold, question=q, text=text,
                        generated_tokens=len(generated), option_probs=first_probs,
                        selected_slot=physical_slot, selected_original_image=identity,
                        deepstack=False, **scored)) + '\n')
            for im in images:
                im.close()
            print('DONE', index, flush=True)


def summarize():
    from collections import Counter
    rows = [json.loads(line) for p in OUT.glob('rows_*.jsonl') for line in p.open()]
    assert len(rows) == len({(r['index'],r['method'],r['mode']) for r in rows}) == 332 * 8
    lookup = {(r['index'], r['method'], r['mode']): r for r in rows}
    result = []
    for task in sorted(TASKS):
        for method in ('base', 'adapter'):
            for mode in MODES:
                rs = [r for r in rows if (r['task'],r['method'],r['mode']) == (task,method,mode)]
                originals = [lookup[r['index'], method, 'original'] for r in rs]
                result.append(dict(task=task, method=method, mode=mode, n=len(rs),
                    accuracy=100*sum(r['score'] for r in rs)/len(rs),
                    predictions=dict(Counter(r['prediction'] for r in rs)),
                    physical_slots=dict(Counter(r['selected_slot'] for r in rs)),
                    same_image_pct=100*sum(r['selected_original_image'] is not None and
                        r['selected_original_image']==o['selected_original_image'] for r,o in zip(rs,originals))/len(rs)))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs, logs = [], []
    for shard in range(8):
        log = (OUT/f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.mmiu_order_binding_factorial',str(shard)],
            cwd=ROOT, env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs:
        log.close()
    assert not any(codes), codes
    summarize()


if __name__ == '__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
