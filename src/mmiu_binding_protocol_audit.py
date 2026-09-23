"""Controlled image-binding diagnostic; never overwrites official result rows."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALL_TASKS = os.environ.get('MMIU_BINDING_ALL_TASKS') == '1'
DEDUP = os.environ.get('MMIU_DEDUP_OPTIONS') == '1'
NATIVE_IDS = os.environ.get('MMIU_NATIVE_VISION_IDS') == '1'
assert not (DEDUP and ALL_TASKS)
assert not (DEDUP and NATIVE_IDS)
OUT = ROOT / ('artifacts/diagnostics/mmiu_native_vision_ids_all1000_20260914' if NATIVE_IDS else
    'artifacts/diagnostics/mmiu_dedup_options_20260914' if DEDUP else
    'artifacts/diagnostics/mmiu_binding_protocol_all1000_20260914' if ALL_TASKS
    else 'artifacts/diagnostics/mmiu_binding_protocol_audit_20260914')
MANIFEST = ROOT / 'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
TASKS = ({'visual_quality_assessment_q_bench+', 'visual_quality_assessment_ve_lol_l'} if DEDUP
    else {'forensic_detection_blink', 'visual_quality_assessment_ve_lol_l'})
LAYOUTS = ('plain', 'native_ids') if NATIVE_IDS else ('plain', 'dedup') if DEDUP else ('plain', 'before', 'after')


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import build_benchmark_prompt, score_prediction
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval().requires_grad_(False)
    from src.qwen_deepstack import disable_qwen_deepstack
    disable_qwen_deepstack(model)
    processor = AutoProcessor.from_pretrained(MODEL)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT, model.model.language_model,
        torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.model.language_model.register_forward_pre_hook(
        lambda m, a, k: (a, dict(k, deepstack_visual_embeds=None)), with_kwargs=True)
    def reject(*a, **k):
        raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process = reject
    ds = QwenBenchmarkDataset(str(MANIFEST), processor, 'mmiu', max_samples=1000,
        data_root=str(ROOT / 'data/benchmarks/mmiu'), prompt_layout='media_first_v1')
    indices = [i for i, row in enumerate(ds.rows) if NATIVE_IDS or ALL_TASKS or row['task'] in TASKS]
    path = OUT / f'rows_{shard}.jsonl'
    assert not path.exists(), 'Use a fresh output directory; do not mix runs'
    with torch.inference_mode(), path.open('w', buffering=1) as out:
        for index in indices[shard::8]:
            row = ds.rows[index]
            item = ds[index]
            question = build_benchmark_prompt(row, ds.spec)
            files = ds._image_paths(row)
            images = [Image.open(p).convert('RGB') for p in files]
            # Independent training input constructor, no answers supplied.
            train_inputs, _, _, _ = ref.prepare_qwen3vl_batch_inputs(processor,
                [dict(row, question=question)], ROOT / 'data/benchmarks/mmiu', torch.device('cpu'),
                include_answers=False)
            equal = {}
            for key in ('input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw'):
                expected = item[key].unsqueeze(0) if key in ('input_ids', 'attention_mask', 'mm_token_type_ids') else item[key]
                equal[key] = torch.equal(train_inputs[key], expected)
            assert all(equal.values()), (index, equal)
            for layout in LAYOUTS:
                variant_question = question
                if layout == 'dedup':
                    option_block = '\n'.join(f'{chr(65+n)}. {choice}' for n, choice in enumerate(row['choices']))
                    expected = row['question'] + '\n' + option_block + '\n' + ds.spec.answer_instruction
                    assert question == expected, (index, 'Unexpected prompt; cannot isolate duplicate removal')
                    variant_question = row['question'] + '\n' + ds.spec.answer_instruction
                content = []
                for n, image in enumerate(images, 1):
                    if layout == 'before':
                        content.append({'type': 'text', 'text': f'Image {n}:\n'})
                    content.append({'type': 'image', 'image': image})
                    if layout == 'after':
                        content.append({'type': 'text', 'text': f'\n[End of Image {n}]\n'})
                content.append({'type': 'text', 'text': variant_question})
                prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=False, add_generation_prompt=True, add_vision_id=layout == 'native_ids')
                if layout == 'native_ids':
                    assert all(f'Picture {n}: <|vision_start|>' in prompt for n in range(1, len(images) + 1))
                initial = processor(text=[prompt], images=images, return_tensors='pt', padding=True)
                if layout == 'plain':
                    assert all(torch.equal(initial[k], train_inputs[k]) for k in train_inputs)
                else:
                    assert torch.equal(initial['pixel_values'], train_inputs['pixel_values'])
                    assert torch.equal(initial['image_grid_thw'], train_inputs['image_grid_thw'])
                initial = {k: v.cuda() for k, v in initial.items()}
                for method in ('base', 'adapter'):
                    inputs = dict(initial)
                    generated = []
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    for step in range(128):
                        model.model.rope_deltas = None
                        logits = (model(**inputs, use_cache=False, logits_to_keep=1).logits if method == 'base'
                            else ref.qwen_embedding_adapter_logits(model, adapter, inputs, logits_to_keep=1)[0])
                        token = int(logits[0, -1].argmax())
                        generated.append(token)
                        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65 + n) for n in range(len(row['choices']))]:
                            break
                        new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                        inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
                        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
                    scored = score_prediction(metric=ds.spec.metric, prediction_text=text,
                        answer=row['answer'], choices=row['choices'], question=question)
                    out.write(json.dumps(dict(index=index, task=row['task'], layout=layout,
                        method=method, text=text, generated_tokens=len(generated),
                        train_eval_input_equal=equal, deepstack=False, **scored)) + '\n')
            for image in images:
                image.close()
            print('DONE', index, flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = []
    logs = []
    for shard in range(8):
        log = (OUT / f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable, '-m', 'src.mmiu_binding_protocol_audit', str(shard)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs:
        log.close()
    assert not any(codes), codes
    rows = [json.loads(line) for p in OUT.glob('rows_*.jsonl') for line in p.open()]
    expected = 1000 if ALL_TASKS or NATIVE_IDS else 400 if DEDUP else 332
    assert len(rows) == len({(r['index'], r['layout'], r['method']) for r in rows}) == expected * len(LAYOUTS) * 2
    summary = []
    for task in sorted({r['task'] for r in rows}):
        for method in ('base', 'adapter'):
            for layout in LAYOUTS:
                rs = [r for r in rows if (r['task'], r['method'], r['layout']) == (task, method, layout)]
                summary.append(dict(task=task, method=method, layout=layout, samples=len(rs),
                    accuracy=100 * sum(r['score'] for r in rs) / len(rs)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run() if len(sys.argv) == 1 else worker(int(sys.argv[1]))
