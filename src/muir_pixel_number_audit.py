"""Paired pixel-number binding diagnostic; never changes official evaluation.

All 132 image-choice matching cases, blank/numbered top margin, original/rotated
media. The original pixels are preserved; the two arms have identical geometry.
Numbers denote display positions, never the gold answer. No training.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT = ROOT / 'artifacts/diagnostics/muir_pixel_numbers_20260914'
METHODS = ('base', 'embedding_adapter')
STYLES = ('blank', 'numbered')
ORDERS = ('original', 'rotate_media_fixed_choices')


def bordered(image, number, numbered):
    from PIL import Image, ImageDraw, ImageFont
    # Same canvas/font geometry in both arms. Never resize/crop the source.
    size = max(16, round(min(image.size) * .07))
    font = ImageFont.load_default(size=size)
    label = f'Image {number}'
    box = font.getbbox(label)
    while box[2] - box[0] > image.width - 8 and size > 2:
        size -= 1
        font = ImageFont.load_default(size=size)
        box = font.getbbox(label)
    height = max(24, size * 2)
    canvas = Image.new('RGB', (image.width, image.height + height), 'white')
    canvas.paste(image, (0, height))
    if numbered:
        x = (image.width - (box[2]-box[0])) // 2 - box[0]
        y = (height - (box[3]-box[1])) // 2 - box[1]
        ImageDraw.Draw(canvas).text((x, y), label, font=font, fill='black')
    assert canvas.crop((0, height, canvas.width, canvas.height)).tobytes() == image.tobytes()
    return canvas, height


def prepare(ds, row, style):
    from PIL import Image
    from src.benchmarks import build_benchmark_prompt
    question = build_benchmark_prompt(row, ds.spec, ds.answer_instruction)
    images = []
    geometry = []
    for number, path in enumerate(ds._image_paths(row), 1):
        with Image.open(path) as raw:
            original = raw.convert('RGB')
            image, height = bordered(original, number, style == 'numbered')
            geometry.append(dict(original_size=list(original.size), border_height=height))
            original.close()
        images.append(image)
    try:
        content = ds._qwen_message_content(row, question, images, [])
        text = ds.processor.apply_chat_template([dict(role='user', content=content)],
                                                tokenize=False, add_generation_prompt=True)
        inputs = ds.processor(text=[text], images=images, return_tensors='pt', padding=True)
    finally:
        for image in images:
            image.close()
    item = dict(inputs)
    for name in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
        item[name] = item[name].squeeze(0)
    return item, text, geometry


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL, ADAPTER_CHECKPOINTS
    from src.muir_binding_diagnostic import permute_row
    from src.audit_mmiu_random_results import extract_answer
    from src.muir_random_matching_permutation_audit import image_position
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    model.eval().requires_grad_(False)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        ADAPTER_CHECKPOINTS['embedding_adapter'], model.model.language_model,
        torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.adapter_start_layer == 0 and adapter.active_adapter_layers == 0
    assert adapter.mode == 'embedding_adapter' and not adapter.native_ffn_carriers
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k: (a,dict(k,deepstack_visual_embeds=None)), with_kwargs=True)
    def reject(*a, **kw):
        raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process = reject
    vision_forward = model.model.visual.forward
    vision_cache = {}
    def cached_vision(*a, **k):
        if not vision_cache:
            value = vision_forward(*a, **k)
            vision_cache['value'] = (type(value), dict(value))
        cls, fields = vision_cache['value']
        return cls(**fields)
    model.model.visual.forward = cached_vision
    ds = QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'), processor, 'muirbench',
        data_root=str(ROOT/'data/benchmarks/muirbench'), max_samples=1000, prompt_layout='media_first_v1')
    indices = [i for i,r in enumerate(ds.rows) if r['task']=='Image-Text Matching' and
               any(re.fullmatch(r'<\|image_\d+\|>', c.strip()) for c in r['choices'])]
    assert len(indices) == 132
    with torch.inference_mode(), (OUT/f'rows_{shard}.jsonl').open('w', buffering=1) as output:
        for index in indices[shard::8]:
            original = ds.rows[index]
            original_predictions = {}
            for order in ORDERS:
                row = original if order == 'original' else permute_row(original, order)
                control = None
                for style in STYLES:
                    item, prompt, geometry = prepare(ds, row, style)
                    if control is None:
                        control = (prompt, {k:item[k].clone() for k in
                            ('input_ids','attention_mask','mm_token_type_ids','image_grid_thw')}, geometry)
                    else:
                        assert prompt == control[0] and geometry == control[2]
                        assert all(torch.equal(item[k], v) for k,v in control[1].items())
                    vision_cache.clear()  # never reuse features across different pixels/order
                    inputs0 = _qwen_inputs_from_item(item, torch.device('cuda'))
                    initial, positions = ref.build_qwen_initial_context(model, inputs0)
                    for method in METHODS:
                        inputs = dict(inputs0)
                        h, pos, generated = initial, positions, []
                        eos = model.generation_config.eos_token_id
                        eos = eos if isinstance(eos, list) else [eos]
                        for step in range(8):
                            model.model.rope_deltas = None
                            if method == 'base':
                                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
                            else:
                                logits = ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                                    initial_hidden=h, position_ids=pos, logits_to_keep=1)[0]
                            token = int(logits[0,-1].argmax())
                            generated.append(token)
                            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                            if token in eos or text in [chr(65+j) for j in range(len(row['choices']))]:
                                break
                            new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                            inputs['input_ids'] = torch.cat([inputs['input_ids'],new],1)
                            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                            h = torch.cat([h,model.model.get_input_embeddings()(new)],1)
                            pos = torch.cat([pos,pos[:,:,-1:]+1],2)
                        prediction = extract_answer(text, row['choices'])
                        if order == 'original':
                            original_predictions[style,method] = prediction
                        prior = original_predictions[style,method]
                        before = image_position(prior, original['choices'])
                        after = image_position(prediction, row['choices'])
                        output.write(json.dumps(dict(index=index, source_index=original['index'],
                            method=method, style=style, order=order, text=text, prediction=prediction,
                            gold=row['answer'], score=int(prediction==row['answer']),
                            same_content_choice=prediction is not None and prediction==prior,
                            same_display_position=(before==after) if before is not None and after is not None else None,
                            choices=row['choices'], images=row['images'], geometry=geometry,
                            image_grid_thw=item['image_grid_thw'].tolist(),
                            generated_tokens=len(generated), deepstack=False))+'\n')
            print('DONE',shard,index,flush=True)


def summarize(codes):
    rows = [json.loads(line) for p in OUT.glob('rows_*.jsonl') for line in p.open()]
    results = []
    for method in METHODS:
        for style in STYLES:
            for order in ORDERS:
                group = [r for r in rows if (r['method'],r['style'],r['order'])==(method,style,order)]
                positions = [r for r in group if r['same_display_position'] is not None]
                if group:
                    results.append(dict(method=method, style=style, order=order,n=len(group),
                        accuracy=100*sum(r['score'] for r in group)/len(group),
                        same_content_choice=100*sum(r['same_content_choice'] for r in group)/len(group),
                        both_choose_images=len(positions),
                        same_display_position=100*sum(r['same_display_position'] for r in positions)/len(positions) if positions else None))
    payload = dict(expected=1056, completed=len(rows), exit_codes=codes, results=results)
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)
    assert not any(codes) and len(rows)==len({(r['index'],r['method'],r['style'],r['order']) for r in rows})==1056


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    logs, jobs = [], []
    for shard in range(8):
        log = (OUT/f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_pixel_number_audit',str(shard)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for f in logs:
        f.close()
    summarize(codes)


if __name__ == '__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
