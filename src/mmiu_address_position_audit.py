"""Isolate image-label semantics, added text positions, and RoPE offsets.

Diagnostic interventions only: constant labels and gapped positions are not
benchmark protocols. No training, vision changes, or replacement of old scores.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/mmiu_address_position_audit_20260914'
MODES = ('plain', 'native_ids', 'constant_ids', 'position_only')
TASKS = {'forensic_detection_blink', 'visual_quality_assessment_ve_lol_l'}


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
    processor = AutoProcessor.from_pretrained(MODEL)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,
        model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.model.language_model.register_forward_pre_hook(
        lambda m, a, k: (a, dict(k, deepstack_visual_embeds=None)), with_kwargs=True)
    ds = QwenBenchmarkDataset(str(MANIFEST), processor, 'mmiu', max_samples=1000,
        data_root=str(ROOT / 'data/benchmarks/mmiu'), prompt_layout='media_first_v1')
    indices = [i for i, r in enumerate(ds.rows) if r['task'] in TASKS]
    zero_id = processor.tokenizer.encode('0', add_special_tokens=False)
    assert len(zero_id) == 1
    records = OUT / f'rows_{shard}.jsonl'
    assert not records.exists()

    def predict(inputs, hidden, pos, method):
        inputs = dict(inputs)
        generated = []
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        for step in range(128):
            model.model.rope_deltas = None
            logits = (model(**inputs, position_ids=pos, use_cache=False, logits_to_keep=1).logits
                if method == 'base' else ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                    initial_hidden=hidden, position_ids=pos, logits_to_keep=1)[0])
            token = int(logits[0, -1].argmax())
            generated.append(token)
            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            if token in eos or text in 'ABCD' and len(text) == 1:
                return text, len(generated)
            new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
            hidden = torch.cat([hidden, model.model.get_input_embeddings()(new)], 1)
            pos = torch.cat([pos, pos[:, :, -1:] + 1], 2)
        return text, len(generated)

    with torch.inference_mode(), records.open('w', buffering=1) as out:
        for ordinal, index in enumerate(indices[shard::8]):
            row = ds.rows[index]
            item = ds[index]
            plain = {k: item[k].unsqueeze(0).cuda() for k in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
            plain.update({k: item[k].cuda() for k in ('pixel_values', 'image_grid_thw')})
            question = build_benchmark_prompt(row, ds.spec)
            images = [Image.open(p).convert('RGB') for p in ds._image_paths(row)]
            content = [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': question}]
            prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                tokenize=False, add_generation_prompt=True, add_vision_id=True)
            numbered = {k: v.cuda() for k, v in processor(text=[prompt], images=images,
                return_tensors='pt', padding=True).items()}
            for im in images:
                im.close()
            assert torch.equal(plain['pixel_values'], numbered['pixel_values'])
            assert torch.equal(plain['image_grid_thw'], numbered['image_grid_thw'])
            # Remove exactly the template-inserted label tokens. Remaining IDs
            # must be the original prompt byte-for-byte at tokenizer level.
            starts = numbered['input_ids'][0].eq(model.config.vision_start_token_id).nonzero().flatten().tolist()
            assert len(starts) == len(images)
            keep = torch.ones(numbered['input_ids'].shape[1], device='cuda', dtype=torch.bool)
            constant = dict(numbered)
            constant['input_ids'] = numbered['input_ids'].clone()
            for n, start in enumerate(starts, 1):
                label = processor.tokenizer.encode(f'Picture {n}: ', add_special_tokens=False)
                begin = start - len(label)
                assert numbered['input_ids'][0, begin:start].tolist() == label
                digit = processor.tokenizer.encode(str(n), add_special_tokens=False)
                assert len(digit) == 1 and label.count(digit[0]) == 1
                constant['input_ids'][0, begin + label.index(digit[0])] = zero_id[0]
                keep[begin:start] = False
            assert torch.equal(numbered['input_ids'][:, keep], plain['input_ids'])
            assert torch.equal(numbered['mm_token_type_ids'][:, keep], plain['mm_token_type_ids'])
            model.model.rope_deltas = None
            hp, pp = ref.build_qwen_initial_context(model, plain)
            model.model.rope_deltas = None
            hn, pn = ref.build_qwen_initial_context(model, numbered)
            assert torch.equal(hp, hn[:, keep])
            hc = hn.clone()
            text = constant['mm_token_type_ids'][0].eq(0)
            hc[:, text] = model.model.get_input_embeddings()(constant['input_ids'][:, text])
            assert torch.equal(ref.qwen_position_ids(model, constant), pn)
            if ordinal == 0:
                # Dispatch-only control: same preprocessed patch tensor and
                # one-frame grids, not a claim about real temporal preprocessing.
                video = dict(plain)
                video['pixel_values_videos'] = video.pop('pixel_values')
                video['video_grid_thw'] = video.pop('image_grid_thw')
                video['input_ids'] = video['input_ids'].clone()
                mask = video['mm_token_type_ids'].ne(0)
                video['input_ids'][mask] = model.config.video_token_id
                video['mm_token_type_ids'] = video['mm_token_type_ids'].clone()
                video['mm_token_type_ids'][mask] = 2
                model.model.rope_deltas = None
                hv, pv = ref.build_qwen_initial_context(model, video)
                assert torch.equal(hp, hv) and torch.equal(pp, pv)
                lp = ref.qwen_embedding_adapter_logits(model, adapter, plain, initial_hidden=hp,
                    position_ids=pp, logits_to_keep=1)[0]
                lv = ref.qwen_embedding_adapter_logits(model, adapter, video, initial_hidden=hv,
                    position_ids=pv, logits_to_keep=1)[0]
                assert torch.equal(lp, lv)
                print('MODALITY_DISPATCH_EXACT', index, flush=True)
            variants = {'plain': (plain, hp, pp), 'native_ids': (numbered, hn, pn),
                'constant_ids': (constant, hc, pn), 'position_only': (plain, hp, pn[:, :, keep])}
            for mode, (inputs, hidden, pos) in variants.items():
                for method in ('base', 'adapter'):
                    prediction, tokens = predict(inputs, hidden, pos, method)
                    scored = score_prediction(metric=ds.spec.metric, prediction_text=prediction,
                        answer=row['answer'], choices=row['choices'], question=question)
                    out.write(json.dumps(dict(index=index, task=row['task'], mode=mode, method=method,
                        text=prediction, generated_tokens=tokens, deepstack=False, **scored)) + '\n')
            print('DONE', index, flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs, logs = [], []
    for shard in range(8):
        log = (OUT / f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable, '-m', 'src.mmiu_address_position_audit', str(shard)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs:
        log.close()
    assert not any(codes), codes
    rows = [json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows) == len({(r['index'], r['mode'], r['method']) for r in rows}) == 332 * 8
    summary = []
    for task in sorted(TASKS):
        for method in ('base', 'adapter'):
            for mode in MODES:
                rs = [r for r in rows if (r['task'], r['method'], r['mode']) == (task, method, mode)]
                summary.append(dict(task=task, method=method, mode=mode, n=len(rs),
                    accuracy=100 * sum(r['score'] for r in rs) / len(rs)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run() if len(sys.argv) == 1 else worker(int(sys.argv[1]))
