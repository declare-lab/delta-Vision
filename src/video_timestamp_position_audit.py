"""Diagnostic deletion of timestamp text, separating RoPE from text-token effects."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/video_timestamp_position_audit_20260914'
MODES = ('native', 'remove_compact_positions', 'remove_keep_positions')


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from src.mmiu_binding_protocol_audit import MODEL, CHECKPOINT
    torch.set_num_threads(4)
    torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
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
    manifests = json.loads((ROOT / 'configs/eval_mmiu_video_matched_20260914.json').read_text())
    record_path = OUT / f'rows_{shard}.jsonl'
    assert not record_path.exists()
    with torch.inference_mode(), record_path.open('w', buffering=1) as out:
        for benchmark in ('videomme', 'mvbench'):
            ds = QwenBenchmarkDataset(manifests[benchmark], processor, benchmark,
                data_root=str(ROOT / 'data/benchmarks' / benchmark), prompt_layout='media_first_v1',
                cache_dir=ROOT / 'artifacts/diagnostics/mmiu_video_all_methods_matched_20260914/processed' / benchmark)
            # Deterministic indices spread across the entire fixed manifest;
            # never selected by correctness or response to an intervention.
            indices = [round(i * (len(ds) - 1) / 63) for i in range(64)]
            assert len(set(indices)) == 64
            for index in indices[shard::8]:
                item = ds[index]
                original = {k: item[k].unsqueeze(0).cuda() for k in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
                original.update({k: item[k].cuda() for k in ('pixel_values_videos', 'video_grid_thw')})
                ids = original['input_ids'][0].tolist()
                types = original['mm_token_type_ids'][0]
                last_visual = int(types.ne(0).nonzero()[-1])
                rendered = processor.tokenizer.decode(ids[:last_visual + 1])
                timestamps = re.findall(r'<\d+\.\d+ seconds>', rendered)
                expected = int(original['video_grid_thw'][:, 0].sum())
                assert len(timestamps) == expected, (benchmark, index, timestamps, expected)
                keep = torch.ones(len(ids), dtype=torch.bool, device='cuda')
                cursor = 0
                removed_positions = []
                for timestamp in timestamps:
                    needle = processor.tokenizer.encode(timestamp, add_special_tokens=False)
                    starts = [i for i in range(cursor, last_visual - len(needle) + 1) if ids[i:i+len(needle)] == needle]
                    assert starts, (index, timestamp, needle)
                    start = starts[0]
                    assert bool(types[start:start+len(needle)].eq(0).all())
                    keep[start:start+len(needle)] = False
                    removed_positions.extend(range(start, start+len(needle)))
                    cursor = start + len(needle)
                compact = dict(original)
                for k in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
                    compact[k] = original[k][:, keep]
                model.model.rope_deltas = None
                hidden, original_pos = ref.build_qwen_initial_context(model, original)
                compact_pos = ref.qwen_position_ids(model, compact)
                kept_pos = original_pos[:, :, keep]
                assert torch.equal(compact['pixel_values_videos'], original['pixel_values_videos'])
                assert torch.equal(compact['video_grid_thw'], original['video_grid_thw'])
                assert int(compact['mm_token_type_ids'].ne(0).sum()) == int(types.ne(0).sum())
                variants = {'native': (original, hidden, original_pos),
                    'remove_compact_positions': (compact, hidden[:, keep], compact_pos),
                    'remove_keep_positions': (compact, hidden[:, keep], kept_pos)}
                for mode, (initial, h0, p0) in variants.items():
                    for method in ('base', 'adapter'):
                        inputs, h, pos = dict(initial), h0, p0
                        generated = []
                        eos = model.generation_config.eos_token_id
                        eos = eos if isinstance(eos, list) else [eos]
                        for step in range(128):
                            model.model.rope_deltas = None
                            logits = (model(**inputs, position_ids=pos, use_cache=False, logits_to_keep=1).logits
                                if method == 'base' else ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                                    initial_hidden=h, position_ids=pos, logits_to_keep=1)[0])
                            token = int(logits[0, -1].argmax())
                            generated.append(token)
                            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                            if token in eos or text in [chr(65+n) for n in range(len(item['choices']))]:
                                break
                            new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                            inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
                            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
                            h = torch.cat([h, model.model.get_input_embeddings()(new)], 1)
                            pos = torch.cat([pos, pos[:, :, -1:] + 1], 2)
                        scored = score_prediction(metric=ds.spec.metric, prediction_text=text,
                            answer=item['answer'], choices=item['choices'], question=item['row']['question'])
                        out.write(json.dumps(dict(benchmark=benchmark, index=index, source_index=item['index'],
                            method=method, mode=mode, text=text, generated_tokens=len(generated),
                            removed_timestamp_tokens=len(removed_positions), timestamps=timestamps,
                            deepstack=False, **scored)) + '\n')
                print('DONE', benchmark, index, flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs, logs = [], []
    for shard in range(8):
        log = (OUT / f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable, '-m', 'src.video_timestamp_position_audit', str(shard)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs:
        log.close()
    assert not any(codes), codes
    rows = [json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows) == len({(r['benchmark'], r['index'], r['mode'], r['method']) for r in rows}) == 128 * 6
    summary = []
    for b in ('videomme', 'mvbench'):
        for method in ('base', 'adapter'):
            for mode in MODES:
                rs = [r for r in rows if (r['benchmark'], r['method'], r['mode']) == (b, method, mode)]
                summary.append(dict(benchmark=b, method=method, mode=mode, n=len(rs),
                    accuracy=100 * sum(r['score'] for r in rs) / len(rs)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run() if len(sys.argv) == 1 else worker(int(sys.argv[1]))
