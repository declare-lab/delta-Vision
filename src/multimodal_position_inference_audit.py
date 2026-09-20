"""Position-only inference regression on actual benchmark inputs; no interventions.

Independently reconstruct M-RoPE from token runs/grids; compare packed masks to
physical causal order, and cached decode to full recomputation on common tokens.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/diagnostics/multimodal_position_inference_20260914'
BENCHES = ('muirbench', 'mmiu', 'videomme', 'mvbench')


def independent_positions(inputs, config):
    import torch
    ids = inputs['input_ids'][0].tolist()
    types = inputs['mm_token_type_ids'][0].tolist()
    valid = inputs['attention_mask'][0].tolist()
    merge = config.vision_config.spatial_merge_size
    images = iter(inputs.get('image_grid_thw', torch.empty(0, 3)).tolist())
    videos = []
    for t, h, w in inputs.get('video_grid_thw', torch.empty(0, 3)).tolist():
        videos.extend([(1, h, w)] * int(t))
    videos = iter(videos)
    result = torch.zeros_like(inputs['input_ids']).unsqueeze(0).expand(3, -1, -1).clone()
    cursor = 0
    offset = 0
    spans = []
    while cursor < len(ids):
        if not valid[cursor]:
            cursor += 1
            continue
        token = ids[cursor]
        modality = 1 if token == config.image_token_id else 2 if token == config.video_token_id else 0
        assert modality == types[cursor], (cursor, token, modality, types[cursor])
        if not modality:
            result[:, 0, cursor] = offset
            cursor += 1
            offset += 1
            continue
        t, h, w = [int(x) for x in next(images if modality == 1 else videos)]
        assert t == 1, 'Image or timestamp-separated video frame must have one temporal grid'
        h, w = h // merge, w // merge
        count = h * w
        assert ids[cursor:cursor+count] == [token] * count
        assert types[cursor:cursor+count] == [modality] * count
        assert all(valid[cursor:cursor+count])
        flat = torch.arange(count, device=result.device)
        result[0, 0, cursor:cursor+count] = offset
        result[1, 0, cursor:cursor+count] = offset + flat // w
        result[2, 0, cursor:cursor+count] = offset + flat % w
        spans.append(dict(modality=modality, start=cursor, end=cursor+count,
                          grid=[h, w], rope_start=offset, rope_end=offset+max(h,w)-1))
        offset += max(h, w)
        cursor += count
    assert next(images, None) is None and next(videos, None) is None
    return result, spans


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL, CHECKPOINT
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    benchmark = BENCHES[shard // 2]
    manifest = ROOT/f'data/benchmarks/{benchmark}/test.jsonl'
    if benchmark == 'mmiu':
        manifest = ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
    dataset = QwenBenchmarkDataset(str(manifest), processor, benchmark, max_samples=1000,
        data_root=str(ROOT/f'data/benchmarks/{benchmark}'), prompt_layout='media_first_v1')
    # Selection uses IDs/media counts, never answers, errors, or scores.
    indices = [0, 62, 125, 187, 250, 312, 375, 437] if shard % 2 == 0 else [500, 562, 625, 687, 750, 812, 875, 999]
    if benchmark in ('muirbench', 'mmiu') and shard % 2:
        indices[-2] = max(range(1000), key=lambda i: len(dataset.rows[i].get('images', [])))
    indices = list(dict.fromkeys(indices))
    class Captured(Exception):
        pass
    def compare(a, b):
        a, b = a[0, -1].float(), b[0, -1].float()
        return dict(kl=float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum()),
                    same_argmax=bool(a.argmax() == b.argmax()))
    with torch.inference_mode(), (OUTPUT/f'rows_{shard}.jsonl').open('w', buffering=1) as out:
        for number, index in enumerate(indices):
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, torch.device('cuda'))
            captured = {}
            def capture(module, args, kwargs):
                captured['hidden'] = kwargs['inputs_embeds']
                captured['positions'] = kwargs['position_ids']
                raise Captured()
            handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                model.model.rope_deltas = None
                model(**inputs, use_cache=False)
            except Captured:
                pass
            finally:
                handle.remove()
            assert captured
            hidden, pos = ref.build_qwen_initial_context(model, inputs)
            expected, spans = independent_positions(inputs, model.config)
            assert torch.equal(hidden, captured['hidden']), (benchmark,index,'native embedding')
            assert torch.equal(pos, captured['positions']), (benchmark,index,'native positions')
            assert torch.equal(pos, expected), (benchmark,index,'independent coordinates')
            del captured
            prepared = ref.prepare_qwen_embedding_adapter_inputs(model, adapter,
                inputs['input_ids'], inputs['attention_mask'], inputs['mm_token_type_ids'], hidden, pos)
            vp = prepared['image_positions'][0]
            tp = prepared['text_positions'][0]
            all_keys = torch.cat([vp, tp])
            assert torch.equal(all_keys.sort().values, torch.arange(hidden.shape[1], device=hidden.device))
            assert torch.equal(prepared['prefix_attention_mask'][0,0], all_keys[None,:] <= tp[:,None])
            assert torch.equal(prepared['visual_position_ids'], expected[:,:,vp])
            assert torch.equal(prepared['text_position_ids'], expected[:,:,tp])
            full_rope = model.model.language_model.rotary_emb(hidden, expected)
            for channel in range(2):
                assert torch.equal(prepared['visual_position_embeddings'][channel], full_rope[channel][:,vp])
                assert torch.equal(prepared['text_position_embeddings'][channel], full_rope[channel][:,tp])
            # Exercise both padding sides with real grids, not synthetic positions.
            for side in ('left', 'right'):
                pads = torch.zeros((1,17), device=hidden.device, dtype=inputs['input_ids'].dtype)
                padded = dict(inputs)
                for key in ('input_ids','attention_mask','mm_token_type_ids'):
                    padded[key] = torch.cat([pads, inputs[key]],1) if side == 'left' else torch.cat([inputs[key], pads],1)
                padpos, _ = independent_positions(padded, model.config)
                native_padpos = ref.qwen_position_ids(model, padded)
                assert torch.equal(padpos, native_padpos)
                pad_hidden = hidden.new_zeros((1,17,hidden.shape[-1]))
                ph = torch.cat([pad_hidden, hidden],1) if side == 'left' else torch.cat([hidden,pad_hidden],1)
                pp = ref.prepare_qwen_embedding_adapter_inputs(model, adapter,
                    padded['input_ids'],padded['attention_mask'],padded['mm_token_type_ids'],ph,padpos)
                assert torch.equal(pp['prefix_attention_mask'],prepared['prefix_attention_mask'])
                assert torch.equal(pp['text_position_ids'],prepared['text_position_ids'])
                assert torch.equal(pp['visual_position_ids'],prepared['visual_position_ids'])
            record = dict(benchmark=benchmark,index=index,sequence_length=hidden.shape[1],
                visual_tokens=vp.numel(),spans=spans,native_input_exact=True,independent_positions_exact=True,
                physical_mask_exact=True,split_rope_exact=True,padding_exact=True,steps=[])
            del prepared, pp, full_rope, ph
            if number == 0:
                logits, _, cache = ref.qwen_embedding_adapter_prefill_cache(model,adapter,
                    inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,pos)
                caches = {name:copy.deepcopy(cache) for name in ('dynamic','hf_static','shape_exact')}
                del cache
                ref.qwen_embedding_adapter_attach_hf_static_cache(model,caches['hf_static'],max_new_tokens=8)
                methods = dict(dynamic=ref.qwen_embedding_adapter_decode_step,
                    hf_static=ref.qwen_embedding_adapter_decode_step_hf_static,
                    shape_exact=ref.qwen_embedding_adapter_decode_step_shape_exact)
                for step in range(8):
                    token = logits[:,-1].argmax(-1).view(1,1)
                    physical_next = inputs['input_ids'].shape[1]
                    for cache in caches.values():
                        assert int(cache['next_text_positions'][0,0]) == physical_next
                        assert torch.equal(cache['next_position_ids'],pos[:,:,-1:]+1)
                    for key in ('input_ids','attention_mask','mm_token_type_ids'):
                        added = token if key == 'input_ids' else torch.ones_like(token) if key == 'attention_mask' else torch.zeros_like(token)
                        inputs[key] = torch.cat([inputs[key],added],1)
                    hidden = torch.cat([hidden,model.model.get_input_embeddings()(token)],1)
                    pos = torch.cat([pos,pos[:,:,-1:]+1],2)
                    independent, _ = independent_positions(inputs,model.config)
                    assert torch.equal(pos,independent)
                    assert torch.equal(pos,ref.qwen_position_ids(model,inputs))
                    logits = ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                        initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                    scores = {}
                    for name,cache in caches.items():
                        mask = ref._qwen_decode_attention_mask(cache,cache['next_position_ids'])
                        assert bool(mask.all()), (benchmark,index,name,step,'masked existing key')
                        candidate,caches[name] = methods[name](model,adapter,token,cache)
                        scores[name] = compare(logits,candidate)
                    record['steps'].append(scores)
                del caches, logits, candidate
            out.write(json.dumps(record)+'\n')
            print('CHECKED',benchmark,index,'tokens',record['sequence_length'],'decode_steps',len(record['steps']),flush=True)
            del hidden, pos, inputs
    print('COMPLETE',shard,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for shard in range(8):
        with (OUTPUT/f'worker{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-m','src.multimodal_position_inference_audit',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
                stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    assert not any(codes), codes
    rows=[json.loads(line) for p in OUTPUT.glob('rows_*.jsonl') for line in p.open()]
    summary=dict(samples=len(rows),benchmarks={b:sum(r['benchmark']==b for r in rows) for b in BENCHES},
        max_visual_tokens=max(r['visual_tokens'] for r in rows),checks='all exact',decode={})
    for name in ('dynamic','hf_static','shape_exact'):
        ss=[s[name] for r in rows for s in r['steps']]
        summary['decode'][name]=dict(steps=len(ss),same_argmax=sum(s['same_argmax'] for s in ss),max_kl=max(s['kl'] for s in ss))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
