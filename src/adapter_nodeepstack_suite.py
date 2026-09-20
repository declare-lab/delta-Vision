"""Fresh matched multi-image/video evaluation of four adapters and native base.

Media-first prompt layout, DeepStack injection forbidden, no training. Resume
only exact, fingerprinted records. The native base is actually evaluated again.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from src.multimodal_eval_inputs import benchmark_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'vision-kv-inject-attention-sink/src'
EXPERIMENT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911'
OUTPUT = ROOT / 'artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913'
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
METHODS = ('base', 'static_kl', 'recurrent_kl', 'sft', 'opd')
NAMES = ('Base model', 'Embedding Adapter + KL', 'Recurrent Adapter + KL', 'Embedding Adapter + SFT', 'Embedding Adapter + OPD')
BENCHES = ('mmiu', 'videomme', 'mvbench', 'muirbench')
LAYOUT = 'media_first_v1'
MAX_NEW_TOKENS = 128


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def dump(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def checkpoints():
    return {m: EXPERIMENT / m / 'checkpoints' / (
        'qwen_recurrent_embedding_adapter_step2000.pt' if m == 'recurrent_kl' else 'qwen_embedding_adapter_step2000.pt')
        for m in METHODS if m != 'base'}


def worker(shard, limit):
    import src
    src.__path__.insert(0, str(REFERENCE))
    import torch
    import types
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
    plan = json.loads((OUTPUT / 'plan.json').read_text())
    assert (plan['prompt_layout'],plan['max_new_tokens']) == (LAYOUT,MAX_NEW_TOKENS)
    assert sha(__file__) == plan['source_sha256']
    assert sha(REFERENCE / 'data.py') == plan['data_source_sha256']
    assert sha(REFERENCE / 'model.py') == plan['model_source_sha256']
    assert sha(ROOT/'src/multimodal_eval_inputs.py') == plan['input_source_sha256']
    assert all(sha(benchmark_manifest(b))==plan['datasets'][b] for b in BENCHES)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    model.eval().requires_grad_(False)
    adapters = {}
    for method, path in checkpoints().items():
        assert sha(path) == plan['checkpoints'][method]['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['global_step'] == 2000
        del saved
        adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(path, model.model.language_model,
                                                                 torch.device('cuda'), torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected']
        assert adapter.adapter_start_layer == 0 and adapter.active_adapter_layers == 0
        assert adapter.native_prefix_memory == 'legacy' and not adapter.native_ffn_carriers
        assert adapter.mode == ('recurrent_embedding_adapter' if method == 'recurrent_kl' else 'embedding_adapter')
        adapter.eval().requires_grad_(False)
        adapters[method] = adapter
    calls = [0]
    def disable(module, positional, keyword):
        calls[0] += 1
        return positional, dict(keyword, deepstack_visual_embeds=None)
    def reject(*args, **kwargs):
        raise AssertionError('DeepStack injection executed')
    model.model.language_model.register_forward_pre_hook(disable, with_kwargs=True)
    model.model.language_model._deepstack_process = reject
    # Only one media kind per request in these four datasets. Native feature
    # extraction mutates its ModelOutput, so return a fresh container each call.
    vision_cache = {}
    original_vision = model.model.visual.forward
    def cached_vision(*args, **kwargs):
        pixels = args[0] if args else kwargs['hidden_states']
        grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
        key = (tuple(pixels.shape), tuple(grid.flatten().tolist()))
        if key not in vision_cache:
            result = original_vision(*args, **kwargs)
            assert hasattr(result, 'pooler_output') and torch.is_tensor(result.pooler_output)
            vision_cache[key] = (type(result), dict(result))
        cls, fields = vision_cache[key]
        return cls(**fields)
    model.model.visual.forward = cached_vision
    path = OUTPUT / f'rows_{shard}.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            assert record['deepstack_enabled'] is False and record['prompt_layout'] == LAYOUT
            assert record['plan_sha256'] == sha(OUTPUT / 'plan.json')
            done.add((record['benchmark'], record['index'], record['method']))
    plan_hash = sha(OUTPUT / 'plan.json')
    with path.open('a', buffering=1) as out, torch.inference_mode():
        for benchmark in BENCHES:
            dataset = QwenBenchmarkDataset(str(benchmark_manifest(benchmark)),
                processor, benchmark, max_samples=limit, prompt_layout=LAYOUT,
                data_root=str(ROOT/f'data/benchmarks/{benchmark}'))
            for index in range(shard, len(dataset), 8):
                if all((benchmark, index, m) in done for m in METHODS):
                    continue
                item = dataset[index]
                inputs0 = _qwen_inputs_from_item(item, torch.device('cuda'))
                assert not ('pixel_values' in inputs0 and 'pixel_values_videos' in inputs0)
                assert inputs0['attention_mask'].bool().all()
                vision_cache.clear()
                initial, positions = ref.build_qwen_initial_context(model, inputs0)
                for method in METHODS:
                    if (benchmark, index, method) in done:
                        continue
                    started = time.time()
                    inputs = dict(inputs0)
                    hidden, pos = initial, positions
                    generated = []
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    choices = item.get('choices') or []
                    adapter = adapters.get(method)
                    original_memories = None
                    if adapter is not None:
                        # Fixed full-adapter memory depends only on the input
                        # images, never on generated answer tokens.
                        visual = inputs['mm_token_type_ids'][0].ne(0)
                        memory = adapter.all_visual_memories_batched(initial[:, visual])
                        original_memories = adapter.all_visual_memories_batched
                        adapter.all_visual_memories_batched = types.MethodType(lambda self, *a, _m=memory, **kw: _m, adapter)
                    try:
                        for step in range(MAX_NEW_TOKENS):
                            model.model.rope_deltas = None
                            if method == 'base':
                                before = calls[0]
                                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
                                assert calls[0] == before + 1
                            else:
                                logits = ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                                    initial_hidden=hidden, position_ids=pos, logits_to_keep=1)[0]
                            token = int(logits[0, -1].argmax())
                            generated.append(token)
                            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                            if token in eos or text in [chr(65+j) for j in range(len(choices))]:
                                break
                            new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                            inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
                            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
                            if adapter is not None:
                                hidden = torch.cat([hidden, model.model.get_input_embeddings()(new)], 1)
                                pos = torch.cat([pos, pos[:, :, -1:] + 1], 2)
                    finally:
                        if original_memories is not None:
                            adapter.all_visual_memories_batched = original_memories
                    scored = score_prediction(metric=dataset.spec.metric, prediction_text=text,
                        answer=item['answer'], answers=item.get('answers'), choices=choices, question=item['row'].get('question'))
                    out.write(json.dumps(dict(benchmark=benchmark, index=index, source_index=item['index'],
                        method=method, task=item['row'].get('task'), text=text, **scored, generated_tokens=len(generated),
                        visual_tokens=int(inputs0['mm_token_type_ids'].ne(0).sum()), deepstack_enabled=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                        prompt_layout=LAYOUT, plan_sha256=plan_hash, seconds=time.time()-started), ensure_ascii=False)+'\n')
                    if adapter is not None:
                        del memory
                vision_cache.clear()
                if index % 80 == shard:
                    print('PROGRESS', benchmark, index, flush=True)
    print('COMPLETE', shard, flush=True)


def aggregate(complete=False):
    rows = []
    for p in OUTPUT.glob('rows_*.jsonl'):
        data = p.read_text()
        lines = data.splitlines()
        if not complete and data and not data.endswith('\n'):
            lines = lines[:-1]
        rows.extend(json.loads(line) for line in lines)
    keys = [(r['benchmark'], r['index'], r['method']) for r in rows]
    assert len(keys) == len(set(keys))
    limit = json.loads((OUTPUT / 'plan.json').read_text())['limit']
    if complete:
        assert set(keys) == {(b, i, m) for b in BENCHES for i in range(limit) for m in METHODS}
    results = {}
    for method in METHODS:
        results[method] = {}
        for bench in BENCHES:
            selected = [r for r in rows if r['method'] == method and r['benchmark'] == bench]
            results[method][bench] = dict(samples=len(selected), accuracy=100*sum(r['score'] for r in selected)/len(selected) if selected else None)
    dump(OUTPUT / 'results.json', results)
    lines = ['# Four adapters and native base: no DeepStack, media-first prompts', '',
        'Fresh evaluation, all five models. Original weights unchanged; no training. Each benchmark uses the same first 1000 examples. MuirBench media references become numbered text references; all question and choice text follows the media. Media order and labels are preserved. MMIU/video prompts without markers are unchanged.', '',
        f'All five disable DeepStack injection. Native language-model prehook plus a rejecting _deepstack_process enforce this. Full adapters already use text-only computation and never call that injection. Native BF16 SDPA; adapters use the saved efficient attention path. Identical greedy uncached decoding, max {MAX_NEW_TOKENS} tokens, standalone-option/EOS stop. Video: 8 full-window sampled frames, real timestamps, no subtitles. No image pruning.', '',
        'This table is NOT directly comparable to historical interleaved/DeepStack-on results or the old-layout pruning table. Scores below are final only when status.json is complete. The 84-example question-relocation diagnostic is a different layout and is not substituted for these benchmark scores.', '',
        '| Method | MuirBench | MMIU | Video-MME | MVBench |', '|---|---:|---:|---:|---:|']
    for method, name in zip(METHODS, NAMES):
        values = []
        for b in ('muirbench', 'mmiu', 'videomme', 'mvbench'):
            r = results[method][b]
            values.append(f"{r['accuracy']:.2f} ({r['samples']})" if r['samples'] else 'pending')
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    (OUTPUT / 'README.md').write_text('\n'.join(lines)+'\n')
    return len(rows)


def run(limit):
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(model=MODEL, limit=limit, methods=list(METHODS), benchmarks=list(BENCHES),
        prompt_layout=LAYOUT, max_new_tokens=MAX_NEW_TOKENS, deepstack_enabled=False, source_sha256=sha(__file__),
        data_source_sha256=sha(REFERENCE/'data.py'), model_source_sha256=sha(REFERENCE/'model.py'),
        input_source_sha256=sha(ROOT/'src/multimodal_eval_inputs.py'),
        checkpoints={m: dict(path=str(p), sha256=sha(p)) for m,p in checkpoints().items()},
        datasets={b:sha(benchmark_manifest(b)) for b in BENCHES})
    if (OUTPUT / 'plan.json').exists():
        assert json.loads((OUTPUT / 'plan.json').read_text()) == plan, 'Do not mix incompatible runs'
    else:
        dump(OUTPUT / 'plan.json', plan)
    jobs, logs, retries = {}, [], {i:0 for i in range(8)}
    def start(shard):
        log = (OUTPUT / f'worker{shard}.log').open('a')
        logs.append(log)
        jobs[shard] = subprocess.Popen([sys.executable, '-m', 'src.adapter_nodeepstack_suite', '--shard', str(shard), '--limit', str(limit), '--output', str(OUTPUT),
            '--prompt-layout',LAYOUT,'--max-new-tokens',str(MAX_NEW_TOKENS)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'),
            stdout=log, stderr=subprocess.STDOUT)
    for i in range(8):
        start(i)
    started = time.time()
    try:
        while True:
            for shard, p in list(jobs.items()):
                if p.poll() not in (None, 0):
                    if retries[shard] >= 1:
                        raise RuntimeError(f'Worker {shard} failed twice; inspect log')
                    retries[shard] += 1
                    start(shard)
            count = aggregate()
            dump(OUTPUT/'status.json', dict(state='running', rows=count, expected=20*limit,
                elapsed=time.time()-started, workers={i:p.pid for i,p in jobs.items()}, retries=retries))
            print('ROWS', count, '/', 20*limit, 'elapsed', round(time.time()-started), flush=True)
            if all(p.poll() == 0 for p in jobs.values()):
                aggregate(complete=True)
                dump(OUTPUT/'status.json', dict(state='complete', rows=count, elapsed=time.time()-started, retries=retries))
                break
            time.sleep(15)
    finally:
        for p in jobs.values():
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prompt-layout', choices=('interleaved','media_first_v1'), default=LAYOUT)
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    OUTPUT = args.output
    LAYOUT = args.prompt_layout
    MAX_NEW_TOKENS = args.max_new_tokens
    if MAX_NEW_TOKENS<1:raise ValueError('max_new_tokens must be positive')
    run(args.limit) if args.shard is None else worker(args.shard, args.limit)
