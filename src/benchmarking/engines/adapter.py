"""Paired original/optimized adapter speed and exactness on Video-MME 999.

Each subprocess owns one model, making warmed allocated peak memory meaningful.
Both variants run on each GPU; order is reversed on alternating shards. Every
request rebuilds vision and KV, then generates exactly N tokens with growing KV.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
MANIFEST = ROOT / 'artifacts/diagnostics/video_balanced_base_adapter_20260914/videomme_selected.jsonl'
MANIFEST_SHA = '44f433831305e6278b25b0ea905e683674e50cef5a359de390fee9218068c145'
EXACT_FIELDS = ['source_index', 'input_sha256', 'tokens', 'logits_sha256',
                'prefill_kv_sha256', 'final_kv_sha256']


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def tensor_sha(tensors):
    import torch
    hasher = hashlib.sha256()
    for tensor in tensors:
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        hasher.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


def cache_tensors(cache):
    return [t for layer in cache['_native_cache'].layers for t in (layer.keys, layer.values)]


def worker(args):
    import torch
    from unittest.mock import patch
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
    from src.benchmarking.common.comparison import decoder_flops
    from src.data import QwenBenchmarkDataset
    from src.model import load_qwen_embedding_adapter_checkpoint
    from src.attention import optimize_qwen_attention_metadata
    from src.model_setup import disable_qwen_deepstack

    assert file_sha(MANIFEST) == MANIFEST_SHA
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model, processor = load_baseline_model('base', args.model, torch.bfloat16, device, 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    optimize_qwen_attention_metadata(model)
    adapter, meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode == 'embedding_adapter'
    level = args.reference_level if args.variant == 'original' else args.optimized_level
    settings = SimpleNamespace(last_logits_only=True, attn_implementation='flash_attention_2',
        cuda_graph=True, cuda_graph_context=True, compile_verify=True, compile_max_diff=0.,
        cuda_graph_warmup=3, adapter_decode_cache_mode='fast',
        adapter_blocked_visual_layers=getattr(args, 'blocked_visual_layers', ()),
        adapter_exact_optimizations=level != 'legacy', adapter_max_optimizations=level == 'max')
    prefill = build_qwen_fast_adapter_prefill(model, adapter, settings)
    decoder = model._adapter_decode_graph_runner
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT / 'data/benchmarks/videomme'), cache_dir=args.input_cache)
    assert len(dataset) == 999
    indices = args.indices if args.indices is not None else list(range(args.shard, 999, args.shards))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    other = 'original' if args.variant == 'optimized' else 'optimized'
    paired_path = output / f'{other}_{args.shard}.jsonl'
    paired = {r['index']: r for r in map(json.loads, paired_path.read_text().splitlines())} if paired_path.exists() else {}
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos

    def request(inputs, *, check=False):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logits, text_mask, _, _, cache = prefill(inputs)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start
        if '_native_cache' not in cache:
            raise AssertionError('Video request did not use the native fast decode cache')
        if check:
            nv = int(inputs['mm_token_type_ids'].ne(0).sum())
            nt = int(inputs['mm_token_type_ids'].eq(0).sum())
            blocked = set(getattr(args, 'blocked_visual_layers', ()))
            expected = [nt if i in blocked else nt + nv for i in range(adapter.num_layers)]
            assert [layer.keys.shape[-2] for layer in cache['_native_cache'].layers] == expected
        # Only inspect or transfer validation data outside measured requests.
        kv_mb = sum(t.numel() * t.element_size() for t in cache_tensors(cache)) / 1024**2
        prefix_hash = tensor_sha(cache_tensors(cache)) if check else None
        times, tokens, logits_hashes = [], [], []
        for step in range(args.tokens):
            if check:
                logits_hashes.append(tensor_sha([logits]))
            scores = logits[:, -1].to(dtype=torch.float32, copy=True)
            scores[:, eos] = -float('inf')
            token = scores.argmax(-1).view(1, 1)
            tokens.append(token)
            if step + 1 == args.tokens:
                break
            torch.cuda.synchronize()
            begin = time.perf_counter()
            logits, cache = decoder(model, adapter, token, cache, logits_to_keep=1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
        torch.cuda.synchronize()
        total_s = time.perf_counter() - start
        peak_mb = torch.cuda.max_memory_allocated()/1024**2
        result = dict(total_s=total_s, prefill_s=prefill_s, decode_s=sum(times),
            decode_ms_per_token=1000 * sum(times) / len(times), decode_steps=len(times),
            tokens=torch.cat(tokens, dim=1)[0].tolist(), kv_cache_mb=kv_mb, peak_memory_mb=peak_mb,
            peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2)
        if check:
            result.update(logits_sha256=logits_hashes, prefill_kv_sha256=prefix_hash,
                          final_kv_sha256=tensor_sha(cache_tensors(cache)))
        return result

    def forbid_sdpa(*a, **kw):
        raise AssertionError('SDPA was called in the FA2-only adapter benchmark')

    with torch.inference_mode(), patch('torch.nn.functional.scaled_dot_product_attention', forbid_sdpa), (output / f'{args.variant}_{args.shard}.jsonl').open('w', buffering=1) as file:
        for ordinal, index in enumerate(indices):
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, device)
            assert inputs['attention_mask'].bool().all()
            assert 'pixel_values_videos' in inputs and 'pixel_values' not in inputs
            input_hash = tensor_sha([inputs[k] for k in sorted(inputs)])
            decoder.allow_capture = True
            checked = request(inputs, check=True)
            if getattr(args, 'verify_eager_decode', False):
                decoder.enabled = False
                eager = request(inputs, check=True)
                decoder.enabled = True
                for field in ('tokens', 'logits_sha256', 'prefill_kv_sha256', 'final_kv_sha256'):
                    assert eager[field] == checked[field], (index, field, 'graph/eager mismatch')
            # A second untimed replay removes first-use allocation/kernel effects.
            warm = request(inputs)
            assert warm['tokens'] == checked['tokens']
            decoder.allow_capture = False
            before = decoder.stats()
            graph_ids = tuple(id(r.graph) for r in prefill.graph_runners)
            trials = []
            for _ in range(args.runs):
                trial = request(inputs)
                assert trial['tokens'] == checked['tokens']
                assert tuple(id(r.graph) for r in prefill.graph_runners) == graph_ids, 'Capture occurred inside timing'
                trials.append(trial)
            after = decoder.stats()
            assert before['captures'] == after['captures'] and before['cold_fallbacks'] == after['cold_fallbacks']
            visual = int(inputs['mm_token_type_ids'].ne(0).sum())
            text = int(inputs['mm_token_type_ids'].eq(0).sum())
            row = dict(index=index, source_index=item['index'], duration=item['row']['duration'],
                variant=args.variant, optimization_level=level, shard=args.shard, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                input_sha256=input_hash, tokens=checked['tokens'],
                logits_sha256=checked['logits_sha256'], prefill_kv_sha256=checked['prefill_kv_sha256'],
                final_kv_sha256=checked['final_kv_sha256'], visual_tokens=visual, text_tokens=text,
                first_token_text=processor.tokenizer.decode(checked['tokens'][:1], skip_special_tokens=True),
                flops=(None if getattr(args, 'blocked_visual_layers', ()) else decoder_flops(model.model.language_model.config, [visual+text]*adapter.num_layers,
                    text_tokens=text, image_tokens=visual, adapter_rank=adapter.visual_adapter_rank)),
                blocked_visual_layers=list(getattr(args, 'blocked_visual_layers', ())),
                eager_decode_bitwise_equal=True if getattr(args, 'verify_eager_decode', False) else None,
                trials=trials, timed_captures=0, timed_fallbacks=0)
            if index in paired:
                mismatch = [key for key in EXACT_FIELDS if row[key] != paired[index][key]]
                if mismatch:
                    dump(output / f'exact_failure_{args.shard}.json', dict(index=index, fields=mismatch))
                    raise AssertionError(f'Sample {index} exact validation failed: {mismatch}')
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps(dict(variant=args.variant, shard=args.shard, done=ordinal+1, expected=len(indices),
                index=index, prefill_ms=1000*statistics.median(t['prefill_s'] for t in trials),
                decode_ms=statistics.median(t['decode_ms_per_token'] for t in trials))), flush=True)
            del item, inputs, checked, warm, trials
    decoder.remove()
    del decoder, prefill, adapter, model
    gc.collect()
    torch.cuda.empty_cache()


def aggregate(args):
    output = Path(args.output)
    variants = {}
    for variant in ['original', 'optimized']:
        rows = [json.loads(line) for path in output.glob(f'{variant}_*.jsonl') for line in path.read_text().splitlines()]
        variants[variant] = {row['index']: row for row in rows}
        assert len(variants[variant]) == len(rows), 'Duplicate samples'
    expected = set(args.indices) if args.indices is not None else set(range(999))
    assert all(set(rows) == expected for rows in variants.values()), {v:len(r) for v,r in variants.items()}
    mismatches = []
    for index in sorted(expected):
        a, b = (variants[v][index] for v in ['original', 'optimized'])
        for key in EXACT_FIELDS:
            if a[key] != b[key]:
                mismatches.append(dict(index=index, field=key))
    summary = dict(samples=len(expected), tokens_per_request=args.tokens, attention='flash_attention_2',
        optimization_levels=dict(original=args.reference_level, optimized=args.optimized_level),
        deepstack='off', exact_mismatches=mismatches, all_logits_and_kv_bitwise_equal=not mismatches,
        timing='Sum of per-sample medians; CPU preprocessing, capture, warmup and exactness hashing excluded',
        prefill='Request start through fresh vision, position preparation, adapter decoder, first logits and owned prefill KV',
        decode='Exactly N-1 growing-KV forward calls; EOS suppressed equally for both variants', variants={})
    for variant, rows in variants.items():
        totals = {key:sum(statistics.median(t[key] for t in row['trials']) for row in rows.values())
                  for key in ['total_s', 'prefill_s', 'decode_s']}
        summary['variants'][variant] = dict(totals,
            decode_ms_per_token=1000*totals['decode_s']/(len(rows)*(args.tokens-1)),
            kv_cache_mb=statistics.mean(row['trials'][0]['kv_cache_mb'] for row in rows.values()),
            flops=statistics.mean(row['flops'] for row in rows.values()),
            peak_memory_mb=max(t['peak_memory_mb'] for row in rows.values() for t in row['trials']),
            decode_tokens_per_second=len(rows)*(args.tokens-1)/totals['decode_s'])
        assert all(row['timed_captures'] == row['timed_fallbacks'] == 0 for row in rows.values())
        assert all(t['decode_steps'] == args.tokens-1 for row in rows.values() for t in row['trials'])
    a, b = summary['variants']['original'], summary['variants']['optimized']
    summary['speedup_vs_original_adapter'] = {k:a[k]/b[k] for k in ['total_s', 'prefill_s', 'decode_s']}
    dump(output / 'summary.json', summary)
    dump(output / 'validation.json', dict(samples=len(expected), logits_checked=len(expected)*args.tokens,
        prefix_and_final_kv_layers_checked=len(expected)*36*2,
        all_logits_and_kv_bitwise_equal=not mismatches, mismatches=mismatches,
        timed_requests=2*len(expected)*args.runs, timed_captures=0, timed_fallbacks=0,
        all_requests_rebuild_vision_and_kv=True, sdpa_calls=0))
    print(json.dumps(summary, indent=2), flush=True)
    if mismatches:
        raise AssertionError(f'Exact validation failed: {mismatches[:10]}')
    return summary


def launch(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('original_*.jsonl')) or any(output.glob('optimized_*.jsonl')):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    assert file_sha(MANIFEST) == MANIFEST_SHA
    source_paths = ['src/model.py', 'src/benchmarking/common/prefill.py', 'src/benchmarking/engines/adapter.py',
        'src/kernels.py', 'src/kernels.py', 'src/attention.py',
        'src/kernels.py', 'src/kernels.py',
        'src/graphs.py',
        'src/kernels.py',
        'src/graphs.py', 'src/graphs.py', 'src/attention.py',
        'src/graphs.py', 'src/attention.py', 'src/model_setup.py',
        'baselines/eval_baselines.py', 'src/benchmarks.py',
        'src/data.py', 'src/video.py', 'src/video.py']
    dump(output / 'protocol.json', dict(vars(args), manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA,
        checkpoint_sha256=file_sha(args.checkpoint), seed=42, video_frames=8, subtitles=False,
        sampling='full_timestamp_v1', prompt_layout='media_first_v1',
        variants=dict(original=args.reference_level, optimized=args.optimized_level),
        dtype='bfloat16', batch_size=1,
        peak_memory='Maximum warmed per-process allocated peak, including model, adapter and graph pools; MiB',
        flops='Analytic decoder prefill core, including visual KV and adapter projections; 2 FLOPs/MAC; excludes vision, LM head, norms, softmax and FA2 packing duplication',
        source_sha256={p:file_sha(ROOT/p) for p in source_paths}))
    running = {}
    orders = {shard:(['original', 'optimized'] if shard%2 == 0 else ['optimized', 'original'])
              for shard in range(len(args.gpus))}
    def start(shard, variant):
        command = [sys.executable, '-m', 'src.benchmarking.engines.adapter', '--worker', '--variant', variant,
            '--shard', str(shard), '--shards', str(len(args.gpus)), '--output', str(output),
            '--model', args.model, '--checkpoint', str(args.checkpoint), '--tokens', str(args.tokens), '--runs', str(args.runs)]
        command += ['--reference-level', args.reference_level, '--optimized-level', args.optimized_level]
        if args.input_cache:
            command += ['--input-cache', str(args.input_cache)]
        if args.indices is not None:
            command += ['--indices', *map(str, args.indices[shard::len(args.gpus)])]
        log = (output / f'{variant}_{shard}.log').open('w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpus[shard]), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1',
            QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        running[shard] = (variant, process)
        print('START', shard, variant, process.pid, flush=True)
    for shard in orders:
        start(shard, orders[shard].pop(0))
    try:
        while running:
            for shard, (variant, process) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'{variant} shard {shard} failed: {(output/f"{variant}_{shard}.log").read_text()[-3000:]}')
                print('DONE', shard, variant, flush=True)
                del running[shard]
                if orders[shard]:
                    start(shard, orders[shard].pop(0))
            time.sleep(2)
    finally:
        for _, process in running.values():
            process.terminate()
        for _, process in running.values():
            process.wait()
    expected_sources = json.loads((output/'protocol.json').read_text())['source_sha256']
    changed = [p for p, sha in expected_sources.items() if file_sha(ROOT/p) != sha]
    dump(output/'source_validation.json', dict(all_sources_unchanged=not changed, changed=changed))
    if changed:
        raise AssertionError(f'Source changed during comparison: {changed}')
    aggregate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='test/results/adapter_exact_20260915/videomme999')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--checkpoint', default=str(CHECKPOINT))
    parser.add_argument('--input-cache', default='')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--reference-level', choices=['legacy', 'exact'], default='legacy')
    parser.add_argument('--optimized-level', choices=['exact', 'max'], default='exact')
    parser.add_argument('--indices', nargs='*', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--variant', choices=['original', 'optimized'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.tokens < 2 or args.runs < 1:
        raise ValueError('Need at least two tokens and one timed run')
    if args.worker:
        worker(args)
    elif args.aggregate:
        aggregate(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
