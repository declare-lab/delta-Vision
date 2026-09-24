"""Establish a native FA2/DeepStack-off Qwen base on the saved Video-MME 999.

The native variant uses the unmodified HF FA2 forwards and generate loop, with
only DeepStack disabled. Other variants retain the earlier optimized baselines.
Run shards on separate GPUs; every sample checks all eight logits and final KV.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from src.benchmarking.engines.adapter import ROOT, MODEL, MANIFEST, MANIFEST_SHA, file_sha, dump, tensor_sha


def worker(args):
    import torch
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    from src.benchmarking.common.comparison import decoder_flops
    from src.data import QwenBenchmarkDataset
    from src.benchmarking.common.generation_timing import GenerationStageTimer
    from src.attention import optimize_qwen_attention_metadata
    from src.model_setup import disable_qwen_deepstack
    from src.kernels import FusedQwenNorms
    from src.kernels import QwenExactRoPE
    from src.kernels import QwenFusedProjections
    from src.graphs import fixed_greedy
    from src.graphs import NativeDecoderGraphs
    from src.graphs import QwenWholePrefillGraphs

    assert file_sha(MANIFEST) == MANIFEST_SHA
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / f'{args.variant}_{args.shard}.jsonl'
    if rows_path.exists():
        raise FileExistsError(rows_path)
    model, processor = load_baseline_model('base', args.model, torch.bfloat16, device, 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    native = args.variant == 'native'
    metadata = None if native else optimize_qwen_attention_metadata(model)
    optimized = args.variant == 'optimized'
    norms = FusedQwenNorms(model) if optimized else None
    rope = QwenExactRoPE(model) if optimized else None
    projections = QwenFusedProjections(model) if optimized else None
    graphs = None if native else NativeDecoderGraphs(model, max_shapes=8, prefill_layers=not optimized,
        packed_kv=bool(args.packed_decode_kv), max_prefill_shapes=1)
    whole = QwenWholePrefillGraphs(model, max_shapes=1) if optimized else None
    timer = GenerationStageTimer(model, measure_memory=True)
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=args.input_cache)
    assert len(dataset) == 999
    indices = args.indices if args.indices is not None else list(range(args.shard, 999, args.shards))
    saved_rows = {}
    if native and args.reference_results:
        saved_rows = {r['index']:r for p in Path(args.reference_results).glob('optimized_*.jsonl')
                      for line in p.read_text().splitlines() if (r := json.loads(line))}
        assert len(saved_rows) == 999
    if native:
        # No project graph/fusion/metadata wrapper may be installed in native mode.
        assert all('forward' not in module.__dict__ for module in model.modules())
        assert not any(getattr(model, name, None) for name in
            ('_benchmark_fused_qwen_norms', '_adapter_decode_graph_runner', '_adapter_exact_rope'))
    sources = ['src/benchmarking/engines/base.py', 'src/graphs.py', 'src/graphs.py',
        'src/kernels.py', 'src/kernels.py', 'src/kernels.py',
        'src/kernels.py', 'src/graphs.py',
        'src/attention.py', 'src/benchmarking/common/generation_timing.py', 'src/data.py', 'src/benchmarks.py',
        'src/video.py', 'src/video.py']
    dump(output/f'{args.variant}_{args.shard}.protocol.json', dict(vars(args), model=args.model,
        manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA, attention='flash_attention_2', deepstack='off',
        dtype='bfloat16', video_frames=8, subtitles=False, video_sampling='full_timestamp_v1',
        prompt_layout='media_first_v1', gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(),
        source_sha256={p:file_sha(ROOT/p) for p in sources},
        precision='Compare all generated logits and final layer KV to a separate native unfused eager FA2 request; native also matches saved input hashes and generated tokens',
        execution=('HF generate and native eager FA2; no CUDA graphs, compile, fused norms/RoPE/projections or attention metadata optimization'
                   if native else 'Previous project optimized graph configuration; packed decode KV when supported'),
        timing='Synchronized native generate; fresh vision/prefill and N-1 q_len=1 growing-KV steps; excludes preprocessing, warmup, capture, validation',
        request_prefill='Outer request start through generation setup, native position preparation, vision, language prefill, first logits and owned KV',
        forward_prefill='First model forward only; generation-side setup is outside this interval',
        peak_memory='Maximum warmed per-process allocated peak; one model per GPU process, model weights and graph pools included',
        flops='Analytic decoder prefill core; 2 FLOPs/MAC; excludes vision, LM head, norms and softmax'))

    def enable(value, capture=False):
        if graphs:
            graphs.enabled, graphs.allow_capture = value, capture
        if metadata:
            metadata.enabled = value
        if norms:
            norms.enabled = value
            rope.enabled = value
            projections.enabled = value
            whole.enabled, whole.allow_capture = value, capture

    def generate(inputs, *, check=False, measure=False):
        model.model.rope_deltas = None
        torch.cuda.synchronize()
        if measure:
            timer.begin()
        start = time.perf_counter()
        if measure:
            timer.mark_request_start(start)
        if optimized and graphs.enabled:
            result = fixed_greedy(model, inputs, args.tokens, output_logits=check)
        else:
            result = model.generate(**inputs, min_new_tokens=args.tokens, max_new_tokens=args.tokens,
                do_sample=False, return_dict_in_generate=True, output_logits=check,
                **({'disable_compile': True} if native else {}))
        torch.cuda.synchronize()
        total = time.perf_counter()-start
        if not measure:
            return None, result
        measured = timer.finish(total, args.tokens)
        assert measured['request_prefill_time_s'] >= measured['generation_prefill_time_s']
        assert measured['request_prefill_time_s'] < total
        measured.update(total_time_s=total, decode_ms_per_token=measured['decode_time_s']*1000/(args.tokens-1))
        return measured, result

    def forbid_sdpa(*a, **kw):
        raise AssertionError('FA2 base unexpectedly fell back to SDPA')

    with torch.inference_mode(), patch('torch.nn.functional.scaled_dot_product_attention', forbid_sdpa), rows_path.open('w', buffering=1) as file:
        for ordinal, index in enumerate(indices):
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, device)
            assert 'pixel_values_videos' in inputs and 'pixel_values' not in inputs
            assert inputs['attention_mask'].bool().all()
            input_hash = tensor_sha([inputs[k] for k in sorted(inputs)])
            if saved_rows:
                assert input_hash == saved_rows[index]['input_sha256'], (index, 'saved input')
            enable(False)
            _, reference = generate(inputs, check=True)
            enable(True, True)
            begin = time.perf_counter()
            _, candidate = generate(inputs, check=True)
            preparation_s = time.perf_counter()-begin
            assert torch.equal(reference.sequences, candidate.sequences), (index, 'tokens')
            assert len(reference.logits) == len(candidate.logits) == args.tokens
            assert all(torch.equal(a, b) for a, b in zip(reference.logits, candidate.logits)), (index, 'logits')
            assert all(torch.equal(getattr(a, k), getattr(b, k))
                for a,b in zip(reference.past_key_values.layers, candidate.past_key_values.layers)
                for k in ('keys', 'values')), (index, 'kv')
            tokens = candidate.sequences[0, inputs['input_ids'].shape[1]:].tolist()
            if saved_rows:
                assert tokens == saved_rows[index]['tokens'], (index, 'saved tokens')
            checked_logits = [tensor_sha([logits]) for logits in candidate.logits]
            checked_final_kv = tensor_sha([t for layer in candidate.past_key_values.layers
                                           for t in (layer.keys, layer.values)])
            del reference, candidate
            _, warm = generate(inputs)
            del warm
            enable(True, False)
            before = graphs.stats() if graphs else None, whole.stats() if whole else None
            trials = []
            for repetition in range(args.runs):
                trial, result = generate(inputs, measure=True)
                assert result.sequences[0, inputs['input_ids'].shape[1]:].tolist() == tokens
                trials.append(trial)
                del result
            after = graphs.stats() if graphs else None, whole.stats() if whole else None
            if graphs:
                assert before[0]['captures'] == after[0]['captures']
                assert before[0]['cold_layer_fallbacks'] == after[0]['cold_layer_fallbacks']
            if whole:
                assert before[1]['captures'] == after[1]['captures'] and before[1]['fallbacks'] == after[1]['fallbacks']
                assert after[1]['replays']-before[1]['replays'] == args.runs
            visual = int(inputs['mm_token_type_ids'].ne(0).sum())
            text = int(inputs['mm_token_type_ids'].eq(0).sum())
            row = dict(index=index, source_index=item['index'], duration=item['row']['duration'],
                variant=args.variant, shard=args.shard, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                input_sha256=input_hash, tokens=tokens, logits_sha256=checked_logits,
                final_kv_sha256=checked_final_kv, cuda_graphs_enabled=graphs is not None,
                saved_input_and_tokens_match=True if saved_rows else None,
                text=processor.tokenizer.decode(tokens, skip_special_tokens=True),
                visual_tokens=visual, text_tokens=text, logits_and_kv_bitwise_equal=True,
                flops=decoder_flops(model.model.language_model.config, [visual+text]*len(model.model.language_model.layers),
                    text_tokens=text, image_tokens=visual), capture_and_validation_s=preparation_s,
                timed_captures=0, timed_fallbacks=0, trials=trials)
            file.write(json.dumps(row, ensure_ascii=False)+'\n')
            print(json.dumps(dict(variant=args.variant, shard=args.shard, done=ordinal+1, expected=len(indices),
                index=index, prefill_ms=1000*statistics.median(t['request_prefill_time_s'] for t in trials),
                forward_prefill_ms=1000*statistics.median(t['generation_prefill_time_s'] for t in trials),
                decode_ms=statistics.median(t['decode_ms_per_token'] for t in trials), exact=True)), flush=True)
            del item, inputs, trials
    timer.remove()
    if whole: whole.remove()
    if graphs: graphs.remove()
    if rope: rope.remove()
    if projections: projections.remove()
    if norms: norms.remove()
    if metadata: metadata.remove()


def aggregate(args):
    output = Path(args.output)
    rows = [json.loads(line) for path in output.glob(f'{args.variant}_*.jsonl') for line in path.read_text().splitlines()]
    expected = set(args.indices) if args.indices is not None else set(range(999))
    assert len(rows) == len(expected) and {r['index'] for r in rows} == expected, (len(rows), len(expected))
    assert all(r['logits_and_kv_bitwise_equal'] and r['timed_captures'] == r['timed_fallbacks'] == 0 for r in rows)
    result = dict(method='base', variant=args.variant, samples=len(rows), tokens_per_request=args.tokens,
        attention='flash_attention_2', deepstack='off', all_logits_and_kv_bitwise_equal=True,
        timing='Sum of per-sample median seconds; full model requests, fresh vision, fixed generation length',
        total_speedup=1., prefill_speedup=1.)
    for key in ['total_time_s', 'generation_prefill_time_s', 'decode_time_s', 'generation_overhead_s']:
        result[key] = sum(statistics.median(t[key] for t in r['trials']) for r in rows)
    if all('request_prefill_time_s' in t for r in rows for t in r['trials']):
        result['request_prefill_time_s'] = sum(statistics.median(t['request_prefill_time_s'] for t in r['trials']) for r in rows)
        result['prefill_time_s'] = result['request_prefill_time_s']
        result['prefill_definition'] = 'Full request prefill including generation setup, position preparation, fresh vision and first logits/KV'
    if args.variant == 'native':
        assert all(not r['cuda_graphs_enabled'] for r in rows)
        result['execution'] = 'Native HF generate and eager FA2; DeepStack off; no project runtime optimizations'
        result['saved_input_and_tokens_match'] = all(r['saved_input_and_tokens_match'] for r in rows)
    result['decode_ms_per_token'] = result['decode_time_s']*1000/(len(rows)*(args.tokens-1))
    result['decode_tokens_per_s'] = 1000/result['decode_ms_per_token']
    result['kv_cache_mb'] = statistics.mean(r['trials'][0]['actual_prefill_kv_cache_mb'] for r in rows)
    result['flops'] = statistics.mean(r['flops'] for r in rows)
    result['peak_memory_mb'] = max(t['peak_memory_mb'] for r in rows for t in r['trials'])
    result['visual_tokens_mean'] = statistics.mean(r['visual_tokens'] for r in rows)
    result['text_tokens_mean'] = statistics.mean(r['text_tokens'] for r in rows)
    result['groups'] = {d:sum(r['duration']==d for r in rows) for d in ['short', 'medium', 'long']}
    dump(output/f'{args.variant}_summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['native', 'current', 'optimized'], default='native')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--output', default='test/results/video_base_native_fa2_20260915/videomme999')
    parser.add_argument('--input-cache', default='test/results/adapter_exact_20260915/inputs')
    parser.add_argument('--reference-results', default='test/results/video_base_20260915/videomme999',
                        help='Saved optimized base rows for native input/token validation; empty disables this cross-check.')
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--packed-decode-kv', action='store_true',
                        help='Use packed KV input/output for native decode CUDA Graphs when all layer cache shapes match.')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    assert args.tokens >= 2 and args.runs >= 1 and 0 <= args.shard < args.shards
    aggregate(args) if args.aggregate else worker(args)


if __name__ == '__main__':
    main()
