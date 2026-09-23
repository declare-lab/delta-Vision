"""FA2 timing and pruning/decode audits on the fixed Video-MME 999.

Uses each maintained Qwen port's HF generate entry. The fa2_metadata execution
reuses sequence metadata within a forward without changing the attention kernel,
RoPE, pruning or model arithmetic. Optional selector graphs and exact RMSNorm
reduce launch/memory traffic while preserving every result bit.
Audits and exact native parity checks are outside timing; every request recomputes
vision, pruning and its own sequence metadata.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from src.benchmark_adapter_optimizations import ROOT, MODEL, MANIFEST, MANIFEST_SHA, dump, file_sha, tensor_sha

METHODS = ['fastv', 'dart', 'visionzip', 'divprune', 'zoo', 'sparsevlm']
INPUT_CACHE = ROOT/'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme'


def label(method, retention):
    return f'{method}_r{round(retention*100):03d}'


def execution_description(execution, selector_graphs=False, exact_norms=False, full_cuda_graphs=False, exact_rope_projections=False):
    description = ('Repository Qwen port, native HF generate, no project runtime optimizations'
            if execution == 'native' else
            'Repository Qwen port, HF generate, FA2 sequence metadata reused within each forward; no graphs, compile or fused model kernels')
    if full_cuda_graphs:
        description = description.replace(
            'no graphs, compile',
            'CUDA Graphs for vision, language prefill layers, full q_len=1 decode forwards, and selector tensor operations; no compile',
        )
    if selector_graphs:
        description = description.replace('no graphs, compile', 'CUDA Graphs of original selector tensor operations only; no vision/decoder graphs, compile')
    if exact_norms:
        description = description.replace('or fused model kernels', 'or fused RoPE/projections; RMSNorm uses the exact installed PyTorch reduction order in one kernel')
    if exact_rope_projections:
        description = description.replace('no compile or fused RoPE/projections', 'no compile; exact RoPE/projection fusion enabled')
    return description


def layer_kv(cache):
    return [t for layer in cache.layers for t in (layer.keys, layer.values)]


def worker(args):
    import torch
    from unittest.mock import patch
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from baselines.multimodal_pruning_utils import visual_budget
    from src.benchmark_comparison import decoder_flops, native_cached_step
    from src.data import QwenBenchmarkDataset
    from src.generation_timing import GenerationStageTimer
    from src.qwen_attention_metadata import optimize_qwen_attention_metadata
    from src.qwen_deepstack import disable_qwen_deepstack
    from src.qwen_fixed_greedy import fixed_greedy

    assert file_sha(MANIFEST) == MANIFEST_SHA
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    output = Path(args.output)/label(args.method, args.retention)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output/f'{args.execution}_{args.shard}.jsonl'
    if rows_path.exists():
        raise FileExistsError(rows_path)
    model, processor = load_baseline_model(args.method, args.model, torch.bfloat16, device,
                                           args.retention, 'flash_attention_2')
    disable_qwen_deepstack(model)
    assert all('forward' not in module.__dict__ for module in model.modules())
    metadata = optimize_qwen_attention_metadata(model) if args.execution == 'fa2_metadata' else None
    graphs = None
    if args.full_cuda_graphs:
        from src.qwen_native_graph import NativeDecoderGraphs
        graphs = NativeDecoderGraphs(model, max_shapes=args.graph_max_shapes, vision=True, full_decode=True,
            prefill_layers=True, packed_kv=bool(args.packed_decode_kv), max_prefill_shapes=1)
    elif args.selector_graphs and args.method in ('dart', 'divprune', 'zoo'):
        from src.qwen_native_graph import NativeDecoderGraphs
        graphs = NativeDecoderGraphs(model, max_shapes=args.graph_max_shapes, vision=False, full_decode=False, prefill_layers=False)
    norms = None
    if args.exact_norms:
        from src.qwen_fused_norm import FusedQwenNorms
        norms = FusedQwenNorms(model)
        norms.native_order = True
    rope = projections = None
    if args.exact_rope_projections:
        from src.qwen_exact_rope import QwenExactRoPE
        from src.qwen_fused_projections import QwenFusedProjections
        rope = QwenExactRoPE(model)
        projections = QwenFusedProjections(model)
    timer = GenerationStageTimer(model, measure_memory=True)
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=args.input_cache)
    assert len(dataset) == 999
    indices = args.indices if args.indices is not None else list(range(args.shard, 999, args.shards))
    saved_base = {r['index']:r for p in (ROOT/'test/results/video_base_20260915/videomme999').glob('optimized_*.jsonl')
                  for line in p.read_text().splitlines() if (r := json.loads(line))}
    assert len(saved_base) == 999
    state = dict(check=False, prefix=None, next_positions=None, calls=[])

    def observe(current, positional, kwargs, result):
        if not state['check']:
            return
        cache = result.past_key_values
        ids = kwargs['input_ids']
        state['calls'].append(dict(input_tokens=ids.shape[-1],
            has_pixels=kwargs.get('pixel_values_videos') is not None,
            layer_lengths=[layer.keys.shape[-2] for layer in cache.layers]))
        if state['prefix'] is None:
            prefix = copy.copy(cache)
            prefix.layers = [copy.copy(layer) for layer in cache.layers]
            state['prefix'] = prefix
            positions = kwargs['position_ids']
            assert positions.shape[0] == 4
            state['next_positions'] = positions[1:, :, -1:] + 1
    observer = model.register_forward_hook(observe, with_kwargs=True)

    def audited(value):
        for module in (model.model, model.model.language_model):
            module._pruning_audit_enabled = value
            module._pruning_audit = []

    rng_cpu = rng_cuda = None
    def generate(inputs, *, check=False, measure=False):
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        model.model.rope_deltas = None
        audited(check)
        state.update(check=check, prefix=None, next_positions=None, calls=[])
        torch.cuda.synchronize()
        if measure:
            timer.begin()
        start = time.perf_counter()
        if measure:
            timer.mark_request_start(start)
        if args.fixed_greedy:
            result = fixed_greedy(model, inputs, args.tokens, output_logits=check)
        else:
            result = model.generate(**inputs, min_new_tokens=args.tokens, max_new_tokens=args.tokens,
                do_sample=False, disable_compile=True, return_dict_in_generate=True, output_logits=check)
        torch.cuda.synchronize()
        total = time.perf_counter()-start
        state['check'] = False
        measured = timer.finish(total, args.tokens) if measure else None
        if measured:
            measured['total_time_s'] = total
            measured['decode_ms_per_token'] = 1000*measured['decode_time_s']/(args.tokens-1)
            assert measured['request_prefill_time_s'] >= measured['generation_prefill_time_s']
        return measured, result

    def forbid_sdpa(*args, **kwargs):
        raise AssertionError('SDPA called in native FA2 pruning benchmark')

    with torch.inference_mode(), patch('torch.nn.functional.scaled_dot_product_attention', forbid_sdpa), rows_path.open('w', buffering=1) as file:
        for ordinal, index in enumerate(indices):
            if graphs is not None:
                graphs.begin_request()
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, device)
            assert inputs['attention_mask'].bool().all()
            assert 'pixel_values_videos' in inputs and 'pixel_values' not in inputs
            input_hash = tensor_sha([inputs[k] for k in sorted(inputs)])
            assert input_hash == saved_base[index]['input_sha256']
            visual = inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
            text_count = inputs['input_ids'].shape[-1]-visual.numel()
            keep = visual_budget(visual.numel(), args.retention)
            configure_baseline(model, args.method, args.retention, int(visual[0]), visual.numel())
            torch.manual_seed(42+index)
            rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            # Count FA2 metadata construction only during the untimed audit.
            # Pruned text positions retain gaps, triggering the stock HF varlen
            # preparation in each remaining prefill layer.
            import transformers.modeling_flash_attention_utils as fa_utils
            metadata_counts = dict(prefill=0, decode=0)
            original_prepare = fa_utils.prepare_fa_kwargs_from_position_ids
            def count_metadata(*a, **kw):
                metadata_counts['prefill' if not state['calls'] else 'decode'] += 1
                return original_prepare(*a, **kw)
            with patch.object(fa_utils, 'prepare_fa_kwargs_from_position_ids', count_metadata):
                if metadata is not None:
                    metadata.enabled = False
                if graphs is not None:
                    graphs.enabled = graphs.allow_capture = False
                if norms is not None: norms.enabled = False
                if rope is not None:
                    rope.enabled = False
                    projections.enabled = False
                _, checked = generate(inputs, check=True)
            original_metadata_counts = dict(metadata_counts)
            original_hashes = dict(logits=[tensor_sha([logits]) for logits in checked.logits],
                prefix=tensor_sha(layer_kv(state['prefix'])), final=tensor_sha(layer_kv(checked.past_key_values)),
                tokens=checked.sequences[0, inputs['input_ids'].shape[-1]:].tolist(),
                audits=model.model._pruning_audit+model.model.language_model._pruning_audit)
            if metadata is not None:
                del checked
                state.update(prefix=None, calls=[], next_positions=None)
                metadata.enabled = True
                if norms is not None: norms.enabled = True
                if rope is not None:
                    rope.enabled = True
                    projections.enabled = True
                if graphs is not None:
                    graphs.enabled = graphs.allow_capture = True
                metadata_counts = dict(prefill=0, decode=0)
                with patch.object(fa_utils, 'prepare_fa_kwargs_from_position_ids', count_metadata):
                    _, checked = generate(inputs, check=True)
                if graphs is not None: graphs.allow_capture = False
            tokens = checked.sequences[0, inputs['input_ids'].shape[-1]:].tolist()
            assert len(tokens) == len(checked.logits) == args.tokens
            audits = model.model._pruning_audit+model.model.language_model._pruning_audit
            assert audits and audits[0]['before_visual'] == visual.numel()
            assert audits[-1]['after_visual'] == keep
            original_visual = set(visual.tolist())
            assert all(a['text_tokens'] == text_count for a in audits)
            assert all(set(a['selected_positions']).issubset(set(a['visual_positions'])) for a in audits)
            assert set(audits[0]['visual_positions']) == original_visual
            front = 2 if args.method in ('fastv', 'dart', 'sparsevlm') else 0
            expected_lengths = [text_count+visual.numel()]*front+[text_count+keep]*(36-front)
            prefix = state['prefix']
            assert len(prefix.layers) == 36
            lengths = [layer.keys.shape[-2] for layer in prefix.layers]
            assert lengths == expected_lengths, (index, args.method, lengths, expected_lengths)
            calls = state['calls']
            assert len(calls) == args.tokens
            for step, call in enumerate(calls):
                assert call['input_tokens'] == (inputs['input_ids'].shape[-1] if step == 0 else 1)
                assert call['has_pixels'] == (step == 0)
                assert call['layer_lengths'] == [length+step for length in lengths]
            prefix_hash = tensor_sha(layer_kv(prefix))
            logits_hashes = [tensor_sha([logits]) for logits in checked.logits]
            final_hash = tensor_sha(layer_kv(checked.past_key_values))
            assert logits_hashes == original_hashes['logits'], (index, args.method, 'native logits parity')
            assert prefix_hash == original_hashes['prefix'], (index, args.method, 'native prefix KV parity')
            assert final_hash == original_hashes['final'], (index, args.method, 'native final KV parity')
            assert tokens == original_hashes['tokens'] and audits == original_hashes['audits']
            # Independent native layer loop, same single-token GEMM shapes and
            # original M-RoPE coordinates. It must agree with HF generate despite
            # different cache lengths across the 36 layers.
            positions = state['next_positions']
            for step in range(args.tokens-1):
                next_token = torch.tensor([[tokens[step]]], device=device)
                actual, prefix = native_cached_step(model, next_token, prefix, positions+step)
                assert torch.equal(actual[:, -1].float(), checked.logits[step+1].float()), (index, args.method, 'decode', step)
            assert tensor_sha(layer_kv(prefix)) == final_hash, (index, args.method, 'final KV')
            state.update(prefix=None, calls=[], next_positions=None)
            del prefix, checked, actual, positions
            audited(False)
            _, warm = generate(inputs)
            assert warm.sequences[0, inputs['input_ids'].shape[-1]:].tolist() == tokens
            del warm
            if args.compare_native_timing:
                assert metadata is not None
                metadata.enabled = False
                if norms is not None: norms.enabled = False
                if rope is not None:
                    rope.enabled = False
                    projections.enabled = False
                if graphs is not None: graphs.enabled = False
                _, warm = generate(inputs)
                del warm
                metadata.enabled = True
                if norms is not None: norms.enabled = True
                if rope is not None:
                    rope.enabled = True
                    projections.enabled = True
                if graphs is not None: graphs.enabled = True
            trials = []
            native_trials = []
            graph_stats = graphs.stats() if graphs is not None else {}
            for repeat in range(args.runs):
                order = ([False, True] if repeat % 2 == 0 else [True, False]) if args.compare_native_timing else [True]
                for enabled in order:
                    if metadata is not None: metadata.enabled = enabled
                    if graphs is not None: graphs.enabled = enabled
                    if norms is not None: norms.enabled = enabled
                    if rope is not None:
                        rope.enabled = enabled
                        projections.enabled = enabled
                    trial, result = generate(inputs, measure=True)
                    assert result.sequences[0, inputs['input_ids'].shape[-1]:].tolist() == tokens
                    assert trial['generation_stages'][0]['cache_lengths'] == lengths
                    (trials if enabled else native_trials).append(trial)
                    del result
            if metadata is not None: metadata.enabled = True
            if norms is not None: norms.enabled = True
            if rope is not None:
                rope.enabled = True
                projections.enabled = True
            if graphs is not None:
                graphs.enabled = True
                final_stats = graphs.stats()
                assert final_stats['captures'] == graph_stats['captures']
                assert final_stats['cold_layer_fallbacks'] == graph_stats['cold_layer_fallbacks']
                graph_stats = dict(final_stats,
                    timed_replays=final_stats['layer_replays']-graph_stats['layer_replays'],
                    timed_captures=0, timed_fallbacks=0)
            row = dict(index=index, source_index=item['index'], duration=item['row']['duration'],
                method=args.method, retention=args.retention, shard=args.shard,
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), input_sha256=input_hash, tokens=tokens,
                logits_sha256=logits_hashes, prefill_kv_sha256=prefix_hash, final_kv_sha256=final_hash,
                visual_tokens=visual.numel(), text_tokens=text_count, retained_visual_tokens=keep,
                prefill_layer_lengths=lengths, pruning_audit=audits,
                fa2_sequence_metadata_builds=metadata_counts,
                original_fa2_sequence_metadata_builds=original_metadata_counts,
                flops=decoder_flops(model.model.language_model.config, lengths,
                    text_tokens=text_count, image_tokens=visual.numel()),
                native_decode_logits_and_final_kv_bitwise_equal=True,
                original_native_logits_prefix_and_final_kv_bitwise_equal=True,
                original_native_selected_positions_equal=True,
                execution=args.execution, cuda_graphs_enabled=graphs is not None,
                exact_rmsnorm_enabled=norms is not None,
                exact_rope_projection_enabled=rope is not None,
                selector_cuda_graphs_enabled=graphs is not None and (args.full_cuda_graphs or args.method in ('dart','divprune','zoo')),
                decoder_cuda_graphs_enabled=bool(args.full_cuda_graphs), vision_cuda_graphs_enabled=bool(args.full_cuda_graphs),
                graph_stats=graph_stats, selector_graph_stats=graph_stats, sdpa_calls=0,
                trials=trials, native_trials=native_trials)
            file.write(json.dumps(row)+'\n')
            print(json.dumps(dict(method=args.method, retention=args.retention, shard=args.shard,
                done=ordinal+1, expected=len(indices), index=index,
                prefill_ms=1000*statistics.median(t['request_prefill_time_s'] for t in trials),
                decode_ms=statistics.median(t['decode_ms_per_token'] for t in trials), exact=True)),flush=True)
            del inputs, item, trials, row, audits, calls
    observer.remove()
    timer.remove()
    if metadata is not None: metadata.remove()
    if graphs is not None: graphs.remove()
    if norms is not None: norms.remove()
    if projections is not None: projections.remove()
    if rope is not None: rope.remove()


def aggregate(args):
    output = Path(args.output)
    rows = []
    expected = set(args.indices) if args.indices is not None else set(range(999))
    for method in args.methods:
        for retention in args.retentions:
            directory = output/label(method, retention)
            data = [json.loads(line) for path in directory.glob(f'{args.execution}_*.jsonl') for line in path.read_text().splitlines()]
            assert len(data) == len(expected) and {r['index'] for r in data} == expected
            assert all(r['native_decode_logits_and_final_kv_bitwise_equal'] and r['sdpa_calls']==0 for r in data)
            if args.full_cuda_graphs:
                assert all(r['decoder_cuda_graphs_enabled'] and r['vision_cuda_graphs_enabled'] for r in data)
            else:
                assert all(not r['decoder_cuda_graphs_enabled'] and not r['vision_cuda_graphs_enabled'] for r in data)
            assert all(r['selector_graph_stats'].get('timed_captures',0)==r['selector_graph_stats'].get('timed_fallbacks',0)==0 for r in data)
            assert all(r['original_native_logits_prefix_and_final_kv_bitwise_equal'] and r['original_native_selected_positions_equal'] for r in data)
            assert all(len(r['trials']) == args.runs and all(t['decode_steps']==args.tokens-1 for t in r['trials']) for r in data)
            timing = {key:sum(statistics.median(t[key] for t in r['trials']) for r in data)
                      for key in ['total_time_s','request_prefill_time_s','generation_prefill_time_s','decode_time_s','generation_overhead_s']}
            summary = dict(method=method, retention=retention, samples=len(data), tokens_per_request=args.tokens,
                attention='flash_attention_2', deepstack='off',
                execution=execution_description(args.execution,args.selector_graphs and method in ('dart','divprune','zoo'),args.exact_norms,args.full_cuda_graphs,args.exact_rope_projections),
                all_original_native_logits_prefix_and_final_kv_bitwise_equal=True,
                all_native_decode_logits_and_final_kv_bitwise_equal=True, **timing,
                prefill_time_s=timing['request_prefill_time_s'],
                decode_ms_per_token=1000*timing['decode_time_s']/(len(data)*(args.tokens-1)),
                decode_tokens_per_s=len(data)*(args.tokens-1)/timing['decode_time_s'],
                kv_cache_mb=statistics.mean(r['trials'][0]['actual_prefill_kv_cache_mb'] for r in data),
                flops=statistics.mean(r['flops'] for r in data),
                peak_memory_mb=max(t['peak_memory_mb'] for r in data for t in r['trials']),
                groups={d:sum(r['duration']==d for r in data) for d in ['short','medium','long']})
            dump(directory/'summary.json', summary)
            rows.append(summary)
    dump(output/'summary.json', rows)
    print(json.dumps(rows,indent=2),flush=True)


def launch(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('*/*_*.jsonl')):
        raise FileExistsError(output)
    sources = ['src/benchmark_video_pruning.py','src/benchmark_comparison.py','src/generation_timing.py',
        'baselines/eval_baselines.py','baselines/multimodal_pruning_utils.py','src/qwen_deepstack.py',
        'src/data.py','src/benchmarks.py','src/video_benchmark_inputs.py','src/benchmark_video_sampling.py',
        'src/qwen_attention_metadata.py','src/qwen_native_graph.py','src/qwen_fused_norm.py','src/qwen_native_order_norm.py']
    sources += [f'baselines/{m}/qwen3_vl/modeling_qwen3_vl_{m}.py' for m in args.methods]
    hashes = {p:file_sha(ROOT/p) for p in sources}
    dump(output/'protocol.json',dict(vars(args),manifest=str(MANIFEST),manifest_sha256=file_sha(MANIFEST),
        dtype='bfloat16',attention='flash_attention_2',deepstack='off',video_frames=8,
        video_sampling='full_timestamp_v1',prompt_layout='media_first_v1',subtitles=False,
        execution=execution_description(args.execution,args.selector_graphs,args.exact_norms,args.full_cuda_graphs,args.exact_rope_projections),
        seed='42+sample_index; restore RNG outside timing before every request so Zoo-Prune uses identical fresh stochastic selection',
        timing='Sum of per-sample medians; CPU preprocessing, input transfer, warmup and validation excluded; selector execution included',
        generation_loop='fixed_greedy' if args.fixed_greedy else 'HF generate',
        prefill='Outer request start through native generation setup, positions, vision, pruning, language and first logits/KV',
        flops='Analytic decoder prefill core using observed layer lengths; excludes vision, selectors/merging, head, norm, softmax',
        peak_memory='Maximum warmed per-request CUDA allocated peak, MiB, model included, one model per GPU process',
        source_sha256=hashes))
    running = {}
    try:
        for method in args.methods:
            for retention in args.retentions:
                directory = output/label(method,retention)
                directory.mkdir(parents=True, exist_ok=True)
                for shard,gpu in enumerate(args.gpus):
                    selected = args.indices[shard::len(args.gpus)] if args.indices is not None else None
                    if selected == []:
                        continue
                    command = [sys.executable,'-m','src.benchmark_video_pruning','--worker','--method',method,
                        '--retention',str(retention),'--shard',str(shard),'--shards',str(len(args.gpus)),
                        '--tokens',str(args.tokens),'--runs',str(args.runs),'--model',args.model,
                        '--execution',args.execution,
                        '--input-cache',str(args.input_cache),'--output',str(output)]
                    if args.compare_native_timing:command += ['--compare-native-timing']
                    if args.selector_graphs:command += ['--selector-graphs']
                    if args.full_cuda_graphs:command += ['--full-cuda-graphs','--graph-max-shapes',str(args.graph_max_shapes)]
                    if args.fixed_greedy:command += ['--fixed-greedy']
                    if args.packed_decode_kv:command += ['--packed-decode-kv']
                    if args.exact_rope_projections:command += ['--exact-rope-projections']
                    if args.exact_norms:command += ['--exact-norms']
                    if selected is not None:command += ['--indices',*map(str,selected)]
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                        TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',
                        QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
                    with (directory/f'{args.execution}_{shard}.log').open('w') as log:
                        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[shard]=process
                    print('START',method,retention,shard,process.pid,flush=True)
                while running:
                    for shard,process in list(running.items()):
                        if process.poll() is None:continue
                        if process.returncode:
                            raise RuntimeError((directory/f'{args.execution}_{shard}.log').read_text()[-4000:])
                        print('DONE',method,retention,shard,flush=True)
                        del running[shard]
                    time.sleep(2)
    finally:
        for process in running.values():process.terminate()
        for process in running.values():process.wait()
    changed = [p for p,h in hashes.items() if file_sha(ROOT/p) != h]
    dump(output/'source_validation.json',dict(all_sources_unchanged=not changed,changed=changed))
    assert not changed,changed
    aggregate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='test/results/video_pruning_fa2_metadata_20260915/videomme999')
    parser.add_argument('--execution',choices=['native','fa2_metadata'],default='fa2_metadata')
    parser.add_argument('--compare-native-timing',action='store_true',help='Paired native/metadata timings for the same model and input; pilot diagnosis only')
    parser.add_argument('--selector-graphs',action='store_true',help='Replay original DART/DivPrune/Zoo selector operations only; vision and decoder stay ungraphed')
    parser.add_argument('--full-cuda-graphs',action='store_true',help='Replay original vision, prefill-layer, full decode and selector tensor operations with CUDA Graphs')
    parser.add_argument('--graph-max-shapes',type=int,default=8)
    parser.add_argument('--fixed-greedy',action='store_true',help='Use the same fixed-length greedy loop as optimized base instead of HF generate')
    parser.add_argument('--packed-decode-kv',action='store_true',help='Use packed KV input/output for full decode CUDA Graphs when layer cache shapes match')
    parser.add_argument('--exact-rope-projections',action='store_true',help='Use the same exact RoPE/projection runtime fusion as optimized base')
    parser.add_argument('--exact-norms',action='store_true',help='Existing bitwise-exact RMSNorm kernel, with original FP32 reduction and BF16 rounding')
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--input-cache',default=str(INPUT_CACHE))
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--retentions',nargs='+',type=float,default=[.05,.2])
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    parser.add_argument('--indices',nargs='+',type=int)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--tokens',type=int,default=8)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--retention',type=float)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    assert args.tokens>=2 and args.runs>=1 and 0<=args.shard<args.shards
    assert not args.selector_graphs or args.execution=='fa2_metadata'
    assert not args.full_cuda_graphs or args.execution=='fa2_metadata'
    assert not args.exact_norms or args.execution=='fa2_metadata'
    if args.worker:
        assert args.method and 0<args.retention<1
        worker(args)
    elif args.aggregate:aggregate(args)
    else:launch(args)


if __name__ == '__main__':
    main()
