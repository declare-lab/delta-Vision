"""Locate why optimized adapter/base decode have the same throughput.

Both use one model instance and identical native decode graphs/kernels. Profiling
timings diagnose components; the saved 999-row speed tables remain authoritative.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8',
                 TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')

import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.benchmark_adapter_optimizations import MODEL, CHECKPOINT, MANIFEST, tensor_sha
from src.benchmark_prefill import build_qwen_fast_adapter_prefill
from src.data import QwenBenchmarkDataset
from src.model import load_qwen_embedding_adapter_checkpoint
from src.qwen_attention_metadata import optimize_qwen_attention_metadata
from src.qwen_deepstack import disable_qwen_deepstack
from src.qwen_native_graph import NativeDecodeGraph, copy_tree, clone_tree, clone_kv_tensors
from decode_operator_breakdown import analyze_trace

OUTPUT = ROOT / 'test/results/adapter_exact_20260915/decode_base_parity'
INPUT_CACHE = ROOT / 'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme'


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, torch.device('cuda:0'), 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    optimize_qwen_attention_metadata(model)
    adapter, _ = load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT), model.model.language_model, torch.device('cuda:0'), torch.bfloat16)
    settings = SimpleNamespace(last_logits_only=True, attn_implementation='flash_attention_2',
        cuda_graph=True, cuda_graph_context=True, compile_verify=True, compile_max_diff=0.,
        cuda_graph_warmup=3, adapter_decode_cache_mode='fast', adapter_exact_optimizations=True)
    prefill = build_qwen_fast_adapter_prefill(model, adapter, settings)
    decoder = model._adapter_decode_graph_runner
    decoder.native.max_shapes = 32
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=str(INPUT_CACHE))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos

    def prefix(inputs, method):
        model.model.rope_deltas = None
        if method == 'adapter':
            logits, _, _, _, cache = prefill(inputs)
            return logits, cache
        positions = model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
        output = model(**inputs, position_ids=positions, use_cache=True, return_dict=True, logits_to_keep=1)
        cache = dict(_native_cache=output.past_key_values,
            _native_positions=positions[:, :, -1:] + torch.arange(1, 9, device=positions.device), _native_steps=0)
        return output.logits, cache

    def token(logits):
        scores = logits[:, -1].float().clone()
        scores[:, eos] = -float('inf')
        return scores.argmax(-1).view(1, 1)

    def request(inputs, method, *, check=False):
        logits, cache = prefix(inputs, method)
        prefix_lengths = [l.keys.shape[-2] for l in cache['_native_cache'].layers]
        times, hashes = [], []
        for step in range(7):
            next_token = token(logits)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits, cache = decoder(model, adapter, next_token, cache, logits_to_keep=1)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
            if check:
                hashes.append(tensor_sha([logits]))
        return dict(decode_ms=statistics.mean(times), steps_ms=times, logits_hashes=hashes,
                    prefix_lengths=prefix_lengths)

    components, rows, active_method = [], [], None

    def measured_replay(graph, kwargs, cache, rope):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        events[0].record()
        copy_tree(graph.inputs, (kwargs, [(l.keys, l.values) for l in cache.layers], rope))
        events[1].record()
        graph.graph.replay()
        events[2].record()
        values = clone_tree({k:v for k,v in graph.output.items() if k != 'past_key_values'})
        copied = clone_kv_tensors([t for pair in graph.output_kv for t in pair])
        for layer, keys, vals in zip(cache.layers, copied[::2], copied[1::2]):
            layer.keys, layer.values = keys, vals
        values['past_key_values'] = cache
        result = type(graph.output)(**values)
        events[3].record()
        torch.cuda.synchronize()
        components.append(dict(method=active_method, wall_ms=(time.perf_counter()-start)*1000,
            input_copy_ms=events[0].elapsed_time(events[1]),
            graph_ms=events[1].elapsed_time(events[2]),
            output_copy_ms=events[2].elapsed_time(events[3])))
        return result

    with torch.inference_mode():
        for index in [0, 333, 666]:
            inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda:0'))
            decoder.enabled, decoder.allow_capture = True, True
            expected = {m: request(inputs, m, check=True) for m in ['base', 'adapter']}
            decoder.allow_capture = False
            before = decoder.stats()
            trials = []
            for repetition in range(8):
                pair = {}
                for method in (['base', 'adapter'] if repetition % 2 == 0 else ['adapter', 'base']):
                    pair[method] = request(inputs, method)
                trials.append(pair)
            assert decoder.stats()['captures'] == before['captures']
            assert decoder.stats()['cold_fallbacks'] == before['cold_fallbacks']
            row = dict(index=index, prefix_lengths={m:v['prefix_lengths'] for m,v in expected.items()},
                decode_ms={m:statistics.median(t[m]['decode_ms'] for t in trials) for m in ['base','adapter']})
            begin = len(components)
            with patch.object(NativeDecodeGraph, 'replay', measured_replay):
                for repetition in range(4):
                    for method in ['base', 'adapter']:
                        active_method = method
                        checked = request(inputs, method, check=True)
                        assert checked['logits_hashes'] == expected[method]['logits_hashes']
            row['instrumented_components_ms'] = {m:{k:statistics.median(r[k] for r in components[begin:] if r['method']==m)
                for k in ['wall_ms', 'input_copy_ms', 'graph_ms', 'output_copy_ms']} for m in ['base','adapter']}
            rows.append(row)
            print(json.dumps(row), flush=True)
            (OUTPUT/'components.json').write_text(json.dumps(dict(rows=rows, components=components,
                note='CUDA-event component diagnostics add instrumentation; use uninstrumented decode_ms for the paired pilot.'), indent=2))
        # Attribute the native optimized kernels, with graph dispatch disabled so
        # profiler can link every kernel to its exact generating operation.
        from transformers.cache_utils import DynamicCache
        import transformers.integrations.flash_attention as fa
        import src.qwen_exact_rope as rope_module
        profiles = {}
        inputs = _qwen_inputs_from_item(dataset[0], torch.device('cuda:0'))
        for method in ['base', 'adapter']:
            decoder.enabled = True
            logits, cache = prefix(inputs, method)
            next_token = token(logits)
            handles, pending = [], defaultdict(list)
            for name, module in model.named_modules():
                label = ('lm_head' if name == 'lm_head' else 'mlp_linear' if '.mlp.' in name else 'attention_projection') if isinstance(module, torch.nn.Linear) else 'rmsnorm' if 'RMSNorm' in type(module).__name__ else None
                if label is None:
                    continue
                def before(current, args, _label=label):
                    scope = torch.profiler.record_function('decode/'+_label)
                    pending[id(current)].append(scope)
                    scope.__enter__()
                def after(current, args, result):
                    pending[id(current)].pop().__exit__(None, None, None)
                handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
            def scoped(function, label):
                def call(*args, **kwargs):
                    with torch.profiler.record_function('decode/'+label):
                        return function(*args, **kwargs)
                return call
            decoder.enabled = False
            try:
                with patch.object(fa, '_flash_attention_forward', scoped(fa._flash_attention_forward, 'attention_core')), \
                     patch.object(DynamicCache, 'update', scoped(DynamicCache.update, 'cache_update')), \
                     patch.object(rope_module, 'exact_rope', scoped(rope_module.exact_rope, 'rope')), \
                     torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function('decode/other'):
                        result = decoder(model, adapter, next_token, cache, logits_to_keep=1)
                        torch.cuda.synchronize()
                trace = OUTPUT/f'{method}.trace.json'
                prof.export_chrome_trace(str(trace))
                profiles[method] = analyze_trace(trace)
                (OUTPUT/f'{method}.ops.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
            finally:
                for handle in handles:
                    handle.remove()
                decoder.enabled = True
        (OUTPUT/'operator_summary.json').write_text(json.dumps(profiles, indent=2))
        print(json.dumps(profiles), flush=True)
    decoder.remove()


if __name__ == '__main__':
    main()
