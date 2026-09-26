"""Isolated Qwen3-VL adapter ablations; production inference stays unchanged."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def install_probe(model, mode, path, reuse_buffers=True, profile=False, share_rotary=True, cuda_profile=False, fused_qk=False, flat_inputs=False, attention_splits=0, batch_visual_kv=False, reuse_indices=False):
    import torch
    import src.vllm_graphs as graphs
    assert model.config.model_type == 'qwen3_vl'
    model.fast_prefill = mode in ('split_q', 'combined', 'all')
    model.batch_memories = mode in ('batch_mlp', 'combined', 'all')
    model.reuse_prefill_buffers = reuse_buffers
    model.batch_visual_kv = batch_visual_kv
    model._runtime_graphs = graphs.RuntimeGraphs(model)
    runtime = model._runtime_graphs
    runtime.fused_qk = fused_qk
    runtime.flat_inputs = flat_inputs
    runtime.attention_splits = attention_splits
    runtime.reuse_indices = reuse_indices
    runtime.share_rotary = runtime.share_rotary and share_rotary
    runtime.fast_metadata = mode in ('metadata', 'all')
    runtime._probe_events = []
    original = torch.cuda.CUDAGraph.replay
    def replay(graph):
        phase = None
        for kind, entries in runtime.entries.items():
            if any(entry[-2] is graph for entry in entries.values()): phase = kind
        if runtime.head_entry is not None and runtime.head_entry[2] is graph: phase = 'head'
        if phase is None: return original(graph)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = original(graph)
        end.record()
        runtime._probe_events.append((phase, start, end))
        return result
    torch.cuda.CUDAGraph.replay = replay
    model._probe_dest = path
    if profile:
        import cProfile
        runtime._probe_cpu = cProfile.Profile()
        runtime._probe_cpu_done = False
        forward = model.forward
        def profiled(*args, _forward=forward, **kwargs):
            if runtime.allow_capture or runtime._probe_cpu_done:
                return _forward(*args, **kwargs)
            return runtime._probe_cpu.runcall(_forward, *args, **kwargs)
        model.forward = profiled
    if cuda_profile:
        forward = model.forward
        profiled_phases = set()
        def gpu_profiled(*args, **kwargs):
            x = kwargs.get('inputs_embeds')
            if x is None: x = kwargs['input_ids']
            phase = 'decode' if x.shape[0] == 1 else 'prefill'
            if runtime.allow_capture or phase in profiled_phases:
                return forward(*args, **kwargs)
            profiled_phases.add(phase)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                result = forward(*args, **kwargs)
            dest = Path(path).parent
            prof.export_chrome_trace(str(dest/f'{phase}_trace.json'))
            (dest/f'{phase}_kernels.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
            return result
        model.forward = gpu_profiled
    return runtime.stats()


def main(a):
    import src.vllm_graphs as graphs
    import src.benchmarking.realworldqa_vllm as bench
    path = str(Path(a.run).absolute()/'gpu_replays.jsonl')
    def install(model): return install_probe(model, a.mode, path, a.reuse_prefill_buffers, a.profile, a.share_rotary, a.cuda_profile, a.fused_qk, a.flat_inputs, a.attention_splits, a.batch_visual_kv, a.reuse_indices)
    graphs.install_runtime_graphs = install
    if a.validate:
        import importlib.util
        spec = importlib.util.spec_from_file_location('graph_validation', Path(__file__).with_name('test_vllm_graphs.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(SimpleNamespace(family='qwen', method=a.method, samples=a.samples,
                                    output=a.run, hf_reference=False, allow_rounding_differences=True))
    else:
        # Avoid recursively patching the callable used inside the worker callback.
        original = bench.take_execution_times
        def take(model):
            values = original(model)
            result = dict(outer_s=values, replay_s={})
            for phase, start, end in model._runtime_graphs._probe_events:
                result['replay_s'].setdefault(phase, []).append(start.elapsed_time(end)/1000.)
            model._runtime_graphs._probe_events.clear()
            with open(model._probe_dest, 'a') as f: f.write(json.dumps(result)+'\n')
            runtime = model._runtime_graphs
            if hasattr(runtime, '_probe_cpu') and not runtime.allow_capture and not runtime._probe_cpu_done:
                import pstats
                runtime._probe_cpu_done = True
                with open(str(Path(model._probe_dest).with_name('model_cpu_profile.txt')), 'w') as f:
                    pstats.Stats(runtime._probe_cpu, stream=f).sort_stats('cumulative').print_stats(60)
            return values
        bench.take_execution_times = take
        bench.worker(SimpleNamespace(run=a.run, family='qwen', method=a.method, backend='vllm',
                                     shard=0, shards=1, limit=a.samples, repeats=3))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=['current', 'split_q', 'batch_mlp', 'combined', 'metadata', 'all'], required=True)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--method', choices=['base', 'adapter'], default='adapter')
    p.add_argument('--reuse-prefill-buffers', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--share-rotary', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cuda-profile', action='store_true')
    p.add_argument('--fused-qk', action='store_true')
    p.add_argument('--flat-inputs', action='store_true')
    p.add_argument('--attention-splits', type=int, choices=[0, 1], default=0)
    p.add_argument('--batch-visual-kv', action='store_true')
    p.add_argument('--reuse-indices', action='store_true')
    main(p.parse_args())
