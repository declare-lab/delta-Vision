"""Attribute native one-token decode kernels; replay captured attention separately."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def analyze_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    cpu = {e.get("args", {}).get("External id"): e for e in events if e.get("cat") == "cpu_op"}
    scopes = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("decode/")]
    totals = defaultdict(lambda: dict(kernel_us=0., kernels=0))
    for e in events:
        if e.get("cat") != "kernel":
            continue
        origin = cpu.get(e.get("args", {}).get("External id"))
        if origin is None:
            continue
        containing = [s for s in scopes if s["tid"] == origin["tid"] and s["ts"] <= origin["ts"] < s["ts"] + s["dur"]]
        if not containing:
            continue
        label = min(containing, key=lambda s: s["dur"])["name"]
        totals[label]["kernel_us"] += e["dur"]
        totals[label]["kernels"] += 1
    whole = [s for s in scopes if s["name"] == "decode/other"]
    scalars = [e for e in cpu.values() if e["name"] == "aten::_local_scalar_dense" and any(s["ts"] <= e["ts"] < s["ts"] + s["dur"] for s in whole)]
    return dict(kernel_groups=dict(totals), decode_cpu_scope_us=[s["dur"] for s in whole],
                decode_scalar_count=len(scalars), decode_scalar_us=sum(e["dur"] for e in scalars),
                note="Each GPU kernel attributed through its originating CPU op External id to the innermost decode scope. Kernel time excludes gaps, host work and contention waits. Profiler overhead affects scope durations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--methods", nargs="+", default=["base", "divprune"])
    parser.add_argument("--deepstack", choices=['native', 'off'], default='native')
    parser.add_argument("--output", default='test/results/decode_operator_breakdown_20260915')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import transformers.integrations.flash_attention as fa
    from transformers.cache_utils import DynamicCache
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.data import QwenBenchmarkDataset
    from src.benchmarking.common.prefill import set_global_seed
    from src.attention import optimize_qwen_attention_metadata
    from src.benchmarking.common.generation_timing import GenerationStageTimer
    torch.set_num_threads(4)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    captured = {}
    original_fa = fa._flash_attention_forward
    original_update = DynamicCache.update

    for method in args.methods:
        set_global_seed(42)
        model, processor = load_baseline_model(method,
            str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
            torch.bfloat16, torch.device("cuda:0"), .05, "flash_attention_2")
        if args.deepstack == 'off':
            from src.model_setup import disable_qwen_deepstack
            disable_qwen_deepstack(model)
        data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
        dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
        inputs = _qwen_inputs_from_item(dataset[0], torch.device("cuda:0"))
        visual = inputs["mm_token_type_ids"][0].nonzero().flatten()
        configure_baseline(model, method, .05, int(visual[0]), len(visual))
        optimization = optimize_qwen_attention_metadata(model)
        timer = GenerationStageTimer(model)

        def request():
            set_global_seed(42)
            model.model.rope_deltas = None
            return model.generate(**inputs, max_new_tokens=8, do_sample=False)

        with torch.inference_mode():
            for _ in range(3):
                request()
            timer.begin()
            import time
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = request()
            torch.cuda.synchronize()
            plain = timer.finish(time.perf_counter() - start, value.shape[-1] - inputs["input_ids"].shape[-1])
        timer.remove()
        active = False
        whole_scope = None
        records = []
        handles = []
        pending = defaultdict(list)

        def before_model(module, positional, kwargs):
            nonlocal active, whole_scope
            active = kwargs["input_ids"].shape[-1] == 1
            if active:
                whole_scope = torch.profiler.record_function("decode/other")
                whole_scope.__enter__()

        def after_model(module, positional, kwargs, output):
            nonlocal active
            if active:
                whole_scope.__exit__(None, None, None)
            active = False

        handles += [model.register_forward_pre_hook(before_model, with_kwargs=True),
                    model.register_forward_hook(after_model, with_kwargs=True)]

        def hook_scope(module, label):
            def before(current, positional):
                scope = torch.profiler.record_function("decode/" + label) if active else None
                pending[id(current)].append(scope)
                if scope:
                    scope.__enter__()
            def after(current, positional, output):
                scope = pending[id(current)].pop()
                if scope:
                    scope.__exit__(None, None, None)
            handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                label = "lm_head" if name == "lm_head" else "mlp_linear" if ".mlp." in name else "attention_projection"
                hook_scope(module, label)
            elif "TextRMSNorm" in type(module).__name__:
                hook_scope(module, "rmsnorm")

        def wrapped_fa(*positional, **kwargs):
            if not active:
                return original_fa(*positional, **kwargs)
            records.append((positional, kwargs))
            with torch.profiler.record_function("decode/attention_core"):
                return original_fa(*positional, **kwargs)

        def wrapped_update(cache, *positional, **kwargs):
            if not active:
                return original_update(cache, *positional, **kwargs)
            with torch.profiler.record_function("decode/cache_update"):
                return original_update(cache, *positional, **kwargs)

        fa._flash_attention_forward = wrapped_fa
        DynamicCache.update = wrapped_update
        try:
            with torch.inference_mode(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                result = request()
                torch.cuda.synchronize()
            assert torch.equal(value, result)
            prof.export_chrome_trace(str(out / f"{method}.trace.json"))
        finally:
            fa._flash_attention_forward = original_fa
            DynamicCache.update = original_update
            for handle in handles:
                handle.remove()
            optimization.remove()
        assert len(records) == 36, len(records)
        captured[method] = records
        shapes = [dict(layer=i, q=list(p[0].shape), k=list(p[1].shape), v=list(p[2].shape), mask=None if p[3] is None else list(p[3].shape)) for i, (p, k) in enumerate(records)]
        row = dict(method=method, shared_gpu=True, sample=0, retention=.05, plain_stages=plain,
                   decode_attention_shapes=shapes, profile=analyze_trace(out / f"{method}.trace.json"))
        summaries.append(row)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(row), flush=True)
        del model, module
        gc.collect()
        torch.cuda.empty_cache()

    # Replay the exact 36 attention calls with captured Q/K/V, without model or Python dispatch gaps.
    # This diagnoses attention itself; it is NOT an end-to-end decode speed measurement.
    graphs = {}
    with torch.inference_mode():
        for method, records in captured.items():
            for _ in range(3):
                expected = [original_fa(*p, **k) for p, k in records]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                results = [original_fa(*p, **k) for p, k in records]
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(expected, results))
            graphs[method] = (graph, results)
        pairs = []
        for repetition in range(30):
            pair = {}
            for method in (args.methods if repetition % 2 == 0 else args.methods[::-1]):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    graphs[method][0].replay()
                end.record()
                end.synchronize()
                pair[method] = begin.elapsed_time(end) / 20
            pairs.append(pair)
    (out / "attention_replay.json").write_text(json.dumps(dict(pairs_ms= pairs,
        medians_ms={m: statistics.median(p[m] for p in pairs) for m in args.methods},
        shared_gpu=True, scope="Exact attention-only graph replay, 36 layer calls, real q/k/v from first cached decode; excludes projections, norms, cache update, MLP, head and native dispatch."), indent=2))
    print("ATTENTION", {m: statistics.median(p[m] for p in pairs) for m in args.methods}, flush=True)


if __name__ == "__main__":
    main()
