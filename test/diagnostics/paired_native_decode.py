"""Interleave native base/pruning generation on identical inputs on a shared GPU."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 25, 125, 133])
    parser.add_argument("--retentions", type=float, nargs="+", default=[.05, .2])
    parser.add_argument("--native-cuda-graphs", action="store_true")
    parser.add_argument("--output-dir", default="test/results/prefill_decode_corrected_20260915/paired_decode")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    import torch
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.benchmarking.common.prefill import set_global_seed
    from src.data import QwenBenchmarkDataset
    from src.benchmarking.common.generation_timing import GenerationStageTimer
    from src.attention import optimize_qwen_attention_metadata
    from src.graphs import NativeDecoderGraphs

    torch.set_num_threads(4)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct")
    device = torch.device("cuda:0")
    base, processor = load_baseline_model("base", model_path, torch.bfloat16, device, 1., "flash_attention_2")
    base_metadata = optimize_qwen_attention_metadata(base)
    base_timer = GenerationStageTimer(base)
    base_graphs = NativeDecoderGraphs(base) if args.native_cuda_graphs else None
    data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
    inputs_by_sample = {i: _qwen_inputs_from_item(dataset[i], device) for i in args.samples}
    protocol = dict(vars(args), shared_gpu=True, native_generation=True, max_new_tokens=8,
                    stop="original EOS", metadata_optimization=True,
                    ordering="alternate base/method then method/base each repetition",
                    timing="synchronized wall time of actual native model forwards; no custom decode loop",
                    model=model_path, data=str(data), argv=sys.argv)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def request(model, timer, inputs):
        set_global_seed(42)
        model.model.rope_deltas = None
        torch.cuda.synchronize(device)
        timer.begin()
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize(device)
        total = time.perf_counter() - start
        tokens = output[0, inputs["input_ids"].shape[-1]:].tolist()
        result = timer.finish(total, len(tokens))
        result.update(total_time_s=total, tokens=tokens)
        return result

    summaries = []
    with torch.inference_mode():
        for method in ["fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"]:
            model, _ = load_baseline_model(method, model_path, torch.bfloat16, device, .05, "flash_attention_2")
            metadata = optimize_qwen_attention_metadata(model)
            timer = GenerationStageTimer(model)
            method_graphs = NativeDecoderGraphs(model) if args.native_cuda_graphs else None
            for retention in args.retentions:
                trials = []
                for sample, inputs in inputs_by_sample.items():
                    visual = inputs["mm_token_type_ids"][0].nonzero().flatten()
                    configure_baseline(base, "base", 1., int(visual[0]), len(visual))
                    configure_baseline(model, method, retention, int(visual[0]), len(visual))
                    if args.native_cuda_graphs:
                        base_graphs.allow_capture = method_graphs.allow_capture = True
                    expected = {}
                    for _ in range(3):
                        for label, current, observer in [("base", base, base_timer), ("method", model, timer)]:
                            expected[label] = request(current, observer, inputs)["tokens"]
                    if args.native_cuda_graphs:
                        base_graphs.allow_capture = method_graphs.allow_capture = False
                        before = [g.stats() for g in [base_graphs, method_graphs]]
                    for repetition in range(args.repetitions):
                        pair = dict(sample=sample, repetition=repetition)
                        order = [("base", base, base_timer), ("method", model, timer)]
                        if repetition % 2:
                            order.reverse()
                        for label, current, observer in order:
                            pair[label] = request(current, observer, inputs)
                            assert pair[label]["tokens"] == expected[label], (method, retention, sample, label)
                        trials.append(pair)
                    if args.native_cuda_graphs:
                        for graph, previous in zip([base_graphs, method_graphs], before):
                            now = graph.stats()
                            assert now["captures"] == previous["captures"]
                            assert now["cold_layer_fallbacks"] == previous["cold_layer_fallbacks"]
                summary = dict(method=method, retention=retention, pairs=len(trials), shared_gpu=True)
                for label in ["base", "method"]:
                    steps = [s["seconds"] * 1000 for pair in trials for s in pair[label]["generation_stages"][1:]]
                    assert steps, (method, retention, label)
                    summary[label + "_metrics"] = dict(decode_median_ms=statistics.median(steps),
                                          decode_mean_ms=statistics.mean(steps),
                                          decode_steps=len(steps),
                                          prefill_median_ms=statistics.median(p[label]["generation_prefill_time_s"] * 1000 for p in trials))
                ratios = [(p["base"]["decode_time_s"] / p["base"]["decode_steps"]) /
                          (p["method"]["decode_time_s"] / p["method"]["decode_steps"]) for p in trials]
                summary["paired_decode_speedup_median"] = statistics.median(ratios)
                summary["method_faster_pairs"] = sum(r > 1 for r in ratios)
                summaries.append(summary)
                (out / f"{method}_ret{int(retention*100):02d}.json").write_text(json.dumps(trials, indent=2))
                (out / "summary.json").write_text(json.dumps(summaries, indent=2))
                print(json.dumps(summary), flush=True)
            if method_graphs is not None:
                method_graphs.remove()
                del method_graphs
            timer.remove()
            metadata.remove()
            del model
            # Loop variables retain the last module until explicitly cleared.
            del current, observer, order
            gc.collect()
            torch.cuda.empty_cache()
    if base_graphs is not None:
        base_graphs.remove()
    base_timer.remove()
    base_metadata.remove()


if __name__ == "__main__":
    main()
