"""Profile canonical baseline generate on an idle GPU, with per-forward CUDA events."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["base", "divprune"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--retention", type=float, default=.05)
    parser.add_argument("--allow-busy", action="store_true", help="Explicit shared-GPU profiling; not isolated latency")
    parser.add_argument("--optimize-metadata", action="store_true")
    parser.add_argument("--paired-original", action="store_true", help="Alternate disabled/enabled metadata fix on the same model/input")
    parser.add_argument("--deepstack", choices=['off','native'], default='off')
    parser.add_argument("--screenshot-fastv", action="store_true")
    parser.add_argument("--output", default="test/results/prefill_slowdown_investigation_20260915/gpu_profile")
    args = parser.parse_args()
    processes = subprocess.check_output(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    live = []
    for value in processes.splitlines():
        state = Path(f"/proc/{value.strip()}/status").read_text()
        if not any(line.startswith("State:") and "T (stopped)" in line for line in state.splitlines()):
            live.append(value)
    if live and not args.allow_busy:
        raise SystemExit(f"GPU {args.physical_gpu} is occupied by PID(s) {processes}; refusing contested speed measurements")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.data import QwenBenchmarkDataset
    from src.benchmarking.common.prefill import set_global_seed
    from src.attention import optimize_qwen_attention_metadata
    torch.set_num_threads(4)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in args.methods:
        set_global_seed(42)
        model, processor = load_baseline_model(method,
            str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
            torch.bfloat16, torch.device("cuda:0"), args.retention, "flash_attention_2")
        if args.deepstack == 'off':
            from src.model_setup import disable_qwen_deepstack
            disable_qwen_deepstack(model)
        if args.screenshot_fastv and method == 'fastv':
            from reproduce_screenshot_fastv import restore_screenshot_forward
            restore_screenshot_forward(model)
        data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
        dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
        inputs = _qwen_inputs_from_item(dataset[args.sample_index], torch.device("cuda:0"))
        visual = inputs["mm_token_type_ids"][0].nonzero().flatten()
        configure_baseline(model, method, args.retention, int(visual[0]), len(visual))
        metadata_optimization = optimize_qwen_attention_metadata(model) if args.optimize_metadata else None
        calls, event_records, handles = [], [], []

        def request():
            model.model.rope_deltas = None
            return model.generate(**inputs, max_new_tokens=8, do_sample=False)

        def hook_pair(label):
            pending = []
            def pre(module, positional, kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                scope = torch.profiler.record_function(label)
                scope.__enter__()
                start.record()
                pending.append((time.perf_counter(), start, end, scope))
                if label == "model_forward":
                    cache = kwargs.get("past_key_values")
                    calls.append({"input_tokens": kwargs["input_ids"].shape[-1],
                        "has_pixels": kwargs.get("pixel_values") is not None,
                        "cache_lengths": [layer.get_seq_length() for layer in cache.layers] if cache is not None else None})
            def post(module, positional, kwargs, result):
                begin, start, end, scope = pending.pop()
                end.record()
                event_records.append((label, time.perf_counter() - begin, start, end))
                scope.__exit__(None, None, None)
            return pre, post

        with torch.inference_mode():
            for _ in range(3):
                request()
            paired_trials = []
            if args.paired_original:
                if metadata_optimization is None:
                    raise ValueError("--paired-original requires --optimize-metadata")
                for repetition in range(12):
                    trial = {}
                    outputs = []
                    for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                        metadata_optimization.enabled = enabled
                        set_global_seed(42)
                        model.model.rope_deltas = None
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        direct = model(**inputs, logits_to_keep=1)
                        torch.cuda.synchronize()
                        trial['optimized_prefill_ms' if enabled else 'original_prefill_ms'] = (time.perf_counter()-start)*1000
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        value = request()
                        torch.cuda.synchronize()
                        trial["optimized_ms" if enabled else "original_ms"] = (time.perf_counter() - start) * 1000
                        outputs.append(value)
                    trial["tokens_identical"] = torch.equal(*outputs)
                    paired_trials.append(trial)
                metadata_optimization.enabled = True
            plain_ms = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = request()
                torch.cuda.synchronize()
                plain_ms.append((time.perf_counter() - start) * 1000)
            stage_requests = []
            active_stage = []
            def stage_pre(module, positional, kwargs):
                kind = "prefill" if kwargs["input_ids"].shape[-1] > 1 else "decode"
                torch.cuda.synchronize()
                active_stage.append((kind, time.perf_counter()))
            def stage_post(module, positional, kwargs, output):
                torch.cuda.synchronize()
                kind, begin = active_stage.pop()
                stage_requests[-1][kind + "_forward_ms"].append((time.perf_counter() - begin) * 1000)
            stage_handles = [model.register_forward_pre_hook(stage_pre, with_kwargs=True),
                             model.register_forward_hook(stage_post, with_kwargs=True)]
            for _ in range(5):
                row = {"prefill_forward_ms": [], "decode_forward_ms": []}
                stage_requests.append(row)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = request()
                torch.cuda.synchronize()
                row["total_ms"] = (time.perf_counter() - start) * 1000
                row["new_token_ids"] = result[0, inputs["input_ids"].shape[-1]:].tolist()
                row["generation_bookkeeping_ms"] = row["total_ms"] - sum(row["prefill_forward_ms"]) - sum(row["decode_forward_ms"])
            for handle in stage_handles:
                handle.remove()
            for label, module in [("model_forward", model), ("vision", model.model.visual),
                                  ("language", model.model.language_model), ("lm_head", model.lm_head)]:
                pre, post = hook_pair(label)
                handles.extend([module.register_forward_pre_hook(pre, with_kwargs=True),
                                module.register_forward_hook(post, with_kwargs=True)])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                result = request()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out / f"{method}.trace.json"))
            (out / f"{method}.operators.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        for handle in handles:
            handle.remove()
        row = {"method": method, "physical_gpu": args.physical_gpu, "retention": args.retention,
               "deepstack": args.deepstack, "screenshot_fastv": args.screenshot_fastv,
               "shared_gpu": args.allow_busy, "metadata_optimized": args.optimize_metadata,
               "paired_trials": paired_trials,
               "sample_index": args.sample_index, "total_without_profiler_ms": plain_ms,
               "synchronized_stage_requests": stage_requests,
               "new_token_ids": result[0, inputs["input_ids"].shape[-1]:].tolist(),
               "decoded_text": processor.tokenizer.decode(result[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True),
               "calls": calls, "profiled_stages": [{"label": label, "host_ms": host * 1000,
                   "cuda_event_ms": start.elapsed_time(end)} for label, host, start, end in event_records],
               "note": "Profiler/CUDA-event stages are diagnostic; total_without_profiler_ms uses uninstrumented original generate"}
        summaries.append(row)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(row), flush=True)
        if metadata_optimization is not None:
            metadata_optimization.remove()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
