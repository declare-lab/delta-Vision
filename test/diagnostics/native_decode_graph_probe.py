"""Replay one fixed native cached forward to quantify dispatch cost, without changing kernels."""
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.attention import optimize_qwen_attention_metadata


def main():
    torch.set_num_threads(4)
    out = ROOT / "test/results/decode_operator_breakdown_20260915"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ["base", "divprune"]:
        model, processor = load_baseline_model(method,
            str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
            torch.bfloat16, torch.device("cuda:0"), .05, "flash_attention_2")
        optimization = optimize_qwen_attention_metadata(model)
        data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
        dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
        inputs = _qwen_inputs_from_item(dataset[0], torch.device("cuda:0"))
        vis = inputs["mm_token_type_ids"][0].nonzero().flatten()
        configure_baseline(model, method, .05, int(vis[0]), len(vis))
        captured = []

        def capture(module, args, kwargs):
            if kwargs["input_ids"].shape[-1] == 1 and not captured:
                captured.append(copy.deepcopy(kwargs))

        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        with torch.inference_mode():
            torch.manual_seed(42)
            model.generate(**inputs, max_new_tokens=8, do_sample=False)
        handle.remove()
        assert len(captured) == 1
        kwargs = captured[0]
        cache = kwargs["past_key_values"]
        initial = [(layer.keys, layer.values) for layer in cache.layers]

        def reset():
            for layer, (keys, values) in zip(cache.layers, initial):
                layer.keys, layer.values = keys, values

        def forward():
            reset()
            return model(**kwargs)

        with torch.inference_mode():
            native = forward()
            expected_logits = native.logits.clone()
            expected_kv = [(l.keys.clone(), l.values.clone()) for l in native.past_key_values.layers]
            # Native FA2 drops an all-ones mask; resolve that invariant before graph capture.
            mask = kwargs.get("attention_mask")
            assert mask is None or bool(torch.all(mask == 1))
            kwargs["attention_mask"] = None
            for _ in range(3):
                warm = forward()
            assert torch.equal(warm.logits, expected_logits)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            reset()
            with torch.cuda.graph(graph):
                result = model(**kwargs)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(result.logits, expected_logits)
            assert all(torch.equal(l.keys, k) and torch.equal(l.values, v)
                       for l, (k, v) in zip(result.past_key_values.layers, expected_kv))
            graph_cache = [(l.keys, l.values) for l in result.past_key_values.layers]
            trials = []
            for repeat in range(30):
                row = {}
                for kind in (["eager", "graph"] if repeat % 2 == 0 else ["graph", "eager"]):
                    if kind == "eager":
                        reset()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if kind == "eager":
                        eager = model(**kwargs)
                    else:
                        graph.replay()
                    torch.cuda.synchronize()
                    row[kind + "_ms"] = (time.perf_counter() - start) * 1000
                    assert torch.equal(eager.logits if kind == "eager" else result.logits, expected_logits)
                trials.append(row)
            assert all(torch.equal(gk, k) and torch.equal(gv, v)
                       for (gk, gv), (k, v) in zip(graph_cache, expected_kv))
        row = dict(method=method, shared_gpu=True, sample_index=0, trials=trials,
                   logits_and_kv_exact=True, native_eager_median_ms=statistics.median(r["eager_ms"] for r in trials),
                   native_graph_median_ms=statistics.median(r["graph_ms"] for r in trials),
                   scope="One fixed cached native forward at a fixed position and KV length; exact original kernels. Diagnostic only, not multi-token generation throughput. All-ones mask resolved before capture.")
        rows.append(row)
        (out / "native_graph_probe.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        optimization.remove()
        del model, graph
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
