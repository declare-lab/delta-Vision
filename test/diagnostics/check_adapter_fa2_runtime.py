"""Verify real FA2 dispatch and FA2 graph/eager parity on the adapter fast path."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.model import load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint
from src.model import qwen_embedding_adapter_decode_step, qwen_embedding_adapter_decode_step_shape_exact
from src.benchmark_prefill import build_qwen_fast_adapter_prefill
from src.data import QwenBenchmarkDataset
from baselines.eval_baselines import _qwen_inputs_from_item


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    processor, model = load_frozen_qwen3vl("/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct",
        torch.bfloat16, device, "flash_attention_2")
    adapter, _ = load_qwen_embedding_adapter_checkpoint(str(ROOT / "artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt"),
        model.model.language_model, device, torch.bfloat16)
    path = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(path), processor, "mmstar", data_root=str(path.parent), max_samples=200)
    args = SimpleNamespace(last_logits_only=True, attn_implementation="flash_attention_2", cuda_graph=False,
        cuda_graph_context=True, compile_verify=True, compile_max_diff=0., cuda_graph_warmup=3)
    eager = build_qwen_fast_adapter_prefill(model, adapter, args)
    args.cuda_graph = True
    graphed = build_qwen_fast_adapter_prefill(model, adapter, args)
    rows = []
    def forbid_sdpa(*args, **kwargs):
        raise AssertionError("Unexpected SDPA fallback in adapter FA2 execution")
    with torch.inference_mode(), patch("torch.nn.functional.scaled_dot_product_attention", forbid_sdpa):
        for index in [0, 25, 125, 133]:
            inputs = _qwen_inputs_from_item(dataset[index], device)
            ref = eager(inputs)
            actual = graphed(inputs)
            assert torch.equal(ref[0], actual[0]), index
            for a, b in zip(ref[-1]["layers"], actual[-1]["layers"]):
                for key in a:
                    assert torch.equal(a[key], b[key]), (index, key)
            row = dict(index=index, graph_eager_logits_kv_exact=True, sdpa_calls=0)
            # Exercise both public decode implementations, even though MMStar
            # usually stops immediately after the first answer token.
            for decode in [qwen_embedding_adapter_decode_step, qwen_embedding_adapter_decode_step_shape_exact]:
                cache = graphed(inputs)[-1]
                token = ref[0][:, -1].argmax(-1).view(1, 1)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    for step in range(3):
                        logits, cache = decode(model, adapter, token, cache)
                        assert torch.isfinite(logits).all()
                        token = logits[:, -1].argmax(-1).view(1, 1)
                events = {e.key: e.count for e in prof.key_averages() if "flash_attn" in e.key.lower()}
                assert events, "No FA2 operator observed"
                row[decode.__name__] = dict(steps=3, operators=events)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                eager(inputs)
            row["prefill_operators"] = {e.key: e.count for e in prof.key_averages() if "flash_attn" in e.key.lower()}
            assert row["prefill_operators"]
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "test/results/all_fa2_20260915/adapter_runtime_check.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
