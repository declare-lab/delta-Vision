"""Trace real baseline generation control flow with tiny CPU models; no timing claims."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, LogitsProcessor
from baselines.eval_baselines import configure_baseline
from src.benchmarking.common.generation_timing import GenerationStageTimer


class TwoTokens(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores.fill_(-float("inf"))
        scores[:, 3 if input_ids.shape[-1] == 8 else 2] = 0
        return scores


def main():
    torch.set_num_threads(1)
    results = []
    for name in ["base", "fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"]:
        cfg = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000., "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                               patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                               out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
            image_token_id=49, video_token_id=48, vision_start_token_id=47, vision_end_token_id=46,
            bos_token_id=1, eos_token_id=2, pad_token_id=0)
        cfg._attn_implementation = "sdpa"
        cls = Qwen3VLForConditionalGeneration
        if name != "base":
            path = ROOT / f"baselines/{name}/qwen3_vl/modeling_qwen3_vl_{name}.py"
            spec = importlib.util.spec_from_file_location(f"cache_trace_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            cls = mod.Qwen3VLForConditionalGeneration
        model = cls(cfg).eval()
        configure_baseline(model, name, .5, 2, 4)
        row = {"method": name, "calls": [], "vision_calls": 0}

        def pre(module, positional, kwargs):
            cache = kwargs.get("past_key_values")
            row["calls"].append({"input_tokens": kwargs["input_ids"].shape[-1],
                "has_pixels": kwargs.get("pixel_values") is not None,
                "use_cache": kwargs.get("use_cache"),
                "cache_lengths_before": [layer.get_seq_length() for layer in cache.layers] if cache is not None else None})

        def vision(*unused):
            row["vision_calls"] += 1

        model.register_forward_pre_hook(pre, with_kwargs=True)
        model.model.visual.register_forward_pre_hook(vision)
        ids = torch.tensor([[1, 47, 49, 49, 49, 49, 46, 4]])
        timer = GenerationStageTimer(model)
        timer.begin()
        begin = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                mm_token_type_ids=(ids == 49).long(), image_grid_thw=torch.tensor([[1, 4, 4]]),
                pixel_values=torch.randn(16, 12), do_sample=False, max_new_tokens=2,
                eos_token_id=2, pad_token_id=0, logits_processor=[TwoTokens()])
        row["stages"] = timer.finish(time.perf_counter() - begin, int(output.shape[-1] - 8))
        timer.remove()
        row["new_tokens"] = output[0, 8:].tolist()
        assert [c["input_tokens"] for c in row["calls"]] == [8, 1], row
        assert row["vision_calls"] == 1, row
        assert all(n > 0 for n in row["calls"][1]["cache_lengths_before"]), row
        results.append(row)
        print(json.dumps(row), flush=True)
    out = ROOT / "test/results/prefill_slowdown_investigation_20260915"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu_cache_trace.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
