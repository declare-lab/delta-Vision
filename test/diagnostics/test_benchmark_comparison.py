"""Resource accounting and cached decode after unequal per-layer pruning."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from src.benchmarking.common.comparison import cache_metrics, decoder_flops, native_cached_step, parse_methods, tensor_storage_bytes


class ComparisonTest(unittest.TestCase):
    def test_fast_prefill_replays_both_original_graph_runners(self):
        from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
        from src.benchmarking.common.comparison import RequestRunner
        args = SimpleNamespace(last_logits_only=True, cuda_graph=True, cuda_graph_context=True,
                               cuda_graph_warmup=3, compile_verify=True, compile_max_diff=0.)
        inputs = {key: torch.ones(1) for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
        hidden, positions, logits, mask = [torch.ones(1) for _ in range(4)]
        cache = {"layers": [], "layer_inputs": [], "layer_after_attention": []}
        model = SimpleNamespace(model=SimpleNamespace(rope_deltas=None,
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))))
        with patch("src.benchmarking.common.prefill.QwenContextCudaGraphRunner") as context_cls, \
             patch("src.benchmarking.common.prefill.QwenAdapterPrefillCacheCudaGraphRunner") as prefill_cls:
            context_cls.return_value = Mock(return_value=(hidden, positions))
            prefill_cls.return_value = Mock(return_value=(logits, mask, cache))
            fn = build_qwen_fast_adapter_prefill(model, object(), args)
            runner = RequestRunner(model, "embedding_adapter", object(), "fast", fn)
            actual, actual_cache, _ = runner.prefill(inputs)
            self.assertIs(actual, logits)
            self.assertIs(actual_cache, cache)
            context_cls.return_value.assert_called_once()
            prefill_cls.return_value.assert_called_once_with(inputs["input_ids"], inputs["attention_mask"],
                inputs["mm_token_type_ids"], hidden, positions, topology=None)

    def test_aliases_and_reference(self):
        self.assertEqual(parse_methods(["zooprune,visionzup", "fastv"]), ["base", "zoo", "visionzip", "fastv"])
        with self.assertRaises(ValueError):
            parse_methods(["bogus"])

    def test_cache_counts_visual_kv_and_shared_storage(self):
        tensor = torch.zeros(1, 2, 6, 4)
        layer = {"text_key": tensor[:, :, :2], "text_value": tensor[:, :, :2],
                 "visual_key": tensor[:, :, 2:], "visual_value": tensor[:, :, 2:]}
        state = {"layers": [layer], "layer_inputs": [torch.zeros(1, 2, 16)]}
        measured = cache_metrics(state, True)
        self.assertEqual(measured["layer_cache_lengths"], [6])
        self.assertEqual(measured["kv_cache_mb"] * 1024**2, tensor.numel() * 4)
        self.assertGreater(measured["decode_cache_mb"], measured["kv_cache_mb"])
        self.assertEqual(tensor_storage_bytes([tensor, tensor.view(-1)]), tensor.numel() * 4)

    def test_flops_uses_every_layer_length(self):
        cfg = SimpleNamespace(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=8, intermediate_size=64)
        full = decoder_flops(cfg, [10, 10, 10], text_tokens=2, image_tokens=8)
        staged = decoder_flops(cfg, [10, 6, 4], text_tokens=2, image_tokens=8)
        early = decoder_flops(cfg, [10, 4, 4], text_tokens=2, image_tokens=8)
        self.assertGreater(full, staged)
        self.assertGreater(staged, early)

    def test_cached_decode_matches_full_replay_with_original_positions(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        torch.manual_seed(42)
        cfg = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=50,
                               rope_parameters={"rope_type": "default", "rope_theta": 10000., "mrope_section": [1, 1, 2]})
        cfg._attn_implementation = "sdpa"
        language = Qwen3VLTextModel(cfg).eval()
        wrapper = SimpleNamespace(model=SimpleNamespace(language_model=language), lm_head=torch.nn.Linear(32, 50, bias=False))
        # Frozen pruning decision at layer 1 removes original prompt positions 1,3.
        def prune(module, args, kwargs):
            h = args[0]
            if h.shape[1] > 1:
                original_length = kwargs["position_embeddings"][0].shape[1]
                keep = torch.tensor([i for i in range(original_length) if i not in (1, 3)])
                if module is language.layers[1]:
                    h = h[:, keep]
                kwargs = dict(kwargs, attention_mask=None,
                              position_embeddings=tuple(v[:, keep] for v in kwargs["position_embeddings"]))
            return (h,) + args[1:], kwargs
        handles = [layer.register_forward_pre_hook(prune, with_kwargs=True) for layer in language.layers[1:]]
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            output = language(input_ids=prompt, use_cache=True)
            self.assertEqual([l.keys.shape[-2] for l in output.past_key_values.layers], [6, 4, 4])
            logits, _ = native_cached_step(wrapper, torch.tensor([[7]]), output.past_key_values,
                                           torch.full((3, 1, 1), 6))
            expected = language(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), use_cache=False)
            expected = wrapper.lm_head(expected.last_hidden_state[:, -1:])
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
