"""Pruning budgets, selector fidelity, causal positions and hybrid-cache checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from baselines import qwen35_pruning as pruning


class PruningTests(unittest.TestCase):
    def test_selector_fidelity(self):
        # The port must match the existing selector including tie/set ordering.
        for file, names in [
            ('baselines/divprune/qwen3_vl/modeling_qwen3_vl_divprune.py', ['_divprune_select_tokens']),
            ('baselines/dart/qwen3_vl/modeling_qwen3_vl_dart.py',
             ['_dart_neighbor_indices', 'dart_get_retained_image_token']),
        ]:
            originals = {n.name: ast.dump(n) for n in ast.parse(Path(file).read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name in names}
            port = {n.name: ast.dump(n) for n in ast.parse(Path(pruning.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in names}
            self.assertEqual(originals, port)

    def test_budgets_and_hybrid_cache(self):
        torch.manual_seed(44)
        torch.set_num_threads(2)
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, rope_parameters={'rope_type':'default', 'rope_theta':10000.,
                'partial_rotary_factor':1., 'mrope_section':[3,3,2]}, attn_implementation='eager')
        model = Qwen3_5TextModel(config).eval().requires_grad_(False)
        controller = pruning.VisualPruningController(SimpleNamespace(model=SimpleNamespace(language_model=model)))
        embedding = torch.randn(1, 29, 64)
        mask = torch.tensor([[False]*2+[True]*20+[False]*7])
        with torch.no_grad():
            native = model(inputs_embeds=embedding, use_cache=True)
            for method in ('divprune', 'dart'):
                for retention in (1., .05, .2):
                    with self.subTest(method=method, retention=retention):
                        positions = torch.arange(29)[None]
                        with controller.activate(method, retention, mask):
                            cached = model(inputs_embeds=embedding, position_ids=positions, use_cache=True)
                        selected = controller.audit['selected_visual_indices']
                        self.assertEqual(len(selected), round(20*retention))
                        if retention == 1.:
                            torch.testing.assert_close(cached.last_hidden_state, native.last_hidden_state, atol=0, rtol=0)
                        first = 29 if method == 'dart' else 9+len(selected)
                        self.assertEqual(cached.past_key_values.get_seq_length(3), first)
                        self.assertEqual(cached.past_key_values.get_seq_length(7), 9+len(selected))
                        prefix, full_mask = embedding, mask
                        for step in range(3):
                            token = torch.randn(1, 1, 64)
                            prefix = torch.cat((prefix, token), 1)
                            full_mask = torch.cat((full_mask, torch.zeros(1, 1, dtype=torch.bool)), 1)
                            with controller.activate(method, retention, mask):
                                cached = model(inputs_embeds=token, position_ids=torch.tensor([[29+step]]),
                                    past_key_values=cached.past_key_values, use_cache=True)
                            with controller.activate(method, retention, full_mask, fixed_visual=selected):
                                full = model(inputs_embeds=prefix, position_ids=torch.arange(prefix.shape[1])[None], use_cache=False)
                            torch.testing.assert_close(cached.last_hidden_state[:, -1], full.last_hidden_state[:, -1],
                                                       atol=3e-5, rtol=3e-5)
        controller.close()


if __name__ == '__main__':
    unittest.main()
