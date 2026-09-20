"""CPU reference tests: oracle parity, hybrid cache, and the unchanged KL objective."""
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from src.qwen35_embedding import StaticVisualAdapter, VisualAdapterController
from src.qwen35_experiment import teacher_targets, student_loss


class Qwen35Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(44)

    def test_oracle_and_hybrid_cache(self):
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, rope_parameters={'rope_type':'default', 'rope_theta':10000.,
                'partial_rotary_factor':1., 'mrope_section':[3,3,2]}, attn_implementation='eager')
        model = Qwen3_5TextModel(config).eval().requires_grad_(False)
        adapter = StaticVisualAdapter(64, 4, 8)
        controller = VisualAdapterController(SimpleNamespace(model=SimpleNamespace(language_model=model)), adapter)
        embedding = torch.randn(1, 12, 64)
        mask = torch.tensor([[False]*2+[True]*5+[False]*5])
        with torch.no_grad(), controller.activate('capture', mask):
            reference = model(inputs_embeds=embedding, use_cache=False).last_hidden_state
        with torch.no_grad(), controller.activate('oracle', mask):
            oracle = model(inputs_embeds=embedding, use_cache=False).last_hidden_state
        torch.testing.assert_close(reference[~mask], oracle[~mask], atol=1e-6, rtol=1e-6)
        # Random learned weights also must preserve incremental/full-prefix equivalence.
        for layer in adapter.up:
            torch.nn.init.normal_(layer.weight, std=.03)
        with torch.no_grad(), controller.activate('adapter', mask):
            cached = model(inputs_embeds=embedding, use_cache=True)
        for _ in range(3):
            token = torch.randn(1, 1, 64)
            embedding = torch.cat((embedding, token), 1)
            mask = torch.cat((mask, torch.zeros(1, 1, dtype=torch.bool)), 1)
            with torch.no_grad(), controller.activate('adapter', mask):
                cached = model(inputs_embeds=token, past_key_values=cached.past_key_values, use_cache=True)
            with torch.no_grad(), controller.activate('adapter', mask):
                full = model(inputs_embeds=embedding, use_cache=False)
            torch.testing.assert_close(cached.last_hidden_state[:, -1], full.last_hidden_state[:, -1],
                                       atol=2e-5, rtol=2e-5)
        with controller.activate('adapter', mask, checkpoint_layers=True):
            value = model(inputs_embeds=embedding, use_cache=False).last_hidden_state
            value[~mask].sin().mean().backward()
        self.assertTrue(all(layer.weight.grad.norm() > 0 for layer in adapter.up))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        controller.close()

    def test_original_topk_kl_value_and_gradient(self):
        from src.train import masked_topk_kl_stats
        hidden_teacher = torch.randn(1, 42, 16)
        hidden_student = torch.randn(1, 42, 16, requires_grad=True)
        head = torch.nn.Linear(16, 70, bias=False).requires_grad_(False)
        def lm(inputs_embeds): return SimpleNamespace(last_hidden_state=inputs_embeds)
        model = SimpleNamespace(model=SimpleNamespace(language_model=lm), lm_head=head)
        plen, topk, temperature = 5, 12, 2.
        ids = torch.randint(0, 70, (1, 42))
        mask = torch.arange(42)[None, :] >= plen
        idx, prob = teacher_targets(model, {'inputs_embeds':hidden_teacher}, plen,
                                   ids[:, plen:], topk, temperature)
        actual = student_loss(model, {'inputs_embeds':hidden_student}, plen, idx, prob, temperature)
        expected, _, _ = masked_topk_kl_stats(head(hidden_student), head(hidden_teacher),
                                             ids, mask, temperature, topk)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual_grad = torch.autograd.grad(actual, hidden_student, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, hidden_student)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)

    def test_accumulation_preserves_original_token_weighting(self):
        from src.train import masked_topk_kl_stats
        logits = torch.randn(4, 13, 70, requires_grad=True)
        teacher = torch.randn_like(logits)
        ids = torch.randint(0, 70, (4, 13))
        counts = torch.tensor([1, 3, 7, 11])
        mask = torch.arange(13)[None] >= (13-counts[:, None])
        expected, _, _ = masked_topk_kl_stats(logits, teacher, ids, mask, 2., 12)
        actual = logits.new_zeros(())
        for i in range(4):
            loss, _, _ = masked_topk_kl_stats(logits[i:i+1], teacher[i:i+1], ids[i:i+1], mask[i:i+1], 2., 12)
            actual = actual + loss*counts[i]/counts.sum()
        torch.testing.assert_close(actual, expected)
        g1 = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(g1, g2)


if __name__ == '__main__': unittest.main()
