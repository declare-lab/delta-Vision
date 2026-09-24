import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from analysis.fig03_visual_effect.visual_effect_shared_rank import NativeTraceOracle, project, uncentered_basis, reconstruction_error


class TinyMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        config = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
        config._attn_implementation = 'eager'
        self.backbone = LlamaModel(config).eval()
        self.model = SimpleNamespace(language_model=self.backbone)
        self.lm_head = nn.Linear(32, 40, bias=False)

    def forward(self, logits_to_keep=1, **kwargs):
        out = self.backbone(**kwargs)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state[:, -logits_to_keep:]))


class RankTests(unittest.TestCase):
    def test_fp32_cancellation_of_bf16_values(self):
        joint = torch.tensor([1e-7, 1.], dtype=torch.bfloat16)
        blocked = torch.tensor([100., 0.5], dtype=torch.bfloat16)
        delta = joint.float() - blocked.float()
        audit = reconstruction_error(joint, blocked, delta)
        self.assertEqual(audit['nonidentical_elements'], 1)
        self.assertLess(audit['max_abs'], 2e-7)
        with self.assertRaises(AssertionError):
            reconstruction_error(joint, blocked, delta + 0.01)

    def test_uncentered_basis_keeps_large_shared_mean(self):
        x = torch.tensor([[100., 1., 0.], [100., -1., 0.]], dtype=torch.float64)
        b, spectrum = uncentered_basis(x.T @ x, 1)
        self.assertAlmostEqual(abs(float(b[0, 0])), 1.)
        self.assertGreater(float(spectrum[0]), 10000.)
        self.assertTrue(torch.equal(project(x, b, 0), torch.zeros_like(x).float()))
        torch.testing.assert_close(project(x, b, 1), torch.tensor([[100., 0., 0.], [100., 0., 0.]]))

    def test_full_restore_rank_zero_and_frozen_teacher_trajectory(self):
        torch.manual_seed(44)
        model = TinyMultimodal().eval()
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]), attention_mask=torch.ones(1, 6, dtype=torch.long))
        types = torch.tensor([[0, 1, 1, 1, 0, 0]])
        oracle = NativeTraceOracle(model)
        with torch.inference_mode():
            expected = model(**inputs).logits[:, -1]
            trace = oracle.trace(inputs, types, 'llava')
            torch.testing.assert_close(trace['native_logits'], expected, rtol=0, atol=0)
            full = oracle.rollout(trace, None, None)
            torch.testing.assert_close(full, expected, rtol=1e-5, atol=1e-6)
            bases = {i: torch.eye(32) for i in range(3)}
            torch.testing.assert_close(oracle.rollout(trace, bases, 32), full, rtol=0, atol=0)
            # Rank zero equals a true text-only model using ORIGINAL positions.
            zero = oracle.rollout(trace, bases, 0)
            pos = torch.tensor([0, 4, 5])
            actual_text = model(input_ids=inputs['input_ids'][:, pos], attention_mask=torch.ones(1, 3, dtype=torch.long),
                                position_ids=pos[None]).logits[:, -1]
            torch.testing.assert_close(zero, actual_text, rtol=1e-5, atol=1e-6)
            after = oracle.trace(inputs, types, 'llava')
            for i in range(3):
                torch.testing.assert_close(after['effects'][i], trace['effects'][i], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
