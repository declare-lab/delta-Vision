"""CPU reference checks independent of the native-model smoke evaluation."""
import unittest

import torch

from src.visual_cross_token_ablation import decompose, flatten_heads


class VisualEdgesTest(unittest.TestCase):
    def test_gqa_causal_visual_edges_without_renormalization(self):
        torch.manual_seed(18)
        q = torch.randn(4, 11, 8, dtype=torch.float64)
        k, v = [torch.randn(2, 11, 8, dtype=torch.float64).repeat_interleave(2, 0) for _ in range(2)]
        p = torch.tensor([1, 2, 5, 8, 9])
        own, cross, text, mass = decompose(q, k, v, p, 8**-.5, chunk=2)
        logits = q @ k.transpose(-1, -2) * 8**-.5
        logits.masked_fill_(torch.ones(11, 11, dtype=torch.bool).triu(1), -torch.inf)
        a = logits.softmax(-1)
        full = (a @ v)[:, p]
        torch.testing.assert_close(own+cross+text, full, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(mass.sum(-1), torch.ones_like(mass[..., 0]))
        visual = torch.zeros(11, dtype=torch.bool); visual[p] = True
        for j, pos in enumerate(p.tolist()):
            self_a = a[:, pos].clone()
            self_a[:, visual] = 0
            self_a[:, pos] = a[:, pos, pos]
            cross_a = a[:, pos].clone()
            cross_a[:, pos] = 0
            torch.testing.assert_close(torch.einsum('ht,htd->hd', self_a, v), own[:, j]+text[:, j])
            torch.testing.assert_close(torch.einsum('ht,htd->hd', cross_a, v), cross[:, j]+text[:, j])
        # The first visual query has no preceding visual token, but can read text.
        torch.testing.assert_close(cross[:, 0], torch.zeros_like(cross[:, 0]))
        self.assertTrue(bool((mass[:, 0, 2] > 0).all()))
        # Later cross rows are nonzero; this would fail for accidentally diagonal-only masks.
        self.assertGreater(float(cross[:, -1].norm()), 0)

    def test_no_text_and_linear_output_projection(self):
        torch.manual_seed(21)
        q, k, v = [torch.randn(2, 7, 4, dtype=torch.float64) for _ in range(3)]
        p = torch.arange(7)
        own, cross, text, _ = decompose(q, k, v, p, .5)
        self.assertTrue(torch.equal(text, torch.zeros_like(text)))
        weight = torch.randn(5, 8, dtype=torch.float64)
        projected_sum = (flatten_heads(own) + flatten_heads(cross)) @ weight.T
        torch.testing.assert_close(projected_sum, flatten_heads(own+cross+text) @ weight.T)


if __name__ == '__main__':
    unittest.main()
