"""Structural tests: one-token dependence, initialization and no teacher bypass."""
import unittest
import copy
from types import SimpleNamespace

import torch
from torch import nn

from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import TokenMLP, PathHook, Bank, DEPTH


class Tests(unittest.TestCase):
    def test_checkpointing_preserves_loss_and_gradients(self):
        torch.manual_seed(44)
        stats = dict(input=dict(mean=torch.randn(8), std=torch.rand(8) + .1),
                     targets={k: dict(mean=torch.randn(8), std=torch.rand(8) + .1)
                              for k in ('cross_0', 'rest_0', 'hidden_0')})
        a = Bank(stats, 'small')
        b = copy.deepcopy(a)
        a.activation_checkpointing = False
        x = torch.randn(17, 8)
        targets = {k: torch.randn(17, 8) for k in a.heads}
        la, _ = a(x, targets, [4, 13]); la.backward()
        lb, _ = b(x, targets, [4, 13]); lb.backward()
        torch.testing.assert_close(la, lb, rtol=0, atol=0)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_token_independence_and_initialization(self):
        torch.manual_seed(44)
        for init in ('zero', 'small'):
            head = TokenMLP(torch.randn(8), torch.rand(8) + .1, torch.randn(8), torch.rand(8) + .1, init)
            x, y = torch.randn(5, 8), torch.randn(5, 8)
            optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
            if init == 'zero':
                torch.testing.assert_close(head(x), head.mean_prediction(x), rtol=0, atol=0)
            for step in range(2):
                optimizer.zero_grad()
                head.loss(x, y, [2, 3]).backward()
                self.assertGreater(float(head.up.weight.grad.norm()), 0)
                if step == 1:
                    self.assertGreater(float(head.down.weight.grad.norm()), 0)
                optimizer.step()
            changed = x.clone(); changed[1:] *= 50
            torch.testing.assert_close(head(x)[0], head(changed)[0], rtol=0, atol=0)
            perm = torch.tensor([4, 1, 0, 3, 2])
            torch.testing.assert_close(head(x)[perm], head(x[perm]), rtol=0, atol=0)
            # A different sequence length must not change any single-row output.
            torch.testing.assert_close(head(x)[:2], head(x[:2]), rtol=1e-6, atol=1e-6)

    def test_student_cannot_receive_teacher_states(self):
        layers = []
        for _ in range(DEPTH):
            layer = nn.Identity()
            layer.self_attn = nn.Identity()
            layer.self_attn.o_proj = nn.Linear(8, 8)
            layers.append(layer)
        model = SimpleNamespace(config=SimpleNamespace(image_token_id=99),
                                model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
        hook = PathHook(model)
        inp = dict(input_ids=torch.tensor([[1, 99, 99, 2]]), attention_mask=torch.ones(1, 4))
        with self.assertRaises(AssertionError):
            hook.begin(inp, 'hidden_mlp', native={1: torch.zeros(2, 8)})
        hook.begin(inp, 'hidden_identity')
        h = torch.randn(1, 4, 8)
        hook.layer_input(0)(None, (h,), {})
        altered = torch.randn_like(h)
        args, _ = hook.layer_input(1)(None, (altered,), {})
        out = args[0]
        torch.testing.assert_close(out[:, 1:3], h[:, 1:3], rtol=0, atol=0)
        torch.testing.assert_close(out[:, [0, 3]], altered[:, [0, 3]], rtol=0, atol=0)
        self.assertFalse(hook.targets)
        self.assertFalse(hook.native)
        for handle in hook.handles:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
