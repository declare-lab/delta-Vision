"""Check actual attention boundaries and independent one-token prediction."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from analysis.fig01b_hidden_prediction.initial_token_prediction_probe import Capture, LAYERS, vector_metrics
from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import TokenMLP


class Attention(nn.Module):
    def __init__(self, width, index):
        super().__init__()
        self.o_proj = nn.Linear(width, width)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(width) * .03)
            self.o_proj.bias.fill_(index * .007)

    def forward(self, hidden_states):
        return self.o_proj(hidden_states)


class Layer(nn.Module):
    def __init__(self, width, index):
        super().__init__()
        self.self_attn = Attention(width, index)
        self.post_attention_layernorm = nn.Identity()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        hidden_states = hidden_states + self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + .123  # FFN stand-in, must not enter delta.


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([Layer(4, i) for i in range(36)])

    def forward(self, hidden_states, **kwargs):
        for block in self.language_model.layers:
            hidden_states = block(hidden_states)
        return hidden_states


class Tests(unittest.TestCase):
    def test_attention_boundary_layer_numbering_and_early_stop(self):
        torch.manual_seed(44)
        body = Body().to(torch.bfloat16)
        model = SimpleNamespace(model=body, config=SimpleNamespace(image_token_id=99))
        hidden = torch.randn(1, 7, 4).to(torch.bfloat16)
        inputs = dict(input_ids=torch.tensor([[1, 99, 99, 99, 2, 3, 4]]),
                      attention_mask=torch.ones(1, 7), hidden_states=hidden)
        reference_hidden, reference_delta = {}, {}
        h = hidden.clone()
        with torch.no_grad():
            for i, layer in enumerate(body.language_model.layers):
                reference_hidden[i] = h[:, 1:4].flatten(0, 1).float().clone()
                a = h + layer.self_attn(h)
                reference_delta[i] = (a.float()-h.float())[:, 1:4].flatten(0, 1)
                h = a + .123
        hook = Capture(model, cross=False)
        x, targets, sizes = hook.collect(inputs)
        self.assertEqual(sizes, [3])
        for i in LAYERS:
            torch.testing.assert_close(targets[f'hidden_{i}']+x.float(), reference_hidden[i], rtol=0, atol=0)
            torch.testing.assert_close(targets[f'delta_{i}'], reference_delta[i], rtol=0, atol=0)
        self.assertTrue(all(layer.calls == 0 for layer in body.language_model.layers[18:]))
        short = {k:v.clone() for k,v in targets.items()}
        _, full, _ = hook.collect(inputs, stop_early=False)
        for key in short:
            torch.testing.assert_close(short[key], full[key], rtol=0, atol=0)
        hook.close()

    def test_initialization_and_no_other_token_input(self):
        torch.manual_seed(44)
        head = TokenMLP(torch.randn(8), torch.rand(8)+.1, torch.randn(8), torch.rand(8)+.1)
        x, y = torch.randn(11, 8), torch.randn(11, 8)
        torch.testing.assert_close(head(x), head.mean_prediction(x), rtol=0, atol=0)
        opt = torch.optim.AdamW(head.parameters(), lr=.001)
        for step in range(2):
            opt.zero_grad(); head.loss(x, y, [2, 9]).backward()
            self.assertGreater(float(head.up.weight.grad.norm()), 0)
            if step == 1:
                self.assertGreater(float(head.down.weight.grad.norm()), 0)
            opt.step()
        changed = x.clone(); changed[1:] = torch.randn_like(changed[1:])*100
        torch.testing.assert_close(head(x)[0], head(changed)[0], rtol=0, atol=0)
        torch.testing.assert_close(head(x)[0], head(x[:1])[0], rtol=1e-6, atol=1e-6)

    def test_metrics_use_original_units_and_per_token_cosine(self):
        target = torch.tensor([[1.,0.],[0.,2.]])
        pred = torch.tensor([[2.,0.],[1.,0.]])
        m = vector_metrics(pred, target, torch.zeros_like(target))
        self.assertAlmostEqual(m['mse'], 1.5)
        self.assertAlmostEqual(m['cosine'], .5)
        same = vector_metrics(target, target, torch.zeros_like(target))
        self.assertEqual(same['mse'], 0.)
        self.assertAlmostEqual(same['cosine'], 1.)


if __name__ == '__main__':
    unittest.main()
