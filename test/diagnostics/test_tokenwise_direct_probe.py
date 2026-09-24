"""CPU checks for probe initialization and pooled regression statistics."""
import torch

from analysis.fig01b_hidden_prediction.tokenwise_direct_probe import Metrics, Probe


def test_initialization_and_gradient():
    for kind, width in [('cross', 4096), ('hidden', 2560)]:
        torch.manual_seed(44)
        probe = Probe(kind)
        x = torch.randn(3, 2560)
        pred = probe(x)
        assert pred.shape == (3, width)
        expected = torch.zeros_like(pred) if kind == 'cross' else x
        torch.testing.assert_close(pred, expected, rtol=0, atol=0)
        (pred - torch.randn_like(pred)).square().mean().backward()
        assert probe.up.weight.grad.norm() > 0


def test_pooled_statistics_equal_unsharded():
    torch.manual_seed(44)
    whole = Metrics(5)
    shards = [Metrics(5), Metrics(5)]
    for i, n in enumerate([3, 7, 11, 4]):
        y = torch.randn(n, 5) + torch.arange(5)
        p = y + .2 * torch.randn_like(y)
        whole.add(p, y)
        shards[i % 2].add(p, y)
    a = Metrics.merged([whole.state()])
    b = Metrics.merged([s.state() for s in shards])
    for key in a:
        assert abs(a[key] - b[key]) < 1e-10, key


def test_identity_metrics():
    metric = Metrics(2)
    x = torch.tensor([[1., 2.], [3., 5.]])
    metric.add(x, x)
    result = Metrics.merged([metric.state()])
    assert result['r2'] == 1
    assert result['mse'] == 0
    assert abs(result['cosine'] - 1) < 1e-10


def test_hidden_rank1024():
    probe = Probe('hidden', rank=1024)
    assert probe.down.weight.shape == (1024, 2560)
    assert probe.up.weight.shape == (2560, 1024)
    assert sum(p.numel() for p in probe.parameters()) == 5242880
    x = torch.randn(2, 2560)
    pred = probe(x)
    torch.testing.assert_close(pred, x, rtol=0, atol=0)
    pred.square().mean().backward()
    assert probe.up.weight.grad.norm() > 0


if __name__ == '__main__':
    test_initialization_and_gradient()
    test_pooled_statistics_equal_unsharded()
    test_identity_metrics()
    test_hidden_rank1024()
    print('PASS: initialization, backward, pooled/sharded metrics, identity')
