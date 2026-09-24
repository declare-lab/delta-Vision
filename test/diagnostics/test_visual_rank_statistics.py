"""Numerical definition checks independent of a checkpoint or GPU."""
import math
import torch
from analysis.table05_native_rank.visual_rank_statistics import spectral_metrics, feature_singular_values, score_and_query_spectra


def test_metric_definitions():
    s = torch.tensor([4., 1.], dtype=torch.float64)
    erank, r95 = spectral_metrics(s)
    expected = math.exp(-.8*math.log(.8)-.2*math.log(.2))
    assert abs(float(erank)-expected) < 1e-12
    assert r95 == 2  # 16/17 < .95
    erank, r95 = spectral_metrics(torch.zeros(8))
    assert erank == 0 and r95 == 0


def test_factorized_score_spectrum():
    torch.manual_seed(44)
    for tokens, width in [(17, 8), (4, 8)]:
        q = torch.randn(3, tokens, width, dtype=torch.float64)
        k = torch.randn_like(q)
        s, qs = score_and_query_spectra(q, k, .25)
        direct = torch.linalg.svdvals(q @ k.transpose(-1, -2)*.25)
        assert torch.allclose(s, direct[..., :min(tokens,width)], atol=1e-10, rtol=1e-10)
        assert torch.allclose(qs, torch.linalg.svdvals(q), atol=1e-10, rtol=1e-10)


def test_feature_gram():
    torch.manual_seed(44)
    for shape in [(20, 30), (30, 20)]:
        x = torch.randn(*shape, dtype=torch.float64)
        assert torch.allclose(feature_singular_values(x), torch.linalg.svdvals(x), atol=1e-10, rtol=1e-10)
