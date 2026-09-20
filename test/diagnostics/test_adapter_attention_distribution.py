import math
import torch
from src.adapter_visual_attention_distribution import distribution_rows, reduce_rows


def test_identity():
    torch.manual_seed(5)
    s = torch.randn(3, 4, 31)
    z = torch.logaddexp(s.logsumexp(-1), torch.zeros(3, 4))
    m = distribution_rows(s, s.clone(), z, z)
    torch.testing.assert_close(m['cosine'], torch.ones(3, 4))
    assert m['js_bits'].max() < 1e-7
    assert m['top10_overlap'].min() == 1
    assert m['top10pct_overlap'].min() == 1
    assert m['visual_mass_abs_gap'].max() == 0


def test_conditional_shape_is_separate_from_total_mass():
    s = torch.tensor([[[0., 1., 2.]]])
    shifted = s + 5
    text_logsum = torch.tensor([[3.]])
    zn = torch.logaddexp(s.logsumexp(-1), text_logsum)
    za = torch.logaddexp(shifted.logsumexp(-1), text_logsum)
    m = distribution_rows(s, shifted, zn, za)
    assert m['js_bits'].item() < 1e-7 and m['cosine'].item() > .999999
    assert m['adapter_visual_mass'].item() > m['native_visual_mass'].item() + .5
    expected = torch.cat((s[0, 0], text_logsum.flatten())).softmax(0)[:3].sum()
    torch.testing.assert_close(m['native_visual_mass'].float().flatten()[0], expected)


def test_heads_are_not_averaged_before_comparison():
    # Head-averaged maps would both be [0.5,0.5], despite every head disagreeing.
    s = torch.tensor([[[30., -30.]], [[-30., 30.]]])
    t = s.flip(0)
    m = distribution_rows(s, t, s.logsumexp(-1), t.logsumexp(-1))
    stats = reduce_rows(m)
    assert stats['means']['cosine'] < 1e-20
    assert abs(stats['means']['js_bits'] - 1) < 1e-6
    assert stats['means']['top1_match'] == 0


def test_js_is_symmetric_and_topk_uses_logits_without_underflow():
    s = torch.arange(40).float().view(1, 1, -1) - 10000
    t = s.flip(-1)
    zn = torch.zeros(1, 1)
    a = distribution_rows(s, t, zn, zn)
    b = distribution_rows(t, s, zn, zn)
    torch.testing.assert_close(a['js_bits'], b['js_bits'])
    assert a['top10_overlap'].item() == 0
    assert a['top10pct_overlap'].item() == 0
    assert a['js_bits'].item() > .99
