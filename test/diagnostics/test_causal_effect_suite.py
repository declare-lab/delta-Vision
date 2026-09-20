"""Checks for matching scope definitions and channel-only projection."""
import torch
from src.causal_effect_benchmark_suite import configure, MODELS


def test_scopes():
    for model, middle, last in [('qwen', 13, 26), ('llava', 11, 22)]:
        scopes = configure(model)
        assert scopes['first5'] == list(range(5))
        assert scopes['first10'] == list(range(10))
        assert scopes['middle10'] == list(range(middle, middle+10))
        assert scopes['last10'] == list(range(last, last+10))
        assert scopes['all'] == list(range(MODELS[model][1]))


def test_projection_limits():
    torch.manual_seed(44)
    joint, blocked = torch.randn(7, 16), torch.randn(7, 16)
    delta = joint - blocked
    basis = torch.eye(16)
    torch.testing.assert_close(blocked + delta @ basis @ basis.T, joint)
    empty = basis[:, :0]
    torch.testing.assert_close(blocked + delta @ empty @ empty.T, blocked)
    narrow = basis[:, :4]
    projected = delta @ narrow @ narrow.T
    assert projected.shape == delta.shape  # No token is removed.
    assert torch.linalg.matrix_rank(projected) <= 4


if __name__ == '__main__':
    test_scopes()
    test_projection_limits()
    print('PASS: model-specific scopes and rank0/full/channel-projection limits')
