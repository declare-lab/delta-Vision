import unittest
from types import SimpleNamespace

import torch

from src.qwen35_full_attention_ablation import block_visual_readout, text_indices


def reference(q, k, v, visual=None):
    """Explicit original-position mask, independent of compressed execution."""
    nk, nq = k.shape[2], q.shape[2]
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    qp = torch.arange(nk - nq, nk, device=q.device)
    kp = torch.arange(nk, device=q.device)
    allowed = kp[None, :] <= qp[:, None]
    if visual is not None:
        vm = torch.cat((visual[0], visual.new_zeros(nk - visual.shape[1])))
        allowed &= ~(~vm[qp, None] & vm[None, :])
    score = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** .5
    out = score.masked_fill(~allowed, -torch.inf).softmax(-1) @ v.float()
    return out.transpose(1, 2).to(q.dtype)


def eager_interface(module, q, k, v, mask, **kwargs):
    assert mask is None
    return reference(q, k, v), None


def check_readout(device='cpu', interface=eager_interface, dtype=torch.float32):
    torch.manual_seed(44)
    mask = torch.tensor([[False, False, True, True, True, False, False]], device=device)
    module = SimpleNamespace(scaling=32 ** -.5, is_causal=True,
                             config=SimpleNamespace(_attn_implementation='flash_attention_2'), layer_idx=19)
    for nq, nk in [(7, 7), (1, 8), (1, 9)]:
        q = torch.randn(1, 4, nq, 32, device=device, dtype=dtype)
        k = torch.randn(1, 2, nk, 32, device=device, dtype=dtype)
        v = torch.randn_like(k)
        old_k, old_v = k.clone(), v.clone()
        out, _ = block_visual_readout(module, q, k, v, mask, interface)
        torch.testing.assert_close(out, reference(q, k, v, mask),
                                   rtol=.02 if dtype == torch.bfloat16 else 1e-5,
                                   atol=.01 if dtype == torch.bfloat16 else 1e-6)
        torch.testing.assert_close(k, old_k, rtol=0, atol=0)
        torch.testing.assert_close(v, old_v, rtol=0, atol=0)
        if nq == nk:
            native, _ = interface(module, q, k, v, None, dropout=0.0, scaling=module.scaling)
            torch.testing.assert_close(out[:, mask[0]], native[:, mask[0]], rtol=0, atol=0)
        # Changing visual KV must not change the blocked text outputs.
        changed_k, changed_v = k.clone(), v.clone()
        visual = mask[0].nonzero().flatten()
        changed_k[:, :, visual] *= 100
        changed_v[:, :, visual] += 100
        other, _ = block_visual_readout(module, q, changed_k, changed_v, mask, interface)
        qi, _ = text_indices(mask, nq, nk)
        torch.testing.assert_close(out[:, qi], other[:, qi], rtol=0, atol=0)


class FullAttentionAblationTests(unittest.TestCase):
    def test_prefill_decode_and_visual_preservation(self):
        check_readout()

    def test_reject_unsupported_partial_prefill(self):
        with self.assertRaises(AssertionError):
            text_indices(torch.tensor([[False, True, False]]), 2, 4)


if __name__ == '__main__':
    unittest.main()
