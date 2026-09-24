"""Check that visual replay resets only visual states, never text trajectories."""
import unittest

import torch

from analysis.fig01a_hidden_channels.visual_channel_native_cache import NativeVisualHook, project_cache


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states):
        self.seen = hidden_states.clone()
        out = hidden_states.clone()
        # A text token reads visual content. Both rows then change downstream.
        out[:, 1] += hidden_states[:, 0]
        out[:, 0] *= 2
        return out


class NativeCacheTest(unittest.TestCase):
    def test_native_visual_reset_preserves_intervened_text(self):
        layers = [Block(), Block()]
        hook = NativeVisualHook(layers)
        hook.positions = torch.tensor([0])
        initial = torch.tensor([[[2., 4.], [10., 20.]]])
        hook.begin("capture")
        native = initial
        for layer in layers:
            native = layer(native)
        hook.check()
        cache = {k: v.clone() for k, v in hook.native.items()}
        bases = {i: (torch.zeros(1, 1, 2), torch.eye(2)) for i in range(2)}
        hook.replacements, _ = project_cache(hook.native, bases, 1)
        hook.begin("replay")
        out = initial
        for layer in layers:
            out = layer(out)
        hook.check()
        # Layer 1 takes its visual rows from its native cache (not layer 0's
        # compressed output), and its text from layer 0's intervened output.
        self.assertTrue(torch.equal(layers[1].seen[:, 0], torch.tensor([[4., 0.]])))
        self.assertTrue(torch.equal(layers[1].seen[:, 1], torch.tensor([[12., 20.]])))
        self.assertTrue(torch.equal(out[:, 1], torch.tensor([[16., 20.]])))
        for k in cache:
            self.assertTrue(torch.equal(hook.native[k], cache[k]))
        hook.replacements = hook.native
        hook.begin("replay")
        restored = initial
        for layer in layers:
            restored = layer(restored)
        self.assertTrue(torch.equal(restored, native))
        saved_native = hook.native
        hook.begin("capture")
        self.assertEqual(len(saved_native), 2)
        self.assertTrue(torch.equal(saved_native[0], cache[0]))
        hook.close()

    def test_zero_rank_uses_fixed_mean_without_image_content(self):
        bases = {0: (torch.tensor([[[3., 5.]]]), torch.eye(2))}
        a, _ = project_cache({0: torch.randn(1, 4, 2)}, bases, 0)
        b, _ = project_cache({0: torch.randn(1, 4, 2) * 100}, bases, 0)
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_complete_basis_reconstructs_input(self):
        basis, _ = torch.linalg.qr(torch.randn(16, 16))
        x = torch.randn(1, 7, 16)
        projected, error = project_cache({0: x}, {0: (torch.randn(1, 1, 16), basis)}, 16)
        torch.testing.assert_close(projected[0], x, atol=1e-5, rtol=1e-5)
        self.assertLess(error, 1e-10)


if __name__ == "__main__":
    unittest.main()
