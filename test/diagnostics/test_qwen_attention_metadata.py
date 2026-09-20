import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from src.qwen_attention_metadata import optimize_qwen_attention_metadata
from transformers.modeling_flash_attention_utils import _is_packed_sequence


class Attention(torch.nn.Module):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, **kwargs):
        self.received = (position_embeddings, attention_mask, position_ids, kwargs)
        return _is_packed_sequence(position_ids, batch_size=1) or all(
            kwargs.get(k) is not None for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"))


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        layer = torch.nn.Module()
        layer.self_attn = Attention()
        self.layers = torch.nn.ModuleList([layer])

    def forward(self, position_ids, position_embeddings, attention_mask):
        # Represent a layer pruning positions 2 and 3; RoPE is independently prepared.
        retained = torch.tensor([0, 1, 4, 5])
        return self.layers[0].self_attn(torch.zeros(1, 4, 4),
            position_ids=position_ids[0, :, retained],
            position_embeddings=position_embeddings, attention_mask=attention_mask)


class MetadataTest(unittest.TestCase):
    def test_single_token_decode_does_not_read_a_device_scalar(self):
        class SingleTokenLanguage(Language):
            def forward(self, position_ids):
                return self.layers[0].self_attn(torch.zeros(1, 1, 4),
                    position_ids=position_ids[0], position_embeddings=None)
        language = SingleTokenLanguage()
        handle = optimize_qwen_attention_metadata(SimpleNamespace(model=SimpleNamespace(language_model=language)))
        positions = torch.tensor([137]).view(1, 1, 1).expand(4, 1, 1)
        with patch("transformers.modeling_flash_attention_utils._is_packed_sequence",
                   side_effect=AssertionError("single-token decode must not inspect tensor values")):
            self.assertFalse(language(position_ids=positions))
        self.assertIsNone(language.layers[0].self_attn.received[2])
        handle.remove()

    def test_pruned_gaps_keep_original_kernel_and_rope_mask(self):
        language = Language()
        model = SimpleNamespace(model=SimpleNamespace(language_model=language))
        positions = torch.arange(6).view(1, 1, 6).expand(4, 1, 6)
        rotary = (torch.randn(1, 4, 4), torch.randn(1, 4, 4))
        mask = torch.ones(1, 6)
        self.assertTrue(language(position_ids=positions, position_embeddings=rotary, attention_mask=mask))
        handle = optimize_qwen_attention_metadata(model)
        self.assertTrue(language(position_ids=positions, position_embeddings=rotary, attention_mask=mask))
        got_rotary, got_mask, got_positions, metadata = language.layers[0].self_attn.received
        self.assertIs(got_rotary, rotary)
        self.assertIs(got_mask, mask)
        self.assertIsNone(got_positions)
        torch.testing.assert_close(metadata["cu_seq_lens_q"], torch.tensor([0, 4], dtype=torch.int32))
        self.assertIs(metadata["cu_seq_lens_q"], metadata["cu_seq_lens_k"])
        self.assertEqual(metadata["max_length_q"], 4)
        handle.remove()
        self.assertTrue(language(position_ids=positions, position_embeddings=rotary, attention_mask=mask))

    def test_real_packed_resets_are_preserved(self):
        language = Language()
        handle = optimize_qwen_attention_metadata(SimpleNamespace(model=SimpleNamespace(language_model=language)))
        positions = torch.tensor([0, 1, 2, 3, 0, 1]).view(1, 1, 6).expand(4, 1, 6)
        self.assertTrue(language(position_ids=positions, position_embeddings=None, attention_mask=None))
        torch.testing.assert_close(language.layers[0].self_attn.received[2], torch.tensor([[0, 1, 0, 1]]))
        handle.remove()


if __name__ == "__main__":
    unittest.main()
