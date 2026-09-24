"""Independent matrix-size checks for the Video-MME FLOP accounting."""
import unittest
import torch
from src.benchmarking.common.resource_flops import _flash_dense, _flash_varlen, matrix_flop_counter


class ResourceFlopsTest(unittest.TestCase):
    def test_gqa_counts_query_heads_not_kv_heads(self):
        q = torch.empty(2, 11, 4, 32)
        k = torch.empty(2, 13, 2, 32)
        v = torch.empty_like(k)
        self.assertEqual(_flash_dense(q, k, v), 4 * 2 * 11 * 13 * 4 * 32)

    def test_decode_one_query(self):
        q = torch.empty(1, 1, 32, 128)
        k = torch.empty(1, 1000, 8, 128)
        self.assertEqual(_flash_dense(q, k, k), 4 * 1000 * 32 * 128)

    def test_varlen_uses_each_segment(self):
        q = torch.empty(11, 4, 32)
        k = torch.empty(13, 2, 32)
        cuq = torch.tensor([0, 3, 11], dtype=torch.int32)
        cuk = torch.tensor([0, 5, 13], dtype=torch.int32)
        self.assertEqual(_flash_varlen(q, k, k, cuq, cuk, 8, 8),
                         4 * (3 * 5 + 8 * 8) * 4 * 32)

    def test_shared_prefix_used_lengths_override_offsets(self):
        q = torch.empty(11, 4, 32)
        k = torch.empty(13, 2, 32)
        cuq = torch.tensor([0, 3, 11], dtype=torch.int32)
        cuk = torch.tensor([0, 0, 13], dtype=torch.int32)
        used = torch.tensor([5, 13], dtype=torch.int32)
        expected = 4 * (3 * 5 + 8 * 13) * 4 * 32
        self.assertEqual(_flash_varlen(q, k, k, cuq, cuk, 8, 13, seqused_k=used), expected)
        # The external FA2 operator passes seqused_k as optional argument 10.
        self.assertEqual(_flash_varlen(q, k, k, cuq, cuk, 8, 13,
                         0., 1., True, -1, -1, 0., None, False, None, None, used), expected)

    def test_counter_keeps_pytorch_matmul_accounting(self):
        x, weight = torch.ones(7, 11), torch.ones(11, 13)
        counter = matrix_flop_counter()
        with counter:
            _ = x @ weight
        self.assertEqual(counter.get_total_flops(), 2 * 7 * 11 * 13)


if __name__ == '__main__':
    unittest.main()
