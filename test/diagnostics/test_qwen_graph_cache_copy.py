"""KV ownership, layout and storage accounting for graph output copies."""
import unittest
import torch
from src.graphs import clone_kv_tensors


class CacheCopyTest(unittest.TestCase):
    def test_values_layout_and_owned_storage(self):
        source = torch.arange(2*3*4*8, dtype=torch.float32).reshape(2,3,4,8)
        inputs = [source, source.transpose(1,2), source[:, :, :2], source[:, :, :0]]
        outputs = clone_kv_tensors(inputs)
        self.assertEqual(len(outputs), len(inputs))
        for original, copied in zip(inputs, outputs):
            self.assertTrue(torch.equal(original, copied))
            self.assertEqual(copied.stride(), torch.empty_like(original).stride())
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in outputs}), 1)
        self.assertEqual(outputs[0].untyped_storage().nbytes(), sum(t.numel()*t.element_size() for t in inputs))
        saved = outputs[1].clone()
        outputs[0].fill_(-1)
        self.assertTrue(torch.equal(outputs[1], saved))
        self.assertTrue(torch.equal(source, torch.arange(source.numel()).reshape(source.shape)))


if __name__ == "__main__":
    unittest.main()
