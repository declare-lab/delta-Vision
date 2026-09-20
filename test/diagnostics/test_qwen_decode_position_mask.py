"""Regression: causal ordering is not the compressed M-RoPE coordinate."""
import ast
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/'src/model.py', ROOT.parent/'vision-kv-inject-attention-sink/src/model.py']


def load_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'torch': torch, 'Tensor': torch.Tensor, 'Any': object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def cache():
    return dict(text_mask=torch.tensor([[True, True, False], [True, True, True]]),
        image_mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
        image_positions=torch.tensor([[10, 1000, 2000, 0], [20, 900, 0, 0]]),
        next_text_positions=torch.tensor([[2400], [1100]]),
        hf_generated_mask=torch.tensor([[True], [False]]), hf_static_max_cache_len=12)


class DecodePositionMaskTest(unittest.TestCase):
    def test_dynamic_preserves_all_past_images(self):
        for path in SOURCES:
            with self.subTest(path=path):
                c = cache()
                rope = torch.tensor([[[60], [45]]]).expand(3,-1,-1)
                result = load_function(path, '_qwen_decode_attention_mask')(c, rope, torch.tensor([True,False]))
                expected = torch.cat([c['image_mask'], c['text_mask'], torch.tensor([[True],[False]])], 1)
                self.assertTrue(torch.equal(result[:,0,0], expected))

    def test_mask_is_invariant_to_rope_coordinate(self):
        for path in SOURCES:
            with self.subTest(path=path):
                fn = load_function(path, '_qwen_decode_attention_mask')
                self.assertTrue(torch.equal(fn(cache(), torch.zeros(3,2,1,dtype=torch.long)),
                    fn(cache(), torch.full((3,2,1), 10000, dtype=torch.long))))

    def test_static_preserves_images_and_masks_unused_capacity(self):
        fn = load_function(SOURCES[1], '_qwen_decode_hf_static_attention_mask')
        c = cache()
        actual = fn(c, torch.zeros(3,2,1,dtype=torch.long), torch.tensor([True,False]))[:,0,0]
        active = torch.cat([c['image_mask'],c['text_mask'],c['hf_generated_mask'],torch.tensor([[True],[False]])],1)
        expected = torch.zeros((2,12),dtype=torch.bool)
        expected[:,:active.shape[1]]=active
        self.assertTrue(torch.equal(actual, expected))

    def test_uses_physical_order_not_unconditional_image_unmask(self):
        for path in SOURCES:
            with self.subTest(path=path):
                c = cache()
                c['next_text_positions'][0,0] = 1500
                result = load_function(path, '_qwen_decode_attention_mask')(c, torch.full((3,2,1), 10000))
                self.assertEqual(result[0,0,0,:4].tolist(), [True,True,False,False])


if __name__ == '__main__':
    unittest.main()
