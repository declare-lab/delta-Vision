"""Check no-DeepStack execution and checkpoint compatibility with real tiny models."""
import copy
import importlib.util
from pathlib import Path
import unittest
import sys
from unittest.mock import patch
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from src.model_setup import disable_qwen_deepstack

ROOT = Path(__file__).resolve().parents[2]


def config():
    return Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_parameters=dict(rope_type='default', rope_theta=10000,
                mrope_section=[1,1,2])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
            num_heads=4, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0,1]),
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123)


class DeepStackPolicyTest(unittest.TestCase):
    def test_native_preserves_main_features_and_weights(self):
        torch.manual_seed(44)
        model = Qwen3VLForConditionalGeneration(config()).eval()
        visual = model.model.visual
        pixels = torch.randn(16,24)
        grid = torch.tensor([[1,4,4]])
        with torch.no_grad():
            original = visual(pixels,grid_thw=grid)
        self.assertEqual(len(original.deepstack_features),2)
        keys = set(model.state_dict())
        arch = copy.deepcopy(model.config.vision_config.deepstack_visual_indexes)
        disable_qwen_deepstack(model)
        disable_qwen_deepstack(model.model)
        disable_qwen_deepstack(model)
        self.assertEqual(len(model.model._deepstack_guard_handles),2)
        self.assertEqual(keys,set(model.state_dict()))
        self.assertEqual(arch,model.config.vision_config.deepstack_visual_indexes)
        with torch.no_grad():
            off = visual(pixels,grid_thw=grid)
        self.assertEqual(off.deepstack_features,[])
        torch.testing.assert_close(off.pooler_output,original.pooler_output,rtol=0,atol=0)
        with self.assertRaisesRegex(AssertionError,'disabled project-wide'):
            visual.deepstack_merger_list[0](torch.randn(16,32))
        with self.assertRaisesRegex(AssertionError,'disabled project-wide'):
            model.model.language_model._deepstack_process(None,None,None)

    def test_shared_teacher_loader_enforces_off(self):
        from src import model as shared
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                teacher = Qwen3VLForConditionalGeneration(config())
                with patch.object(shared.AutoProcessor,'from_pretrained',return_value=object()), \
                     patch.object(shared.Qwen3VLForConditionalGeneration,'from_pretrained',return_value=teacher), \
                     patch.object(shared,'_load_qwen3vl_zero3_sharded',return_value=teacher):
                    _,loaded=shared.load_frozen_qwen3vl('/unused',torch.float32,torch.device('cpu'),
                        zero3_sharded_load=sharded,deepspeed_config={})
                self.assertEqual(loaded.model.visual.deepstack_visual_indexes,[])
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_all_six_baseline_constructors_enforce_off(self):
        for method in ('fastv','dart','divprune','visionzip','sparsevlm','zoo'):
            with self.subTest(method=method):
                path=ROOT/f'baselines/{method}/qwen3_vl/modeling_qwen3_vl_{method}.py'
                spec=importlib.util.spec_from_file_location(f'test_{method}',path)
                mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
                model=mod.Qwen3VLModel(config())
                self.assertEqual(model.visual.deepstack_visual_indexes,[])
                with self.assertRaisesRegex(AssertionError,'disabled project-wide'):
                    model.language_model._deepstack_process(None,None,None)


if __name__=='__main__':unittest.main()
