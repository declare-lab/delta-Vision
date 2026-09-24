"""Checkpoint construction must preserve architecture, parameters, and dtype."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from src.model import QwenEmbeddingAdapter, PerLayerKVAdapter
from src.model_setup import (create_qwen_adapter, create_llava_kv_adapter,
                             load_qwen_embedding_adapter_checkpoint)


class SharedSetupTests(unittest.TestCase):
    def test_static_and_recurrent_checkpoint_roundtrip(self):
        lm=SimpleNamespace(config=SimpleNamespace(hidden_size=32,num_attention_heads=4,head_dim=8),layers=[None]*3)
        for mode in ('embedding_adapter','recurrent_embedding_adapter'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                torch.manual_seed(44)
                old=QwenEmbeddingAdapter.from_language_model(lm,mode=mode,visual_adapter_rank=8)
                torch.manual_seed(44)
                current=create_qwen_adapter(lm,mode=mode,rank=8)
                self.assertEqual(old.state_dict().keys(),current.state_dict().keys())
                for key,value in old.state_dict().items():
                    self.assertTrue(torch.equal(value,current.state_dict()[key]))
                # Nonzero trained weights detect lost or partially loaded parameters.
                with torch.no_grad():
                    for param in current.parameters():param.add_(.01)
                path=Path(d)/'adapter.pt'
                torch.save(dict(state_dict=current.state_dict(),global_step=2,
                    adapter_config=dict(output_mode=mode,visual_adapter_rank=8)),path)
                loaded,meta=load_qwen_embedding_adapter_checkpoint(path,lm,torch.device('cpu'),torch.float32)
                self.assertEqual(loaded.mode,mode)
                self.assertEqual(meta['missing'],[]);self.assertEqual(meta['unexpected'],[])
                self.assertFalse(loaded.training)
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                for key,value in current.state_dict().items():
                    self.assertTrue(torch.equal(value,loaded.state_dict()[key]))

    def test_llava_kv_initialization(self):
        config=dict(num_llm_layers=3,num_source_layers=2,source_dim=32,num_heads=4,head_dim=8,bottleneck_dim=8)
        torch.manual_seed(44);old=PerLayerKVAdapter(**config)
        torch.manual_seed(44);new=create_llava_kv_adapter(**config)
        self.assertEqual(old.state_dict().keys(),new.state_dict().keys())
        for key,value in old.state_dict().items():self.assertTrue(torch.equal(value,new.state_dict()[key]))


if __name__=='__main__':unittest.main()
