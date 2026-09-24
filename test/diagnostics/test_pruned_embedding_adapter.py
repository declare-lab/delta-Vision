"""FP32 algebra/cache validation, independent full-query vs text-only execution.

Native attention uses a dense causal oracle; the split path independently
interprets its FA2 varlen plan. CPU FP32 avoids BF16 matrix-shape roundoff.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, ALL_ATTENTION_FUNCTIONS
from src.model import (QwenEmbeddingAdapter, qwen_embedding_adapter_decode_step,
    prepare_qwen_embedding_adapter_inputs, qwen_embedding_adapter_prefill_cache_prepared)
from analysis.table13_pruning_adapter.qwen_pruned_embedding_adapter import native_prefix, select_visual, adapter_prefill, native_reference_generate


def dense_attention(query, key, value, scale, causal):
    groups = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(groups, 1)
    v = value.repeat_interleave(groups, 1)
    scores = (query @ k.transpose(-1,-2)) * scale
    if causal:
        nq, nk = query.shape[2], key.shape[2]
        allowed = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk-nq
        scores = scores.masked_fill(~allowed, -float('inf'))
    return (scores.softmax(-1) @ v).transpose(1,2).contiguous()


def native_attention(module, query, key, value, attention_mask, scaling, **kwargs):
    assert attention_mask is None
    return dense_attention(query,key,value,scaling,True), None


def planned_attention(query, key, value, *, scaling, plan):
    if plan.get('dense_decode'):
        return dense_attention(query,key,value,scaling,False)
    pieces=[]
    for i in range(len(plan['cu_q'])-1):
        a,b=int(plan['cu_q'][i]),int(plan['cu_q'][i+1])
        c,d=int(plan['cu_k'][i]),int(plan['cu_k'][i+1])
        ids=plan['key_indices'][c:d]
        pieces.append(dense_attention(query[:,:,a:b],key[:,:,ids],value[:,:,ids],scaling,True))
    return torch.cat(pieces,1)


class PrunedEmbeddingAdapterTest(unittest.TestCase):
    def test_blocked_layers_skip_adapter_and_match_full_cache_attention_masking(self):
        from analysis.table14_15_layer_skipping.qwen_adapter_layer_blocking import native_blocked_visual_attention
        torch.manual_seed(19);torch.set_num_threads(2)
        config=Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            rope_parameters={'rope_type':'default','mrope_section':[1,1,2]})
        config._attn_implementation='sdpa'
        lm=Qwen3VLTextModel(config).eval();config._attn_implementation='flash_attention_2'
        model=SimpleNamespace(model=SimpleNamespace(language_model=lm,get_input_embeddings=lambda:lm.embed_tokens),
            lm_head=nn.Linear(32,64,bias=False),_adapter_attention_implementation='flash_attention_2')
        adapter=QwenEmbeddingAdapter.from_language_model(lm,mode='embedding_adapter',visual_adapter_rank=4).eval()
        for up in adapter.visual_adapter_up:nn.init.normal_(up.weight,std=.1)
        ids=torch.randint(0,64,(1,12));types=torch.zeros_like(ids);types[:,2:8]=1
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=types)
        positions=torch.arange(12)[None,None,:].expand(3,1,-1).clone()
        def cpu_flash(q,k,v,softmax_scale,causal,**kwargs):
            return dense_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),softmax_scale,causal)
        with patch.dict(ALL_ATTENTION_FUNCTIONS, {'flash_attention_2':native_attention}), \
             patch('src.attention.attention_heads',planned_attention), \
             patch('flash_attn.flash_attn_interface.flash_attn_func',cpu_flash),torch.inference_mode():
            initial=lm.embed_tokens(ids)
            selection=select_visual(model,inputs,initial,'divprune',1.)
            for blocked in [[],[0],[3],[0,3],list(range(4))]:
                with self.subTest(blocked=blocked):
                    calls=[];handles=[]
                    for i,down in enumerate(adapter.visual_adapter_down):
                        handles.append(down.register_forward_hook(lambda m,a,o,i=i:calls.append(i)))
                    prepared=prepare_qwen_embedding_adapter_inputs(model,adapter,ids,inputs['attention_mask'],types,initial,positions)
                    logits,_,cache=qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prepared,
                        retain_prefix_states=False,blocked_visual_layers=blocked)
                    for handle in handles:handle.remove()
                    self.assertEqual(calls,[i for i in range(4) if i not in blocked])
                    self.assertEqual([c['visual_key'].shape[2] for c in cache['layers']],
                                     [0 if i in blocked else 6 for i in range(4)])
                    with native_blocked_visual_attention(selection[2],12,blocked):
                        expected,ref,_=native_reference_generate(model,adapter,initial,positions,'divprune',selection,4,set())
                    torch.testing.assert_close(logits,ref,atol=2e-6,rtol=2e-5)
                    tokens=[]
                    for step in range(4):
                        token=logits[0,-1].argmax();tokens.append(int(token))
                        if step<3:logits,cache=qwen_embedding_adapter_decode_step(model,adapter,token.view(1,1),cache)
                    self.assertEqual(tokens,expected)
                    for i in blocked:self.assertEqual(cache['layers'][i]['visual_key'].shape[2],0)

    def test_native_prefix_and_mixed_length_decode_equal_full_query_oracle(self):
        torch.manual_seed(9); torch.set_num_threads(2)
        config=Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            rope_parameters={'rope_type':'default','mrope_section':[1,1,2]})
        config._attn_implementation='sdpa'
        lm=Qwen3VLTextModel(config).eval()
        config._attn_implementation='flash_attention_2'
        model=SimpleNamespace(model=SimpleNamespace(language_model=lm,get_input_embeddings=lambda:lm.embed_tokens),
            lm_head=nn.Linear(32,64,bias=False),_adapter_attention_implementation='flash_attention_2')
        adapter=QwenEmbeddingAdapter.from_language_model(lm,mode='embedding_adapter',visual_adapter_rank=4).eval()
        for up in adapter.visual_adapter_up: nn.init.normal_(up.weight,std=.1)
        ids=torch.randint(0,64,(1,12));types=torch.zeros_like(ids);types[:,2:8]=1
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=types)
        positions=torch.arange(12)[None,None,:].expand(3,1,-1).clone()
        positions[1,0,2:8]=torch.tensor([2,2,3,3,4,4])
        positions[2,0,2:8]=torch.tensor([2,3,2,3,2,3])
        with patch.dict(ALL_ATTENTION_FUNCTIONS, {'flash_attention_2':native_attention}), \
             patch('src.attention.attention_heads',planned_attention), torch.inference_mode():
            initial=lm.embed_tokens(ids)
            prefix=native_prefix(model,initial,positions)
            saved=prefix[1].layers[0].keys.clone()
            for method in ['dart','divprune']:
                for ratio in [1.,.5,.2,.05]:
                    with self.subTest(method=method,ratio=ratio):
                        selection=select_visual(model,inputs,initial,method,ratio,prefix)
                        logits,mask,cache,audit=adapter_prefill(model,adapter,inputs,initial,positions,method,selection,prefix)
                        expected,ref,lengths=native_reference_generate(model,adapter,initial,positions,method,selection,4,set())
                        torch.testing.assert_close(logits,ref,atol=2e-6,rtol=2e-5)
                        self.assertEqual(lengths,[n+6 for n in audit['layer_visual']])
                        tokens=[]
                        for step in range(4):
                            token=logits[0,-1].argmax();tokens.append(int(token))
                            if step<3:logits,cache=qwen_embedding_adapter_decode_step(model,adapter,token.view(1,1),cache)
                        self.assertEqual(tokens,expected)
                        torch.testing.assert_close(prefix[1].layers[0].keys,saved,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
