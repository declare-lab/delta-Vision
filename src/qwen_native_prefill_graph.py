"""Whole native Qwen text prefill graph for unpadded, uncompressed requests.

The HF forward, attention kernel and cache values stay unchanged. Multimodal
positions keep their three RoPE axes; only the verified consecutive text axis is
omitted from the FA2 packed-sequence heuristic. Vision runs on every request.
"""
from collections import OrderedDict
import copy
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from src.qwen_native_graph import clone_tree, copy_tree, clone_kv_tensors, signature


class LanguagePrefillGraph:
    def __init__(self, original, kwargs, cache):
        self.inputs = clone_tree(kwargs)
        self.cache = copy.copy(cache)
        self.cache.layers = [copy.copy(layer) for layer in cache.layers]
        states = [dict(layer.__dict__) for layer in self.cache.layers]
        def forward():
            for layer, state in zip(self.cache.layers, states):
                layer.__dict__.clear()
                layer.__dict__.update(state)
            return original(**self.inputs, past_key_values=self.cache)
        device = self.inputs['inputs_embeds'].device
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.kv = [(layer.keys, layer.values) for layer in self.cache.layers]
            self.graph.replay()

    def replay(self, kwargs, cache):
        copy_tree(self.inputs, kwargs)
        self.graph.replay()
        output = clone_tree({key: value for key, value in self.output.items() if key != 'past_key_values'})
        owned = clone_kv_tensors([t for pair in self.kv for t in pair])
        for i, (layer, source) in enumerate(zip(cache.layers, self.cache.layers)):
            layer.__dict__.update(source.__dict__)
            layer.keys, layer.values = owned[2*i:2*i+2]
        output['past_key_values'] = cache
        return type(self.output)(**output)


class QwenWholePrefillGraphs:
    def __init__(self, model, max_shapes=1):
        if model.model.language_model.config._attn_implementation != 'flash_attention_2':
            raise ValueError('Whole prefill graphs require FA2')
        self.language = model.model.language_model
        self.original = self.language.forward
        self.enabled = True
        self.allow_capture = False
        self.max_shapes = max_shapes
        self.entries = OrderedDict()
        self.captures = self.replays = self.fallbacks = 0
        self.language.forward = self.forward

    def forward(self, *args, **kwargs):
        hidden = kwargs.get('inputs_embeds')
        cache = kwargs.get('past_key_values')
        positions = kwargs.get('position_ids')
        if (not self.enabled or args or torch.is_grad_enabled() or hidden is None
            or hidden.shape[0] != 1 or hidden.shape[1] <= 1 or not hidden.is_cuda
            or not kwargs.get('use_cache', self.language.config.use_cache)
            or kwargs.get('deepstack_visual_embeds')
            or positions is None or positions.ndim != 3 or positions.shape[0] != 4
            or not bool((positions[0].diff(dim=-1) == 1).all())
            or (cache is not None and (type(cache) is not DynamicCache or cache.offloading
                or any(type(layer) is not DynamicLayer or layer.is_initialized for layer in cache.layers)))):
            return self.original(*args, **kwargs)
        mask = kwargs.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or mask.shape != hidden.shape[:2] or not bool(mask.bool().all())):
            return self.original(*args, **kwargs)
        prepared = dict(kwargs)
        prepared.pop('past_key_values', None)
        prepared['attention_mask'] = None
        prepared['position_ids'] = positions[1:]
        prepared['visual_pos_masks'] = None  # No DeepStack branch uses this mask.
        prepared['deepstack_visual_embeds'] = None
        key = signature(prepared)
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                self.fallbacks += 1
                return self.original(*args, **kwargs)
            if cache is None:
                cache = DynamicCache(config=self.language.config)
            entry = LanguagePrefillGraph(self.original, prepared, cache)
            self.entries[key] = entry
            self.captures += 1
            while len(self.entries) > self.max_shapes:
                self.entries.popitem(last=False)
        if cache is None:
            cache = DynamicCache(config=self.language.config)
        self.entries.move_to_end(key)
        self.replays += 1
        return entry.replay(prepared, cache)

    def stats(self):
        return dict(captures=self.captures, replays=self.replays, fallbacks=self.fallbacks,
                    max_shapes=self.max_shapes, retained_shapes=len(self.entries))

    def remove(self):
        self.language.forward = self.original
        self.entries.clear()
