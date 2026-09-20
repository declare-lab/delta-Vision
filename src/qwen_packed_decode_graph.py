"""Native FA2 decode with packed, owned KV input/output allocations.

The DynamicLayer update still concatenates the same keys/values in the same
order, writing directly into a preallocated contiguous slice. Graph outputs are
cloned once before returning, so successive requests cannot overwrite old KV.
"""
import copy
import torch
from transformers.cache_utils import DynamicLayer
from src.qwen_native_graph import clone_tree, copy_tree


class PackedDynamicLayer(DynamicLayer):
    def update(self, key_states, value_states, *args, **kwargs):
        torch.cat((self.keys, key_states), dim=-2, out=self.output_keys)
        torch.cat((self.values, value_states), dim=-2, out=self.output_values)
        self.keys, self.values = self.output_keys, self.output_values
        return self.keys, self.values


def install_packed_cache(cache, packed):
    keys, values = packed.unbind(0)
    views = list(zip(keys.unbind(0), values.unbind(0)))
    for layer, (key, value) in zip(cache.layers, views):
        layer.keys, layer.values = key, value
    cache._graph_packed_kv = packed
    cache._graph_packed_kv_views = views


def can_pack(cache):
    first = cache.layers[0].keys
    return first.shape[0] == 1 and all(
        layer.keys.shape == layer.values.shape == first.shape
        and layer.keys.is_contiguous() and layer.values.is_contiguous()
        for layer in cache.layers)


class PackedNativeDecodeGraph:
    def __init__(self, owner, original, kwargs, cache, rope):
        self.inputs = clone_tree((kwargs, rope))
        shape = cache.layers[0].keys.shape
        self.packed_input = cache.layers[0].keys.new_empty((2, len(cache.layers), *shape))
        out_shape = (*shape[:-2], shape[-2]+1, shape[-1])
        self.packed_output = cache.layers[0].keys.new_empty((2, len(cache.layers), *out_shape))
        self.cache = copy.copy(cache)
        self.cache.layers = [copy.copy(layer) for layer in cache.layers]
        install_packed_cache(self.cache, self.packed_input)
        self.input_tensors = [tensor for pair in self.cache._graph_packed_kv_views for tensor in pair]
        self._copy_cache(cache)
        output_keys, output_values = self.packed_output.unbind(0)
        self.states = []
        for layer, output_key, output_value in zip(self.cache.layers, output_keys.unbind(0), output_values.unbind(0)):
            layer.__class__ = PackedDynamicLayer
            layer.output_keys, layer.output_values = output_key, output_value
            self.states.append(dict(layer.__dict__))

        def forward():
            for layer, state in zip(self.cache.layers, self.states):
                layer.__dict__.update(state)
            enabled, active, old_rope = owner.enabled, owner.in_full_decode, owner.model.model.rope_deltas
            owner.enabled, owner.in_full_decode = False, True
            owner.model.model.rope_deltas = self.inputs[1]
            try:
                return original(**self.inputs[0], past_key_values=self.cache)
            finally:
                owner.enabled, owner.in_full_decode = enabled, active
                owner.model.model.rope_deltas = old_rope

        with torch.cuda.device(kwargs['input_ids'].device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.graph.replay()

    def _copy_cache(self, cache):
        packed = getattr(cache, '_graph_packed_kv', None)
        views = getattr(cache, '_graph_packed_kv_views', ())
        if (packed is not None and packed.shape == self.packed_input.shape
            and len(views) == len(cache.layers)
            and all(layer.keys is key and layer.values is value for layer, (key, value) in zip(cache.layers, views))):
            self.packed_input.copy_(packed)
        else:
            torch._foreach_copy_(self.input_tensors, [t for layer in cache.layers for t in (layer.keys, layer.values)])

    def replay(self, kwargs, cache, rope):
        copy_tree(self.inputs, (kwargs, rope))
        self._copy_cache(cache)
        self.graph.replay()
        values = clone_tree({k:v for k,v in self.output.items() if k != 'past_key_values'})
        install_packed_cache(cache, self.packed_output.clone())
        values['past_key_values'] = cache
        return type(self.output)(**values)
