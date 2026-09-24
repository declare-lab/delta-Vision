"""Qwen3-VL native/adapter CUDA Graph execution and fixed-length greedy decoding."""


# CUDA Graph replay of native Qwen decoder layers, preserving pruning and kernels.
from collections import OrderedDict
import copy
import sys
from functools import wraps

import torch
from transformers.cache_utils import DynamicLayer
from transformers.modeling_flash_attention_utils import _is_packed_sequence
from transformers.utils import ModelOutput


def signature(value):
    if torch.is_tensor(value):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, dict):
        return tuple((k, signature(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(signature(v) for v in value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(type(value))


def clone_tree(value):
    sources, targets = [], []
    result = _clone_structure(value, sources, targets)
    if targets:
        torch._foreach_copy_(targets, sources)
    return result


def clone_kv_tensors(tensors):
    """Own disjoint KV views with one allocation, preserving dense layouts.

    All tensors must share dtype/device. Keep logits/metadata in separate
    allocations so a KV storage-byte measurement does not count them as KV.
    """
    if not tensors:
        return []
    first = tensors[0]
    assert all(t.device == first.device and t.dtype == first.dtype for t in tensors)
    buffer = torch.empty(sum(t.numel() for t in tensors), device=first.device, dtype=first.dtype)
    targets, offset = [], 0
    for tensor in tensors:
        stride = tensor.stride()
        expected, dense = 1, True
        for step, size in sorted((s, n) for s, n in zip(stride, tensor.shape) if n > 1):
            dense &= step == expected
            expected *= size
        if not dense and tensor.numel() > 0:
            strides, step = [], 1
            for size in reversed(tensor.shape):
                strides.append(step)
                step *= max(size, 1)
            stride = tuple(reversed(strides))
        targets.append(buffer.as_strided(tensor.shape, stride, offset))
        offset += tensor.numel()
    torch._foreach_copy_(targets, tensors)
    return targets


def _clone_structure(value, sources, targets):
    if torch.is_tensor(value):
        target = torch.empty_like(value, memory_format=torch.preserve_format)
        sources.append(value)
        targets.append(target)
        return target
    if isinstance(value, ModelOutput):
        return type(value)(**{k: _clone_structure(v, sources, targets) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _clone_structure(v, sources, targets) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_structure(v, sources, targets) for v in value)
    if isinstance(value, list):
        return [_clone_structure(v, sources, targets) for v in value]
    return value


def copy_tree(target, source):
    targets, sources = [], []
    def collect(a, b):
        if torch.is_tensor(a):
            targets.append(a)
            sources.append(b)
        elif isinstance(a, dict):
            for key in a:
                collect(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            for aa, bb in zip(a, b):
                collect(aa, bb)
    collect(target, source)
    if targets:
        torch._foreach_copy_(targets, sources)


class LayerCache:
    def __init__(self, layer, index):
        self.layer = copy.copy(layer)
        self.index = index
        self.initial = dict(self.layer.__dict__)
        for key in ("keys", "values"):
            self.initial[key] = clone_tree(self.initial.get(key))

    def reset(self):
        self.layer.__dict__.update(self.initial)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        assert layer_idx == self.index
        return self.layer.update(key_states, value_states, *args, **kwargs)


class LayerGraph:
    def __init__(self, original, hidden, kwargs, layer, index):
        self.inputs = clone_tree((hidden, kwargs))
        self.cache = LayerCache(layer, index) if layer is not None else None

        def forward():
            h, kw = self.inputs
            if self.cache is not None:
                self.cache.reset()
            return original(h, past_key_values=self.cache, **kw)

        with torch.cuda.device(hidden.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.cache_output = (self.cache.layer.keys, self.cache.layer.values) if self.cache is not None else None
            self.graph.replay()

    def replay(self, hidden, kwargs, layer):
        copy_tree(self.inputs, (hidden, kwargs))
        if self.cache is not None:
            for key in ("keys", "values"):
                initial = self.cache.initial.get(key)
                if torch.is_tensor(initial):
                    initial.copy_(getattr(layer, key))
        self.graph.replay()
        if self.cache_output is not None:
            # Own each returned cache tensor; later graph replays cannot overwrite
            # a caller's earlier generation or an earlier layer's cached states.
            layer.keys, layer.values = (t.clone() for t in self.cache_output)
            layer.is_initialized = True
            layer.dtype, layer.device = layer.keys.dtype, layer.keys.device
        return self.output.clone()


class NativeDecoderGraphs:
    def __init__(self, model, max_shapes=8, vision=True, full_decode=True, prefill_layers=True, fused_norms=False, packed_kv=False, max_prefill_shapes=1):
        if model.model.language_model.config._attn_implementation not in (
            "flash_attention_2", "flash_attention_3", "flash_attention_4",
            "kernels-community/vllm-flash-attn3", "kernels-community/flash-attn4",
        ):
            raise ValueError("Native decoder graphs require a native FlashAttention backend")
        self.allow_capture = False
        self.packed_kv = packed_kv
        self.norm_optimizer = None
        if fused_norms and getattr(model, '_benchmark_fused_qwen_norms', None) is None:
            from src.kernels import FusedQwenNorms
            self.norm_optimizer = FusedQwenNorms(model)
            model._benchmark_fused_qwen_norms = self.norm_optimizer
        self.enabled = True
        self.captures = self.replays = self.fallbacks = 0
        self.originals = []
        self.entries = []
        self.max_shapes = max_shapes
        # A prefill shape owns full-sequence activations at every layer. Its
        # cache must not inherit the larger growing-KV decode-shape budget.
        # Keep only the current prefill/vision shape, as the adapter does.
        if max_shapes < 1 or max_prefill_shapes < 1:
            raise ValueError('Graph cache capacities must be positive')
        self.max_prefill_shapes = max_prefill_shapes
        self.position_metadata = {}
        self.model = model
        self.in_full_decode = False
        self.decode_handles = []
        self.model_forward = None
        self.function_originals = []
        self.selector_entries = []
        self.audit_flags = []
        for module in (model.model, model.model.language_model):
            self.audit_flags.append((module, getattr(module, "_pruning_audit_enabled", None)))
            module._pruning_audit_enabled = False

        implementation = sys.modules[type(model.model.language_model).__module__]
        for name in ("_divprune_select_tokens", "_zoo_select_tokens", "_dart_neighbor_indices"):
            if hasattr(implementation, name):
                original = getattr(implementation, name)
                entries = OrderedDict()
                self.selector_entries.append(entries)
                self.entries.append(entries)
                self.function_originals.append((implementation, name, original))
                setattr(implementation, name, self._wrap_function(original, entries))

        def begin_language(module, args, kwargs):
            self.position_metadata.clear()
        self.handle = model.model.language_model.register_forward_pre_hook(begin_language, with_kwargs=True)
        for index, layer in enumerate(model.model.language_model.layers if prefill_layers else []):
            original = layer.forward
            entries = OrderedDict()
            self.originals.append((layer, original))
            self.entries.append(entries)
            layer.forward = self._wrap(original, index, entries)
        if vision:
            visual = model.model.visual
            original = visual.forward
            entries = OrderedDict()
            self.entries.append(entries)
            self.originals.append((visual, original))
            visual.forward = self._wrap_vision(original, entries)
        if full_decode:
            entries = OrderedDict()
            self.entries.append(entries)
            self.model_forward = model.forward
            model.forward = self._wrap_model_decode(self.model_forward, entries)
            def decode_positions(module, args, kwargs):
                # FA2's packed-sequence test is always false for q_len=1.
                # Resolve it on the host without a scalar read inside capture.
                if self.in_full_decode and "position_ids" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs.pop("position_ids")
                    return args, kwargs
            for layer in model.model.language_model.layers:
                self.decode_handles.append(layer.self_attn.register_forward_pre_hook(decode_positions, with_kwargs=True))

    def _wrap_model_decode(self, original, entries):
        @wraps(original)
        def forward(*args, **kwargs):
            ids = kwargs.get("input_ids")
            cache = kwargs.get("past_key_values")
            if (not self.enabled or args or torch.is_grad_enabled() or ids is None or ids.shape != (1, 1)
                or kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None
                or cache is None or getattr(cache, "offloading", False)
                or kwargs.get("return_dict") is False
                or any(type(layer) is not DynamicLayer or not layer.is_initialized for layer in cache.layers)):
                return original(*args, **kwargs)
            prepared = dict(kwargs)
            prepared.pop("past_key_values")
            mask = prepared.get("attention_mask")
            if mask is not None:
                if not bool(torch.all(mask == 1)):
                    return original(*args, **kwargs)
                prepared["attention_mask"] = None
            rope = self.model.model.rope_deltas
            key = signature((prepared, [(layer.keys, layer.values) for layer in cache.layers], rope))
            entry = entries.get(key)
            if entry is None:
                if not self.allow_capture:
                    self.fallbacks += 1
                    return original(*args, **kwargs)
                graph_type = NativeDecodeGraph
                if self.packed_kv:
                    from src.graphs import (PackedNativeDecodeGraph, can_pack,
                        GroupedPackedNativeDecodeGraph, can_pack_grouped)
                    if can_pack(cache):
                        graph_type = PackedNativeDecodeGraph
                    elif can_pack_grouped(cache):
                        graph_type = GroupedPackedNativeDecodeGraph
                entry = graph_type(self, original, prepared, cache, rope)
                entries[key] = entry
                self.captures += 1
                while len(entries) > self.max_shapes:
                    entries.popitem(last=False)
            entries.move_to_end(key)
            self.replays += 1
            return entry.replay(prepared, cache, rope)
        return forward

    def _wrap_function(self, original, entries):
        def forward(*args, **kwargs):
            if not self.enabled or torch.is_grad_enabled():
                return original(*args, **kwargs)
            values = (*args, kwargs)
            key = signature(values)
            entry = entries.get(key)
            if entry is None:
                if not self.allow_capture:
                    self.fallbacks += 1
                    return original(*args, **kwargs)
                entry = PlainGraph(lambda *packed: original(*packed[:-1], **packed[-1]), values)
                entries[key] = entry
                self.captures += 1
                while len(entries) > self.max_shapes:
                    entries.popitem(last=False)
            entries.move_to_end(key)
            self.replays += 1
            return entry.replay(values)
        return forward

    def _wrap_vision(self, original, entries):
        def forward(hidden_states, grid_thw, **kwargs):
            if not self.enabled or torch.is_grad_enabled():
                return original(hidden_states, grid_thw, **kwargs)
            from src.model import qwen_visual_grid_metadata
            metadata = qwen_visual_grid_metadata(self.model, grid_thw)
            prepared = dict(kwargs, **metadata)
            if hasattr(self.model.model.visual, "_visionzip_attn_mean"):
                prepared["visionzip_boundaries"] = metadata["cu_seqlens"].tolist()
            values = (hidden_states, grid_thw, prepared)
            key = signature(values)
            entry = entries.get(key)
            if entry is None:
                if not self.allow_capture:
                    self.fallbacks += 1
                    return original(hidden_states, grid_thw, **kwargs)
                entry = PlainGraph(lambda h, grid, kw: original(h, grid, **kw), values)
                entries[key] = entry
                self.captures += 1
                while len(entries) > self.max_prefill_shapes:
                    entries.popitem(last=False)
            entries.move_to_end(key)
            self.replays += 1
            output = entry.replay(values)
            if hasattr(self.model.model.visual, "_visionzip_attn_mean"):
                # The caller consumes and clears these side outputs after each
                # image. Python assignments inside capture do not run on replay.
                self.model.model.visual._visionzip_attn_mean = output[1]
                self.model.model.visual._visionzip_attn_key = output[2]
            return output
        return forward

    def _prepare_kwargs(self, kwargs):
        result = dict(kwargs)
        positions = result.pop("position_ids", None)
        if positions is not None:
            key = (positions.data_ptr(), tuple(positions.shape), tuple(positions.stride()))
            if key not in self.position_metadata:
                metadata = {}
                if positions.shape[-1] > 1:
                    if not bool(torch.all(positions.diff(dim=-1) > 0)):
                        return None
                    if bool(_is_packed_sequence(positions, batch_size=positions.shape[0])):
                        size = positions.shape[-1]
                        cu = torch.tensor([0, size], dtype=torch.int32, device=positions.device)
                        metadata = dict(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=size, max_length_k=size)
                self.position_metadata[key] = (positions, metadata)
            # Preserve explicit complete FA metadata; do not reinterpret partial metadata.
            keys = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            supplied = [result.get(k) is not None for k in keys]
            if any(supplied) and not all(supplied):
                return None
            if not any(supplied):
                result.update(self.position_metadata[key][1])
        return result

    def _wrap(self, original, index, entries):
        def forward(hidden_states, *args, **kwargs):
            if not self.enabled or args or torch.is_grad_enabled() or hidden_states.device.type != "cuda":
                return original(hidden_states, *args, **kwargs)
            cache = kwargs.get("past_key_values")
            if cache is not None and (getattr(cache, "offloading", False) or index >= len(cache.layers) or type(cache.layers[index]) is not DynamicLayer):
                return original(hidden_states, **kwargs)
            # A nonempty padding mask needs dynamic unpadding; keep that path native.
            if kwargs.get("attention_mask") is not None:
                return original(hidden_states, **kwargs)
            graph_kwargs = self._prepare_kwargs(kwargs)
            if graph_kwargs is None:
                return original(hidden_states, **kwargs)
            graph_kwargs.pop("past_key_values", None)
            layer = cache.layers[index] if cache is not None else None
            try:
                key = signature((hidden_states, graph_kwargs, None if layer is None else
                                 (layer.is_initialized, layer.keys, layer.values)))
            except TypeError:
                return original(hidden_states, **kwargs)
            entry = entries.get(key)
            if entry is None:
                if not self.allow_capture:
                    self.fallbacks += 1
                    return original(hidden_states, **kwargs)
                entry = LayerGraph(original, hidden_states, graph_kwargs, layer, index)
                entries[key] = entry
                self.captures += 1
                limit = self.max_prefill_shapes if hidden_states.shape[1] > 1 else self.max_shapes
                while len(entries) > limit:
                    entries.popitem(last=False)
            entries.move_to_end(key)
            self.replays += 1
            return entry.replay(hidden_states, graph_kwargs, layer)
        return forward

    def begin_request(self):
        """Drop selector graphs belonging to earlier requests before warmup.

        DART can need multiple helper shapes within one request. Keep those
        during its warmup/replays, without retaining prior-request helpers.
        """
        for entries in self.selector_entries:
            entries.clear()

    def stats(self):
        return dict(captures=self.captures, layer_replays=self.replays, cold_layer_fallbacks=self.fallbacks,
                    max_decode_shapes=self.max_shapes, max_prefill_shapes=self.max_prefill_shapes,
                    max_vision_shapes=self.max_prefill_shapes,
                    selector_shapes=[len(entries) for entries in self.selector_entries])

    def remove(self):
        if self.norm_optimizer is not None:
            self.norm_optimizer.remove()
            if getattr(self.model, '_benchmark_fused_qwen_norms', None) is self.norm_optimizer:
                del self.model._benchmark_fused_qwen_norms
            self.norm_optimizer = None
        if self.model_forward is not None:
            self.model.forward = self.model_forward
        for handle in self.decode_handles:
            handle.remove()
        for layer, original in self.originals:
            layer.forward = original
        for module, name, original in self.function_originals:
            setattr(module, name, original)
        for module, previous in self.audit_flags:
            if previous is None:
                del module._pruning_audit_enabled
            else:
                module._pruning_audit_enabled = previous
        self.handle.remove()
        self.entries.clear()
        self.position_metadata.clear()


class NativeDecodeGraph:
    """One native forward graph for each growing KV length; no fixed-step replay."""
    def __init__(self, owner, original, kwargs, cache, rope):
        self.inputs = clone_tree((kwargs, [(layer.keys, layer.values) for layer in cache.layers], rope))
        self.cache = copy.copy(cache)
        self.cache.layers = [copy.copy(layer) for layer in cache.layers]
        states = []
        for layer, (keys, values) in zip(self.cache.layers, self.inputs[1]):
            state = dict(layer.__dict__, keys=keys, values=values)
            states.append(state)

        def forward():
            for layer, state in zip(self.cache.layers, states):
                layer.__dict__.update(state)
            enabled, active, old_rope = owner.enabled, owner.in_full_decode, owner.model.model.rope_deltas
            owner.enabled, owner.in_full_decode = False, True
            owner.model.model.rope_deltas = self.inputs[2]
            try:
                return original(**self.inputs[0], past_key_values=self.cache)
            finally:
                owner.enabled, owner.in_full_decode = enabled, active
                owner.model.model.rope_deltas = old_rope

        with torch.cuda.device(kwargs["input_ids"].device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.output_kv = [(layer.keys, layer.values) for layer in self.cache.layers]
            self.graph.replay()

    def replay(self, kwargs, cache, rope):
        copy_tree(self.inputs, (kwargs, [(layer.keys, layer.values) for layer in cache.layers], rope))
        self.graph.replay()
        values = clone_tree({k:v for k,v in self.output.items() if k != "past_key_values"})
        copied = clone_kv_tensors([tensor for pair in self.output_kv for tensor in pair])
        output_kv = list(zip(copied[::2], copied[1::2]))
        for layer, (keys, vals) in zip(cache.layers, output_kv):
            layer.keys, layer.values = keys, vals
        values["past_key_values"] = cache
        return type(self.output)(**values)


class PlainGraph:
    def __init__(self, forward, inputs):
        self.inputs = clone_tree(inputs)
        device = inputs[0].device
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward(*self.inputs)
            self.graph.replay()

    def replay(self, inputs):
        copy_tree(self.inputs, inputs)
        self.graph.replay()
        return clone_tree(self.output)


# Native FA2 decode with packed, owned KV input/output allocations.
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


def install_grouped_cache(cache, groups):
    """Own one contiguous allocation per KV length; preserve layer order."""
    views = [None] * len(cache.layers)
    for indices, buffer in groups:
        for index, key, value in zip(indices, buffer[0].unbind(0), buffer[1].unbind(0)):
            cache.layers[index].keys, cache.layers[index].values = key, value
            views[index] = (key, value)
    assert all(view is not None for view in views)
    cache._graph_grouped_kv = groups
    cache._graph_grouped_kv_views = views


def can_pack_grouped(cache):
    return all(layer.keys.shape == layer.values.shape and layer.keys.shape[0] == 1
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


class GroupedPackedNativeDecodeGraph:
    """The same native decoder graph with separate buffers for each KV length."""
    def __init__(self, owner, original, kwargs, cache, rope):
        self.inputs = clone_tree((kwargs, rope))
        indices_by_shape = {}
        for index, layer in enumerate(cache.layers):
            indices_by_shape.setdefault(tuple(layer.keys.shape), []).append(index)
        self.input_groups, self.output_groups = [], []
        for shape, indices in indices_by_shape.items():
            tensor = cache.layers[indices[0]].keys
            self.input_groups.append((tuple(indices), tensor.new_empty((2, len(indices), *shape))))
            out_shape = (*shape[:-2], shape[-2] + 1, shape[-1])
            self.output_groups.append((tuple(indices), tensor.new_empty((2, len(indices), *out_shape))))
        self.cache = copy.copy(cache)
        self.cache.layers = [copy.copy(layer) for layer in cache.layers]
        install_grouped_cache(self.cache, self.input_groups)
        self.input_tensors = [t for pair in self.cache._graph_grouped_kv_views for t in pair]
        self._copy_cache(cache)
        for indices, buffer in self.output_groups:
            for index, key, value in zip(indices, buffer[0].unbind(0), buffer[1].unbind(0)):
                layer = self.cache.layers[index]
                layer.__class__ = PackedDynamicLayer
                layer.output_keys, layer.output_values = key, value
        self.states = [dict(layer.__dict__) for layer in self.cache.layers]

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
        groups = getattr(cache, '_graph_grouped_kv', ())
        views = getattr(cache, '_graph_grouped_kv_views', ())
        valid = (len(views) == len(cache.layers) and len(groups) == len(self.input_groups)
                 and all(layer.keys is key and layer.values is value
                         for layer, (key, value) in zip(cache.layers, views))
                 and all(i == j and a.shape == b.shape
                         for (i, a), (j, b) in zip(groups, self.input_groups)))
        if valid:
            for (_, source), (_, target) in zip(groups, self.input_groups):
                target.copy_(source)
        else:
            torch._foreach_copy_(self.input_tensors, [t for layer in cache.layers for t in (layer.keys, layer.values)])

    def replay(self, kwargs, cache, rope):
        copy_tree(self.inputs, (kwargs, rope))
        self._copy_cache(cache)
        self.graph.replay()
        values = clone_tree({k: v for k, v in self.output.items() if k != 'past_key_values'})
        install_grouped_cache(cache, [(indices, buffer.clone()) for indices, buffer in self.output_groups])
        values['past_key_values'] = cache
        return type(self.output)(**values)


# Whole native Qwen text prefill graph for unpadded, uncompressed requests.
from transformers.cache_utils import DynamicCache, DynamicLayer


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


# Warmed FA2 graphs for genuine single-token adapter decode with growing KV.
def cache_containers(cache):
    result = dict(cache)
    result["layers"] = [dict(layer) for layer in cache["layers"]]
    return result


class AdapterDecodeGraph:
    def __init__(self, model, adapter, tokens, cache, kwargs, plan):
        from src.model import qwen_embedding_adapter_decode_step
        self.inputs = clone_tree((tokens, cache, kwargs, plan))

        def forward():
            tokens, initial, kwargs, plan = self.inputs
            return qwen_embedding_adapter_decode_step(model, adapter, tokens,
                cache_containers(initial), attention_plan=plan, **kwargs)

        with torch.cuda.device(tokens.device):
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

    def replay(self, tokens, cache, kwargs, plan):
        copy_tree(self.inputs, (tokens, cache, kwargs, plan))
        self.graph.replay()
        logits, output_cache = self.output
        # Visual K/V and prefix metadata are immutable during fast decode.
        # Own only the updated tensors; retain the caller's visual cache.
        updated = clone_tree((logits, output_cache["text_mask"],
            output_cache["next_position_ids"], output_cache["next_text_positions"]))
        copied = clone_kv_tensors([layer[name] for layer in output_cache["layers"] for name in ("text_key", "text_value")])
        result = cache_containers(cache)
        for layer, (key, value) in zip(result["layers"], zip(copied[::2], copied[1::2])):
            layer["text_key"], layer["text_value"] = key, value
        result["text_mask"], result["next_position_ids"], result["next_text_positions"] = updated[1:]
        result["dense_decode_ready"] = output_cache.get("dense_decode_ready", False)
        return updated[0], result


class QwenAdapterDecodeGraphs:
    def __init__(self, model, adapter, max_shapes=12):
        self.model, self.adapter = model, adapter
        self.entries = OrderedDict()
        self.max_shapes = max_shapes
        self.enabled = True
        self.allow_capture = False
        self.captures = self.replays = self.cold_fallbacks = 0

    def __call__(self, model, adapter, tokens, cache, **kwargs):
        from src.model import qwen_embedding_adapter_decode_step, _qwen_decode_attention_mask
        from src.attention import decode_plan
        if not self.enabled:
            return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, **kwargs)
        if cache.get("attention_implementation") != "flash_attention_2":
            raise ValueError("Adapter decode graphs require FA2")
        if cache.get("dense_decode_ready", False) and kwargs.get("token_active_mask") is None:
            plan = {"dense_decode": True}
        else:
            mask = _qwen_decode_attention_mask(cache, cache["next_position_ids"], kwargs.get("token_active_mask"))
            plan = decode_plan(mask)
        key = signature((tokens, cache, kwargs, plan))
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                self.cold_fallbacks += 1
                return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, attention_plan=plan, **kwargs)
            entry = AdapterDecodeGraph(model, adapter, tokens, cache, kwargs, plan)
            self.entries[key] = entry
            self.captures += 1
            while len(self.entries) > self.max_shapes:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        self.replays += 1
        return entry.replay(tokens, cache, kwargs, plan)

    def stats(self):
        return dict(captures=self.captures, replays=self.replays, cold_fallbacks=self.cold_fallbacks)


# Single-token graphs sharing immutable visual KV across a request's steps.
class SharedVisualDecodeGraph(AdapterDecodeGraph):
    def __init__(self, model, adapter, tokens, cache, kwargs, plan, visual):
        from src.model import qwen_embedding_adapter_decode_step
        mutable = cache_containers(cache)
        mutable["layers"] = [{k:v for k,v in layer.items() if k not in ("visual_key", "visual_value")}
            for layer in cache["layers"]]
        # The inherited replay copies only this mutable portion. Graph kernels
        # read the separately owned visual tensors shared by all growing lengths.
        self.inputs = clone_tree((tokens, mutable, kwargs, plan))
        initial = cache_containers(self.inputs[1])
        for layer, (key, value) in zip(initial["layers"], visual):
            layer.update(visual_key=key, visual_value=value)

        def forward():
            token, _, kw, prepared_plan = self.inputs
            return qwen_embedding_adapter_decode_step(model, adapter, token,
                cache_containers(initial), attention_plan=prepared_plan, **kw)

        with torch.cuda.device(tokens.device):
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


class SharedVisualDecodeGraphs(QwenAdapterDecodeGraphs):
    """Fast-path visual KV is immutable and newly owned by each prefill.

    Retain source tensor references and check their identities, preventing an
    allocator's reused address from being mistaken for the previous request.
    Every new request refreshes the shared visual buffers before its first step.
    As with the existing graph runners, calls are sequential on one CUDA stream.
    """
    def __init__(self, model, adapter, max_shapes=12):
        super().__init__(model, adapter, max_shapes)
        self.visual_banks = OrderedDict()
        self.visual_sources = ()
        self.current_visual = None
        self.visual_prefix_copies = 0

    def _visual(self, cache):
        sources = tuple(t for layer in cache["layers"] for t in (layer["visual_key"], layer["visual_value"]))
        if len(sources) == len(self.visual_sources) and all(a is b for a,b in zip(sources,self.visual_sources)):
            return self.current_visual
        key = signature(sources)
        visual = self.visual_banks.get(key)
        if visual is None:
            cloned = clone_tree(sources)
            visual = tuple(zip(cloned[::2],cloned[1::2]))
            self.visual_banks[key] = visual
            while len(self.visual_banks) > 2:
                self.visual_banks.popitem(last=False)
        else:
            copy_tree(visual, tuple(zip(sources[::2],sources[1::2])))
        self.visual_banks.move_to_end(key)
        self.visual_sources, self.current_visual = sources, visual
        self.visual_prefix_copies += 1
        return visual

    def __call__(self, model, adapter, tokens, cache, **kwargs):
        from src.model import qwen_embedding_adapter_decode_step, _qwen_decode_attention_mask
        from src.attention import decode_plan
        if not self.enabled:
            return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, **kwargs)
        if cache.get("attention_implementation") != "flash_attention_2":
            raise ValueError("Adapter decode graphs require FA2")
        if cache.get("dense_decode_ready", False) and kwargs.get("token_active_mask") is None:
            plan = {"dense_decode": True}
        else:
            plan = decode_plan(_qwen_decode_attention_mask(cache, cache["next_position_ids"], kwargs.get("token_active_mask")))
        visual = self._visual(cache)
        key = (id(visual), signature((tokens, cache, kwargs, plan)))
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                self.cold_fallbacks += 1
                return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, attention_plan=plan, **kwargs)
            entry = SharedVisualDecodeGraph(model, adapter, tokens, cache, kwargs, plan, visual)
            self.entries[key] = entry
            self.captures += 1
            while len(self.entries) > self.max_shapes:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        self.replays += 1
        return entry.replay(tokens, cache, kwargs, plan)

    def stats(self):
        return dict(super().stats(), visual_prefix_copies=self.visual_prefix_copies)


# Adapter fast prefill followed by the native Qwen FA2 cached decoder.
class NativeAdapterDecodeGraphs:
    def __init__(self, model, adapter, max_shapes=8, packed_kv=False):
        self.model, self.adapter = model, adapter
        # Match the native/base speed benchmark decode-graph budget.
        self.max_shapes = max_shapes
        self.native = NativeDecoderGraphs(model, max_shapes=max_shapes, vision=False, prefill_layers=False, packed_kv=packed_kv)
        self.native.enabled = False  # Teacher/prefill calls retain their own execution path.
        self.fallback = SharedVisualDecodeGraphs(model, adapter, max_shapes)
        self.enabled = True
        self.cache_conversions = 0

    @property
    def allow_capture(self):
        return self.native.allow_capture

    @allow_capture.setter
    def allow_capture(self, value):
        self.native.allow_capture = self.fallback.allow_capture = value

    def _convert(self, cache):
        layers = cache['layers']
        grouped = cache.get('_packed_native_groups')
        # Pack across layers with six copies rather than two concatenations and
        # a DynamicCache.update allocation per layer. Values and order are exact.
        if grouped:
            owned_groups=[(indices,buffer.clone()) for indices,buffer in grouped]
        elif '_packed_native_kv' in cache:
            # The optimized prefill writes every layer into its final layout.
            # One owned copy keeps later graph replays independent of this KV.
            packed = cache['_packed_native_kv'].clone()
            keys, values = packed.unbind(0)
        elif len({layer['visual_key'].shape[2] for layer in layers}) > 1:
            groups={}
            for i,layer in enumerate(layers):groups.setdefault(layer['visual_key'].shape[2],[]).append(i)
            owned_groups=[]
            for indices in groups.values():
                tensors=[]
                for field in ['key','value']:
                    tensors.append(torch.cat([torch.stack([layers[i]['visual_'+field] for i in indices]),
                                              torch.stack([layers[i]['text_'+field] for i in indices])],dim=-2))
                owned_groups.append((tuple(indices),torch.stack(tensors)))
            grouped=True
        else:
            keys = torch.cat([torch.stack([l['visual_key'] for l in layers]),
                              torch.stack([l['text_key'] for l in layers])], dim=-2)
            values = torch.cat([torch.stack([l['visual_value'] for l in layers]),
                                torch.stack([l['text_value'] for l in layers])], dim=-2)
        native = DynamicCache(config=self.model.model.language_model.config)
        assert len(native.layers) == len(layers)
        if grouped:
            from src.graphs import install_grouped_cache
            for indices,buffer in owned_groups:
                for i,key,value in zip(indices,buffer[0].unbind(0),buffer[1].unbind(0)):
                    native.layers[i].lazy_initialization(key,value)
            install_grouped_cache(native,owned_groups)
        for layer, key, value in ([] if grouped else zip(native.layers, keys.unbind(0), values.unbind(0))):
            if type(layer) is not DynamicLayer:
                raise TypeError('Native adapter decode requires ordinary DynamicLayer cache')
            layer.lazy_initialization(key, value)
            layer.keys, layer.values = key, value
        if self.native.packed_kv and '_packed_native_kv' in cache:
            native._graph_packed_kv = packed
            native._graph_packed_kv_views = [(layer.keys, layer.values) for layer in native.layers]
        positions = torch.cat([cache['next_text_positions'].unsqueeze(0), cache['next_position_ids']], dim=0)
        positions = positions + torch.arange(self.max_shapes + 1, device=positions.device)
        cache.clear()
        cache.update(attention_implementation='flash_attention_2', layers=[],
            _native_cache=native, _native_positions=positions, _native_steps=0)
        self.cache_conversions += 1

    def prepare_cache(self, cache):
        if ('_native_cache' not in cache and cache.get('dense_decode_ready', False)
            and cache.get('attention_implementation') == 'flash_attention_2'
            and cache['text_mask'].shape[0] == 1):
            self._convert(cache)
        return cache

    def __call__(self, model, adapter, tokens, cache, *, logits_to_keep=1, token_active_mask=None):
        if '_native_cache' not in cache:
            if (tokens.shape != (1,1) or token_active_mask is not None
                or not cache.get('dense_decode_ready', False)
                or cache.get('attention_implementation') != 'flash_attention_2'):
                self.fallback.enabled = self.enabled
                return self.fallback(model, adapter, tokens, cache,
                    logits_to_keep=logits_to_keep, token_active_mask=token_active_mask)
            self._convert(cache)
        if token_active_mask is not None:
            raise ValueError('Native adapter cache requires the unpadded single-request fast path')
        step = cache['_native_steps']
        positions = cache['_native_positions']
        positions = positions[:, :, step:step+1] if step < positions.shape[-1] else positions[:, :, :1] + step
        previous = self.native.enabled
        self.native.enabled = self.enabled
        try:
            output = model(input_ids=tokens, position_ids=positions,
                past_key_values=cache['_native_cache'], use_cache=True, return_dict=True,
                logits_to_keep=logits_to_keep)
        finally:
            self.native.enabled = previous
        cache['_native_cache'] = output.past_key_values
        cache['_native_steps'] = step + 1
        return output.logits, cache

    def stats(self):
        native, fallback = self.native.stats(), self.fallback.stats()
        return dict(captures=native['captures']+fallback['captures'],
            replays=native['layer_replays']+fallback['replays'],
            cold_fallbacks=native['cold_layer_fallbacks']+fallback['cold_fallbacks'],
            cache_conversions=self.cache_conversions)

    def remove(self):
        self.native.remove()


# Minimal fixed-length greedy loop around native Qwen forwards.
from types import SimpleNamespace


def fixed_greedy(model, inputs, tokens, *, output_logits=False):
    if inputs['input_ids'].shape[0] != 1 or tokens < 2:
        raise ValueError('Fixed greedy speed test requires one request and at least two tokens')
    model.model.rope_deltas = None
    positions = model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
    output = model(**inputs, position_ids=positions, use_cache=True, logits_to_keep=1, return_dict=True)
    next_positions = positions[:, :, -1:] + torch.arange(1, tokens, device=positions.device)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    generated, logits = [], []
    for step in range(tokens):
        scores = output.logits[:, -1].to(torch.float32, copy=True)
        if output_logits:
            logits.append(scores.clone())
        scores[:, eos] = -float('inf')
        token = scores.argmax(-1).view(1, 1)
        generated.append(token)
        if step + 1 < tokens:
            output = model(input_ids=token, position_ids=next_positions[:, :, step:step+1],
                past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
    return SimpleNamespace(sequences=torch.cat([inputs['input_ids'], *generated], dim=1),
        logits=tuple(logits), past_key_values=output.past_key_values)
