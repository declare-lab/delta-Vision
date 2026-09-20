"""CUDA Graph replay of native Qwen decoder layers, preserving pruning and kernels.

Only explicit warmup may capture a new shape. Timed calls replay warmed graphs or
fall back to the original forward. Selection and DeepStack remain in native code.
"""
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
    def __init__(self, model, max_shapes=12, vision=True, full_decode=True, prefill_layers=True, fused_norms=False, packed_kv=False):
        if model.model.language_model.config._attn_implementation not in (
            "flash_attention_2", "flash_attention_3", "flash_attention_4",
            "kernels-community/vllm-flash-attn3", "kernels-community/flash-attn4",
        ):
            raise ValueError("Native decoder graphs require a native FlashAttention backend")
        self.allow_capture = False
        self.packed_kv = packed_kv
        self.norm_optimizer = None
        if fused_norms and getattr(model, '_benchmark_fused_qwen_norms', None) is None:
            from src.qwen_fused_norm import FusedQwenNorms
            self.norm_optimizer = FusedQwenNorms(model)
            model._benchmark_fused_qwen_norms = self.norm_optimizer
        self.enabled = True
        self.captures = self.replays = self.fallbacks = 0
        self.originals = []
        self.entries = []
        self.max_shapes = max_shapes
        self.position_metadata = {}
        self.model = model
        self.in_full_decode = False
        self.decode_handles = []
        self.model_forward = None
        self.function_originals = []
        self.audit_flags = []
        for module in (model.model, model.model.language_model):
            self.audit_flags.append((module, getattr(module, "_pruning_audit_enabled", None)))
            module._pruning_audit_enabled = False

        implementation = sys.modules[type(model.model.language_model).__module__]
        for name in ("_divprune_select_tokens", "_zoo_select_tokens", "_dart_neighbor_indices"):
            if hasattr(implementation, name):
                original = getattr(implementation, name)
                entries = OrderedDict()
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
                    from src.qwen_packed_decode_graph import PackedNativeDecodeGraph, can_pack
                    if can_pack(cache):
                        graph_type = PackedNativeDecodeGraph
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
                while len(entries) > 2:
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
                while len(entries) > self.max_shapes:
                    entries.popitem(last=False)
            entries.move_to_end(key)
            self.replays += 1
            return entry.replay(hidden_states, graph_kwargs, layer)
        return forward

    def stats(self):
        return dict(captures=self.captures, layer_replays=self.replays, cold_layer_fallbacks=self.fallbacks)

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
