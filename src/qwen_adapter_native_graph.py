"""Adapter fast prefill followed by the native Qwen FA2 cached decoder."""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from src.qwen_native_graph import NativeDecoderGraphs
from src.qwen_adapter_shared_graph import SharedVisualDecodeGraphs


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
        # Pack across layers with six copies rather than two concatenations and
        # a DynamicCache.update allocation per layer. Values and order are exact.
        if '_packed_native_kv' in cache:
            # The optimized prefill writes every layer into its final layout.
            # One owned copy keeps later graph replays independent of this KV.
            packed = cache['_packed_native_kv'].clone()
            keys, values = packed.unbind(0)
        else:
            keys = torch.cat([torch.stack([l['visual_key'] for l in layers]),
                              torch.stack([l['text_key'] for l in layers])], dim=-2)
            values = torch.cat([torch.stack([l['visual_value'] for l in layers]),
                                torch.stack([l['text_value'] for l in layers])], dim=-2)
        native = DynamicCache(config=self.model.model.language_model.config)
        assert len(native.layers) == len(layers)
        for layer, key, value in zip(native.layers, keys.unbind(0), values.unbind(0)):
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
