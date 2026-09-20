"""Static visual embedding adapter for Qwen3.5's hybrid decoder.

The native mixer processes the ORIGINAL sequence, including every visual state
write and causal-convolution boundary. Only visual mixer outputs and visual FFNs
are discarded. This is the correctness implementation, not a speed benchmark.
"""
from contextlib import contextmanager
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def install_fast_kernels():
    """Explicit, fail-closed binding; never silently use the Python recurrence."""
    import triton
    from packaging.version import Version
    assert Version(triton.__version__) >= Version('3.7.1'), (
        'FLA gated backward on H200 requires Triton >=3.7.1; use the isolated qwen35_python dependencies')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    def chunk(q, k, v, **kw):
        allowed = {x: kw[x] for x in ('g', 'beta', 'initial_state', 'output_final_state',
                   'use_qk_l2norm_in_kernel', 'cu_seqlens') if x in kw}
        return chunk_gated_delta_rule(q, k, v, **allowed)

    def recurrent(q, k, v, **kw):
        allowed = {x: kw[x] for x in ('g', 'beta', 'initial_state', 'output_final_state',
                   'use_qk_l2norm_in_kernel', 'cu_seqlens') if x in kw}
        return fused_recurrent_gated_delta_rule(q, k, v, **allowed)

    def conv(x, w, bias=None, activation=None, **kw):
        return causal_conv1d_fn(x, w, bias, activation=activation)

    m.torch_chunk_gated_delta_rule = chunk
    m.torch_recurrent_gated_delta_rule = recurrent
    m.causal_conv1d_fn = conv
    m.causal_conv1d_update = causal_conv1d_update
    return {'linear_prefill': 'fla.chunk_gated_delta_rule',
            'linear_decode': 'fla.fused_recurrent_gated_delta_rule',
            'convolution': 'causal_conv1d CUDA', 'full_attention': 'flash_attention_2'}


class StaticVisualAdapter(nn.Module):
    def __init__(self, hidden_size=2560, num_layers=32, rank=128):
        super().__init__()
        self.down = nn.ModuleList(nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers))
        self.up = nn.ModuleList(nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers))
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, embeddings):
        # FP32 optimizer/master weights; matmuls use the backbone BF16 dtype.
        with torch.autocast(device_type=embeddings.device.type, dtype=embeddings.dtype,
                            enabled=embeddings.dtype in (torch.float16, torch.bfloat16)):
            return tuple(embeddings + up(F.silu(down(embeddings)))
                         for down, up in zip(self.down, self.up))


class VisualAdapterController:
    """Instance-local hooks. Native teacher forwards remain completely native."""
    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter
        self.mode = 'native'
        self.mask = self.visual_idx = self.text_idx = None
        self.predictions = None
        self.captured = []
        self.checkpoint_layers = False
        self.originals = []
        self.hook = model.model.language_model.register_forward_pre_hook(self._start, with_kwargs=True)
        for index, layer in enumerate(model.model.language_model.layers):
            original = layer.forward
            self.originals.append(original)
            def wrapped(layer, hidden_states, *args, _i=index, _orig=original, **kwargs):
                return self._layer(_i, layer, _orig, hidden_states, *args, **kwargs)
            layer.forward = types.MethodType(wrapped, layer)

    def _start(self, module, args, kwargs):
        h = kwargs.get('inputs_embeds')
        self.predictions = None
        self.visual_idx = self.text_idx = None
        if self.mode == 'native' or h is None or self.mask is None:
            return
        cache = kwargs.get('past_key_values')
        if cache is not None and cache.get_seq_length() > 0:
            return  # Native incremental text decode with hybrid states from prefill.
        assert h.shape[:2] == self.mask.shape and h.shape[0] == 1
        self.visual_idx = self.mask[0].nonzero().flatten()
        self.text_idx = (~self.mask[0]).nonzero().flatten()
        assert self.visual_idx.numel() and self.text_idx.numel()
        if self.mode == 'adapter':
            self.predictions = self.adapter(h.index_select(1, self.visual_idx).detach())
        elif self.mode == 'oracle':
            self.predictions = tuple(self.captured)
            assert len(self.predictions) == len(module.layers)
        elif self.mode == 'capture':
            self.captured = []

    def _layer(self, index, layer, original, hidden, *args, **kwargs):
        if self.mode == 'capture' and self.visual_idx is not None:
            self.captured.append(hidden.index_select(1, self.visual_idx).detach().clone())
        if self.predictions is None:
            return original(hidden, *args, **kwargs)
        assert not args, 'Expected native named decoder arguments'
        visual_idx, text_idx = self.visual_idx, self.text_idx
        prediction = self.predictions[index]

        def run(h, v):
            h = h.index_copy(1, visual_idx, v.to(h.dtype))
            normalized = layer.input_layernorm(h)
            call_kwargs = dict(kwargs)
            positions = call_kwargs.pop('position_embeddings')
            attention_mask = call_kwargs.pop('attention_mask', None)
            position_ids = call_kwargs.pop('position_ids', None)
            cache = call_kwargs.pop('past_key_values', None)
            if layer.block_type == 'linear_attention':
                mixed = layer.linear_attn(normalized, cache_params=cache,
                                          attention_mask=attention_mask, **call_kwargs)
            else:
                mixed, _ = layer.self_attn(normalized, position_embeddings=positions,
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=cache, **call_kwargs)
            text = h.index_select(1, text_idx) + mixed.index_select(1, text_idx)
            text = text + layer.mlp(layer.post_attention_layernorm(text))
            return h.index_copy(1, text_idx, text)

        if self.checkpoint_layers and torch.is_grad_enabled():
            assert kwargs.get('past_key_values') is None, 'No mutable caches in checkpointed training'
            return checkpoint(run, hidden, prediction, use_reentrant=False)
        return run(hidden, prediction)

    @contextmanager
    def activate(self, mode, mask=None, checkpoint_layers=False):
        assert self.mode == 'native', 'Nested adapter contexts are not supported'
        self.mode, self.mask, self.checkpoint_layers = mode, mask, checkpoint_layers
        try:
            yield self
        finally:
            self.mode, self.mask = 'native', None
            self.predictions = self.visual_idx = self.text_idx = None
            self.checkpoint_layers = False

    def close(self):
        self.hook.remove()
        for layer, original in zip(self.model.model.language_model.layers, self.originals):
            layer.forward = original
