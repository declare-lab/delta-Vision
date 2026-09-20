"""Combine native Q/K/V and gate/up GEMMs for single-token Qwen inference.

Parameters remain separate views of one allocation, preserving state_dict names,
values, training fallback and total parameter memory. Full prefill stays native:
splitting a multi-row merged GEMM would make the norm inputs noncontiguous.
"""
import torch
import torch.nn.functional as F


class QwenFusedProjections:
    def __init__(self, model):
        self.enabled = True
        self.groups = []
        self.originals = []
        for layer in model.model.language_model.layers:
            self._group([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            self._group([layer.mlp.gate_proj, layer.mlp.up_proj])

    def _group(self, modules):
        if any(m.bias is not None for m in modules):
            raise ValueError('Qwen projection fusion currently requires bias-free layers')
        sizes = [m.weight.shape[0] for m in modules]
        with torch.no_grad():
            weight = torch.cat([m.weight for m in modules], dim=0)
            for module, view in zip(modules, weight.split(sizes, dim=0)):
                module.weight.data = view
        state = dict(weight=weight, sizes=sizes, source=None, outputs=None)
        self.groups.append(state)
        for index, module in enumerate(modules):
            original = module.forward
            self.originals.append((module, original))
            def forward(x, _index=index, _original=original):
                if not self.enabled or torch.is_grad_enabled() or x.ndim != 3 or x.shape[:2] != (1, 1):
                    return _original(x)
                if _index == 0:
                    state['source'] = x
                    state['outputs'] = F.linear(x, state['weight']).split(state['sizes'], dim=-1)
                if state['source'] is not x or state['outputs'] is None:
                    return _original(x)
                output = state['outputs'][_index]
                if _index + 1 == len(modules):
                    state['source'] = state['outputs'] = None
                return output
            module.forward = forward

    def remove(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.groups.clear()
