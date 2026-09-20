"""Use exact BF16 RoPE kernels in one native Qwen model instance."""
from types import FunctionType, MethodType
import torch
from src.qwen_adapter_kernels import exact_rope


class QwenExactRoPE:
    def __init__(self, model):
        self.enabled = True
        self.fuse_norm = False
        self.originals = []
        self.norm_originals = []
        for layer in model.model.language_model.layers:
            attention = layer.self_attn
            original = attention.forward
            function = original.__func__
            original_rope = function.__globals__['apply_rotary_pos_emb']
            state = dict(active=False)
            def rope(query, key, cos, sin, unsqueeze_dim=1, _original=original_rope, _state=state, _attention=attention):
                if _state['active']:
                    from src.qwen_native_order_norm import native_order_norm_rope
                    return (native_order_norm_rope(query, _attention.q_norm.weight, _attention.q_norm.variance_epsilon, (cos,sin)),
                            native_order_norm_rope(key, _attention.k_norm.weight, _attention.k_norm.variance_epsilon, (cos,sin)))
                if (self.enabled and not torch.is_grad_enabled() and query.is_cuda
                    and query.dtype == key.dtype == cos.dtype == sin.dtype == torch.bfloat16
                    and unsqueeze_dim == 1):
                    return exact_rope(query, (cos, sin)), exact_rope(key, (cos, sin))
                return _original(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
            # Bind the unchanged native attention forward with a private globals
            # dictionary so other models and adapter/reference paths are untouched.
            local = FunctionType(function.__code__, dict(function.__globals__, apply_rotary_pos_emb=rope),
                function.__name__, function.__defaults__, function.__closure__)
            local.__kwdefaults__ = function.__kwdefaults__
            self.originals.append((attention, original))
            bound = MethodType(local, attention)
            for norm in (attention.q_norm, attention.k_norm):
                norm_forward = norm.forward
                self.norm_originals.append((norm, norm_forward))
                def maybe_defer(x, _state=state, _original=norm_forward):
                    return x if _state['active'] else _original(x)
                norm.forward = maybe_defer
            def forward(*args, _bound=bound, _state=state, _attention=attention, **kwargs):
                x = args[0] if args else kwargs.get('hidden_states')
                previous = _state['active']
                _state['active'] = (self.enabled and self.fuse_norm and not torch.is_grad_enabled()
                    and x is not None and x.is_cuda and x.dtype == torch.bfloat16
                    and _attention.head_dim == 128
                    and not any(n._forward_hooks or n._forward_pre_hooks for n in (_attention.q_norm, _attention.k_norm)))
                try:
                    return _bound(*args, **kwargs)
                finally:
                    _state['active'] = previous
            attention.forward = forward

    def remove(self):
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()
        for norm, original in self.norm_originals:
            norm.forward = original
        self.norm_originals.clear()
