"""Disable delta-rule writes at visual positions without changing decay or conv."""
from contextlib import contextmanager

import torch


def mask_write_logits(logits, visual_mask):
    assert logits.ndim == 3 and logits.shape[0] == 1
    if logits.shape[1] == 1:  # Native cached text decoding.
        return logits
    assert logits.shape[:2] == visual_mask.shape
    return logits.masked_fill(visual_mask[..., None], -torch.inf)


@contextmanager
def no_visual_write(model, visual_mask, *, enabled=True, verify=False):
    """b -> sigmoid(b)=beta; -inf at image positions makes beta exactly zero.

    All native Q/K/V projections, convolution, decay g, readout, residuals, FFNs,
    cache updates and text-position beta remain untouched. Applies to all LA layers.
    """
    handles = []
    audit = {'enabled': enabled, 'layers': [], 'prefill_calls': {}, 'decode_calls': {}}

    def hook_for(index):
        def hook(module, args, output):
            is_decode = output.shape[1] == 1
            bucket = audit['decode_calls' if is_decode else 'prefill_calls']
            bucket[index] = bucket.get(index, 0) + 1
            result = mask_write_logits(output, visual_mask) if enabled else output
            if verify and enabled:
                if is_decode:
                    assert result is output
                else:
                    torch.testing.assert_close(result[~visual_mask], output[~visual_mask], rtol=0, atol=0)
                    assert bool(result[visual_mask].sigmoid().eq(0).all())
            return result
        return hook

    try:
        for i, layer in enumerate(model.model.language_model.layers):
            if layer.block_type == 'linear_attention':
                handles.append(layer.linear_attn.in_proj_b.register_forward_hook(hook_for(i)))
                audit['layers'].append(i)
        assert len(handles) == 24
        yield audit
    finally:
        for handle in handles:
            handle.remove()
