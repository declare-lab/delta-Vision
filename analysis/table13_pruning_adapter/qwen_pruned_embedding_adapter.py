"""Compose existing Qwen DART/DivPrune selection with a static embedding adapter.

DART runs native layers 0/1; only layers 2..35 use M_l(E_selected).
DivPrune selects E before the decoder; all layers use M_l(E_selected).
Original text tokens and M-RoPE coordinates are retained. No retraining.
"""
from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache
from baselines.multimodal_pruning_utils import visual_budget, keep_visual_subset
from baselines.dart.qwen3_vl.modeling_qwen3_vl_dart import dart_get_retained_image_token
from baselines.divprune.qwen3_vl.modeling_qwen3_vl_divprune import _divprune_select_tokens
from src.model import (
    prepare_qwen_embedding_adapter_inputs, qwen_embedding_adapter_prefill_cache_prepared,
)


def native_prefix(model, initial, positions):
    """Execute the same two native FA2 layers once for all DART retentions."""
    lm = model.model.language_model
    cache = DynamicCache(config=lm.config)
    hidden = initial
    rotary = lm.rotary_emb(hidden, positions)
    for layer in lm.layers[:2]:
        hidden = layer(hidden, attention_mask=None, position_ids=positions[0],
                       position_embeddings=rotary, past_key_values=cache, use_cache=True)
    return hidden, cache


def select_visual(model, inputs, initial, method, retention, prefix=None):
    if method not in ('dart', 'divprune') or not 0 < retention <= 1:
        raise ValueError((method, retention))
    assert initial.shape[0] == 1 and bool(inputs['attention_mask'].all())
    visual = inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
    n = len(visual)
    assert n > 0
    count = visual_budget(n, retention)
    if count == n:
        selected = visual
    elif method == 'divprune':
        selected = visual[_divprune_select_tokens(initial[0, visual], count)].sort().values
    else:
        assert prefix is not None
        hidden, cache = prefix
        start = int(visual[0])
        before = torch.arange(start, device=visual.device)
        after = torch.arange(start, initial.shape[1], device=visual.device)
        after = after[~torch.isin(after, visual)]
        order = torch.cat((before, visual, after))
        config = dict(pivot_image_token=5, pivot_text_token=3,
                      reduction_ratio=1-retention, target_retention=retention)
        local = dart_get_retained_image_token(config,
            model.model.language_model.norm(hidden)[:, order],
            cache.layers[1].keys[:, :, order], start, n)
        selected = order[local].sort().values
    assert len(selected) == count and selected.unique().numel() == count
    assert bool(torch.isin(selected, visual).all())
    keep = keep_visual_subset(initial.shape[1], visual, selected)
    text = inputs['mm_token_type_ids'][0].eq(0).nonzero().flatten()
    assert bool(torch.isin(text, keep).all()) and len(keep) == len(text)+count
    return selected, keep, visual, text


def adapter_prefill(model, adapter, inputs, initial, positions, method, selection, prefix=None):
    if adapter.mode != 'embedding_adapter':
        raise ValueError('This experiment uses the existing static adapter, not recurrent mode')
    selected, keep, visual, text = selection
    prepared = prepare_qwen_embedding_adapter_inputs(model, adapter,
        inputs['input_ids'][:, keep], inputs['attention_mask'][:, keep],
        inputs['mm_token_type_ids'][:, keep], initial[:, keep], positions[:, :, keep])
    assert torch.equal(prepared['visual_memory'], initial[:, selected])
    prefix_caches = []
    start = 2 if method == 'dart' else 0
    if start:
        assert prefix is not None
        hidden, native_cache = prefix
        # Text carries the actual native prefix evolution. Adapter input stays E.
        prepared['h'] = hidden[:, text]
        for cached in native_cache.layers[:start]:
            prefix_caches.append(dict(
                visual_key=cached.keys[:, :, visual].contiguous(),
                visual_value=cached.values[:, :, visual].contiguous(),
                text_key=cached.keys[:, :, text].contiguous(),
                text_value=cached.values[:, :, text].contiguous()))
    logits, mask, cache = qwen_embedding_adapter_prefill_cache_prepared(
        model, adapter, **prepared, retain_prefix_states=False,
        start_layer=start, prefix_layer_caches=prefix_caches)
    assert cache['dense_decode_ready'] and cache['attention_implementation'] == 'flash_attention_2'
    counts = [int(c['visual_key'].shape[2]) for c in cache['layers']]
    expected = [len(visual)]*start + [len(selected)]*(len(cache['layers'])-start)
    assert counts == expected, (counts, expected)
    # Decoder positions must remain those of the original unpruned request.
    assert torch.equal(cache['next_position_ids'], positions[:, :, -1:]+1)
    audit = dict(original_visual=len(visual), retained_visual=len(selected),
        selected_original_positions=selected.tolist(), layer_visual=counts,
        excluded_full_layers=list(range(start)), adapter_layers=list(range(start, len(counts))),
        prunable_visual_ratio=sum(counts[start:])/(len(visual)*(len(counts)-start)),
        all_layer_visual_ratio=sum(counts)/(len(visual)*len(counts)))
    return logits, mask, cache, audit


@contextmanager
def native_reference_hooks(model, adapter, initial, positions, method, selection):
    """Independent native HF reference: slice, then replace visual layer inputs.

Reference executes native visual Q/FFN too, but discards their result at the
next adapter layer. Text and native DynamicCache remain HF-managed.
With adapter=None, this is a native baseline with exactly the same selected IDs.
"""
    selected, keep, _, _ = selection
    start = 2 if method == 'dart' else 0
    slots = torch.searchsorted(keep, selected)
    lm = model.model.language_model
    handles = []
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if hidden.shape[1] == 1 or index < start:
                return
            kwargs = dict(kwargs)
            assert kwargs.get('attention_mask') is None
            if index == start:
                hidden = hidden[:, keep]
            if adapter is not None:
                hidden = hidden.clone()
                hidden[:, slots] = adapter.visual_memory_for_layer(initial[:, selected], index)
            kwargs['position_embeddings'] = tuple(p[:, keep] for p in kwargs['position_embeddings'])
            if torch.is_tensor(kwargs.get('position_ids')):
                kwargs['position_ids'] = kwargs['position_ids'][:, keep]
            if args:
                args = (hidden, *args[1:])
            else:
                kwargs['hidden_states'] = hidden
            return args, kwargs
        return apply
    try:
        for index, layer in enumerate(lm.layers):
            handles.append(layer.register_forward_pre_hook(hook(index), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def native_reference_generate(model, adapter, initial, positions, method, selection, max_tokens, eos):
    lm = model.model.language_model
    tokens = []
    with native_reference_hooks(model, adapter, initial, positions, method, selection):
        result = lm(inputs_embeds=initial, attention_mask=None, position_ids=positions, use_cache=True)
        logits = model.lm_head(result.last_hidden_state[:, -1:])
        first_logits = logits.clone()
        cache = result.past_key_values
        native_lengths = [c.keys.shape[2] for c in cache.layers]
        for step in range(max_tokens):
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            if token in eos or step+1 == max_tokens:
                break
            result = lm(input_ids=torch.tensor([[token]], device=initial.device),
                attention_mask=None, position_ids=positions[:, :, -1:]+step+1,
                past_key_values=cache, use_cache=True)
            logits = model.lm_head(result.last_hidden_state[:, -1:])
    return tokens, first_logits, native_lengths


def compare_logits(actual, reference, *, relative_limit=.01, absolute_limit=.5, enforce=True):
    a, b = actual.float(), reference.float()
    relative = float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8))
    maximum = float((a-b).abs().max())
    same = int(a[0, -1].argmax()) == int(b[0, -1].argmax())
    if enforce:
        assert relative <= relative_limit and maximum <= absolute_limit and same, (relative, maximum, same)
    return dict(relative_rms_error=relative, max_abs_logit_error=maximum, same_first_token=same)
