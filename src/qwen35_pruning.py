"""Qwen3.5 visual pruning, keeping native FA2 / GatedDeltaNet and hybrid caches.

Selectors are copied unchanged from this project's Qwen3-VL baselines.
DART keeps the first full-attention block (layer 3), then prunes before layer 4.
Its requested retention applies to layers AFTER pruning, as requested by the user.
DivPrune prunes before layer 0. All original text and M-RoPE positions survive.
"""
from contextlib import contextmanager
from decimal import Decimal
from functools import partial

import torch


def visual_budget(count, retention):
    ratio = Decimal(str(round(float(retention), 12)))
    assert 0 <= ratio <= 1
    return max(1, min(count, round(Decimal(count) * ratio))) if count else 0


def _divprune_select_tokens(visual_feature_vectors: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep_count = min(max(int(keep_count), 1), visual_feature_vectors.shape[0])
    features = torch.nn.functional.normalize(visual_feature_vectors.float(), dim=-1)
    dist_matrix = 1.0 - torch.mm(features, features.t())

    selected = torch.empty(keep_count, dtype=torch.long, device=visual_feature_vectors.device)
    nearest = None
    for i in range(keep_count):
        if i == 0:
            scores = (dist_matrix[0] if dist_matrix.shape[0] == 1 else
                      torch.topk(dist_matrix, 2, dim=0, largest=False).values[1, :])
        else:
            row = dist_matrix.index_select(0, selected[i-1:i]).squeeze(0)
            nearest = row.clone() if nearest is None else torch.minimum(nearest, row)
            scores = nearest.clone()
            scores.index_fill_(0, selected[:i], -float("inf"))
        selected[i] = torch.argmax(scores)
    return selected


def _dart_neighbor_indices(last_layer_state, valid_indices, pivot_index, topk):
    # Preserve the original candidate order and torch.topk tie behavior. The
    # tensor work can be graphed while the original Python set remains intact.
    import torch.nn.functional as F_dart
    valid_vectors = last_layer_state[0].index_select(0, valid_indices)
    pivot = last_layer_state[0].index_select(0, pivot_index).squeeze(0)
    scores = -F_dart.cosine_similarity(pivot, valid_vectors, dim=-1)
    return valid_indices.index_select(0, scores.topk(topk).indices)


def dart_get_retained_image_token(config_dart, last_layer_state, k_states, image_token_start_index, image_token_length):
    """DART token selection: L1 norm pivots + cosine similarity neighbors.

    Directly from baselines/dart/Qwen2_5-VL/Qwen2_5VL_DART/modeling_qwen2_5_vl_self.py
    """
    import torch.nn.functional as F_dart

    pivot_image_token = config_dart['pivot_image_token']
    pivot_text_token = config_dart['pivot_text_token']
    reduction_ratio = config_dart['reduction_ratio']
    target_keep = visual_budget(image_token_length, config_dart.get('target_retention', 1 - reduction_ratio))
    TOKEN_TOPK = max(1, round(target_keep / max(1, pivot_image_token + pivot_text_token)))

    device = last_layer_state.device

    # k_states: (batch, kv_heads, seq, head_dim) -> flatten heads
    k_flat = k_states.permute(0, 2, 1, 3).reshape(k_states.shape[0], k_states.shape[2], -1)

    k_states_image_token = k_flat[0][image_token_start_index:image_token_start_index + image_token_length, :]
    k_states_query_token = k_flat[0][image_token_start_index + image_token_length:, :]

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

    actual_pivot_img = min(pivot_image_token, image_token_length, target_keep)
    actual_pivot_text = min(pivot_text_token, max(k_states_query_token.shape[0], 1))

    image_indices = (k_states_image_token_L1_norm.topk(actual_pivot_img).indices + image_token_start_index).tolist()
    query_indices = []
    if k_states_query_token.shape[0] > 0:
        query_indices = (k_states_query_token_L1_norm.topk(actual_pivot_text).indices + image_token_start_index + image_token_length).tolist()
    indices_set = set(image_indices + query_indices)

    valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)
    valid_indices_list = list(valid_indices)

    for item in list(indices_set):
        selected_image_indices = indices_set - set(query_indices)
        if len(selected_image_indices) >= target_keep or not valid_indices_list:
            break
        actual_topk = min(TOKEN_TOPK, target_keep - len(selected_image_indices), len(valid_indices_list))
        if actual_topk <= 0:
            break
        top_k_real_indices = _dart_neighbor_indices(last_layer_state,
            torch.tensor(valid_indices_list, device=device, dtype=torch.long),
            torch.tensor([item], device=device, dtype=torch.long), actual_topk).tolist()
        indices_set.update(top_k_real_indices)
        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)

    indices_set.difference_update(query_indices)
    if len(indices_set) < target_keep and valid_indices_list:
        remaining = target_keep - len(indices_set)
        valid_tensor = torch.tensor(valid_indices_list, device=device, dtype=torch.long)
        fill_scores = last_layer_state[0][valid_tensor, :].norm(dim=-1)
        fill_idx = valid_tensor[torch.topk(fill_scores, k=min(remaining, valid_tensor.numel()), largest=True).indices]
        indices_set.update(int(idx) for idx in fill_idx.tolist())
    retained_image_tokens_index = torch.tensor(sorted(indices_set), device=device)
    if retained_image_tokens_index.numel() > target_keep:
        scores = last_layer_state[0][retained_image_tokens_index, :].norm(dim=-1)
        retained_image_tokens_index = retained_image_tokens_index[
            torch.topk(scores, k=target_keep, largest=True).indices
        ].sort().values
    return retained_image_tokens_index


class VisualPruningController:
    """Batch-one, unpadded single-image inference with native cached generation.

    Earlier layer caches remain intact. Later layers build their caches from the
    shorter stream; recurrent states are never sliced or reset during decode.
    ``fixed_visual`` is only for full-prefix cache-reference tests: DART selection
    must remain the selection made on the original prompt, excluding new answers.
    """

    def __init__(self, model):
        self.text = model.model.language_model
        self.enabled = False
        self.audit = None
        self.handles = [self.text.register_forward_pre_hook(self._start, with_kwargs=True)]
        for index, layer in enumerate(self.text.layers):
            self.handles.append(layer.register_forward_pre_hook(
                partial(self._layer, index), with_kwargs=True))

    @contextmanager
    def activate(self, method, retention, mask, fixed_visual=None):
        assert not self.enabled and method in ('dart', 'divprune')
        assert mask.ndim == 2 and mask.shape[0] == 1 and mask.any()
        self.enabled = True
        self.method, self.retention, self.mask = method, retention, mask
        self.prune_layer = 4 if method == 'dart' else 0
        self.fixed_visual = fixed_visual
        try:
            yield self
        finally:
            self.enabled = False
            self.source = self.keep = None

    def _start(self, module, args, kwargs):
        if not self.enabled:
            return
        cache = kwargs.get('past_key_values')
        self.prefill = cache is None or cache.get_seq_length() == 0
        if self.prefill:
            self.keep = self.source = None
            self.original_length = kwargs['inputs_embeds'].shape[1]
            assert self.mask.shape[1] == self.original_length
            attention = kwargs.get('attention_mask')
            assert attention is None or (attention.ndim == 2 and bool(attention.all()))
            self.visual = self.mask[0].nonzero().flatten()
            self.text_indices = (~self.mask[0]).nonzero().flatten()
            assert torch.equal(self.visual, torch.arange(self.visual[0], self.visual[-1]+1,
                                                         device=self.visual.device))

    def _select(self, hidden):
        count = self.visual.numel()
        budget = visual_budget(count, self.retention)
        if self.fixed_visual is not None:
            selected = torch.as_tensor(self.fixed_visual, device=hidden.device, dtype=torch.long)
        elif budget == count:
            selected = self.visual
        elif self.method == 'divprune':
            selected = self.visual[_divprune_select_tokens(hidden[0, self.visual], budget)]
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            source, positions = self.source
            layer = self.text.layers[self.prune_layer - 1]
            attn = layer.self_attn
            source = layer.input_layernorm(source)
            key = attn.k_norm(attn.k_proj(source).view(*source.shape[:-1], -1, attn.head_dim)).transpose(1, 2)
            _, key = apply_rotary_pos_emb(key, key, *positions)
            selected = dart_get_retained_image_token(
                dict(pivot_image_token=5, pivot_text_token=3, reduction_ratio=1-self.retention,
                     target_retention=self.retention),
                self.text.norm(hidden), key, int(self.visual[0]), count)
        selected = selected.sort().values
        assert selected.numel() == budget and selected.unique().numel() == budget
        assert bool(self.mask[0, selected].all())
        self.keep = torch.cat((self.text_indices, selected)).sort().values
        layers = [count]*self.prune_layer + [budget]*(len(self.text.layers)-self.prune_layer)
        self.audit = dict(original_visual_tokens=count, retained_visual_tokens=budget,
            visual_tokens_per_layer=layers, post_pruning_retention=budget/count,
            all_layer_visual_retention=sum(layers)/(len(layers)*count),
            prune_before_layer=self.prune_layer, selected_visual_indices=selected.tolist(),
            original_sequence_length=self.original_length, retained_sequence_length=self.keep.numel(),
            text_tokens=self.text_indices.numel())
        self.source = None

    def _layer(self, index, module, args, kwargs):
        if not self.enabled:
            return
        if not self.prefill:
            # Eager/SDPA reference tests need each layer's own cache length.
            # Native unpadded FA2 uses no explicit attention mask here.
            attention = kwargs.get('attention_mask')
            if self.text.config.layer_types[index] == 'full_attention' and attention is not None and attention.ndim == 4:
                length = kwargs['past_key_values'].get_seq_length(index) + args[0].shape[1]
                kwargs['attention_mask'] = attention[..., :length]
                return args, kwargs
            return
        hidden = args[0]
        if self.method == 'dart' and index == self.prune_layer-1:
            self.source = (hidden, kwargs['position_embeddings'])
        if index < self.prune_layer:
            return
        if index == self.prune_layer:
            self._select(hidden)
            hidden = hidden.index_select(1, self.keep)
        assert hidden.shape[1] == self.keep.numel()
        kwargs['position_embeddings'] = tuple(p.index_select(-2, self.keep)
                                              for p in kwargs['position_embeddings'])
        if kwargs.get('position_ids') is not None:
            kwargs['position_ids'] = kwargs['position_ids'].index_select(-1, self.keep)
        attention = kwargs.get('attention_mask')
        if attention is not None:
            attention = attention.index_select(-1, self.keep)
            if attention.ndim == 4:
                attention = attention.index_select(-2, self.keep)
            kwargs['attention_mask'] = attention
        return (hidden, *args[1:]), kwargs

    def close(self):
        for handle in self.handles:
            handle.remove()
