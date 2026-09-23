"""Cached LLaVA DART: previous-layer RoPE keys, layer-2 pruning, exact budget.

The DART pivot/diversity rule follows the vendored upstream implementation.
Its approximate per-pivot quota is replaced by an exact quota including image
pivots. Retention is the sum of visual tokens over prunable layers 2..L-1
divided by (L-2)*original_visual_count. BOTH compulsory full-retention layers
0 and 1 are excluded from the budget.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


def retained_budget(image_len, layers, retention):
    if not 0 < retention <= 1 or layers <= 2:
        raise ValueError('Invalid retention or layer count')
    return max(1, min(image_len, round(image_len * retention)))


@contextmanager
def native_pruning_reference(model, selected, image_start, image_len, sequence_len):
    """Independent validation path: native HF forward/generate plus slice hooks.

    No custom decoder loop. Native HF maintains all cache/position bookkeeping.
    Fix the selected image IDs to separate cache validation from selection.
    """
    device=selected.device
    indices=torch.cat((torch.arange(image_start,device=device),selected,
        torch.arange(image_start+image_len,sequence_len,device=device)))
    handles=[]
    def hook(index):
        def apply(module,args,kwargs):
            h=kwargs.get('hidden_states',args[0] if args else None)
            if h.shape[1]==1:return args,kwargs
            kwargs=dict(kwargs)
            assert kwargs.get('attention_mask') is None
            if index==2:
                h=h[:,indices]
                if args:args=(h,*args[1:])
                else:kwargs['hidden_states']=h
            kwargs['position_ids']=kwargs['position_ids'][:,indices]
            kwargs['position_embeddings']=tuple(x[:,indices] for x in kwargs['position_embeddings'])
            return args,kwargs
        return apply
    try:
        for index,layer in enumerate(model.model.language_model.layers):
            if index>=2:handles.append(layer.register_forward_pre_hook(hook(index),with_kwargs=True))
        yield
    finally:
        for handle in handles:handle.remove()


def select_dart(hidden, keys, image_start, image_len, keep, norm,
                image_pivots=4, text_pivots=4):
    assert hidden.shape[0] == keys.shape[0] == 1
    assert keys.shape[2] == hidden.shape[1]
    keep = max(1, min(image_len, int(keep)))
    device = hidden.device
    if keep == image_len:
        return torch.arange(image_start, image_start + image_len, device=device), {}
    # Preserve token rows: [B,H,S,D] -> [B,S,H*D].
    flat = keys.transpose(1, 2).reshape(1, keys.shape[2], -1)
    end = image_start + image_len
    ni = min(image_pivots, keep, image_len)
    nt = min(text_pivots, hidden.shape[1] - end)
    ip = (flat[0, image_start:end].norm(p=1, dim=-1).topk(ni).indices + image_start).tolist()
    tp = (flat[0, end:].norm(p=1, dim=-1).topk(nt).indices + end).tolist() if nt else []
    pivots = sorted(ip + tp)
    features = F.normalize(norm(hidden)[0].float(), dim=-1)
    selected = set(ip)
    available = set(range(image_start, end)) - selected
    quotas = []
    for j, pivot in enumerate(pivots):
        remaining = keep - len(selected)
        quota = (remaining + len(pivots) - j - 1) // (len(pivots) - j)
        quotas.append(quota)
        if not quota:
            continue
        candidates = torch.tensor(sorted(available), device=device)
        # Same per-pivot negative cosine diversity objective as upstream.
        scores = -(features[candidates] * features[pivot]).sum(-1)
        chosen = candidates[scores.topk(quota).indices].tolist()
        selected.update(chosen)
        available.difference_update(chosen)
    result = torch.tensor(sorted(selected), device=device)
    assert result.numel() == keep and result.unique().numel() == keep
    return result, dict(image_pivots=ip, text_pivots=tp, pivot_order=pivots, quotas=quotas)


class DartDecoder:
    def __init__(self, model):
        self.model = model
        self.lm = model.model.language_model
        assert self.lm.config._attn_implementation == 'flash_attention_2'
        self.cache = None

    def prefill(self, embeds, image_start, image_len, retention, fixed_indices=None):
        lm = self.lm
        hidden = embeds
        positions = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
        self.cache = DynamicCache(config=lm.config)
        counts = []
        selected = None
        meta = {}
        keep = retained_budget(image_len, len(lm.layers), retention)
        for i, layer in enumerate(lm.layers):
            if i == 2:
                keys = self.cache.layers[1].keys
                if fixed_indices is None:
                    selected, meta = select_dart(hidden, keys, image_start, image_len, keep, lm.norm)
                else:
                    selected = fixed_indices
                indices = torch.cat((torch.arange(image_start, device=hidden.device), selected,
                    torch.arange(image_start + image_len, embeds.shape[1], device=hidden.device)))
                hidden = hidden[:, indices]
                positions = positions[:, indices]
            counts.append(hidden.shape[1])
            hidden = layer(hidden, attention_mask=None, position_ids=positions,
                position_embeddings=lm.rotary_emb(hidden, position_ids=positions),
                past_key_values=self.cache, use_cache=True)
        self.next_position = embeds.shape[1]
        text = embeds.shape[1] - image_len
        visual_counts = [n - text for n in counts]
        assert visual_counts == [image_len] * 2 + [keep] * (len(lm.layers) - 2)
        self.audit = dict(original_visual=image_len, retained_visual=keep,
            text=text, visual_tokens_per_layer=visual_counts,
            all_layer_visual_retention=sum(visual_counts)/(image_len*len(lm.layers)),
            excluded_first_layer_retention=sum(visual_counts[1:])/(image_len*(len(lm.layers)-1)),
            prunable_layer_retention=sum(visual_counts[2:])/(image_len*(len(lm.layers)-2)),
            requested_prunable_layer_retention=retention,
            excluded_compulsory_full_layers=[0,1],
            post_pruning_retention=keep/image_len, selected_indices=selected.tolist(),
            key_source_layer=1, pruning_layer=2, **meta)
        return self.model.lm_head(lm.norm(hidden[:, -1:]))[:, -1]

    def decode(self, token):
        hidden = self.lm.embed_tokens(torch.tensor([[token]], device=next(self.model.parameters()).device))
        positions = torch.tensor([[self.next_position]], device=hidden.device)
        for layer in self.lm.layers:
            hidden = layer(hidden, attention_mask=None, position_ids=positions,
                position_embeddings=self.lm.rotary_emb(hidden, position_ids=positions),
                past_key_values=self.cache, use_cache=True)
        self.next_position += 1
        return self.model.lm_head(self.lm.norm(hidden))[:, -1]

    @torch.inference_mode()
    def generate(self, embeds, image_start, image_len, retention, max_new_tokens, eos_ids):
        logits = self.prefill(embeds, image_start, image_len, retention)
        tokens = []
        for j in range(max_new_tokens):
            token = int(logits[0].argmax())
            tokens.append(token)
            if token in eos_ids or j + 1 == max_new_tokens:
                break
            logits = self.decode(token)
        return tokens, self.audit
