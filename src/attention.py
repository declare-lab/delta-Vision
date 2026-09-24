"""Qwen3-VL adapter FA2 attention, input preparation and attention metadata."""
from __future__ import annotations


# FA2 for adapter text queries with keys in their original sequence order.
import torch


def prefix_plan(text_positions, image_positions, text_mask, image_mask):
    if text_positions.shape[0] != 1:
        raise ValueError("Adapter FA2 benchmark currently requires batch size 1")
    if not bool(text_mask.all()) or not bool(image_mask.all()):
        raise ValueError("Adapter FA2 benchmark requires unpadded inputs")
    text = text_positions[0].tolist()
    images = image_positions[0].tolist()
    return prefix_plan_from_positions(text, images, text_positions.device)


def prefix_plan_from_positions(text, images, device, *, shared_prefix=False, async_copy=False):
    """Build the same plan from already-read CPU sequence positions."""
    positions = images + text
    order = sorted(range(len(positions)), key=positions.__getitem__)
    if len(set(positions)) != len(positions) or text != sorted(text):
        raise ValueError("Adapter FA2 requires unique, increasing sequence positions")
    rank = {positions[index]: i for i, index in enumerate(order)}
    runs = []
    start = 0
    for i in range(1, len(text)):
        if rank[text[i]] != rank[text[i - 1]] + 1:
            runs.append((start, i))
            start = i
    runs.append((start, len(text)))
    cu_q, cu_k, indices = [0], [0], []
    for start, end in runs:
        length = rank[text[end - 1]] + 1
        if not shared_prefix:
            indices.extend(order[:length])
        cu_q.append(cu_q[-1] + end - start)
        cu_k.append(cu_k[-1] + length)
    plan = _plan(order if shared_prefix else indices, cu_q, cu_k, device, causal=True, async_copy=async_copy)
    if shared_prefix:
        # Each text run reads the same sorted KV buffer, truncated at its own
        # causal endpoint. seqused_k retains the ordinary unpaged FA2 kernel.
        plan['shared_prefix'] = True
        plan['cu_k_shared'] = device_tensor([0] * len(runs) + [len(order)], torch.int32, device, async_copy)
        plan['used_k'] = device_tensor([b-a for a,b in zip(cu_k,cu_k[1:])], torch.int32, device, async_copy)
    # These masks were checked above. Certify once that subsequent generated
    # queries see the entire prefix, without a device-to-host mask read per step.
    plan["dense_decode_ready"] = bool(text) and (not images or max(images) < text[-1])
    return plan


def decode_plan(attention_mask):
    # A single cached query can attend every allowed key, including masks for
    # inactive samples. Keep this preparation outside the 36-layer loop.
    batch, _, queries, keys = attention_mask.shape
    if queries != 1:
        raise ValueError("Cached adapter FA2 decode requires one query")
    masks = attention_mask[:, 0, 0].tolist()
    indices, cu_k = [], [0]
    for b, mask in enumerate(masks):
        indices.extend(b * keys + i for i, allowed in enumerate(mask) if allowed)
        cu_k.append(len(indices))
    plan = _plan(indices, list(range(batch + 1)), cu_k, attention_mask.device, causal=False)
    plan["dense_decode"] = all(all(mask) for mask in masks)
    return plan


def device_tensor(data, dtype, device, async_copy=False):
    if async_copy:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(data, dtype=dtype, device=device)


def _plan(indices, cu_q, cu_k, device, *, causal, async_copy=False):
    return dict(
        key_indices=device_tensor(indices, torch.long, device, async_copy),
        cu_q=device_tensor(cu_q, torch.int32, device, async_copy),
        cu_k=device_tensor(cu_k, torch.int32, device, async_copy),
        max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])),
        max_k=max(b - a for a, b in zip(cu_k, cu_k[1:])),
        causal=causal,
    )


def attention_heads(query, key, value, *, scaling, plan):
    from flash_attn.flash_attn_interface import flash_attn_func

    if plan.get("dense_decode", False):
        # With one query and every key valid, no gather or varlen packing is
        # needed. This is the native FA2 decode kernel used by the base model.
        return flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=0., softmax_scale=scaling, causal=False).contiguous()

    batch, heads, length, dim = query.shape
    q = query.transpose(1, 2).reshape(-1, heads, dim)
    k = key.transpose(1, 2).reshape(-1, key.shape[1], dim).index_select(0, plan["key_indices"])
    v = value.transpose(1, 2).reshape(-1, value.shape[1], dim).index_select(0, plan["key_indices"])
    output = varlen_attention(q, k, v, scaling=scaling, plan=plan)
    return output.reshape(batch, length, heads, dim).contiguous()


def varlen_attention(query, key, value, *, scaling, plan):
    """Run the ordinary FA2 kernel for either duplicated or shared prefixes."""
    if plan.get('shared_prefix', False):
        # The public autograd wrapper does not expose seqused_k. Use its own
        # inference operator with the same non-paged kernel and causal mask.
        if torch.is_grad_enabled():
            raise ValueError('Shared-prefix FA2 is an inference-only path')
        from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward
        return _wrapped_flash_attn_varlen_forward(query, key, value,
            plan['cu_q'], plan['cu_k_shared'], plan['max_q'], plan['max_k'], 0., scaling,
            causal=plan['causal'], seqused_k=plan['used_k'])[0]
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    return flash_attn_varlen_func(query, key, value, plan['cu_q'], plan['cu_k'],
        plan['max_q'], plan['max_k'], dropout_p=0., softmax_scale=scaling, causal=plan['causal'])


# Prepare batch-one FA2 adapter inputs with one CPU mask read.
def prepare_fa2_inputs(model, adapter, input_ids, attention_mask, mm_token_type_ids,
                       initial_hidden, position_ids, *, reuse_position_embeddings=True,
                       defer_tensor_ops=False, shared_prefix=False, topology=None, async_copy=False,
                       prepare_text_only_plan=False):
    from src.attention import prefix_plan_from_positions, device_tensor
    assert input_ids.shape[0] == 1
    device = input_ids.device
    valid, types = topology if topology is not None else torch.stack((attention_mask[0], mm_token_type_ids[0])).tolist()
    text = [i for i,(active,kind) in enumerate(zip(valid,types)) if active and kind == 0]
    images = [i for i,(active,kind) in enumerate(zip(valid,types)) if active and kind != 0]
    if not text or not images:
        raise ValueError("Qwen sample must contain both text and image positions")
    text_pos = device_tensor([text], torch.long, device, async_copy)
    image_pos = device_tensor([images], torch.long, device, async_copy)
    metadata = dict(initial_hidden=initial_hidden, position_ids=position_ids,
        text_positions=text_pos, image_positions=image_pos,
        attention_plan=prefix_plan_from_positions(text, images, device, shared_prefix=shared_prefix, async_copy=async_copy),
        reuse_position_embeddings=reuse_position_embeddings)
    if prepare_text_only_plan:
        metadata['text_only_attention_plan']=prefix_plan_from_positions(text,[],device,shared_prefix=shared_prefix,async_copy=async_copy)
    return metadata if defer_tensor_ops else materialize_fa2_inputs(model, adapter, metadata)


def materialize_fa2_inputs(model, adapter, metadata):
    """Gather current request features and compute RoPE inside the prefill graph."""
    from src.model import gather_batched_positions, qwen_visual_position_ids, qwen_prefix_causal_attention_mask
    initial_hidden, position_ids = metadata['initial_hidden'], metadata['position_ids']
    text_pos, image_pos = metadata['text_positions'], metadata['image_positions']
    text_mask = torch.ones_like(text_pos, dtype=torch.bool)
    image_mask = torch.ones_like(image_pos, dtype=torch.bool)
    text_position_ids = position_ids[:, :1, text_pos[0]]
    visual_position_ids = qwen_visual_position_ids(position_ids, image_pos, image_mask)
    dtype = next(adapter.parameters()).dtype
    h = gather_batched_positions(initial_hidden, text_pos, text_mask).to(dtype=dtype)
    visual_memory = gather_batched_positions(initial_hidden, image_pos, image_mask).to(dtype=dtype)
    mask = qwen_prefix_causal_attention_mask(text_mask, image_mask, h.device,
        text_positions=text_pos, image_positions=image_pos)
    text_embeddings = visual_embeddings = None
    if metadata['reuse_position_embeddings']:
        rotary = model.model.language_model.rotary_emb
        text_embeddings = rotary(h, text_position_ids)
        visual_embeddings = rotary(visual_memory, visual_position_ids)
    return dict(h=h, visual_memory=visual_memory, text_mask=text_mask, image_mask=image_mask,
        text_positions=text_pos, image_positions=image_pos, text_position_ids=text_position_ids,
        visual_position_ids=visual_position_ids, prefix_attention_mask=mask,
        text_position_embeddings=text_embeddings, visual_position_embeddings=visual_embeddings,
        attention_plan=metadata['attention_plan'],
        **({'text_only_attention_plan':metadata['text_only_attention_plan']} if 'text_only_attention_plan' in metadata else {}))


# Keep single-request FlashAttention metadata separate from sparse RoPE positions.
from dataclasses import dataclass, field



@dataclass
class AttentionMetadataOptimization:
    handles: list = field(default_factory=list)
    enabled: bool = True
    unpacked: bool = False
    metadata: dict = field(default_factory=dict)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.metadata.clear()


def optimize_qwen_attention_metadata(model) -> AttentionMetadataOptimization:
    """Cache packed-position inference for verified increasing text positions.

Preserve genuine position resets, supplied cu_seqlens, padding masks, all cache
tensors and position_embeddings. Evaluate the sequence condition once before any
layer prunes tokens. This also avoids redundant packed checks in one-token decode.
"""
    language = model.model.language_model
    state = AttentionMetadataOptimization()
    if language.config._attn_implementation != "flash_attention_2":
        state.enabled = False
        return state

    def before_language(module, args, kwargs):
        state.unpacked = False
        state.metadata.clear()
        if not state.enabled:
            return
        positions = kwargs.get("position_ids")
        if positions is None:
            return
        if positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[0]
        elif positions.ndim != 2:
            return
        # A strictly increasing sequence may have pruned gaps, but has no resets.
        state.unpacked = positions.shape[-1] <= 1 or bool(torch.all(positions.diff(dim=-1) > 0))

    def before_attention(module, args, kwargs):
        if state.enabled and state.unpacked and "position_ids" in kwargs:
            positions = kwargs["position_ids"]
            if positions is None:
                return
            metadata_keys = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            supplied = [kwargs.get(key) is not None for key in metadata_keys]
            if any(supplied) and not all(supplied):
                return
            kwargs = dict(kwargs)
            kwargs.pop("position_ids")
            if not all(supplied):
                key = (positions.data_ptr(), tuple(positions.shape), tuple(positions.stride()))
                if key not in state.metadata:
                    from transformers.modeling_flash_attention_utils import _is_packed_sequence
                    metadata = {}
                    # A single position is always equal to its own minimum, so
                    # the FA heuristic is false without any device scalar read.
                    if positions.shape[-1] > 1 and bool(_is_packed_sequence(positions, batch_size=positions.shape[0])):
                        length = positions.shape[-1]
                        cumulative = torch.tensor([0, length], device=positions.device, dtype=torch.int32)
                        metadata = dict(cu_seq_lens_q=cumulative, cu_seq_lens_k=cumulative,
                                        max_length_q=length, max_length_k=length)
                    # Retain the source tensor so a later pruning allocation cannot reuse its pointer.
                    state.metadata[key] = (positions, metadata)
                kwargs.update(state.metadata[key][1])
            return args, kwargs

    state.handles.append(language.register_forward_pre_hook(before_language, with_kwargs=True))
    for layer in language.layers:
        state.handles.append(layer.self_attn.register_forward_pre_hook(before_attention, with_kwargs=True))
    return state
