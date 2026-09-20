"""FA2 for adapter text queries with keys in their original sequence order.

Visual keys are stored ahead of text keys in the adapter cache. Reorder them
for attention and pack each consecutive text run as a separate causal sequence.
FA2's bottom-right causal mask then matches the original position-based mask,
including text before images. Plans are built outside CUDA graph capture.
"""
from __future__ import annotations

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
