"""Prepare batch-one FA2 adapter inputs with one CPU mask read."""
import torch


def prepare_fa2_inputs(model, adapter, input_ids, attention_mask, mm_token_type_ids,
                       initial_hidden, position_ids, *, reuse_position_embeddings=True,
                       defer_tensor_ops=False, shared_prefix=False, topology=None, async_copy=False):
    from src.qwen_adapter_fa2 import prefix_plan_from_positions, device_tensor
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
        attention_plan=metadata['attention_plan'])
