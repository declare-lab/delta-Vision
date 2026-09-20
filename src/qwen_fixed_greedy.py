"""Minimal fixed-length greedy loop around native Qwen forwards.

Used only for equal-work speed tests. Native M-RoPE preparation, FA2 attention,
dynamic KV growth and EOS suppression match generate(min_new_tokens=N).
"""
from types import SimpleNamespace
import torch


def fixed_greedy(model, inputs, tokens, *, output_logits=False):
    if inputs['input_ids'].shape[0] != 1 or tokens < 2:
        raise ValueError('Fixed greedy speed test requires one request and at least two tokens')
    model.model.rope_deltas = None
    positions = model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
    output = model(**inputs, position_ids=positions, use_cache=True, logits_to_keep=1, return_dict=True)
    next_positions = positions[:, :, -1:] + torch.arange(1, tokens, device=positions.device)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    generated, logits = [], []
    for step in range(tokens):
        scores = output.logits[:, -1].to(torch.float32, copy=True)
        if output_logits:
            logits.append(scores.clone())
        scores[:, eos] = -float('inf')
        token = scores.argmax(-1).view(1, 1)
        generated.append(token)
        if step + 1 < tokens:
            output = model(input_ids=token, position_ids=next_positions[:, :, step:step+1],
                past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
    return SimpleNamespace(sequences=torch.cat([inputs['input_ids'], *generated], dim=1),
        logits=tuple(logits), past_key_values=output.past_key_values)
