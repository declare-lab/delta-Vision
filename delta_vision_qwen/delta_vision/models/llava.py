from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def get_language_model(vlm: torch.nn.Module) -> torch.nn.Module:
    if hasattr(vlm, "language_model"):
        return vlm.language_model
    if hasattr(vlm, "model") and hasattr(vlm.model, "language_model"):
        return vlm.model.language_model
    raise AttributeError("could not locate LLaVA language model")


def get_lm_layers(language_model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        return language_model.model.layers
    if hasattr(language_model, "layers"):
        return language_model.layers
    raise AttributeError("could not locate LLM layers")


def get_lm_norm(language_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(language_model, "model") and hasattr(language_model.model, "norm"):
        return language_model.model.norm
    if hasattr(language_model, "norm"):
        return language_model.norm
    raise AttributeError("could not locate LLM norm")


def get_lm_embed_tokens(language_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(language_model, "model") and hasattr(language_model.model, "embed_tokens"):
        return language_model.model.embed_tokens
    if hasattr(language_model, "embed_tokens"):
        return language_model.embed_tokens
    raise AttributeError("could not locate LLM token embedding")


@torch.no_grad()
def build_llava_initial_hidden(
    vlm: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor | None = None,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
) -> torch.Tensor:
    """Build LLaVA multimodal input embeddings without running the LLM blocks."""
    llava_model = vlm.model if hasattr(vlm, "model") else vlm
    inputs_embeds = llava_model.get_input_embeddings()(input_ids)
    image_features = llava_model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        image_sizes=image_sizes,
        return_dict=True,
    ).pooler_output
    image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    special_image_mask = llava_model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
    )
    return inputs_embeds.masked_scatter(special_image_mask, image_features)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def llava15_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def get_text_and_image_positions(
    input_ids: torch.Tensor,
    merged_len: int,
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return merged-sequence text positions, image positions, and teacher positions.

    Assumes one sample and one image placeholder, matching the LLaVA-1.5 VQA
    prompt used in this project.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("position extraction currently expects batch size 1")
    ids = input_ids[0].tolist()
    image_indices = [i for i, token_id in enumerate(ids) if token_id == image_token_id]
    if len(image_indices) > 1:
        text_positions = [i for i, token_id in enumerate(ids) if token_id != image_token_id]
        return (
            torch.tensor(text_positions, dtype=torch.long),
            torch.tensor(image_indices, dtype=torch.long),
            torch.tensor(text_positions, dtype=torch.long),
        )
    if len(image_indices) != 1:
        raise ValueError(f"Expected at least one <image> token, found {len(image_indices)}")

    image_idx = image_indices[0]
    num_image_tokens = merged_len - (input_ids.shape[1] - 1)
    if num_image_tokens <= 0:
        raise ValueError("Could not infer a positive number of merged image tokens")

    text_positions: list[int] = []
    teacher_positions: list[int] = []
    for src_idx in range(input_ids.shape[1]):
        if src_idx == image_idx:
            continue
        merged_idx = src_idx if src_idx < image_idx else src_idx + num_image_tokens - 1
        text_positions.append(merged_idx)
        teacher_positions.append(merged_idx)

    image_positions = list(range(image_idx, image_idx + num_image_tokens))
    return (
        torch.tensor(text_positions, dtype=torch.long),
        torch.tensor(image_positions, dtype=torch.long),
        torch.tensor(teacher_positions, dtype=torch.long),
    )


@dataclass
class BatchedModalPositions:
    text_positions: torch.Tensor
    image_positions: torch.Tensor
    text_position_ids: torch.Tensor
    text_mask: torch.Tensor
    image_mask: torch.Tensor
    full_mask: torch.Tensor


def get_batched_text_and_image_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    merged_len: int,
    image_token_id: int,
    image_seq_length: int,
) -> BatchedModalPositions:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, source_len]")
    if image_seq_length <= 0:
        raise ValueError("image_seq_length must be positive")

    batch, source_len = input_ids.shape
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    attention_mask = attention_mask.to(device=device)

    rows_text_positions: list[list[int]] = []
    rows_text_position_ids: list[list[int]] = []
    rows_image_positions: list[list[int]] = []
    rows_full_valid: list[list[int]] = []
    max_text = 0
    max_image = image_seq_length

    for batch_idx in range(batch):
        valid_src = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        if not valid_src:
            raise ValueError("empty input row in batch")
        image_indices = [
            int(src_idx)
            for src_idx in valid_src
            if int(input_ids[batch_idx, src_idx].item()) == int(image_token_id)
        ]
        if len(image_indices) == 1:
            image_idx = image_indices[0]
            text_positions = []
            text_position_ids = []
            for src_idx in valid_src:
                src_idx = int(src_idx)
                if src_idx == image_idx:
                    continue
                merged_idx = src_idx if src_idx < image_idx else src_idx + image_seq_length - 1
                if merged_idx >= merged_len:
                    raise ValueError(f"merged text position {merged_idx} exceeds merged_len={merged_len}")
                text_positions.append(merged_idx)
                text_position_ids.append(merged_idx)
            image_positions = list(range(image_idx, image_idx + image_seq_length))
            if image_positions[-1] >= merged_len:
                raise ValueError(f"merged image position {image_positions[-1]} exceeds merged_len={merged_len}")
            expanded_len = max(text_positions + image_positions) + 1
            full_valid = list(range(expanded_len))
        elif len(image_indices) > 1:
            image_set = set(image_indices)
            text_positions = [int(src_idx) for src_idx in valid_src if int(src_idx) not in image_set]
            text_position_ids = text_positions.copy()
            image_positions = image_indices
            if max(valid_src) >= merged_len:
                raise ValueError(f"source position {max(valid_src)} exceeds merged_len={merged_len}")
            full_valid = [int(src_idx) for src_idx in valid_src]
        else:
            raise ValueError("expected at least one image token per row")
        rows_text_positions.append(text_positions)
        rows_text_position_ids.append(text_position_ids)
        rows_image_positions.append(image_positions)
        rows_full_valid.append(full_valid)
        max_text = max(max_text, len(text_positions))
        max_image = max(max_image, len(image_positions))

    text_positions_tensor = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    text_position_ids_tensor = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    text_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    image_positions_tensor = torch.zeros((batch, max_image), device=device, dtype=torch.long)
    image_mask = torch.zeros((batch, max_image), device=device, dtype=torch.bool)
    full_mask = torch.zeros((batch, merged_len), device=device, dtype=torch.bool)

    for batch_idx in range(batch):
        text_len = len(rows_text_positions[batch_idx])
        image_len = len(rows_image_positions[batch_idx])
        full_len = len(rows_full_valid[batch_idx])
        text_positions_tensor[batch_idx, :text_len] = torch.tensor(
            rows_text_positions[batch_idx],
            device=device,
            dtype=torch.long,
        )
        text_position_ids_tensor[batch_idx, :text_len] = torch.tensor(
            rows_text_position_ids[batch_idx],
            device=device,
            dtype=torch.long,
        )
        text_mask[batch_idx, :text_len] = True
        image_positions_tensor[batch_idx, :image_len] = torch.tensor(
            rows_image_positions[batch_idx],
            device=device,
            dtype=torch.long,
        )
        image_mask[batch_idx, :image_len] = True
        full_mask[batch_idx, :full_len] = True

    return BatchedModalPositions(
        text_positions=text_positions_tensor,
        image_positions=image_positions_tensor,
        text_position_ids=text_position_ids_tensor,
        text_mask=text_mask,
        image_mask=image_mask,
        full_mask=full_mask,
    )


def gather_batched_positions(hidden_states: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3 or positions.ndim != 2 or mask.shape != positions.shape:
        raise ValueError("expected hidden_states [B,S,H], positions/mask [B,T]")
    gather_idx = positions.to(device=hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    gathered = torch.gather(hidden_states, dim=1, index=gather_idx)
    return gathered * mask.to(device=hidden_states.device, dtype=gathered.dtype).unsqueeze(-1)


def scatter_batched_positions(
    base_hidden_states: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if base_hidden_states.ndim != 3 or values.ndim != 3:
        raise ValueError("expected base_hidden_states and values to be rank-3")
    if positions.shape != values.shape[:2] or mask.shape != positions.shape:
        raise ValueError("positions/mask must match values batch and sequence shape")
    out = base_hidden_states.clone()
    for batch_idx in range(out.shape[0]):
        valid = mask[batch_idx].to(device=out.device).bool()
        if bool(valid.any()):
            idx = positions[batch_idx, valid].to(device=out.device)
            out[batch_idx, idx] = values[batch_idx, valid].to(device=out.device, dtype=out.dtype)
    return out


def make_causal_mask(batch: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)


def make_causal_padding_mask(
    padding_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if padding_mask.ndim != 2:
        raise ValueError("padding_mask must have shape [batch, seq_len]")
    batch, seq_len = padding_mask.shape
    device = padding_mask.device
    min_value = torch.finfo(dtype).min
    causal = torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).view(1, 1, seq_len, seq_len)
    key_pad = torch.zeros((batch, 1, 1, seq_len), device=device, dtype=dtype)
    key_pad = key_pad.masked_fill(padding_mask[:, None, None, :], min_value)
    return causal + key_pad


def llama_attention_output(
    self_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run a LLaMA attention module through o_proj without cache updates.

    This mirrors the local transformers LlamaAttention implementation while
    keeping the query/key/value set explicit for attention-effect distillation.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    num_key_value_groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_mask = None
    if attention_mask is not None and not is_causal:
        attn_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=self_attn.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self_attn.o_proj(attn_output)


def _llama_attention_qkv(
    self_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    num_key_value_groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    return query_states, key_states, value_states


def _explicit_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    logits = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) * float(scaling)
    logits = logits + attention_mask.to(device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1).to(dtype=value_states.dtype)
    return torch.matmul(probs, value_states)


def compute_llama_factorized_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: torch.Tensor,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decompose and reconstruct the attention effect for one LLaMA layer.

    Returns tensors after the original attention output projection. The
    factorized reconstruction is:

        DeltaA = m_V * (A_vis - A_text)

    computed per attention head before concatenation and o_proj.
    """
    layer = get_lm_layers(language_model)[layer_idx]
    batch, full_len, _ = full_hidden_states.shape
    if batch != 1:
        raise ValueError("factorized attention-effect extraction expects batch size 1")

    device = full_hidden_states.device
    text_positions = text_positions.to(device=device)
    image_positions = image_positions.to(device=device)
    text_len = int(text_positions.numel())

    full_position_ids = torch.arange(full_len, device=device).unsqueeze(0)
    text_position_ids = text_positions.unsqueeze(0)
    text_hidden_states = full_hidden_states.index_select(1, text_positions)

    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    full_normed = layer.input_layernorm(full_hidden_states)
    text_normed = layer.input_layernorm(text_hidden_states)

    full_position_embeddings = rotary_owner.rotary_emb(full_normed, full_position_ids)
    q_full, k_full, v_full = _llama_attention_qkv(layer.self_attn, full_normed, full_position_embeddings)
    q_text_from_full = q_full.index_select(2, text_positions)

    full_key_positions = torch.arange(full_len, device=device)
    causal_mask = full_key_positions.view(1, 1, 1, full_len) <= text_positions.view(1, 1, text_len, 1)
    image_key_mask = torch.zeros((full_len,), device=device, dtype=torch.bool)
    image_key_mask[image_positions] = True
    text_key_mask = torch.zeros((full_len,), device=device, dtype=torch.bool)
    text_key_mask[text_positions] = True

    logits = torch.matmul(q_text_from_full.float(), k_full.float().transpose(-2, -1)) * float(layer.self_attn.scaling)
    logits = logits.masked_fill(~causal_mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)

    joint_head = torch.matmul(probs.to(dtype=v_full.dtype), v_full)
    visual_mask = image_key_mask.view(1, 1, 1, full_len) & causal_mask
    text_mask = text_key_mask.view(1, 1, 1, full_len) & causal_mask
    visual_probs = probs * visual_mask.to(dtype=probs.dtype)
    text_probs = probs * text_mask.to(dtype=probs.dtype)
    visual_mass = visual_probs.sum(dim=-1)
    text_mass = text_probs.sum(dim=-1)

    visual_norm_probs = visual_probs / visual_mass.clamp_min(1e-20).unsqueeze(-1)
    text_norm_probs = text_probs / text_mass.clamp_min(1e-20).unsqueeze(-1)
    visual_head = torch.matmul(visual_norm_probs.to(dtype=v_full.dtype), v_full)
    text_head_from_joint = torch.matmul(text_norm_probs.to(dtype=v_full.dtype), v_full)
    factorized_head_delta = visual_mass.to(dtype=v_full.dtype).unsqueeze(-1) * (visual_head - text_head_from_joint)

    text_position_embeddings = rotary_owner.rotary_emb(text_normed, text_position_ids)
    q_text, k_text, v_text = _llama_attention_qkv(layer.self_attn, text_normed, text_position_embeddings)
    text_causal = torch.ones((text_len, text_len), device=device, dtype=torch.bool).tril()
    text_attention_mask = torch.zeros((1, 1, text_len, text_len), device=device, dtype=torch.float32)
    text_attention_mask = text_attention_mask.masked_fill(~text_causal.view(1, 1, text_len, text_len), torch.finfo(torch.float32).min)
    text_head = _explicit_attention(q_text, k_text, v_text, text_attention_mask, layer.self_attn.scaling)
    direct_head_delta = joint_head - text_head
    text_equiv_head = text_head_from_joint - text_head

    def project(head_states: torch.Tensor) -> torch.Tensor:
        projected = head_states.transpose(1, 2).contiguous().reshape(1, text_len, -1)
        return layer.self_attn.o_proj(projected)

    return {
        "factorized_delta": project(factorized_head_delta),
        "direct_delta": project(direct_head_delta),
        "visual_only_delta": project(visual_head),
        "text_equiv_delta": project(text_equiv_head),
        "visual_mass": visual_mass.detach(),
    }


def compute_llama_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: torch.Tensor,
    text_positions: torch.Tensor,
) -> torch.Tensor:
    """Return A_joint(text positions) - A_text for one teacher layer."""
    layer = get_lm_layers(language_model)[layer_idx]
    batch, full_len, _ = full_hidden_states.shape
    if batch != 1:
        raise ValueError("attention-effect extraction currently expects batch size 1")

    full_position_ids = torch.arange(full_len, device=full_hidden_states.device).unsqueeze(0)
    text_position_ids = text_positions.to(device=full_hidden_states.device).unsqueeze(0)
    text_hidden_states = full_hidden_states.index_select(1, text_positions.to(device=full_hidden_states.device))

    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    full_normed = layer.input_layernorm(full_hidden_states)
    text_normed = layer.input_layernorm(text_hidden_states)

    full_position_embeddings = rotary_owner.rotary_emb(full_normed, full_position_ids)
    joint_attention = llama_attention_output(
        layer.self_attn,
        full_normed,
        full_position_embeddings,
        None,
        is_causal=True,
    )

    text_position_embeddings = rotary_owner.rotary_emb(text_normed, text_position_ids)
    text_attention = llama_attention_output(
        layer.self_attn,
        text_normed,
        text_position_embeddings,
        None,
        is_causal=True,
    )
    return joint_attention.index_select(1, text_positions.to(device=full_hidden_states.device)) - text_attention


def compute_llama_attention_effect_batched(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: torch.Tensor,
    text_hidden_states: torch.Tensor,
    text_positions: torch.Tensor,
    text_position_ids: torch.Tensor,
    full_mask: torch.Tensor,
    text_mask: torch.Tensor,
) -> torch.Tensor:
    """Return batched A_joint(text positions) - A_text with padding masks."""
    layer = get_lm_layers(language_model)[layer_idx]
    if full_hidden_states.ndim != 3 or text_hidden_states.ndim != 3:
        raise ValueError("hidden states must be rank-3")
    if text_positions.shape != text_mask.shape or text_position_ids.shape != text_positions.shape:
        raise ValueError("text positions, position ids, and masks must have matching shapes")
    batch, full_len, _ = full_hidden_states.shape
    if full_mask.shape != (batch, full_len):
        raise ValueError("full_mask has incompatible shape")

    device = full_hidden_states.device
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    full_position_ids = torch.arange(full_len, device=device).unsqueeze(0).expand(batch, -1)

    full_normed = layer.input_layernorm(full_hidden_states)
    text_normed = layer.input_layernorm(text_hidden_states)

    full_attention_mask = make_causal_padding_mask(~full_mask.to(device=device), full_hidden_states.dtype)
    full_position_embeddings = rotary_owner.rotary_emb(full_normed, full_position_ids)
    joint_attention = llama_attention_output(
        layer.self_attn,
        full_normed,
        full_position_embeddings,
        full_attention_mask,
        is_causal=False,
    )

    text_attention_mask = make_causal_padding_mask(~text_mask.to(device=device), text_hidden_states.dtype)
    text_position_embeddings = rotary_owner.rotary_emb(text_normed, text_position_ids.to(device=device))
    text_attention = llama_attention_output(
        layer.self_attn,
        text_normed,
        text_position_embeddings,
        text_attention_mask,
        is_causal=False,
    )
    joint_text = gather_batched_positions(joint_attention, text_positions.to(device=device), text_mask.to(device=device))
    return (joint_text - text_attention) * text_mask.to(device=device, dtype=text_attention.dtype).unsqueeze(-1)


def compute_llama_text_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    layer: torch.nn.Module | None = None,
    rotary_owner: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Return text-only attention output after o_proj, before residual add."""
    if layer is None:
        layer = get_lm_layers(language_model)[layer_idx]
    if rotary_owner is None:
        rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    normed = layer.input_layernorm(hidden_states)
    position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
    if attention_mask is None and padding_mask is not None:
        attention_mask = make_causal_padding_mask(padding_mask.to(device=hidden_states.device), hidden_states.dtype)
    return llama_attention_output(
        layer.self_attn,
        normed,
        position_embeddings,
        attention_mask,
        is_causal=attention_mask is None,
    )


def run_llama_layer_text_with_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_delta: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    layer: torch.nn.Module | None = None,
    rotary_owner: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Run a text-only LLaMA layer and inject delta after self-attention."""
    if layer is None:
        layer = get_lm_layers(language_model)[layer_idx]
    if rotary_owner is None:
        rotary_owner = language_model.model if hasattr(language_model, "model") else language_model

    residual = hidden_states
    position_embeddings = rotary_owner.rotary_emb(hidden_states, position_ids)
    normed = layer.input_layernorm(hidden_states)
    if attention_mask is None and padding_mask is not None:
        attention_mask = make_causal_padding_mask(padding_mask.to(device=hidden_states.device), hidden_states.dtype)
    attn_out = llama_attention_output(
        layer.self_attn,
        normed,
        position_embeddings,
        attention_mask,
        is_causal=attention_mask is None,
    )
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    if padding_mask is not None:
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return hidden_states


def run_llama_layer_text_with_attention_delta_cache(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Any,
    attention_delta: torch.Tensor | None = None,
    causal_mask: torch.Tensor | None = None,
    layer: torch.nn.Module | None = None,
    rotary_owner: torch.nn.Module | None = None,
    position_embeddings: Any | None = None,
) -> torch.Tensor:
    """Run one text-only LLaMA layer with KV cache and inject attention delta."""
    if layer is None:
        layer = get_lm_layers(language_model)[layer_idx]
    if rotary_owner is None:
        rotary_owner = language_model.model if hasattr(language_model, "model") else language_model

    residual = hidden_states
    if position_embeddings is None:
        position_embeddings = rotary_owner.rotary_emb(hidden_states, position_ids)
    normed = layer.input_layernorm(hidden_states)
    attention_mask = causal_mask
    attn_kwargs: dict[str, Any] = {}
    attn_kwargs["cache_position"] = cache_position
    attn_out, _ = layer.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        position_embeddings=position_embeddings,
        **attn_kwargs,
    )
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def run_llama_text_attention_residual(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_delta: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the text-only attention sublayer through its residual add."""
    layer = get_lm_layers(language_model)[layer_idx]
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model

    normed = layer.input_layernorm(hidden_states)
    position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
    attention_mask = None
    is_causal = True
    if padding_mask is not None:
        attention_mask = make_causal_padding_mask(padding_mask.to(device=hidden_states.device), hidden_states.dtype)
        is_causal = False
    attn_out = llama_attention_output(layer.self_attn, normed, position_embeddings, attention_mask, is_causal=is_causal)
    hidden_states = hidden_states + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta
    if padding_mask is not None:
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return hidden_states


def run_llama_layer_text_only(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one frozen LLaMA/Vicuna decoder layer on text states only."""
    layer = get_lm_layers(language_model)[layer_idx]
    batch, seq_len, _ = hidden_states.shape
    causal_mask = make_causal_mask(batch, seq_len, hidden_states.device, hidden_states.dtype)
    cache_position = torch.arange(seq_len, device=hidden_states.device)

    kwargs: dict[str, Any] = {}
    sig = inspect.signature(layer.forward)
    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = causal_mask
    if "position_ids" in sig.parameters:
        kwargs["position_ids"] = position_ids
    if "output_attentions" in sig.parameters:
        kwargs["output_attentions"] = False
    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False
    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = cache_position
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    if "position_embeddings" in sig.parameters and hasattr(rotary_owner, "rotary_emb"):
        kwargs["position_embeddings"] = rotary_owner.rotary_emb(hidden_states, position_ids)

    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def run_llama_layer_with_attention_mask(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one frozen LLaMA/Vicuna decoder layer with an explicit 4D mask."""
    layer = get_lm_layers(language_model)[layer_idx]
    seq_len = hidden_states.shape[1]
    cache_position = torch.arange(seq_len, device=hidden_states.device)

    kwargs: dict[str, Any] = {}
    sig = inspect.signature(layer.forward)
    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = attention_mask
    if "position_ids" in sig.parameters:
        kwargs["position_ids"] = position_ids
    if "output_attentions" in sig.parameters:
        kwargs["output_attentions"] = False
    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False
    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = cache_position
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    if "position_embeddings" in sig.parameters and hasattr(rotary_owner, "rotary_emb"):
        kwargs["position_embeddings"] = rotary_owner.rotary_emb(hidden_states, position_ids)

    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def make_text_to_vision_block_mask(
    seq_len: int,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal mask plus text-query to vision-key blocking for method B."""
    mask = make_causal_mask(1, seq_len, device, dtype).clone()
    min_value = torch.finfo(dtype).min
    mask[:, :, text_positions[:, None], image_positions[None, :]] = min_value
    return mask
