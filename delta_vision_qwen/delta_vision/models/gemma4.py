from __future__ import annotations

from collections import UserDict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import (
    create_masks_for_generate,
    create_masks_for_vision_model,
    get_block_sequence_ids_for_mask,
)


def gemma4_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_frozen_gemma4(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Gemma4ForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(model_path)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


def get_gemma4_language_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("could not locate Gemma4 language model")


def gemma4_image_token_id(model: torch.nn.Module, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is None and hasattr(model, "model"):
        token_id = getattr(model.model.config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(processor, "image_token_id", None)
    if token_id is None:
        token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    return int(token_id)


def prepare_gemma4_batch_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    image_key: str,
    question_key: str,
    answer_key: str,
    image_root: Path | None,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, list[str]]:
    full_texts: list[str] = []
    answer_token_lens: list[int] = []
    images: list[Image.Image] = []
    image_paths: list[str] = []
    eos = processor.tokenizer.eos_token or ""
    for row in rows:
        image_path = Path(row[image_key])
        if image_root is not None and not image_path.is_absolute():
            image_path = image_root / image_path
        prompt = gemma4_prompt(processor, str(row[question_key]))
        answer = str(row[answer_key]).strip()
        answer_suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
        full_texts.append(f"{prompt}{answer_suffix}")
        answer_token_lens.append(len(processor.tokenizer(answer_suffix, add_special_tokens=False).input_ids))
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
        image_paths.append(str(image_path))

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    try:
        inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    return inputs, torch.tensor(answer_token_lens, device=device, dtype=torch.long), image_paths


def prepare_gemma4_eval_inputs(
    processor: Any,
    row: dict[str, Any],
    question: str,
    device: torch.device,
) -> dict[str, Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=gemma4_prompt(processor, question), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


@torch.no_grad()
def build_gemma4_initial_context(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor, dict[str, Tensor]]:
    gemma_model = model.model
    input_ids = inputs["input_ids"]
    image_mask, video_mask, audio_mask = gemma_model.get_placeholder_mask(input_ids=input_ids)
    multimodal_mask = image_mask | video_mask | audio_mask
    llm_input_ids = torch.where(
        multimodal_mask,
        torch.full_like(input_ids, int(gemma_model.config.text_config.pad_token_id)),
        input_ids,
    )
    inputs_embeds = gemma_model.get_input_embeddings()(llm_input_ids)

    language_model = gemma_model.language_model
    per_layer_inputs = None
    if getattr(language_model, "hidden_size_per_layer_input", 0):
        pad_embedding = language_model.embed_tokens.weight[gemma_model.config.text_config.pad_token_id, :]
        multimodal_mask_device = multimodal_mask.to(inputs_embeds.device)
        llm_inputs_embeds = torch.where(multimodal_mask_device[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
        per_layer_inputs = language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)

    if inputs.get("pixel_values") is not None:
        image_features = gemma_model.get_image_features(
            inputs["pixel_values"],
            inputs.get("image_position_ids"),
            return_dict=True,
        ).pooler_output
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        image_scatter_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_scatter_mask, image_features)

    if getattr(language_model, "hidden_size_per_layer_input", 0):
        per_layer_inputs = language_model.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    position_ids = inputs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

    masks = gemma4_attention_masks(
        model,
        inputs_embeds,
        inputs.get("attention_mask"),
        position_ids,
        inputs.get("mm_token_type_ids"),
    )
    return inputs_embeds, position_ids, per_layer_inputs, image_mask, masks


def gemma4_attention_masks(
    model: torch.nn.Module,
    inputs_embeds: Tensor,
    attention_mask: Tensor | None,
    position_ids: Tensor,
    mm_token_type_ids: Tensor | None,
) -> dict[str, Tensor]:
    gemma_model = model.model
    mask_kwargs = {
        "config": gemma_model.config.get_text_config(),
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    text_config = gemma_model.config.get_text_config()
    use_bidir = getattr(text_config, "use_bidirectional_attention", None) == "vision"
    if use_bidir and mm_token_type_ids is not None:
        block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
        return create_masks_for_vision_model(block_sequence_ids=block_sequence_ids, **mask_kwargs)
    return create_masks_for_generate(**mask_kwargs)


def gemma4_text_attention_masks(
    language_model: torch.nn.Module,
    text_hidden: Tensor,
    text_mask: Tensor,
    position_ids: Tensor,
) -> dict[str, Tensor]:
    mask_kwargs = {
        "config": language_model.config,
        "inputs_embeds": text_hidden,
        "attention_mask": text_mask.to(dtype=torch.long),
        "past_key_values": None,
        "position_ids": position_ids,
    }
    return create_masks_for_generate(**mask_kwargs)


def get_gemma4_text_image_positions(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_mask: Tensor,
    position_ids: Tensor,
    answer_token_lens: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    batch, _ = input_ids.shape
    rows_text: list[list[int]] = []
    rows_image: list[list[int]] = []
    max_text = 0
    max_image = 0
    for batch_idx in range(batch):
        valid = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        image_set = {int(i) for i in torch.nonzero(image_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()}
        text = [int(i) for i in valid if int(i) not in image_set]
        image = [int(i) for i in valid if int(i) in image_set]
        if not text or not image:
            raise ValueError("Gemma4 sample must contain both text and image positions")
        rows_text.append(text)
        rows_image.append(image)
        max_text = max(max_text, len(text))
        max_image = max(max_image, len(image))

    device = input_ids.device
    text_positions = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    image_positions = torch.zeros((batch, max_image), device=device, dtype=torch.long)
    text_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    image_pos_mask = torch.zeros((batch, max_image), device=device, dtype=torch.bool)
    text_position_ids = torch.zeros((batch, max_text), device=device, dtype=position_ids.dtype)
    answer_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    for batch_idx in range(batch):
        t = torch.tensor(rows_text[batch_idx], device=device, dtype=torch.long)
        v = torch.tensor(rows_image[batch_idx], device=device, dtype=torch.long)
        text_positions[batch_idx, : t.numel()] = t
        image_positions[batch_idx, : v.numel()] = v
        text_mask[batch_idx, : t.numel()] = True
        image_pos_mask[batch_idx, : v.numel()] = True
        text_position_ids[batch_idx, : t.numel()] = position_ids[batch_idx].index_select(0, t)
        if answer_token_lens is not None:
            answer_len = min(int(answer_token_lens[batch_idx].item()), int(t.numel()))
            answer_mask[batch_idx, int(t.numel()) - answer_len : int(t.numel())] = True
    return text_positions, image_positions, text_position_ids, text_mask, image_pos_mask, answer_mask


def gather_batched_positions(hidden_states: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
    gather_idx = positions.to(device=hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    gathered = torch.gather(hidden_states, dim=1, index=gather_idx)
    return gathered * mask.to(device=hidden_states.device, dtype=gathered.dtype).unsqueeze(-1)


def scatter_batched_positions(base_hidden_states: Tensor, positions: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    out = base_hidden_states.clone()
    valid = mask.to(device=out.device).bool()
    if bool(valid.any()):
        batch_idx = torch.arange(out.shape[0], device=out.device).unsqueeze(1).expand_as(positions)
        out[batch_idx[valid], positions.to(device=out.device)[valid]] = values.to(device=out.device, dtype=out.dtype)[
            valid
        ]
    return out


def gemma4_position_embeddings(language_model: torch.nn.Module, hidden_states: Tensor, position_ids: Tensor) -> dict[str, Any]:
    return {
        layer_type: language_model.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in language_model.unique_layer_types
    }


def gemma4_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_masks: dict[str, Tensor],
    position_ids: Tensor,
    shared_kv_states: UserDict,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    layer_type = language_model.config.layer_types[layer_idx]
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids, layer_type)
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        hidden_states=normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_masks[layer_type],
        shared_kv_states=shared_kv_states,
        position_ids=position_ids,
        past_key_values=None,
    )
    return attn_out


def run_gemma4_layer_text_from_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    text_attention: Tensor,
    attention_delta: Tensor | None,
    per_layer_input: Tensor | None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    residual = hidden_states
    hidden_states = text_attention.to(dtype=hidden_states.dtype)
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if getattr(layer, "enable_moe_block", False):
        hidden_states_1 = layer.post_feedforward_layernorm_1(hidden_states)
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = layer.router(hidden_states_flat)
        hidden_states_2 = layer.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = layer.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = layer.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = hidden_states_1 + hidden_states_2
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if getattr(layer, "hidden_size_per_layer_input", 0):
        if per_layer_input is None:
            raise ValueError("Gemma4 layer requires per_layer_input")
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states * layer.layer_scalar


def run_gemma4_layer_full_from_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_output: Tensor,
    text_positions: Tensor | None,
    text_delta: Tensor | None,
    text_mask: Tensor | None,
    per_layer_input: Tensor | None,
) -> Tensor:
    if text_delta is not None:
        if text_positions is None or text_mask is None:
            attention_output = attention_output + text_delta.to(dtype=attention_output.dtype)
        else:
            current = gather_batched_positions(attention_output, text_positions, text_mask)
            attention_output = scatter_batched_positions(
                attention_output,
                text_positions,
                current + text_delta.to(dtype=current.dtype),
                text_mask,
            )
    return run_gemma4_layer_text_from_attention_output(
        language_model,
        layer_idx,
        hidden_states,
        attention_output,
        None,
        per_layer_input,
    )


def run_gemma4_full_layer_with_text_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_masks: dict[str, Tensor],
    position_ids: Tensor,
    shared_kv_states: UserDict,
    text_positions: Tensor | None,
    text_delta: Tensor | None,
    text_mask: Tensor | None,
    per_layer_input: Tensor | None,
) -> Tensor:
    attn_out = gemma4_attention_output(
        language_model,
        layer_idx,
        hidden_states,
        attention_masks,
        position_ids,
        shared_kv_states,
    )
    return run_gemma4_layer_full_from_attention_output(
        language_model,
        layer_idx,
        hidden_states,
        attn_out,
        text_positions,
        text_delta,
        text_mask,
        per_layer_input,
    )


def run_gemma4_layer_text_with_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    attention_masks: dict[str, Tensor],
    position_ids: Tensor,
    shared_kv_states: UserDict,
    attention_delta: Tensor | None,
    per_layer_input: Tensor | None,
) -> Tensor:
    text_attention = gemma4_attention_output(
        language_model,
        layer_idx,
        hidden_states,
        attention_masks,
        position_ids,
        shared_kv_states,
    )
    return run_gemma4_layer_text_from_attention_output(
        language_model,
        layer_idx,
        hidden_states,
        text_attention,
        attention_delta,
        per_layer_input,
    )


def compute_gemma4_attention_effect_batched(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: Tensor,
    text_hidden_states: Tensor,
    full_attention_masks: dict[str, Tensor],
    text_attention_masks: dict[str, Tensor],
    full_position_ids: Tensor,
    text_position_ids: Tensor,
    text_positions: Tensor,
    text_mask: Tensor,
    full_shared_kv_states: UserDict,
    text_shared_kv_states: UserDict,
    precomputed_text_attention: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    joint = gemma4_attention_output(
        language_model,
        layer_idx,
        full_hidden_states,
        full_attention_masks,
        full_position_ids,
        full_shared_kv_states,
    )
    if precomputed_text_attention is None:
        text = gemma4_attention_output(
            language_model,
            layer_idx,
            text_hidden_states,
            text_attention_masks,
            text_position_ids,
            text_shared_kv_states,
        )
    else:
        text = precomputed_text_attention
    joint_text = gather_batched_positions(joint, text_positions, text_mask)
    return joint_text - text, text


def gemma4_lm_logits(model: torch.nn.Module, hidden_states: Tensor) -> Tensor:
    logits = model.lm_head(hidden_states)
    final_logit_softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return logits
