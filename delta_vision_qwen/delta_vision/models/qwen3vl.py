from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from delta_vision.models.llava import get_language_model


def qwen3vl_prompt(processor: Any, question: str) -> str:
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


def load_frozen_qwen3vl(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
) -> tuple[Any, Qwen3VLForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


def qwen3vl_image_token_id(model: torch.nn.Module, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(processor, "image_token_id", None)
    if token_id is None:
        token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return int(token_id)


def prepare_qwen3vl_sample_inputs(
    processor: Any,
    row: dict[str, Any],
    image_key: str,
    question_key: str,
    answer_key: str,
    image_root: Path | None,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor, str]:
    inputs, text_ids, answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
        processor,
        [row],
        image_key,
        question_key,
        answer_key,
        image_root,
        device,
    )
    return inputs, text_ids, answer_mask, image_paths[0]


def prepare_qwen3vl_batch_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    image_key: str,
    question_key: str,
    answer_key: str,
    image_root: Path | None,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor, list[str]]:
    prompts: list[str] = []
    full_texts: list[str] = []
    answer_token_lens: list[int] = []
    images: list[Image.Image] = []
    image_paths: list[str] = []
    eos = processor.tokenizer.eos_token or ""
    for row in rows:
        image_path = Path(row[image_key])
        if image_root is not None and not image_path.is_absolute():
            image_path = image_root / image_path
        prompt = qwen3vl_prompt(processor, str(row[question_key]))
        answer = str(row[answer_key]).strip()
        answer_suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
        full_text = f"{prompt}{answer_suffix}"
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
        prompts.append(prompt)
        full_texts.append(full_text)
        answer_token_lens.append(len(processor.tokenizer(answer_suffix, add_special_tokens=False).input_ids))
        image_paths.append(str(image_path))

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    try:
        inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    text_ids, answer_mask, text_mask = qwen3vl_text_ids_and_answer_mask(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        [int(inputs["input_ids"].shape[1]) + 1 for _ in rows],
        int(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id),
    )
    answer_mask.zero_()
    for batch_idx, answer_len in enumerate(answer_token_lens):
        text_len = int(text_mask[batch_idx].sum().item())
        start = max(0, text_len - int(answer_len))
        answer_mask[batch_idx, start:text_len] = True
    return inputs, text_ids.to(device), answer_mask.to(device), image_paths


def qwen3vl_text_ids_and_answer_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    prompt_lens: list[int],
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    rows_ids: list[list[int]] = []
    rows_answer_mask: list[list[bool]] = []
    max_text = 0
    for batch_idx in range(input_ids.shape[0]):
        valid_src = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        text_ids: list[int] = []
        answer_mask: list[bool] = []
        prompt_len = int(prompt_lens[batch_idx])
        for src_idx in valid_src:
            src_idx = int(src_idx)
            if int(mm_token_type_ids[batch_idx, src_idx].item()) != 0:
                continue
            text_ids.append(int(input_ids[batch_idx, src_idx].item()))
            answer_mask.append(src_idx >= prompt_len)
        if not text_ids:
            raise ValueError("empty text sequence after removing Qwen3-VL image tokens")
        rows_ids.append(text_ids)
        rows_answer_mask.append(answer_mask)
        max_text = max(max_text, len(text_ids))

    text_ids_tensor = torch.full(
        (input_ids.shape[0], max_text),
        int(pad_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    answer_mask_tensor = torch.zeros((input_ids.shape[0], max_text), device=input_ids.device, dtype=torch.bool)
    text_mask_tensor = torch.zeros((input_ids.shape[0], max_text), device=input_ids.device, dtype=torch.bool)
    for batch_idx, ids in enumerate(rows_ids):
        text_len = len(ids)
        text_ids_tensor[batch_idx, :text_len] = torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)
        answer_mask_tensor[batch_idx, :text_len] = torch.tensor(
            rows_answer_mask[batch_idx],
            device=input_ids.device,
            dtype=torch.bool,
        )
        text_mask_tensor[batch_idx, :text_len] = True
    return text_ids_tensor, answer_mask_tensor, text_mask_tensor


@torch.no_grad()
def build_qwen3vl_initial_hidden(model: torch.nn.Module, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    inputs_embeds, position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    return inputs_embeds, position_ids


@torch.no_grad()
def build_qwen3vl_initial_context(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    qwen_model = model.model
    input_ids = inputs["input_ids"]
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_outputs = qwen_model.get_image_features(
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        return_dict=True,
    )
    image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    visual_pos_masks = image_mask[..., 0]
    deepstack_visual_embeds = image_outputs.deepstack_features
    position_ids = qwen_model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.get("attention_mask"),
        past_key_values=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    if position_ids is None:
        raise RuntimeError("Qwen3-VL position_ids could not be computed")
    return inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds


def get_qwen3vl_text_image_positions(
    input_ids: Tensor,
    attention_mask: Tensor,
    mm_token_type_ids: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, seq_len = input_ids.shape
    rows_text: list[list[int]] = []
    rows_image: list[list[int]] = []
    max_text = 0
    max_image = 0
    for batch_idx in range(batch):
        valid = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
        text = [int(i) for i in valid if int(mm_token_type_ids[batch_idx, int(i)].item()) == 0]
        image = [int(i) for i in valid if int(mm_token_type_ids[batch_idx, int(i)].item()) == 1]
        if not text or not image:
            raise ValueError("Qwen3-VL sample must contain both text and image positions")
        rows_text.append(text)
        rows_image.append(image)
        max_text = max(max_text, len(text))
        max_image = max(max_image, len(image))

    device = input_ids.device
    text_positions = torch.zeros((batch, max_text), device=device, dtype=torch.long)
    image_positions = torch.zeros((batch, max_image), device=device, dtype=torch.long)
    text_mask = torch.zeros((batch, max_text), device=device, dtype=torch.bool)
    image_mask = torch.zeros((batch, max_image), device=device, dtype=torch.bool)
    full_mask = attention_mask.to(device=device).bool()
    text_position_ids = torch.zeros((3, batch, max_text), device=device, dtype=position_ids.dtype)
    for batch_idx in range(batch):
        t = torch.tensor(rows_text[batch_idx], device=device, dtype=torch.long)
        v = torch.tensor(rows_image[batch_idx], device=device, dtype=torch.long)
        text_positions[batch_idx, : t.numel()] = t
        image_positions[batch_idx, : v.numel()] = v
        text_mask[batch_idx, : t.numel()] = True
        image_mask[batch_idx, : v.numel()] = True
        text_position_ids[:, batch_idx, : t.numel()] = position_ids[:, batch_idx].index_select(1, t)
    return text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask


def gather_batched_positions(hidden_states: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
    gather_idx = positions.to(device=hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    gathered = torch.gather(hidden_states, dim=1, index=gather_idx)
    return gathered * mask.to(device=hidden_states.device, dtype=gathered.dtype).unsqueeze(-1)


def qwen3vl_static_visual_memory(
    hidden0: Tensor,
    image_positions: Tensor,
    image_mask: Tensor,
    deepstack_visual_embeds: list[Tensor] | None,
    mode: str,
) -> Tensor:
    memory = gather_batched_positions(hidden0, image_positions, image_mask)
    if mode == "v0":
        return memory
    if mode != "vdeep":
        raise ValueError(f"unsupported Qwen3-VL static visual memory mode: {mode}")
    if not deepstack_visual_embeds:
        return memory
    deep_sum = torch.stack(
        [x.to(device=memory.device, dtype=memory.dtype) for x in deepstack_visual_embeds],
        dim=0,
    ).sum(dim=0)
    return memory + deep_sum.unsqueeze(0)


def qwen3vl_visual_memory_by_layer(
    hidden0: Tensor,
    image_positions: Tensor,
    image_mask: Tensor,
    deepstack_visual_embeds: list[Tensor] | None,
    mode: str,
    num_layers: int,
) -> list[Tensor]:
    base = gather_batched_positions(hidden0, image_positions, image_mask)
    if mode == "v0":
        return [base for _ in range(num_layers)]
    if mode == "vdeep":
        memory = qwen3vl_static_visual_memory(hidden0, image_positions, image_mask, deepstack_visual_embeds, mode)
        return [memory for _ in range(num_layers)]
    if mode != "vcum":
        raise ValueError(f"unsupported Qwen3-VL visual memory mode: {mode}")
    memories = []
    running_deep = torch.zeros_like(base)
    for layer_idx in range(num_layers):
        memories.append(base + running_deep)
        if deepstack_visual_embeds and layer_idx < len(deepstack_visual_embeds):
            running_deep = running_deep + deepstack_visual_embeds[layer_idx].to(
                device=base.device,
                dtype=base.dtype,
            ).unsqueeze(0)
    return memories


@torch.no_grad()
def qwen3vl_prefix_visual_memory_by_layer(
    language_model: torch.nn.Module,
    hidden0: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    image_positions: Tensor,
    image_mask: Tensor,
    visual_pos_masks: Tensor,
    deepstack_visual_embeds: list[Tensor] | None,
) -> list[Tensor]:
    """Compute per-layer visual states from the image prefix only.

    Qwen3-VL places image tokens before the question text. Under the causal mask,
    those image tokens cannot attend to later question tokens, so their layerwise
    states can be precomputed from the prefix ending at the last image token.
    """
    valid_positions = image_positions[image_mask.to(device=image_positions.device).bool()]
    if valid_positions.numel() == 0:
        raise ValueError("empty Qwen3-VL image positions")
    prefix_len = int(valid_positions.max().item()) + 1
    h = hidden0[:, :prefix_len].contiguous()
    prefix_position_ids = position_ids[:, :, :prefix_len].contiguous()
    prefix_attention_mask = attention_mask[:, :prefix_len].contiguous()
    prefix_visual_pos_masks = visual_pos_masks[:, :prefix_len].contiguous()
    memories: list[Tensor] = []
    for layer_idx in range(len(language_model.layers)):
        memories.append(gather_batched_positions(h, image_positions, image_mask))
        h = run_qwen3vl_full_layer_with_text_delta(
            language_model,
            layer_idx,
            h,
            prefix_position_ids,
            prefix_attention_mask,
            image_positions,
            text_delta=None,
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            h = language_model._deepstack_process(
                h,
                prefix_visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )
    return memories


def scatter_batched_positions(base_hidden_states: Tensor, positions: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    out = base_hidden_states.clone()
    valid = mask.to(device=out.device).bool()
    if bool(valid.any()):
        batch_idx = torch.arange(out.shape[0], device=out.device).unsqueeze(1).expand_as(positions)
        out[batch_idx[valid], positions.to(device=out.device)[valid]] = values.to(device=out.device, dtype=out.dtype)[
            valid
        ]
    return out


def qwen3vl_attention_output(
    self_attn: torch.nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
) -> Tensor:
    attn_output, _ = self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    return attn_output


def run_qwen3vl_layer_text_with_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_delta: Tensor | None,
    padding_mask: Tensor | None = None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    attention_mask_2d = None if padding_mask is None else (~padding_mask).to(dtype=torch.long)
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out = qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def qwen3vl_text_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Return the text-only attention sublayer output before residual add."""
    layer = language_model.layers[layer_idx]
    attention_mask_2d = None if padding_mask is None else (~padding_mask).to(dtype=torch.long)
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    normed = layer.input_layernorm(hidden_states)
    return qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)


def qwen3vl_text_attention_heads(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Return text-only attention output before o_proj as [batch, text_len, heads, head_dim]."""
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    if padding_mask is None:
        valid_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
    else:
        valid_mask = ~padding_mask.to(device=hidden_states.device, dtype=torch.bool)
    seq_len = hidden_states.shape[1]
    causal = torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool).tril()
    attention_mask = causal.view(1, 1, seq_len, seq_len) & valid_mask.view(valid_mask.shape[0], 1, 1, seq_len)
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        *language_model.rotary_emb(normed, position_ids),
    )
    key_states = repeat_kv(key_states, int(self_attn.num_key_value_groups))
    value_states = repeat_kv(value_states, int(self_attn.num_key_value_groups))
    out = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(self_attn.scaling),
    )
    return out.transpose(1, 2).contiguous()


def qwen3vl_text_attention_output_with_visual_kv(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    vision_states: Tensor,
    visual_position_ids: Tensor,
    text_padding_mask: Tensor | None = None,
    vision_padding_mask: Tensor | None = None,
) -> Tensor:
    """Run Qwen native text attention over text K/V plus injected visual K/V."""
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn

    normed_text = layer.input_layernorm(hidden_states)
    text_shape = normed_text.shape[:-1]
    hidden_shape = (*text_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_key_states = self_attn.k_norm(self_attn.k_proj(normed_text).view(hidden_shape)).transpose(1, 2)
    text_value_states = self_attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
    query_states, text_key_states = apply_rotary_pos_emb(
        query_states,
        text_key_states,
        *language_model.rotary_emb(normed_text, position_ids),
    )

    normed_visual = layer.input_layernorm(vision_states)
    visual_shape = normed_visual.shape[:-1]
    visual_hidden_shape = (*visual_shape, -1, self_attn.head_dim)
    visual_key_states = self_attn.k_norm(self_attn.k_proj(normed_visual).view(visual_hidden_shape)).transpose(1, 2)
    visual_value_states = self_attn.v_proj(normed_visual).view(visual_hidden_shape).transpose(1, 2)
    _, visual_key_states = apply_rotary_pos_emb(
        visual_key_states,
        visual_key_states,
        *language_model.rotary_emb(normed_visual, visual_position_ids),
    )

    num_key_value_groups = int(self_attn.num_key_value_groups)
    text_key_states = repeat_kv(text_key_states, num_key_value_groups)
    text_value_states = repeat_kv(text_value_states, num_key_value_groups)
    visual_key_states = repeat_kv(visual_key_states, num_key_value_groups)
    visual_value_states = repeat_kv(visual_value_states, num_key_value_groups)

    key_states = torch.cat([visual_key_states, text_key_states], dim=2)
    value_states = torch.cat([visual_value_states, text_value_states], dim=2)

    batch, text_len = hidden_states.shape[:2]
    visual_len = vision_states.shape[1]
    device = hidden_states.device
    if text_padding_mask is None:
        valid_text = torch.ones((batch, text_len), device=device, dtype=torch.bool)
    else:
        valid_text = ~text_padding_mask.to(device=device, dtype=torch.bool)
    if vision_padding_mask is None:
        valid_visual = torch.ones((batch, visual_len), device=device, dtype=torch.bool)
    else:
        valid_visual = ~vision_padding_mask.to(device=device, dtype=torch.bool)
    causal = torch.ones((text_len, text_len), device=device, dtype=torch.bool).tril()
    text_allowed = causal.view(1, text_len, text_len) & valid_text.view(batch, 1, text_len)
    visual_allowed = valid_visual.view(batch, 1, visual_len).expand(batch, text_len, visual_len)
    attention_mask = torch.cat([visual_allowed, text_allowed], dim=-1).unsqueeze(1)

    heads = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(self_attn.scaling),
    )
    heads = heads.transpose(1, 2).contiguous()
    return self_attn.o_proj(heads.reshape(*text_shape, -1).contiguous())


def qwen3vl_text_attention_outputs_cache(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    past_key_values: Any,
    attention_mask_2d: Tensor | None = None,
    position_embeddings: tuple[Tensor, Tensor] | None = None,
    mask_past_key_values: Any | None = None,
    return_query_states: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Return cached text attention outputs and update the text KV cache once.

    The first tensor is the post-o_proj attention output. The second tensor is
    the pre-o_proj output in query-head layout [batch, tokens, heads, head_dim].
    This is needed by factorized_native_head_o Sidecar decode, where the delta
    depends on text-only attention heads but the language-model cache must still
    be updated exactly once for the current token.
    """
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=mask_past_key_values,
        position_ids=text_position_ids,
    )
    if position_embeddings is None:
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self_attn.layer_idx)
    key_states = repeat_kv(key_states, int(self_attn.num_key_value_groups))
    value_states = repeat_kv(value_states, int(self_attn.num_key_value_groups))
    heads = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(self_attn.scaling),
    )
    heads = heads.transpose(1, 2).contiguous()
    output = self_attn.o_proj(heads.reshape(*input_shape, -1).contiguous())
    if return_query_states:
        query_merged = query_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return output, heads, query_merged
    return output, heads


def run_qwen3vl_layer_text_from_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    text_attention: Tensor,
    attention_delta: Tensor | None,
) -> Tensor:
    """Run a Qwen text block tail after a precomputed attention output."""
    layer = language_model.layers[layer_idx]
    residual = hidden_states
    hidden_states = residual + text_attention.to(dtype=hidden_states.dtype)
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def run_qwen3vl_layer_text_with_attention_delta_cache(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_delta: Tensor | None,
    past_key_values: Any,
    attention_mask_2d: Tensor | None = None,
    position_embeddings: tuple[Tensor, Tensor] | None = None,
    mask_past_key_values: Any | None = None,
) -> Tensor:
    """Run one Qwen3-VL text layer with KV cache and inject attention delta."""
    layer = language_model.layers[layer_idx]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=mask_past_key_values,
        position_ids=text_position_ids,
    )
    if position_embeddings is None:
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        hidden_states=normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def run_qwen3vl_full_layer_with_text_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_mask_2d: Tensor | None,
    text_positions: Tensor,
    text_delta: Tensor | None,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out = qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
    hidden_states = residual + attn_out
    if text_delta is not None:
        hidden_states = hidden_states.clone()
        for batch_idx in range(hidden_states.shape[0]):
            hidden_states[batch_idx, text_positions[batch_idx]] = (
                hidden_states[batch_idx, text_positions[batch_idx]] + text_delta[batch_idx].to(hidden_states.dtype)
            )
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def run_qwen3vl_full_layer_with_text_delta_cache(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_mask_2d: Tensor | None,
    text_positions: Tensor | None,
    text_delta: Tensor | None,
    past_key_values: Any,
    position_embeddings: tuple[Tensor, Tensor] | None = None,
    mask_past_key_values: Any | None = None,
) -> Tensor:
    """Run one Qwen3-VL full layer with KV cache and optional text delta.

    During decode `text_positions` can be None because the current sequence
    contains only the newly generated text token.
    """
    layer = language_model.layers[layer_idx]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask_2d,
        past_key_values=mask_past_key_values,
        position_ids=text_position_ids,
    )
    if position_embeddings is None:
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        hidden_states=normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    hidden_states = residual + attn_out
    if text_delta is not None:
        if text_positions is None:
            hidden_states = hidden_states + text_delta.to(hidden_states.dtype)
        else:
            hidden_states = hidden_states.clone()
            for batch_idx in range(hidden_states.shape[0]):
                hidden_states[batch_idx, text_positions[batch_idx]] = (
                    hidden_states[batch_idx, text_positions[batch_idx]] + text_delta[batch_idx].to(hidden_states.dtype)
                )
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def compute_qwen3vl_attention_effect_batched(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: Tensor,
    text_hidden_states: Tensor,
    full_position_ids: Tensor,
    text_position_ids: Tensor,
    text_positions: Tensor,
    full_mask: Tensor,
    text_mask: Tensor,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    full_normed = layer.input_layernorm(full_hidden_states)
    text_normed = layer.input_layernorm(text_hidden_states)
    full_attention = create_causal_mask(
        config=language_model.config,
        inputs_embeds=full_normed,
        attention_mask=full_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=full_position_ids[0],
    )
    text_attention = create_causal_mask(
        config=language_model.config,
        inputs_embeds=text_normed,
        attention_mask=text_mask.to(dtype=torch.long),
        past_key_values=None,
        position_ids=text_position_ids[0],
    )
    joint = qwen3vl_attention_output(
        layer.self_attn,
        full_normed,
        language_model.rotary_emb(full_normed, full_position_ids),
        full_attention,
    )
    text = qwen3vl_attention_output(
        layer.self_attn,
        text_normed,
        language_model.rotary_emb(text_normed, text_position_ids),
        text_attention,
    )
    joint_text = gather_batched_positions(joint, text_positions, text_mask)
    return joint_text - text


def compute_qwen3vl_visual_attention_mass_batched(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: Tensor,
    full_position_ids: Tensor,
    text_positions: Tensor,
    image_positions: Tensor,
    full_mask: Tensor,
    text_mask: Tensor,
    image_mask: Tensor,
    reduce_heads: str = "mean",
    text_chunk_size: int = 128,
) -> Tensor:
    """Return visual attention mass for text queries in Qwen native attention.

    The mass is computed from the same Qwen q/k projections, q/k norms, M-RoPE
    and GQA repeat used by the native self-attention:

        m_vis = sum_{j in image keys} softmax(q_text k_j)

    Output shape is [batch, text_len, 1] when reduce_heads="mean", otherwise
    [batch, text_len, num_query_heads].
    """
    if reduce_heads not in {"mean", "none"}:
        raise ValueError("reduce_heads must be 'mean' or 'none'")
    layer = language_model.layers[layer_idx]
    self_attn = layer.self_attn
    normed = layer.input_layernorm(full_hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        *language_model.rotary_emb(normed, full_position_ids),
    )
    key_states = repeat_kv(key_states, int(self_attn.num_key_value_groups))

    batch, heads, full_len, head_dim = query_states.shape
    text_len = text_positions.shape[1]
    text_query_idx = text_positions.to(device=query_states.device, dtype=torch.long)[:, None, :, None].expand(
        batch,
        heads,
        text_len,
        head_dim,
    )
    query_text = torch.gather(query_states, dim=2, index=text_query_idx)

    key_t = key_states.float().transpose(-2, -1)
    key_idx = torch.arange(full_len, device=query_states.device).view(1, 1, 1, full_len)
    valid_keys = full_mask.to(device=query_states.device, dtype=torch.bool).view(batch, 1, 1, full_len)
    image_key_idx = image_positions.to(device=query_states.device, dtype=torch.long).view(batch, 1, 1, -1)
    image_valid_base = image_mask.to(device=query_states.device, dtype=torch.bool).view(batch, 1, 1, -1)
    image_gather_idx = image_positions.to(device=query_states.device, dtype=torch.long)[:, None, None, :].expand(
        batch,
        heads,
        -1,
        image_positions.shape[1],
    )

    chunks: list[Tensor] = []
    chunk_size = max(1, int(text_chunk_size))
    for start in range(0, text_len, chunk_size):
        end = min(start + chunk_size, text_len)
        q_chunk = query_text[:, :, start:end].float()
        scores = torch.matmul(q_chunk, key_t) * float(self_attn.scaling)

        query_pos = text_positions.to(device=query_states.device, dtype=torch.long)[:, None, start:end, None]
        causal_valid = valid_keys & (key_idx <= query_pos)
        scores_all = scores.masked_fill(~causal_valid, torch.finfo(scores.dtype).min)
        all_lse = torch.logsumexp(scores_all, dim=-1)

        image_scores = torch.gather(
            scores,
            dim=-1,
            index=image_gather_idx.expand(batch, heads, end - start, image_positions.shape[1]),
        )
        image_valid = image_valid_base & (image_key_idx <= query_pos)
        image_scores = image_scores.masked_fill(~image_valid, torch.finfo(image_scores.dtype).min)
        image_lse = torch.logsumexp(image_scores, dim=-1)
        chunk_mass = torch.exp(image_lse - all_lse).masked_fill(
            ~text_mask.to(device=query_states.device, dtype=torch.bool)[:, None, start:end],
            0.0,
        )
        chunks.append(chunk_mass)

    mass = torch.cat(chunks, dim=-1).transpose(1, 2).contiguous()
    if reduce_heads == "mean":
        return mass.mean(dim=-1, keepdim=True).to(dtype=full_hidden_states.dtype)
    return mass.to(dtype=full_hidden_states.dtype)


def qwen3vl_lm_norm(model: torch.nn.Module) -> torch.nn.Module:
    return get_language_model(model).norm
