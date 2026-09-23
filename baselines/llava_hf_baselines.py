"""HF LLaVA baseline evaluation helpers using the project benchmark registry."""
from __future__ import annotations

import time
from typing import Any

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb

from src.benchmarks import get_benchmark_spec, score_prediction, summarize_metric
from src.model import _get_language_model, llava_projected_image_features, load_frozen_llava


LLAVA_LAYER_PRUNING_METHODS = {"fastv", "dart", "sparsevlm", "divprune", "zoo"}
LLAVA_SUPPORTED_METHODS = LLAVA_LAYER_PRUNING_METHODS | {"base", "visionzip"}
LLAVA_PRUNING_LAYERS = {
    "fastv": (2,),
    "dart": (2,),
    "sparsevlm": (2, 6, 15),
    "divprune": (0,),
    "zoo": (0,),
}


def load_llava_baseline_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
):
    return load_frozen_llava(
        model_path,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )


def evaluate_llava_baseline(
    model,
    processor,
    dataset,
    *,
    method: str,
    benchmark: str,
    retention: float,
    max_new_tokens: int,
    log_every: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = get_benchmark_spec(benchmark)
    device = next(model.parameters()).device
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    predictions: list[dict[str, Any]] = []
    total_time = 0.0

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        image_sizes = item.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.unsqueeze(0).to(device) if torch.is_tensor(image_sizes) else image_sizes

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            text = generate_llava_baseline(
                model,
                processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_token_id=image_token_id,
                method=method,
                retention=retention,
                max_new_tokens=max_new_tokens,
                image_sizes=image_sizes,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_time += t1 - t0

        row = dataset.rows[idx]
        scored = score_prediction(
            metric=spec.metric,
            prediction_text=text,
            answer=row.get("answer"),
            answers=row.get("answers"),
            choices=row.get("choices"),
            question=row.get("question"),
        )
        predictions.append(
            {
                "index": row.get("index", idx),
                "prediction_text": text,
                "time": t1 - t0,
                **scored,
            }
        )

        if (idx + 1) % log_every == 0 or idx == len(dataset) - 1:
            running_score = sum(p["score"] for p in predictions) / max(len(predictions), 1)
            print(f"  [{idx+1}/{len(dataset)}] score={running_score:.4f} last={text[:50]!r}", flush=True)

    summary = summarize_metric(spec.metric, predictions, dataset.rows)
    summary["total_time"] = total_time
    summary["avg_time"] = total_time / max(len(predictions), 1)
    return predictions, summary


@torch.inference_mode()
def generate_llava_baseline(
    model,
    processor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_id: int,
    method: str,
    retention: float,
    max_new_tokens: int,
    image_sizes=None,
) -> str:
    method_key = method.lower()
    if method_key not in LLAVA_SUPPORTED_METHODS:
        raise ValueError(f"Unsupported LLaVA baseline method={method!r}")

    tokenizer = processor.tokenizer
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if method_key == "base" or float(retention) >= 1.0:
        native_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if image_sizes is not None:
            native_inputs["image_sizes"] = image_sizes
        outputs = model.generate(**native_inputs, **generation_kwargs)
        generated = outputs[0, input_ids.shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    visual_memory = llava_projected_image_features(model, pixel_values, image_sizes=image_sizes)
    if method_key == "dart":
        from src.llava_dart_corrected import DartDecoder
        embeds, _, start, length = build_llava_inputs_embeds_with_image_span(
            model, input_ids=input_ids, attention_mask=attention_mask,
            image_token_id=image_token_id, visual_memory=visual_memory)
        tokens, _ = DartDecoder(model).generate(
            embeds, start, length, retention, max_new_tokens, _eos_token_ids(tokenizer))
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()
    visual_memory = reduce_visual_memory(visual_memory, method=method_key, retention=retention)

    inputs_embeds, lm_attention_mask, _, _ = build_llava_inputs_embeds_with_image_span(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        visual_memory=visual_memory,
    )

    outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=lm_attention_mask, **generation_kwargs)
    # With inputs_embeds only, HF returns generated token IDs only. Length is
    # not a reliable indication of whether a prompt prefix is present.
    generated = outputs[0]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_llava_inputs_embeds(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    visual_memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_embeds, lm_attention_mask, _, _ = build_llava_inputs_embeds_with_image_span(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        visual_memory=visual_memory,
    )
    return inputs_embeds, lm_attention_mask


def build_llava_inputs_embeds_with_image_span(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    visual_memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    language_model = _get_language_model(model)
    ids = input_ids[0]
    valid = attention_mask[0].bool()
    ids = ids[valid]
    image_positions = torch.where(ids == image_token_id)[0]
    if image_positions.numel() == 0:
        inputs_embeds = language_model.embed_tokens(ids.unsqueeze(0))
        return inputs_embeds, torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device), 0, 0

    start = int(image_positions[0].item())
    end = int(image_positions[-1].item()) + 1
    before_ids = ids[:start].unsqueeze(0)
    after_ids = ids[end:].unsqueeze(0)

    before = language_model.embed_tokens(before_ids) if before_ids.numel() else None
    after = language_model.embed_tokens(after_ids) if after_ids.numel() else None
    visual_memory = visual_memory.to(device=ids.device, dtype=language_model.embed_tokens.weight.dtype)
    parts = [part for part in (before, visual_memory, after) if part is not None]
    inputs_embeds = torch.cat(parts, dim=1)
    lm_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return inputs_embeds, lm_attention_mask, start, int(visual_memory.shape[1])


def greedy_decode_language_model(
    model,
    tokenizer,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    language_model = _get_language_model(model)
    eos_ids = _eos_token_ids(tokenizer)
    generated: list[int] = []

    outputs = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
    logits = model.lm_head(outputs.last_hidden_state[:, -1])
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(logits[0], dim=-1).item())
        generated.append(next_token)
        if next_token in eos_ids:
            break
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat([attention_mask, torch.ones_like(token_tensor)], dim=1)
        outputs = language_model(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1])
        past_key_values = outputs.past_key_values
    return generated


def greedy_decode_language_model_with_layer_pruning(
    model,
    tokenizer,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_start: int,
    image_len: int,
    method: str,
    retention: float,
    max_new_tokens: int,
    pivot_image_token: int = 4,
    pivot_text_token: int = 4,
) -> list[int]:
    language_model = _get_language_model(model)
    eos_ids = _eos_token_ids(tokenizer)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = llava_layer_pruned_last_logits(
            model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_start=image_start,
            image_len=image_len,
            method=method,
            retention=retention,
            pivot_image_token=pivot_image_token,
            pivot_text_token=pivot_text_token,
        )
        next_token = int(torch.argmax(logits[0], dim=-1).item())
        generated.append(next_token)
        if next_token in eos_ids:
            break
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=inputs_embeds.device)
        token_embed = language_model.embed_tokens(token_tensor).to(dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, token_embed], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(token_tensor)], dim=1)
    return generated


def llava_layer_pruned_last_logits(
    model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_start: int,
    image_len: int,
    method: str,
    retention: float,
    pivot_image_token: int,
    pivot_text_token: int,
) -> torch.Tensor:
    language_model = _get_language_model(model)
    hidden_states = inputs_embeds
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long).unsqueeze(0)
    current_attention_mask = attention_mask
    num_layers = int(getattr(language_model.config, "num_hidden_layers", len(language_model.layers)))
    pruning_layers = tuple(layer for layer in LLAVA_PRUNING_LAYERS[method] if 0 <= layer < num_layers)
    original_image_len = int(image_len)
    current_image_len = int(image_len)
    pruned_once = False

    for layer_idx, decoder_layer in enumerate(language_model.layers[:num_layers]):
        if (
            layer_idx in pruning_layers
            and current_image_len > 0
            and hidden_states.shape[1] > 1
            and (method == "sparsevlm" or not pruned_once)
        ):
            retained_idx = llava_retained_image_token_indices(
                decoder_layer,
                language_model,
                hidden_states,
                position_ids=position_ids,
                method=method,
                image_start=image_start,
                image_len=current_image_len,
                original_image_len=original_image_len,
                retention=retention,
                pivot_image_token=pivot_image_token,
                pivot_text_token=pivot_text_token,
            )
            if 0 < retained_idx.numel() < current_image_len:
                hidden_states, position_ids, current_attention_mask = prune_llava_image_tokens(
                    hidden_states,
                    position_ids,
                    current_attention_mask,
                    image_start=image_start,
                    image_len=current_image_len,
                    retained_idx=retained_idx,
                )
                current_image_len = int(retained_idx.numel())
                pruned_once = True

        position_embeddings = language_model.rotary_emb(hidden_states, position_ids=position_ids)
        causal_mask = llava_causal_mask(language_model, hidden_states, current_attention_mask, position_ids)
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

    hidden_states = language_model.norm(hidden_states)
    return model.lm_head(hidden_states[:, -1])


def prune_llava_image_tokens(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    image_start: int,
    image_len: int,
    retained_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    seq_len = int(hidden_states.shape[1])
    pre_img = torch.arange(image_start, device=hidden_states.device)
    post_img = torch.arange(image_start + image_len, seq_len, device=hidden_states.device)
    keep_indices = torch.cat([pre_img, retained_idx.to(device=hidden_states.device), post_img]).sort().values
    hidden_states = hidden_states[:, keep_indices, :]
    position_ids = position_ids[:, keep_indices]
    if attention_mask is not None:
        attention_mask = attention_mask[:, keep_indices]
    return hidden_states, position_ids, attention_mask


def llava_causal_mask(
    language_model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor | None:
    mask_fn = create_causal_mask
    if getattr(language_model.config, "sliding_window", None) is not None:
        mask_fn = create_sliding_window_causal_mask
    return mask_fn(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )


def llava_retained_image_token_indices(
    decoder_layer,
    language_model,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    method: str,
    image_start: int,
    image_len: int,
    original_image_len: int,
    retention: float,
    pivot_image_token: int,
    pivot_text_token: int,
) -> torch.Tensor:
    image_len = max(0, min(int(image_len), int(hidden_states.shape[1]) - int(image_start)))
    if image_len <= 0:
        return torch.empty(0, device=hidden_states.device, dtype=torch.long)

    base_len = original_image_len if method == "sparsevlm" else image_len
    target_keep = _llava_target_keep(base_len, retention)
    target_keep = min(target_keep, image_len)
    if target_keep >= image_len:
        return torch.arange(image_start, image_start + image_len, device=hidden_states.device)

    if method == "fastv":
        return llava_fastv_retained_image_token_indices(
            decoder_layer,
            language_model,
            hidden_states,
            position_ids=position_ids,
            image_start=image_start,
            image_len=image_len,
            target_keep=target_keep,
        )
    if method == "dart":
        return llava_dart_retained_image_token_indices(
            decoder_layer,
            language_model,
            hidden_states,
            image_start=image_start,
            image_len=image_len,
            retention=retention,
            pivot_image_token=pivot_image_token,
            pivot_text_token=pivot_text_token,
        )
    if method == "sparsevlm":
        return llava_sparsevlm_retained_image_token_indices(
            decoder_layer,
            language_model,
            hidden_states,
            position_ids=position_ids,
            image_start=image_start,
            image_len=image_len,
            target_keep=target_keep,
        )
    if method == "divprune":
        selected = _divprune_select_tokens(hidden_states[0, image_start:image_start + image_len], target_keep)
        return (selected + image_start).sort().values
    if method == "zoo":
        visual_features = hidden_states[0, image_start:image_start + image_len]
        importance = _zoo_token_sensitivity(
            visual_features,
            decoder_layer,
            num_refine=64,
            noise_scale=0.01,
        )
        selected = _zoo_select_tokens(visual_features, importance, target_keep)
        return (selected + image_start).sort().values
    raise ValueError(f"Unsupported LLaVA layer pruning method={method!r}")


def _llava_target_keep(n_tokens: int, retention: float) -> int:
    n_tokens = max(int(n_tokens), 0)
    if n_tokens <= 0:
        return 0
    keep = round(n_tokens * float(retention))
    return max(1, min(n_tokens, keep))


def llava_fastv_retained_image_token_indices(
    decoder_layer,
    language_model,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    image_start: int,
    image_len: int,
    target_keep: int,
) -> torch.Tensor:
    q_states, k_states, scaling = llava_qk_states(decoder_layer, language_model, hidden_states, position_ids)
    q_last = q_states[:, :, -1:, :]
    k_img = k_states[:, :, image_start:image_start + image_len, :]
    attn_scores = torch.matmul(q_last, k_img.transpose(-1, -2)) * scaling
    attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
    image_scores = attn_scores.mean(dim=1)[0, 0]
    return _global_topk_from_scores(image_scores, target_keep, image_start)


def llava_sparsevlm_retained_image_token_indices(
    decoder_layer,
    language_model,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    image_start: int,
    image_len: int,
    target_keep: int,
) -> torch.Tensor:
    q_states, k_states, scaling = llava_qk_states(decoder_layer, language_model, hidden_states, position_ids)
    attn_logits = torch.matmul(q_states, k_states.transpose(-1, -2)) * scaling
    attn_logits = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32)
    attn_avg = attn_logits.mean(dim=1)[0]

    cur_seq = int(hidden_states.shape[1])
    text_start = image_start + image_len
    if text_start < cur_seq:
        v_to_t = attn_avg[image_start:image_start + image_len, text_start:].mean(dim=0)
        if v_to_t.numel() > 0:
            text_raters = torch.nonzero(v_to_t > v_to_t.mean(), as_tuple=True)[0] + text_start
        else:
            text_raters = torch.empty(0, device=hidden_states.device, dtype=torch.long)
    else:
        text_raters = torch.empty(0, device=hidden_states.device, dtype=torch.long)
    if text_raters.numel() == 0:
        text_raters = torch.tensor([cur_seq - 1], device=hidden_states.device, dtype=torch.long)

    valid_raters = text_raters[text_raters < cur_seq]
    if valid_raters.numel() > 0:
        text_to_vis = attn_avg[valid_raters, image_start:image_start + image_len].sum(dim=0)
    else:
        text_to_vis = attn_avg[-1, image_start:image_start + image_len]
    return _global_topk_from_scores(text_to_vis, target_keep, image_start)


def llava_qk_states(
    decoder_layer,
    language_model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    normed = decoder_layer.input_layernorm(hidden_states)
    attn = decoder_layer.self_attn
    hidden_shape = (*normed.shape[:-1], -1, attn.head_dim)
    q_states = attn.q_proj(normed).view(hidden_shape)
    k_states = attn.k_proj(normed).view(hidden_shape)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if q_norm is not None:
        q_states = q_norm(q_states)
    if k_norm is not None:
        k_states = k_norm(k_states)
    q_states = q_states.transpose(1, 2)
    k_states = k_states.transpose(1, 2)
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids=position_ids)
    q_states, k_states = llama_apply_rotary_pos_emb(q_states, k_states, *position_embeddings)
    if k_states.shape[1] != q_states.shape[1]:
        if q_states.shape[1] % k_states.shape[1] == 0:
            k_states = k_states.repeat_interleave(q_states.shape[1] // k_states.shape[1], dim=1)
        else:
            q_states = q_states.mean(dim=1, keepdim=True)
            k_states = k_states.mean(dim=1, keepdim=True)
    scaling = float(getattr(attn, "scaling", attn.head_dim**-0.5))
    return q_states, k_states, scaling


def _global_topk_from_scores(scores: torch.Tensor, keep: int, offset: int) -> torch.Tensor:
    keep = max(1, min(int(keep), int(scores.numel())))
    return (torch.topk(scores, k=keep, largest=True).indices + offset).sort().values


def llava_dart_retained_image_token_indices(
    decoder_layer,
    language_model,
    hidden_states: torch.Tensor,
    *,
    image_start: int,
    image_len: int,
    retention: float,
    pivot_image_token: int,
    pivot_text_token: int,
) -> torch.Tensor:
    device = hidden_states.device
    image_len = max(0, min(int(image_len), int(hidden_states.shape[1]) - int(image_start)))
    if image_len <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    target_keep = max(1, min(image_len, round(image_len * float(retention))))
    normed = decoder_layer.input_layernorm(hidden_states)
    attn = decoder_layer.self_attn
    hidden_shape = (*normed.shape[:-1], -1, attn.head_dim)
    k_states = attn.k_proj(normed).view(hidden_shape)
    k_norm = getattr(attn, "k_norm", None)
    if k_norm is not None:
        k_states = k_norm(k_states)
    k_flat = k_states.reshape(k_states.shape[0], k_states.shape[1], -1)
    last_layer_state = language_model.norm(hidden_states)

    image_slice = slice(image_start, image_start + image_len)
    text_start = image_start + image_len
    image_states = k_flat[0][image_slice]
    text_states = k_flat[0][text_start:]
    actual_pivot_img = min(int(pivot_image_token), image_len, target_keep)
    if actual_pivot_img <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    image_indices = (image_states.norm(p=1, dim=-1).topk(actual_pivot_img).indices + image_start).tolist()

    actual_pivot_text = min(int(pivot_text_token), int(text_states.shape[0]))
    query_indices: list[int] = []
    if actual_pivot_text > 0:
        query_indices = (text_states.norm(p=1, dim=-1).topk(actual_pivot_text).indices + text_start).tolist()

    selected = set(int(idx) for idx in image_indices)
    valid = set(range(image_start, image_start + image_len)) - selected
    pivots = image_indices + query_indices
    for item in pivots:
        if len(selected) >= target_keep or not valid:
            break
        valid_list = list(valid)
        remaining = target_keep - len(selected)
        per_pivot = max(1, (remaining + len(pivots) - 1) // max(1, len(pivots)))
        topk = min(per_pivot, remaining, len(valid_list))
        valid_vectors = last_layer_state[0][valid_list, :]
        cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[0][item, :], valid_vectors, dim=-1)
        chosen = cos_sim.topk(topk).indices.tolist()
        chosen_real = [valid_list[int(idx)] for idx in chosen]
        selected.update(chosen_real)
        valid.difference_update(chosen_real)

    if len(selected) > target_keep:
        scored = torch.tensor(sorted(selected), device=device, dtype=torch.long)
        scores = last_layer_state[0][scored, :].norm(dim=-1)
        scored = scored[torch.topk(scores, k=target_keep, largest=True).indices]
        return scored.sort().values
    return torch.tensor(sorted(selected), device=device, dtype=torch.long)


def reduce_visual_memory(visual_memory: torch.Tensor, *, method: str, retention: float) -> torch.Tensor:
    n_tokens = int(visual_memory.shape[1])
    keep = max(1, min(n_tokens, round(n_tokens * float(retention))))
    if keep >= n_tokens:
        return visual_memory

    method = method.lower()
    features = visual_memory[0].float()
    if method == "fastv":
        indices = _top_norm_indices(features, keep)
    elif method == "dart":
        indices = _dart_indices(features, keep)
    elif method == "sparsevlm":
        indices = _sparsevlm_indices(features, keep)
    elif method == "visionzip":
        indices = _visionzip_indices(features, keep)
    elif method == "divprune":
        indices = _diversity_indices(features, keep)
    elif method == "zoo":
        indices = _zoo_indices(features, keep)
    else:
        raise ValueError(f"Unsupported LLaVA baseline method={method!r}")
    return visual_memory[:, indices.to(device=visual_memory.device), :]


def _top_norm_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    scores = features.norm(dim=-1)
    return torch.topk(scores, k=keep, largest=True).indices.sort().values


def _dart_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    scores = features.abs().mean(dim=-1) + centered.norm(dim=-1)
    return torch.topk(scores, k=keep, largest=True).indices.sort().values


def _sparsevlm_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    n_tokens = features.shape[0]
    norm_scores = features.norm(dim=-1)
    grid = torch.linspace(-1.0, 1.0, n_tokens, device=features.device)
    central_prior = 1.0 - grid.abs()
    scores = 0.8 * _normalize_scores(norm_scores) + 0.2 * central_prior
    return torch.topk(scores, k=keep, largest=True).indices.sort().values


def _visionzip_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    dominant = max(1, min(keep, round(keep * 0.8)))
    contextual = keep - dominant
    dominant_idx = _top_norm_indices(features, dominant)
    if contextual <= 0:
        return dominant_idx
    mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
    mask[dominant_idx] = False
    rest_idx = torch.where(mask)[0]
    if rest_idx.numel() <= contextual:
        return torch.cat([dominant_idx, rest_idx]).sort().values
    rest = features[rest_idx]
    anchor = features[dominant_idx].mean(dim=0, keepdim=True)
    scores = torch.nn.functional.cosine_similarity(rest, anchor, dim=-1).neg()
    contextual_idx = rest_idx[torch.topk(scores, k=contextual, largest=True).indices]
    return torch.cat([dominant_idx, contextual_idx]).sort().values


def _divprune_select_tokens(visual_feature_vectors: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep_count = min(max(int(keep_count), 1), int(visual_feature_vectors.shape[0]))
    features = torch.nn.functional.normalize(visual_feature_vectors.float(), dim=-1)
    dist_matrix = 1.0 - torch.mm(features, features.t())

    selected = torch.empty(keep_count, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(keep_count):
        if i == 0:
            if dist_matrix.shape[0] == 1:
                scores = dist_matrix[0]
            else:
                scores = torch.topk(dist_matrix, 2, dim=0, largest=False).values[1, :]
        else:
            selected_dists = torch.index_select(dist_matrix, 0, selected[:i])
            scores = torch.min(selected_dists, dim=0).values
            scores[selected[:i]] = -float("inf")
        selected[i] = torch.argmax(scores)
    return selected


def _diversity_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    return _divprune_select_tokens(features, keep).sort().values


def _zoo_token_sensitivity(
    visual_feature_vectors: torch.Tensor,
    decoder_layer,
    num_refine: int,
    noise_scale: float,
) -> torch.Tensor:
    num_refine = max(int(num_refine), 1)
    noise_scale = max(float(noise_scale), 1e-6)
    num_tokens, hidden_size = visual_feature_vectors.shape
    dtype = visual_feature_vectors.dtype
    device = visual_feature_vectors.device

    directions = torch.randn(num_refine, hidden_size, device=device, dtype=dtype)
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-12)
    directions = directions.unsqueeze(1).expand(-1, num_tokens, -1)
    plus = visual_feature_vectors.unsqueeze(0) + noise_scale * directions
    minus = visual_feature_vectors.unsqueeze(0) - noise_scale * directions

    flat_plus = plus.reshape(num_refine * num_tokens, hidden_size)
    flat_minus = minus.reshape(num_refine * num_tokens, hidden_size)
    proj_plus = decoder_layer.self_attn.v_proj(decoder_layer.input_layernorm(flat_plus))
    proj_minus = decoder_layer.self_attn.v_proj(decoder_layer.input_layernorm(flat_minus))
    coeff = (proj_plus - proj_minus).reshape(num_refine, num_tokens, -1).float().norm(dim=-1) / (2 * noise_scale)
    return coeff.mean(dim=0)


def _zoo_select_tokens(
    visual_feature_vectors: torch.Tensor,
    importance_scores: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    keep_count = min(max(int(keep_count), 1), int(visual_feature_vectors.shape[0]))
    features = torch.nn.functional.normalize(visual_feature_vectors.float(), dim=-1)
    dist_matrix = 1.0 - torch.mm(features, features.t())

    if importance_scores.max() > importance_scores.min():
        sens_weight = (importance_scores - importance_scores.min()) / (
            importance_scores.max() - importance_scores.min() + 1e-8
        )
    else:
        sens_weight = torch.ones_like(importance_scores, dtype=torch.float32)
    sens_weight = sens_weight.float()

    selected = torch.empty(keep_count, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(keep_count):
        if i == 0:
            scores = sens_weight
        else:
            selected_dists = torch.index_select(dist_matrix, 0, selected[:i])
            scores = torch.min(selected_dists, dim=0).values * sens_weight
            scores[selected[:i]] = -float("inf")
        selected[i] = torch.argmax(scores)
    return selected


def _zoo_indices(features: torch.Tensor, keep: int) -> torch.Tensor:
    importance = _normalize_scores(features.norm(dim=-1))
    return _zoo_select_tokens(features, importance, keep).sort().values


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    denom = scores.max() - scores.min()
    if float(denom.item()) <= 1e-12:
        return torch.zeros_like(scores)
    return (scores - scores.min()) / denom


def _eos_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(item) for item in eos)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("</s>", "<|im_end|>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
    return ids
