#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    make_causal_mask,
    read_jsonl,
)
from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option
from delta_vision.models.modeling import image_token_id, load_frozen_llava
from delta_vision.models.vision_kv_adapter import (
    HeadwiseSplitVisionKVAdapter,
    LayerwiseVisionKVAdapter,
    SplitVisionKVAdapter,
)


def parse_int_list(spec: str) -> list[int]:
    values = [int(x) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    return values


def _llava_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _vision_tower(model: torch.nn.Module) -> torch.nn.Module:
    llava_model = _llava_model(model)
    if hasattr(llava_model, "vision_tower"):
        return llava_model.vision_tower
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    raise AttributeError("could not locate LLaVA vision tower")


def build_train_inputs(
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    prompt = llava15_prompt(str(row["question"]).strip())
    answer = str(row.get("answer", "")).strip()
    text = f"{prompt} {answer}" if answer else prompt
    with Image.open(row["image"]) as image:
        full = processor(text=text, images=image.convert("RGB"), return_tensors="pt")
        prompt_only = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in full.items()}
    return inputs, int(prompt_only["input_ids"].shape[1])


@torch.no_grad()
def vision_source_states(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    source_layers: list[int],
) -> torch.Tensor:
    out = _vision_tower(model)(pixel_values, output_hidden_states=True, return_dict=True)
    states = []
    for idx in source_layers:
        hidden = out.hidden_states[int(idx)]
        states.append(hidden[:, 1:].float())
    return torch.stack(states, dim=1).detach().clone()


@torch.no_grad()
def vision_source_kv_cache(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    source_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    vision_tower = _vision_tower(model)
    vision_model = vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower
    hidden = vision_model.embeddings(pixel_values)
    hidden = vision_model.pre_layrnorm(hidden)
    layers = vision_model.encoder.layers
    wanted = {int(idx) for idx in source_layers}
    keys_by_layer = {}
    values_by_layer = {}
    for idx, layer in enumerate(layers):
        normed = layer.layer_norm1(hidden)
        if idx in wanted:
            keys_by_layer[idx] = layer.self_attn.k_proj(normed)[:, 1:].float()
            values_by_layer[idx] = layer.self_attn.v_proj(normed)[:, 1:].float()
        layer_out = layer(hidden, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    if len(keys_by_layer) != len(source_layers):
        raise ValueError(f"could not collect all source_layers={source_layers}")
    return (
        torch.stack([keys_by_layer[int(idx)] for idx in source_layers], dim=1).detach().clone(),
        torch.stack([values_by_layer[int(idx)] for idx in source_layers], dim=1).detach().clone(),
    )


def adapter_forward(adapter: torch.nn.Module, source: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(source, tuple):
        return adapter(source[0], source[1])
    return adapter(source)


def validate_kv_shapes(
    source: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    pred_k: torch.Tensor,
    pred_v: torch.Tensor,
    target_k: torch.Tensor,
    target_v: torch.Tensor,
) -> None:
    if pred_k.shape != target_k.shape:
        raise ValueError(f"pred_k/target_k shape mismatch: {tuple(pred_k.shape)} vs {tuple(target_k.shape)}")
    if pred_v.shape != target_v.shape:
        raise ValueError(f"pred_v/target_v shape mismatch: {tuple(pred_v.shape)} vs {tuple(target_v.shape)}")
    source_tokens = source[0].shape[2] if isinstance(source, tuple) else source.shape[2]
    if int(source_tokens) != int(target_k.shape[2]):
        raise ValueError(f"source/target visual token mismatch: source={source_tokens} target={target_k.shape[2]}")


def build_source_features(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    source_layers: list[int],
    source_kind: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if source_kind == "hidden":
        return vision_source_states(model, pixel_values, source_layers)
    if source_kind == "vision_kv":
        return vision_source_kv_cache(model, pixel_values, source_layers)
    raise ValueError(f"unknown source_kind={source_kind}")


@torch.no_grad()
def teacher_rollout_targets(
    model: torch.nn.Module,
    processor: Any,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hidden = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
        vision_feature_layer=getattr(model.config, "vision_feature_layer", None),
        vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", None),
    ).to(dtype=dtype)
    image_id = image_token_id(model, processor)
    text_positions, image_positions, _teacher_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden.shape[1],
        image_id,
    )
    text_positions = text_positions.to(device=hidden.device)
    image_positions = image_positions.to(device=hidden.device)

    layers = get_lm_layers(language_model)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention_mask = make_causal_mask(1, hidden.shape[1], hidden.device, hidden.dtype)
    target_keys = []
    target_values = []
    for layer in layers:
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        image_normed = normed.index_select(1, image_positions)
        image_shape = image_normed.shape[:-1]
        image_hidden_shape = (*image_shape, -1, attn.head_dim)
        target_keys.append(attn.k_proj(image_normed).view(image_hidden_shape).float())
        target_values.append(attn.v_proj(image_normed).view(image_hidden_shape).float())

        residual = hidden
        position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
        attn_out = _llava_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(hidden)

    logits = model.lm_head(get_lm_norm(language_model)(hidden))
    return {
        "logits": logits.index_select(1, text_positions).float().detach().clone(),
        "target_k": torch.stack(target_keys, dim=1).detach().clone(),
        "target_v": torch.stack(target_values, dim=1).detach().clone(),
        "text_positions": text_positions,
        "image_positions": image_positions,
    }


def _llava_attention_output(
    self_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key_states = repeat_kv(key_states, groups)
    value_states = repeat_kv(value_states, groups)
    out = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask[:, :, :, : key_states.shape[-2]],
        dropout_p=0.0,
        is_causal=False,
        scale=self_attn.scaling,
    )
    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self_attn.o_proj(out)


def _apply_rope_single(
    rotary_owner: torch.nn.Module,
    reference: torch.Tensor,
    states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cos, sin = rotary_owner.rotary_emb(reference, position_ids)
    rotated, _ = apply_rotary_pos_emb(states, states, cos, sin)
    return rotated


def _external_visual_attention(
    self_attn: torch.nn.Module,
    normed_text: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_position_ids: torch.Tensor,
    image_k_content: torch.Tensor,
    image_value: torch.Tensor,
    rotary_owner: torch.nn.Module,
) -> torch.Tensor:
    input_shape = normed_text.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)
    query = self_attn.q_proj(normed_text).view(hidden_shape).transpose(1, 2)
    text_key = self_attn.k_proj(normed_text).view(hidden_shape).transpose(1, 2)
    text_value = self_attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)

    query = _apply_rope_single(rotary_owner, normed_text, query, text_position_ids)
    text_key = _apply_rope_single(rotary_owner, normed_text, text_key, text_position_ids)
    image_key = image_k_content.transpose(1, 2)
    image_value = image_value.transpose(1, 2)
    image_key = _apply_rope_single(rotary_owner, normed_text, image_key, image_position_ids)

    key = torch.cat([image_key, text_key], dim=2)
    value = torch.cat([image_value, text_value], dim=2)
    groups = getattr(
        self_attn,
        "num_key_value_groups",
        self_attn.config.num_attention_heads // self_attn.config.num_key_value_heads,
    )
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)

    text_pos = text_position_ids[0]
    image_pos = image_position_ids[0]
    image_allowed = image_pos.view(1, -1) <= text_pos.view(-1, 1)
    text_allowed = text_pos.view(1, -1) <= text_pos.view(-1, 1)
    allowed = torch.cat([image_allowed, text_allowed], dim=1)
    min_value = torch.finfo(normed_text.dtype).min
    mask = torch.zeros((1, 1, text_pos.numel(), allowed.shape[1]), device=normed_text.device, dtype=normed_text.dtype)
    mask = mask.masked_fill(~allowed.view(1, 1, *allowed.shape), min_value)

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=self_attn.scaling,
    )
    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self_attn.o_proj(out)


def student_logits_external_kv(
    model: torch.nn.Module,
    processor: Any,
    language_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    image_positions: torch.Tensor,
    predicted_k: torch.Tensor,
    predicted_v: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_id = image_token_id(model, processor)
    input_ids = inputs["input_ids"]
    text_source_positions = torch.nonzero(input_ids[0] != image_id, as_tuple=False).flatten()
    text_ids = input_ids.index_select(1, text_source_positions)
    text_embeds = _llava_model(model).get_input_embeddings()(text_ids).to(dtype=dtype)

    full_hidden_len = input_ids.shape[1] - 1 + int(image_positions.numel())
    text_positions, _image_positions, _teacher_positions = get_text_and_image_positions(input_ids, full_hidden_len, image_id)
    text_position_ids = text_positions.to(device=text_embeds.device).unsqueeze(0)
    image_position_ids = image_positions.to(device=text_embeds.device).unsqueeze(0)

    hidden = text_embeds
    layers = get_lm_layers(language_model)
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    for layer_idx, layer in enumerate(layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn_out = _external_visual_attention(
            layer.self_attn,
            normed,
            text_position_ids,
            image_position_ids,
            predicted_k[:, layer_idx].to(dtype=dtype),
            predicted_v[:, layer_idx].to(dtype=dtype),
            rotary_owner,
        )
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(hidden)
    logits = model.lm_head(get_lm_norm(language_model)(hidden)).float()
    return logits, text_source_positions


def topk_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int) -> torch.Tensor:
    k = min(int(topk), teacher_logits.shape[-1])
    _values, indices = teacher_logits.topk(k, dim=-1)
    teacher_top = teacher_logits.gather(-1, indices).float()
    student_top = student_logits.gather(-1, indices).float()
    teacher_prob = F.softmax(teacher_top, dim=-1)
    student_log_prob = F.log_softmax(student_top, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = pred.float() - target.float()
    reduce_dims = tuple(range(1, diff.ndim))
    numerator = diff.square().mean(dim=reduce_dims)
    denominator = target.float().square().mean(dim=reduce_dims).clamp_min(eps)
    return (numerator / denominator).mean()


def select_answer_text_indices(
    input_ids: torch.Tensor,
    text_source_positions: torch.Tensor,
    prompt_source_len: int,
    max_answer_positions: int,
) -> torch.Tensor:
    # Causal LM logits at source position t predict token t+1. Supervise the
    # prompt-final position plus subsequent answer positions, not the answer
    # token states themselves.
    start = max(0, int(prompt_source_len) - 1)
    end = start + int(max_answer_positions)
    answer_mask = (text_source_positions >= start) & (text_source_positions < end)
    answer_indices = torch.nonzero(answer_mask, as_tuple=False).flatten()
    if answer_indices.numel() == 0:
        answer_indices = torch.tensor([text_source_positions.numel() - 1], device=text_source_positions.device)
    if answer_indices.numel() > max_answer_positions:
        answer_indices = answer_indices[:max_answer_positions]
    return answer_indices


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    distributed = "LOCAL_RANK" in __import__("os").environ
    if distributed:
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)
    is_main = rank == 0
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, max_samples=args.max_samples)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    layers = get_lm_layers(language_model)
    cfg = language_model.config
    source_layers = parse_int_list(args.source_layers)
    if args.adapter_kind == "auto":
        adapter_kind = "split_headwise" if args.source_kind == "vision_kv" else "mlp"
    else:
        adapter_kind = args.adapter_kind
    if adapter_kind == "split_headwise":
        if args.source_kind != "vision_kv":
            raise ValueError("split_headwise adapter requires --source-kind vision_kv")
        adapter_core = HeadwiseSplitVisionKVAdapter(
            num_layers=len(layers),
            source_layers=source_layers,
            source_dim=int(model.config.vision_config.hidden_size),
            source_num_heads=int(model.config.vision_config.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        ).to(device=device, dtype=torch.float32)
    else:
        adapter_cls = SplitVisionKVAdapter if args.source_kind == "vision_kv" else LayerwiseVisionKVAdapter
        adapter_core = adapter_cls(
            num_layers=len(layers),
            source_layers=source_layers,
            source_dim=int(model.config.vision_config.hidden_size),
            num_key_value_heads=int(cfg.num_key_value_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            bottleneck_dim=args.bottleneck_dim,
            shared_down=not args.per_layer_down,
            identity_down=args.identity_down,
        ).to(device=device, dtype=torch.float32)
    adapter_core.adapter_kind = adapter_kind
    adapter = DDP(adapter_core, device_ids=[device.index]) if distributed else adapter_core
    trainable_params = sum(param.numel() for param in adapter_core.parameters() if param.requires_grad)
    if is_main:
        print(
            json.dumps(
                {
                    "adapter_trainable_params": trainable_params,
                    "adapter_trainable_millions": trainable_params / 1_000_000,
                    "source_layers": source_layers,
                    "source_kind": args.source_kind,
                    "adapter_kind": adapter_kind,
                    "bottleneck_dim": args.bottleneck_dim,
                    "shared_down": not args.per_layer_down,
                    "identity_down": args.identity_down,
                    "world_size": world_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    wandb_run = None
    if is_main and args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.output_dir).name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "adapter_trainable_params": trainable_params,
                "adapter_trainable_millions": trainable_params / 1_000_000,
                "world_size": world_size,
            },
        )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"

    step = 0
    while step < args.max_steps:
        for local_pos in range(rank, len(rows), world_size):
            row = rows[local_pos]
            if step >= args.max_steps:
                break
            inputs, prompt_len = build_train_inputs(processor, row, device)
            with torch.no_grad():
                source_states = build_source_features(model, inputs["pixel_values"], source_layers, args.source_kind)
                targets = teacher_rollout_targets(model, processor, language_model, inputs, dtype)
            pred_k, pred_v = adapter_forward(adapter, source_states)
            validate_kv_shapes(source_states, pred_k, pred_v, targets["target_k"], targets["target_v"])
            kv_mse = (
                F.mse_loss(pred_k.float(), targets["target_k"].float())
                + F.mse_loss(pred_v.float(), targets["target_v"].float())
            )
            kv_loss = (
                normalized_mse_loss(pred_k, targets["target_k"])
                + normalized_mse_loss(pred_v, targets["target_v"])
            )
            student_logits, text_source_positions = student_logits_external_kv(
                model,
                processor,
                language_model,
                inputs,
                targets["image_positions"],
                pred_k,
                pred_v,
                dtype,
            )
            answer_indices = select_answer_text_indices(
                inputs["input_ids"],
                text_source_positions,
                prompt_len,
                args.max_answer_positions,
            )
            teacher_answer_logits = targets["logits"].index_select(1, answer_indices)
            student_answer_logits = student_logits.index_select(1, answer_indices)
            kl = topk_kl_loss(student_answer_logits.reshape(-1, student_answer_logits.shape[-1]), teacher_answer_logits.reshape(-1, teacher_answer_logits.shape[-1]), args.kl_topk)
            loss = args.lambda_kv * kv_loss + args.lambda_kl * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()

            if distributed:
                metrics_tensor = torch.tensor(
                    [loss.detach(), kv_loss.detach(), kl.detach(), kv_mse.detach(), torch.as_tensor(grad_norm, device=device)],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            else:
                metrics_tensor = torch.tensor(
                    [loss.detach(), kv_loss.detach(), kl.detach(), kv_mse.detach(), torch.as_tensor(grad_norm, device=device)],
                    device=device,
                    dtype=torch.float32,
                )
            if is_main and step % args.log_every == 0:
                item = {
                    "step": step,
                    "loss": float(metrics_tensor[0].detach().cpu()),
                    "kv_nmse": float(metrics_tensor[1].detach().cpu()),
                    "kl_top1024": float(metrics_tensor[2].detach().cpu()),
                    "kv_mse": float(metrics_tensor[3].detach().cpu()),
                    "grad_norm": float(metrics_tensor[4].detach().cpu()),
                    "answer_positions": int(answer_indices.numel()),
                    "image": row.get("image", ""),
                }
                print(json.dumps(item, ensure_ascii=False), flush=True)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": item["loss"],
                            "train/kv_nmse": item["kv_nmse"],
                            "train/kv_mse": item["kv_mse"],
                            "train/kl_top1024": item["kl_top1024"],
                            "train/grad_norm": item["grad_norm"],
                            "train/answer_positions": item["answer_positions"],
                            "train/step": step,
                        },
                        step=step,
                    )
            if is_main and args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"step{step}.pt", adapter_core, args)
            step += 1
    if is_main:
        save_checkpoint(out_dir / f"step{step}.pt", adapter_core, args)
        save_checkpoint(out_dir / "latest.pt", adapter_core, args)
    if distributed:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


def save_checkpoint(
    path: Path,
    adapter: HeadwiseSplitVisionKVAdapter | LayerwiseVisionKVAdapter | SplitVisionKVAdapter,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "source_layers": adapter.source_layers,
            "source_kind": getattr(args, "source_kind", "hidden"),
            "adapter_kind": getattr(adapter, "adapter_kind", getattr(args, "adapter_kind", "mlp")),
            "args": vars(args),
        },
        path,
    )
    print(f"saved {path}", flush=True)


def load_adapter_checkpoint(
    path: str,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
) -> HeadwiseSplitVisionKVAdapter | LayerwiseVisionKVAdapter | SplitVisionKVAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    source_layers = [int(x) for x in ckpt.get("source_layers", parse_int_list(ckpt_args.get("source_layers", "22,23,24")))]
    source_kind = str(ckpt.get("source_kind", ckpt_args.get("source_kind", "hidden")))
    adapter_kind = str(ckpt.get("adapter_kind", ckpt_args.get("adapter_kind", "auto")))
    if adapter_kind == "auto":
        state_dict = ckpt["state_dict"]
        if any(key.startswith("proj_k.") or key.startswith("proj_v.") for key in state_dict):
            adapter_kind = "split_headwise"
        elif any(key.startswith("down_k.") or key.startswith("up_k.") for key in state_dict):
            adapter_kind = "mlp"
        else:
            adapter_kind = "split_headwise" if source_kind == "vision_kv" else "mlp"
    cfg = language_model.config
    if adapter_kind == "split_headwise":
        adapter = HeadwiseSplitVisionKVAdapter(
            num_layers=len(get_lm_layers(language_model)),
            source_layers=source_layers,
            source_dim=int(model.config.vision_config.hidden_size),
            source_num_heads=int(model.config.vision_config.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        ).to(device=device, dtype=torch.float32)
    else:
        adapter_cls = SplitVisionKVAdapter if source_kind == "vision_kv" else LayerwiseVisionKVAdapter
        adapter = adapter_cls(
            num_layers=len(get_lm_layers(language_model)),
            source_layers=source_layers,
            source_dim=int(model.config.vision_config.hidden_size),
            num_key_value_heads=int(cfg.num_key_value_heads),
            head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            bottleneck_dim=int(ckpt_args.get("bottleneck_dim", 32)),
            shared_down=not bool(ckpt_args.get("per_layer_down", False)),
            identity_down=bool(ckpt_args.get("identity_down", False)),
        ).to(device=device, dtype=torch.float32)
    adapter.source_kind = source_kind
    adapter.adapter_kind = adapter_kind
    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval()
    return adapter


@torch.inference_mode()
def eval_mmstar(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    adapter = load_adapter_checkpoint(args.checkpoint, model, language_model, device)
    option_ids = option_token_id_lists(processor.tokenizer)
    stats = {
        "llava": {"scored": 0, "correct": 0},
        "adapter_kv": {"scored": 0, "correct": 0, "agree": 0, "ret": 0, "kl": 0.0},
    }
    predictions = []
    for idx, row in enumerate(rows):
        gold = str(row["answer"]).strip().upper()[:1]
        inputs, _prompt_len = build_train_inputs(processor, {"image": row["image"], "question": row["question"], "answer": ""}, device)
        source_states = build_source_features(
            model,
            inputs["pixel_values"],
            adapter.source_layers,
            str(getattr(adapter, "source_kind", "hidden")),
        )
        targets = teacher_rollout_targets(model, processor, language_model, inputs, dtype)
        pred_k, pred_v = adapter_forward(adapter, source_states)
        validate_kv_shapes(source_states, pred_k, pred_v, targets["target_k"], targets["target_v"])
        student_logits, _text_source_positions = student_logits_external_kv(
            model,
            processor,
            language_model,
            inputs,
            targets["image_positions"],
            pred_k,
            pred_v,
            dtype,
        )
        teacher_logits = targets["logits"][0, -1]
        pred_logits = student_logits[0, -1]
        teacher_pred = predict_option(teacher_logits, option_ids)
        pred = predict_option(pred_logits, option_ids)
        teacher_correct = teacher_pred == gold
        teacher_dist = option_distribution(teacher_logits, option_ids)
        pred_dist = option_distribution(pred_logits, option_ids)
        stats["llava"]["scored"] += 1
        stats["llava"]["correct"] += int(teacher_correct)
        stats["adapter_kv"]["scored"] += 1
        stats["adapter_kv"]["correct"] += int(pred == gold)
        stats["adapter_kv"]["agree"] += int(pred == teacher_pred)
        stats["adapter_kv"]["ret"] += int(teacher_correct and pred == gold)
        stats["adapter_kv"]["kl"] += float(F.kl_div(pred_dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item())
        predictions.append(
            {
                "index": row.get("index", args.start_index + idx),
                "gold": gold,
                "llava": teacher_pred,
                "adapter_kv": pred,
            }
        )
        if (idx + 1) % args.log_every == 0:
            print(f"evaluated {idx + 1}/{len(rows)}", flush=True)
    teacher_n = max(1, stats["llava"]["scored"])
    adapter_n = max(1, stats["adapter_kv"]["scored"])
    teacher_correct = max(1, stats["llava"]["correct"])
    result = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "results": [
            {
                "setting": "llava",
                "scored": stats["llava"]["scored"],
                "correct": stats["llava"]["correct"],
                "accuracy": stats["llava"]["correct"] / teacher_n,
            },
            {
                "setting": "adapter_kv",
                "scored": stats["adapter_kv"]["scored"],
                "correct": stats["adapter_kv"]["correct"],
                "accuracy": stats["adapter_kv"]["correct"] / adapter_n,
                "llava_agreement": stats["adapter_kv"]["agree"] / adapter_n,
                "llava_correct_retention": stats["adapter_kv"]["ret"] / teacher_correct,
                "output_kl_to_llava": stats["adapter_kv"]["kl"] / adapter_n,
            },
        ],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions_jsonl:
        pred_out = Path(args.predictions_jsonl)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LLaVA vision-source -> external visual-KV adapter.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", default="data/pixmo_ama_50k.jsonl")
    train_parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--source-layers", default="22,23,24")
    train_parser.add_argument("--source-kind", default="hidden", choices=["hidden", "vision_kv"])
    train_parser.add_argument("--adapter-kind", default="auto", choices=["auto", "mlp", "split_headwise"])
    train_parser.add_argument("--bottleneck-dim", type=int, default=32)
    train_parser.add_argument("--per-layer-down", action="store_true")
    train_parser.add_argument("--identity-down", action="store_true")
    train_parser.add_argument("--max-samples", type=int, default=1000)
    train_parser.add_argument("--max-steps", type=int, default=100)
    train_parser.add_argument("--max-answer-positions", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--lambda-kv", type=float, default=0.0)
    train_parser.add_argument("--lambda-kl", type=float, default=1.0)
    train_parser.add_argument("--kl-topk", type=int, default=1024)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    train_parser.add_argument("--attn-implementation", default="eager")
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--dist-backend", default="nccl")
    train_parser.add_argument("--seed", type=int, default=1234)
    train_parser.add_argument("--log-every", type=int, default=1)
    train_parser.add_argument("--save-every", type=int, default=1000)
    train_parser.add_argument("--shuffle", action="store_true")
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--wandb-project", default="delta-vision")
    train_parser.add_argument("--wandb-run-name", default="")
    train_parser.add_argument("--wandb-mode", default="online")

    eval_parser = sub.add_parser("eval-mmstar")
    eval_parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    eval_parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output-json", required=True)
    eval_parser.add_argument("--predictions-jsonl", default="")
    eval_parser.add_argument("--start-index", type=int, default=0)
    eval_parser.add_argument("--max-samples", type=int, default=1000)
    eval_parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    eval_parser.add_argument("--attn-implementation", default="eager")
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval-mmstar":
        eval_mmstar(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
