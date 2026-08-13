#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.evaluation.metrics import OPTIONS, option_scores, option_token_id_lists
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_visual_attention_mass_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prompt,
    qwen3vl_prefix_visual_memory_by_layer,
    qwen3vl_text_attention_heads,
    qwen3vl_text_attention_output,
    qwen3vl_visual_memory_by_layer,
    run_qwen3vl_full_layer_with_text_delta,
    run_qwen3vl_layer_text_from_attention_output,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)
from delta_vision.models.sidecar import DeltaVisionModule
from delta_vision.cli.qwen.train_qwen3vl_sidecar import (
    attach_qwen_anchor_trainable_visual_kv,
    attach_qwen_first_layer_film_visual_kv,
    attach_qwen_trainable_native_qkv,
    qwen_native_sidecar_query,
    qwen_native_visual_kv,
    qwen_anchor_trainable_visual_kv,
    qwen_first_layer_film_visual_kv,
    qwen_trainable_native_sidecar_query,
    qwen_trainable_native_visual_kv,
    parse_int_set,
    residual_from_cached_coeff,
)
from delta_vision.runtime.qwen_analytic_sidecar import QwenNativeAttentionSidecar, run_analytic_qwen3vl_sidecar_only_rollout


CHOICE_RE = re.compile(r"(?m)(?:^|\b)([A-D])(?:[.)：:]|\\s*:)")


def row_image_path(row: dict[str, Any]) -> Path:
    path = Path(str(row["image"]))
    image_root = os.environ.get("DELTA_VISION_IMAGE_ROOT", "")
    if image_root and not path.is_absolute():
        path = Path(image_root) / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL delta-vision sidecar/hybrid prompt logits.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--sidecar-backend",
        choices=("learned", "analytic_attention", "native_attention"),
        default="learned",
        help=(
            "`learned` uses a trained predictor. `native_attention`/`analytic_attention` "
            "are oracle teacher paths that recompute Qwen's native visual attention effect; "
            "they are not learned prediction backends."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=0)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--reader-activation", choices=("gelu", "silu", "swiglu", "situ_glu"), default="gelu")
    parser.add_argument("--layer-adapter-rank", type=int, default=128)
    parser.add_argument("--reader-mode", choices=("cross_attention", "pooled"), default="cross_attention")
    parser.add_argument("--latent-tokens", type=int, default=64)
    parser.add_argument("--block-corrector-groups", default="")
    parser.add_argument("--block-corrector-dim", type=int, default=0)
    parser.add_argument("--shared-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-mode",
        choices=(
            "residual",
            "residual_full",
            "factorized_lowrank",
            "factorized_full",
            "factorized_native_o",
            "factorized_native_head_o",
            "factorized_native_head_o_pure",
            "factorized_native_head_o_residual",
            "native_cross_attention",
        ),
        default="residual",
    )
    parser.add_argument("--visual-memory-mode", choices=("v0", "vdeep", "vcum", "vprefix"), default="v0")
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--use-rope", action="store_true", default=False)
    parser.add_argument("--sidecar-query-source", choices=("sidecar", "qwen_native", "qwen_trainable_native"), default="sidecar")
    parser.add_argument(
        "--sidecar-visual-kv-source",
        choices=(
            "sidecar",
            "qwen_native",
            "qwen_first_layer",
            "qwen_first_layer_film",
            "qwen_trainable_native",
            "qwen_anchor_trainable",
        ),
        default="sidecar",
    )
    parser.add_argument(
        "--qwen-anchor-layers",
        default="4,16,30",
        help="Comma-separated Qwen layer ids used by sidecar_visual_kv_source=qwen_anchor_trainable.",
    )
    parser.add_argument(
        "--sidecar-active-layers",
        default="",
        help="Comma-separated layers where Sidecar reads visual memory; inactive layers reuse latest active coeff.",
    )
    parser.add_argument(
        "--factorized-mass-mode",
        choices=("learned", "analytic", "oracle", "fixed"),
        default="learned",
    )
    parser.add_argument("--fixed-visual-mass", type=float, default=0.12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def build_prompt(processor: Any, row: dict[str, Any], benchmark: str) -> str:
    question = str(row["question"]).strip()
    if benchmark == "mmstar":
        question = f"{question}\nAnswer directly with only the letter of the correct option."
    else:
        question = f"{question}\nAnswer directly with the final answer only."
    return qwen3vl_prompt(processor, question)


def candidate_kind(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "mmstar":
        return "abcd"
    question = str(row.get("question", ""))
    answer = str(row.get("answer", "")).strip()
    if CHOICE_RE.search(question) or answer.upper()[:1] in OPTIONS:
        return "abcd"
    if answer.lower() in {"yes", "no"}:
        return "yesno"
    return "skip"


def candidate_ids(tokenizer: Any, kind: str) -> dict[str, list[int]]:
    if kind == "abcd":
        return option_token_id_lists(tokenizer)
    if kind == "yesno":
        out: dict[str, list[int]] = {}
        for key, variants in {
            "Yes": ("Yes", " Yes", "yes", " yes"),
            "No": ("No", " No", "no", " no"),
        }.items():
            ids = set()
            for text in variants:
                encoded = tokenizer(text, add_special_tokens=False).input_ids
                if encoded:
                    ids.add(int(encoded[-1]))
            out[key] = sorted(ids)
        return out
    raise ValueError(f"unsupported candidate kind: {kind}")


def normalize_gold(row: dict[str, Any], kind: str) -> str:
    answer = str(row.get("answer", "")).strip()
    if kind == "abcd":
        return answer.upper()[:1]
    if kind == "yesno":
        return "Yes" if answer.lower().startswith("yes") else "No"
    return answer


def score_candidates(logits: torch.Tensor, ids: dict[str, list[int]]) -> torch.Tensor:
    if set(ids.keys()) == set(OPTIONS):
        return option_scores(logits, ids)
    scores = []
    for key in ids:
        idx = torch.tensor(ids[key], device=logits.device, dtype=torch.long)
        scores.append(logits.index_select(0, idx).max())
    return torch.stack(scores)


def predict(logits: torch.Tensor, ids: dict[str, list[int]]) -> str:
    keys = list(ids)
    scores = score_candidates(logits.float(), ids)
    return keys[int(scores.argmax().item())]


def distribution(logits: torch.Tensor, ids: dict[str, list[int]]) -> torch.Tensor:
    return F.softmax(score_candidates(logits.float(), ids), dim=0)


def load_sidecar(
    args: argparse.Namespace,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> DeltaVisionModule:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for name in (
        "rank",
        "sidecar_dim",
        "num_heads",
        "state_tokens",
        "reader_mlp_ratio",
        "reader_activation",
        "layer_adapter_rank",
        "shared_basis",
        "output_mode",
        "corrector_layers",
        "corrector_dim",
        "block_corrector_groups",
        "block_corrector_dim",
        "use_rope",
        "sidecar_query_source",
        "sidecar_visual_kv_source",
        "qwen_anchor_layers",
        "sidecar_active_layers",
        "factorized_mass_mode",
        "fixed_visual_mass",
        "layer_condition_mode",
        "reader_mode",
        "latent_tokens",
        "visual_transform_mode",
        "visual_transform_rank",
        "visual_transform_activation",
    ):
        if name in checkpoint_args:
            setattr(args, name, checkpoint_args[name])
    if args.visual_memory_mode == "v0" and checkpoint_args.get("visual_memory_mode") in {"v0", "vdeep", "vcum", "vprefix"}:
        args.visual_memory_mode = checkpoint_args["visual_memory_mode"]
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if (
        args.output_mode == "factorized_native_head_o"
        and "coeff_head.weight" not in state_dict
        and "basis" not in state_dict
    ):
        args.output_mode = "factorized_native_head_o_pure"
    elif args.output_mode == "factorized_native_head_o":
        args.output_mode = "factorized_native_head_o_residual"
    hidden_size = int(language_model.config.hidden_size)
    num_layers = len(language_model.layers)
    sidecar = DeltaVisionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rank=args.rank,
        sidecar_dim=args.sidecar_dim,
        num_heads=args.num_heads,
        state_tokens=args.state_tokens,
        dropout=0.0,
        gate_init=1.0,
        basis=None,
        train_basis=args.output_mode != "residual_full",
        reader_mlp_ratio=args.reader_mlp_ratio,
        reader_activation=args.reader_activation,
        layer_adapter_rank=args.layer_adapter_rank,
        reader_concat_query=True,
        normalize_basis_rows=True,
        shared_basis=args.shared_basis,
        output_mode=args.output_mode,
        corrector_layers=getattr(args, "corrector_layers", ""),
        corrector_dim=getattr(args, "corrector_dim", 0),
        block_corrector_groups=getattr(args, "block_corrector_groups", ""),
        block_corrector_dim=getattr(args, "block_corrector_dim", 0),
        use_rope=getattr(args, "use_rope", False),
        layer_condition_mode=getattr(args, "layer_condition_mode", "query"),
        reader_mode=getattr(args, "reader_mode", "cross_attention"),
        latent_tokens=getattr(args, "latent_tokens", 64),
        visual_transform_mode=getattr(args, "visual_transform_mode", "none"),
        visual_transform_rank=getattr(args, "visual_transform_rank", 128),
        visual_transform_activation=getattr(args, "visual_transform_activation", "gelu"),
    ).to(device=device, dtype=dtype)
    if args.sidecar_query_source == "qwen_trainable_native" or args.sidecar_visual_kv_source == "qwen_trainable_native":
        attach_qwen_trainable_native_qkv(
            sidecar,
            language_model,
            train_query=args.sidecar_query_source == "qwen_trainable_native",
            train_visual_kv=args.sidecar_visual_kv_source == "qwen_trainable_native",
        )
    if args.sidecar_visual_kv_source == "qwen_anchor_trainable":
        attach_qwen_anchor_trainable_visual_kv(sidecar, language_model, args.qwen_anchor_layers)
    if args.sidecar_visual_kv_source == "qwen_first_layer_film":
        attach_qwen_first_layer_film_visual_kv(sidecar, language_model)
    missing, unexpected = sidecar.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"checkpoint load missing={missing} unexpected={unexpected}", flush=True)
    sidecar.eval()
    for param in sidecar.parameters():
        param.requires_grad_(False)
    sidecar.runtime_fold_output_basis = True
    sidecar.prepare_inference_cache(device, dtype)
    return sidecar


def load_native_sidecar(checkpoint_path: str, num_layers: int, device: torch.device, dtype: torch.dtype) -> QwenNativeAttentionSidecar:
    sidecar = QwenNativeAttentionSidecar(num_layers=num_layers, gate_init=1.0, train_gates=False).to(device=device, dtype=dtype)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        sidecar.load_state_dict(state_dict, strict=True)
        sidecar.to(device=device, dtype=dtype)
    sidecar.eval()
    return sidecar


def qwen3vl_visual_position_ids(
    full_position_ids: torch.Tensor,
    image_positions: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    visual_position_ids = torch.zeros(
        3,
        image_positions.shape[0],
        image_positions.shape[1],
        device=image_positions.device,
        dtype=full_position_ids.dtype,
    )
    valid = image_mask.bool()
    batch_idx = torch.arange(image_positions.shape[0], device=image_positions.device).unsqueeze(1).expand_as(image_positions)
    for dim_idx in range(3):
        dim_positions = full_position_ids[dim_idx]
        visual_position_ids[dim_idx][valid] = dim_positions[batch_idx[valid], image_positions[valid].long()]
    return visual_position_ids


def build_full_from_text_and_memory(
    reference_full: torch.Tensor,
    text_positions: torch.Tensor,
    text_hidden: torch.Tensor,
    text_mask: torch.Tensor,
    image_positions: torch.Tensor,
    image_memory: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    full = scatter_batched_positions(reference_full, text_positions, text_hidden, text_mask)
    return scatter_batched_positions(full, image_positions, image_memory, image_mask)


def factorized_visual_mass_override(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    layer_idx: int,
    text_hidden: torch.Tensor,
    reference_full: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    text_mask: torch.Tensor,
    image_mask: torch.Tensor,
    full_mask: torch.Tensor,
    image_memory: torch.Tensor,
    mode: str,
    fixed_visual_mass: float,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not sidecar.output_mode.startswith("factorized") or mode == "learned":
        return None
    if mode == "fixed":
        return torch.full(
            (text_hidden.shape[0], text_hidden.shape[1], 1),
            float(fixed_visual_mass),
            device=text_hidden.device,
            dtype=dtype,
        )
    if mode not in {"analytic", "oracle"}:
        raise ValueError(f"unsupported factorized mass mode: {mode}")
    full_state = build_full_from_text_and_memory(
        reference_full.to(dtype=dtype),
        text_positions,
        text_hidden,
        text_mask,
        image_positions,
        image_memory.to(dtype=dtype),
        image_mask,
    )
    return compute_qwen3vl_visual_attention_mass_batched(
        language_model,
        layer_idx,
        full_state,
        full_position_ids,
        text_positions,
        image_positions,
        full_mask,
        text_mask,
        image_mask,
        reduce_heads=(
            "none"
            if sidecar.output_mode
            in {
                "factorized_native_head_o",
                "factorized_native_head_o_pure",
                "factorized_native_head_o_residual",
            }
            else "mean"
        ),
    )


def sidecar_rope_visual_kv(
    sidecar: DeltaVisionModule,
    language_model: torch.nn.Module,
    visual_memory: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    image_positions: torch.Tensor,
    image_mask: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor] | None]:
    if not sidecar.use_rope:
        return sidecar.prepare_visual_kv(visual_memory.to(dtype=dtype), ~image_mask), None
    visual_position_ids = qwen3vl_visual_position_ids(full_position_ids, image_positions, image_mask)
    visual_pos_emb = language_model.rotary_emb(visual_memory.to(dtype=dtype), visual_position_ids)
    text_pos_emb = language_model.rotary_emb(visual_memory.to(dtype=dtype), text_position_ids)
    return (
        sidecar.prepare_visual_kv(visual_memory.to(dtype=dtype), ~image_mask, position_embeddings=visual_pos_emb),
        text_pos_emb,
    )


@torch.inference_mode()
def qwen_logits(processor: Any, model: torch.nn.Module, row: dict[str, Any], benchmark: str, device: torch.device) -> torch.Tensor:
    with Image.open(row_image_path(row)) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    out = model(**inputs, return_dict=True, use_cache=False)
    text_pos, _, _, text_mask, _, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        model.model.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            inputs_embeds=model.model.get_input_embeddings()(inputs["input_ids"]),
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        ),
    )
    last_text = int(text_pos[0, int(text_mask[0].sum().item()) - 1].item())
    return out.logits[0, last_text]


@torch.inference_mode()
def sidecar_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
    sidecar_scale: float,
    visual_memory_mode: str,
    sidecar_query_source: str = "sidecar",
    sidecar_visual_kv_source: str = "sidecar",
    factorized_mass_mode: str = "learned",
    fixed_visual_mass: float = 0.12,
    sidecar_active_layers_spec: str = "",
) -> torch.Tensor:
    with Image.open(row_image_path(row)) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    h = gather_batched_positions(hidden0, text_pos, text_mask).to(dtype=dtype)
    initial_text_hidden = h
    if visual_memory_mode == "vprefix":
        visual_memories = qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            hidden0.to(dtype=dtype),
            position_ids,
            inputs["attention_mask"],
            image_pos,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
    else:
        visual_memories = qwen3vl_visual_memory_by_layer(
            hidden0.to(dtype=dtype),
            image_pos,
            image_mask,
            deepstack_visual_embeds,
            visual_memory_mode,
            len(language_model.layers),
        )
    shared_visual_kv = None
    shared_text_pos_emb = None
    visual_position_ids = qwen3vl_visual_position_ids(position_ids, image_pos, image_mask)
    if (
        visual_memory_mode in {"v0", "vdeep"}
        and sidecar_visual_kv_source == "sidecar"
        and getattr(sidecar, "visual_transform_mode", "none") == "none"
    ):
        shared_visual_kv, shared_text_pos_emb = sidecar_rope_visual_kv(
            sidecar,
            language_model,
            visual_memories[0],
            position_ids,
            text_position_ids,
            image_pos,
            image_mask,
            dtype,
        )
    elif sidecar_visual_kv_source in {"qwen_first_layer", "qwen_first_layer_film"}:
        shared_visual_kv = qwen_native_visual_kv(
            language_model,
            0,
            visual_memories[0].to(dtype=dtype),
            visual_position_ids,
            ~image_mask,
        )
        shared_text_pos_emb = None
    sidecar_state = sidecar.initial_state(visual_memories[0].to(dtype=dtype), ~image_mask) if sidecar.state_tokens > 0 else None
    visual_memory_state = sidecar.initial_visual_memory_state(visual_memories[0].to(dtype=dtype))
    text_padding_mask = ~text_mask
    sidecar_active_layers = (
        parse_int_set(sidecar_active_layers_spec, len(language_model.layers))
        if str(sidecar_active_layers_spec).strip()
        else set(range(len(language_model.layers)))
    )
    if sidecar_active_layers != set(range(len(language_model.layers))) and sidecar.output_mode != "residual":
        raise ValueError("--sidecar-active-layers currently requires residual output checkpoints")
    cached_anchor_coeff: torch.Tensor | None = None
    for layer_idx in range(len(language_model.layers)):
        layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
        visual_memory_for_layer, visual_memory_state = sidecar.visual_memory_for_layer(
            visual_memories[layer_idx].to(dtype=dtype),
            layer_idx,
            hidden_states=h,
            initial_hidden_states=initial_text_hidden,
            current_visual_memory=visual_memory_state,
            vision_padding_mask=~image_mask,
        )
        if layer_idx not in sidecar_active_layers:
            delta = residual_from_cached_coeff(sidecar, cached_anchor_coeff, layer_tensor, h)
            delta = delta * float(sidecar_scale)
            h = run_qwen3vl_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                padding_mask=text_padding_mask,
            )
            continue
        visual_kv = shared_visual_kv
        text_pos_emb = shared_text_pos_emb
        visual_pos_emb = None
        if getattr(sidecar, "reader_mode", "cross_attention") == "pooled":
            visual_kv = None
            text_pos_emb = None
        elif visual_kv is None:
            if sidecar_visual_kv_source == "qwen_native":
                visual_kv = qwen_native_visual_kv(
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_trainable_native":
                visual_kv = qwen_trainable_native_visual_kv(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_anchor_trainable":
                visual_kv = qwen_anchor_trainable_visual_kv(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_first_layer":
                visual_kv = shared_visual_kv
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_first_layer_film":
                if shared_visual_kv is None:
                    raise RuntimeError("qwen_first_layer_film requires shared layer-0 visual K/V")
                visual_kv = qwen_first_layer_film_visual_kv(sidecar, shared_visual_kv, layer_idx)
                text_pos_emb = None
            else:
                if getattr(sidecar, "use_rope", False):
                    visual_pos_emb = language_model.rotary_emb(visual_memory_for_layer, visual_position_ids)
                    text_pos_emb = language_model.rotary_emb(visual_memory_for_layer, text_position_ids)
                else:
                    visual_kv, text_pos_emb = sidecar_rope_visual_kv(
                        sidecar,
                        language_model,
                        visual_memory_for_layer,
                        position_ids,
                        text_position_ids,
                        image_pos,
                        image_mask,
                        dtype,
                    )
        query_override = None
        if sidecar_query_source == "qwen_native":
            query_override = qwen_native_sidecar_query(language_model, layer_idx, h, text_position_ids)
        elif sidecar_query_source == "qwen_trainable_native":
            query_override = qwen_trainable_native_sidecar_query(
                sidecar,
                language_model,
                layer_idx,
                h,
                text_position_ids,
            )
        text_attention = None
        if sidecar.output_mode.startswith("factorized"):
            text_attention = qwen3vl_text_attention_output(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
        text_attention_heads = None
        if sidecar.output_mode in {
            "factorized_native_head_o",
            "factorized_native_head_o_pure",
            "factorized_native_head_o_residual",
        }:
            text_attention_heads = qwen3vl_text_attention_heads(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                padding_mask=text_padding_mask,
            )
        visual_mass = factorized_visual_mass_override(
            sidecar,
            language_model,
            layer_idx,
            h,
            hidden0,
            position_ids,
            text_pos,
            image_pos,
            text_mask,
            image_mask,
            full_mask,
            visual_memory_for_layer,
            factorized_mass_mode,
            fixed_visual_mass,
            dtype,
        )
        call_vision_states = None if visual_kv is not None else visual_memory_for_layer
        call_vision_mask = None if visual_kv is not None else ~image_mask
        if getattr(sidecar, "visual_transform_mode", "none") == "latent_compressor":
            call_vision_mask = None
        if sidecar.state_tokens > 0:
            delta, sidecar_state, coeff = sidecar(
                h,
                call_vision_states,
                layer_tensor,
                sidecar_state=sidecar_state,
                visual_kv=visual_kv,
                vision_padding_mask=call_vision_mask,
                text_attention=text_attention,
                visual_mass=visual_mass,
                output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                text_attention_heads=text_attention_heads,
                position_embeddings=text_pos_emb,
                visual_position_embeddings=visual_pos_emb,
                query_states=query_override,
                initial_hidden_states=initial_text_hidden,
                return_state=True,
                return_coefficients=True,
            )
        else:
            delta, coeff = sidecar(
                h,
                call_vision_states,
                layer_tensor,
                visual_kv=visual_kv,
                vision_padding_mask=call_vision_mask,
                text_attention=text_attention,
                visual_mass=visual_mass,
                output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                text_attention_heads=text_attention_heads,
                position_embeddings=text_pos_emb,
                visual_position_embeddings=visual_pos_emb,
                query_states=query_override,
                initial_hidden_states=initial_text_hidden,
                return_coefficients=True,
            )
        cached_anchor_coeff = coeff
        delta = delta * float(sidecar_scale)
        if text_attention is None:
            h = run_qwen3vl_layer_text_with_attention_delta(
                language_model,
                layer_idx,
                h,
                text_position_ids,
                delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
                padding_mask=text_padding_mask,
            )
        else:
            h = run_qwen3vl_layer_text_from_attention_output(
                language_model,
                layer_idx,
                h,
                text_attention,
                delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
            )
    logits = model.lm_head(language_model.norm(h))
    return logits[0, int(text_mask[0].sum().item()) - 1]


@torch.inference_mode()
def analytic_sidecar_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
    visual_memory_mode: str,
    native_sidecar: QwenNativeAttentionSidecar | None = None,
) -> torch.Tensor:
    with Image.open(row_image_path(row)) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    h = run_analytic_qwen3vl_sidecar_only_rollout(
        language_model,
        hidden0,
        position_ids,
        inputs["attention_mask"],
        text_pos,
        image_pos,
        text_position_ids,
        text_mask,
        image_mask,
        full_mask,
        visual_pos_masks,
        deepstack_visual_embeds,
        visual_memory_mode,
        dtype,
        native_sidecar=native_sidecar,
    )
    logits = model.lm_head(language_model.norm(h))
    return logits[0, int(text_mask[0].sum().item()) - 1]


@torch.inference_mode()
def no_visual_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with Image.open(row_image_path(row)) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0, position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
    text_pos, _, text_position_ids, text_mask, _, _ = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    h = gather_batched_positions(hidden0, text_pos, text_mask).to(dtype=dtype)
    text_padding_mask = ~text_mask
    for layer_idx in range(len(language_model.layers)):
        h = run_qwen3vl_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            text_position_ids,
            attention_delta=None,
            padding_mask=text_padding_mask,
        )
    logits = model.lm_head(language_model.norm(h))
    return logits[0, int(text_mask[0].sum().item()) - 1]


@torch.inference_mode()
def hybrid_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    sidecar: DeltaVisionModule,
    row: dict[str, Any],
    benchmark: str,
    device: torch.device,
    dtype: torch.dtype,
    sidecar_scale: float,
    visual_memory_mode: str,
    sidecar_query_source: str = "sidecar",
    sidecar_visual_kv_source: str = "sidecar",
    factorized_mass_mode: str = "learned",
    fixed_visual_mass: float = 0.12,
    sidecar_active_layers_spec: str = "",
) -> torch.Tensor:
    with Image.open(row_image_path(row)) as image:
        inputs = processor(text=build_prompt(processor, row, benchmark), images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    h, position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
    text_pos, image_pos, text_position_ids, text_mask, image_mask, full_mask = get_qwen3vl_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        position_ids,
    )
    if visual_memory_mode == "vprefix":
        visual_memories = qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            h.to(dtype=dtype),
            position_ids,
            inputs["attention_mask"],
            image_pos,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
    else:
        visual_memories = qwen3vl_visual_memory_by_layer(
            h.to(dtype=dtype),
            image_pos,
            image_mask,
            deepstack_visual_embeds,
            visual_memory_mode,
            len(language_model.layers),
        )
    shared_visual_kv = None
    shared_text_pos_emb = None
    visual_position_ids = qwen3vl_visual_position_ids(position_ids, image_pos, image_mask)
    if (
        visual_memory_mode in {"v0", "vdeep"}
        and sidecar_visual_kv_source == "sidecar"
        and getattr(sidecar, "visual_transform_mode", "none") == "none"
    ):
        shared_visual_kv, shared_text_pos_emb = sidecar_rope_visual_kv(
            sidecar,
            language_model,
            visual_memories[0],
            position_ids,
            text_position_ids,
            image_pos,
            image_mask,
            dtype,
        )
    elif sidecar_visual_kv_source in {"qwen_first_layer", "qwen_first_layer_film"}:
        shared_visual_kv = qwen_native_visual_kv(
            language_model,
            0,
            visual_memories[0].to(dtype=dtype),
            visual_position_ids,
            ~image_mask,
        )
        shared_text_pos_emb = None
    sidecar_state = sidecar.initial_state(visual_memories[0].to(dtype=dtype), ~image_mask) if sidecar.state_tokens > 0 else None
    visual_memory_state = sidecar.initial_visual_memory_state(visual_memories[0].to(dtype=dtype))
    attention_mask_2d = inputs.get("attention_mask")
    initial_text_hidden = gather_batched_positions(h, text_pos, text_mask).to(dtype=dtype)
    sidecar_active_layers = (
        parse_int_set(sidecar_active_layers_spec, len(language_model.layers))
        if str(sidecar_active_layers_spec).strip()
        else set(range(len(language_model.layers)))
    )
    if sidecar_active_layers != set(range(len(language_model.layers))) and sidecar.output_mode != "residual":
        raise ValueError("--sidecar-active-layers currently requires residual output checkpoints")
    cached_anchor_coeff: torch.Tensor | None = None
    for layer_idx in range(len(language_model.layers)):
        layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
        text_hidden = gather_batched_positions(h, text_pos, text_mask)
        visual_memory_for_layer, visual_memory_state = sidecar.visual_memory_for_layer(
            visual_memories[layer_idx].to(dtype=dtype),
            layer_idx,
            hidden_states=text_hidden,
            initial_hidden_states=initial_text_hidden,
            current_visual_memory=visual_memory_state,
            vision_padding_mask=~image_mask,
        )
        if layer_idx not in sidecar_active_layers:
            delta = residual_from_cached_coeff(sidecar, cached_anchor_coeff, layer_tensor, text_hidden)
            delta = delta * float(sidecar_scale)
            h = run_qwen3vl_full_layer_with_text_delta(
                language_model,
                layer_idx,
                h,
                position_ids,
                attention_mask_2d,
                text_pos,
                delta,
            )
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                h = language_model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
            continue
        visual_kv = shared_visual_kv
        text_pos_emb = shared_text_pos_emb
        visual_pos_emb = None
        if getattr(sidecar, "reader_mode", "cross_attention") == "pooled":
            visual_kv = None
            text_pos_emb = None
        elif visual_kv is None:
            if sidecar_visual_kv_source == "qwen_native":
                visual_kv = qwen_native_visual_kv(
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_trainable_native":
                visual_kv = qwen_trainable_native_visual_kv(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_anchor_trainable":
                visual_kv = qwen_anchor_trainable_visual_kv(
                    sidecar,
                    language_model,
                    layer_idx,
                    visual_memory_for_layer,
                    visual_position_ids,
                    ~image_mask,
                )
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_first_layer":
                visual_kv = shared_visual_kv
                text_pos_emb = None
            elif sidecar_visual_kv_source == "qwen_first_layer_film":
                if shared_visual_kv is None:
                    raise RuntimeError("qwen_first_layer_film requires shared layer-0 visual K/V")
                visual_kv = qwen_first_layer_film_visual_kv(sidecar, shared_visual_kv, layer_idx)
                text_pos_emb = None
            else:
                if getattr(sidecar, "use_rope", False):
                    visual_pos_emb = language_model.rotary_emb(visual_memory_for_layer, visual_position_ids)
                    text_pos_emb = language_model.rotary_emb(visual_memory_for_layer, text_position_ids)
                else:
                    visual_kv, text_pos_emb = sidecar_rope_visual_kv(
                        sidecar,
                        language_model,
                        visual_memory_for_layer,
                        position_ids,
                        text_position_ids,
                        image_pos,
                        image_mask,
                        dtype,
                    )
        query_override = None
        if sidecar_query_source == "qwen_native":
            query_override = qwen_native_sidecar_query(language_model, layer_idx, text_hidden, text_position_ids)
        elif sidecar_query_source == "qwen_trainable_native":
            query_override = qwen_trainable_native_sidecar_query(
                sidecar,
                language_model,
                layer_idx,
                text_hidden,
                text_position_ids,
            )
        text_attention = None
        if sidecar.output_mode.startswith("factorized"):
            text_attention = qwen3vl_text_attention_output(
                language_model,
                layer_idx,
                text_hidden,
                text_position_ids,
                padding_mask=~text_mask,
            )
        text_attention_heads = None
        if sidecar.output_mode in {
            "factorized_native_head_o",
            "factorized_native_head_o_pure",
            "factorized_native_head_o_residual",
        }:
            text_attention_heads = qwen3vl_text_attention_heads(
                language_model,
                layer_idx,
                text_hidden,
                text_position_ids,
                padding_mask=~text_mask,
            )
        visual_mass = factorized_visual_mass_override(
            sidecar,
            language_model,
            layer_idx,
            text_hidden,
            h,
            position_ids,
            text_pos,
            image_pos,
            text_mask,
            image_mask,
            full_mask,
            visual_memory_for_layer,
            factorized_mass_mode,
            fixed_visual_mass,
            dtype,
        )
        call_vision_states = None if visual_kv is not None else visual_memory_for_layer
        call_vision_mask = None if visual_kv is not None else ~image_mask
        if getattr(sidecar, "visual_transform_mode", "none") == "latent_compressor":
            call_vision_mask = None
        if sidecar.state_tokens > 0:
            delta, sidecar_state, coeff = sidecar(
                text_hidden,
                call_vision_states,
                layer_tensor,
                sidecar_state=sidecar_state,
                visual_kv=visual_kv,
                vision_padding_mask=call_vision_mask,
                text_attention=text_attention,
                visual_mass=visual_mass,
                output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                text_attention_heads=text_attention_heads,
                position_embeddings=text_pos_emb,
                visual_position_embeddings=visual_pos_emb,
                query_states=query_override,
                initial_hidden_states=initial_text_hidden,
                return_state=True,
                return_coefficients=True,
            )
        else:
            delta, coeff = sidecar(
                text_hidden,
                call_vision_states,
                layer_tensor,
                visual_kv=visual_kv,
                vision_padding_mask=call_vision_mask,
                text_attention=text_attention,
                visual_mass=visual_mass,
                output_projection=language_model.layers[layer_idx].self_attn.o_proj,
                text_attention_heads=text_attention_heads,
                position_embeddings=text_pos_emb,
                visual_position_embeddings=visual_pos_emb,
                query_states=query_override,
                initial_hidden_states=initial_text_hidden,
                return_coefficients=True,
            )
        cached_anchor_coeff = coeff
        delta = delta * float(sidecar_scale)
        h = run_qwen3vl_full_layer_with_text_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_mask_2d,
            text_pos,
            delta,
        )
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            h = language_model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
    logits = model.lm_head(language_model.norm(h))
    return logits[0, int(text_pos[0, int(text_mask[0].sum().item()) - 1].item())]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, None)[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    sidecar = None
    native_sidecar = None
    if args.sidecar_backend == "learned" and args.checkpoint:
        sidecar = load_sidecar(args, language_model, device, dtype)
    if args.sidecar_backend == "native_attention":
        native_sidecar = load_native_sidecar(args.checkpoint, len(language_model.layers), device, dtype)
    if args.sidecar_backend in {"analytic_attention", "native_attention"} and args.visual_memory_mode == "v0":
        args.visual_memory_mode = "vprefix"

    metrics: dict[str, dict[str, float | int]] = {
        "qwen": {"correct": 0, "scored": 0},
        "no_visual": {"correct": 0, "agree": 0, "qwen_correct_retention": 0, "kl": 0.0, "scored": 0},
    }
    if sidecar is not None or args.sidecar_backend in {"analytic_attention", "native_attention"}:
        metrics["sidecar_only"] = {"correct": 0, "agree": 0, "qwen_correct_retention": 0, "kl": 0.0, "scored": 0}
    if sidecar is not None:
        metrics["hybrid"] = {"correct": 0, "agree": 0, "qwen_correct_retention": 0, "kl": 0.0, "scored": 0}
    predictions = []
    skipped = 0
    qwen_correct_total = 0
    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            skipped += 1
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        q_logits = qwen_logits(processor, model, row, args.benchmark, device)
        q_pred = predict(q_logits, ids)
        q_dist = distribution(q_logits, ids)
        q_correct = q_pred == gold
        qwen_correct_total += int(q_correct)
        metrics["qwen"]["correct"] += int(q_correct)
        metrics["qwen"]["scored"] += 1
        sample = {"index": row.get("index", idx), "gold": gold, "qwen": q_pred, "qwen_correct": q_correct}
        nv_logits = no_visual_logits(processor, model, language_model, row, args.benchmark, device, dtype)
        nv_pred = predict(nv_logits, ids)
        nv_dist = distribution(nv_logits, ids)
        metrics["no_visual"]["correct"] += int(nv_pred == gold)
        metrics["no_visual"]["agree"] += int(nv_pred == q_pred)
        metrics["no_visual"]["qwen_correct_retention"] += int(q_correct and nv_pred == q_pred)
        metrics["no_visual"]["kl"] += float(F.kl_div(nv_dist.log(), q_dist, reduction="sum").item())
        metrics["no_visual"]["scored"] += 1
        sample["no_visual"] = nv_pred
        if args.sidecar_backend in {"analytic_attention", "native_attention"}:
            logits = analytic_sidecar_logits(
                processor,
                model,
                language_model,
                row,
                args.benchmark,
                device,
                dtype,
                args.visual_memory_mode,
                native_sidecar=native_sidecar,
            )
            pred = predict(logits, ids)
            dist = distribution(logits, ids)
            metrics["sidecar_only"]["correct"] += int(pred == gold)
            metrics["sidecar_only"]["agree"] += int(pred == q_pred)
            metrics["sidecar_only"]["qwen_correct_retention"] += int(q_correct and pred == q_pred)
            metrics["sidecar_only"]["kl"] += float(F.kl_div(dist.log(), q_dist, reduction="sum").item())
            metrics["sidecar_only"]["scored"] += 1
            sample["sidecar_only"] = pred
        elif sidecar is not None:
            for key, fn in (
                ("sidecar_only", sidecar_logits),
                ("hybrid", hybrid_logits),
            ):
                if key == "sidecar_only":
                    logits = fn(
                        processor,
                        model,
                        language_model,
                        sidecar,
                        row,
                        args.benchmark,
                        device,
                        dtype,
                        args.sidecar_scale,
                        args.visual_memory_mode,
                        args.sidecar_query_source,
                        args.sidecar_visual_kv_source,
                        args.factorized_mass_mode,
                        args.fixed_visual_mass,
                        args.sidecar_active_layers,
                    )
                else:
                    logits = fn(
                        processor,
                        model,
                        language_model,
                        sidecar,
                        row,
                        args.benchmark,
                        device,
                        dtype,
                        args.sidecar_scale,
                        args.visual_memory_mode,
                        args.sidecar_query_source,
                        args.sidecar_visual_kv_source,
                        args.factorized_mass_mode,
                        args.fixed_visual_mass,
                        args.sidecar_active_layers,
                    )
                pred = predict(logits, ids)
                dist = distribution(logits, ids)
                metrics[key]["correct"] += int(pred == gold)
                metrics[key]["agree"] += int(pred == q_pred)
                metrics[key]["qwen_correct_retention"] += int(q_correct and pred == q_pred)
                metrics[key]["kl"] += float(F.kl_div(dist.log(), q_dist, reduction="sum").item())
                metrics[key]["scored"] += 1
                sample[key] = pred
        predictions.append(sample)
        if (idx + 1) % 10 == 0 or idx + 1 == len(rows):
            print(f"evaluated {idx + 1}/{len(rows)} scored={metrics['qwen']['scored']} skipped={skipped}", flush=True)

    results = []
    qwen_n = max(int(metrics["qwen"]["scored"]), 1)
    results.append(
        {
            "setting": "qwen",
            "scored": int(metrics["qwen"]["scored"]),
            "correct": int(metrics["qwen"]["correct"]),
            "accuracy": float(metrics["qwen"]["correct"]) / qwen_n,
        }
    )
    for key in ("no_visual", "sidecar_only", "hybrid"):
        if key not in metrics:
            continue
        row_metrics = metrics[key]
        n = max(int(row_metrics["scored"]), 1)
        results.append(
            {
                "setting": key,
                "scored": int(row_metrics["scored"]),
                "correct": int(row_metrics["correct"]),
                "accuracy": float(row_metrics["correct"]) / n,
                "qwen_agreement": float(row_metrics["agree"]) / n,
                "qwen_correct_retention": float(row_metrics["qwen_correct_retention"]) / max(qwen_correct_total, 1),
                "output_kl_to_qwen": float(row_metrics["kl"]) / n,
            }
        )

    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "sidecar_backend": args.sidecar_backend,
        "rank": args.rank,
        "sidecar_dim": args.sidecar_dim,
        "num_heads": args.num_heads,
        "state_tokens": args.state_tokens,
        "layer_adapter_rank": args.layer_adapter_rank,
        "reader_activation": args.reader_activation,
        "shared_basis": args.shared_basis,
        "output_mode": args.output_mode,
        "visual_memory_mode": args.visual_memory_mode,
        "sidecar_query_source": args.sidecar_query_source,
        "sidecar_visual_kv_source": args.sidecar_visual_kv_source,
        "sidecar_scale": args.sidecar_scale,
        "skipped": skipped,
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
