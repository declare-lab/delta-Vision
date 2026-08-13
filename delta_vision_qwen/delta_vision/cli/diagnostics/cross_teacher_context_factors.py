#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.masking_utils import create_causal_mask

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    compute_llama_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llama_attention_output,
    llava15_prompt,
    make_causal_mask,
    read_jsonl,
)
from delta_vision.models.modeling import image_token_id, load_frozen_llava
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prompt,
    qwen3vl_attention_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cross-teacher shared context-factor diagnostic.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teachers", default="llava,qwen3vl", help="Comma-separated: llava,qwen3vl")
    parser.add_argument("--llava-model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--qwen-model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def mmstar_question(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    return f"{question}\nAnswer directly with only the letter of the correct option."


def _llava_full_layer(language_model: torch.nn.Module, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    layer = get_lm_layers(language_model)[layer_idx]
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention_mask = make_causal_mask(1, hidden.shape[1], hidden.device, hidden.dtype)
    residual = hidden
    normed = layer.input_layernorm(hidden)
    position_embeddings = rotary_owner.rotary_emb(normed, position_ids)
    attn_out = llama_attention_output(layer.self_attn, normed, position_embeddings, attention_mask, is_causal=False)
    hidden = residual + attn_out
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    return residual + hidden


@torch.inference_mode()
def extract_llava_features(
    rows: list[dict[str, Any]],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    layer_stride: int,
    log_every: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    processor, model = load_frozen_llava(model_path, dtype, device, attn_implementation)
    language_model = get_language_model(model)
    layers = get_lm_layers(language_model)
    selected_layers = list(range(0, len(layers), layer_stride))
    img_id = image_token_id(model, processor)
    feats: list[torch.Tensor] = []

    for idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=llava15_prompt(mmstar_question(row)), images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        hidden = build_llava_initial_hidden(
            model,
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_sizes=inputs.get("image_sizes"),
            vision_feature_layer=getattr(model.config, "vision_feature_layer", None),
            vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", None),
        ).to(dtype=dtype)
        text_pos, _image_pos, _teacher_pos = get_text_and_image_positions(inputs["input_ids"], hidden.shape[1], img_id)
        sample_parts: list[torch.Tensor] = []
        for layer_idx in range(len(layers)):
            if layer_idx in selected_layers:
                delta = compute_llama_attention_effect(language_model, layer_idx, hidden, text_pos)
                sample_parts.append(delta[0, -1].float().cpu())
            hidden = _llava_full_layer(language_model, layer_idx, hidden)
        feats.append(torch.cat(sample_parts, dim=0))
        if (idx + 1) % log_every == 0:
            print(f"llava extracted {idx + 1}/{len(rows)}", flush=True)
    meta = {
        "teacher": "llava",
        "model_path": model_path,
        "selected_layers": selected_layers,
        "feature_shape": list(feats[0].shape) if feats else [0],
        "summary": "concatenated last-text-token attention effect for selected layers",
    }
    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(feats, dim=0), meta


def _qwen_full_layer(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> torch.Tensor:
    layer = language_model.layers[layer_idx]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden, position_ids)
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attn_out = qwen3vl_attention_output(layer.self_attn, normed, position_embeddings, attention_mask)
    hidden = residual + attn_out
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    return residual + hidden


@torch.inference_mode()
def extract_qwen_features(
    rows: list[dict[str, Any]],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    layer_stride: int,
    log_every: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    processor, model = load_frozen_qwen3vl(model_path, dtype, device, attn_implementation)
    language_model = get_language_model(model)
    selected_layers = list(range(0, len(language_model.layers), layer_stride))
    feats: list[torch.Tensor] = []

    for idx, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            inputs = processor(text=qwen3vl_prompt(processor, mmstar_question(row)), images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        hidden, full_position_ids, _visual_pos_masks, _deepstack = build_qwen3vl_initial_context(model, inputs)
        hidden = hidden.to(dtype=dtype)
        text_pos, _image_pos, text_position_ids, text_mask, _image_mask, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        sample_parts: list[torch.Tensor] = []
        for layer_idx in range(len(language_model.layers)):
            if layer_idx in selected_layers:
                text_hidden = gather_batched_positions(hidden, text_pos, text_mask)
                delta = compute_qwen3vl_attention_effect_batched(
                    language_model,
                    layer_idx,
                    hidden,
                    text_hidden,
                    full_position_ids,
                    text_position_ids,
                    text_pos,
                    full_mask,
                    text_mask,
                )
                last_idx = int(torch.nonzero(text_mask[0].bool(), as_tuple=False).flatten()[-1].item())
                sample_parts.append(delta[0, last_idx].float().cpu())
            hidden = _qwen_full_layer(language_model, layer_idx, hidden, full_position_ids, inputs["attention_mask"])
        feats.append(torch.cat(sample_parts, dim=0))
        if (idx + 1) % log_every == 0:
            print(f"qwen3vl extracted {idx + 1}/{len(rows)}", flush=True)
    meta = {
        "teacher": "qwen3vl",
        "model_path": model_path,
        "selected_layers": selected_layers,
        "feature_shape": list(feats[0].shape) if feats else [0],
        "summary": "concatenated last-text-token attention effect for selected layers",
    }
    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(feats, dim=0), meta


def standardize(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = y.mean(dim=0, keepdim=True)
    std = y.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (y - mean) / std, mean, std


def pca_reconstruct(y: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    y0 = y - y.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(y0, full_matrices=False)
    r = min(rank, u.shape[1], vh.shape[0])
    recon = (u[:, :r] * s[:r]) @ vh[:r] + y.mean(dim=0, keepdim=True)
    return recon, u[:, :r]


def r2_score(y: torch.Tensor, recon: torch.Tensor) -> float:
    sse = (y - recon).pow(2).sum()
    sst = (y - y.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    return float((1.0 - sse / sst).item())


def analyze(features: dict[str, torch.Tensor], ranks: list[int]) -> dict[str, Any]:
    standardized: dict[str, torch.Tensor] = {}
    for name, y in features.items():
        standardized[name] = standardize(y.float())[0]

    weighted_concat = torch.cat(
        [y / (y.shape[1] ** 0.5) for y in standardized.values()],
        dim=1,
    )
    shared_center = weighted_concat - weighted_concat.mean(dim=0, keepdim=True)
    u_shared_all, _s_shared, _vh_shared = torch.linalg.svd(shared_center, full_matrices=False)

    names = list(standardized)
    shuffled = dict(standardized)
    if len(names) >= 2:
        generator = torch.Generator().manual_seed(1234)
        perm = torch.randperm(standardized[names[-1]].shape[0], generator=generator)
        shuffled[names[-1]] = standardized[names[-1]][perm]
        shuffled_concat = torch.cat(
            [y / (y.shape[1] ** 0.5) for y in shuffled.values()],
            dim=1,
        )
        shuffled_center = shuffled_concat - shuffled_concat.mean(dim=0, keepdim=True)
        u_shuffled_all, _s_shuffled, _vh_shuffled = torch.linalg.svd(shuffled_center, full_matrices=False)
    else:
        u_shuffled_all = u_shared_all

    results = []
    for rank in ranks:
        item: dict[str, Any] = {"rank": rank, "teachers": {}}
        shared_mean = []
        independent_mean = []
        for name, y in standardized.items():
            indep_recon, _u = pca_reconstruct(y, rank)
            r = min(rank, u_shared_all.shape[1])
            u = u_shared_all[:, :r]
            decoder = u.transpose(0, 1) @ y
            shared_recon = u @ decoder
            u_neg = u_shuffled_all[:, :r]
            neg_decoder = u_neg.transpose(0, 1) @ y
            neg_recon = u_neg @ neg_decoder
            indep_r2 = r2_score(y, indep_recon)
            shared_r2 = r2_score(y, shared_recon)
            shuffled_shared_r2 = r2_score(y, neg_recon)
            item["teachers"][name] = {
                "independent_r2": indep_r2,
                "shared_r2": shared_r2,
                "shuffled_shared_r2": shuffled_shared_r2,
                "shared_drop": indep_r2 - shared_r2,
                "shuffled_gap": shared_r2 - shuffled_shared_r2,
            }
            shared_mean.append(shared_r2)
            independent_mean.append(indep_r2)
        item["mean_independent_r2"] = float(sum(independent_mean) / len(independent_mean))
        item["mean_shared_r2"] = float(sum(shared_mean) / len(shared_mean))
        item["mean_shared_drop"] = item["mean_independent_r2"] - item["mean_shared_r2"]
        results.append(item)
    return {"ranks": results}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.data)
    rows = rows[args.start_index : args.start_index + args.max_samples]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    teachers = [x.strip() for x in args.teachers.split(",") if x.strip()]
    features: dict[str, torch.Tensor] = {}
    metas: dict[str, Any] = {}

    for teacher in teachers:
        if teacher == "llava":
            feats, meta = extract_llava_features(
                rows,
                args.llava_model_path,
                device,
                dtype,
                args.attn_implementation,
                args.layer_stride,
                args.log_every,
            )
        elif teacher == "qwen3vl":
            feats, meta = extract_qwen_features(
                rows,
                args.qwen_model_path,
                device,
                dtype,
                args.attn_implementation,
                args.layer_stride,
                args.log_every,
            )
        else:
            raise ValueError(f"unknown teacher: {teacher}")
        features[teacher] = feats
        metas[teacher] = meta
        torch.save({"features": feats, "meta": meta}, out_dir / f"{teacher}_effect_features.pt")
        print(f"{teacher} feature matrix {tuple(feats.shape)}", flush=True)

    report = {
        "data": args.data,
        "start_index": args.start_index,
        "max_samples": len(rows),
        "teachers": metas,
        "analysis": analyze(features, ranks),
        "interpretation": {
            "independent_r2": "PCA upper bound with each teacher owning its own context latent U_m(c).",
            "shared_r2": "Same shared sample latent U(c), with teacher-specific linear decoders.",
            "shuffled_shared_r2": "Negative control: shared latent after shuffling the last teacher's context order.",
            "shared_drop": "Independent R2 minus shared R2; small drop supports cross-teacher common context factors.",
            "shuffled_gap": "Shared R2 minus shuffled-shared R2; positive gap means context alignment matters.",
        },
    }
    (out_dir / "shared_context_factor_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
