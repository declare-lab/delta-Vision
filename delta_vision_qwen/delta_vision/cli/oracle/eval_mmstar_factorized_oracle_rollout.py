#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from delta_vision.evaluation.metrics import option_distribution, option_token_id_lists, predict_option
from delta_vision.models.llava import (
    compute_llama_attention_effect,
    compute_llama_factorized_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    read_jsonl,
    run_llama_layer_text_with_attention_delta,
)
from delta_vision.runtime.basis import load_layer_basis, project_delta_to_coefficients, reconstruct_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate MMStar direct-vs-factorized attention oracle rollout.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--effects-dir", default="artifacts/effects/mmstar_attention_effects_1000_teacherpos_storehidden")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--output-json", default="artifacts/eval/oracle/mmstar_factorized_oracle_rollout_100.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ranks", default="64,128,256,512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def _setting_key(mode: str, rank: int | None = None) -> str:
    return mode if rank is None else f"{mode}_{rank}"


@torch.inference_mode()
def rollout_with_deltas(
    language_model: torch.nn.Module,
    initial_text_hidden: torch.Tensor,
    text_positions: torch.Tensor,
    deltas: list[torch.Tensor] | None,
    basis: torch.Tensor | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    layers = get_lm_layers(language_model)
    h = initial_text_hidden
    position_ids = text_positions.unsqueeze(0)
    for layer_idx in range(len(layers)):
        attn_delta = None
        if deltas is not None:
            attn_delta = deltas[layer_idx].to(device=h.device, dtype=h.dtype)
            if basis is not None and rank is not None:
                layer_basis = basis[layer_idx : layer_idx + 1, :rank].expand(attn_delta.shape[0], -1, -1)
                coeff = project_delta_to_coefficients(attn_delta, layer_basis)
                attn_delta = reconstruct_delta(coeff, layer_basis)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=attn_delta,
            layer=layers[layer_idx],
        )
    return h


def _new_metrics() -> dict[str, float]:
    return {
        "correct": 0,
        "agree": 0,
        "teacher_correct_retention": 0,
        "kl": 0.0,
        "hidden32_mse": 0.0,
        "scored": 0,
    }


def prepare_prompt_only_inputs(
    processor: AutoProcessor,
    row: dict,
    image_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image = Image.open(row["image"]).convert("RGB")
    try:
        prompt = llava15_prompt(str(row["question"]).strip())
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    finally:
        image.close()
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, None)[args.start_index :]
    rows = rows[: args.max_samples]
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    language_model = get_language_model(model)
    layers = get_lm_layers(language_model)
    norm = get_lm_norm(language_model)
    num_layers = len(layers)
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    basis = load_layer_basis(args.basis, max(ranks), num_layers, 4096).to(device=device, dtype=dtype)
    option_ids = option_token_id_lists(processor.tokenizer)

    settings: list[tuple[str, int | None]] = [
        ("teacher", None),
        ("no_visual", None),
        ("v1_direct_full", None),
        ("v2_factorized_full", None),
    ]
    settings += [("v1_direct_rank", rank) for rank in ranks]
    settings += [("v2_factorized_rank", rank) for rank in ranks]
    metrics = {_setting_key(mode, rank): _new_metrics() for mode, rank in settings}
    teacher_correct = 0
    effect_cos_sum = 0.0
    effect_norm_ratio_sum = 0.0
    effect_count = 0

    for sample_idx, row in enumerate(rows):
        inputs = prepare_prompt_only_inputs(
            processor,
            row,
            image_token_id,
            device,
        )
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden_states = tuple(x.detach().to(dtype=dtype) for x in teacher.hidden_states[:-1])
        merged_len = hidden_states[0].shape[1]
        text_positions, image_positions, _ = get_text_and_image_positions(inputs["input_ids"], merged_len, image_token_id)
        text_positions = text_positions.to(device)
        image_positions = image_positions.to(device)
        last_text_idx = int(text_positions[-1].item())
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_logits = teacher.logits[0, last_text_idx]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        is_teacher_correct = teacher_pred == gold
        teacher_correct += int(is_teacher_correct)
        initial_text_hidden = hidden_states[0].index_select(1, text_positions)
        teacher_final_h = teacher.hidden_states[-1].detach().to(dtype=dtype).index_select(1, text_positions)

        direct_deltas: list[torch.Tensor] = []
        factorized_deltas: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            direct = compute_llama_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx],
                text_positions,
            ).detach()
            factor = compute_llama_factorized_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx],
                text_positions,
                image_positions,
            )["factorized_delta"].detach()
            direct_deltas.append(direct)
            factorized_deltas.append(factor)
            direct_f = direct.float().reshape(-1, direct.shape[-1])
            factor_f = factor.float().reshape(-1, factor.shape[-1])
            effect_cos_sum += float(F.cosine_similarity(factor_f, direct_f, dim=-1, eps=1e-6).mean().item())
            effect_norm_ratio_sum += float(
                (factor_f.norm(dim=-1) / direct_f.norm(dim=-1).clamp_min(1e-6)).mean().item()
            )
            effect_count += 1

        h_cache: dict[str, torch.Tensor] = {
            "teacher": teacher_final_h,
            "no_visual": rollout_with_deltas(language_model, initial_text_hidden, text_positions, None),
            "v1_direct_full": rollout_with_deltas(language_model, initial_text_hidden, text_positions, direct_deltas),
            "v2_factorized_full": rollout_with_deltas(language_model, initial_text_hidden, text_positions, factorized_deltas),
        }
        for rank in ranks:
            h_cache[f"v1_direct_rank_{rank}"] = rollout_with_deltas(
                language_model,
                initial_text_hidden,
                text_positions,
                direct_deltas,
                basis=basis,
                rank=rank,
            )
            h_cache[f"v2_factorized_rank_{rank}"] = rollout_with_deltas(
                language_model,
                initial_text_hidden,
                text_positions,
                factorized_deltas,
                basis=basis,
                rank=rank,
            )

        for mode, rank in settings:
            key = _setting_key(mode, rank)
            h = h_cache[key]
            if key == "teacher":
                logits = teacher_logits
            else:
                logits = model.lm_head(norm(h))[0, -1]
            pred = predict_option(logits, option_ids)
            dist = option_distribution(logits, option_ids)
            metrics[key]["correct"] += int(pred == gold)
            metrics[key]["agree"] += int(pred == teacher_pred)
            metrics[key]["teacher_correct_retention"] += int(is_teacher_correct and pred == teacher_pred)
            metrics[key]["kl"] += float(F.kl_div(dist.log(), teacher_dist, reduction="sum").item())
            metrics[key]["hidden32_mse"] += float((h.float() - teacher_final_h.float()).pow(2).mean().item())
            metrics[key]["scored"] += 1

        if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(rows):
            print(f"processed {sample_idx + 1}/{len(rows)}", flush=True)

    results = []
    for mode, rank in settings:
        key = _setting_key(mode, rank)
        m = metrics[key]
        n = max(int(m["scored"]), 1)
        results.append(
            {
                "setting": key,
                "mode": mode,
                "rank": rank,
                "num_samples": m["scored"],
                "accuracy": m["correct"] / n,
                "teacher_agreement": m["agree"] / n,
                "teacher_correct_retention": m["teacher_correct_retention"] / max(teacher_correct, 1),
                "output_kl": m["kl"] / n,
                "hidden32_mse": m["hidden32_mse"] / n,
            }
        )

    payload = {
        "data": args.data,
        "effects_dir": args.effects_dir,
        "model_path": args.model_path,
        "basis": args.basis,
        "num_samples": len(rows),
        "teacher_correct": teacher_correct,
        "effect_factorized_vs_direct": {
            "cosine": effect_cos_sum / max(effect_count, 1),
            "norm_ratio": effect_norm_ratio_sum / max(effect_count, 1),
            "count": effect_count,
        },
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
