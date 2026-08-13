#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from delta_vision.models.llava import dtype_from_name, get_language_model, read_jsonl
from delta_vision.evaluation.metrics import OPTIONS, option_distribution, option_token_id_lists, predict_option
from delta_vision.models.qwen3vl import (
    build_qwen3vl_initial_context,
    compute_qwen3vl_attention_effect_batched,
    gather_batched_positions,
    get_qwen3vl_text_image_positions,
    load_frozen_qwen3vl,
    qwen3vl_prefix_visual_memory_by_layer,
    qwen3vl_prompt,
    run_qwen3vl_layer_text_with_attention_delta,
    scatter_batched_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL MMStar attention-effect oracles.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--basis-output", default="artifacts/basis/qwen3vl_mmstar_attention_pca_rank512.pt")
    parser.add_argument("--reuse-basis", action="store_true")
    parser.add_argument("--max-rank", type=int, default=512)
    parser.add_argument("--ranks", default="32,64,128,256,512")
    parser.add_argument("--max-tokens-per-layer", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def build_prompt(processor: Any, row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    question = f"{question}\nAnswer directly with only the letter of the correct option."
    return qwen3vl_prompt(processor, question)


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


def memory_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    reference_full: torch.Tensor,
    text_hidden: torch.Tensor,
    image_memory: torch.Tensor,
    full_position_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    full_mask: torch.Tensor,
    text_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    full = build_full_from_text_and_memory(
        reference_full,
        text_positions,
        text_hidden,
        text_mask,
        image_positions,
        image_memory,
        image_mask,
    )
    return compute_qwen3vl_attention_effect_batched(
        language_model,
        layer_idx,
        full,
        text_hidden,
        full_position_ids,
        text_position_ids,
        text_positions,
        full_mask,
        text_mask,
    )


@torch.inference_mode()
def prompt_inputs(processor: Any, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    with Image.open(row["image"]) as image:
        inputs = processor(text=build_prompt(processor, row), images=image.convert("RGB"), return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def update_metrics(metrics: dict[str, float | int], pred: str, teacher_pred: str, gold: str, kl: float) -> None:
    metrics["correct"] = int(metrics["correct"]) + int(pred == gold)
    metrics["agree"] = int(metrics["agree"]) + int(pred == teacher_pred)
    metrics["teacher_correct_and_agree"] = int(metrics["teacher_correct_and_agree"]) + int(
        teacher_pred == gold and pred == teacher_pred
    )
    metrics["output_kl_sum"] = float(metrics["output_kl_sum"]) + float(kl)
    metrics["scored"] = int(metrics["scored"]) + 1


def finalize_metrics(metrics: dict[str, float | int], teacher_correct: int) -> dict[str, float | int]:
    scored = max(int(metrics["scored"]), 1)
    teacher_correct = max(int(teacher_correct), 1)
    return {
        "num_samples": int(metrics["scored"]),
        "correct": int(metrics["correct"]),
        "accuracy": float(metrics["correct"]) / scored,
        "teacher_agreement": float(metrics["agree"]) / scored,
        "teacher_correct_and_agree": int(metrics["teacher_correct_and_agree"]),
        "teacher_correct_retention": float(metrics["teacher_correct_and_agree"]) / teacher_correct,
        "output_kl": float(metrics["output_kl_sum"]) / scored,
    }


def parse_ranks(spec: str, max_rank: int) -> list[int]:
    ranks = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not ranks:
        raise ValueError("--ranks cannot be empty")
    if ranks[-1] > max_rank:
        raise ValueError(f"rank {ranks[-1]} exceeds --max-rank {max_rank}")
    return ranks


def project_reconstruct(delta: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    layer_basis = basis[:, :rank].to(device=delta.device)
    coeff = torch.matmul(delta.float(), layer_basis.float().transpose(1, 2))
    return torch.matmul(coeff, layer_basis.float()).to(dtype=delta.dtype)


@torch.inference_mode()
def build_or_load_basis(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = Path(args.basis_output)
    if args.reuse_basis and out.exists():
        loaded = torch.load(out, map_location="cpu")
        raw = loaded["basis"]
        if isinstance(raw, dict):
            basis = torch.stack([raw[layer] for layer in range(len(language_model.layers))], dim=0)
        else:
            basis = raw
        return basis[:, : args.max_rank].to(device=device, dtype=dtype)

    num_layers = len(language_model.layers)
    hidden_size = int(language_model.config.hidden_size)
    banks: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(num_layers)}
    counts = {layer: 0 for layer in range(num_layers)}
    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, _, _ = build_qwen3vl_initial_context(model, inputs)
        text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen3vl_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach() for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach() for state in teacher_states
        ]
        valid = text_mask[0].to(device=device).bool()
        for layer_idx in range(num_layers):
            if counts[layer_idx] >= args.max_tokens_per_layer:
                continue
            delta = compute_qwen3vl_attention_effect_batched(
                language_model,
                layer_idx,
                teacher_states[layer_idx].to(dtype=dtype),
                teacher_text_states[layer_idx].to(dtype=dtype),
                full_position_ids,
                text_position_ids,
                text_positions,
                full_mask,
                text_mask,
            )[0, valid]
            remaining = args.max_tokens_per_layer - counts[layer_idx]
            if delta.shape[0] > remaining:
                idxs = torch.randperm(delta.shape[0], device=device)[:remaining]
                delta = delta.index_select(0, idxs)
            if delta.numel() > 0:
                banks[layer_idx].append(delta.detach().cpu().to(torch.float16))
                counts[layer_idx] += int(delta.shape[0])
        if (idx + 1) % 25 == 0:
            print(f"basis collect {idx + 1}/{len(rows)} min_tokens={min(counts.values())}", flush=True)
        if all(counts[layer] >= args.max_tokens_per_layer for layer in range(num_layers)):
            print(f"basis token banks full after {idx + 1}/{len(rows)} samples", flush=True)
            break

    bases = {}
    energies = {}
    for layer_idx in range(num_layers):
        if not banks[layer_idx]:
            raise RuntimeError(f"empty PCA bank for layer {layer_idx}")
        matrix = torch.cat(banks[layer_idx], dim=0).float()
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        matrix = matrix.to(device)
        _, svals, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(args.max_rank, vh.shape[0])
        bases[layer_idx] = vh[:rank].detach().cpu().to(torch.float16)
        energies[layer_idx] = svals.detach().cpu().float().pow(2)
        print(f"basis svd layer={layer_idx} tokens={matrix.shape[0]} rank={rank}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": args.data,
            "model_path": args.model_path,
            "max_samples": len(rows),
            "max_rank": args.max_rank,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "counts": counts,
            "basis": bases,
            "singular_energy": energies,
        },
        out,
    )
    out.with_suffix(".metrics.json").write_text(
        json.dumps({"counts": counts, "max_rank": args.max_rank}, indent=2),
        encoding="utf-8",
    )
    return torch.stack([bases[layer] for layer in range(num_layers)], dim=0).to(device=device, dtype=dtype)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, args.max_samples)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    ranks = parse_ranks(args.ranks, args.max_rank)
    basis = build_or_load_basis(args, rows, processor, model, language_model, device, dtype)
    option_ids = option_token_id_lists(processor.tokenizer)
    num_layers = len(language_model.layers)
    names = ("no_visual", "v0", "vdeep", "vcum", "vprefix", "vteacher", "full_target") + tuple(
        f"rank_{rank}" for rank in ranks
    )
    metrics = {
        name: {"correct": 0, "agree": 0, "teacher_correct_and_agree": 0, "output_kl_sum": 0.0, "scored": 0}
        for name in names
    }
    teacher_correct = 0
    predictions = []

    for idx, row in enumerate(rows):
        inputs = prompt_inputs(processor, row, device)
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden0, full_position_ids, visual_pos_masks, deepstack_visual_embeds = build_qwen3vl_initial_context(model, inputs)
        text_positions, image_positions, text_position_ids, text_mask, image_mask, full_mask = (
            get_qwen3vl_text_image_positions(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["mm_token_type_ids"],
                full_position_ids,
            )
        )
        teacher_states = [state.detach() for state in teacher.hidden_states]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach() for state in teacher_states
        ]
        last_text_idx = int(text_mask[0].sum().item()) - 1
        teacher_logits = teacher.logits[0, int(text_positions[0, last_text_idx].item())]
        teacher_pred = predict_option(teacher_logits, option_ids)
        teacher_dist = option_distribution(teacher_logits, option_ids)
        gold = str(row["answer"]).strip().upper()[:1]
        teacher_correct += int(teacher_pred == gold)

        v0 = gather_batched_positions(hidden0, image_positions, image_mask).to(dtype=dtype)
        if deepstack_visual_embeds:
            deep_sum = torch.stack([x.to(device=device, dtype=dtype) for x in deepstack_visual_embeds], dim=0).sum(dim=0)
            vdeep = v0 + deep_sum.unsqueeze(0)
        else:
            vdeep = v0
        vcum_by_layer = []
        running_deep = torch.zeros_like(v0)
        for layer_idx in range(num_layers):
            vcum_by_layer.append(v0 + running_deep)
            if layer_idx < len(deepstack_visual_embeds):
                running_deep = running_deep + deepstack_visual_embeds[layer_idx].to(
                    device=device, dtype=dtype
                ).unsqueeze(0)
        vprefix_by_layer = qwen3vl_prefix_visual_memory_by_layer(
            language_model,
            hidden0.to(dtype=dtype),
            full_position_ids,
            inputs["attention_mask"],
            image_positions,
            image_mask,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
        vteacher_by_layer = [
            gather_batched_positions(state, image_positions, image_mask).to(dtype=dtype)
            for state in teacher_states[:num_layers]
        ]

        sample_preds = {"index": row.get("index", idx), "gold": gold, "teacher": teacher_pred}
        for name in names:
            h = teacher_text_states[0].to(dtype=dtype)
            for layer_idx in range(num_layers):
                if name == "no_visual":
                    delta = None
                elif name == "full_target" or name.startswith("rank_"):
                    delta = compute_qwen3vl_attention_effect_batched(
                        language_model,
                        layer_idx,
                        teacher_states[layer_idx].to(dtype=dtype),
                        teacher_text_states[layer_idx].to(dtype=dtype),
                        full_position_ids,
                        text_position_ids,
                        text_positions,
                        full_mask,
                        text_mask,
                    )
                    if name.startswith("rank_"):
                        delta = project_reconstruct(delta, basis[layer_idx : layer_idx + 1], int(name.split("_", 1)[1]))
                else:
                    if name == "v0":
                        memory = v0
                    elif name == "vdeep":
                        memory = vdeep
                    elif name == "vcum":
                        memory = vcum_by_layer[layer_idx]
                    elif name == "vprefix":
                        memory = vprefix_by_layer[layer_idx]
                    elif name == "vteacher":
                        memory = vteacher_by_layer[layer_idx]
                    else:
                        raise ValueError(name)
                    delta = memory_attention_effect(
                        language_model,
                        layer_idx,
                        teacher_states[layer_idx].to(dtype=dtype),
                        h,
                        memory,
                        full_position_ids,
                        text_position_ids,
                        text_positions,
                        image_positions,
                        full_mask,
                        text_mask,
                        image_mask,
                    )
                h = run_qwen3vl_layer_text_with_attention_delta(
                    language_model,
                    layer_idx,
                    h,
                    text_position_ids,
                    delta,
                    padding_mask=~text_mask,
                )
            logits = model.lm_head(language_model.norm(h))[0, last_text_idx]
            pred = predict_option(logits, option_ids)
            dist = option_distribution(logits, option_ids)
            kl = F.kl_div(dist.clamp_min(1e-8).log(), teacher_dist, reduction="sum").item()
            update_metrics(metrics[name], pred, teacher_pred, gold, kl)
            sample_preds[name] = pred
        predictions.append(sample_preds)
        if (idx + 1) % 25 == 0:
            print(f"evaluated {idx + 1}/{len(rows)} teacher_correct={teacher_correct}", flush=True)

    results = {
        "benchmark": "mmstar",
        "model_path": args.model_path,
        "num_samples": len(rows),
        "teacher": {
            "correct": teacher_correct,
            "accuracy": teacher_correct / max(len(rows), 1),
        },
        "oracles": {name: finalize_metrics(metrics[name], teacher_correct) for name in names},
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions_jsonl:
        pred_out = Path(args.predictions_jsonl)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
