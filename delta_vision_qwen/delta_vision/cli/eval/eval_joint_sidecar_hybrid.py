#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    llava15_prompt,
    read_jsonl,
)
from delta_vision.evaluation.metrics import OPTIONS, option_distribution, option_scores, option_token_id_lists
from delta_vision.models.modeling import build_rollout_model, image_token_id, load_frozen_llava, load_rollout_checkpoint


CHOICE_RE = re.compile(r"(?m)(?:^|\b)([A-D])(?:[.)：:]|\\s*:)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate full joint LLaVA + Shared Sidecar hybrid prompt logits.")
    parser.add_argument("--benchmark", choices=("mmstar", "realworldqa"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sidecar-scales", default="1.0")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--reader-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=256)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--active-layers", default="all")
    parser.add_argument("--ignore-mismatched-checkpoint-shapes", action="store_true")
    parser.add_argument("--slice-mismatched-checkpoint-prefix", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def parse_scales(spec: str) -> list[float]:
    scales = [float(x) for x in spec.split(",") if x.strip()]
    if not scales:
        raise ValueError("--sidecar-scales cannot be empty")
    return scales


def parse_active_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(num_layers))
    if spec.strip() == "":
        return set()
    layers = {int(x) for x in spec.split(",") if x.strip()}
    if any(x < 0 or x >= num_layers for x in layers):
        raise ValueError(f"--active-layers must contain zero-based layer ids in [0, {num_layers - 1}]")
    return layers


def build_prompt(row: dict[str, Any], benchmark: str) -> str:
    question = str(row["question"]).strip()
    if benchmark == "mmstar":
        suffix = "Answer directly with only the letter of the correct option."
    else:
        suffix = ""
    return llava15_prompt(f"{question}\n{suffix}".strip())


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


def run_full_joint_layer_with_text_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    text_positions: torch.Tensor,
    text_delta: torch.Tensor | None,
) -> torch.Tensor:
    layer = get_lm_layers(language_model)[layer_idx]
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    residual = hidden_states
    position_embeddings = rotary_owner.rotary_emb(hidden_states, position_ids)
    normed = layer.input_layernorm(hidden_states)
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    attn_out, _ = layer.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + attn_out
    if text_delta is not None:
        hidden_states = hidden_states.clone()
        hidden_states[:, text_positions] = hidden_states[:, text_positions] + text_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


@torch.inference_mode()
def llava_prompt_logits(
    processor: Any,
    model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    image_token: int,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(row["image"]).convert("RGB")
    try:
        inputs = processor(text=build_prompt(row, benchmark), images=image, return_tensors="pt")
    finally:
        image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    out = model(**inputs, return_dict=True, use_cache=False)
    text_pos, _, _ = get_text_and_image_positions(inputs["input_ids"], out.logits.shape[1], image_token)
    return out.logits[0, int(text_pos[-1].item())]


@torch.inference_mode()
def hybrid_prompt_logits(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    row: dict[str, Any],
    benchmark: str,
    image_token: int,
    device: torch.device,
    dtype: torch.dtype,
    active_layers: set[int],
    sidecar_scale: float,
) -> torch.Tensor:
    sidecar = rollout_model.sidecar
    image = Image.open(row["image"]).convert("RGB")
    try:
        inputs = processor(text=build_prompt(row, benchmark), images=image, return_tensors="pt")
    finally:
        image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach().to(dtype=dtype)
    text_pos, image_pos, _ = get_text_and_image_positions(inputs["input_ids"], hidden0.shape[1], image_token)
    text_pos = text_pos.to(device)
    image_pos = image_pos.to(device)
    vision_tokens = hidden0.index_select(1, image_pos)
    visual_kv = sidecar.prepare_visual_kv(vision_tokens, None)
    state = None
    if sidecar.state_tokens > 0 and sidecar.runtime_use_state:
        state = sidecar.initial_state(vision_tokens, None)

    h = hidden0
    position_ids = torch.arange(h.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(len(get_lm_layers(language_model))):
        delta = None
        if layer_idx in active_layers and sidecar_scale != 0.0:
            layer_tensor = torch.full((1,), layer_idx, device=device, dtype=torch.long)
            text_hidden = h.index_select(1, text_pos)
            if sidecar.state_tokens > 0 and sidecar.runtime_use_state:
                delta, state = sidecar(
                    text_hidden,
                    None,
                    layer_tensor,
                    sidecar_state=state,
                    visual_kv=visual_kv,
                    return_state=True,
                )
            else:
                delta = sidecar(text_hidden, None, layer_tensor, visual_kv=visual_kv)
            delta = delta * sidecar_scale
        h = run_full_joint_layer_with_text_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            text_pos,
            delta,
        )
    logits = model.lm_head(get_lm_norm(language_model)(h))
    return logits[0, int(text_pos[-1].item())]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data, args.max_samples)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    scales = parse_scales(args.sidecar_scales)
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    img_token = image_token_id(model, processor)

    rollout_model = build_rollout_model(args, dtype, device)
    load_rollout_checkpoint(
        rollout_model,
        args.checkpoint,
        ignore_mismatched_checkpoint_shapes=args.ignore_mismatched_checkpoint_shapes,
        slice_mismatched_checkpoint_prefix=args.slice_mismatched_checkpoint_prefix,
    )
    rollout_model.eval()
    for param in rollout_model.sidecar.parameters():
        param.requires_grad_(False)
    rollout_model.sidecar.basis.requires_grad_(False)
    rollout_model.sidecar.runtime_use_state = False
    rollout_model.sidecar.prepare_inference_cache(device, dtype)

    active_layers = parse_active_layers(args.active_layers, args.num_layers)
    metrics: dict[str, dict[str, float | int]] = {
        "llava": {"correct": 0, "scored": 0}
    }
    for scale in scales:
        metrics[f"hybrid_scale_{scale:g}"] = {
            "correct": 0,
            "agree": 0,
            "llava_correct_retention": 0,
            "kl": 0.0,
            "scored": 0,
        }
    predictions = []
    skipped = 0
    llava_correct_total = 0

    for idx, row in enumerate(rows):
        kind = candidate_kind(row, args.benchmark)
        if kind == "skip":
            skipped += 1
            continue
        ids = candidate_ids(processor.tokenizer, kind)
        gold = normalize_gold(row, kind)
        llava_logits = llava_prompt_logits(processor, model, row, args.benchmark, img_token, device)
        llava_pred = predict(llava_logits, ids)
        llava_dist = distribution(llava_logits, ids)
        llava_correct = llava_pred == gold
        llava_correct_total += int(llava_correct)
        metrics["llava"]["correct"] += int(llava_correct)
        metrics["llava"]["scored"] += 1
        sample = {
            "index": row.get("index", idx),
            "gold": gold,
            "kind": kind,
            "llava": llava_pred,
            "llava_correct": llava_correct,
        }
        for scale in scales:
            key = f"hybrid_scale_{scale:g}"
            logits = hybrid_prompt_logits(
                processor,
                model,
                language_model,
                rollout_model,
                row,
                args.benchmark,
                img_token,
                device,
                dtype,
                active_layers,
                scale,
            )
            pred = predict(logits, ids)
            dist = distribution(logits, ids)
            metrics[key]["correct"] += int(pred == gold)
            metrics[key]["agree"] += int(pred == llava_pred)
            metrics[key]["llava_correct_retention"] += int(llava_correct and pred == llava_pred)
            metrics[key]["kl"] += float(F.kl_div(dist.log(), llava_dist, reduction="sum").item())
            metrics[key]["scored"] += 1
            sample[key] = pred
        predictions.append(sample)
        done = idx + 1
        if done % 25 == 0 or done == len(rows):
            print(f"evaluated {done}/{len(rows)} scored={metrics['llava']['scored']} skipped={skipped}", flush=True)

    results = []
    llava_n = max(int(metrics["llava"]["scored"]), 1)
    results.append(
        {
            "setting": "llava",
            "scored": int(metrics["llava"]["scored"]),
            "correct": int(metrics["llava"]["correct"]),
            "accuracy": float(metrics["llava"]["correct"]) / llava_n,
        }
    )
    for scale in scales:
        key = f"hybrid_scale_{scale:g}"
        row_metrics = metrics[key]
        n = max(int(row_metrics["scored"]), 1)
        results.append(
            {
                "setting": key,
                "sidecar_scale": scale,
                "scored": int(row_metrics["scored"]),
                "correct": int(row_metrics["correct"]),
                "accuracy": float(row_metrics["correct"]) / n,
                "llava_agreement": float(row_metrics["agree"]) / n,
                "llava_correct_retention": float(row_metrics["llava_correct_retention"]) / max(llava_correct_total, 1),
                "output_kl_to_llava": float(row_metrics["kl"]) / n,
            }
        )

    payload = {
        "benchmark": args.benchmark,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "active_layers": sorted(active_layers),
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
