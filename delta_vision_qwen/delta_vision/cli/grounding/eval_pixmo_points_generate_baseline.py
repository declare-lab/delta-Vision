#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache

from delta_vision.data import JsonlDataset, collate_rows
from delta_vision.models.llava import (
    build_llava_initial_hidden,
    dtype_from_name,
    get_language_model,
    get_lm_embed_tokens,
    get_lm_layers,
    get_lm_norm,
    get_text_and_image_positions,
    run_llama_layer_text_with_attention_delta,
    run_llama_layer_text_with_attention_delta_cache,
)
from delta_vision.models.modeling import (
    build_rollout_model,
    image_token_id,
    load_frozen_llava,
    load_rollout_checkpoint,
)
from delta_vision.grounding.pointing import parse_points, point_distance, point_in_masks_xy100


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_active_layers(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(num_layers))
    values = {int(x) for x in spec.split(",") if x.strip()}
    if any(x < 0 or x >= num_layers for x in values):
        raise ValueError(f"--active-layers must contain zero-based layer ids in [0, {num_layers - 1}]")
    return values


def parse_xy(text: str) -> tuple[float, float] | None:
    tail = text.rsplit("ASSISTANT:", 1)[-1]
    values = [float(x) for x in NUMBER_RE.findall(tail)]
    if len(values) < 2:
        return None
    x = max(0.0, min(100.0, values[0]))
    y = max(0.0, min(100.0, values[1]))
    return x, y


def prompt(label: str) -> str:
    return (
        "USER: <image>\n"
        f"Point to the {str(label).strip()}. "
        "Answer with only two numbers: x y. Use 0-100 image coordinates, "
        "where 0 0 is top-left and 100 100 is bottom-right.\n"
        "ASSISTANT:"
    )


def mark_compiled_sidecar_step() -> None:
    if torch.cuda.is_available():
        torch.compiler.cudagraph_mark_step_begin()


def layer_id_tensor(layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.full((hidden_states.shape[0],), layer_idx, device=hidden_states.device, dtype=torch.long)


def make_layer_id_tensors(num_layers: int, hidden_states: torch.Tensor) -> list[torch.Tensor]:
    return [layer_id_tensor(layer_idx, hidden_states) for layer_idx in range(num_layers)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate coordinate-generation baselines on PixMo-Points-Eval.")
    parser.add_argument("--data", default="data/pixmo_points/eval_test.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--mode", choices=("llava", "blank", "center", "sidecar"), default="llava")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--basis", default="artifacts/basis/delta_attn_pca_rank768.pt")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=24)
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
    parser.add_argument("--prefill-active-layers", default=None)
    parser.add_argument("--decode-active-layers", default=None)
    parser.add_argument("--disable-sidecar-state", action="store_true")
    parser.add_argument("--use-sidecar-state", action="store_true")
    parser.add_argument("--no-freeze-basis-for-inference", action="store_true")
    parser.add_argument("--no-compile-sidecar", action="store_true")
    parser.add_argument("--static-sidecar-layers", action="store_true")
    parser.add_argument("--fold-sidecar-output-basis", action="store_true")
    parser.add_argument("--hf-decode-when-no-sidecar", action="store_true")
    return parser.parse_args()


def sidecar_logits(
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    initial_text_hidden: torch.Tensor,
    generated_ids: list[int],
    prompt_position_ids: torch.Tensor,
    vision_tokens: torch.Tensor,
    active_layers: set[int] | None = None,
) -> torch.Tensor:
    if active_layers is None:
        active_layers = set(range(len(get_lm_layers(language_model))))
    sidecar = rollout_model.sidecar
    if generated_ids:
        token_ids = torch.tensor([generated_ids], device=initial_text_hidden.device, dtype=torch.long)
        gen_hidden = get_lm_embed_tokens(language_model)(token_ids).to(dtype=initial_text_hidden.dtype)
        h = torch.cat([initial_text_hidden, gen_hidden], dim=1)
        start = int(prompt_position_ids[0, -1].item()) + 1
        gen_pos = torch.arange(start, start + len(generated_ids), device=prompt_position_ids.device).unsqueeze(0)
        position_ids = torch.cat([prompt_position_ids, gen_pos], dim=1)
    else:
        h = initial_text_hidden
        position_ids = prompt_position_ids

    visual_kv = sidecar.prepare_visual_kv(vision_tokens, None)
    state = sidecar.initial_state(vision_tokens, None) if sidecar.state_tokens > 0 and sidecar.runtime_use_state else None
    lm_layers = list(get_lm_layers(language_model))
    lm_norm = get_lm_norm(language_model)
    num_layers = len(lm_layers)
    layer_tensors = make_layer_id_tensors(num_layers, h)
    for layer_idx in range(num_layers):
        if layer_idx not in active_layers:
            delta = None
        elif sidecar.state_tokens > 0 and sidecar.runtime_use_state:
            mark_compiled_sidecar_step()
            layer_arg = layer_tensors[layer_idx]
            delta, state = sidecar(
                h,
                None,
                layer_arg,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
            )
            state = state.clone()
        else:
            mark_compiled_sidecar_step()
            delta = sidecar(h, None, layer_tensors[layer_idx], visual_kv=visual_kv)
        h = run_llama_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            position_ids,
            attention_delta=delta,
        )
    return model.lm_head(lm_norm(h))


def sidecar_generate_one(
    processor: Any,
    model: torch.nn.Module,
    language_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    row: dict[str, Any],
    img_token: int,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    prefill_active_layers: set[int],
    decode_active_layers: set[int],
    hf_decode_when_no_sidecar: bool,
) -> str:
    image = Image.open(row["image"]).convert("RGB")
    try:
        inputs = processor(text=prompt(str(row["label"])), images=image, return_tensors="pt")
    finally:
        image.close()
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    hidden0 = build_llava_initial_hidden(
        model,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes"),
    ).detach()
    text_pos, image_pos, prompt_positions = get_text_and_image_positions(
        inputs["input_ids"],
        hidden0.shape[1],
        img_token,
    )
    initial_text_hidden = hidden0.index_select(1, text_pos.to(device)).to(dtype=dtype)
    vision_tokens = hidden0.index_select(1, image_pos.to(device)).to(dtype=dtype)
    prompt_position_ids = prompt_positions.to(device).unsqueeze(0)

    sidecar = rollout_model.sidecar
    visual_kv = sidecar.prepare_visual_kv(vision_tokens, None)
    state0 = sidecar.initial_state(vision_tokens, None) if sidecar.state_tokens > 0 and sidecar.runtime_use_state else None
    cache = DynamicCache(config=language_model.config)
    h = initial_text_hidden
    cache_position = torch.arange(h.shape[1], device=h.device)
    state = state0
    lm_layers = list(get_lm_layers(language_model))
    rotary_owner = language_model.model if hasattr(language_model, "model") else language_model
    lm_embed = get_lm_embed_tokens(language_model)
    lm_norm = get_lm_norm(language_model)
    num_layers = len(lm_layers)
    layer_tensors = make_layer_id_tensors(num_layers, h)
    position_embeddings = rotary_owner.rotary_emb(h, prompt_position_ids)
    for layer_idx in range(num_layers):
        if layer_idx not in prefill_active_layers:
            delta = None
        elif sidecar.state_tokens > 0 and sidecar.runtime_use_state:
            mark_compiled_sidecar_step()
            layer_arg = layer_tensors[layer_idx]
            delta, state = sidecar(
                h,
                None,
                layer_arg,
                sidecar_state=state,
                visual_kv=visual_kv,
                return_state=True,
            )
            state = state.clone()
        else:
            mark_compiled_sidecar_step()
            delta = sidecar(h, None, layer_tensors[layer_idx], visual_kv=visual_kv)
        h = run_llama_layer_text_with_attention_delta_cache(
            language_model,
            layer_idx,
            h,
            prompt_position_ids,
            cache_position,
            cache,
            attention_delta=delta,
            layer=lm_layers[layer_idx],
            rotary_owner=rotary_owner,
            position_embeddings=position_embeddings,
        )
    logits = model.lm_head(lm_norm(h))
    next_id = int(logits[0, -1].float().argmax().item())

    generated: list[int] = []
    eos = processor.tokenizer.eos_token_id
    decode_layer_tensors = make_layer_id_tensors(len(lm_layers), initial_text_hidden[:, :1])
    for _ in range(max_new_tokens):
        if next_id == eos:
            break
        generated.append(next_id)
        token = processor.tokenizer.decode([next_id], skip_special_tokens=True)
        if "\n" in token and generated:
            break
        position_id = int(prompt_position_ids[0, -1].item()) + len(generated)
        if not decode_active_layers and hf_decode_when_no_sidecar:
            token_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            pos = torch.tensor([[position_id]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([position_id], device=device, dtype=torch.long)
            outputs = language_model(
                input_ids=token_tensor,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
                return_dict=True,
            )
            logits = model.lm_head(outputs.last_hidden_state)
        else:
            token_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            h = lm_embed(token_tensor).to(dtype=initial_text_hidden.dtype)
            pos = torch.tensor([[position_id]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([position_id], device=device, dtype=torch.long)
            position_embeddings = rotary_owner.rotary_emb(h, pos)
            state = state0
            sidecar_decode = getattr(rollout_model, "sidecar_decode", sidecar)
            static_sidecar_layers = getattr(rollout_model, "sidecar_decode_layers", None)
            layer_tensors = decode_layer_tensors
            for layer_idx in range(num_layers):
                if layer_idx not in decode_active_layers:
                    delta = None
                elif sidecar.state_tokens > 0 and sidecar.runtime_use_state:
                    mark_compiled_sidecar_step()
                    layer_arg = layer_tensors[layer_idx]
                    delta, state = sidecar_decode(
                        h,
                        None,
                        layer_arg,
                        sidecar_state=state,
                        visual_kv=visual_kv,
                        return_state=True,
                    )
                    state = state.clone()
                else:
                    mark_compiled_sidecar_step()
                    if static_sidecar_layers is not None:
                        delta = static_sidecar_layers[layer_idx](h, visual_kv)
                    elif getattr(rollout_model, "sidecar_decode_no_state", False):
                        delta = sidecar_decode(h, layer_tensors[layer_idx], visual_kv)
                    else:
                        delta = sidecar_decode(h, None, layer_tensors[layer_idx], visual_kv=visual_kv)
                h = run_llama_layer_text_with_attention_delta_cache(
                    language_model,
                    layer_idx,
                    h,
                    pos,
                    cache_pos,
                    cache,
                    attention_delta=delta,
                    layer=lm_layers[layer_idx],
                    rotary_owner=rotary_owner,
                    position_embeddings=position_embeddings,
                )
            logits = model.lm_head(lm_norm(h))
        next_id = int(logits[0, -1].float().argmax().item())
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dataset = JsonlDataset(args.data, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
        pin_memory=True,
    )
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor = None
    model = None
    language_model = None
    rollout_model = None
    img_token = None
    if args.mode != "center":
        processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
        language_model = get_language_model(model)
        img_token = image_token_id(model, processor)
    if args.mode == "sidecar":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode sidecar")
        rollout_model = build_rollout_model(args, dtype, device)
        load_rollout_checkpoint(rollout_model, args.checkpoint)
        rollout_model.eval()
        for param in rollout_model.sidecar.parameters():
            param.requires_grad_(False)
        if not args.no_freeze_basis_for_inference:
            rollout_model.sidecar.basis.requires_grad_(False)
        rollout_model.sidecar.prepare_inference_cache(device, dtype)
        rollout_model.sidecar.runtime_fold_output_basis = bool(args.fold_sidecar_output_basis)
        rollout_model.sidecar.runtime_use_state = bool(args.use_sidecar_state and not args.disable_sidecar_state)
        if not args.no_compile_sidecar:
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
            if rollout_model.sidecar.runtime_use_state:
                rollout_model.sidecar_decode = torch.compile(rollout_model.sidecar, mode="reduce-overhead")
                rollout_model.sidecar_decode_no_state = False
            else:
                rollout_model.sidecar_decode = torch.compile(rollout_model.sidecar.decode_no_state, mode="reduce-overhead")
                rollout_model.sidecar_decode_no_state = True
        if args.static_sidecar_layers and not rollout_model.sidecar.runtime_use_state:
            def make_static_sidecar_layer(layer_idx: int):
                def static_sidecar_layer(hidden_states: torch.Tensor, visual_kv: Any) -> torch.Tensor:
                    return rollout_model.sidecar.decode_no_state_layer(hidden_states, layer_idx, visual_kv)

                return static_sidecar_layer

            rollout_model.sidecar_decode_layers = [
                make_static_sidecar_layer(layer_idx)
                for layer_idx in range(len(get_lm_layers(language_model)))
            ]
    prefill_active_layers = set()
    decode_active_layers = set()
    if args.mode == "sidecar":
        assert language_model is not None
        base_active_layers = parse_active_layers(args.active_layers, len(get_lm_layers(language_model)))
        prefill_active_layers = (
            parse_active_layers(args.prefill_active_layers, len(get_lm_layers(language_model)))
            if args.prefill_active_layers is not None
            else base_active_layers
        )
        decode_active_layers = (
            parse_active_layers(args.decode_active_layers, len(get_lm_layers(language_model)))
            if args.decode_active_layers is not None
            else base_active_layers
        )

    total = 0
    parsed = 0
    correct = 0
    dist_sum = 0.0
    predictions: list[dict[str, Any]] = []
    for batch_idx, rows in enumerate(loader):
        if args.mode == "center":
            decoded = ["50 50"] * len(rows)
        elif args.mode == "sidecar":
            assert processor is not None and model is not None and language_model is not None
            assert rollout_model is not None and img_token is not None
            decoded = [
                sidecar_generate_one(
                    processor,
                    model,
                    language_model,
                    rollout_model,
                    row,
                    img_token,
                    args.max_new_tokens,
                    device,
                    dtype,
                    prefill_active_layers,
                    decode_active_layers,
                    args.hf_decode_when_no_sidecar,
                )
                for row in rows
            ]
        else:
            assert processor is not None and model is not None
            images = []
            prompts = []
            try:
                for row in rows:
                    prompts.append(prompt(str(row["label"])))
                    if args.mode == "blank":
                        images.append(Image.new("RGB", (336, 336), (127, 127, 127)))
                    else:
                        images.append(Image.open(row["image"]).convert("RGB"))
                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            finally:
                for image in images:
                    image.close()
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for text, row in zip(decoded, rows, strict=True):
            xy = parse_xy(text)
            if xy is None:
                xy = (50.0, 50.0)
                parsed_ok = False
            else:
                parsed_ok = True
                parsed += 1
            x, y = xy
            points = parse_points(row["points"])
            pred_tensor = torch.tensor([x, y], device=device)
            dist = float(point_distance(pred_tensor, points.to(device)).cpu().item())
            in_mask = point_in_masks_xy100((x, y), row["mask_path"]) if row.get("mask_path") else False
            total += 1
            correct += int(in_mask)
            dist_sum += dist
            predictions.append(
                {
                    "index": row.get("index", total - 1),
                    "label": row["label"],
                    "prediction": {"x": x, "y": y},
                    "parsed": parsed_ok,
                    "raw_output": text,
                    "points": row["points"],
                    "point_distance": dist,
                    "point_in_mask": in_mask,
                    "image": row["image"],
                }
            )
        if total % 25 == 0 or batch_idx == len(loader) - 1:
            print(
                f"evaluated {total}/{len(dataset)} acc={correct / max(total, 1):.4f} "
                f"parse={parsed / max(total, 1):.4f}",
                flush=True,
            )

    metrics = {
        "data": args.data,
        "mode": args.mode,
        "num_samples": total,
        "parsed": parsed,
        "parse_rate": parsed / max(total, 1),
        "point_in_mask_correct": correct,
        "point_in_mask_accuracy": correct / max(total, 1),
        "mean_point_distance_xy100": dist_sum / max(total, 1),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_path = Path(args.predictions_jsonl) if args.predictions_jsonl else out.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
