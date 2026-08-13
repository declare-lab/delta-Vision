#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from delta_vision.models.llava import (
    compute_llama_attention_effect,
    dtype_from_name,
    get_language_model,
    get_lm_layers,
    get_text_and_image_positions,
    read_jsonl,
)
from delta_vision.runtime.rollout import prepare_sample_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect sampled attention-delta tokens online.")
    parser.add_argument("--data", default="data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=125)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tokens-per-layer", type=int, default=2048)
    parser.add_argument("--token-mode", choices=("all", "answer", "prompt"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = read_jsonl(args.data, None)[args.start_index :]
    rows = rows[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    language_model = get_language_model(model)
    num_layers = len(get_lm_layers(language_model))
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    banks: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(num_layers)}
    counts = {layer: 0 for layer in range(num_layers)}
    for sample_idx, row in enumerate(rows):
        inputs, _, answer_mask, _ = prepare_sample_inputs(
            processor,
            row,
            "image",
            "question",
            "answer",
            None,
            image_token_id,
            device,
        )
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = tuple(x.detach() for x in outputs.hidden_states[:-1])
        merged_len = hidden_states[0].shape[1]
        text_positions, _, _ = get_text_and_image_positions(inputs["input_ids"], merged_len, image_token_id)
        text_positions = text_positions.to(device)

        for layer_idx in range(num_layers):
            if counts[layer_idx] >= args.max_tokens_per_layer:
                continue
            delta = compute_llama_attention_effect(
                language_model,
                layer_idx,
                hidden_states[layer_idx].to(dtype=dtype),
                text_positions,
            ).squeeze(0)
            if args.token_mode == "answer":
                keep = answer_mask[0].to(device=delta.device)
                delta = delta.index_select(0, torch.nonzero(keep, as_tuple=False).flatten())
            elif args.token_mode == "prompt":
                keep = ~answer_mask[0].to(device=delta.device)
                delta = delta.index_select(0, torch.nonzero(keep, as_tuple=False).flatten())
            if delta.shape[0] == 0:
                continue
            remaining = args.max_tokens_per_layer - counts[layer_idx]
            if delta.shape[0] > remaining:
                idx = torch.randperm(delta.shape[0], device=delta.device)[:remaining]
                delta = delta.index_select(0, idx)
            banks[layer_idx].append(delta.detach().cpu().to(torch.float16))
            counts[layer_idx] += delta.shape[0]

        if (sample_idx + 1) % 5 == 0:
            print(f"processed {sample_idx + 1}/{len(rows)}", flush=True)
        if all(counts[layer] >= args.max_tokens_per_layer for layer in range(num_layers)):
            print(f"token banks full after {sample_idx + 1}/{len(rows)} samples", flush=True)
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": args.data,
            "model_path": args.model_path,
            "start_index": args.start_index,
            "max_samples": args.max_samples,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "token_mode": args.token_mode,
            "tokens": {
                layer: torch.cat(parts, dim=0) if parts else torch.empty(0, 4096, dtype=torch.float16)
                for layer, parts in banks.items()
            },
            "counts": counts,
        },
        output,
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
