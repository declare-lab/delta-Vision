#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.evaluation.mmstar import TraceEvalConfig, evaluate_mmstar_traces
from delta_vision.models.modeling import build_rollout_model, load_frozen_llava, load_rollout_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate attention-level coefficient Sidecar rollout on MMStar traces.")
    parser.add_argument("--effects-dir", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=0)
    parser.add_argument("--reader-mlp-ratio", type=float, default=0.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=0)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument("--sidecar-output-mode", choices=("residual", "factorized_lowrank", "factorized_full"), default="residual")
    parser.add_argument("--ignore-mismatched-checkpoint-shapes", action="store_true")
    parser.add_argument("--slice-mismatched-checkpoint-prefix", action="store_true")
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--active-layers", default="all")
    parser.add_argument("--hidden-distance-layers", default="0,8,16,24,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_llava(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)

    rollout_model = build_rollout_model(args, dtype, device)
    load_rollout_checkpoint(
        rollout_model,
        args.checkpoint,
        ignore_mismatched_checkpoint_shapes=args.ignore_mismatched_checkpoint_shapes,
        slice_mismatched_checkpoint_prefix=args.slice_mismatched_checkpoint_prefix,
    )
    rollout_model.eval()

    evaluate_mmstar_traces(
        processor=processor,
        model=model,
        language_model=language_model,
        rollout_model=rollout_model,
        config=TraceEvalConfig(
            effects_dir=Path(args.effects_dir),
            output_json=Path(args.output_json),
            start_sample=args.start_sample,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            hidden_distance_layers=args.hidden_distance_layers,
            active_layers=args.active_layers,
            sidecar_scale=args.sidecar_scale,
        ),
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
