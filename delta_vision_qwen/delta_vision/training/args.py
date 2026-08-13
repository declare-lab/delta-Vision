from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Online attention Sidecar rollout distillation with ZeRO-2.")
    parser.add_argument("--data", default="data/pixmo_ama_full_valid.jsonl")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--basis", required=True)
    parser.add_argument(
        "--allow-mmstar-basis",
        action="store_true",
        help="Diagnostic-only override for overfitting MMStar traces; keep disabled for PixMo training.",
    )
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-deepspeed-dir", default=None)
    parser.add_argument("--resume-deepspeed-tag", default=None)
    parser.add_argument("--resume-global-step", type=int, default=0)
    parser.add_argument(
        "--ignore-checkpoint-basis",
        action="store_true",
        help="When loading an init checkpoint, keep the basis loaded from --basis instead of checkpoint basis tensors.",
    )
    parser.add_argument(
        "--ignore-mismatched-checkpoint-shapes",
        action="store_true",
        help="Skip init checkpoint tensors whose shape does not match the current Sidecar model.",
    )
    parser.add_argument(
        "--slice-mismatched-checkpoint-prefix",
        action="store_true",
        help=(
            "When loading a wider init checkpoint into a narrower model, slice "
            "mismatched tensors from the leading prefix dimensions instead of skipping them."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2_coeff.json")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--state-tokens", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--reader-mlp-ratio", type=float, default=0.0)
    parser.add_argument("--layer-adapter-rank", type=int, default=0)
    parser.add_argument("--reader-fuse-query", action="store_true")
    parser.add_argument("--reader-concat-query", action="store_true")
    parser.add_argument(
        "--sidecar-output-mode",
        choices=("residual", "factorized_lowrank", "factorized_full"),
        default="residual",
        help="Sidecar output parameterization. factorized_lowrank predicts m*(A_vis_lowrank - A_text).",
    )
    parser.add_argument("--gate-init", type=float, default=1.0)
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--sidecar-scale-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--rollout-teacher-forcing-steps",
        type=int,
        default=0,
        help="Feed teacher hidden states into the next layer for this many optimizer steps after computing losses.",
    )
    parser.add_argument(
        "--rollout-mixed-steps",
        type=int,
        default=0,
        help="Linearly decay teacher hidden-state mixing to zero after teacher forcing.",
    )
    parser.add_argument(
        "--rollout-teacher-mix-start",
        type=float,
        default=0.0,
        help="Initial teacher/student hidden-state mixing value during rollout-mixed-steps.",
    )
    parser.add_argument("--basis-lr-mult", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU dataloader batch size.")
    parser.add_argument(
        "--batched-rollout",
        action="store_true",
        help="Run each dataloader batch as one Teacher/Student batch instead of looping over samples.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--required-world-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches prefetched by each DataLoader worker when num_workers > 0.",
    )
    parser.add_argument(
        "--decode-images-in-workers",
        action="store_true",
        help="Decode PIL images inside dataloader workers instead of the training loop.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--trajectory-layers", default="all")
    parser.add_argument(
        "--active-layers",
        default="all",
        help="Zero-based LLM block indices where Sidecar is called. Use all for the full-depth model.",
    )
    parser.add_argument("--effect-layers-per-sample", type=int, default=2)
    parser.add_argument(
        "--effect-loss-mode",
        choices=("mixed", "coeff", "residual"),
        default="mixed",
        help="Local effect KD target: legacy mixed coefficient+residual, coefficient-only, or residual-only.",
    )
    parser.add_argument(
        "--effect-input-state",
        choices=("teacher", "student"),
        default="student",
        help=(
            "Hidden state used as the Sidecar input for local effect KD. "
            "The Student rollout path always uses Student hidden states and cascades predictions."
        ),
    )
    parser.add_argument(
        "--effect-target-state",
        choices=("teacher", "student"),
        default="student",
        help="State used for sampled attention-effect targets. student reduces rollout/teacher-forcing mismatch.",
    )
    parser.add_argument(
        "--sidecar-token-mode",
        choices=("all", "prompt_only"),
        default="all",
        help="Token positions where Sidecar residual is applied during rollout.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-loss-mode",
        choices=("nmse", "direction"),
        default="nmse",
        help="Trajectory loss target: legacy normalized MSE or RMS-normalized direction matching.",
    )
    parser.add_argument("--lambda-trajectory", type=float, default=1.0)
    parser.add_argument("--lambda-trajectory-cos", type=float, default=0.0)
    parser.add_argument("--lambda-trajectory-rms", type=float, default=0.0)
    parser.add_argument("--trajectory-late-start", type=int, default=24)
    parser.add_argument("--trajectory-late-weight", type=float, default=1.0)
    parser.add_argument("--lambda-logit", type=float, default=0.5)
    parser.add_argument("--lambda-topk-logit", type=float, default=0.0)
    parser.add_argument("--topk-logit-k", type=int, default=64)
    parser.add_argument("--lambda-answer-margin", type=float, default=0.0)
    parser.add_argument("--answer-margin-topk", type=int, default=64)
    parser.add_argument("--lambda-image-negative", type=float, default=0.0)
    parser.add_argument("--image-negative-margin", type=float, default=0.15)
    parser.add_argument("--image-negative-mode", choices=("zero",), default="zero")
    parser.add_argument("--lambda-task", type=float, default=0.0)
    parser.add_argument("--lambda-effect", type=float, default=0.0)
    parser.add_argument("--lambda-effect-cos", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--sync-perf-timing",
        action="store_true",
        help="Synchronize CUDA around profiled training sections for accurate per-step timing.",
    )
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-vision")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()
