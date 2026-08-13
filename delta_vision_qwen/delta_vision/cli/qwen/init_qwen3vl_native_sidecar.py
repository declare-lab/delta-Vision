#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from delta_vision.models.llava import dtype_from_name, get_language_model
from delta_vision.models.qwen3vl import load_frozen_qwen3vl
from delta_vision.runtime.qwen_analytic_sidecar import QwenNativeAttentionSidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Initialize a Qwen native-attention sidecar gate checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-init", type=float, default=1.0)
    parser.add_argument("--train-gates", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    _, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = get_language_model(model)
    sidecar = QwenNativeAttentionSidecar(
        num_layers=len(language_model.layers),
        gate_init=args.gate_init,
        train_gates=args.train_gates,
    )
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in sidecar.state_dict().items()},
        "args": {
            "model_path": args.model_path,
            "gate_init": args.gate_init,
            "train_gates": args.train_gates,
            "sidecar_backend": "native_attention",
            "visual_memory_mode": "vprefix",
            "num_layers": len(language_model.layers),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(payload["args"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), **payload["args"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
