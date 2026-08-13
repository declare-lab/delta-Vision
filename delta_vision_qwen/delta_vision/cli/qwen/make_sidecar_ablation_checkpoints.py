#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create inference-only ablation checkpoints for a Qwen Sidecar checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ablations",
        default="full,no_visual_transform,no_reader_mlp,fixed_mass_012,low_mass_003,half_gate",
        help="Comma-separated ablations to write.",
    )
    return parser.parse_args()


def _clone_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in ckpt.items():
        if key == "state_dict":
            out[key] = {name: tensor.detach().cpu().clone() for name, tensor in value.items()}
        elif key == "args" and isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def _zero_prefix(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> int:
    count = 0
    for key, tensor in state_dict.items():
        if key.startswith(prefixes) and torch.is_tensor(tensor):
            tensor.zero_()
            count += 1
    return count


def _set_fixed_mass(state_dict: dict[str, torch.Tensor], mass: float) -> None:
    mass = min(max(float(mass), 1e-6), 1.0 - 1e-6)
    bias = math.log(mass / (1.0 - mass))
    if "mass_head.weight" in state_dict:
        state_dict["mass_head.weight"].zero_()
    if "mass_head.bias" in state_dict:
        state_dict["mass_head.bias"].fill_(bias)


def apply_ablation(ckpt: dict[str, Any], ablation: str) -> dict[str, Any]:
    out = _clone_checkpoint(ckpt)
    state_dict = out["state_dict"]
    args = out.setdefault("args", {})
    args["inference_ablation"] = ablation

    if ablation == "full":
        return out
    if ablation == "no_visual_transform":
        # layer_kv_adapter is residual: V_l = V0 + Up(act(Down(V0))).
        # Zeroing Up makes this exactly static V0 while preserving checkpoint shape.
        changed = _zero_prefix(state_dict, ("visual_adapter_up.", "visual_stage_up.", "visual_cascade_up."))
        args["ablation_note"] = f"zeroed {changed} visual transform up-projection tensors"
        return out
    if ablation == "no_reader_mlp":
        changed = _zero_prefix(state_dict, ("reader_mlp.",))
        args["ablation_note"] = f"zeroed {changed} reader MLP tensors"
        return out
    if ablation == "fixed_mass_012":
        _set_fixed_mass(state_dict, 0.12)
        args["ablation_note"] = "mass_head outputs constant sigmoid mass 0.12"
        return out
    if ablation == "low_mass_003":
        _set_fixed_mass(state_dict, 0.03)
        args["ablation_note"] = "mass_head outputs constant sigmoid mass 0.03"
        return out
    if ablation == "half_gate":
        if "gate" in state_dict:
            state_dict["gate"].mul_(0.5)
        args["ablation_note"] = "multiplied per-layer residual gate by 0.5"
        return out
    if ablation == "zero_gate":
        if "gate" in state_dict:
            state_dict["gate"].zero_()
        args["ablation_note"] = "zeroed per-layer residual gate"
        return out
    raise ValueError(f"unsupported ablation: {ablation}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        ckpt = {"state_dict": ckpt, "args": {}}
    manifest = []
    for ablation in [item.strip() for item in args.ablations.split(",") if item.strip()]:
        out = apply_ablation(ckpt, ablation)
        out_path = output_dir / f"{Path(args.checkpoint).stem}.{ablation}.pt"
        torch.save(out, out_path)
        manifest.append({"ablation": ablation, "checkpoint": str(out_path), "args": out.get("args", {})})
        print(f"{ablation}\t{out_path}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
