#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build attention-delta PCA basis from token shards.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rank", type=int, default=768)
    parser.add_argument("--max-tokens-per-layer", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    shard_paths = sorted(Path(args.shard_dir).glob("tokens_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no tokens_shard_*.pt files found in {args.shard_dir}")

    loaded = [torch.load(path, map_location="cpu") for path in shard_paths]
    layers = sorted(int(layer) for layer in loaded[0]["tokens"])
    bases = {}
    energies = {}
    counts = {}
    for layer in layers:
        matrix = torch.cat([shard["tokens"][layer].float() for shard in loaded], dim=0)
        if matrix.shape[0] > args.max_tokens_per_layer:
            idx = torch.randperm(matrix.shape[0])[: args.max_tokens_per_layer]
            matrix = matrix.index_select(0, idx)
        counts[layer] = int(matrix.shape[0])
        matrix = matrix.to(args.device)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        _, svals, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(args.max_rank, vh.shape[0])
        bases[layer] = vh[:rank].cpu().to(torch.float16)
        energies[layer] = svals.float().cpu().pow(2)
        print(f"layer={layer} tokens={matrix.shape[0]} rank={rank}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "shards": [str(path) for path in shard_paths],
            "layers": layers,
            "max_rank": args.max_rank,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "counts": counts,
            "basis": bases,
            "singular_energy": energies,
        },
        output,
    )
    (output.with_suffix(".metrics.json")).write_text(
        json.dumps(
            {
                "layers": layers,
                "counts": counts,
                "ranks": {str(layer): int(bases[layer].shape[0]) for layer in layers},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
