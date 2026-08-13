#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time

import torch

from delta_vision.runtime.basis import reconstruct_delta
from delta_vision.runtime.ops import cross_attention, split_heads


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(name: str, fn, warmup: int, iters: int) -> None:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed * 1000.0 / iters:.3f} ms/iter", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark Sidecar low-level ops.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--text-len", type=int, default=96)
    parser.add_argument("--vision-len", type=int, default=576)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--sidecar-dim", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    coeff = torch.randn(args.batch, args.text_len, args.rank, device=device, dtype=dtype)
    basis = torch.randn(args.batch, args.rank, args.hidden_size, device=device, dtype=dtype)
    q = torch.randn(args.batch, args.text_len, args.sidecar_dim, device=device, dtype=dtype)
    k = split_heads(
        torch.randn(args.batch, args.vision_len, args.sidecar_dim, device=device, dtype=dtype),
        args.heads,
    ).transpose(1, 2).contiguous()
    v = split_heads(
        torch.randn(args.batch, args.vision_len, args.sidecar_dim, device=device, dtype=dtype),
        args.heads,
    ).transpose(1, 2).contiguous()

    print(
        f"device={device} dtype={dtype} batch={args.batch} text={args.text_len} "
        f"vision={args.vision_len} rank={args.rank} hidden={args.hidden_size} "
        f"triton_basis={os.environ.get('VISUAL_SIDECAR_USE_TRITON_BASIS', '0')}",
        flush=True,
    )
    bench("basis_reconstruct", lambda: reconstruct_delta(coeff, basis), args.warmup, args.iters)
    bench(
        "sidecar_cross_attention",
        lambda: cross_attention(split_heads(q, args.heads), k, v),
        args.warmup,
        args.iters,
    )


if __name__ == "__main__":
    main()
