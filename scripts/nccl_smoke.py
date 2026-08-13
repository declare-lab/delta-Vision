#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
    dist.all_reduce(value)
    if dist.get_rank() == 0:
        print(f"world_size={dist.get_world_size()} all_reduce_sum={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
