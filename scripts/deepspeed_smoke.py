#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback
import faulthandler

import deepspeed
import torch
import torch.distributed as dist
from torch import nn


def main() -> None:
    faulthandler.enable()
    print("deepspeed_smoke start", flush=True)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"before torch init local_rank={local_rank}", flush=True)
        dist.init_process_group("nccl")
        print(f"after torch init rank={dist.get_rank()}", flush=True)
        value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
        dist.all_reduce(value)
        if dist.get_rank() == 0:
            print(f"world_size={dist.get_world_size()} all_reduce_sum={value.item()}", flush=True)
        model = nn.Linear(4, 4).cuda()
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
            config={
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": False},
                "zero_optimization": {"stage": 2},
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
            },
        )
        y = engine(torch.ones(1, 4, device="cuda")).sum()
        engine.backward(y)
        engine.step()
        if dist.get_rank() == 0:
            print("deepspeed initialize/step ok", flush=True)
        dist.destroy_process_group()
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
