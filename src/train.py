"""DeepSpeed ZeRO-2 training: distill LLaVA via vision KV adapter."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.model import (
    PerLayerKVAdapter,
    extract_vision_kv,
    load_frozen_llava,
    student_forward_with_visual_kv,
    teacher_forward,
)
from src.data import VQADataset, OPDDataset, collate_fn


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 1024,
) -> torch.Tensor:
    """KL divergence on top-K teacher logits."""
    k = min(topk, teacher_logits.shape[-1])
    _, indices = teacher_logits.topk(k, dim=-1)
    t_topk = teacher_logits.gather(-1, indices)
    s_topk = student_logits.gather(-1, indices)
    t_prob = F.softmax(t_topk, dim=-1)
    s_logprob = F.log_softmax(s_topk, dim=-1)
    return F.kl_div(s_logprob, t_prob, reduction="batchmean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../delta-vision/models/llava-1.5-7b-hf")
    parser.add_argument("--data", default="../delta-vision/data/pixmo_ama_train.jsonl")
    parser.add_argument("--data-root", default="../delta-vision", help="Root for resolving image paths in JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Load adapter weights from this checkpoint before training")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-answer-tokens", type=int, default=9999)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--source-layers", default="22,23", help="Comma-separated ViT layer indices")
    parser.add_argument("--bottleneck-dim", type=int, default=0, help="If >0, use bottleneck: source_dim -> bottleneck -> target_dim")
    parser.add_argument("--concat-source", action="store_true", help="Concat source layers instead of weighted sum")
    parser.add_argument("--dataset-type", default="vqa", choices=["vqa", "opd"], help="Dataset format")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device=str(device))
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    source_layers = [int(x) for x in args.source_layers.split(",")]
    language_model = model.model.language_model
    num_llm_layers = len(language_model.layers)
    num_heads = getattr(language_model.config, "num_key_value_heads", language_model.config.num_attention_heads)
    head_dim = language_model.config.hidden_size // language_model.config.num_attention_heads
    if is_main:
        print(f"LLM: {num_llm_layers} layers, {num_heads} heads, head_dim={head_dim}")
    adapter = PerLayerKVAdapter(
        num_llm_layers=num_llm_layers,
        num_source_layers=len(source_layers),
        source_dim=1024,
        num_heads=num_heads,
        head_dim=head_dim,
        bottleneck_dim=args.bottleneck_dim,
        concat_source=args.concat_source,
    )

    trainable_params = sum(p.numel() for p in adapter.parameters())
    if is_main:
        print(f"Adapter trainable params: {trainable_params / 1e6:.2f}M")

    wandb_run = None
    if is_main and args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.output_dir).name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "adapter_trainable_params": trainable_params,
                "adapter_trainable_millions": trainable_params / 1e6,
                "world_size": world_size,
            },
        )

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt["state_dict"])
        if is_main:
            print(f"Loaded init checkpoint: {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    engine, optimizer, _, _ = deepspeed.initialize(
        model=adapter,
        optimizer=optimizer,
        config=args.deepspeed_config,
    )

    DatasetCls = OPDDataset if args.dataset_type == "opd" else VQADataset
    dataset = DatasetCls(
        args.data,
        processor,
        data_root=args.data_root,
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
    step = 0

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            B = input_ids.shape[0]
            pixel_values = batch["pixel_values"].to(device) if torch.is_tensor(batch["pixel_values"]) else batch["pixel_values"]
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            with torch.no_grad():
                image_sizes = batch.get("image_sizes")
                if isinstance(pixel_values, list):
                    # Variable crops: process per-sample
                    source_k_list, source_v_list, teacher_logits_list = [], [], []
                    for i in range(B):
                        pv_i = pixel_values[i].unsqueeze(0).to(device)
                        sk, sv = extract_vision_kv(model, pv_i, source_layer_indices=source_layers)
                        source_k_list.append(sk)
                        source_v_list.append(sv)
                        isz = image_sizes[i:i+1].to(device) if image_sizes is not None and torch.is_tensor(image_sizes) else None
                        tl = teacher_forward(model, input_ids[i:i+1], pv_i, image_sizes=isz)
                        teacher_logits_list.append(tl)
                    source_k = source_v = teacher_logits = None
                else:
                    if image_sizes is not None and torch.is_tensor(image_sizes):
                        image_sizes = image_sizes.to(device)
                    source_k, source_v = extract_vision_kv(model, pixel_values, source_layer_indices=source_layers)
                    teacher_logits = teacher_forward(model, input_ids, pixel_values, attention_mask, image_sizes=image_sizes)
                    source_k_list = source_v_list = teacher_logits_list = None

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for i in range(B):
                single_ids = input_ids[i:i+1]
                if source_k_list is not None:
                    single_sk = source_k_list[i]
                    single_sv = source_v_list[i]
                else:
                    single_sk = source_k[i:i+1]
                    single_sv = source_v[i:i+1]

                student_logits = student_forward_with_visual_kv(
                    model, single_ids, engine.module, single_sk, single_sv, image_token_id
                )

                # prompt_len is text-only (image tokens excluded)
                text_prompt_len = int(prompt_lens[i].item())
                num_text = student_logits.shape[1]

                # Student: text-only, answer starts at text_prompt_len-1 (causal: predict next)
                s_start = max(0, text_prompt_len - 1)
                # Exclude pad tokens from answer range
                n_image_tokens = (single_ids[0] == image_token_id).sum().item()
                actual_len = int(attention_mask[i].sum().item()) - n_image_tokens
                s_end = min(s_start + args.max_answer_tokens, actual_len)
                s_answer = student_logits[0, s_start:s_end]

                # Teacher: full sequence with image tokens expanded
                # Answer starts at (576 + text_prompt_len - 1) in teacher space
                n_image_tokens = (single_ids[0] == image_token_id).sum().item()
                if teacher_logits_list is not None:
                    t_logits_i = teacher_logits_list[i][0]
                    t_end = min(t_start + args.max_answer_tokens, t_logits_i.shape[0])
                    t_answer = t_logits_i[t_start:t_end]
                else:
                    t_end = min(t_start + args.max_answer_tokens, teacher_logits.shape[1])
                    t_answer = teacher_logits[i, t_start:t_end]

                if s_answer.shape[0] > 0 and t_answer.shape[0] > 0:
                    min_len = min(s_answer.shape[0], t_answer.shape[0])
                    kl = topk_kl_loss(
                        s_answer[:min_len].float(),
                        t_answer[:min_len].float(),
                        topk=args.kl_topk,
                    )
                    total_loss = total_loss + kl

            loss = total_loss / B
            engine.backward(loss)
            engine.step()

            if is_main and step % args.log_every == 0:
                item = {"step": step, "loss": float(loss.item())}
                print(json.dumps(item), flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(item) + "\n")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": item["loss"], "train/step": step}, step=step)

            if is_main and step > 0 and step % args.save_every == 0:
                ckpt_path = Path(args.output_dir) / f"step_{step}.pt"
                torch.save({
                    "state_dict": engine.module.state_dict(),
                    "step": step,
                    "args": vars(args),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

            step += 1

        if step >= args.max_steps:
            break

    if is_main:
        final_path = Path(args.output_dir) / "final.pt"
        torch.save({
            "state_dict": engine.module.state_dict(),
            "step": step,
            "args": vars(args),
        }, final_path)
        print(f"Training complete. Final checkpoint: {final_path}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
