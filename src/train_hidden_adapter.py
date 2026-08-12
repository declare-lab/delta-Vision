"""Train HiddenStateAdapter (Method 2): adapter -> LLM frozen k_proj/v_proj."""
import argparse, json, os, random
from pathlib import Path
import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.insert(0, ".")
from src.model import extract_vision_kv_qwen
from src.hidden_adapter import HiddenStateAdapter, student_forward_hidden_adapter
from src.train_qwen import QwenVQADataset, collate_fn, topk_kl_loss
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
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

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_dim = model.model.language_model.config.hidden_size
    num_layers = len(model.model.language_model.layers)
    adapter = HiddenStateAdapter(source_dim=1024, hidden_dim=hidden_dim, num_source_layers=2, num_llm_layers=num_layers)

    trainable = sum(p.numel() for p in adapter.parameters())
    if is_main:
        print(f"Adapter trainable: {trainable/1e6:.1f}M, hidden_dim={hidden_dim}, layers={num_layers}")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    engine, optimizer, _, _ = deepspeed.initialize(model=adapter, optimizer=optimizer, config=args.deepspeed_config)

    wandb_run = None
    if is_main and args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or Path(args.output_dir).name, config=vars(args))

    dataset = QwenVQADataset(args.data, processor, args.data_root, shuffle=True, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    image_token_id = 151655
    metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
    step = 0
    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break
            input_ids = batch["input_ids"][0].unsqueeze(0).to(device)
            pixel_values = batch["pixel_values"][0].to(device)
            grid_thw = batch["image_grid_thw"][0].to(device)
            prompt_len = batch["prompt_lens"][0]

            with torch.no_grad():
                source_k, _ = extract_vision_kv_qwen(model, pixel_values, grid_thw, [22, 23])
                mm_ids = batch["mm_token_type_ids"][0]
                mm_ids = mm_ids.unsqueeze(0).to(device) if mm_ids is not None else None
                teacher_out = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, mm_token_type_ids=mm_ids)
                teacher_logits = teacher_out.logits

            student_logits = student_forward_hidden_adapter(model, input_ids, engine.module, source_k, image_token_id, image_grid_thw=grid_thw)

            text_mask = input_ids[0] != image_token_id
            num_text = student_logits.shape[1]
            s_start = max(0, prompt_len - 1)
            s_end = num_text
            s_answer = student_logits[0, s_start:s_end]

            n_img = (input_ids[0] == image_token_id).sum().item()
            t_start = n_img + prompt_len - 1
            t_end = teacher_logits.shape[1]
            t_answer = teacher_logits[0, t_start:t_end]

            if s_answer.shape[0] > 0 and t_answer.shape[0] > 0:
                min_len = min(s_answer.shape[0], t_answer.shape[0])
                loss = topk_kl_loss(s_answer[:min_len].float(), t_answer[:min_len].float(), 1024, 1.0)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            engine.backward(loss)
            engine.step()

            if is_main and step % args.log_every == 0:
                item = {"step": step, "loss": float(loss.item())}
                print(json.dumps(item), flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(item) + "\n")
                if wandb_run:
                    wandb_run.log({"train/loss": item["loss"]}, step=step)
            if is_main and step > 0 and step % args.save_every == 0:
                torch.save({"state_dict": engine.module.state_dict(), "step": step}, Path(args.output_dir) / f"step_{step}.pt")
            step += 1
        if step >= args.max_steps:
            break
    if is_main:
        torch.save({"state_dict": engine.module.state_dict(), "step": step}, Path(args.output_dir) / "final.pt")
        print("Training complete.")
    if wandb_run:
        wandb_run.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
