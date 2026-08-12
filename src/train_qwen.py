"""DeepSpeed ZeRO-2 training for Qwen3-VL vision KV adapter."""
import argparse, json, os, random
from pathlib import Path
import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import sys
sys.path.insert(0, ".")
from src.model import PerLayerKVAdapter, extract_vision_kv_qwen, student_forward_qwen


def topk_kl_loss(student_logits, teacher_logits, topk=1024):
    k = min(topk, teacher_logits.shape[-1])
    _, indices = teacher_logits.topk(k, dim=-1)
    t_topk = teacher_logits.gather(-1, indices)
    s_topk = student_logits.gather(-1, indices)
    return F.kl_div(F.log_softmax(s_topk, dim=-1), F.softmax(t_topk, dim=-1), reduction="batchmean")


class QwenVQADataset(Dataset):
    def __init__(self, jsonl_path, processor, data_root, max_samples=None, shuffle=False, seed=42):
        self.processor = processor
        self.data_root = Path(data_root)
        with open(jsonl_path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        if shuffle:
            random.Random(seed).shuffle(self.rows)
        if max_samples:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.data_root / row["image"]).convert("RGB")
        question = row["question"]
        answer = row.get("answer", "")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = text + " " + answer
        inputs = self.processor(text=[full_text], images=[img], return_tensors="pt", padding=True)
        prompt_inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        # text-only prompt len
        image_token_id = 151655
        prompt_len = prompt_inputs["input_ids"].shape[1] - (prompt_inputs["input_ids"] == image_token_id).sum().item()
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "prompt_len": prompt_len, "mm_token_type_ids": inputs.get("mm_token_type_ids", [None])[0] if "mm_token_type_ids" in inputs else None,
        }


def collate_fn(batch):
    # Qwen pixel_values vary in size, keep as list
    return {
        "input_ids": [item["input_ids"] for item in batch],
        "pixel_values": [item["pixel_values"] for item in batch],
        "image_grid_thw": [item["image_grid_thw"] for item in batch],
        "prompt_lens": [item["prompt_len"] for item in batch], "mm_token_type_ids": [item.get("mm_token_type_ids") for item in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="../delta-vision/data/pixmo_ama_train.jsonl")
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-answer-tokens", type=int, default=9999)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-kv-mse", type=float, default=0.5)
    parser.add_argument("--use-activation", action="store_true")
    parser.add_argument("--bottleneck-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-run-name", default="")
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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lm = model.model.language_model
    num_llm_layers = len(lm.layers)
    num_kv_heads = lm.config.num_key_value_heads
    head_dim = lm.layers[0].self_attn.head_dim
    if is_main:
        print(f"LLM: {num_llm_layers} layers, {num_kv_heads} kv_heads, head_dim={head_dim}")

    adapter = PerLayerKVAdapter(bottleneck_dim=args.bottleneck_dim, use_activation=args.use_activation,
        num_llm_layers=num_llm_layers, num_source_layers=2,
        source_dim=4096, num_heads=num_kv_heads, head_dim=head_dim,
    )
    trainable = sum(p.numel() for p in adapter.parameters())
    if is_main:
        print(f"Adapter: {trainable/1e6:.1f}M params")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt["state_dict"])
        if is_main:
            print(f"Loaded init checkpoint: {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    engine, optimizer, _, _ = deepspeed.initialize(model=adapter, optimizer=optimizer, config=args.deepspeed_config)

    wandb_run = None
    if is_main and args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or Path(args.output_dir).name,
                               config={**vars(args), "params_M": trainable/1e6})

    dataset = QwenVQADataset(args.data, processor, args.data_root, max_samples=args.max_samples, shuffle=True, seed=args.seed)
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
                source_k, source_v = extract_vision_kv_qwen(model, pixel_values, grid_thw, [22, 23])
                # Hook teacher LLM layers to get KV at image positions
                teacher_kvs = {}
                hooks = []
                lm_layers = model.model.language_model.layers
                img_mask = (input_ids[0] == image_token_id)
                for layer_idx in range(len(lm_layers)):
                    def make_hook(idx):
                        def hook_fn(module, input, output):
                            h = input[0] if isinstance(input, tuple) else input
                            attn = lm_layers[idx].self_attn
                            # Get KV at image positions
                            img_h = h[:, img_mask]
                            k = attn.k_proj(img_h)
                            v = attn.v_proj(img_h)
                            teacher_kvs[idx] = (k.float().detach(), v.float().detach())
                        return hook_fn
                    h = lm_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                    hooks.append(h)
                mm_ids = batch["mm_token_type_ids"][0]; mm_ids = mm_ids.unsqueeze(0).to(device) if mm_ids is not None else None
                teacher_out = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, mm_token_type_ids=mm_ids)
                teacher_logits = teacher_out.logits
                for h in hooks:
                    h.remove()

            # Student forward (text only)
            text_mask = input_ids[0] != image_token_id
            text_ids = input_ids[:, text_mask]
            student_logits = student_forward_qwen(model, text_ids, engine.module, source_k, source_v, image_grid_thw=grid_thw, spatial_merge_size=2)

            # Align answer positions
            num_text = student_logits.shape[1]
            s_start = max(0, prompt_len - 1)
            s_end = min(s_start + args.max_answer_tokens, num_text)
            s_answer = student_logits[0, s_start:s_end]

            n_image_tokens = (input_ids[0] == image_token_id).sum().item()
            t_start = n_image_tokens + prompt_len - 1
            t_end = min(t_start + args.max_answer_tokens, teacher_logits.shape[1])
            t_answer = teacher_logits[0, t_start:t_end]

            if s_answer.shape[0] > 0 and t_answer.shape[0] > 0:
                min_len = min(s_answer.shape[0], t_answer.shape[0])
                loss = topk_kl_loss(s_answer[:min_len].float(), t_answer[:min_len].float(), args.kl_topk)
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
                torch.save({"state_dict": engine.module.state_dict(), "step": step, "args": vars(args)},
                           Path(args.output_dir) / f"step_{step}.pt")

            step += 1
        if step >= args.max_steps:
            break

    if is_main:
        torch.save({"state_dict": engine.module.state_dict(), "step": step, "args": vars(args)},
                   Path(args.output_dir) / "final.pt")
        print(f"Training complete. Final: {Path(args.output_dir)/chr(102)+chr(105)+chr(110)+chr(97)+chr(108)+chr(46)+chr(112)+chr(116)}")
    if wandb_run:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
