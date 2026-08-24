"""Train Qwen3-VL embedding adapter for rendered-text OCR distillation.

This is an experimental entrypoint kept outside the main trainer on purpose.
It expects paired rendered-text rows with:
  - images or image: rendered page image path(s)
  - image_root: optional image root
  - text_context: raw textual context for the teacher
  - question or rendered_question: question for the student
  - answer: short answer suffix supervised for both teacher/student
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from src.model import (
    QwenEmbeddingAdapter,
    dtype_from_name,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    qwen_embedding_adapter_logits,
)
from src.train import lr_multiplier, optimizer_param_groups, save_checkpoint, set_engine_lr, trainable_parameters_for_mode


class JsonlRows:
    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        require_answer_visible: bool = False,
        shuffle: bool = True,
        seed: int = 49,
    ) -> None:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if require_answer_visible and row.get("answer_visible") is False:
                    continue
                if not str(row.get("text_context") or "").strip():
                    continue
                if not str(row.get("answer") or "").strip():
                    continue
                rows.append(row)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if not rows:
            raise RuntimeError(f"no usable rows found in {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, start: int, batch_size: int) -> list[dict[str, Any]]:
        return [self.rows[(start + offset) % len(self.rows)] for offset in range(batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OCR/rendered-text teacher -> rendered-image adapter student training.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/rendered_text_copy_2048/paired_train.jsonl")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="artifacts/experiments/test_rendered_text_teacher")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--micro-batch-size-per-gpu", "--batch-size", dest="micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--required-world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--require-answer-visible", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=0, help="0 means no teacher text truncation.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-start-lr-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--visual-adapter-rank", type=int, default=128)
    parser.add_argument("--output-mode", default="embedding_adapter")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deepspeed-config", default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vision-kv-inject")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed_is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(value.item()) for key, value in zip(keys, values)}


RENDERED_PAGE_INSTRUCTION = "Read the ordered page images and answer using only their text."
QA_IMAGE_INSTRUCTION = "Use the image text to answer the question."
QA_FINAL_ANSWER_INSTRUCTION = "Return only the final answer, with no explanation."
COPY_TRANSCRIPTION_INSTRUCTION = "Transcribe all visible text in the image exactly. Preserve line breaks."


def is_copy_transcription(row: dict[str, Any]) -> bool:
    return str(row.get("task_type") or "").strip() in {"copy", "copy_transcription"}


def cleaned_rendered_question(row: dict[str, Any]) -> str:
    question = str(row.get("raw_question") or row.get("question") or row.get("rendered_question") or "").strip()
    prefixes = (
        RENDERED_PAGE_INSTRUCTION,
        "The attached page images are consecutive pages in order.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix) :].lstrip("\n ").strip()
                changed = True
    return question


def text_teacher_prompt(processor: Any, row: dict[str, Any], max_context_chars: int) -> str:
    context = str(row["text_context"]).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]
    question = cleaned_rendered_question(row)
    if is_copy_transcription(row):
        instruction = question or COPY_TRANSCRIPTION_INSTRUCTION
        text = f"Text:\n{context}\n\n{instruction}"
    else:
        text = f"Context:\n{context}\n\nQuestion:\n{question}\n\n{QA_FINAL_ANSWER_INSTRUCTION}"
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_text_teacher_inputs(
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    max_context_chars: int,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    texts: list[str] = []
    answer_lens: list[int] = []
    eos = processor.tokenizer.eos_token or ""
    for row in rows:
        prompt = text_teacher_prompt(processor, row, max_context_chars)
        answer = str(row.get("answer", "")).strip()
        suffix = f" {answer}{eos if eos and not answer.endswith(eos) else ''}"
        texts.append(f"{prompt}{suffix}")
        answer_lens.append(len(processor.tokenizer(suffix, add_special_tokens=False).input_ids))

    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "right"
    try:
        encoded = processor.tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch_idx, answer_len in enumerate(answer_lens):
        valid_len = int(attention_mask[batch_idx].sum().item())
        start = max(0, valid_len - int(answer_len))
        answer_mask[batch_idx, start:valid_len] = True
    return inputs, input_ids, answer_mask


def student_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        question = cleaned_rendered_question(copy)
        if is_copy_transcription(copy):
            copy["question"] = question or COPY_TRANSCRIPTION_INSTRUCTION
        else:
            copy["question"] = f"{QA_IMAGE_INSTRUCTION}\n{QA_FINAL_ANSWER_INSTRUCTION}\n\nQuestion: {question}"
        result.append(copy)
    return result


def masked_unaligned_topk_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_ids: Tensor,
    student_answer_mask: Tensor,
    teacher_answer_mask: Tensor,
    *,
    temperature: float,
    topk: int,
) -> tuple[Tensor, Tensor]:
    losses: list[Tensor] = []
    counts: list[int] = []
    for batch_idx in range(student_logits.shape[0]):
        s_mask = student_answer_mask[batch_idx, 1:].bool()
        t_mask = teacher_answer_mask[batch_idx, 1:].bool()
        s = student_logits[batch_idx, :-1][s_mask]
        t = teacher_logits[batch_idx, :-1][t_mask]
        targets = student_ids[batch_idx, 1:][s_mask]
        count = min(int(s.shape[0]), int(t.shape[0]), int(targets.shape[0]))
        counts.append(count)
        if count <= 0:
            continue
        s = s[-count:].float()
        t = t[-count:].float()
        targets = targets[-count:].long()
        k_eff = min(int(topk), int(t.shape[-1]))
        topk_idx = torch.topk(t, k=k_eff, dim=-1).indices
        target_idx = targets.unsqueeze(-1)
        if k_eff < int(t.shape[-1]):
            target_in_topk = topk_idx.eq(target_idx).any(dim=-1, keepdim=True)
            gather_idx = torch.where(target_in_topk, topk_idx, torch.cat([topk_idx[..., :-1], target_idx], dim=-1))
        else:
            gather_idx = topk_idx
        s_gathered = torch.gather(s, dim=-1, index=gather_idx) / float(temperature)
        t_gathered = torch.gather(t, dim=-1, index=gather_idx) / float(temperature)
        kl = F.kl_div(
            F.log_softmax(s_gathered, dim=-1),
            F.softmax(t_gathered, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        token_loss = kl * (float(temperature) * float(temperature))
        losses.append(token_loss)
    answer_counts = torch.tensor(counts, device=student_logits.device, dtype=torch.float32)
    if not losses:
        return student_logits.new_zeros(()), answer_counts
    flat_losses = torch.cat(losses)
    return flat_losses.float().mean().to(dtype=student_logits.dtype), answer_counts


def main() -> None:
    args = parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        if args.required_world_size > 1 and dist.get_world_size() != args.required_world_size:
            raise RuntimeError(f"expected {args.required_world_size} ranks, got {dist.get_world_size()}")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    rank = dist.get_rank() if distributed_is_initialized() else 0
    world_size = dist.get_world_size() if distributed_is_initialized() else 1

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if distributed_is_initialized():
        dist.barrier()

    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = model.model.language_model
    adapter = QwenEmbeddingAdapter.from_language_model(
        language_model,
        mode=args.output_mode,
        visual_adapter_rank=args.visual_adapter_rank,
    ).to(device=device, dtype=dtype)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
        print(f"loaded init checkpoint {args.init_checkpoint} missing={list(missing)} unexpected={list(unexpected)}", flush=True)

    trainable = trainable_parameters_for_mode(adapter)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if distributed:
        ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
        ds_config["train_micro_batch_size_per_gpu"] = int(args.micro_batch_size_per_gpu)
        ds_config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
        ds_config["gradient_clipping"] = float(args.grad_clip)
        engine, _, _, _ = deepspeed.initialize(
            model=adapter,
            model_parameters=trainable,
            optimizer=optimizer,
            config=ds_config,
        )
    else:
        from src.train import SimpleEngine

        engine = SimpleEngine(adapter, optimizer, args.grad_clip)
    engine.train()
    base_lrs = [float(group.get("lr", args.lr)) for group in optimizer_param_groups(engine)]

    data = JsonlRows(
        args.data,
        max_samples=args.max_samples,
        require_answer_visible=args.require_answer_visible,
        shuffle=True,
        seed=args.seed,
    )
    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    metrics_path = Path(args.metrics_jsonl) if str(args.metrics_jsonl).strip() else output_dir / "train_metrics.jsonl"
    if is_rank0():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"rendered text-teacher train rows={len(data)} world_size={world_size} "
            f"micro_batch={args.micro_batch_size_per_gpu} grad_accum={args.gradient_accumulation_steps} "
            f"global_batch={world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps} "
            f"steps={args.max_steps} trainable={sum(p.numel() for p in trainable)/1e6:.2f}M",
            flush=True,
        )

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled" and is_rank0():
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.parent.name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "dataset_size": len(data),
                "world_size": world_size,
                "global_batch": world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps,
            },
        )

    for step in range(1, int(args.max_steps) + 1):
        start_s = time.perf_counter()
        current_lr = set_engine_lr(engine, base_lrs, lr_multiplier(args, step - 1))
        accum: dict[str, float] = {
            "logit_kl": 0.0,
            "loss": 0.0,
            "answer_tokens": 0.0,
        }
        image_paths: list[str] = []
        for micro_idx in range(int(args.gradient_accumulation_steps)):
            sample_base = (
                (step - 1) * int(args.gradient_accumulation_steps) * world_size * int(args.micro_batch_size_per_gpu)
                + micro_idx * world_size * int(args.micro_batch_size_per_gpu)
                + rank * int(args.micro_batch_size_per_gpu)
            )
            rows = data.batch(sample_base, int(args.micro_batch_size_per_gpu))
            teacher_inputs, _, teacher_answer_mask = prepare_text_teacher_inputs(
                processor,
                rows,
                device,
                max_context_chars=args.max_context_chars,
            )
            student_inputs, student_ids, student_answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
                processor,
                student_rows(rows),
                image_root,
                device,
                include_answers=True,
            )
            assert student_ids is not None and student_answer_mask is not None

            with torch.no_grad():
                teacher = model(**teacher_inputs, return_dict=True, use_cache=False)
                teacher_logits = teacher.logits.detach()
            student_logits, _, _ = qwen_embedding_adapter_logits(model, engine.module, student_inputs)

            kl, answer_counts = masked_unaligned_topk_kl(
                student_logits,
                teacher_logits,
                student_ids,
                student_answer_mask,
                teacher_answer_mask,
                temperature=args.temperature,
                topk=args.kl_topk,
            )
            unscaled_loss = float(args.lambda_logit) * kl
            loss = unscaled_loss / float(args.gradient_accumulation_steps)
            engine.backward(loss)
            accum["logit_kl"] += float(kl.detach()) / float(args.gradient_accumulation_steps)
            accum["loss"] += float(unscaled_loss.detach()) / float(args.gradient_accumulation_steps)
            accum["answer_tokens"] += float(answer_counts.mean().item()) / float(args.gradient_accumulation_steps)
        engine.step()

        if step % int(args.log_every) == 0 or step == 1:
            accum["lr"] = float(current_lr)
            accum["sec_per_step"] = time.perf_counter() - start_s
            reduced = reduce_metrics(accum, device)
            payload = {
                "step": step,
                **reduced,
                "global_batch": int(world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps),
                "image": image_paths[0] if image_paths else "",
            }
            if is_rank0():
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(
                    f"step={step} loss={payload['loss']:.6f} kl={payload['logit_kl']:.6f} "
                    f"answer_tokens={payload['answer_tokens']:.1f} lr={payload['lr']:.3e} "
                    f"global_batch={payload['global_batch']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"train/{k}": v for k, v in payload.items() if isinstance(v, (int, float))}, step=step)

        if step % int(args.save_every) == 0:
            if hasattr(engine, "save_checkpoint"):
                engine.save_checkpoint(str(output_dir / "optimizer"), tag=f"step{step}")
            if is_rank0():
                save_checkpoint(engine.module, output_dir / f"qwen_rendered_text_teacher_step{step}.pt", args, step)
                save_checkpoint(engine.module, output_dir / f"qwen_embedding_adapter_step{step}.pt", args, step)

    if hasattr(engine, "save_checkpoint"):
        engine.save_checkpoint(str(output_dir / "optimizer"), tag="final")
    if is_rank0():
        save_checkpoint(engine.module, output_dir / "qwen_rendered_text_teacher_final.pt", args, int(args.max_steps))
        save_checkpoint(engine.module, output_dir / "qwen_embedding_adapter_final.pt", args, int(args.max_steps))
    if wandb_run is not None:
        import wandb

        wandb.finish()
    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
