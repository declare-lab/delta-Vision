"""Train Qwen3-VL embedding adapter with text-context teacher and rendered-image student.

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
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.model import (
    QwenEmbeddingAdapter,
    dtype_from_name,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
    qwen_embedding_adapter_logits,
    resolve_row_image_paths,
)
from src.train import SimpleEngine, lr_multiplier, save_checkpoint, set_engine_lr, trainable_parameters_for_mode


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
    parser = argparse.ArgumentParser("Rendered text-context teacher -> rendered-image adapter student experiment.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/rendered_context_qa_eval_v1/paired.jsonl")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="artifacts/experiments/test_rendered_text_teacher")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--require-answer-visible", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-start-lr-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--lambda-logit", type=float, default=2.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0)
    parser.add_argument("--visual-adapter-rank", type=int, default=128)
    parser.add_argument("--output-mode", default="embedding_adapter")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def text_teacher_prompt(processor: Any, row: dict[str, Any], max_context_chars: int) -> str:
    context = str(row["text_context"]).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]
    question = str(row.get("question") or row.get("rendered_question") or "").strip()
    question = question.replace("The attached page images are consecutive pages in order.\n", "").strip()
    text = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer directly with a short phrase."
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
        if "rendered_question" in copy:
            copy["question"] = copy["rendered_question"]
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
        losses.append(kl * (float(temperature) * float(temperature)))
    answer_counts = torch.tensor(counts, device=student_logits.device, dtype=torch.float32)
    if not losses:
        return student_logits.new_zeros(()), answer_counts
    return torch.cat(losses).mean().to(dtype=student_logits.dtype), answer_counts


def masked_unaligned_ce(
    student_logits: Tensor,
    student_ids: Tensor,
    student_answer_mask: Tensor,
) -> Tensor:
    shift_mask = student_answer_mask[:, 1:].bool()
    if int(shift_mask.sum().item()) == 0:
        return student_logits.new_zeros(())
    logits = student_logits[:, :-1][shift_mask].float()
    targets = student_ids[:, 1:][shift_mask].long()
    return F.cross_entropy(logits, targets).to(dtype=student_logits.dtype)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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
    engine = SimpleEngine(adapter, optimizer, args.grad_clip)
    engine.train()
    base_lrs = [float(group.get("lr", args.lr)) for group in optimizer.param_groups]

    data = JsonlRows(
        args.data,
        max_samples=args.max_samples,
        require_answer_visible=args.require_answer_visible,
        shuffle=True,
        seed=args.seed,
    )
    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    metrics_path = output_dir / "train_metrics.jsonl"
    print(
        f"rendered text-teacher experiment rows={len(data)} batch={args.batch_size} "
        f"steps={args.max_steps} trainable={sum(p.numel() for p in trainable)/1e6:.2f}M",
        flush=True,
    )

    for step in range(1, int(args.max_steps) + 1):
        start_s = time.perf_counter()
        current_lr = set_engine_lr(engine, base_lrs, lr_multiplier(args, step - 1))
        rows = data.batch((step - 1) * int(args.batch_size), int(args.batch_size))

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
        student_logits, _, _ = qwen_embedding_adapter_logits(model, engine.module, student_inputs, collect_states=False)

        kl, answer_counts = masked_unaligned_topk_kl(
            student_logits,
            teacher_logits,
            student_ids,
            student_answer_mask,
            teacher_answer_mask,
            temperature=args.temperature,
            topk=args.kl_topk,
        )
        ce = masked_unaligned_ce(student_logits, student_ids, student_answer_mask)
        loss = float(args.lambda_logit) * kl + float(args.lambda_ce) * ce
        engine.backward(loss)
        engine.step()

        if step % int(args.log_every) == 0 or step == 1:
            payload = {
                "step": step,
                "loss": float(loss.detach()),
                "logit_kl": float(kl.detach()),
                "ce": float(ce.detach()),
                "lr": float(current_lr),
                "answer_tokens": float(answer_counts.mean().item()),
                "sec_per_step": time.perf_counter() - start_s,
                "image": image_paths[0] if image_paths else "",
                "source": str(rows[0].get("source", "")),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"step={step} loss={payload['loss']:.6f} kl={payload['logit_kl']:.6f} "
                f"ce={payload['ce']:.6f} answer_tokens={payload['answer_tokens']:.1f} "
                f"lr={payload['lr']:.3e}",
                flush=True,
            )

        if step % int(args.save_every) == 0:
            save_checkpoint(adapter, output_dir / f"qwen_rendered_text_teacher_step{step}.pt", args, step)

    save_checkpoint(adapter, output_dir / "qwen_rendered_text_teacher_final.pt", args, int(args.max_steps))


if __name__ == "__main__":
    main()
