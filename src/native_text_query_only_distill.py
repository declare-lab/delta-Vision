"""Full-student distillation with visual tokens using residual + FFN only.

Student intervention:
- During prompt prefill, visual-token rows after self-attention o_proj are set
  to zero in every language layer.
- Visual tokens therefore do not act as useful attention queries: they do not
  read other visual tokens, themselves, or preceding text through attention.
- Text-token rows are unchanged and can attend to visual/text K/V under native
  causal attention.

Teacher is frozen native Qwen3-VL with FA2 and DeepStack disabled.  The student
starts from the same native weights and is trained with PixMo answer-position
top-k logit KL, matching the previous teacher-student setup.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from src.model import load_frozen_qwen3vl, prepare_qwen3vl_batch_inputs
from src.qwen_deepstack import disable_qwen_deepstack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/native_text_query_only_ffn_distill_20260917"
MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEPTH = 36


class VisualFFNOnly:
    """Zero visual rows of the attention output after o_proj."""

    def __init__(self, model):
        self.mask: torch.Tensor | None = None
        self.calls = 0
        self.enabled = True
        self.handles = []
        for _, block in enumerate(model.model.language_model.layers):
            self.handles.append(block.self_attn.o_proj.register_forward_hook(self.replace()))

    def replace(self):
        def hook(module, args, output):
            if not self.enabled or output.shape[1] == 1:
                return
            if self.mask is None:
                raise RuntimeError("VisualFFNOnly.mask is not set")
            if self.mask.shape != output.shape[:2]:
                raise RuntimeError(f"mask shape {tuple(self.mask.shape)} != output shape {tuple(output.shape[:2])}")
            self.calls += 1
            return torch.where(self.mask[:, :, None], torch.zeros_like(output), output)

        return hook


def select_positions(inputs, answer_mask):
    bi, pi, labels = [], [], []
    for batch_idx in range(len(answer_mask)):
        text_pos = ((inputs["mm_token_type_ids"][batch_idx] == 0) & inputs["attention_mask"][batch_idx].bool()).nonzero().flatten()
        selected = answer_mask[batch_idx, 1 : len(text_pos)].nonzero().flatten()
        bi.append(torch.full_like(selected, batch_idx))
        pi.append(text_pos[:-1][selected])
        labels.append(inputs["input_ids"][batch_idx, text_pos[1:][selected]])
    return torch.cat(bi), torch.cat(pi), torch.cat(labels)


class Student(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.intervention = VisualFFNOnly(model)

    def forward(self, inputs, bi, pi, idx, target):
        self.intervention.mask = inputs["input_ids"] == self.model.config.image_token_id
        self.intervention.calls = 0
        self.model.model.rope_deltas = None
        hidden = self.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[bi, pi]
        if self.intervention.calls != DEPTH:
            raise RuntimeError(f"expected {DEPTH} visual-FFN hooks, got {self.intervention.calls}")
        losses = []
        for start in range(0, len(hidden), 128):
            # Checkpoint LM-head chunks so full-vocabulary logits are not retained.
            def part(h, indices, t):
                logits = self.model.lm_head(h).float().gather(-1, indices) / 2
                return F.kl_div(F.log_softmax(logits, -1), F.softmax(t / 2, -1), reduction="none").sum(-1) * 4

            losses.append(
                torch.utils.checkpoint.checkpoint(
                    part,
                    hidden[start : start + 128],
                    idx[start : start + 128],
                    target[start : start + 128],
                    use_reentrant=False,
                )
            )
        return torch.cat(losses).mean()


def ensure_plan() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    plan = {
        "model": "Qwen3-VL-4B-Instruct",
        "teacher": "frozen native attention, FA2, DeepStack off",
        "student": "native initialization; in every LM layer visual-token attention output rows are zeroed, so visual tokens do not attend to self, other visual tokens, or preceding text; text causal attention unchanged",
        "train_vision": True,
        "steps": 2000,
        "world_size": 8,
        "micro_batch": 4,
        "global_batch": 32,
        "loss": "answer-position teacher top1024 KL, include ground-truth token if outside topk, T2, lambda1, CE0",
        "lr": 5e-5,
        "seed": 44,
        "optimizer": "AdamW betas0.9,0.95 wd0.01 clip1, ZeRO2 BF16",
        "schedule": "warmup60, cosine to0.1",
        "data": "Pixmo135995 original pixel_bucket512 order",
        "evaluation": "RealWorldQA765 MMStar1000 SQA1000, native/text-query-only-untrained/text-query-only-trained",
        "deepstack": False,
        "attention": "FA2 text; visual attention output after o_proj is exactly zero during prefill",
        "checkpointing": "nonreentrant, every500step HF checkpoints require VisualFFNOnly hook",
    }
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise RuntimeError(f"Existing plan differs: {plan_path}")
        return existing
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def train(steps: int, smoke: bool = False) -> None:
    import deepspeed

    torch.set_num_threads(4)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    deepspeed.init_distributed()
    torch.manual_seed(44)

    plan = ensure_plan()
    folder = OUT / ("smoke" if smoke else "train")
    folder.mkdir(exist_ok=True)

    def log_rank0(message: str) -> None:
        if rank == 0:
            print(json.dumps({"event": message, "time": time.time()}), flush=True)

    log_rank0("load_teacher_start")
    processor, teacher = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device("cuda", rank), "flash_attention_2")
    disable_qwen_deepstack(teacher)
    teacher.eval()
    log_rank0("load_teacher_done")

    log_rank0("load_student_start")
    _, student = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device("cuda", rank), "flash_attention_2")
    disable_qwen_deepstack(student)
    student.requires_grad_(True)
    if not plan["train_vision"]:
        student.model.visual.requires_grad_(False)
    # Disabled DeepStack branches cannot receive useful gradients.
    student.model.visual.deepstack_merger_list.requires_grad_(False)

    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    wrapped = Student(student)
    log_rank0("load_student_done")

    conf = json.loads((ROOT / "configs/ds_zero2.json").read_text())
    conf.update(train_micro_batch_size_per_gpu=4, gradient_accumulation_steps=1, train_batch_size=32)
    conf["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }
    log_rank0("trainable_parameter_collect_start")
    trainable_parameters = [p for p in wrapped.parameters() if p.requires_grad]
    log_rank0("trainable_parameter_collect_done")
    log_rank0("deepspeed_init_start")
    engine, opt, _, _ = deepspeed.initialize(model=wrapped, model_parameters=trainable_parameters, config=conf)
    log_rank0("deepspeed_init_done")

    log_rank0("data_load_start")
    data = [json.loads(s) for s in (ROOT / "data/train/pixmo/pixmo_ama_full_valid.clean.jsonl").read_text().splitlines()]
    areas = json.loads((ROOT / "data/train/pixmo/pixmo_ama_full_valid.clean.jsonl.pixel_areas.json").read_text())
    areas = areas["areas"] if isinstance(areas, dict) else areas
    sized = sorted((area, idx) for idx, area in enumerate(areas))
    buckets = [[idx for _, idx in sized[j : j + 512]] for j in range(0, len(sized), 512)]
    generator = torch.Generator().manual_seed(44)
    order = []
    for bucket_idx in torch.randperm(len(buckets), generator=generator).tolist():
        bucket = buckets[bucket_idx]
        order.extend(bucket[i] for i in torch.randperm(len(bucket), generator=generator).tolist())
    assert len(order) >= steps * 32
    log_rank0("data_load_done")

    wb = None
    if rank == 0:
        (folder / "setup.json").write_text(
            json.dumps(
                {
                    "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad),
                    "sample_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                    "train_vision": plan["train_vision"],
                },
                indent=2,
            )
        )
        if not smoke:
            import wandb

            wb = wandb.init(
                project="vision-kv-inject",
                name="native_text_query_only_ffn_full_distill_2000_20260917",
                config=plan,
                mode="online",
                dir=str(OUT),
            )
            (OUT / "wandb.json").write_text(json.dumps({"url": wb.url}))

    start = time.time()
    for step in range(steps):
        step_t0 = time.time()
        if rank == 0:
            print(json.dumps({"event": "step_start", "step": step + 1, "elapsed": step_t0 - start}), flush=True)
        ids = order[step * 32 + rank * 4 : step * 32 + rank * 4 + 4]
        t_prepare = time.time()
        inputs, _, answer_mask, _ = prepare_qwen3vl_batch_inputs(
            processor,
            [data[i] for i in ids],
            ROOT / "data/train/pixmo",
            torch.device("cuda", rank),
            include_answers=True,
        )
        bi, pi, labels = select_positions(inputs, answer_mask)
        assert len(labels) > 0
        if rank == 0:
            print(json.dumps({"event": "prepare_done", "step": step + 1, "seconds": time.time() - t_prepare, "tokens": int(inputs["attention_mask"].sum())}), flush=True)

        with torch.no_grad():
            t_teacher = time.time()
            teacher.model.rope_deltas = None
            h = teacher.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[bi, pi]
            inds, vals = [], []
            for j in range(0, len(h), 128):
                logits = teacher.lm_head(h[j : j + 128]).float()
                idx = logits.topk(1024, -1).indices
                lab = labels[j : j + 128, None]
                idx = torch.where(idx.eq(lab).any(-1, keepdim=True), idx, torch.cat((idx[:, :-1], lab), -1))
                inds.append(idx)
                vals.append(logits.gather(-1, idx))
            idx = torch.cat(inds)
            target = torch.cat(vals)
            del h, logits
            if rank == 0:
                print(json.dumps({"event": "teacher_done", "step": step + 1, "seconds": time.time() - t_teacher, "answer_tokens": int(len(labels))}), flush=True)

        t_student = time.time()
        loss = engine(inputs, bi, pi, idx, target)
        assert torch.isfinite(loss)
        if rank == 0:
            print(json.dumps({"event": "student_loss_done", "step": step + 1, "seconds": time.time() - t_student, "loss": float(loss.detach())}), flush=True)
        t_backward = time.time()
        engine.backward(loss)
        if rank == 0:
            print(json.dumps({"event": "backward_done", "step": step + 1, "seconds": time.time() - t_backward}), flush=True)

        if step == 0:
            t_grad = time.time()
            from deepspeed.utils import safe_get_full_grad

            checks = {}
            for name, param in [
                ("language_v", student.model.language_model.layers[17].self_attn.v_proj.weight),
                ("language_mlp", student.model.language_model.layers[17].mlp.gate_proj.weight),
                ("vision", next(student.model.visual.parameters())),
            ]:
                if param.requires_grad:
                    grad = safe_get_full_grad(param)
                    assert grad is not None and torch.isfinite(grad).all() and grad.float().norm() > 0, name
                    checks[name] = float(grad.float().norm())
            if rank == 0:
                (folder / "gradient_check.json").write_text(json.dumps(checks, indent=2))
                print(json.dumps({"event": "gradient_check_done", "step": step + 1, "seconds": time.time() - t_grad, "checks": checks}), flush=True)

        warmup = 60
        if step < warmup:
            mult = (step + 1) / warmup
        else:
            mult = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, 2000 - warmup)))
        for group in engine.optimizer.param_groups:
            group["lr"] = 5e-5 * mult
        engine.step()

        if rank == 0 and (step == 0 or (step + 1) % 5 == 0 or smoke):
            row = {"step": step + 1, "loss": float(loss.detach()), "lr": 5e-5 * mult, "elapsed": time.time() - start}
            print(json.dumps(row), flush=True)
            with (folder / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if wb:
                wb.log(row, step=step + 1)

        if not smoke and (step + 1) % 500 == 0:
            torch.distributed.barrier()
            if rank == 0:
                dest = OUT / f"checkpoint-{step + 1}"
                student.save_pretrained(dest, safe_serialization=True)
                processor.save_pretrained(dest)
                (dest / "VISUAL_FFN_ONLY_REQUIRED.json").write_text(
                    json.dumps(
                        {
                            "deepstack": False,
                            "visual_attention": "zero_attention_output_visual_ffn_only",
                            "requires_hook": "src.native_text_query_only_distill.VisualFFNOnly",
                        },
                        indent=2,
                    )
                )
            torch.distributed.barrier()

    if wb:
        wb.finish()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    ensure_plan()
    train(int(sys.argv[1]), len(sys.argv) > 2 and sys.argv[2] == "smoke")
