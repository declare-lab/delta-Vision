"""Diagnose Qwen embedding_adapter loss plateaus.

This script intentionally lives under test/diagnostics. It compares the trained adapter
against simple oracle visual-memory candidates and prints token/layer diagnostics for a
fixed batch. Outputs are written under test/results by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import (  # noqa: E402
    QwenEmbeddingAdapter,
    dtype_from_name,
    gather_batched_positions,
    get_qwen_text_image_positions,
    load_frozen_qwen3vl,
    load_qwen_embedding_adapter_checkpoint,
    prepare_qwen3vl_batch_inputs,
    prepare_qwen_embedding_adapter_inputs,
    qwen_lm_head_logits,
    qwen_position_ids,
)
from src.model import (  # noqa: E402
    _apply_rope_one_from_embeddings,
    _compile_exact_qwen_apply_rotary_pos_emb,
    _prefix_causal_attention_heads,
)
from src.train import masked_directional_mse, parse_trajectory_layers  # noqa: E402


def read_jsonl_rows(path: Path, *, start_index: int, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < start_index or not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= count:
                break
    if not rows:
        raise ValueError(f"no rows read from {path} at start_index={start_index}")
    return rows


def scalar(value: Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def masked_stats(values: Tensor, mask: Tensor) -> dict[str, float]:
    flat = values.detach().float()[mask.bool()]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().cpu()),
        "p50": float(flat.quantile(0.50).cpu()),
        "p90": float(flat.quantile(0.90).cpu()),
        "p99": float(flat.quantile(0.99).cpu()),
        "max": float(flat.max().cpu()),
    }


def per_token_topk_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_ids: Tensor,
    answer_mask: Tensor,
    *,
    temperature: float,
    topk: int,
) -> tuple[Tensor, Tensor]:
    shift_mask = answer_mask[:, 1:].bool()
    out = student_logits.new_zeros(answer_mask[:, 1:].shape, dtype=torch.float32)
    if int(shift_mask.sum().item()) == 0 or topk <= 0:
        return out, shift_mask
    shift_student = student_logits[:, :-1][shift_mask].float()
    shift_teacher = teacher_logits[:, :-1][shift_mask].float()
    shift_targets = target_ids[:, 1:][shift_mask]
    k_eff = min(int(topk), int(shift_teacher.shape[-1]))
    topk_idx = torch.topk(shift_teacher, k=k_eff, dim=-1).indices
    target_idx = shift_targets.unsqueeze(-1)
    if k_eff == shift_teacher.shape[-1]:
        gather_idx = topk_idx
    else:
        target_in_topk = topk_idx.eq(target_idx).any(dim=-1, keepdim=True)
        gather_idx = torch.where(target_in_topk, topk_idx, torch.cat([topk_idx[..., :-1], target_idx], dim=-1))
    t = torch.gather(shift_teacher, -1, gather_idx) / float(temperature)
    s = torch.gather(shift_student, -1, gather_idx) / float(temperature)
    kl = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none").sum(dim=-1)
    kl = kl * float(temperature) * float(temperature)
    out[shift_mask] = kl.to(device=out.device, dtype=torch.float32)
    return out, shift_mask


def masked_token_mean(per_token: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def masked_sample_mean(per_token: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(device=per_token.device, dtype=per_token.dtype)
    per_sample = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    has_valid = valid.sum(dim=1) > 0
    if int(has_valid.sum().item()) == 0:
        return per_token.new_zeros(())
    return per_sample[has_valid].mean()


def top1_agreement(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    student_top = student_logits[:, :-1].argmax(dim=-1)
    teacher_top = teacher_logits[:, :-1].argmax(dim=-1)
    agree = student_top.eq(teacher_top)[mask.bool()].float().mean()
    return float(agree.cpu())


def tensor_norm_stats(name: str, tensor: Tensor, mask: Tensor | None = None) -> dict[str, float | str]:
    norms = tensor.detach().float().pow(2).mean(dim=-1).sqrt()
    if mask is None:
        active = torch.ones(norms.shape, device=norms.device, dtype=torch.bool)
    else:
        active = mask.to(device=norms.device, dtype=torch.bool)
    stats = masked_stats(norms, active)
    return {"name": name, **stats}


def gate_rows(adapter: torch.nn.Module) -> list[dict[str, float | str]]:
    rows = []
    for name, param in adapter.named_parameters():
        if "gate" not in name:
            continue
        value = param.detach().float()
        rows.append(
            {
                "name": name,
                "raw_mean": float(value.mean().cpu()),
                "raw_min": float(value.min().cpu()),
                "raw_max": float(value.max().cpu()),
                "sigmoid_mean": float(value.sigmoid().mean().cpu()),
                "sigmoid_min": float(value.sigmoid().min().cpu()),
                "sigmoid_max": float(value.sigmoid().max().cpu()),
            }
        )
    return rows


def prepared_forward_kwargs(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "h": prepared["h"],
        "visual_memory": prepared["visual_memory"],
        "text_mask": prepared["text_mask"],
        "text_position_ids": prepared["text_position_ids"],
        "visual_position_ids": prepared["visual_position_ids"],
        "prefix_attention_mask": prepared["prefix_attention_mask"],
        "text_position_embeddings": prepared["text_position_embeddings"],
        "visual_position_embeddings": prepared["visual_position_embeddings"],
    }


def oracle_logits_from_visual_memories(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    prepared: dict[str, Any],
    all_visual_memories: Tensor,
    *,
    collect_state_indices: set[int],
) -> tuple[Tensor, Tensor, list[Tensor]]:
    language_model = model.model.language_model
    layers = language_model.layers
    h = prepared["h"]
    text_mask = prepared["text_mask"]
    text_position_ids = prepared["text_position_ids"]
    visual_position_ids = prepared["visual_position_ids"]
    prefix_attention_mask = prepared["prefix_attention_mask"]
    text_position_embeddings = prepared["text_position_embeddings"]
    visual_position_embeddings = prepared["visual_position_embeddings"]
    states = [h.new_empty(0) for _ in range(len(layers) + 1)]
    if 0 in collect_state_indices:
        states[0] = h
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        normed_text = layer.input_layernorm(h)
        text_shape = normed_text.shape[:-1]
        hidden_shape = (*text_shape, -1, attn.head_dim)
        raw_query = attn.q_proj(normed_text).view(hidden_shape)
        raw_text_key = attn.k_proj(normed_text).view(hidden_shape)
        query = attn.q_norm(raw_query).transpose(1, 2)
        text_key = attn.k_norm(raw_text_key).transpose(1, 2)
        text_value = attn.v_proj(normed_text).view(hidden_shape).transpose(1, 2)
        layer_text_pos = text_position_embeddings
        if layer_text_pos is None:
            layer_text_pos = language_model.rotary_emb(normed_text, text_position_ids)
        query, text_key = _compile_exact_qwen_apply_rotary_pos_emb(query, text_key, layer_text_pos)

        vision_states = all_visual_memories[layer_idx].to(device=h.device, dtype=h.dtype)
        normed_vision = layer.input_layernorm(vision_states)
        vision_shape = normed_vision.shape[:-1]
        vision_hidden_shape = (*vision_shape, -1, attn.head_dim)
        raw_visual_key = attn.k_proj(normed_vision).view(vision_hidden_shape)
        visual_key = attn.k_norm(raw_visual_key).transpose(1, 2)
        visual_value = attn.v_proj(normed_vision).view(vision_hidden_shape).transpose(1, 2)
        layer_visual_pos = visual_position_embeddings
        if layer_visual_pos is None:
            layer_visual_pos = language_model.rotary_emb(normed_vision, visual_position_ids)
        visual_key = _apply_rope_one_from_embeddings(visual_key, layer_visual_pos)

        heads = _prefix_causal_attention_heads(
            query,
            visual_key,
            visual_value,
            text_key,
            text_value,
            attention_mask=prefix_attention_mask,
            scaling=float(attn.scaling),
        )
        text_attention = attn.o_proj(heads.reshape(*text_shape, -1).contiguous())
        h = h + text_attention.to(dtype=h.dtype)
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h
        state_idx = layer_idx + 1
        if state_idx in collect_state_indices:
            states[state_idx] = h
    return qwen_lm_head_logits(model, language_model, h, text_mask), text_mask, states


def build_oracle_memories(
    mode: str,
    *,
    prepared: dict[str, Any],
    teacher_hidden_states: tuple[Tensor, ...],
    position_ids: Tensor,
    inputs: dict[str, Tensor],
    adapter: QwenEmbeddingAdapter,
) -> Tensor:
    mode = mode.strip()
    base_visual = prepared["visual_memory"]
    num_layers = int(adapter.num_layers)
    if mode == "adapter":
        return adapter.all_visual_memories_batched(base_visual)
    if mode == "recurrent_adapter":
        memories = []
        current = base_visual
        for layer_idx in range(num_layers):
            current = adapter.visual_memory_for_layer(current, layer_idx)
            memories.append(current)
        return torch.stack(memories, dim=0)
    if mode == "initial":
        return base_visual.unsqueeze(0).expand(num_layers, -1, -1, -1).contiguous()
    if mode == "teacher_layer_input":
        memories = []
        _, image_pos, _, _, image_mask, _ = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            position_ids,
        )
        for layer_idx in range(num_layers):
            source_idx = min(layer_idx, len(teacher_hidden_states) - 1)
            memories.append(gather_batched_positions(teacher_hidden_states[source_idx], image_pos, image_mask))
        return torch.stack(memories, dim=0).to(dtype=base_visual.dtype)
    if mode == "teacher_layer_output":
        memories = []
        _, image_pos, _, _, image_mask, _ = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            position_ids,
        )
        for layer_idx in range(num_layers):
            source_idx = min(layer_idx + 1, len(teacher_hidden_states) - 1)
            memories.append(gather_batched_positions(teacher_hidden_states[source_idx], image_pos, image_mask))
        return torch.stack(memories, dim=0).to(dtype=base_visual.dtype)
    raise ValueError(f"unknown oracle mode: {mode}")


def adapter_diagnostics(
    model: torch.nn.Module,
    adapter: QwenEmbeddingAdapter,
    inputs: dict[str, Tensor],
    text_ids: Tensor,
    answer_mask: Tensor,
    *,
    temperature: float,
    topk: int,
    trajectory_layers: set[int],
    oracle_modes: list[str],
    normalization: str,
    timing_runs: int,
) -> dict[str, Any]:
    with torch.no_grad():
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        language_model = model.model.language_model
        num_layers = len(language_model.layers)
        position_ids = qwen_position_ids(model, inputs)
        text_pos, image_pos, _, text_mask, image_mask, _ = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            position_ids,
        )
        teacher_logits = gather_batched_positions(teacher.logits.detach(), text_pos, text_mask)
        teacher_text_states = {
            idx: gather_batched_positions(teacher.hidden_states[idx].detach(), text_pos, text_mask)
            for idx in sorted(trajectory_layers)
            if idx < len(teacher.hidden_states)
        }
        initial_hidden = teacher.hidden_states[0].detach()
        prepared = prepare_qwen_embedding_adapter_inputs(
            model,
            adapter,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            initial_hidden,
            position_ids,
            reuse_position_embeddings=True,
        )
        from src.model import qwen_embedding_adapter_logits_prepared

        student_logits, student_text_mask, student_states = qwen_embedding_adapter_logits_prepared(
            model,
            adapter,
            **prepared_forward_kwargs(prepared),
            collect_states=True,
            collect_state_indices=trajectory_layers,
        )
        if student_states is None:
            raise RuntimeError("adapter states were not collected")
        token_kl, token_mask = per_token_topk_kl(
            student_logits,
            teacher_logits,
            text_ids,
            answer_mask,
            temperature=temperature,
            topk=topk,
        )
        normed_kl = masked_token_mean(token_kl, token_mask) if normalization == "token" else masked_sample_mean(token_kl, token_mask)
        layer_rows = []
        for idx in sorted(trajectory_layers):
            if idx not in teacher_text_states or idx >= len(student_states) or student_states[idx].numel() == 0:
                continue
            pred_state = student_states[idx]
            if idx == num_layers:
                pred_state = language_model.norm(pred_state)
            traj = masked_directional_mse(
                pred_state,
                teacher_text_states[idx].to(dtype=pred_state.dtype),
                student_text_mask,
                normalization=normalization,
            )
            layer_rows.append(
                {
                    "mode": "adapter",
                    "layer": idx,
                    "trajectory": scalar(traj),
                    "student_norm": scalar(pred_state.detach().float().pow(2).mean(dim=-1).sqrt()[student_text_mask].mean()),
                    "teacher_norm": scalar(teacher_text_states[idx].detach().float().pow(2).mean(dim=-1).sqrt()[student_text_mask].mean()),
                }
            )

        all_adapter_memories = adapter.all_visual_memories_batched(prepared["visual_memory"])
        visual_delta = all_adapter_memories - prepared["visual_memory"].unsqueeze(0)
        adapter_visual_rows = []
        for layer_idx in range(adapter.num_layers):
            adapter_visual_rows.append(
                {
                    "layer": layer_idx + 1,
                    "visual_memory_norm": scalar(all_adapter_memories[layer_idx].detach().float().pow(2).mean(dim=-1).sqrt()[image_mask].mean()),
                    "visual_delta_norm": scalar(visual_delta[layer_idx].detach().float().pow(2).mean(dim=-1).sqrt()[image_mask].mean()),
                }
            )

        oracle_rows = [
            {
                "mode": "adapter",
                "kl": scalar(normed_kl),
                "token_kl_mean": scalar(masked_token_mean(token_kl, token_mask)),
                "sample_kl_mean": scalar(masked_sample_mean(token_kl, token_mask)),
                "top1_agreement": top1_agreement(student_logits, teacher_logits, token_mask),
            }
        ]
        oracle_layer_rows = []
        timing_rows = []
        if timing_runs > 0:
            from src.model import qwen_embedding_adapter_logits_prepared

            def time_fn(name: str, fn) -> None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fn()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(int(timing_runs)):
                    fn()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / max(1, int(timing_runs))
                timing_rows.append({"mode": name, "seconds": seconds, "ms": seconds * 1000.0})

            time_fn(
                "adapter_static_src",
                lambda: qwen_embedding_adapter_logits_prepared(
                    model,
                    adapter,
                    **prepared_forward_kwargs(prepared),
                    collect_states=False,
                    collect_state_indices=None,
                ),
            )
        for mode in oracle_modes:
            if mode == "adapter":
                continue
            all_memories = build_oracle_memories(
                mode,
                prepared=prepared,
                teacher_hidden_states=teacher.hidden_states,
                position_ids=position_ids,
                inputs=inputs,
                adapter=adapter,
            )
            if timing_runs > 0 and mode == "recurrent_adapter":
                time_fn(
                    "recurrent_adapter_test",
                    lambda all_memories=all_memories: oracle_logits_from_visual_memories(
                        model,
                        adapter,
                        prepared,
                        all_memories,
                        collect_state_indices=set(),
                    ),
                )
            oracle_logits, oracle_text_mask, oracle_states = oracle_logits_from_visual_memories(
                model,
                adapter,
                prepared,
                all_memories,
                collect_state_indices=trajectory_layers,
            )
            oracle_token_kl, oracle_token_mask = per_token_topk_kl(
                oracle_logits,
                teacher_logits,
                text_ids,
                answer_mask,
                temperature=temperature,
                topk=topk,
            )
            oracle_kl = (
                masked_token_mean(oracle_token_kl, oracle_token_mask)
                if normalization == "token"
                else masked_sample_mean(oracle_token_kl, oracle_token_mask)
            )
            oracle_rows.append(
                {
                    "mode": mode,
                    "kl": scalar(oracle_kl),
                    "token_kl_mean": scalar(masked_token_mean(oracle_token_kl, oracle_token_mask)),
                    "sample_kl_mean": scalar(masked_sample_mean(oracle_token_kl, oracle_token_mask)),
                    "top1_agreement": top1_agreement(oracle_logits, teacher_logits, oracle_token_mask),
                }
            )
            for idx in sorted(trajectory_layers):
                if idx not in teacher_text_states or idx >= len(oracle_states) or oracle_states[idx].numel() == 0:
                    continue
                pred_state = oracle_states[idx]
                if idx == num_layers:
                    pred_state = language_model.norm(pred_state)
                traj = masked_directional_mse(
                    pred_state,
                    teacher_text_states[idx].to(dtype=pred_state.dtype),
                    oracle_text_mask,
                    normalization=normalization,
                )
                oracle_layer_rows.append(
                    {
                        "mode": mode,
                        "layer": idx,
                        "trajectory": scalar(traj),
                        "student_norm": scalar(pred_state.detach().float().pow(2).mean(dim=-1).sqrt()[oracle_text_mask].mean()),
                        "teacher_norm": scalar(teacher_text_states[idx].detach().float().pow(2).mean(dim=-1).sqrt()[oracle_text_mask].mean()),
                    }
                )

        token_summary = masked_stats(token_kl, token_mask)
        per_sample = []
        valid = token_mask.to(dtype=token_kl.dtype)
        sample_kl = (token_kl * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        answer_counts = valid.sum(dim=1)
        for batch_idx in range(token_kl.shape[0]):
            per_sample.append(
                {
                    "sample": batch_idx,
                    "answer_tokens": scalar(answer_counts[batch_idx]),
                    "adapter_kl": scalar(sample_kl[batch_idx]),
                }
            )
        norm_rows = [
            tensor_norm_stats("teacher_initial_text", prepared["h"], prepared["text_mask"]),
            tensor_norm_stats("teacher_initial_visual", prepared["visual_memory"], image_mask),
        ]
        return {
            "summary": {
                "batch_size": int(inputs["input_ids"].shape[0]),
                "sequence_len": int(inputs["input_ids"].shape[1]),
                "text_tokens": int(text_mask.sum().item()),
                "image_tokens": int(image_mask.sum().item()),
                "answer_tokens": int(token_mask.sum().item()),
                "normalization": normalization,
                "temperature": float(temperature),
                "topk": int(topk),
                "visual_update_mode": str(getattr(adapter, "visual_update_mode", "static")),
                "adapter_kl": scalar(normed_kl),
                "adapter_token_kl": token_summary,
                "adapter_top1_agreement": top1_agreement(student_logits, teacher_logits, token_mask),
                "has_gate": bool(gate_rows(adapter)),
            },
            "oracle_rows": oracle_rows,
            "layer_rows": layer_rows + oracle_layer_rows,
            "visual_rows": adapter_visual_rows,
            "norm_rows": norm_rows,
            "gate_rows": gate_rows(adapter),
            "timing_rows": timing_rows,
            "per_sample_rows": per_sample,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen embedding_adapter plateau diagnostics")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "test/results/qwen_plateau_diagnostics"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-topk", type=int, default=1024)
    parser.add_argument("--loss-normalization", choices=("token", "sample"), default="token")
    parser.add_argument("--trajectory-layers", default="4,8,12,16,20,24,28,32,36")
    parser.add_argument(
        "--oracle-modes",
        default="adapter,recurrent_adapter,initial,teacher_layer_input,teacher_layer_output",
        help="Comma-separated modes: adapter, recurrent_adapter, initial, teacher_layer_input, teacher_layer_output.",
    )
    parser.add_argument("--timing-runs", type=int, default=0, help="Optional timing runs for static src vs test-only recurrent adapter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    adapter, meta = load_qwen_embedding_adapter_checkpoint(args.checkpoint, model.model.language_model, device, dtype)
    adapter.eval()
    rows = read_jsonl_rows(Path(args.data), start_index=args.start_index, count=args.batch_size)
    inputs, text_ids, answer_mask, image_paths = prepare_qwen3vl_batch_inputs(
        processor,
        rows,
        Path(args.image_root) if args.image_root else None,
        device,
        include_answers=True,
    )
    if text_ids is None or answer_mask is None:
        raise RuntimeError("diagnostics require answer labels")
    num_layers = len(model.model.language_model.layers)
    trajectory_layers = parse_trajectory_layers(args.trajectory_layers, num_layers)
    oracle_modes = [mode.strip() for mode in args.oracle_modes.split(",") if mode.strip()]
    diagnostics = adapter_diagnostics(
        model,
        adapter,
        inputs,
        text_ids,
        answer_mask,
        temperature=float(args.temperature),
        topk=int(args.kl_topk),
        trajectory_layers=trajectory_layers,
        oracle_modes=oracle_modes,
        normalization=str(args.loss_normalization),
        timing_runs=int(args.timing_runs),
    )
    payload = {
        "args": vars(args),
        "checkpoint_meta": meta,
        "image_paths": image_paths,
        "rows": rows,
        **diagnostics,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "oracle.csv", diagnostics["oracle_rows"])
    write_csv(output_dir / "trajectory_by_layer.csv", diagnostics["layer_rows"])
    write_csv(output_dir / "adapter_visual_norms.csv", diagnostics["visual_rows"])
    write_csv(output_dir / "norms.csv", diagnostics["norm_rows"])
    write_csv(output_dir / "gates.csv", diagnostics["gate_rows"])
    write_csv(output_dir / "timing.csv", diagnostics["timing_rows"])
    write_csv(output_dir / "per_sample.csv", diagnostics["per_sample_rows"])

    print("=== Qwen embedding_adapter plateau diagnostics ===")
    print(f"output_dir={output_dir}")
    print(json.dumps(diagnostics["summary"], indent=2, ensure_ascii=False))
    print("oracle:")
    for row in diagnostics["oracle_rows"]:
        print(
            f"  {row['mode']:22} kl={row['kl']:.6f} "
            f"token_kl={row['token_kl_mean']:.6f} sample_kl={row['sample_kl_mean']:.6f} "
            f"top1_agreement={row['top1_agreement']:.4f}"
        )
    if diagnostics["timing_rows"]:
        print("timing:")
        for row in diagnostics["timing_rows"]:
            print(f"  {row['mode']:22} {row['ms']:.2f} ms")


if __name__ == "__main__":
    main()
