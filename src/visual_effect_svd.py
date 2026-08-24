"""OCR copy diagnostic for low-rank native visual attention effects.

This module is standalone inside this repository. It evaluates whether the
native Qwen3-VL visual-token attention contribution to text tokens can be
approximated by a low-rank matrix without hurting copy/OCR token logits.

The evaluation is teacher-forced: the gold copy text is included in the prompt,
then answer-token logits are decoded by argmax and scored against the reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers.masking_utils import create_causal_mask

from src.model import (
    build_qwen_initial_context,
    dtype_from_name,
    gather_batched_positions,
    get_qwen_text_image_positions,
    load_frozen_qwen3vl,
    prepare_qwen3vl_batch_inputs,
)
from src.ocr_eval import COPY_TRANSCRIPTION_INSTRUCTION, score_text, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OCR copy native visual-effect SVD diagnostic.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/rendered_text_copy_300/paired_eval.jsonl")
    parser.add_argument("--image-root", default="data/train/rendered_text_copy_300")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def load_rows(path: Path, max_samples: int, num_shards: int, shard_id: int) -> tuple[list[dict[str, Any]], int, int, int]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    total = len(rows)
    per_shard = (total + int(num_shards) - 1) // int(num_shards)
    start = int(shard_id) * per_shard
    end = min(start + per_shard, total)
    return rows[start:end], total, start, end


def prepare_copy_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["question"] = COPY_TRANSCRIPTION_INSTRUCTION
    return out


def qwen_attention_output(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_mask_2d: Tensor,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    normed = layer.input_layernorm(hidden_states)
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=normed,
        attention_mask=attention_mask_2d.to(dtype=torch.long),
        past_key_values=None,
        position_ids=text_position_ids,
    )
    attn_output, _ = layer.self_attn(
        hidden_states=normed,
        position_embeddings=language_model.rotary_emb(normed, position_ids),
        attention_mask=attention_mask,
        past_key_values=None,
    )
    return attn_output


def qwen_layer_text_with_attention_delta(
    language_model: torch.nn.Module,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
    attention_delta: Tensor | None,
    text_mask: Tensor,
) -> Tensor:
    layer = language_model.layers[layer_idx]
    residual = hidden_states
    attn_out = qwen_attention_output(language_model, layer_idx, hidden_states, position_ids, text_mask)
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(dtype=hidden_states.dtype)
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def qwen_visual_attention_effect(
    language_model: torch.nn.Module,
    layer_idx: int,
    full_hidden_states: Tensor,
    text_hidden_states: Tensor,
    full_position_ids: Tensor,
    text_position_ids: Tensor,
    text_positions: Tensor,
    full_mask: Tensor,
    text_mask: Tensor,
) -> Tensor:
    joint = qwen_attention_output(language_model, layer_idx, full_hidden_states, full_position_ids, full_mask)
    text = qwen_attention_output(language_model, layer_idx, text_hidden_states, text_position_ids, text_mask)
    joint_text = gather_batched_positions(joint, text_positions, text_mask)
    return joint_text - text


def low_rank_delta(delta: Tensor, text_mask: Tensor, rank: int) -> Tensor:
    out = torch.zeros_like(delta)
    for batch_idx in range(delta.shape[0]):
        valid = text_mask[batch_idx].to(device=delta.device).bool()
        x = delta[batch_idx, valid]
        if x.numel() == 0:
            continue
        k = min(int(rank), x.shape[0], x.shape[1])
        if k >= min(x.shape[0], x.shape[1]):
            out[batch_idx, valid] = x
            continue
        u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
        out[batch_idx, valid] = ((u[:, :k] * s[:k]) @ vh[:k]).to(dtype=delta.dtype)
    return out


def answer_prediction_from_logits(processor: Any, logits: Tensor, answer_mask: Tensor) -> str:
    pred_ids: list[int] = []
    shifted = answer_mask[:, 1:].bool()
    pred = logits[:, :-1].argmax(dim=-1)
    for token_id in pred[0, shifted[0]].detach().cpu().tolist():
        pred_ids.append(int(token_id))
    return processor.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def run_text_oracle(
    model: Any,
    language_model: torch.nn.Module,
    inputs: dict[str, Tensor],
    *,
    rank: int | None,
    dtype: torch.dtype,
) -> Tensor:
    teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden0, full_position_ids = build_qwen_initial_context(model, inputs)
    text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states]
    teacher_text_states = [
        gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype) for state in teacher_states
    ]
    h = teacher_text_states[0]
    for layer_idx in range(len(language_model.layers)):
        delta = qwen_visual_attention_effect(
            language_model,
            layer_idx,
            teacher_states[layer_idx],
            teacher_text_states[layer_idx],
            full_position_ids,
            text_position_ids,
            text_positions,
            full_mask,
            text_mask,
        )
        if rank is not None:
            delta = low_rank_delta(delta, text_mask, rank)
        h = qwen_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            text_position_ids,
            delta.masked_fill(~text_mask.unsqueeze(-1), 0.0),
            text_mask,
        )
    return model.lm_head(language_model.norm(h))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, total, start, end = load_rows(Path(args.data), args.max_samples, args.num_shards, args.shard_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    processor, model = load_frozen_qwen3vl(args.model_path, dtype, device, args.attn_implementation)
    language_model = model.model.language_model

    scores_full: list[dict[str, float]] = []
    scores_rank: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []
    elapsed = 0.0
    print(f"rank={args.rank} shard={args.shard_id}/{args.num_shards} samples=[{start},{end})/{total}", flush=True)
    for local_idx, row in enumerate(rows, start=1):
        t0 = time.perf_counter()
        inputs, text_ids, answer_mask, _ = prepare_qwen3vl_batch_inputs(
            processor,
            [prepare_copy_row(row)],
            Path(args.image_root),
            device,
            include_answers=True,
        )
        assert text_ids is not None and answer_mask is not None
        full_logits = run_text_oracle(model, language_model, inputs, rank=None, dtype=dtype)
        rank_logits = run_text_oracle(model, language_model, inputs, rank=args.rank, dtype=dtype)
        elapsed += time.perf_counter() - t0
        ref = str(row.get("answer") or "")
        full_pred = answer_prediction_from_logits(processor, full_logits, answer_mask)
        rank_pred = answer_prediction_from_logits(processor, rank_logits, answer_mask)
        full_score = score_text(full_pred, ref)
        rank_score = score_text(rank_pred, ref)
        scores_full.append(full_score)
        scores_rank.append(rank_score)
        predictions.append(
            {
                "index": row.get("index", start + local_idx - 1),
                "answer": ref,
                "full_effect_prediction": full_pred,
                "rank_effect_prediction": rank_pred,
                "full_effect_eval": full_score,
                "rank_effect_eval": rank_score,
            }
        )
        if local_idx % int(args.log_every) == 0 or local_idx == len(rows):
            full = summarize(scores_full)
            rank = summarize(scores_rank)
            print(
                f"[{local_idx}/{len(rows)}] full_f1={full['token_f1']:.4f} rank{args.rank}_f1={rank['token_f1']:.4f} "
                f"full_cer={full['cer']:.4f} rank_cer={rank['cer']:.4f}",
                flush=True,
            )

    result = {
        "task": "ocr_copy_native_visual_effect_svd",
        "rank": int(args.rank),
        "data": str(args.data),
        "total_samples": len(predictions),
        "source_total_samples": total,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "sample_start": start,
        "sample_end": end,
        "metrics": {
            "full_effect": summarize(scores_full),
            f"rank_{int(args.rank)}": summarize(scores_rank),
        },
        "timing": {"total_s": elapsed, "avg_s": elapsed / max(1, len(predictions))},
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
