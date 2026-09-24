"""Low-rank oracle diagnostic for native Qwen3-VL visual attention effects.

This follows the oracle path used in ``delta-vision``:

    DeltaA_l = A_l^joint(text positions) - A_l^text-only

The script first builds a per-layer PCA/SVD basis from sampled ``DeltaA_l``
tokens. Evaluation then rolls out the text-only language model and injects
either the full true ``DeltaA_l`` or its low-rank projection after the attention
residual and before the MLP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import parse_qwen_device_map, parse_qwen_max_memory
from src.model import (
    build_qwen_initial_context,
    dtype_from_name,
    gather_batched_positions,
    get_qwen_text_image_positions,
    load_frozen_qwen3vl,
    module_device,
    prepare_qwen3vl_batch_inputs,
    qwen_input_device,
)
from analysis.common.text_reconstruction_metrics import COPY_TRANSCRIPTION_INSTRUCTION, score_text, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL visual-effect low-rank oracle diagnostic.")
    parser.add_argument("--model-path", default="/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", default="data/train/rendered_text_copy_300/paired_eval.jsonl")
    parser.add_argument("--image-root", default="data/train/rendered_text_copy_300")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basis", default="", help="Basis checkpoint path. Defaults to output-dir/visual_effect_basis_rank{max_rank}.pt.")
    parser.add_argument("--reuse-basis", action="store_true", help="Load --basis if it exists instead of rebuilding it.")
    parser.add_argument(
        "--mode",
        choices=("build-and-eval", "build-basis", "eval"),
        default="build-and-eval",
        help="Build the PCA basis, evaluate with an existing basis, or do both.",
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Rows used for evaluation. 0 means all rows.")
    parser.add_argument(
        "--basis-max-samples",
        type=int,
        default=0,
        help="Rows used to build the basis. 0 means use --max-samples rows.",
    )
    parser.add_argument("--max-rank", type=int, default=512)
    parser.add_argument("--ranks", default="32,64,128,256,512")
    parser.add_argument("--max-tokens-per-layer", type=int, default=16384)
    parser.add_argument("--basis-token-mode", choices=("all", "answer", "prompt"), default="all")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--qwen-device-map", default="", help="Optional HF device_map, e.g. auto.")
    parser.add_argument("--qwen-max-memory", default="", help="Optional HF max_memory JSON or comma list.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--compute-kl", action="store_true", help="Compute full-vocab KL to teacher on answer-token logits.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def select_contiguous_shard(rows: list[dict[str, Any]], num_shards: int, shard_id: int) -> tuple[list[dict[str, Any]], int, int]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    per_shard = (len(rows) + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, len(rows))
    selected = rows[start:end]
    if not selected:
        raise RuntimeError(f"no rows selected for shard {shard_id}/{num_shards}")
    return selected, start, end


def parse_ranks(spec: str, max_rank: int) -> list[int]:
    ranks = sorted({int(item) for item in str(spec).split(",") if item.strip()})
    if not ranks:
        raise ValueError("--ranks cannot be empty")
    if ranks[0] <= 0:
        raise ValueError("--ranks must be positive")
    if ranks[-1] > max_rank:
        raise ValueError(f"requested rank {ranks[-1]} exceeds --max-rank {max_rank}")
    return ranks


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
    device = module_device(layer, hidden_states.device)
    hidden_states = hidden_states.to(device=device)
    position_ids = position_ids.to(device=device)
    attention_mask_2d = attention_mask_2d.to(device=device)

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
    device = module_device(layer, hidden_states.device)
    hidden_states = hidden_states.to(device=device)
    position_ids = position_ids.to(device=device)
    text_mask = text_mask.to(device=device)

    residual = hidden_states
    attn_out = qwen_attention_output(language_model, layer_idx, hidden_states, position_ids, text_mask)
    hidden_states = residual + attn_out
    if attention_delta is not None:
        hidden_states = hidden_states + attention_delta.to(device=hidden_states.device, dtype=hidden_states.dtype)

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
    joint_text = gather_batched_positions(joint, text_positions.to(device=joint.device), text_mask.to(device=joint.device))
    return joint_text - text


def basis_token_mask(text_mask: Tensor, answer_mask: Tensor, mode: str) -> Tensor:
    if mode == "all":
        return text_mask
    if mode == "answer":
        return text_mask & answer_mask
    if mode == "prompt":
        return text_mask & ~answer_mask
    raise ValueError(f"unsupported basis token mode: {mode}")


def project_reconstruct_delta(delta: Tensor, layer_basis: Tensor, rank: int) -> Tensor:
    basis = layer_basis[:rank].to(device=delta.device)
    coeff = torch.matmul(delta.float(), basis.float().transpose(0, 1))
    return torch.matmul(coeff, basis.float()).to(dtype=delta.dtype)


def save_basis_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


@torch.inference_mode()
def build_attention_basis(
    *,
    rows: list[dict[str, Any]],
    processor: Any,
    model: Any,
    language_model: torch.nn.Module,
    image_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_rank: int,
    max_tokens_per_layer: int,
    token_mode: str,
    log_every: int,
) -> dict[str, Any]:
    num_layers = len(language_model.layers)
    hidden_size = int(language_model.config.hidden_size)
    banks: dict[int, list[Tensor]] = {layer_idx: [] for layer_idx in range(num_layers)}
    counts = {layer_idx: 0 for layer_idx in range(num_layers)}

    for sample_idx, row in enumerate(rows, start=1):
        inputs, _text_ids, answer_mask, _ = prepare_qwen3vl_batch_inputs(
            processor,
            [prepare_copy_row(row)],
            image_root,
            device,
            include_answers=True,
        )
        if answer_mask is None:
            raise RuntimeError("answer_mask missing while building visual-effect basis")
        teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        _initial_hidden, full_position_ids = build_qwen_initial_context(model, inputs)
        text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen_text_image_positions(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["mm_token_type_ids"],
            full_position_ids,
        )
        teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states[:-1]]
        teacher_text_states = [
            gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype) for state in teacher_states
        ]
        keep_mask = basis_token_mask(text_mask, answer_mask.to(device=text_mask.device), token_mode)

        for layer_idx in range(num_layers):
            if counts[layer_idx] >= max_tokens_per_layer:
                continue
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
            valid = keep_mask.to(device=delta.device).bool()
            tokens = delta[valid]
            if tokens.numel() == 0:
                continue
            remaining = max_tokens_per_layer - counts[layer_idx]
            if tokens.shape[0] > remaining:
                perm = torch.randperm(tokens.shape[0], device=tokens.device)[:remaining]
                tokens = tokens.index_select(0, perm)
            banks[layer_idx].append(tokens.detach().to(device="cpu", dtype=torch.float16))
            counts[layer_idx] += int(tokens.shape[0])

        if sample_idx % int(log_every) == 0 or sample_idx == len(rows):
            print(
                f"basis collect {sample_idx}/{len(rows)} min_tokens={min(counts.values())} max_tokens={max(counts.values())}",
                flush=True,
            )
        if all(counts[layer_idx] >= max_tokens_per_layer for layer_idx in range(num_layers)):
            print(f"basis token banks full after {sample_idx}/{len(rows)} samples", flush=True)
            break

    bases: dict[int, Tensor] = {}
    energies: dict[int, Tensor] = {}
    svd_device = qwen_input_device(model)
    for layer_idx in range(num_layers):
        if not banks[layer_idx]:
            raise RuntimeError(f"empty SVD token bank for layer {layer_idx}; try a larger --basis-max-samples")
        matrix = torch.cat(banks[layer_idx], dim=0).float()
        if matrix.shape[1] != hidden_size:
            raise RuntimeError(f"layer {layer_idx} hidden size mismatch: got {matrix.shape[1]} expected {hidden_size}")
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        matrix = matrix.to(device=svd_device)
        _, svals, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(int(max_rank), int(vh.shape[0]))
        bases[layer_idx] = vh[:rank].detach().cpu().to(torch.float16)
        energies[layer_idx] = svals.detach().cpu().float().pow(2)
        print(f"basis svd layer={layer_idx} tokens={matrix.shape[0]} rank={rank}", flush=True)

    return {
        "format_version": 2,
        "basis": bases,
        "singular_energy": energies,
        "counts": counts,
        "max_rank": int(max_rank),
        "max_tokens_per_layer": int(max_tokens_per_layer),
        "basis_token_mode": token_mode,
        "num_basis_rows": len(rows),
    }


def load_layer_basis(path: Path, *, max_rank: int, num_layers: int, hidden_size: int, dtype: torch.dtype) -> Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    raw_basis = loaded["basis"]
    if isinstance(raw_basis, dict):
        layers = []
        for layer_idx in range(num_layers):
            key: Any = layer_idx if layer_idx in raw_basis else str(layer_idx)
            if key not in raw_basis:
                raise ValueError(f"basis checkpoint is missing layer {layer_idx}")
            layers.append(raw_basis[key].float())
        basis = torch.stack(layers, dim=0)
    else:
        basis = raw_basis.float()
    if basis.ndim != 3:
        raise ValueError(f"basis must have shape [layers, rank, hidden], got {tuple(basis.shape)}")
    if basis.shape[0] != num_layers:
        raise ValueError(f"basis has {basis.shape[0]} layers, expected {num_layers}")
    if basis.shape[2] != hidden_size:
        raise ValueError(f"basis hidden size {basis.shape[2]} does not match model hidden size {hidden_size}")
    if max_rank > basis.shape[1]:
        raise ValueError(f"requested max rank {max_rank} exceeds stored rank {basis.shape[1]}")
    return basis[:, :max_rank].contiguous().to(dtype=dtype)


def answer_prediction_from_logits(processor: Any, logits: Tensor, answer_mask: Tensor) -> str:
    mask = answer_mask[:, 1:].to(device=logits.device).bool()
    if not bool(mask.any().item()):
        return ""
    pred = logits[:, :-1].argmax(dim=-1)
    pred_ids = [int(token_id) for token_id in pred[mask].detach().cpu().tolist()]
    return processor.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def answer_kl_to_teacher(student_logits: Tensor, teacher_logits: Tensor, answer_mask: Tensor) -> tuple[float, int]:
    mask = answer_mask[:, 1:].to(device=student_logits.device).bool()
    if not bool(mask.any().item()):
        return 0.0, 0
    student = student_logits[:, :-1][mask].float()
    teacher = teacher_logits.to(device=student_logits.device)[:, :-1][mask].float()
    kl = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="batchmean")
    return float(kl.item()), int(student.shape[0])


def score_prediction(processor: Any, prediction: str, reference: str) -> dict[str, float]:
    score = score_text(prediction, reference)
    score["pred_tokens"] = float(len(processor.tokenizer(str(prediction), add_special_tokens=False).input_ids))
    return score


def logits_from_text_hidden(model: Any, language_model: torch.nn.Module, hidden_states: Tensor) -> Tensor:
    norm_device = module_device(language_model.norm, hidden_states.device)
    normed = language_model.norm(hidden_states.to(device=norm_device))
    head_device = module_device(model.lm_head, normed.device)
    return model.lm_head(normed.to(device=head_device))


@torch.inference_mode()
def build_eval_trace(
    *,
    row: dict[str, Any],
    processor: Any,
    model: Any,
    language_model: torch.nn.Module,
    image_root: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs, text_ids, answer_mask, _ = prepare_qwen3vl_batch_inputs(
        processor,
        [prepare_copy_row(row)],
        image_root,
        device,
        include_answers=True,
    )
    if text_ids is None or answer_mask is None:
        raise RuntimeError("teacher-forced OCR eval requires text_ids and answer_mask")
    teacher = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    _initial_hidden, full_position_ids = build_qwen_initial_context(model, inputs)
    text_positions, _, text_position_ids, text_mask, _, full_mask = get_qwen_text_image_positions(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["mm_token_type_ids"],
        full_position_ids,
    )
    teacher_states = [state.detach().to(dtype=dtype) for state in teacher.hidden_states[:-1]]
    teacher_text_states = [
        gather_batched_positions(state, text_positions, text_mask).detach().to(dtype=dtype) for state in teacher_states
    ]
    teacher_text_logits = gather_batched_positions(teacher.logits.detach(), text_positions, text_mask)
    attention_deltas: list[Tensor] = []
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
        delta = delta.masked_fill(~text_mask.to(device=delta.device).unsqueeze(-1), 0.0)
        attention_deltas.append(delta.detach().cpu().to(torch.float16))

    return {
        "text_hidden0": teacher_text_states[0].detach().cpu().to(torch.float16),
        "text_position_ids": text_position_ids.detach().cpu(),
        "text_mask": text_mask.detach().cpu(),
        "answer_mask": answer_mask.detach().cpu(),
        "teacher_text_logits": teacher_text_logits,
        "attention_deltas": attention_deltas,
    }


@torch.inference_mode()
def run_text_rollout(
    *,
    model: Any,
    language_model: torch.nn.Module,
    trace: dict[str, Any],
    mode: str,
    basis: Tensor | None,
    rank: int | None,
    dtype: torch.dtype,
) -> Tensor:
    if mode not in {"no_visual", "full_effect", "rank"}:
        raise ValueError(f"unsupported rollout mode: {mode}")
    if mode == "rank" and (basis is None or rank is None):
        raise ValueError("rank rollout requires basis and rank")

    h = trace["text_hidden0"].to(device=qwen_input_device(model), dtype=dtype)
    text_position_ids = trace["text_position_ids"]
    text_mask = trace["text_mask"]
    attention_deltas = trace["attention_deltas"]
    for layer_idx in range(len(language_model.layers)):
        delta = None
        if mode != "no_visual":
            delta = attention_deltas[layer_idx].to(device=h.device, dtype=dtype)
            if mode == "rank":
                assert basis is not None and rank is not None
                layer_basis = basis[layer_idx].to(device=delta.device, dtype=dtype)
                delta = project_reconstruct_delta(delta, layer_basis, rank)
        h = qwen_layer_text_with_attention_delta(
            language_model,
            layer_idx,
            h,
            text_position_ids,
            delta,
            text_mask,
        )
    return logits_from_text_hidden(model, language_model, h)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "singular_energy":
            continue
        if isinstance(value, dict):
            safe[key] = {str(k): int(v) if isinstance(v, int) else v for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_path = Path(args.basis) if args.basis else out_dir / f"visual_effect_basis_rank{int(args.max_rank)}.pt"
    image_root = Path(args.image_root)
    ranks = parse_ranks(args.ranks, int(args.max_rank))

    eval_limit = None if int(args.max_samples) == 0 else int(args.max_samples)
    eval_source_rows = read_jsonl(Path(args.data), eval_limit)
    eval_rows, sample_start, sample_end = select_contiguous_shard(eval_source_rows, int(args.num_shards), int(args.shard_id))

    basis_limit = int(args.basis_max_samples) if int(args.basis_max_samples) > 0 else int(args.max_samples)
    basis_limit = None if basis_limit == 0 else basis_limit
    basis_rows = read_jsonl(Path(args.data), basis_limit)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    qwen_device_map = parse_qwen_device_map(args.qwen_device_map)
    qwen_max_memory = parse_qwen_max_memory(args.qwen_max_memory)
    processor, model = load_frozen_qwen3vl(
        args.model_path,
        dtype,
        device,
        args.attn_implementation,
        device_map=qwen_device_map,
        max_memory=qwen_max_memory,
    )
    if qwen_device_map is not None:
        device = qwen_input_device(model)
        print(f"qwen_device_map={qwen_device_map} input_device={device} max_memory={qwen_max_memory or 'auto'}", flush=True)
    language_model = model.model.language_model
    num_layers = len(language_model.layers)
    hidden_size = int(language_model.config.hidden_size)

    basis_meta: dict[str, Any] = {}
    if args.mode in {"build-and-eval", "build-basis"}:
        if args.reuse_basis and basis_path.exists():
            print(f"reuse basis: {basis_path}", flush=True)
            loaded_meta = torch.load(basis_path, map_location="cpu", weights_only=False)
            basis_meta = {key: value for key, value in loaded_meta.items() if key != "basis"}
        else:
            print(
                f"build basis rows={len(basis_rows)} max_rank={args.max_rank} "
                f"max_tokens_per_layer={args.max_tokens_per_layer} token_mode={args.basis_token_mode}",
                flush=True,
            )
            payload = build_attention_basis(
                rows=basis_rows,
                processor=processor,
                model=model,
                language_model=language_model,
                image_root=image_root,
                device=device,
                dtype=dtype,
                max_rank=int(args.max_rank),
                max_tokens_per_layer=int(args.max_tokens_per_layer),
                token_mode=args.basis_token_mode,
                log_every=int(args.log_every),
            )
            payload.update(
                {
                    "task": "qwen3vl_visual_attention_effect_pca_basis",
                    "data": str(args.data),
                    "image_root": str(args.image_root),
                    "model_path": str(args.model_path),
                    "seed": int(args.seed),
                }
            )
            save_basis_atomic(payload, basis_path)
            basis_meta = {key: value for key, value in payload.items() if key != "basis"}
            print(f"wrote basis: {basis_path}", flush=True)

    if args.mode == "build-basis":
        result = {
            "task": "qwen3vl_visual_effect_svd_build_basis",
            "basis": str(basis_path),
            "basis_meta": json_safe_meta(basis_meta),
        }
        (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return

    if not basis_path.exists():
        raise FileNotFoundError(f"basis not found: {basis_path}; run --mode build-basis first or use --mode build-and-eval")
    if not basis_meta:
        loaded_meta = torch.load(basis_path, map_location="cpu", weights_only=False)
        basis_meta = {key: value for key, value in loaded_meta.items() if key != "basis"}
    basis = load_layer_basis(basis_path, max_rank=int(args.max_rank), num_layers=num_layers, hidden_size=hidden_size, dtype=dtype)
    available_rank = int(basis.shape[1])
    for rank in ranks:
        if rank > available_rank:
            raise ValueError(f"requested rank {rank} exceeds available basis rank {available_rank}")

    mode_names = ["teacher", "no_visual", "full_effect"] + [f"rank_{rank}" for rank in ranks]
    scores: dict[str, list[dict[str, float]]] = {name: [] for name in mode_names}
    timings: dict[str, float] = {name: 0.0 for name in mode_names}
    kl_sums: dict[str, float] = {name: 0.0 for name in mode_names}
    kl_counts: dict[str, int] = {name: 0 for name in mode_names}
    predictions: list[dict[str, Any]] = []

    print(
        f"eval rows={len(eval_rows)} shard={args.shard_id}/{args.num_shards} "
        f"samples=[{sample_start},{sample_end}) ranks={','.join(str(rank) for rank in ranks)}",
        flush=True,
    )
    total_start = time.perf_counter()
    for local_idx, row in enumerate(eval_rows, start=1):
        reference = str(row.get("answer") or "")
        trace_start = time.perf_counter()
        trace = build_eval_trace(
            row=row,
            processor=processor,
            model=model,
            language_model=language_model,
            image_root=image_root,
            device=device,
            dtype=dtype,
        )
        teacher_logits = trace["teacher_text_logits"]
        answer_mask = trace["answer_mask"]
        timings["teacher"] += time.perf_counter() - trace_start

        record: dict[str, Any] = {
            "index": row.get("index", sample_start + local_idx - 1),
            "id": row.get("id", sample_start + local_idx - 1),
            "answer": reference,
        }

        teacher_pred = answer_prediction_from_logits(processor, teacher_logits, answer_mask)
        teacher_score = score_prediction(processor, teacher_pred, reference)
        scores["teacher"].append(teacher_score)
        record["teacher_prediction"] = teacher_pred
        record["teacher_eval"] = teacher_score

        rollout_specs: list[tuple[str, str, int | None]] = [("no_visual", "no_visual", None), ("full_effect", "full_effect", None)]
        rollout_specs.extend((f"rank_{rank}", "rank", rank) for rank in ranks)
        for name, rollout_mode, rank in rollout_specs:
            start = time.perf_counter()
            logits = run_text_rollout(
                model=model,
                language_model=language_model,
                trace=trace,
                mode=rollout_mode,
                basis=basis,
                rank=rank,
                dtype=dtype,
            )
            timings[name] += time.perf_counter() - start
            pred = answer_prediction_from_logits(processor, logits, answer_mask)
            score = score_prediction(processor, pred, reference)
            scores[name].append(score)
            record[f"{name}_prediction"] = pred
            record[f"{name}_eval"] = score
            if args.compute_kl:
                kl, token_count = answer_kl_to_teacher(logits, teacher_logits, answer_mask)
                kl_sums[name] += kl * max(token_count, 1)
                kl_counts[name] += token_count

        predictions.append(record)
        if local_idx % int(args.log_every) == 0 or local_idx == len(eval_rows):
            parts = []
            for name in ("teacher", "full_effect", f"rank_{ranks[-1]}"):
                summary = summarize(scores[name])
                parts.append(f"{name}_f1={summary['token_f1']:.4f} {name}_cer={summary['cer']:.4f}")
            print(f"[{local_idx}/{len(eval_rows)}] " + " ".join(parts), flush=True)

    elapsed = time.perf_counter() - total_start
    metrics = {name: summarize(scores[name]) for name in mode_names}
    if args.compute_kl:
        for name in mode_names:
            if name == "teacher":
                metrics[name]["answer_token_kl_to_teacher"] = 0.0
            else:
                metrics[name]["answer_token_kl_to_teacher"] = kl_sums[name] / max(kl_counts[name], 1)

    result = {
        "task": "ocr_copy_qwen3vl_visual_attention_effect_lowrank_oracle",
        "oracle": "DeltaA_l = A_joint_l(text_positions) - A_text_l, injected after attention residual before MLP",
        "data": str(args.data),
        "image_root": str(args.image_root),
        "model_path": str(args.model_path),
        "basis": str(basis_path),
        "basis_meta": json_safe_meta(basis_meta),
        "basis_token_mode": str(args.basis_token_mode),
        "max_rank": int(args.max_rank),
        "ranks": ranks,
        "total_samples": len(predictions),
        "source_total_samples": len(eval_source_rows),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "metrics": metrics,
        "timing": {
            "total_s": elapsed,
            "avg_s": elapsed / max(1, len(predictions)),
            "by_mode_s": timings,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
