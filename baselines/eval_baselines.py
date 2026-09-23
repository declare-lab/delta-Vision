"""Unified baseline evaluation for project baselines and five supported VLMs.

Usage:
    .venv/bin/python baselines/eval_baselines.py \
        --method fastv --retention 0.10 \
        --benchmark mmstar --max-samples 1000
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    .venv/bin/python baselines/eval_baselines.py \
        --method all --retention 0.10 \
        --benchmark all --max-samples 1000
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoConfig, AutoProcessor
from src.benchmarks import (
    format_seconds_minsec, get_benchmark_spec, score_prediction, summarize_metric,
    parse_benchmark_names, canonical_benchmark_name,
)
from src.data import LlavaBenchmarkDataset, QwenBenchmarkDataset
from baselines.llava_hf_baselines import evaluate_llava_baseline, load_llava_baseline_model


DATA_ROOT = os.environ.get("DATA_ROOT", "/lustre-data/leijingdi/code/delta-vision")
MODEL_PATH = os.environ.get("MODEL_PATH", f"{DATA_ROOT}/models/Qwen3-VL-4B-Instruct")

BASE_METHOD = "base"
DYVTE_METHOD = "dyvte"
REDUNDANCYLENS_METHOD = "redundancylens"
BASELINE_METHODS = ["fastv", "dart", "visionzip", "sparsevlm", "divprune", "zoo", DYVTE_METHOD, REDUNDANCYLENS_METHOD]
METHODS = [BASE_METHOD] + BASELINE_METHODS
NO_TRAIN_METHODS = METHODS

MODEL_SPECS = {
    "llava-1.5-7b-hf": {
        "kind": "llava",
        "path": f"{DATA_ROOT}/models/llava-1.5-7b-hf",
    },
    "llava-1.5-13b-hf": {
        "kind": "llava",
        "path": str(ROOT / "model" / "llava-1.5-13b-hf"),
    },
    "llava-v1.6-mistral-7b-hf": {
        "kind": "llava",
        "path": str(ROOT / "model" / "llava-v1.6-mistral-7b-hf"),
    },
    "qwen3-vl-8b": {
        "kind": "qwen",
        "path": str(ROOT / "model" / "Qwen3-VL-8B-Instruct"),
    },
    "qwen3-vl-30b-a3b": {
        "kind": "qwen",
        "path": str(ROOT / "model" / "Qwen3-VL-30B-A3B-Instruct"),
    },
}


def parse_model_labels(value: str | None) -> list[str | None]:
    if value is None or not value.strip():
        return [None]
    raw = [item for item in value.replace(",", " ").split() if item]
    if not raw or any(item.lower() == "all" for item in raw):
        return list(MODEL_SPECS)
    unknown = [item for item in raw if item not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unsupported model-label={unknown}; choose from {sorted(MODEL_SPECS)} or all")
    return raw


def resolve_model(label: str | None, model_path: str) -> tuple[str, str, str | None]:
    if label is None:
        return "qwen", model_path, None
    spec = MODEL_SPECS[label]
    return str(spec["kind"]), str(spec["path"]), label


def resolve_data_path(data_path: str, data_root: str) -> str:
    path = Path(data_path)
    if path.is_absolute():
        return str(path)
    repo_path = ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return str(Path(data_root) / path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _qwen_from_pretrained_kwargs(dtype, device, attn_implementation: str | None) -> dict:
    kwargs = {"torch_dtype": dtype, "device_map": device}
    if attn_implementation and str(attn_implementation).strip() and str(attn_implementation).strip() != "auto":
        kwargs["attn_implementation"] = str(attn_implementation).strip()
    return kwargs


def _assert_requested_attn_implementation(model, requested: str | None) -> None:
    if not requested or str(requested).strip() == "auto":
        return
    expected = str(requested).strip()
    actual = getattr(model.model.language_model.config, "_attn_implementation", None)
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
    allowed = {expected}
    if expected in FLASH_ATTN_KERNEL_FALLBACK:
        allowed.add(FLASH_ATTN_KERNEL_FALLBACK[expected])
    visual_actual = getattr(model.model.visual.config, "_attn_implementation", None)
    if visual_actual not in allowed:
        raise RuntimeError(f"requested attn_implementation={expected}, but Qwen vision uses {visual_actual}")
    if actual not in allowed:
        raise RuntimeError(f"requested attn_implementation={expected}, but loaded Qwen text config uses {actual}")


def load_baseline_model(method: str, model_path: str, dtype, device, retention: float, attn_implementation: str | None = None):
    """Load model with baseline-specific modifications."""
    from src.qwen_deepstack import disable_qwen_deepstack

    config = AutoConfig.from_pretrained(model_path)
    load_kwargs = _qwen_from_pretrained_kwargs(dtype, device, attn_implementation)
    if method == BASE_METHOD:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        model_cls = Qwen3VLMoeForConditionalGeneration if getattr(config, "model_type", None) == "qwen3_vl_moe" else Qwen3VLForConditionalGeneration
        model = model_cls.from_pretrained(model_path, **load_kwargs)
        _assert_requested_attn_implementation(model, attn_implementation)
        processor = AutoProcessor.from_pretrained(model_path)
        return disable_qwen_deepstack(model), processor

    if method in {DYVTE_METHOD, REDUNDANCYLENS_METHOD}:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        model_cls = Qwen3VLMoeForConditionalGeneration if getattr(config, "model_type", None) == "qwen3_vl_moe" else Qwen3VLForConditionalGeneration
        model = model_cls.from_pretrained(model_path, **load_kwargs)
        _assert_requested_attn_implementation(model, attn_implementation)
        processor = AutoProcessor.from_pretrained(model_path)
        patch_qwen3_vl_baseline_forward(model)
        return disable_qwen_deepstack(model), processor

    if getattr(config, "model_type", None) == "qwen3_vl_moe":
        if method not in {"dart", "divprune"}:
            raise ValueError(f"Qwen3-VL-MoE baseline is implemented for dart/divprune, got method={method!r}")
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        _assert_requested_attn_implementation(model, attn_implementation)
        processor = AutoProcessor.from_pretrained(model_path)
        patch_qwen3_vl_moe_pruning_forward(model)
        return disable_qwen_deepstack(model), processor

    mod_path = ROOT / "baselines" / method / "qwen3_vl"
    sys.path.insert(0, str(mod_path))

    mod_name = f"modeling_qwen3_vl_{method}"
    mod = __import__(mod_name)
    ModelClass = mod.Qwen3VLForConditionalGeneration

    model = ModelClass.from_pretrained(model_path, **load_kwargs)
    _assert_requested_attn_implementation(model, attn_implementation)
    processor = AutoProcessor.from_pretrained(model_path)

    return disable_qwen_deepstack(model), processor


def patch_qwen3_vl_moe_pruning_forward(model):
    language_model = model.model.language_model
    if getattr(language_model, "_vision_kv_inject_pruning_forward", False):
        return
    language_model.forward = types.MethodType(qwen3_vl_moe_pruning_forward, language_model)
    language_model._vision_kv_inject_pruning_forward = True


def patch_qwen3_vl_baseline_forward(model):
    language_model = model.model.language_model
    if getattr(language_model, "_vision_kv_inject_baseline_forward", False):
        return
    language_model._vision_kv_inject_original_forward = language_model.forward
    model_type = getattr(model.config, "model_type", "") or getattr(language_model.config, "model_type", "")
    if "moe" in str(model_type):
        language_model.forward = types.MethodType(qwen3_vl_moe_pruning_forward, language_model)
    else:
        language_model.forward = types.MethodType(qwen3_vl_pruning_forward, language_model)
    language_model._vision_kv_inject_baseline_forward = True


def _redundancylens_config_has_active_layers(config: dict | None) -> bool:
    if not config:
        return False
    return bool(config.get("attn_reduction_layers") or config.get("ffn_reduction_layers"))


def _qwen_baseline_passthrough(config, hidden_configs: tuple[dict | None, dict | None, dict | None, dict | None]) -> bool:
    dart_config, divprune_config, dyvte_config, redundancylens_config = hidden_configs
    return (
        not dart_config
        and not divprune_config
        and not dyvte_config
        and not _redundancylens_config_has_active_layers(redundancylens_config)
        and getattr(config, "_vision_kv_inject_original_forward", None) is not None
    )


def qwen3_vl_pruning_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs,
):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    return _qwen3_vl_text_forward_with_baselines(
        self,
        qwen3_vl,
        qwen3_vl.BaseModelOutputWithPast,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )


def qwen3_vl_moe_pruning_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs,
):
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as qwen3_vl_moe

    return _qwen3_vl_text_forward_with_baselines(
        self,
        qwen3_vl_moe,
        qwen3_vl_moe.MoeModelOutputWithPast,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )


def _qwen3_vl_text_forward_with_baselines(
    self,
    module_ops,
    output_cls,
    *,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    dart_config = getattr(self.config, "DART_config", None)
    divprune_config = getattr(self.config, "divprune_config", None)
    dyvte_config = getattr(self.config, "dyvte_config", None)
    redundancylens_config = getattr(self.config, "redundancylens_config", None)
    if _qwen_baseline_passthrough(self, (dart_config, divprune_config, dyvte_config, redundancylens_config)):
        return self._vision_kv_inject_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

    if use_cache is None:
        use_cache = getattr(self.config, "use_cache", True)
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = module_ops.DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    input_attention_mask = attention_mask
    self.dyvte_last_exit_layer = None

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    attention_mask = module_ops.create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    if dart_config or divprune_config:
        deepstack_visual_embeds = None
    divprune_pruned = False
    dyvte_exited = False
    dart_source_hidden = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if dart_config and layer_idx == dart_config["K"] - 1:
            dart_source_hidden = hidden_states
        if dart_config and layer_idx == dart_config["K"] and hidden_states.shape[1] > 1:
            img_start, img_len = _qwen_image_span(visual_pos_masks, dart_config, hidden_states)
            if img_len > 0:
                # DART uses the PREVIOUS layer's post-RoPE keys, just as the
                # dense Qwen port does. Preserve the token/head axes.
                assert dart_source_hidden is not None, "DART requires K >= 1"
                source_layer = self.layers[layer_idx - 1]
                normed = source_layer.input_layernorm(dart_source_hidden)
                attn_mod = source_layer.self_attn
                hidden_shape = (*normed.shape[:-1], -1, attn_mod.head_dim)
                k_states = attn_mod.k_norm(attn_mod.k_proj(normed).view(hidden_shape)).transpose(1, 2)
                _, k_states = module_ops.apply_rotary_pos_emb(k_states, k_states, *position_embeddings)
                last_layer_state = self.norm(hidden_states)
                from baselines.multimodal_pruning_utils import visual_indices, audit_prune
                visual = visual_indices(visual_pos_masks, hidden_states, img_start, img_len)
                prefix = torch.arange(img_start, device=hidden_states.device)
                other = torch.arange(img_start, hidden_states.shape[1], device=hidden_states.device)
                other = other[~torch.isin(other, visual)]
                selector_order = torch.cat((prefix, visual, other))
                local = _qwen_dart_retained_image_token(dart_config,
                    last_layer_state[:, selector_order], k_states[:, :, selector_order], img_start, img_len)
                retained_idx = selector_order[local]
                audit_prune(self, hidden_states.shape[1], visual, retained_idx, layer_idx)
                hidden_states, position_ids, text_position_ids, position_embeddings, visual_pos_masks = (
                    _prune_qwen_image_tokens(
                        hidden_states,
                        position_ids,
                        text_position_ids,
                        visual_pos_masks,
                        self.rotary_emb,
                        img_start,
                        img_len,
                        retained_idx - img_start,
                    )
                )
                attention_mask = None

        if (
            divprune_config
            and not divprune_pruned
            and layer_idx == divprune_config["layer"]
            and hidden_states.shape[1] > 1
        ):
            img_start, img_len = _qwen_image_span(visual_pos_masks, divprune_config, hidden_states)
            if img_len > 0:
                from baselines.multimodal_pruning_utils import visual_budget, visual_indices, audit_prune
                visual = visual_indices(visual_pos_masks, hidden_states, img_start, img_len)
                keep_count = visual_budget(img_len, divprune_config["target_retention"])
                selected = visual[_divprune_select_tokens(hidden_states[0, visual], keep_count)] - img_start
                audit_prune(self, hidden_states.shape[1], visual, selected + img_start, layer_idx)
                hidden_states, position_ids, text_position_ids, position_embeddings, visual_pos_masks = (
                    _prune_qwen_image_tokens(
                        hidden_states,
                        position_ids,
                        text_position_ids,
                        visual_pos_masks,
                        self.rotary_emb,
                        img_start,
                        img_len,
                        selected,
                    )
                )
                attention_mask = None
                divprune_pruned = True

        if _redundancylens_layer_active(redundancylens_config, layer_idx, hidden_states):
            hidden_states = _qwen_redundancylens_decoder_layer(
                decoder_layer,
                module_ops,
                layer_idx,
                hidden_states,
                attention_mask,
                text_position_ids,
                past_key_values,
                position_embeddings,
                visual_pos_masks,
                redundancylens_config,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

        dyvte_dynamic_exit = False
        dyvte_gate = getattr(self, "dyvte_gate", None)
        if (
            dyvte_config
            and dyvte_config.get("enabled", True)
            and dyvte_gate is not None
            and not dyvte_exited
            and hidden_states.shape[1] > 1
        ):
            min_exit_layer = int(dyvte_config.get("min_exit_layer", 0))
            max_exit_layer = int(dyvte_config.get("max_exit_layer", len(self.layers) - 1))
            if min_exit_layer <= layer_idx <= max_exit_layer:
                gate_device = next(dyvte_gate.parameters()).device
                gate_dtype = next(dyvte_gate.parameters()).dtype
                from src.dyvte_qwen import qwen_dyvte_features_from_hidden
                features = qwen_dyvte_features_from_hidden(
                    hidden_states,
                    visual_pos_masks=visual_pos_masks,
                    attention_mask=input_attention_mask,
                ).to(device=gate_device, dtype=gate_dtype)
                gate_logits = dyvte_gate(layer_idx, features)
                exit_mask = gate_logits[..., 1] > gate_logits[..., 0]
                dyvte_dynamic_exit = bool(exit_mask.all().item())

        dyvte_static_exit = (
            dyvte_config
            and dyvte_config.get("enabled", True)
            and dyvte_gate is None
            and not dyvte_exited
            and layer_idx == dyvte_config.get("exit_layer")
            and hidden_states.shape[1] > 1
        )
        if dyvte_dynamic_exit or dyvte_static_exit:
            hidden_states, position_ids, text_position_ids, position_embeddings, visual_pos_masks = (
                _drop_qwen_visual_tokens(
                    hidden_states,
                    position_ids,
                    text_position_ids,
                    visual_pos_masks,
                    self.rotary_emb,
                    dyvte_config,
                )
            )
            attention_mask = None
            input_attention_mask = None
            deepstack_visual_embeds = None
            dyvte_exited = True
            self.dyvte_last_exit_layer = int(layer_idx)

    hidden_states = self.norm(hidden_states)
    return output_cls(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _qwen_image_span(visual_pos_masks, config: dict, hidden_states: torch.Tensor) -> tuple[int, int]:
    if visual_pos_masks is not None:
        img_idx = torch.nonzero(visual_pos_masks[0], as_tuple=True)[0]
        if img_idx.numel() == 0:
            return 0, 0
        img_start = int(img_idx[0].item())
        img_len = int(img_idx.numel())
    else:
        img_start = int(config.get("image_token_start_index", 0))
        img_len = int(config.get("image_token_length", 0))
    if img_start < 0 or img_start >= hidden_states.shape[1]:
        return 0, 0
    img_len = max(0, min(img_len, hidden_states.shape[1] - img_start))
    return img_start, img_len


def _prune_qwen_image_tokens(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
    visual_pos_masks: torch.Tensor | None,
    rotary_emb,
    img_start: int,
    img_len: int,
    selected_relative: torch.Tensor,
):
    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    from baselines.multimodal_pruning_utils import visual_indices, keep_visual_subset
    visual = visual_indices(visual_pos_masks, hidden_states, img_start, img_len)
    keep_indices = keep_visual_subset(seq_len, visual, selected_relative.to(device=device) + img_start)

    hidden_states = hidden_states[:, keep_indices, :]
    position_ids = position_ids[:, :, keep_indices] if position_ids is not None else None
    if text_position_ids is not None:
        text_position_ids = text_position_ids[:, keep_indices]
    position_embeddings = rotary_emb(hidden_states, position_ids)
    if visual_pos_masks is not None:
        visual_pos_masks = visual_pos_masks[:, keep_indices]
    return hidden_states, position_ids, text_position_ids, position_embeddings, visual_pos_masks


def _drop_qwen_visual_tokens(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
    visual_pos_masks: torch.Tensor | None,
    rotary_emb,
    config: dict,
):
    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    if visual_pos_masks is not None and visual_pos_masks.shape[-1] == seq_len:
        keep_indices = torch.nonzero(~visual_pos_masks[0].to(device=device).bool(), as_tuple=True)[0]
    else:
        img_start, img_len = _qwen_image_span(visual_pos_masks, config, hidden_states)
        if img_len <= 0:
            return hidden_states, position_ids, text_position_ids, rotary_emb(hidden_states, position_ids), visual_pos_masks
        keep_indices = torch.cat(
            [
                torch.arange(img_start, device=device),
                torch.arange(img_start + img_len, seq_len, device=device),
            ]
        )

    if keep_indices.numel() == seq_len:
        return hidden_states, position_ids, text_position_ids, rotary_emb(hidden_states, position_ids), visual_pos_masks

    hidden_states = hidden_states[:, keep_indices, :]
    position_ids = position_ids[:, :, keep_indices] if position_ids is not None else None
    if text_position_ids is not None:
        text_position_ids = text_position_ids[:, keep_indices]
    position_embeddings = rotary_emb(hidden_states, position_ids)
    return hidden_states, position_ids, text_position_ids, position_embeddings, None


def _qwen_visual_mask_like(
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    batch, seq_len = hidden_states.shape[:2]
    device = hidden_states.device
    if visual_pos_masks is not None and visual_pos_masks.shape[-1] == seq_len:
        return visual_pos_masks.to(device=device).bool()

    img_start, img_len = _qwen_image_span(visual_pos_masks, config, hidden_states)
    mask = torch.zeros((batch, seq_len), device=device, dtype=torch.bool)
    if img_len > 0:
        mask[:, img_start : img_start + img_len] = True
    return mask


def _redundancylens_layer_active(config: dict | None, layer_idx: int, hidden_states: torch.Tensor) -> bool:
    if not config or hidden_states.shape[1] <= 1:
        return False
    attn_reduction_layers = set(int(layer) for layer in config.get("attn_reduction_layers", []))
    ffn_reduction_layers = set(int(layer) for layer in config.get("ffn_reduction_layers", []))
    hollow_active = bool(config.get("hollow_attention", True)) and int(layer_idx) in attn_reduction_layers
    ffn_filter_active = int(layer_idx) in ffn_reduction_layers
    return hollow_active or ffn_filter_active


def _qwen_hollow_attention_mask(
    attention_mask: torch.Tensor | None,
    visual_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    window: int,
) -> torch.Tensor | None:
    if window <= 0 or not bool(visual_mask.any()):
        return attention_mask

    batch, seq_len = visual_mask.shape
    dtype = hidden_states.dtype if hidden_states.dtype.is_floating_point else torch.float32
    device = hidden_states.device
    min_dtype = torch.finfo(dtype).min

    if attention_mask is None or attention_mask.ndim != 4:
        mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask[None, None, :, :].expand(batch, 1, -1, -1).clone()
    else:
        mask = attention_mask.to(device=device).clone()
        if not mask.dtype.is_floating_point:
            mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask.bool(), min_dtype)

    for batch_idx in range(batch):
        visual_idx = torch.nonzero(visual_mask[batch_idx], as_tuple=True)[0]
        if visual_idx.numel() == 0:
            continue
        for query_pos in visual_idx:
            mask[batch_idx, :, query_pos, visual_idx] = min_dtype
            allowed = visual_idx[(visual_idx <= query_pos) & ((query_pos - visual_idx) < int(window))]
            mask[batch_idx, :, query_pos, allowed] = 0
    return mask


def _qwen_attention_forward(
    attn_mod,
    module_ops,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values,
    position_embeddings,
    **kwargs,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

    query_states = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = module_ops.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, attn_mod.layer_idx)

    attention_interface = module_ops.ALL_ATTENTION_FUNCTIONS.get_interface(
        attn_mod.config._attn_implementation,
        module_ops.eager_attention_forward,
    )
    attn_output, _ = attention_interface(
        attn_mod,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn_mod.training else attn_mod.attention_dropout,
        scaling=attn_mod.scaling,
        **kwargs,
    )
    return attn_mod.o_proj(attn_output.reshape(*input_shape, -1).contiguous())


def _qwen_hollow_flash_attention_forward(
    attn_mod,
    module_ops,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values,
    position_embeddings,
    visual_mask: torch.Tensor,
    window: int,
    **kwargs,
) -> torch.Tensor:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

    query_states = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = module_ops.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, attn_mod.layer_idx)

    base_attention_mask = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
    attention_interface = module_ops.ALL_ATTENTION_FUNCTIONS.get_interface(
        attn_mod.config._attn_implementation,
        module_ops.eager_attention_forward,
    )
    attn_output, _ = attention_interface(
        attn_mod,
        query_states,
        key_states,
        value_states,
        base_attention_mask,
        dropout=0.0 if not attn_mod.training else attn_mod.attention_dropout,
        scaling=attn_mod.scaling,
        **kwargs,
    )

    batch, _, key_len, _ = key_states.shape
    if key_len != visual_mask.shape[-1]:
        return attn_mod.o_proj(attn_output.reshape(*input_shape, -1).contiguous())

    key_positions = torch.arange(key_len, device=hidden_states.device)
    query_segments = []
    key_segments = []
    value_segments = []
    key_lengths = []
    scatter_batch = []
    scatter_pos = []

    for batch_idx in range(batch):
        visual_idx = torch.nonzero(visual_mask[batch_idx], as_tuple=True)[0]
        if visual_idx.numel() == 0:
            continue
        valid_keys = None
        if attention_mask is not None and attention_mask.ndim == 2:
            valid_keys = attention_mask[batch_idx, :key_len].to(device=hidden_states.device).bool()
        for query_pos in visual_idx.tolist():
            if attention_mask is not None and attention_mask.ndim == 4:
                mask_row = attention_mask[batch_idx, 0, query_pos, :key_len].to(device=hidden_states.device)
                if mask_row.dtype.is_floating_point:
                    allowed = mask_row > (torch.finfo(mask_row.dtype).min / 2)
                else:
                    allowed = mask_row.bool()
            else:
                allowed = key_positions <= int(query_pos)
                if valid_keys is not None:
                    allowed = allowed & valid_keys

            local_visual = (
                visual_mask[batch_idx]
                & (key_positions <= int(query_pos))
                & ((int(query_pos) - key_positions) < int(window))
            )
            allowed = (allowed & ~visual_mask[batch_idx]) | local_visual
            allowed_idx = torch.nonzero(allowed, as_tuple=True)[0]
            if allowed_idx.numel() == 0:
                continue

            query_segments.append(query_states[batch_idx, :, query_pos, :].unsqueeze(0))
            key_segments.append(key_states[batch_idx, :, allowed_idx, :].transpose(0, 1).contiguous())
            value_segments.append(value_states[batch_idx, :, allowed_idx, :].transpose(0, 1).contiguous())
            key_lengths.append(int(allowed_idx.numel()))
            scatter_batch.append(batch_idx)
            scatter_pos.append(int(query_pos))

    if not query_segments:
        return attn_mod.o_proj(attn_output.reshape(*input_shape, -1).contiguous())

    query_cat = torch.cat(query_segments, dim=0).contiguous()
    key_cat = torch.cat(key_segments, dim=0).contiguous()
    value_cat = torch.cat(value_segments, dim=0).contiguous()
    cu_seqlens_q = torch.arange(
        0,
        len(query_segments) + 1,
        device=hidden_states.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.empty(len(key_lengths) + 1, device=hidden_states.device, dtype=torch.int32)
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = torch.tensor(key_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0)

    visual_out = flash_attn_varlen_func(
        query_cat,
        key_cat,
        value_cat,
        cu_seqlens_q,
        cu_seqlens_k,
        1,
        max(key_lengths),
        dropout_p=0.0 if not attn_mod.training else attn_mod.attention_dropout,
        softmax_scale=attn_mod.scaling,
        causal=False,
    )
    if isinstance(visual_out, tuple):
        visual_out = visual_out[0]

    attn_output = attn_output.clone()
    attn_output[
        torch.tensor(scatter_batch, device=hidden_states.device, dtype=torch.long),
        torch.tensor(scatter_pos, device=hidden_states.device, dtype=torch.long),
    ] = visual_out
    return attn_mod.o_proj(attn_output.reshape(*input_shape, -1).contiguous())


def _qwen_redundancylens_mlp(
    mlp,
    hidden_states: torch.Tensor,
    visual_mask: torch.Tensor,
    ratio: float,
    sample_divisor: int,
) -> torch.Tensor:
    if ratio >= 1.0 or not bool(visual_mask.any()):
        return mlp(hidden_states)
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        return mlp(hidden_states)

    ratio = max(0.0, min(1.0, float(ratio)))
    out = torch.empty_like(hidden_states)
    text_mask = ~visual_mask
    if bool(text_mask.any()):
        out[text_mask] = mlp(hidden_states[text_mask])

    visual_states = hidden_states[visual_mask]
    gate = mlp.act_fn(mlp.gate_proj(visual_states))
    middle = gate * mlp.up_proj(visual_states)
    keep_dim = max(1, min(middle.shape[-1], round(middle.shape[-1] * ratio)))
    if keep_dim >= middle.shape[-1]:
        out[visual_mask] = mlp.down_proj(middle)
        return out

    sample_count = max(1, int(visual_states.shape[0]) // max(1, int(sample_divisor)))
    sample_count = min(sample_count, int(visual_states.shape[0]))
    if sample_count == int(visual_states.shape[0]):
        probe = middle
    else:
        probe_idx = torch.linspace(
            0,
            int(visual_states.shape[0]) - 1,
            sample_count,
            device=hidden_states.device,
            dtype=torch.long,
        )
        probe = middle.index_select(0, probe_idx)
    channel_scores = probe.detach().float().abs().mean(dim=0)
    keep_idx = torch.topk(channel_scores, k=keep_dim, largest=True).indices.sort().values
    reduced_middle = middle.index_select(-1, keep_idx)
    reduced_weight = mlp.down_proj.weight.index_select(1, keep_idx)
    out[visual_mask] = torch.nn.functional.linear(reduced_middle, reduced_weight, mlp.down_proj.bias)
    return out


def _qwen_redundancylens_decoder_layer(
    decoder_layer,
    module_ops,
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
    past_key_values,
    position_embeddings,
    visual_pos_masks: torch.Tensor | None,
    config: dict,
    **kwargs,
) -> torch.Tensor:
    visual_mask = _qwen_visual_mask_like(hidden_states, visual_pos_masks, config)
    residual = hidden_states
    normed = decoder_layer.input_layernorm(hidden_states)

    layer_attention_mask = attention_mask
    attn_reduction_layers = set(int(layer) for layer in config.get("attn_reduction_layers", []))
    use_hollow_attention = config.get("hollow_attention", True) and int(layer_idx) in attn_reduction_layers
    if use_hollow_attention and decoder_layer.self_attn.config._attn_implementation == "flash_attention_2":
        attn_out = _qwen_hollow_flash_attention_forward(
            decoder_layer.self_attn,
            module_ops,
            normed,
            attention_mask,
            past_key_values,
            position_embeddings,
            visual_mask,
            int(config.get("attention_range", 256)),
            **kwargs,
        )
    else:
        if use_hollow_attention:
            layer_attention_mask = _qwen_hollow_attention_mask(
                attention_mask,
                visual_mask,
                normed,
                int(config.get("attention_range", 256)),
            )

        attn_out = _qwen_attention_forward(
            decoder_layer.self_attn,
            module_ops,
            normed,
            layer_attention_mask,
            past_key_values,
            position_embeddings,
            **kwargs,
        )
    hidden_states = residual + attn_out

    residual = hidden_states
    normed = decoder_layer.post_attention_layernorm(hidden_states)
    ffn_reduction_layers = set(int(layer) for layer in config.get("ffn_reduction_layers", []))
    if int(layer_idx) in ffn_reduction_layers:
        mlp_out = _qwen_redundancylens_mlp(
            decoder_layer.mlp,
            normed,
            visual_mask,
            float(config.get("ffn_channel_ratio", 0.2)),
            int(config.get("ffn_sample_divisor", 10)),
        )
    else:
        mlp_out = decoder_layer.mlp(normed)
    return residual + mlp_out


def _qwen_dart_retained_image_token(config_dart, last_layer_state, k_states, img_start: int, img_len: int):
    # One selector for dense and MoE; do not reintroduce direct B,H,S,D reshape.
    from baselines.dart.qwen3_vl.modeling_qwen3_vl_dart import dart_get_retained_image_token
    return dart_get_retained_image_token(config_dart, last_layer_state, k_states, img_start, img_len)


def _divprune_select_tokens(visual_feature_vectors: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep_count = min(max(int(keep_count), 1), int(visual_feature_vectors.shape[0]))
    features = torch.nn.functional.normalize(visual_feature_vectors.float(), dim=-1)
    dist_matrix = 1.0 - torch.mm(features, features.t())

    selected = torch.empty(keep_count, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(keep_count):
        if i == 0:
            if dist_matrix.shape[0] == 1:
                scores = dist_matrix[0]
            else:
                scores = torch.topk(dist_matrix, 2, dim=0, largest=False).values[1, :]
        else:
            selected_dists = torch.index_select(dist_matrix, 0, selected[:i])
            scores = torch.min(selected_dists, dim=0).values
            scores.index_fill_(0, selected[:i], -float("inf"))
        selected[i] = torch.argmax(scores)
    return selected


def _parse_layer_list(value: str | None, *, num_layers: int, name: str) -> list[int] | None:
    layers = _parse_layer_sequence(value, num_layers=num_layers, name=name)
    return None if layers is None else sorted(set(layers))


def _parse_layer_sequence(value: str | None, *, num_layers: int, name: str) -> list[int] | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    layers = []
    seen = set()
    for raw in value.replace(",", " ").split():
        layer = int(raw)
        if layer < 0 or layer >= int(num_layers):
            raise ValueError(f"{name} contains layer {layer}, but valid range is [0, {int(num_layers) - 1}]")
        if layer not in seen:
            layers.append(layer)
            seen.add(layer)
    return layers


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _arg_or_env(args, attr: str, env_name: str, default=None):
    value = getattr(args, attr, None) if args is not None else None
    if value is not None:
        return value
    return os.environ.get(env_name, default)


def _resolve_redundancylens_reduction_layers(
    args,
    *,
    component: str,
    num_layers: int,
) -> list[int]:
    if component == "attn":
        direct_attr = "redundancylens_attn_reduction_layers"
        direct_env = "REDUNDANCYLENS_ATTN_REDUCTION_LAYERS"
        priority_attr = "redundancylens_attn_priority_layers"
        priority_env = "REDUNDANCYLENS_ATTN_PRIORITY_LAYERS"
        count_attr = "redundancylens_l_ra"
        count_env = "REDUNDANCYLENS_L_RA"
        count_name = "L_RA"
        reduction_name = "attention"
    elif component == "ffn":
        direct_attr = "redundancylens_ffn_reduction_layers"
        direct_env = "REDUNDANCYLENS_FFN_REDUCTION_LAYERS"
        priority_attr = "redundancylens_ffn_priority_layers"
        priority_env = "REDUNDANCYLENS_FFN_PRIORITY_LAYERS"
        count_attr = "redundancylens_l_rf"
        count_env = "REDUNDANCYLENS_L_RF"
        count_name = "L_RF"
        reduction_name = "FFN"
    else:
        raise ValueError(f"unknown RedundancyLens component={component!r}")

    direct_raw = _arg_or_env(args, direct_attr, direct_env)
    if direct_raw is not None and not str(direct_raw).strip():
        return []
    direct_layers = _parse_layer_list(
        direct_raw,
        num_layers=num_layers,
        name=f"RedundancyLens {reduction_name} reduction layers",
    )
    if direct_layers is not None:
        return direct_layers

    priority_raw = _arg_or_env(args, priority_attr, priority_env)
    count_raw = _arg_or_env(args, count_attr, count_env)
    if count_raw is not None and int(count_raw) == 0:
        return []
    if priority_raw is None or count_raw is None:
        raise ValueError(
            f"RedundancyLens is LR/L_RA/L_RF based, not retention based. Set "
            f"--redundancylens-{component}-reduction-layers directly, or set "
            f"--redundancylens-{component}-priority-layers with --redundancylens-{count_name.lower().replace('_', '-')}. "
            f"Env alternatives: {direct_env}, or {priority_env}+{count_env}."
        )

    priority_layers = _parse_layer_sequence(
        priority_raw,
        num_layers=num_layers,
        name=f"RedundancyLens {reduction_name} LR priority layers",
    )
    if priority_layers is None:
        raise ValueError(f"RedundancyLens {reduction_name} LR priority list is empty")
    count = int(count_raw)
    if count < 0 or count > len(priority_layers):
        raise ValueError(
            f"RedundancyLens {count_name}={count} must be in [0, {len(priority_layers)}] "
            f"for the provided LR priority list"
        )
    return sorted(priority_layers[:count])


def configure_baseline(model, method: str, retention: float, img_start: int, img_len: int, args=None):
    """Set baseline-specific config."""
    if method == BASE_METHOD:
        return
    text_config = model.model.language_model.config

    if method == "fastv":
        text_config.fastv_config = {
            "fastv_k": 2,
            "fastv_r": 1.0 - retention,
            "target_retention": retention,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
        }
    elif method == "dart":
        text_config.DART_config = {
            "K": 2,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "reduction_ratio": 1.0 - retention,
            "target_retention": retention,
            "pivot_image_token": 5,
            "pivot_text_token": 3,
        }
    elif method == "visionzip":
        model.config.visionzip_config = {
            "dominant_ratio": retention * 0.8,
            "target_retention": retention,
            "contextual_ratio": retention * 0.2,
        }
    elif method == "sparsevlm":
        text_config.sparse_config = {
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "pruning_loc": [2, 6, 15],
            "target_retention": retention,
        }
    elif method == "h2o":
        # H2O: budget computed dynamically in forward from visual_pos_masks
        text_config.h2o_config = {
            "retention": retention,
        }
    elif method == "prefixkv":
        # PrefixKV: budget computed dynamically in forward from visual_pos_masks
        text_config.prefixkv_config = {
            "retention": retention,
        }
    elif method == "divprune":
        text_config.divprune_config = {
            "layer": 0,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "target_retention": retention,
        }
    elif method == "zoo":
        text_config.zoo_config = {
            "layer": 0,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "target_retention": retention,
            "num_refine": 64,
            "noise_scale": 0.01,
        }
    elif method == DYVTE_METHOD:
        num_layers = int(getattr(text_config, "num_hidden_layers", len(model.model.language_model.layers)))
        gate_checkpoint = _arg_or_env(args, "dyvte_gate_checkpoint", "DYVTE_GATE_CHECKPOINT")
        if gate_checkpoint:
            language_model = model.model.language_model
            gate_device = next(language_model.parameters()).device
            gate_dtype = next(language_model.parameters()).dtype
            loaded_path = getattr(language_model, "dyvte_gate_checkpoint", None)
            if str(loaded_path) != str(gate_checkpoint):
                from src.dyvte_qwen import load_qwen_dyvte_gate_checkpoint
                gate, gate_meta = load_qwen_dyvte_gate_checkpoint(
                    gate_checkpoint,
                    language_model,
                    device=gate_device,
                    dtype=gate_dtype,
                )
                language_model.dyvte_gate = gate
                language_model.dyvte_gate_checkpoint = str(gate_checkpoint)
                language_model.dyvte_gate_meta = gate_meta
            min_exit_layer = int(_arg_or_env(args, "dyvte_min_exit_layer", "DYVTE_MIN_EXIT_LAYER", 0))
            max_exit_layer = int(_arg_or_env(args, "dyvte_max_exit_layer", "DYVTE_MAX_EXIT_LAYER", num_layers - 1))
            if min_exit_layer < 0 or max_exit_layer >= num_layers or min_exit_layer > max_exit_layer:
                raise ValueError(f"invalid DyVTE gate exit range [{min_exit_layer}, {max_exit_layer}] for {num_layers} layers")
            text_config.dyvte_config = {
                "enabled": True,
                "gate_checkpoint": str(gate_checkpoint),
                "min_exit_layer": min_exit_layer,
                "max_exit_layer": max_exit_layer,
                "image_token_start_index": img_start,
                "image_token_length": img_len,
            }
        elif (cli_exit_layer := getattr(args, "dyvte_exit_layer", None) if args is not None else None) is not None:
            exit_layer = int(cli_exit_layer)
            if exit_layer < 0 or exit_layer >= num_layers:
                raise ValueError(f"dyvte exit_layer={exit_layer} is outside [0, {num_layers - 1}]")
            text_config.dyvte_config = {
                "enabled": True,
                "exit_layer": exit_layer,
                "image_token_start_index": img_start,
                "image_token_length": img_len,
            }
        elif "DYVTE_EXIT_LAYER" in os.environ:
            exit_layer = int(os.environ["DYVTE_EXIT_LAYER"])
            if exit_layer < 0 or exit_layer >= num_layers:
                raise ValueError(f"dyvte exit_layer={exit_layer} is outside [0, {num_layers - 1}]")
            text_config.dyvte_config = {
                "enabled": True,
                "exit_layer": exit_layer,
                "image_token_start_index": img_start,
                "image_token_length": img_len,
            }
        else:
            raise ValueError(
                "DyVTE does not have token retention. Set --dyvte-gate-checkpoint for the trained PixMo gate, "
                "or --dyvte-exit-layer/DYVTE_EXIT_LAYER for a static-exit ablation."
            )
    elif method == REDUNDANCYLENS_METHOD:
        num_layers = int(getattr(text_config, "num_hidden_layers", len(model.model.language_model.layers)))
        attn_reduction_layers = _resolve_redundancylens_reduction_layers(
            args,
            component="attn",
            num_layers=num_layers,
        )
        ffn_reduction_layers = _resolve_redundancylens_reduction_layers(
            args,
            component="ffn",
            num_layers=num_layers,
        )
        ffn_channel_ratio_value = _arg_or_env(args, "redundancylens_ffn_channel_ratio", "REDUNDANCYLENS_FFN_CHANNEL_RATIO", 0.2)
        hollow_attention_value = _arg_or_env(args, "redundancylens_hollow_attention", "REDUNDANCYLENS_HOLLOW_ATTENTION", "1")
        attention_range_value = _arg_or_env(args, "redundancylens_attention_range", "REDUNDANCYLENS_ATTENTION_RANGE", 256)
        ffn_sample_divisor_value = _arg_or_env(args, "redundancylens_ffn_sample_divisor", "REDUNDANCYLENS_FFN_SAMPLE_DIVISOR", 10)
        text_config.redundancylens_config = {
            "attn_reduction_layers": attn_reduction_layers,
            "ffn_reduction_layers": ffn_reduction_layers,
            "l_ra": len(attn_reduction_layers),
            "l_rf": len(ffn_reduction_layers),
            "ffn_channel_ratio": max(0.0, min(1.0, float(ffn_channel_ratio_value))),
            "ffn_sample_divisor": int(ffn_sample_divisor_value),
            "hollow_attention": _parse_bool(hollow_attention_value),
            "attention_range": int(attention_range_value),
            "image_token_start_index": img_start,
            "image_token_length": img_len,
        }
    elif method == "epic":
        text_config.DART_config = {
            "K": 2,
            "K2": 20,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "reduction_ratio": 1.0 - retention,
        }


def _layer_count_from_raw(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return 0
    return len(set(value.replace(",", " ").split()))


def _baseline_output_tag(method: str, retention: float, args=None) -> str:
    if method == BASE_METHOD:
        return "base"
    if method == DYVTE_METHOD:
        gate_checkpoint = _arg_or_env(args, "dyvte_gate_checkpoint", "DYVTE_GATE_CHECKPOINT")
        if gate_checkpoint:
            digest = hashlib.sha1(str(gate_checkpoint).encode("utf-8")).hexdigest()[:8]
            return f"gate_{Path(str(gate_checkpoint)).stem}_{digest}"
        exit_layer = _arg_or_env(args, "dyvte_exit_layer", "DYVTE_EXIT_LAYER")
        return f"exit{exit_layer}" if exit_layer is not None else "exit-required"
    if method == REDUNDANCYLENS_METHOD:
        attn_layers_raw = _arg_or_env(args, "redundancylens_attn_reduction_layers", "REDUNDANCYLENS_ATTN_REDUCTION_LAYERS")
        ffn_layers_raw = _arg_or_env(args, "redundancylens_ffn_reduction_layers", "REDUNDANCYLENS_FFN_REDUCTION_LAYERS")
        l_ra = _arg_or_env(args, "redundancylens_l_ra", "REDUNDANCYLENS_L_RA")
        l_rf = _arg_or_env(args, "redundancylens_l_rf", "REDUNDANCYLENS_L_RF")
        if l_ra is None:
            l_ra = _layer_count_from_raw(attn_layers_raw)
        if l_rf is None:
            l_rf = _layer_count_from_raw(ffn_layers_raw)
        raw_parts = [
            str(attn_layers_raw),
            str(ffn_layers_raw),
            str(_arg_or_env(args, "redundancylens_attn_priority_layers", "REDUNDANCYLENS_ATTN_PRIORITY_LAYERS")),
            str(_arg_or_env(args, "redundancylens_ffn_priority_layers", "REDUNDANCYLENS_FFN_PRIORITY_LAYERS")),
            str(l_ra),
            str(l_rf),
            str(_arg_or_env(args, "redundancylens_ffn_channel_ratio", "REDUNDANCYLENS_FFN_CHANNEL_RATIO", 0.2)),
            str(_arg_or_env(args, "redundancylens_attention_range", "REDUNDANCYLENS_ATTENTION_RANGE", 256)),
        ]
        digest = hashlib.sha1("|".join(raw_parts).encode("utf-8")).hexdigest()[:8]
        return f"lra{l_ra}_lrf{l_rf}_{digest}"
    return f"ret{int(round(float(retention) * 100)):02d}"


def _method_config_summary(model, method: str) -> dict:
    if method not in {DYVTE_METHOD, REDUNDANCYLENS_METHOD}:
        return {}
    text_config = model.model.language_model.config
    attr = "dyvte_config" if method == DYVTE_METHOD else "redundancylens_config"
    return dict(getattr(text_config, attr, None) or {})


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return max(1, torch.tensor([], dtype=dtype).element_size())


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _qwen_token_counts(item: dict) -> tuple[int, int]:
    attention_mask = item["attention_mask"].bool()
    mm_ids = item["mm_token_type_ids"]
    image_tokens = int(((mm_ids != 0) & attention_mask).sum().item())
    text_tokens = int(((mm_ids == 0) & attention_mask).sum().item())
    return text_tokens, image_tokens


def _baseline_retained_image_tokens(method: str, retention: float, image_tokens: int) -> int:
    image_tokens = max(0, int(image_tokens))
    if image_tokens == 0:
        return 0
    if method == BASE_METHOD or float(retention) >= 1.0:
        return image_tokens
    if method == DYVTE_METHOD:
        return 0
    if method == REDUNDANCYLENS_METHOD:
        return image_tokens
    return max(1, min(image_tokens, round(image_tokens * float(retention))))


def _baseline_prune_layer(method: str) -> int | None:
    if method == BASE_METHOD:
        return None
    if method in {"visionzip", "divprune", "zoo"}:
        return 0
    if method in {"fastv", "dart", "sparsevlm"}:
        return 2
    if method == REDUNDANCYLENS_METHOD:
        return None
    return 0


def _qwen_layerwise_kv_cache_mb(
    language_config,
    *,
    text_tokens: int,
    image_tokens: int,
    retained_image_tokens: int,
    prune_layer: int | None,
    dtype_bytes: int,
) -> float:
    layers = int(getattr(language_config, "num_hidden_layers", 0))
    kv_heads = int(getattr(language_config, "num_key_value_heads", getattr(language_config, "num_attention_heads", 0)))
    head_dim = int(getattr(language_config, "head_dim", language_config.hidden_size // language_config.num_attention_heads))
    full_seq = int(text_tokens) + int(image_tokens)
    retained_seq = int(text_tokens) + int(retained_image_tokens)
    bytes_per_token_layer = kv_heads * head_dim * 2 * int(dtype_bytes)
    if prune_layer is None or retained_image_tokens >= image_tokens:
        total_bytes = layers * full_seq * bytes_per_token_layer
    else:
        full_layers = max(0, min(layers, int(prune_layer)))
        retained_layers = max(0, layers - full_layers)
        total_bytes = (full_layers * full_seq + retained_layers * retained_seq) * bytes_per_token_layer
    return total_bytes / (1024.0**2)


def _qwen_llm_prefill_flops_per_layers(language_config, seq_len: int, layers: int) -> float:
    layers = max(0, int(layers))
    if layers == 0:
        return 0.0
    hidden = int(getattr(language_config, "hidden_size"))
    heads = int(getattr(language_config, "num_attention_heads"))
    kv_heads = int(getattr(language_config, "num_key_value_heads", heads))
    head_dim = int(getattr(language_config, "head_dim", hidden // heads))
    intermediate = int(getattr(language_config, "intermediate_size", hidden * 4))
    seq_len = int(seq_len)
    qkv_o_macs = hidden * (heads * head_dim + 2 * kv_heads * head_dim + heads * head_dim)
    mlp_macs = 3 * hidden * intermediate
    linear = 2.0 * seq_len * layers * (qkv_o_macs + mlp_macs)
    attention = 4.0 * layers * heads * (seq_len**2) * head_dim
    return linear + attention


def _qwen_baseline_prefill_flops(
    language_config,
    *,
    text_tokens: int,
    image_tokens: int,
    retained_image_tokens: int,
    prune_layer: int | None,
) -> float:
    layers = int(getattr(language_config, "num_hidden_layers", 0))
    full_seq = int(text_tokens) + int(image_tokens)
    retained_seq = int(text_tokens) + int(retained_image_tokens)
    if prune_layer is None or retained_image_tokens >= image_tokens:
        return _qwen_llm_prefill_flops_per_layers(language_config, full_seq, layers)
    full_layers = max(0, min(layers, int(prune_layer)))
    retained_layers = max(0, layers - full_layers)
    return _qwen_llm_prefill_flops_per_layers(
        language_config,
        full_seq,
        full_layers,
    ) + _qwen_llm_prefill_flops_per_layers(
        language_config,
        retained_seq,
        retained_layers,
    )


def _qwen_redundancylens_prefill_flops(
    language_config,
    *,
    text_tokens: int,
    image_tokens: int,
) -> float:
    layers = int(getattr(language_config, "num_hidden_layers", 0))
    full_seq = int(text_tokens) + int(image_tokens)
    config = getattr(language_config, "redundancylens_config", None) or {}
    attn_reduction_layers = set(int(layer) for layer in config.get("attn_reduction_layers", []))
    ffn_reduction_layers = set(int(layer) for layer in config.get("ffn_reduction_layers", []))
    if not attn_reduction_layers and not ffn_reduction_layers:
        return _qwen_llm_prefill_flops_per_layers(language_config, full_seq, layers)

    hidden = int(getattr(language_config, "hidden_size"))
    heads = int(getattr(language_config, "num_attention_heads"))
    kv_heads = int(getattr(language_config, "num_key_value_heads", heads))
    head_dim = int(getattr(language_config, "head_dim", hidden // heads))
    intermediate = int(getattr(language_config, "intermediate_size", hidden * 4))
    ffn_ratio = max(0.0, min(1.0, float(config.get("ffn_channel_ratio", 1.0))))
    attention_range = max(1, int(config.get("attention_range", image_tokens or 1)))
    hollow_attention = bool(config.get("hollow_attention", True))

    qkv_o_macs = hidden * (heads * head_dim + 2 * kv_heads * head_dim + heads * head_dim)
    mlp_macs = 3 * hidden * intermediate
    full_attn_pairs = full_seq**2
    hollow_attn_pairs = max(0, full_attn_pairs - image_tokens**2 + image_tokens * min(image_tokens, attention_range))

    total = 0.0
    for layer_idx in range(layers):
        seq_linear = 2.0 * full_seq * qkv_o_macs
        if layer_idx in ffn_reduction_layers:
            mlp_tokens = int(text_tokens) + int(image_tokens) * ffn_ratio
        else:
            mlp_tokens = full_seq
        if hollow_attention and layer_idx in attn_reduction_layers:
            attention_pairs = hollow_attn_pairs
        else:
            attention_pairs = full_attn_pairs
        total += seq_linear
        total += 2.0 * mlp_tokens * mlp_macs
        total += 4.0 * heads * attention_pairs * head_dim
    return total


def _qwen_speed_resources(model, item: dict, method: str, retention: float, dtype: torch.dtype) -> dict[str, float | int]:
    language_config = model.model.language_model.config
    text_tokens, image_tokens = _qwen_token_counts(item)
    if method == DYVTE_METHOD:
        dyvte_config = getattr(language_config, "dyvte_config", None) or {}
        last_exit_layer = getattr(model.model.language_model, "dyvte_last_exit_layer", None)
        if dyvte_config.get("enabled", True) and last_exit_layer is not None:
            retained_image_tokens = 0
            prune_layer = int(last_exit_layer) + 1
        elif dyvte_config.get("enabled", True) and "exit_layer" in dyvte_config:
            retained_image_tokens = 0
            prune_layer = int(dyvte_config["exit_layer"]) + 1
        else:
            retained_image_tokens = image_tokens
            prune_layer = None
    else:
        retained_image_tokens = _baseline_retained_image_tokens(method, retention, image_tokens)
        prune_layer = _baseline_prune_layer(method)
    if method == REDUNDANCYLENS_METHOD:
        prefill_flops = _qwen_redundancylens_prefill_flops(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
        )
    else:
        prefill_flops = _qwen_baseline_prefill_flops(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            retained_image_tokens=retained_image_tokens,
            prune_layer=prune_layer,
        )
    return {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "retained_image_tokens": retained_image_tokens,
        "kv_cache_mb": _qwen_layerwise_kv_cache_mb(
            language_config,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            retained_image_tokens=retained_image_tokens,
            prune_layer=prune_layer,
            dtype_bytes=_dtype_bytes(dtype),
        ),
        "prefill_flops": prefill_flops,
    }


def _qwen_inputs_from_item(item: dict, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {
        "input_ids": item["input_ids"].unsqueeze(0).to(device),
        "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
        "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0).to(device),
    }
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        value = item.get(key)
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def _warmup_qwen_speed(model, dataset, device: torch.device, *, max_new_tokens: int, rounds: int) -> None:
    rounds = max(0, int(rounds))
    if rounds <= 0 or len(dataset) == 0:
        return
    item = dataset[0]
    inputs = _qwen_inputs_from_item(item, device)
    with torch.inference_mode():
        for _ in range(rounds):
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            _ = model(**inputs, logits_to_keep=1).logits
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    _sync_cuda()


def evaluate_single(
    model,
    processor,
    dataset,
    benchmark: str,
    max_new_tokens: int,
    log_every: int = 50,
    *,
    method: str = BASE_METHOD,
    retention: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    measure_prefill: bool = False,
    speed_warmup: int = 0,
    measure_decode: bool = False,
    native_graphs=None,
    measure_peak_memory: bool = False,
):
    """Run evaluation on dataset, return predictions and summary."""
    spec = get_benchmark_spec(benchmark)
    device = next(model.parameters()).device
    predictions = []
    total_time = 0.0
    prefill_time = 0.0
    kv_cache_values: list[float] = []
    flops_values: list[float] = []
    text_token_values: list[float] = []
    image_token_values: list[float] = []
    retained_image_token_values: list[float] = []
    stage_timer = None
    if measure_decode or measure_peak_memory:
        from src.generation_timing import GenerationStageTimer
        stage_timer = GenerationStageTimer(model, measure_memory=measure_peak_memory)
    if measure_prefill:
        _warmup_qwen_speed(model, dataset, device, max_new_tokens=max_new_tokens, rounds=speed_warmup)

    for idx in range(len(dataset)):
        item = dataset[idx]
        inputs = _qwen_inputs_from_item(item, device)

        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None

        graph_prepare_s = None
        graph_before = None
        if native_graphs is not None:
            # Match the adapter's per-input graph warmup policy. Capture and
            # correctness checks are explicitly outside steady-state timing.
            prepare_start = time.perf_counter()
            with torch.inference_mode():
                # Warmup must not consume ZooPrune's random selector draws or
                # compare different selections. Mirror the timed request pair.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(device)
                python_rng = random.getstate()
                def restore_rng():
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng, device)
                    random.setstate(python_rng)
                native_graphs.enabled = False
                if measure_prefill:
                    _ = model(**inputs, logits_to_keep=1).logits
                    model.model.rope_deltas = None
                reference = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    return_dict_in_generate=True, output_logits=True)
                restore_rng()
                native_graphs.enabled = native_graphs.allow_capture = True
                if measure_prefill:
                    model.model.rope_deltas = None
                    _ = model(**inputs, logits_to_keep=1).logits
                model.model.rope_deltas = None
                candidate = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    return_dict_in_generate=True, output_logits=True)
                native_graphs.allow_capture = False
                if not torch.equal(reference.sequences, candidate.sequences):
                    raise RuntimeError(f"Native CUDA Graph changed tokens at sample {idx}")
                if len(reference.logits) != len(candidate.logits) or any(not torch.equal(a, b) for a,b in zip(reference.logits,candidate.logits)):
                    raise RuntimeError(f"Native CUDA Graph changed logits at sample {idx}")
                if any(not torch.equal(getattr(a, key), getattr(b, key))
                       for a,b in zip(reference.past_key_values.layers,candidate.past_key_values.layers)
                       for key in ("keys", "values")):
                    raise RuntimeError(f"Native CUDA Graph changed KV at sample {idx}")
                del reference, candidate
                restore_rng()
            _sync_cuda()
            graph_prepare_s = time.perf_counter() - prepare_start
            graph_before = native_graphs.stats()
            model.model.rope_deltas = None

        resources = None
        item_prefill_s = None
        if measure_prefill:
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            _sync_cuda()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                _ = model(**inputs, logits_to_keep=1).logits
            _sync_cuda()
            item_prefill_s = time.perf_counter() - prefill_start
            prefill_time += item_prefill_s
            resources = _qwen_speed_resources(model, item, method, retention, dtype)
            kv_cache_values.append(float(resources["kv_cache_mb"]))
            flops_values.append(float(resources["prefill_flops"]))
            text_token_values.append(float(resources["text_tokens"]))
            image_token_values.append(float(resources["image_tokens"]))
            retained_image_token_values.append(float(resources["retained_image_tokens"]))

        _sync_cuda()
        if stage_timer is not None:
            stage_timer.begin()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        _sync_cuda()
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        stage_metrics = stage_timer.finish(t1 - t0, int(out.shape[-1] - inputs["input_ids"].shape[-1])) if stage_timer is not None else {}

        text = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        row = dataset.rows[idx]
        scored = score_prediction(
            metric=spec.metric,
            prediction_text=text,
            answer=row.get("answer"),
            answers=row.get("answers"),
            choices=row.get("choices"),
            question=row.get("question"),
        )
        prediction = {
            "index": idx,
            "prediction_text": text,
            "time": t1 - t0,
            "total_time_s": t1 - t0,
            **scored,
            **stage_metrics,
        }
        if native_graphs is not None:
            graph_after = native_graphs.stats()
            if graph_after["captures"] != graph_before["captures"] or graph_after["cold_layer_fallbacks"] != graph_before["cold_layer_fallbacks"]:
                raise RuntimeError(f"Native CUDA Graph capture/miss during timed sample {idx}")
            prediction.update(native_graph_prepare_s=graph_prepare_s, native_graph_logits_kv_exact=True,
                              native_graph_stats=graph_after)
        if item_prefill_s is not None:
            prediction["prefilling_time_s"] = item_prefill_s
        if resources is not None:
            prediction.update(resources)
        predictions.append(prediction)

        if (idx + 1) % log_every == 0 or idx == len(dataset) - 1:
            running_score = sum(p["score"] for p in predictions) / max(len(predictions), 1)
            print(f"  [{idx+1}/{len(dataset)}] score={running_score:.4f} last={text[:50]!r}", flush=True)

    summary = summarize_metric(spec.metric, predictions, dataset.rows)
    summary["total_time"] = total_time
    summary["total_time_s"] = total_time
    if native_graphs is not None:
        summary["native_cuda_graphs"] = True
        summary["native_graph_stats"] = native_graphs.stats()
        summary["native_graph_prepare_s"] = sum(p["native_graph_prepare_s"] for p in predictions)
        summary["native_graph_logits_kv_exact"] = all(p["native_graph_logits_kv_exact"] for p in predictions)
    summary["total_time_minsec"] = format_seconds_minsec(total_time)
    summary["avg_time"] = total_time / max(len(predictions), 1)
    summary["avg_total_time_s"] = total_time / max(len(predictions), 1)
    if stage_timer is not None:
        stage_timer.remove()
        for key in ("generation_prefill_time_s", "decode_time_s", "generation_overhead_s", "decode_steps", "generated_tokens"):
            summary[key] = sum(p[key] for p in predictions)
        summary["decode_ms_per_step"] = 1000 * summary["decode_time_s"] / summary["decode_steps"] if summary["decode_steps"] else None
        summary["actual_prefill_kv_cache_mb"] = _mean([p["actual_prefill_kv_cache_mb"] for p in predictions])
    if measure_peak_memory:
        from src.peak_memory import summarize_peak_memory
        summary.update(summarize_peak_memory(predictions))
        for key in ('peak_memory_mb', 'peak_reserved_mb', 'peak_memory_delta_mb',
                    'prefill_peak_allocated_mb', 'decode_peak_allocated_mb',
                    'prefill_peak_reserved_mb', 'decode_peak_reserved_mb'):
            values = [p[key] for p in predictions if p.get(key) is not None]
            summary[key] = max(values) if values else None
            summary[key + '_mean'] = _mean(values)
        summary['peak_memory_definition'] = 'Maximum across requests of process CUDA allocated bytes during warmed generation, including weights and retained graph pools; MiB. Capture itself excluded; reserved memory reported separately.'
    if measure_prefill:
        summary["prefilling_time_s"] = prefill_time
        summary["prefilling_time_minsec"] = format_seconds_minsec(prefill_time)
        summary["avg_prefilling_time_s"] = prefill_time / max(len(predictions), 1)
        summary["flops"] = _mean(flops_values)
        summary["kv_cache_mb"] = _mean(kv_cache_values)
        summary["text_tokens_avg"] = _mean(text_token_values)
        summary["image_tokens_avg"] = _mean(image_token_values)
        summary["retained_image_tokens_avg"] = _mean(retained_image_token_values)
        summary["timing"] = {
            "total_time_s": total_time,
            "total_time_minsec": summary["total_time_minsec"],
            "prefilling_time_s": prefill_time,
            "prefilling_time_minsec": summary["prefilling_time_minsec"],
        }
        summary["resources"] = {
            "flops": summary["flops"],
            "kv_cache_mb": summary["kv_cache_mb"],
            "text_tokens_avg": summary["text_tokens_avg"],
            "image_tokens_avg": summary["image_tokens_avg"],
            "retained_image_tokens_avg": summary["retained_image_tokens_avg"],
        }
    return predictions, summary


def main():
    parser = argparse.ArgumentParser("Baseline evaluation for project baselines")
    parser.add_argument("--method", default="fastv", help="Method name or 'all' or 'no-train'")
    parser.add_argument("--retention", type=float, default=0.10, help="Image token retention ratio for token-pruning baselines")
    parser.add_argument("--benchmark", default="mmstar", help="Benchmark name or 'all'")
    parser.add_argument("--model-label", default=None, help="One of the five supported model labels, comma-list, or all")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data", default=None, help="Override benchmark JSONL path, e.g. fixed speed-test subset.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--measure-peak-memory", action="store_true", help="Record warmed generation CUDA allocated/reserved peaks; summarize with dataset maximum.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--deepstack", choices=("off",), default="off")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--measure-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure Qwen prefill time and resource metrics in the benchmark_prefill.py table style.",
    )
    parser.add_argument("--speed-warmup", type=int, default=1, help="Unmeasured warmup rounds for --measure-prefill.")
    parser.add_argument("--native-cuda-graphs", action="store_true", help="Graph native vision and decoder layers; verify each input and report unmeasured per-input preparation.")
    parser.add_argument("--measure-decode", action="store_true", help="Time real prefill/decode forwards inside native generate; record token counts and bookkeeping separately.")
    parser.add_argument("--optimize-attention-metadata", action="store_true", help="Avoid mistaking gaps in pruned RoPE positions for packed sequences in FA2.")
    parser.add_argument("--dyvte-gate-checkpoint", default=None, help="Trained Qwen3-VL DyVTE gate checkpoint.")
    parser.add_argument("--dyvte-exit-layer", type=int, default=None, help="Static DyVTE visual-token exit layer for ablation.")
    parser.add_argument("--dyvte-min-exit-layer", type=int, default=None, help="Minimum layer where a trained DyVTE gate may exit.")
    parser.add_argument("--dyvte-max-exit-layer", type=int, default=None, help="Maximum layer where a trained DyVTE gate may exit.")
    parser.add_argument(
        "--redundancylens-attn-reduction-layers",
        default=None,
        help="0-based layers where RedundancyLens applies Hollow Attention. This is the concrete L_RA layer set.",
    )
    parser.add_argument(
        "--redundancylens-ffn-reduction-layers",
        default=None,
        help="0-based layers where RedundancyLens applies Probe-Activated Dynamic FFN. This is the concrete L_RF layer set.",
    )
    parser.add_argument(
        "--redundancylens-attn-priority-layers",
        default=None,
        help="Layer Ranking priority list for attention; the first L_RA layers are reduced.",
    )
    parser.add_argument(
        "--redundancylens-ffn-priority-layers",
        default=None,
        help="Layer Ranking priority list for FFN; the first L_RF layers are reduced.",
    )
    parser.add_argument("--redundancylens-l-ra", type=int, default=None, help="Number of attention-reduction layers chosen from the attention LR list.")
    parser.add_argument("--redundancylens-l-rf", type=int, default=None, help="Number of FFN-reduction layers chosen from the FFN LR list.")
    parser.add_argument("--redundancylens-attention-range", type=int, default=None, help="Hollow Attention R_A for visual tokens.")
    parser.add_argument("--redundancylens-ffn-channel-ratio", type=float, default=None, help="Probe-Activated FFN K ratio for visual tokens.")
    parser.add_argument("--redundancylens-ffn-sample-divisor", type=int, default=None, help="Visual-token sample divisor for the Probe-Activated FFN probe.")
    parser.add_argument("--redundancylens-hollow-attention", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    set_global_seed(int(args.seed))

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # Determine methods and benchmarks to run
    if args.method == "all":
        methods = METHODS
    elif args.method == "no-train":
        methods = NO_TRAIN_METHODS
    else:
        methods = [item for item in args.method.replace(",", " ").split() if item]
    unknown_methods = [method for method in methods if method not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unsupported method={unknown_methods}; choose from {METHODS}")

    benchmarks = parse_benchmark_names(args.benchmark)
    # Exclude ocrbench and textvqa by default if running "all"
    if args.benchmark == "all":
        benchmarks = [b for b in benchmarks if b not in ("ocrbench", "textvqa")]

    base_output = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "eval" / "baselines"
    model_labels = parse_model_labels(args.model_label)
    base_summaries: dict[tuple[str, str], dict] = {}

    for model_label in model_labels:
        model_kind, model_path, resolved_label = resolve_model(model_label, args.model_path)
        model_name = resolved_label or Path(model_path).name
        for method in methods:
            method_tag = _baseline_output_tag(method, args.retention, args)
            print(f"\n{'='*60}")
            print(
                f"Model: {model_name} ({model_kind}) | Method: {method} | "
                f"Config: {method_tag} | Benchmarks: {benchmarks}"
            )
            print(f"{'='*60}")

            if model_kind == "qwen":
                model, processor = load_baseline_model(
                    method,
                    model_path,
                    dtype,
                    device,
                    args.retention,
                    attn_implementation=args.attn_implementation,
                )
                if args.deepstack == "off":
                    from src.qwen_deepstack import disable_qwen_deepstack
                    disable_qwen_deepstack(model)
            elif model_kind == "llava":
                processor, model = load_llava_baseline_model(
                    model_path,
                    dtype=dtype,
                    device=str(device),
                    attn_implementation=args.attn_implementation,
                )
            else:
                raise ValueError(f"Unsupported model kind={model_kind!r}")

            for benchmark in benchmarks:
                spec = get_benchmark_spec(benchmark)
                max_new_tokens = args.max_new_tokens or spec.max_new_tokens
                data_path = resolve_data_path(args.data or spec.default_data, args.data_root)

                print(f"\n  --- {benchmark} (max_new_tokens={max_new_tokens}) ---")

                max_samples = None if args.max_samples <= 0 else args.max_samples
                if resolved_label:
                    out_dir = base_output / model_name / method / method_tag / benchmark
                else:
                    out_dir = base_output / method / method_tag / benchmark
                out_dir.mkdir(parents=True, exist_ok=True)

                if model_kind == "qwen":
                    dataset = QwenBenchmarkDataset(
                        data_path, processor, benchmark,
                        data_root=str(Path(data_path).parent),
                        max_samples=max_samples,
                    )

                    # Get image token range from first sample for config
                    item0 = dataset[0]
                    mm = item0["mm_token_type_ids"]
                    img_indices = torch.nonzero(mm != 0, as_tuple=True)[0]
                    img_start = img_indices[0].item() if img_indices.numel() > 0 else 0
                    img_len = img_indices.numel()

                    # Configure baseline behavior after the visual token span is known.
                    configure_baseline(model, method, args.retention, img_start, img_len, args=args)
                    metadata_optimization = None
                    if args.optimize_attention_metadata:
                        from src.qwen_attention_metadata import optimize_qwen_attention_metadata
                        metadata_optimization = optimize_qwen_attention_metadata(model)

                    native_graphs = None
                    if args.native_cuda_graphs:
                        if method not in {BASE_METHOD, "fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"}:
                            raise ValueError(f"--native-cuda-graphs is not validated for {method}")
                        from src.qwen_native_graph import NativeDecoderGraphs
                        native_graphs = NativeDecoderGraphs(model)

                    predictions, summary = evaluate_single(
                        model, processor, dataset, benchmark,
                        max_new_tokens=max_new_tokens,
                        log_every=args.log_every,
                        method=method,
                        retention=args.retention,
                        dtype=dtype,
                        measure_prefill=bool(args.measure_prefill),
                        speed_warmup=int(args.speed_warmup),
                        measure_decode=bool(args.measure_decode),
                        native_graphs=native_graphs,
                        measure_peak_memory=bool(args.measure_peak_memory),
                    )
                    if native_graphs is not None:
                        native_graphs.remove()
                    if metadata_optimization is not None:
                        metadata_optimization.remove()
                    summary["attention_metadata_optimized"] = bool(args.optimize_attention_metadata)
                    summary["actual_attention_implementation"] = model.model.language_model.config._attn_implementation
                    summary["deepstack"] = args.deepstack
                else:
                    dataset = LlavaBenchmarkDataset(
                        data_path, processor, benchmark,
                        data_root=str(Path(data_path).parent),
                        max_samples=max_samples,
                    )
                    predictions, summary = evaluate_llava_baseline(
                        model,
                        processor,
                        dataset,
                        method=method,
                        benchmark=benchmark,
                        retention=args.retention,
                        max_new_tokens=max_new_tokens,
                        log_every=args.log_every,
                    )

                # Save results
                summary["method"] = method
                summary["model_kind"] = model_kind
                summary["model_label"] = resolved_label
                summary["model_path"] = model_path
                summary["data_path"] = data_path
                summary["config_tag"] = method_tag
                summary["retention"] = None if method in {DYVTE_METHOD, REDUNDANCYLENS_METHOD} else args.retention
                if model_kind == "qwen":
                    summary["method_config"] = _method_config_summary(model, method)
                summary["benchmark"] = benchmark
                summary["display_name"] = spec.display_name
                summary["metric"] = spec.metric
                if method == BASE_METHOD:
                    if args.measure_prefill:
                        summary["speedup_total"] = 1.0
                        summary["speedup_prefilling"] = 1.0
                    base_summaries[(model_name, benchmark)] = summary
                elif args.measure_prefill:
                    reference = base_summaries.get((model_name, benchmark))
                    if reference:
                        total_time_s = float(summary.get("total_time_s", 0.0))
                        prefill_time_s = float(summary.get("prefilling_time_s", 0.0))
                        ref_total = float(reference.get("total_time_s", 0.0))
                        ref_prefill = float(reference.get("prefilling_time_s", 0.0))
                        summary["speedup_total"] = ref_total / total_time_s if total_time_s > 0 else None
                        summary["speedup_prefilling"] = ref_prefill / prefill_time_s if prefill_time_s > 0 else None
                        summary["base_reference"] = {
                            "method": BASE_METHOD,
                            "total_time_s": ref_total,
                            "prefilling_time_s": ref_prefill,
                        }
                    else:
                        summary["speedup_total"] = None
                        summary["speedup_prefilling"] = None
                (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))
                (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, default=str))
                if args.measure_prefill and "prefilling_time_s" in summary:
                    print(
                        f"  => {benchmark}: score={summary['score']:.4f} "
                        f"total={summary['total_time_s']:.4f}s prefill={summary['prefilling_time_s']:.4f}s "
                        f"kv={summary.get('kv_cache_mb', 0.0):.2f}MB flops={summary.get('flops', 0.0):.4e} "
                        f"speedup_total={summary.get('speedup_total')} speedup_prefill={summary.get('speedup_prefilling')} "
                        f"({out_dir})"
                    )
                else:
                    print(f"  => {benchmark}: score={summary['score']:.4f} ({out_dir})")

            del model
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
