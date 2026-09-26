"""Shared model/adapter construction, checkpoint loading and DeepStack policy."""
from __future__ import annotations


# Model/adapter setup shared by training and evaluation.
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from src.model import QwenEmbeddingAdapter


def load_frozen_qwen3vl(*args, **kwargs):
    from src.model import load_frozen_qwen3vl as load
    return load(*args, **kwargs)


def load_frozen_llava(*args, **kwargs):
    from src.model import load_frozen_llava as load
    return load(*args, **kwargs)


def create_qwen_adapter(language_model, *, mode, rank, device=None, dtype=None):
    from src.model import QwenEmbeddingAdapter
    return QwenEmbeddingAdapter.from_language_model(
        language_model, mode=mode, visual_adapter_rank=rank,
    ).to(device=device, dtype=dtype)


def create_llava_kv_adapter(**config):
    from src.model import PerLayerKVAdapter
    return PerLayerKVAdapter(**config)


def load_adapter_checkpoint(*args, **kwargs):
    from src.model import load_adapter_checkpoint as load
    return load(*args, **kwargs)


def load_qwen_embedding_adapter_checkpoint(
    checkpoint_path: str | Path,
    language_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[QwenEmbeddingAdapter, dict[str, Any]]:
    import torch
    from src.model import canonical_adapter_mode, QWEN_EMBEDDING_ADAPTER_MODES
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    saved_config = checkpoint.get("adapter_config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    mode = canonical_adapter_mode(str(saved_config.get("output_mode") or checkpoint_args.get("output_mode", "embedding_adapter")))
    if mode not in QWEN_EMBEDDING_ADAPTER_MODES:
        raise ValueError(f"checkpoint output_mode={mode!r} is not a Qwen embedding adapter mode")
    adapter = create_qwen_adapter(
        language_model,
        mode=mode,
        rank=int(
            saved_config.get(
                "visual_adapter_rank",
                checkpoint_args.get(
                    "visual_adapter_rank",
                    checkpoint_args.get("visual_transform_rank", 128),
                ),
            )
        ),
        device=device, dtype=dtype,
    )
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    meta = {
        "args": checkpoint_args,
        "global_step": checkpoint.get("global_step", checkpoint.get("step", None)) if isinstance(checkpoint, dict) else None,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
    return adapter, meta

def load_qwen35(config, device):
    import json
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from src.qwen35 import install_fast_kernels, StaticVisualAdapter, VisualAdapterController
    from src.model_setup import disable_qwen_deepstack

    kernels = install_fast_kernels()
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(config['model_path'],
        dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': str(device)}, local_files_only=True).eval().requires_grad_(False)
    disable_qwen_deepstack(model)
    c = model.config.text_config
    assert c.hidden_size == 2560 and c.num_hidden_layers == 32
    assert c.layer_types.count('linear_attention') == 24 and c.layer_types.count('full_attention') == 8
    assert not model.config.vision_config.deepstack_visual_indexes
    assert model.model.language_model.config._attn_implementation == 'flash_attention_2'
    assert model.model.visual.config._attn_implementation == 'flash_attention_2'
    adapter = StaticVisualAdapter(rank=config['rank']).to(device)
    assert sum(p.numel() for p in adapter.parameters()) == 20971520
    controller = VisualAdapterController(model, adapter)
    print('QWEN35_BACKBONE_FROZEN', json.dumps(kernels), flush=True)
    return processor, model, adapter, controller


# Default DeepStack-off policy; training may explicitly retain a native teacher.
def _reject_deepstack(*args, **kwargs):
    raise AssertionError("DeepStack execution is disabled for this model")


def disable_qwen_deepstack(model):
    """Disable auxiliary vision mergers and language injection before any forward.

    Accept either the outer generation model or its multimodal backbone. Keep
    checkpoint parameters/configuration intact so loading and saving stay compatible.
    The main vision merger is unchanged. Qwen3.5 already has no DeepStack branches.
    """
    backbone = model if hasattr(model, "visual") else model.model
    visual = backbone.visual
    language = backbone.language_model
    visual.deepstack_visual_indexes = []
    if hasattr(language, "_deepstack_process"):
        language._deepstack_process = _reject_deepstack
    if not hasattr(backbone, "_deepstack_guard_handles"):
        backbone._deepstack_guard_handles = [
            module.register_forward_pre_hook(_reject_deepstack)
            for module in getattr(visual, "deepstack_merger_list", ())
        ]
    backbone._benchmark_deepstack = "off"
    model._benchmark_deepstack = "off"
    return model
