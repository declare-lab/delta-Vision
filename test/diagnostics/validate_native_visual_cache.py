"""GPU integration validation before the full native-visual-cache experiment."""
import argparse
from pathlib import Path

import torch

from src.visual_channel_native_cache import (
    NativeVisualHook, _layers, _to_device_item, configure_runtime, dataset,
    dump_json, extend, forward, generate, get_model, positions,
)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-kind", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    configure_runtime()
    root = Path(args.output)
    processor, model = get_model(args.model_kind)
    hook = NativeVisualHook(_layers(args.model_kind, model))
    checks = []
    for benchmark in ("mmstar", "realworldqa"):
        ds = dataset(root, args.model_kind, processor, benchmark)
        inputs = _to_device_item(ds[0], torch.device("cuda:0"))
        hook.positions = positions(args.model_kind, model, inputs)
        hook.mode = "capture"
        logits = forward(model, inputs, hook).logits[0, -1].float()
        native = dict(hook.native)
        hook.mode = "off"
        native_text, tokens = generate(model, processor, inputs, hook, logits, 8)
        hook.replacements = native
        hook.mode = "replay"
        replay_logits = forward(model, inputs, hook).logits[0, -1].float()
        replay_text, replay_tokens = generate(model, processor, inputs, hook, replay_logits, 8)
        diff = float((logits - replay_logits).abs().max())
        assert diff == 0 and tokens == replay_tokens, (diff, tokens, replay_tokens)
        hook.mode = "capture"
        forward(model, extend(inputs, tokens[0]), hook)
        prefix_error = max(float((hook.native[l].float() - x.float()).abs().max()) for l, x in native.items())
        hook.replacements = {l: torch.zeros_like(x) for l, x in native.items()}
        hook.mode = "replay"
        removed_logits = forward(model, inputs, hook).logits[0, -1].float()
        removed_diff = float((logits - removed_logits).abs().max())
        assert removed_diff > 0
        row = {"benchmark": benchmark, "visual_tokens": len(hook.positions), "layers": len(hook.layers),
               "native_text": native_text, "replay_text": replay_text, "native_replay_max_logit_abs": diff,
               "appended_token_visual_max_abs": prefix_error, "zero_visual_max_logit_abs": removed_diff}
        checks.append(row)
        print(row, flush=True)
    hook.close()
    dump_json(root / f"validation_{args.model_kind}.json", checks)


if __name__ == "__main__":
    main()
