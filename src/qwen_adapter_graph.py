"""Warmed FA2 graphs for genuine single-token adapter decode with growing KV."""
from collections import OrderedDict

import torch
from src.qwen_native_graph import clone_tree, copy_tree, signature, clone_kv_tensors


def cache_containers(cache):
    result = dict(cache)
    result["layers"] = [dict(layer) for layer in cache["layers"]]
    return result


class AdapterDecodeGraph:
    def __init__(self, model, adapter, tokens, cache, kwargs, plan):
        from src.model import qwen_embedding_adapter_decode_step
        self.inputs = clone_tree((tokens, cache, kwargs, plan))

        def forward():
            tokens, initial, kwargs, plan = self.inputs
            return qwen_embedding_adapter_decode_step(model, adapter, tokens,
                cache_containers(initial), attention_plan=plan, **kwargs)

        with torch.cuda.device(tokens.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.graph.replay()

    def replay(self, tokens, cache, kwargs, plan):
        copy_tree(self.inputs, (tokens, cache, kwargs, plan))
        self.graph.replay()
        logits, output_cache = self.output
        # Visual K/V and prefix metadata are immutable during fast decode.
        # Own only the updated tensors; retain the caller's visual cache.
        updated = clone_tree((logits, output_cache["text_mask"],
            output_cache["next_position_ids"], output_cache["next_text_positions"]))
        copied = clone_kv_tensors([layer[name] for layer in output_cache["layers"] for name in ("text_key", "text_value")])
        result = cache_containers(cache)
        for layer, (key, value) in zip(result["layers"], zip(copied[::2], copied[1::2])):
            layer["text_key"], layer["text_value"] = key, value
        result["text_mask"], result["next_position_ids"], result["next_text_positions"] = updated[1:]
        result["dense_decode_ready"] = output_cache.get("dense_decode_ready", False)
        return updated[0], result


class QwenAdapterDecodeGraphs:
    def __init__(self, model, adapter, max_shapes=12):
        self.model, self.adapter = model, adapter
        self.entries = OrderedDict()
        self.max_shapes = max_shapes
        self.enabled = True
        self.allow_capture = False
        self.captures = self.replays = self.cold_fallbacks = 0

    def __call__(self, model, adapter, tokens, cache, **kwargs):
        from src.model import qwen_embedding_adapter_decode_step, _qwen_decode_attention_mask
        from src.qwen_adapter_fa2 import decode_plan
        if not self.enabled:
            return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, **kwargs)
        if cache.get("attention_implementation") != "flash_attention_2":
            raise ValueError("Adapter decode graphs require FA2")
        if cache.get("dense_decode_ready", False) and kwargs.get("token_active_mask") is None:
            plan = {"dense_decode": True}
        else:
            mask = _qwen_decode_attention_mask(cache, cache["next_position_ids"], kwargs.get("token_active_mask"))
            plan = decode_plan(mask)
        key = signature((tokens, cache, kwargs, plan))
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                self.cold_fallbacks += 1
                return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, attention_plan=plan, **kwargs)
            entry = AdapterDecodeGraph(model, adapter, tokens, cache, kwargs, plan)
            self.entries[key] = entry
            self.captures += 1
            while len(self.entries) > self.max_shapes:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        self.replays += 1
        return entry.replay(tokens, cache, kwargs, plan)

    def stats(self):
        return dict(captures=self.captures, replays=self.replays, cold_fallbacks=self.cold_fallbacks)
