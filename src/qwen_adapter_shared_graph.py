"""Single-token graphs sharing immutable visual KV across a request's steps."""
from collections import OrderedDict

import torch
from src.qwen_adapter_graph import AdapterDecodeGraph, QwenAdapterDecodeGraphs, cache_containers
from src.qwen_native_graph import clone_tree, copy_tree, signature


class SharedVisualDecodeGraph(AdapterDecodeGraph):
    def __init__(self, model, adapter, tokens, cache, kwargs, plan, visual):
        from src.model import qwen_embedding_adapter_decode_step
        mutable = cache_containers(cache)
        mutable["layers"] = [{k:v for k,v in layer.items() if k not in ("visual_key", "visual_value")}
            for layer in cache["layers"]]
        # The inherited replay copies only this mutable portion. Graph kernels
        # read the separately owned visual tensors shared by all growing lengths.
        self.inputs = clone_tree((tokens, mutable, kwargs, plan))
        initial = cache_containers(self.inputs[1])
        for layer, (key, value) in zip(initial["layers"], visual):
            layer.update(visual_key=key, visual_value=value)

        def forward():
            token, _, kw, prepared_plan = self.inputs
            return qwen_embedding_adapter_decode_step(model, adapter, token,
                cache_containers(initial), attention_plan=prepared_plan, **kw)

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


class SharedVisualDecodeGraphs(QwenAdapterDecodeGraphs):
    """Fast-path visual KV is immutable and newly owned by each prefill.

    Retain source tensor references and check their identities, preventing an
    allocator's reused address from being mistaken for the previous request.
    Every new request refreshes the shared visual buffers before its first step.
    As with the existing graph runners, calls are sequential on one CUDA stream.
    """
    def __init__(self, model, adapter, max_shapes=12):
        super().__init__(model, adapter, max_shapes)
        self.visual_banks = OrderedDict()
        self.visual_sources = ()
        self.current_visual = None
        self.visual_prefix_copies = 0

    def _visual(self, cache):
        sources = tuple(t for layer in cache["layers"] for t in (layer["visual_key"], layer["visual_value"]))
        if len(sources) == len(self.visual_sources) and all(a is b for a,b in zip(sources,self.visual_sources)):
            return self.current_visual
        key = signature(sources)
        visual = self.visual_banks.get(key)
        if visual is None:
            cloned = clone_tree(sources)
            visual = tuple(zip(cloned[::2],cloned[1::2]))
            self.visual_banks[key] = visual
            while len(self.visual_banks) > 2:
                self.visual_banks.popitem(last=False)
        else:
            copy_tree(visual, tuple(zip(sources[::2],sources[1::2])))
        self.visual_banks.move_to_end(key)
        self.visual_sources, self.current_visual = sources, visual
        self.visual_prefix_copies += 1
        return visual

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
            plan = decode_plan(_qwen_decode_attention_mask(cache, cache["next_position_ids"], kwargs.get("token_active_mask")))
        visual = self._visual(cache)
        key = (id(visual), signature((tokens, cache, kwargs, plan)))
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                self.cold_fallbacks += 1
                return qwen_embedding_adapter_decode_step(model, adapter, tokens, cache, attention_plan=plan, **kwargs)
            entry = SharedVisualDecodeGraph(model, adapter, tokens, cache, kwargs, plan, visual)
            self.entries[key] = entry
            self.captures += 1
            while len(self.entries) > self.max_shapes:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        self.replays += 1
        return entry.replay(tokens, cache, kwargs, plan)

    def stats(self):
        return dict(super().stats(), visual_prefix_copies=self.visual_prefix_copies)
