"""Verified batch-one Qwen3.5 prefill/decode graphs for latency measurement."""
import copy
import torch
from src.graphs import clone_tree


def clone_cache(source):
    cache = copy.copy(source)
    cache.layers = [copy.copy(layer) for layer in source.layers]
    for layer in cache.layers:
        layer.__dict__ = clone_tree(layer.__dict__)
    return cache


class CallGraph:
    def __init__(self, fn):
        self.fn = fn  # Keep every tensor referenced by the captured kernels alive.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn()
        self.graph.replay()

    def replay(self):
        self.graph.replay()
        return self.output


class Qwen35Graphs:
    def __init__(self, model, adapter, controller):
        self.model, self.adapter, self.controller = model, adapter, controller
        self.language = model.model.language_model
        self.captures = 0
        self.native_checked = False
        self.allow_capture = False
        self.entries = []
        self.prefill_entry = self.vision_entry = None
        # Static visual/text indices are computed outside capture. Prediction
        # itself is still recomputed from the current initial embedding in graph.
        controller.hook.remove()
        def start(module, args, kwargs):
            h = kwargs.get('inputs_embeds')
            controller.predictions = None
            if adapter is not None and h is not None and h.shape[1] > 1:
                controller.visual_idx, controller.text_idx = self.visual_idx, self.text_idx
                controller.predictions = adapter(h.index_select(1, self.visual_idx))
        self.start = start
        self.hook = self.language.register_forward_pre_hook(self.start, with_kwargs=True)

    def prepare(self, inputs):
        from src.qwen35 import initial_context
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids, get_vision_attention_seqlens
        from transformers.cache_utils import DynamicCache
        self.release()
        self.inputs = inputs
        self.visual_idx = inputs['mm_token_type_ids'][0].eq(1).nonzero().flatten()
        self.text_idx = inputs['mm_token_type_ids'][0].eq(0).nonzero().flatten()
        self.model.model.rope_deltas = None
        reference_context = initial_context(self.model, inputs)
        self.positions = reference_context['position_ids']
        # The sequential axis only controls packing, which is absent for batch1.
        if self.positions.shape[0] == 4:
            self.positions = self.positions[1:]
        length = inputs['input_ids'].shape[1]
        self.next_positions = (torch.arange(length, length+7, device='cuda').view(1,1,7)
                               + self.model.model.rope_deltas.view(1,1,1)).expand(3,1,7)
        vision = self.model.model.visual
        grid = inputs['image_grid_thw']
        indices, weights = get_vision_interpolation_indices_and_weights(grid,
            num_grid_per_side=vision.num_grid_per_side, mode=vision.interpolation_mode,
            align_corners=vision.interpolation_align_corners, spatial_merge_size=vision.config.spatial_merge_size)
        position_ids = get_vision_position_ids(grid, vision.spatial_merge_size)
        cu, maximum = get_vision_attention_seqlens(grid, vision.config)
        # Precomputed metadata avoids host reads; vision kernels are unchanged.
        metadata = dict(position_ids=position_ids, cu_seqlens=cu, max_seqlen=maximum,
                        interp_indices=indices, interp_weights=weights)
        self.mask = inputs['mm_token_type_ids'].eq(1).unsqueeze(-1).expand_as(reference_context['inputs_embeds'])
        def context():
            features = vision(inputs['pixel_values'], grid_thw=grid, **metadata).pooler_output
            return self.model.get_input_embeddings()(inputs['input_ids']).masked_scatter(self.mask, features)
        self.vision_entry = CallGraph(context)
        self.hidden = self.vision_entry.output
        assert torch.equal(self.hidden, reference_context['inputs_embeds']), 'Qwen3.5 vision graph mismatch'
        self.empty_cache = DynamicCache(config=self.language.config)
        self.expected_logits, self.expected_tokens = self.eager_request()
        if not self.native_checked:
            # Also check the ordinary multimodal HF entrypoint, not only the
            # prepared language-model call used by the graph implementation.
            eos = self.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            # Restore the ordinary controller too: a second prepared call using
            # our static-index hook would not independently validate that hook.
            self.hook.remove()
            ordinary_hook = self.language.register_forward_pre_hook(self.controller._start, with_kwargs=True)
            try:
                mode = 'adapter' if self.adapter is not None else 'native'
                with self.controller.activate(mode, inputs['mm_token_type_ids'].eq(1)):
                    out = self.model(**inputs, use_cache=True, logits_to_keep=1)
                    for i, expected in enumerate(self.expected_logits):
                        assert torch.equal(out.logits, expected), (i, 'Native multimodal/graph setup mismatch')
                        scores = out.logits[:, -1].float().clone()
                        scores[:, eos] = -float('inf')
                        if i < 7:
                            out = self.model(input_ids=scores.argmax(-1).view(1,1),
                                past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                ordinary_hook.remove()
                self.hook = self.language.register_forward_pre_hook(self.start, with_kwargs=True)
            self.native_checked = True

    def build_context(self):
        self.hidden = self.vision_entry.replay()

    def capture(self, enabled):
        self.allow_capture = enabled

    def stats(self):
        return (self.captures, 0, len(self.entries))

    def forward(self, hidden, positions, cache):
        out = self.language(inputs_embeds=hidden, position_ids=positions,
            attention_mask={'full_attention':None, 'linear_attention':None},
            past_key_values=cache, use_cache=True)
        return self.model.lm_head(out.last_hidden_state[:, -1:]), out.past_key_values

    def prefill(self):
        if self.prefill_entry is None:
            assert self.allow_capture
            self.prefill_entry = CallGraph(lambda: self.forward(self.hidden, self.positions, clone_cache(self.empty_cache)))
            self.captures += 1
        return self.prefill_entry.replay()

    def step(self, token, cache, i):
        if i == len(self.entries):
            assert self.allow_capture
            static_token = token.clone()
            entry = CallGraph(lambda: self.forward(self.model.get_input_embeddings()(static_token),
                self.next_positions[:, :, i:i+1], clone_cache(cache)))
            self.entries.append((static_token, entry))
            self.captures += 1
        static_token, entry = self.entries[i]
        static_token.copy_(token)
        return entry.replay()

    def eager_request(self):
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        logits, cache = self.forward(self.hidden, self.positions, clone_cache(self.empty_cache))
        outputs, tokens = [], []
        for i in range(8):
            outputs.append(logits.clone())
            scores = logits[:, -1].float().clone()
            scores[:, eos] = -float('inf')
            token = scores.argmax(-1).view(1,1)
            tokens.append(token)
            if i != 7:
                logits, cache = self.forward(self.model.get_input_embeddings()(token), self.next_positions[:, :, i:i+1], cache)
        return outputs, torch.cat(tokens, 1)[0].tolist()

    def verify(self, logits, tokens, eos):
        assert tokens == self.expected_tokens, 'Qwen3.5 graph/eager greedy mismatch'
        assert all(torch.equal(a,b) for a,b in zip(logits,self.expected_logits)), 'Qwen3.5 graph/eager logits mismatch'
        # Recompute eagerly after all timed graph replays to detect state leakage.
        outputs, expected = self.eager_request()
        assert expected == tokens and all(torch.equal(a,b) for a,b in zip(outputs,logits)), 'Hybrid state reset mismatch'

    def release(self):
        self.entries.clear()
        self.prefill_entry = self.vision_entry = None
        self.expected_logits = []
