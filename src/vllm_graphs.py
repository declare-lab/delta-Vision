"""Bounded, shape-specific CUDA graphs over vLLM-owned KV and recurrent caches.

Capture must not advance recurrent memory: restore the affected request slots
after every warmup/capture, then replay once. No model weights or cache ownership
are substituted. Prefill topology and sequence lengths are explicit graph keys.
"""
import copy
import dataclasses
from collections import OrderedDict

import torch


def map_tree(x, tensor_fn, memo=None):
    memo = {} if memo is None else memo
    if id(x) in memo:
        return memo[id(x)]
    if torch.is_tensor(x):
        y = tensor_fn(x)
    elif dataclasses.is_dataclass(x):
        y = copy.copy(x)
        for f in dataclasses.fields(x):
            setattr(y, f.name, map_tree(getattr(x, f.name), tensor_fn, memo))
        for name in ('_graph_starts', '_graph_lengths'):
            if hasattr(x, name): setattr(y, name, getattr(x, name))
    elif isinstance(x, dict):
        y = {k:map_tree(v, tensor_fn, memo) for k,v in x.items()}
    elif isinstance(x, (list, tuple)):
        y = type(x)(map_tree(v, tensor_fn, memo) for v in x)
    else:
        return x
    memo[id(x)] = y
    return y


def signature(x):
    if torch.is_tensor(x):
        return ('tensor', tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
    if dataclasses.is_dataclass(x):
        return (type(x).__name__, tuple((f.name,signature(getattr(x,f.name))) for f in dataclasses.fields(x)))
    if isinstance(x, dict): return tuple((k,signature(v)) for k,v in x.items())
    if isinstance(x, (tuple,list)): return tuple(signature(v) for v in x)
    return x


def inspect_inputs(x, collect=True):
    import torch
    import dataclasses
    memo = {}
    cuda, cpu = [], []
    def visit(value):
        compound = torch.is_tensor(value) or dataclasses.is_dataclass(value) or isinstance(value, (dict, tuple, list))
        if not compound: return value
        if id(value) in memo: return ('alias', memo[id(value)])
        memo[id(value)] = len(memo)
        if torch.is_tensor(value):
            if collect:
                (cuda if value.device.type == 'cuda' else cpu).append(value)
            return ('tensor', tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        if dataclasses.is_dataclass(value):
            return (type(value).__name__, tuple((f.name, visit(getattr(value, f.name))) for f in dataclasses.fields(value)))
        if isinstance(value, dict): return tuple((k, visit(v)) for k, v in value.items())
        return tuple(visit(v) for v in value)
    key = visit(x)
    return key, (cuda, cpu)


def alias_signature(x):
    return inspect_inputs(x, collect=False)[0]



def copy_inputs(target, source):
    targets, sources, seen = [], [], set()
    def collect(a,b):
        if id(a) in seen: return
        seen.add(id(a))
        if torch.is_tensor(a):
            if a.device.type == 'cuda': targets.append(a); sources.append(b)
            else: a.copy_(b)
        elif dataclasses.is_dataclass(a):
            for f in dataclasses.fields(a): collect(getattr(a,f.name),getattr(b,f.name))
        elif isinstance(a,dict):
            for k in a: collect(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            for aa,bb in zip(a,b): collect(aa,bb)
    collect(target,source)
    if targets: torch._foreach_copy_(targets,sources)


class RuntimeGraphs:
    def __init__(self, model, prefill_capacity=1, decode_capacity=8):
        self.model = model
        self.fast_metadata = (getattr(model.config, 'model_type', None) == 'qwen3_vl'
                              and getattr(model.config, 'delta_vision_fast_metadata', True))
        self.original = model.forward
        self.prefill_capacity, self.decode_capacity = prefill_capacity, decode_capacity
        self.entries = {'prefill':OrderedDict(), 'decode':OrderedDict()}
        self.allow_capture = True
        self.enabled = True
        self.captures = {'prefill':0, 'decode':0}
        self.replays = {'prefill':0, 'decode':0}
        self.max_hidden_diff = 0.
        self.cache_checked = 0
        self.head_original = model.compute_logits
        self.head_entry = None
        self.head_captures = self.head_replays = 0
        self.share_rotary = (self.fast_metadata
                             and not getattr(model, 'hf_reference', False)
                             and getattr(model.config, 'delta_vision_share_rotary', True))
        self._rotary_tables = None
        self.fused_qk = self.share_rotary and getattr(model.config, 'delta_vision_fused_qk', True)
        self.flat_inputs = self.fast_metadata and getattr(model.config, 'delta_vision_flat_inputs', True)
        self.attention_splits = getattr(model.config, 'delta_vision_attention_splits', 0)
        self.reuse_indices = self.fast_metadata and getattr(model.config, 'delta_vision_reuse_indices', True)
        self._index_topology = None
        if self.share_rotary:
            self._install_shared_rotary()
            self._install_fused_qk()
        model.forward = self.forward
        model.compute_logits = self.logits

    def _install_fused_qk(self):
        from src.kernels import vllm_qk_norm_rope
        for layer in self.model.language_model.model.layers:
            attn = layer.self_attn
            original = attn.forward
            def fused(query, key, positions, query_indices=None, _attn=attn):
                return vllm_qk_norm_rope(_attn, query, key, positions, query_indices)
            attn._delta_fused_qk = fused
            attn._delta_fused_qk_enabled = lambda: self.enabled and self.fused_qk
            attn._delta_attention_splits = lambda: self.attention_splits
            def forward(positions, hidden_states, _attn=attn, _original=original, _fused=fused):
                if not self.enabled or not self.fused_qk or positions.ndim != 2:
                    return _original(positions, hidden_states)
                qkv, _ = _attn.qkv_proj(hidden_states)
                q, k, v = qkv.split([_attn.q_size, _attn.kv_size, _attn.kv_size], -1)
                q, k = _fused(q, k, positions)
                out = _attn.attn(q.flatten(1), k.flatten(1), v)
                return _attn.o_proj(out)[0]
            attn.forward = forward

    def _install_shared_rotary(self):
        """Reuse position lookup across layers while keeping native MRoPE math.

        Scope is one graph execution, never another request. Both base and adapter
        use this path, with the native eager implementation as the reference.
        """
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding, triton_mrope
        seen = set()
        for layer in self.model.language_model.model.layers:
            rope = layer.self_attn.rotary_emb
            if id(rope) in seen or type(rope) is not MRotaryEmbedding:
                continue
            seen.add(id(rope))
            original = rope.forward
            def forward(positions, query, key=None, offsets=None, _rope=rope, _original=original):
                tables = self._rotary_tables
                if tables is None or positions.ndim != 2 or key is None or offsets is not None:
                    return _original(positions, query, key, offsets)
                table_key = (id(_rope), id(positions), query.dtype)
                values = tables.get(table_key)
                if values is None:
                    cos, sin = _rope._match_cos_sin_cache_dtype(query)[positions].chunk(2, -1)
                    values = cos.contiguous(), sin.contiguous()
                    tables[table_key] = values
                q, k = triton_mrope(query, key, *values, _rope.mrope_section,
                                   _rope.head_size, _rope.rotary_dim, _rope.mrope_interleaved)
                return q.reshape(query.shape), k.reshape(key.shape)
            rope.forward = forward

    def logits(self, hidden, *args, **kwargs):
        if not self.enabled: return self.head_original(hidden,*args,**kwargs)
        key = signature((hidden,args,kwargs))
        if self.head_entry is None or self.head_entry[0] != key:
            assert self.allow_capture, 'Unwarmed LM head graph'
            static = map_tree((hidden,args,kwargs),lambda x:x.clone())
            def run(): return self.head_original(static[0],*static[1],**static[2])
            for _ in range(3): run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): output = run()
            self.head_entry = (key,static,graph,output)
            self.head_captures += 1
        _,static,graph,output = self.head_entry
        copy_inputs(static,(hidden,args,kwargs))
        graph.replay(); self.head_replays += 1
        return output

    def state_snapshots(self, metadata):
        saved = []
        for layer in self.model.language_model.model.layers:
            module = getattr(layer,'linear_attn',None)
            if module is None: continue
            md = metadata[module.prefix]
            idx = md.non_spec_state_indices_tensor.long().unique()
            assert bool((idx >= 0).all()), 'Invalid active recurrent slot'
            for state in module.kv_cache:
                saved.append((state,idx,state.index_select(0,idx).clone()))
        return saved

    @staticmethod
    def restore(saved):
        for state,idx,value in saved: state.index_copy_(0,idx,value)

    def forward(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context, override_forward_context
        context = get_forward_context()
        if not self.enabled or context.attn_metadata is None:
            self.model._graph_visual_idx = self.model._graph_text_idx = None
            self.model._graph_text_segments = None
            return self.original(*args,**kwargs)
        assert not args, 'Graph model expects named vLLM inputs'
        metadata = context.attn_metadata
        assert isinstance(metadata,dict), 'DBO is not supported'
        # Read small host metadata once per forward, not once per decoder layer.
        descriptions, seen = [], set()
        for md in metadata.values():
            if id(md) in seen: continue
            seen.add(id(md))
            if hasattr(md,'seq_lens'):
                if self.fast_metadata and md.seq_lens.numel() == 1:
                    md._graph_starts = (0, md.num_actual_tokens)
                    md._graph_lengths = (md.max_seq_len,)
                else:
                    md._graph_starts = tuple(md.query_start_loc.tolist())
                    md._graph_lengths = tuple(md.seq_lens.tolist())
                descriptions.append((md._graph_starts,md._graph_lengths))
        mask = getattr(self.model,'_adapter_mm_mask',None)
        host_mask = tuple(mask.tolist()) if mask is not None else None
        segments = None
        if host_mask is not None and descriptions:
            starts,lengths = descriptions[0]
            if all(b-a==n for a,b,n in zip(starts,starts[1:],lengths)):
                segments = []
                for lo,hi in zip(starts,starts[1:]):
                    i = lo
                    while i<hi:
                        if host_mask[i]: i+=1; continue
                        end=i+1
                        while end<hi and not host_mask[end]: end+=1
                        qs, qe = i, end
                        segments.append((i,end,lo,qe,qs)); i=end
                segments = tuple(segments)
        self.model._graph_text_segments = segments
        if mask is not None:
            # Dynamic nonzero belongs to preparation, never CUDA capture.
            device = kwargs['inputs_embeds'].device if kwargs.get('inputs_embeds') is not None else kwargs['input_ids'].device
            topology_key = (host_mask, device)
            cached = self._index_topology if self.reuse_indices else None
            if cached is None or cached[0] != topology_key:
                cached = (topology_key, mask.nonzero().flatten().to(device),
                          (~mask).nonzero().flatten().to(device))
                if self.reuse_indices: self._index_topology = cached
            self.model._graph_visual_idx, self.model._graph_text_idx = cached[1:]
        else:
            self.model._graph_visual_idx = self.model._graph_text_idx = None
        phases = [max(b-a for a,b in zip(s,s[1:])) for s,_ in descriptions]
        phase = 'prefill' if phases and max(phases)>1 else 'decode'
        inputs = (kwargs,metadata,context.slot_mapping,
                  getattr(self.model,'_graph_visual_idx',None),getattr(self.model,'_graph_text_idx',None))
        if self.flat_inputs:
            shape_key, input_tensors = inspect_inputs(inputs)
        else:
            shape_key = alias_signature(inputs) if self.fast_metadata else signature(inputs)
        key = (tuple(descriptions), host_mask, shape_key)
        entries = self.entries[phase]
        entry = entries.get(key)
        if entry is None:
            assert self.allow_capture, ('Unwarmed vLLM graph',phase)
            static = map_tree(inputs,lambda x:x.clone())
            static_context = copy.copy(context)
            static_context.attn_metadata, static_context.slot_mapping = static[1:3]
            if self.flat_inputs:
                static_context._copy_targets = inspect_inputs(static)[1]
            if self.fast_metadata and self.attention_splits:
                for md in static_context.attn_metadata.values():
                    if hasattr(md, 'max_num_splits'):
                        md.max_num_splits = self.attention_splits
            text_cu = tuple((torch.tensor([0,ke-qs],dtype=torch.int32,device='cuda'),
                             torch.tensor([ke-ks],dtype=torch.int32,device='cuda'))
                            for _,_,ks,ke,qs in (segments or ()))
            def run():
                self.model._graph_visual_idx, self.model._graph_text_idx = static[3:5]
                self.model._graph_text_segments = segments
                self.model._graph_text_cu = text_cu
                self.model._adapter_mm_mask = None
                call = dict(static[0])
                # vLLM's fused residual/RMSNorm updates the initial embedding
                # buffer in-place. Every replay must start from the original E.
                if call.get('inputs_embeds') is not None:
                    call['inputs_embeds'] = call['inputs_embeds'].clone()
                self._rotary_tables = {} if self.share_rotary else None
                try:
                    with override_forward_context(static_context):
                        return self.original(**call)
                finally:
                    self._rotary_tables = None
            saved = self.state_snapshots(metadata)
            try:
                reference = run().clone()
                final_states = [(s,idx,s.index_select(0,idx).clone()) for s,idx,_ in saved]
                self.restore(saved)
                for _ in range(2): run(); self.restore(saved)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): output = run()
                self.restore(saved)
                graph.replay()
                # Fail closed: compare actual model output and all active GDN states.
                diff = (output-reference).abs().max().item()
                self.max_hidden_diff = max(self.max_hidden_diff,diff)
                assert torch.equal(output,reference), (phase,'Graph/eager hidden mismatch',diff)
                for state,idx,value in final_states:
                    assert torch.equal(state.index_select(0,idx),value), 'Graph/eager recurrent cache mismatch'
                self.cache_checked += len(final_states)
            except BaseException:
                self.restore(saved)
                raise
            self.captures[phase] += 1
            entry = (static,static_context,run,graph,output)
            entries[key] = entry
            capacity = self.prefill_capacity if phase=='prefill' else self.decode_capacity
            while len(entries)>capacity: entries.popitem(last=False)
        else:
            static,static_context,_,graph,output = entry
            if self.flat_inputs:
                targets, cpu_targets = static_context._copy_targets
                sources, cpu_sources = input_tensors
                if targets: torch._foreach_copy_(targets, sources)
                for target, source in zip(cpu_targets, cpu_sources): target.copy_(source)
            else:
                copy_inputs(static,inputs)
            graph.replay()
        entries.move_to_end(key)
        self.model._adapter_mm_mask = None
        self.replays[phase] += 1
        return entry[-1]

    def stats(self):
        return dict(captures=dict(self.captures), replays=dict(self.replays),
            head_captures=self.head_captures, head_replays=self.head_replays,
            max_hidden_diff=self.max_hidden_diff, recurrent_cache_checks=self.cache_checked)


def install_runtime_graphs(model):
    if hasattr(model,'_runtime_graphs'): raise RuntimeError('Graphs already installed')
    model._runtime_graphs = RuntimeGraphs(model)
    return model._runtime_graphs.stats()


def graph_stats(model):
    return model._runtime_graphs.stats()


def capture_on(model):
    model._runtime_graphs.allow_capture = True
    return True


def capture_off(model):
    model._runtime_graphs.allow_capture = False
    return model._runtime_graphs.stats()
