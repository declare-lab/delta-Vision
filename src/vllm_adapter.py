"""Qwen3-VL / Qwen3.5 visual adapters for vLLM 0.23.

src.vllm_graphs installs bounded prefill/decode CUDA graphs over vLLM caches.
Qwen3-VL executes attention and the output projection on text queries only. Both adapters run the decoder FFN only on text rows. Qwen3.5 keeps
visual GDN writes and causal convolution, including per-request cache state.
"""
from pathlib import Path
import argparse
import hashlib
import json
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def register():
    from vllm import ModelRegistry
    for name in ARCHITECTURES.values():
        ModelRegistry.register_model(name, f'src.vllm_adapter:{name}')


ARCHITECTURES = {'qwen3_vl': 'DeltaVisionQwen3VLForConditionalGeneration',
                 'qwen3_5': 'DeltaVisionQwen35ForConditionalGeneration'}
MODES = ('embedding_adapter', 'recurrent_embedding_adapter')


def export(base, checkpoint, output):
    """Export a local HF-style directory, linking rather than copying base weights."""
    from safetensors.torch import save_file
    base, checkpoint, output = map(lambda p: Path(p).resolve(), (base, checkpoint, output))
    config = json.loads((base / 'config.json').read_text())
    family = config['model_type']
    if family not in ARCHITECTURES:
        raise ValueError('This port supports dense Qwen3-VL and Qwen3.5 only')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args, adapter_config = saved.get('args', {}), saved.get('adapter_config', {})
    mode = adapter_config.get('output_mode', args.get('output_mode'))
    if family == 'qwen3_5':
        architecture = saved.get('config', {}).get('architecture')
        mode = {'static_embedding_adapter': 'embedding_adapter',
                'recurrent_embedding_adapter': 'recurrent_embedding_adapter'}.get(architecture, mode)
    if mode not in MODES:
        raise ValueError(f'Unsupported adapter mode {mode!r}')
    state = saved['state_dict']
    prefix = '' if family == 'qwen3_5' else 'visual_adapter_'
    rank = state[f'{prefix}down.0.weight'].shape[0]
    hidden, layers = config['text_config']['hidden_size'], config['text_config']['num_hidden_layers']
    weights = {}
    expected = set()
    for i in range(layers):
        for name, shape in [('down', (rank, hidden)), ('up', (hidden, rank))]:
            key = f'{prefix}{name}.{i}.weight'
            expected.add(key)
            if tuple(state[key].shape) != shape:
                raise ValueError(f'{key}: expected {shape}, got {state[key].shape}')
            weights[f'{name}.{i}.weight'] = state[key].contiguous()
    if set(state) != expected:
        raise ValueError(f'Unexpected checkpoint tensors: {set(state) - expected}')
    output.mkdir(parents=True, exist_ok=False)
    for path in base.iterdir():
        if path.is_file() and path.name != 'config.json':
            (output / path.name).symlink_to(path)
    config['architectures'] = [ARCHITECTURES[family]]
    config['vision_config']['deepstack_visual_indexes'] = []
    with checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    config['delta_vision_adapter'] = dict(mode=mode, rank=rank,
        checkpoint_sha256=checkpoint_sha,
        source_checkpoint=str(checkpoint), base_model=str(base),
        implementation='vllm-0.23-visual-adapter-runtime-graphs')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    # Deliberately not *.safetensors: the base-model loader must not consume this file.
    save_file(weights, str(output / 'adapter.weights'))
    return output


class VisualAdapter(nn.Module):
    def __init__(self, hidden, layers, rank):
        super().__init__()
        self.down = nn.ModuleList(nn.Linear(hidden, rank, bias=False) for _ in range(layers))
        self.up = nn.ModuleList(nn.Linear(rank, hidden, bias=False) for _ in range(layers))
        self.register_buffer('_down_stacked', None, persistent=False)
        self.register_buffer('_up_stacked', None, persistent=False)

    def forward(self, embedding, layer):
        return embedding + self.up[layer](F.silu(self.down[layer](embedding)))

    def all_memories(self, embedding):
        """Frozen static adapter: evaluate all independent layers with two BMMs."""
        if self._down_stacked is None:
            self._down_stacked = torch.stack([m.weight.detach() for m in self.down])
            self._up_stacked = torch.stack([m.weight.detach() for m in self.up])
        x = embedding.unsqueeze(0).expand(len(self.down), -1, -1)
        x = F.silu(torch.bmm(x, self._down_stacked.transpose(1, 2)))
        return embedding.unsqueeze(0) + torch.bmm(x, self._up_stacked.transpose(1, 2))


def _hf_gemma_norm(module, x, residual=None):
    """Preserve Transformers' BF16 residual boundary before FP32 normalization."""
    if residual is not None:
        x = x + residual
        residual = x
    f = x.float()
    f = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + module.variance_epsilon)
    out = (f * (1.0 + module.weight.float())).to(x.dtype)
    return out if residual is None else (out, residual)


def _hf_gated_norm(module, x, z):
    """Qwen3.5 rounds normalized values and weighted values before FP32 SiLU."""
    f = x.float()
    f = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + module.eps)
    f = module.weight * f.to(x.dtype)
    return (f * F.silu(z.float())).to(x.dtype)


def _hf_delta_core(module, mixed_qkv, b, a, core_attn_out):
    """Use the training FLA arithmetic with vLLM-owned per-request caches.

    vLLM's fused gate/core keeps beta in FP32 and uses different chunk/decode
    kernels. HF rounds sigmoid(beta) to the activation dtype. Match both phases,
    including mixed decode/prefill batches, without changing cache ownership.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        core_attn_out.zero_()
        return
    md = metadata[module.prefix]
    if md.spec_sequence_masks is not None:
        raise ValueError('Speculative GDN state is unsupported')
    count = md.num_actual_tokens
    x = mixed_qkv[:count]
    conv_state, ssm_state = module.kv_cache
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    # The vLLM Triton convolution otherwise rounds each BF16 product before
    # adding it to the FP32 accumulator; causal-conv1d CUDA multiplies in FP32.
    # Promoting the already BF16-rounded weights preserves their exact values.
    weight = module.conv1d.weight.squeeze(1).float()
    slots = md.non_spec_state_indices_tensor
    if md.num_prefills:
        x = causal_conv1d_fn(x.transpose(0, 1), weight, module.conv1d.bias,
            activation=module.activation, conv_states=conv_state,
            has_initial_state=md.has_initial_state, cache_indices=slots,
            query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
    else:
        x = causal_conv1d_update(x, conv_state, weight, module.conv1d.bias,
            module.activation, conv_state_indices=slots[:count], validate_data=True)
    q, k, v = x.split([module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.reshape(1, count, module.num_k_heads, module.head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(1, count, module.num_v_heads, module.head_v_dim)
    repeat = module.num_v_heads // module.num_k_heads
    q, k = q.repeat_interleave(repeat, 2), k.repeat_interleave(repeat, 2)
    beta = b[:count].sigmoid().unsqueeze(0)
    g = (-module.A_log.float().exp() * F.softplus(a[:count].float() + module.dt_bias)).unsqueeze(0)
    nd = md.num_decode_tokens
    # vLLM stores [V,K], whereas the training FLA default stores [K,V].
    def run(start, end, indices, cu, prefill):
        state = ssm_state[indices].transpose(-1, -2).contiguous()
        if prefill:
            state = state.masked_fill(~md.prefill_has_initial_state[:,None,None,None],0)
        kernel = chunk_gated_delta_rule if prefill else fused_recurrent_gated_delta_rule
        out, state = kernel(q[:,start:end], k[:,start:end], v[:,start:end],
            g=g[:,start:end], beta=beta[:,start:end], initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        ssm_state[indices] = state.transpose(-1, -2).to(ssm_state.dtype)
        core_attn_out[start:end].copy_(out[0])
    if md.num_decodes:
        run(0, nd, slots[:md.num_decodes], md.non_spec_query_start_loc[:md.num_decodes+1], False)
    if md.num_prefills:
        run(nd, count, md.prefill_state_indices, md.prefill_query_start_loc, True)


def _hf_flash_forward(impl, layer, query, key, value, kv_cache, attn_metadata,
                      output, output_scale=None, output_block_scale=None):
    """Strict HF arithmetic; retain vLLM cache writes and request isolation.

    FA2's fixed-length and varlen kernels differ numerically even for a single
    unpadded sequence. This compatibility path uses the reference fixed-length
    kernel per request. Decode gathers paged KV, so this is not a speed port.
    """
    from flash_attn import flash_attn_func
    if attn_metadata is None:
        return output.zero_()
    if output_scale is not None or output_block_scale is not None:
        raise ValueError('Quantized attention output is unsupported')
    md = attn_metadata
    starts = getattr(md, '_graph_starts', None)
    lengths = getattr(md, '_graph_lengths', None)
    if starts is None:
        starts, lengths = md.query_start_loc.tolist(), md.seq_lens.tolist()
    # Attention's do_kv_cache_update has already written this step's K/V.
    kc, vc = kv_cache.unbind(1)
    block_size = kc.shape[1]
    for i, length in enumerate(lengths):
        start, end = starts[i:i+2]
        if start == end:
            continue
        if end - start == length:
            k, v = key[start:end], value[start:end]
        else:
            blocks = md.block_table[i, :(length + block_size - 1) // block_size].long()
            k = kc.index_select(0, blocks).flatten(0, 1)[:length]
            v = vc.index_select(0, blocks).flatten(0, 1)[:length]
        out = flash_attn_func(query[start:end].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            softmax_scale=impl.scale, causal=md.causal,
            window_size=tuple(impl.sliding_window) if impl.sliding_window else (-1, -1),
            softcap=impl.logits_soft_cap)
        output[start:end].copy_(out[0])
    return output


def _qwen_text_attention_full_q(attn, hidden, positions, segments, cumulative, text_indices):
    """Text-only query/output projection with native paged K/V writes.

    Each text span uses its causal prefix, including prefix text before images.
    The shorter GEMMs can change BF16 rounding relative to full-row attention.
    """
    from vllm.forward_context import get_forward_context
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    qkv, _ = attn.qkv_proj(hidden)
    q,k,v = qkv.split([attn.q_size,attn.kv_size,attn.kv_size],dim=-1)
    q = attn.q_norm(q.view(-1,attn.num_heads,attn.head_dim)).view_as(q)
    k = attn.k_norm(k.view(-1,attn.num_kv_heads,attn.head_dim)).view_as(k)
    q,k = attn.rotary_emb(positions,q,k)
    q = q.view(-1,attn.num_heads,attn.head_dim)
    k = k.view(-1,attn.num_kv_heads,attn.head_dim)
    v = v.view_as(k)
    attention = attn.attn
    context = get_forward_context()
    attention.impl.do_kv_cache_update(attention,k,v,attention.kv_cache,
        context.slot_mapping[attention.layer_name])
    md = context.attn_metadata[attention.layer_name]
    kc,vc = attention.kv_cache.unbind(1)
    outputs = []
    for (ts,te,ks,ke,qs),(cuq,used) in zip(segments,cumulative):
        # Use native vLLM FA2 arithmetic, matching the reference paged backend.
        req = md._graph_starts.index(ks)
        out = flash_attn_varlen_func(q[qs:ke],kc,vc,
            cu_seqlens_q=cuq,seqused_k=used,block_table=md.block_table[req:req+1],
            max_seqlen_q=ke-qs,max_seqlen_k=ke-ks,
            softmax_scale=attn.scaling,causal=True,fa_version=2)
        outputs.append(out[ts-qs:te-qs].reshape(te-ts,-1))
    compact = torch.cat(outputs,dim=0)
    projected,_ = attn.o_proj(compact)
    return projected


def _qwen_text_attention(attn, hidden, positions, segments, cumulative, text_indices, rotary_cache=None):
    weight, bias = attn.qkv_proj.weight, attn.qkv_proj.bias
    q = F.linear(hidden.index_select(0, text_indices), weight[:attn.q_size],
                 bias[:attn.q_size] if bias is not None else None)
    kv = F.linear(hidden, weight[attn.q_size:],
                  bias[attn.q_size:] if bias is not None else None)
    return _qwen_projected_text_attention(attn, q, kv, positions, segments, cumulative, text_indices, rotary_cache)


def _qwen_projected_text_attention(attn, q, kv, positions, segments, cumulative, text_indices, rotary_cache=None):
    from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
    k, v = kv.split(attn.kv_size, dim=-1)
    if getattr(attn, '_delta_fused_qk_enabled', lambda: False)():
        q, k = attn._delta_fused_qk(q, k, positions, text_indices)
        return _qwen_paged_text_output(attn, q, k, v.reshape_as(k), segments, cumulative)
    q = attn.q_norm(q.view(-1, attn.num_heads, attn.head_dim))
    k = attn.k_norm(k.reshape(-1, attn.num_kv_heads, attn.head_dim))
    rope = attn.rotary_emb
    assert rope.rotary_dim == attn.head_dim
    cached = rotary_cache.get(id(rope)) if rotary_cache is not None else None
    if cached is None:
        cos, sin = rope.cos_sin_cache[positions].chunk(2, dim=-1)
        if positions.ndim == 2:
            if rope.mrope_interleaved:
                cos, sin = (apply_interleaved_rope(x, rope.mrope_section) for x in (cos, sin))
            else:
                cos, sin = (torch.cat([part[i] for i, part in enumerate(x.split(rope.mrope_section, -1))], -1)
                            for x in (cos, sin))
        cached = cos, sin, cos.index_select(0, text_indices), sin.index_select(0, text_indices)
        if rotary_cache is not None:
            rotary_cache[id(rope)] = cached
    cos, sin, text_cos, text_sin = cached
    q = rope.apply_rotary_emb(q, text_cos, text_sin)
    k = rope.apply_rotary_emb(k, cos, sin)
    v = v.reshape_as(k)
    return _qwen_paged_text_output(attn, q, k, v, segments, cumulative)


def _qwen_paged_text_output(attn, q, k, v, segments, cumulative):
    from vllm.forward_context import get_forward_context
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    attention = attn.attn
    context = get_forward_context()
    attention.impl.do_kv_cache_update(attention, k, v, attention.kv_cache,
                                    context.slot_mapping[attention.layer_name])
    md = context.attn_metadata[attention.layer_name]
    kc, vc = attention.kv_cache.unbind(1)
    outputs, offset = [], 0
    for (ts, te, ks, ke, qs), (cuq, used) in zip(segments, cumulative):
        assert (qs, ke) == (ts, te)
        count = te-ts
        request = md._graph_starts.index(ks)
        out = flash_attn_varlen_func(q[offset:offset+count], kc, vc,
            cu_seqlens_q=cuq, seqused_k=used, block_table=md.block_table[request:request+1],
            max_seqlen_q=count, max_seqlen_k=ke-ks,
            softmax_scale=attn.scaling, causal=True, fa_version=2,
            num_splits=getattr(attn, '_delta_attention_splits', lambda: 0)())
        outputs.append(out.reshape(count, -1))
        offset += count
    projected, _ = attn.o_proj(torch.cat(outputs, 0))
    return projected



def __getattr__(name):
    # Export and CPU validation do not import vLLM or initialize any CUDA state.
    if name not in ARCHITECTURES.values():
        raise AttributeError(name)
    from safetensors.torch import load_file
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration, Qwen3VLMultiModalProcessor,
        Qwen3VLProcessingInfo, Qwen3VLDummyInputsBuilder)
    hybrid = name == ARCHITECTURES['qwen3_5']
    parent, info = Qwen3VLForConditionalGeneration, Qwen3VLProcessingInfo
    if hybrid:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo)
        parent, info = Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor, info=info,
        dummy_inputs=Qwen3VLDummyInputsBuilder)
    class DeltaVisionModel(parent):
        supports_lora = False
        supports_pp = False

        def __init__(self, *, vllm_config, prefix='model'):
            p = vllm_config.parallel_config
            if p.tensor_parallel_size != 1 or p.pipeline_parallel_size != 1:
                raise ValueError('Initial adapter port requires TP=PP=1')
            if (not vllm_config.model_config.enforce_eager and
                    vllm_config.compilation_config.mode != 0):
                raise ValueError('Use shape-specific adapter graphs with compilation_config.mode=0')
            if vllm_config.scheduler_config.async_scheduling:
                raise ValueError('Initial adapter port requires async_scheduling=False')
            if vllm_config.scheduler_config.enable_chunked_prefill:
                raise ValueError('Chunked prefill is not yet validated; disable it')
            if vllm_config.cache_config.enable_prefix_caching:
                raise ValueError('Prefix caching is not yet validated; disable it')
            if vllm_config.quant_config is not None:
                raise ValueError('Initial adapter port requires unquantized weights')
            if vllm_config.speculative_config is not None:
                raise ValueError('Speculative decoding is not yet validated')
            if hybrid and vllm_config.cache_config.mamba_ssm_cache_dtype != 'float32':
                raise ValueError('Qwen3.5 HF reference keeps recurrent state in FP32; '
                                 'set mamba_ssm_cache_dtype="float32"')
            c = vllm_config.model_config.hf_config
            if c.vision_config.deepstack_visual_indexes:
                raise ValueError('Export with DeepStack disabled first')
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.use_deepstack = False
            # Native vLLM kernels are the speed path. Exact HF arithmetic is
            # available only as an explicit numerical diagnostic reference.
            self.hf_reference = bool(getattr(c, 'delta_vision_hf_reference', False))
            self.fast_prefill = not hybrid and not self.hf_reference and bool(getattr(c, 'delta_vision_fast_prefill', True))
            self.batch_memories = self.fast_prefill and bool(getattr(c, 'delta_vision_batch_memories', True))
            self.reuse_prefill_buffers = self.fast_prefill and bool(getattr(c, 'delta_vision_reuse_prefill_buffers', True))
            self.batch_visual_kv = self.fast_prefill and bool(getattr(c, 'delta_vision_batch_visual_kv', True))
            self.register_buffer('_batched_visual_kv_weights', None, persistent=False)
            self.register_buffer('_batched_visual_norm_weights', None, persistent=False)
            if self.hf_reference:
                # vLLM 0.23 rounds interpolation weights to BF16 before multiplying;
                # our Transformers 5.15 reference keeps them FP32 through reduction.
                # This changes the actual visual embedding, not just decoder kernels.
                self.visual.fast_pos_embed_interpolate = self._vision_positions
                # Match the reference's FP32 vision sin/cos and Conv3d patch projection.
                # The stock backend caches vision sin/cos in BF16 and substitutes GEMM
                # for Conv3d, both of which change the initial adapter input numerically.
                rope = self.visual.rotary_pos_emb
                rope.cos_sin_cache = rope._compute_cos_sin_cache().to(
                    device=self.visual.device, dtype=torch.float32)
                self.visual.patch_embed.proj.enable_linear = False
                for block in self.visual.blocks:
                    block.attn.apply_rotary_emb.enable_fp32_compute = True
                    if hybrid:
                        # The fused rotary kernel's FP32 FMA changes BF16 rounding.
                        block.attn.apply_rotary_emb.forward = block.attn.apply_rotary_emb.forward_native
                if hybrid:
                    # Same GPU kernels as adapter training; fail closed if absent.
                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                    from flash_attn import flash_attn_func
                    decoder = self.language_model.model
                    decoder.norm.forward = MethodType(_hf_gemma_norm, decoder.norm)
                    for layer in decoder.layers:
                        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                            norm.forward = MethodType(_hf_gemma_norm, norm)
                        layer.mlp.act_fn.forward = layer.mlp.act_fn.forward_native
                        if layer.layer_type == 'linear_attention':
                            layer.linear_attn._forward_core = MethodType(_hf_delta_core, layer.linear_attn)
                            norm = layer.linear_attn.norm
                            norm.forward = MethodType(_hf_gated_norm, norm)
                        else:
                            impl = layer.self_attn.attn.impl
                            if getattr(impl, 'vllm_flash_attn_version', None) != 2:
                                raise ValueError('Qwen3.5 adapter compatibility requires FA2')
                            impl.forward = MethodType(_hf_flash_forward, impl)
                            for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
                                norm.forward = MethodType(_hf_gemma_norm, norm)
                            rope = layer.self_attn.rotary_emb
                            rope.forward = rope.forward_native
            spec = c.delta_vision_adapter
            if spec['mode'] not in MODES:
                raise ValueError(f"Unsupported adapter mode {spec['mode']!r}")
            self.adapter_mode = spec['mode']
            if getattr(c.text_config, 'layer_scale', False):
                raise ValueError('Layer-scale variants have not been validated')
            self.visual_adapter = VisualAdapter(c.text_config.hidden_size,
                c.text_config.num_hidden_layers, spec['rank'])
            self.visual_adapter.load_state_dict(load_file(str(
                Path(vllm_config.model_config.model) / 'adapter.weights')), strict=True)
            self._adapter_mm_mask = None

        def _vision_positions(self, grid_thw):
            from transformers.vision_utils import get_vision_interpolation_indices_and_weights
            grid = torch.as_tensor(grid_thw, device=self.visual.device)
            indices, weights = get_vision_interpolation_indices_and_weights(
                grid, self.visual.num_grid_per_side, mode='bilinear', align_corners=True,
                spatial_merge_size=self.config.vision_config.spatial_merge_size)
            return (self.visual.pos_embed(indices) * weights[:, :, None]).sum(1).to(self.visual.dtype)

        def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
            # vLLM supplies inputs_embeds even for ordinary decode. An empty
            # multimodal list is authoritative host metadata: no GPU mask scan,
            # nonzero, or device-to-host synchronization is needed in that case.
            if hybrid:
                self._adapter_mm_mask = (is_multimodal.clone() if is_multimodal is not None
                                         else torch.zeros_like(input_ids, dtype=torch.bool))
            else:
                has_visual = multimodal_embeddings is not None and len(multimodal_embeddings) > 0
                self._adapter_mm_mask = (is_multimodal.clone() if has_visual and is_multimodal is not None
                                         else None)
                if has_visual and is_multimodal is None:
                    raise ValueError('Visual embeddings require the vLLM multimodal token mask')
            return super().embed_input_ids(input_ids, multimodal_embeddings,
                                            is_multimodal=is_multimodal)

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
            if intermediate_tensors is not None:
                raise ValueError('Pipeline intermediate states are unsupported')
            decoder = self.language_model.model
            mask, self._adapter_mm_mask = self._adapter_mm_mask, None
            graph_vi = getattr(self, '_graph_visual_idx', None)
            if inputs_embeds is None:
                if mask is not None and bool(mask.any()):
                    raise RuntimeError('Visual mask without image embeddings')
                # Decode (and ordinary text-only prefill) uses native cache machinery.
                return super().forward(input_ids, positions, inputs_embeds=None, **kwargs)
            if not hybrid and ((mask is None and graph_vi is None)
                               or (graph_vi is not None and graph_vi.numel() == 0)):
                return super().forward(input_ids, positions, inputs_embeds=inputs_embeds, **kwargs)
            h = inputs_embeds
            if graph_vi is not None:
                vi, ti = graph_vi, self._graph_text_idx
            else:
                vi = ti = None
            if mask is None:
                # vLLM profile/warm-up forwards supply synthetic inputs_embeds.
                mask = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
            if mask.numel() > h.shape[0]:
                raise RuntimeError('Multimodal mask exceeds the current token batch')
            if mask.numel() < h.shape[0]:
                mask = F.pad(mask, (0, h.shape[0] - mask.numel()), value=False)
            mask = mask.to(device=h.device, dtype=torch.bool)
            if vi is None:
                vi = mask.nonzero().flatten()
            if vi.numel() == 0:
                return super().forward(input_ids, positions, inputs_embeds=h, **kwargs)
            if ti is None:
                ti = (~mask).nonzero().flatten()
            embedding = h.index_select(0, vi)
            compact_prefill = (self.fast_prefill and self.reuse_prefill_buffers
                               and getattr(self, '_graph_text_segments', None) is not None)
            # Own the residual buffer: input embeddings may be reused by callers.
            # Visual rows are replaced each layer, so copying every unchanged row
            # twice per layer is unnecessary. Rotary tensors are request-local.
            if compact_prefill:
                h = h.clone()
            rotary_cache = {} if compact_prefill else None
            memories = (self.visual_adapter.all_memories(embedding)
                        if self.batch_memories and self.adapter_mode == 'embedding_adapter' else None)
            if compact_prefill and self.batch_visual_kv and memories is not None:
                return self._compact_visual_prefill(h, memories, positions, vi, ti)
            for i, layer in enumerate(decoder.layers):
                prediction = memories[i] if memories is not None else self.visual_adapter(embedding, i)
                if self.adapter_mode == 'recurrent_embedding_adapter':
                    embedding = prediction
                h = h.index_copy_(0, vi, prediction) if compact_prefill else h.index_copy(0, vi, prediction)
                normalized = layer.input_layernorm(h)
                if hybrid:
                    # Keep native visual Q/K/V, gates, causal convolution, and GDN
                    # writes. vLLM owns per-request convolution/recurrent caches.
                    mixed = torch.empty_like(normalized)
                    if layer.layer_type == 'linear_attention':
                        layer.linear_attn(hidden_states=normalized, output=mixed)
                    else:
                        layer.self_attn(hidden_states=normalized, output=mixed, positions=positions)
                elif getattr(self,'_graph_text_segments',None) is not None:
                    attention_fn = _qwen_text_attention if self.fast_prefill else _qwen_text_attention_full_q
                    extra = {'rotary_cache': rotary_cache} if self.fast_prefill else {}
                    mixed = attention_fn(layer.self_attn,normalized,positions,self._graph_text_segments,self._graph_text_cu,ti,**extra)
                    text = h.index_select(0,ti) + mixed
                    text = text + layer.mlp(layer.post_attention_layernorm(text))
                    h = h.index_copy_(0,ti,text) if compact_prefill else h.index_copy(0,ti,text)
                    continue
                else:
                    mixed = layer.self_attn(positions, normalized)
                text = h.index_select(0, ti) + mixed.index_select(0, ti)
                if ti.numel():
                    text = text + layer.mlp(layer.post_attention_layernorm(text))
                    h = h.index_copy(0, ti, text)
            return decoder.norm(h)

        def _compact_visual_prefill(self, h, memories, positions, vi, ti):
            from src.kernels import layerwise_rmsnorm
            layers = self.language_model.model.layers
            if self._batched_visual_kv_weights is None:
                assert all(layer.self_attn.qkv_proj.bias is None for layer in layers)
                assert len({layer.input_layernorm.variance_epsilon for layer in layers}) == 1
                self._batched_visual_kv_weights = torch.stack([
                    layer.self_attn.qkv_proj.weight.detach()[layer.self_attn.q_size:] for layer in layers])
                self._batched_visual_norm_weights = torch.stack([
                    layer.input_layernorm.weight.detach() for layer in layers])
            normalized = layerwise_rmsnorm(memories, self._batched_visual_norm_weights,
                                           layers[0].input_layernorm.variance_epsilon)
            visual_kv = torch.bmm(normalized, self._batched_visual_kv_weights.transpose(1, 2))
            text = h.index_select(0, ti)
            rotary_cache = {}
            for i, layer in enumerate(layers):
                attn = layer.self_attn
                qkv, _ = attn.qkv_proj(layer.input_layernorm(text))
                q, text_kv = qkv.split([attn.q_size, 2 * attn.kv_size], -1)
                kv = torch.empty((h.shape[0], 2 * attn.kv_size), device=h.device, dtype=h.dtype)
                kv.index_copy_(0, vi, visual_kv[i])
                kv.index_copy_(0, ti, text_kv)
                mixed = _qwen_projected_text_attention(attn, q, kv, positions,
                    self._graph_text_segments, self._graph_text_cu, ti, rotary_cache)
                text = text + mixed
                text = text + layer.mlp(layer.post_attention_layernorm(text))
            h.index_copy_(0, vi, memories[-1])
            h.index_copy_(0, ti, text)
            return self.language_model.model.norm(h)

        def load_weights(self, weights):
            # Disabled DeepStack parameters remain in the base checkpoint.
            weights = ((n, w) for n, w in weights if not n.startswith(
                ('model.visual.deepstack_merger_list.', 'visual.deepstack_merger_list.')))
            loaded = super().load_weights(weights)
            return loaded | {f'visual_adapter.{n}' for n, _ in self.visual_adapter.named_parameters()}

    DeltaVisionModel.__name__ = DeltaVisionModel.__qualname__ = name
    globals()[name] = DeltaVisionModel
    return DeltaVisionModel


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(export(a.base, a.checkpoint, a.output))
