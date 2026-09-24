"""Controlled Qwen3.5 visual recurrent-memory experiments (inference only).

No-visual-state means identity recurrent transitions over the visual span, NOT
beta=0 with forgetting still enabled. Native convolution is held fixed in this
state-only experiment. It is a distinct visual path and is audited separately.
"""
from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F

RANKS = (0, 8, 16, 32, 64, 128)


def matrix_similarity(pred, ref):
    p, r = pred.float(), ref.float()
    dims = tuple(range(1, r.ndim))
    if not dims:
        dims = (0,)
    dot = (p*r).sum(dims)
    pn, rn = p.square().sum(dims), r.square().sum(dims)
    valid = (pn > 1e-20) & (rn > 1e-20)
    cosine = dot/(pn*rn).sqrt().clamp_min(1e-30)
    error = (p-r).square().sum(dims)/rn.clamp_min(1e-30)
    return dict(cosine=cosine[valid].mean().item() if valid.any() else None,
                normalized_mse=error[rn > 1e-20].mean().item() if (rn > 1e-20).any() else None,
                normalized_frobenius=error[rn > 1e-20].sqrt().mean().item() if (rn > 1e-20).any() else None,
                zero_reference_count=int((rn <= 1e-20).sum()))


def spectrum_metrics(s, full_rank):
    s = s.float()
    if s.ndim == 1: s = s[None]
    energy = s.square()
    totals = energy.sum(-1, keepdim=True)
    cdf = energy.cumsum(-1)/totals.clamp_min(1e-30)
    p = s/s.sum(-1, keepdim=True).clamp_min(1e-30)
    er = (-(p*p.clamp_min(1e-30).log()).sum(-1)).exp()
    nonzero = totals[:, 0] > 1e-20
    result = {'full_rank': full_rank, 'zero_matrices': int((~nonzero).sum())}
    for name, fraction in [('r90', .9), ('r95', .95)]:
        ranks = torch.where(nonzero, (cdf < fraction).sum(-1)+1, 0).clamp_max(s.shape[-1])
        result[name] = ranks.float().mean().item()
        result[name+'_per_head'] = ranks.tolist()
        result[name+'_fraction'] = result[name]/full_rank
    effective=torch.where(nonzero, er, 0)
    result['effective_rank'] = effective.mean().item()
    result['effective_rank_per_head'] = effective.tolist()
    result['energy_by_rank'] = {str(r): (cdf[:, min(r, s.shape[-1])-1].mean().item() if r else 0.) for r in RANKS}
    return result


def decompose(state):
    # Each value head has its own key x value matrix; never flatten heads together.
    # LAPACK on these small matrices is faster than cuSOLVER's serial QR path.
    # Solve in FP64 (no randomized/truncated approximation), then restore FP32
    # factors on the state device for native FP32 recurrent interventions.
    factors=torch.linalg.svd(state.detach().to(device='cpu',dtype=torch.float64),full_matrices=False)
    return tuple(x.to(device=state.device,dtype=torch.float32) for x in factors)


def truncate(svd, rank):
    u, s, vh = svd
    return (u[..., :rank]*s[..., None, :rank]) @ vh[..., :rank, :]


def subspace_overlap(a, b, ranks=(8, 16, 32, 64)):
    result = {}
    for r in ranks:
        def overlap(x, y):
            return ((x[..., :r].transpose(-1, -2) @ y[..., :r]).square().sum((-1, -2))/r).mean().item()
        result[str(r)] = {'left': overlap(a[0], b[0]),
                          'right': overlap(a[2].transpose(-1,-2), b[2].transpose(-1,-2)),
                          'teacher_gap_ratio': (a[1][..., r-1]/a[1][..., r].clamp_min(1e-20)).mean().item()}
    return result


def next_token_kl(student, teacher):
    lp, lq = teacher.float().log_softmax(-1), student.float().log_softmax(-1)
    return (lp.exp()*(lp-lq)).sum(-1).clamp_min(0)


def span(mask):
    idx = mask[0].nonzero().flatten()
    assert mask.shape[0] == 1 and idx.numel() > 0
    start, end = int(idx[0]), int(idx[-1])+1
    assert idx.numel() == end-start, 'Only a single contiguous image span is supported'
    assert end < mask.shape[1], 'Need post-visual text for functional readout'
    return start, end


def project_linear(module, hidden, *, erase_visual_conv=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    batch, length, _ = hidden.shape
    raw = module.in_proj_qkv(hidden).transpose(1, 2)
    if erase_visual_conv is not None:
        start, end = erase_visual_conv
        raw = raw.clone(); raw[:, :, start:end] = 0
    conv = m.causal_conv1d_fn(raw, module.conv1d.weight.squeeze(1), module.conv1d.bias,
                            activation=module.activation).transpose(1, 2)
    q, k, v = torch.split(conv, [module.key_dim, module.key_dim, module.value_dim], -1)
    q=q.reshape(batch,length,module.num_k_heads,module.head_k_dim)
    k=k.reshape_as(q)
    v=v.reshape(batch,length,module.num_v_heads,module.head_v_dim)
    q=q.repeat_interleave(module.num_v_heads//module.num_k_heads,2)
    k=k.repeat_interleave(module.num_v_heads//module.num_k_heads,2)
    g=-module.A_log.float().exp()*F.softplus(module.in_proj_a(hidden).float()+module.dt_bias)
    beta=module.in_proj_b(hidden).sigmoid()
    z=module.in_proj_z(hidden).reshape(batch,length,-1,module.head_v_dim)
    return dict(q=q,k=k,v=v,g=g,beta=beta,z=z)


def kernel(parts, start=0, end=None, state=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    end = parts['q'].shape[1] if end is None else end
    if start == end:
        return parts['v'][:, :0], state
    return m.torch_chunk_gated_delta_rule(
        *(parts[k][:,start:end].contiguous() for k in ('q','k','v')),
        g=parts['g'][:,start:end].contiguous(), beta=parts['beta'][:,start:end].contiguous(),
        initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)


def finish_linear(module, core, z):
    batch, length = core.shape[:2]
    return module.out_proj(module.norm(core.reshape(-1,module.head_v_dim),
        z.reshape(-1,module.head_v_dim)).reshape(batch,length,-1))


def boundary_states(parts, start, end):
    _, sin = kernel(parts, 0, start)
    if sin is None:
        q,v = parts['q'],parts['v']
        sin=torch.zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3],device=q.device,dtype=torch.float32)
    # Preserve the native prefix chunk alignment. Starting a new chunk at the
    # first image token would create a different BF16 rounding trajectory.
    _, sout = kernel(parts,0,end)
    return sin, sout


def linear_state_output(module, parts, start, end, state, original_core=None):
    if original_core is None: original_core, _ = kernel(parts)
    _,sout=boundary_states(parts,start,end)
    suffix, final = kernel(parts,end,None,state)
    control, _ = kernel(parts,end,None,sout)
    # For fixed suffix inputs the recurrence is affine in its initial state.
    # Apply the paired response difference to the ORIGINAL native core output.
    # This cancels split-kernel roundoff at the full-restoration endpoint.
    corrected=(original_core[:,end:].float()+(suffix.float()-control.float())).to(original_core.dtype)
    combined=torch.cat((original_core[:,:end],corrected),dim=1)
    return finish_linear(module,combined,parts['z']), final


def text_effect_rank(effect):
    matrix=effect.float().reshape(-1,effect.shape[-1])
    s=torch.linalg.svdvals(matrix, driver='gesvd' if matrix.is_cuda else None)
    return spectrum_metrics(s, min(matrix.shape))


def mixer_output(layer, h, kwargs):
    norm=layer.input_layernorm(h)
    if layer.block_type=='linear_attention':
        return layer.linear_attn(norm, cache_params=None, attention_mask=None)
    return layer.self_attn(norm, position_embeddings=kwargs['position_embeddings'],
                          position_ids=kwargs.get('position_ids'),attention_mask=None,past_key_values=None)[0]


def layer_finish(layer, h, mixed, *, text_idx=None):
    if text_idx is None:
        out=h+mixed
        return out+layer.mlp(layer.post_attention_layernorm(out))
    text=h.index_select(1,text_idx)+mixed.index_select(1,text_idx)
    text=text+layer.mlp(layer.post_attention_layernorm(text))
    return h.index_copy(1,text_idx,text)


@torch.inference_mode()
def capture(model, controller, context, mask, method):
    traces=[]; handles=[]
    def before(layer, args, kwargs):
        h=args[0] if args else kwargs['hidden_states']
        if method=='adapter':
            h=h.index_copy(1,controller.visual_idx,controller.predictions[len(traces)].to(h.dtype))
        assert kwargs.get('attention_mask') is None, 'Probe expects unpadded batch1 FA2 inputs'
        traces.append({'hidden':h.detach(), 'kwargs':{k:v for k,v in kwargs.items() if k in ('position_embeddings','position_ids')}})
    for layer in model.model.language_model.layers:
        handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
    try:
        with controller.activate(method,mask):
            output=model.model.language_model(**context)
        logits=model.lm_head(output.last_hidden_state[:,-1]).float()
    finally:
        for handle in handles: handle.remove()
    return traces,logits


@torch.inference_mode()
def tail_logits(model, traces, index, hidden, predictions=None, mask=None):
    layers=model.model.language_model.layers
    for j in range(index+1,len(layers)):
        layer=layers[j]
        text_idx=None
        if predictions is not None:
            vi=mask[0].nonzero().flatten(); text_idx=(~mask[0]).nonzero().flatten()
            hidden=hidden.index_copy(1,vi,predictions[j].expand(hidden.shape[0],-1,-1).to(hidden.dtype))
        mixed=mixer_output(layer,hidden,traces[j]['kwargs'])
        hidden=layer_finish(layer,hidden,mixed,text_idx=text_idx)
    return model.lm_head(model.model.language_model.norm(hidden[:,-1])).float()


@contextmanager
def state_intervention(model, mask, *, rank=None, memory=None, layers=None):
    """Live/cascaded per-layer intervention; prefix and convolution stay native.

    A rank128 intervention is still computed, not bypassed. Generation decode
    consumes the modified prefill recurrent state through the native cache API.
    memory holds prenorm adapter predictions for a controlled state-only swap.
    """
    start,end=span(mask)
    originals=[]
    for i,layer in enumerate(model.model.language_model.layers):
        if layer.block_type!='linear_attention' or (layers is not None and i not in layers): continue
        module=layer.linear_attn; original=module.forward
        def wrapped(self,hidden_states,cache_params=None,attention_mask=None,_i=i,_layer=layer,_orig=original,**kwargs):
            if hidden_states.shape[1]!=mask.shape[1]:
                return _orig(hidden_states,cache_params=cache_params,attention_mask=attention_mask,**kwargs)
            assert attention_mask is None
            from transformers.models.qwen3_5 import modeling_qwen3_5 as m
            original_kernel=m.torch_chunk_gated_delta_rule
            def intervene(q,k,v,**kw):
                assert kw.get('initial_state') is None, 'Prefill must start with an empty recurrent cache'
                # Helpers use the original bound native FLA implementation.
                m.torch_chunk_gated_delta_rule=original_kernel
                try:
                    core, original_final = original_kernel(q,k,v,**{**kw,'output_final_state':True})
                    parts=dict(q=q,k=k,v=v,g=kw['g'],beta=kw['beta'])
                    sin,sout=boundary_states(parts,start,end)
                    if memory is None:
                        delta=sout-sin
                        target=sin if rank==0 else sout if rank==128 else sin+truncate(decompose(delta),rank)
                    else:
                        replaced=hidden_states.clone()
                        replaced[:,start:end]=_layer.input_layernorm(memory[_i].to(hidden_states.dtype))
                        alt=project_linear(self,replaced)
                        alt_in,alt_out=boundary_states(alt,start,end)
                        torch.testing.assert_close(alt_in,sin,rtol=0,atol=0)
                        target=alt_out
                    suffix, final=kernel(parts,end,None,target)
                    control, control_final=kernel(parts,end,None,sout)
                    corrected=(core[:,end:].float()+(suffix.float()-control.float())).to(core.dtype)
                    corrected_final=original_final+(final-control_final)
                    return torch.cat((core[:,:end],corrected),1), corrected_final
                finally:
                    m.torch_chunk_gated_delta_rule=intervene
            m.torch_chunk_gated_delta_rule=intervene
            try:
                return _orig(hidden_states,cache_params=cache_params,attention_mask=attention_mask,**kwargs)
            finally:
                m.torch_chunk_gated_delta_rule=original_kernel
        originals.append((module,original)); module.forward=types.MethodType(wrapped,module)
    try: yield
    finally:
        for module,original in originals: module.forward=original
