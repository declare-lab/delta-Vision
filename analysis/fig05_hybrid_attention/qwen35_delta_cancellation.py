"""GDN delta-residual diagnostics on native post-convolution Q/K/V and gates.

FP32 sequential replay measures the algebraic recurrence, not bitwise BF16
chunk intermediates. Native tensors are untouched in observation mode. A paired
replay difference implements the intervention while retaining native roundoff;
an independent replay-only control quantifies that numerical approximation.
"""
from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T', 'START', 'END'])
def _scan(Q, K0, V0, G, BETA, INITIAL, FINAL, OUT, STATS,
          T, START, END, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          BV: tl.constexpr, REMOVE: tl.constexpr):
    iv, ih = tl.program_id(0), tl.program_id(1)
    kk = tl.arange(0, K)
    vv = iv * BV + tl.arange(0, BV)
    offset = ih*K*V + kk[:, None]*V + vv[None, :]
    stride = H*K*V
    state = tl.load(INITIAL + offset).to(tl.float32)
    changed = state
    for t in tl.range(0, T):
        q = tl.load(Q + (t*H+ih)*K + kk).to(tl.float32) * (K ** -0.5)
        k = tl.load(K0 + (t*H+ih)*K + kk).to(tl.float32)
        v = tl.load(V0 + (t*H+ih)*V + vv).to(tl.float32)
        beta = tl.load(BETA + t*H+ih).to(tl.float32)
        decay = tl.exp(tl.load(G + t*H+ih).to(tl.float32))
        decayed = state * decay
        predicted = tl.sum(k[:, None] * decayed, 0)
        residual = v-predicted
        update = k[:, None] * (beta*residual)[None, :]
        after = decayed + update
        # Sum over value tiles before sqrt: Frobenius norms per TOKEN/HEAD.
        factor2 = beta*beta*tl.sum(k*k)
        sp = ((t*H+ih)*(V//BV)+iv)*8
        tl.store(STATS+sp, tl.sum(v*v))
        tl.store(STATS+sp+1, tl.sum(residual*residual))
        tl.store(STATS+sp+2, tl.sum(predicted*predicted))
        tl.store(STATS+sp+3, tl.sum(v*predicted))
        tl.store(STATS+sp+4, factor2*tl.sum(v*v))
        tl.store(STATS+sp+5, factor2*tl.sum(residual*residual))
        forgetting = decayed-state
        step = after-state
        tl.store(STATS+sp+6, tl.sum(tl.sum(forgetting*forgetting, 0)))
        tl.store(STATS+sp+7, tl.sum(tl.sum(step*step, 0)))
        state = after
        if REMOVE:
            changed *= decay
            cp = tl.sum(k[:, None]*changed, 0)
            is_vis = (t >= START) & (t < END)
            changed += k[:, None]*(beta*(v-tl.where(is_vis, 0., cp)))[None, :]
        else:
            changed = state
        op = (t*H+ih)*V+vv
        tl.store(OUT+op, tl.sum(q[:, None]*state, 0))
        tl.store(OUT+T*H*V+op, tl.sum(q[:, None]*changed, 0))
    tl.store(FINAL+offset, state)
    tl.store(FINAL+stride+offset, changed)


def replay(q, k, v, g, beta, start, end, *, initial=None, remove=False):
    from fla.modules.l2norm import l2norm_fwd
    assert q.shape[0] == 1 and q.shape == k.shape
    _, length, heads, kd = q.shape
    vd = v.shape[-1]
    assert kd == vd == 128 and v.shape[2] == heads
    assert 0 <= start <= end <= length and length > 1
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    if initial is None:
        initial = torch.zeros(heads, kd, vd, device=q.device, dtype=torch.float32)
    out = torch.empty(2, length, heads, vd, device=q.device, dtype=torch.float32)
    final = torch.empty(2, heads, kd, vd, device=q.device, dtype=torch.float32)
    stats = torch.empty(length, heads, vd//8, 8, device=q.device, dtype=torch.float32)
    _scan[(vd//8, heads)](q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
        beta.contiguous(), initial.contiguous(), final, out, stats, length, start, end,
        heads, kd, vd, 8, remove, num_warps=1, num_stages=1, enable_fp_fusion=False)
    return out, final, stats.sum(2)


def relerr(a, b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)).item()


def summarize(stats, *, compact=False):
    # Inputs [positions, heads, fields]. Preserve exact pooled sufficient stats.
    s = stats.double()
    n = s.shape[0]
    if not n:
        return {'positions': 0}
    value, residual, predicted = (s[..., i].clamp_min(0).sqrt() for i in range(3))
    raw, delta = (s[..., i].clamp_min(0).sqrt() for i in (4, 5))
    valid = value > 0
    c = 1-residual/value.clamp_min(1e-30)
    fields = torch.stack([raw.sum(0), delta.sum(0), value.sum(0), residual.sum(0),
        predicted.sum(0), s[..., 3].sum(0), s[..., 0].sum(0), s[..., 2].sum(0),
        s[..., 6].clamp_min(0).sqrt().sum(0), s[..., 7].clamp_min(0).sqrt().sum(0)], -1)
    result=dict(positions=n, heads=s.shape[1],
        fields=['raw_write_norm_sum','delta_write_norm_sum','value_norm_sum','residual_norm_sum',
                'predicted_norm_sum','value_predicted_dot_sum','value_squared_sum','predicted_squared_sum',
                'forgetting_norm_sum','total_state_step_norm_sum'],
        per_head_sums=fields.cpu().tolist(),
        cancellation_ratio=1-(delta.sum()/raw.sum().clamp_min(1e-30)).item(),
        value_norm_mean=value.mean().item(), residual_norm_mean=residual.mean().item(),
        token_head_c_quantiles=torch.quantile(c[valid], s.new_tensor([0,.05,.25,.5,.75,.95,1])).cpu().tolist(),
        zero_value_count=int((~valid).sum()), zero_raw_write_count=int((raw==0).sum()),
        negative_c_count=int(((c<0)&valid).sum()), c_gt_09_count=int(((c>.9)&valid).sum()))
    if compact:
        result['pooled_sums']=fields.sum(0).cpu().tolist()
        del result['per_head_sums']
    return result


class DeltaCancellation:
    def __init__(self, model, mask, mode='observe'):
        assert mode in ('observe','no_cancel','replay_control','no_cancel_pure','noop')
        self.model, self.mode = model, mode
        idx=mask[0].nonzero().flatten()
        self.start, self.end = int(idx[0]), int(idx[-1])+1
        assert idx.numel() == self.end-self.start
        self.length=mask.shape[1]
        self.records, self.states = {}, {}

    def call(self, layer, fn, q, k, v, **kw):
        native, final = fn(q,k,v,**kw)
        if q.shape[1] == 1:  # Decode is entirely native with the intervened cache.
            return native, final
        assert layer not in self.records and q.shape[1] == self.length
        assert kw['use_qk_l2norm_in_kernel'] and kw['output_final_state']
        assert kw.get('initial_state') is None and kw.get('cu_seqlens') is None
        remove = self.mode in ('no_cancel','no_cancel_pure')
        out, state, stats = replay(q,k,v,kw['g'],kw['beta'],self.start,self.end,remove=remove)
        checks=dict(replay_vs_native_core=relerr(out[0],native[0]),
                    replay_vs_native_final_state=relerr(state[0],final[0]))
        assert max(checks.values()) < .03, (layer, checks)
        rec=dict(layer=layer, checks=checks)
        if self.mode == 'observe':
            rec.update(visual=summarize(stats[self.start:self.end]),
                text=summarize(torch.cat((stats[:self.start],stats[self.end:]),0)),
                text_before=summarize(stats[:self.start]),text_after=summarize(stats[self.end:]),
                visual_position_bins=[summarize(x,compact=True) for x in stats[self.start:self.end].tensor_split(16)])
        if self.mode in ('observe','noop'):
            result=(native,final)
        elif self.mode == 'replay_control':
            result=(out[0:1].to(native.dtype),state[0:1])
        elif self.mode == 'no_cancel_pure':
            result=(out[1:2].to(native.dtype),state[1:2])
        else:
            result=((native.float()+(out[1:2]-out[0:1])).to(native.dtype),final+(state[1:2]-state[0:1]))
            rec['local_state_relative_change']=relerr(result[1], final)
        assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all(), (layer,self.mode)
        self.records[layer]=rec
        self.states[layer]=result[1].detach().clone()
        return result

    @contextmanager
    def activate(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        originals=[]
        try:
            for index, layer in enumerate(self.model.model.language_model.layers):
                if layer.block_type != 'linear_attention':
                    continue
                module=layer.linear_attn
                original=module.forward
                def wrapped(this,*args,_i=index,_orig=original,**kwargs):
                    chunk=m.torch_chunk_gated_delta_rule
                    def intercept(q,k,v,**kw):
                        return self.call(_i,chunk,q,k,v,**kw)
                    m.torch_chunk_gated_delta_rule=intercept
                    try:
                        return _orig(*args,**kwargs)
                    finally:
                        m.torch_chunk_gated_delta_rule=chunk
                originals.append((module,original))
                module.forward=MethodType(wrapped,module)
            yield self
            assert len(self.records)==24, (self.mode,len(self.records))
        finally:
            for module,original in originals:
                module.forward=original
