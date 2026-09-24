"""Fixed-vanilla-trajectory decomposition of Gated DeltaNet writes by position.

A_t = exp(g_t) (I - beta_t k_t k_t^T), B_t = beta_t k_t v_t^T.
All three replays use identical A. Only the additive B is routed by source.
This diagnostic never replaces the model's native output or cache.
"""
from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T', 'START', 'END', 'NT'])
def _sources(Q, K0, V0, G, BETA, INITIAL, FINAL, OUT, STATS, BOUNDARY,
             T, START, END, NT, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             BV: tl.constexpr, NORMALIZE: tl.constexpr):
    # Each value tile is independent under left multiplication by A_t.
    iv, ih = tl.program_id(0), tl.program_id(1)
    kk = tl.arange(0, K)
    vv = iv * BV + tl.arange(0, BV)
    state_offset = ih*K*V + kk[:, None]*V + vv[None, :]
    stride = H*K*V
    total = tl.load(INITIAL + state_offset).to(tl.float32)
    vis = tl.load(INITIAL + stride + state_offset).to(tl.float32)
    text = tl.load(INITIAL + 2*stride + state_offset).to(tl.float32)
    for t in tl.range(0, T):
        q = tl.load(Q + (t*H+ih)*K + kk).to(tl.float32)
        k = tl.load(K0 + (t*H+ih)*K + kk).to(tl.float32)
        v = tl.load(V0 + (t*H+ih)*V + vv).to(tl.float32)
        if NORMALIZE:
            q = q / tl.sqrt(tl.sum(q*q)+1e-6)
            k = k / tl.sqrt(tl.sum(k*k)+1e-6)
        q = q * (K ** -0.5)
        beta = tl.load(BETA + t*H+ih).to(tl.float32)
        decay = tl.exp(tl.load(G + t*H+ih).to(tl.float32))
        total *= decay
        vis *= decay
        text *= decay
        is_vis = (t >= START) & (t < END)
        # IMPORTANT: retain the -beta*k*k^T term for BOTH sources at ALL positions.
        total += k[:, None] * (beta * (v-tl.sum(k[:, None]*total, 0)))[None, :]
        vis += k[:, None] * (beta * (tl.where(is_vis, v, 0.)-tl.sum(k[:, None]*vis, 0)))[None, :]
        text += k[:, None] * (beta * (tl.where(is_vis, 0., v)-tl.sum(k[:, None]*text, 0)))[None, :]
        if t == END-1:
            tl.store(BOUNDARY + ih*(V//BV)+iv, tl.sum(tl.sum(vis*vis, 0), 0))
        if not is_vis:
            ti = tl.where(t < START, t, t-(END-START))
            op = (ti*H+ih)*V+vv
            o = tl.sum(q[:, None]*total, 0)
            ov = tl.sum(q[:, None]*vis, 0)
            ot = tl.sum(q[:, None]*text, 0)
            tl.store(OUT+op, o)
            tl.store(OUT+NT*H*V+op, ov)
            tl.store(OUT+2*NT*H*V+op, ot)
            sp = ((ti*H+ih)*(V//BV)+iv)*4
            tl.store(STATS+sp, tl.sum(tl.sum(vis*vis, 0), 0))
            tl.store(STATS+sp+1, tl.sum(tl.sum(total*total, 0), 0))
            err = total-vis-text
            tl.store(STATS+sp+2, tl.sum(tl.sum(err*err, 0), 0))
            tl.store(STATS+sp+3, tl.sum(tl.sum(text*text, 0), 0))
    tl.store(FINAL+state_offset, total)
    tl.store(FINAL+stride+state_offset, vis)
    tl.store(FINAL+2*stride+state_offset, text)


def split_sources(q, k, v, g, beta, start, end, *, initial=None, prefill=True):
    assert q.shape[0] == 1 and q.shape == k.shape
    _, length, heads, kd = q.shape
    vd = v.shape[-1]
    assert kd == vd == 128 and v.shape[2] == heads
    assert 0 <= start <= end <= length
    if prefill:
        # Match native chunk's BF16 l2norm rounding. Native decode instead
        # normalizes inside its recurrent kernel, in FP32.
        from fla.modules.l2norm import l2norm_fwd
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    nt = length-(end-start)
    if initial is None:
        initial = torch.zeros(3, heads, kd, vd, device=q.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    out = torch.empty(3, nt, heads, vd, device=q.device, dtype=torch.float32)
    stats = torch.empty(nt, heads, vd//8, 4, device=q.device, dtype=torch.float32)
    boundary = torch.zeros(heads, vd//8, device=q.device, dtype=torch.float32)
    _sources[(vd//8, heads)](q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
        beta.contiguous(), initial, final, out, stats, boundary, length, start, end, nt,
        heads, kd, vd, 8, not prefill, num_warps=1, num_stages=1, enable_fp_fusion=False)
    return out, stats.sum(2), final, boundary.sum(1)


def relative_error(a, b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)).item()


class StateSourceTracker:
    def __init__(self, model, visual_mask):
        self.model = model
        idx = visual_mask[0].nonzero().flatten()
        self.start, self.end = int(idx[0]), int(idx[-1])+1
        assert idx.numel() == self.end-self.start
        self.prompt_length = visual_mask.shape[1]
        self.states, self.boundaries, self.traces, self.checks = {}, {}, {}, {}

    def record(self, layer, q, k, v, kw, native_core, native_final):
        prefill = layer not in self.states
        if prefill:
            assert q.shape[1] == self.prompt_length and kw.get('initial_state') is None
            start, end = self.start, self.end
        else:
            assert q.shape[1] == 1
            start = end = 0
        out, stats, final, boundary = split_sources(q, k, v, kw['g'], kw['beta'], start, end,
            initial=self.states.get(layer), prefill=prefill)
        self.states[layer] = final
        if prefill:
            self.boundaries[layer] = boundary
        original_text = torch.cat((native_core[0, :start], native_core[0, end:]), 0)
        total, visual, text = out
        decomposition_output = relative_error(visual+text, total)
        decomposition_state = relative_error(final[1]+final[2], final[0])
        numerical_core = relative_error(total, original_text)
        numerical_state = relative_error(final[0], native_final[0])
        assert decomposition_output < 2e-5, (layer, decomposition_output)
        assert decomposition_state < 2e-5, (layer, decomposition_state)
        # Replay vs native mixed-precision chunk rounding is measured separately.
        assert numerical_core < .03, (layer, 'native core', numerical_core)
        assert numerical_state < .03, (layer, 'native state', numerical_state)
        checks = dict(source_output_relative_error=decomposition_output,
                      source_final_state_relative_error=decomposition_state,
                      fp32_replay_vs_native_core=numerical_core,
                      fp32_replay_vs_native_final_state=numerical_state,
                      source_state_max_position_error=(stats[..., 2].sum(-1)/stats[..., 1].sum(-1).clamp_min(1e-30)).sqrt().max().item())
        self.checks.setdefault(layer, []).append(checks)
        norm = total.flatten(1).norm(dim=-1)
        visual_ratio = visual.flatten(1).norm(dim=-1)/norm.clamp_min(1e-30)
        # Also compare against the actual BF16 native readout as a sensitivity check.
        native_ratio = visual.flatten(1).norm(dim=-1)/original_text.float().flatten(1).norm(dim=-1).clamp_min(1e-30)
        survival = (stats[..., 0].sum(-1)/self.boundaries[layer].sum().clamp_min(1e-30)).sqrt()
        positions = list(range(start))+list(range(end, q.shape[1])) if prefill else [self.prompt_length+len(self.traces[layer])-self.prompt_text_count]
        if prefill:
            self.prompt_text_count = len(positions)
        for j, position in enumerate(positions):
            rec = dict(position=position, phase='prompt' if prefill else 'decode',
                distance_after_visual=position-self.end+1,
                visual_readout_ratio=visual_ratio[j].item(),
                visual_readout_ratio_native_denominator=native_ratio[j].item(),
                visual_state_survival=survival[j].item(),
                total_readout_norm=norm[j].item(), visual_readout_norm=visual[j].norm().item(),
                visual_state_norm=stats[j, :, 0].sum().sqrt().item(),
                visual_readout_ratio_per_head=(visual[j].norm(dim=-1)/total[j].norm(dim=-1).clamp_min(1e-30)).tolist(),
                visual_state_survival_per_head=(stats[j, :, 0]/self.boundaries[layer].clamp_min(1e-30)).sqrt().tolist())
            self.traces.setdefault(layer, []).append(rec)

    @contextmanager
    def activate(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        originals = []
        try:
            for index, layer in enumerate(self.model.model.language_model.layers):
                if layer.block_type != 'linear_attention':
                    continue
                module = layer.linear_attn
                original = module.forward
                def wrapped(this, *args, _i=index, _orig=original, **kwargs):
                    chunk, recurrent = m.torch_chunk_gated_delta_rule, m.torch_recurrent_gated_delta_rule
                    def intercept(fn):
                        def call(q, k, v, **kw):
                            assert kw.get('output_final_state'), 'Use native generation with cache'
                            core, final = fn(q, k, v, **kw)
                            self.record(_i, q, k, v, kw, core, final)
                            return core, final  # Return native tensors, untouched.
                        return call
                    m.torch_chunk_gated_delta_rule = intercept(chunk)
                    m.torch_recurrent_gated_delta_rule = intercept(recurrent)
                    try:
                        return _orig(*args, **kwargs)
                    finally:
                        m.torch_chunk_gated_delta_rule = chunk
                        m.torch_recurrent_gated_delta_rule = recurrent
                originals.append((module, original))
                module.forward = MethodType(wrapped, module)
            yield self
        finally:
            for module, original in originals:
                module.forward = original

    def result(self):
        assert len(self.traces) == 24
        return [dict(layer=i, visual_end_state_norm=self.boundaries[i].sum().sqrt().item(),
                     positions=self.traces[i], checks=self.checks[i]) for i in sorted(self.traces)]
