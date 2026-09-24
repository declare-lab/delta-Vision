"""Native-gate interventions and paired fixed-trajectory GDN diagnostics."""
from contextlib import contextmanager
from types import MethodType
import torch
import triton
import triton.language as tl

LA=[i for i in range(32) if i%4!=3]
FA=list(range(3,32,4))


@contextmanager
def visual_gates(model,mask,mode='native',layers=None,conv_boundary=None):
    """g/beta edits AFTER native projections; original FLA kernels execute.

conv_boundary resets raw depthwise-conv history before the selected token.
Prefill prefix output is native; suffix convolution starts with zero history.
"""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    assert mode in ('native','no_write','no_forget','state_skip')
    selected=LA if layers is None else layers
    originals=[];audit={'layers':selected,'prefill':{},'mode':mode,'conv_boundary':conv_boundary}
    try:
        for index in selected:
            module=model.model.language_model.layers[index].linear_attn
            original=module.forward
            def wrapped(this,*args,_i=index,_orig=original,**kwargs):
                kernel=m.torch_chunk_gated_delta_rule;conv=m.causal_conv1d_fn
                def gate(q,k,v,**kw):
                    if q.shape[1]>1:
                        assert q.shape[:2]==mask.shape and kw.get('initial_state') is None
                        audit['prefill'][_i]=audit['prefill'].get(_i,0)+1
                        kw=dict(kw)
                        if mode in ('no_forget','state_skip'):kw['g']=kw['g'].masked_fill(mask[...,None],0.)
                        if mode in ('no_write','state_skip'):kw['beta']=kw['beta'].masked_fill(mask[...,None],0.)
                    return kernel(q,k,v,**kw)
                def reset_conv(x,*aa,**kk):
                    native=conv(x,*aa,**kk)
                    if conv_boundary is None:return native
                    assert x.shape[-1]==mask.shape[1]
                    assert x.shape[-1]-conv_boundary>=this.conv_kernel_size
                    assert 0<conv_boundary<x.shape[-1]
                    suffix=conv(x[:,:,conv_boundary:].contiguous(),*aa,**kk)
                    return torch.cat((native[:,:,:conv_boundary],suffix),-1)
                m.torch_chunk_gated_delta_rule=gate
                if conv_boundary is not None:m.causal_conv1d_fn=reset_conv
                try:return _orig(*args,**kwargs)
                finally:
                    m.torch_chunk_gated_delta_rule=kernel;m.causal_conv1d_fn=conv
            originals.append((module,original));module.forward=MethodType(wrapped,module)
        yield audit
        assert set(audit['prefill'])==set(selected) and set(audit['prefill'].values())=={1}
    finally:
        for module,original in originals:module.forward=original


@triton.jit(do_not_specialize=['T','START','END','PRE','POST'])
def _trace(Q,K0,V0,G,BETA,INITIAL,FINAL,OUT,STATS,BOUNDARY,
           T,START,END,PRE,POST,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
           BV:tl.constexpr,NORMALIZE:tl.constexpr):
    iv,ih=tl.program_id(0),tl.program_id(1)
    kk=tl.arange(0,K);vv=iv*BV+tl.arange(0,BV)
    off=ih*K*V+kk[:,None]*V+vv[None,:];stride=H*K*V
    state=tl.load(INITIAL+off).to(tl.float32)
    skip=tl.load(INITIAL+stride+off).to(tl.float32)
    for t in tl.range(0,T):
        q=tl.load(Q+(t*H+ih)*K+kk).to(tl.float32)
        k=tl.load(K0+(t*H+ih)*K+kk).to(tl.float32)
        v=tl.load(V0+(t*H+ih)*V+vv).to(tl.float32)
        if NORMALIZE:
            q=q/tl.sqrt(tl.sum(q*q)+1e-6);k=k/tl.sqrt(tl.sum(k*k)+1e-6)
        q=q*(K**-.5)
        b=tl.load(BETA+t*H+ih).to(tl.float32)
        decay=tl.exp(tl.load(G+t*H+ih).to(tl.float32))
        before=state;state*=decay
        residual=v-tl.sum(k[:,None]*state,0)
        update=k[:,None]*(b*residual)[None,:]
        state+=update
        if not ((t>=START)&(t<END)):
            skip*=decay
            skip+=k[:,None]*(b*(v-tl.sum(k[:,None]*skip,0)))[None,:]
        difference=state-skip
        o=tl.sum(q[:,None]*state,0);d=tl.sum(q[:,None]*difference,0)
        tl.store(OUT+(t*H+ih)*V+vv,o)
        sp=((t*H+ih)*(V//BV)+iv)*7
        tl.store(STATS+sp,tl.sum(v*v))
        tl.store(STATS+sp+1,tl.sum(residual*residual))
        tl.store(STATS+sp+2,tl.sum(tl.sum(update*update,0)))
        tl.store(STATS+sp+3,tl.sum(tl.sum(before*before,0)))
        tl.store(STATS+sp+4,tl.sum(tl.sum(difference*difference,0)))
        tl.store(STATS+sp+5,tl.sum(o*o))
        tl.store(STATS+sp+6,tl.sum(d*d))
        if t==PRE:tl.store(BOUNDARY+off,state)
        if t==POST:
            tl.store(BOUNDARY+stride+off,state)
            tl.store(BOUNDARY+2*stride+off,skip)
    tl.store(FINAL+off,state);tl.store(FINAL+stride+off,skip)


def trace(q,k,v,g,beta,start,end,pre,post,initial=None,prefill=True):
    assert q.shape[0]==1 and q.shape==k.shape
    _,length,heads,kd=q.shape;vd=v.shape[-1];assert kd==vd==128
    if prefill:
        from fla.modules.l2norm import l2norm_fwd
        q,_=l2norm_fwd(q);k,_=l2norm_fwd(k)
    if initial is None:initial=torch.zeros(2,heads,kd,vd,device=q.device,dtype=torch.float32)
    final=torch.empty_like(initial)
    out=torch.empty(length,heads,vd,device=q.device,dtype=torch.float32)
    stats=torch.empty(length,heads,vd//8,7,device=q.device,dtype=torch.float32)
    boundary=torch.zeros(3,heads,kd,vd,device=q.device,dtype=torch.float32)
    _trace[(vd//8,heads)](q.contiguous(),k.contiguous(),v.contiguous(),g.contiguous(),beta.contiguous(),
        initial.contiguous(),final,out,stats,boundary,length,start,end,pre,post,heads,kd,vd,8,not prefill,
        num_warps=1,num_stages=1,enable_fp_fusion=False)
    ss=stats.sum(2).clamp_min(0).sqrt()
    # state=0 makes relative perturbation undefined; retain NaN, don't report zero.
    relative=torch.where(ss[:,:,3]>0,ss[:,:,2]/ss[:,:,3],float('nan'))
    readratio=torch.where(ss[:,:,5]>0,ss[:,:,6]/ss[:,:,5],float('nan'))
    raw_q_norm=q[0].float().norm(dim=-1)
    qnorm=raw_q_norm/(kd**.5) if prefill else raw_q_norm/(raw_q_norm.square()+1e-6).sqrt()/(kd**.5)
    alignment=torch.where(ss[:,:,4]*qnorm>0,ss[:,:,6]/(ss[:,:,4]*qnorm),float('nan'))
    metrics=torch.stack((beta[0].float(),g[0].float().exp(),ss[:,:,0],ss[:,:,1],ss[:,:,2],
        ss[:,:,3],relative,ss[:,:,4],ss[:,:,5],ss[:,:,6],readratio,alignment),-1)
    return out,final,metrics,boundary


METRICS=['beta','retention','value_norm','residual_norm','delta_write_norm','previous_state_norm',
         'relative_delta_write','native_minus_skip_state_norm','native_readout_norm',
         'native_minus_skip_readout_norm','relative_readout','query_state_alignment']


class WriteForgetTracker:
    def __init__(self,model,inputs):
        self.model=model;mask=inputs['mm_token_type_ids'].eq(1)
        pos=mask[0].nonzero().flatten();self.start=int(pos[0]);self.end=int(pos[-1])+1
        assert pos.numel()==self.end-self.start
        ids=inputs['input_ids'][0]
        assert int(ids[self.start-1])==model.config.vision_start_token_id
        assert int(ids[self.end])==model.config.vision_end_token_id
        self.pre=self.start-2;self.post=self.end;self.length=len(ids)
        assert self.pre>=0
        self.states={};self.metrics={};self.boundaries={};self.checks={}

    @contextmanager
    def activate(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        originals=[]
        try:
            for i in LA:
                module=self.model.model.language_model.layers[i].linear_attn;original=module.forward
                def wrapped(this,*args,_i=i,_orig=original,**kwargs):
                    chunk,recurrent=m.torch_chunk_gated_delta_rule,m.torch_recurrent_gated_delta_rule
                    def intercept(fn):
                        def call(q,k,v,**kw):
                            native,final=fn(q,k,v,**kw);prefill=_i not in self.states
                            o,s,metrics,bounds=trace(q,k,v,kw['g'],kw['beta'],
                                self.start if prefill else 0,self.end if prefill else 0,
                                self.pre if prefill else -1,self.post if prefill else -1,
                                initial=self.states.get(_i),prefill=prefill)
                            err=((o-native[0].float()).norm()/native.float().norm().clamp_min(1e-30)).item()
                            serr=((s[0]-final[0]).norm()/final[0].norm().clamp_min(1e-30)).item()
                            assert max(err,serr)<.03,(_i,err,serr)
                            self.checks.setdefault(_i,[]).append([err,serr])
                            self.states[_i]=s
                            self.metrics.setdefault(_i,[]).append(metrics.cpu())
                            if prefill:self.boundaries[_i]=bounds.cpu()
                            return native,final
                        return call
                    m.torch_chunk_gated_delta_rule=intercept(chunk);m.torch_recurrent_gated_delta_rule=intercept(recurrent)
                    try:return _orig(*args,**kwargs)
                    finally:m.torch_chunk_gated_delta_rule=chunk;m.torch_recurrent_gated_delta_rule=recurrent
                originals.append((module,original));module.forward=MethodType(wrapped,module)
            yield self
            assert len(self.metrics)==24
        finally:
            for module,original in originals:module.forward=original
