"""CUDA scan vs independent FP64 CPU recurrence, including intervention."""
import torch


def reference(q,k,v,g,beta,start,end):
    q,k,v,g,beta=[x.double().cpu() for x in (q,k,v,g,beta)]
    _,length,heads,kd=q.shape
    state=torch.zeros(2,heads,kd,v.shape[-1],dtype=torch.float64)
    outputs=[];stats=[]
    for t in range(length):
        old=state.clone()
        state*=g[0,t].exp()[None,:,None,None]
        kt,vt,b=k[0,t],v[0,t],beta[0,t]
        pred=(state*kt[None,:,:,None]).sum(-2)
        residual=vt-pred[0]
        update=b[:,None,None]*kt[:,:,None]*residual[:,None,:]
        factor=b.square()*kt.square().sum(-1)
        forget=state[0]-old[0]
        stats.append(torch.stack([vt.square().sum(-1),residual.square().sum(-1),
            pred[0].square().sum(-1),(vt*pred[0]).sum(-1),factor*vt.square().sum(-1),
            factor*residual.square().sum(-1),forget.square().sum((-1,-2)),
            (forget+update).square().sum((-1,-2))],-1))
        state[0]+=update
        r1=vt if start<=t<end else vt-pred[1]
        state[1]+=b[:,None,None]*kt[:,:,None]*r1[:,None,:]
        outputs.append((state*(q[0,t]/kd**.5)[None,:,:,None]).sum(-2))
    return torch.stack(outputs,1),state,torch.stack(stats)


def run_tests():
    from analysis.fig05_hybrid_attention.qwen35_delta_cancellation import replay, summarize
    from fla.modules.l2norm import l2norm_fwd
    torch.manual_seed(44)
    q,k,v=[torch.randn(1,19,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(1,19,2,device='cuda')*.2
    b=torch.rand(1,19,2,device='cuda',dtype=torch.bfloat16)
    actual=replay(q,k,v,g,b,3,11,remove=True)
    qn,_=l2norm_fwd(q);kn,_=l2norm_fwd(k)
    expected=reference(qn,kn,v,g,b,3,11)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a.cpu().double(),e,rtol=5e-5,atol=5e-6)
    noop=replay(q,k,v,g,b,3,11,remove=False)
    assert torch.equal(noop[0][0],noop[0][1]) and torch.equal(noop[1][0],noop[1][1])
    # First orthonormal-key write has no correction, second identical write is
    # fully predicted. Removing correction makes the second state twice as large.
    q=k=torch.zeros(1,3,1,128,device='cuda',dtype=torch.bfloat16);k[...,0]=1
    v=torch.ones_like(k);g=torch.zeros(1,3,1,device='cuda');b=torch.ones_like(g).bfloat16()
    o,s,st=replay(q,k,v,g,b,1,3,remove=True)
    assert summarize(st[:1])['cancellation_ratio']==0
    assert summarize(st[1:])['cancellation_ratio']>.9999
    torch.testing.assert_close(s[1],s[0]*3,rtol=1e-5,atol=1e-5)
    return {'fp64_recurrence_match':True,'no_op_exact':True,'known_full_cancellation':True}
