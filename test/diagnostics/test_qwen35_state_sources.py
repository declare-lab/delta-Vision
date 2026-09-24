"""Run with Qwen3.5 dependency path; CUDA test compares against FP64 recurrence."""
import torch


def reference(q, k, v, g, beta, start, end, initial=None):
    heads, kd, vd = k.shape[2], k.shape[-1], v.shape[-1]
    s = torch.zeros(3, heads, kd, vd, dtype=torch.float64) if initial is None else initial.double().cpu().clone()
    outputs=[];norms=[];boundary=None
    for t in range(q.shape[1]):
        qt, kt, vt = q[0,t].double().cpu()/kd**.5, k[0,t].double().cpu(), v[0,t].double().cpu()
        decay = g[0,t].double().cpu().exp()
        b = beta[0,t].double().cpu()
        s *= decay[None,:,None,None]
        for source in range(3):
            write = vt if source==0 or (source==1)==(start<=t<end) else torch.zeros_like(vt)
            delta=b[:,None]*(write-(kt[None,:,:,None]*s[source:source+1]).sum(2)[0])
            s[source] += kt[:,:,None]*delta[:,None,:]
        if t==end-1:boundary=s[1].square().sum((-1,-2))
        if not start<=t<end:
            outputs.append((s*qt[None,:,:,None]).sum(2))
            norms.append(torch.stack((s[1].square().sum((-1,-2)),s[0].square().sum((-1,-2)),
                                     (s[0]-s[1]-s[2]).square().sum((-1,-2)),s[2].square().sum((-1,-2))),-1))
    return torch.stack(outputs,1),torch.stack(norms),s,boundary


def run_tests():
    from analysis.fig05_hybrid_attention.qwen35_state_sources import split_sources
    from fla.modules.l2norm import l2norm_fwd
    torch.manual_seed(44)
    q,k,v=[torch.randn(1,13,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(1,13,2,device='cuda')*.2
    beta=torch.rand(1,13,2,device='cuda',dtype=torch.bfloat16)
    out,stats,state,boundary=split_sources(q,k,v,g,beta,2,7)
    qn,_=l2norm_fwd(q);kn,_=l2norm_fwd(k)
    expected=reference(qn,kn,v,g,beta,2,7)
    for actual,ref in zip((out,stats,state,boundary),expected):
        torch.testing.assert_close(actual.cpu().double(),ref,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(out[1]+out[2],out[0],rtol=2e-5,atol=2e-6)
    assert out[1,:2].eq(0).all()
    assert torch.all(stats[2:, :, 0] <= boundary[None,:]+1e-5)
    # Decode retains both sources and uses FP32 q/k normalization (native fused kernel).
    dq,dk,dv=[torch.randn(1,1,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    dg=-torch.rand(1,1,2,device='cuda')*.2;db=torch.rand(1,1,2,device='cuda',dtype=torch.bfloat16)
    actual=split_sources(dq,dk,dv,dg,db,0,0,initial=state,prefill=False)
    dqn=dq.float()/(dq.float().square().sum(-1,keepdim=True)+1e-6).sqrt()
    dkn=dk.float()/(dk.float().square().sum(-1,keepdim=True)+1e-6).sqrt()
    expected=reference(dqn,dkn,dv,dg,db,0,0,state)
    for a,r in zip(actual[:3],expected[:3]):torch.testing.assert_close(a.cpu().double(),r,rtol=3e-5,atol=3e-6)
    return {'fp64_reference_prefill':True,'fp64_reference_decode':True,'complete_A_in_both_sources':True,
            'zero_visual_source_before_image':True,'visual_state_contracts_after_image':True}


if __name__=='__main__':
    print(run_tests())
