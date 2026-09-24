"""FP64 recurrence oracle validates all token/head diagnostics and state split."""
import torch


def run_tests():
    from analysis.fig05_hybrid_attention.qwen35_write_forget import trace
    from fla.modules.l2norm import l2norm_fwd
    torch.manual_seed(44)
    q,k,v=[torch.randn(1,13,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(1,13,2,device='cuda')*.2;b=torch.rand_like(g).bfloat16()
    o,final,metrics,bounds=trace(q,k,v,g,b,3,8,1,8)
    qn,_=l2norm_fwd(q);kn,_=l2norm_fwd(k)
    qn,kn,vv,gg,bb=[x.double().cpu()[0] for x in [qn,kn,v,g,b]]
    states=torch.zeros(2,2,128,128,dtype=torch.float64);outs=[];boundary=torch.zeros_like(bounds.cpu().double())
    for t in range(13):
        old=states[0].clone();bar=old*gg[t].exp()[:,None,None]
        residual=vv[t]-(kn[t,:, :,None]*bar).sum(-2)
        update=bb[t,:,None,None]*kn[t,:,:,None]*residual[:,None,:]
        states[0]=bar+update
        if not 3<=t<8:
            states[1]*=gg[t].exp()[:,None,None]
            states[1]+=bb[t,:,None,None]*kn[t,:,:,None]*(vv[t]-(kn[t,:,:,None]*states[1]).sum(-2))[:,None,:]
        diff=states[0]-states[1];qt=qn[t]/128**.5
        actual_o=(qt[:,:,None]*states[0]).sum(-2);d=(qt[:,:,None]*diff).sum(-2);outs.append(actual_o)
        dn=diff.flatten(1).norm(dim=-1);on=actual_o.norm(dim=-1);rn=d.norm(dim=-1);pn=old.flatten(1).norm(dim=-1);un=update.flatten(1).norm(dim=-1)
        expected=torch.stack([bb[t],gg[t].exp(),vv[t].norm(dim=-1),residual.norm(dim=-1),un,pn,
            un/pn,dn,on,rn,rn/on,rn/(dn*qt.norm(dim=-1))],-1)
        valid=torch.isfinite(expected)
        torch.testing.assert_close(metrics[t].double().cpu()[valid],expected[valid],rtol=1e-4,atol=2e-5)
        if t==1:boundary[0]=states[0]
        if t==8:boundary[1:]=states
    torch.testing.assert_close(o.double().cpu(),torch.stack(outs),rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(final.double().cpu(),states,rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(bounds.double().cpu(),boundary,rtol=5e-5,atol=5e-6)
    assert torch.isnan(metrics[0,:,6]).all(),'Initial zero state must not yield a fake finite relative perturbation'
    return dict(fp64_metrics_match=True,fp64_boundary_match=True,fp64_dual_replay_match=True)
