import torch

from src.embedding_task_subspace import Stats, noise_pair, projection_stats


def test_projection_partition_and_metrics():
    torch.manual_seed(44)
    y=torch.randn(9,12,dtype=torch.float64)+2
    p=y+torch.randn_like(y)*.3
    b=torch.linalg.qr(torch.randn(12,5,dtype=torch.float64)).Q
    (pt,yt),(pn,yn)=projection_stats(p,y,b,5)
    torch.testing.assert_close((p-y).square().sum(),(pt-yt).square().sum()+(pn-yn).square().sum())
    torch.testing.assert_close(pn@b,torch.zeros(9,5,dtype=torch.float64),atol=1e-12,rtol=0)
    whole=Stats(12);whole.add_vectors(p,y)
    s1=Stats(12);s1.add_vectors(p[:4],y[:4])
    s2=Stats(12);s2.add_vectors(p[4:],y[4:])
    a=Stats.merge([whole.state()]);c=Stats.merge([s1.state(),s2.state()])
    for key in ('r2','cosine','mse','relative_error'):
        assert abs(a[key]-c[key])<1e-12
    expected=1-float((p-y).square().sum()/((y-y.mean(0)).square().sum()))
    assert abs(expected-a['r2'])<1e-12
    assert abs(a['relative_error']-float((p-y).norm()/y.norm()))<1e-12


def test_equal_token_noise_norms_and_subspaces():
    torch.manual_seed(44)
    h=torch.randn(7,32)
    b=torch.linalg.qr(torch.randn(32,5)).Q
    task,complement=noise_pair(h,b,45)
    torch.testing.assert_close(task.norm(dim=-1),h.norm(dim=-1))
    torch.testing.assert_close(complement.norm(dim=-1),h.norm(dim=-1))
    torch.testing.assert_close(task,task@b@b.T,atol=3e-6,rtol=1e-5)
    torch.testing.assert_close(complement@b,torch.zeros(7,5),atol=3e-6,rtol=0)
    t2,n2=noise_pair(h,b,45)
    assert torch.equal(task,t2) and torch.equal(complement,n2)


if __name__=='__main__':
    test_projection_partition_and_metrics()
    test_equal_token_noise_norms_and_subspaces()
    print('PASS: pooled metrics, projection/complement partition, norm-matched deterministic noise')
