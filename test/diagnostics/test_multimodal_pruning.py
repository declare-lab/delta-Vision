import torch
from baselines.multimodal_pruning_utils import visual_budget, prune_deepstack
from baselines.multimodal_pruning_utils import visual_indices, keep_visual_subset, sparse_rater_scores

def test_fastv_preserves_global_head_weighting():
    from baselines.multimodal_pruning_utils import fastv_visual_scores
    q=torch.ones(1,2,1,1)
    keys=torch.tensor([[[[100.],[2.],[0.]],[[-100.],[0.],[1.]]]])
    visual=torch.tensor([1,2])
    score=fastv_visual_scores(q,keys,visual,1.)
    assert score.argmax().item()==1
    conditional=(q@keys[:,:,visual].transpose(-1,-2)).softmax(-1).mean(1)[0,0]
    assert conditional.argmax().item()==0
    torch.testing.assert_close(score,(q@keys.transpose(-1,-2)).softmax(-1).mean(1)[0,0,visual])

def test_budget_decimal_boundary():
    assert visual_budget(1250,.05)==62
    assert visual_budget(1250,1-.95)==62
    assert visual_budget(1250,.05*.8+.05*.2)==62
    for n in range(1,20000):
        for r in (.05,.2):
            assert visual_budget(n,r)==visual_budget(n,1-(1-r))==visual_budget(n,r*.8+r*.2)

def test_deepstack_mask_alignment():
    mask=torch.tensor([[0,1,1,0,1,0]],dtype=torch.bool)
    feature=torch.arange(12).reshape(3,4)
    keep=torch.tensor([0,2,3,4,5])
    m,f=prune_deepstack(mask,[feature],keep)
    torch.testing.assert_close(m,mask[:,keep])
    torch.testing.assert_close(f[0],feature[[1,2]])

def test_discontiguous_keeps_all_text():
    h=torch.randn(1,11,8);mask=torch.tensor([[0,1,1,0,0,1,1,0,1,1,0]])
    visual=visual_indices(mask,h,1,6)
    torch.testing.assert_close(visual,torch.tensor([1,2,5,6,8,9]))
    kept=keep_visual_subset(11,visual,torch.tensor([2,8]))
    torch.testing.assert_close(kept,torch.tensor([0,2,3,4,7,8,10]))

def test_sparse_chunks_equal_dense():
    torch.manual_seed(0)
    q=torch.randn(1,4,17,8);k=torch.randn_like(q)
    visual=torch.tensor([1,2,3,7,8,12]);scale=8**-.5
    dense=((q@k.transpose(-1,-2))*scale).softmax(-1).mean(1)[0]
    candidates=torch.tensor([4,5,6,9,10,11,13,14,15,16])
    received=dense[visual][:,candidates].sum(0)
    raters=candidates[received>received.mean()]
    scores,got=sparse_rater_scores(q,k,visual,None,scale,chunk=2)
    torch.testing.assert_close(got,raters)
    torch.testing.assert_close(scores,dense[raters][:,visual].sum(0))

def test_contiguous_keep_matches_legacy():
    vis=torch.arange(3,13);sel=torch.tensor([4,8,10])
    torch.testing.assert_close(keep_visual_subset(18,vis,sel),torch.cat((torch.arange(3),sel,torch.arange(13,18))))

def test_diversity_running_min_is_exact():
    # Load only selectors from AST: no checkpoint/custom model import needed.
    import ast
    from pathlib import Path
    for method,name in [('divprune','_divprune_select_tokens'),('zoo','_zoo_select_tokens')]:
        path=Path(f'baselines/{method}/qwen3_vl/modeling_qwen3_vl_{method}.py')
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
        for seed in (1,5,41):
            torch.manual_seed(seed);x=torch.randn(30,12);importance=torch.rand(30)
            for count in (1,3,15,30):
                got=scope[name](x,count) if method=='divprune' else scope[name](x,importance,count)
                normalized=torch.nn.functional.normalize(x,dim=-1);dist=1-normalized@normalized.T
                weight=(importance-importance.min())/(importance.max()-importance.min()+1e-8)
                selected=[]
                for i in range(count):
                    if i==0:score=dist.topk(2,dim=0,largest=False).values[1] if method=='divprune' else weight
                    else:
                        score=dist[selected].min(dim=0).values
                        if method=='zoo':score=score*weight
                        score[selected]=-float('inf')
                    selected.append(int(score.argmax()))
                torch.testing.assert_close(got,torch.tensor(selected))
