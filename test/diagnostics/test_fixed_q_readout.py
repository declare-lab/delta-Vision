import types
import torch
from src.fixed_q_visual_readout import fixed_readout,visible_top_overlap

def toy_layer():
    torch.manual_seed(31)
    attn=types.SimpleNamespace(head_dim=4,scaling=.5,
        q_proj=torch.nn.Linear(8,8,bias=False),k_proj=torch.nn.Linear(8,4,bias=False),
        v_proj=torch.nn.Linear(8,4,bias=False),o_proj=torch.nn.Linear(8,8,bias=False),
        q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity())
    return types.SimpleNamespace(self_attn=attn,input_layernorm=torch.nn.LayerNorm(8))

def test_identity_and_full_denominator():
    layer=toy_layer();h=torch.randn(1,9,8);visual=torch.tensor([2,4,5])
    pos=(torch.ones(1,9,4),torch.zeros(1,9,4))
    kv,blocks=fixed_readout(layer,h,h[0,visual],visual,pos)
    b=blocks[0]
    torch.testing.assert_close(b['ids'],torch.tensor([3,6,7,8]))
    for p,t in [('cp','ct'),('sqp','sqt'),('outp','outt'),('avp','avt')]:
        torch.testing.assert_close(b[p],b[t],atol=0,rtol=0)
    assert b['kl'].abs().max()==0
    assert (b['mass_t']<1).all()
    # For the interleaved text query at 3, visual keys 4/5 are future.
    assert (b['avt'][0,:,1:]==0).all()

def test_content_change_changes_readout():
    layer=toy_layer();h=torch.randn(1,9,8);visual=torch.tensor([2,4,5])
    pos=(torch.ones(1,9,4),torch.zeros(1,9,4))
    _,blocks=fixed_readout(layer,h,torch.randn(3,8),visual,pos)
    b=blocks[0]
    assert b['kl'].min()>-1e-6 and b['kl'].max()>1e-5
    assert (b['cp']-b['ct']).norm()>1e-4

def test_topk_masks_future_and_avoids_underflow():
    # All visual probabilities could underflow against a huge nonvisual sink.
    # Logit ordering must still be measured, without selecting future keys.
    t=torch.tensor([[[-1000.,-1001.,5000.]]])
    p=torch.tensor([[[-1001.,-1000.,5000.]]])
    visible=torch.tensor([[True,True,False]])
    assert visible_top_overlap(t,p,visible,top=1).item()==0
    assert visible_top_overlap(t,t,visible,top=2).item()==1

def test_matches_independent_gqa_attention_with_rope():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    layer=toy_layer();h=torch.randn(1,9,8);visual=torch.tensor([2,4,5])
    angle=torch.randn(1,9,2).repeat(1,1,2)
    pos=(angle.cos(),angle.sin())
    _,blocks=fixed_readout(layer,h,h[0,visual],visual,pos)
    n=layer.input_layernorm(h);a=layer.self_attn
    q=a.q_proj(n).view(1,9,2,4).transpose(1,2)
    k=a.k_proj(n).view(1,9,1,4).transpose(1,2)
    v=a.v_proj(n).view(1,9,1,4).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*pos)
    mask=torch.ones(9,9,dtype=torch.bool).tril()
    full=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
    vv=v.clone();vv[:,:,[0,1,3,6,7,8]]=0
    contribution=torch.nn.functional.scaled_dot_product_attention(q,k,vv,attn_mask=mask,enable_gqa=True)
    for b in blocks:
        expected=full[0,:,b['ids']].transpose(0,1).flatten(1)
        vis_expected=contribution[0,:,b['ids']].transpose(0,1).flatten(1)
        torch.testing.assert_close(b['fullt'],expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(b['ct'],vis_expected,rtol=1e-5,atol=1e-6)
