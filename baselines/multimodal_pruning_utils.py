"""Index-safe extensions of the repository's Qwen pruning baselines.

Budgets count all visual positions in a request. Text, separators, timestamps,
and original M-RoPE coordinates must never be mistaken for visual tokens.
"""
import torch
from decimal import Decimal


def visual_budget(count, retention):
    """Decimal retention, nearest integer (ties-to-even), independent of 1-r noise."""
    if count == 0:return 0
    ratio = Decimal(str(round(float(retention), 12)))
    if not 0 <= ratio <= 1:raise ValueError(retention)
    return max(1, min(int(count), round(Decimal(int(count))*ratio)))


def prune_deepstack(mask, features, keep):
    """Use the same original visual indices for hidden and native DeepStack."""
    if mask is None:
        assert features is None
        return None, None
    visual=mask[0].bool().nonzero().flatten()
    selected=torch.isin(visual,keep).nonzero().flatten()
    if features is not None:
        assert all(x.shape[0]==len(visual) for x in features)
        features=[x[selected.to(x.device)] for x in features]
    return mask[:,keep],features


def fastv_visual_scores(q_last, keys, visual, scale):
    """Previous-layer last-query attention, global denominator, mean over heads."""
    groups=q_last.shape[1]//keys.shape[1]
    assert groups*keys.shape[1]==q_last.shape[1]
    if groups>1:keys=keys.repeat_interleave(groups,dim=1)
    logits=(q_last@keys.transpose(-1,-2))*scale
    return logits.softmax(-1,dtype=torch.float32).mean(1)[0,0,visual]


def visual_indices(mask, hidden, start, length):
    if mask is not None:
        assert hidden.shape[0] == 1 and mask.shape[-1] == hidden.shape[1]
        return mask[0].bool().nonzero().flatten().to(hidden.device)
    return torch.arange(start, start+length, device=hidden.device)


def keep_visual_subset(seq_len, visual, selected):
    mask=torch.ones(seq_len,dtype=torch.bool,device=visual.device)
    assert torch.isin(selected,visual).all()
    assert selected.unique().numel()==selected.numel()
    mask[visual]=False;mask[selected]=True
    return mask.nonzero().flatten()


def audit_prune(module, seq_len, visual, selected, layer):
    if not getattr(module, '_pruning_audit_enabled', True):
        return
    if not hasattr(module,'_pruning_audit'):module._pruning_audit=[]
    module._pruning_audit.append(dict(layer=layer,before_visual=int(visual.numel()),
        after_visual=int(selected.numel()),text_tokens=int(seq_len-visual.numel()),
        visual_positions=visual.detach().cpu().tolist(),selected_positions=selected.detach().cpu().tolist()))


def sparse_rater_scores(q,k,visual,raters,scale,chunk=128):
    """Same unmasked selector softmax as existing SparseVLM, bounded row chunks.

This is a pruning score over the known prompt, not the causal LM attention.
"""
    seq=q.shape[2]
    if raters is None:
        candidates=torch.arange(seq,device=q.device)
        candidates=candidates[(candidates>visual[0]) & ~torch.isin(candidates,visual)]
        if len(candidates):
            received=torch.zeros(len(candidates),device=q.device,dtype=torch.float32)
            for block in visual.split(chunk):
                a=((q[:,:,block]@k.transpose(-1,-2))*scale).softmax(-1,dtype=torch.float32).mean(1)[0]
                received+=a[:,candidates].sum(0)
            raters=candidates[received>received.mean()]
        if raters is None or len(raters)==0:raters=torch.tensor([seq-1],device=q.device)
    scores=torch.zeros(len(visual),device=q.device,dtype=torch.float32)
    for block in raters.split(chunk):
        a=((q[:,:,block]@k.transpose(-1,-2))*scale).softmax(-1,dtype=torch.float32).mean(1)[0]
        scores+=a[:,visual].sum(0)
    return scores,raters
