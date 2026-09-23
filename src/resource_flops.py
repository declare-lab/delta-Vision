"""Executed matrix/conv/FlashAttention FLOPs, outside benchmark timing.

Two FLOPs per MAC, dense QK/AV convention (also for causal attention), matching
PyTorch FlopCounterMode. Includes vision, decoder, head, adapter and selector
matrix operations. Does not claim to count scalar normalization/softmax,
activation, sorting/comparison or indexing operations.
"""
from math import prod
import torch
from torch.utils.flop_counter import FlopCounterMode


def _flash_dense(q,k,v,*args,out_val=None,**kwargs):
    assert q.ndim==k.ndim==v.ndim==4
    return 2*int(q.shape[0])*int(q.shape[1])*int(k.shape[1])*int(q.shape[2])*(int(q.shape[3])+int(v.shape[3]))


def _flash_varlen(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,*args,out_val=None,**kwargs):
    qlens=cu_seqlens_q.diff().tolist();klens=cu_seqlens_k.diff().tolist()
    # FlashAttention uses actual lengths for shared-prefix KV packing.
    used=kwargs.get('seqused_k',args[10] if len(args)>10 else None)
    if used is not None:klens=used.tolist()
    assert len(qlens)==len(klens)
    return 2*sum(int(a)*int(b) for a,b in zip(qlens,klens))*int(q.shape[1])*(int(q.shape[2])+int(v.shape[2]))


_flash_dense._get_raw=True
_flash_varlen._get_raw=True


def matrix_flop_counter():
    import flash_attn.flash_attn_interface  # register external FA2 custom ops
    return FlopCounterMode(display=False,custom_mapping={
        torch.ops.flash_attn._flash_attn_forward:_flash_dense,
        torch.ops.flash_attn._flash_attn_varlen_forward:_flash_varlen})


def counts(counter):
    return {str(k):int(v) for k,v in counter.flop_counts.get('Global',{}).items()}
