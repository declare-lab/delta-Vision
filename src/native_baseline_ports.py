"""Native-HF ports of attention pruning for LLaVA and hybrid/MoE Qwen.

No decoder attention kernel is replaced. All text tokens and original position
coordinates survive. Qwen3.5 uses the first full-attention block (3) for FastV,
and full-attention blocks 3/7/15 for SparseVLM. These are explicit hybrid
adaptations, not algorithms published by the original authors for Qwen3.5.
"""
from contextlib import contextmanager
from functools import partial
import sys
import torch
from torch.nn import functional as F
from baselines.multimodal_pruning_utils import visual_budget
from src.qwen_baseline_author_comparison import cluster_recycle,sparse_budgets


def stage_budgets(n,layers,ratio,stages):
    if stages==(2,6,15):return sparse_budgets(n,layers,ratio)
    widths=(stages[1]-stages[0],stages[2]-stages[1],layers-stages[2]-1)
    total=round(n*sum(widths)*ratio);profile=(303,110,36)
    scale=total/sum(w*p for w,p in zip(widths,profile));ideal=[p*scale for p in profile]
    choices=[]
    for a in range(max(1,round(ideal[0])-12),min(n,round(ideal[0])+12)+1):
        for b in range(max(1,round(ideal[1])-12),min(a,round(ideal[1])+12)+1):
            c=max(1,min(b,round((total-widths[0]*a-widths[1]*b)/widths[2])))
            err=abs(sum(w*v for w,v in zip(widths,(a,b,c)))-total)
            choices.append(((err,sum((v-t)**2 for v,t in zip((a,b,c),ideal))),(a,b,c)))
    return min(choices)[1]


def native_qk(layer,h,positions):
    a=layer.self_attn;x=layer.input_layernorm(h)
    shape=(*x.shape[:-1],-1,a.head_dim)
    q=a.q_proj(x)
    # Qwen3.5 interleaves each head's Q with a head-sized sigmoid gate.
    if 'qwen3_5' in type(a).__module__:
        q=q.view(*x.shape[:-1],-1,2*a.head_dim).chunk(2,-1)[0]
    q=q.reshape(shape);k=a.k_proj(x).view(shape)
    if hasattr(a,'q_norm'):q=a.q_norm(q);k=a.k_norm(k)
    q=q.transpose(1,2);k=k.transpose(1,2)
    apply=sys.modules[type(a).__module__].apply_rotary_pos_emb
    return apply(q,k,*positions)


def visual_attention_scores(layer,h,positions,queries,visual):
    q,k=native_qk(layer,h,positions)
    k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    # Chunk query positions only; preserve the complete causal denominator.
    result=torch.zeros(len(visual),device=h.device,dtype=torch.float32)
    for ids in queries.split(16):
        logits=(q[:,:,ids]@k.transpose(-1,-2))*layer.self_attn.scaling
        future=torch.arange(h.shape[1],device=h.device)[None,:]>ids[:,None]
        logits.masked_fill_(future[None,None],-float('inf'))
        result+=logits.softmax(-1,dtype=torch.float32).mean(1)[0,:,visual].sum(0)
    return result/max(1,len(queries))


class NativeVisualPruning:
    def __init__(self,model):
        self.model=model;self.text=model.model.language_model;self.enabled=False
        self.hybrid='qwen3_5' in type(self.text).__module__
        self.stages=(3,7,15) if self.hybrid else (2,6,15)
        self.first=4 if self.hybrid else 2
        self.handles=[self.text.register_forward_pre_hook(self.start,with_kwargs=True)]
        for i,layer in enumerate(self.text.layers):
            self.handles.append(layer.register_forward_pre_hook(partial(self.pre,i),with_kwargs=True))
            self.handles.append(layer.register_forward_hook(partial(self.post,i),with_kwargs=True))

    @contextmanager
    def activate(self,method,ratio,mask,features=None,fixed=None):
        assert not self.enabled and mask.shape[0]==1 and mask.any()
        self.enabled=True;self.method=method;self.ratio=ratio;self.mask=mask
        self.features=features;self.fixed=fixed;self.audit=[];self.replay=[]
        try:yield self
        finally:self.enabled=False;self.features=None

    def start(self,module,args,kw):
        if not self.enabled:return
        cache=kw.get('past_key_values');self.prefill=cache is None or cache.get_seq_length()==0
        if not self.prefill:return
        h=kw.get('inputs_embeds');assert h is not None
        self.indices=torch.arange(h.shape[1],device=h.device);self.current_mask=self.mask[0].clone()
        self.original_n=int(self.mask.sum());self.rater_original=None
        if self.method=='sparsevlm' and self.ratio<1:
            visual=self.current_mask.nonzero().flatten();text=(~self.current_mask).nonzero().flatten()
            trailing=text[text>visual[-1]];assert len(trailing),'Missing trailing question'
            affinity=h[:,visual]@h[:,trailing].transpose(1,2)
            importance=affinity.softmax(-1).mean(1)[0]
            raters=(importance>importance.mean()).nonzero().flatten()
            if not len(raters):raters=importance.argmax().view(1)
            self.rater_original=trailing[raters]
            self.targets=stage_budgets(self.original_n,len(self.text.layers),self.ratio,self.stages)

    def shrink(self,h,chosen,values,layer,phase):
        visual=self.current_mask.nonzero().flatten();nonvisual=(~self.current_mask).nonzero().flatten()
        # chosen is in current sequence coordinates. Merged rows inherit their
        # center's original positions; text, delimiters and timestamps survive.
        retained=torch.cat((nonvisual,chosen)).sort().values
        result=h.clone()
        if values is not None:result[:,chosen]=values[None]
        result=result[:,retained]
        self.audit.append(dict(layer=layer,phase=phase,before_visual=len(visual),after_visual=len(chosen),selected_original_positions=self.indices[chosen].tolist()))
        self.replay.append(dict(layer=layer,phase=phase,retained=retained.detach().clone(),chosen=chosen.detach().clone(),values=None if values is None else values.detach().clone()))
        self.indices=self.indices[retained];self.current_mask=self.current_mask[retained]
        return result

    def pre(self,i,layer,args,kw):
        if not self.enabled or not self.prefill or self.ratio>=1:return
        h=kw.get('hidden_states',args[0] if args else None);kw=dict(kw)
        visual=self.current_mask.nonzero().flatten()
        if self.method in ('visionzip','zoo','divprune') and i==0:
            if self.fixed is not None:
                entry=next(e for e in self.fixed if e['layer']==i and e['phase']=='pre')
                chosen,values=entry['chosen'],entry['values']
            else:
                k=visual_budget(len(visual),self.ratio)
                if self.method=='divprune':
                    from baselines.llava_hf_baselines import _divprune_select_tokens
                    selected=_divprune_select_tokens(h[0,visual],k);values=None
                else:selected,values=self.features.select(self.method,h[0,visual],k,self.ratio)
                chosen=visual[selected]
            h=self.shrink(h,chosen,values,i,'pre')
        elif self.method=='fastv' and i==self.first:
            if self.fixed is not None:chosen=next(e for e in self.fixed if e['layer']==i)['chosen']
            else:chosen=visual[self.last_scores.topk(visual_budget(len(visual),self.ratio)).indices].sort().values
            h=self.shrink(h,chosen,None,i,'pre')
        if len(self.indices)!=self.mask.shape[1]:
            kw['position_embeddings']=tuple(x.index_select(-2,self.indices) for x in kw['position_embeddings'])
            if kw.get('position_ids') is not None:kw['position_ids']=kw['position_ids'].index_select(-1,self.indices)
            attention=kw.get('attention_mask')
            if attention is not None:
                assert attention.ndim in (2,4)
                attention=attention.index_select(-1,self.indices)
                if attention.ndim==4:attention=attention.index_select(-2,self.indices)
                kw['attention_mask']=attention
        visual=self.current_mask.nonzero().flatten()
        if self.fixed is None:
            if self.method=='fastv' and i==self.first-1:
                text=(~self.current_mask).nonzero().flatten()
                self.last_scores=visual_attention_scores(layer,h,kw['position_embeddings'],text[-1:],visual)
            elif self.method=='sparsevlm' and i in self.stages:
                queries=torch.searchsorted(self.indices,self.rater_original)
                assert torch.equal(self.indices[queries],self.rater_original)
                self.last_scores=visual_attention_scores(layer,h,kw['position_embeddings'],queries,visual)
        if args:return (h,*args[1:]),kw
        return args,dict(kw,hidden_states=h)

    def post(self,i,layer,args,kw,output):
        if not self.enabled or not self.prefill or self.ratio>=1 or self.method!='sparsevlm' or i not in self.stages:return
        h=output;visual=self.current_mask.nonzero().flatten();n=len(visual)
        target=min(n,self.targets[self.stages.index(i)])
        if n==target:return
        if self.fixed is not None:
            entry=next(e for e in self.fixed if e['layer']==i and e['phase']=='post');chosen,values=entry['chosen'],entry['values']
        else:
            keep=min(range(target),key=lambda k:abs(k+(int((int((n-k)*.3)+1)/10)+1)-target))
            recycled=target-keep;selected=self.last_scores.topk(keep).indices.sort().values
            dropped=torch.ones(n,dtype=torch.bool,device=h.device);dropped[selected]=False;ids=dropped.nonzero().flatten()
            pool_count=min(len(ids),max(recycled,int(len(ids)*.3)+1));pool=ids[self.last_scores[ids].topk(pool_count).indices]
            merged,centers=cluster_recycle(h[0,visual[pool]],recycled)
            chosen=torch.cat((visual[selected],visual[pool[centers]]));values=torch.cat((h[0,visual[selected]],merged))
            order=chosen.argsort();chosen=chosen[order];values=values[order]
        return self.shrink(h,chosen,values,i,'post')

    def close(self):
        for h in self.handles:h.remove()
