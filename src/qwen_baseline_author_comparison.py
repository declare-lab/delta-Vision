"""Explicit Qwen adaptations for a paired audit, not silently changed baselines.

FastV: unchanged; previous-layer last-query attention already matches the reference.
VisionZip: retain Qwen's native final vision features / received-attention proxy
(Qwen has no CLS), but merge spatial feature groups BEFORE the nonlinear merger.
Zoo: perturb the input of the native vision merger, not the language V projection.
SparseVLM v1: initial-embedding text raters, causal attention scores, prune AFTER
layers 2/6/15, and density-peak recycling. Rescale the author's [303,110,36]
stage profile to the requested mean visual budget over layers 3..L-1, including
recycled tokens. Preserve original Qwen M-RoPE using each cluster's center.
"""
import functools
import types
import torch
from baselines.multimodal_pruning_utils import visual_budget


def sparse_budgets(n, layers=36, retention=.2):
    widths = (4, 9, layers-16)
    total = round(n * sum(widths) * retention)
    profile = (303, 110, 36)
    scale = total / sum(w*p for w,p in zip(widths,profile))
    ideal = [scale*p for p in profile]
    # Exact integer layer sum where feasible; otherwise nearest achievable sum.
    candidates=[]
    for a in range(max(1,round(ideal[0])-12), min(n,round(ideal[0])+12)+1):
        for b in range(max(1,round(ideal[1])-12), min(a,round(ideal[1])+12)+1):
            c=max(1,min(b,round((total-4*a-9*b)/widths[2])))
            actual=4*a+9*b+widths[2]*c
            candidates.append(((abs(actual-total),sum((v-t)**2 for v,t in zip((a,b,c),ideal))), (a,b,c)))
    return min(candidates)[1]


def cluster_recycle(x, count):
    """Author v1 density-peak assignment and uniform merge, plus center IDs."""
    n,d=x.shape
    count=min(count,n)
    # Keep the same subtraction/norm arithmetic, but bound the temporary
    # [query, key, hidden] allocation. Full broadcasting can exceed 128 GiB.
    distance=torch.empty(n,n,device=x.device,dtype=x.dtype)
    rows=max(1,min(n,(128*1024*1024)//max(1,n*d*x.element_size())))
    for start in range(0,n,rows):
        distance[start:start+rows]=(x[start:start+rows,None,:]-x[None,:,:]).norm(dim=-1)/(d**.5)
    density=(-(distance.topk(count,dim=-1,largest=False).values**2).mean(-1)).exp()
    density=density+torch.rand_like(density)*1e-6
    higher=(density[None,:]>density[:,None]).to(x.dtype)
    delta=(distance*higher+distance.max()*(1-higher)).min(-1).values
    centers=(delta*density).topk(count).indices
    assignment=distance[centers].argmin(0)
    assignment[centers]=torch.arange(count,device=x.device)
    weights=torch.zeros(count,device=x.device,dtype=x.dtype)
    weights.index_add_(0,assignment,torch.ones(n,device=x.device,dtype=x.dtype))
    normalized=1/(weights[assignment]+1e-6)
    result=torch.zeros(count,d,device=x.device,dtype=x.dtype)
    result.index_add_(0,assignment,x*normalized[:,None])
    return result,centers


class AuthorSparse:
    def __init__(self, model):
        self.text=model.model.language_model
        self.active=False
        self.handles=[self.text.register_forward_pre_hook(self.start,with_kwargs=True)]
        for i,layer in enumerate(self.text.layers):
            self.handles.append(layer.register_forward_pre_hook(functools.partial(self.pre,i),with_kwargs=True))
            self.handles.append(layer.register_forward_hook(functools.partial(self.post,i),with_kwargs=True))

    def start(self,module,args,kw):
        if not self.active:return
        cache=kw.get('past_key_values')
        self.prefill=cache is None or cache.get_seq_length()==0
        if not self.prefill:return
        h=kw['inputs_embeds'];mask=kw['visual_pos_masks'][0]
        self.visual=mask.nonzero().flatten()
        assert len(self.visual) and torch.equal(self.visual,torch.arange(self.visual[0],self.visual[-1]+1,device=h.device))
        self.original_n=int(len(self.visual));self.n=self.original_n
        self.start_index=int(self.visual[0]);self.end=self.start_index+self.n
        self.indices=torch.arange(h.shape[1],device=h.device)
        self.targets=sparse_budgets(self.n,len(self.text.layers),self.retention)
        # Exactly the author's text-rater criterion, on initial embeddings.
        affinity=h[:,self.visual]@h[:,self.end:].transpose(1,2)
        importance=affinity.softmax(-1).mean(1)[0]
        raters=(importance>importance.mean()).nonzero().flatten()
        if not len(raters):raters=importance.argmax().view(1)
        self.raters=raters # relative to start of the trailing text block
        self.audit=[]

    def pre(self,i,layer,args,kw):
        if not self.active or not self.prefill:return
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        h=kw.get('hidden_states',args[0] if args else None)
        kw=dict(kw)
        kw['position_embeddings']=tuple(p.index_select(-2,self.indices) for p in kw['position_embeddings'])
        if kw.get('position_ids') is not None:kw['position_ids']=kw['position_ids'].index_select(-1,self.indices)
        assert kw.get('attention_mask') is None, 'Native unpadded FA2 required'
        if i in (2,6,15):
            norm=layer.input_layernorm(h);a=layer.self_attn;shape=(*norm.shape[:-1],-1,a.head_dim)
            q=a.q_norm(a.q_proj(norm).view(shape)).transpose(1,2)
            k=a.k_norm(a.k_proj(norm).view(shape)).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*kw['position_embeddings'])
            k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
            raters=self.raters+self.start_index+self.n
            logits=(q[:,:,raters]@k.transpose(-1,-2))*a.scaling
            causal=torch.arange(h.shape[1],device=h.device)[None,:]>raters[:,None]
            logits=logits.masked_fill(causal[None,None],-float('inf'))
            self.scores=logits.softmax(-1,dtype=torch.float32).mean(1)[0,:,self.start_index:self.start_index+self.n].mean(0)
        return args,kw

    def post(self,i,layer,args,kw,output):
        if not self.active or not self.prefill or i not in (2,6,15):return
        h=output;start=self.start_index;n=self.n;target=min(n,self.targets[(2,6,15).index(i)])
        if target==n:return
        # Include recycled tokens in the exact stage budget. The reference
        # recycles top 30% of dropped tokens into ~one center per ten tokens.
        keep=min(range(target),key=lambda k:abs(k+(int((int((n-k)*.3)+1)/10)+1)-target))
        merged_count=target-keep
        selected=self.scores.topk(keep).indices.sort().values
        dropped=torch.ones(n,dtype=torch.bool,device=h.device);dropped[selected]=False
        dropped_ids=dropped.nonzero().flatten()
        pool_count=min(len(dropped_ids),max(merged_count,int(len(dropped_ids)*.3)+1))
        pool=dropped_ids[self.scores[dropped_ids].topk(pool_count).indices]
        merged,centers=cluster_recycle(h[0,start+pool],merged_count)
        visual_hidden=torch.cat((h[0,start+selected],merged),0)
        # Assign merged features their center's original M-RoPE coordinates.
        anchors=torch.cat((selected,pool[centers])).sort()
        visual_hidden=visual_hidden[anchors.indices]
        retained=torch.cat((torch.arange(start,device=h.device),start+anchors.values,
                            torch.arange(start+n,h.shape[1],device=h.device)))
        result=torch.cat((h[:,:start],visual_hidden[None],h[:,start+n:]),1)
        self.indices=self.indices[retained]
        self.n=target
        self.audit.append(dict(after_layer=i,before_visual=n,after_visual=target,
                               selected_visual=keep,recycled_visual=merged_count))
        return result


class AuthorCorrections:
    def __init__(self,model,method,module):
        self.model,self.method,self.module=model,method,module
        self.active=False;self.raw=None;self.handles=[]
        if method=='sparsevlm':self.sparse=AuthorSparse(model)
        if method in ('visionzip','zoo'):
            self.handles.append(model.model.visual.merger.register_forward_pre_hook(self.capture))
        if method=='zoo':
            self.original_sensitivity=module._zoo_token_sensitivity
            module._zoo_token_sensitivity=self.sensitivity
        if method=='visionzip':
            # Bind the already-audited model forward, changing ONLY where the
            # contextual feature merge occurs. Keep Qwen's attention proxy.
            import inspect,textwrap
            source=textwrap.dedent(inspect.getsource(type(model.model).forward))
            # Decorators refer to module globals; remove them before compilation.
            source=source[source.index('def forward('):]
            marker='contextual_tokens = target_hidden + aggregated_hidden'
            assert source.count(marker)==1
            source=source.replace(marker,'contextual_tokens = self._author_contextual_merge(contextual_mask, target_indices, assign_one_hot, counts, target_hidden + aggregated_hidden)')
            env=dict(vars(module));exec(compile(source,'author_visionzip_forward','exec'),env)
            model.model.forward=types.MethodType(env['forward'],model.model)
            model.model._author_contextual_merge=self.merge_contextual

    def capture(self,module,args):
        # Recursive merger calls must not overwrite the original vision features.
        if not getattr(self,'inside',False):self.raw=args[0].detach()

    def set_variant(self,variant,retention=.2):
        self.active=variant=='after';self.raw=None
        if self.method=='sparsevlm':
            self.sparse.active=self.active;self.sparse.retention=retention
            self.model.model.language_model.config.sparse_config=None if self.active else self.before_sparse

    def sensitivity(self,features,layer,num_refine,noise_scale):
        if not self.active:return self.original_sensitivity(features,layer,num_refine,noise_scale)
        assert self.raw is not None
        merger=self.model.model.visual.merger
        count=features.shape[0];raw=self.raw.reshape(count,-1);dim=raw.shape[-1]
        direction=torch.randn(num_refine,dim,device=raw.device,dtype=raw.dtype)
        direction=direction/(direction.norm(dim=-1,keepdim=True)+1e-12)
        parts=[];self.inside=True
        try:
            for u in direction.split(4):
                plus=(raw[None]+noise_scale*u[:,None]).reshape(-1,self.raw.shape[-1])
                minus=(raw[None]-noise_scale*u[:,None]).reshape(-1,self.raw.shape[-1])
                a=merger(plus).reshape(len(u),count,-1)
                b=merger(minus).reshape(len(u),count,-1)
                parts.append((a-b).float().norm(dim=-1)/(2*noise_scale))
        finally:self.inside=False
        return torch.cat(parts).mean(0)

    def merge_contextual(self,mask,targets,assignment,counts,old):
        if not self.active:return old
        assert self.raw is not None
        group=self.raw.reshape(len(mask),-1)
        remaining=group[mask][None]
        source=remaining[:,~torch.isin(torch.arange(remaining.shape[1],device=remaining.device),targets)]
        aggregate=torch.bmm(assignment.transpose(1,2).float(),source.float())/counts.float()
        merged=(remaining[:,targets]+aggregate.to(remaining.dtype)).reshape(-1,self.raw.shape[-1])
        self.inside=True
        try:return self.model.model.visual.merger(merged)[None]
        finally:self.inside=False
