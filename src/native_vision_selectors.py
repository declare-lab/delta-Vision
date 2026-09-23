"""Vision-side evidence for native baseline ports.

CLIP: penultimate-block CLS attention and keys, merge raw features then project.
Qwen: final-block received attention and keys, merge raw spatial groups then
native merger. There is no CLS in Qwen. Zoo perturbs the actual vision projector
input. For LLaVA-Next, apply native unpadding also to raw features/statistics;
keep image-newline structural vectors (counted in the visual budget).
"""
import sys
import torch
from torch.nn import functional as F
from baselines.llava_hf_baselines import _zoo_select_tokens


class VisionEvidence:
    def __init__(self,model,method):
        self.model=model;self.method=method;self.enabled=False;self.inside=False
        self.clip=hasattr(model.model,'vision_tower');self.chunks=[];self.handles=[]
        if self.clip:
            tower=model.model.vision_tower;core=getattr(tower,'vision_model',tower)
            idx=model.config.vision_feature_layer;assert isinstance(idx,int) and idx==-2
            self.handles.append(core.encoder.layers[-2].self_attn.register_forward_pre_hook(self.clip_attention,with_kwargs=True))
            self.handles.append(core.encoder.layers[-2].register_forward_hook(self.clip_hidden,with_kwargs=True))
            self.projector=model.model.multi_modal_projector
            self.handles.append(self.projector.register_forward_pre_hook(self.clip_raw))
        else:
            vision=model.model.visual;self.projector=vision.merger;self.group=vision.spatial_merge_size**2
            self.handles.append(vision.blocks[-1].attn.register_forward_pre_hook(self.qwen_attention,with_kwargs=True))
            self.handles.append(vision.merger.register_forward_pre_hook(self.qwen_raw))

    def reset(self,sizes=None):
        self.enabled=True;self.chunks=[];self.sizes=sizes;self.packed=None

    def clip_attention(self,a,args,kw):
        if not self.enabled or self.inside or self.method!='visionzip':return
        h=kw.get('hidden_states',args[0] if args else None);b,s,_=h.shape
        q=a.q_proj(h).view(b,s,a.num_heads,a.head_dim).transpose(1,2)
        k=a.k_proj(h).view(b,s,a.num_heads,a.head_dim).transpose(1,2)
        self.cls_scores=((q[:,:,:1]@k.transpose(-1,-2))*a.scale).softmax(-1).sum(1)[:,0,1:]
        self.keys=k.mean(1)[:,1:]

    def clip_hidden(self,module,args,kw,out):
        if not self.enabled or self.inside:return
        h=out[0] if isinstance(out,tuple) else out
        self.cls=h[:,0].detach()

    def clip_raw(self,module,args):
        if not self.enabled or self.inside:return
        self.raw=args[0].detach()

    def qwen_attention(self,a,args,kw):
        if not self.enabled or self.inside:return
        h=kw.get('hidden_states',args[0] if args else None);s=h.shape[0]
        cu=kw['cu_seqlens'].tolist();self.last_boundaries=cu
        if self.method!='visionzip':return
        q,k,v=a.qkv(h).reshape(s,3,a.num_heads,-1).permute(1,0,2,3).unbind(0)
        apply=sys.modules[type(a).__module__].apply_rotary_pos_emb_vision
        q,k=apply(q,k,*kw['position_embeddings']);q=q.transpose(0,1);k=k.transpose(0,1)
        scores=torch.zeros(s,device=h.device,dtype=torch.float32)
        for begin,end in zip(cu[:-1],cu[1:]):
            for start in range(begin,end,128):
                logits=q[:,start:min(start+128,end)]@k[:,begin:end].transpose(-1,-2)
                scores[begin:end]+=(logits/(q.shape[-1]**.5)).softmax(-1).mean(0).float().sum(0)
        self.last_scores=scores.view(-1,self.group).mean(-1)
        self.last_keys=k.transpose(0,1).reshape(-1,self.group,a.num_heads,k.shape[-1]).mean(1).mean(1)

    def qwen_raw(self,module,args):
        if not self.enabled or self.inside:return
        raw=args[0].detach();count=raw.shape[0]//self.group
        group_ids=torch.empty(count,device=raw.device,dtype=torch.long)
        for i,(a,b) in enumerate(zip(self.last_boundaries[:-1],self.last_boundaries[1:])):
            assert a%self.group==b%self.group==0;group_ids[a//self.group:b//self.group]=i
        self.chunks.append(dict(raw=raw.reshape(count,-1),raw_dim=raw.shape[-1],groups=group_ids,
            scores=self.last_scores if self.method=='visionzip' else None,keys=self.last_keys if self.method=='visionzip' else None))

    def project(self,x):
        self.inside=True
        try:
            return self.projector(x if self.clip else x.reshape(-1,self.chunks[0]['raw_dim']))
        finally:self.inside=False

    def pack_clip(self):
        raw=self.raw;b,n,d=raw.shape
        groups=torch.arange(b,device=raw.device)[:,None,None].expand(b,n,1)
        structural=torch.zeros(b,n,1,device=raw.device,dtype=raw.dtype)
        if self.method=='visionzip':extra=torch.cat((self.cls_scores[...,None].to(raw.dtype),self.keys.to(raw.dtype)),dim=-1)
        else:extra=torch.zeros(b,n,1,device=raw.device,dtype=raw.dtype)
        combined=torch.cat((raw,extra,groups.to(raw.dtype),structural),-1)
        if hasattr(self.model.model,'image_newline'):
            assert self.sizes is not None and self.sizes.shape[0]==1
            newline=torch.zeros(combined.shape[-1],device=raw.device,dtype=raw.dtype);newline[-1]=1;newline[-2]=-1
            packed,_=self.model.model.pack_image_features([combined],self.sizes,'default',image_newline=newline)
            joined=packed[0]
        else:assert b==1;joined=combined[0]
        return dict(raw=joined[:,:d],scores=joined[:,d],keys=joined[:,d+1:-2],groups=joined[:,-2].long(),structural=joined[:,-1].bool())

    def evidence(self):
        if self.packed is not None:return self.packed
        if self.clip:data=self.pack_clip()
        else:
            assert self.chunks
            group_pieces=[];offset=0
            for c in self.chunks:group_pieces.append(c['groups']+offset);offset+=int(c['groups'].max())+1
            data=dict(raw=torch.cat([c['raw'] for c in self.chunks]),groups=torch.cat(group_pieces))
            data['structural']=torch.zeros(len(data['raw']),device=data['raw'].device,dtype=torch.bool)
            if self.method=='visionzip':data.update(scores=torch.cat([c['scores'] for c in self.chunks]),keys=torch.cat([c['keys'] for c in self.chunks]))
        self.packed=data;return data

    def sensitivity(self,raw):
        # Same random direction shared across tokens, 64 directions, +/- .01.
        direction=torch.randn(64,raw.shape[-1],device=raw.device,dtype=raw.dtype)
        direction=direction/(direction.norm(dim=-1,keepdim=True)+1e-12)
        result=[]
        for u in direction.split(4):
            plus=raw[None]+.01*u[:,None];minus=raw[None]-.01*u[:,None]
            a=self.project(plus.reshape(-1,raw.shape[-1])).reshape(len(u),len(raw),-1)
            b=self.project(minus.reshape(-1,raw.shape[-1])).reshape(len(u),len(raw),-1)
            result.append((a-b).float().norm(dim=-1)/.02)
        return torch.cat(result).mean(0)

    def select(self,method,projected,budget,ratio):
        data=self.evidence();raw=data['raw'];assert len(raw)==len(projected),(raw.shape,projected.shape)
        structural=data['structural'];fixed=structural.nonzero().flatten()
        candidates=(~structural).nonzero().flatten();available=budget-len(fixed)
        assert available>0,(budget,len(fixed),'Budget below preserved structural-newline count')
        if method=='zoo':
            importance=self.sensitivity(raw[candidates]);selected=candidates[_zoo_select_tokens(projected[candidates],importance,available)]
            indices=torch.cat((fixed,selected)).sort().values
            return indices,None
        # Allocate the exact global budget over native image/frame/crop groups.
        # Never merge contextual information across independent media groups.
        ids=data['groups'][candidates].unique(sorted=True);sizes=torch.tensor([int((data['groups']==g).sum()) for g in ids],device=raw.device)
        minimum=2 if self.clip else 1
        assert available>=len(ids)*minimum,(available,len(ids),'Not enough budget for native groups')
        quotas=torch.full_like(sizes,minimum);room=sizes-quotas
        remaining=available-int(quotas.sum());ideal=remaining*room.float()/room.sum().clamp_min(1)
        quotas+=ideal.floor().long();remaining=available-int(quotas.sum())
        if remaining:quotas[(ideal-ideal.floor()).topk(remaining).indices]+=1
        chosen=[fixed];values=[projected[fixed]]
        for group,k in zip(ids.tolist(),quotas.tolist()):
            indices=(data['groups']==group).nonzero().flatten();n=len(indices)
            # Native CLIP CLS is included in the total budget, not extra.
            cls_count=1 if self.clip else 0
            contextual=min(k-cls_count,max(0,round(k*.2)));dominant=k-contextual-cls_count
            scores=data['scores'][indices];top=indices[scores.topk(dominant).indices].sort().values
            selected_mask=torch.zeros(len(raw),device=raw.device,dtype=torch.bool);selected_mask[top]=True
            remaining_ids=indices[~selected_mask[indices]]
            this_indices=[top];this_values=[projected[top]]
            if contextual:
                keys=F.normalize(data['keys'][remaining_ids],dim=-1)
                targets=torch.arange(0,len(keys),max(1,len(keys)//contextual),device=raw.device)[:contextual]
                source_mask=torch.ones(len(keys),device=raw.device,dtype=torch.bool);source_mask[targets]=False
                assignment=(keys[source_mask]@keys[targets].T).argmax(-1)
                onehot=F.one_hot(assignment,num_classes=contextual).to(raw.dtype)
                counts=onehot.sum(0).clamp_min(1)[:,None]
                merged=raw[remaining_ids[targets]]+(onehot.T.float()@raw[remaining_ids[source_mask]].float()/counts.float()).to(raw.dtype)
                this_indices.append(remaining_ids[targets]);this_values.append(self.project(merged))
                selected_mask[remaining_ids[targets]]=True
            if self.clip:
                # Native CLIP has no LM slot for CLS. Anchor the added CLS at
                # the first discarded patch; keep original text coordinates.
                anchor=indices[~selected_mask[indices]][:1];assert len(anchor)==1
                this_indices.append(anchor);this_values.append(self.project(self.cls[group:group+1]))
            chosen.extend(this_indices);values.extend(this_values)
        chosen=torch.cat(chosen);values=torch.cat(values);order=chosen.argsort()
        assert len(chosen)==budget and chosen.unique().numel()==budget
        return chosen[order],values[order]

    def close(self):
        for h in self.handles:h.remove()
