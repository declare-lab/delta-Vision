"""Late-layer error concentration vs native visual attention reception."""
import json,os,sys,subprocess,time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.initial_token_postnorm_probe import Bank,teacher,runtime,ROOT
from src.model import qwen_apply_rotary_pos_emb
from src.data import QwenBenchmarkDataset
from src.visual_channel_rank_grid import _to_device_item
P=ROOT/'artifacts/diagnostics/initial_token_postnorm_all36_20260916'
OUT=ROOT/'artifacts/diagnostics/mse_fourway_sink_audit_20260916'
EXP=ROOT/'artifacts/experiments/adapter_mse_fourway_20260916'
LS=(15,16,17,20,28,29,33,35);KEYS=tuple(f'norm_{l}' for l in LS)
def worker(shard):
 runtime();processor,model=teacher()
 from src.model import load_qwen_embedding_adapter_checkpoint
 adapters={}
 for name in ('pre_uniform','pre_norm3'):
  adapters[name],meta=load_qwen_embedding_adapter_checkpoint(EXP/name/'checkpoints/qwen_embedding_adapter_step2000.pt',model.model.language_model,model.device,torch.bfloat16)
  assert not meta['missing'] and not meta['unexpected']
 h={};z={};rope={};handles=[]
 def raw(l):
  def hook(m,args,kw):h[l]=kw.get('hidden_states',args[0] if args else None).detach()
  return hook
 def att(l):
  def hook(m,args,kw):z[l]=kw['hidden_states'].detach();rope[l]=kw['position_embeddings']
  return hook
 layers=model.model.language_model.layers
 for l in (0,*LS):handles.append(layers[l].register_forward_pre_hook(raw(l),with_kwargs=True))
 for l in LS:handles.append(layers[l].self_attn.register_forward_pre_hook(att(l),with_kwargs=True))
 gamma={str(l):{'rms':float(layers[l].input_layernorm.weight.float().square().mean().sqrt()),'max_abs':float(layers[l].input_layernorm.weight.float().abs().max())} for l in LS}
 if shard==0:(OUT/'gamma.json').write_text(json.dumps(gamma,indent=2))
 with torch.no_grad(),(OUT/f'rows{shard}.jsonl').open('w',buffering=1) as log:
  for b in ('realworldqa','mmstar','sqa'):
   ds=QwenBenchmarkDataset(str(P/f'{b}_eval.jsonl'),processor,b)
   for i in sorted(set(round(j*(len(ds)-1)/63) for j in range(64))):
    inp=_to_device_item(ds[i],model.device);inp={k:v for k,v in inp.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')}
    h.clear();z.clear();rope.clear();model.model.rope_deltas=None;model.model(**inp,use_cache=False)
    vis=(inp['mm_token_type_ids'][0]==1).nonzero().flatten();text=((inp['mm_token_type_ids'][0]==0)&(torch.arange(inp['input_ids'].shape[1],device='cuda')>vis[-1])).nonzero().flatten();E=h[0][0,vis];metrics={};predictions={name:adapter.all_visual_memories_batched(E.unsqueeze(0))[:,0] for name,adapter in adapters.items()}
    for l in LS:
     block=layers[l];a=block.self_attn;shape=(*z[l].shape[:-1],-1,a.head_dim)
     q=a.q_norm(a.q_proj(z[l]).view(shape)).transpose(1,2);k=a.k_norm(a.k_proj(z[l]).view(shape)).transpose(1,2);q,k=qwen_apply_rotary_pos_emb(q,k,*rope[l]);k=k.repeat_interleave(q.shape[1]//k.shape[1],1)
     incoming=torch.zeros(len(vis),device='cuda',dtype=torch.float32)
     keys=torch.arange(z[l].shape[1],device='cuda')
     for tq in text.split(32):
      logits=(q[0,:,tq].float()@k[0].float().transpose(-1,-2))*float(a.scaling);logits.masked_fill_(keys[None,None,:]>tq[None,:,None],-torch.inf);incoming+=logits.softmax(-1)[:,:,vis].sum((0,1))
     incoming/=q.shape[1]*len(text);sink=int(incoming.argmax())
     raw=h[l][0,vis].float();norm=raw.norm(dim=-1);outliers=norm>3*norm.median();ordinary=~outliers
     retain=torch.ones(len(vis),device='cuda',dtype=torch.bool);retain[sink]=False
     stats={'outlier_fraction':float(outliers.float().mean()),'outlier_visual_attention_share':float(incoming[outliers].sum()/incoming.sum()),'sink_visual_share':float(incoming[sink]/incoming.sum()),'sink_is_norm_outlier':bool(outliers[sink])}
     for name,predictions_by_layer in predictions.items():
      pred=predictions_by_layer[l]
      for space in ('pre','post'):
       pp,yy=(pred.float(),raw) if space=='pre' else (block.input_layernorm.forward(pred).float(),z[l][0,vis].float())
       err=(pp.double()-yy.double()).square().mean(-1);cos=F.cosine_similarity(pp.double(),yy.double(),dim=-1);total=err.sum()
       stats[name+'_'+space]={'mse':float(err.mean()),'cosine':float(cos.mean()),'outlier_sse_share':float(err[outliers].sum()/total.clamp_min(1e-20)), 'sink_sse_share':float(err[sink]/total.clamp_min(1e-20)), 'ordinary_mse':float(err[ordinary].mean()),'ordinary_cosine':float(cos[ordinary].mean()),'without_sink_mse':float(err[retain].mean()),'without_sink_cosine':float(cos[retain].mean())}
     metrics[str(l)]=stats
    log.write(json.dumps({'benchmark':b,'sample':i,'visual_tokens':len(vis),'metrics':metrics})+'\n')
    if i//8%30==0:print(b,i,flush=True)
 (OUT/f'done{shard}.json').write_text(json.dumps({'complete':True}))

if __name__=='__main__':
 OUT.mkdir(exist_ok=False)
 worker(0)
 rows=[json.loads(line) for line in (OUT/'rows0.jsonl').read_text().splitlines()]
 summary={}
 for b in ('realworldqa','mmstar','sqa'):
  rr=[r for r in rows if r['benchmark']==b];assert len(rr)==64;summary[b]={}
  for l in LS:
   values=[r['metrics'][str(l)] for r in rr];v={}
   for key in values[0]:
    if isinstance(values[0][key],dict):v[key]={k:sum(x[key][k] for x in values)/len(values) for k in values[0][key]}
    else:v[key]=sum(x[key] for x in values)/len(values)
   summary[b][str(l)]=v
 (OUT/'results.json').write_text(json.dumps(summary,indent=2))
 (OUT/'status.json').write_text(json.dumps({'state':'complete','samples':192,'selection':'64 evenly spaced dataset indices per benchmark; no error-based selection'}))
