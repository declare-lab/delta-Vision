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
OUT=ROOT/'artifacts/diagnostics/postnorm_late_sink_20260916'
LS=tuple(range(28,36));KEYS=tuple(f'norm_{l}' for l in LS)
def worker(shard):
 runtime();processor,model=teacher();saved=torch.load(P/'final.pt',map_location='cpu',weights_only=False,mmap=True)
 bank=Bank(saved['stats'],'zero',keys=KEYS).cuda().eval();bank.load_state_dict({k:v for k,v in saved['bank'].items() if k.split('.')[1] in KEYS});del saved
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
   for i in range(shard,len(ds),8):
    inp=_to_device_item(ds[i],model.device);inp={k:v for k,v in inp.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')}
    h.clear();z.clear();rope.clear();model.model.rope_deltas=None;model.model(**inp,use_cache=False)
    vis=(inp['mm_token_type_ids'][0]==1).nonzero().flatten();text=((inp['mm_token_type_ids'][0]==0)&(torch.arange(inp['input_ids'].shape[1],device='cuda')>vis[-1])).nonzero().flatten();E=h[0][0,vis];metrics={}
    for l in LS:
     block=layers[l];a=block.self_attn;shape=(*z[l].shape[:-1],-1,a.head_dim)
     q=a.q_norm(a.q_proj(z[l]).view(shape)).transpose(1,2);k=a.k_norm(a.k_proj(z[l]).view(shape)).transpose(1,2);q,k=qwen_apply_rotary_pos_emb(q,k,*rope[l]);k=k.repeat_interleave(q.shape[1]//k.shape[1],1)
     incoming=torch.zeros(len(vis),device='cuda',dtype=torch.float32)
     keys=torch.arange(z[l].shape[1],device='cuda')
     for tq in text.split(32):
      logits=(q[0,:,tq].float()@k[0].float().transpose(-1,-2))*float(a.scaling);logits.masked_fill_(keys[None,None,:]>tq[None,:,None],-torch.inf);incoming+=logits.softmax(-1)[:,:,vis].sum((0,1))
     incoming/=q.shape[1]*len(text);sink=int(incoming.argmax())
     target=z[l][0,vis].float();pred=block.input_layernorm.forward(E).float()+bank.heads[f'norm_{l}'](E)
     err=(pred.double()-target.double()).square().mean(-1);cos=F.cosine_similarity(pred.double(),target.double(),dim=-1)
     norm=h[l][0,vis].float().norm(dim=-1);outliers=norm>10*norm.median();outidx=int(norm.argmax())
     retain=torch.ones(len(vis),device='cuda',dtype=torch.bool);retain[sink]=False
     nonout=~outliers;total=err.sum();energy=target.double().square().mean()
     metrics[str(l)]={'mse':float(err.mean()),'cosine':float(cos.mean()),'target_mean_square':float(energy),'normalized_mse':float(err.mean()/energy),'top1_error_share':float(err.max()/total), 'sink_error_share':float(err[sink]/total),'sink_full_attention':float(incoming[sink]),'sink_visual_share':float(incoming[sink]/incoming.sum()),'sink_token_fraction':1/len(vis),'mse_without_top_attention_token':float(err[retain].mean()),'cosine_without_top_attention_token':float(cos[retain].mean()),'outlier_token_fraction':float(outliers.float().mean()),'outlier_error_share':float(err[outliers].sum()/total),'mse_without_norm_outliers':float(err[nonout].mean()),'cosine_without_norm_outliers':float(cos[nonout].mean()),'max_norm_ratio':float(norm.max()/norm.median()),'max_norm_error_share':float(err[outidx]/total),'max_norm_visual_attention_rank':int((incoming>incoming[outidx]).sum())+1}
    log.write(json.dumps({'benchmark':b,'sample':i,'visual_tokens':len(vis),'metrics':metrics})+'\n')
    if i//8%30==0:print(b,i,flush=True)
 (OUT/f'done{shard}.json').write_text(json.dumps({'complete':True}))

def launch():
 OUT.mkdir(exist_ok=False);jobs=[]
 try:
  for i in range(8):
   f=(OUT/f'gpu{i}.log').open('w');p=subprocess.Popen([sys.executable,'-u','-m','src.postnorm_late_sink_audit',str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f))
  while any(p.poll() is None for p,_ in jobs):
   if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('Worker failed')
   time.sleep(3)
  assert all(p.returncode==0 for p,_ in jobs)
  rows=[json.loads(s) for i in range(8) for s in (OUT/f'rows{i}.jsonl').read_text().splitlines()];summary={}
  original=json.loads((P/'results.json').read_text());diff=0.
  for b,n in [('realworldqa',765),('mmstar',1000),('sqa',1000)]:
   rr=[r for r in rows if r['benchmark']==b];assert len(rr)==n and {r['sample'] for r in rr}==set(range(n));summary[b]={}
   for l in LS:
    summary[b][str(l)]={k:sum(r['metrics'][str(l)][k] for r in rr)/n for k in rr[0]['metrics'][str(l)]}
    old=next(r for r in original if r['dataset']==b and r['layer']==l)
    diff=max(diff,abs(old['mlp_mse']-summary[b][str(l)]['mse']));assert abs(old['mlp_mse']-summary[b][str(l)]['mse'])<1e-5
  (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'status.json').write_text(json.dumps({'state':'complete','samples':len(rows),'max_mse_repeat_difference':diff}))
  print(json.dumps(summary),flush=True)
 finally:
  for p,f in jobs:
   if p.poll() is None:p.terminate()
   f.close()
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[1]))
 else:launch()
