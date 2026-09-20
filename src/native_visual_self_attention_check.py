"""Remove visual->other-visual attention edges, renormalizing over prefix text+self."""
import json,os,sys,time,subprocess,hashlib
from pathlib import Path
import torch
from flash_attn import flash_attn_func
from src.initial_token_mlp_probe import runtime,teacher
from src.visual_cross_token_ablation import prepare
from src.model import qwen_apply_rotary_pos_emb
from src.data import QwenBenchmarkDataset
from src.benchmarks import get_benchmark_spec,score_prediction
from src.eval_benchmarks import generate_teacher_qwen
R=Path(__file__).resolve().parents[1];O=R/'artifacts/diagnostics/native_visual_self_attention_check_20260917'
class Intervention:
 def __init__(self,model):
  self.enabled=False;self.pending={};self.changed=[];self.validate=False;self.errors=[]
  for l,b in enumerate(model.model.language_model.layers):
   b.self_attn.register_forward_pre_hook(self.capture(l),with_kwargs=True)
   b.self_attn.o_proj.register_forward_pre_hook(self.replace(l))
 def capture(self,l):
  def hook(m,args,kw):
   h=kw.get('hidden_states',args[0] if args else None)
   if not self.enabled or h.shape[1]==1:return
   assert h.shape[0]==1
   shape=(*h.shape[:-1],-1,m.head_dim)
   q=m.q_norm(m.q_proj(h).view(shape)).transpose(1,2);k=m.k_norm(m.k_proj(h).view(shape)).transpose(1,2);v=m.v_proj(h).view(shape).transpose(1,2)
   q,k=qwen_apply_rotary_pos_emb(q,k,*kw['position_embeddings']);q=q[0].transpose(0,1);k=k[0].transpose(0,1);v=v[0].transpose(0,1)
   p=self.pos;n=len(p);prefix=int(p[0]);Q=q[p,None].contiguous()
   K=torch.cat((k[:prefix].unsqueeze(0).expand(n,-1,-1,-1),k[p,None]),1).contiguous();V=torch.cat((v[:prefix].unsqueeze(0).expand(n,-1,-1,-1),v[p,None]),1).contiguous()
   keys=torch.arange(len(k),device=p.device)
   allowed=(keys[None,:]<prefix)|(keys[None,:]==p[:,None])
   groups=q.shape[1]//k.shape[1]
   out=torch.nn.functional.scaled_dot_product_attention(q[p].transpose(0,1).unsqueeze(0),k.repeat_interleave(groups,1).transpose(0,1).unsqueeze(0),v.repeat_interleave(groups,1).transpose(0,1).unsqueeze(0),attn_mask=allowed[None,None],dropout_p=0.,scale=float(m.scaling))[0].transpose(0,1)
   if self.validate:
    # Independent explicit sparse-mask reference on first/middle/last visual query.
    ix=torch.tensor([0,n//2,n-1],device=p.device);groups=Q.shape[2]//K.shape[2]
    kk=k.float().repeat_interleave(groups,1).transpose(0,1);vv=v.float().repeat_interleave(groups,1).transpose(0,1);qq=q[p[ix]].float().transpose(0,1)
    logits=qq@kk.transpose(-1,-2)*float(m.scaling);keys=torch.arange(len(k),device=p.device)
    allowed=(keys[None,:]<prefix)|(keys[None,:]==p[ix,None]);logits.masked_fill_(~allowed[None],-torch.inf);ref=(logits.softmax(-1)@vv).transpose(0,1)
    err=float((out[ix].float()-ref).norm()/ref.norm().clamp_min(1e-12));assert err<.02,(l,err);self.errors.append(err)
   self.pending[l]=out.reshape(n,-1)
  return hook
 def replace(self,l):
  def hook(m,args):
   if not self.enabled or args[0].shape[1]==1:return
   x=args[0];y=x.clone();y[0,self.pos]=self.pending.pop(l)
   if self.validate:
    mask=torch.ones(x.shape[1],device=x.device,dtype=torch.bool);mask[self.pos]=False;assert torch.equal(x[:,mask],y[:,mask])
   self.changed.append(l);return (y,)+args[1:]
  return hook

def worker(shard,limit):
 runtime();proc,model=teacher();control=Intervention(model)
 with torch.inference_mode(),(O/f'rows{shard}.jsonl').open('w',buffering=1) as out:
  for name in ['realworldqa']:
   path=R/f'artifacts/diagnostics/channel_native_cache_20260916/{name}_eval.jsonl';ds=QwenBenchmarkDataset(str(path),proc,name);digest=hashlib.sha256(path.read_bytes()).hexdigest()
   for i in range(shard,min(len(ds),limit) if limit else len(ds),8):
    item=ds[i];inputs=prepare(item,model.device);p=(inputs['input_ids'][0]==model.config.image_token_id).nonzero().flatten();assert len(p)>0;assert torch.equal(p,torch.arange(int(p[0]),int(p[-1])+1,device=p.device));control.pos=p
    for mode in ['native','visual_self_only']:
     control.enabled=mode!='native';control.changed=[];control.errors=[];control.validate=i==shard
     _,answer=generate_teacher_qwen(model,proc,**inputs,max_new_tokens=8)
     assert control.changed==([] if mode=='native' else list(range(36)));assert not control.pending
     score=score_prediction(metric=get_benchmark_spec(name).metric,prediction_text=answer,answer=item['answer'],choices=item.get('choices'),question=item['row'].get('question'))
     out.write(json.dumps({'dataset':name,'sample':i,'mode':mode,'answer':answer,**score,'manifest_sha256':digest,'visual_tokens':len(p),'changed_layers':control.changed,'sparse_reference_max_relative_error':max(control.errors,default=0.)})+'\n')
    if i//8%20==0:print(name,i,flush=True)
 (O/f'done{shard}.json').write_text(json.dumps({'complete':True}))

def launch(limit):
 O.mkdir(exist_ok=False);(O/'PROTOCOL.md').write_text('Qwen3-VL-4B native versus visual self-only, all36 LM layers, FA2, DeepStack off in both. Contiguous visual block required. Visual Q can attend to preceding text and itself; all other visual keys removed BEFORE softmax (renormalized). Text Q unchanged, causal; native KV cache/decode. No training or adapters. Same 765/1000/1000 manifests, greedy max8. 8 GPUs. Initial E includes encoder mixing. First sample per dataset/GPU/layer checked against explicit sparse-mask FP32 reference; nonvisual attention-output rows bitwise preserved before o_proj.\n')
 jobs=[];status={'state':'running','started':time.time(),'limit':limit};(O/'status.json').write_text(json.dumps(status))
 try:
  for i in range(8):
   log=(O/f'gpu{i}.log').open('w');p=subprocess.Popen([sys.executable,'-u','-m','src.native_visual_self_attention_check','worker',str(i),str(limit)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
  while any(p.poll() is None for p,_ in jobs):
   if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('worker failed')
   time.sleep(3)
  rows=[json.loads(s) for i in range(8) for s in (O/f'rows{i}.jsonl').read_text().splitlines()];summary={}
  for ds,n in [('realworldqa',765)]:
   n=min(n,limit) if limit else n;summary[ds]={}
   for mode in ['native','visual_self_only']:
    rr=[r for r in rows if r['dataset']==ds and r['mode']==mode];assert len(rr)==n and {r['sample'] for r in rr}==set(range(n));summary[ds][mode]={'samples':n,'accuracy_pct':100*sum(r['score'] for r in rr)/n}
  (O/'results.json').write_text(json.dumps(summary,indent=2));status.update(state='complete',finished=time.time())
 except BaseException as e:status.update(state='failed',error=repr(e));raise
 finally:
  (O/'status.json').write_text(json.dumps(status,indent=2))
  for p,f in jobs:
   if p.poll() is None:p.terminate()
   f.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
 else:launch(32)
