"""Full-student logit distillation with strictly self-only visual attention."""
import os,json,time,math,sys,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from src.model import load_frozen_qwen3vl,prepare_qwen3vl_batch_inputs
from src.qwen_deepstack import disable_qwen_deepstack
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/native_visual_self_distill_20260917'
MODEL='/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
class SelfOnlyVisual:
 def __init__(self,model):
  self.mask=None;self.values={};self.calls=0;self.enabled=True
  for l,b in enumerate(model.model.language_model.layers):
   b.self_attn.register_forward_pre_hook(self.capture(l),with_kwargs=True)
   b.self_attn.o_proj.register_forward_pre_hook(self.replace(l))
 def capture(self,l):
  def hook(m,args,kw):
   h=kw.get('hidden_states',args[0] if args else None)
   if not self.enabled or h.shape[1]==1:return
   # A singleton softmax is exactly 1: visual attention output is its own V.
   v=m.v_proj(h).view(*h.shape[:2],-1,m.head_dim)
   self.values[l]=v.repeat_interleave(m.config.num_attention_heads//v.shape[2],dim=2).reshape(*h.shape[:2],-1)
  return hook
 def replace(self,l):
  def hook(m,args):
   if not self.enabled or args[0].shape[1]==1:return
   v=self.values.pop(l);self.calls+=1
   assert self.mask.shape==args[0].shape[:2]
   return (torch.where(self.mask[:,:,None],v,args[0]),)+args[1:]
  return hook

def select_positions(inputs,answer_mask):
 bi=[];pi=[];labels=[]
 for b in range(len(answer_mask)):
  pos=((inputs['mm_token_type_ids'][b]==0)&inputs['attention_mask'][b].bool()).nonzero().flatten()
  sel=answer_mask[b,1:len(pos)].nonzero().flatten()
  bi.append(torch.full_like(sel,b));pi.append(pos[:-1][sel]);labels.append(inputs['input_ids'][b,pos[1:][sel]])
 return torch.cat(bi),torch.cat(pi),torch.cat(labels)
class Student(torch.nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model;self.intervention=SelfOnlyVisual(model)
 def forward(self,inputs,bi,pi,idx,target):
  self.intervention.mask=inputs['input_ids']==self.model.config.image_token_id
  self.model.model.rope_deltas=None
  hidden=self.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[bi,pi]
  losses=[]
  for start in range(0,len(hidden),128):
   # Checkpoint LM head chunks so full-vocabulary logits are not retained.
   def part(h,indices,t):
    logits=self.model.lm_head(h).float().gather(-1,indices)/2
    return F.kl_div(F.log_softmax(logits,-1),F.softmax(t/2,-1),reduction='none').sum(-1)*4
   losses.append(torch.utils.checkpoint.checkpoint(part,hidden[start:start+128],idx[start:start+128],target[start:start+128],use_reentrant=False))
  return torch.cat(losses).mean()

def train(steps,smoke=False):
 import deepspeed
 from deepspeed.ops.adam import FusedAdam
 torch.set_num_threads(4);rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);deepspeed.init_distributed();torch.manual_seed(44)
 folder=OUT/('smoke' if smoke else 'train');folder.mkdir(exist_ok=True)
 processor,teacher=load_frozen_qwen3vl(MODEL,torch.bfloat16,torch.device('cuda',rank),'flash_attention_2');disable_qwen_deepstack(teacher);teacher.eval()
 _,student=load_frozen_qwen3vl(MODEL,torch.bfloat16,torch.device('cuda',rank),'flash_attention_2');disable_qwen_deepstack(student);student.requires_grad_(True)
 plan=json.loads((OUT/'plan.json').read_text())
 if not plan['train_vision']:student.model.visual.requires_grad_(False)
 # Disabled DeepStack branches cannot receive gradients.
 student.model.visual.deepstack_merger_list.requires_grad_(False)
 student.train();student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});wrapped=Student(student)
 opt=FusedAdam([p for p in wrapped.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),weight_decay=.01)
 conf=json.loads((ROOT/'configs/ds_zero2.json').read_text());conf.update(train_micro_batch_size_per_gpu=4,gradient_accumulation_steps=1,train_batch_size=32)
 engine,opt,_,_=deepspeed.initialize(model=wrapped,optimizer=opt,config=conf)
 data=[json.loads(s) for s in (ROOT/'data/train/pixmo/pixmo_ama_full_valid.clean.jsonl').read_text().splitlines()]
 areas=json.loads((ROOT/'data/train/pixmo/pixmo_ama_full_valid.clean.jsonl.pixel_areas.json').read_text());areas=areas['areas'] if isinstance(areas,dict) else areas
 sized=sorted((x,i) for i,x in enumerate(areas));buckets=[[i for _,i in sized[j:j+512]] for j in range(0,len(sized),512)];g=torch.Generator().manual_seed(44);order=[]
 for j in torch.randperm(len(buckets),generator=g).tolist():
  bucket=buckets[j];order.extend(bucket[i] for i in torch.randperm(len(bucket),generator=g).tolist())
 assert len(order)>=steps*32
 wb=None
 if rank==0:
  (folder/'setup.json').write_text(json.dumps({'trainable_params':sum(p.numel() for p in student.parameters() if p.requires_grad),'sample_order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),'train_vision':plan['train_vision']},indent=2))
  if not smoke:
   import wandb
   wb=wandb.init(project='vision-kv-inject',name='native_visual_self_only_full_distill_2000_20260917',config=plan,mode='online',dir=str(OUT));(OUT/'wandb.json').write_text(json.dumps({'url':wb.url}))
 start=time.time()
 for step in range(steps):
  ids=order[step*32+rank*4:step*32+rank*4+4];inputs,_,am,_=prepare_qwen3vl_batch_inputs(processor,[data[i] for i in ids],ROOT/'data/train/pixmo',torch.device('cuda',rank),include_answers=True)
  bi,pi,labels=select_positions(inputs,am);assert len(labels)>0
  with torch.no_grad():
   teacher.model.rope_deltas=None;h=teacher.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[bi,pi];inds=[];vals=[]
   for j in range(0,len(h),128):
    logits=teacher.lm_head(h[j:j+128]).float();idx=logits.topk(1024,-1).indices;lab=labels[j:j+128,None];idx=torch.where(idx.eq(lab).any(-1,keepdim=True),idx,torch.cat((idx[:,:-1],lab),-1));inds.append(idx);vals.append(logits.gather(-1,idx))
   idx=torch.cat(inds);target=torch.cat(vals);del h,logits
  loss=engine(inputs,bi,pi,idx,target);assert torch.isfinite(loss)
  engine.backward(loss)
  if step==0:
   # Verify language and (when enabled) vision receive actual gradients via ZeRO.
   from deepspeed.utils import safe_get_full_grad
   checks={}
   for name,p in [('language_v',student.model.language_model.layers[17].self_attn.v_proj.weight),('vision',next(student.model.visual.parameters()))]:
    if p.requires_grad:
     grad=safe_get_full_grad(p);assert grad is not None and torch.isfinite(grad).all() and grad.float().norm()>0,name;checks[name]=float(grad.float().norm())
   if rank==0:(folder/'gradient_check.json').write_text(json.dumps(checks))
  warm=60;mult=(step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,2000-warm)))
  for pg in engine.optimizer.param_groups:pg['lr']=5e-5*mult
  engine.step()
  if rank==0 and (step==0 or (step+1)%5==0 or smoke):
   row={'step':step+1,'loss':float(loss.detach()),'lr':5e-5*mult,'elapsed':time.time()-start};print(json.dumps(row),flush=True)
   with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if wb:wb.log(row,step=step+1)
  if not smoke and ((step+1)%500==0):
   torch.distributed.barrier()
   if rank==0:
    dest=OUT/f'checkpoint-{step+1}';student.save_pretrained(dest,safe_serialization=True);processor.save_pretrained(dest);(dest/'SELF_ONLY_REQUIRED.json').write_text(json.dumps({'deepstack':False,'visual_attention':'self_only','requires_hook':'src.native_visual_self_distill.SelfOnlyVisual'}))
   torch.distributed.barrier()
 if wb:wb.finish()
 torch.distributed.destroy_process_group()
if __name__=='__main__':train(int(sys.argv[1]),len(sys.argv)>2 and sys.argv[2]=='smoke')
