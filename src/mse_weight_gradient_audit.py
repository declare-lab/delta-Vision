"""Read-only paired checkpoint and gradient audit, no optimization of saved models."""
import json,sys,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from src.initial_token_mlp_probe import runtime,teacher
from src.model import load_qwen_embedding_adapter_checkpoint,prepare_qwen3vl_batch_inputs
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'artifacts/experiments/adapter_mse_fourway_20260916';OUT=ROOT/'artifacts/diagnostics/mse_weight_gradient_20260916';OUT.mkdir(exist_ok=True)
LS=[15,16,17,20,29,33,35]
spec=importlib.util.spec_from_file_location('loss_audit_helpers',EXP/'code/src/layerwise_hidden_mse.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def cosine(a,b):return float(F.cosine_similarity(a.double().reshape(1,-1),b.double().reshape(1,-1)))
def vec(state,l):return torch.cat([state[f'visual_adapter_{name}.{l}.weight'].float().flatten() for name in ['down','up']])
# Actual checkpoint movement, not a simulated optimizer step.
states={}
for group in ['pre_uniform','pre_norm3']:
 states[group]={step:torch.load(EXP/group/f'checkpoints/qwen_embedding_adapter_step{step}.pt',map_location='cpu',weights_only=False)['state_dict'] for step in [500,1000,1500,2000]}
params={}
for l in range(36):
 a=vec(states['pre_uniform'][2000],l);b=vec(states['pre_norm3'][2000],l)
 upa=states['pre_uniform'][2000][f'visual_adapter_up.{l}.weight'].float();upb=states['pre_norm3'][2000][f'visual_adapter_up.{l}.weight'].float()
 params[str(l)]={'final_parameter_relative_difference':float((a-b).norm()/a.norm().clamp_min(1e-20)),'up_weight_relative_difference':float((upa-upb).norm()/upa.norm().clamp_min(1e-20)),'up_weight_cosine':cosine(upa,upb),'intervals':{}}
 for s,t in [(500,1000),(1000,1500),(1500,2000)]:
  da=vec(states['pre_uniform'][t],l)-vec(states['pre_uniform'][s],l);db=vec(states['pre_norm3'][t],l)-vec(states['pre_norm3'][s],l)
  params[str(l)]['intervals'][f'{s}_{t}']={'update_cosine':cosine(da,db),'update_norm_uniform':float(da.norm()),'update_norm_weighted':float(db.norm()),'update_relative_difference':float((da-db).norm()/da.norm().clamp_min(1e-20))}
(OUT/'parameters.json').write_text(json.dumps(params,indent=2));del states
runtime();processor,model=teacher();cap=helper.Capture(model)
args=json.loads((EXP/'pre_uniform/checkpoints/args.json').read_text());data=[json.loads(x) for x in Path(args['data']).open()]
areas=json.loads(Path(args['pixel_area_cache']).read_text());areas=areas['areas'] if isinstance(areas,dict) else areas
sized=sorted((area,i) for i,area in enumerate(areas));buckets=[[i for _,i in sized[start:start+512]] for start in range(0,len(sized),512)];gen=torch.Generator().manual_seed(44);order=[]
for j in torch.randperm(len(buckets),generator=gen).tolist():
 bucket=buckets[j];perm=torch.randperm(len(bucket),generator=gen).tolist();order.extend(bucket[i] for i in perm)
import hashlib
assert hashlib.sha256(json.dumps(order).encode()).hexdigest()==json.loads((EXP/'pre_uniform/checkpoints/paired_setup_rank0.json').read_text())['sample_order_sha256']
# First two global batches, fixed by original training order, no outlier-based selection.
xs=[];ys={l:[] for l in LS};sizes=[]
for start in range(0,64,4):
 inputs,*_=prepare_qwen3vl_batch_inputs(processor,[data[i] for i in order[start:start+4]],Path(args['image_root']),model.device,include_answers=False)
 raw=cap.collect(inputs);xs+=raw[0];sizes += [len(v) for v in raw[0]]
 for l in LS:ys[l]+=raw[l]
 print('CAPTURE',start+4,flush=True)
E=torch.cat(xs,0).unsqueeze(0);del xs;results={}
for group in ['pre_uniform','pre_norm3']:
 adapter,meta=load_qwen_embedding_adapter_checkpoint(EXP/group/'checkpoints/qwen_embedding_adapter_step2000.pt',model.model.language_model,model.device,torch.bfloat16);assert not meta['missing'] and not meta['unexpected'];adapter.requires_grad_(True)
 results[group]={}
 for l in LS:
  pred=adapter.visual_memory_for_layer(E,l)[0];ls={k:[] for k in ['uniform','weighted','ordinary','outlier']};wlist=[];outcount=0
  for pp,tt in zip(pred.split(sizes),ys[l]):
   err=(pp.float()-tt.float()).square().mean(-1);w=helper.token_weights(tt);mask=w<1;outcount+=int(mask.sum());wlist+=w[mask].tolist()
   ls['uniform'].append(err.mean());ls['weighted'].append((err*w).sum()/w.sum());ls['ordinary'].append((err*(~mask)).mean());ls['outlier'].append((err*mask).mean())
  losses={k:torch.stack(v).mean()/36 for k,v in ls.items()};parameters=[adapter.visual_adapter_down[l].weight,adapter.visual_adapter_up[l].weight];grad={}
  for k,loss in losses.items():
   gg=torch.autograd.grad(loss,parameters,retain_graph=True);grad[k]=torch.cat([v.float().flatten() for v in gg])
  u,w,n,o=[grad[k] for k in ['uniform','weighted','ordinary','outlier']]
  val={'outlier_token_fraction':outcount/sum(sizes),'outlier_weight_mean':sum(wlist)/len(wlist) if wlist else None,'losses':{k:float(v.detach()*36) for k,v in losses.items()},'grad_norms':{k:float(v.norm()) for k,v in grad.items()},'sink_normal_gradient_cosine':cosine(o,n),'uniform_normal_gradient_cosine':cosine(u,n),'weighted_normal_gradient_cosine':cosine(w,n),'uniform_weighted_gradient_cosine':cosine(u,w),'gradient_relative_change':float((u-w).norm()/u.norm().clamp_min(1e-20)), 'gradient_decomposition_relative_error':float((u-n-o).norm()/u.norm().clamp_min(1e-20))}
  results[group][str(l)]=val;print('GRAD',group,l,json.dumps(val),flush=True)
  del pred,ls,losses,grad,u,w,n,o
 del adapter
(OUT/'gradients.json').write_text(json.dumps({'training_samples':64,'selection':'first2 global batches in exact paired training order','checkpoint_step':2000,'results':results},indent=2))
(OUT/'status.json').write_text(json.dumps({'state':'complete','historical_adam_state_available':False,'note':'Checkpoint intervals are real accumulated parameter changes; gradients are controlled current-checkpoint probes, not recovered historical optimizer updates.'}))
