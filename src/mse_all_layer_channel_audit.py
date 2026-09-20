"""Read-only paired checkpoint and gradient audit, no optimization of saved models."""
import json,sys,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from src.initial_token_mlp_probe import runtime,teacher
from src.model import load_qwen_embedding_adapter_checkpoint,prepare_qwen3vl_batch_inputs
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'artifacts/experiments/adapter_mse_fourway_20260916';OUT=ROOT/'artifacts/diagnostics/mse_all_layer_channel_audit_20260916';OUT.mkdir(exist_ok=True)
LS=list(range(36))
spec=importlib.util.spec_from_file_location('loss_audit_helpers',EXP/'code/src/layerwise_hidden_mse.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def cosine(a,b):return float(F.cosine_similarity(a.double().reshape(1,-1),b.double().reshape(1,-1)))
def vec(state,l):return torch.cat([state[f'visual_adapter_{name}.{l}.weight'].float().flatten() for name in ['down','up']])
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

results={}
for l in LS:
 target=torch.cat(ys[l],0).float();delta=target-E[0].float();weights=[];out=[]
 for t in ys[l]:
  w=helper.token_weights(t);ordinary=w==1
  weights.append(ordinary.float()/ordinary.sum()/64);out.append(~ordinary)
 tw=torch.cat(weights);out=torch.cat(out);sqrtw=tw.sqrt().unsqueeze(-1)
 results[str(l)]={}
 for group in ['pre_uniform','pre_norm3']:
  results[str(l)][group]={}
  for step in [2000]:
   adapter,meta=load_qwen_embedding_adapter_checkpoint(EXP/group/f'checkpoints/qwen_embedding_adapter_step{step}.pt',model.model.language_model,model.device,torch.bfloat16)
   with torch.no_grad():
    pred=adapter.visual_memory_for_layer(E,l)[0].float();error=(pred-target).square();channel=(error*tw[:,None]).sum(0)
    up=adapter.visual_adapter_up[l].weight.float();basis=torch.linalg.qr(up,mode='reduced').Q
    oracle=delta@basis@basis.T
    lower=float(((delta-oracle).square()*tw[:,None]).sum()/2560)
    metrics={'outlier_token_count':int(out.sum()),'total_token_count':len(out),'ordinary_mse_top_channel':int(channel.argmax()),'ordinary_mse_top_channel_share':float(channel.max()/channel.sum()) if channel.sum()>0 else None,'ordinary_mse':float(channel.mean()),'channel4_ordinary_mse':float(channel[4]),'other_channels_ordinary_mse':float((channel.sum()-channel[4])/2559),'best_fit_with_current_up_basis_mse':lower,'ordinary_mse_top8_channel_share':float(channel.topk(8).values.sum()/channel.sum())}
   if step==2000:
    adapter.requires_grad_(True);pred=adapter.visual_memory_for_layer(E,l)[0];losses={k:[] for k in ['normal','outlier','uniform','weighted']}
    for pp,tt in zip(pred.split(sizes),ys[l]):
     err=(pp.float()-tt.float()).square().mean(-1);w=helper.token_weights(tt);mask=w<1
     losses['normal'].append((err*(~mask)).mean());losses['outlier'].append((err*mask).mean());losses['uniform'].append(err.mean());losses['weighted'].append((err*w).sum()/w.sum())
    grads={}
    for name,values in losses.items():
     gg=torch.autograd.grad(torch.stack(values).mean()/36,[adapter.visual_adapter_down[l].weight,adapter.visual_adapter_up[l].weight],retain_graph=True);grads[name]=[g.float() for g in gg]
    metrics['gradient_split']={}
    for j,part in enumerate(['down','up']):
     nn,oo,uu,ww=[grads[k][j] for k in ['normal','outlier','uniform','weighted']]
     metrics['gradient_split'][part]={'normal_norm':float(nn.norm()),'outlier_norm':float(oo.norm()),'outlier_normal_cosine':cosine(oo.flatten(),nn.flatten()),'uniform_normal_cosine':cosine(uu.flatten(),nn.flatten()),'weighted_normal_cosine':cosine(ww.flatten(),nn.flatten())}
    channelpower=grads['outlier'][1].square().sum(-1);ids=channelpower.topk(8).indices
    metrics['outlier_up_gradient_top_channel']=int(channelpower.argmax()) if channelpower.sum()>0 else None
    metrics['outlier_up_gradient_top_channel_share']=float(channelpower.max()/channelpower.sum()) if channelpower.sum()>0 else None
    metrics['outlier_up_gradient_top8_channel_share']=float(channelpower[ids].sum()/channelpower.sum());metrics['outlier_up_gradient_channel4_share']=float(channelpower[4]/channelpower.sum());metrics['outlier_up_gradient_top8_channels']=ids.tolist()
    del pred,losses,grads,gg
   results[str(l)][group][str(step)]=metrics;print(l,group,step,json.dumps(metrics),flush=True);del adapter
 (OUT/'results.json').write_text(json.dumps(results,indent=2))
(OUT/'status.json').write_text(json.dumps({'state':'complete','samples':64,'selection':'same first2 paired training batches','no_optimizer_retraining':True}))
