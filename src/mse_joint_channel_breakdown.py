"""Read-only matched held-out channel and token error decomposition."""
import json,importlib.util
from pathlib import Path
import torch
from src.initial_token_mlp_probe import runtime,teacher
from src.model import load_qwen_embedding_adapter_checkpoint
from src.data import QwenBenchmarkDataset
from src.visual_channel_rank_grid import _to_device_item
R=Path('/lustre-data/leijingdi/code/vision-kv-inject');O=R/'artifacts/diagnostics/mse_joint_breakdown_20260917';O.mkdir(exist_ok=True)
S=R/'artifacts/experiments';spec=importlib.util.spec_from_file_location('helper',S/'adapter_mse_joint_20260917/code/src/layerwise_hidden_mse.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
runtime();proc,model=teacher();cap=h.Capture(model)
paths={'uniform':S/'adapter_mse_fourway_20260916/pre_uniform','token':S/'adapter_mse_fourway_20260916/pre_norm3','joint':S/'adapter_mse_joint_20260917/pre_token_channel3'}
adapters={k:load_qwen_embedding_adapter_checkpoint(p/'checkpoints/qwen_embedding_adapter_step2000.pt',model.model.language_model,model.device,torch.bfloat16)[0].eval() for k,p in paths.items()}
results={}
with torch.no_grad(),(O/'rows.jsonl').open('w',buffering=1) as log:
 for dsname in ['realworldqa','mmstar','sqa']:
  ds=QwenBenchmarkDataset(str(R/f'artifacts/diagnostics/channel_native_cache_20260916/{dsname}_eval.jsonl'),proc,dsname)
  ids=torch.linspace(0,len(ds)-1,32).round().long().tolist();acc={};channels={}
  for j,i in enumerate(ids):
   inputs=_to_device_item(ds[i],model.device);inputs={k:v for k,v in inputs.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')};raw=cap.collect(inputs);E=raw[0][0].unsqueeze(0)
   mem={k:a.all_visual_memories_batched(E) for k,a in adapters.items()}
   for l,block in enumerate(model.model.language_model.layers):
    t=raw[l][0];w=h.token_weights(t);cw=h.channel_weights(t);cm=cw<1;tm=w<1;vals={'downweighted_channels':float(cm.sum()),'channel4_weight':float(cw[4]),'outlier_token_fraction':float(tm.float().mean())}
    for group in adapters:
     p=mem[group][l,0]
     for space in ['pre','post']:
      pp,tt=(p,t) if space=='pre' else (block.input_layernorm(p),block.input_layernorm(t))
      err=(pp.float()-tt.float()).square();key=f'{group}_{space}';ch=err.mean(0).cpu();channels.setdefault(f'{l}_{key}',torch.zeros(2560));channels[f'{l}_{key}']+=ch/32
      vals[key+'_mse']=float(err.mean());vals[key+'_downchannel_contrib']=float(err[:,cm].sum()/err.numel());vals[key+'_otherchannel_contrib']=float(err[:,~cm].sum()/err.numel());vals[key+'_outlier_contrib']=float(err[tm].sum()/err.numel());vals[key+'_ordinary_contrib']=float(err[~tm].sum()/err.numel())
      vals[key+'_channel4_contrib']=float(err[:,4].mean()/2560)
    dest=acc.setdefault(str(l),{k:0. for k in vals})
    for k,v in vals.items():dest[k]+=v/32
    log.write(json.dumps({'dataset':dsname,'sample':i,'layer':l,**vals})+'\n')
   del mem;raw.clear();cap.raw={};print(dsname,j+1,flush=True)
  results[dsname]={'samples':32,'indices':ids,'layers':acc,'channels':{k:v.tolist() for k,v in channels.items()}}
  (O/'results.json').write_text(json.dumps(results))
(O/'status.json').write_text(json.dumps({'state':'complete','samples_per_dataset':32,'selection':'evenly spaced fixed indices','no_training':True}))
