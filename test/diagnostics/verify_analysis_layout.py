"""Numerical replay of paper interventions against an independent source snapshot.

Run --side old/new in fresh processes; the old process imports only the saved
src implementations. Real models, real RWQA input, no checkpoint training run.
"""
import argparse,hashlib,importlib,json,sys
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'artifacts/maintenance/paper_analysis_layout_20260924'
p=argparse.ArgumentParser();p.add_argument('--side',choices=['old','new'],required=True);p.add_argument('--family',choices=['qwen','qwen35'],required=True);a=p.parse_args()
code=AUDIT/'reference' if a.side=='old' else ROOT
sys.path.insert(0,str(code))
if a.family=='qwen35':sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
torch.set_num_threads(4);torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
device=torch.device('cuda:0');dtype=torch.bfloat16
mapping=json.loads((AUDIT/'migration.json').read_text())['mapping']
if a.side=='old':
 for f,h in json.loads((AUDIT/'reference_sha256.json').read_text()).items():
  assert hashlib.sha256((code/f).read_bytes()).hexdigest()==h,f

def mod(name):
 path='src/'+name+'.py'
 return importlib.import_module(('src.'+name) if a.side=='old' else mapping[path][:-3].replace('/','.'))
row=json.loads((ROOT/'data/benchmarks/realworldqa/test.jsonl').read_text().splitlines()[0])
results={};meta={'family':a.family,'side':a.side,'input':'realworldqa[0]','device':torch.cuda.get_device_name(0)}
if a.family=='qwen':
 from src.model_setup import load_frozen_qwen3vl
 from src.model import prepare_qwen3vl_batch_inputs
 processor,model=load_frozen_qwen3vl('/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',dtype,device,'flash_attention_2')
 inputs,_,_,_=prepare_qwen3vl_batch_inputs(processor,[row],ROOT/'data/benchmarks/realworldqa',device,include_answers=False)
 nh=mod('visual_channel_native_cache');hook=nh.NativeVisualHook(model.model.language_model.layers);hook.positions=inputs['mm_token_type_ids'][0].eq(1).nonzero().flatten()
 with torch.inference_mode():
  hook.mode='capture';out=nh.forward(model,inputs,hook);results['native_logits']=out.logits.cpu()
  native={i:t.clone() for i,t in hook.native.items()}
  for rank in [0,32]:
   hook.replacements={}
   for i,t in native.items():
    x=torch.zeros_like(t);x[...,:rank]=t[...,:rank];hook.replacements[i]=x
   hook.mode='replay';out=nh.forward(model,inputs,hook);results[f'channel_rank{rank}_logits']=out.logits.cpu()
 hook.close()
 # Capture postnorm residuals from actual native visual states.
 post=mod('initial_token_postnorm_probe');capture=post.Capture(model)
 x,target,sizes=capture.collect(inputs);capture.close()
 mlp=mod('initial_token_mlp_probe');keys=['norm_0','norm_15','norm_35'];d=x.shape[-1]
 stats={'input':{'mean':torch.zeros(d),'std':torch.ones(d)},'targets':{k:{'mean':torch.zeros(d),'std':torch.ones(d)} for k in keys}}
 torch.manual_seed(44);bank=mlp.Bank(stats,keys=keys).to(device);bank.activation_checkpointing=False
 x=x[:16].detach();target={k:target[k][:16] for k in keys};optim=torch.optim.AdamW(bank.parameters(),lr=3e-4)
 meta['mlp_steps']=[]
 for step in range(2):
  optim.zero_grad(set_to_none=True);loss,_=bank(x,target,[len(x)]);loss.backward()
  gh=hashlib.sha256()
  for param in bank.parameters():gh.update(param.grad.detach().float().cpu().numpy().tobytes())
  meta['mlp_steps'].append({'loss':float(loss.detach()),'gradient_sha256':gh.hexdigest()});optim.step()
  results[f'mlp_step{step}_loss']=loss.detach().cpu()
 for key in keys:results['mlp_'+key]=bank.heads[key](x).detach().cpu()
else:
 from src.qwen35 import load_model,prepare_inputs
 from src.benchmarks import get_benchmark_spec,build_benchmark_prompt
 processor,model,adapter,controller=load_model({'model_path':str(ROOT/'model/Qwen3.5-4B'),'rank':128},device)
 inputs,_=prepare_inputs(processor,row,str(ROOT/'data/benchmarks/realworldqa'),device,question=build_benchmark_prompt(row,get_benchmark_spec('realworldqa')))
 memory=mod('qwen35_memory_probe');full=mod('qwen35_full_attention_ablation');mask=inputs['mm_token_type_ids'].eq(1)
 for case in ['native','boundary0','fa11_15_off']:
  context=memory.state_intervention(model,mask,rank=0) if case=='boundary0' else full.remove_full_attention_visual_effect(model,mask,[11,15]) if case=='fa11_15_off' else nullcontext()
  with torch.inference_mode(),context:
   model.model.rope_deltas=None
   out=model.generate(**inputs,max_new_tokens=8,do_sample=False,return_dict_in_generate=True,output_scores=True)
  results[case+'_tokens']=out.sequences.cpu();results[case+'_logits']=torch.stack(out.scores).cpu()
meta['input_sha256']=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
torch.save(results,AUDIT/f'{a.family}_{a.side}_outputs.pt');(AUDIT/f'{a.family}_{a.side}_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
