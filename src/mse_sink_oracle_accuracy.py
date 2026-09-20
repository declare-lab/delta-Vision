"""Diagnostic native-memory restoration, no training or deployable claim."""
import json,torch
from pathlib import Path
from src.initial_token_mlp_probe import runtime,teacher
from src.model import load_qwen_embedding_adapter_checkpoint,qwen_embedding_adapter_logits
from src.data import QwenBenchmarkDataset
from src.visual_channel_rank_grid import _to_device_item
from src.benchmarks import get_benchmark_spec,score_prediction
from src.eval_benchmarks import extract_option_from_text
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'artifacts/experiments/adapter_mse_fourway_20260916';OUT=ROOT/'artifacts/diagnostics/mse_sink_oracle_accuracy_20260916';OUT.mkdir(exist_ok=True)
runtime();processor,model=teacher();model._adapter_attention_implementation='flash_attention_2';adapters={}
for name in ['pre_uniform','pre_norm3']:
 adapters[name],meta=load_qwen_embedding_adapter_checkpoint(EXP/name/'checkpoints/qwen_embedding_adapter_step2000.pt',model.model.language_model,model.device,torch.bfloat16);assert not meta['missing'] and not meta['unexpected']
raw={};enabled=False;vis=None
for l,block in enumerate(model.model.language_model.layers):
 def hook(m,args,kw,l=l):
  if enabled:raw[l]=kw.get('hidden_states',args[0] if args else None)[:,vis].detach().clone()
 block.register_forward_pre_hook(hook,with_kwargs=True)
eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(processor.tokenizer.eos_token_id)
def generate(adapter,memory,inputs):
 original=adapter.all_visual_memories_batched;adapter.all_visual_memories_batched=lambda *a,**kw:memory
 generated=[];current=dict(inputs)
 try:
  for _ in range(8):
   model.model.rope_deltas=None;logits=qwen_embedding_adapter_logits(model,adapter,current,logits_to_keep=1)[0]
   token=int(logits[0,-1].float().argmax());generated.append(token);text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
   if token in eos or extract_option_from_text(text) in ['A','B','C','D']:break
   new=torch.tensor([[token]],device=model.device);current=dict(current,input_ids=torch.cat((current['input_ids'],new),1),attention_mask=torch.cat((current['attention_mask'],torch.ones_like(new)),1),mm_token_type_ids=torch.cat((current['mm_token_type_ids'],torch.zeros_like(new)),1))
  return text
 finally:adapter.all_visual_memories_batched=original
with torch.no_grad(),(OUT/'rows.jsonl').open('w',buffering=1) as log:
 for b in ['realworldqa','mmstar','sqa']:
  ds=QwenBenchmarkDataset(str(ROOT/f'artifacts/diagnostics/channel_native_cache_20260916/{b}_eval.jsonl'),processor,b)
  old={name:{r['sample']:r for f in (EXP/name/'eval').glob('rows*.jsonl') for r in map(json.loads,f.open()) if r['benchmark']==b} for name in adapters}
  for i in sorted(set(round(j*(len(ds)-1)/63) for j in range(64))):
   item=ds[i];inputs=_to_device_item(item,model.device);inputs={k:v for k,v in inputs.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')};vis=(inputs['mm_token_type_ids'][0]==1).nonzero().flatten()
   raw.clear();enabled=True;model.model.rope_deltas=None;model.model(**inputs,use_cache=False);enabled=False
   native=torch.stack([raw[l] for l in range(36)]);norm=native.float().norm(dim=-1);mask=(norm>3*norm.median(dim=-1,keepdim=True).values).unsqueeze(-1);results={}
   for name,adapter in adapters.items():
    memory=adapter.all_visual_memories_batched(raw[0]);variants={'restore_outliers':torch.where(mask,native,memory),'restore_ordinary':torch.where(mask,memory,native)}
    if name=='pre_uniform':variants['restore_all']=native
    results[name]={'original':{'text':old[name][i]['answer'],'score':old[name][i]['score']}}
    if i==0:
     repeat=generate(adapter,memory,inputs);assert repeat==old[name][i]['answer'],(name,b,repeat,old[name][i]['answer'])
    for mode,m in variants.items():
     answer=generate(adapter,m,inputs);score=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=answer,answer=item['answer'],choices=item.get('choices'),question=item['row'].get('question'));results[name][mode]={'text':answer,**score}
   log.write(json.dumps({'benchmark':b,'sample':i,'results':results,'outlier_fraction':float(mask.float().mean())})+'\n')
   if i==0 or i>len(ds)-20:print(b,i,flush=True)
rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines()];summary={}
for b in ['realworldqa','mmstar','sqa']:
 rr=[r for r in rows if r['benchmark']==b];assert len(rr)==64;summary[b]={}
 for name in adapters:
  summary[b][name]={mode:100*sum(r['results'][name][mode]['score'] for r in rr)/64 for mode in rr[0]['results'][name]}
(OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'status.json').write_text(json.dumps({'state':'complete','samples':192,'protocol':'Same64 evenly spaced indices per dataset as fixed-Q audit. Per-layer native raw norm >3×median defines outliers. Teacher visual memory restored at all36 layers, text states propagate along the student trajectory. Diagnostic oracle only.'}));print(summary,flush=True)
