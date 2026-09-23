"""Validate and execute the remaining native-HF baseline ports."""
import argparse
from contextlib import nullcontext,contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/eval/all_baselines_seed44_20260922_restart'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def prepare(run):
 run.mkdir(parents=True,exist_ok=False)
 for d in ['source','logs','rows']:(run/d).mkdir()
 cfg=json.loads((PARENT/'config.json').read_text());cfg['parent_run']=str(PARENT)
 cfg['ports']={'fastv':'Native last-text-query attention, previous full-attention layer. Hybrid: after3.',
 'sparsevlm':'Initial visual/text raters, causal attention, post-block pruning+recycling. Dense2/6/15; hybrid3/7/15; exact mean budget excludes initial full blocks.',
 'visionzip':'CLIP CLS attention+keys and preprojector merge, CLS counted inside budget. Anyres raw features/statistics natively unpadded; structural newlines preserved and counted. Qwen native final-vision received attention, groupwise premerger merge.',
 'zoo':'Actual visual projector/merger finite differences,64 directions,+/-.01,SADS. Anyres newlines preserved/count in budget.',
 'positions':'Native original positions preserved; merged centers anchor new rows; CLS anchored at a discarded patch.',
 'precision':'BF16 native model; FP32 selection distances and accumulation; no eager replacement of FA2 model attention.'}
 dump(run/'config.json',cfg)
 jobs=[j for j in json.loads((PARENT/'jobs.json').read_text()) if not j['ready']]
 dump(run/'jobs.json',jobs)
 files=list((ROOT/'src').glob('*.py'))+[ROOT/'baselines'/n for n in ['eval_baselines.py','llava_hf_baselines.py','multimodal_pruning_utils.py']]+[Path(__file__).resolve()]
 for method in cfg['methods']:files+=list((ROOT/'baselines'/method/'qwen3_vl').glob('*.py'))
 for p in files:
  dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 shutil.copy2(PARENT/'source/src/scoring_reference.py',run/'source/src/scoring_reference.py')
 dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
 dump(run/'status.json',dict(state='prepared',jobs=len(jobs)))

@contextmanager
def independent_replay(lm,events,sequence_length):
 """Only fixed row gather/replacement and position slicing, no selector code."""
 import torch
 from functools import partial
 state={'indices':None};handles=[]
 def pre(i,module,args,kw):
  h=kw.get('hidden_states',args[0] if args else None)
  if h.shape[1]==1:return
  if i==0:state['indices']=torch.arange(sequence_length,device=h.device)
  for e in events:
   if e['layer']==i and e['phase']=='pre':
    h=h.clone()
    if e['values'] is not None:h[:,e['chosen']]=e['values'][None]
    h=h[:,e['retained']];state['indices']=state['indices'][e['retained']]
  kw=dict(kw);ix=state['indices']
  kw['position_embeddings']=tuple(v.index_select(-2,ix) for v in kw['position_embeddings'])
  if kw.get('position_ids') is not None:kw['position_ids']=kw['position_ids'].index_select(-1,ix)
  assert kw.get('attention_mask') is None,'Unpadded FA2 reference required'
  return ((h,*args[1:]),kw) if args else (args,dict(kw,hidden_states=h))
 def post(i,module,args,kw,out):
  h=out
  if h.shape[1]==1:return
  for e in events:
   if e['layer']==i and e['phase']=='post':
    h=h.clone()
    if e['values'] is not None:h[:,e['chosen']]=e['values'][None]
    h=h[:,e['retained']];state['indices']=state['indices'][e['retained']]
  return h
 for i,l in enumerate(lm.layers):
  handles.append(l.register_forward_pre_hook(partial(pre,i),with_kwargs=True));handles.append(l.register_forward_hook(partial(post,i),with_kwargs=True))
 try:yield
 finally:
  for h in handles:h.remove()

def worker(run,model_name,method,suite,shard,smoke):
 sys.path.insert(0,str(run/'source'))
 import torch
 from src.data import QwenBenchmarkDataset,LlavaBenchmarkDataset
 from src.scoring_reference import score_prediction,get_benchmark_spec,build_benchmark_prompt
 from src.divprune_rerun import input_digest
 from src.native_baseline_ports import NativeVisualPruning,stage_budgets
 from src.native_vision_selectors import VisionEvidence
 from src.qwen_deepstack import disable_qwen_deepstack
 from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item,configure_baseline
 from baselines.llava_hf_baselines import load_llava_baseline_model,build_llava_inputs_embeds_with_image_span
 from src.model import llava_projected_image_features
 from baselines.multimodal_pruning_utils import visual_budget
 cfg=json.loads((run/'config.json').read_text());mi=cfg['models'][model_name];kind=mi['kind']
 for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h,p
 torch.set_num_threads(4);torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
 custom_dart=suite=='multimodal' and method=='dart'
 if kind=='llava':processor,model=load_llava_baseline_model(mi['path'],dtype=torch.bfloat16,device='cuda:0',attn_implementation='flash_attention_2')
 elif kind=='qwen35':
  from src.qwen35_embedding import install_fast_kernels
  from src.qwen35_experiment import prepare_inputs
  from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
  install_fast_kernels();processor=AutoProcessor.from_pretrained(mi['path'],local_files_only=True)
  model=Qwen3_5ForConditionalGeneration.from_pretrained(mi['path'],dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True);disable_qwen_deepstack(model)
 else:model,processor=load_baseline_model('dart' if custom_dart else 'base',mi['path'],torch.bfloat16,'cuda:0',.05,'flash_attention_2')
 model.eval().requires_grad_(False);lm=model.model.language_model
 evidence=VisionEvidence(model,method) if method in ('visionzip','zoo') else None
 controller=None if custom_dart else NativeVisualPruning(model)
 counts={}
 def capture(i):
  def hook(m,args,kw):
   h=kw.get('hidden_states',args[0] if args else None)
   if i not in counts:counts[i]=h.shape[1]
  return hook
 handles=[l.register_forward_pre_hook(capture(i),with_kwargs=True) for i,l in enumerate(lm.layers)]
 tag=f'{suite}__{model_name}__{method}__'+('smoke' if smoke else f'shard{shard}');path=run/'rows'/f'{tag}.jsonl'
 done=set()
 if path.exists():done={(r['benchmark'],r['sample'],r['retention']) for r in map(json.loads,path.read_text().splitlines())}
 datasets=cfg['multimodal'] if suite=='multimodal' else cfg['evaluation'];started=time.time();written=0;checks=[]
 eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 with torch.inference_mode(),path.open('a',buffering=1) as out:
  for bi,(b,info) in enumerate(datasets.items()):
   assert sha(info['path'])==info['sha256'];rows=[json.loads(s) for s in Path(info['path']).read_text().splitlines() if s]
   if suite=='multimodal':
    from src.fixed_multimodal_inputs import FixedMultimodalDataset
    ds=FixedMultimodalDataset(info['path'],processor,b,data_root=str(Path(cfg['original_root'])/'data/benchmarks'/b))
    historical=json.loads((Path(cfg['parent_run']).parent/'divprune_fixed_multimodal_random44_five_models_20260920_112257/multimodal_input_cache.json').read_text())[b]
   elif kind=='qwen35':ds=None
   else:ds=(LlavaBenchmarkDataset if kind=='llava' else QwenBenchmarkDataset)(info['path'],processor,b,data_root=info['image_root'])
   indices=range(1) if smoke else range(shard,len(rows),8)
   for i in indices:
    if all((b,i,r) in done for r in cfg['retentions']):continue
    row=rows[i]
    if kind=='qwen35':
     inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,get_benchmark_spec(b)));digest=input_digest(inputs)
    else:item=ds[i];digest=input_digest(item)
    if suite=='multimodal':assert digest==historical[i]['input_sha256'],(b,i,'Historical processor input mismatch',digest,historical[i]['input_sha256'])
    if kind=='llava':
     ids=item['input_ids'][None].cuda();am=item['attention_mask'][None].cuda();pix=item['pixel_values'][None].cuda();sizes=item.get('image_sizes');sizes=sizes[None].cuda() if torch.is_tensor(sizes) else sizes
     if evidence:evidence.reset(sizes)
     memory=llava_projected_image_features(model,pix,image_sizes=sizes)
     embeds,mask,start,n=build_llava_inputs_embeds_with_image_span(model,input_ids=ids,attention_mask=am,image_token_id=model.config.image_token_index,visual_memory=memory)
     inputs=dict(inputs_embeds=embeds,attention_mask=mask)
     visual_mask=torch.zeros(embeds.shape[:2],device='cuda:0',dtype=torch.bool);visual_mask[:,start:start+n]=True;prefix=0;nt=embeds.shape[1]-n
    else:
     if kind!='qwen35':inputs=_qwen_inputs_from_item(item,torch.device('cuda:0'))
     visual_mask=inputs['mm_token_type_ids'].ne(0);n=int(visual_mask.sum());nt=inputs['input_ids'].shape[1]-n;prefix=inputs['input_ids'].shape[1]
    def reset_evidence():
     if evidence and kind!='llava':evidence.reset()
     if hasattr(model.model,'rope_deltas'):model.model.rope_deltas=None
    if smoke and controller:
     reset_evidence();native=model(**inputs,use_cache=True,logits_to_keep=1).logits
     reset_evidence()
     with controller.activate(method,1.,visual_mask,evidence):same=model(**inputs,use_cache=True,logits_to_keep=1).logits
     torch.testing.assert_close(native,same,rtol=0,atol=0);checks.append(dict(benchmark=b,test='100% native logits',exact=True))
     del native,same
    for ratio in cfg['retentions']:
     if (b,i,ratio) in done:continue
     torch.manual_seed(44+bi*10000+i);counts.clear();reset_evidence();begin=time.time()
     if custom_dart:configure_baseline(model,method,ratio,int(visual_mask[0].nonzero()[0]),n)
     if smoke and controller:
      with controller.activate(method,ratio,visual_mask,evidence):pruned=model(**inputs,use_cache=True,logits_to_keep=1).logits
      events=controller.replay;reset_evidence()
      with independent_replay(lm,events,n+nt):ref=model(**inputs,use_cache=True,logits_to_keep=1).logits
      torch.testing.assert_close(pruned,ref,rtol=0,atol=0)
      checks.append(dict(benchmark=b,ratio=ratio,test='independent native row-hook logits',exact=True));del pruned,ref
      torch.manual_seed(44+bi*10000+i);counts.clear();reset_evidence()
     context=nullcontext() if custom_dart else controller.activate(method,ratio,visual_mask,evidence)
     with context:
      if suite=='multimodal':
       # Retain historical fresh-prefix / standalone-option stopping.
       current=dict(inputs);tokens=[]
       for step in range(info['max_new_tokens']):
        reset_evidence();logits=model(**current,use_cache=False,logits_to_keep=1).logits;token=int(logits[0,-1].argmax());tokens.append(token)
        text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        if token in eos or text in [chr(65+x) for x in range(len(row['choices']))]:break
        new=torch.tensor([[token]],device='cuda:0');current['input_ids']=torch.cat((current['input_ids'],new),1);current['attention_mask']=torch.ones_like(current['input_ids']);current['mm_token_type_ids']=torch.cat((current['mm_token_type_ids'],torch.zeros_like(new)),1)
        if controller:controller.mask=current['mm_token_type_ids'].ne(0)
      else:
       result=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=info['max_new_tokens'])
       tokens=result[0,prefix:].tolist();text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
     actual=[counts[j]-nt for j in range(len(lm.layers))]
     exclude=(4 if kind=='qwen35' else 2) if method in ('fastv','dart') else ((4 if kind=='qwen35' else 3) if method=='sparsevlm' else 0)
     if method=='sparsevlm':
      stages=controller.stages;budgets=stage_budgets(n,len(lm.layers),ratio,stages);expected=[n]*(stages[0]+1)
      for stage,nextstage,k in zip(stages,(*stages[1:],len(lm.layers)-1),budgets):expected.extend([k]*(nextstage-stage))
     else:expected=[n]*exclude+[visual_budget(n,ratio)]*(len(lm.layers)-exclude)
     assert actual==expected,(model_name,method,b,i,ratio,n,actual,expected)
     score=score_prediction(metric='multi_choice' if suite=='multimodal' else get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
     record=dict(model=model_name,method=method,suite=suite,benchmark=b,sample=i,retention=ratio,prediction_text=text,generated_token_ids=tokens,input_sha256=digest,max_new_tokens=info['max_new_tokens'],seconds=time.time()-begin,token_audit=dict(original_visual=n,layer_visual=actual,excluded_full_layers=list(range(exclude)),prunable_visual_ratio=sum(actual[exclude:])/(n*(len(actual)-exclude)),events=[] if custom_dart else controller.audit),**score)
     out.write(json.dumps(record,ensure_ascii=False)+'\n');written+=1
    if smoke or written%40==0:
     print('PROGRESS',tag,b,i,written,round(time.time()-started),flush=True)
     dump(run/f'progress_{tag}.json',dict(rows=written,benchmark=b,sample=i,elapsed=time.time()-started))
 dump(run/'rows'/f'{tag}.done.json',dict(passed=True,rows=written,checks=checks,elapsed=time.time()-started))

def queue(run):
 cfg=json.loads((run/'config.json').read_text());root=Path(cfg['original_root']);jobs=json.loads((run/'jobs.json').read_text());failed=[];active=[]
 env=dict(os.environ,PYTHONPATH=str(root/'artifacts/dependencies/qwen35_python')+':'+str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 # Finish all smoke checks first, so every required implementation is checked
 # before spending hours completing a single model's full suite.
 for smoke in (True,False):
  for job in jobs:
   m,t,s=job['model'],job['method'],job['suite'];base=f'{s}__{m}__{t}'
   if not smoke and not (run/'rows'/f'{base}__smoke.done.json').exists():continue
   active=[];logs=[]
   try:
    for shard in ([0] if smoke else range(8)):
     tag=base+'__'+('smoke' if smoke else f'shard{shard}')
     if (run/'rows'/f'{tag}.done.json').exists():continue
     log=(run/'logs'/f'{tag}.log').open('a');logs.append(log)
     cmd=[str(root/'.venv/bin/python'),str(run/'source/scripts/repair_baseline_suite.py'),'worker','--run',str(run),'--model',m,'--method',t,'--suite',s,'--shard',str(shard)]
     if smoke:cmd+=['--smoke']
     proc=subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT);active.append(proc)
    while any(p.poll() is None for p in active):
     if any(p.poll() not in (None,0) for p in active):raise RuntimeError('worker failed')
     dump(run/'status.json',dict(state='smoke' if smoke else 'running',job=job,pids=[p.pid for p in active if p.poll() is None],failed=failed));time.sleep(5)
    assert all(p.returncode==0 for p in active)
   except Exception as e:
    for p in active:
     if p.poll() is None:p.terminate()
    for p in active:p.wait()
    failed.append(dict(job=job,phase='smoke' if smoke else 'full',error=repr(e)));dump(run/'failed_jobs.json',failed)
   finally:
    for log in logs:log.close()
 dump(run/'status.json',dict(state='failed_jobs' if failed else 'complete',failed=failed))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','queue']);p.add_argument('--run',type=Path,required=True);p.add_argument('--model');p.add_argument('--method');p.add_argument('--suite',default='image');p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.mode=='prepare':prepare(a.run)
 elif a.mode=='worker':worker(a.run,a.model,a.method,a.suite,a.shard,a.smoke)
 else:queue(a.run)
