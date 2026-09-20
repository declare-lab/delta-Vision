import sys,os,json,subprocess,time
import torch
from src.native_visual_self_distill import ROOT,OUT,MODEL,SelfOnlyVisual
from src.initial_token_mlp_probe import runtime
from src.model import load_frozen_qwen3vl
from src.qwen_deepstack import disable_qwen_deepstack
from src.visual_cross_token_ablation import prepare
from src.data import QwenBenchmarkDataset
from src.benchmarks import get_benchmark_spec,score_prediction
from src.eval_benchmarks import generate_teacher_qwen

def worker(stage,rank):
 runtime();source=MODEL if stage=='before' else str(OUT/'checkpoint-2000');proc,model=load_frozen_qwen3vl(source,torch.bfloat16,torch.device('cuda:0'),'flash_attention_2');disable_qwen_deepstack(model);hook=SelfOnlyVisual(model)
 with torch.inference_mode(),(OUT/f'eval_{stage}/rows{rank}.jsonl').open('w',buffering=1) as f:
  for dsname in ['realworldqa','mmstar','sqa']:
   ds=QwenBenchmarkDataset(str(ROOT/f'artifacts/diagnostics/channel_native_cache_20260916/{dsname}_eval.jsonl'),proc,dsname)
   for i in range(rank,len(ds),8):
    item=ds[i];inputs=prepare(item,model.device);hook.mask=inputs['input_ids']==model.config.image_token_id
    for mode in (['native','self_untrained'] if stage=='before' else ['self_trained']):
     hook.enabled=mode!='native';hook.calls=0
     _,answer=generate_teacher_qwen(model,proc,**inputs,max_new_tokens=8)
     assert hook.calls==(0 if mode=='native' else 36)
     score=score_prediction(metric=get_benchmark_spec(dsname).metric,prediction_text=answer,answer=item['answer'],choices=item.get('choices'),question=item['row'].get('question'))
     f.write(json.dumps({'dataset':dsname,'sample':i,'mode':mode,'answer':answer,**score})+'\n')
    if i//8%30==0:print(dsname,i,flush=True)

def launch(stage):
 folder=OUT/f'eval_{stage}';folder.mkdir(exist_ok=False);jobs=[]
 try:
  for i in range(8):
   log=(folder/f'gpu{i}.log').open('w');p=subprocess.Popen([sys.executable,'-u','-m','src.native_visual_self_distill_eval',stage,str(i)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
  while any(p.poll() is None for p,_ in jobs):
   if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('eval failed')
   time.sleep(3)
  rows=[json.loads(s) for i in range(8) for s in (folder/f'rows{i}.jsonl').read_text().splitlines()];result={}
  for ds,n in [('realworldqa',765),('mmstar',1000),('sqa',1000)]:
   result[ds]={}
   for mode in (['native','self_untrained'] if stage=='before' else ['self_trained']):
    rr=[r for r in rows if r['dataset']==ds and r['mode']==mode];assert len(rr)==n and {r['sample'] for r in rr}==set(range(n));result[ds][mode]={'samples':n,'accuracy_pct':100*sum(r['score'] for r in rr)/n}
  (folder/'results.json').write_text(json.dumps(result,indent=2))
 finally:
  for p,f in jobs:
   if p.poll() is None:p.terminate()
   f.close()
if __name__=='__main__':
 if len(sys.argv)>2:worker(sys.argv[1],int(sys.argv[2]))
 else:launch(sys.argv[1])
