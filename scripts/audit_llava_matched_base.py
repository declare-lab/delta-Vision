"""Native LLaVA controls on the exact baseline rerun inputs and scorer."""
import argparse,csv,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/eval/all_baselines_seed44_20260922_restart'
RUN=ROOT/'artifacts/eval/llava_matched_native_20260923'
MODELS=['llava-1.6-mistral-7b','llava-1.5-13b','llava-1.5-7b']
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def worker(model_name,shard):
 sys.path.insert(0,str(REFERENCE/'source'))
 import torch
 from baselines.llava_hf_baselines import load_llava_baseline_model,build_llava_inputs_embeds_with_image_span
 from src.model import llava_projected_image_features
 from src.data import LlavaBenchmarkDataset
 from src.scoring_reference import get_benchmark_spec,score_prediction
 from src.divprune_rerun import input_digest
 cfg=json.loads((REFERENCE/'config.json').read_text());torch.set_num_threads(4);torch.manual_seed(44)
 processor,model=load_llava_baseline_model(cfg['models'][model_name]['path'],dtype=torch.bfloat16,device='cuda:0',attn_implementation='flash_attention_2')
 model.eval().requires_grad_(False)
 prior={}
 for line in (REFERENCE/'rows'/f'{model_name}__dart__shard{shard}.jsonl').read_text().splitlines():
  r=json.loads(line)
  if r['retention']==.2:prior[r['benchmark'],r['sample']]=r
 path=RUN/'rows'/f'{model_name}__shard{shard}.jsonl';path.parent.mkdir(exist_ok=True)
 done=set()
 if path.exists():done={(r['benchmark'],r['sample']) for r in map(json.loads,path.read_text().splitlines())}
 checks=[];started=time.time();count=0
 # Put SQA/RealWorldQA first because these triggered the concern.
 names=['sqa','realworldqa']+[b for b in cfg['evaluation'] if b not in ('sqa','realworldqa')]
 with torch.inference_mode(),path.open('a',buffering=1) as out:
  for b in names:
   info=cfg['evaluation'][b];ds=LlavaBenchmarkDataset(info['path'],processor,b,data_root=info['image_root'])
   for i in range(shard,len(ds),8):
    if (b,i) in done:continue
    item=ds[i];row=item['row'];digest=input_digest(item)
    assert digest==prior[b,i]['input_sha256'],(model_name,b,i,'not paired')
    inputs=dict(input_ids=item['input_ids'][None].cuda(),attention_mask=item['attention_mask'][None].cuda(),pixel_values=item['pixel_values'][None].cuda())
    sizes=item.get('image_sizes')
    if torch.is_tensor(sizes):inputs['image_sizes']=sizes[None].cuda()
    result=model.generate(**inputs,max_new_tokens=info['max_new_tokens'],use_cache=True,do_sample=False)
    tokens=result[0,inputs['input_ids'].shape[1]:].tolist();text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
    if i==shard:
     # Verify that labels are not consulted by the input constructor.
     saved=ds.rows[i];ds.rows[i]=dict(saved,answer='SENTINEL_NOT_AN_ANSWER',answers=['SENTINEL_NOT_AN_ANSWER'])
     redacted=ds[i];ds.rows[i]=saved;assert input_digest(redacted)==digest
     # Native image route versus unchanged full visual embedding route.
     memory=llava_projected_image_features(model,inputs['pixel_values'],image_sizes=inputs.get('image_sizes'))
     emb,am,_,_=build_llava_inputs_embeds_with_image_span(model,input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],image_token_id=model.config.image_token_index,visual_memory=memory)
     a=model(**inputs,use_cache=True,logits_to_keep=1).logits
     z=model(inputs_embeds=emb,attention_mask=am,use_cache=True,logits_to_keep=1).logits
     torch.testing.assert_close(a,z,rtol=0,atol=0)
     raw=model.generate(inputs_embeds=emb,attention_mask=am,use_cache=True,do_sample=False,max_new_tokens=info['max_new_tokens'])[0].tolist()
     assert raw==tokens,(b,i,'native/full-embedding generation differs')
     checks.append(dict(benchmark=b,sample=i,label_independent_inputs=True,native_full_embedding_logits_exact=True,native_full_embedding_generation_exact=True))
    score=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
    out.write(json.dumps(dict(model=model_name,method='native',benchmark=b,sample=i,prediction_text=text,generated_token_ids=tokens,input_sha256=digest,max_new_tokens=info['max_new_tokens'],**score),ensure_ascii=False)+'\n');count+=1
    if count%40==0:print('PROGRESS',model_name,shard,b,i,count,round(time.time()-started),flush=True)
 dump(RUN/'rows'/f'{model_name}__shard{shard}.done.json',dict(checks=checks,elapsed=time.time()-started))
def report():
 cfg=json.loads((REFERENCE/'config.json').read_text());benchmarks=list(cfg['evaluation']);groups={}
 for p in (RUN/'rows').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   groups.setdefault((r['model'],r['benchmark']),[]).append(r)
 rows=[]
 for m in MODELS:
  row=dict(model=m,method='native',retention=1.)
  for b in benchmarks:
   records=groups.get((m,b),[])
   if len(records)==cfg['evaluation'][b]['samples']:
    assert sorted(r['sample'] for r in records)==list(range(len(records)))
    row[b]=100*sum(r['score'] for r in records)/len(records)
  if all(b in row for b in benchmarks):row['AVG']=sum(row[b] for b in benchmarks)/9
  rows.append(row)
 old=json.loads((REFERENCE/'summary.json').read_text())['rows'];comparison=[]
 for base in rows:
  comparison.append(base)
  comparison.extend(r for r in old if r['model']==base['model'] and r['method'] in ('dart','divprune') and 'AVG' in r)
 dump(RUN/'summary.json',dict(native=rows,comparison=comparison))
 lines=['# LLaVA: matched native controls','', 'Same inputs, prompt, caps, FA2/BF16, pinned7f266415 scorer.','', '| Model | Method | Ratio | '+' | '.join(benchmarks+['AVG'])+' |','|---|---|---:|'+'---:|'*10]
 for r in comparison:lines.append('| '+r['model']+' | '+r['method']+f" | {r['retention']:.0%} | "+' | '.join(f'{r[b]:.2f}' if b in r else 'pending' for b in benchmarks+['AVG'])+' |')
 (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n');return rows

def queue():
 RUN.mkdir(exist_ok=True);(RUN/'logs').mkdir(exist_ok=True);(RUN/'rows').mkdir(exist_ok=True)
 active=[]
 env=dict(os.environ,PYTHONPATH=str(REFERENCE/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
 # Paused repair scheduler's current worker may still be completing a smoke.
 while any(int(m)>1024 for m in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()):time.sleep(5)
 try:
  for m in MODELS:
   active=[];logs=[]
   for shard in range(8):
    if (RUN/'rows'/f'{m}__shard{shard}.done.json').exists():continue
    log=(RUN/'logs'/f'{m}_{shard}.log').open('a');logs.append(log)
    p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'worker','--model',m,'--shard',str(shard)],env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT);active.append(p)
   while any(p.poll() is None for p in active):
    assert all(p.poll() in (None,0) for p in active),'Native worker failed'
    dump(RUN/'status.json',dict(state='running',model=m,pids=[p.pid for p in active if p.poll() is None]));report();time.sleep(15)
   assert all(p.returncode==0 for p in active)
   for log in logs:log.close()
  rows=report();assert all('AVG' in r for r in rows);dump(RUN/'status.json',dict(state='complete'))
 except BaseException as e:
  for p in active:
   if p.poll() is None:p.terminate()
  for p in active:p.wait()
  dump(RUN/'status.json',dict(state='failed',error=repr(e)));raise
 finally:
  pid=int((ROOT/'artifacts/eval/all_baselines_repaired_20260923/queue.pid').read_text())
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['queue','worker','report']);p.add_argument('--model');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.mode=='queue':queue()
 elif a.mode=='worker':worker(a.model,a.shard)
 else:print(report())
