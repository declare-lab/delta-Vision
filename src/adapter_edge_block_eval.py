"""Existing static adapter, paired all-layer and first6+last10 visual-edge blocking."""
import json, os, sys, subprocess, time, hashlib
from pathlib import Path
import torch
from src import causal_effect_benchmark_suite as suite
ROOT=suite.ROOT
OUT=ROOT/'artifacts/diagnostics/adapter_first6_last10_20260916'
CHECKPOINT=ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
BLOCKED=set(range(6))|set(range(26,36))

def worker(shard):
 from src import model as ref
 from src.qwen_deepstack import disable_qwen_deepstack
 from src.data import QwenBenchmarkDataset
 from src.benchmarks import get_benchmark_spec, score_prediction
 from src.eval_benchmarks import extract_option_from_text
 from flash_attn import flash_attn_func
 torch.set_num_threads(4);torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
 processor,model=ref.load_frozen_qwen3vl(suite.MODELS['qwen'][0],torch.bfloat16,torch.device('cuda:0'),'flash_attention_2')
 disable_qwen_deepstack(model);model._adapter_attention_implementation='flash_attention_2'
 adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda:0'),torch.bfloat16)
 assert adapter.mode=='embedding_adapter' and getattr(adapter,'adapter_start_layer',0)==0 and getattr(adapter,'active_adapter_layers',0)==0
 model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
 original=ref._prefix_causal_attention_heads
 state={'layer':0,'blocked':False,'audit':False};errors=[]
 def attention(q,vk,vv,tk,tv,**kw):
  l=state['layer'];state['layer']+=1
  assert kw['attention_plan'] is not None
  if not state['blocked'] or l not in BLOCKED:return original(q,vk,vv,tk,tv,**kw)
  heads=flash_attn_func(q.transpose(1,2),tk.transpose(1,2),tv.transpose(1,2),dropout_p=0.,softmax_scale=kw['scaling'],causal=True)
  if state['audit']:
   k=torch.cat((vk,tk),2).float().repeat_interleave(q.shape[1]//tk.shape[1],1)
   v=torch.cat((vv,tv),2).float().repeat_interleave(q.shape[1]//tk.shape[1],1)
   mask=kw['attention_mask'].clone();mask[...,:vk.shape[2]]=False
   expected=torch.nn.functional.scaled_dot_product_attention(q.float(),k,v,attn_mask=mask,scale=kw['scaling']).transpose(1,2)
   err=float((heads.float()-expected).norm()/expected.norm().clamp_min(1e-10));assert err<.02,err;errors.append(err)
  return heads
 ref._prefix_causal_attention_heads=attention
 vision=[];get=model.model.get_image_features
 def cached(*a,**kw):
  if not vision:vision.append(get(*a,**kw))
  return vision[0]
 model.model.get_image_features=cached
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(processor.tokenizer.eos_token_id)
 with torch.inference_mode():
  for benchmark,(relative,n) in suite.DATASETS.items():
   path=ROOT/relative;ds=QwenBenchmarkDataset(str(path),processor,benchmark,data_root=str(path.parent),max_samples=n)
   old=json.loads((ROOT/f'artifacts/diagnostics/causal_effect_2models_2bench_20260912/qwen_{benchmark}/plan.json').read_text())
   assert hashlib.sha256(json.dumps(ds.rows,sort_keys=True).encode()).hexdigest()==old['selection_sha256']
   folder=OUT/benchmark;folder.mkdir(exist_ok=True)
   with (folder/f'shard{shard}.jsonl').open('w') as f:
    for i in range(shard,n,8):
     item=ds[i];inputs=suite.prepare(item,model,'qwen');vision.clear()
     initial,pos=ref.build_qwen_initial_context(model,inputs)
     visual=inputs['mm_token_type_ids'][0].ne(0)
     memories=adapter.all_visual_memories_batched(initial[:,visual]);method=adapter.all_visual_memories_batched
     adapter.all_visual_memories_batched=lambda *a,**kw:memories
     results={}
     try:
      for blocked in [False,True]:
       state['blocked']=blocked;state['audit']=i==shard
       current=dict(inputs);generated=[]
       for step in range(8):
        state['layer']=0
        # Rebuild text embeddings/positions for extended prompts, cached vision unchanged.
        h,p=(initial,pos) if step==0 else ref.build_qwen_initial_context(model,current)
        logits=ref.qwen_embedding_adapter_logits(model,adapter,current,initial_hidden=h,position_ids=p,logits_to_keep=1)[0]
        assert state['layer']==36
        token=int(logits[0,-1].float().argmax());generated.append(token)
        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
        if token in eos or extract_option_from_text(text) in ['A','B','C','D']:break
        new=torch.tensor([[token]],device='cuda')
        current=dict(current,input_ids=torch.cat((current['input_ids'],new),1),attention_mask=torch.cat((current['attention_mask'],torch.ones_like(new)),1),mm_token_type_ids=torch.cat((current['mm_token_type_ids'],torch.zeros_like(new)),1))
       key='adapter_blocked' if blocked else 'adapter'
       score=score_prediction(metric=get_benchmark_spec(benchmark).metric,prediction_text=text,answer=item['answer'],choices=item.get('choices'),question=item['row'].get('question'))
       results[key]={'text':text,**score}
     finally:adapter.all_visual_memories_batched=method
     f.write(json.dumps({'sample':i,'results':results})+'\n');f.flush()
     if i//8%20==0:print(benchmark,i,flush=True)
   (folder/f'done{shard}.json').write_text(json.dumps({'max_attention_relative_error':max(errors)}))

def launch():
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'plan.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT),'blocked_layers':sorted(BLOCKED),'backend':'FA2','deepstack':'off','max_new_tokens':8,'world':8,'no_training':True},indent=2))
 jobs=[]
 try:
  for i in range(8):
   log=(OUT/f'gpu{i}.log').open('w');p=subprocess.Popen([sys.executable,'-u','-m','src.adapter_edge_block_eval',str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
  while any(p.poll() is None for p,_ in jobs):
   if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('Worker failed')
   time.sleep(3)
  assert all(p.returncode==0 for p,_ in jobs)
  summary={}
  for b,(_,n) in suite.DATASETS.items():
   rows=[r for i in range(8) for r in map(json.loads,(OUT/b/f'shard{i}.jsonl').open())]
   assert len(rows)==n and {r['sample'] for r in rows}==set(range(n))
   summary[b]={k:100*sum(r['results'][k]['score'] for r in rows)/n for k in ['adapter','adapter_blocked']}
  (OUT/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 finally:
  for p,log in jobs:
   if p.poll() is None:p.terminate()
   log.close()
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[1]))
 else:launch()
