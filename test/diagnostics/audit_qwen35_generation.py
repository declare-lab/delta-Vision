"""Paired generation-limit diagnostic, identical questions and prompts across six methods."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35_experiment import load_model,prepare_inputs,dump
from src.qwen35_pruning import VisualPruningController
from src.benchmarks import get_benchmark_spec,build_benchmark_prompt,score_prediction
from scripts.qwen35_worker import read_rows

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--method',required=True);a=p.parse_args()
 run=a.run_dir;cfg=json.loads((run/'config.json').read_text());ref=Path(cfg['reference_run'])
 processor,model,adapter,old=load_model(cfg,torch.device('cuda:0'))
 if a.method=='adapter':
  ckpt=torch.load(ref/'checkpoints/qwen35_embedding_adapter_step2000.pt',map_location='cpu',weights_only=False)
  adapter.load_state_dict(ckpt['state_dict'],strict=True)
 pruning=VisualPruningController(model)
 if a.method not in ('native','adapter'):old.close()
 root=ref if a.method in ('native','adapter') else run
 records=[]
 for case in json.loads((run/'generation_audit_cases.json').read_text()):
  name,index=case['benchmark'],case['index'];info=cfg['evaluation'][name];row=read_rows(info['path'])[index];spec=get_benchmark_spec(name)
  prior=next(p for f in (root/'eval'/a.method).glob(f'{name}.shard*.jsonl') for p in read_rows(f) if p['index']==index)
  inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
  mask=inputs['mm_token_type_ids'].eq(1)
  for cap in (256,1024):
   context=old.activate(a.method,mask) if a.method in ('native','adapter') else pruning.activate(a.method.rsplit('_',1)[0],int(a.method.rsplit('_',1)[1])/100,mask)
   with context:
    out=model.generate(**inputs,do_sample=False,max_new_tokens=cap,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id)
   ids=out[0,inputs['input_ids'].shape[1]:];txt=processor.tokenizer.decode(ids,skip_special_tokens=True)
   eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
   finished=int(ids[-1]) in eos
   if finished:break
  short=processor.tokenizer.decode(ids[:prior['generated_tokens']],skip_special_tokens=True)
  score=lambda text:score_prediction(metric=info['metric'],prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
  rec=dict(method=a.method,benchmark=name,index=index,old_text=prior['prediction_text'],old_score=prior['score'],short_rescored=score(prior['prediction_text']),long_text=txt,long_scored=score(txt),generated_tokens=len(ids),finished_eos=finished,cap=cap,short_prefix_reproduced=short==prior['prediction_text'])
  records.append(rec);dump(run/f'generation_audit_{a.method}.json',records)
  print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
