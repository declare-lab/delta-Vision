"""Real-model smoke test of shared EOS-aware evaluation on an audited error."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35_experiment import load_model,prepare_inputs,generate_evaluation_answer,dump
from src.qwen35_pruning import VisualPruningController
from src.benchmarks import get_benchmark_spec,build_benchmark_prompt

@torch.inference_mode()
def main():
 run=ROOT/'artifacts/experiments/qwen35_pruning/qwen35_4b_dart_divprune_5_20_20260920_082533'
 config=json.loads((run/'config.json').read_text());config['evaluation_generation']=dict(max_new_tokens=4096,do_sample=False,unfinished_response='invalid_zero')
 processor,model,adapter,old=load_model(config,torch.device('cuda:0'));old.close();del adapter,old
 pruning=VisualPruningController(model);info=config['evaluation']['mmstar'];row=json.loads(Path(info['path']).read_text().splitlines()[797]);spec=get_benchmark_spec('mmstar')
 inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
 results=[]
 native=generate_evaluation_answer(model,processor,inputs,row,spec,config);results.append(dict(method='native',**native))
 with pruning.activate('divprune',.2,inputs['mm_token_type_ids'].eq(1)):
  result=generate_evaluation_answer(model,processor,inputs,row,spec,config);results.append(dict(method='divprune_20',**result))
 assert native['stopped_by_eos'] and native['prediction']=='A' and native['score']==1
 assert result['stopped_by_eos'] and result['prediction']=='B' and result['score']==0
 dump(run/'corrected_eval_smoke.json',results)
 for x in results:print(x['method'],x['prediction'],x['score'],x['generated_tokens'],x['stopped_by_eos'],flush=True)
if __name__=='__main__':main()
