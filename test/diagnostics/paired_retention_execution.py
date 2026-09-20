"""Alternate base, 5% and 20% on each identical input with exactly eight tokens."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
import torch
from paired_runtime_execution import MODEL, assert_cache, sync
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.generation_timing import GenerationStageTimer
from src.qwen_native_graph import NativeDecoderGraphs
from src.qwen_deepstack import disable_qwen_deepstack


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',default=['fastv','dart','divprune','zoo','sparsevlm','visionzip'])
    parser.add_argument('--samples',type=int,nargs='+',default=[0,25,125,133])
    parser.add_argument('--rounds',type=int,default=12)
    parser.add_argument('--output',default='test/results/deepstack_off_20260915/retention_interleaved')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(vars(args),deepstack='off',attention='flash_attention_2',
        tokens=8,decode_steps=7,shared_with_training=True,orders='Cycle all six permutations of base / 5% / 20% within each input; no timing exclusions'),indent=2))
    base,processor=load_baseline_model('base',MODEL,torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(base)
    base_graph=NativeDecoderGraphs(base,max_shapes=32)
    base_timer=GenerationStageTimer(base)
    path=ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    dataset=QwenBenchmarkDataset(str(path),processor,'mmstar',data_root=str(path.parent),max_samples=200)
    orders=list(itertools.permutations(['base','ret05','ret20']))
    summaries=[]

    def request(model,timer,inputs,check=False):
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model.model.rope_deltas=None
        sync();timer.begin();start=time.perf_counter()
        result=model.generate(**inputs,max_new_tokens=8,min_new_tokens=8,do_sample=False,
            return_dict_in_generate=True,output_logits=check)
        sync();elapsed=time.perf_counter()-start
        measured=timer.finish(elapsed,8)
        measured['total_time_s']=elapsed
        measured['tokens']=result.sequences[0,inputs['input_ids'].shape[-1]:].tolist()
        assert len(measured['tokens'])==8 and measured['decode_steps']==7
        return measured,result

    with torch.inference_mode():
        for method in args.methods:
            model,_=load_baseline_model(method,MODEL,torch.bfloat16,device,.05,'flash_attention_2')
            disable_qwen_deepstack(model)
            graph=NativeDecoderGraphs(model,max_shapes=32)
            timer=GenerationStageTimer(model)
            trials=[]
            for index in args.samples:
                inputs=_qwen_inputs_from_item(dataset[index],device)
                visual=inputs['mm_token_type_ids'][0].nonzero().flatten()
                expected={}
                for label in ['base','ret05','ret20']:
                    m,g,t=(base,base_graph,base_timer) if label=='base' else (model,graph,timer)
                    if label!='base':configure_baseline(m,method,.05 if label=='ret05' else .2,int(visual[0]),len(visual))
                    g.enabled=False
                    reference,a=request(m,t,inputs,True)
                    g.enabled=g.allow_capture=True
                    candidate,b=request(m,t,inputs,True)
                    assert reference['tokens']==candidate['tokens']
                    assert len(a.logits)==len(b.logits)==8 and all(torch.equal(x,y) for x,y in zip(a.logits,b.logits))
                    assert_cache(a.past_key_values,b.past_key_values)
                    g.allow_capture=False
                    expected[label]=reference['tokens']
                    del a,b
                before=[g.stats() for g in [base_graph,graph]]
                for repetition in range(args.rounds):
                    trial=dict(index=index,repetition=repetition,order=orders[repetition%len(orders)],exact_vs_eager=True)
                    for label in trial['order']:
                        m,t=(base,base_timer) if label=='base' else (model,timer)
                        if label!='base':configure_baseline(m,method,.05 if label=='ret05' else .2,int(visual[0]),len(visual))
                        trial[label],_=request(m,t,inputs)
                        assert trial[label]['tokens']==expected[label]
                    trials.append(trial)
                for old,g in zip(before,[base_graph,graph]):
                    now=g.stats()
                    assert old['captures']==now['captures'] and old['cold_layer_fallbacks']==now['cold_layer_fallbacks']
                (out/f'{method}.json').write_text(json.dumps(trials,indent=2))
                print(method,index,'5% / 20% / base verified and timed',flush=True)
            summary=dict(method=method,triples=len(trials))
            for key,name in [('generation_prefill_time_s','prefill'),('decode_time_s','decode'),('total_time_s','total')]:
                ratios=[t['ret20'][key]/t['ret05'][key] for t in trials]
                summary[name+'_ret05_speedup_over_ret20']=statistics.median(ratios)
                summary[name+'_ret05_faster_triples']=sum(r>1 for r in ratios)
                for label in ['base','ret05','ret20']:
                    summary[label+'_'+name+'_median_ms']=statistics.median(t[label][key]*1000/(7 if name=='decode' else 1) for t in trials)
            summaries.append(summary)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2))
            print(json.dumps(summary),flush=True)
            timer.remove();graph.remove()
            del model,graph,timer,_
            gc.collect();torch.cuda.empty_cache()
    base_timer.remove();base_graph.remove()


if __name__=='__main__':main()
