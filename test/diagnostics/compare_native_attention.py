"""Same-device alternating comparison of warmed native base FlashAttention backends."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.generation_timing import GenerationStageTimer
from src.qwen_native_graph import NativeDecoderGraphs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backends',nargs='+',default=['flash_attention_2','flash_attention_3'])
    parser.add_argument('--samples',nargs='+',type=int,default=[0,25,125,133])
    parser.add_argument('--pairs',type=int,default=10)
    parser.add_argument('--output',default='test/results/native_base_optimized_20260915/backend_pairs.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    loaded={}
    for backend in args.backends:
        model,processor=load_baseline_model('base','/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
            torch.bfloat16,torch.device('cuda:0'),1.,backend)
        loaded[backend]=(model,NativeDecoderGraphs(model),GenerationStageTimer(model))
    data=ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    dataset=QwenBenchmarkDataset(str(data),processor,'mmstar',data_root=str(data.parent),max_samples=200)
    out=ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for index in args.samples:
            inputs=_qwen_inputs_from_item(dataset[index],torch.device('cuda:0'))
            expected={}
            for backend,(model,graphs,timer) in loaded.items():
                graphs.allow_capture=True
                model.model.rope_deltas=None
                expected[backend]=model.generate(**inputs,max_new_tokens=8,do_sample=False)
                graphs.allow_capture=False
            trials=[]
            for repetition in range(args.pairs):
                trial={}
                for backend in (args.backends if repetition%2==0 else args.backends[::-1]):
                    model,graphs,timer=loaded[backend]
                    model.model.rope_deltas=None
                    before=graphs.stats()
                    torch.cuda.synchronize()
                    timer.begin()
                    start=time.perf_counter()
                    result=model.generate(**inputs,max_new_tokens=8,do_sample=False)
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    row=timer.finish(elapsed,result.shape[-1]-inputs['input_ids'].shape[-1])
                    row['total_time_s']=elapsed
                    assert torch.equal(result,expected[backend])
                    assert graphs.captures==before['captures'] and graphs.fallbacks==before['cold_layer_fallbacks']
                    trial[backend]=row
                trials.append(trial)
            row=dict(index=index,trials=trials,shared_gpu=True,generated_tokens={b:expected[b][0,inputs['input_ids'].shape[-1]:].tolist() for b in args.backends})
            row['median_ms']={b:{k:statistics.median(t[b][k]*1000 for t in trials)
                for k in ['total_time_s','generation_prefill_time_s','decode_time_s']} for b in args.backends}
            rows.append(row)
            out.write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
    for model,graphs,timer in loaded.values():
        graphs.remove()
        timer.remove()


if __name__=='__main__':
    main()
