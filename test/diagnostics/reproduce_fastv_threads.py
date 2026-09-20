"""Check one historical-environment hypothesis, only native base and FastV 5%."""
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.qwen_deepstack import disable_qwen_deepstack


def main():
    device=torch.device('cuda:0')
    path=ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    out=ROOT/'test/results/screenshot_adapter_fastv_20260915/threads.json'
    records=[]
    with torch.inference_mode():
        for method in ['base','fastv']:
            torch.set_num_threads(4)
            model,processor=load_baseline_model(method,'/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',torch.bfloat16,device,.05,'flash_attention_2')
            disable_qwen_deepstack(model)
            data=QwenBenchmarkDataset(str(path),processor,'mmstar',data_root=str(path.parent),max_samples=200)
            for index in [0,25,50,75,100,125,150,175]:
                inputs=_qwen_inputs_from_item(data[index],device)
                visual=inputs['mm_token_type_ids'][0].nonzero().flatten()
                configure_baseline(model,method,.05,int(visual[0]),len(visual))
                row=dict(method=method,index=index)
                reference=None
                for count in ([4,112] if index%50==0 else [112,4]):
                    torch.set_num_threads(count)
                    for keep in [1,0]:
                        times=[]
                        for repetition in range(4):
                            model.model.rope_deltas=None
                            torch.cuda.synchronize();start=time.perf_counter()
                            logits=model(**inputs,logits_to_keep=keep).logits
                            torch.cuda.synchronize();elapsed=time.perf_counter()-start
                            if repetition:times.append(elapsed*1000)
                        row[f'threads{count}_logits{keep}_ms']=statistics.median(times)
                        if keep:
                            if reference is None:reference=logits.clone()
                            else:assert torch.equal(reference,logits)
                records.append(row);out.write_text(json.dumps(records,indent=2));print(row,flush=True)
            del model,processor,data,inputs,reference,logits
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
