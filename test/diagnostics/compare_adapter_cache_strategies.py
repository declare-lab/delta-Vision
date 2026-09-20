"""Same-model alternating decode test of per-step versus shared visual copies."""
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import torch
import src.model
from paired_runtime_execution import MODEL,CHECKPOINT,assert_cache,sync
from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
from src.model import load_qwen_embedding_adapter_checkpoint,qwen_embedding_adapter_decode_step
from src.benchmark_prefill import build_qwen_fast_adapter_prefill
from src.qwen_adapter_graph import QwenAdapterDecodeGraphs
from src.qwen_adapter_shared_graph import SharedVisualDecodeGraphs
from src.qwen_deepstack import disable_qwen_deepstack
from src.data import QwenBenchmarkDataset
from src.qwen_adapter_prepare import prepare_fa2_inputs


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(model)
    adapter,_=load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT),model.model.language_model,device,torch.bfloat16)
    args=SimpleNamespace(last_logits_only=True,attn_implementation='flash_attention_2',cuda_graph=True,
        cuda_graph_context=True,compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,adapter_decode_cache_mode='fast')
    prefill=build_qwen_fast_adapter_prefill(model,adapter,args,native_decode=False)
    original_prepare=src.model.prepare_qwen_embedding_adapter_inputs
    src.model.prepare_qwen_embedding_adapter_inputs=prepare_fa2_inputs
    try:
        new_prefill=build_qwen_fast_adapter_prefill(model,adapter,args,native_decode=False)
    finally:
        src.model.prepare_qwen_embedding_adapter_inputs=original_prepare
    graphs={'per_step':QwenAdapterDecodeGraphs(model,adapter),'shared':SharedVisualDecodeGraphs(model,adapter)}
    path=ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    dataset=QwenBenchmarkDataset(str(path),processor,'mmstar',data_root=str(path.parent),max_samples=200)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    trials=[];checks=[];prefill_trials=[]
    out=ROOT/'test/results/deepstack_off_20260915/adapter_cache_strategies.json'

    def request(inputs,graph,check=False):
        logits,_,_,_,cache=prefill(inputs)
        all_logits=[];tokens=[];times=[]
        copies=graph.stats()['visual_prefix_copies'] if isinstance(graph,SharedVisualDecodeGraphs) else None
        for step in range(8):
            if check:all_logits.append(logits.clone())
            scores=logits[:,-1].to(dtype=torch.float32,copy=True);scores[:,eos]=-float('inf')
            token=scores.argmax(-1).view(1,1);tokens.append(int(token))
            if step==7:break
            sync();start=time.perf_counter()
            logits,cache=graph(model,adapter,token,cache,logits_to_keep=1)
            sync();times.append(time.perf_counter()-start)
        if copies is not None:assert graph.stats()['visual_prefix_copies']==copies+1
        return dict(tokens=tokens,decode_ms_per_step=1000*sum(times)/7),all_logits,cache

    with torch.inference_mode():
        for index in [0,25,125,133]:
            inputs=_qwen_inputs_from_item(dataset[index],device)
            old_payload=prefill(inputs);new_payload=new_prefill(inputs)
            assert torch.equal(old_payload[0],new_payload[0])
            assert_cache(old_payload[-1],new_payload[-1])
            for repetition in range(24):
                pair=dict(index=index,repetition=repetition)
                for label,fn in ([('old',prefill),('one_read',new_prefill)] if repetition%2==0 else [('one_read',new_prefill),('old',prefill)]):
                    sync();start=time.perf_counter();payload=fn(inputs);sync()
                    pair[label]=1000*(time.perf_counter()-start)
                prefill_trials.append(pair)
            reference,expected,reference_cache=request(inputs,qwen_embedding_adapter_decode_step,True)
            for label,graph in graphs.items():
                graph.allow_capture=True
                candidate,logits,cache=request(inputs,graph,True)
                assert reference['tokens']==candidate['tokens']
                assert all(torch.equal(a,b) for a,b in zip(expected,logits))
                assert_cache(reference_cache,cache)
                graph.allow_capture=False
            checks.append(dict(index=index,all_logits_tokens_kv_exact=True))
            for repetition in range(16):
                pair=dict(index=index,repetition=repetition)
                for label in (['per_step','shared'] if repetition%2==0 else ['shared','per_step']):
                    pair[label],_,_=request(inputs,graphs[label])
                    assert pair[label]['tokens']==reference['tokens']
                trials.append(pair)
            if index==0:
                changed=dict(inputs,pixel_values=inputs['pixel_values'].flip(0))
                expected_tokens,expected_logits,expected_cache=request(changed,qwen_embedding_adapter_decode_step,True)
                actual_tokens,actual_logits,actual_cache=request(changed,graphs['shared'],True)
                assert expected_tokens['tokens']==actual_tokens['tokens']
                assert all(torch.equal(a,b) for a,b in zip(expected_logits,actual_logits))
                assert_cache(expected_cache,actual_cache)
                checks.append(dict(index=index,altered_pixels_same_geometry=True,shared_prefix_refreshed=True))
            out.write_text(json.dumps(dict(checks=checks,trials=trials,prefill_trials=prefill_trials,deepstack='off',attention='flash_attention_2',
                one_read_prefill_speedup=statistics.median(p['old']/p['one_read'] for p in prefill_trials),
                shared_speedup=statistics.median(t['per_step']['decode_ms_per_step']/t['shared']['decode_ms_per_step'] for t in trials),
                medians_ms={label:statistics.median(t[label]['decode_ms_per_step'] for t in trials) for label in graphs}),indent=2))
            print(index,'strategies checked and timed',flush=True)
    print(out.read_text()[-250:],flush=True)


if __name__=='__main__':main()
