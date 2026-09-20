"""Check adapter KV with the exact native Qwen FA2 cached-decode path."""
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import torch
from transformers.cache_utils import DynamicCache
from paired_runtime_execution import MODEL,CHECKPOINT,sync
from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
from src.model import load_qwen_embedding_adapter_checkpoint,qwen_embedding_adapter_decode_step
from src.benchmark_prefill import build_qwen_fast_adapter_prefill
from src.qwen_native_graph import NativeDecoderGraphs
from src.qwen_deepstack import disable_qwen_deepstack
from src.data import QwenBenchmarkDataset


def main():
    torch.set_num_threads(4);device=torch.device('cuda:0')
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(model)
    adapter,_=load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT),model.model.language_model,device,torch.bfloat16)
    args=SimpleNamespace(last_logits_only=True,attn_implementation='flash_attention_2',cuda_graph=True,
        cuda_graph_context=True,compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,adapter_decode_cache_mode='fast')
    prefill=build_qwen_fast_adapter_prefill(model,adapter,args,native_decode=False)
    shared=model._adapter_decode_graph_runner
    native=NativeDecoderGraphs(model,vision=False,max_shapes=16)
    path=ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    dataset=QwenBenchmarkDataset(str(path),processor,'mmstar',data_root=str(path.parent),max_samples=200)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    checks=[];trials=[]
    out=ROOT/'test/results/deepstack_off_20260915/native_adapter_probe.json'

    def request(inputs,kind,check=False):
        sync();start=time.perf_counter()
        logits,_,_,_,cache=prefill(inputs)
        past=None
        positions=None
        if kind=='native':
            positions=torch.cat([cache['next_text_positions'].unsqueeze(0),cache['next_position_ids']],dim=0)
            past=DynamicCache(config=model.model.language_model.config)
            for i,layer in enumerate(cache['layers']):
                past.update(torch.cat([layer['visual_key'],layer['text_key']],dim=2),
                    torch.cat([layer['visual_value'],layer['text_value']],dim=2),i)
        sync();prefill_ms=1000*(time.perf_counter()-start)
        tokens=[];all_logits=[];times=[]
        for step in range(8):
            if check:all_logits.append(logits.clone())
            scores=logits[:,-1].to(dtype=torch.float32,copy=True);scores[:,eos]=-float('inf')
            token=scores.argmax(-1).view(1,1);tokens.append(int(token))
            if step==7:break
            sync();start=time.perf_counter()
            if kind=='native':
                result=model(input_ids=token,position_ids=positions+(step),past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=1)
                logits,past=result.logits,result.past_key_values
            else:
                fn=qwen_embedding_adapter_decode_step if kind=='eager' else shared
                logits,cache=fn(model,adapter,token,cache,logits_to_keep=1)
            sync();times.append(time.perf_counter()-start)
        return dict(tokens=tokens,prefill_ms=prefill_ms,decode_ms=1000*sum(times)/7),all_logits,past if kind=='native' else cache

    with torch.inference_mode():
        for index in [0,25,125,133]:
            inputs=_qwen_inputs_from_item(dataset[index],device)
            ref,expected,cache=request(inputs,'eager',True)
            shared.allow_capture=native.allow_capture=True
            for kind in ['shared','native']:
                result,logits,actual=request(inputs,kind,True)
                differences=[float((a-b).abs().max()) for a,b in zip(expected,logits)]
                checks.append(dict(index=index,kind=kind,max_logit_differences=differences,tokens_equal=ref['tokens']==result['tokens']))
                out.write_text(json.dumps(dict(checks=checks,trials=trials),indent=2))
                assert ref['tokens']==result['tokens'] and max(differences)==0,(index,kind,differences)
                for i,layer in enumerate(cache['layers']):
                    for suffix,attr in [('key','keys'),('value','values')]:
                        expected_kv=torch.cat([layer['visual_'+suffix],layer['text_'+suffix]],dim=2)
                        actual_kv=getattr(actual.layers[i],attr) if kind=='native' else torch.cat([actual['layers'][i]['visual_'+suffix],actual['layers'][i]['text_'+suffix]],dim=2)
                        assert torch.equal(expected_kv,actual_kv)
            shared.allow_capture=native.allow_capture=False
            before=[g.stats() for g in [shared,native]]
            for repetition in range(16):
                pair=dict(index=index,repetition=repetition)
                for kind in (['shared','native'] if repetition%2==0 else ['native','shared']):
                    pair[kind],_,_=request(inputs,kind)
                    assert pair[kind]['tokens']==ref['tokens']
                trials.append(pair)
            for old,g in zip(before,[shared,native]):
                now=g.stats();assert old['captures']==now['captures']
                key='cold_fallbacks' if 'cold_fallbacks' in old else 'cold_layer_fallbacks';assert old[key]==now[key]
            summary=dict(checks=checks,trials=trials,native_decode_speedup_over_shared=statistics.median(t['shared']['decode_ms']/t['native']['decode_ms'] for t in trials),
                decode_medians_ms={k:statistics.median(t[k]['decode_ms'] for t in trials) for k in ['shared','native']},
                prefill_medians_ms={k:statistics.median(t[k]['prefill_ms'] for t in trials) for k in ['shared','native']},
                note='Native prefill includes conversion of adapter KV to DynamicCache; all eight logits and final KV checked against the existing manual eager decoder.')
            out.write_text(json.dumps(summary,indent=2));print(index,json.dumps({k:v for k,v in summary.items() if k not in ['checks','trials']}),flush=True)


if __name__=='__main__':main()
