"""Alternate 5%/20% on identical inputs to audit small cross-wave differences."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.benchmarking.engines.adapter import MODEL, MANIFEST, tensor_sha
from src.benchmarking.engines.pruning import INPUT_CACHE, layer_kv, label
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.generation_timing import GenerationStageTimer
from src.attention import optimize_qwen_attention_metadata
from src.model_setup import disable_qwen_deepstack
from src.kernels import FusedQwenNorms
from src.graphs import NativeDecoderGraphs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',default='fastv')
    parser.add_argument('--indices',nargs='+',type=int,default=[0,333,666])
    parser.add_argument('--runs',type=int,default=12)
    parser.add_argument('--output',default='test/results/video_pruning_fa2_metadata_20260915/paired_retention')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model(args.method,MODEL,torch.bfloat16,torch.device('cuda:0'),.05,'flash_attention_2')
    disable_qwen_deepstack(model)
    metadata=optimize_qwen_attention_metadata(model)
    selectors=NativeDecoderGraphs(model,max_shapes=32,vision=False,full_decode=False,prefill_layers=False)
    norms=FusedQwenNorms(model);norms.native_order=True
    timer=GenerationStageTimer(model,measure_memory=True)
    dataset=QwenBenchmarkDataset(str(MANIFEST),processor,'videomme',data_root=str(ROOT/'data/benchmarks/videomme'),cache_dir=str(INPUT_CACHE))
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    reference={}
    for retention in [.05,.2]:
        p=ROOT/'test/results/video_pruning_fa2_metadata_20260915/videomme999'/label(args.method,retention)
        reference[retention]={r['index']:r for f in p.glob('fa2_metadata_*.jsonl') for line in f.read_text().splitlines() if (r:=json.loads(line))}
        assert len(reference[retention])==999
    rows=[]
    with torch.inference_mode():
        for index in args.indices:
            inputs=_qwen_inputs_from_item(dataset[index],torch.device('cuda:0'))
            visual=inputs['mm_token_type_ids'][0].nonzero().flatten()
            assert tensor_sha([inputs[k] for k in sorted(inputs)])==reference[.05][index]['input_sha256']
            def request(retention,check=False):
                configure_baseline(model,args.method,retention,int(visual[0]),len(visual))
                for module in [model.model,model.model.language_model]:
                    module._pruning_audit_enabled=False
                torch.manual_seed(42+index)
                model.model.rope_deltas=None
                torch.cuda.synchronize();timer.begin();start=time.perf_counter();timer.mark_request_start(start)
                result=model.generate(**inputs,min_new_tokens=8,max_new_tokens=8,do_sample=False,disable_compile=True,
                    return_dict_in_generate=True,output_logits=check)
                torch.cuda.synchronize();total=time.perf_counter()-start
                metrics=timer.finish(total,8)
                metrics['total_time_s']=total
                assert result.sequences[0,inputs['input_ids'].shape[-1]:].tolist()==reference[retention][index]['tokens']
                if check:
                    assert [tensor_sha([x]) for x in result.logits]==reference[retention][index]['logits_sha256']
                    assert tensor_sha(layer_kv(result.past_key_values))==reference[retention][index]['final_kv_sha256']
                return metrics
            selectors.allow_capture=True
            for retention in [.05,.2]:request(retention,True)
            selectors.allow_capture=False
            for retention in [.05,.2]:request(retention)
            before=selectors.stats()
            for repeat in range(args.runs):
                order=[.05,.2] if repeat%2==0 else [.2,.05]
                row=dict(index=index,repeat=repeat,order=order,exact_vs_full_results=True)
                for retention in order:row[str(retention)]=request(retention)
                rows.append(row)
            after=selectors.stats()
            assert after['captures']==before['captures'] and after['cold_layer_fallbacks']==before['cold_layer_fallbacks']
    summaries={}
    for key in ['total_time_s','request_prefill_time_s','decode_time_s']:
        ratios=[r['0.2'][key]/r['0.05'][key] for r in rows]
        summaries[key]=dict(paired_median_5percent_speedup=statistics.median(ratios),
            pairs_5percent_faster=sum(r>1 for r in ratios),pairs=len(ratios),
            ret05_mean_sample_median_ms=1000*statistics.mean(statistics.median(r['0.05'][key] for r in rows if r['index']==i) for i in args.indices),
            ret20_mean_sample_median_ms=1000*statistics.mean(statistics.median(r['0.2'][key] for r in rows if r['index']==i) for i in args.indices))
    result=dict(method=args.method,indices=args.indices,runs=args.runs,tokens=8,
        purpose='Paired diagnostic, not a replacement for the full 999 table; no decoder/vision graphs',
        summaries=summaries,trials=rows)
    (output/f'{args.method}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(method=args.method,summaries=summaries),indent=2),flush=True)
    timer.remove();norms.remove();selectors.remove();metadata.remove()


if __name__=='__main__':main()
