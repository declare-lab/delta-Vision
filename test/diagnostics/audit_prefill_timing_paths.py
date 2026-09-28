"""Compare original benchmark timing paths on one byte-identical input."""
from pathlib import Path
import json
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from src.benchmarking.common import prefill as bp
from src import model as vm
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.comparison import RequestRunner

OUT=ROOT/'test/results/prefill_timing_audit_20260915'
OUT.mkdir(parents=True,exist_ok=True)
sys.argv=['audit','--model-path',str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
          '--checkpoint',str(ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt')]
args=bp.parse_args()
bp.configure_torch_runtime()
bp.set_global_seed(42)
processor,model=vm.load_frozen_qwen3vl(args.model_path,torch.bfloat16,torch.device('cuda:0'),args.attn_implementation)
adapter,meta=vm.load_qwen_embedding_adapter_checkpoint(args.checkpoint[0],model.model.language_model,torch.device('cuda:0'),torch.bfloat16)
ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/mmstar/mmstar_val.jsonl'),processor,'mmstar',data_root=str(ROOT/'data/benchmarks/mmstar'),max_samples=1)
item=ds[0]
inputs={k:(v.unsqueeze(0) if k in ('input_ids','attention_mask','mm_token_type_ids') else v).cuda() for k,v in item.items() if torch.is_tensor(v)}
results=[]

def measure(name,fn,n=20):
    with torch.inference_mode():
        original=bp.benchmark(fn,warmup=5,n_runs=n)
        times=[]
        for _ in range(5):
            torch.cuda.synchronize()
            begin=time.perf_counter()
            value=fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-begin)
    row=dict(path=name,original_loop_ms=original*1000,sync_each_call_median_ms=statistics.median(times)*1000)
    print(json.dumps(row),flush=True)
    results.append(row)
    (OUT/'paths.json').write_text(json.dumps(results,indent=2))
    return value

with torch.inference_mode():
    print('INPUT', inputs['input_ids'].shape, int(inputs['mm_token_type_ids'].ne(0).sum()),flush=True)
    eager_base=lambda:model(**inputs,use_cache=True,logits_to_keep=1).logits
    measure('original_base_prefill',eager_base)
    e2e=bp.build_qwen_benchmark_e2e_fn(model,adapter,args=args,logits_to_keep=1,
        build_qwen_initial_context=vm.build_qwen_initial_context,qwen_embedding_adapter_logits=vm.qwen_embedding_adapter_logits,
        prepare_qwen_embedding_adapter_inputs=vm.prepare_qwen_embedding_adapter_inputs,
        qwen_embedding_adapter_logits_prepared=vm.qwen_embedding_adapter_logits_prepared,
        qwen_position_ids=vm.qwen_position_ids,qwen_visual_grid_metadata=vm.qwen_visual_grid_metadata)
    measure('original_prefill_e2e_fastpath_no_decode_cache',lambda:e2e(inputs))
    initial,positions=vm.build_qwen_initial_context(model,inputs)
    cached=bp.build_qwen_benchmark_delta_fn(model,adapter,args=args,logits_to_keep=1,
        qwen_embedding_adapter_logits_from_tensors=vm.qwen_embedding_adapter_logits_from_tensors,
        prepare_qwen_embedding_adapter_inputs=vm.prepare_qwen_embedding_adapter_inputs,
        qwen_embedding_adapter_logits_prepared=vm.qwen_embedding_adapter_logits_prepared)
    measure('original_cached_visual_prefill_diagnostic',lambda:cached(inputs,initial,positions))
    cache_prefill=bp.build_qwen_fast_adapter_prefill(model,adapter,args)
    measure('original_metric_prefill_with_decode_cache',lambda:cache_prefill(inputs))
    def hf_generate(fixed):
        model.model.rope_deltas=None
        kwargs=dict(max_new_tokens=32,do_sample=False,return_dict_in_generate=True)
        if fixed:kwargs['min_new_tokens']=32
        return model.generate(**inputs,**kwargs)
    output=measure('original_hf_generate_natural_stop',lambda:hf_generate(False),n=3)
    print('NATURAL_GENERATED_TOKENS',output.sequences.shape[-1]-inputs['input_ids'].shape[-1],flush=True)
    measure('hf_generate_fixed_32',lambda:hf_generate(True),n=3)
    runner=RequestRunner(model,'base')
    measure('new_custom_base_loop_fixed_32',lambda:runner.request(inputs,32),n=3)
    runner=RequestRunner(model,'embedding_adapter',adapter,'fast',cache_prefill)
    measure('new_custom_adapter_loop_fixed_32',lambda:runner.request(inputs,32),n=3)
    from src.evaluate import generate_adapter_qwen_decode_cache
    def original_adapter_generate():
        logits,mask,hidden,pos,cache=cache_prefill(inputs)
        return generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,32,
            initial_hidden=hidden,position_ids=pos,prefill_logits=logits,prefill_text_mask=mask,decode_cache=cache,
            decode_cache_mode='fast',early_stop_metric=ds.spec.metric,choices=item.get('choices'))
    output=measure('original_adapter_generate_natural_structured_stop',original_adapter_generate,n=3)
    print('ORIGINAL_ADAPTER_OUTPUT',output,flush=True)
