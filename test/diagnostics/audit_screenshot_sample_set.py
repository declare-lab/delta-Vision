"""Match the screenshot's analytic resource fingerprint to local MMStar subsets."""
from pathlib import Path
import sys,json,statistics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from transformers import AutoProcessor,AutoConfig
from src.data import QwenBenchmarkDataset
from src.benchmarks import estimate_qwen_kv_cache_mb,estimate_qwen_prefill_flops
model='/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
processor=AutoProcessor.from_pretrained(model)
config=AutoConfig.from_pretrained(model).text_config
out=[]
for name in ['mmstar_speedtest_200.jsonl','mmstar_val.jsonl']:
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/mmstar'/name),processor,'mmstar',data_root=str(ROOT/'data/benchmarks/mmstar'),max_samples=200)
    values=[]
    for i in range(len(ds)):
        item=ds[i]
        v=int(item['mm_token_type_ids'].ne(0).sum()); t=item['input_ids'].numel()-v
        values.append(dict(index=item['index'],text=t,visual=v,
            base_kv=estimate_qwen_kv_cache_mb(config,text_tokens=t,image_tokens=v,dtype_bytes=2,adapter=False),
            base_flops=estimate_qwen_prefill_flops(config,text_tokens=t,image_tokens=v),
            adapter_flops=estimate_qwen_prefill_flops(config,text_tokens=t,image_tokens=v,adapter_mode='embedding_adapter',visual_adapter_rank=128)))
        if (i+1)%50==0:print(name,i+1,flush=True)
    row={'dataset':name,'samples':len(ds),**{k:statistics.mean(v[k] for v in values) for k in ['base_kv','base_flops','adapter_flops','text','visual']}}
    print(json.dumps(row),flush=True)
    out.append(row)
    (ROOT/'test/results/prefill_timing_audit_20260915/sample_fingerprints.json').write_text(json.dumps(out,indent=2))
