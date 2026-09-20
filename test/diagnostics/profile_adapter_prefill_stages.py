"""Locate host preparation/copy costs around the warmed adapter fast path."""
import functools
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import src.model as model_module
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.benchmark_prefill import build_qwen_fast_adapter_prefill, QwenContextCudaGraphRunner, QwenAdapterPrefillCacheCudaGraphRunner
from src.qwen_adapter_native_graph import NativeAdapterDecodeGraphs
from src.qwen_deepstack import disable_qwen_deepstack
from src.data import QwenBenchmarkDataset


def main():
    torch.set_num_threads(4)
    model, processor = load_baseline_model('base', '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct', torch.bfloat16, 'cuda:0', 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    checkpoint = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
    adapter, _ = model_module.load_qwen_embedding_adapter_checkpoint(str(checkpoint), model.model.language_model, torch.device('cuda:0'), torch.bfloat16)
    active = False
    stages = {}
    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            if not active:
                return original(*args, **kwargs)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            stages.setdefault(label, []).append((time.perf_counter()-begin)*1000)
            return result
        setattr(owner, name, call)
    wrap(QwenContextCudaGraphRunner, '__call__', 'context')
    wrap(QwenAdapterPrefillCacheCudaGraphRunner, '__call__', 'adapter_prefill_inclusive')
    wrap(model_module, 'prepare_qwen_embedding_adapter_inputs', 'prepare_adapter_inputs')
    wrap(NativeAdapterDecodeGraphs, 'prepare_cache', 'pack_owned_decode_cache')
    settings = SimpleNamespace(last_logits_only=True, attn_implementation='flash_attention_2', cuda_graph=True,
        cuda_graph_context=True, compile_verify=True, compile_max_diff=0., cuda_graph_warmup=3, adapter_decode_cache_mode='fast')
    prefill = build_qwen_fast_adapter_prefill(model, adapter, settings)
    path = ROOT/'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    dataset = QwenBenchmarkDataset(str(path), processor, 'mmstar', data_root=str(path.parent), max_samples=200)
    rows=[]
    output=ROOT/'test/results/screenshot_adapter_fastv_20260915/adapter_stage_profile.json'
    with torch.inference_mode():
        for index in [0,25,125,133]:
            inputs=_qwen_inputs_from_item(dataset[index],torch.device('cuda:0'))
            active=False
            for _ in range(3): prefill(inputs)
            times=[]
            for _ in range(12):
                torch.cuda.synchronize();start=time.perf_counter()
                prefill(inputs)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            active=True;stages={}
            for _ in range(6):prefill(inputs)
            active=False
            row=dict(index=index,uninstrumented_prefill_median_ms=statistics.median(times),
                synchronized_stage_median_ms={k:statistics.median(v) for k,v in stages.items()},
                note='Stage timings add synchronization and adapter_prefill_inclusive includes preparation/cache packing; do not sum nested stages.')
            rows.append(row);output.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)


if __name__=='__main__':main()
