"""Verify independent requests, growing KV and packed-cache alias fallback."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8',HF_HUB_DISABLE_PROGRESS_BARS='1')
import torch
from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
from src.benchmarking.engines.adapter import MODEL, CHECKPOINT, MANIFEST, tensor_sha, cache_tensors
from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
from src.model import load_qwen_embedding_adapter_checkpoint
from src.attention import optimize_qwen_attention_metadata
from src.model_setup import disable_qwen_deepstack
from src.data import QwenBenchmarkDataset


def main():
    torch.set_num_threads(4)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,torch.device('cuda:0'),1.,'flash_attention_2')
    disable_qwen_deepstack(model)
    optimize_qwen_attention_metadata(model)
    adapter,_=load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT),model.model.language_model,torch.device('cuda:0'),torch.bfloat16)
    args=SimpleNamespace(last_logits_only=True,attn_implementation='flash_attention_2',cuda_graph=True,
        cuda_graph_context=True,compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,
        adapter_decode_cache_mode='fast',adapter_exact_optimizations=True,adapter_max_optimizations=True)
    prefill=build_qwen_fast_adapter_prefill(model,adapter,args)
    decoder=model._adapter_decode_graph_runner
    decoder.allow_capture=True
    dataset=QwenBenchmarkDataset(str(MANIFEST),processor,'videomme',data_root=str(ROOT/'data/benchmarks/videomme'),
        cache_dir=str(ROOT/'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme'))
    row=json.loads((ROOT/'test/results/adapter_exact_20260915/videomme999_final/optimized_0.jsonl').read_text().splitlines()[0])
    assert row['index']==0
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    def token(logits):
        scores=logits[:,-1].float().clone();scores[:,eos]=-float('inf')
        return scores.argmax(-1).view(1,1)
    with torch.inference_mode():
        inputs=_qwen_inputs_from_item(dataset[0],torch.device('cuda:0'))
        modified=dict(inputs,pixel_values_videos=inputs['pixel_values_videos']*0.9)
        a,_,_,_,cache_a=prefill(inputs)
        a=a.clone()
        assert tensor_sha([a])==row['logits_sha256'][0]
        prefix_a=tensor_sha(cache_tensors(cache_a))
        assert prefix_a==row['prefill_kv_sha256']
        b,_,_,_,cache_b=prefill(modified)
        b=b.clone()
        assert tensor_sha(cache_tensors(cache_a))==prefix_a
        assert tensor_sha([a])!=tensor_sha([b]),'Modified frames did not change first logits'
        ownership_checks=1
        for step in range(7):
            if step==3:
                # Replacing a view with an equal tensor must invalidate the
                # bulk-copy shortcut and copy the actual per-layer cache.
                cache_a['_native_cache'].layers[0].keys=cache_a['_native_cache'].layers[0].keys.clone()
            hash_a=tensor_sha(cache_tensors(cache_a));logits_a=tensor_sha([a])
            b,cache_b=decoder(model,adapter,token(b),cache_b,logits_to_keep=1)
            assert tensor_sha(cache_tensors(cache_a))==hash_a and tensor_sha([a])==logits_a
            hash_b=tensor_sha(cache_tensors(cache_b));logits_b=tensor_sha([b])
            a,cache_a=decoder(model,adapter,token(a),cache_a,logits_to_keep=1)
            assert tensor_sha(cache_tensors(cache_b))==hash_b and tensor_sha([b])==logits_b
            assert tensor_sha([a])==row['logits_sha256'][step+1]
            assert all(l.keys.shape[-2]==1138+step+1 for l in cache_a['_native_cache'].layers)
            ownership_checks+=2
        assert tensor_sha(cache_tensors(cache_a))==row['final_kv_sha256']
    result=dict(independent_cache_and_logits_checks=ownership_checks,
        all_logits_and_final_kv_match_saved_exact_adapter=True,changed_frames_change_output=True,
        replaced_layer_view_falls_back_correctly=True,actual_decode_steps_per_request=7)
    path=ROOT/'test/results/adapter_max_20260915/ownership.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    decoder.remove()


if __name__=='__main__':main()
