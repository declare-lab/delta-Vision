"""Locate Zoo prefill cost and compare exact existing RMSNorm fusion in place."""
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.benchmarking.engines.adapter import MODEL, MANIFEST, tensor_sha
from src.benchmarking.engines.pruning import INPUT_CACHE, layer_kv
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.generation_timing import GenerationStageTimer
from src.attention import optimize_qwen_attention_metadata
from src.model_setup import disable_qwen_deepstack
from src.kernels import FusedQwenNorms
from src.graphs import NativeDecoderGraphs


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('zoo', MODEL, torch.bfloat16, torch.device('cuda:0'), .2, 'flash_attention_2')
    disable_qwen_deepstack(model)
    metadata = optimize_qwen_attention_metadata(model)
    selectors = NativeDecoderGraphs(model, max_shapes=12, vision=False, full_decode=False, prefill_layers=False)
    norms = FusedQwenNorms(model)
    norms.native_order = True
    timer = GenerationStageTimer(model, measure_memory=True)
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme', data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=str(INPUT_CACHE))
    implementation = sys.modules[type(model.model.language_model).__module__]
    output = ROOT/'test/results/video_pruning_fa2_metadata_20260915/zoo_stage_diagnosis.json'
    all_rows = []
    with torch.inference_mode():
        for index in [0,333,666]:
            inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda:0'))
            visual = inputs['mm_token_type_ids'][0].nonzero().flatten()
            configure_baseline(model,'zoo',.2,int(visual[0]),len(visual))
            def request(check=False):
                torch.manual_seed(42+index)
                model.model.rope_deltas=None
                torch.cuda.synchronize()
                timer.begin();start=time.perf_counter();timer.mark_request_start(start)
                result=model.generate(**inputs,min_new_tokens=8,max_new_tokens=8,do_sample=False,
                    disable_compile=True,return_dict_in_generate=True,output_logits=check)
                torch.cuda.synchronize()
                measured=timer.finish(time.perf_counter()-start,8)
                if check:
                    hashes=[tensor_sha([x]) for x in result.logits]+[tensor_sha(layer_kv(result.past_key_values))]
                    return measured,hashes
                return measured,None
            selectors.allow_capture=True
            norms.enabled=False
            _,reference=request(True)
            norms.enabled=True
            _,candidate=request(True)
            assert candidate==reference,(index,'exact norm parity')
            selectors.allow_capture=False
            for enabled in [False,True]:
                norms.enabled=enabled
                request()
            timings={False:[],True:[]}
            for repeat in range(5):
                for enabled in ([False,True] if repeat%2==0 else [True,False]):
                    norms.enabled=enabled
                    row,_=request()
                    timings[enabled].append(row)
            stages={}
            for enabled in [False,True]:
                norms.enabled=enabled
                records={}
                def wrap(name,fn):
                    def forward(*a,**kw):
                        torch.cuda.synchronize();start=time.perf_counter()
                        result=fn(*a,**kw)
                        torch.cuda.synchronize()
                        records.setdefault(name,[]).append(1000*(time.perf_counter()-start))
                        return result
                    return forward
                with patch.object(model.model.visual,'forward',wrap('vision',model.model.visual.forward)), \
                     patch.object(model.model.language_model,'forward',wrap('language',model.model.language_model.forward)), \
                     patch.object(implementation,'_zoo_token_sensitivity',wrap('sensitivity',implementation._zoo_token_sensitivity)), \
                     patch.object(implementation,'_zoo_select_tokens',wrap('selector',implementation._zoo_select_tokens)):
                    for _ in range(3):request()
                stages[str(enabled)]={k:statistics.mean(v if k!='language' else v[::8]) for k,v in records.items()}
            row=dict(index=index,exact_norm_logits_and_kv=True,
                timing={str(enabled):dict(prefill_ms=statistics.median(t['request_prefill_time_s'] for t in rows)*1000,
                    decode_ms_per_token=statistics.median(t['decode_time_s'] for t in rows)*1000/7,
                    peak_mib=max(t['peak_memory_mb'] for t in rows)) for enabled,rows in timings.items()},
                synchronous_stage_ms=stages)
            all_rows.append(row);output.write_text(json.dumps(all_rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    timer.remove();norms.remove();selectors.remove();metadata.remove()


if __name__=='__main__':main()
