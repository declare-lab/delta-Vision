"""Base only: isolate the cost of native Qwen RMSNorm without changing FA2."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.generation_timing import GenerationStageTimer
from src.model_setup import disable_qwen_deepstack
from src.kernels import FusedQwenNorms
from src.graphs import NativeDecoderGraphs
from paired_runtime_execution import MODEL, assert_cache, sync


def main():
    torch.set_num_threads(4)
    out = ROOT / 'test/results/focused_base_zip_adapter_20260915/base_norms_exact'
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    models, graphs, timers = {}, {}, {}
    for label in ['original', 'fused']:
        models[label], processor = load_baseline_model('base', MODEL, torch.bfloat16, device, 1., 'flash_attention_2')
        disable_qwen_deepstack(models[label])
        if label == 'fused':
            norms = FusedQwenNorms(models[label])
            assert len(norms.originals) == 145
        graphs[label] = NativeDecoderGraphs(models[label], max_shapes=16, fused_norms=False)
        timers[label] = GenerationStageTimer(models[label])
    path = ROOT / 'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    data = QwenBenchmarkDataset(str(path), processor, 'mmstar', data_root=str(path.parent), max_samples=200)
    trials, checks = [], []

    def request(label, inputs, check=False):
        model, timer = models[label], timers[label]
        model.model.rope_deltas = None
        sync();timer.begin();start = time.perf_counter()
        result = model.generate(**inputs, min_new_tokens=8, max_new_tokens=8, do_sample=False,
            return_dict_in_generate=True, output_logits=check)
        sync();elapsed = time.perf_counter()-start
        measured = timer.finish(elapsed, 8)
        measured.update(total_time_s=elapsed, tokens=result.sequences[0, inputs['input_ids'].shape[-1]:].tolist())
        return measured, result

    with torch.inference_mode():
        for index in [0,25,125,133]:
            inputs = _qwen_inputs_from_item(data[index], device)
            references = {}
            for label in models:
                graphs[label].enabled = False
                references[label] = request(label, inputs, True)
            a, b = references['original'][1], references['fused'][1]
            differences = [float((x-y).abs().max()) for x,y in zip(a.logits,b.logits)]
            tokens_equal = torch.equal(a.sequences,b.sequences)
            assert_cache(a.past_key_values, b.past_key_values)
            checks.append(dict(index=index,tokens_equal=tokens_equal,logits_bitwise_equal=all(d==0 for d in differences),max_logit_diff=differences))
            (out/'checks.json').write_text(json.dumps(checks,indent=2))
            assert tokens_equal, (index, differences)
            # Validate each graph against its own arithmetic separately.
            for label in models:
                graphs[label].enabled = graphs[label].allow_capture = True
                measured, candidate = request(label, inputs, True)
                reference = references[label][1]
                assert torch.equal(reference.sequences,candidate.sequences)
                assert all(torch.equal(x,y) for x,y in zip(reference.logits,candidate.logits))
                assert_cache(reference.past_key_values,candidate.past_key_values)
                graphs[label].allow_capture = False
            del a,b,candidate,reference,references
            before = {label:g.stats() for label,g in graphs.items()}
            for repetition in range(12):
                pair = dict(index=index,repetition=repetition)
                for label in (['original','fused'] if repetition%2==0 else ['fused','original']):
                    pair[label], result = request(label,inputs)
                    del result
                trials.append(pair)
            for label,g in graphs.items():
                assert g.stats()['captures'] == before[label]['captures']
                assert g.stats()['cold_layer_fallbacks'] == before[label]['cold_layer_fallbacks']
            summary = {}
            for stage,key,div in [('prefill','generation_prefill_time_s',1),('decode','decode_time_s',7),('total','total_time_s',1)]:
                summary[stage] = dict(speedup=statistics.median(t['original'][key]/t['fused'][key] for t in trials),
                    **{label+'_median_ms':statistics.median(t[label][key]*1000/div for t in trials) for label in models})
            (out/'results.json').write_text(json.dumps(dict(checks=checks,trials=trials,summary=summary,
                deepstack='off',attention='flash_attention_2',tokens=8,shared_gpu=True),indent=2))
            print(index, json.dumps(summary), flush=True)
    for g in graphs.values():g.remove()
    for t in timers.values():t.remove()
    norms.remove()


if __name__ == '__main__':main()
