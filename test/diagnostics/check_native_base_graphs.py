"""Exercise native base generate with warmed decoder graphs and exact output checks."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import LogitsProcessor
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.generation_timing import GenerationStageTimer
from src.graphs import NativeDecoderGraphs


class Capture(LogitsProcessor):
    def __init__(self):
        self.values = []

    def __call__(self, ids, scores):
        self.values.append(scores.clone())
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--method", default="base")
    parser.add_argument("--retention", type=float, default=.05)
    parser.add_argument("--samples", nargs="+", type=int, default=[0, 25, 125, 133])
    parser.add_argument("--all-samples", action="store_true", help="Run the complete screenshot-matched MMStar 200 subset")
    parser.add_argument("--output", default="test/results/native_base_optimized_20260915/graph_check.json")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--audit-attention", action="store_true")
    parser.add_argument("--deepstack", choices=['off', 'native'], default='off')
    parser.add_argument("--screenshot-fastv", action='store_true')
    args = parser.parse_args()
    if args.all_samples:
        args.samples = list(range(200))
    torch.set_num_threads(4)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_baseline_model(args.method,
        "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct", torch.bfloat16,
        torch.device("cuda:0"), args.retention, args.attention)
    if args.deepstack == 'off':
        from src.model_setup import disable_qwen_deepstack
        disable_qwen_deepstack(model)
    if args.screenshot_fastv and args.method == 'fastv':
        from reproduce_screenshot_fastv import restore_screenshot_forward
        restore_screenshot_forward(model)
    data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
    graphs = NativeDecoderGraphs(model, max_shapes=8)
    timer = GenerationStageTimer(model)
    rows = []
    with torch.inference_mode():
        for index in args.samples:
            inputs = _qwen_inputs_from_item(dataset[index], torch.device("cuda:0"))
            visual = inputs["mm_token_type_ids"][0].nonzero().flatten()
            configure_baseline(model, args.method, args.retention, int(visual[0]), len(visual))

            def generate(capture=None):
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                model.model.rope_deltas = None
                return model.generate(**inputs, max_new_tokens=8, do_sample=False,
                    return_dict_in_generate=True, logits_processor=[capture] if capture is not None else None)

            graphs.enabled = False
            reference_capture = Capture()
            operators = None
            if args.audit_attention:
                def forbid_sdpa(*args, **kwargs):
                    raise AssertionError("Unexpected SDPA fallback")
                with patch("torch.nn.functional.scaled_dot_product_attention", forbid_sdpa), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    reference = generate(reference_capture)
                operators = {event.key: event.count for event in prof.key_averages() if "flash_attn" in event.key.lower()}
                assert operators, "No FlashAttention operator observed"
            else:
                reference = generate(reference_capture)
            graphs.enabled = graphs.allow_capture = True
            prepare_start = time.perf_counter()
            optimized_capture = Capture()
            optimized = generate(optimized_capture)
            torch.cuda.synchronize()
            prepare_s = time.perf_counter() - prepare_start
            graphs.allow_capture = False
            assert torch.equal(reference.sequences, optimized.sequences), index
            max_logit_diff = max(float((a-b).abs().max()) for a,b in zip(reference_capture.values, optimized_capture.values))
            max_kv_diff = max(float((getattr(a,key)-getattr(b,key)).abs().max())
                for a,b in zip(reference.past_key_values.layers,optimized.past_key_values.layers) for key in ['keys','values'])
            assert max_logit_diff == max_kv_diff == 0, (index,max_logit_diff,max_kv_diff)
            trials = []
            before = graphs.stats()
            for repetition in range(args.pairs):
                pair = {}
                for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                    graphs.enabled = enabled
                    torch.cuda.synchronize()
                    timer.begin()
                    start = time.perf_counter()
                    result = generate()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    count = result.sequences.shape[-1] - inputs['input_ids'].shape[-1]
                    measured = timer.finish(seconds,count)
                    measured['total_time_s'] = seconds
                    assert torch.equal(reference.sequences,result.sequences)
                    pair['graphs' if enabled else 'eager'] = measured
                trials.append(pair)
            after = graphs.stats()
            assert after['captures'] == before['captures']
            assert after['cold_layer_fallbacks'] == before['cold_layer_fallbacks']
            row = dict(deepstack=args.deepstack,screenshot_fastv=args.screenshot_fastv,attention_operators=operators,index=index,method=args.method,retention=args.retention,attention=args.attention,prepare_s=prepare_s,graphs=after,
                       logits_max_diff=max_logit_diff,kv_max_diff=max_kv_diff,outputs_equal=True,trials=trials,shared_gpu=True)
            for label in ['eager','graphs']:
                row[label+'_median_ms'] = {key:statistics.median(t[label][key]*1000 for t in trials)
                    for key in ['total_time_s','generation_prefill_time_s','decode_time_s']}
            rows.append(row)
            out.write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
    graphs.remove()
    timer.remove()


if __name__=='__main__':
    main()
