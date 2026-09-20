"""Locate exact adapter optimizations on real inputs, without changing reports."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.model import (load_qwen_embedding_adapter_checkpoint, build_qwen_initial_context,
    prepare_qwen_embedding_adapter_inputs, qwen_embedding_adapter_prefill_cache_prepared)
from src.qwen_deepstack import disable_qwen_deepstack
from src.qwen_fused_norm import FusedQwenNorms

MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    return graph, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='+', type=int, default=[0, 125])
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--benchmark', choices=['mmstar', 'videomme'], default='mmstar')
    parser.add_argument('--output', default='test/results/adapter_exact_20260915/probe.json')
    args = parser.parse_args()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    model._adapter_attention_implementation = 'flash_attention_2'
    adapter, _ = load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT), model.model.language_model,
        torch.device('cuda:0'), torch.bfloat16)
    norms = FusedQwenNorms(model)
    norms.enabled = False
    path = (ROOT / 'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl' if args.benchmark == 'mmstar'
            else ROOT / 'artifacts/diagnostics/video_balanced_base_adapter_20260914/videomme_selected.jsonl')
    dataset = QwenBenchmarkDataset(str(path), processor, args.benchmark,
        data_root=str(ROOT / 'data/benchmarks' / args.benchmark))
    rows = []
    with torch.inference_mode():
        for index in args.samples:
            norms.enabled = False
            inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda:0'))
            hidden, positions = build_qwen_initial_context(model, inputs)
            prepared = prepare_qwen_embedding_adapter_inputs(model, adapter, inputs['input_ids'],
                inputs['attention_mask'], inputs['mm_token_type_ids'], hidden, positions)
            vm = prepared['visual_memory']
            batched = adapter.all_visual_memories_batched(vm)
            serial = torch.stack([adapter.visual_memory_for_layer(vm, i) for i in range(adapter.num_layers)])
            row = dict(index=index, text_tokens=prepared['h'].shape[1], visual_tokens=vm.shape[1],
                batched_visual_bitwise_equal=torch.equal(batched, serial),
                batched_visual_max_diff=float((batched-serial).abs().max()),
                batched_visual_unequal=int((batched != serial).sum()))
            del batched, serial
            def forward():
                return qwen_embedding_adapter_prefill_cache_prepared(model, adapter, **prepared,
                    logits_to_keep=1, retain_prefix_states=False, batch_visual_memories=batched_enabled,
                    exact_kernels=kernels_enabled)
            graphs, outputs = {}, {}
            labels = ['original', 'norms', 'batched', 'norms_batched', 'norms_batched_kernels']
            for label in labels:
                norms.enabled = 'norms' in label
                batched_enabled = 'batched' in label
                kernels_enabled = 'kernels' in label
                graphs[label], outputs[label] = capture(forward)
            a, b = outputs['original'], outputs['norms']
            row.update(norms_logits_bitwise_equal=torch.equal(a[0], b[0]),
                norms_logits_max_diff=float((a[0]-b[0]).abs().max()),
                norms_kv_bitwise_equal=all(torch.equal(x[k], y[k])
                    for x, y in zip(a[2]['layers'], b[2]['layers']) for k in x))
            row['parity'] = {label:dict(logits_exact=torch.equal(a[0], outputs[label][0]),
                logits_max_diff=float((a[0]-outputs[label][0]).abs().max()),
                kv_exact=all(torch.equal(x[k], y[k]) for x,y in zip(a[2]['layers'], outputs[label][2]['layers']) for k in x))
                for label in labels[1:]}
            trials = []
            for repetition in range(args.pairs):
                trial = {}
                for label in (labels if repetition % 2 == 0 else labels[::-1]):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graphs[label].replay()
                    torch.cuda.synchronize()
                    trial[label] = (time.perf_counter()-start)*1000
                trials.append(trial)
            row['prefill_decoder_median_ms'] = {label:statistics.median(t[label] for t in trials) for label in graphs}
            row['trials'] = trials
            norms.enabled = False
            batched_enabled = False
            kernels_enabled = False
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                forward()
                torch.cuda.synchronize()
            (output_path.parent/f'original_ops_{index}.txt').write_text(prof.key_averages().table(
                sort_by='self_cuda_time_total', row_limit=35))
            rows.append(row)
            output_path.write_text(json.dumps(rows, indent=2))
            print(json.dumps({k:v for k,v in row.items() if k != 'trials'}), flush=True)
            del graphs, outputs, a, b
    norms.remove()


if __name__ == '__main__':
    main()
