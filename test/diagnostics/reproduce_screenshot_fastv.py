"""Recover screenshot FastV selection, retaining explicitly requested FA2/DeepStack off.

The archived forward is used only in this diagnostic, not the production baseline.
Source: historical_evidence/line_11936.txt and provenance.json in the result folder.
"""
import gc
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, evaluate_single, set_global_seed
from src.data import QwenBenchmarkDataset
from src.qwen_deepstack import disable_qwen_deepstack


def restore_screenshot_forward(model):
    language = model.model.language_model
    source = Path(__file__).with_name('fastv_screenshot_forward.txt').read_text()
    scope = vars(sys.modules[type(language).__module__]).copy()
    tree = ast.parse(source)
    # Documentation generation expects a class-qualified method; it has no
    # runtime effect. Retain the config-default and output-capture decorators.
    tree.body[0].decorator_list = [d for d in tree.body[0].decorator_list
        if not isinstance(d, ast.Name) or d.id != 'auto_docstring']
    exec(compile(tree, 'fastv_screenshot_forward.txt', 'exec'), scope)
    language.forward = types.MethodType(scope['forward'], language)
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-base', action='store_true', help='Reuse the saved base after an interrupted FastV run.')
    parser.add_argument('--output', default='test/results/screenshot_adapter_fastv_20260915/recovered_fastv_fa2')
    parser.add_argument('--native-cuda-graphs', action='store_true')
    parser.add_argument('--optimize-attention-metadata', action='store_true')
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()
    set_global_seed(42)
    output = ROOT / args.output
    output.mkdir(exist_ok=True)
    data_path = ROOT / 'data/benchmarks/mmstar/mmstar_speedtest_200.jsonl'
    model_path = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
    reference = None
    for method in ['base', 'fastv']:
        if method == 'base' and args.resume_base:
            reference = json.loads((output/'base.json').read_text())
            continue
        model, processor = load_baseline_model(method, model_path, torch.bfloat16, 'cuda:0', .05, 'flash_attention_2')
        disable_qwen_deepstack(model)
        source_hash = restore_screenshot_forward(model) if method == 'fastv' else None
        data = QwenBenchmarkDataset(str(data_path), processor, 'mmstar', data_root=str(data_path.parent), max_samples=args.samples)
        visual = data[0]['mm_token_type_ids'].nonzero().flatten()
        configure_baseline(model, method, .05, int(visual[0]), len(visual))
        metadata = graphs = None
        if args.optimize_attention_metadata:
            from src.qwen_attention_metadata import optimize_qwen_attention_metadata
            metadata = optimize_qwen_attention_metadata(model)
        if args.native_cuda_graphs:
            from src.qwen_native_graph import NativeDecoderGraphs
            graphs = NativeDecoderGraphs(model)
        predictions, summary = evaluate_single(model, processor, data, 'mmstar', 8,
            log_every=25, method=method, retention=.05, measure_prefill=True, speed_warmup=1,
            measure_decode=True, measure_peak_memory=True, native_graphs=graphs)
        if method == 'base':
            reference = summary.copy()
        summary.update(method=method, attention='flash_attention_2', deepstack='off',
            native_cuda_graphs=args.native_cuda_graphs, attention_metadata_optimized=args.optimize_attention_metadata,
            source_forward_sha256=source_hash, source_selection='screenshot_20260825' if method == 'fastv' else 'native',
            speedup_total=reference['total_time_s']/summary['total_time_s'],
            speedup_prefilling=reference['prefilling_time_s']/summary['prefilling_time_s'])
        (output/f'{method}.json').write_text(json.dumps(summary, indent=2))
        (output/f'{method}.predictions.json').write_text(json.dumps(predictions, indent=2))
        print(json.dumps(summary), flush=True)
        if graphs is not None: graphs.remove()
        if metadata is not None: metadata.remove()
        del model, processor, data
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
