"""Evaluate DART / DivPrune on the frozen Qwen3.5 nine-benchmark manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/dependencies/qwen35_python'))

import torch
from src.benchmarks import build_benchmark_prompt, get_benchmark_spec, score_prediction
from src.qwen35_experiment import dump, load_model, prepare_inputs, sha, generate_evaluation_answer
from src.qwen35_pruning import VisualPruningController
from scripts.qwen35_worker import read_rows, difference, compare_caches


@torch.inference_mode()
def validate(config, run):
    processor, model, adapter, old = load_model(config, torch.device('cuda:0'))
    old.close()
    del adapter, old
    pruning = VisualPruningController(model)
    report = {'checks': []}
    for benchmark in ('mmstar', 'realworldqa'):
        info = config['evaluation'][benchmark]
        row = read_rows(info['path'])[0]
        inputs, _ = prepare_inputs(processor, row, info['image_root'], torch.device('cuda:0'),
            question=build_benchmark_prompt(row, get_benchmark_spec(benchmark)))
        mask = inputs['mm_token_type_ids'].eq(1)
        native = model(**inputs, use_cache=True, logits_to_keep=1)
        for method in ('divprune', 'dart'):
            with pruning.activate(method, 1., mask):
                same = model(**inputs, use_cache=True, logits_to_keep=1)
            torch.testing.assert_close(native.logits, same.logits, atol=0, rtol=0)
            tensors = compare_caches(native.past_key_values, same.past_key_values)
            report['checks'].append(dict(benchmark=benchmark, method=method, test='100% native identity',
                                         exact_logits=True, exact_cache_tensors=tensors))
        del native, same
        native_errors = []
        for method, retention in [('native', 1.), ('divprune', .05), ('divprune', .2), ('dart', .05), ('dart', .2)]:
            from contextlib import nullcontext
            def context(mask, selected=None):
                return nullcontext() if method == 'native' else pruning.activate(method, retention, mask, selected)
            seq = {k: v.clone() for k, v in inputs.items()}
            with context(mask):
                cached = model(**seq, use_cache=True, logits_to_keep=1)
            selected = None if method == 'native' else pruning.audit['selected_visual_indices']
            if method != 'native':
                report['checks'].append(dict(benchmark=benchmark, method=method, retention=retention,
                                             test='token_budget', **pruning.audit))
            for step in range(3):
                token = cached.logits[:, -1].argmax(-1, keepdim=True)
                for key, extension in [('input_ids', token), ('attention_mask', torch.ones_like(token)),
                                       ('mm_token_type_ids', torch.zeros_like(token))]:
                    seq[key] = torch.cat((seq[key], extension), 1)
                positions = model._prepare_position_ids_for_generation(seq['input_ids'],
                    {'attention_mask': seq['attention_mask'], 'past_key_values': cached.past_key_values})[..., -1:]
                with context(mask):
                    cached = model(input_ids=token, attention_mask=seq['attention_mask'], position_ids=positions,
                        past_key_values=cached.past_key_values, use_cache=True, logits_to_keep=1)
                with context(seq['mm_token_type_ids'].eq(1), selected):
                    full = model(**seq, use_cache=False, logits_to_keep=1)
                error = difference(full.logits, cached.logits)
                report['checks'].append(dict(benchmark=benchmark, method=method, retention=retention,
                                             test='cached_vs_full_prefix', step=step, **error))
                if method == 'native':
                    native_errors.append(error['relative_rms'])
                else:
                    assert error['relative_rms'] < max(.02, 2*max(native_errors)), error
                del full
            del cached
        print('VALIDATED', benchmark, flush=True)
    report['passed'] = True
    dump(run/'validation.json', report)


@torch.inference_mode()
def evaluate(config, run, method, retention, shard, shards):
    processor, model, adapter, old = load_model(config, torch.device('cuda:0'))
    old.close()
    del adapter, old
    pruning = VisualPruningController(model)
    name = f'{method}_{round(retention*100)}'
    dest = run/'eval'/name
    dest.mkdir(parents=True, exist_ok=True)
    for benchmark, info in config['evaluation'].items():
        assert sha(info['path']) == info['sha256']
        rows = read_rows(info['path'])
        assert len(rows) == info['samples']
        spec = get_benchmark_spec(benchmark)
        path = dest/f'{benchmark}.shard{shard}.jsonl'
        completed = read_rows(path) if path.exists() else []
        expected = list(range(shard, len(rows), shards))
        assert [r['index'] for r in completed] == expected[:len(completed)]
        with path.open('a') as handle:
            for index in expected[len(completed):]:
                row = rows[index]
                inputs, _ = prepare_inputs(processor, row, info['image_root'], torch.device('cuda:0'),
                    question=build_benchmark_prompt(row, spec))
                with pruning.activate(method, retention, inputs['mm_token_type_ids'].eq(1)):
                    generated = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                          max_new_tokens=info['max_new_tokens'])
                result = dict(index=index, benchmark=benchmark, **generated,
                    input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                    image_grid_thw=inputs['image_grid_thw'].tolist(),
                    token_audit=pruning.audit)
                handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                handle.flush()
                if index % 40 < shards:
                    print(name, benchmark, index, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['validate', 'eval'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', choices=['divprune', 'dart'])
    parser.add_argument('--retention', type=float)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    config = json.loads((args.run_dir/'config.json').read_text())
    if args.stage == 'validate':
        validate(config, args.run_dir)
    else:
        evaluate(config, args.run_dir, args.method, args.retention, args.shard, args.shards)


if __name__ == '__main__':
    main()
