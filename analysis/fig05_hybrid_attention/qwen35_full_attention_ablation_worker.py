"""Paired native/adapter accuracy with two FA visual-readout paths removed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'artifacts/dependencies/qwen35_python'))
import torch

from src.benchmarks import build_benchmark_prompt, get_benchmark_spec
from src.qwen35 import load_model, prepare_inputs, generate_evaluation_answer, sha
from analysis.fig05_hybrid_attention.qwen35_full_attention_ablation import remove_full_attention_visual_effect


@torch.inference_mode()
def validate(model, processor, controller, inputs, mask, row, spec, config, info):
    # Test actual FA2 against a dense reference on small tensors, including GQA,
    # image masking, unchanged visual outputs and persistent decode blocking.
    import importlib.util
    path = ROOT / 'test/diagnostics/test_qwen35_full_attention_ablation.py'
    loader = importlib.util.spec_from_file_location('fa_ablation_tests', path)
    tests = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(tests)
    from transformers.integrations.flash_attention import flash_attention_forward
    tests.check_readout('cuda', flash_attention_forward, torch.bfloat16)
    checks = []
    for method in config['methods']:
        with controller.activate(method, mask):
            ref = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            generated = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                    max_new_tokens=info['max_new_tokens'])
        with controller.activate(method, mask), remove_full_attention_visual_effect(
                model, mask, config['ablated_layers'], block=False) as audit:
            restored = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            control = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                  max_new_tokens=info['max_new_tokens'])
        torch.testing.assert_close(ref, restored, rtol=0, atol=0)
        assert generated['generated_token_ids'] == control['generated_token_ids']
        assert {c['layer'] for c in audit['calls']} == set(config['ablated_layers'])
        checks.append(dict(method=method, no_op_logits_exact=True, no_op_generation_exact=True))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=['validate', 'accuracy'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(44)
    processor, model, adapter, controller = load_model(config, torch.device('cuda:0'))
    torch.set_float32_matmul_precision('highest')
    assert config['methods'] and set(config['methods']) <= {'native', 'adapter'}
    if 'adapter' in config['methods']:
        assert sha(config['adapter_checkpoint']) == config['adapter_checkpoint_sha256']
        checkpoint = torch.load(config['adapter_checkpoint'], map_location='cpu', weights_only=False)
        assert checkpoint['global_step'] == 2000
        adapter.load_state_dict(checkpoint['state_dict'], strict=True)
        adapter.eval().requires_grad_(False)
        del checkpoint
    for benchmark, info in config['evaluation'].items():
        assert sha(info['path']) == info['sha256']
        rows = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
        assert len(rows) == info['samples']
        indices = [0] if args.stage == 'validate' else list(range(args.shard, len(rows), args.shards))
        path = args.run / args.stage / f'{benchmark}.shard{args.shard}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        assert [r['index'] for r in old] == indices[:len(old)]
        with path.open('a') as handle, torch.inference_mode():
            for index in indices[len(old):]:
                start = time.time()
                row = rows[index]
                spec = get_benchmark_spec(benchmark)
                inputs, _ = prepare_inputs(processor, row, info['image_root'], torch.device('cuda:0'),
                                           question=build_benchmark_prompt(row, spec))
                mask = inputs['mm_token_type_ids'].eq(1)
                result = dict(benchmark=benchmark, index=index, layers=config['ablated_layers'],
                              input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                              sequence_length=mask.shape[1], visual_tokens=int(mask.sum()), variants=[])
                if args.stage == 'validate':
                    result['checks'] = validate(model, processor, controller, inputs, mask, row, spec, config, info)
                for method in config['methods']:
                    with controller.activate(method, mask):
                        baseline = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                               max_new_tokens=info['max_new_tokens'])
                    with controller.activate(method, mask), remove_full_attention_visual_effect(
                            model, mask, config['ablated_layers']) as audit:
                        blocked = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                              max_new_tokens=info['max_new_tokens'])
                    # Both selected layers must execute for every generation step;
                    # all image KV remain physically present through cached decode.
                    for layer in config['ablated_layers']:
                        calls = [c for c in audit['calls'] if c['layer'] == layer]
                        assert len(calls) == blocked['generated_tokens']
                        assert calls[0]['query_length'] == mask.shape[1]
                        assert [c['cache_length'] for c in calls] == list(range(mask.shape[1], mask.shape[1] + len(calls)))
                        assert all(c['query_length'] == 1 for c in calls[1:])
                    result['variants'].extend([
                        dict(method=method, condition='unmodified', **baseline),
                        dict(method=method, condition=config.get('ablation_condition', 'block_two_fa'), **blocked, audit=audit)])
                result['elapsed_s'] = time.time() - start
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                print(json.dumps(dict(stage=args.stage, benchmark=benchmark, index=index,
                                      elapsed_s=result['elapsed_s'])), flush=True)
                del inputs, result
    controller.close()


if __name__ == '__main__':
    main()
