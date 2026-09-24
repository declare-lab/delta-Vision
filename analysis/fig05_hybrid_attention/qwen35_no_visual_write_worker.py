"""RealWorldQA: fresh native control versus beta=0 at all visual LA positions."""
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
from src.qwen35 import load_model, prepare_inputs, generate_evaluation_answer, sha
from src.benchmarks import get_benchmark_spec, build_benchmark_prompt
from analysis.fig05_hybrid_attention.qwen35_no_visual_write import no_visual_write


@torch.inference_mode()
def kernel_check():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    # With beta=0 the exact recurrent trajectory is just decayed initial state.
    torch.manual_seed(44)
    q, k, v = [torch.randn(1, 17, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(1, 17, 2, device='cuda') * .1
    beta = torch.zeros(1, 17, 2, device='cuda', dtype=torch.bfloat16)
    initial = torch.randn(1, 2, 128, 128, device='cuda')
    output, state = m.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
        initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected = initial * g.sum(1).exp()[..., None, None]
    torch.testing.assert_close(state, expected, rtol=2e-5, atol=2e-5)
    return dict(zero_beta_decay_only_state=True, state_max_abs_error=(state-expected).abs().max().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    args = p.parse_args()
    config = json.loads((args.run/'config.json').read_text())
    info = config['evaluation']['realworldqa']
    assert sha(info['path']) == info['sha256']
    rows = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
    assert len(rows) == info['samples'] == 765
    indices = list(range(args.shard, len(rows), 8))
    path = args.run/'accuracy'/f'realworldqa.shard{args.shard}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['index'] for r in old] == indices[:len(old)]
    torch.set_num_threads(4)
    torch.manual_seed(44)
    processor, model, adapter, controller = load_model(config, torch.device('cuda:0'))
    # Controller remains native throughout; no adapter checkpoint is loaded or used.
    torch.set_float32_matmul_precision('highest')
    gate = kernel_check()
    spec = get_benchmark_spec('realworldqa')
    with path.open('a') as handle, torch.inference_mode():
        for index in indices[len(old):]:
            start = time.time()
            row = rows[index]
            inputs, _ = prepare_inputs(processor, row, info['image_root'], torch.device('cuda:0'),
                                       question=build_benchmark_prompt(row, spec))
            mask = inputs['mm_token_type_ids'].eq(1)
            def generate():
                return generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                  max_new_tokens=info['max_new_tokens'])
            baseline = generate()
            first = index == indices[len(old)]
            if first:
                with no_visual_write(model, mask, enabled=False):
                    control = generate()
                assert baseline['generated_token_ids'] == control['generated_token_ids'], 'No-op generation mismatch'
                with no_visual_write(model, mask, enabled=False):
                    noop = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
                original = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
                torch.testing.assert_close(noop, original, rtol=0, atol=0)
                gate.update(no_op_logits_exact=True, no_op_generation_exact=True)
                (args.run/'validation'/f'shard{args.shard}.json').write_text(json.dumps(gate, indent=2)+'\n')
                del noop, original
            with no_visual_write(model, mask, verify=first) as audit:
                blocked = generate()
            assert len(audit['prefill_calls']) == 24
            assert all(n == 1 for n in audit['prefill_calls'].values())
            for i in audit['layers']:
                assert audit['decode_calls'].get(i, 0) == blocked['generated_tokens']-1
            result = dict(index=index, benchmark='realworldqa',
                input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                sequence_length=mask.shape[1], visual_tokens=int(mask.sum()), audit=audit,
                native=baseline, no_visual_write=blocked, elapsed_s=time.time()-start)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            handle.flush()
            print(json.dumps(dict(index=index, elapsed_s=result['elapsed_s'])), flush=True)
    controller.close()


if __name__ == '__main__':
    main()
