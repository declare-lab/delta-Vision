"""Independent HF input and per-image batching checks; no model intervention.

Eight preselected Image-Text Matching examples, not a benchmark re-evaluation.
Compare custom input preparation against inputs captured from actual HF forward;
also compare joint/per-image vision encoding and stacked/ordinary adapter MLPs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/diagnostics/muir_input_independence_20260914'


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL, CHECKPOINT, sha
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    dataset = QwenBenchmarkDataset(str(ROOT / 'data/benchmarks/muirbench/test.jsonl'),
        processor, 'muirbench', max_samples=1000, prompt_layout='media_first_v1')
    candidates = [i for i, row in enumerate(dataset.rows) if row['task'] == 'Image-Text Matching']
    index = candidates[shard * len(candidates) // 8]
    inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda'))
    captured = {}

    class CapturedInput(Exception):
        pass

    def capture(module, args, kwargs):
        # HF has already computed embeddings and positions independently here.
        captured['hidden'] = kwargs['inputs_embeds'].clone()
        captured['positions'] = kwargs['position_ids'].clone()
        raise CapturedInput()

    def error(a, b):
        assert a.shape == b.shape, (a.shape, b.shape)
        a, b = a.float(), b.float()
        return dict(equal=bool(torch.equal(a, b)), max_abs=float((a-b).abs().max()),
                    relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)))

    with torch.inference_mode():
        hook = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            model(**inputs, use_cache=False)
        except CapturedInput:
            pass
        finally:
            hook.remove()
        assert captured, 'Native forward did not reach language-model input'
        initial, positions = ref.build_qwen_initial_context(model, inputs)
        visual_mask = inputs['input_ids'][0].eq(model.config.image_token_id)
        assert torch.equal(visual_mask, inputs['mm_token_type_ids'][0].ne(0))
        result = dict(index=index, checkpoint_sha256=sha(CHECKPOINT), deepstack_executed=False,
            native_custom_embedding=error(captured['hidden'], initial),
            native_custom_positions=error(captured['positions'], positions))
        # No language block ran, so DeepStack could not execute on either path.
        assert result['native_custom_embedding']['equal']
        assert result['native_custom_positions']['equal']
        grids = inputs['image_grid_thw']
        patch_counts = grids.prod(-1).tolist()
        token_counts = [n // model.model.visual.spatial_merge_size**2 for n in patch_counts]
        result['image_token_counts'] = token_counts
        visual = initial[:, visual_mask]
        independent = []
        offset = 0
        for i, patches in enumerate(patch_counts):
            output = model.model.visual(
                inputs['pixel_values'][offset:offset+patches].to(model.model.visual.dtype),
                grid_thw=grids[i:i+1], return_dict=True)
            independent.append(output.pooler_output)
            offset += patches
        assert offset == inputs['pixel_values'].shape[0]
        result['joint_vs_separate_vision'] = error(visual[0], torch.cat(independent, dim=0))
        stacked = adapter.all_visual_memories_batched(visual)
        result['layers'] = []
        for layer in range(len(model.model.language_model.layers)):
            ordinary = adapter.visual_memory_for_layer(visual, layer)
            separate = torch.cat([adapter.visual_memory_for_layer(chunk, layer)
                for chunk in visual.split(token_counts, dim=1)], dim=1)
            result['layers'].append(dict(layer=layer,
                stacked_vs_ordinary=error(stacked[layer], ordinary),
                joint_vs_separate_adapter=error(ordinary, separate)))
        result['parameter_stack_matches_modules'] = all(
            torch.equal(adapter._down_stacked[i], adapter.visual_adapter_down[i].weight)
            and torch.equal(adapter._up_stacked[i], adapter.visual_adapter_up[i].weight)
            for i in range(len(adapter.visual_adapter_down)))
        assert result['parameter_stack_matches_modules']
    (OUTPUT / f'row_{shard}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    try:
        for shard in range(8):
            log = (OUTPUT / f'worker{shard}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, '-m', 'src.muir_input_independence_audit', '--shard', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard)),
                stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        assert not any(codes), codes
    finally:
        for log in logs:
            log.close()
    rows = [json.loads((OUTPUT / f'row_{i}.json').read_text()) for i in range(8)]
    summary = dict(samples=len(rows), indices=[r['index'] for r in rows],
        embedding_exact=all(r['native_custom_embedding']['equal'] for r in rows),
        positions_exact=all(r['native_custom_positions']['equal'] for r in rows),
        parameter_stacks_exact=all(r['parameter_stack_matches_modules'] for r in rows),
        max_separate_vision_relative_l2=max(r['joint_vs_separate_vision']['relative_l2'] for r in rows),
        max_stacked_adapter_relative_l2=max(l['stacked_vs_ordinary']['relative_l2'] for r in rows for l in r['layers']),
        max_separate_adapter_relative_l2=max(l['joint_vs_separate_adapter']['relative_l2'] for r in rows for l in r['layers']))
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    run() if args.shard is None else worker(args.shard)
