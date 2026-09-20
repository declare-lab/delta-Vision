"""Causal debugging of visual-memory layer ranges, frozen original checkpoint.

Teacher inputs are captured before each native decoder block, DeepStack off.
Only selected layer memories are restored; all text states evolve normally.
This is an oracle diagnostic, NOT a deployable method or benchmark score.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/diagnostics/muir_layer_localization_20260914'
RANGES = {'adapter': (0, 0), **{f'prefix_{k}': (0, k) for k in (1, 4, 8, 12, 18, 24, 30, 36)},
          **{f'suffix_{k}': (k, 36) for k in (4, 8, 12, 18, 24, 30)}}
if os.environ.get('MUIR_LOCALIZE_PHASE') == 'kv':
    OUTPUT = ROOT/'artifacts/diagnostics/muir_layer_kv_localization_20260914'
    RANGES = {'adapter': (0,0), 'prefix_36': (0,36), 'middle_12_18': (12,18),
              **{f'layer_{l}': (l,l+1) for l in range(12,18)},
              'key_12_18': (12,18), 'value_12_18': (12,18)}


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL, CHECKPOINT
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    def off(module, args, kwargs):
        return args, dict(kwargs, deepstack_visual_embeds=None)
    def reject(*args, **kwargs):
        raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off, with_kwargs=True)
    model.model.language_model._deepstack_process = reject
    dataset = QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),
        processor, 'muirbench', max_samples=1000, prompt_layout='media_first_v1')
    indices = [i for i, row in enumerate(dataset.rows) if row['task'] == 'Image-Text Matching']
    previous = {}
    for path in (ROOT/'artifacts/diagnostics/muir_readout_trace_validated_20260914').glob('rows_*.jsonl'):
        for line in path.open():
            row = json.loads(line)
            if row['layer'] == 0:
                previous[row['index'], row['method']] = row['prediction']
    token_ids = [processor.tokenizer.encode(c, add_special_tokens=False) for c in 'ABCD']
    assert all(len(x) == 1 for x in token_ids)
    token_ids = [x[0] for x in token_ids]
    original_memories = adapter.all_visual_memories_batched
    with torch.inference_mode(), (OUTPUT/f'rows_{shard}.jsonl').open('w', buffering=1) as out:
        for index in indices[shard::8]:
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, torch.device('cuda'))
            initial, pos = ref.build_qwen_initial_context(model, inputs)
            visual = inputs['mm_token_type_ids'][0].ne(0)
            predicted = original_memories(initial[:, visual])
            captured = {}
            def make_hook(layer):
                def capture(module, args, kwargs):
                    h = args[0] if args else kwargs['hidden_states']
                    captured[layer] = h[:, visual].clone()
                return capture
            hooks = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
                     for i, layer in enumerate(model.model.language_model.layers)]
            try:
                teacher = model(**inputs, use_cache=False, logits_to_keep=1).logits[0,-1].float()
            finally:
                for h in hooks:
                    h.remove()
            assert len(captured) == 36
            native = torch.stack([captured[l] for l in range(36)])
            assert native.shape == predicted.shape
            assert torch.equal(native[0], initial[:,visual])
            def record(name, logits):
                text = processor.tokenizer.decode([int(logits.argmax())])
                if name in ('native', 'adapter'):
                    key = 'base' if name == 'native' else 'static_kl'
                    assert text == previous[index, key], (index, name, text, previous[index, key])
                gold = ord(item['answer'])-65
                scores = logits[token_ids]
                competitors = scores.clone()
                competitors[gold] = -float('inf')
                out.write(json.dumps(dict(index=index, condition=name, prediction=text,
                    correct=text.strip()==item['answer'], gold=item['answer'],
                    gold_margin=float(scores[gold]-competitors.max()), option_logits=scores.tolist(),
                    native_relative_logit_error=float((logits-teacher).norm()/teacher.norm().clamp_min(1e-12)),
                    native_kl=float((teacher.softmax(-1)*(teacher.log_softmax(-1)-logits.log_softmax(-1))).sum()),
                    deepstack=False, diagnostic_only=True))+'\n')
            record('native', teacher)
            for condition, (start, end) in RANGES.items():
                memory = predicted.clone()
                projection = 'k_proj' if condition.startswith('key_') else 'v_proj' if condition.startswith('value_') else None
                patch_hooks = []
                if projection:
                    assert initial.shape[1]-int(visual.sum()) != int(visual.sum())
                    for l in range(start, end):
                        layer = model.model.language_model.layers[l]
                        projected = getattr(layer.self_attn, projection)(layer.input_layernorm(native[l]))
                        def replace(module, args, output, replacement=projected):
                            # Only the visual call; text keys/values remain live.
                            if output.shape[1] == replacement.shape[1]:
                                return replacement
                            return output
                        patch_hooks.append(getattr(layer.self_attn, projection).register_forward_hook(replace))
                else:
                    memory[start:end] = native[start:end]
                adapter.all_visual_memories_batched = types.MethodType(lambda self,*a,_m=memory,**kw:_m, adapter)
                try:
                    logits = ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                        initial_hidden=initial, position_ids=pos, logits_to_keep=1)[0][0,-1].float()
                finally:
                    adapter.all_visual_memories_batched = original_memories
                    for handle in patch_hooks:
                        handle.remove()
                assert torch.isfinite(logits).all(), (index, condition)
                record(condition, logits)
            print('DONE', index, flush=True)


def run():
    retry = os.environ.get('MUIR_LOCALIZE_RETRY')
    OUTPUT.mkdir(parents=True, exist_ok=bool(retry))
    jobs, logs = [], []
    try:
        for i in ([int(x) for x in retry.split(',')] if retry else range(8)):
            log = (OUTPUT/f'worker{i}.log').open('w')
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable, '-m', 'src.muir_layer_localization', str(i)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OMP_NUM_THREADS='4'),
                stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in jobs]
        assert not any(codes), codes
    finally:
        for log in logs:
            log.close()
    rows = [json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows) == len({(r['index'], r['condition']) for r in rows}) == 84*(len(RANGES)+1)
    result = {}
    for name in ['native', *RANGES]:
        rs = [r for r in rows if r['condition']==name]
        result[name] = dict(samples=len(rs), correct=sum(r['correct'] for r in rs),
            accuracy=100*sum(r['correct'] for r in rs)/len(rs),
            mean_gold_margin=sum(r['gold_margin'] for r in rs)/len(rs),
            mean_native_kl=sum(r['native_kl'] for r in rs)/len(rs))
    (OUTPUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
