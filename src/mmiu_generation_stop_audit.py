"""Check whether option-prefix stopping changes actual completed answers.

Same immutable inputs, checkpoint, greedy decoding and 128-token ceiling as the
recorded MMIU run. Only stop on native EOS, not on an intermediate A/B/... token.
This writes separate diagnostic outputs, never overwrites benchmark results.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/diagnostics/mmiu_random1000_seed42_all_methods_20260914'
OUT = ROOT / 'artifacts/diagnostics/mmiu_generation_stop_audit_20260914'


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.mmiu_binding_protocol_audit import MODEL, CHECKPOINT
    from src.audit_mmiu_random_results import extract_answer

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa'
    ).eval().requires_grad_(False)
    from src.qwen_deepstack import disable_qwen_deepstack
    disable_qwen_deepstack(model)
    processor = AutoProcessor.from_pretrained(MODEL)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode == 'embedding_adapter' and adapter.adapter_start_layer == 0
    adapter.eval().requires_grad_(False)
    def reject(*args, **kwargs):
        raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process = reject
    ds = QwenBenchmarkDataset(str(SOURCE / 'mmiu_random1000.jsonl'), processor, 'mmiu',
        data_root=str(ROOT / 'data/benchmarks/mmiu'), max_samples=1000,
        cache_dir=SOURCE / 'processed/mmiu', prompt_layout='media_first_v1')
    old = {r['index']: r for p in SOURCE.glob('embedding_adapter_shard*.jsonl')
           for line in p.open() if (r := json.loads(line))}
    assert len(old) == 1000
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    original_memories = adapter.all_visual_memories_batched
    with torch.inference_mode(), (OUT / f'rows_{shard}.jsonl').open('w', buffering=1) as out:
        for index in range(shard, 1000, 8):
            item = ds[index]
            inputs = {k: v.cuda() for k, v in item.items()
                      if torch.is_tensor(v) and k in ('input_ids', 'attention_mask', 'mm_token_type_ids',
                                                      'pixel_values', 'image_grid_thw')}
            for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
                inputs[key] = inputs[key].unsqueeze(0)
            hidden, pos = ref.build_qwen_initial_context(model, inputs)
            visual = inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
            memory = original_memories(hidden[:, visual])
            adapter.all_visual_memories_batched = types.MethodType(lambda self, *a, _m=memory, **kw: _m, adapter)
            tokens = []
            try:
                for step in range(128):
                    model.model.rope_deltas = None
                    logits = ref.qwen_embedding_adapter_logits(model, adapter, inputs,
                        initial_hidden=hidden, position_ids=pos, logits_to_keep=1)[0]
                    token = int(logits[0, -1].argmax())
                    tokens.append(token)
                    if step == 0:
                        assert processor.tokenizer.decode([token], skip_special_tokens=True).strip() == old[index]['text'].strip(), index
                    if token in eos:
                        break
                    new = torch.tensor([[token]], device='cuda', dtype=inputs['input_ids'].dtype)
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], new], 1)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                    inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(new)], 1)
                    hidden = torch.cat([hidden, model.model.get_input_embeddings()(new)], 1)
                    pos = torch.cat([pos, pos[:, :, -1:] + 1], 2)
            finally:
                adapter.all_visual_memories_batched = original_memories
            text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            prediction = extract_answer(text, item['choices'])
            out.write(json.dumps(dict(index=index, source_index=item['index'], task=item['row']['task'],
                previous_text=old[index]['text'], previous_prediction=old[index]['prediction'],
                previous_correct=old[index]['prediction'] == item['answer'],
                text=text, prediction=prediction, gold=item['answer'], correct=prediction == item['answer'],
                generated_tokens=len(tokens), ended_on_eos=tokens[-1] in eos,
                same_first_token=True, deepstack=False, image_count=len(item['row']['images']))) + '\n')
            if index % 40 == shard:
                print('DONE', index, 'tokens', len(tokens), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = []
    logs = []
    for shard in range(8):
        log = (OUT / f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable, '-m', 'src.mmiu_generation_stop_audit', str(shard)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs:
        log.close()
    assert not any(codes), codes
    rows = [json.loads(line) for p in OUT.glob('rows_*.jsonl') for line in p.open()]
    assert len(rows) == len({r['index'] for r in rows}) == 1000
    summary = dict(n=1000, first_tokens_reproduced=all(r['same_first_token'] for r in rows),
        original_accuracy=sum(r['previous_correct'] for r in rows) / 10,
        eos_only_accuracy=sum(r['correct'] for r in rows) / 10,
        prediction_changes=[r for r in rows if r['prediction'] != r['previous_prediction']],
        nontrivial_completions=[r for r in rows if r['text'].strip() != r['previous_text'].strip()],
        capped=[r['index'] for r in rows if not r['ended_on_eos']])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    run() if len(sys.argv) == 1 else worker(int(sys.argv[1]))
