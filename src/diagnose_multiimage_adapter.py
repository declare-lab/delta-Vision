"""Read-only checkpoint audit: cached evaluation versus training forward path.

The subset contains ALL Image-Text Matching (MuirBench) and forensic BLINK
(MMIU) examples from the historical first-1000 evaluation. It is diagnostic,
not a replacement benchmark score. No weights or production code are changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '/lustre-data/leijingdi/code/vision-kv-inject-attention-sink/src'
CHECKPOINT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
HISTORY = CHECKPOINT.parents[1] / 'eval'
OUTPUT = ROOT / 'artifacts/diagnostics/multiimage_adapter_audit_20260913'


def worker(shard):
    import src
    src.__path__.insert(0, REFERENCE)
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.eval_benchmarks import generate_adapter_qwen_decode_cache, generate_adapter_qwen_recompute
    from src.benchmarks import score_prediction
    torch.set_num_threads(4)
    processor, model = ref.load_frozen_qwen3vl(
        '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
        torch.bfloat16, torch.device('cuda'), 'flash_attention_2')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    records = []
    datasets = {}
    for benchmark, task in [('muirbench', 'Image-Text Matching'), ('mmiu', 'forensic_detection_blink')]:
        datasets[benchmark] = QwenBenchmarkDataset(
            str(ROOT / f'data/benchmarks/{benchmark}/test.jsonl'), processor, benchmark, max_samples=1000)
        old = json.loads((HISTORY / benchmark / 'predictions.json').read_text())
        records.extend((benchmark, row) for row in old if row['row']['task'] == task)
    with torch.inference_mode(), (OUTPUT / f'rows_{shard}.jsonl').open('w', buffering=1) as out:
        for number in range(shard, len(records), 8):
            benchmark, old = records[number]
            item = datasets[benchmark][int(old['index'])]
            inputs = {k: (v.unsqueeze(0) if k in ('input_ids', 'attention_mask', 'mm_token_type_ids') else v).cuda()
                      for k, v in item.items() if torch.is_tensor(v)}
            assert inputs['attention_mask'].bool().all()
            choices = item['choices']
            cache = generate_adapter_qwen_decode_cache(
                model, processor, adapter, inputs, 16, early_stop_metric=benchmark, choices=choices)[1][0]
            recompute = generate_adapter_qwen_recompute(
                model, processor, adapter, inputs, 16, early_stop_metric=benchmark, choices=choices)[1][0]
            def score(text):
                return score_prediction(metric=benchmark, prediction_text=text, answer=item['answer'], choices=choices)
            result = dict(benchmark=benchmark, index=old['index'], gold=item['answer'],
                          history=old['adapter_eval'], native_history=old['teacher_eval'],
                          cache=score(cache), recompute=score(recompute), cache_text=cache, recompute_text=recompute)
            out.write(json.dumps(result) + '\n')
            print(benchmark, old['index'], cache, recompute, flush=True)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    processes = []
    files = []
    try:
        for shard in range(8):
            log = (OUTPUT / f'worker{shard}.log').open('w')
            files.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, '-m', 'src.diagnose_multiimage_adapter', '--shard', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        if any(codes):
            raise RuntimeError(f'Audit failed: {codes}; see {OUTPUT}')
    finally:
        for log in files:
            log.close()
    rows = [json.loads(line) for shard in range(8) for line in (OUTPUT / f'rows_{shard}.jsonl').read_text().splitlines()]
    summary = {}
    for benchmark, count in [('muirbench', 84), ('mmiu', 132)]:
        selected = [r for r in rows if r['benchmark'] == benchmark]
        assert len(selected) == count and len({r['index'] for r in selected}) == count
        summary[benchmark] = dict(samples=count,
            accuracy={method: sum(r[method]['score'] for r in selected) / count * 100
                      for method in ['native_history', 'history', 'cache', 'recompute']},
            cache_vs_history_changed=sum(r['cache']['prediction'] != r['history']['prediction'] for r in selected),
            cache_vs_recompute_changed=sum(r['cache']['prediction'] != r['recompute']['prediction'] for r in selected))
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def prompt_worker(shard):
    """Move question after images; keep choices, image order and model fixed."""
    import src
    src.__path__.insert(0, REFERENCE)
    import torch
    import types
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.eval_benchmarks import generate_adapter_qwen_decode_cache
    from src.benchmarks import score_prediction
    torch.set_num_threads(4)
    processor, model = ref.load_frozen_qwen3vl(
        '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
        torch.bfloat16, torch.device('cuda'), 'flash_attention_2')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    dataset = QwenBenchmarkDataset(str(ROOT / 'data/benchmarks/muirbench/test.jsonl'),
                                   processor, 'muirbench', max_samples=1000)
    original = dataset._qwen_message_content
    def moved(self, row, question, images, videos):
        content = original(row, question, images, videos)
        q = row['question']
        assert content[0]['type'] == 'text' and content[0]['text'].startswith(q)
        content[0] = dict(content[0], text=content[0]['text'][len(q):])
        instruction = self.spec.answer_instruction
        assert content[-1]['type'] == 'text' and content[-1]['text'].endswith(instruction)
        content[-1] = dict(content[-1], text=content[-1]['text'][:-len(instruction)] + q + '\n' + instruction)
        return content
    old = json.loads((HISTORY / 'muirbench/predictions.json').read_text())
    records = [r for r in old if r['row']['task'] == 'Image-Text Matching']
    with torch.inference_mode(), (OUTPUT / f'prompt_rows_{shard}.jsonl').open('w', buffering=1) as out:
        for j in range(shard, len(records), 8):
            row = records[j]
            result = dict(index=row['index'], gold=row['row']['answer'])
            pixels = None
            for condition in ['original', 'question_after_images']:
                dataset._qwen_message_content = original if condition == 'original' else types.MethodType(moved, dataset)
                item = dataset[int(row['index'])]
                inputs = {k: (v.unsqueeze(0) if k in ('input_ids', 'attention_mask', 'mm_token_type_ids') else v).cuda()
                          for k, v in item.items() if torch.is_tensor(v)}
                if pixels is None:
                    pixels = inputs['pixel_values'].clone()
                else:
                    assert torch.equal(pixels, inputs['pixel_values']), 'Image order/content changed'
                model.model.rope_deltas = None
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                native = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                pred = generate_adapter_qwen_decode_cache(model, processor, adapter, inputs, 16,
                    early_stop_metric='muirbench', choices=item['choices'])[1][0]
                def score(text):
                    return score_prediction(metric='muirbench', prediction_text=text,
                                            answer=item['answer'], choices=item['choices'])
                result[condition] = dict(native=score(native), adapter=score(pred), native_text=native, adapter_text=pred)
            out.write(json.dumps(result) + '\n')
            print(row['index'], result['original']['adapter_text'], result['question_after_images']['adapter_text'], flush=True)


def prompt_run():
    processes, files = [], []
    try:
        for shard in range(8):
            log = (OUTPUT / f'prompt_worker{shard}.log').open('w')
            files.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, '-m', 'src.diagnose_multiimage_adapter', '--prompt-check', '--shard', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        assert not any(codes), codes
    finally:
        for f in files:
            f.close()
    rows = [json.loads(line) for shard in range(8) for line in (OUTPUT / f'prompt_rows_{shard}.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['index'] for r in rows}) == 84
    result = {condition: {model: 100 * sum(r[condition][model]['score'] for r in rows) / len(rows)
                         for model in ['native', 'adapter']}
              for condition in ['original', 'question_after_images']}
    (OUTPUT / 'prompt_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int)
    parser.add_argument('--prompt-check', action='store_true')
    args = parser.parse_args()
    if args.prompt_check:
        prompt_run() if args.shard is None else prompt_worker(args.shard)
    else:
        run() if args.shard is None else worker(args.shard)
