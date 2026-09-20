"""Qwen/LLaVA causal effect rank suite; baselines first, then five scopes.

Reuses the tested post-W_O current-trajectory intervention, preserving the
original RealWorldQA script/results. All bases are shared, prompt-only and
transductive; no model or adapter training is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src import realworldqa_causal_effect_rank as core

ROOT = core.ROOT
MODELS = {
    'qwen': (core.MODEL, 36, 2560),
    'llava': ('/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf', 32, 4096),
}
DATASETS = {'realworldqa': ('data/benchmarks/realworldqa/test.jsonl', 765),
            'mmstar': ('data/benchmarks/mmstar/mmstar_val.jsonl', 1000)}


def configure(kind):
    depth = MODELS[kind][1]
    mid = (depth - 10)//2
    scopes = {'first5': list(range(5)), 'first10': list(range(10)),
              'middle10': list(range(mid, mid+10)), 'last10': list(range(depth-10, depth)),
              'all': list(range(depth))}
    # Process-local settings only; the historical source file is not edited.
    core.SCOPES = scopes
    core.LAYERS = list(range(depth))
    return scopes


def prepare(item, model, kind):
    if kind == 'qwen':
        return core.prepare(item, torch.device('cuda:0'))
    inputs = {k: item[k].unsqueeze(0).cuda() for k in ['input_ids', 'attention_mask', 'pixel_values']}
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    if 'image_sizes' in item:
        inputs['image_sizes'] = item['image_sizes'].unsqueeze(0).cuda()
    assert bool(inputs['attention_mask'].all())
    image_id = model.config.image_token_index
    # Modern processor expands <image> into its 576 visual positions.
    assert int((inputs['input_ids'] == image_id).sum()) == 576
    inputs['mm_token_type_ids'] = (inputs['input_ids'] == image_id).long()
    return inputs


def model_inputs(inputs, kind):
    return {k: v for k, v in inputs.items() if kind == 'qwen' or k != 'mm_token_type_ids'}


def generate(model, processor, inputs, hook, kind, mode, scope=None, rank=128):
    from src.eval_benchmarks import extract_option_from_text
    current = dict(inputs)
    generated = []
    eos = {processor.tokenizer.eos_token_id}
    extra = model.generation_config.eos_token_id
    eos.update(extra if isinstance(extra, list) else [extra])
    for _ in range(8):
        hook.reset(current, mode, scope, rank)
        if hasattr(model.model, 'rope_deltas'):
            model.model.rope_deltas = None
        output = model(**model_inputs(current, kind), use_cache=False, return_dict=True, logits_to_keep=1)
        if mode != 'native':
            assert hook.calls == {l: 1 for l in hook.selected}
        token = int(output.logits[0, -1].float().argmax())
        generated.append(token)
        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if token in eos or extract_option_from_text(text) in ['A', 'B', 'C', 'D']:
            return text
        new = torch.tensor([[token]], device=current['input_ids'].device)
        current = dict(current, input_ids=torch.cat((current['input_ids'], new), 1),
                       attention_mask=torch.cat((current['attention_mask'], torch.ones_like(new)), 1),
                       mm_token_type_ids=torch.cat((current['mm_token_type_ids'], torch.zeros_like(new)), 1))
    return text


def worker(args):
    from src.model import load_frozen_qwen3vl, load_frozen_llava
    from src.data import QwenBenchmarkDataset, LlavaBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    scopes = configure(args.model)
    torch.set_num_threads(4); torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    path, depth, width = MODELS[args.model]
    if args.model == 'qwen':
        processor, model = load_frozen_qwen3vl(path, torch.bfloat16, torch.device('cuda:0'), 'sdpa')
        cls = QwenBenchmarkDataset
    else:
        processor, model = load_frozen_llava(path, torch.bfloat16, 'cuda:0', 'sdpa')
        cls = LlavaBenchmarkDataset
    assert len(model.model.language_model.layers) == depth
    assert model.config.text_config.hidden_size == width
    data = ROOT/DATASETS[args.benchmark][0]
    dataset = cls(str(data), processor, args.benchmark, data_root=str(data.parent), max_samples=args.samples)
    assert len(dataset) == args.samples
    root = Path(args.output)
    plan = json.loads((root/'plan.json').read_text())
    assert hashlib.sha256(json.dumps(dataset.rows, sort_keys=True).encode()).hexdigest() == plan['selection_sha256']
    hook = core.Intervention(model)
    features = []
    original_features = model.model.get_image_features
    def cached(*a, **kw):
        if not features: features.append(original_features(*a, **kw))
        return features[0]
    model.model.get_image_features = cached
    previous = {}
    if args.phase == 'eval':
        payload = torch.load(root/'basis.pt', map_location='cpu', weights_only=False)
        hook.bases = {l: b.cuda().float() for l, b in payload['basis'].items()}
        previous = {r['sample']: r['results']['native'] for r in
                    map(json.loads, (root/f'baseline_shard{args.shard}.jsonl').open())}
    started = time.time()
    with torch.inference_mode(), (root/f'{args.phase}_shard{args.shard}.jsonl').open('w') as f:
        for i in range(args.shard, len(dataset), args.world):
            item = dataset[i]; inputs = prepare(item, model, args.model); features.clear()
            if args.phase == 'collect':
                hook.reset(inputs, 'collect')
                if hasattr(model.model, 'rope_deltas'): model.model.rope_deltas = None
                model.model(**model_inputs(inputs, args.model), use_cache=False, return_dict=True)
                assert hook.calls == {l: 1 for l in core.LAYERS}
                row = {'sample': i, 'index': item['index']}
            else:
                modes = [('native', None, width)]
                if args.phase == 'eval':
                    modes += [('full', None, width)]
                    modes += [('rank', scope, r) for scope in scopes for r in core.RANKS]
                results = {}
                for mode, scope, r in modes:
                    key = mode if scope is None else f'{scope}_r{r}'
                    begin = time.time()
                    answer = generate(model, processor, inputs, hook, args.model, mode, scope, r)
                    score = score_prediction(metric=get_benchmark_spec(args.benchmark).metric,
                        prediction_text=answer, answer=item['answer'], choices=item.get('choices'), question=item['row'].get('question'))
                    results[key] = {'text': answer, **score, 'seconds': time.time()-begin}
                if args.phase == 'eval':
                    assert results['native']['text'] == results['full']['text'], ('Full identity failed', i)
                    assert results['native']['text'] == previous[i]['text'], ('Baseline repeat mismatch', i)
                row = {'sample': i, 'index': item['index'], 'results': results}
            f.write(json.dumps(row)+'\n'); f.flush()
            if i//args.world % 10 == 0:
                print(args.model, args.benchmark, args.phase, 'shard', args.shard, 'sample', i,
                      'seconds', round(time.time()-started, 1), flush=True)
    if args.phase == 'collect':
        torch.save({'cov': {l: c.cpu() for l, c in hook.covs.items()}, 'counts': hook.counts}, root/f'cov{args.shard}.pt')
    (root/f'{args.phase}_shard{args.shard}.done.json').write_text(json.dumps({'seconds': time.time()-started}))


def merge(root, phase, samples, world):
    rows = []
    for shard in range(world):
        assert (root/f'{phase}_shard{shard}.done.json').exists()
        rows.extend(map(json.loads, (root/f'{phase}_shard{shard}.jsonl').open()))
    assert len(rows) == samples and {r['sample'] for r in rows} == set(range(samples))
    summaries = {}
    reference = np.array([r['results']['native']['score'] for r in rows])
    for name in rows[0]['results']:
        scores = np.array([r['results'][name]['score'] for r in rows])
        diff = reference-scores
        summaries[name] = {'samples': samples, 'correct': float(scores.sum()), 'accuracy_pct': float(scores.mean()*100),
            'drop_pp': float(diff.mean()*100), 'harmed': int((diff>0).sum()), 'helped': int((diff<0).sum()),
            'invalid': sum(bool(r['results'][name]['invalid']) for r in rows)}
    filename = 'baseline.json' if phase == 'baseline' else 'results.json'
    (root/filename).write_text(json.dumps(summaries, indent=2))
    return summaries


def report(root, plans):
    lines = ['# Causal visual-effect rank: two models × two benchmarks', '',
        'Accuracy (%). RealWorldQA all765; MMStar first1000 (smoke runs explicitly override sample counts).', '',
        '| Model | Benchmark | N | Native baseline | Full-effect |', '|---|---|---:|---:|---:|']
    for p in plans:
        folder = root/p['name']
        b = json.loads((folder/'baseline.json').read_text()) if (folder/'baseline.json').exists() else {}
        r = json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else {}
        native = f"{b['native']['accuracy_pct']:.2f}" if b else 'pending'
        full = f"{r['full']['accuracy_pct']:.2f}" if r else 'pending'
        lines.append(f"| {p['model']} | {p['benchmark']} | {p['samples']} | {native} | {full} |")
    for p in plans:
        lines += ['', f"## {p['model']} / {p['benchmark']}", '',
                  '| Scope | Layers (0-based) | r0 | r32 | r64 | r128 |', '|---|---|---:|---:|---:|---:|']
        file = root/p['name']/'results.json'
        r = json.loads(file.read_text()) if file.exists() else {}
        for scope, layers in p['scopes'].items():
            values = [f"{r[f'{scope}_r{k}']['accuracy_pct']:.2f}" if r else 'pending' for k in core.RANKS]
            lines.append(f"| {scope} | {layers[0]}–{layers[-1]} | "+' | '.join(values)+' |')
    lines += ['', '## Protocol', '',
        '- Delta = joint attention output minus text-to-visual-blocked attention output; AFTER native W_O, BEFORE residual. Both use the current trajectory, not cached native decoder states.',
        '- Output = blocked + rank-r projection of delta, on text queries only. Visual query outputs and visual residual paths remain native. Rank0 blocks visual reading and renormalizes attention.',
        '- Five independent scopes × ranks0/32/64/128, plus native and full-effect identity:22 conditions per model/benchmark.',
        '- Each model/dataset has its own uncentered shared per-layer basis. FP64 second moments/eigendecomposition; FP32 projection; no TF32. Full channel widths: Qwen2560, LLaVA4096.',
        '- Bases use the selected evaluation prompts (no gold answers). This is transductive compression analysis, not held-out generalization. No adapter or backbone training.',
        '- Native model-specific processor/templates/images. BF16/SDPA, greedy max8, option/EOS early stop, fresh no-cache forward on every generated token. Qwen native DeepStack retained.',
        '- All four baselines finish before compression evaluation begins. Eval repeats the baseline and requires exact text equality; full-effect reconstruction also requires exact answer-text equality per sample.',
        '- Same subset order for both models, stored data/selection/code hashes. Existing training processes are not stopped.',
        '- Intermediate numerical choices match the prior RealWorldQA experiment; earlier historical FA2/other precision baselines should not be substituted for this suite’s measured baselines.', '',
        '[Plan](plan.json) · [Status](status.json)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def launch(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    plans = []
    for kind in args.models.split(','):
        for benchmark in args.benchmarks.split(','):
            path, n = DATASETS[benchmark]; n = min(n, args.limit) if args.limit else n
            rows = [json.loads(l) for l in (ROOT/path).open()][:n]
            p = {'model': kind, 'model_path': MODELS[kind][0], 'benchmark': benchmark, 'samples': n,
                 'name': f'{kind}_{benchmark}', 'scopes': configure(kind), 'ranks': core.RANKS,
                 'data_sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                 'selection_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                 'hidden_width': MODELS[kind][2], 'world': args.world}
            plans.append(p); (root/p['name']).mkdir(exist_ok=True)
            (root/p['name']/'plan.json').write_text(json.dumps(p, indent=2))
    metadata = {'experiments': plans, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'intervention_source_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                'basis_protocol': 'shared uncentered prompt-only basis on evaluation subset; transductive oracle, no training'}
    (root/'plan.json').write_text(json.dumps(metadata, indent=2)); report(root, plans)
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status = {'state': 'running', 'completed': []}
    def save():
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (root/'status.json').write_text(json.dumps(status, indent=2))
    tasks = [(p, 'baseline') for p in plans]
    tasks += [(p, phase) for p in plans for phase in ['collect', 'basis', 'eval']]
    try:
        for p, phase in tasks:
            folder = root/p['name']; key = f"{p['name']}/{phase}"
            status.update(experiment=p['name'], phase=phase); save()
            cmd = [sys.executable, '-u', '-m', 'src.causal_effect_benchmark_suite', 'worker', '--output', str(folder),
                '--model', p['model'], '--benchmark', p['benchmark'], '--samples', str(p['samples']), '--world', str(args.world), '--phase', phase]
            print('START', key, flush=True)
            if phase == 'basis':
                cmd[4] = 'basis'
                with (folder/'basis.log').open('w') as log:
                    subprocess.run(cmd, env=dict(env, CUDA_VISIBLE_DEVICES='0'), stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                jobs = []
                try:
                    for shard in range(args.world):
                        log = (folder/f'{phase}_gpu{shard}.log').open('w')
                        proc = subprocess.Popen(cmd+['--shard', str(shard)], env=dict(env, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
                        jobs.append((proc, log))
                    while any(proc.poll() is None for proc, _ in jobs):
                        if any(proc.poll() not in (None, 0) for proc, _ in jobs): raise RuntimeError(f'{key}: worker failed')
                        time.sleep(5)
                    assert all(proc.returncode == 0 for proc, _ in jobs)
                finally:
                    for proc, log in jobs:
                        if proc.poll() is None: proc.terminate()
                        log.close()
                if phase in ['baseline', 'eval']:
                    result = merge(folder, phase, p['samples'], args.world)
                    print('RESULT', key, json.dumps(result), flush=True)
                    report(root, plans)
            status['completed'].append(key); save()
        status.update(state='complete', phase='complete'); save()
    except BaseException as e:
        status.update(state='failed', error=repr(e)); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['launch', 'worker', 'basis'])
    parser.add_argument('--output', default=str(ROOT/'artifacts/diagnostics/causal_effect_2models_2bench_20260912'))
    parser.add_argument('--models', default='qwen,llava')
    parser.add_argument('--benchmarks', default='realworldqa,mmstar')
    parser.add_argument('--model', choices=list(MODELS), default='qwen')
    parser.add_argument('--benchmark', choices=list(DATASETS), default='realworldqa')
    parser.add_argument('--phase', choices=['baseline', 'collect', 'basis', 'eval'], default='baseline')
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--samples', type=int, default=765)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'basis':
        configure(args.model); core.basis(args)
    else:
        {'launch': launch, 'worker': worker}[args.command](args)
