"""Frozen native-model causal visual-effect rank intervention, not hidden rank.

Each layer recomputes joint and blocked attention from its CURRENT input.
Only text-query outputs change; visual residuals remain native. Shared channel
bases are calibrated on benchmark prompts without answers (transductive oracle).
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

from analysis.common.visual_cross_token_ablation import ROOT, MODEL, prepare

SCOPES = {'first5': list(range(5)), 'first10': list(range(10)),
          'middle10': list(range(13, 23)), 'last10': list(range(26, 36))}
LAYERS = sorted(set(sum(SCOPES.values(), [])))
RANKS = [0, 32, 64, 128]
DATA = ROOT/'data/benchmarks/realworldqa/test.jsonl'


class Intervention:
    def __init__(self, model):
        self.model = model
        self.mode = 'native'
        self.selected = set(LAYERS)
        self.rank = 128
        self.bases = {}
        self.covs = {}
        self.counts = {}
        self.handles = []
        self.calls = {}
        self.reconstruction_error = 0.
        for layer in LAYERS:
            attn = model.model.language_model.layers[layer].self_attn
            self.handles.append(attn.register_forward_hook(self.hook(layer), with_kwargs=True))

    def reset(self, inputs, mode, scope=None, rank=128):
        assert inputs['input_ids'].shape[0] == 1
        assert bool(inputs['attention_mask'].all())
        self.types = inputs['mm_token_type_ids']
        self.mode, self.rank = mode, rank
        self.selected = set(SCOPES[scope]) if scope else set(LAYERS)
        self.calls = {}
        self.reconstruction_error = 0.

    def hook(self, layer):
        def run(module, args, kw, output):
            if self.mode == 'native' or layer not in self.selected:
                return
            assert kw.get('past_key_values') is None, 'Only fresh no-cache rollouts supported'
            h = kw.get('hidden_states', args[0] if args else None)
            n = h.shape[1]
            assert self.types.shape[1] == n
            text = self.types[0] == 0
            visual = self.types[0] == 1
            assert bool(visual.any())
            mask = kw.get('attention_mask')
            if mask is None:
                blocked_mask = torch.zeros((1, 1, n, n), device=h.device, dtype=h.dtype)
                forbidden = torch.ones((n, n), device=h.device, dtype=torch.bool).triu(1)
                blocked_mask.masked_fill_(forbidden, torch.finfo(h.dtype).min)
            else:
                assert mask.ndim == 4
                blocked_mask = mask.clone()
            edges = text[None, None, :, None] & visual[None, None, None, :]
            blocked_mask.masked_fill_(edges, False if blocked_mask.dtype == torch.bool else torch.finfo(blocked_mask.dtype).min)
            blocked_kw = dict(kw, attention_mask=blocked_mask)
            # Call forward directly, deliberately bypassing this module's hooks.
            blocked = module.forward(*args, **blocked_kw)[0]
            joint = output[0]
            delta = joint.float() - blocked.float()
            self.calls[layer] = self.calls.get(layer, 0) + 1
            if self.mode == 'collect':
                x = delta[0, text].double()
                if layer not in self.covs:
                    self.covs[layer] = torch.zeros((x.shape[-1], x.shape[-1]), dtype=torch.float64, device=x.device)
                    self.counts[layer] = 0
                self.covs[layer].addmm_(x.T, x)
                self.counts[layer] += len(x)
                return  # Collection does NOT change the native trajectory.
            if self.mode == 'full':
                reconstructed = blocked.float() + delta
                self.reconstruction_error = max(self.reconstruction_error, float((reconstructed - joint.float()).abs().max()))
                replacement = reconstructed.to(joint.dtype)
            elif self.rank == 0:
                replacement = blocked
            else:
                b = self.bases[layer][:, :self.rank]
                assert b.shape == (joint.shape[-1], self.rank)
                replacement = (blocked.float() + (delta @ b) @ b.T).to(joint.dtype)
            patched = torch.where(text[None, :, None], replacement, joint)
            assert torch.equal(patched[:, visual], joint[:, visual])
            return (patched,) + output[1:]
        return run


def load_worker():
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor, model = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device('cuda:0'), 'sdpa')
    dataset = QwenBenchmarkDataset(str(DATA), processor, 'realworldqa', data_root=str(DATA.parent), max_samples=765)
    return processor, model, dataset


def generate(model, processor, inputs, hook, mode, scope=None, rank=128):
    from src.evaluate import extract_option_from_text
    current = dict(inputs)
    generated = []
    eos = {processor.tokenizer.eos_token_id}
    extra = model.generation_config.eos_token_id
    eos.update(extra if isinstance(extra, list) else [extra])
    for _ in range(8):
        hook.reset(current, mode, scope, rank)
        model.model.rope_deltas = None
        out = model(**current, use_cache=False, return_dict=True, logits_to_keep=1)
        if mode != 'native':
            assert hook.calls == {l: 1 for l in hook.selected}
        token = int(out.logits[0, -1].float().argmax())
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
    from src.benchmarks import get_benchmark_spec, score_prediction
    processor, model, dataset = load_worker()
    hook = Intervention(model)
    root = Path(args.output)
    original_features = model.model.get_image_features
    features = []
    def cached(*a, **kw):
        if not features:
            features.append(original_features(*a, **kw))
        return features[0]
    model.model.get_image_features = cached
    if args.phase == 'eval':
        payload = torch.load(root/'basis.pt', map_location='cpu', weights_only=False)
        hook.bases = {l: b.cuda().float() for l, b in payload['basis'].items()}
    limit = min(args.samples, len(dataset))
    start = time.time()
    path = root/f'{args.phase}_shard{args.shard}.jsonl'
    with torch.inference_mode(), path.open('w') as f:
        for i in range(args.shard, limit, args.world):
            item = dataset[i]
            inputs = prepare(item, torch.device('cuda:0'))
            features.clear()
            if args.phase == 'collect':
                hook.reset(inputs, 'collect')
                model.model.rope_deltas = None
                model.model(**inputs, use_cache=False, return_dict=True)
                assert hook.calls == {l: 1 for l in LAYERS}
                record = {'sample': i, 'index': item['index']}
            else:
                modes = [('native', None, 2560), ('full', None, 2560)]
                modes += [('rank', scope, rank) for scope in SCOPES for rank in RANKS]
                results = {}
                for mode, scope, rank in modes:
                    name = mode if scope is None else f'{scope}_r{rank}'
                    t = time.time()
                    text = generate(model, processor, inputs, hook, mode, scope, rank)
                    score = score_prediction(metric=get_benchmark_spec('realworldqa').metric,
                        prediction_text=text, answer=item['answer'], choices=item.get('choices'), question=item['row'].get('question'))
                    results[name] = {'text': text, **score, 'seconds': time.time()-t}
                assert results['native']['text'] == results['full']['text'], 'Full effect failed identity control'
                record = {'sample': i, 'index': item['index'], 'results': results}
            f.write(json.dumps(record)+'\n'); f.flush()
            if i//args.world % 10 == 0:
                print(args.phase, 'shard', args.shard, 'sample', i, 'seconds', round(time.time()-start, 1), flush=True)
    if args.phase == 'collect':
        torch.save({'cov': {l: x.cpu() for l, x in hook.covs.items()}, 'counts': hook.counts}, root/f'cov{args.shard}.pt')
    (root/f'{args.phase}_shard{args.shard}.done.json').write_text(json.dumps({'seconds': time.time()-start}))


def basis(args):
    root = Path(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    sums = {}
    counts = {}
    for shard in range(args.world):
        data = torch.load(root/f'cov{shard}.pt', map_location='cpu', weights_only=False)
        for l, x in data['cov'].items():
            if l not in sums:
                sums[l] = x
                counts[l] = data['counts'][l]
            else:
                sums[l] += x
                counts[l] += data['counts'][l]
        del data
    bases, energy = {}, {}
    for l in LAYERS:
        cov = sums.pop(l).cuda()
        values, vectors = torch.linalg.eigh((cov+cov.T)*.5)
        values = values.flip(0).clamp_min(0)
        b = vectors[:, -128:].flip(1).float()
        assert float((b.T@b-torch.eye(128, device=b.device)).abs().max()) < 1e-4
        bases[l] = b.cpu()
        energy[l] = {str(r): float(values[:r].sum()/values.sum().clamp_min(1e-30)) for r in RANKS}
        print('basis layer', l, energy[l], flush=True)
    torch.save({'basis': bases, 'energy': energy, 'counts': counts}, root/'basis.pt')
    (root/'basis_energy.json').write_text(json.dumps(energy, indent=2))


def merge(args):
    root = Path(args.output)
    rows = []
    for shard in range(args.world):
        rows += [json.loads(l) for l in (root/f'eval_shard{shard}.jsonl').open()]
    assert len(rows) == args.samples and {r['sample'] for r in rows} == set(range(args.samples))
    summaries = {}
    for name in rows[0]['results']:
        ref = np.array([r['results']['native']['score'] for r in rows])
        score = np.array([r['results'][name]['score'] for r in rows])
        summaries[name] = {'samples': len(rows), 'accuracy_pct': float(score.mean()*100),
            'drop_pp': float((ref-score).mean()*100), 'harmed': int((ref>score).sum()), 'helped': int((ref<score).sum())}
    (root/'results.json').write_text(json.dumps(summaries, indent=2))
    lines = ['# RealWorldQA causal visual-effect rank', '', f'Qwen3-VL-4B; {args.samples} samples; no training; zero-based layers.', '',
        '| Scope | Layers | r0 | r32 | r64 | r128 |', '|---|---|---:|---:|---:|---:|']
    for scope, layers in SCOPES.items():
        lines.append(f'| {scope} | {layers[0]}–{layers[-1]} | '+' | '.join(f'{summaries[f"{scope}_r{r}"]["accuracy_pct"]:.2f}%' for r in RANKS)+' |')
    lines += ['', f'Native: {summaries["native"]["accuracy_pct"]:.2f}%; full-effect identity: {summaries["full"]["accuracy_pct"]:.2f}%.', '',
        'Delta = native attention output minus text-to-visual-blocked output, AFTER W_O and BEFORE residual. Both are computed from the current intervened trajectory.',
        'Text outputs = blocked + delta projected onto shared per-layer channel basis. Visual query outputs are untouched. Rank0 is blocked attention, with softmax renormalization.',
        'Bases: uncentered FP64 second moments on all evaluation prompts, no gold answer tokens. This is a transductive diagnostic, NOT held-out basis generalization and NOT adapter src.training.',
        'Full native prefix, DeepStack disabled, SDPA/BF16, fresh no-cache greedy forward each step, at most8 generated tokens. Only vision-encoder features are reused within an image.',
        'Earlier HotpotQA basis collection included answer trajectories; this run is prompt-only, so it is not an exact reproduction of that calibration protocol.',
        '[Configuration](plan.json) · [Detailed results](results.json) · [Basis energy](basis_energy.json)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def launch(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    plan = {'model': MODEL, 'data': str(DATA), 'data_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'samples': args.samples, 'world': args.world, 'scopes': SCOPES, 'ranks': RANKS,
        'basis': 'shared per layer; uncentered prompt-only second moments on this evaluation subset; transductive oracle',
        'native_deepstack': False, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'plan.json').write_text(json.dumps(plan, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status = {'state': 'running'}
    def save():
        (root/'status.json').write_text(json.dumps(status, indent=2))
    try:
        for phase in ['collect', 'basis', 'eval', 'merge']:
            status['phase'] = phase; save()
            if phase in ['basis', 'merge']:
                cmd = [sys.executable, '-u', '-m', 'analysis.table06_layer_effect.realworldqa_causal_effect_rank', phase, '--output', str(root), '--world', str(args.world), '--samples', str(args.samples)]
                with (root/f'{phase}.log').open('w') as log:
                    subprocess.run(cmd, env=dict(env, CUDA_VISIBLE_DEVICES='0'), stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                jobs = []
                try:
                    for shard in range(args.world):
                        log = (root/f'{phase}_gpu{shard}.log').open('w')
                        cmd = [sys.executable, '-u', '-m', 'analysis.table06_layer_effect.realworldqa_causal_effect_rank', 'worker', '--phase', phase,
                            '--output', str(root), '--samples', str(args.samples), '--world', str(args.world), '--shard', str(shard)]
                        proc = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
                        jobs.append((proc, log))
                    while any(p.poll() is None for p, _ in jobs):
                        if any(p.poll() not in (None, 0) for p, _ in jobs):
                            raise RuntimeError(f'{phase} worker failed; see per-GPU logs')
                        time.sleep(5)
                    assert all(p.returncode == 0 for p, _ in jobs)
                finally:
                    for p, log in jobs:
                        if p.poll() is None: p.terminate()
                        log.close()
            print('DONE', phase, flush=True)
        status['state'] = 'complete'; save()
    except BaseException as e:
        status.update(state='failed', error=repr(e)); save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['worker', 'basis', 'merge', 'launch'])
    p.add_argument('--output', default=str(ROOT/'artifacts/diagnostics/realworldqa_causal_effect_rank_765_20260912'))
    p.add_argument('--phase', choices=['collect', 'eval'], default='collect')
    p.add_argument('--samples', type=int, default=765)
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    args = p.parse_args()
    {'worker': worker, 'basis': basis, 'merge': merge, 'launch': launch}[args.command](args)
