"""Read-only native visual attention spectra; no oracle, training, or answer input.

Each sample is measured independently. Q is post-QNorm/post-RoPE. Scores
are raw scaled Q_visual K_visual.T, WITHOUT a causal mask; masked -inf is
not a finite matrix. Native attention remains causal and unchanged. Output
is native attention after O projection and before residual, at visual rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
from src.model import load_frozen_llava, load_frozen_qwen3vl, qwen_apply_rotary_pos_emb
from src.data import LlavaBenchmarkDataset, QwenBenchmarkDataset
from src.benchmarks import get_benchmark_spec

SHARED = Path('/lustre-data/leijingdi/code/vision-kv-inject')
MODELS = {
    'qwen': '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
    'llava': '/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf',
}
COUNTS = {'mmstar': 1000, 'realworldqa': 765, 'sqa': 1000}


def spectral_metrics(s):
    """s can be batched, descending; entropy uses s, NOT s**2."""
    s = s.double().clamp_min(0)
    total = s.sum(-1, keepdim=True)
    p = s / total.clamp_min(1e-300)
    erank = (-(p * p.clamp_min(1e-300).log()).sum(-1)).exp()
    energy = s.square()
    fractions = energy.cumsum(-1) / energy.sum(-1, keepdim=True).clamp_min(1e-300)
    r95 = (fractions < .95).sum(-1) + 1
    zero = total.squeeze(-1) == 0
    return torch.where(zero, 0., erank), torch.where(zero, 0, r95)


def feature_singular_values(x):
    """Exact smaller Gram eigenspectrum in FP64 (no random/token subsampling)."""
    x = x.double()
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    vals = torch.linalg.eigvalsh((gram + gram.T) * .5).flip(-1).clamp_min(0)
    return vals.sqrt()


def score_and_query_spectra(q, k, scale):
    """Nonzero singular values of QK.T via thin QR, without materializing Nv².

    Q=Uq Rq, K=Uk Rk; Uq/Uk are column orthonormal, hence Rq Rk.T
    has the same nonzero spectrum as QK.T. All heads included, FP64.
    """
    rq = torch.linalg.qr(q.double(), mode='r').R
    rk = torch.linalg.qr(k.double(), mode='r').R
    core = rq @ rk.transpose(-1, -2) * scale
    def gram_values(x):
        gram = x @ x.transpose(-1, -2)
        return torch.linalg.eigvalsh((gram + gram.transpose(-1, -2))*.5).flip(-1).clamp_min(0).sqrt()
    return gram_values(core), gram_values(rq)


def packed(s):
    e, r = spectral_metrics(s)
    return {'effective_rank': float(e), 'rank95': int(r)}


class NativeObserver:
    def __init__(self, model, kind):
        self.kind = kind
        self.layers = model.model.language_model.layers
        self.positions = None
        self.records = {}
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self.hook(i), with_kwargs=True))

    def hook(self, index):
        def observe(module, args, kwargs, output):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if hidden is None or hidden.shape[0] != 1:
                raise ValueError('Expected native batch-one attention hidden')
            pos = self.positions
            if pos is None or len(pos) == 0 or int(pos.max()) >= hidden.shape[1]:
                raise ValueError('Missing or misaligned visual positions')
            if index in self.records:
                raise ValueError('Expected one prompt-only forward per sample')
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_proj(hidden).view(shape)
            k = module.k_proj(hidden).view(shape)
            if self.kind == 'qwen':
                q, k = module.q_norm(q), module.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            cos, sin = kwargs['position_embeddings']
            rope = qwen_apply_rotary_pos_emb if self.kind == 'qwen' else llama_rope
            q, k = rope(q, k, cos, sin)
            q = q[0].index_select(1, pos)
            k = k[0].index_select(1, pos)
            groups = q.shape[0] // k.shape[0]
            k = k.repeat_interleave(groups, dim=0)
            scores_s, q_s = score_and_query_spectra(q, k, float(module.scaling))
            score_e, score_r = spectral_metrics(scores_s)
            q_e, q_r = spectral_metrics(q_s)
            score_e, score_r, q_e, q_r = [x.cpu().tolist() for x in (score_e, score_r, q_e, q_r)]
            attn = output[0][0].index_select(0, pos)
            record = {
                'layer': index, 'visual_tokens': len(pos), 'heads': q.shape[0],
                'head_dim': q.shape[-1], 'q_width': q.shape[0]*q.shape[-1],
                'output_width': attn.shape[-1],
                'q_concat': packed(feature_singular_values(q.transpose(0, 1).reshape(len(pos), -1))),
                'attention_output': packed(feature_singular_values(attn)),
                'q_per_head': [{'head': h, 'effective_rank': float(q_e[h]), 'rank95': int(q_r[h])}
                               for h in range(len(q_e))],
                'qkt_per_head': [{'head': h, 'effective_rank': float(score_e[h]), 'rank95': int(score_r[h])}
                                 for h in range(len(score_e))],
            }
            self.records[index] = record
            # Forward hooks returning None never replace the original output.
        return observe

    def close(self):
        for h in self.handles:
            h.remove()


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.kind}_shard{args.shard}.jsonl'
    if path.exists() and not args.resume:
        raise FileExistsError(path)
    done = set()
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            done.add((row['benchmark'], row['sample_position']))
    load = load_frozen_qwen3vl if args.kind == 'qwen' else load_frozen_llava
    processor, model = load(MODELS[args.kind], torch.bfloat16, device, 'flash_attention_2')
    observer = NativeObserver(model, args.kind)
    started = time.time()
    checked = False
    processed = 0
    with torch.inference_mode(), path.open('a') as out:
        for bench in args.benchmarks.split(','):
            n = min(COUNTS[bench], args.limit) if args.limit else COUNTS[bench]
            data = SHARED / get_benchmark_spec(bench).default_data
            cls = QwenBenchmarkDataset if args.kind == 'qwen' else LlavaBenchmarkDataset
            ds = cls(str(data), processor, bench, data_root=str(data.parent), max_samples=n)
            selection_hash = hashlib.sha256(json.dumps(ds.rows, sort_keys=True).encode()).hexdigest()
            for j in range(args.shard, len(ds), args.shards):
                if (bench, j) in done:
                    continue
                item = ds[j]
                inputs = {}
                for key in ['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw', 'image_sizes']:
                    if key not in item:
                        continue
                    x = item[key]
                    if key in ('input_ids', 'attention_mask', 'mm_token_type_ids') or args.kind == 'llava':
                        x = x.unsqueeze(0)
                    inputs[key] = x.to(device=device, dtype=torch.bfloat16 if x.is_floating_point() else x.dtype)
                image_id = model.config.image_token_id if args.kind == 'qwen' else model.config.image_token_index
                pos = (inputs['input_ids'][0] == image_id).nonzero().flatten()
                if not len(pos):
                    raise RuntimeError('Processor must expand image token positions before observation')
                observer.positions = pos
                observer.records = {}
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                if len(observer.records) != len(observer.layers):
                    raise RuntimeError('Not all native language layers were captured')
                if not checked:
                    # Remove hooks temporarily; assert observation leaves native logits identical.
                    observer.close()
                    plain = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                    if not torch.equal(plain.logits, result.logits):
                        raise RuntimeError(f'Observation changed logits: maxdiff={(plain.logits-result.logits).abs().max()}')
                    del plain
                    saved = observer.records
                    observer = NativeObserver(model, args.kind)
                    observer.records = saved
                    checked = True
                    print('PASS: observed and unobserved native logits bitwise identical', flush=True)
                record = {'model': args.kind, 'benchmark': bench, 'sample_position': j,
                          'sample_id': item['index'], 'selection_sha256': selection_hash,
                          'input_tokens': inputs['input_ids'].shape[1], 'visual_tokens': len(pos),
                          'layers': [observer.records[k] for k in sorted(observer.records)]}
                out.write(json.dumps(record) + '\n')
                out.flush()
                processed += 1
                print(f'{args.kind} shard={args.shard} {bench} sample={j} nv={len(pos)} done={processed} elapsed={time.time()-started:.1f}s', flush=True)
                del result, inputs, item
    observer.close()
    (root / f'{args.kind}_shard{args.shard}.done.json').write_text(json.dumps({
        'seconds': time.time()-started, 'processed': processed, 'args': vars(args),
        'observation_bitwise_check': checked}, indent=2))


def merge(args):
    root = Path(args.output)
    groups = {}
    for path in sorted(root.glob('*_shard[0-9]*.jsonl')):
        for line in path.open():
            r = json.loads(line)
            key = (r['model'], r['benchmark'])
            group = groups.setdefault(key, {})
            if r['sample_position'] in group:
                raise ValueError(f'Duplicate sample {key} {r["sample_position"]}')
            group[r['sample_position']] = r
    layers_table, heads_table, overall = [], [], []
    for (model, bench), records in sorted(groups.items()):
        expected = min(COUNTS[bench], args.limit) if args.limit else COUNTS[bench]
        if set(records) != set(range(expected)):
            raise ValueError(f'Incomplete {model}/{bench}: {len(records)}/{expected}')
        rows = list(records.values())
        if len({r['selection_sha256'] for r in rows}) != 1:
            raise ValueError('Different data selections across shards')
        for li in range(len(rows[0]['layers'])):
            entries = [r['layers'][li] for r in rows]
            line = {'model': model, 'benchmark': bench, 'layer': li, 'samples': len(rows),
                    'visual_tokens_mean': sum(r['visual_tokens'] for r in rows)/len(rows)}
            for metric in ['rank95', 'effective_rank']:
                for name in ['q_concat', 'attention_output']:
                    line[f'{name}_{metric}'] = sum(e[name][metric] for e in entries)/len(entries)
                for name in ['q_per_head', 'qkt_per_head']:
                    line[f'{name}_{metric}'] = sum(sum(h[metric] for h in e[name])/len(e[name]) for e in entries)/len(entries)
            layers_table.append(line)
            for hi in range(entries[0]['heads']):
                hline = {'model': model, 'benchmark': bench, 'layer': li, 'head': hi, 'samples': len(rows)}
                for name in ['q_per_head', 'qkt_per_head']:
                    for metric in ['rank95', 'effective_rank']:
                        hline[f'{name}_{metric}'] = sum(e[name][hi][metric] for e in entries)/len(entries)
                heads_table.append(hline)
        subset = [r for r in layers_table if r['model'] == model and r['benchmark'] == bench]
        overall.append({'model': model, 'benchmark': bench, 'samples': len(rows), 'layers': len(subset),
                        **{k: sum(r[k] for r in subset)/len(subset) for k in subset[0] if k.endswith(('_rank95', '_effective_rank'))}})
    for name, rows in [('per_layer', layers_table), ('per_head', heads_table), ('overall', overall)]:
        with (root/f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (root/f'{name}.json').write_text(json.dumps(rows, indent=2))
    lines = ['# Native visual-token rank statistics', '',
             'Only prompt visual positions; no answers, token subsampling, intervention, or src.training. Native causal forward and DeepStack retained.', '',
             'r95 uses squared singular-value energy. Effective rank uses p_i = sigma_i / sum(sigma), exp(-sum p log p).', '',
             'Q: post-normalization/post-RoPE, concatenated heads (per-head Q also in CSV). QK: raw scaled visual-to-visual scores, per-head mean; no causal masking before spectral analysis. Output: native W_O output before residual.', '',
             'FP64 smaller-Gram eigenspectra for feature matrices; exact QR factorization for per-head QK. Each sample measured separately, arithmetic sample averages. Layer indices are zero-based.', '',
             '| Model | Dataset | N | Q r95 | Q effective | QK r95 | QK effective | Output r95 | Output effective |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def table_line(r, title):
        return '| '+title+' | '+' | '.join(f'{r[k]:.2f}' for k in ['q_concat_rank95','q_concat_effective_rank','qkt_per_head_rank95','qkt_per_head_effective_rank','attention_output_rank95','attention_output_effective_rank'])+' |'
    for r in overall:
        lines.append(table_line(r, f'{r["model"]} | {r["benchmark"]} | {r["samples"]}'))
    for model, bench in sorted(groups):
        lines += ['', f'## {model} / {bench}', '', '| Layer | Q r95 | Q effective | QK r95 | QK effective | Output r95 | Output effective |', '|---:|---:|---:|---:|---:|---:|---:|']
        for r in layers_table:
            if r['model'] == model and r['benchmark'] == bench:
                lines.append(table_line(r, str(r['layer'])))
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(overall, indent=2), flush=True)


def launch(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for gpu in range(8):
            kind = 'qwen' if gpu < 4 else 'llava'
            shard = gpu % 4
            cmd = [sys.executable, '-m', 'analysis.table05_native_rank.visual_rank_statistics', 'run', '--kind', kind,
                   '--shard', str(shard), '--shards', '4', '--output', str(root), '--benchmarks', args.benchmarks]
            if args.limit: cmd += ['--limit', str(args.limit)]
            if args.resume: cmd += ['--resume']
            log = (root/f'{kind}_shard{shard}.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            children.append((proc, log))
        while any(p.poll() is None for p, _ in children):
            if any(p.poll() not in (None, 0) for p, _ in children):
                raise RuntimeError('Worker failed; inspect worker logs. Completed sample records retained.')
            time.sleep(10)
        if any(p.returncode != 0 for p, _ in children): raise RuntimeError('Worker failed')
        merge(args)
    finally:
        for p, log in children:
            if p.poll() is None: p.terminate()
            log.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['run', 'launch', 'merge'])
    p.add_argument('--kind', choices=list(MODELS), default='qwen')
    p.add_argument('--output', required=True)
    p.add_argument('--benchmarks', default='mmstar,realworldqa,sqa')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    {'run': run, 'merge': merge, 'launch': launch}[args.command](args)


if __name__ == '__main__':
    main()
