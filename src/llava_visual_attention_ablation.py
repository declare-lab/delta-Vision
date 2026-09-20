"""Paired native / visual self-only / visual FFN-only LLaVA evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from src.benchmarks import get_benchmark_spec, score_prediction
from src.data import LlavaBenchmarkDataset
from src.eval_benchmarks import generate_teacher_llava
from src.model import load_frozen_llava
from src.visual_channel_rank_grid import _layers, _to_device_item

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf'
DATA = ROOT / 'artifacts/diagnostics/channel_native_cache_20260916'
MODES = ('native', 'visual_self_only', 'text_query_only_visual_ffn')


class Intervention:
    def __init__(self, model):
        self.mode = 'native'
        self.positions = None
        self.pending = {}
        self.changed = []
        self.errors = []
        self.validate = False
        self.layers = _layers('llava', model)
        for li, layer in enumerate(self.layers):
            layer.self_attn.register_forward_pre_hook(self.capture(li), with_kwargs=True)
            layer.self_attn.o_proj.register_forward_pre_hook(self.replace_self(li))
            layer.self_attn.o_proj.register_forward_hook(self.zero_visual(li))

    def capture(self, li):
        def hook(m, args, kw):
            h = kw.get('hidden_states', args[0] if args else None)
            if self.mode != 'visual_self_only' or h.shape[1] == 1:
                return
            assert h.shape[0] == 1
            shape = (*h.shape[:-1], -1, m.head_dim)
            q = m.q_proj(h).view(shape).transpose(1, 2)
            k = m.k_proj(h).view(shape).transpose(1, 2)
            v = m.v_proj(h).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *kw['position_embeddings'])
            q, k, v = (x[0].transpose(0, 1) for x in (q, k, v))
            p = self.positions
            n, prefix = len(p), int(p[0])
            Q = q[p, None].contiguous()
            K = torch.cat((k[:prefix].unsqueeze(0).expand(n, -1, -1, -1), k[p, None]), 1).contiguous()
            V = torch.cat((v[:prefix].unsqueeze(0).expand(n, -1, -1, -1), v[p, None]), 1).contiguous()
            out = flash_attn_func(Q, K, V, dropout_p=0., softmax_scale=m.scaling, causal=False)[:, 0]
            if self.validate:
                ix = torch.tensor([0, n // 2, n - 1], device=p.device)
                groups = q.shape[1] // k.shape[1]
                kk = k.float().repeat_interleave(groups, 1).transpose(0, 1)
                vv = v.float().repeat_interleave(groups, 1).transpose(0, 1)
                qq = q[p[ix]].float().transpose(0, 1)
                logits = qq @ kk.transpose(-1, -2) * m.scaling
                keys = torch.arange(len(k), device=p.device)
                allowed = (keys[None] < prefix) | (keys[None] == p[ix, None])
                ref = (logits.masked_fill(~allowed[None], -torch.inf).softmax(-1) @ vv).transpose(0, 1)
                error = float((out[ix].float() - ref).norm() / ref.norm().clamp_min(1e-12))
                assert error < .02, (li, error)
                self.errors.append(error)
            self.pending[li] = out.reshape(n, -1)
        return hook

    def replace_self(self, li):
        def hook(m, args):
            if self.mode != 'visual_self_only' or args[0].shape[1] == 1:
                return
            x = args[0]
            y = x.clone()
            y[0, self.positions] = self.pending.pop(li)
            if self.validate:
                text = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                text[self.positions] = False
                assert torch.equal(x[:, text], y[:, text])
            self.changed.append(li)
            return (y, *args[1:])
        return hook

    def zero_visual(self, li):
        def hook(m, args, output):
            if self.mode != 'text_query_only_visual_ffn' or output.shape[1] == 1:
                return
            y = output.clone()
            y[:, self.positions] = 0
            if self.validate:
                text = torch.ones(y.shape[1], dtype=torch.bool, device=y.device)
                text[self.positions] = False
                assert torch.equal(y[:, text], output[:, text])
                assert torch.count_nonzero(y[:, self.positions]) == 0
            self.changed.append(li)
            return y
        return hook


def worker(args):
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor, model = load_frozen_llava(MODEL, torch.bfloat16, 'cuda:0', 'flash_attention_2')
    assert model.config.text_config._attn_implementation == 'flash_attention_2'
    control = Intervention(model)
    with torch.inference_mode(), (args.output / f'rows{args.rank}.jsonl').open('w', buffering=1) as handle:
        for name in ('realworldqa', 'mmstar', 'sqa'):
            path = DATA / f'{name}_eval.jsonl'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            ds = LlavaBenchmarkDataset(str(path), processor, name, max_samples=args.limit or None)
            for i in range(args.rank, len(ds), args.workers):
                item = ds[i]
                batch = _to_device_item(item, torch.device('cuda:0'))
                inputs = {k: batch[k] for k in ('input_ids', 'pixel_values', 'attention_mask', 'image_sizes') if k in batch}
                p = (inputs['input_ids'][0] == model.config.image_token_index).nonzero().flatten()
                assert len(p) == 576, len(p)
                assert torch.equal(p, torch.arange(int(p[0]), int(p[-1]) + 1, device=p.device))
                control.positions = p
                for mode in MODES:
                    control.mode = mode
                    control.changed, control.errors = [], []
                    control.validate = i == args.rank
                    _, answer = generate_teacher_llava(model, processor, **inputs, max_new_tokens=get_benchmark_spec(name).max_new_tokens)
                    assert control.changed == ([] if mode == 'native' else list(range(len(control.layers))))
                    assert not control.pending
                    score = score_prediction(metric=get_benchmark_spec(name).metric, prediction_text=answer,
                                             answer=item.get('answer'), answers=item.get('answers'),
                                             choices=item.get('choices'), question=item['row'].get('question'))
                    handle.write(json.dumps(dict(dataset=name, sample=i, mode=mode, answer=answer, **score,
                                                 manifest_sha256=digest, visual_tokens=len(p),
                                                 changed_layers=control.changed,
                                                 sparse_reference_max_relative_error=max(control.errors, default=0.)), ensure_ascii=False) + '\n')
                if i // args.workers % 20 == 0:
                    print(name, i, flush=True)
    (args.output / f'done{args.rank}.json').write_text(json.dumps({'complete': True}))


def launch(args):
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='running', started=time.time(), samples_limit=args.limit, workers=args.workers)
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        model=MODEL, attention='flash_attention_2', deepstack='not present in LLaVA-1.5', dtype='bfloat16',
        layers=32, training=False, modes=list(MODES),
        self_only='visual Q attends preceding text and itself; excludes all other visual keys before softmax',
        ffn_only='visual attention output zero after o_proj; residual and FFN retained',
        text='native causal text queries read visual and text K/V',
        generation='greedy native KV cache, benchmark max_new_tokens, same prompts across modes',
        data=str(DATA), validation='first sample per shard/dataset: FP32 sparse reference and unchanged text rows'
    ), indent=2))
    jobs = []
    try:
        (args.output / 'status.json').write_text(json.dumps(state, indent=2))
        for rank in range(args.workers):
            log = (args.output / f'gpu{rank}.log').open('w')
            cmd = [sys.executable, '-u', '-m', 'src.llava_visual_attention_ablation', '--output', str(args.output),
                   '--workers', str(args.workers), '--rank', str(rank), '--limit', str(args.limit)]
            p = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                                 OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'), stdout=log, stderr=subprocess.STDOUT)
            jobs.append((p, log))
        while any(p.poll() is None for p, _ in jobs):
            if any(p.poll() not in (None, 0) for p, _ in jobs):
                raise RuntimeError('LLaVA worker failed; inspect gpu logs')
            time.sleep(3)
        rows = [json.loads(line) for rank in range(args.workers) for line in (args.output / f'rows{rank}.jsonl').read_text().splitlines()]
        result = {}
        for name, n in (('realworldqa', 765), ('mmstar', 1000), ('sqa', 1000)):
            n = min(n, args.limit) if args.limit else n
            result[name] = {}
            for mode in MODES:
                selected = [r for r in rows if r['dataset'] == name and r['mode'] == mode]
                assert len(selected) == n and {r['sample'] for r in selected} == set(range(n))
                result[name][mode] = dict(samples=n, accuracy_pct=100 * sum(r['score'] for r in selected) / n,
                                          invalid_pct=100 * sum(bool(r.get('invalid')) for r in selected) / n)
        (args.output / 'results.json').write_text(json.dumps(result, indent=2))
        state.update(state='complete', finished=time.time())
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        (args.output / 'status.json').write_text(json.dumps(state, indent=2))
        for p, log in jobs:
            if p.poll() is None:
                p.terminate()
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.rank is not None else launch(args)
