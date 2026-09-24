"""Test what can be predicted from ONE initial visual embedding E_i.

Targets come from a frozen FA2 / DeepStack-off Qwen teacher. In a student
forward, no teacher layer cache is accessible. The visual encoder is unchanged.
This is a functional experiment; native visual work is still executed and then
overwritten, so runtimes are deliberately not interpreted as acceleration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from analysis.fig01b_hidden_prediction.tokenwise_direct_probe import image_signature, Metrics, TRAIN
from analysis.common.visual_cross_token_ablation import decompose, flatten_heads
from analysis.fig01a_hidden_channels.visual_channel_native_cache import ROOT, dump_json, load_rows, digest, get_model
from analysis.fig01a_hidden_channels.visual_channel_rank_grid import _to_device_item

PREVIOUS = ROOT / 'artifacts/diagnostics/pixmo_tokenwise_direct_1000_20260912'
EVAL_SOURCE = ROOT / 'artifacts/diagnostics/channel_native_cache_20260916'
PROTOCOL = 'initial_single_visual_token_mlp_v1'
HEADS = ('cross', 'rest', 'hidden')
DEPTH = 36
WIDTH = 2560


def prepare(root, steps):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        plan = json.loads((root / 'plan.json').read_text())
        assert plan['protocol'] == PROTOCOL and plan['steps'] == steps
        for item in plan['manifests'].values():
            assert digest(root / item['file']) == item['sha256']
        return plan
    manifests = {}
    eval_paths = []
    for name in ('mmstar', 'sqa', 'realworldqa'):
        src = EVAL_SOURCE / f'{name}_eval.jsonl'
        dst = root / src.name
        dst.write_bytes(src.read_bytes())
        rows = load_rows(dst)
        eval_paths.extend(r['image'] for r in rows)
        manifests[name] = dict(file=dst.name, sha256=digest(dst), samples=len(rows))
    with ThreadPoolExecutor(max_workers=16) as pool:
        es = list(pool.map(image_signature, sorted(set(eval_paths))))
        esha, eph = {s[0] for s in es}, {s[1] for s in es}
        candidates = load_rows(EVAL_SOURCE / 'pixmo_calibration.jsonl')
        sig = list(pool.map(image_signature, [r['image'] for r in candidates]))
    val = []
    for row, (sha, ph) in zip(candidates, sig):
        if sha in esha or any((ph ^ p).bit_count() <= 4 for p in eph):
            continue
        val.append(dict(image=row['image'], question=row['question'], sha256_rgb=sha, phash=str(ph)))
        if len(val) == 128:
            break
    assert len(val) == 128
    blocked_sha = esha | {r['sha256_rgb'] for r in val}
    blocked_ph = eph | {int(r['phash']) for r in val}
    audit = json.loads((PREVIOUS / 'image_audit.json').read_text())['train_images']
    allowed = {}
    for image, s in audit.items():
        allowed[image] = s['sha256_rgb'] not in blocked_sha and not any(
            (int(s['phash']) ^ p).bit_count() <= 4 for p in blocked_ph)
    old = load_rows(PREVIOUS / 'train_rows.jsonl')
    train = [dict(image=str(TRAIN.parent / r['image']), question=r['question'], source_row=r['source_row'])
             for r in old if allowed[r['image']]]
    assert len(train) > 30000
    for name, rows in (('train', train), ('validation', val)):
        dst = root / f'{name}.jsonl'
        dst.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifests[name] = dict(file=dst.name, sha256=digest(dst), samples=len(rows))
    plan = dict(protocol=PROTOCOL, model='Qwen3-VL-4B-Instruct', hidden_dim=WIDTH, depth=DEPTH,
                attention='flash_attention_2', deepstack='off', initial_input='E_i only; before language layer 0',
                steps=steps, world_size=8, batch_per_gpu=4, seed=44, lr=3e-4,
                mlp_width=WIDTH, heads=list(HEADS), manifests=manifests,
                train_unique_images=len({r['image'] for r in train}), validation_images=128,
                image_split='exact decoded RGB and DCT pHash Hamming<=4 excluded across train/val/eval',
                normalization_images=1024,
                target_layers='cross/rest: 0..34; hidden: inputs 1..35',
                cross_target='W_O(sum_{visual j!=i} a_ij V_j), original causal denominator',
                rest_target='actual native attention output minus cross target',
                hidden_target='native layer input minus E_i',
                training_loss='per-image per-channel-standardized MSE, equal head/layer weight; no answer supervision',
                input_normalization='fixed channel mean/std fitted on training images; no cross-token operation',
                output_initialization='training target mean + zero or small-Xavier residual; paired pilot chooses on validation',
                inference='MLP predictions from E_i only; all relevant layers simultaneously; student text propagates',
                caveat='initial E_i already contextualized by frozen vision encoder; does not remove encoder mixing',
                source_sha256=digest(__file__))
    dump_json(root / 'plan.json', plan)
    return plan


def inputs_for(processor, rows, device):
    from src.model import prepare_qwen3vl_batch_inputs
    inputs, *_ = prepare_qwen3vl_batch_inputs(processor, rows, TRAIN.parent, device, include_answers=False)
    return inputs


class PathHook:
    def __init__(self, model):
        self.model = model
        self.layers = model.model.language_model.layers
        self.mode = 'off'
        self.handles = []
        self.targets = {}
        self.cross = {}
        self.initial = None
        self.native = {}
        self.predictions = {}
        self.bank = None
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self.layer_input(li), with_kwargs=True))
            if li < DEPTH - 1:
                self.handles.append(layer.self_attn.register_forward_pre_hook(self.attention_input(li), with_kwargs=True))
                self.handles.append(layer.self_attn.o_proj.register_forward_hook(self.attention_output(li)))

    def begin(self, inputs, mode, bank=None, native=None, cache_native=True):
        self.mode = mode
        self.bank = bank
        self.cache_native = cache_native
        self.pos = [(r == self.model.config.image_token_id).nonzero().flatten() for r in inputs['input_ids']]
        self.lengths = [int(m.sum()) for m in inputs['attention_mask']]
        self.sizes = [len(p) for p in self.pos]
        for p, m, n in zip(self.pos, inputs['attention_mask'], self.lengths):
            assert len(p) > 0 and int(p[-1]) < n
            assert bool(m[:n].all()) and not bool(m[n:].any()), 'Right padding required'
            assert bool((p[1:] - p[:-1] == 1).all()), 'Single contiguous visual block required'
        self.calls = {}
        self.initial = None
        self.targets = {}
        self.cross = {}
        self.native = {} if native is None else native
        self.predictions = {}
        if mode not in ('capture', 'replay'):
            assert not self.native, 'Student must not receive native layer states'

    def take(self, h):
        return torch.cat([h[b, p] for b, p in enumerate(self.pos)])

    def put(self, h, x):
        out = h.clone()
        for b, (p, v) in enumerate(zip(self.pos, x.split(self.sizes))):
            out[b, p] = v.to(h.dtype)
        return out

    def layer_input(self, li):
        def hook(module, args, kwargs):
            h = kwargs.get('hidden_states', args[0] if args else None)
            if self.mode == 'off' or h.shape[1] == 1:
                return
            self.calls[li] = self.calls.get(li, 0) + 1
            if li == 0:
                self.initial = self.take(h).detach()
                if self.mode not in ('capture', 'native', 'replay', 'hidden_identity'):
                    assert not self.targets and not self.native
                    if self.mode.startswith('hidden'):
                        names = ['hidden']
                    else:
                        names = ['cross', 'rest']
                    self.predictions = self.bank.predict(self.initial, names, mean_only=self.mode.endswith('_mean'))
                    if self.mode == 'cross_mean':
                        self.predictions = self.bank.predict(self.initial, names, mean_only=False)
                        for l in range(DEPTH - 1):
                            self.predictions[f'cross_{l}'] = self.bank.heads[f'cross_{l}'].mean_prediction(self.initial)
            if self.mode == 'capture':
                visual = self.take(h).detach()
                if self.cache_native:
                    self.native[li] = visual.clone()
                if li:
                    self.targets[f'hidden_{li-1}'] = (visual.float() - self.initial.float()).detach()
                return
            if li == 0:
                return
            if self.mode == 'replay':
                value = self.native[li]
            elif self.mode == 'hidden_identity':
                value = self.initial
            elif self.mode.startswith('hidden'):
                value = self.initial.float() + self.predictions[f'hidden_{li-1}']
            else:
                return
            out = self.put(h, value)
            if 'hidden_states' in kwargs:
                return args, {**kwargs, 'hidden_states': out}
            return (out, *args[1:]), kwargs
        return hook

    def attention_input(self, li):
        def hook(module, args, kwargs):
            if self.mode != 'capture':
                return
            h = kwargs.get('hidden_states', args[0] if args else None)
            if h.shape[1] == 1:
                return
            shape = (*h.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(h).view(shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(shape)).transpose(1, 2)
            v = module.v_proj(h).view(shape).transpose(1, 2)
            from src.model import qwen_apply_rotary_pos_emb
            q, k = qwen_apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            groups = q.shape[1] // k.shape[1]
            vals = []
            for b, p in enumerate(self.pos):
                n = int(p[-1]) + 1  # later question tokens are causally invisible
                _, cross, _, _ = decompose(q[b, :, :n], k[b, :, :n].repeat_interleave(groups, 0),
                                           v[b, :, :n].repeat_interleave(groups, 0), p, float(module.scaling))
                vals.append(F.linear(flatten_heads(cross), module.o_proj.weight.float()))
            self.cross[li] = torch.cat(vals).detach()
        return hook

    def attention_output(self, li):
        def hook(module, args, output):
            if self.mode == 'off' or output.shape[1] == 1:
                return
            if self.mode == 'capture':
                full = self.take(output).float()
                self.targets[f'cross_{li}'] = self.cross.pop(li)
                self.targets[f'rest_{li}'] = full - self.targets[f'cross_{li}']
                return
            if self.mode in ('attn_mlp', 'attn_mean', 'cross_mean'):
                assert not self.targets and not self.native
                return self.put(output, self.predictions[f'cross_{li}'] + self.predictions[f'rest_{li}'])
        return hook

    def check(self):
        assert self.calls == {i: 1 for i in range(DEPTH)}, self.calls


def run_model(model, inputs, hook, body=True):
    model.model.rope_deltas = None
    with torch.no_grad():
        out = (model.model if body else model)(**inputs, use_cache=False, return_dict=True,
                                               **({} if body else {'logits_to_keep': 1}))
    hook.check()
    return out


class TokenMLP(nn.Module):
    def __init__(self, xmean, xstd, ymean, ystd, init='zero', width=None):
        super().__init__()
        dim = len(xmean)
        self.register_buffer('xmean', xmean.float())
        self.register_buffer('xstd', xstd.float())
        self.register_buffer('ymean', ymean.float())
        self.register_buffer('ystd', ystd.float())
        self.down = nn.Linear(dim, width or dim)
        self.up = nn.Linear(width or dim, len(ymean))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
        if init == 'zero':
            nn.init.zeros_(self.up.weight)
        elif init == 'small':
            nn.init.xavier_uniform_(self.up.weight, gain=.01)
        else:
            raise ValueError(init)

    def normalized(self, x):
        z = (x.float() - self.xmean) / self.xstd
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.up(F.silu(self.down(z)))

    def forward(self, x):
        return self.ymean + self.ystd * self.normalized(x).float()

    def mean_prediction(self, x):
        return self.ymean.expand(len(x), -1)

    def loss(self, x, y, sizes):
        pred = self.normalized(x).float()
        target = (y.float() - self.ymean) / self.ystd
        losses = (pred - target).square().mean(-1).split(sizes)
        return torch.stack([part.mean() for part in losses]).mean()


class Bank(nn.Module):
    def __init__(self, stats, init='zero', keys=None):
        super().__init__()
        self.activation_checkpointing = True
        self.heads = nn.ModuleDict()
        for name in (keys if keys is not None else stats['targets']):
            s = stats['targets'][name]
            self.heads[name] = TokenMLP(stats['input']['mean'], stats['input']['std'], s['mean'], s['std'], init)

    def forward(self, x, targets, sizes):
        losses = {}
        for key, head in self.heads.items():
            if torch.is_grad_enabled() and self.activation_checkpointing:
                losses[key] = checkpoint(head.loss, x, targets[key], sizes, use_reentrant=False, preserve_rng_state=False)
            else:
                losses[key] = head.loss(x, targets[key], sizes)
        total = torch.stack(list(losses.values())).mean()
        return total, {k: v.detach() for k, v in losses.items()}

    def predict(self, x, names=HEADS, mean_only=False):
        with torch.no_grad():
            return {key: (head.mean_prediction(x) if mean_only else head(x))
                    for key, head in self.heads.items() if key.split('_')[0] in names}


def runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))


def teacher():
    from src.model import load_frozen_qwen3vl
    from src.model_setup import disable_qwen_deepstack
    from analysis.fig01a_hidden_channels.visual_channel_native_cache import MODELS
    processor, model = load_frozen_qwen3vl(MODELS['qwen'], torch.bfloat16, torch.device('cuda', torch.cuda.current_device()), 'flash_attention_2')
    disable_qwen_deepstack(model)
    assert model.config.text_config._attn_implementation == 'flash_attention_2'
    assert all(not p.requires_grad for p in model.parameters())
    return processor, model


def statistics(processor, model, hook, rows, rank, world):
    import torch.distributed as dist
    accum = {}
    pilot = []
    seen = set()
    unique = []
    for row in rows:
        if row['image'] not in seen:
            seen.add(row['image']); unique.append(row)
    random.Random(44).shuffle(unique)
    selected = unique[:1024]
    for i in range(rank, len(selected), world):
        inputs = inputs_for(processor, [selected[i]], model.device)
        hook.begin(inputs, 'capture', cache_native=False); run_model(model, inputs, hook)
        values = {'input': hook.initial, **hook.targets}
        for key, value in values.items():
            x = value.double()
            if key not in accum:
                accum[key] = torch.zeros((3, x.shape[-1]), dtype=torch.float64, device=x.device)
            accum[key][0] += x.sum(0)
            accum[key][1] += x.square().sum(0)
            accum[key][2] += len(x)
        if len(pilot) < 8:
            pilot.append((hook.initial.detach().clone(), {k: hook.targets[k].clone() for k in ('cross_17', 'rest_17', 'hidden_17')}, hook.sizes[:]))
    stats = {'targets': {}}
    for key in sorted(accum):
        a = accum[key]
        dist.all_reduce(a)
        mean = a[0] / a[2]
        std = (a[1] / a[2] - mean.square()).clamp_min(0).sqrt()
        std = std.clamp_min(max(float(std.mean()) * .01, 1e-6))
        value = dict(mean=mean.float().cpu(), std=std.float().cpu(), count=int(a[2, 0]))
        if key == 'input': stats['input'] = value
        else: stats['targets'][key] = value
    stats['training_images'] = [r['image'] for r in selected]
    return stats, pilot


def init_pilot(stats, samples, heldout, rank):
    report = {}
    keys = ['cross_17', 'rest_17', 'hidden_17']
    for init in ('zero', 'small'):
        torch.manual_seed(44 + rank)
        bank = Bank(stats, init, keys).cuda()
        opt = torch.optim.AdamW(bank.parameters(), lr=3e-4, weight_decay=.01, fused=True)
        def measure(data):
            with torch.no_grad():
                return float(torch.stack([bank(x, y, size)[0] for x, y, size in data]).mean())
        first = measure(samples)
        grads = []
        for step in range(200):
            x, y, sizes = samples[step % len(samples)]
            opt.zero_grad(set_to_none=True)
            loss, _ = bank(x, y, sizes)
            loss.backward()
            if step < 2:
                grads.append({k: dict(down=float(h.down.weight.grad.norm()), up=float(h.up.weight.grad.norm())) for k, h in bank.heads.items()})
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
            opt.step()
        last = measure(samples)
        report[init] = dict(initial_train=first, final_train=last, validation=measure(heldout), gradients_first_two_steps=grads)
        assert math.isfinite(last) and last < first, 'Pilot did not learn'
        assert all(v['down'] > 0 and v['up'] > 0 for v in grads[1].values()), 'Dead branch after first update'
        del bank, opt
    return report


def validate(bank, processor, model, hook, rows, rank, world):
    import torch.distributed as dist
    total = torch.zeros(2, device=model.device, dtype=torch.float64)
    with torch.no_grad():
        for i in range(rank, len(rows), world):
            inp = inputs_for(processor, [rows[i]], model.device)
            hook.begin(inp, 'capture', cache_native=False); run_model(model, inp, hook)
            loss, _ = bank(hook.initial, hook.targets, hook.sizes)
            total[0] += loss.double(); total[1] += 1
    dist.all_reduce(total)
    return float(total[0] / total[1])


def train(root, pilot_only=False):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    runtime()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    plan = json.loads((root / 'plan.json').read_text())
    dump_json(root / f'runtime_rank{rank}.json', dict(source_sha256=digest(__file__), plan_sha256=digest(root / 'plan.json'),
                                                   torch=torch.__version__, cuda=torch.version.cuda,
                                                   rank=rank, world=world, started=time.time()))
    rows, valrows = load_rows(root / 'train.jsonl'), load_rows(root / 'validation.jsonl')
    processor, model = teacher()
    hook = PathHook(model)
    started = time.time()
    inp = inputs_for(processor, [valrows[rank]], model.device)
    hook.begin(inp, 'capture')
    captured_logits = run_model(model, inp, hook, body=False).logits
    native_states = hook.native
    hook.begin(inp, 'native')
    native_logits = run_model(model, inp, hook, body=False).logits
    hook.begin(inp, 'replay', native=native_states)
    replay_logits = run_model(model, inp, hook, body=False).logits
    check = dict(capture_max_logit_error=float((captured_logits.float() - native_logits.float()).abs().max()),
                 replay_max_logit_error=float((replay_logits.float() - native_logits.float()).abs().max()),
                 attention='flash_attention_2', deepstack='off')
    assert check['capture_max_logit_error'] == check['replay_max_logit_error'] == 0, check
    dump_json(root / f'path_validation_rank{rank}.json', check)
    del native_states, captured_logits, native_logits, replay_logits
    saved = None
    if (root / 'resume.pt').exists():
        saved = torch.load(root / 'resume.pt', map_location='cpu', mmap=True, weights_only=False)
        assert saved['plan']['manifests'] == plan['manifests'] and saved['plan']['steps'] == plan['steps']
        stats, init = saved['stats'], saved['init']
    elif (root / 'normalization.pt').exists() and (root / 'initialization.json').exists():
        stats = torch.load(root / 'normalization.pt', map_location='cpu', weights_only=False)
        init = json.loads((root / 'initialization.json').read_text())['selected']
    else:
        stats, samples = statistics(processor, model, hook, rows, rank, world)
        heldout = []
        for i in range(rank, 32, world):
            inp = inputs_for(processor, [valrows[i]], model.device)
            hook.begin(inp, 'capture', cache_native=False); run_model(model, inp, hook)
            heldout.append((hook.initial.clone(), {k: hook.targets[k].clone() for k in ('cross_17', 'rest_17', 'hidden_17')}, hook.sizes[:]))
        pilot = init_pilot(stats, samples, heldout, rank)
        dump_json(root / f'pilot_rank{rank}.json', pilot)
        scores = torch.tensor([pilot[k]['validation'] for k in ('zero', 'small')], device=model.device)
        dist.all_reduce(scores); scores /= world
        init = ('zero', 'small')[int(scores.argmin())]
        if rank == 0:
            torch.save(stats, root / 'normalization.pt')
            dump_json(root / 'initialization.json', dict(selected=init, paired_validation=dict(zip(('zero', 'small'), scores.tolist())),
                                                       pilot_steps=200, seeds=list(range(44, 52)),
                                                       note='pilot on training images; choice by separate Pixmo validation, never benchmark accuracy'))
        del samples, heldout
    hook.begin(inp, 'native')
    torch.cuda.empty_cache()
    if pilot_only:
        dist.barrier(); dist.destroy_process_group(); return
    torch.manual_seed(44)
    bank = Bank(stats, init).cuda()
    if saved is not None:
        bank.load_state_dict(saved['bank'])
    ddp = DistributedDataParallel(bank, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(bank.parameters(), lr=plan['lr'], betas=(.9, .95), weight_decay=.01, fused=True)
    first_step = 0 if saved is None else saved['step']
    best = float('inf') if saved is None else saved['best_validation']
    if saved is not None:
        opt.load_state_dict(saved['optimizer'])
        rng = torch.load(root / f'rng_rank{rank}.pt', map_location='cpu', weights_only=False)
        assert rng['step'] == first_step
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'])
        del saved, rng
    log = (root / f'train_rank{rank}.jsonl').open('a' if first_step else 'w', buffering=1)
    for step in range(first_step, plan['steps']):
        start = (step * 32 + rank * 4) % len(rows)
        batch = [rows[(start + j) % len(rows)] for j in range(4)]
        inp = inputs_for(processor, batch, model.device)
        hook.begin(inp, 'capture', cache_native=False); run_model(model, inp, hook)
        warm = max(1, round(plan['steps'] * .03))
        scale = (step + 1) / warm if step < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, plan['steps'] - warm - 1)))
        opt.param_groups[0]['lr'] = plan['lr'] * scale
        opt.zero_grad(set_to_none=True)
        loss, per = ddp(hook.initial, hook.targets, hook.sizes)
        assert bool(torch.isfinite(loss))
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
        assert bool(torch.isfinite(grad))
        opt.step()
        number = loss.detach().clone(); dist.all_reduce(number); number /= world
        entry = dict(step=step + 1, loss=float(number), local_loss=float(loss.detach()), grad_norm=float(grad),
                     lr=opt.param_groups[0]['lr'], seconds=time.time() - started,
                     peak_memory_bytes=torch.cuda.max_memory_allocated(),
                     by_head={h: float(torch.stack([v for k, v in per.items() if k.startswith(h + '_')]).mean()) for h in HEADS})
        if (step + 1) in (250, 500, 1000, plan['steps']):
            entry['validation'] = validate(bank, processor, model, hook, valrows, rank, world)
            if entry['validation'] < best:
                best = entry['validation']
                if rank == 0:
                    tmp = root / 'best.pt.tmp'
                    torch.save(dict(bank=bank.state_dict(), stats=stats, step=step + 1, init=init,
                                    validation=best, plan=plan, source_sha256=digest(__file__)), tmp)
                    tmp.replace(root / 'best.pt')
            if rank == 0:
                tmp = root / 'resume.pt.tmp'
                torch.save(dict(bank=bank.state_dict(), optimizer=opt.state_dict(), stats=stats, step=step + 1,
                                init=init, best_validation=best, plan=plan, source_sha256=digest(__file__)), tmp)
                tmp.replace(root / 'resume.pt')
            torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), step=step + 1),
                       root / f'rng_rank{rank}.pt')
            dist.barrier()
        log.write(json.dumps(entry) + '\n')
        if rank == 0 and (step == 0 or (step + 1) % 10 == 0):
            print('TRAIN', json.dumps(entry), flush=True)
        del loss, per
    log.close()
    dump_json(root / f'train_rank{rank}.done.json', dict(seconds=time.time() - started, steps=plan['steps'], best_validation=best))
    dist.barrier(); dist.destroy_process_group()


def evaluate(root, shard, limit=None):
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    from src.evaluate import generate_teacher_qwen
    runtime()
    processor, model = teacher()
    checkpoint = torch.load(root / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = Bank(checkpoint['stats'], checkpoint['init']).cuda().eval()
    bank.load_state_dict(checkpoint['bank'])
    del checkpoint
    hook = PathHook(model)
    original = model.model.get_image_features
    features = []
    def cached(*a, **kw):
        if not features: features.append(original(*a, **kw))
        return features[0]
    model.model.get_image_features = cached
    modes = ['native', 'hidden_identity', 'hidden_mean', 'hidden_mlp', 'attn_mean', 'cross_mean', 'attn_mlp']
    allmetrics = {}
    with (root / f'eval_{shard}.jsonl').open('w', buffering=1) as f, torch.no_grad():
        for name in ('mmstar', 'sqa', 'realworldqa'):
            ds = QwenBenchmarkDataset(str(root / f'{name}_eval.jsonl'), processor, name, max_samples=limit)
            metrics = {k: Metrics(WIDTH) for k in bank.heads}
            meanmetrics = {k: Metrics(WIDTH) for k in bank.heads}
            for i in range(shard, len(ds), 8):
                item = ds[i]
                inp = _to_device_item(item, model.device)
                inp = {k: v for k, v in inp.items() if k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids')}
                features.clear()
                hook.begin(inp, 'capture'); native_logits = run_model(model, inp, hook, body=False).logits
                saved = hook.native
                native_initial = hook.initial.clone()
                prediction = bank.predict(native_initial)
                for key, target in hook.targets.items():
                    pred = prediction[key]
                    mean = bank.heads[key].mean_prediction(native_initial)
                    if key.startswith('hidden_'):
                        target, pred, mean = [t + native_initial.float() for t in (target, pred, mean)]
                    metrics[key].add(pred, target)
                    meanmetrics[key].add(mean, target)
                hook.begin(inp, 'replay', native=saved)
                replay_logits = run_model(model, inp, hook, body=False).logits
                error = float((native_logits.float() - replay_logits.float()).abs().max())
                assert error == 0, ('Native replay changed logits', name, i, error)
                del saved, prediction, native_initial
                results = {}
                for mode in modes:
                    hook.begin(inp, mode, bank)
                    _, text = generate_teacher_qwen(model, processor, **inp, max_new_tokens=get_benchmark_spec(name).max_new_tokens)
                    hook.check()
                    if mode != 'native':
                        assert not hook.native and not hook.targets, 'Teacher information leaked into student'
                    score = score_prediction(metric=get_benchmark_spec(name).metric, prediction_text=text,
                                             answer=item['answer'], choices=item.get('choices'), question=item['row'].get('question'))
                    results[mode] = dict(text=text, **score)
                f.write(json.dumps(dict(benchmark=name, sample=i, sample_id=item['index'], predictions=results,
                                        native_replay_max_logit_error=error), ensure_ascii=False) + '\n')
                if i // 8 % 10 == 0:
                    print('EVAL', shard, name, i, flush=True)
            allmetrics[name] = dict(mlp={k: m.state() for k, m in metrics.items()}, mean={k: m.state() for k, m in meanmetrics.items()})
    dump_json(root / f'metrics_{shard}.json', allmetrics)
    dump_json(root / f'eval_{shard}.done.json', dict(complete=True))


def report(root):
    rows = [r for s in range(8) for r in load_rows(root / f'eval_{s}.jsonl')]
    plan = json.loads((root / 'plan.json').read_text())
    states = [json.loads((root / f'metrics_{s}.json').read_text()) for s in range(8)]
    results = {}
    doc = ['# Initial-token MLP: Qwen3-VL-4B', '', 'FA2；DeepStack 关闭；所有 MLP 只输入初始单个 visual token Eᵢ。', '',
           '| Dataset | N | Native | E identity | Hidden mean | Hidden MLP | Attention mean | Cross mean + Rest MLP | Cross MLP + Rest MLP |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('mmstar', 'sqa', 'realworldqa'):
        rr = sorted([r for r in rows if r['benchmark'] == name], key=lambda r: r['sample'])
        n = plan['manifests'][name]['samples']
        assert [r['sample'] for r in rr] == list(range(n))
        modes = list(rr[0]['predictions'])
        accuracy = {m: 100 * np.mean([r['predictions'][m]['score'] for r in rr]) for m in modes}
        paired = {}
        idx = np.random.default_rng(44).integers(0, n, (10000, n))
        for m in modes[1:]:
            d = np.array([r['predictions'][m]['score'] - r['predictions']['native']['score'] for r in rr])
            paired[m] = dict(delta_pp=float(d.mean() * 100), ci95_pp=np.quantile(d[idx].mean(1) * 100, [.025, .975]).tolist())
        fit = {which: {k: Metrics.merged([s[name][which][k] for s in states]) for k in states[0][name][which]} for which in ('mlp', 'mean')}
        results[name] = dict(samples=n, accuracy_pct=accuracy, paired_vs_native=paired, fit=fit)
        doc.append(f'| {name} | {n} | ' + ' | '.join(f'{accuracy[m]:.2f}' for m in modes) + ' |')
    doc += ['', '## Interpretation', '',
            '- Cross/Rest MLP 同时替换语言层 0–34 的完整视觉 attention 输出；原生 self/prefix 权重、softmax 分母不作为学生输入。原生 FFN 保留。',
            '- Hidden MLP 从 Eᵢ 直接预测层 1–35 的 visual 输入并同时替换；文本状态始终来自学生轨迹。',
            '- 最后一层视觉输出不能影响文本，故不将该位置的准确率当作证据。',
            '- 视觉编码器保持原生，其 token 已经过视觉编码器内部交互；结论只针对之后的语言 Transformer。',
            '- 模型均保留全部视觉 token 和完整 2560 维；MLP 中间宽度 2560，不进行低秩瓶颈或 token selection。',
            '- 原生视觉计算在功能实验中仍执行后丢弃；此表不测速度。R²/cosine/MSE、训练均值对照和配对区间见 results.json。',
            '- 较差结果只能说明当前训练配置未能替代，不能证明单 token 中不存在相应信息。']
    dump_json(root / 'results.json', results)
    (root / 'RESULTS.md').write_text('\n'.join(doc) + '\n')


def launch(root, steps, pilot_only=False):
    plan = prepare(root, steps)
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_MODE='disabled')
    status = dict(protocol=PROTOCOL, state='running', stage='initialization_and_training', started=time.time())
    dump_json(root / 'status.json', status)
    try:
        cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '-m',
               'analysis.fig01b_hidden_prediction.initial_token_mlp_probe', 'train', '--output', str(root)]
        if pilot_only: cmd.append('--pilot-only')
        with (root / 'train.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not pilot_only:
            status['stage'] = 'evaluation'; dump_json(root / 'status.json', status)
            procs = []
            try:
                for shard in range(8):
                    log = (root / f'eval_{shard}.log').open('w')
                    p = subprocess.Popen([sys.executable, '-u', '-m', 'analysis.fig01b_hidden_prediction.initial_token_mlp_probe', 'eval', '--output', str(root), '--shard', str(shard)],
                                         cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
                    procs.append((p, log))
                while any(p.poll() is None for p, _ in procs):
                    for p, _ in procs:
                        if p.poll() not in (None, 0): raise RuntimeError(f'Eval failed: {p.pid}')
                    time.sleep(5)
            finally:
                for p, log in procs:
                    if p.poll() is None: p.terminate()
                    log.close()
            report(root)
        status.update(state='complete', stage='pilot_complete' if pilot_only else 'complete', finished=time.time())
    except BaseException as exc:
        status.update(state='failed', error=repr(exc)); raise
    finally:
        dump_json(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'eval', 'report', 'launch'))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/diagnostics/initial_token_mlp_20260916')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output, args.steps)
    elif args.command == 'train': train(args.output, args.pilot_only)
    elif args.command == 'eval': evaluate(args.output, args.shard, args.limit)
    elif args.command == 'report': report(args.output)
    elif args.command == 'launch': launch(args.output, args.steps, args.pilot_only)
