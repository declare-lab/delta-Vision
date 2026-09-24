"""Independent predictions from E_i: attention increment and layer-input hidden.

Frozen native Qwen3-VL-4B, FA2, no DeepStack. All teacher layers up to the
targets remain untouched. Students see one initial visual token at a time.
No answer generation, suffix intervention, or teacher state in student inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import Bank, PathHook, inputs_for, runtime, teacher
from analysis.fig01a_hidden_channels.visual_channel_native_cache import ROOT, dump_json, load_rows, digest

LAYERS = (15, 16, 17)  # Zero-based language-layer indices, NOT output indices.
KINDS = ('delta', 'cross', 'hidden')
KEYS = tuple(f'{kind}_{layer}' for kind in KINDS for layer in LAYERS)
OLD = ROOT / 'artifacts/diagnostics/initial_token_mlp_qwen_20260916'
PROTOCOL = 'initial_token_attention_delta_and_hidden_l15_16_17_v1'


class TargetsReady(Exception):
    """Stop the frozen teacher only after every requested target was captured."""


class Capture(PathHook):
    def __init__(self, model, *, cross=True):
        self.model = model
        self.layers = model.model.language_model.layers
        self.mode = 'off'
        self.want_cross = cross
        self.stop_early = True
        self.handles = []
        for layer in (0, *LAYERS):
            self.handles.append(self.layers[layer].register_forward_pre_hook(self.layer_input(layer), with_kwargs=True))
        for layer in LAYERS:
            block = self.layers[layer]
            if cross:
                self.handles.append(block.self_attn.register_forward_pre_hook(self.attention_input(layer), with_kwargs=True))
            self.handles.append(block.self_attn.o_proj.register_forward_hook(self.attention_output(layer)))
            self.handles.append(block.post_attention_layernorm.register_forward_pre_hook(self.after_attention(layer)))

    def begin(self, inputs):
        super().begin(inputs, 'capture', cache_native=False)
        self.before = {}
        self.raw_attention = {}

    def layer_input(self, layer):
        def hook(module, args, kwargs):
            if self.mode == 'off':
                return
            h = kwargs.get('hidden_states', args[0] if args else None)
            self.calls[layer] = self.calls.get(layer, 0) + 1
            visual = self.take(h).detach()
            if layer == 0:
                self.initial = visual
            else:
                assert self.initial is not None
                self.before[layer] = visual
                # Residual parameterization, final prediction is E_i + MLP(E_i).
                self.targets[f'hidden_{layer}'] = visual.float() - self.initial.float()
        return hook

    def attention_output(self, layer):
        def hook(module, args, output):
            if self.mode == 'off':
                return
            self.raw_attention[layer] = self.take(output).detach()
            if self.want_cross:
                self.targets[f'cross_{layer}'] = self.cross.pop(layer)
        return hook

    def after_attention(self, layer):
        def hook(module, args):
            if self.mode == 'off':
                return
            before = self.before.pop(layer)
            after = self.take(args[0]).detach()
            # Subtract after casting: preserves the *actual* BF16 residual-add
            # result, instead of silently treating o_proj output as identical.
            expected = before + self.raw_attention[layer]
            assert torch.equal(after, expected), ('Incorrect attention boundary', layer)
            self.targets[f'delta_{layer}'] = after.float() - before.float()
            if layer == max(LAYERS) and self.stop_early:
                raise TargetsReady()
        return hook

    def collect(self, inputs, *, stop_early=True):
        self.begin(inputs)
        self.stop_early = stop_early
        self.model.model.rope_deltas = None
        with torch.no_grad():
            try:
                self.model.model(**inputs, use_cache=False, return_dict=True)
            except TargetsReady:
                assert stop_early
        assert self.calls == {layer: 1 for layer in (0, *LAYERS)}, self.calls
        expected = set(KEYS) if self.want_cross else {f'{k}_{l}' for k in ('delta', 'hidden') for l in LAYERS}
        assert set(self.targets) == expected
        assert not self.before and not self.cross and not self.native
        return self.initial, self.targets, self.sizes

    def close(self):
        self.mode = 'off'
        for handle in self.handles:
            handle.remove()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        plan = json.loads((root / 'plan.json').read_text())
        assert plan['protocol'] == PROTOCOL
        for info in plan['manifests'].values():
            assert digest(root / info['file']) == info['sha256']
        return plan
    old = json.loads((OLD / 'plan.json').read_text())
    manifests = {k: old['manifests'][k] for k in ('train', 'validation', 'mmstar', 'realworldqa')}
    for info in manifests.values():
        source = OLD / info['file']
        assert digest(source) == info['sha256']
        shutil.copyfile(source, root / info['file'])
    plan = dict(protocol=PROTOCOL, model='Qwen3-VL-4B-Instruct', hidden_dim=2560,
                attention='flash_attention_2', deepstack='off', layers=list(LAYERS), layer_numbering='zero-based',
                manifests=manifests, image_split=old['image_split'], train_unique_images=old['train_unique_images'],
                input='only initial individual visual embedding E_i, before language layer 0',
                delta_target='H_after_attention_residual_i - H_before_attention_i; before FFN, actual BF16 residual addition',
                cross_target='W_O sum_{visual j!=i} A_ij V_j; native causal denominator; supplementary isolation of visual cross mixing',
                hidden_target='input H_i of language layer 15/16/17; no compression or student predictions upstream',
                hidden_parameterization='prediction = E_i + MLP(E_i); loss on H_i - E_i, metrics on reconstructed full H_i',
                mlp='separate 2560 -> 2560 -> 2560 SiLU per target and layer; biases; no token mixing, pooling, or positional input',
                heads=list(KEYS), initialization='fresh zero output residual about TRAIN target mean; old validated initialization scheme',
                initialization_reference=str(OLD / 'initialization.json'), previous_weights_loaded=False,
                normalization='input and per-target channel mean/std from 1024 TRAIN images only',
                loss='per-image channel-standardized MSE, equal weight for nine independent heads',
                steps=2000, world_size=8, batch_per_gpu=4, seed=44, lr=3e-4,
                optimizer='AdamW betas=(0.9,0.95), weight_decay=0.01, clip=1; cosine, 3% warmup, 10% final LR',
                image_processing='original processor limits; every visual token, original resolution policy, no token subsampling',
                selection='best separate Pixmo validation MSE per independent head; never benchmark selection',
                evaluation='MMStar 1000, RealWorldQA 765; per-token cosine and original-unit MSE, equal image average',
                controls='training-mean prediction per target; additionally E_i identity for hidden',
                interpretation='predictability from contextualized E_i, not an accuracy or acceleration experiment',
                source_sha256=digest(__file__))
    dump_json(root / 'plan.json', plan)
    shutil.copyfile(__file__, root / 'source_snapshot.py')
    return plan


def normalization(root, processor, model, rank, world):
    """Reuse exact old cross/hidden statistics, fit the new residual delta."""
    import torch.distributed as dist
    if (root / 'normalization.pt').exists():
        return torch.load(root / 'normalization.pt', weights_only=False, map_location='cpu')
    previous = torch.load(OLD / 'normalization.pt', weights_only=False, map_location='cpu')
    stats = dict(input=previous['input'], targets={}, training_images=previous['training_images'])
    for l in LAYERS:
        stats['targets'][f'cross_{l}'] = previous['targets'][f'cross_{l}']
        stats['targets'][f'hidden_{l}'] = previous['targets'][f'hidden_{l-1}']
    rows = load_rows(root / 'train.jsonl')
    by_image = {}
    for row in rows:
        by_image.setdefault(row['image'], row)
    selected = [by_image[p] for p in stats['training_images']]
    assert len(selected) == 1024
    hook = Capture(model, cross=False)
    accum = {f'delta_{l}': torch.zeros(3, 2560, dtype=torch.float64, device=model.device) for l in LAYERS}
    for i in range(rank, len(selected), world):
        _, targets, _ = hook.collect(inputs_for(processor, [selected[i]], model.device))
        for key, a in accum.items():
            x = targets[key].double()
            a[0] += x.sum(0); a[1] += x.square().sum(0); a[2] += len(x)
        if rank == 0 and (i // world) % 16 == 0:
            print('NORMALIZATION', i, len(selected), flush=True)
    hook.close()
    for key, a in accum.items():
        dist.all_reduce(a)
        mean = a[0] / a[2]
        std = (a[1] / a[2] - mean.square()).clamp_min(0).sqrt()
        std = std.clamp_min(max(float(std.mean()) * .01, 1e-6))
        stats['targets'][key] = dict(mean=mean.float().cpu(), std=std.float().cpu(), count=int(a[2, 0]))
    stats['targets'] = {k: stats['targets'][k] for k in KEYS}
    if rank == 0:
        atomic_save(stats, root / 'normalization.pt')
    dist.barrier()
    return stats


def vector_metrics(pred, target, training_mean):
    """One sample. No large-image weighting, no hidden-state normalization."""
    pred, target, training_mean = pred.double(), target.double(), training_mean.double()
    squared = (pred - target).square().mean()
    mse_mean = (training_mean - target).square().mean()
    return dict(mse=float(squared), cosine=float(F.cosine_similarity(pred, target, dim=-1).mean()),
                relative_l2=float((squared / target.square().mean().clamp_min(1e-30)).sqrt()),
                mse_over_training_mean=float(squared / mse_mean.clamp_min(1e-30)))


def validation(bank, processor, model, hook, rows, rank, world):
    import torch.distributed as dist
    values = torch.zeros(len(KEYS) + 1, dtype=torch.float64, device=model.device)
    with torch.no_grad():
        for i in range(rank, len(rows), world):
            x, targets, sizes = hook.collect(inputs_for(processor, [rows[i]], model.device))
            for j, key in enumerate(KEYS):
                values[j] += bank.heads[key].loss(x, targets[key], sizes).double()
            values[-1] += 1
    dist.all_reduce(values)
    return dict(zip(KEYS, (values[:-1] / values[-1]).tolist()))


def atomic_save(value, path):
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, tmp)
    tmp.replace(path)


def train(root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    runtime()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256'], 'Source changed after the run was prepared'
    processor, model = teacher()
    rows, valrows = load_rows(root / 'train.jsonl'), load_rows(root / 'validation.jsonl')
    started = time.time()
    stats = normalization(root, processor, model, rank, world)
    hook = Capture(model)
    torch.manual_seed(plan['seed'])
    bank = Bank(stats, 'zero', keys=KEYS).cuda()
    ddp = DistributedDataParallel(bank, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=plan['lr'], betas=(.9, .95), weight_decay=.01, fused=True)
    first = 0
    best = {key: float('inf') for key in KEYS}
    best_steps = {key: 0 for key in KEYS}
    best_state = {}
    if (root / 'resume.pt').exists():
        saved = torch.load(root / 'resume.pt', weights_only=False, map_location='cpu', mmap=True)
        assert saved['plan'] == plan
        bank.load_state_dict(saved['bank']); optimizer.load_state_dict(saved['optimizer'])
        first, best, best_steps = saved['step'], saved['best'], saved['best_steps']
        if rank == 0:
            best_state = torch.load(root / 'best.pt', weights_only=False, map_location='cpu', mmap=True)['bank']
        del saved
    dump_json(root / f'runtime_rank{rank}.json', dict(source_sha256=digest(__file__),
              parameters=sum(p.numel() for p in bank.parameters()), world=world, rank=rank,
              processor_size=dict(processor.image_processor.size), start_step=first))
    with (root / f'train_rank{rank}.jsonl').open('a' if first else 'w', buffering=1) as log:
        for step in range(first, plan['steps']):
            tick = time.time()
            start = (step * 32 + rank * 4) % len(rows)
            batch = [rows[(start + j) % len(rows)] for j in range(4)]
            inp = inputs_for(processor, batch, model.device)
            x, targets, sizes = hook.collect(inp)
            warm = round(plan['steps'] * .03)
            scale = (step + 1) / warm if step < warm else .1 + .45 * (1 + math.cos(math.pi * (step-warm) / (plan['steps']-warm-1)))
            optimizer.param_groups[0]['lr'] = plan['lr'] * scale
            optimizer.zero_grad(set_to_none=True)
            loss, per = ddp(x, targets, sizes)
            assert bool(torch.isfinite(loss))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
            assert bool(torch.isfinite(grad))
            if step < 2:
                grads = {key: dict(down=float(head.down.weight.grad.norm()), up=float(head.up.weight.grad.norm()))
                         for key, head in bank.heads.items()}
                assert all(g['up'] > 0 and (step == 0 or g['down'] > 0) for g in grads.values())
                dump_json(root / f'initialization_step{step+1}_rank{rank}.json', grads)
                assert all(p.grad is None for p in model.parameters())
            optimizer.step()
            means = torch.stack([per[k] for k in KEYS]); dist.all_reduce(means); means /= world
            entry = dict(step=step+1, loss=float(means.mean()), by_target=dict(zip(KEYS, means.tolist())),
                         grad_norm=float(grad), lr=optimizer.param_groups[0]['lr'], visual_tokens=sizes,
                         seconds=time.time()-started, step_seconds=time.time()-tick,
                         peak_memory_bytes=torch.cuda.max_memory_allocated())
            if step == 0 or (step+1) % 250 == 0 or step+1 == plan['steps']:
                scores = validation(bank, processor, model, hook, valrows, rank, world)
                entry['validation'] = scores
                for key in KEYS:
                    if scores[key] < best[key]:
                        best[key], best_steps[key] = scores[key], step+1
                        if rank == 0:
                            best_state.update({f'heads.{key}.{k}': v.detach().cpu().clone()
                                               for k, v in bank.heads[key].state_dict().items()})
                if rank == 0:
                    atomic_save(dict(bank=best_state, stats=stats, best=best, best_steps=best_steps, plan=plan), root/'best.pt')
                    atomic_save(dict(bank=bank.state_dict(), optimizer=optimizer.state_dict(), stats=stats,
                                     step=step+1, best=best, best_steps=best_steps, plan=plan), root/'resume.pt')
                dist.barrier()
            log.write(json.dumps(entry, allow_nan=False)+'\n')
            if rank == 0 and (step == 0 or (step+1) % 10 == 0):
                print('TRAIN', json.dumps(entry), flush=True)
                dump_json(root / 'progress.json', dict(stage='training', **entry))
            del x, targets, loss, per, inp
    dump_json(root / f'train_rank{rank}.done.json', dict(steps=plan['steps'], seconds=time.time()-started, best=best, best_steps=best_steps))
    hook.close()
    dist.barrier(); dist.destroy_process_group()


def evaluate(root, shard):
    from src.data import QwenBenchmarkDataset
    from analysis.fig01a_hidden_channels.visual_channel_rank_grid import _to_device_item
    runtime()
    processor, model = teacher()
    saved = torch.load(root/'best.pt', weights_only=False, map_location='cpu', mmap=True)
    bank = Bank(saved['stats'], 'zero', keys=KEYS).cuda().eval()
    bank.load_state_dict(saved['bank'])
    best_steps = saved['best_steps']
    del saved
    hook = Capture(model)
    with (root / f'eval_{shard}.jsonl').open('w', buffering=1) as log, torch.no_grad():
        for benchmark in ('mmstar', 'realworldqa'):
            ds = QwenBenchmarkDataset(str(root/f'{benchmark}_eval.jsonl'), processor, benchmark)
            for i in range(shard, len(ds), 8):
                item = ds[i]
                inp = _to_device_item(item, model.device)
                inp = {k: v for k, v in inp.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')}
                x, targets, sizes = hook.collect(inp)
                metrics = {}
                for key, head in bank.heads.items():
                    pred, target, mean = head(x), targets[key], head.mean_prediction(x)
                    if key.startswith('hidden_'):
                        pred, target, mean = [v + x.float() for v in (pred, target, mean)]
                    values = dict(mlp=vector_metrics(pred, target, mean), mean=vector_metrics(mean, target, mean))
                    if key.startswith('hidden_'):
                        values['identity'] = vector_metrics(x, target, mean)
                        values['hidden_change'] = vector_metrics(pred-x.float(), target-x.float(), mean-x.float())
                    metrics[key] = values
                log.write(json.dumps(dict(benchmark=benchmark, sample=i, sample_id=item['index'],
                                          visual_tokens=sizes[0], metrics=metrics), allow_nan=False)+'\n')
                if i // 8 % 10 == 0:
                    print('EVAL', shard, benchmark, i, flush=True)
    hook.close()
    dump_json(root / f'eval_{shard}.done.json', dict(complete=True, best_steps=best_steps))


def report(root):
    plan = json.loads((root/'plan.json').read_text())
    rows = [r for shard in range(8) for r in load_rows(root/f'eval_{shard}.jsonl')]
    results = {}
    table = []
    doc = ['# 单个初始 visual token 的预测实验', '',
           'Qwen3-VL-4B，FA2，DeepStack 关闭；0-based 层 15/16/17；完整 hidden 维度 2560。', '',
           'attention 增量取残差相加后减去 attention 前的 hidden（FFN 前）；cross 单独排除 self 和文本前缀贡献。',
           'hidden 取对应层的输入。每个目标/层各自一个 MLP，只读取 E_i；原生教师不被替换、压缩或级联干预。', '',
           '| Dataset | Target | Layer | MLP MSE ↓ | MLP cosine ↑ | Mean MSE ↓ | Mean cosine ↑ | E identity MSE ↓ | E identity cosine ↑ |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for benchmark in ('mmstar','realworldqa'):
        rr = sorted((r for r in rows if r['benchmark']==benchmark), key=lambda r:r['sample'])
        assert [r['sample'] for r in rr] == list(range(plan['manifests'][benchmark]['samples']))
        results[benchmark] = dict(samples=len(rr), metrics={})
        for key in KEYS:
            scores = {mode: {metric: sum(r['metrics'][key][mode][metric] for r in rr)/len(rr)
                             for metric in rr[0]['metrics'][key][mode]} for mode in rr[0]['metrics'][key]}
            results[benchmark]['metrics'][key] = scores
            kind, layer = key.split('_')
            record = dict(dataset=benchmark, target=kind, layer=int(layer), samples=len(rr))
            for mode in ('mlp','mean','identity'):
                for metric in ('mse','cosine'):
                    record[f'{mode}_{metric}'] = scores.get(mode,{}).get(metric)
            table.append(record)
            numbers = [record[f'{m}_{s}'] for m in ('mlp','mean','identity') for s in ('mse','cosine')]
            doc.append(f'| {benchmark} | {kind} | {layer} | '+' | '.join('—' if v is None else f'{v:.6f}' for v in numbers)+' |')
    doc += ['', 'MSE 使用原始 hidden 数值；cosine 先逐 token 计算，再按图像等权平均。',
            'hidden 使用 E_i + MLP(E_i) 参数化；对应均值对照为 E_i + 训练集平均变化，identity 为直接 E_i。',
            '这些结果只检验数值可预测性；不能单凭 cosine/MSE 宣称 attention 分布或任务功能一致。',
            '训练为 Pixmo 2000 步，8 卡、每卡 4 条；输入维度和 MLP 中间宽度均为 2560，保留全部视觉 token。']
    dump_json(root/'results.json', results)
    with (root/'results.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    (root/'RESULTS.md').write_text('\n'.join(doc)+'\n')


def launch(root):
    prepare(root)
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_MODE='disabled')
    state = dict(protocol=PROTOCOL, state='running', stage='normalization_and_training', started=time.time(), pid=os.getpid())
    dump_json(root/'status.json', state)
    try:
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
             '-m','analysis.fig01b_hidden_prediction.initial_token_prediction_probe','train','--output',str(root)]
        with (root/'train.log').open('a') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['stage']='evaluation'; dump_json(root/'status.json',state)
        procs=[]
        try:
            for shard in range(8):
                log=(root/f'eval_{shard}.log').open('w')
                p=subprocess.Popen([sys.executable,'-u','-m','analysis.fig01b_hidden_prediction.initial_token_prediction_probe','eval','--output',str(root),'--shard',str(shard)],
                                   cwd=ROOT, env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
                procs.append((p,log))
            while any(p.poll() is None for p,_ in procs):
                if any(p.poll() not in (None,0) for p,_ in procs):
                    raise RuntimeError('An evaluation shard failed')
                time.sleep(5)
        finally:
            for p,log in procs:
                if p.poll() is None:p.terminate()
                log.close()
        report(root)
        state.update(state='complete',stage='complete',finished=time.time())
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),finished=time.time())
        raise
    finally:
        dump_json(root/'status.json',state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','eval','report','launch'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.action=='eval':evaluate(args.output,args.shard)
    else:globals()[args.action](args.output)
