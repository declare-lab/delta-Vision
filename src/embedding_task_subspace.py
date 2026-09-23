"""Frozen full embedding-adapter task-subspace diagnostic on MMStar.

Native input-hidden gradients, shared feature bases, and one-layer native
suffix interventions. Same-set, label-informed diagnostic; not generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
CHECKPOINT = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
REFERENCE_SRC = Path('/lustre-data/leijingdi/code/vision-kv-inject-attention-sink/src')
DATA = ROOT/'data/benchmarks/mmstar/mmstar_val.jsonl'
LAYERS = (33, 34, 35)
RANKS = (16, 32, 64, 128, 256)
CAUSAL_RANKS = (64, 128)
EPS = (.01, .05, .1)
SEEDS = (44, 45, 46)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def reference_module():
    # Import the exact maintained adapter path used by the selected checkpoint,
    # without changing this repository's imports or modifying either worktree.
    package = types.ModuleType('_embedding_reference')
    package.__path__ = [str(REFERENCE_SRC)]
    sys.modules.setdefault('_embedding_reference', package)
    return importlib.import_module('_embedding_reference.model')


class Stats:
    """Additive pooled feature metrics, also valid for ambient null coordinates."""
    def __init__(self, width, intrinsic=None):
        self.n = self.images = self.valid = 0
        self.sse = self.y2 = self.cos = 0.
        self.sy = torch.zeros(width, dtype=torch.float64)
        self.dim = width if intrinsic is None else intrinsic

    def add(self, p2, y2, dot, e2, sy):
        self.n += len(y2); self.images += 1
        self.sse += float(e2.sum()); self.y2 += float(y2.sum())
        self.sy += sy.detach().double().cpu()
        denominator = (p2.clamp_min(0)*y2.clamp_min(0)).sqrt()
        valid = denominator > 1e-20
        self.valid += int(valid.sum())
        self.cos += float((dot[valid]/denominator[valid]).clamp(-1, 1).sum())

    def add_vectors(self, p, y):
        p, y = p.double(), y.double()
        self.add(p.square().sum(-1), y.square().sum(-1), (p*y).sum(-1),
                 (p-y).square().sum(-1), y.sum(0))

    def state(self):
        return dict(n=self.n, images=self.images, valid=self.valid, sse=self.sse,
                    y2=self.y2, cos=self.cos, sy=self.sy.tolist(), dim=self.dim)

    @staticmethod
    def merge(states):
        n = sum(s['n'] for s in states)
        sy = np.sum([s['sy'] for s in states], axis=0)
        sse = sum(s['sse'] for s in states); y2 = sum(s['y2'] for s in states)
        sst = max(0., y2-float(sy@sy)/n)
        valid = sum(s['valid'] for s in states)
        return dict(images=sum(s['images'] for s in states), visual_tokens=n,
                    dimension=states[0]['dim'], sse=sse, teacher_energy=y2,
                    mse=sse/(n*states[0]['dim']), r2=1-sse/sst if sst>1e-20 else None,
                    cosine=sum(s['cos'] for s in states)/valid if valid else None,
                    relative_error=math.sqrt(sse/y2) if y2>1e-20 else None,
                    centered_teacher_sst=sst, valid_cosine_tokens=valid)


def projection_stats(p, y, basis, rank):
    """FP64 task and complement sufficient statistics, no dense projector."""
    p, y, b = p.double(), y.double(), basis[:, :rank].double()
    pp, yp = p@b, y@b
    # Explicit residual is used for stability; only N x D, never D x D.
    pn, yn = p-pp@b.T, y-yp@b.T
    return (pp, yp), (pn, yn)


def noise_pair(hidden, basis, seed):
    """Each token: orthogonal directions with norm equal to teacher token norm."""
    generator = torch.Generator(device=hidden.device).manual_seed(seed)
    h = hidden.float()
    z = torch.randn(h.shape, device=h.device, generator=generator)
    b = basis.float()
    task = (z@b)@b.T
    complement = z-task
    norm = h.norm(dim=-1, keepdim=True)
    task = task/task.norm(dim=-1, keepdim=True).clamp_min(1e-20)*norm
    complement = complement/complement.norm(dim=-1, keepdim=True).clamp_min(1e-20)*norm
    return task, complement


class Capture:
    def __init__(self, model):
        self.layers = model.model.language_model.layers
        self.enabled = True; self.grad = False
        self.h = {}; self.kw = {}; self.initial = None
        self.patch_layer = None; self.patch = None; self.positions = None
        self.handles = [layer.register_forward_pre_hook(self.hook(l), with_kwargs=True)
                        for l, layer in enumerate(self.layers) if l in (0, *LAYERS)]

    def reset(self, positions, grad=False):
        self.enabled = True; self.grad = grad; self.positions = positions
        self.h.clear(); self.kw.clear(); self.initial = None
        self.patch_layer = None; self.patch = None

    def hook(self, layer):
        def run(module, args, kw):
            if not self.enabled:
                return
            h = kw.get('hidden_states', args[0] if args else None)
            if layer == 0:
                self.initial = h.detach().clone()
                return
            if self.grad and layer == min(LAYERS):
                h = h.detach().requires_grad_(True)
            if self.grad:
                assert h.requires_grad
                h.retain_grad()
            self.h[layer] = h
            self.kw[layer] = {k: v for k, v in kw.items() if k != 'hidden_states'}
            if self.patch_layer == layer:
                h = h.clone()
                h[:, self.positions] = (h[:, self.positions].float()+self.patch).to(h.dtype)
            if args:
                return (h,)+args[1:], kw
            return args, dict(kw, hidden_states=h)
        return run


def setup(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    plan = json.loads((Path(args.output)/'plan.json').read_text())
    assert sha(__file__) == plan['source_sha256'], 'Diagnostic code changed after launch'
    assert sha(REFERENCE_SRC/'model.py') == plan['reference_model_source_sha256']
    assert sha(CHECKPOINT) == plan['checkpoint_sha256'], 'Checkpoint changed after launch'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(44)
    processor, model = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device('cuda:0'), 'flash_attention_2')
    ref = reference_module()
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT, model.model.language_model,
                                                              torch.device('cuda:0'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected'], meta
    assert adapter.visual_adapter_rank == 128 and adapter.mode == 'embedding_adapter'
    assert adapter.adapter_start_layer == 0 and adapter.active_adapter_layers == 0
    assert adapter.adapter_attention_backend == 'efficient'
    assert len(model.model.language_model.layers) == 36
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(p.requires_grad for p in adapter.parameters())
    dataset = QwenBenchmarkDataset(str(DATA), processor, 'mmstar', max_samples=args.samples)
    assert len(dataset) == args.samples
    assert hashlib.sha256(json.dumps(dataset.rows, sort_keys=True).encode()).hexdigest() == plan['selected_rows_sha256']
    ids = {letter: processor.tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCD'}
    assert all(len(x)==1 for x in ids.values()), ids
    # Native visual features are constant within an image across evaluation variants.
    image_cache = {}
    original_features = model.model.get_image_features
    def features(*a, **kw):
        if 'features' not in image_cache:
            image_cache['features'] = original_features(*a, **kw)
        return image_cache['features']
    model.model.get_image_features = features
    return processor, model, adapter, ref, dataset, ids, image_cache


def native_forward(model, inputs):
    model.model.rope_deltas = None
    return model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1].float()


def inputs_for(item):
    from src.visual_cross_token_ablation import prepare
    return prepare(item, torch.device('cuda:0'))


def collect(args):
    processor, model, adapter, ref, data, ids, image_cache = setup(args)
    root = Path(args.output); cap = Capture(model)
    cov = {l: torch.zeros((2560, 2560), device='cuda', dtype=torch.float64) for l in LAYERS}
    counts = {l: 0 for l in LAYERS}
    start = time.time()
    with (root/f'collect_{args.shard}.jsonl').open('w') as f:
        for number, i in enumerate(range(args.shard, args.samples, args.world)):
            image_cache.clear(); item = data[i]; inputs = inputs_for(item)
            positions = (inputs['mm_token_type_ids'][0] == 1).nonzero().flatten()
            gold = str(item['answer']).strip().upper(); assert gold in ids, gold
            cap.reset(positions, grad=True)
            with torch.enable_grad():
                logits = native_forward(model, inputs)
                loss = F.cross_entropy(logits[None], torch.tensor([ids[gold][0]], device='cuda'))
                loss.backward()
            norms = {}
            for l in LAYERS:
                g = cap.h[l].grad[0, positions].double()
                assert torch.isfinite(g).all() and float(g.norm()) > 0
                cov[l].addmm_(g.T, g)
                counts[l] += len(g); norms[l] = float(g.norm())
            assert all(p.grad is None for p in model.parameters())
            f.write(json.dumps(dict(sample=i, gold=gold, ce=float(loss.detach()), visual_tokens=len(positions),
                                    gradient_norm=norms, first_token=int(logits.argmax())))+'\n'); f.flush()
            cap.h.clear(); cap.kw.clear(); cap.initial=None
            del logits, loss, g
            if number % 10 == 0:
                print('collect', args.shard, number+1, 'sample', i, 'seconds', round(time.time()-start,1), flush=True)
    torch.save(dict(cov={l: x.cpu() for l,x in cov.items()}, counts=counts), root/f'cov_{args.shard}.pt')
    dump(root/f'collect_{args.shard}.done.json', dict(seconds=time.time()-start))


def basis(args):
    root = Path(args.output)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    sums = {}; counts = {l:0 for l in LAYERS}
    for shard in range(args.world):
        data = torch.load(root/f'cov_{shard}.pt', weights_only=False, map_location='cpu')
        for l in LAYERS:
            sums[l] = sums.get(l, 0)+data['cov'][l]
            counts[l] += data['counts'][l]
    bases = {}; random = {}; spectra = {}
    for l in LAYERS:
        matrix = sums[l].cuda()
        eigenvalues, vectors = torch.linalg.eigh((matrix+matrix.T)*.5)
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        b = vectors[:, -max(RANKS):].flip(1)
        torch.testing.assert_close(b.T@b, torch.eye(max(RANKS),device='cuda',dtype=torch.float64),atol=1e-9,rtol=1e-9)
        bases[l] = b.cpu()
        generator = torch.Generator().manual_seed(4400+l)
        random[l] = torch.linalg.qr(torch.randn(2560,max(RANKS),generator=generator,dtype=torch.float64),mode='reduced').Q
        energy = eigenvalues.cumsum(0)/eigenvalues.sum()
        spectra[l] = dict(visual_tokens=counts[l], eigenvalues=eigenvalues.cpu().tolist(),
                          singular_values=eigenvalues.sqrt().cpu().tolist(), cumulative_energy=energy.cpu().tolist(),
                          r90=int(torch.searchsorted(energy,.9))+1, r95=int(torch.searchsorted(energy,.95))+1,
                          energy={str(r):float(energy[r-1]) for r in RANKS})
        print('basis', l, spectra[l]['energy'], flush=True)
    torch.save(dict(task=bases,random=random),root/'basis.pt')
    dump(root/'spectrum.json',spectra)


def suffix_logits(model, cap, layer, delta=None):
    """Prefix is native and reused ONLY for a single-layer intervention."""
    h = cap.h[layer].detach()
    if delta is not None:
        h = h.clone()
        h[:, cap.positions] = (h[:,cap.positions].float()+delta).to(h.dtype)
    cap.enabled = False
    for l in range(layer,36):
        h = cap.layers[l](h, **cap.kw[l])
    return model.lm_head(model.model.language_model.norm(h)[:, -1:])[0,-1].float()


def decode(first_logits, processor, continuation):
    from src.eval_benchmarks import extract_option_from_text
    tokens = []
    logits = first_logits
    for _ in range(8):
        token = int(logits.argmax()); tokens.append(token)
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        option = extract_option_from_text(text)
        if option in ('A', 'B', 'C', 'D'):
            return text, option
        if token == processor.tokenizer.eos_token_id or len(tokens)==8:
            return text, None
        logits = continuation(tokens)
    raise AssertionError('unreachable')


def extended(inputs, tokens):
    new = torch.tensor([tokens],device=inputs['input_ids'].device)
    return dict(inputs,input_ids=torch.cat((inputs['input_ids'],new),1),
                attention_mask=torch.cat((inputs['attention_mask'],torch.ones_like(new)),1),
                mm_token_type_ids=torch.cat((inputs['mm_token_type_ids'],torch.zeros_like(new)),1))


def evaluate(args):
    from src.model import qwen_position_ids
    processor, model, adapter, ref, data, ids, image_cache = setup(args)
    root = Path(args.output); cap = Capture(model)
    payload = torch.load(root/'basis.pt',weights_only=False,map_location='cuda')
    stats = {}; start=time.time()
    def add(key,p,y):
        if key not in stats: stats[key]=Stats(y.shape[-1])
        stats[key].add_vectors(p,y)
    with torch.no_grad(), (root/f'eval_{args.shard}.jsonl').open('w') as f:
        for number,i in enumerate(range(args.shard,args.samples,args.world)):
            image_cache.clear(); item=data[i]; inputs=inputs_for(item)
            positions=(inputs['mm_token_type_ids'][0]==1).nonzero().flatten()
            gold=str(item['answer']).strip().upper(); gold_id=ids[gold][0]
            cap.reset(positions)
            native=native_forward(model,inputs)
            cap.enabled=False
            saved_h=dict(cap.h); saved_kw=dict(cap.kw); initial=cap.initial
            # Native-prefix suffix reuse must preserve the entire logit vector.
            identity_errors={}
            for l in LAYERS:
                reconstructed=suffix_logits(model,cap,l)
                err=float((native-reconstructed).abs().max()); identity_errors[l]=err
                torch.testing.assert_close(reconstructed,native,rtol=0,atol=0)
            memories=adapter.all_visual_memories_batched(initial[:,positions])
            for l in LAYERS:
                teacher=saved_h[l][0,positions].double(); pred=memories[l,0].double()
                add(f'{l}/raw',pred,teacher)
                for kind in ('task','random'):
                    for r in RANKS:
                        projected,complement=projection_stats(pred,teacher,payload[kind][l],r)
                        add(f'{l}/{kind}/{r}/projected',*projected)
                        key=f'{l}/{kind}/{r}/complement'
                        if key not in stats:stats[key]=Stats(2560,intrinsic=2560-r)
                        stats[key].add_vectors(*complement)
            logpt=native.log_softmax(-1); pt=logpt.exp()
            native_ce=float(-logpt[gold_id])
            def native_continue(tokens):
                cap.enabled=False
                return native_forward(model,extended(inputs,tokens))
            native_text,native_option=decode(native,processor,native_continue)
            pos_ids=qwen_position_ids(model,inputs)
            full_logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,
                                                         position_ids=pos_ids,logits_to_keep=1)[0][0,-1].float()
            def adapter_continue(tokens):
                inp=extended(inputs,tokens); model.model.rope_deltas=None
                # Append text embeddings only; the original image embedding is identical.
                ext_h=torch.cat((initial,model.model.get_input_embeddings()(inp['input_ids'][:,initial.shape[1]:])),1)
                return ref.qwen_embedding_adapter_logits(model,adapter,inp,initial_hidden=ext_h,
                         position_ids=qwen_position_ids(model,inp),logits_to_keep=1)[0][0,-1].float()
            adapter_text,adapter_option=decode(full_logits,processor,adapter_continue)
            a_logp=full_logits.log_softmax(-1)
            row=dict(sample=i,gold=gold,native=dict(text=native_text,score=int(native_option==gold),ce=native_ce),
                     adapter=dict(text=adapter_text,score=int(adapter_option==gold),ce=float(-a_logp[gold_id]),
                                  kl=float((pt*(logpt-a_logp)).sum().clamp_min(0))),
                     suffix_identity_max_logit_error=identity_errors, perturbations={})
            for l in LAYERS:
                for r in CAUSAL_RANKS:
                    for seed in SEEDS:
                        noise=noise_pair(saved_h[l][0,positions],payload['task'][l][:,:r],seed+i*101+l*100003)
                        for epsilon in EPS:
                            for space, direction in zip(('task','complement'),noise):
                                delta=epsilon*direction
                                cap.h=dict(saved_h);cap.kw=dict(saved_kw);cap.positions=positions
                                logits=suffix_logits(model,cap,l,delta)
                                lp=logits.log_softmax(-1)
                                def continuation(tokens):
                                    inp=extended(inputs,tokens)
                                    cap.reset(positions); cap.patch_layer=l;cap.patch=delta
                                    out=native_forward(model,inp);cap.enabled=False
                                    return out
                                text,option=decode(logits,processor,continuation)
                                h=saved_h[l][0,positions]
                                actual=(h.float()+delta).to(h.dtype).float()-h.float()
                                norms=actual.norm(dim=-1)/(epsilon*h.float().norm(dim=-1)).clamp_min(1e-20)
                                key=f'{l}/r{r}/eps{epsilon}/seed{seed}/{space}'
                                row['perturbations'][key]=dict(text=text,score=int(option==gold),
                                    ce=float(-lp[gold_id]),kl=float((pt*(logpt-lp)).sum().clamp_min(0)),
                                    actual_norm_ratio_mean=float(norms.mean()),
                                    actual_norm_ratio_min=float(norms.min()),actual_norm_ratio_max=float(norms.max()))
            f.write(json.dumps(row)+'\n');f.flush()
            cap.h.clear();cap.kw.clear();cap.initial=None
            if number%5==0:
                print('eval',args.shard,number+1,'sample',i,'seconds',round(time.time()-start,1),flush=True)
    torch.save({k:v.state() for k,v in stats.items()},root/f'metrics_{args.shard}.pt')
    dump(root/f'eval_{args.shard}.done.json',dict(seconds=time.time()-start))


def merge(args):
    root=Path(args.output)
    rows=[json.loads(line) for s in range(args.world) for line in (root/f'eval_{s}.jsonl').open()]
    assert len(rows)==args.samples and {r['sample'] for r in rows}==set(range(args.samples))
    rows.sort(key=lambda x:x['sample'])
    count=len(LAYERS)*len(CAUSAL_RANKS)*len(EPS)*len(SEEDS)*2
    assert all(len(r['perturbations'])==count for r in rows)
    assert all(v==0 for r in rows for v in r['suffix_identity_max_logit_error'].values())
    shards=[torch.load(root/f'metrics_{s}.pt',weights_only=False) for s in range(args.world)]
    assert all(set(s)==set(shards[0]) for s in shards)
    metrics={k:Stats.merge([s[k] for s in shards]) for k in shards[0]}
    assert all(v['images']==args.samples for v in metrics.values())
    for key,value in metrics.items():
        value['error_energy_fraction']=value['sse']/metrics[key.split('/')[0]+'/raw']['sse']
    baselines={kind:dict(accuracy_pct=100*np.mean([r[kind]['score'] for r in rows]),
                        ce=np.mean([r[kind]['ce'] for r in rows])) for kind in ('native','adapter')}
    baselines['adapter']['output_kl']=np.mean([r['adapter']['kl'] for r in rows])
    causal={}
    for l in LAYERS:
        for rank in CAUSAL_RANKS:
            for epsilon in EPS:
                for space in ('task','complement'):
                    keys=[f'{l}/r{rank}/eps{epsilon}/seed{s}/{space}' for s in SEEDS]
                    selected=[[r['perturbations'][k] for k in keys] for r in rows]
                    score=np.array([[x['score'] for x in ss] for ss in selected])
                    ref=np.array([r['native']['score'] for r in rows])
                    diff=ref-score.mean(1)
                    rng=np.random.default_rng(44)
                    boot=np.array([diff[rng.integers(0,len(rows),len(rows))].mean()*100 for _ in range(2000)])
                    key=f'{l}/r{rank}/eps{epsilon}/{space}'
                    causal[key]=dict(images=len(rows),seeds=len(SEEDS),accuracy_pct=float(score.mean()*100),
                        accuracy_seed_std_pp=float(score.mean(0).std()*100),drop_pp=float(diff.mean()*100),
                        drop_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
                        kl=float(np.mean([[x['kl'] for x in ss] for ss in selected])),
                        ce=float(np.mean([[x['ce'] for x in ss] for ss in selected])),
                        actual_norm_ratio_mean=float(np.mean([[x['actual_norm_ratio_mean'] for x in ss] for ss in selected])))
    dump(root/'results.json',dict(samples=len(rows),baselines=baselines,similarity=metrics,causal=causal))
    report(root)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec=json.loads((root/'spectrum.json').read_text())
    result=json.loads((root/'results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for l in LAYERS:
        x=spec[str(l)]
        axes[0].semilogy(np.arange(1,2561),np.maximum(x['singular_values'],1e-30),label=f'Layer {l}')
        axes[1].plot(np.arange(1,257),x['cumulative_energy'][:256],label=f'Layer {l}')
    axes[0].set(xlabel='Rank',ylabel='Gradient singular value')
    axes[1].set(xlabel='Rank',ylabel='Cumulative gradient energy',ylim=(0,1.01))
    for a in axes:a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'gradient_spectrum.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,l in enumerate(LAYERS):
        m=result['similarity'];raw=m[f'{l}/raw']
        for label,key in [('Task','task'),('Random','random')]:
            for row,metric in enumerate(('r2','relative_error')):
                axes[row,col].plot(RANKS,[m[f'{l}/{key}/{r}/projected'][metric] for r in RANKS],marker='o',label=label)
                if label=='Task':
                    axes[row,col].plot(RANKS,[m[f'{l}/task/{r}/complement'][metric] for r in RANKS],marker='s',label='Complement')
                    axes[row,col].axhline(raw[metric],color='k',linestyle='--',label='Raw')
                axes[row,col].set(xlabel='Rank',ylabel=metric,title=f'Layer {l}')
        for row in range(2):axes[row,col].legend();axes[row,col].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'subspace_similarity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,l in enumerate(LAYERS):
        for r in CAUSAL_RANKS:
            for space in ('task','complement'):
                vals=[result['causal'][f'{l}/r{r}/eps{e}/{space}'] for e in EPS]
                for row,metric in enumerate(('drop_pp','kl')):
                    axes[row,col].plot(EPS,[v[metric] for v in vals],marker='o',label=f'{space} r{r}')
                    axes[row,col].set(xlabel='Relative per-token noise norm',ylabel=metric,title=f'Layer {l}')
        for row in range(2):axes[row,col].legend();axes[row,col].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'causal_perturbations.png',dpi=160);plt.close(fig)
    b=result['baselines']
    lines=['# 完整 embedding adapter：Task-subspace 诊断','',
        f'MMStar {result["samples"]} 条；0-based 层 33/34/35；原生 hidden width 2560；不训练。','',
        '**同集合、使用答案标签的诊断**：这批 MMStar 的真实选项 CE 梯度建立基底，也在这批数据上分析；不是泛化证据。',
        '补空间不预先称为任务无关空间。此前单层 MLP 的 R² 和准确率不能移用于本实验。','',
        f'Checkpoint: `{CHECKPOINT}`','',
        f'原模型 accuracy={b["native"]["accuracy_pct"]:.4f}%；完整 adapter={b["adapter"]["accuracy_pct"]:.4f}%；adapter 输出 KL={b["adapter"]["output_kl"]:.6g}。','',
        '## 1. 梯度谱','', '| Layer | r90 | r95 | E16 | E32 | E64 | E128 | E256 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for l in LAYERS:
        x=spec[str(l)];lines.append(f'| {l} | {x["r90"]} | {x["r95"]} | '+' | '.join(f'{x["energy"][str(r)]:.5f}' for r in RANKS)+' |')
    lines+=['','![梯度谱](gradient_spectrum.png)','','## 2. 原生与 adapter memory 拟合','',
            '| Layer | Basis | Rank | Space | R² | Cosine | MSE | Relative error | Error energy fraction |',
            '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for k,v in result['similarity'].items():
        parts=k.split('/');l=parts[0]
        basis,rank,space=('—','2560','raw') if len(parts)==2 else parts[1:]
        lines.append(f'| {l} | {basis} | {rank} | {space} | {v["r2"]:.6f} | {v["cosine"]:.6f} | {v["mse"]:.6g} | {v["relative_error"]:.6f} | {v["error_energy_fraction"]:.6f} |')
    lines+=['','![拟合](subspace_similarity.png)','','## 3. 等范数扰动','',
            '逐层单点干预原模型。准确率、KL 为三个种子平均；置信区间按样本配对 bootstrap，先在样本内平均种子。正 drop 表示掉分。','',
            '| Layer | Rank | Epsilon | Space | Accuracy % | Drop pp | Drop 95% CI | Full-vocab KL | Answer CE | Actual/requested norm |',
            '|---|---|---|---|---:|---:|---|---:|---:|---:|']
    for k,v in result['causal'].items():
        l,r,e,space=k.split('/');ci=v['drop_ci95_pp']
        lines.append(f'| {l} | {r} | {e} | {space} | {v["accuracy_pct"]:.4f} | {v["drop_pp"]:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {v["kl"]:.6g} | {v["ce"]:.6g} | {v["actual_norm_ratio_mean"]:.4f} |')
    lines+=['','![扰动](causal_perturbations.png)','','## 定义与核查','',
        '- Native H：目标层 attention RMSNorm 前的视觉 hidden；完整原生前缀保留，DeepStack 关闭。',
        '- Predicted H：确认的完整 adapter 以初始 merged embedding E 为锚点生成的对应层 memory；使用原 checkpoint 的参考实现，不是单层 MSE predictor。',
        '- 梯度：原模型对正确 A/B/C/D 单 token 的 full-vocab CE，在提示最后位置求导；不改变参数。各层梯度为所有视觉位置未经中心化的二阶矩。',
        '- R²：所有样本/视觉 token 汇总的每通道中心化 SST；cosine 逐 token 平均。relative error = Frobenius error / teacher norm。补空间 MSE 除以其内在维度 D-r。',
        '- 随机基底为独立固定种子的正交基底，相同 rank；补空间统计不是把未解释梯度方向预先判为无用。',
        '- 扰动在 FP32 中逐 token 匹配原生 hidden 范数，再乘 epsilon；task 与补空间来自同一噪声。BF16 注入后的实际范数另行记录，包含量化偏差。',
        '- 仅复用未干预的原生前缀；从目标层开始执行完整原生 suffix。三个 suffix 的无扰动重放均要求完整 logits bitwise 一致。',
        '- Output KL/CE：第一答案 token 的完整词表分布。Accuracy：greedy 最多 8 token，选项/EOS 早停；必要时继续执行同一位置干预。',
        '- Native FlashAttention2/BF16；完整 adapter 保持该 checkpoint 的 efficient SDPA 路径；投影和统计使用 FP64，关闭 TF32。',
        '- 不从“不显著掉分”推出等价；没有预设 task 投影必须比 raw 或随机基底拟合得好。','',
        '[Plan](plan.json) · [Results](results.json) · [Status](status.json)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def launch(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'plan.json').exists(), 'Choose a fresh output directory; never overwrite prior results'
    rows=[json.loads(line) for line in DATA.open()][:args.samples]
    assert len(rows)==args.samples
    plan=dict(model=MODEL,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
              data=str(DATA),samples=args.samples,selected_rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
              layers=LAYERS,ranks=RANKS,causal_ranks=CAUSAL_RANKS,epsilons=EPS,seeds=SEEDS,world=args.world,
              source_sha256=sha(__file__),reference_model_source_sha256=sha(REFERENCE_SRC/'model.py'),
              same_set_label_informed=True,training=False,hidden_width=2560,
              answer_loss='full-vocabulary CE of true single-token A/B/C/D at first answer position',
              experiment='full embedding adapter, native hidden input layers 33/34/35; one-layer perturbations')
    dump(root/'plan.json',plan)
    started=time.time()
    for phase in ('collect','basis','eval','merge'):
        dump(root/'status.json',dict(state='running',phase=phase,elapsed_seconds=time.time()-started))
        if phase in ('basis','merge'):
            getattr(sys.modules[__name__],phase)(args)
            continue
        procs=[];logs=[]
        for shard in range(args.world):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            log=(root/f'{phase}_gpu{shard}.log').open('w');logs.append(log)
            command=[sys.executable,'-u','-m','src.embedding_task_subspace',phase,'--output',str(root),
                     '--samples',str(args.samples),'--world',str(args.world),'--shard',str(shard)]
            procs.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        try:
            while any(p.poll() is None for p in procs):
                failures=[(s,p.returncode) for s,p in enumerate(procs) if p.poll() not in (None,0)]
                if failures:raise RuntimeError(f'{phase} workers failed: {failures}')
                time.sleep(3)
            assert all(p.returncode==0 for p in procs)
        except BaseException as exc:
            for p in procs:
                if p.poll() is None:p.terminate()
            for p in procs:p.wait()
            dump(root/'status.json',dict(state='failed',phase=phase,error=str(exc)))
            raise
        finally:
            for log in logs:log.close()
    dump(root/'status.json',dict(state='complete',samples=args.samples,layers=LAYERS,elapsed_seconds=time.time()-started))
    print('COMPLETE',root/'README.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['launch','collect','basis','eval','merge'])
    parser.add_argument('--output',default=str(ROOT/'artifacts/diagnostics/embedding_task_subspace_mmstar1500_20260912'))
    parser.add_argument('--samples',type=int,default=1500)
    parser.add_argument('--world',type=int,default=8)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    globals()[args.command if args.command!='eval' else 'evaluate'](args)
