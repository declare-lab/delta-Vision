"""Native Qwen3-VL visual-edge ablations, with unchanged softmax denominator.

Single-layer interventions only. Full native prefill components can therefore
be reused at the intervention layer after checking its native input is identical.
No training, token removal, renormalization, or intervention on text query rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'


def decompose(q, k, v, positions, scale, chunk=128):
    """[H,T,D] -> visual self/cross/text [H,Nv,D], original causal weights.

    Single unpadded sequence; caller repeats GQA keys/values. All valid keys
    remain in the denominator. Return FP32 (or FP64 for a FP64 input).
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    keys = torch.arange(k.shape[1], device=k.device)
    visual = torch.zeros(k.shape[1], dtype=torch.bool, device=k.device)
    visual[positions] = True
    parts = [[], [], []]
    masses = []
    for p in positions.split(chunk):
        logits = (q[:, p] @ k.transpose(-1, -2)) * scale
        logits.masked_fill_(keys[None, None, :] > p[None, :, None], -torch.inf)
        a = logits.softmax(-1)
        diag = a.gather(-1, p[None, :, None].expand(q.shape[0], -1, 1))
        own = diag * v[:, p]
        cross_a = a * visual[None, None, :]
        cross_a.scatter_(-1, p[None, :, None].expand(q.shape[0], -1, 1), 0.)
        cross = cross_a @ v
        text_a = a * (~visual)[None, None, :]
        text = text_a @ v
        for dest, x in zip(parts, (own, cross, text)):
            dest.append(x)
        masses.append(torch.stack((diag.squeeze(-1), cross_a.sum(-1), text_a.sum(-1)), -1))
    return *(torch.cat(xs, 1) for xs in parts), torch.cat(masses, 1)


def flatten_heads(x):
    return x.transpose(0, 1).reshape(x.shape[1], -1)


def ratio(a, b):
    denom = torch.linalg.vector_norm(b.float())
    return float(torch.linalg.vector_norm(a.float()) / denom) if float(denom) > 1e-12 else None


class VisualIntervention:
    def __init__(self, model):
        self.layers = model.model.language_model.layers
        self.enabled = True
        self.mode = 'full'
        self.target = -1
        self.positions = None
        self.pending = {}
        self.cache = {}
        self.metrics = {}
        self.changed = []
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self.capture(li), with_kwargs=True))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self.replace(li)))

    def reset(self, positions):
        self.positions = positions
        self.pending.clear()
        self.cache.clear()
        self.metrics.clear()
        self.set_mode('full', -1)

    def set_mode(self, mode, target):
        self.mode, self.target = mode, target
        self.changed = []

    def capture(self, li):
        def hook(module, args, kwargs):
            if not self.enabled or self.mode != 'full':
                return
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if hidden.shape[1] == 1:
                return  # native generated-text decode, never intervene
            assert hidden.shape[0] == 1
            assert li not in self.cache, 'Expected one prefill per generation'
            mask = kwargs.get('attention_mask')
            assert mask is None or (mask.ndim == 2 and bool(mask.all())), 'Only unpadded native causal supported'
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            v = module.v_proj(hidden).view(shape).transpose(1, 2)
            from src.model import qwen_apply_rotary_pos_emb
            q, k = qwen_apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            groups = q.shape[1] // k.shape[1]
            self.pending[li] = decompose(q[0], k[0].repeat_interleave(groups, 0),
                                         v[0].repeat_interleave(groups, 0), self.positions,
                                         float(module.scaling))
        return hook

    def replace(self, li):
        def hook(module, args):
            native = args[0]
            if not self.enabled or native.shape[1] == 1:
                return
            p = self.positions
            if self.mode == 'full':
                own, cross, text, masses = self.pending.pop(li)
                vv = own + cross
                all_parts = vv + text
                flat = [flatten_heads(x) for x in (own, cross, text, vv, all_parts)]
                wo_cross = F.linear(flat[1], module.weight.float(), None)
                wo_vv = F.linear(flat[3], module.weight.float(), None)
                wo_all = F.linear(flat[4], module.weight.float(), None)
                token_denom = torch.linalg.vector_norm(vv, dim=-1)
                valid = token_denom > 1e-12
                token_ratios = torch.linalg.vector_norm(cross, dim=-1)[valid] / token_denom[valid]
                native_visual = native[0, p].float()
                err = ratio(flat[4] - native_visual, native_visual)
                assert err is not None and err < .025, ('Native reconstruction discrepancy', li, err)
                self.metrics[li] = {
                    'layer': li, 'visual_tokens': len(p),
                    'cross_over_visual_fro': ratio(cross, vv),
                    'cross_over_total_fro': ratio(cross, all_parts),
                    'wo_cross_over_visual_fro': ratio(wo_cross, wo_vv),
                    'wo_cross_over_total_fro': ratio(wo_cross, wo_all),
                    'head_token_ratio_mean': float(token_ratios.mean()) if len(token_ratios) else None,
                    'head_token_ratio_p95': float(token_ratios.quantile(.95)) if len(token_ratios) else None,
                    'near_zero_visual_head_tokens': int((~valid).sum()),
                    'self_mass': float(masses[..., 0].mean()),
                    'cross_mass': float(masses[..., 1].mean()),
                    'text_mass': float(masses[..., 2].mean()),
                    'reconstruction_relative_error': err,
                    'cross_norm': float(cross.norm()), 'visual_norm': float(vv.norm()),
                    'total_norm': float(all_parts.norm()),
                    'per_head': [{'head': h, 'cross_over_visual_fro': ratio(cross[h], vv[h]),
                                  'cross_over_total_fro': ratio(cross[h], all_parts[h])}
                                 for h in range(cross.shape[0])],
                }
                self.cache[li] = {
                    'native': native.clone(),
                    'self_only': (flat[0] + flat[2]).to(native.dtype),
                    'cross_only': (flat[1] + flat[2]).to(native.dtype),
                    'reconstructed': flat[4].to(native.dtype),
                }
                return  # Full is bitwise native
            if li != self.target:
                return
            cached = self.cache[li]
            # This cache reuse is valid only for ONE intervened layer per run.
            assert torch.equal(native, cached['native']), ('Upstream native path changed', li)
            replacement = native.clone()
            replacement[0, p] = cached[self.mode]
            text_mask = torch.ones(native.shape[1], dtype=torch.bool, device=native.device)
            text_mask[p] = False
            assert torch.equal(replacement[:, text_mask], native[:, text_mask])
            self.changed.append(li)
            return (replacement,) + args[1:]
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def prepare(item, device):
    result = {}
    for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids'):
        x = item[key]
        if key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
            x = x.unsqueeze(0)
        result[key] = x.to(device=device, dtype=torch.bfloat16 if x.is_floating_point() else x.dtype)
    assert bool(result['attention_mask'].all())
    return result


def run(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    from src.evaluate import generate_teacher_qwen
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f'shard{args.shard}.jsonl'
    finished = set()
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(result_path)
        for line in result_path.open():
            r = json.loads(line)
            finished.add((r['sample_position'], r['condition']))
    spec = get_benchmark_spec('realworldqa')
    processor, model = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device('cuda:0'), 'flash_attention_2')
    ds = QwenBenchmarkDataset(str(ROOT / spec.default_data), processor, 'realworldqa',
                              data_root=str((ROOT / spec.default_data).parent), max_samples=args.limit or None)
    selection_hash = hashlib.sha256(json.dumps(ds.rows, sort_keys=True).encode()).hexdigest()
    intervene = VisualIntervention(model)
    target_layers = list(range(len(intervene.layers))) if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
    conditions = [('full', -1)] + [(mode, li) for li in target_layers for mode in ('reconstructed', 'self_only', 'cross_only')]
    # Native image encoder features (DeepStack disabled) are identical for all
    # single-layer variants of an image. Cache the unchanged native output.
    original_features = model.model.get_image_features
    image_cache = []
    def get_cached_features(*a, **kw):
        if not image_cache:
            image_cache.append(original_features(*a, **kw))
        return image_cache[0]
    model.model.get_image_features = get_cached_features
    start = time.time()
    checked = False
    processed = 0
    with torch.inference_mode(), result_path.open('a') as out:
        for index in range(args.shard, len(ds), args.shards):
            names = [f'{mode}:{li}' for mode, li in conditions]
            if all((index, name) in finished for name in names):
                continue
            item = ds[index]
            inputs = prepare(item, torch.device('cuda:0'))
            p = (inputs['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
            assert len(p) > 0
            intervene.reset(p)
            image_cache.clear()
            def generate():
                _, text = generate_teacher_qwen(model, processor, **inputs, max_new_tokens=args.max_new_tokens)
                return text
            plain = None
            if not checked:
                intervene.enabled = False
                plain = generate()
                # Recompute the vision path for the observer self-check too.
                image_cache.clear()
                intervene.enabled = True
            for mode, li in conditions:
                name = f'{mode}:{li}'
                if mode != 'full' and (index, name) in finished:
                    continue
                intervene.set_mode(mode, li)
                t0 = time.time()
                prediction = generate()
                if mode == 'full':
                    assert len(intervene.cache) == len(intervene.layers)
                    if plain is not None:
                        assert prediction == plain, ('Observation changed native generated answer', plain, prediction)
                        checked = True
                        print(f'PASS native observer/cache self-check shard={args.shard}', flush=True)
                    baseline_prediction = prediction
                else:
                    assert intervene.changed == [li], intervene.changed
                    if li == len(intervene.layers)-1:
                        assert prediction == baseline_prediction, 'Final-layer visual-only change affected text prediction'
                scored = score_prediction(metric='realworldqa', prediction_text=prediction, answer=item['answer'],
                                          choices=item.get('choices'), question=item['row'].get('question'))
                row = {'sample_position': index, 'sample_id': item['index'], 'condition': name,
                       'mode': mode, 'layer': li, 'prediction_text': prediction, **scored,
                       'selection_sha256': selection_hash, 'visual_tokens': len(p),
                       'seconds': time.time()-t0}
                if mode == 'full':
                    row['mixing'] = [intervene.metrics[k] for k in sorted(intervene.metrics)]
                if (index, name) not in finished:
                    out.write(json.dumps(row, allow_nan=False)+'\n')
                    out.flush()
                    processed += 1
                if mode == 'full' or li % 6 == 5:
                    print(f'shard={args.shard} sample={index} condition={name} score={scored["score"]} elapsed={time.time()-start:.1f}s', flush=True)
            # Matched numeric-path control: all edges retained, FP32 decomposition
            # recomposed at the earliest target layer, on first sample per worker.
            if plain is not None:
                intervene.set_mode('reconstructed', target_layers[0])
                reconstructed = generate()
                check = {'sample_position': index, 'native_prediction': baseline_prediction,
                         'reconstructed_prediction': reconstructed, 'observer_prediction_identical': True,
                         'max_reconstruction_relative_error': max(x['reconstruction_relative_error'] for x in intervene.metrics.values())}
                # Equal decomposition up to native BF16/FlashAttention rounding
                # can change e.g. answer capitalization; quantify rather than hide.
                check['reconstructed_text_identical'] = reconstructed == baseline_prediction
                (root / f'shard{args.shard}.check.json').write_text(json.dumps(check, indent=2))
            print(f'SAMPLE COMPLETE shard={args.shard} sample={index} nv={len(p)} elapsed={time.time()-start:.1f}s', flush=True)
    intervene.close()
    model.model.get_image_features = original_features
    (root / f'shard{args.shard}.done.json').write_text(json.dumps({'args': vars(args), 'seconds': time.time()-start,
                                                               'new_records': processed, 'self_check': checked}, indent=2))


def self_test():
    torch.manual_seed(4)
    q, k, v = [torch.randn(3, 9, 4, dtype=torch.float64) for _ in range(3)]
    p = torch.tensor([2, 3, 5, 7])
    own, cross, text, mass = decompose(q, k, v, p, .5, chunk=2)
    logits = q @ k.transpose(-1, -2) * .5
    logits.masked_fill_(torch.ones(9, 9, dtype=torch.bool).triu(1), -torch.inf)
    a = logits.softmax(-1)
    full = (a @ v)[:, p]
    assert torch.allclose(own+cross+text, full, atol=1e-12)
    assert torch.allclose(mass.sum(-1), torch.ones_like(mass[..., 0]))
    keep = torch.zeros(9, dtype=torch.bool); keep[p] = True
    for j, pos in enumerate(p):
        av = a[:, pos].clone()
        av[:, keep] = 0
        av[:, pos] = a[:, pos, pos]
        assert torch.allclose(torch.einsum('ht,htd->hd', av, v), own[:, j]+text[:, j])
        av = a[:, pos].clone(); av[:, pos] = 0
        assert torch.allclose(torch.einsum('ht,htd->hd', av, v), cross[:, j]+text[:, j])
    # A single visual token has exactly zero visual cross contribution.
    _, c, _, _ = decompose(q, k, v, torch.tensor([2]), .5)
    assert torch.equal(c, torch.zeros_like(c))
    print('PASS: decomposition, unchanged denominator, causal mask, text preservation, single-visual edge case')


def merge(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.output)
    protocol = json.loads((root / 'protocol.json').read_text())
    n = protocol['samples']
    layers = list(range(36)) if protocol['layers'] == 'all' else [int(x) for x in protocol['layers'].split(',')]
    records = {}
    hashes = set()
    for path in sorted(root.glob('shard[0-9]*.jsonl')):
        for line in path.open():
            r = json.loads(line)
            key = r['sample_position'], r['condition']
            assert key not in records, ('Duplicate', key)
            records[key] = r
            hashes.add(r['selection_sha256'])
    conditions = ['full:-1'] + [f'{m}:{li}' for li in layers for m in ('reconstructed', 'self_only', 'cross_only')]
    expected = {(i, c) for i in range(n) for c in conditions}
    assert set(records) == expected, f'Incomplete records: {len(records)}/{len(expected)}'
    assert len(hashes) == 1
    for i in range(n):
        assert len({records[(i,c)]['sample_id'] for c in conditions}) == 1
    base = np.array([records[(i,'full:-1')]['score'] for i in range(n)])
    rng = np.random.default_rng(44)
    bootstrap = rng.integers(0, n, (10000, n))
    summary = []
    mixing_rows = []
    for li in layers:
        re_rows = [records[(i,f'reconstructed:{li}')] for i in range(n)]
        re_scores = np.array([r['score'] for r in re_rows])
        metrics = [next(x for x in records[(i, 'full:-1')]['mixing'] if x['layer'] == li) for i in range(n)]
        mix = {'layer':li, 'samples':n}
        for key in metrics[0]:
            if key in ('layer','per_head'):continue
            vals = [x[key] for x in metrics if x[key] is not None]
            mix[key] = float(np.mean(vals)) if vals else None
        mixing_rows.append(mix)
        for mode in ('self_only','cross_only'):
            rows = [records[(i,f'{mode}:{li}')] for i in range(n)]
            scores = np.array([r['score'] for r in rows])
            diff = base-scores
            matched_diff = re_scores-scores
            ci = np.quantile(diff[bootstrap].mean(-1)*100, [.025,.975])
            matched_ci = np.quantile(matched_diff[bootstrap].mean(-1)*100, [.025,.975])
            row = {'layer':li,'mode':mode,'samples':n,'full_accuracy_pct':float(base.mean()*100),
                   'reconstructed_full_accuracy_pct':float(re_scores.mean()*100),
                   'numeric_control_drop_pp':float((base-re_scores).mean()*100),
                   'numeric_control_harmed':int((base>re_scores).sum()),
                   'numeric_control_helped':int((base<re_scores).sum()),
                   'numeric_control_changed_text':sum(r['prediction_text'] != records[(i,'full:-1')]['prediction_text'] for i,r in enumerate(re_rows)),
                   'accuracy_pct':float(scores.mean()*100),'drop_pp':float(diff.mean()*100),
                   'drop_ci95_low_pp':float(ci[0]),'drop_ci95_high_pp':float(ci[1]),
                   'matched_drop_pp':float(matched_diff.mean()*100),
                   'matched_drop_ci95_low_pp':float(matched_ci[0]),'matched_drop_ci95_high_pp':float(matched_ci[1]),
                   'matched_harmed':int((matched_diff>0).sum()),'matched_helped':int((matched_diff<0).sum()),
                   'harmed':int((diff>0).sum()),'helped':int((diff<0).sum()),
                   'changed_answer_text':sum(r['prediction_text'] != records[(i,'full:-1')]['prediction_text'] for i,r in enumerate(rows)),
                   'invalid':sum(bool(r['invalid']) for r in rows)}
            summary.append(row)
    for name, rows in [('accuracy_per_layer',summary),('mixing_per_layer',mixing_rows)]:
        (root/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        with (root/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig, axes = plt.subplots(2,1,figsize=(13,8),sharex=True)
    for mode,label,color in [('self_only','Remove visual cross-token mixing','#d95f02'),
                             ('cross_only','Remove visual self-read','#1b9e77')]:
        rows=[r for r in summary if r['mode']==mode]
        x=[r['layer'] for r in rows]; y=np.array([r['matched_drop_pp'] for r in rows])
        low=np.array([r['matched_drop_ci95_low_pp'] for r in rows]); high=np.array([r['matched_drop_ci95_high_pp'] for r in rows])
        axes[0].plot(x,y,label=label,color=color,marker='.',linewidth=1.5)
        axes[0].fill_between(x,low,high,color=color,alpha=.15)
    axes[0].axhline(0,color='gray',linewidth=.8)
    axes[0].set_ylabel('Drop vs matched reconstructed Full (pp)')
    axes[0].set_title(f'Qwen3-VL-4B-Instruct / RealWorldQA n={n}; native Full={base.mean()*100:.2f}%')
    axes[0].legend()
    axes[1].plot(layers,[r['cross_over_visual_fro'] for r in mixing_rows],label='Cross / visual readout (pre W_O)',marker='.')
    axes[1].plot(layers,[r['wo_cross_over_visual_fro'] for r in mixing_rows],label='Cross / visual readout (post W_O)',marker='.')
    axes[1].plot(layers,[r['cross_over_total_fro'] for r in mixing_rows],label='Cross / all-context readout (pre W_O)',linestyle='--')
    axes[1].set_xlabel('Intervened language-model layer (0-based)')
    axes[1].set_ylabel('Mean per-sample Frobenius norm ratio')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'accuracy_drop_and_mixing.png',dpi=180);plt.close(fig)
    table=['# RealWorldQA：视觉 cross-token mixing 消融','',
           f'Qwen3-VL-4B-Instruct，完整 {n} 条。Full 原生准确率 **{base.mean()*100:.2f}%**。', '',
           '## 口径','',
           '- 无训练、无 adapter；DeepStack 关闭，原生图像预处理和文本路径保留。层号 0–35。',
           '- 每个实验只干预一层的视觉 query。所有文本 query 不改；视觉 query 对此前可见文本/模板的读取保留。',
           '- Full = text + visual-self + visual-cross；Self-only = text + visual-self；Cross-only = text + visual-cross。',
           '- 使用原始 causal softmax 权重及原始分母，不对剩余边重新归一化；残差和 FFN 不删。',
           '- 改动发生在各 head 输出拼接后、原生 W_O 之前。decode 新文本 token 不干预；其 KV cache 来自对应干预后的 prefill。',
           '- 同样本各变体缓存原生视觉编码器特征（DeepStack 关闭）；单层干预处逐次检查原生输入与 Full 缓存逐位相同。',
           '- BF16 原生 FlashAttention；分项用 FP32 重算，转回 BF16 交给原生 W_O。每条样本、每个目标层额外跑一次 Full 重构：所有分项相加、不删除任何边，作为相同计算精度的数值对照。',
           '- 沿用现有 RealWorldQA prompt、贪心生成（最多 8 tokens）及本仓库 scorer，不用选项 logits 代替生成。',
           '- 下表 Drop = 同层 Full 重构 − 干预准确率，单位百分点；原生 Full 与重构 Full 同时列出。相对原生的 Drop 另存 CSV/JSON。95% CI 为样本配对 bootstrap（10000 次，seed 44），逐层区间未作多重比较校正。',
           '- H/G = 相对同层 Full 重构，答对改错 / 答错改对。净 Drop 接近零不代表没有个体样本变化。', '',
           '## 逐层准确率','',
           '| 层 | 原生 Full % | Full 重构 % | Self-only % | Drop pp [95% CI] | H/G | Cross-only % | Drop pp [95% CI] | H/G |',
           '|---:|---:|---:|---:|---|---|---:|---|---|']
    for li in layers:
        a,b=[next(r for r in summary if r['layer']==li and r['mode']==m) for m in ('self_only','cross_only')]
        def interval(r):return f"{r['matched_drop_pp']:.2f} [{r['matched_drop_ci95_low_pp']:.2f}, {r['matched_drop_ci95_high_pp']:.2f}]"
        table.append(f"| {li} | {base.mean()*100:.2f} | {a['reconstructed_full_accuracy_pct']:.2f} | {a['accuracy_pct']:.2f} | {interval(a)} | {a['matched_harmed']}/{a['matched_helped']} | {b['accuracy_pct']:.2f} | {interval(b)} | {b['matched_harmed']}/{b['matched_helped']} |")
    table += ['', '## 原生 Full 的实际 mixing 幅度','',
              '先对单样本的所有视觉 query/head 取 Frobenius 范数比，再对样本等权平均。`M_visual = ||cross|| / ||self+cross||`；`M_total = ||cross|| / ||self+cross+text||`。W_O 后也统计视觉分项之比。范数比不是概率，分项抵消时可以大于 1；不是“有效信息比例”。逐 head/query 比值均值、P95 和近零分母计数另存 JSON/CSV。','',
              '| 层 | M_visual | M_total | M_visual（W_O 后） | Self mass | Cross mass | Text mass | 重构相对误差 |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in mixing_rows:
        vals=[r[k] for k in ('cross_over_visual_fro','cross_over_total_fro','wo_cross_over_visual_fro','self_mass','cross_mass','text_mass','reconstruction_relative_error')]
        table.append('| '+str(r['layer'])+' | '+' | '.join(f'{x:.5f}' if x is not None else 'undefined' for x in vals)+' |')
    table += ['', '## 解读边界','',
              '单层消融影响小，说明在其余计算保留时该层视觉跨位置写入的边际影响较小；不证明同时删除多层也无损，也不证明所有跨位置信息都冗余。这里没有去掉视觉编码器和早先语言层已经建立的上下文。', '',
              '最后一层只改变视觉输出，之后没有新的一层 attention 让文本读取它，因此预期准确率完全不变；代码逐样本检查这一点，不能将其当成独立的冗余证据。','',
              '![逐层准确率下降和 mixing 幅度](accuracy_drop_and_mixing.png)','',
              '原始逐样本结果：`shard*.jsonl`；配置：[protocol.json](protocol.json)；准确率：[accuracy_per_layer.csv](accuracy_per_layer.csv)；幅度：[mixing_per_layer.csv](mixing_per_layer.csv)。']
    (root/'README.md').write_text('\n'.join(table)+'\n')
    print(f'MERGED {len(records)} records; Full {base.mean()*100:.2f}%; report {root / "README.md"}',flush=True)


def launch(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'model': MODEL, 'benchmark': 'realworldqa', 'samples': args.limit or 765,
                'layers': args.layers, 'max_new_tokens': args.max_new_tokens, 'dtype': 'bfloat16',
                'attention': 'native flash_attention_2; intervention components FP32, cast to BF16 before native W_O',
                'scope': 'single-layer visual query -> visual key, DeepStack disabled, text keys preserved',
                'denominator': 'unchanged full causal denominator; no edge renormalization',
                'numeric_control': 'reconstructed Full at every layer on every sample; paired drops vs native and reconstructed Full',
                'seed': 44, 'args': vars(args), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'start_time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    manifest_path = root / 'protocol.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))
    procs = []
    try:
        for shard in range(args.shards):
            log = (root / f'shard{shard}.log').open('a')
            command = [sys.executable, '-u', '-m', 'analysis.common.visual_cross_token_ablation', 'run',
                       '--output', str(root), '--shard', str(shard), '--shards', str(args.shards),
                       '--limit', str(args.limit), '--layers', args.layers, '--max-new-tokens', str(args.max_new_tokens)]
            if args.resume: command.append('--resume')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            procs.append((proc, log))
            print(f'worker GPU={shard} pid={proc.pid} log={log.name}', flush=True)
        while any(proc.poll() is None for proc, _ in procs):
            for proc, _ in procs:
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f'worker {proc.pid} exited {proc.returncode}; inspect logs')
            time.sleep(5)
        if any(proc.returncode != 0 for proc, _ in procs):
            raise RuntimeError('worker failure')
        print('ALL WORKERS COMPLETE', flush=True)
        merge(args)
    finally:
        for proc, log in procs:
            if proc.poll() is None: proc.terminate()
            log.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=('run', 'launch', 'merge', 'self-test'))
    p.add_argument('--output', default=str(ROOT / 'artifacts/diagnostics/realworldqa_visual_cross_token_20260912'))
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--layers', default='all')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    {'run': run, 'launch': launch, 'merge': merge, 'self-test': lambda _: self_test()}[args.command](args)


if __name__ == '__main__':
    main()
