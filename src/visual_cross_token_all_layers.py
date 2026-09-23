"""All-layer visual-edge interventions on the CURRENT trajectory (no layer cache)."""
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

from src.visual_cross_token_ablation import MODEL, ROOT, decompose, flatten_heads, prepare, ratio

MODES = ('full', 'reconstructed', 'self_only', 'cross_only')


class AllLayerIntervention:
    def __init__(self, model):
        self.layers = model.model.language_model.layers
        self.mode = 'full'
        self.positions = None
        self.pending = {}
        self.changed = []
        self.errors = []
        self.native_inputs = {}
        self.check_inputs = False
        self.different_inputs = []
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self.capture(li), with_kwargs=True))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self.replace(li)))

    def set_mode(self, mode):
        assert not self.pending
        self.mode = mode
        self.changed = []
        self.errors = []
        self.different_inputs = []

    def capture(self, li):
        def hook(module, args, kwargs):
            h = kwargs.get('hidden_states', args[0] if args else None)
            if h.shape[1] == 1:
                return  # generated text, not a visual prefill row
            assert h.shape[0] == 1
            if self.check_inputs:
                current = h[0, self.positions]
                if self.mode == 'full':
                    self.native_inputs[li] = current.clone()
                elif not torch.equal(current, self.native_inputs[li]):
                    self.different_inputs.append(li)
            if self.mode == 'full':
                return
            mask = kwargs.get('attention_mask')
            assert mask is None or (mask.ndim == 2 and bool(mask.all()))
            shape = (*h.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(h).view(shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(shape)).transpose(1, 2)
            v = module.v_proj(h).view(shape).transpose(1, 2)
            from src.model import qwen_apply_rotary_pos_emb
            q, k = qwen_apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            groups = q.shape[1] // k.shape[1]
            # Recompute at EVERY layer from this variant's actual input states.
            self.pending[li] = decompose(q[0], k[0].repeat_interleave(groups, 0),
                                         v[0].repeat_interleave(groups, 0), self.positions,
                                         float(module.scaling))
        return hook

    def replace(self, li):
        def hook(module, args):
            native = args[0]
            if native.shape[1] == 1 or self.mode == 'full':
                return
            own, cross, prefix, masses = self.pending.pop(li)
            full = flatten_heads(own + cross + prefix)
            err = ratio(full-native[0, self.positions].float(), native[0, self.positions])
            assert err is not None and err < .025, (li, err)
            assert float((masses.sum(-1)-1).abs().max()) < 1e-5
            self.errors.append(err)
            chosen = {'reconstructed': lambda: full,
                      'self_only': lambda: flatten_heads(own + prefix),
                      'cross_only': lambda: flatten_heads(cross + prefix)}[self.mode]()
            result = native.clone()
            result[0, self.positions] = chosen.to(native.dtype)
            mask = torch.ones(native.shape[1], dtype=torch.bool, device=native.device)
            mask[self.positions] = False
            assert torch.equal(result[:, mask], native[:, mask]), 'Changed nonvisual query rows'
            self.changed.append(li)
            return (result,) + args[1:]
        return hook


def run(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    from src.eval_benchmarks import generate_teacher_qwen
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'shard{args.shard}.jsonl'
    if path.exists():
        raise FileExistsError(path)
    processor, model = load_frozen_qwen3vl(MODEL, torch.bfloat16, torch.device('cuda:0'), 'flash_attention_2')
    spec = get_benchmark_spec('realworldqa')
    data = ROOT / spec.default_data
    ds = QwenBenchmarkDataset(str(data), processor, 'realworldqa', data_root=str(data.parent), max_samples=args.limit or None)
    sha = hashlib.sha256(json.dumps(ds.rows, sort_keys=True).encode()).hexdigest()
    intervention = AllLayerIntervention(model)
    # Only cache unchanged native image encoder features (DeepStack disabled), NEVER LM states.
    original_features = model.model.get_image_features
    features = []
    def get_features(*a, **kw):
        if not features:
            features.append(original_features(*a, **kw))
        return features[0]
    model.model.get_image_features = get_features
    start = time.time()
    total = 0
    with torch.inference_mode(), path.open('w') as out:
        for index in range(args.shard, len(ds), args.shards):
            item = ds[index]
            inputs = prepare(item, torch.device('cuda:0'))
            positions = (inputs['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
            assert len(positions) > 0
            assert torch.equal(positions, torch.arange(int(positions[0]), int(positions[-1])+1, device=positions.device))
            prefix = processor.tokenizer.decode(inputs['input_ids'][0, :positions[0]], skip_special_tokens=False)
            assert prefix == '<|im_start|>user\n<|vision_start|>', prefix
            intervention.positions = positions
            intervention.check_inputs = total == 0
            intervention.native_inputs.clear()
            features.clear()
            for mode in MODES:
                intervention.set_mode(mode)
                t0 = time.time()
                _, prediction = generate_teacher_qwen(model, processor, **inputs, max_new_tokens=8)
                expected = [] if mode == 'full' else list(range(len(intervention.layers)))
                assert intervention.changed == expected
                assert not intervention.pending
                if intervention.check_inputs and mode in ('self_only','cross_only'):
                    assert 0 not in intervention.different_inputs
                    assert 1 in intervention.different_inputs, 'All-layer change did not propagate to next layer'
                score = score_prediction(metric='realworldqa', prediction_text=prediction, answer=item['answer'],
                                         choices=item.get('choices'), question=item['row'].get('question'))
                row = {'sample_position':index,'sample_id':item['index'],'mode':mode,
                       'prediction_text':prediction,**score,'selection_sha256':sha,
                       'visual_tokens':len(positions),'prefix':prefix,'changed_layers':intervention.changed,
                       'reconstruction_error_max':max(intervention.errors, default=0.),
                       'seconds':time.time()-t0}
                if intervention.check_inputs:
                    row['downstream_inputs_different_from_native'] = intervention.different_inputs
                out.write(json.dumps(row,allow_nan=False)+'\n');out.flush()
                total += 1
            print(f'SAMPLE COMPLETE shard={args.shard} sample={index} elapsed={time.time()-start:.1f}s',flush=True)
    (root/f'shard{args.shard}.done.json').write_text(json.dumps({'records':total,'seconds':time.time()-start},indent=2))


def merge(args):
    import numpy as np
    root = Path(args.output)
    n = args.limit or 765
    rows = {}
    for f in sorted(root.glob('shard[0-9]*.jsonl')):
        for line in f.open():
            r = json.loads(line);k = r['sample_position'],r['mode']
            assert k not in rows
            rows[k] = r
    assert set(rows)=={(i,m) for i in range(n) for m in MODES}
    assert len({r['selection_sha256'] for r in rows.values()})==1
    for i in range(n):assert len({rows[(i,m)]['sample_id'] for m in MODES})==1
    native = np.array([rows[(i,'full')]['score'] for i in range(n)])
    ref = np.array([rows[(i,'reconstructed')]['score'] for i in range(n)])
    boot = np.random.default_rng(44).integers(0,n,(10000,n))
    summary = []
    for mode in MODES:
        score = np.array([rows[(i,mode)]['score'] for i in range(n)])
        diff = ref-score;ci=np.quantile(diff[boot].mean(-1)*100,[.025,.975])
        summary.append({'mode':mode,'samples':n,'correct':int(score.sum()),'accuracy_pct':float(score.mean()*100),
                        'drop_vs_native_pp':float((native-score).mean()*100),'drop_vs_reconstructed_pp':float(diff.mean()*100),
                        'ci95_low_pp':float(ci[0]),'ci95_high_pp':float(ci[1]),
                        'harmed_vs_reconstructed':int((diff>0).sum()),'helped_vs_reconstructed':int((diff<0).sum())})
    old = {}
    previous = ROOT/'artifacts/diagnostics/realworldqa_visual_cross_token_20260912'
    for f in sorted(previous.glob('shard[0-9]*.jsonl')):
        for line in f.open():
            if '"condition": "full:-1"' not in line:continue
            r=json.loads(line);old[r['sample_position']]=r
    mismatch = [i for i in range(n) if i in old and rows[(i,'full')]['prediction_text']!=old[i]['prediction_text']]
    result={'summary':summary,'baseline_text_mismatches_vs_previous':mismatch,
            'baseline_compared_samples':sum(i in old for i in range(n)),
            'records':len(rows),'max_reconstruction_error':max(r['reconstruction_error_max'] for r in rows.values())}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    labels={'full':'原生 Full','reconstructed':'全层 Full 重构','self_only':'全层 Self-only','cross_only':'全层 Cross-only'}
    doc=['# RealWorldQA：全部层同时消融视觉 mixing','',
         f'Qwen3-VL-4B-Instruct，{n} 条，层号 0–35 全部同时干预，8 卡，无训练。','',
         '| 条件 | 正确数 | 准确率 | 相对原生下降 pp | 相对重构下降 pp [95% CI] | 答对→错 / 答错→对（相对重构） |',
         '|---|---:|---:|---:|---|---|']
    for r in summary:
        doc.append(f"| {labels[r['mode']]} | {r['correct']}/{n} | {r['accuracy_pct']:.2f}% | {r['drop_vs_native_pp']:.2f} | {r['drop_vs_reconstructed_pp']:.2f} [{r['ci95_low_pp']:.2f}, {r['ci95_high_pp']:.2f}] | {r['harmed_vs_reconstructed']}/{r['helped_vs_reconstructed']} |")
    doc += ['', '## 干预定义','',
            '- Self-only：全部语言层删除视觉 query 对其他视觉 token 的读取，保留读自己和前置模板标记。',
            '- Cross-only：全部语言层删除视觉 query 对自己的读取，保留其他视觉 token 和前置模板标记。',
            '- 前置模板实际为 `<|im_start|>user\\n<|vision_start|>` 共 4 个 token；图片后的问题不可见，仍严格因果。',
            '- 每层基于该条件当前的 hidden 重新计算 Q/K/V、RoPE 和原始 causal softmax 分母，随后删除对应边；不重新归一化。绝不复用原生逐层分项缓存。',
            '- 在原生 W_O 前只替换视觉 query 行；文本 query 行、原生 residual/FFN 保留，DeepStack 关闭。后续文本状态可因读取已改变的视觉状态而变化。',
            '- native decode 使用各变体自己的 prefill KV cache，不干预新生成的文本 query。沿用原有 prompt、greedy、最多 8 tokens 和 scorer。',
            '- Full 重构在全部层保留 self+cross+prefix，控制 FP32 分项重算、BF16 输出的数值差异。',
            '- 各条件逐样本断言全部 36 层实际执行干预，且非视觉 query 行未被直接改写。首样本进一步检查第 0 层输入不变、第 1 层已受到上层干预影响。',
            '- 95% CI：逐样本配对 bootstrap 10000 次，seed 44；下降为正代表变差。', '',
            f'与上一轮逐层实验共有 {result["baseline_compared_samples"]} 条基线对照，原生答案文本不一致 {len(mismatch)} 条。', '',
            '[原始汇总 JSON](results.json)；原始逐样本记录：`shard*.jsonl`。']
    (root/'README.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def launch(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest={'model':MODEL,'samples':args.limit or 765,'scope':'all 36 layers simultaneously',
              'modes':MODES,'shards':args.shards,'max_new_tokens':8,'native_deepstack':False,
              'current_state_recompute':True,'layer_component_cache':False,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'decomposition_source_sha256':hashlib.sha256((ROOT/'src/visual_cross_token_ablation.py').read_bytes()).hexdigest()}
    (root/'protocol.json').write_text(json.dumps(manifest,indent=2))
    procs=[]
    try:
        for shard in range(args.shards):
            log=(root/f'shard{shard}.log').open('w')
            cmd=[sys.executable,'-u','-m','src.visual_cross_token_all_layers','run','--output',str(root),
                 '--shards',str(args.shards),'--shard',str(shard),'--limit',str(args.limit)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',
                     TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            procs.append((proc,log));print(f'GPU {shard}: PID {proc.pid}',flush=True)
        while any(p.poll() is None for p,_ in procs):
            for p,_ in procs:
                if p.poll() not in (None,0):raise RuntimeError(f'Worker {p.pid} failed; inspect logs')
            time.sleep(3)
        assert all(p.returncode==0 for p,_ in procs)
        merge(args)
    finally:
        for p,f in procs:
            if p.poll() is None:p.terminate()
            f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['run','launch','merge'])
    p.add_argument('--output',default=str(ROOT/'artifacts/diagnostics/realworldqa_visual_cross_token_all_layers_20260912'))
    p.add_argument('--shards',type=int,default=8)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    {'run':run,'launch':launch,'merge':merge}[args.command](args)
