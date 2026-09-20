"""One prefix length at a time, FA2 with DeepStack off and rank0 edge blocking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from src import causal_effect_benchmark_suite as suite


class FA2RankZero:
    """Text-only causal FA2 exactly removes visual keys and renormalizes."""
    def __init__(self, model):
        self.mode = 'native'
        self.selected = set()
        self.calls = {}
        self.audit = False
        self.errors = []
        self.handles = [layer.self_attn.register_forward_hook(self.hook(i), with_kwargs=True)
                        for i, layer in enumerate(model.model.language_model.layers)]

    def reset(self, inputs, mode, scope=None, rank=0):
        assert rank == 0
        assert inputs['input_ids'].shape[0] == 1 and bool(inputs['attention_mask'].all())
        self.types = inputs['mm_token_type_ids']
        self.mode = mode
        self.selected = set(suite.core.SCOPES[scope]) if scope else set()
        self.calls = {}

    def hook(self, layer):
        def run(module, args, kw, output):
            if self.mode == 'native' or layer not in self.selected:
                return
            from flash_attn import flash_attn_func
            from src.model import qwen_apply_rotary_pos_emb
            assert kw.get('past_key_values') is None
            h = kw.get('hidden_states', args[0] if args else None)
            shape = (*h.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(h).view(shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(shape)).transpose(1, 2)
            v = module.v_proj(h).view(shape).transpose(1, 2)
            q, k = qwen_apply_rotary_pos_emb(q, k, *kw['position_embeddings'])
            text = (self.types[0] == 0).nonzero().flatten()
            visual = self.types[0] == 1
            assert bool(visual.any()) and self.types.shape[1] == h.shape[1]
            qt, kt, vt = [x.index_select(2, text) for x in (q, k, v)]
            heads = flash_attn_func(qt.transpose(1, 2), kt.transpose(1, 2), vt.transpose(1, 2),
                                   dropout_p=0., softmax_scale=float(module.scaling), causal=True)
            replacement = module.o_proj(heads.reshape(1, len(text), -1))
            if self.audit:
                # Independent full-key FP32 masked attention on sampled text queries.
                positions = torch.linspace(0, len(text)-1, min(8,len(text)),device=h.device).long().unique()
                qq = qt.index_select(2, positions).float()
                kk = k.repeat_interleave(q.shape[1]//k.shape[1],1).float()
                vv = v.repeat_interleave(q.shape[1]//v.shape[1],1).float()
                logits = (qq @ kk.transpose(-1,-2)) * float(module.scaling)
                keys = torch.arange(h.shape[1],device=h.device)
                allowed = (keys[None,:] <= text[positions,None]) & (~visual)[None,:]
                logits.masked_fill_(~allowed[None,None], -torch.inf)
                expected = logits.softmax(-1) @ vv
                actual = heads.transpose(1,2).index_select(2,positions).float()
                error = float((actual-expected).norm()/expected.norm().clamp_min(1e-10))
                assert error < .02, (layer,error)
                self.errors.append(error)
            patched = output[0].clone()
            patched.index_copy_(1,text,replacement)
            assert torch.equal(patched[:,visual],output[0][:,visual])
            self.calls[layer] = self.calls.get(layer,0)+1
            return (patched,)+output[1:]
        return run


def worker(a):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    suite.configure('qwen')
    scope = f'first{a.first}' + (f'_last{a.last}' if a.last else '')
    suite.core.SCOPES[scope] = sorted(set(range(a.first)) | set(range(36-a.last,36)))
    processor, model = load_frozen_qwen3vl(suite.MODELS['qwen'][0], torch.bfloat16, torch.device('cuda:0'), a.backend)
    from src.qwen_deepstack import disable_qwen_deepstack
    if a.deepstack == 'off':
        disable_qwen_deepstack(model)
    assert model.config.text_config._attn_implementation == a.backend
    hook = FA2RankZero(model) if a.backend == 'flash_attention_2' else suite.core.Intervention(model)
    features = []
    original = model.model.get_image_features
    def cached(*args, **kwargs):
        if not features:
            features.append(original(*args, **kwargs))
        return features[0]
    model.model.get_image_features = cached
    root = Path(a.output)
    for benchmark in suite.DATASETS:
        relative, n = suite.DATASETS[benchmark]
        data = suite.ROOT / relative
        dataset = QwenBenchmarkDataset(str(data), processor, benchmark, data_root=str(data.parent), max_samples=n)
        old = Path(a.reference) / f'qwen_{benchmark}'
        plan = json.loads((old/'plan.json').read_text())
        assert hashlib.sha256(json.dumps(dataset.rows, sort_keys=True).encode()).hexdigest() == plan['selection_sha256']
        reference = {}
        if a.first != 11 and not a.fresh:
            for file in (Path(a.baseline)/benchmark).glob('shard*.jsonl'):
                reference.update({r['sample']:r['results'] for r in map(json.loads,file.open())})
        folder = root/benchmark
        folder.mkdir(exist_ok=True)
        with (folder/f'shard{a.shard}.jsonl').open('w') as out, torch.inference_mode():
            for i in range(a.shard, n, a.world):
                item = dataset[i]
                inputs = suite.prepare(item, model, 'qwen')
                features.clear()
                results = {}
                modes = [('native', None, 'native'), ('rank', 'first10', 'first10_r0')] if a.first == 11 else []
                if a.fresh:
                    modes = [('native', None, 'native'), ('rank', 'first10', 'first10_r0')]
                modes.append(('rank', scope, scope+'_r0'))
                if a.first != 11 and not a.fresh:
                    results['native'] = reference[i]['native']
                    results['first10_r0'] = reference[i]['first10_r0']
                hook.audit = i == a.shard
                for mode, current_scope, key in modes:
                    answer = suite.generate(model, processor, inputs, hook, 'qwen', mode, current_scope, 0)
                    score = score_prediction(metric=get_benchmark_spec(benchmark).metric, prediction_text=answer,
                        answer=item['answer'], choices=item.get('choices'), question=item['row'].get('question'))
                    results[key] = {'text':answer, **score}
                if hook.audit:
                    (folder/f'parity{a.shard}.json').write_text(json.dumps({'sample':i,'max_relative_head_error':max(hook.errors) if hasattr(hook,'errors') else None, 'visual_outputs_exact':True}))
                hook.audit = False
                out.write(json.dumps({'sample':i, 'index':item['index'], 'results':results})+'\n')
                out.flush()
                if i//a.world % 20 == 0:
                    print(benchmark, scope, i, flush=True)
        (folder/f'shard{a.shard}.done.json').write_text(json.dumps({'samples':len(range(a.shard,n,a.world))}))


def launch(a):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps(dict(vars(a), protocol=f'BF16/{a.backend}, DeepStack {a.deepstack}; text-to-visual blocked and renormalized; native visual outputs retained; greedy max8; original sample order'), indent=2))
    jobs=[]
    try:
        for shard in range(a.world):
            log=(root/f'gpu{shard}.log').open('w')
            cmd=[sys.executable, '-u', '-m', 'src.causal_first_rank0_sweep', 'worker', '--first', str(a.first), '--output', str(root), '--reference', a.reference, '--world', str(a.world), '--shard', str(shard), '--baseline', a.baseline, '--backend', a.backend, '--deepstack', a.deepstack, '--last', str(a.last)] + (['--fresh'] if a.fresh else [])
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            jobs.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,_ in jobs):
            if any(p.poll() not in (None,0) for p,_ in jobs):
                raise RuntimeError('Worker failed; see GPU logs')
            time.sleep(3)
        assert all(p.returncode==0 for p,_ in jobs)
        scope = f'first{a.first}' + (f'_last{a.last}' if a.last else '')
        summary={}
        for benchmark,(_,n) in suite.DATASETS.items():
            rows=[]
            for shard in range(a.world):
                assert (root/benchmark/f'shard{shard}.done.json').exists()
                rows.extend(map(json.loads,(root/benchmark/f'shard{shard}.jsonl').open()))
            assert len(rows)==n and {r['sample'] for r in rows}==set(range(n))
            summary[benchmark]={}
            for key in ['native','first10_r0',scope+'_r0']:
                correct=sum(r['results'][key]['score'] for r in rows)
                summary[benchmark][key]={'samples':n,'correct':correct,'accuracy_pct':100*correct/n}
            summary[benchmark]['drop_pp']=summary[benchmark]['native']['accuracy_pct']-summary[benchmark][scope+'_r0']['accuracy_pct']
        (root/'results.json').write_text(json.dumps(summary,indent=2))
        (root/'status.json').write_text(json.dumps({'state':'complete'}))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as e:
        (root/'status.json').write_text(json.dumps({'state':'failed','error':repr(e)}))
        raise
    finally:
        for p,log in jobs:
            if p.poll() is None:p.terminate()
            log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['launch','worker'])
    parser.add_argument('--first',type=int,required=True,choices=range(1,21))
    parser.add_argument('--last',type=int,default=0,choices=range(0,37))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default=str(suite.ROOT/'artifacts/diagnostics/causal_effect_2models_2bench_20260912'))
    parser.add_argument('--backend',choices=['flash_attention_2','sdpa'],default='flash_attention_2')
    parser.add_argument('--deepstack',choices=['on','off'],default='off')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--baseline',default='')
    parser.add_argument('--world',type=int,default=8)
    parser.add_argument('--shard',type=int,default=0)
    a=parser.parse_args()
    {'launch':launch,'worker':worker}[a.command](a)
