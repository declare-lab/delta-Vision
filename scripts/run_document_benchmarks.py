"""Resumable, fingerprinted 15-configuration evaluation on three document tasks."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'artifacts/eval/document_benchmarks_1000_20260918'
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
BENCHES = ['chartqa', 'docvqa', 'infographicvqa']
METHODS = ['base', 'fastv', 'dart', 'visionzip', 'sparsevlm', 'divprune', 'zoo',
           'embedding_adapter', 'recurrent_adapter']
CKPT_ROOT = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911'
CHECKPOINTS = {
    'embedding_adapter': CKPT_ROOT/'static_kl/checkpoints/qwen_embedding_adapter_step2000.pt',
    'recurrent_adapter': CKPT_ROOT/'recurrent_kl/checkpoints/qwen_recurrent_embedding_adapter_step2000.pt',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def ratios(method):
    return [1.] if method == 'base' or method in CHECKPOINTS else [.05, .2]


def manifest(bench):
    return ROOT/f'data/benchmarks/{bench}/eval1000_seed42.jsonl'


def selected_indices(stage, shard, shards, rows):
    if stage == 'smoke':
        return sorted({0, 500, max(range(1000), key=lambda i: rows[i]['width']*rows[i]['height'])})
    return list(range(shard, 1000, shards))


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    paths = ['scripts/run_document_benchmarks.py', 'scripts/prepare_document_benchmarks.py',
             'src/benchmarks.py', 'src/document_metrics.py', 'src/data.py', 'src/model.py',
             'src/eval_benchmarks.py', 'src/qwen_adapter_fa2.py', 'src/qwen_deepstack.py',
             'baselines/eval_baselines.py', 'baselines/multimodal_pruning_utils.py']
    paths += [str(p.relative_to(ROOT)) for method in METHODS[1:7]
              for p in (ROOT/f'baselines/{method}/qwen3_vl').glob('*.py')]
    plan = dict(model=MODEL, attention='flash_attention_2', dtype='bfloat16', deepstack=False,
        methods=METHODS, retentions={m: ratios(m) for m in METHODS}, samples_per_benchmark=1000,
        retention_definition='Existing post-pruning visual retention parameter; early full layers are NOT deducted. Actual per-layer visual counts are saved.',
        benchmarks=BENCHES, seed=42, prompt='image first; question; Answer the question using a single word or phrase.',
        max_new_tokens=128, decoding='greedy, EOS stopping, cached, batch size 1',
        image_resolution='Unmodified model processor defaults; no image resizing or token cap overrides.',
        metrics={'chartqa':'relaxed accuracy (5% numeric tolerance)', 'docvqa':'ANLS (reference threshold 0.5)', 'infographicvqa':'ANLS (reference threshold 0.5)'},
        avg='Arithmetic mean of the three unrounded benchmark scores, all displayed x100; no F1.',
        data={b: json.loads((manifest(b).parent/'provenance.json').read_text()) for b in BENCHES},
        checkpoints={m: dict(path=str(p), sha256=sha(p)) for m,p in CHECKPOINTS.items()},
        model_config_sha256=sha(Path(MODEL)/'config.json'),
        processor_config_sha256=sha(Path(MODEL)/'preprocessor_config.json'),
        sources={p: sha(ROOT/p) for p in paths})
    for b, meta in plan['data'].items():
        assert sha(manifest(b)) == meta['manifest_sha256']
        rows = [json.loads(l) for l in manifest(b).read_text().splitlines()]
        assert len(rows) == 1000 and len({r['question_id'] for r in rows}) == 1000
        for row in rows:
            assert row['answers'] and sha(Path(row['image_root'])/row['image']) == row['image_sha256']
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text()) == plan, 'Protocol changed; do not mix results'
    else:
        dump(OUT/'plan.json', plan)
        for p in paths:
            dest = OUT/'source'/p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/p).read_bytes())
    return plan


def load_rows(path):
    if not path.exists():
        return []
    data = path.read_text()
    # A killed writer can leave one incomplete line. Never treat it as a result.
    if data and not data.endswith('\n'):
        data = data[:data.rfind('\n')+1]
    return [json.loads(line) for line in data.splitlines()]


def worker(args):
    import torch
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.benchmarks import score_prediction
    from src.data import QwenBenchmarkDataset
    from src.qwen_deepstack import disable_qwen_deepstack
    from src.model import (load_qwen_embedding_adapter_checkpoint, build_qwen_initial_context,
                          prepare_qwen_embedding_adapter_inputs, qwen_embedding_adapter_prefill_cache_prepared,
                          qwen_embedding_adapter_logits)
    from src.eval_benchmarks import generate_adapter_qwen_decode_cache
    torch.set_num_threads(4)
    torch.manual_seed(42)
    plan = json.loads((OUT/'plan.json').read_text())
    plan_hash = sha(OUT/'plan.json')
    for p,h in plan['sources'].items():
        assert sha(ROOT/p) == h, p
    method = args.method
    model, processor = load_baseline_model('base' if method in CHECKPOINTS else method,
        MODEL, torch.bfloat16, torch.device('cuda:0'), .2, 'flash_attention_2')
    disable_qwen_deepstack(model)
    model.eval().requires_grad_(False)
    adapter = None
    if method in CHECKPOINTS:
        assert sha(CHECKPOINTS[method]) == plan['checkpoints'][method]['sha256']
        adapter, meta = load_qwen_embedding_adapter_checkpoint(CHECKPOINTS[method], model.model.language_model,
                                                              torch.device('cuda:0'), torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected'], meta
        assert adapter.mode == ('embedding_adapter' if method=='embedding_adapter' else 'recurrent_embedding_adapter')
        adapter.eval().requires_grad_(False)
        model._adapter_attention_implementation = 'flash_attention_2'
    for module in (model.model, model.model.language_model):
        module._pruning_audit_enabled = False
    observed = []
    def observe(module, positional, keyword, result):
        if not observed and result.past_key_values is not None:
            observed.extend(int(layer.keys.shape[-2]) for layer in result.past_key_values.layers)
    hook = model.register_forward_hook(observe, with_kwargs=True) if adapter is None else None
    directory = OUT/args.stage
    directory.mkdir(exist_ok=True)
    path = directory/f'{method}_{args.shard}.jsonl'
    saved = load_rows(path)
    done = {(r['benchmark'],r['index'],r['retention']) for r in saved}
    assert len(done) == len(saved)
    assert all(r['plan_sha256'] == plan_hash for r in saved)
    if path.exists() and not path.read_bytes().endswith(b'\n'):
        with path.open('wb') as f:
            for r in saved:
                f.write((json.dumps(r, ensure_ascii=False)+'\n').encode())
    completed = len(saved)
    started = time.time()
    with torch.inference_mode(), path.open('a', buffering=1) as handle:
        for bench in BENCHES:
            assert sha(manifest(bench)) == plan['data'][bench]['manifest_sha256']
            dataset = QwenBenchmarkDataset(str(manifest(bench)), processor, bench)
            for i in selected_indices(args.stage, args.shard, args.shards, dataset.rows):
                if all((bench,i,r) in done for r in ratios(method)):
                    continue
                item = dataset[i]
                h = hashlib.sha256()
                for key in ['input_ids','attention_mask','mm_token_type_ids','image_grid_thw','pixel_values']:
                    tensor = item[key].contiguous()
                    h.update(str((key,tuple(tensor.shape),str(tensor.dtype))).encode())
                    h.update(tensor.view(torch.uint8).numpy().tobytes())
                input_sha = h.hexdigest()
                nv = int(item['mm_token_type_ids'].ne(0).sum())
                nt = int(item['attention_mask'].sum())-nv
                inputs = _qwen_inputs_from_item(item, torch.device('cuda:0'))
                for retention in ratios(method):
                    if (bench,i,retention) in done:
                        continue
                    t0 = time.time()
                    torch.manual_seed(42+i)
                    model.model.rope_deltas = None
                    observed.clear()
                    parity = None
                    if adapter is None:
                        positions = item['mm_token_type_ids'].ne(0).nonzero().flatten()
                        configure_baseline(model, method, retention, int(positions[0]), nv)
                        generated = model.generate(**inputs, do_sample=False, use_cache=True,
                            max_new_tokens=128, return_dict_in_generate=False)
                        tokens = generated[0,inputs['input_ids'].shape[1]:].tolist()
                        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        assert len(observed) == 36, observed
                        layer_visual = [n-nt for n in observed]
                        assert all(0 < n <= nv for n in layer_visual)
                    else:
                        hidden, positions = build_qwen_initial_context(model, inputs)
                        prepared = prepare_qwen_embedding_adapter_inputs(model, adapter,
                            inputs['input_ids'], inputs['attention_mask'], inputs['mm_token_type_ids'], hidden, positions)
                        assert prepared['attention_plan'] is not None
                        logits, mask, cache = qwen_embedding_adapter_prefill_cache_prepared(model, adapter,
                            **prepared, logits_to_keep=1, retain_prefix_states=False)
                        if args.stage == 'smoke':
                            reference = qwen_embedding_adapter_logits(model, adapter, inputs,
                                initial_hidden=hidden, position_ids=positions, logits_to_keep=1)[0]
                            delta = (logits.float()-reference.float()).abs().max().item()
                            parity = dict(prefill_max_abs_diff=delta, same_argmax=bool(logits.argmax(-1).eq(reference.argmax(-1)).all()))
                            assert parity['same_argmax'], parity
                            del reference
                        step_metrics = {}
                        _, texts = generate_adapter_qwen_decode_cache(model, processor, adapter, inputs, 128,
                            initial_hidden=hidden, position_ids=positions,
                            prefill_logits=logits, prefill_text_mask=mask, decode_cache=cache,
                            decode_cache_mode='fast', decode_step_metrics=step_metrics)
                        text = texts[0].strip()
                        tokens = step_metrics['generated_token_ids'][0]
                        layer_visual = [nv]*36
                        del cache, hidden, positions, prepared, logits, mask
                    scored = score_prediction(metric=dataset.spec.metric, prediction_text=text,
                        answer=item['answer'], answers=item['answers'], question=item['row']['question'])
                    record = dict(benchmark=bench, index=i, source_index=item['index'],
                        question_id=item['row']['question_id'], method=method, retention=retention,
                        prediction_text=text, **scored, generated_token_ids=tokens,
                        truncated=len(tokens)==128, input_sha256=input_sha,
                        visual_tokens=nv, text_tokens=nt, layer_visual_tokens=layer_visual,
                        layer_sum_visual_ratio=sum(layer_visual)/(36*nv),
                        deepstack=False, attention='flash_attention_2', parity=parity,
                        seconds=time.time()-t0, plan_sha256=plan_hash)
                    handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                    completed += 1
                    if args.stage=='smoke' or completed % 50 == 0:
                        print('PROGRESS',method,args.shard,bench,i,retention,completed,
                              'elapsed',round(time.time()-started,1),'score',round(scored['score'],4),flush=True)
                del inputs, item
    if hook:
        hook.remove()
    print('COMPLETE',method,args.stage,args.shard,completed,flush=True)


def aggregate(stage='full', require_complete=False):
    records = [r for p in (OUT/stage).glob('*.jsonl') for r in load_rows(p)]
    keys = {(r['method'],r['retention'],r['benchmark'],r['index']) for r in records}
    assert len(keys)==len(records), 'Duplicate predictions'
    hashes = {}
    for r in records:
        key = (r['benchmark'],r['index'])
        if key in hashes:
            assert hashes[key]==r['input_sha256'], ('Mismatched inputs', key)
        hashes[key]=r['input_sha256']
    if require_complete:
        expected = {(m,ret,b,i) for m in METHODS for ret in ratios(m) for b in BENCHES
                    for i in selected_indices(stage,0,1,[json.loads(l) for l in manifest(b).read_text().splitlines()])}
        assert keys == expected, (len(keys),len(expected),list(expected-keys)[:5])
    rows = []
    for m in METHODS:
        for ret in ratios(m):
            row = dict(method=m,retention=ret)
            for b in BENCHES:
                selected=[r for r in records if (r['method'],r['retention'],r['benchmark'])==(m,ret,b)]
                row[b] = 100*sum(r['score'] for r in selected)/len(selected) if selected else None
                row[b+'_samples']=len(selected)
            row['avg']=sum(row[b] for b in BENCHES)/3 if all(row[b] is not None for b in BENCHES) else None
            rows.append(row)
    dump(OUT/f'{stage}_summary.json', dict(complete=require_complete,predictions=len(records),rows=rows))
    if stage=='full':
        with (OUT/'RESULTS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        lines=['# Document benchmark evaluation','',
            'Qwen3-VL-4B-Instruct, FA2, DeepStack off, BF16. Same fixed 1000 examples per task.',
            'ChartQA: test, 500 human + 500 augmented, Relaxed Accuracy. DocVQA/InfographicVQA: validation, ANLS. Scores x100; AVG is their unrounded arithmetic mean.',
            'Retention means the existing post-pruning parameter, not the sum over all layers. Adapter rows retain all visual K/V tokens.',
            'Status: '+('COMPLETE' if require_complete else 'IN PROGRESS; incomplete cells are provisional.'),'',
            '| Method | Retention | ChartQA | DocVQA | InfographicVQA | AVG |',
            '|---|---:|---:|---:|---:|---:|']
        for row in rows:
            values=[]
            for b in BENCHES:
                value=row[b]
                values.append('—' if value is None else f"{value:.2f}"+(f" ({row[b+'_samples']}/1000)" if row[b+'_samples']!=1000 else ''))
            values.append('—' if row['avg'] is None else f"{row['avg']:.2f}")
            lines.append('| '+' | '.join([row['method'],f"{row['retention']:.0%}",*values])+' |')
        (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return len(records)


def run_stage(stage, shards):
    jobs=[(m,s) for m in METHODS for s in range(shards)]
    active={}; failed=[]; status_path=OUT/'status.json'
    while jobs or active:
        for gpu in range(8):
            if gpu in active or not jobs:
                continue
            method,shard=jobs.pop(0)
            logfile=OUT/'logs'/f'{stage}_{method}_{shard}.log'
            handle=logfile.open('a')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--method',method,
                 '--stage',stage,'--shard',str(shard),'--shards',str(shards)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(p,handle,method,shard)
        for gpu,(p,handle,method,shard) in list(active.items()):
            if p.poll() is not None:
                handle.close()
                if p.returncode:
                    failed.append(dict(method=method,shard=shard,returncode=p.returncode))
                del active[gpu]
        count=aggregate(stage)
        dump(status_path,dict(state='running',stage=stage,predictions=count,pending=len(jobs),failed=failed,
            active=[dict(gpu=g,pid=p.pid,method=m,shard=s) for g,(p,h,m,s) in active.items()],updated=time.time()))
        if jobs or active:
            time.sleep(15)
    if failed:
        dump(status_path,dict(state='failed',stage=stage,failed=failed,updated=time.time()))
        raise RuntimeError(failed)
    aggregate(stage,require_complete=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--stage',choices=['smoke','full'],default='smoke')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--aggregate',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    if args.worker:
        worker(args)
    elif args.aggregate:
        aggregate('full',require_complete=True)
    else:
        import fcntl
        OUT.mkdir(parents=True,exist_ok=True)
        lock=(OUT/'.launcher.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight()
        run_stage('smoke',1)
        if not args.smoke_only:
            run_stage('full',2)
            dump(OUT/'status.json',dict(state='complete',predictions=45000,cells=45,finished=time.time()))


if __name__=='__main__':
    main()
