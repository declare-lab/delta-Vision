"""Eight-shard paired DART/DivPrune + rank128 adapter evaluation, seed44."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'artifacts/eval/all_baselines_seed44_20260922_restart'
ADAPTER_REFERENCE = ROOT/'artifacts/eval/qwen4b_adapter_image_reproduction_20260922'
COMMIT = '7f266415a28b3801339da93211a8fd9de2ff319e'
RATIOS = [.50, .20, .05]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    for name in ['source/src', 'data', 'rows', 'logs', 'audits']:
        (run/name).mkdir(parents=True)
    old = json.loads((PARENT/'config.json').read_text())
    ref = json.loads((ADAPTER_REFERENCE/'config.json').read_text())
    assert sha(ref['checkpoint']) == ref['checkpoint_sha256']
    config = dict(model_path=ref['model_path'], checkpoint=ref['checkpoint'],
        checkpoint_sha256=ref['checkpoint_sha256'], seed=44, shards=8,
        attention='flash_attention_2', deepstack=False, dtype='bfloat16',
        methods=['dart', 'divprune'], retentions=RATIOS, scorer_commit=COMMIT,
        decoding='Greedy cached decode, EOS or original cap; no answer-based stop or no-EOS penalty',
        adapter='Existing static rank128 PixMo KL step2000; no retraining',
        dart='Native layers 0/1; existing 5-image/3-text pivot selector; M_l(E_selected) in layers 2..35',
        divprune='Existing FP32 diversity selector on initial projected visual embedding; M_l(E_selected) in all layers',
        retention='Visual tokens only; exclude mandatory native DART layers 0/1; all DivPrune layers count',
        evaluation={}, original_root=str(ROOT), baseline_reference=str(PARENT),
        adapter_reference=str(ADAPTER_REFERENCE))
    data = {}
    for bench, info in old['evaluation'].items():
        assert sha(info['path']) == info['sha256']
        target = run/'data'/f'{bench}.jsonl'
        shutil.copy2(info['path'], target)
        config['evaluation'][bench] = dict(info, path=str(target))
        data[bench] = list(read_rows(target))
    files = (list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py'))) + [Path(__file__).resolve(), ROOT/'baselines/multimodal_pruning_utils.py']
    for method in config['methods']:
        files += list((ROOT/'baselines'/method/'qwen3_vl').glob('*.py'))
    for path in files:
        target = run/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    scorer = run/'source/src/scoring_reference.py'
    scorer.write_bytes(subprocess.check_output(['git', 'show', COMMIT+':src/benchmarks.py'], cwd=ROOT))
    spec = importlib.util.spec_from_file_location('reference_scorer', scorer)
    scoring = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scoring
    spec.loader.exec_module(scoring)
    references, seen = [], set()
    hashes = {b:{} for b in data}
    def add(record, method, ratio, index, source):
        b = record['benchmark']; row = data[b][index]
        scored = scoring.score_prediction(metric=scoring.get_benchmark_spec(b).metric,
            prediction_text=record['prediction_text'], answer=row.get('answer'),
            answers=row.get('answers'), choices=row.get('choices'), question=row.get('question'))
        key = method, ratio, b, index
        assert key not in seen, key
        seen.add(key)
        references.append(dict(benchmark=b, sample=index, method=method, retention=ratio,
            prediction_text=record['prediction_text'], generated_token_ids=record['generated_token_ids'],
            source_run=source, rescored_with=COMMIT, **scored))
    for method in config['methods']:
        for path in (PARENT/'rows').glob(f'qwen3-vl-4b__{method}__shard*.jsonl'):
            for record in read_rows(path):
                if record['retention'] not in [.05, .20]:
                    continue
                b, i = record['benchmark'], record['sample']
                previous = hashes[b].setdefault(str(i), record['input_sha256'])
                assert previous == record['input_sha256']
                add(record, method, record['retention'], i, str(PARENT))
    adapter_hashes = {b:{} for b in data}
    for path in (ADAPTER_REFERENCE/'rows').glob('shard*.jsonl'):
        for record in read_rows(path):
            for member in record['members']:
                if member['cohort'] != 'seed44':
                    continue
                b, i = record['benchmark'], member['index']
                assert record['max_new_tokens'] == config['evaluation'][b]['max_new_tokens']
                previous = adapter_hashes[b].setdefault(str(i), record['input_sha256'])
                assert previous == record['input_sha256']
                add(record, 'base' if record['method']=='native' else 'adapter', 1., i, str(ADAPTER_REFERENCE))
    samples = sum(len(rows) for rows in data.values())
    assert len(references) == samples*6, (len(references), samples*6)
    (run/'references.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in references))
    dump(run/'reference_input_hashes.json', dict(baseline=hashes, adapter=adapter_hashes))
    config['expected_new_predictions'] = samples*8  # six combined + two baseline50 controls
    dump(run/'config.json', config)
    dump(run/'source_hashes.json', {str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    dump(run/'status.json', dict(state='prepared', expected=config['expected_new_predictions']))
    (run/'PROTOCOL.md').write_text('''# DART/DivPrune with the existing Qwen3-VL-4B embedding adapter

Nine fixed seed44 single-image manifests: 1000 questions each, RealWorldQA 765.
Retention: 50%, 20%, 5%. BF16, FA2, DeepStack off. No retraining.
DART runs native layers 0/1 on the full sequence, selects with the existing
previous-layer K/current-H rule, and uses the selected initial E in adapter
layers 2..35. Its first two decode caches retain all visual tokens.
DivPrune selects initial projected E before the LLM; all 36 layers use the
selected E through their own adapter. All text and original M-RoPE positions stay.

Run six combinations and two baseline50 controls. Existing native/adapter100
and baseline5/20 raw predictions are rescored with pinned 7f266415 and included
as references only after input hashes are checked against the fresh inputs.
GQA/VQAv2 cap16, all other benchmarks cap8; greedy EOS/cap stopping.
MME/POPE: question accuracy; VQAv2: soft score; AVG: mean of nine unrounded scores.
Smoke checks compare adapter fast execution with independent native HF slice/
replacement hooks, including the complete generated sequence and cache lengths.
The 100% DivPrune composition must match the original adapter path exactly.
The 100% DART composition is a native-first-two-layer hybrid, not pure adapter.
''')
    report(run)


def worker(run, shard, smoke=False):
    sys.path.insert(0, str(run/'source'))
    import torch
    from src.model import (load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint,
        build_qwen_initial_context, prepare_qwen_embedding_adapter_inputs,
        qwen_embedding_adapter_prefill_cache_prepared)
    from analysis.table13_pruning_adapter.qwen_pruned_embedding_adapter import native_prefix, select_visual, adapter_prefill, native_reference_generate, compare_logits
    from analysis.table10_training_objective.native_initial_visual_eval import inputs_from_item
    from baselines.common import input_digest
    from src.data import QwenBenchmarkDataset
    from src.evaluate import generate_adapter_qwen_decode_cache, _eos_token_ids
    from src.scoring_reference import score_prediction, get_benchmark_spec
    config = json.loads((run/'config.json').read_text())
    for path, digest in json.loads((run/'source_hashes.json').read_text()).items():
        assert sha(run/'source'/path) == digest, path
    assert sha(config['checkpoint']) == config['checkpoint_sha256']
    torch.set_num_threads(2); torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    processor, model = load_frozen_qwen3vl(config['model_path'], torch.bfloat16, device, 'flash_attention_2')
    model._adapter_attention_implementation = 'flash_attention_2'
    assert model.model.visual.deepstack_visual_indexes == []
    adapter, meta = load_qwen_embedding_adapter_checkpoint(config['checkpoint'], model.model.language_model, device, torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected'] and meta['global_step'] == 2000
    assert adapter.mode == 'embedding_adapter' and adapter.visual_adapter_rank == 128
    eos = _eos_token_ids(processor.tokenizer)
    hashes = json.loads((run/'reference_input_hashes.json').read_text())
    tag = 'smoke' if smoke else f'shard{shard}'
    path = run/'rows'/f'{tag}.jsonl'
    done = {(r['benchmark'], r['sample'], r['method'], r['retention']) for r in read_rows(path)} if path.exists() else set()
    started = time.time(); written = 0; validations = []
    with torch.inference_mode(), path.open('a', buffering=1) as stream:
        for bench, info in config['evaluation'].items():
            assert sha(info['path']) == info['sha256']
            ds = QwenBenchmarkDataset(info['path'], processor, bench, data_root=info['image_root'])
            assert len(ds) == info['samples']
            for index in (range(1) if smoke else range(shard, len(ds), config['shards'])):
                jobs = [(method, ratio, True) for method in config['methods'] for ratio in RATIOS]
                jobs += [(method, .50, False) for method in config['methods']]
                if all((bench,index,m+('_adapter' if use else ''),r) in done for m,r,use in jobs):
                    continue
                item = ds[index]; row = item['row']; inputs = inputs_from_item(item, device)
                digest = input_digest(item)
                assert digest == hashes['baseline'][bench][str(index)], (bench,index,'baseline input mismatch')
                h = hashlib.sha256()
                for name, value in sorted(inputs.items()):
                    value = value.cpu().contiguous()
                    h.update(str((name,list(value.shape),str(value.dtype))).encode())
                    h.update(value.view(torch.uint8).numpy().tobytes())
                assert h.hexdigest() == hashes['adapter'][bench][str(index)], (bench,index,'adapter input mismatch')
                model.model.rope_deltas = None
                initial, positions = build_qwen_initial_context(model, inputs)
                prefix = native_prefix(model, initial, positions)
                numerical_controls = {}
                if smoke:
                    # A retained-100% DivPrune composition must be a strict no-op.
                    sel = select_visual(model,inputs,initial,'divprune',1.)
                    a,_,_,_ = adapter_prefill(model,adapter,inputs,initial,positions,'divprune',sel)
                    prep = prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],
                        inputs['attention_mask'],inputs['mm_token_type_ids'],initial,positions)
                    b,_,_ = qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prep,retain_prefix_states=False)
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                    del a,b,prep
                    # BF16 shape-dependent GEMM/FA2 roundoff also exists in the
                    # unpruned original adapter. Record same-input controls.
                    # Algebra is separately tested against FP32 at 2e-6 atol.
                    for m in config['methods']:
                        sel = select_visual(model,inputs,initial,m,1.,prefix)
                        a,_,_,_ = adapter_prefill(model,adapter,inputs,initial,positions,m,sel,prefix)
                        _,b,_ = native_reference_generate(model,adapter,initial,positions,m,sel,1,eos)
                        numerical_controls[m] = compare_logits(a,b,enforce=False)
                        del a,b
                for method, ratio, use_adapter in jobs:
                    label = method+('_adapter' if use_adapter else '')
                    key = bench,index,label,ratio
                    if key in done:
                        continue
                    begin = time.time()
                    selection = select_visual(model,inputs,initial,method,ratio,prefix)
                    if use_adapter:
                        logits, mask, cache, audit = adapter_prefill(model,adapter,inputs,initial,positions,method,selection,prefix)
                        if smoke:
                            reference, ref_logits, lengths = native_reference_generate(model,adapter,
                                initial,positions,method,selection,info['max_new_tokens'],eos)
                            control = numerical_controls[method]
                            check = compare_logits(logits,ref_logits,
                                relative_limit=max(.05,3*control['relative_rms_error']),
                                absolute_limit=max(1.,3*control['max_abs_logit_error']))
                            assert lengths == [v+len(selection[3]) for v in audit['layer_visual']]
                        metrics = {}
                        _, texts = generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,info['max_new_tokens'],
                            initial_hidden=initial,position_ids=positions,prefill_logits=logits,
                            prefill_text_mask=mask,decode_cache=cache,decode_cache_mode='fast',decode_step_metrics=metrics)
                        tokens = metrics['generated_token_ids'][0]
                        if smoke:
                            assert tokens == reference, (bench,method,ratio,tokens,reference)
                            validations.append(dict(benchmark=bench,method=method,retention=ratio,
                                cache_lengths=lengths,generated_tokens_equal=True,
                                unpruned_bf16_control=control,**check))
                        del logits,mask,cache
                    else:
                        tokens,_,lengths = native_reference_generate(model,None,initial,positions,method,
                            selection,info['max_new_tokens'],eos)
                        n,k=len(selection[2]),len(selection[0]); start=2 if method=='dart' else 0
                        visual=[v-len(selection[3]) for v in lengths]
                        assert visual == [n]*start+[k]*(36-start)
                        audit=dict(original_visual=n,retained_visual=k,layer_visual=visual,
                            excluded_full_layers=list(range(start)),selected_original_positions=selection[0].tolist(),
                            prunable_visual_ratio=k/n,all_layer_visual_ratio=sum(visual)/(36*n))
                    text = processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    scored = score_prediction(metric=get_benchmark_spec(bench).metric,prediction_text=text,
                        answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(benchmark=bench,sample=index,method=label,retention=ratio,
                        prediction_text=text,generated_token_ids=tokens,max_new_tokens=info['max_new_tokens'],
                        stop='eos' if tokens and tokens[-1] in eos else 'length',
                        input_sha256=digest,token_audit=audit,seconds=time.time()-begin,**scored)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n'); written+=1;done.add(key)
                dump(run/f'progress_{tag}.json',dict(benchmark=bench,sample=index,rows=len(done),elapsed=time.time()-started))
                if smoke or written%80 == 0:
                    print('PROGRESS',tag,bench,index,len(done),round(time.time()-started),flush=True)
                del prefix,initial,positions,inputs
    dump(run/'audits'/f'{tag}.json',dict(passed=True,rows=len(done),validations=validations,
        unchanged_adapter_100_percent=smoke,all_input_hashes_match=True,elapsed=time.time()-started))


def report(run):
    config = json.loads((run/'config.json').read_text())
    records = list(read_rows(run/'references.jsonl'))
    fresh = []
    for path in (run/'rows').glob('shard*.jsonl'):
        # Appends are line buffered. Ignore only an in-flight final partial line.
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue
                fresh.append(json.loads(line))
    records += fresh
    seen, groups = set(), {}
    for r in records:
        key = r['method'],r['retention'],r['benchmark'],r['sample']
        assert key not in seen, key
        seen.add(key);groups.setdefault(key[:3],[]).append(r['score'])
    benches = ['mmstar','realworldqa','gqa','mmb','mmb-cn','mme','pope','sqa','vqav2']
    variants = [('base',1.),('adapter',1.)]
    for ratio in RATIOS:
        variants += [(m,ratio) for m in ['dart','dart_adapter','divprune','divprune_adapter']]
    table=[]
    for method,ratio in variants:
        r=dict(method=method,retention=ratio)
        for b in benches:
            scores=groups.get((method,ratio,b),[])
            if len(scores)==config['evaluation'][b]['samples']:
                r[b]=100*sum(scores)/len(scores)
        if all(b in r for b in benches):r['AVG']=sum(r[b] for b in benches)/9
        table.append(r)
    dump(run/'summary.json',dict(new_predictions=len(fresh),expected=config['expected_new_predictions'],results=table))
    lines=['# Qwen3-VL-4B: DART / DivPrune + existing rank128 embedding adapter','',
        'FA2, DeepStack off, fixed seed44 manifests, pinned 7f266415 scoring. See PROTOCOL.md.',
        'Base/adapter100 and plain baseline5/20 are rescored historical references; baseline50 and all combinations are new.',
        '', '| Method | Retention | '+' | '.join(benches+['AVG'])+' |',
        '|---|---:|'+'---:|'*10]
    for r in table:
        lines.append('| '+r['method']+f" | {r['retention']:.0%} | "+' | '.join(f'{r[b]:.2f}' if b in r else 'pending' for b in benches+['AVG'])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return len(fresh)


def queue(run):
    config=json.loads((run/'config.json').read_text()); root=Path(config['original_root'])
    script=run/'source/analysis/table13_pruning_adapter/eval_pruned_embedding_adapter.py'
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(run/'source'))
    active=[]
    try:
        for smoke in [True,False]:
            active=[]
            for shard in ([0] if smoke else range(8)):
                tag='smoke' if smoke else f'shard{shard}'
                if (run/'audits'/f'{tag}.json').exists():continue
                cmd=[str(root/'.venv/bin/python'),'-u',str(script),'worker','--run',str(run),'--shard',str(shard)]
                if smoke:cmd+=['--smoke']
                with (run/'logs'/f'{tag}.log').open('a') as log:
                    active.append(subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in active):
                if any(p.poll() not in (None,0) for p in active):raise RuntimeError('Worker failed; inspect logs')
                dump(run/'status.json',dict(state='smoke' if smoke else 'running',pids=[p.pid for p in active if p.poll() is None],
                    new_predictions=report(run),expected=config['expected_new_predictions']))
                time.sleep(15)
            assert all(p.returncode==0 for p in active)
        actual=report(run);assert actual==config['expected_new_predictions']
        dump(run/'status.json',dict(state='complete',new_predictions=actual))
    except BaseException as exc:
        for p in active:
            if p.poll() is None:p.terminate()
        dump(run/'status.json',dict(state='failed',error=repr(exc),new_predictions=report(run)))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','worker','queue','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    if args.action=='worker':worker(run,args.shard,args.smoke)
    else:globals()[args.action](run)


if __name__=='__main__':main()
