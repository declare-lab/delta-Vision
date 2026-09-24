"""Reproduce Video-MME resource comparison with aligned graph residency.

Eight fresh shard processes for one method/retention at a time. All cases
visit the same 999 inputs with identical per-GPU sharding/order. FLOPs are counted in a separate
ungraphed pass; no instrumentation is present during memory/timing trials.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.benchmarking.engines.adapter import MODEL, CHECKPOINT, MANIFEST, MANIFEST_SHA, dump, file_sha

METHODS = ['fastv', 'dart', 'visionzip', 'divprune', 'zoo', 'sparsevlm']
NAMES = dict(base='Qwen3-VL-4B', adapter='Embedding Adapter', fastv='FastV', dart='DART',
             visionzip='VisionZip', divprune='DivPrune', zoo='ZOO-Prune', sparsevlm='SparseVLM')


def label(method, retention):
    return method if method in ('base', 'adapter') else f'{method}_r{round(100*retention):03d}'


def memory(args):
    from src.benchmarking.engines import adapter as ab
    from src.benchmarking.engines import base as bb
    from src.benchmarking.engines import pruning as pb
    options = SimpleNamespace(model=MODEL, checkpoint=str(CHECKPOINT), output=str(args.output),
        input_cache='test/results/adapter_exact_20260915/inputs', reference_results='',
        indices=args.indices, shard=args.shard, shards=args.shards, tokens=8, runs=3, variant='optimized',
        optimized_level='max', reference_level='exact', packed_decode_kv=True,
        method=args.method, retention=args.retention, execution='fa2_metadata',
        full_cuda_graphs=True, selector_graphs=False, graph_max_shapes=8,
        exact_norms=True, exact_rope_projections=True, fixed_greedy=True, compare_native_timing=False)
    if args.method == 'base': bb.worker(options)
    elif args.method == 'adapter': ab.worker(options)
    else: pb.worker(options)


def flops(args):
    import torch
    from src.benchmarking.engines.adapter import tensor_sha
    from src.benchmarking.common.comparison import RequestRunner, decoder_flops, cache_metrics
    from src.benchmarking.common.resource_flops import matrix_flop_counter, counts
    from src.data import QwenBenchmarkDataset
    from src.model import load_qwen_embedding_adapter_checkpoint
    from src.model_setup import disable_qwen_deepstack
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda')
    model, processor = load_baseline_model('base' if args.method == 'adapter' else args.method,
        MODEL, torch.bfloat16, device, args.retention, 'flash_attention_2')
    disable_qwen_deepstack(model)
    adapter = None
    if args.method == 'adapter':
        adapter, meta = load_qwen_embedding_adapter_checkpoint(CHECKPOINT, model.model.language_model, device, torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected']
        model._adapter_attention_implementation = 'flash_attention_2'
    runner = RequestRunner(model, args.method, adapter=adapter, decode_mode='fast')
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root='data/benchmarks/videomme', cache_dir='test/results/adapter_exact_20260915/inputs')
    assert len(dataset) == 999 and file_sha(MANIFEST) == MANIFEST_SHA
    reference = {r['index']: r for p in Path('test/results/video_base_20260915/videomme999').glob('optimized_*.jsonl')
                 for r in map(json.loads, p.read_text().splitlines())}
    vision = [0, 0]
    active = [None]
    def before(*unused):
        if active[0] is not None: vision[0] = active[0].get_total_flops()
    def after(*unused):
        if active[0] is not None: vision[1] += active[0].get_total_flops() - vision[0]
    model.model.visual.register_forward_pre_hook(before)
    model.model.visual.register_forward_hook(after)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    indices = args.indices if args.indices is not None else list(range(args.shard, 999, args.shards))
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output/f'flops_{args.shard}.jsonl'
    prior = list(map(json.loads, rows_path.read_text().splitlines())) if args.resume and rows_path.exists() else []
    assert [r['index'] for r in prior] == indices[:len(prior)], 'Invalid FLOP resume prefix'
    completed_indices = {r['index'] for r in prior}
    with torch.inference_mode(), rows_path.open('a' if args.resume else 'x', buffering=1) as out:
        for ordinal, index in enumerate(indices):
            if index in completed_indices:
                continue
            torch.manual_seed(42 + index)
            item = dataset[index]
            inputs = _qwen_inputs_from_item(item, device)
            digest = tensor_sha([inputs[k] for k in sorted(inputs)])
            assert digest == reference[index]['input_sha256'], (index, 'input mismatch')
            visual = inputs['mm_token_type_ids'][0].ne(0)
            nv = int(visual.sum()); nt = inputs['input_ids'].shape[-1] - nv
            if args.method not in ('base', 'adapter'):
                configure_baseline(model, args.method, args.retention, int(visual.nonzero()[0]), nv)
            vision[:] = [0, 0]
            if adapter is None:
                # Match fixed_greedy exactly: generation prepares all four
                # position rows, including the text row used by FA2 packing.
                model.model.rope_deltas = None
                positions = model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
            counter = matrix_flop_counter(); active[0] = counter
            with counter:
                if adapter is not None:
                    logits, cache, position = runner.prefill(inputs)
                else:
                    output = model(**inputs, position_ids=positions, use_cache=True,
                                   logits_to_keep=1, return_dict=True)
                    logits, cache = output.logits, output.past_key_values
                    position = positions[:, :, -1:] + 1
                    del output
            active[0] = None
            prefill = counter.get_total_flops(); ops = counts(counter)
            lengths = cache_metrics(cache, adapter is not None)['layer_cache_lengths']
            decode = 0; tokens = []; decode_ops = {}
            for step in range(8):
                scores = logits[:, -1].float().clone(); scores[:, eos] = -float('inf')
                token = scores.argmax(-1).view(1, 1); tokens.append(int(token))
                if step == 7: break
                counter = matrix_flop_counter()
                with counter:
                    if adapter is not None:
                        logits, cache, position = runner.step(token, cache, position)
                    else:
                        output = model(input_ids=token, position_ids=position, past_key_values=cache,
                                       use_cache=True, logits_to_keep=1, return_dict=True)
                        logits, cache = output.logits, output.past_key_values
                        position = position + 1
                        del output
                decode += counter.get_total_flops()
                for key, value in counts(counter).items(): decode_ops[key] = decode_ops.get(key, 0) + value
            old = decoder_flops(model.model.language_model.config, lengths, text_tokens=nt, image_tokens=nv,
                                adapter_rank=adapter.visual_adapter_rank if adapter is not None else None)
            row = dict(index=index, method=args.method, retention=args.retention, visual_tokens=nv, text_tokens=nt,
                input_sha256=digest, tokens=tokens, old_decoder_prefill_flops=old, vision_matrix_flops=vision[1],
                prefill_matrix_flops=prefill, decode_matrix_flops=decode, request_matrix_flops=prefill+decode,
                prefill_ops=ops, decode_ops=decode_ops, layer_lengths=lengths)
            assert any('flash_attn' in key for key in ops), ops
            out.write(json.dumps(row)+'\n')
            print(json.dumps(dict(method=args.method, done=ordinal+1, expected=len(indices), index=index)), flush=True)
            del logits, cache, position, scores, token, inputs, item


def aggregate(output, expected, cases=None):
    records = []
    cases = cases or [('base', 1.), ('adapter', 1.)] + [(m, r) for m in METHODS for r in (.05, .2)]
    expected = list(expected)
    def read_unique(paths):
        rows = [r for p in sorted(paths) for r in map(json.loads, p.read_text().splitlines())]
        indexed = {r['index']: r for r in rows}
        assert len(indexed) == len(rows), 'Duplicate request indices'
        return indexed
    for method, retention in cases:
        folder = output / label(method, retention)
        memory_files = folder.glob('optimized_*.jsonl' if method in ('base', 'adapter') else f'{label(method,retention)}/fa2_metadata_*.jsonl')
        mem = read_unique(memory_files)
        flop = read_unique(folder.glob('flops_*.jsonl'))
        assert set(mem) == set(flop) == set(expected), folder
        for index in expected:
            assert mem[index]['input_sha256'] == flop[index]['input_sha256'], (folder, index, 'input')
            assert mem[index]['tokens'] == flop[index]['tokens'], (folder, index, 'tokens')
        trials = [trial for row in mem.values() for trial in row['trials']]
        row = dict(method=NAMES[method], retention='adapter' if method=='adapter' else f'{retention:.0%}', samples=len(mem),
            peak_allocated_GiB=max(t['peak_memory_mb'] for t in trials)/1024,
            old_decoder_prefill_TFLOPs=statistics.mean(r['old_decoder_prefill_flops'] for r in flop.values())/1e12,
            vision_TFLOPs=statistics.mean(r['vision_matrix_flops'] for r in flop.values())/1e12,
            prefill_TFLOPs=statistics.mean(r['prefill_matrix_flops'] for r in flop.values())/1e12,
            decode_TFLOPs=statistics.mean(r['decode_matrix_flops'] for r in flop.values())/1e12,
            request_TFLOPs=statistics.mean(r['request_matrix_flops'] for r in flop.values())/1e12)
        records.append(row)
    for row in records:
        row['request_FLOPs_pct_base'] = 100*row['request_TFLOPs']/records[0]['request_TFLOPs']
    dump(output/'resource_summary.json', records)
    import csv
    with (output/'resource_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    lines = ['# Video-MME resource remeasurement', '', f'Completed cases: {len(records)}/14. Only completed and validated cases appear below; see status.json for overall completion.', '',
        'FA2; DeepStack off; original manifest and processed inputs; 8 generated tokens (7 cached decode forwards).',
        'One method/retention at a time across eight GPUs, one independent process per GPU; identical per-GPU sample assignment/order for all methods. Decode graph capacity 8; prefill and vision graph capacity 1; selector helpers retain only the current request.',
        'Peak = maximum warmed CUDA allocated memory over requests, including weights, working tensors and retained graph pools. GiB = 2^30 bytes.',
        'FLOPs = mean per request of matrix/conv/FA2 operations: 2 FLOPs/MAC, dense QK/AV convention even for causal attention. Includes vision, selectors, adapter, prefill/decode and LM head; excludes scalar norms/softmax/activations, sorting and indexing. This is an operation-count estimate, not a hardware instruction counter.',
        'FLOP instrumentation runs in a separate ungraphed process; it does not enter measured memory/timing. Inputs and generated tokens checked for every paired request.', '',
        '| Method | Retention | Peak (GiB) | Prefill (TFLOPs) | Decode (TFLOPs) | Request (TFLOPs) | FLOPs / base |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['method']} | {r['retention']} | {r['peak_allocated_GiB']:.3f} | {r['prefill_TFLOPs']:.4f} | {r['decode_TFLOPs']:.4f} | {r['request_TFLOPs']:.4f} | {r['request_FLOPs_pct_base']:.2f}% |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['queue', 'memory', 'flops', 'aggregate'], default='queue')
    p.add_argument('--method', choices=['base', 'adapter']+METHODS)
    p.add_argument('--retention', type=float, default=1.)
    p.add_argument('--indices', nargs='+', type=int)
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
    if args.phase == 'memory': return memory(args)
    if args.phase == 'flops': return flops(args)
    if args.phase == 'aggregate': return aggregate(args.output, args.indices or range(999))
    previous = json.loads((args.output/'status.json').read_text()) if args.resume else {}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cases = [('base',1.),('adapter',1.)]+[(m,r) for m in METHODS for r in (.05,.2)]
    # Save code hashes before launching; the snapshot is a reviewable record.
    sources = [Path(__file__).resolve()] + (list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py'))) + list((ROOT/'baselines').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)):file_sha(p) for p in sources}
    revision = 'source_resume_'+time.strftime('%Y%m%d_%H%M%S') if args.resume else 'source'
    if args.resume:
        old_protocol = json.loads((args.output/'protocol.json').read_text())
        assert old_protocol['indices'] == (args.indices or list(range(999)))
        assert old_protocol['gpus'] == args.gpus
        # Only orchestration/counting may change; preserve the measured model
        # and memory paths already validated by completed workers.
        for path, digest in old_protocol['source_sha256'].items():
            if path != str(Path(__file__).resolve().relative_to(ROOT)):
                assert file_sha(ROOT/path) == digest, (path, 'cannot reuse changed model/measurement code')
        dump(args.output/(revision+'_previous_status.json'), previous)
        dump(args.output/(revision+'_previous_protocol.json'), old_protocol)
    for source in sources:
        target = args.output/revision/source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    dump(args.output/'protocol.json', dict(manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=file_sha(CHECKPOINT), source_sha256=hashes,
        indices=args.indices or list(range(999)), shards=len(args.gpus), gpus=args.gpus,
        scheduling='sequential method/retention; all GPUs finish memory, then all finish FLOPs, then next case',
        cases=[dict(method=m, retention=r) for m,r in cases],
        decode_graph_shapes=8, prefill_graph_shapes=1, vision_graph_shapes=1, selector_cache_scope="current request only",
        attention='flash_attention_2', deepstack='off', tokens=8, runs=3,
        source_revision=revision, resumed=bool(args.resume)))
    indices = args.indices or list(range(999))
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        raise ValueError('This queue requires eight distinct GPUs for each sequential case')
    completed = []; completed_cases = list(previous.get('completed_cases', [])); running = {}; failed = []
    start = time.time() - previous.get('elapsed_s', 0.)
    active_case = None; active_phase = None
    def status(state, **extra):
        dump(args.output/'status.json', dict(state=state, elapsed_s=time.time()-start,
            active_case=active_case, phase=active_phase, completed_cases=completed_cases,
            completed=completed, failed=failed, running=[v[2] for v in running.values()],
            pending_cases=len(cases)-len(completed_cases), **extra))
    try:
        for method, retention in cases:
            if any(c['method']==method and c['retention']==retention for c in completed_cases):
                continue
            active_case = dict(method=method, retention=retention)
            folder = args.output/label(method, retention); folder.mkdir(exist_ok=True)
            for phase in ('memory', 'flops'):
                active_phase = phase
                # A barrier between phases and cases prevents method overlap.
                assert not running
                for shard, gpu in enumerate(args.gpus):
                    selected = indices[shard::len(args.gpus)]
                    if not selected: continue
                    existing = (folder/f'flops_{shard}.jsonl' if phase=='flops' else
                        folder/(f'optimized_{shard}.jsonl' if method in ('base','adapter') else
                        f'{label(method,retention)}/fa2_metadata_{shard}.jsonl'))
                    if args.resume and existing.exists():
                        rows = list(map(json.loads, existing.read_text().splitlines()))
                        assert [r['index'] for r in rows] == selected[:len(rows)], (existing, 'invalid shard prefix')
                        if len(rows) == len(selected):
                            completed.append(dict(method=method, retention=retention, phase=phase,
                                shard=shard, samples=len(selected), gpu=gpu, reused=True, returncode=0))
                            continue
                        assert phase == 'flops', (existing, 'partial memory shards require complete replay to preserve graph history')
                    command = [sys.executable,'-u',str(Path(__file__).resolve()),'--phase',phase,
                        '--method',method,'--retention',str(retention),'--output',str(folder),
                        '--shard',str(shard),'--shards',str(len(args.gpus)),
                        '--indices',*map(str,selected)]
                    if args.resume and existing.exists(): command += ['--resume']
                    log = (folder/f'{phase}_{shard}.log').open('a' if args.resume else 'w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM='false')
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    job = dict(method=method, retention=retention, phase=phase, shard=shard,
                               samples=len(selected), gpu=gpu, pid=proc.pid)
                    running[gpu] = (proc, log, job)
                while running:
                    for gpu, (proc, log, job) in list(running.items()):
                        code = proc.poll()
                        if code is None: continue
                        log.close(); del running[gpu]
                        (completed if code == 0 else failed).append(dict(job, returncode=code))
                    if failed: raise RuntimeError(f'Failed jobs: {failed}')
                    status('running')
                    if running: time.sleep(5)
            active_phase = 'validating'
            status('validating')
            for path, digest in hashes.items():
                assert file_sha(ROOT/path) == digest, (path,'source changed during run')
            aggregate(args.output, indices, cases[:len(completed_cases)+1])
            completed_cases.append(dict(method=method, retention=retention, samples=len(indices)))
            status('running')
    except BaseException as error:
        for proc, log, job in running.values():
            if proc.poll() is None: proc.terminate()
        for proc, log, job in running.values():
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()
        running.clear()
        status('failed', error=repr(error))
        raise
    active_case = None; active_phase = None
    status('complete')



if __name__ == '__main__': main()
