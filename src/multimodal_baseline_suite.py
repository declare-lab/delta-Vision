"""Audited Qwen pruning evaluation. One request-wide visual budget, native positions.

Uses maintained multi-image/video dataset reader, never changes its worktree.
Greedy uncached forwards avoid compressed-cache position ambiguity in old ports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from src.multimodal_eval_inputs import benchmark_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent/'vision-kv-inject-attention-sink/src'
MODEL = '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct'
METHODS = ('base', 'embedding_adapter', 'fastv', 'dart', 'visionzip', 'sparsevlm', 'divprune', 'zoo')
BENCHES = ('muirbench', 'mmiu', 'videomme', 'mvbench')
CHECKPOINT = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
EXPERIMENT = CHECKPOINT.parents[2]
ADAPTER_CHECKPOINTS = {
    'embedding_adapter': CHECKPOINT,
    'recurrent_adapter': EXPERIMENT/'recurrent_kl/checkpoints/qwen_recurrent_embedding_adapter_step2000.pt',
    'embedding_adapter_sft': EXPERIMENT/'sft/checkpoints/qwen_embedding_adapter_step2000.pt',
    'embedding_adapter_opd': EXPERIMENT/'opd/checkpoints/qwen_embedding_adapter_step2000.pt',
    'embedding_adapter_mixed': ROOT/'artifacts/experiments/qwen_mixed_adapter/mixed60_25_15_resume1500_2000_20260911/checkpoints/qwen_embedding_adapter_step2000.pt',
    'adapter_start7': ROOT/'artifacts/experiments/qwen_three_point/20260909_025142/B_start7_direct/checkpoints/qwen_embedding_adapter_step2000.pt',
    'adapter_start16': ROOT/'artifacts/experiments/qwen_three_point/20260909_025142/A_start16_direct/checkpoints/qwen_embedding_adapter_step2000.pt',
}
ALL_METHODS = tuple(dict.fromkeys((*METHODS, *ADAPTER_CHECKPOINTS)))

def manifest_path(args, benchmark):
    if getattr(args,'manifest_map',None):
        return Path(json.loads(Path(args.manifest_map).read_text())[benchmark])
    return benchmark_manifest(benchmark)

def method_retentions(method):
    return (1.,) if method == 'base' or method in ADAPTER_CHECKPOINTS else (.2,.05)

def check_protocol(record, args):
    # Historical results omitted these fields and used different inputs/limits.
    # Reject them rather than silently relabelling or mixing their scores.
    if record.get('prompt_layout') != args.prompt_layout:
        raise ValueError('Prompt layout differs or is unrecorded; use a new output directory')
    if record.get('max_new_tokens') != args.max_new_tokens:
        raise ValueError('Generation limit differs or is unrecorded; use a new output directory')

def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def shard_complete(root,method,shard,args):
    path=Path(root)/f'{method}_shard{shard}.jsonl'
    if not path.exists():return False
    got=set()
    for line in path.read_text().splitlines():
        r=json.loads(line)
        check_protocol(r,args)
        if r.get('deepstack_enabled') is not False:return False
        if method=='fastv' and r.get('fastv_scoring')!='previous_layer_global_softmax':return False
        if method=='dart' and r.get('dart_keys')!='previous_layer_post_rope':return False
        key=(r['benchmark'],r['index'],r['retention'])
        assert key not in got
        got.add(key)
    expected=set()
    for b in args.benchmarks:
        count=min(args.limit,sum(bool(l.strip()) for l in manifest_path(args,b).open()))
        expected.update((b,i,r) for i in range(shard,count,args.shards) for r in method_retentions(method))
    return got==expected

def worker(args):
    import src
    src.__path__.insert(0, str(REFERENCE))
    import torch
    import types
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from baselines.multimodal_pruning_utils import visual_budget
    torch.set_num_threads(4)
    torch.manual_seed(42)
    os.environ['QWEN_VIDEO_SAMPLING'] = 'full_timestamp_v1'
    os.environ['QWEN_VIDEO_NUM_FRAMES'] = '8'
    plan=json.loads((Path(args.output)/'plan.json').read_text())
    for name,h in plan['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    for b in args.benchmarks:
        assert hashlib.sha256(manifest_path(args,b).read_bytes()).hexdigest()==plan['dataset_sha256'][b]
    model, processor = load_baseline_model('base' if args.method in ADAPTER_CHECKPOINTS else args.method, MODEL, torch.bfloat16, 'cuda:0', .2, 'sdpa')
    model.eval().requires_grad_(False)
    adapter=None
    if args.method in ADAPTER_CHECKPOINTS:
        from src import model as ref
        checkpoint = ADAPTER_CHECKPOINTS[args.method]
        with checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['adapter_checkpoints'][args.method]['sha256']
        adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(checkpoint,model.model.language_model,torch.device('cuda'),torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected']
        expected_mode = 'recurrent_embedding_adapter' if args.method == 'recurrent_adapter' else 'embedding_adapter'
        expected_start = {'adapter_start7': 7, 'adapter_start16': 16}.get(args.method, 0)
        assert adapter.mode==expected_mode and adapter.adapter_start_layer==expected_start
        assert adapter.active_adapter_layers==(35 if expected_start else 0)
        assert not adapter.visual_boundary_ffn and not adapter.stop_after_adapter_layers
        assert not adapter.native_ffn_carriers and adapter.native_prefix_memory=='legacy'
        adapter.eval().requires_grad_(False)
    # Same intervention for native base and every pruning port. Do not mutate
    # any training model/configuration outside this evaluation process.
    def disable_deepstack(module, positional, keyword):
        keyword = dict(keyword, deepstack_visual_embeds=None)
        module._deepstack_disabled_calls = getattr(module, '_deepstack_disabled_calls', 0) + 1
        return positional, keyword
    def reject_deepstack(*a, **kw):
        raise AssertionError('DeepStack executed in the no-DeepStack suite')
    model.model.language_model.register_forward_pre_hook(disable_deepstack, with_kwargs=True)
    model.model.language_model._deepstack_process = reject_deepstack
    root = Path(args.output)
    path = root/f'{args.method}_shard{args.shard}.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            check_protocol(r,args)
            assert r.get('deepstack_enabled') is False, 'Do not mix historical DeepStack-on results'
            if args.method=='fastv':assert r.get('fastv_scoring')=='previous_layer_global_softmax', 'Archive obsolete FastV scores before resuming'
            if args.method=='dart':assert r.get('dart_keys')=='previous_layer_post_rope', 'Archive obsolete DART scores before resuming'
            done.add((r['benchmark'],r['index'],r['retention']))
    # Reuse only vision-encoder results from this exact request. VisionZip also
    # needs its selection metadata, which its forward normally consumes.
    feature_cache = {}
    original = model.model.get_image_features
    def cached_features(*a, **kw):
        pixels = kw.get('pixel_values', a[0] if a else None)
        key = pixels.data_ptr()
        if key not in feature_cache:
            result = original(*a, **kw)
            meta = {n:getattr(model.model.visual,n,None) for n in ('_visionzip_attn_mean','_visionzip_attn_key')}
            feature_cache[key] = (result,meta)
        result, meta = feature_cache[key]
        for n,v in meta.items():
            if v is not None:setattr(model.model.visual,n,v)
        return result
    model.model.get_image_features = cached_features
    started = time.time()
    prefix_parity_checked = False
    with path.open('a', buffering=1) as file, torch.inference_mode():
        for bench in args.benchmarks:
            dataset = QwenBenchmarkDataset(str(manifest_path(args,bench)), processor,
                bench, data_root=str(ROOT/f'data/benchmarks/{bench}'), max_samples=args.limit,
                cache_dir=root/'processed'/bench,prompt_layout=args.prompt_layout)
            for index in range(args.shard,len(dataset),args.shards):
                retentions = method_retentions(args.method)
                if all((bench,index,r) in done for r in retentions):continue
                item = dataset[index]
                digest=hashlib.sha256()
                for name in ('input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
                    if torch.is_tensor(item.get(name)):
                        value=item[name].contiguous().cpu()
                        digest.update(str((name,tuple(value.shape),str(value.dtype))).encode())
                        digest.update(value.view(torch.uint8).numpy().tobytes())
                input_sha256=digest.hexdigest()
                inputs0 = _qwen_inputs_from_item(item, torch.device('cuda:0'))
                feature_cache.clear()
                visual = inputs0['mm_token_type_ids'][0].ne(0).nonzero().flatten()
                assert len(visual)>0
                if args.method=='visionzip' and 'pixel_values' in inputs0 and 'pixel_values_videos' in inputs0:
                    raise RuntimeError('VisionZip mixed image+video request requires separate vision metadata aggregation')
                for retention in retentions:
                    if (bench,index,retention) in done:continue
                    before = time.time()
                    if adapter is None:configure_baseline(model,args.method,retention,int(visual[0]),len(visual))
                    else:
                        hidden,pos=ref.build_qwen_initial_context(model,inputs0)
                        original_memories=adapter.all_visual_memories_batched
                        if adapter.adapter_start_layer:
                            # The native prefix MUST construct the anchor first.
                            # Never predict late-layer memories from initial embeddings.
                            memory_cache = {}
                            def late_memories(self, anchor, **kw):
                                if not memory_cache:
                                    memory_cache['value'] = original_memories(anchor, **kw)
                                    memory_cache['anchor_delta'] = float((anchor-hidden[:,visual].to(anchor)).float().norm())
                                return memory_cache['value']
                            adapter.all_visual_memories_batched=types.MethodType(late_memories,adapter)
                        else:
                            memory=adapter.all_visual_memories_batched(hidden[:,visual])
                            adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
                    generated=[]; inputs=dict(inputs0); first_audit=None
                    for step in range(args.max_new_tokens):
                        torch.manual_seed(42000+index)
                        model.model.rope_deltas=None
                        for module in (model.model,model.model.language_model):module._pruning_audit=[]
                        calls=getattr(model.model.language_model,'_deepstack_disabled_calls',0)
                        if adapter is None:
                            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                            assert model.model.language_model._deepstack_disabled_calls==calls+1
                        else:
                            logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                        if step==0:
                            audits=model.model._pruning_audit+model.model.language_model._pruning_audit
                            if args.method != 'base' and args.method not in ADAPTER_CHECKPOINTS:
                                assert audits, f'{args.method}: pruning did not run'
                                assert audits[0]['before_visual']==len(visual)
                                expected=visual_budget(len(visual),retention)
                                assert audits[-1]['after_visual']==expected, (args.method,audits[-1]['after_visual'],expected)
                                assert all(x['text_tokens']==inputs0['input_ids'].shape[1]-len(visual) for x in audits)
                            first_audit=[{k:v for k,v in a.items() if not k.endswith('_positions')} for a in audits]
                            if adapter is not None and adapter.adapter_start_layer:
                                memory=memory_cache['value']
                                assert memory.shape[0]==35-adapter.adapter_start_layer
                                assert memory_cache['anchor_delta']>0
                                parity=dict(native_prefix_layers=adapter.adapter_start_layer,
                                            adapter_end_exclusive=35, native_tail_start=35,
                                            anchor='native_prefix_rollout',anchor_delta=memory_cache['anchor_delta'])
                                if not prefix_parity_checked and inputs0['input_ids'].shape[1]<12000:
                                    # Independent full-sequence HF execution. Every adapted
                                    # layer receives the same predicted visual input, while
                                    # native prefix and final layer remain native.
                                    handles=[]
                                    def replace_at(layer_index):
                                        def replace(module,a,k):
                                            h=(a[0] if a else k['hidden_states']).clone()
                                            slot=min(layer_index,34)-adapter.adapter_start_layer
                                            h[:,visual]=memory[slot].to(h)
                                            return ((h,)+a[1:],k) if a else (a,dict(k,hidden_states=h))
                                        return replace
                                    try:
                                        for li in range(adapter.adapter_start_layer,36):
                                            handles.append(model.model.language_model.layers[li].register_forward_pre_hook(replace_at(li),with_kwargs=True))
                                        model.model.rope_deltas=None
                                        expected=model(**inputs0,use_cache=False,logits_to_keep=1).logits.float()
                                    finally:
                                        for handle in handles:handle.remove()
                                    actual=logits.float()
                                    parity['relative_logit_error']=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
                                    parity['kl']=float((expected.softmax(-1)*(expected.log_softmax(-1)-actual.log_softmax(-1))).sum())
                                    parity['same_argmax']=bool(actual.argmax()==expected.argmax())
                                    assert parity['relative_logit_error']<.04 and parity['kl']<.05,parity
                                    prefix_parity_checked=True
                                    print('NATIVE_PREFIX_PARITY',parity,flush=True)
                                first_audit.append(parity)
                        token=int(logits[0,-1].argmax());generated.append(token)
                        eos=model.generation_config.eos_token_id
                        if token in (eos if isinstance(eos,list) else [eos]):break
                        # Stop only at an unambiguous standalone option. Scoring
                        # otherwise uses the identical benchmark answer parser.
                        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                        choices=item.get('choices') or []
                        if text in [chr(65+i) for i in range(len(choices))]:break
                        inputs['input_ids']=torch.cat((inputs['input_ids'],torch.tensor([[token]],device='cuda:0')),dim=1)
                        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids']=torch.cat((inputs['mm_token_type_ids'],torch.zeros((1,1),device='cuda:0',dtype=inputs['mm_token_type_ids'].dtype)),dim=1)
                        if adapter is not None:
                            new=inputs['input_ids'][:,-1:]
                            hidden=torch.cat([hidden,model.model.get_input_embeddings()(new)],1)
                            pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                    if adapter is not None:
                        adapter.all_visual_memories_batched=original_memories
                        del memory
                        if adapter.adapter_start_layer:memory_cache.clear()
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True)
                    scored=score_prediction(metric=dataset.spec.metric,prediction_text=text,answer=item['answer'],
                        answers=item.get('answers'),choices=item.get('choices'),question=item['row'].get('question'))
                    record=dict(benchmark=bench,index=index,source_index=item['index'],method=args.method,
                        task=item['row'].get('task'),annotation_valid=item['row'].get('annotation_valid',True),
                        retention=retention,text=text,**scored,visual_tokens=len(visual),audit=first_audit,
                        seconds=time.time()-before,generated_tokens=len(generated),deepstack_enabled=False,
                        prompt_layout=args.prompt_layout,max_new_tokens=args.max_new_tokens,
                        input_sha256=input_sha256,dataset_sha256=plan['dataset_sha256'][bench],
                        fastv_scoring='previous_layer_global_softmax' if args.method=='fastv' else None,
                        dart_keys='previous_layer_post_rope' if args.method=='dart' else None)
                    file.write(json.dumps(record,ensure_ascii=False)+'\n')
                if index%16==args.shard:print(args.method,bench,index,'elapsed',round(time.time()-started),flush=True)
                feature_cache.clear()
    print('COMPLETE',args.method,args.shard,flush=True)

def aggregate(args, complete=False):
    root=Path(args.output); results=[];lines=['# Qwen3-VL-4B pruning baselines','',
        'Repository Qwen ports; **not official upstream multi-image/video results**. DeepStack is disabled for BOTH native base and all six pruning ports. Every forward asserts that DeepStack injection does not execute. Historical DeepStack-on base scores are not mixed into this table.', '',
        'Retention uses a single decimal-budget function (nearest integer, ties-to-even) in selectors and validation. Failed shards are retried once; persistent failures are reported without cancelling independent jobs. All expected indices must be present before this suite is called complete.', '',
        'First up to 1000 records per benchmark, all images. Video: 8 uniformly sampled full-window frames, true timestamps, no subtitles, ≤262144 pixels/frame. Visual budget is global per request (not per image/frame); all nonvisual tokens and original M-RoPE positions retained.', '',
        f'Prompt layout: {args.prompt_layout}. BF16 SDPA, deterministic greedy decoding up to {args.max_new_tokens} tokens, stop at standalone choice/EOS; fresh uncached forward on every decode step. No latency comparison is claimed. Vision features reused within a request. Zoo seed=42000+sample index.', '',
        'FastV/DART prune at zero-based layer 2. SparseVLM uses the existing absolute target budget at layers 2/6/15 (reaches budget at layer 2). DivPrune/Zoo/VisionZip prune before layer 0. VisionZip dominant:contextual = 80:20; per-frame native ViT attention statistics; contextual merging pools visual positions across the request.', '',
        'FastV saliency comes from layer 1 (the layer before pruning), last query, full causal-key softmax, then head averaging and selection of visual positions. Earlier visual-only normalization/current-layer FastV outputs are archived and excluded.', '',
        'DART pivots use layer-1 keys AFTER native K normalization and M-RoPE, matching the previous-layer source of the upstream port. Diversity uses normalized layer-1 output states. Exact fixed budgets are enforced, unlike approximate pivot quota counts in the upstream implementation.', '',
        '| Method | Retention | '+' | '.join(args.benchmarks)+' |', '|---|---:|'+'---:|'*len(args.benchmarks)]
    if (root/'sampling.json').exists():
        sampling=json.loads((root/'sampling.json').read_text())
        lines.insert(2, f"Sample selection: {sampling['strategy']}; seed={sampling['seed']}; source N={sampling['source_count']}; selected N={sampling['sample_count']}. The first-record limit below applies to this already sampled manifest, not the original corpus.")
    input_hashes={}
    for method in args.methods:
        rows=[]
        for shard in range(args.shards):
            p=root/f'{method}_shard{shard}.jsonl'
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        row=json.loads(line)
                        check_protocol(row,args)
                        key=(row['benchmark'],row['index'])
                        if key in input_hashes:assert input_hashes[key]==row['input_sha256'],f'Input mismatch: {method}, {key}'
                        input_hashes[key]=row['input_sha256']
                        rows.append(row)
                    except json.JSONDecodeError:
                        if complete:raise
        for retention in method_retentions(method):
            cells=[]
            for bench in args.benchmarks:
                sub=[r for r in rows if r['benchmark']==bench and r['retention']==retention]
                expected=min(args.limit,sum(bool(l.strip()) for l in manifest_path(args,bench).open()))
                assert len({r['index'] for r in sub})==len(sub), 'duplicate rows'
                if complete:assert sorted(r['index'] for r in sub)==list(range(expected)),(method,bench,len(sub),expected)
                acc=100*sum(r['score'] for r in sub)/len(sub) if sub else None
                valid=[r for r in sub if r.get('annotation_valid',True)]
                results.append(dict(method=method,retention=retention,benchmark=bench,n=len(sub),expected=expected,accuracy=acc,
                    valid_annotation_n=len(valid),valid_annotation_accuracy=100*sum(r['score'] for r in valid)/len(valid) if valid else None))
                cells.append(f'{acc:.2f} ({len(sub)}/{expected})' if sub else 'pending')
            lines.append('| '+method+' | '+str(round(retention*100))+'% | '+' | '.join(cells)+' |')
    dump(root/'results.json',results)
    if not args.json_only:(root/'README.md').write_text('\n'.join(lines)+'\n')

def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    previous = None
    if (root/'plan.json').exists():
        previous=json.loads((root/'plan.json').read_text())
        assert previous.get('deepstack_enabled') is False, 'Use a new output directory'
        check_protocol(previous.get('args',{}),args)
        for b in args.benchmarks:
            assert previous['dataset_sha256'].get(b)==hashlib.sha256(manifest_path(args,b).read_bytes()).hexdigest(), 'Dataset changed; use a new output directory'
    jobs=[(m,s) for m in args.methods for s in range(args.shards) if not shard_complete(root,m,s,args)]
    hashes={b:hashlib.sha256(manifest_path(args,b).read_bytes()).hexdigest() for b in args.benchmarks}
    sources=[Path(__file__),ROOT/'src/multimodal_eval_inputs.py',ROOT/'baselines/multimodal_pruning_utils.py',ROOT/'baselines/eval_baselines.py',REFERENCE/'model.py',REFERENCE/'data.py',REFERENCE/'benchmarks.py',REFERENCE/'benchmark_video_sampling.py']
    sources += list((ROOT/'baselines').glob('*/qwen3_vl/modeling_qwen3_vl_*.py'))
    with CHECKPOINT.open('rb') as f:checkpoint_hash=hashlib.file_digest(f,'sha256').hexdigest()
    adapter_hashes = {}
    for method in args.methods:
        if method in ADAPTER_CHECKPOINTS:
            checkpoint=ADAPTER_CHECKPOINTS[method]
            with checkpoint.open('rb') as f:
                adapter_hashes[method]=dict(path=str(checkpoint),sha256=hashlib.file_digest(f,'sha256').hexdigest())
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if previous is not None:
        assert previous.get('source_sha256')==source_hashes, 'Source changed; use a fresh output directory'
        assert previous.get('adapter_checkpoints')==adapter_hashes, 'Adapter checkpoints changed; use a fresh output directory'
    dump(root/'plan.json',dict(args=vars(args),dataset_sha256=hashes,model=MODEL,reference_src=str(REFERENCE),deepstack_enabled=False,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_hash,
        adapter_checkpoints=adapter_hashes, source_sha256=source_hashes,
        after_baselines='Fixed native Q readout' if args.readout_after else None))
    running={};failed=[];failure_events=[];attempts={};start=time.time()
    while jobs or running:
        for gpu in args.gpus:
            if gpu in running or not jobs:continue
            method,shard=jobs.pop(0)
            attempts[(method,shard)]=attempts.get((method,shard),0)+1
            log=(root/f'{method}_shard{shard}.log').open('a')
            command=[sys.executable,'-m','src.multimodal_baseline_suite','worker','--method',method,
                '--shard',str(shard),'--shards',str(args.shards),'--limit',str(args.limit),'--output',str(root),
                '--prompt-layout',args.prompt_layout,'--max-new-tokens',str(args.max_new_tokens),
                '--benchmarks',*args.benchmarks]
            if args.manifest_map:command.extend(['--manifest-map',args.manifest_map])
            if getattr(args,'checkpoint',None):command.extend(['--checkpoint',args.checkpoint])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[gpu]=(proc,log,method,shard)
        for gpu,(proc,log,m,s) in list(running.items()):
            code=proc.poll()
            if code is not None:
                log.close();del running[gpu]
                if code:
                    event=dict(method=m,shard=s,code=code,attempt=attempts[(m,s)])
                    failure_events.append(event)
                    print('WORKER FAILED',event,flush=True)
                    if attempts[(m,s)]<2:jobs.append((m,s))
                    else:failed.append(event)
        aggregate(args)
        dump(root/'status.json',dict(state='running',elapsed=time.time()-start,pending=len(jobs),
            active=[dict(gpu=g,pid=v[0].pid,method=v[2],shard=v[3]) for g,v in running.items()],failed=failed,failure_events=failure_events))
        if running:time.sleep(10)
    if failed:
        dump(root/'status.json',dict(state='failed',failed=failed));raise RuntimeError(failed)
    aggregate(args,complete=True)
    dump(root/'status.json',dict(state='complete',elapsed=time.time()-start))
    if args.readout_after:
        subprocess.run([sys.executable,'-m','src.fixed_q_visual_readout','run',
            '--output',str(root.parent/'fixed_q_readout_mmstar1500_20260912')],cwd=ROOT,check=True)

def make_parser():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('worker','run','aggregate'))
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=1000)
    p.add_argument('--method',choices=ALL_METHODS);p.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(METHODS))
    p.add_argument('--benchmarks',nargs='+',default=list(BENCHES));p.add_argument('--shards',type=int,default=8)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--readout-after',action='store_true')
    p.add_argument('--prompt-layout',choices=('interleaved','media_first_v1'),default='media_first_v1')
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--manifest-map')
    p.add_argument('--checkpoint',help='Explicit checkpoint for an embedding_adapter-only evaluation')
    p.add_argument('--json-only',action='store_true')
    return p

def main():
    global CHECKPOINT
    args=make_parser().parse_args()
    if args.checkpoint:
        if args.mode=='worker':
            if args.method!='embedding_adapter':raise ValueError('--checkpoint requires method embedding_adapter')
        elif args.methods!=['embedding_adapter']:
            raise ValueError('--checkpoint requires --methods embedding_adapter only')
        CHECKPOINT=Path(args.checkpoint).resolve()
        if not CHECKPOINT.is_file():raise FileNotFoundError(CHECKPOINT)
        ADAPTER_CHECKPOINTS['embedding_adapter']=CHECKPOINT
    if args.max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
    if args.mode=='worker':worker(args)
    elif args.mode=='run':run(args)
    else:aggregate(args)

if __name__=='__main__':main()
