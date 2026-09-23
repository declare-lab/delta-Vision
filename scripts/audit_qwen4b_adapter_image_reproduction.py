"""Paired re-evaluation on historical and seed44 cohorts; never tune against scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BENCHES=['mmstar','realworldqa','gqa','mmb','mmb-cn','mme','pope','sqa','vqav2']
OLD=ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl'
RANDOM=ROOT/'artifacts/eval/native_initial_visual_random44_20260921/config.json'
TARGET={'native':[64.9,71.2,61.6,87.5,87.7,84.7,89.3,93.3,80.9],
        'adapter':[54.4,62.9,56.7,84.5,82.8,80.1,87.6,82.3,78.0]}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def prepare(run):
    sys.path.insert(0,str(ROOT))
    from src.benchmarks import score_prediction,get_benchmark_spec
    run.mkdir(parents=True,exist_ok=False)
    for name in ('source/src','data','rows','logs','audits'):(run/name).mkdir(parents=True)
    oldplan=json.loads((OLD.parent/'plan.json').read_text());random=json.loads(RANDOM.read_text())
    ckpt=OLD/'checkpoints/qwen_embedding_adapter_step2000.pt'
    cfg=dict(model_path=random['model_path'],checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),
        seed=44,shards=8,attention='flash_attention_2',deepstack=False,dtype='bfloat16',
        methods=['native','adapter'],cohorts=['seed44','historical'],evaluation={},
        screenshot_reference=dict(zip(BENCHES,[{m:TARGET[m][i] for m in TARGET} for i in range(9)])),
        max_tolerance_pp=1.0,decoding='greedy, EOS or benchmark cap; no answer-based early stopping',
        scoring='current unchanged scorer; POPE/MME question accuracy, VQAv2 soft score; no missing-EOS zero rule',
        provenance='Existing September11 PixMo rank128 static KL step2000 checkpoint; August checkpoint unavailable. Historical cohort from preserved predictions; primary cohort remains random seed44.')
    score_audit={}
    for b in BENCHES:
        previous=json.loads((OLD/'eval'/b/'predictions.json').read_text())
        if isinstance(previous,dict):previous=previous['predictions']
        hist=[r['row'] for r in previous]
        randomrows=[json.loads(l) for l in Path(random['evaluation'][b]['path']).read_text().splitlines() if l.strip()]
        score_audit[b]={}
        for m in ('teacher','adapter'):
            values=[score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=r[m+'_text'],
                answer=r['row'].get('answer'),answers=r['row'].get('answers'),choices=r['row'].get('choices'),
                question=r['row'].get('question'))['score'] for r in previous]
            score_audit[b][m]=dict(old=100*sum(r[m+'_eval']['score'] for r in previous)/len(previous),
                rescored=100*sum(values)/len(values),changed=sum(abs(v-r[m+'_eval']['score'])>1e-12 for v,r in zip(values,previous)))
        merged=[];lookup={};members=[]
        for cohort,rows in [('seed44',randomrows),('historical',hist)]:
            for i,row in enumerate(rows):
                # Only identical inference AND scoring fields may share a forward.
                key=json.dumps({k:row.get(k) for k in ('image','images','question','hint','problem','choices','answer','answers')},sort_keys=True,ensure_ascii=False)
                if key not in lookup:lookup[key]=len(merged);merged.append(row);members.append([])
                members[lookup[key]].append(dict(cohort=cohort,index=i))
        path=run/'data'/f'{b}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
        cfg['evaluation'][b]=dict(path=str(path),sha256=sha(path),samples=len(merged),
            image_root=oldplan['benchmarks'][b]['data_root'],members=members,
            cohort_counts={'seed44':len(randomrows),'historical':len(hist)},max_new_tokens=random['evaluation'][b]['max_new_tokens'])
    hashes={}
    for p in (ROOT/'src').glob('*.py'):
        dest=run/'source/src'/p.name;shutil.copy2(p,dest);hashes[str(dest.relative_to(run))]=sha(dest)
    shutil.copy2(__file__,run/'source/worker.py');hashes['source/worker.py']=sha(run/'source/worker.py')
    dump(run/'source_hashes.json',hashes);dump(run/'config.json',cfg);dump(run/'historical_rescoring.json',score_audit)
    dump(run/'status.json',dict(state='prepared',expected_predictions=2*sum(v['samples'] for v in cfg['evaluation'].values())))


def worker(run,shard):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.model import (load_frozen_qwen3vl,load_qwen_embedding_adapter_checkpoint,build_qwen_initial_context,
        prepare_qwen_embedding_adapter_inputs,qwen_embedding_adapter_prefill_cache_prepared,
        qwen_embedding_adapter_logits)
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec,score_prediction
    from src.eval_benchmarks import generate_adapter_qwen_decode_cache,_eos_token_ids
    from src.native_initial_visual_eval import inputs_from_item
    cfg=json.loads((run/'config.json').read_text());assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/p)==h
    torch.set_num_threads(4);torch.manual_seed(44);device=torch.device('cuda:0')
    processor,model=load_frozen_qwen3vl(cfg['model_path'],torch.bfloat16,device,'flash_attention_2')
    model._adapter_attention_implementation='flash_attention_2'
    adapter,meta=load_qwen_embedding_adapter_checkpoint(cfg['checkpoint'],model.model.language_model,device,torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected'] and meta['global_step']==2000
    assert adapter.mode=='embedding_adapter'
    assert model.model.visual.deepstack_visual_indexes==[]
    assert model.model.language_model.config._attn_implementation=='flash_attention_2'
    eos=sorted(_eos_token_ids(processor.tokenizer));done=set();path=run/'rows'/f'shard{shard}.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);done.add((r['benchmark'],r['sample'],r['method']))
    start=time.time();count=0
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for b,info in cfg['evaluation'].items():
            assert sha(info['path'])==info['sha256']
            ds=QwenBenchmarkDataset(info['path'],processor,b,data_root=info['image_root']);audited=False
            for i in range(shard,len(ds),cfg['shards']):
                if all((b,i,m) in done for m in cfg['methods']):continue
                item=ds[i];row=item['row'];inputs=inputs_from_item(item,device);digest=hashlib.sha256()
                for k,v in sorted(inputs.items()):
                    v=v.cpu().contiguous();digest.update(str((k,list(v.shape),str(v.dtype))).encode());digest.update(v.view(torch.uint8).numpy().tobytes())
                for method in cfg['methods']:
                    if (b,i,method) in done:continue
                    model.model.rope_deltas=None;t0=time.time()
                    if method=='native':
                        generated=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=info['max_new_tokens'],eos_token_id=eos)
                        tokens=generated[0,inputs['input_ids'].shape[1]:].tolist();text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                        del generated
                    else:
                        hidden,pos=build_qwen_initial_context(model,inputs)
                        prepared=prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,pos)
                        assert prepared['attention_plan'] is not None
                        logits,mask,cache=qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prepared,logits_to_keep=1,retain_prefix_states=False)
                        assert cache['attention_implementation']=='flash_attention_2'
                        if not audited:
                            # Independent full-sequence native HF forward with per-layer memory scatter.
                            memories=adapter.all_visual_memories_batched(hidden[:,inputs['mm_token_type_ids'][0].ne(0)])
                            vis=inputs['mm_token_type_ids'][0].ne(0);handles=[]
                            def hook(l):
                                def replace(module,args,kw):
                                    h=kw.get('hidden_states',args[0] if args else None).clone();h[:,vis]=memories[l].to(h)
                                    return (args,dict(kw,hidden_states=h)) if 'hidden_states' in kw else ((h,)+args[1:],kw)
                                return replace
                            try:
                                for l,layer in enumerate(model.model.language_model.layers):handles.append(layer.register_forward_pre_hook(hook(l),with_kwargs=True))
                                model.model.rope_deltas=None;ref=model(**inputs,use_cache=False,logits_to_keep=1).logits.float()
                            finally:
                                for h in handles:h.remove()
                            actual=logits.float();relative=float((ref-actual).norm()/ref.norm());kl=float((ref.softmax(-1)*(ref.log_softmax(-1)-actual.log_softmax(-1))).sum())
                            centered_ref=ref-ref.mean(-1,keepdim=True);centered_actual=actual-actual.mean(-1,keepdim=True)
                            audit=dict(benchmark=b,sample=i,native_scatter_relative_error=relative,native_scatter_kl=kl,
                                centered_relative_error=float((centered_ref-centered_actual).norm()/centered_ref.norm()),
                                same_argmax=bool(ref.argmax()==actual.argmax()),deepstack=False,fa2=True)
                            # A common logit offset changes raw relative error but
                            # leaves probabilities unchanged. Keep the original
                            # probability-distribution bound; retain raw diagnostics.
                            assert kl<.05,audit
                            dump(run/'audits'/f'{b}_{shard}.json',audit);del ref,actual,memories;audited=True
                        metrics={}
                        _,texts=generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,info['max_new_tokens'],
                            initial_hidden=hidden,position_ids=pos,prefill_logits=logits,prefill_text_mask=mask,decode_cache=cache,
                            decode_cache_mode='fast',decode_step_metrics=metrics)
                        tokens=metrics['generated_token_ids'][0];text=texts[0].strip()
                        del hidden,pos,prepared,logits,mask,cache
                    score=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),
                        answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(benchmark=b,sample=i,method=method,members=info['members'][i],prediction_text=text,
                        generated_token_ids=tokens,stopped_by_eos=bool(tokens and tokens[-1] in eos),
                        max_new_tokens=info['max_new_tokens'],input_sha256=digest.hexdigest(),seconds=time.time()-t0,**score)
                    out.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
                if count%100==0:print('PROGRESS',b,i,count,round(time.time()-start),flush=True)
    print('COMPLETE',shard,count,round(time.time()-start),flush=True)


def report(run):
    cfg=json.loads((run/'config.json').read_text());records=[]
    for path in (run/'rows').glob('shard*.jsonl'):
        for line in path.read_text().splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:continue
    lookup={(r['benchmark'],r['sample'],r['method']):r for r in records};assert len(lookup)==len(records)
    for (b,i,m),r in lookup.items():
        other=lookup.get((b,i,'native' if m=='adapter' else 'adapter'))
        if other:assert other['input_sha256']==r['input_sha256']
    results={};lines=['# Qwen3-VL-4B and static embedding adapter paired reproduction','',
        'FA2; DeepStack OFF; current unchanged scoring. Both methods regenerate answers. Historical and seed44 cohorts are reported separately.','']
    for cohort in cfg['cohorts']:
        lines += [f'## {cohort}','','| Benchmark | N | Native | Screenshot | Δ pp | Adapter | Screenshot | Δ pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
        results[cohort]={}
        for b,info in cfg['evaluation'].items():
            vals={}
            for m in cfg['methods']:
                rows=[r for r in records if r['benchmark']==b and r['method']==m]
                expanded=[(mem['index'],r) for r in rows for mem in r['members'] if mem['cohort']==cohort]
                if len(expanded)==info['cohort_counts'][cohort]:
                    assert sorted(i for i,_ in expanded)==list(range(len(expanded)))
                    vals[m]=100*sum(r['score'] for _,r in expanded)/len(expanded)
            results[cohort][b]=vals
            t=cfg['screenshot_reference'][b];n=vals.get('native');a=vals.get('adapter')
            nums=[n,t['native'],None if n is None else n-t['native'],a,t['adapter'],None if a is None else a-t['adapter']]
            lines.append('| '+b+' | '+str(info['cohort_counts'][cohort])+' | '+' | '.join('pending' if v is None else f'{v:.2f}' for v in nums)+' |')
        if all(len(v)==2 for v in results[cohort].values()):
            results[cohort]['AVG']={m:sum(v[m] for v in results[cohort].values())/9 for m in cfg['methods']}
            vals=results[cohort]['AVG'];lines.append(f"| AVG | | {vals['native']:.2f} | 80.1 | {vals['native']-80.1:+.2f} | {vals['adapter']:.2f} | 74.4 | {vals['adapter']-74.4:+.2f} |")
        lines.append('')
    dump(run/'summary.json',results);(run/'RESULTS.md').write_text('\n'.join(lines)+'\n');return len(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();run=a.run_dir.resolve()
    if a.worker is not None:return worker(run,a.worker)
    if not run.exists():prepare(run)
    if a.prepare_only:return
    jobs=[];start=time.time()
    for i in range(8):
        log=(run/'logs'/f'worker{i}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
        child=subprocess.Popen([sys.executable,'-u',str(run/'source/worker.py'),'--run-dir',str(run),'--worker',str(i)],cwd=run/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();jobs.append(child)
    expected=2*sum(v['samples'] for v in json.loads((run/'config.json').read_text())['evaluation'].values())
    while True:
        codes=[j.poll() for j in jobs];count=report(run)
        state='failed' if any(c not in (None,0) for c in codes) else ('running' if any(c is None for c in codes) else 'complete')
        dump(run/'status.json',dict(state=state,launcher_pid=os.getpid(),worker_pids=[j.pid for j in jobs],exit_codes=codes,predictions=count,expected=expected,elapsed_seconds=time.time()-start))
        if state=='failed':
            for j in jobs:
                if j.poll() is None:j.terminate()
            raise RuntimeError('Worker failed; inspect logs')
        if state=='complete':assert count==expected;break
        time.sleep(15)


if __name__=='__main__':main()
