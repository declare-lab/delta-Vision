"""Evaluate five completed rank checkpoints under the accepted 2026-09-22 protocol."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'artifacts/eval/qwen4b_adapter_image_reproduction_20260922'
TRAIN=ROOT/'artifacts/experiments/pixmo_static_rank_sweep/qwen3vl4b_pixmo_static_kl_rank_sweep_2000_20260921'
RANKS=[32,64,256,512,1024]
COMMIT='7f266415a28b3801339da93211a8fd9de2ff319e'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def prepare(run):
    import torch
    torch.set_num_threads(4)
    assert json.loads((TRAIN/'status.json').read_text())['state']=='complete'
    assert json.loads((REFERENCE/'status.json').read_text())['state']=='complete'
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','data','rows','logs','audits']:(run/d).mkdir(parents=True)
    ref=json.loads((REFERENCE/'config.json').read_text())
    cfg=dict(reference_run=str(REFERENCE),training_run=str(TRAIN),ranks=RANKS,checkpoints={},evaluation={},
        model_path=ref['model_path'],attention=ref['attention'],deepstack=False,dtype='bfloat16',seed=44,shards=8,
        cohort='historical accepted by user on 2026-09-22; exact same IDs and order',
        decoding=ref['decoding'],scorer_commit=COMMIT,scoring='Exact Git src/benchmarks.py; POPE/MME question accuracy, VQAv2 soft score, AVG mean of nine unrounded scores.',
        reference_summary=json.loads((REFERENCE/'summary.json').read_text())['historical'],expected_predictions=5*8765)
    frozen_hashes=json.loads((REFERENCE/'source_hashes.json').read_text())
    for p in (REFERENCE/'source/src').glob('*.py'):
        assert sha(p)==frozen_hashes[str(p.relative_to(REFERENCE))]
        shutil.copy2(p,run/'source/src'/p.name)
    scorer=subprocess.check_output(['git','show',f'{COMMIT}:src/benchmarks.py'],cwd=ROOT)
    (run/'source/src/scoring_reference_7f266415.py').write_bytes(scorer)
    assert hashlib.sha256(scorer).hexdigest()==json.loads((REFERENCE/'extraction_7f266415/audit.json').read_text())['source_sha256']
    reference_inputs={}
    for p in (REFERENCE/'rows').glob('shard*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['method']!='native':continue
            for member in r['members']:
                if member['cohort']=='historical':reference_inputs[r['benchmark'],member['index']]=r['input_sha256']
    for b,info in ref['evaluation'].items():
        union=[json.loads(l) for l in Path(info['path']).read_text().splitlines()];selected={}
        for row,members in zip(union,info['members']):
            for member in members:
                if member['cohort']=='historical':selected[member['index']]=row
        n=info['cohort_counts']['historical'];assert sorted(selected)==list(range(n))
        path=run/'data'/f'{b}.jsonl';path.write_text(''.join(json.dumps(selected[i],ensure_ascii=False)+'\n' for i in range(n)))
        cfg['evaluation'][b]=dict(path=str(path),sha256=sha(path),samples=n,image_root=info['image_root'],max_new_tokens=info['max_new_tokens'],
            reference_input_hashes=[reference_inputs[b,i] for i in range(n)])
    for rank in RANKS:
        status=json.loads((TRAIN/f'rank{rank}/status.json').read_text());ckpt=Path(status['checkpoint'])
        assert status['state']=='complete' and sha(ckpt)==status['checkpoint_sha256']
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert saved['global_step']==2000 and saved['args']['visual_adapter_rank']==rank
        for layer in range(36):
            assert list(saved['state_dict'][f'visual_adapter_down.{layer}.weight'].shape)==[rank,2560]
            assert list(saved['state_dict'][f'visual_adapter_up.{layer}.weight'].shape)==[2560,rank]
        assert all(torch.isfinite(t).all() for t in saved['state_dict'].values())
        cfg['checkpoints'][str(rank)]=dict(path=str(ckpt),sha256=status['checkpoint_sha256'],step=2000,parameters=sum(t.numel() for t in saved['state_dict'].values()))
        del saved
    shutil.copytree(ROOT/'analysis', run/'source/analysis', dirs_exist_ok=True)
    hashes={str(p.relative_to(run)):sha(p) for p in (run/'source').rglob('*.py')}
    dump(run/'config.json',cfg);dump(run/'source_hashes.json',hashes);dump(run/'status.json',dict(state='prepared',expected=cfg['expected_predictions']))


def worker(run,shard):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.model import load_frozen_qwen3vl,load_qwen_embedding_adapter_checkpoint,build_qwen_initial_context,prepare_qwen_embedding_adapter_inputs,qwen_embedding_adapter_prefill_cache_prepared
    from src.data import QwenBenchmarkDataset
    from analysis.table10_training_objective.native_initial_visual_eval import inputs_from_item
    from src.evaluate import generate_adapter_qwen_decode_cache, _eos_token_ids
    from src.scoring_reference_7f266415 import score_prediction,get_benchmark_spec
    cfg=json.loads((run/'config.json').read_text())
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/p)==h
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);device=torch.device('cuda:0')
    processor,model=load_frozen_qwen3vl(cfg['model_path'],torch.bfloat16,device,'flash_attention_2')
    model._adapter_attention_implementation='flash_attention_2'
    assert model.model.visual.deepstack_visual_indexes==[]
    assert model.model.language_model.config._attn_implementation=='flash_attention_2'
    adapters={}
    for rank in cfg['ranks']:
        info=cfg['checkpoints'][str(rank)];assert sha(info['path'])==info['sha256']
        adapter,meta=load_qwen_embedding_adapter_checkpoint(info['path'],model.model.language_model,device,torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected'] and meta['global_step']==2000 and adapter.mode=='embedding_adapter'
        assert sum(p.numel() for p in adapter.parameters())==info['parameters']
        adapters[rank]=adapter
    eos=sorted(_eos_token_ids(processor.tokenizer));path=run/'rows'/f'shard{shard}.jsonl';done=set()
    if path.exists():
        for l in path.read_text().splitlines():
            r=json.loads(l);done.add((r['benchmark'],r['sample'],r['rank']))
    count=0;start=time.time()
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for b,info in cfg['evaluation'].items():
            assert sha(info['path'])==info['sha256']
            ds=QwenBenchmarkDataset(info['path'],processor,b,data_root=info['image_root'])
            assert len(ds)==info['samples']
            for i in range(shard,len(ds),cfg['shards']):
                if all((b,i,r) in done for r in cfg['ranks']):continue
                item=ds[i];row=item['row'];inputs=inputs_from_item(item,device);digest=hashlib.sha256()
                for k,v in sorted(inputs.items()):
                    v=v.cpu().contiguous();digest.update(str((k,list(v.shape),str(v.dtype))).encode());digest.update(v.view(torch.uint8).numpy().tobytes())
                fingerprint=digest.hexdigest();assert fingerprint==info['reference_input_hashes'][i],(b,i,'input changed')
                model.model.rope_deltas=None;hidden,pos=build_qwen_initial_context(model,inputs)
                for rank,adapter in adapters.items():
                    if (b,i,rank) in done:continue
                    t0=time.time()
                    prepared=prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,pos)
                    assert prepared['attention_plan'] is not None
                    logits,mask,cache=qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prepared,logits_to_keep=1,retain_prefix_states=False)
                    assert cache['attention_implementation']=='flash_attention_2'
                    metrics={}
                    _,texts=generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,info['max_new_tokens'],
                        initial_hidden=hidden,position_ids=pos,prefill_logits=logits,prefill_text_mask=mask,decode_cache=cache,
                        decode_cache_mode='fast',decode_step_metrics=metrics)
                    tokens=metrics['generated_token_ids'][0];text=texts[0].strip()
                    score=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(benchmark=b,sample=i,rank=rank,prediction_text=text,generated_token_ids=tokens,stopped_by_eos=bool(tokens and tokens[-1] in eos),
                        max_new_tokens=info['max_new_tokens'],input_sha256=fingerprint,seconds=time.time()-t0,**score)
                    out.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
                    del prepared,logits,mask,cache
                del hidden,pos,inputs
                if count%200==0:print('PROGRESS',b,i,count,round(time.time()-start),flush=True)
    dump(run/'audits'/f'worker{shard}.json',dict(complete=True,predictions=count,checkpoint_hashes_verified=True,all_input_hashes_match_accepted_run=True,deepstack=False,attention='flash_attention_2'))
    print('COMPLETE',shard,count,round(time.time()-start),flush=True)


def report(run,complete=False):
    cfg=json.loads((run/'config.json').read_text());rows=[]
    for p in (run/'rows').glob('shard*.jsonl'):
        for l in p.read_text().splitlines():
            try:rows.append(json.loads(l))
            except json.JSONDecodeError:
                if complete:raise
    keys={(r['benchmark'],r['sample'],r['rank']) for r in rows};assert len(keys)==len(rows)
    benches=list(cfg['evaluation']);table=[]
    for label,method in [('Base (reference)','native'),('Rank128 (previous checkpoint)','adapter')]:
        table.append(dict(Method=label,**{b:cfg['reference_summary'][b][method] for b in benches},AVG=cfg['reference_summary']['AVG'][method]))
    for rank in cfg['ranks']:
        result={'Method':f'Rank{rank}'}
        for b,info in cfg['evaluation'].items():
            chosen=[r for r in rows if r['benchmark']==b and r['rank']==rank]
            if len(chosen)==info['samples']:
                assert sorted(r['sample'] for r in chosen)==list(range(info['samples']))
                result[b]=100*sum(r['score'] for r in chosen)/len(chosen)
            else:result[b]=None
        result['AVG']=sum(result[b] for b in benches)/9 if all(result[b] is not None for b in benches) else None
        table.append(result)
    order=['Base (reference)','Rank32','Rank64','Rank128 (previous checkpoint)','Rank256','Rank512','Rank1024']
    table.sort(key=lambda r:order.index(r['Method']))
    dump(run/'summary.json',table)
    with (run/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['Method',*benches,'AVG']);writer.writeheader();writer.writerows(table)
    lines=['# PixMo static embedding adapter rank sweep: nine image benchmarks','',
        'Accepted historical sample manifests: 1000 per benchmark except RealWorldQA 765. FA2, BF16, DeepStack off. Exact scorer from Git 7f266415. GQA/VQAv2 cap16, others cap8; greedy EOS stopping. POPE/MME question accuracy, VQAv2 soft score; AVG is mean of nine unrounded scores.',
        'Rank32/64/256/512/1024 are new PixMo step2000 checkpoints, trained with DeepStack off. Base and previous rank128 are copied from the accepted September22 paired re-evaluation on identical inputs; rank128 is an earlier training run, not a newly controlled rank128 training replicate.','',
        '| Method | '+' | '.join(benches)+' | AVG |','|---|'+'---:|'*10]
    for r in table:lines.append('| '+r['Method']+' | '+' | '.join('pending' if r[b] is None else f'{r[b]:.2f}' for b in [*benches,'AVG'])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if complete:
        assert len(rows)==cfg['expected_predictions']
        sys.path.insert(0,str(run/'source'))
        from src.scoring_reference_7f266415 import score_prediction,get_benchmark_spec
        data={b:[json.loads(l) for l in Path(info['path']).read_text().splitlines()] for b,info in cfg['evaluation'].items()}
        for r in rows:
            b=r['benchmark'];row=data[b][r['sample']]
            assert r['input_sha256']==cfg['evaluation'][b]['reference_input_hashes'][r['sample']]
            actual=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=r['prediction_text'],answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
            assert actual['score']==r['score'] and actual['prediction']==r['prediction']
        for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/p)==h
        dump(run/'verification.json',dict(predictions=len(rows),unique=True,complete=True,all_scores_recomputed=True,all_inputs_match_reference=True,source_hashes_verified=True))
    return len(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();run=a.run_dir.resolve()
    if a.worker is not None:return worker(run,a.worker)
    if not run.exists():prepare(run)
    if a.prepare_only:return
    cfg=json.loads((run/'config.json').read_text());jobs=[];start=time.time()
    for i in range(cfg['shards']):
        log=(run/'logs'/f'worker{i}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
        child=subprocess.Popen([sys.executable,'-u',str(run/'source/analysis/fig04_adapter_rank/eval_pixmo_static_rank_sweep.py'),'--run-dir',str(run),'--worker',str(i)],cwd=run/'source',env=env,stdout=log,stderr=subprocess.STDOUT);log.close();jobs.append(child)
    while True:
        codes=[p.poll() for p in jobs];count=report(run)
        state='failed' if any(c not in (None,0) for c in codes) else ('running' if any(c is None for c in codes) else 'complete')
        if state=='complete':report(run,True)
        dump(run/'status.json',dict(state=state,launcher_pid=os.getpid(),worker_pids=[j.pid for j in jobs],exit_codes=codes,predictions=count,expected=cfg['expected_predictions'],elapsed_seconds=time.time()-start))
        if state=='failed':
            for p in jobs:
                if p.poll() is None:p.terminate()
            raise RuntimeError('Worker failed; inspect logs')
        if state=='complete':break
        time.sleep(15)


if __name__=='__main__':main()
