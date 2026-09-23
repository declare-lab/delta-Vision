"""DART rerun; exclude ALL compulsory full-retention layers (0 and 1)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BURN = Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, data):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def prepare(size='7b'):
    run = ROOT/'artifacts/eval'/(f'llava15_{size}_dart_prunable_layers_seed44_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    run.mkdir()
    for name in ['data', 'logs', 'full', 'source']:
        (run/name).mkdir()
    old = json.loads((ROOT/'artifacts/eval/divprune_fixed_multimodal_random44_five_models_20260920_112257/config.json').read_text())
    model_path=old['models'][f'llava-1.5-{size}']['path']
    mc=json.loads((Path(model_path)/'config.json').read_text())
    # The 7B checkpoint omits LlamaConfig's default num_hidden_layers=32.
    layers=mc['text_config'].get('num_hidden_layers',32)
    assert layers=={'7b':32,'13b':40}[size]
    config = dict(model=model_path, model_label=f'LlaVA-1.5-{size.upper()}', layers=layers, data={}, retentions=[.05, .2],
        seed=44, shards=8, attention='flash_attention_2', dtype='bfloat16', prompt_template='auto',
        retention_definition=f'sum visual tokens in layers 2..{layers-1} / ({layers-2} * original visual count); exclude ALL compulsory full-retention layers: 0 and 1',
        pruning_layer=2, key_source_layer=1, image_pivots=4, text_pivots=4,
        scoring='Current project text parser; mean question accuracy for MME/POPE; VQAv2 soft score; no no-EOS-zero override; unrounded nine-score mean',
        resume_burn=True, original_root=str(ROOT))
    for name, info in old['single_image'].items():
        assert sha(info['path']) == info['sha256']
        dest = run/'data'/f'{name}.jsonl'
        shutil.copy2(info['path'], dest)
        config['data'][name] = dict(info, path=str(dest))
    files = list((ROOT/'src').glob('*.py')) + [ROOT/'baselines/llava_hf_baselines.py', Path(__file__).resolve()]
    hashes = {}
    for p in files:
        rel = p.relative_to(ROOT)
        dest = run/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        hashes[str(rel)] = sha(dest)
    dump(run/'config.json', config)
    dump(run/'source_hashes.json', hashes)
    dump(run/'status.json', dict(state='prepared'))
    Path(f'/tmp/llava{size}_dart_active_run').write_text(str(run) + '\n')
    return run


def report(run):
    config = json.loads((run/'config.json').read_text())
    sums, counts = {}, {}
    for f in (run/'full').glob('shard*.jsonl'):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # Last line may be actively written by a worker.
            key = (r['retention'], r['benchmark'])
            sums[key] = sums.get(key, 0.) + r['score']
            counts[key] = counts.get(key, 0) + 1
    names = list(config['data'])
    rows = []
    for retention in config['retentions']:
        result = dict(retention=retention)
        for name in names:
            if counts.get((retention, name)) == config['data'][name]['samples']:
                result[name] = 100*sums[retention, name]/counts[retention, name]
        if all(n in result for n in names):
            result['AVG'] = sum(result[n] for n in names)/len(names)
        rows.append(result)
    dump(run/'summary.json', rows)
    layers=config.get('layers',32)
    budgets=[round(576*r) for r in config['retentions']]
    actual=[k/576 for k in budgets]
    lines = ['# '+config.get('model_label','LLaVA-1.5-7B')+' DART, seed44', '',
        f'Visual retention counts ONLY prunable layers 2–{layers-1}; BOTH compulsory full-retention layers 0 and 1 are excluded. Layers 0/1 keep 576; layers 2–{layers-1} keep {budgets[0]} ({actual[0]:.4%}) or {budgets[1]} ({actual[1]:.4%}).',
        'FA2, BF16; same paired inputs and decoding caps. MME/POPE question accuracy; VQAv2 soft accuracy. AVG is the unrounded nine-benchmark arithmetic mean.', '',
        '| Requested retention | '+' | '.join(names+['AVG'])+' |', '|---|'+'---:|'*(len(names)+1)]
    for r in rows:
        cells = [f'{r[n]:.2f}' if n in r else f"pending ({counts.get((r['retention'],n),0)})" for n in names]
        cells.append(f"{r['AVG']:.2f}" if 'AVG' in r else 'pending')
        lines.append('| '+f"{r['retention']:.0%}"+' | '+' | '.join(cells)+' |')
    lines += ['', 'Old 5% (2026-08-25) and 20% (2026-09-18) records have different/unverified implementations and budgets; they are historical references, not paired controls.']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (run/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['retention',*names,'AVG']);w.writeheader();w.writerows(rows)
    return sum(counts.values()), rows


def validate_selector():
    import torch
    from torch.nn import functional as F
    from src.llava_dart_corrected import select_dart, retained_budget
    torch.manual_seed(44)
    h = torch.randn(1, 610, 32)
    k = torch.randn(1, 4, 610, 8)
    for retention, expected in [(0.05,29),(.2,115),(1.,576)]:
        keep=retained_budget(576,32,retention);assert keep==expected
        actual, _ = select_dart(h,k,4,576,keep,torch.nn.Identity())
        if keep==576:continue
        flat=k.permute(0,2,1,3).reshape(1,610,32)
        ip=(flat[0,4:580].abs().sum(-1).topk(4).indices+4).tolist()
        tp=(flat[0,580:].abs().sum(-1).topk(4).indices+580).tolist()
        selected=set(ip);pivots=sorted(ip+tp)
        for j,p in enumerate(pivots):
            quota=(keep-len(selected)+len(pivots)-j-1)//(len(pivots)-j)
            cand=sorted(set(range(4,580))-selected)
            score=-F.cosine_similarity(h[0,p],h[0,cand],dim=-1)
            selected.update(cand[x] for x in score.topk(quota).indices.tolist())
        assert actual.tolist()==sorted(selected)
    assert retained_budget(576,40,.05)==29
    assert retained_budget(576,40,.2)==115


def worker(run, shard, validate=False):
    import torch
    from src.model import load_frozen_llava, llava_projected_image_features
    from src.data import LlavaBenchmarkDataset
    from src.benchmarks import score_prediction, get_benchmark_spec
    from src.divprune_rerun import input_digest
    from src.llava_dart_corrected import DartDecoder, native_pruning_reference
    from baselines.llava_hf_baselines import build_llava_inputs_embeds_with_image_span, generate_llava_baseline
    c=json.loads((run/'config.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32=False
    processor,model=load_frozen_llava(c['model'],dtype=torch.bfloat16,device='cuda:0',attn_implementation='flash_attention_2')
    eos=processor.tokenizer.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    output=run/'validation_samples.jsonl' if validate else run/'full'/f'shard{shard}.jsonl'
    done=set()
    if output.exists():
        done={(r['benchmark'],r['sample'],r['retention']) for r in map(json.loads,output.read_text().splitlines())}
    started=time.time();processed=len(done);validations=[]
    if validate:validate_selector()
    with torch.inference_mode(),output.open('a',buffering=1) as f:
        for name,info in c['data'].items():
            assert sha(info['path'])==info['sha256']
            dataset=LlavaBenchmarkDataset(info['path'],processor,name,data_root=info['image_root'],prompt_template=c['prompt_template'])
            assert len(dataset)==info['samples']
            indices=range(1) if validate else range(shard,len(dataset),8)
            for index in indices:
                if all((name,index,r) in done for r in c['retentions']):continue
                item=dataset[index];row=item['row'];digest=input_digest(item)
                ids=item['input_ids'][None].cuda();mask=item['attention_mask'][None].cuda();pixels=item['pixel_values'][None].cuda()
                memory=llava_projected_image_features(model,pixels)
                embeds,_,start,length=build_llava_inputs_embeds_with_image_span(model,input_ids=ids,attention_mask=mask,image_token_id=model.config.image_token_index,visual_memory=memory)
                assert length==576 and len(model.model.language_model.layers)==c['layers']
                decoder=DartDecoder(model)
                if validate:
                    logits=decoder.prefill(embeds,start,length,1.)
                    native=model(input_ids=ids,attention_mask=mask,pixel_values=pixels,use_cache=True,logits_to_keep=1).logits[:,-1]
                    torch.testing.assert_close(logits,native,rtol=0,atol=0)
                    tokens,_=decoder.generate(embeds,start,length,1.,info['max_new_tokens'],eos)
                    expected=model.generate(input_ids=ids,attention_mask=mask,pixel_values=pixels,use_cache=True,do_sample=False,max_new_tokens=info['max_new_tokens'])[0,ids.shape[1]:].tolist()
                    assert tokens==expected,(name,'native generation mismatch',tokens,expected)
                    validations.append(dict(benchmark=name,native_logits_exact=True,native_generation_exact=True))
                for retention in c['retentions']:
                    if (name,index,retention) in done:continue
                    begin=time.time()
                    tokens,audit=decoder.generate(embeds,start,length,retention,info['max_new_tokens'],eos)
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    if validate:
                        cached=DartDecoder(model);logit=cached.prefill(embeds,start,length,retention)
                        fixed=torch.tensor(cached.audit['selected_indices'],device='cuda:0')
                        with native_pruning_reference(model,fixed,start,length,embeds.shape[1]):
                            independent=model(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],dtype=torch.long,device='cuda:0'),use_cache=True,logits_to_keep=1).logits[:,-1]
                            torch.testing.assert_close(logit,independent,rtol=0,atol=0)
                            independent_tokens=model.generate(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],dtype=torch.long,device='cuda:0'),use_cache=True,do_sample=False,max_new_tokens=info['max_new_tokens'])[0].tolist()
                            assert independent_tokens==tokens,(name,retention,'native hooked generation mismatch',tokens,independent_tokens)
                        reference=embeds;errors=[];relative_errors=[];replay_argmax=[]
                        for token in tokens[:3]:
                            assert int(logit[0].argmax())==token
                            logit=cached.decode(token)
                            reference=torch.cat((reference,model.model.language_model.embed_tokens(torch.tensor([[token]],device='cuda:0'))),1)
                            fresh=DartDecoder(model).prefill(reference,start,length,retention,fixed_indices=fixed)
                            err=float((logit.float()-fresh.float()).abs().max());errors.append(err)
                            # BF16 GEMMs use different shapes in cached q_len=1
                            # versus complete-prefix replay. Check the full-vector
                            # error and greedy decision, not relative errors of
                            # individual near-zero vocabulary logits.
                            relative=float((logit.float()-fresh.float()).norm()/fresh.float().norm().clamp_min(1e-12))
                            relative_errors.append(relative)
                            assert relative < .1 and err < 2.,(name,retention,relative,err)
                            replay_argmax.append(int(logit.argmax())==int(fresh.argmax()))
                        validations.append(dict(benchmark=name,retention=retention,native_hook_prefill_logits_exact=True,native_hook_generation_exact=True,cached_replay_max_logit_error=max(errors),cached_replay_relative_error=max(relative_errors),cached_replay_argmax_equal=all(replay_argmax)))
                        if name=='mmstar':
                            routed=generate_llava_baseline(model,processor,input_ids=ids,attention_mask=mask,pixel_values=pixels,image_token_id=model.config.image_token_index,method='dart',retention=retention,max_new_tokens=info['max_new_tokens'])
                            assert routed==text
                    score=score_prediction(metric=get_benchmark_spec(name).metric,prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(benchmark=name,sample=index,source_index=row.get('index'),retention=retention,text=text,**score,
                        generated_token_ids=tokens,input_sha256=digest,max_new_tokens=info['max_new_tokens'],
                        stop='eos' if tokens[-1] in eos else 'length',token_audit=audit,seconds=time.time()-begin)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');processed+=1
                if validate or processed%20==0:
                    dump(run/(f'progress{shard}.json' if not validate else 'validation_progress.json'),dict(processed=processed,benchmark=name,sample=index,elapsed_s=time.time()-started))
                    print(name,index,processed,round(time.time()-started,1),flush=True)
                del decoder,embeds,memory,ids,mask,pixels,item
    if validate:dump(run/'validation.json',dict(passed=True,selector_reference_passed=True,checks=validations))
    else:dump(run/'full'/f'shard{shard}.done.json',dict(completed=processed,elapsed_s=time.time()-started))


def execute(run):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root'])
    for path,digest in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/path)==digest
    env=dict(os.environ,PYTHONPATH=str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    python=str(root/'.venv/bin/python');script=str(run/'source/scripts/rerun_llava7b_dart.py')
    active=[];started=time.time()
    try:
        if not (run/'validation.json').exists():
            dump(run/'status.json',dict(state='validating',pid=os.getpid()))
            with (run/'logs/validation.log').open('a') as log:
                p=subprocess.Popen([python,script,'--run',str(run),'--worker','--validate'],cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
                active.append((p,log));rc=p.wait();active.clear();assert rc==0,'Validation failed; see validation.log'
        assert json.loads((run/'validation.json').read_text())['passed']
        for gpu in range(8):
            if (run/'full'/f'shard{gpu}.done.json').exists():continue
            log=(run/'logs'/f'shard{gpu}.log').open('a')
            p=subprocess.Popen([python,script,'--run',str(run),'--worker','--shard',str(gpu)],cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            active.append((p,log))
        while active:
            for p,log in list(active):
                if p.poll() is not None:
                    log.close();active.remove((p,log));assert p.returncode==0,f'Worker failed: {p.pid}'
            count,_=report(run)
            dump(run/'status.json',dict(state='running',completed=count,total=17530,pid=os.getpid(),worker_pids=[p.pid for p,_ in active],elapsed_s=time.time()-started))
            if active:time.sleep(20)
        count,rows=report(run)
        assert count==17530 and all('AVG' in r for r in rows)
        # Verify one and only one paired result for every selected question.
        records=[json.loads(line) for f in (run/'full').glob('shard*.jsonl') for line in f.read_text().splitlines()]
        by_key={(r['benchmark'],r['sample'],r['retention']):r for r in records};assert len(by_key)==len(records)
        for name,info in c['data'].items():
            for i in range(info['samples']):
                a,b=[by_key[name,i,r] for r in c['retentions']];assert a['input_sha256']==b['input_sha256']
        dump(run/'status.json',dict(state='complete',completed=count,total=17530,elapsed_s=time.time()-started))
    except Exception as exc:
        for p,log in active:
            if p.poll() is None:p.terminate()
            p.wait();log.close()
        dump(run/'status.json',dict(state='failed',error=repr(exc)));raise
    finally:
        if c['resume_burn'] and json.loads((run/'status.json').read_text())['state']=='complete':
            time.sleep(5)
            r=subprocess.run([python,str(BURN),'start'],capture_output=True,text=True)
            dump(run/'burn_resume.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path);ap.add_argument('--model-size',choices=['7b','13b'],default='7b');ap.add_argument('--worker',action='store_true');ap.add_argument('--validate',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    if a.worker:worker(a.run,a.shard,a.validate)
    else:
        run=a.run or prepare(a.model_size);print(run,flush=True)
        if not a.prepare_only:execute(run)
