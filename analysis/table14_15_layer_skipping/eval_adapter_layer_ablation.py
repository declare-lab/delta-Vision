"""Disable visual KV in configurable first/last adapter layers; nine fixed benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from analysis.table13_pruning_adapter.eval_pruned_embedding_adapter import sha, dump, read_rows

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'artifacts/eval/qwen4b_pruned_embedding_adapter_20260923'


def prepare(run,first_layers=5,last_layers=10,compare_run=None):
    if first_layers < 0 or last_layers < 0 or first_layers+last_layers >= 36:
        raise ValueError('Keep at least one active visual layer; first/last groups must not overlap')
    blocked=list(range(first_layers))+list(range(36-last_layers,36))
    active=list(range(first_layers,36-last_layers))
    method=f'adapter_first{first_layers}_last{last_layers}_off'
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/scripts','data','rows','logs','audits']:(run/d).mkdir(parents=True)
    c=json.loads((PARENT/'config.json').read_text())
    for key in ['methods','retentions','dart','divprune','retention']:
        c.pop(key,None)
    c.update(blocked_visual_layers=blocked,active_visual_layers=active,
        first_layers=first_layers,last_layers=last_layers,
        methods=[method],reference_methods=['base','adapter'],visual_token_retention_in_active_layers=1.,
        experiment=f'No visual KV in first{first_layers}/last{last_layers} layers; all text layers execute; no token pruning',
        original_root=str(ROOT),wait_for_run=str(PARENT),
        expected_new_predictions=sum(i['samples'] for i in c['evaluation'].values()))
    for b,info in c['evaluation'].items():
        assert sha(info['path'])==info['sha256']
        dest=run/'data'/f'{b}.jsonl';shutil.copy2(info['path'],dest);info['path']=str(dest)
    sources=(list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py')))+[Path(__file__).resolve(),ROOT/'analysis/table13_pruning_adapter/eval_pruned_embedding_adapter.py',
        ROOT/'baselines/multimodal_pruning_utils.py']
    for method in ['dart','divprune']:sources+=list((ROOT/'baselines'/method/'qwen3_vl').glob('*.py'))
    for p in sources:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(PARENT/'source/src/scoring_reference.py',run/'source/src/scoring_reference.py')
    shutil.copy2(PARENT/'reference_input_hashes.json',run/'reference_input_hashes.json')
    refs=[r for r in read_rows(PARENT/'references.jsonl') if r['method'] in ['base','adapter']]
    assert len(refs)==2*c['expected_new_predictions']
    if compare_run is not None:
        compare_run=Path(compare_run).resolve()
        previous=json.loads((compare_run/'config.json').read_text())
        assert json.loads((compare_run/'status.json').read_text())['state']=='complete'
        for key in ['checkpoint_sha256','scorer_commit','seed','attention','deepstack','dtype','decoding']:
            assert c[key]==previous[key],key
        for b,info in c['evaluation'].items():
            assert info['sha256']==previous['evaluation'][b]['sha256']
            assert info['max_new_tokens']==previous['evaluation'][b]['max_new_tokens']
        prev_method=previous['methods'][0];assert prev_method!=method
        extra=[dict(r,source_run=str(compare_run)) for p in sorted((compare_run/'rows').glob('shard*.jsonl')) for r in read_rows(p)]
        assert len(extra)==c['expected_new_predictions'] and all(r['method']==prev_method for r in extra)
        refs+=extra;c['reference_methods'].append(prev_method);c['comparison_run']=str(compare_run)
    (run/'references.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs))
    dump(run/'config.json',c)
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    dump(run/'status.json',dict(state='prepared',expected=c['expected_new_predictions']))
    (run/'PROTOCOL.md').write_text(f'''# Embedding adapter: remove first{first_layers} + last{last_layers} visual injection

Existing Qwen3-VL-4B static rank128 PixMo KL step2000 checkpoint, no retraining.
Blocked layers {blocked}: skip adapter/visual K,V projections; zero-length visual KV
caches; text attends only causal text keys, renormalized. Text attention/FFN
still run in all36 layers. Active layers {active} use every visual token through M_l(E).
No visual token selection, and no initial-E fallback in the disabled layers.
Both prefill and decode obey the same block; original M-RoPE positions remain.

Same seed44 nine manifests/checkpoint/prompts/scorer as the concurrent pruning
composition experiment. RealWorldQA765, all other benchmarks1000. FA2/BF16,
DeepStack off; greedy EOS/cap; GQA/VQAv2 cap16, others8; pinned7f266415 scoring.
Compare to same-input historical full adapter and base, rescored from raw text.
Input hashes for both references must match on every sample. AVG averages nine
unrounded scores. This is an accuracy experiment, not concurrent speed timing.

Validation uses a separate native full-cache path which removes visual keys
only at attention time (including decode), plus CPU FP32 oracle tests. Disabled
adapter modules must never execute in production; all36 text FFNs must execute.
''')
    report(run)


def worker(run,shard,smoke):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.model import (load_frozen_qwen3vl,load_qwen_embedding_adapter_checkpoint,build_qwen_initial_context,
        prepare_qwen_embedding_adapter_inputs,qwen_embedding_adapter_prefill_cache_prepared)
    from analysis.table10_training_objective.native_initial_visual_eval import inputs_from_item
    from baselines.common import input_digest
    from src.data import QwenBenchmarkDataset
    from src.evaluate import generate_adapter_qwen_decode_cache, _eos_token_ids
    from analysis.table13_pruning_adapter.qwen_pruned_embedding_adapter import select_visual, native_reference_generate, compare_logits
    from analysis.table14_15_layer_skipping.qwen_adapter_layer_blocking import native_blocked_visual_attention
    from src.scoring_reference import score_prediction,get_benchmark_spec
    c=json.loads((run/'config.json').read_text())
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h,p
    assert sha(c['checkpoint'])==c['checkpoint_sha256']
    torch.set_num_threads(2);torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    processor,model=load_frozen_qwen3vl(c['model_path'],torch.bfloat16,device,'flash_attention_2')
    model._adapter_attention_implementation='flash_attention_2'
    assert model.model.visual.deepstack_visual_indexes==[]
    adapter,meta=load_qwen_embedding_adapter_checkpoint(c['checkpoint'],model.model.language_model,device,torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected'] and meta['global_step']==2000
    assert adapter.mode=='embedding_adapter' and adapter.visual_adapter_rank==128
    blocked=c['blocked_visual_layers'];active=c['active_visual_layers']
    assert len(model.model.language_model.layers)==36
    assert set(blocked).isdisjoint(active) and sorted(blocked+active)==list(range(36))
    eos=_eos_token_ids(processor.tokenizer)
    hashes=json.loads((run/'reference_input_hashes.json').read_text())
    tag='smoke' if smoke else f'shard{shard}';path=run/'rows'/f'{tag}.jsonl'
    done={(r['benchmark'],r['sample']) for r in read_rows(path)} if path.exists() else set()
    started=time.time();validations=[]
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for b,info in c['evaluation'].items():
            assert sha(info['path'])==info['sha256']
            ds=QwenBenchmarkDataset(info['path'],processor,b,data_root=info['image_root'])
            for i in (range(1) if smoke else range(shard,len(ds),8)):
                if (b,i) in done:continue
                begin=time.time();item=ds[i];row=item['row'];inputs=inputs_from_item(item,device)
                digest=input_digest(item);assert digest==hashes['baseline'][b][str(i)]
                h=hashlib.sha256()
                for name,v in sorted(inputs.items()):
                    v=v.cpu().contiguous();h.update(str((name,list(v.shape),str(v.dtype))).encode());h.update(v.view(torch.uint8).numpy().tobytes())
                assert h.hexdigest()==hashes['adapter'][b][str(i)]
                model.model.rope_deltas=None;initial,pos=build_qwen_initial_context(model,inputs)
                prepared=prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],
                    inputs['mm_token_type_ids'],initial,pos)
                calls=[];ffn_calls=[];handles=[]
                if smoke:
                    for j,down in enumerate(adapter.visual_adapter_down):
                        handles.append(down.register_forward_hook(lambda m,a,o,j=j:calls.append(j)))
                    for j,layer in enumerate(model.model.language_model.layers):
                        handles.append(layer.mlp.register_forward_hook(lambda m,a,o,j=j:ffn_calls.append(j)))
                try:
                    logits,mask,cache=qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prepared,
                        retain_prefix_states=False,blocked_visual_layers=blocked)
                finally:
                    for handle in handles:handle.remove()
                n=int(prepared['visual_memory'].shape[1]);expected=[0 if j in blocked else n for j in range(36)]
                assert [k['visual_key'].shape[2] for k in cache['layers']]==expected
                assert cache['dense_decode_ready'] and cache['blocked_visual_layers']==blocked
                if smoke:
                    assert calls==active and ffn_calls==list(range(36)),(calls,ffn_calls)
                    sel=select_visual(model,inputs,initial,'divprune',1.)
                    ordinary,_,_=qwen_embedding_adapter_prefill_cache_prepared(model,adapter,**prepared,retain_prefix_states=False)
                    _,ref0,_=native_reference_generate(model,adapter,initial,pos,'divprune',sel,1,eos)
                    control=compare_logits(ordinary,ref0,enforce=False)
                    with native_blocked_visual_attention(sel[2],initial.shape[1],blocked):
                        ref_tokens,ref_logits,_=native_reference_generate(model,adapter,initial,pos,'divprune',sel,info['max_new_tokens'],eos)
                    numeric=compare_logits(logits,ref_logits,relative_limit=max(.05,3*control['relative_rms_error']),
                        absolute_limit=max(1.,3*control['max_abs_logit_error']))
                metrics={}
                _,_=generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,info['max_new_tokens'],initial_hidden=initial,
                    position_ids=pos,prefill_logits=logits,prefill_text_mask=mask,decode_cache=cache,decode_cache_mode='fast',decode_step_metrics=metrics)
                tokens=metrics['generated_token_ids'][0]
                assert [k['visual_key'].shape[2] for k in cache['layers']]==expected
                if smoke:
                    assert tokens==ref_tokens,(b,tokens,ref_tokens)
                    validations.append(dict(benchmark=b,adapter_modules_executed=calls,text_ffns_executed=ffn_calls,
                        layer_visual=expected,generated_tokens_equal=True,unblocked_bf16_control=control,**numeric))
                text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                scored=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),
                    answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                out.write(json.dumps(dict(benchmark=b,sample=i,method=c['methods'][0],prediction_text=text,
                    generated_token_ids=tokens,max_new_tokens=info['max_new_tokens'],input_sha256=digest,
                    stop='eos' if tokens and tokens[-1] in eos else 'length',layer_visual=expected,seconds=time.time()-begin,**scored),ensure_ascii=False)+'\n')
                done.add((b,i));dump(run/f'progress_{tag}.json',dict(benchmark=b,sample=i,rows=len(done),elapsed=time.time()-started))
                if smoke or len(done)%50==0:print('PROGRESS',tag,b,i,len(done),round(time.time()-started),flush=True)
                del initial,pos,prepared,logits,mask,cache,inputs
    dump(run/'audits'/f'{tag}.json',dict(passed=True,rows=len(done),validations=validations,input_hashes_match=True,elapsed=time.time()-started))


def report(run):
    c=json.loads((run/'config.json').read_text());rows=list(read_rows(run/'references.jsonl'));fresh=[]
    for p in (run/'rows').glob('shard*.jsonl'):
        for line in p.read_text().splitlines(keepends=True):
            if line.endswith('\n'):fresh.append(json.loads(line))
    rows+=fresh;groups={};seen=set()
    for r in rows:
        k=r['method'],r['benchmark'],r['sample'];assert k not in seen;seen.add(k)
        groups.setdefault(k[:2],[]).append(r['score'])
    benches=['mmstar','realworldqa','gqa','mmb','mmb-cn','mme','pope','sqa','vqav2'];table=[]
    for m in c.get('reference_methods',['base','adapter'])+c['methods']:
        r=dict(method=m)
        for b in benches:
            values=groups.get((m,b),[])
            if len(values)==c['evaluation'][b]['samples']:r[b]=100*sum(values)/len(values)
        if all(b in r for b in benches):r['AVG']=sum(r[b] for b in benches)/9
        table.append(r)
    dump(run/'summary.json',dict(new_predictions=len(fresh),expected=c['expected_new_predictions'],results=table))
    lines=['# Qwen3-VL-4B adapter: visual injection layer ablation','',
        f"No visual token pruning. Blocked layers: {c['blocked_visual_layers']}; text runs all36 layers.",
        'Same seed44 inputs, FA2/BF16, DeepStack off, pinned7f scoring. Base/full adapter are rescored historical references.',
        '', '| Method | '+' | '.join(benches+['AVG'])+' |','|---|'+'---:|'*10]
    for r in table:lines.append('| '+r['method']+' | '+' | '.join(f'{r[b]:.2f}' if b in r else 'pending' for b in benches+['AVG'])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n');return len(fresh)


def queue(run):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root']);active=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(run/'source'))
    try:
        for smoke in [True,False]:
            if not smoke and c.get('wait_for_run'):
                parent=Path(c['wait_for_run'])
                while True:
                    state=json.loads((parent/'status.json').read_text())
                    if state['state'] in ['complete','failed']:
                        break
                    dump(run/'status.json',dict(state='waiting_for_previous_evaluation',wait_for_run=str(parent),
                        previous_state=state['state'],previous_predictions=state.get('new_predictions'),
                        new_predictions=report(run),expected=c['expected_new_predictions']))
                    time.sleep(15)
            active=[]
            for shard in ([0] if smoke else range(8)):
                tag='smoke' if smoke else f'shard{shard}'
                if (run/'audits'/f'{tag}.json').exists():continue
                cmd=[str(root/'.venv/bin/python'),'-u',str(run/'source/analysis/table14_15_layer_skipping/eval_adapter_layer_ablation.py'),'worker','--run',str(run),'--shard',str(shard)]
                if smoke:cmd+=['--smoke']
                with (run/'logs'/f'{tag}.log').open('a') as log:
                    active.append(subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in active):
                if any(p.poll() not in (None,0) for p in active):raise RuntimeError('Worker failed; see logs')
                dump(run/'status.json',dict(state='smoke' if smoke else 'running',pids=[p.pid for p in active if p.poll() is None],
                    new_predictions=report(run),expected=c['expected_new_predictions']))
                time.sleep(15)
            assert all(p.returncode==0 for p in active)
        n=report(run);assert n==c['expected_new_predictions'];dump(run/'status.json',dict(state='complete',new_predictions=n))
    except BaseException as exc:
        for p in active:
            if p.poll() is None:p.terminate()
        dump(run/'status.json',dict(state='failed',error=repr(exc),new_predictions=report(run)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','queue','report'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--first-layers',type=int,default=5);p.add_argument('--last-layers',type=int,default=10)
    p.add_argument('--compare-run',type=Path)
    a=p.parse_args();run=a.run.resolve()
    if a.action=='prepare':prepare(run,a.first_layers,a.last_layers,a.compare_run)
    elif a.action=='worker':worker(run,a.shard,a.smoke)
    else:globals()[a.action](run)
