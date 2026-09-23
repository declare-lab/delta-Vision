"""Frozen, resumable baseline accuracy queue; eight shards per model/method.

Unavailable implementations stay explicitly pending rather than being routed to
surrogate norm selectors. The runner executes every approved ready job.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
METHODS=['dart','divprune','fastv','visionzip','zoo','sparsevlm']
RATIOS=[.05,.10,.15,.20]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source','data','logs','rows']:(run/d).mkdir()
    prior=json.loads((ROOT/'artifacts/eval/divprune_fixed_multimodal_random44_five_models_20260920_112257/config.json').read_text())
    models=prior['models']
    models['qwen3.5-4b']=dict(kind='qwen35',path=str(ROOT/'models/Qwen3.5-4B'))
    # Start with the verified 4B paths; each subsequent model gets all four ratios.
    order=['qwen3-vl-4b','llava-1.5-7b','llava-1.5-13b','qwen3-vl-8b','qwen3-vl-30b-a3b','qwen3.5-4b','llava-1.6-mistral-7b']
    cfg=dict(models={k:models[k] for k in order},evaluation={},multimodal={},retentions=RATIOS,methods=METHODS,
        shards=8,seed=44,attention='flash_attention_2',dtype='bfloat16',deepstack=False,
        scorer_commit='7f266415a28b3801339da93211a8fd9de2ff319e',
        scoring='Pinned text extraction. MME/POPE per-question accuracy; VQAv2 soft; AVG mean of nine unrounded scores; no added no-EOS penalty.',
        retention='Exclude ALL compulsory unpruned layers. Count visual tokens only, including recycled tokens. Record actual per-layer counts.',
        single_image_generation='Existing prompts/caps; greedy cached generation; native EOS or cap. Qwen3.5 thinking disabled.',
        original_root=str(ROOT))
    for b,info in prior['single_image'].items():
        assert sha(info['path'])==info['sha256']
        dest=run/'data'/f'{b}.jsonl';shutil.copy2(info['path'],dest)
        cfg['evaluation'][b]=dict(info,path=str(dest))
    for b,info in prior['multimodal'].items():
        assert sha(info['path'])==info['sha256']
        dest=run/'data'/f'{b}.jsonl';shutil.copy2(info['path'],dest)
        cfg['multimodal'][b]=dict(info,path=str(dest))
    for m,info in cfg['models'].items():assert Path(info['path'],'config.json').exists(),m
    files=list((ROOT/'src').glob('*.py'))+[ROOT/'baselines'/n for n in ['eval_baselines.py','llava_hf_baselines.py','multimodal_pruning_utils.py']]
    for m in METHODS:files+=list((ROOT/'baselines'/m/'qwen3_vl').glob('*.py'))
    files+=[Path(__file__).resolve()]
    for p in files:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (run/'source/src/scoring_reference.py').write_bytes(subprocess.check_output(['git','show',cfg['scorer_commit']+':src/benchmarks.py'],cwd=ROOT))
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})
    jobs=[]
    for model in order:
        for method in METHODS:
            ready=method in ('dart','divprune') or model in ('qwen3-vl-4b','qwen3-vl-8b')
            jobs.append(dict(model=model,method=method,suite='image',ready=ready,
                reason=None if ready else 'Author-logic port and validation required; no norm-selector fallback'))
    for method in METHODS:
        jobs.append(dict(model='qwen3-vl-4b',method=method,suite='multimodal',ready=False,
            reason='Restore fixed-list preprocessing and validate multi-span pruning'))
    dump(run/'config.json',cfg);dump(run/'jobs.json',jobs)
    dump(run/'status.json',dict(state='prepared',total_configurations=7*6*4+6*4,
        total_predictions=7*6*4*8765+6*4*2949,ready_jobs=sum(j['ready'] for j in jobs)))
    (run/'PROTOCOL.md').write_text('Seed44 frozen nine-image manifests; historical fixed MuirBench1000, Video-MME999, MVBench950.\n'
        'All six methods requested for seven models at5/10/15/20%. Unimplemented ports are pending in jobs.json, never counted as finished.\n'
        'FA2 BF16 DeepStack off; pinned7f266415 text scoring; original prompts and generation limits.\n'
        'Eight GPUs shard ONE model/method at a time. Each request evaluates all four retentions on the same processed input.\n')

def worker(run,model_name,method,shard,smoke):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.scoring_reference import score_prediction,get_benchmark_spec,build_benchmark_prompt
    from src.data import QwenBenchmarkDataset,LlavaBenchmarkDataset
    from src.divprune_rerun import input_digest
    from baselines.eval_baselines import load_baseline_model,configure_baseline,_qwen_inputs_from_item
    from baselines.multimodal_pruning_utils import visual_budget
    from src.qwen_deepstack import disable_qwen_deepstack
    from src.qwen_baseline_author_comparison import AuthorCorrections,sparse_budgets
    from baselines import llava_hf_baselines as llava
    cfg=json.loads((run/'config.json').read_text());mi=cfg['models'][model_name];kind=mi['kind'];torch.set_num_threads(4)
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h,p
    torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
    correction=None;pruning=None
    if kind=='llava':
        processor,model=llava.load_llava_baseline_model(mi['path'],dtype=torch.bfloat16,device='cuda:0',attn_implementation='flash_attention_2')
        from src.model import llava_projected_image_features
        from src.llava_dart_corrected import DartDecoder,native_pruning_reference
    elif kind=='qwen35':
        from src.qwen35_embedding import install_fast_kernels
        from src.qwen35_experiment import prepare_inputs
        from src.qwen35_pruning import VisualPruningController
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        install_fast_kernels();processor=AutoProcessor.from_pretrained(mi['path'],local_files_only=True)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(mi['path'],dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True)
        disable_qwen_deepstack(model);pruning=VisualPruningController(model)
    else:
        model,processor=load_baseline_model(method,mi['path'],torch.bfloat16,'cuda:0',.05,'flash_attention_2')
        if method in ('visionzip','zoo','sparsevlm'):
            correction=AuthorCorrections(model,method,sys.modules[type(model).__module__])
    model.eval().requires_grad_(False);lm=model.model.language_model
    assert lm.config._attn_implementation=='flash_attention_2'
    counts={}
    def capture(i):
        def hook(module,args,kw):
            h=kw.get('hidden_states',args[0] if args else None)
            if i not in counts:counts[i]=int(h.shape[1])
        return hook
    # Install AFTER pruning hooks so the audit measures actual decoder input.
    handles=[layer.register_forward_pre_hook(capture(i),with_kwargs=True) for i,layer in enumerate(lm.layers)]
    tag=f'{model_name}__{method}__'+('smoke' if smoke else f'shard{shard}')
    path=run/'rows'/f'{tag}.jsonl';done=set()
    if path.exists():done={(r['benchmark'],r['sample'],r['retention']) for r in map(json.loads,path.read_text().splitlines())}
    started=time.time();written=0
    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for bi,(b,info) in enumerate(cfg['evaluation'].items()):
            assert sha(info['path'])==info['sha256']
            rows=[json.loads(s) for s in Path(info['path']).read_text().splitlines() if s]
            cls=LlavaBenchmarkDataset if kind=='llava' else QwenBenchmarkDataset
            ds=None if kind=='qwen35' else cls(info['path'],processor,b,data_root=info['image_root'])
            indices=range(1) if smoke else range(shard,len(rows),cfg['shards'])
            for i in indices:
                if all((b,i,r) in done for r in cfg['retentions']):continue
                row=rows[i]
                if kind=='qwen35':
                    inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,get_benchmark_spec(b)))
                    digest=input_digest(inputs)
                else:
                    item=ds[i];digest=input_digest(item)
                if kind=='llava':
                    ids=item['input_ids'][None].cuda();mask=item['attention_mask'][None].cuda();pixels=item['pixel_values'][None].cuda()
                    sizes=item.get('image_sizes');sizes=sizes[None].cuda() if torch.is_tensor(sizes) else sizes
                    memory=llava_projected_image_features(model,pixels,image_sizes=sizes)
                    n=int(memory.shape[1]);nt=int(ids.ne(model.config.image_token_index).sum())
                    embeds,_,start,length=llava.build_llava_inputs_embeds_with_image_span(model,input_ids=ids,attention_mask=mask,image_token_id=model.config.image_token_index,visual_memory=memory)
                    assert length==n
                    if smoke and method=='dart':
                        native_inputs=dict(input_ids=ids,attention_mask=mask,pixel_values=pixels)
                        if sizes is not None:native_inputs['image_sizes']=sizes
                        native=model(**native_inputs,use_cache=True,logits_to_keep=1).logits[:,-1]
                        same=DartDecoder(model).prefill(embeds,start,n,1.)
                        torch.testing.assert_close(native,same,atol=0,rtol=0)
                else:
                    if kind!='qwen35':inputs=_qwen_inputs_from_item(item,torch.device('cuda:0'))
                    visual=inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten();n=len(visual);nt=inputs['input_ids'].shape[1]-n
                for ratio in cfg['retentions']:
                    if (b,i,ratio) in done:continue
                    torch.manual_seed(44+bi*10000+i);counts.clear();begin=time.time();detail={}
                    if kind=='llava':
                        if method=='dart':
                            decoder=DartDecoder(model);tokens,detail=decoder.generate(embeds,start,n,ratio,info['max_new_tokens'],set(eos))
                            if smoke:
                                selected=torch.tensor(detail['selected_indices'],device='cuda:0')
                                with native_pruning_reference(model,selected,start,n,embeds.shape[1]):
                                    expected_tokens=model.generate(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],device='cuda:0',dtype=torch.long),max_new_tokens=info['max_new_tokens'],do_sample=False,use_cache=True)[0].tolist()
                                assert tokens==expected_tokens,(b,ratio,'independent native hook mismatch')
                        else:
                            assert method=='divprune','Unaudited LLaVA surrogate forbidden'
                            reduced=llava.reduce_visual_memory(memory,method=method,retention=ratio)
                            reduced_embeds,am,_,_=llava.build_llava_inputs_embeds_with_image_span(model,input_ids=ids,attention_mask=mask,image_token_id=model.config.image_token_index,visual_memory=reduced)
                            tokens=model.generate(inputs_embeds=reduced_embeds,attention_mask=am,do_sample=False,use_cache=True,max_new_tokens=info['max_new_tokens'])[0].tolist()
                    else:
                        if kind=='qwen35':context=pruning.activate(method,ratio,inputs['mm_token_type_ids'].eq(1))
                        else:
                            context=nullcontext();configure_baseline(model,method,ratio,int(visual[0]),n)
                            model.model.rope_deltas=None
                            if correction:
                                if method=='sparsevlm':correction.before_sparse=lm.config.sparse_config
                                correction.set_variant('after',ratio)
                        with context:
                            result=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=info['max_new_tokens'])
                        tokens=result[0,inputs['input_ids'].shape[1]:].tolist()
                        if pruning:detail=pruning.audit
                    excluded=(4 if kind=='qwen35' else 2) if method in ('dart','fastv') else (3 if method=='sparsevlm' else 0)
                    actual=[counts[j]-nt for j in range(len(lm.layers))]
                    if method=='sparsevlm':
                        a,c,d=sparse_budgets(n,len(lm.layers),ratio);expected=[n]*3+[a]*4+[c]*9+[d]*(len(lm.layers)-16)
                    else:expected=[n]*excluded+[visual_budget(n,ratio)]*(len(lm.layers)-excluded)
                    assert actual==expected,(model_name,method,b,i,ratio,n,actual,expected)
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    scored=score_prediction(metric=get_benchmark_spec(b).metric,prediction_text=text,answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    audit=dict(original_visual=n,layer_visual=actual,excluded_full_layers=list(range(excluded)),prunable_visual_ratio=sum(actual[excluded:])/(n*(len(actual)-excluded)),all_layer_visual_ratio=sum(actual)/(n*len(actual)),detail=detail)
                    out.write(json.dumps(dict(model=model_name,method=method,retention=ratio,benchmark=b,sample=i,source_index=row.get('index'),input_sha256=digest,prediction_text=text,generated_token_ids=tokens,max_new_tokens=info['max_new_tokens'],stop='eos' if tokens and tokens[-1] in eos else 'length',token_audit=audit,seconds=time.time()-begin,**scored),ensure_ascii=False)+'\n');written+=1
                    if correction:correction.raw=None
                if smoke or written%40==0:
                    dump(run/f'progress_{tag}.json',dict(rows=written,benchmark=b,sample=i,elapsed=time.time()-started))
                    print('PROGRESS',tag,b,i,written,round(time.time()-started),flush=True)
    dump(run/'rows'/f'{tag}.done.json',dict(passed=True,rows=written,elapsed=time.time()-started))

def report(run):
    cfg=json.loads((run/'config.json').read_text());groups={};seen=set()
    for p in (run/'rows').glob('*__shard*.jsonl'):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            k=(r['model'],r['method'],r['retention'],r['benchmark'],r['sample']);assert k not in seen,k;seen.add(k)
            groups.setdefault(k[:4],[]).append(r['score'])
    results=[];names=list(cfg['evaluation'])
    for model in cfg['models']:
        for method in cfg['methods']:
            for ratio in cfg['retentions']:
                row=dict(model=model,method=method,retention=ratio)
                for b in names:
                    scores=groups.get((model,method,ratio,b),[])
                    if len(scores)==cfg['evaluation'][b]['samples']:row[b]=100*sum(scores)/len(scores)
                if all(b in row for b in names):row['AVG']=sum(row[b] for b in names)/9
                results.append(row)
    dump(run/'summary.json',dict(predictions=len(seen),rows=results))
    lines=['# Fresh seed44 baseline rerun','', '| Model | Method | Retention | '+' | '.join(names+['AVG'])+' |','|---|---|---:|'+'---:|'*10]
    for r in results:lines.append('| '+r['model']+' | '+r['method']+f" | {r['retention']:.0%} | "+' | '.join(f'{r[b]:.2f}' if b in r else 'pending' for b in names+['AVG'])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n');return len(seen)

def queue(run):
    cfg=json.loads((run/'config.json').read_text());root=Path(cfg['original_root']);active=[]
    env=dict(os.environ,PYTHONPATH=str(root/'artifacts/dependencies/qwen35_python')+':'+str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    script=run/'source/scripts/rerun_baseline_suite.py'
    failed=[]
    try:
        for j in json.loads((run/'jobs.json').read_text()):
            if not j['ready']:continue
            assert j['suite']=='image'
            model,method=j['model'],j['method'];base=f'{model}__{method}'
            try:
                for smoke in (True,False):
                    active=[];logs=[]
                    for shard in ([0] if smoke else range(8)):
                        tag=base+'__'+('smoke' if smoke else f'shard{shard}')
                        if (run/'rows'/f'{tag}.done.json').exists():continue
                        log=(run/'logs'/f'{tag}.log').open('a');logs.append(log)
                        cmd=[str(root/'.venv/bin/python'),str(script),'worker','--run',str(run),'--model',model,'--method',method,'--shard',str(shard)]
                        if smoke:cmd+=['--smoke']
                        proc=subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT);active.append(proc)
                    while any(p.poll() is None for p in active):
                        if any(p.poll() not in (None,0) for p in active):raise RuntimeError('Worker failed; see '+base)
                        dump(run/'status.json',dict(state='smoke' if smoke else 'running',model=model,method=method,ratios=cfg['retentions'],pids=[p.pid for p in active if p.poll() is None],predictions=report(run),failed=failed));time.sleep(15)
                    assert all(p.returncode==0 for p in active),base
                    for log in logs:log.close()
            except Exception as exc:
                for p in active:
                    if p.poll() is None:p.terminate()
                for p in active:p.wait()
                for log in logs:log.close()
                failed.append(dict(model=model,method=method,error=repr(exc)));dump(run/'failed_jobs.json',failed)
                # Other independently verified methods continue. Failed methods
                # never generate a completion marker or enter final averages.
        pending=[j for j in json.loads((run/'jobs.json').read_text()) if not j['ready']]
        dump(run/'status.json',dict(state='pending_implementation' if pending else ('failed_jobs' if failed else 'complete'),predictions=report(run),failed=failed,pending=pending))
    except BaseException:
        for p in active:
            if p.poll() is None:p.terminate()
        for p in active:p.wait()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','queue','report']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model');p.add_argument('--method',choices=METHODS);p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare(a.run)
    elif a.mode=='worker':worker(a.run,a.model,a.method,a.shard,a.smoke)
    elif a.mode=='queue':queue(a.run)
    else:print(report(a.run))
