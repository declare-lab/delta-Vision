"""RealWorldQA765: write/forget/read, factorial, groups and convolution controls."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'artifacts/experiments/qwen35_memory/qwen35_memory_mechanism_20260922_102800'
LA=[i for i in range(32) if i%4!=3];FA=list(range(3,32,4))


def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def read(p):
    if not p.exists():return []
    return [json.loads(l) for l in p.read_text().splitlines(keepends=True) if l.endswith('\n')]


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/scripts','source/test/diagnostics','logs','audits','traces','spectra']:(run/d).mkdir(parents=True)
    old=json.loads((REFERENCE/'config.json').read_text())
    c={k:old[k] for k in ['model_path','model_revision','rank','evaluation_generation']}
    c.update(reference_run=str(REFERENCE),evaluation={'realworldqa':old['evaluation']['realworldqa']},
        original_root=str(ROOT),seed=44,deepstack=False,native_only=True,samples=765,
        dependencies=[str(ROOT/'artifacts/eval/qwen4b_no_visual_tokens_20260923'),str(ROOT/'artifacts/diagnostics/video_no_visual_tokens_flops_20260923')],
        conditions={},stages={},group_aliases={})
    conditions=c['conditions']
    for name,mode in [('native','native'),('no_write','no_write'),('no_forget','no_forget'),('state_skip','state_skip')]:
        conditions[name]=dict(mode=mode,layers=LA)
    conditions['boundary_rank0']=dict(boundary_rank0=True)
    c['stages']['core']=['native','no_write','no_forget','state_skip','boundary_rank0']
    conditions['fa_off']=dict(mode='native',layers=LA,fa_off=True)
    conditions['state_skip_fa_off']=dict(mode='state_skip',layers=LA,fa_off=True)
    c['stages']['factorial']=['fa_off','state_skip_fa_off']
    groups=[];seen={tuple(LA):'state_skip'}
    for name,ls in ([(f'group_{g}',list(range(4*g,4*g+3))) for g in range(8)]
            +[(f'prefix_{k}',[i for i in LA if i<4*k]) for k in range(1,9)]
            +[(f'suffix_{k}',[i for i in LA if i>=4*k]) for k in range(8)]):
        if tuple(ls) in seen:c['group_aliases'][name]=seen[tuple(ls)];continue
        seen[tuple(ls)]=name;conditions[name]=dict(mode='state_skip',layers=ls);groups.append(name)
    c['group_aliases'].update(early_four_groups='prefix_4',late_four_groups='suffix_4')
    c['stages']['groups']=groups
    for name,boundary,fa in [('state_skip_conv_after_end','after_end',False),('state_skip_conv_before_end','before_end',False),
                             ('state_skip_conv_before_end_fa_off','before_end',True)]:
        conditions[name]=dict(mode='state_skip',layers=LA,conv_boundary=boundary,fa_off=fa)
    c['stages']['conv']=['state_skip_conv_after_end','state_skip_conv_before_end','state_skip_conv_before_end_fa_off']
    c['stage_order']=['core','analysis','factorial','groups','conv','spectrum']
    c['analysis']='Native post-convolution inputs; fixed native Q/K/V/g/beta dual recurrent replay. All positions/heads/layers saved FP32 in compressed NPZ. Native outputs untouched. Includes cached decode input tokens; final generated token not re-forwarded.'
    c['boundary']='Before vision_start vs after vision_end; counterfactual skips only patch-position recurrent transitions, markers normal. Three boundary states saved; spectra uncentered per head, full128.'
    c['statistics']='Exact two-sided McNemar; paired bootstrap10000 draws seed44; four-way paired correctness groups; no independent-sample CI. Layer/head comparisons descriptive.'
    c['third_LA']='Trace statistics separately preserve all24 layers, including third-in-group2,6,10,14,18,22,26,30.'
    sources=(list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py')))+[Path(__file__).resolve(),ROOT/'analysis/fig05_hybrid_attention/report_qwen35_write_forget.py',ROOT/'test/diagnostics/test_qwen35_write_forget.py']
    for p in sources:
        d=run/'source'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    dump(run/'config.json',c)
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'source').rglob('*.py')})
    dump(run/'status.json',dict(state='prepared',stage=None))


def worker(run,shard,stage):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root'])
    sys.path.insert(0,str(run/'source'));sys.path.insert(0,str(root/'artifacts/dependencies/qwen35_python'))
    import numpy as np
    import torch
    from src.qwen35 import load_model,prepare_inputs,generate_evaluation_answer,sha
    from analysis.fig05_hybrid_attention.qwen35_write_forget import visual_gates, WriteForgetTracker, METRICS
    from analysis.fig05_hybrid_attention.qwen35_memory_probe import state_intervention
    from analysis.fig05_hybrid_attention.qwen35_full_attention_ablation import remove_full_attention_visual_effect
    from src.benchmarks import get_benchmark_spec,build_benchmark_prompt
    for name,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/name)==h
    info=c['evaluation']['realworldqa'];assert sha(info['path'])==info['sha256']
    data=read(Path(info['path']));assert len(data)==765
    ref={r['index']:r for f in (Path(c['reference_run'])/'accuracy').glob('realworldqa.shard*.jsonl') for r in read(f)}
    if stage=='spectrum':return spectrum(run,shard)
    torch.set_num_threads(2);torch.manual_seed(44)
    processor,model,adapter,controller=load_model(c,torch.device('cuda:0'));torch.set_float32_matmul_precision('highest')
    spec=get_benchmark_spec('realworldqa')
    if stage=='validate':
        loader=importlib.util.spec_from_file_location('tests',run/'source/test/diagnostics/test_qwen35_write_forget.py')
        tests=importlib.util.module_from_spec(loader);loader.loader.exec_module(tests);gate=tests.run_tests()
    folder=run/stage;folder.mkdir(exist_ok=True);path=folder/f'shard{shard}.jsonl'
    done={r['index'] for r in read(path)};indices=[0] if stage=='validate' else list(range(shard,765,8))
    with path.open('a',buffering=1) as out,torch.inference_mode():
        for index in indices:
            if index in done:continue
            started=time.time();row=data[index]
            inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
            digest=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
            assert digest==ref[index]['input_ids_sha256']
            expected=next(v for v in ref[index]['variants'] if v['method']=='native' and v['rank'] is None)
            mask=inputs['mm_token_type_ids'].eq(1);end=int(mask[0].nonzero()[-1])+1
            assert int(inputs['input_ids'][0,end])==model.config.vision_end_token_id
            def generate(condition):
                first={}
                def hook(module,args,output):
                    if not first:first['logits']=output.logits[:,-1].detach().float().clone()
                h=model.register_forward_hook(hook)
                try:
                    with ExitStack() as stack:
                        if condition.get('boundary_rank0'):stack.enter_context(state_intervention(model,mask,rank=0))
                        else:
                            cb=condition.get('conv_boundary');cb=end if cb=='before_end' else end+1 if cb=='after_end' else None
                            stack.enter_context(visual_gates(model,mask,condition.get('mode','native'),condition.get('layers'),cb))
                        if condition.get('fa_off'):stack.enter_context(remove_full_attention_visual_effect(model,mask,FA))
                        answer=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=info['max_new_tokens'])
                    return answer,first['logits']
                finally:h.remove()
            if stage=='analysis':
                tracker=WriteForgetTracker(model,inputs)
                with tracker.activate():native=generate_evaluation_answer(model,processor,inputs,row,spec,c,max_new_tokens=info['max_new_tokens'])
                assert native['generated_token_ids']==expected['generated_token_ids']
                metrics=torch.stack([torch.cat(tracker.metrics[i],0) for i in LA]).numpy()
                boundary=torch.stack([tracker.boundaries[i] for i in LA]).numpy()
                ids=inputs['input_ids'][0].tolist()+native['generated_token_ids'][:-1]
                assert metrics.shape==(24,len(ids),32,len(METRICS))
                # Separate question tokens from the assistant-generation template.
                labels=np.zeros(len(ids),dtype=np.int8)
                labels[tracker.start-1]=1;labels[tracker.start:tracker.end]=2;labels[tracker.end]=3
                labels[tracker.end+1:tracker.length]=4
                imend=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
                ends=[i for i in range(tracker.end+1,tracker.length) if ids[i]==imend]
                if ends:labels[ends[-1]:tracker.length]=6
                labels[tracker.length:]=5
                dest=run/'traces'/f'{index:04d}.npz';temp=dest.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,metrics=metrics,boundary=boundary,token_ids=np.array(ids),
                    phase=labels,layer_ids=np.array(LA),metric_names=np.array(METRICS),
                    phase_names=np.array(['prompt_prefix','vision_start','image_patches','vision_end','question_text','answer_input','assistant_template']))
                temp.replace(dest)
                record=dict(index=index,input_ids_sha256=digest,native=native,trace=str(dest),
                    prompt_length=tracker.length,visual_start=tracker.start,visual_end=tracker.end,
                    checks=tracker.checks,elapsed_s=time.time()-started)
                del tracker,metrics,boundary
            else:
                names=c['stages']['core']+c['stages']['factorial']+c['stages']['conv'] if stage=='validate' else c['stages'][stage]
                native,baseline=generate(c['conditions']['native']);assert native['generated_token_ids']==expected['generated_token_ids']
                variants={'native':native};logp=baseline.log_softmax(-1)
                for name in names:
                    if name=='native':continue
                    answer,logits=generate(c['conditions'][name])
                    answer['kl_to_native']=(logp.exp()*(logp-logits.log_softmax(-1))).sum().item()
                    variants[name]=answer
                if stage=='validate':
                    with remove_full_attention_visual_effect(model,mask,FA,block=False):
                        noop,no_logits=generate(c['conditions']['native'])
                    assert torch.equal(no_logits,baseline) and noop['generated_token_ids']==native['generated_token_ids']
                    gate.update(native_reference_tokens_exact=True,fa_noop_logits_exact=True,all_conditions_smoke=True)
                    dump(run/'validation.json',dict(passed=True,checks=gate))
                if stage in ('core','validate'):
                    old=next(v for v in ref[index]['variants'] if v['method']=='native' and v['rank']==0)
                    assert variants['boundary_rank0']['generated_token_ids']==old['generated_token_ids'],(index,'Old rank0 mismatch')
                record=dict(index=index,input_ids_sha256=digest,variants=variants,elapsed_s=time.time()-started)
            out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');done.add(index)
            dump(run/f'progress_{stage}_{shard}.json',dict(completed=len(done),index=index,seconds=record['elapsed_s']))
            if stage=='validate' or len(done)%10==0:print(stage,shard,index,len(done),round(record['elapsed_s'],2),flush=True)
            del inputs
    controller.close()


def spectrum(run,shard):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    folder=run/'spectrum';folder.mkdir(exist_ok=True)
    path=folder/f'shard{shard}.jsonl';done={r['index'] for r in read(path)}
    with path.open('a',buffering=1) as out:
        for i in range(shard,765,8):
            if i in done:continue
            started=time.time()
            with np.load(run/'traces'/f'{i:04d}.npz') as z:boundary=torch.from_numpy(z['boundary']).double()
            # [layer, 3 boundary kinds, head, K,V]; no rank approximation here.
            effect=boundary[:,1]-boundary[:,2]
            mats=torch.cat((boundary,effect[:,None]),1)
            sv=torch.linalg.svdvals(mats)
            energy=sv.square();total=energy.sum(-1);cdf=energy.cumsum(-1)/total[...,None].clamp_min(1e-300)
            p=sv/sv.sum(-1,keepdim=True).clamp_min(1e-300)
            er=torch.where(total>0,(-(p*p.clamp_min(1e-300).log()).sum(-1)).exp(),0)
            r95=torch.where(total>0,(cdf<.95).sum(-1)+1,0)
            a=boundary[:,0].flatten(-2);b=boundary[:,1].flatten(-2)
            an=a.norm(dim=-1);bn=b.norm(dim=-1)
            cosine=torch.where(an*bn>0,(a*b).sum(-1)/(an*bn),float('nan'))
            relative=torch.where(an>0,(b-a).norm(dim=-1)/an,float('nan'))
            dest=run/'spectra'/f'{i:04d}.npz'
            np.savez_compressed(dest,singular_values=sv.numpy(),r95=r95.numpy(),effective_rank=er.numpy(),
                pre_post_cosine=cosine.numpy(),pre_post_relative_change=relative.numpy(),
                kinds=np.array(['before_vision_start','after_vision_end','skip_after_vision_end','native_minus_skip_effect']))
            out.write(json.dumps(dict(index=i,path=str(dest),seconds=time.time()-started))+'\n');done.add(i)
            dump(run/f'progress_spectrum_{shard}.json',dict(completed=len(done),index=i))


def queue(run):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root']);active=[];started=time.time()
    control=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py');burn_stopped=False
    try:
        while True:
            waiting=[]
            for dep in map(Path,c['dependencies']):
                s=json.loads((dep/'status.json').read_text())
                if s['state']=='failed':raise RuntimeError(f'Dependency failed: {dep}')
                pidfile=dep/'queue.pid';alive=False
                if pidfile.exists():
                    proc=Path('/proc')/pidfile.read_text().strip()/'stat'
                    try:alive=proc.read_text().split()[2]!='Z'
                    except FileNotFoundError:pass
                if s['state']!='complete' or alive:waiting.append(str(dep))
            if not waiting:break
            dump(run/'status.json',dict(state='waiting_for_previous_experiments',dependencies=waiting));time.sleep(10)
        if control.exists():subprocess.run([str(root/'.venv/bin/python'),str(control),'stop'],check=True);burn_stopped=True
        for stage in ['validate']+c['stage_order']:
            if (run/'audits'/f'{stage}.json').exists():continue
            active=[]
            for shard in ([0] if stage=='validate' else range(8)):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
                log=(run/'logs'/f'{stage}_{shard}.log').open('a')
                cmd=[str(root/'.venv/bin/python'),'-u',str(run/'source/analysis/fig05_hybrid_attention/qwen35_write_forget_suite.py'),'worker','--run',str(run),'--stage',stage,'--shard',str(shard)]
                p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);active.append((p,log))
            while any(p.poll() is None for p,_ in active):
                if any(p.poll() not in (None,0) for p,_ in active):raise RuntimeError(f'{stage} worker failed')
                done=sum(json.loads(p.read_text())['completed'] for p in run.glob(f'progress_{stage}_*.json'))
                dump(run/'status.json',dict(state='running',stage=stage,completed=done,expected=1 if stage=='validate' else 765,elapsed_s=time.time()-started))
                time.sleep(10)
            assert all(p.returncode==0 for p,_ in active)
            for _,log in active:log.close()
            rr=[r for f in (run/stage).glob('shard*.jsonl') for r in read(f)]
            assert len(rr)==(1 if stage=='validate' else 765) and len({r['index'] for r in rr})==len(rr)
            dump(run/'audits'/f'{stage}.json',dict(passed=True,samples=len(rr)))
            if stage!='validate':
                subprocess.run([str(root/'.venv/bin/python'),str(run/'source/analysis/fig05_hybrid_attention/report_qwen35_write_forget.py'),'--run',str(run),'--partial'],check=True,cwd=root)
        subprocess.run([str(root/'.venv/bin/python'),str(run/'source/analysis/fig05_hybrid_attention/report_qwen35_write_forget.py'),'--run',str(run)],check=True,cwd=root)
        dump(run/'status.json',dict(state='complete',elapsed_s=time.time()-started))
    except BaseException as e:
        for p,_ in active:
            if p.poll() is None:p.terminate()
        for p,_ in active:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        dump(run/'status.json',dict(state='failed',error=repr(e)));raise
    finally:
        if burn_stopped:
            r=subprocess.run([str(root/'.venv/bin/python'),str(control),'start','--coexist'],capture_output=True,text=True)
            dump(run/'burn_restoration.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','worker']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage');p.add_argument('--shard',type=int,default=0);a=p.parse_args();a.run=a.run.resolve()
    if a.action=='worker':worker(a.run,a.shard,a.stage)
    else:globals()[a.action](a.run)
