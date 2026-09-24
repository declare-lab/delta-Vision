"""Video-MME999 no-visual LLM: visual encoder included in prefill and peak.
Only the no-visual condition is run; historical base/adapter are references.
"""
import argparse,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(os.environ.get('RESOURCE_REPO',str(Path(__file__).resolve().parents[3])))
CASES=['no_visual_tokens']
NO_VIS=ROOT/'artifacts/diagnostics/video_no_visual_tokens_flops_20260923'
def dump(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def read_rows(paths):
    out={}
    for p in paths:
        for s in p.read_text().splitlines():
            x=json.loads(s);assert x['index'] not in out;out[x['index']]=x
    return out

def worker(a):
    sys.path.insert(0,str(a.previous/'source'))
    import torch
    from src.benchmarking.engines import adapter as ab
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    from src.data import QwenBenchmarkDataset
    from src.model import build_qwen_initial_context
    from src.model_setup import disable_qwen_deepstack
    from src.attention import optimize_qwen_attention_metadata
    from src.kernels import FusedQwenNorms
    from src.kernels import QwenExactRoPE
    from src.kernels import QwenFusedProjections
    from src.graphs import NativeDecoderGraphs
    from src.graphs import QwenWholePrefillGraphs
    from unittest.mock import patch
    c=json.loads((a.previous/'protocol.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0')
    model,processor=load_baseline_model('base',c['model'],torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(model);optimize_qwen_attention_metadata(model)
    if a.case!='no_visual_tokens':raise ValueError('This profile only measures no_visual_tokens; base/adapter full-pipeline uses resources')
    norms=FusedQwenNorms(model);rope=QwenExactRoPE(model);projections=QwenFusedProjections(model)
    native=NativeDecoderGraphs(model,max_shapes=8,vision=False,prefill_layers=False,packed_kv=True)
    whole=QwenWholePrefillGraphs(model,max_shapes=1)
    def prefill():
        nonlocal next_positions
        model.model.rope_deltas=None
        full_hidden,positions3=build_qwen_initial_context(model,full_inputs)
        hidden=full_hidden[:,keep].contiguous()
        positions3=positions3[:,:,keep].contiguous()
        del full_hidden
        positions4=torch.cat((torch.arange(nt,device=device).view(1,1,nt),positions3),dim=0)
        next_positions=positions4[:,:,-1:]+torch.arange(1,8,device=device)
        out=model(inputs_embeds=hidden,attention_mask=inputs['attention_mask'],position_ids=positions4,
            use_cache=True,logits_to_keep=1,return_dict=True)
        return out.logits,out.past_key_values
    def step(token,cache,i):
        out=model(input_ids=token,position_ids=next_positions[:,:,i:i+1],past_key_values=cache,
            use_cache=True,logits_to_keep=1,return_dict=True)
        return out.logits,out.past_key_values
    def capture_llm(value):native.allow_capture=whole.allow_capture=value
    def stats_llm():return native.stats()['captures'],native.stats()['cold_layer_fallbacks'],whole.stats()['captures'],whole.stats()['fallbacks']
    def tensors(cache):return [t for l in cache.layers for t in (l.keys,l.values)]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    # Same native FA2 encoder graph as the historical full-pipeline benchmark.
    vision=NativeDecoderGraphs(model,max_shapes=8,vision=True,full_decode=False,prefill_layers=False,max_prefill_shapes=1)
    def capture(value):
        capture_llm(value);vision.allow_capture=value
    def stats():
        return stats_llm()+(vision.stats()['captures'],vision.stats()['cold_layer_fallbacks'])
    inside=[False];encoder_calls=[0]
    def count_encoder(*unused):encoder_calls[0]+=1
    model.model.visual.register_forward_pre_hook(count_encoder)
    def request(check=False,continuous=False):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();inside[0]=True
        try:
            encoder_before=encoder_calls[0]
            start=time.perf_counter();logits,cache=prefill();torch.cuda.synchronize();p=time.perf_counter()-start
            assert encoder_calls[0]-encoder_before==1
            prefix_hash=ab.tensor_sha(tensors(cache)) if check else None
            initial_lengths=[t.shape[-2] for t in tensors(cache)[::2]]
            tt=[];hashes=[];d=0.
            decode_start=time.perf_counter()
            for i in range(8):
                if check:hashes.append(ab.tensor_sha([logits[:,-1].float()]))
                scores=logits[:,-1].to(torch.float32,copy=True);scores[:,eos]=-float('inf')
                token=scores.argmax(-1).view(1,1);tt.append(token)
                if i==7:break
                if continuous:logits,cache=step(token,cache,i)
                else:
                    torch.cuda.synchronize();begin=time.perf_counter();logits,cache=step(token,cache,i);torch.cuda.synchronize();d+=time.perf_counter()-begin
            if continuous:
                torch.cuda.synchronize();d=time.perf_counter()-decode_start
            peak=torch.cuda.max_memory_allocated()/1024**3
            row=dict(prefill_s=p,decode_s=d,total_s=p+d,peak_GiB=peak,tokens=torch.cat(tt,dim=1)[0].tolist(),layer_lengths=initial_lengths)
            if check:row.update(logits_sha256=hashes,prefill_kv_sha256=prefix_hash,final_kv_sha256=ab.tensor_sha(tensors(cache)))
            return row
        finally:inside[0]=False
    def reject_sdpa(*unused,**kw):raise AssertionError('SDPA in FA2 benchmark')
    ds=QwenBenchmarkDataset(c['manifest'],processor,'videomme',data_root=str(ROOT/'data/benchmarks/videomme'),
        cache_dir=str(ROOT/'test/results/adapter_exact_20260915/inputs'))
    old=read_rows((NO_VIS/'rows').glob('shard*.jsonl'))
    indices=a.indices or list(range(a.shard,999,8));folder=a.run/a.case;folder.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),patch('torch.nn.functional.scaled_dot_product_attention',reject_sdpa),(folder/f'shard{a.shard}.jsonl').open('x',buffering=1) as out:
        for n,index in enumerate(indices):
            inputs=_qwen_inputs_from_item(ds[index],device)
            digest=ab.tensor_sha([inputs[k] for k in sorted(inputs)]);assert digest==old[index]['input_sha256']
            full_inputs=inputs
            keep=full_inputs['mm_token_type_ids'][0].eq(0)
            nv=int((~keep).sum());nt=int(keep.sum())
            inputs={k:full_inputs[k][:,keep].contiguous() for k in ['input_ids','attention_mask','mm_token_type_ids']}
            assert nv==old[index]['visual_tokens'] and nt==old[index]['text_tokens']
            next_positions=None
            if n==0:
                native.enabled=whole.enabled=vision.enabled=False
                eager=request(True)
                native.enabled=whole.enabled=vision.enabled=True
            capture(True);checked=request(True)
            assert checked['tokens']==old[index]['tokens'],(a.case,index,'tokens')
            if n==0:
                assert eager['tokens']==checked['tokens'],(index,'eager/graph tokens')
                assert eager['logits_sha256']==checked['logits_sha256'],(index,'eager/graph logits')
                assert eager['final_kv_sha256']==checked['final_kv_sha256'],(index,'eager/graph KV')
                del eager
            assert checked['layer_lengths']==[nt]*36
            warm=request();assert warm['tokens']==checked['tokens'];capture(False)
            def replay_counts():return whole.stats()['replays'],native.stats()['layer_replays'],vision.stats()['layer_replays']
            before=stats();replay_before=replay_counts()
            trials=[request() for _ in range(3)]
            continuous_trials=[request(continuous=True) for _ in range(3)]
            assert before==stats(),(index,'capture/fallback in timing')
            replay_delta=tuple(b-a for a,b in zip(replay_before,replay_counts()))
            assert replay_delta==(6,42,6),(index,'Not all forwards replayed a graph',replay_delta)
            assert all(t['tokens']==checked['tokens'] for t in continuous_trials)
            assert all(t['tokens']==checked['tokens'] for t in trials)
            row=dict(index=index,case=a.case,shard=a.shard,input_sha256=digest,tokens=checked['tokens'],
                layer_lengths=checked['layer_lengths'],prefill_kv_sha256=checked['prefill_kv_sha256'],final_kv_sha256=checked['final_kv_sha256'],
                trials=trials,continuous_trials=continuous_trials,graph_prefill_replays=6,graph_decode_replays=42,text_tokens=nt,visual_tokens=nv,timed_encoder_calls=6,graph_encoder_replays=6,timed_captures=0,timed_fallbacks=0)
            out.write(json.dumps(row)+'\n');print(json.dumps(dict(case=a.case,done=n+1,expected=len(indices),index=index)),flush=True)
            del full_inputs,inputs,keep,checked,warm,trials,continuous_trials

def report(a):
    rows=read_rows((a.run/'no_visual_tokens').glob('shard*.jsonl'))
    assert set(rows)==set(range(999))
    for r in rows.values():
        assert r['graph_prefill_replays']==r['graph_encoder_replays']==r['timed_encoder_calls']==6
        assert r['graph_decode_replays']==42 and r['timed_captures']==r['timed_fallbacks']==0
    fresh=dict(case='no_visual_tokens',samples=999,source='fresh',peak_GiB=max(t['peak_GiB'] for x in rows.values() for field in ['trials','continuous_trials'] for t in x[field]))
    for field,prefix in [('trials',''),('continuous_trials','continuous_')]:
        for stage in ['prefill','decode']:
            fresh[prefix+stage+'_ms']=1000*statistics.mean(statistics.median(t[stage+'_s'] for t in x[field]) for x in rows.values())
        fresh[prefix+'total_ms']=fresh[prefix+'prefill_ms']+fresh[prefix+'decode_ms']
    old=ROOT/'artifacts/reports/video_adapter_layers_including_vision_20260923/RESULTS.json'
    encoder_counts=read_rows((a.previous/'base').glob('flops_*.jsonl'))
    llm_counts=read_rows((NO_VIS/'rows').glob('shard*.jsonl'))
    for i,r in rows.items():
        assert r['input_sha256']==encoder_counts[i]['input_sha256']==llm_counts[i]['input_sha256']
        assert r['tokens']==llm_counts[i]['tokens']
    fresh['flops_with_vision_T']=statistics.mean(encoder_counts[i]['vision_matrix_flops']+llm_counts[i]['request_matrix_flops'] for i in rows)/1e12
    fresh['flops_without_vision_T']=statistics.mean(llm_counts[i]['request_matrix_flops'] for i in rows)/1e12
    fresh['flops_source']='Matched-input executed encoder counts plus verified text-only LLM counts; no FLOP counter in timed calls'
    result=[dict(x,source=str(old)) for x in json.loads(old.read_text()) if x['case'] in ['base','adapter']]+[fresh]
    for r in result:
        r['flops_with_vision_pct_base']=100*r['flops_with_vision_T']/result[0]['flops_with_vision_T']
        for stage in ['prefill','decode','total']:r[stage+'_speedup']=result[0][stage+'_ms']/r[stage+'_ms']
    dump(a.run/'RESULTS.json',dict(complete=True,samples=999,includes_visual_encoding=True,results=result))
    lines=['# Video-MME999 including visual encoding','',
        'Only no_visual_tokens is freshly measured. Base/adapter are historical full-pipeline references. Same video tensors verified against archived input hashes; FA2/BF16, DeepStack off, fixed8 outputs; identical frozen model/optimization source. Visual encoding and multimodal context preparation occur on every request INSIDE prefill. Visual features are discarded before the LLM, but timestamps/template markers/text positions retained. Each of six timed trials per input verifies1 encoder graph replay,1 LLM prefill graph replay,7 decoder graph replays. No graph captures or fallbacks during trials.',
        'Main timing matches previous forward-only decode convention; total=prefill+decode. Three repeats per input, stage medians then mean over999. Continuous decode including greedy selection is additionally reported in JSON. Peak allocated GiB includes encoder and LLM stages, all resident weights/input buffers and graph pools. Eight isolated GPUs, single method.','',
        '| Method | Source | Prefill ms | Decode ms | Total ms | Peak GiB | Prefill speedup | Decode speedup | Total speedup |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:
        source='fresh' if r['case']=='no_visual_tokens' else 'historical'
        lines.append(f"| {r['case']} | {source} | {r['prefill_ms']:.2f} | {r['decode_ms']:.2f} | {r['total_ms']:.2f} | {r['peak_GiB']:.3f} | {r['prefill_speedup']:.3f} | {r['decode_speedup']:.3f} | {r['total_speedup']:.3f} |")
    (a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def queue(a):
    burn=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')
    children=[];paused=False;restore_burn=False;start=time.time();done=[]
    def alive(pid):
        p=Path(f'/proc/{pid}/stat')
        return p.exists() and p.read_text().split()[2]!='Z'
    def status(state,**kw):dump(a.run/'status.json',dict(state=state,completed=done,elapsed_s=time.time()-start,**kw))
    def stop(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if a.scheduler_pid and alive(a.scheduler_pid):
            assert b'repair_baseline_suite.py\x00queue' in Path(f'/proc/{a.scheduler_pid}/cmdline').read_bytes()
            os.kill(a.scheduler_pid,signal.SIGSTOP);paused=True
        if burn.exists() and (burn.parent/'pid').exists():
            restore_burn=alive(int((burn.parent/'pid').read_text()))
            if restore_burn:subprocess.run([sys.executable,str(burn),'stop'],check=True)
        dump(a.run/'isolation.json',dict(scheduler_pid=a.scheduler_pid,scheduler_paused=paused,burn_was_running=restore_burn))
        while True:
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            status('waiting_for_current_evaluations_to_finish',gpu_pids=pids)
            if not pids:break
            time.sleep(10)
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        for case in CASES:
            children=[]
            for shard in range(8):
                log=(a.run/f'{case}_{shard}.log').open('w')
                cmd=[sys.executable,'-m','src.benchmarking','videomme','vision-removed','--','worker','--run',str(a.run),'--previous',str(a.previous),'--case',case,'--shard',str(shard)]
                p=subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                log.close();children.append(p)
            while any(p.poll() is None for p in children):
                assert all(p.poll() in (None,0) for p in children),(case,'Worker failed')
                status('running',case=case,pids=[p.pid for p in children if p.poll() is None]);time.sleep(5)
            assert all(p.returncode==0 for p in children);done.append(case)
        report(a);status('complete')
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        if paused and alive(a.scheduler_pid):os.kill(a.scheduler_pid,signal.SIGCONT)
        if restore_burn:
            r=subprocess.run([sys.executable,str(burn),'start','--coexist'],capture_output=True,text=True)
            dump(a.run/'restoration.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','queue','report'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,default=ROOT/'artifacts/diagnostics/video_adapter_layer_ablation_999_20260923')
    p.add_argument('--case',choices=CASES);p.add_argument('--shard',type=int,default=0);p.add_argument('--indices',type=int,nargs='+');p.add_argument('--scheduler-pid',type=int)
    a=p.parse_args();a.run=a.run.resolve();a.previous=a.previous.resolve();a.run.mkdir(parents=True,exist_ok=True)
    os.environ.update(QWEN_VIDEO_NUM_FRAMES='8',QWEN_VIDEO_SAMPLING='full_timestamp_v1',TOKENIZERS_PARALLELISM='false')
    if a.phase=='worker':return worker(a)
    if a.phase=='report':return report(a)
    frozen=a.run/'driver.py'
    if not frozen.exists():shutil.copy2(__file__,frozen)
    dump(a.run/'protocol.json',dict(source_run=str(a.previous),cases=CASES,driver_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        samples=999,frames=8,tokens=8,deepstack=False,FA2=True,vision_in_timing=True,vision_graph_capacity=1,prefill_graph_capacity=1,decode_graph_capacity=8,
        repetitions=3,decode_modes=['legacy_separate_forward_timing','continuous_including_greedy_selection']))
    return queue(a)

if __name__=='__main__':main()
