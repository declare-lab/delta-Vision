"""Corrected scope: prepared visual embeddings -> LLM prefill + cached decode.

Encoder computation and its graph pools are outside the measured region.
Each trial total is prefill + decode, not end-to-end request wall time.
"""
import argparse,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ.get('RESOURCE_REPO',str(Path(__file__).resolve().parents[3])))
CASES=['base','adapter','no_visual_tokens','first5_last10','first10_last10']
NO_VIS=ROOT/'artifacts/diagnostics/video_no_visual_tokens_flops_20260923'
class ReplayCounter:
    def __init__(self,graph):self.graph=graph;self.replays=0
    def replay(self):self.replays+=1;return self.graph.replay()
    def __getattr__(self,name):return getattr(self.graph,name)

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
    from src.model import build_qwen_initial_context,load_qwen_embedding_adapter_checkpoint
    from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
    from src.model_setup import disable_qwen_deepstack
    from src.attention import optimize_qwen_attention_metadata
    from src.kernels import FusedQwenNorms
    from src.kernels import QwenExactRoPE
    from src.kernels import QwenFusedProjections
    from src.graphs import NativeDecoderGraphs
    from src.graphs import QwenWholePrefillGraphs
    from unittest.mock import patch
    c=json.loads((a.previous/'protocol.json').read_text())
    blocked=c['cases'].get(a.case,[])
    is_adapter=a.case in ('adapter','first5_last10','first10_last10')
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0')
    model,processor=load_baseline_model('base',c['model'],torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(model);optimize_qwen_attention_metadata(model)
    adapter=None
    if not is_adapter:
        norms=FusedQwenNorms(model);rope=QwenExactRoPE(model);projections=QwenFusedProjections(model)
        native=NativeDecoderGraphs(model,max_shapes=8,vision=False,prefill_layers=False,packed_kv=True)
        whole=QwenWholePrefillGraphs(model,max_shapes=1)
        def prefill():
            out=model(inputs_embeds=hidden,attention_mask=inputs['attention_mask'],position_ids=positions4,
                use_cache=True,logits_to_keep=1,return_dict=True)
            return out.logits,out.past_key_values
        def step(token,cache,i):
            out=model(input_ids=token,position_ids=next_positions[:,:,i:i+1],past_key_values=cache,
                use_cache=True,logits_to_keep=1,return_dict=True)
            return out.logits,out.past_key_values
        def capture(value):native.allow_capture=whole.allow_capture=value
        def stats():return native.stats()['captures'],native.stats()['cold_layer_fallbacks'],whole.stats()['captures'],whole.stats()['fallbacks']
        def tensors(cache):return [t for l in cache.layers for t in (l.keys,l.values)]
    else:
        adapter,meta=load_qwen_embedding_adapter_checkpoint(c['checkpoint'],model.model.language_model,device,torch.bfloat16)
        settings=SimpleNamespace(last_logits_only=True,attn_implementation='flash_attention_2',cuda_graph=True,
            cuda_graph_context=False,compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,
            adapter_decode_cache_mode='fast',adapter_exact_optimizations=True,adapter_max_optimizations=True,
            adapter_blocked_visual_layers=blocked)
        fast=build_qwen_fast_adapter_prefill(model,adapter,settings)
        runner=fast.graph_runners[1];decoder=model._adapter_decode_graph_runner
        def prefill():
            logits,_,cache=runner(inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,positions3,topology=topology)
            return logits,cache
        def step(token,cache,i):return decoder(model,adapter,token,cache,logits_to_keep=1)
        def capture(value):decoder.allow_capture=value
        def stats():return decoder.stats()['captures'],decoder.stats()['cold_fallbacks'],id(runner.graph)
        def tensors(cache):return ab.cache_tensors(cache)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    inside=[False]
    def no_encoder(*unused):
        if inside[0]:raise AssertionError('Visual encoder executed inside LLM timing')
    model.model.visual.register_forward_pre_hook(no_encoder)
    def request(check=False,continuous=False):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();inside[0]=True
        try:
            start=time.perf_counter();logits,cache=prefill();torch.cuda.synchronize();p=time.perf_counter()-start
            prefix_hash=ab.tensor_sha(tensors(cache)) if check else None
            initial_lengths=[t.shape[-2] for t in tensors(cache)[::2]]
            tt=[];hashes=[];d=0.
            decode_start=time.perf_counter()
            for i in range(8):
                if check:hashes.append(ab.tensor_sha([logits[:,-1].float() if not is_adapter else logits]))
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
    old=read_rows((a.previous/a.case).glob('optimized_*.jsonl')) if a.case!='no_visual_tokens' else read_rows((NO_VIS/'rows').glob('shard*.jsonl'))
    indices=a.indices or list(range(a.shard,999,8));folder=a.run/a.case;folder.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),patch('torch.nn.functional.scaled_dot_product_attention',reject_sdpa),(folder/f'shard{a.shard}.jsonl').open('x',buffering=1) as out:
        for n,index in enumerate(indices):
            inputs=_qwen_inputs_from_item(ds[index],device)
            digest=ab.tensor_sha([inputs[k] for k in sorted(inputs)]);assert digest==old[index]['input_sha256']
            model.model.rope_deltas=None
            # Encoder and multimodal position preparation happen before all timed trials.
            hidden,positions3=build_qwen_initial_context(model,inputs)
            positions4=model._prepare_position_ids_for_generation(inputs['input_ids'],dict(inputs))
            nv=int(inputs['mm_token_type_ids'].ne(0).sum());nt=int(inputs['mm_token_type_ids'].eq(0).sum())
            if a.case=='no_visual_tokens':
                keep=inputs['mm_token_type_ids'][0].eq(0)
                hidden=hidden[:,keep].contiguous()
                positions3=positions3[:,:,keep].contiguous()
                # Text sequence axis must be consecutive for whole-prefill graph;
                # the three original multimodal RoPE axes must NOT be renumbered.
                positions4=torch.cat((torch.arange(nt,device=device).view(1,1,nt),positions3),dim=0)
                inputs={**inputs,**{k:inputs[k][:,keep].contiguous() for k in ['input_ids','attention_mask','mm_token_type_ids']}}
                assert torch.equal(hidden,model.get_input_embeddings()(inputs['input_ids']))
                assert positions4.shape==(4,1,nt)
            next_positions=positions4[:,:,-1:]+torch.arange(1,8,device=device)
            topology=torch.stack((inputs['attention_mask'][0],inputs['mm_token_type_ids'][0])).tolist()
            if a.case=='no_visual_tokens' and n==0:
                native.enabled=whole.enabled=False
                eager=request(True)
                native.enabled=whole.enabled=True
            capture(True);checked=request(True)
            assert checked['tokens']==old[index]['tokens'],(a.case,index,'tokens')
            if a.case!='no_visual_tokens':
                assert checked['final_kv_sha256']==old[index]['final_kv_sha256'],(a.case,index,'final KV')
                assert checked['logits_sha256']==old[index]['logits_sha256'],(a.case,index,'logits')
            elif n==0:
                assert eager['tokens']==checked['tokens'],(index,'eager/graph tokens')
                assert eager['logits_sha256']==checked['logits_sha256'],(index,'eager/graph logits')
                assert eager['final_kv_sha256']==checked['final_kv_sha256'],(index,'eager/graph KV')
                del eager
            assert checked['layer_lengths']==[nt if a.case=='no_visual_tokens' or i in blocked else nt+nv for i in range(36)]
            warm=request();assert warm['tokens']==checked['tokens'];capture(False)
            if is_adapter:
                runner.graph=ReplayCounter(runner.graph)
                def replay_counts():return runner.graph.replays,decoder.stats()['replays']
            else:
                def replay_counts():return whole.stats()['replays'],native.stats()['layer_replays']
            before=stats();replay_before=replay_counts()
            trials=[request() for _ in range(3)]
            continuous_trials=[request(continuous=True) for _ in range(3)]
            assert before==stats(),(index,'capture/fallback in timing')
            replay_delta=tuple(b-a for a,b in zip(replay_before,replay_counts()))
            assert replay_delta==(6,42),(index,'Not all forwards replayed a graph',replay_delta)
            assert all(t['tokens']==checked['tokens'] for t in continuous_trials)
            if is_adapter:runner.graph=runner.graph.graph
            assert all(t['tokens']==checked['tokens'] for t in trials)
            row=dict(index=index,case=a.case,shard=a.shard,input_sha256=digest,tokens=checked['tokens'],
                layer_lengths=checked['layer_lengths'],prefill_kv_sha256=checked['prefill_kv_sha256'],final_kv_sha256=checked['final_kv_sha256'],
                trials=trials,continuous_trials=continuous_trials,graph_prefill_replays=6,graph_decode_replays=42,text_tokens=nt,visual_tokens=nv,timed_encoder_calls=0,timed_captures=0,timed_fallbacks=0)
            out.write(json.dumps(row)+'\n');print(json.dumps(dict(case=a.case,done=n+1,expected=len(indices),index=index)),flush=True)
            del hidden,inputs,checked,warm,trials,continuous_trials

def report(a):
    result=[]
    for case in a.cases:
        rows=read_rows((a.run/case).glob('shard*.jsonl'))
        assert set(rows)==set(a.indices or range(999))
        r=dict(case=case,samples=len(rows),peak_GiB=max(t['peak_GiB'] for x in rows.values() for t in x['continuous_trials']))
        for field,prefix in [('trials','matched_legacy_'),('continuous_trials','continuous_')]:
            for stage in ['prefill','decode','total']:
                r[prefix+stage+'_ms']=1000*statistics.mean(statistics.median(t[stage+'_s'] for t in x[field]) for x in rows.values())
        result.append(r)
    for r in result:
        for prefix in ['matched_legacy_','continuous_']:
            for stage in ['prefill','decode','total']:
                r[prefix+stage+'_speedup']=result[0][prefix+stage+'_ms']/r[prefix+stage+'_ms']
    dump(a.run/'RESULTS.json',result)
    lines=[f'# Matched Video-MME speed and memory ({len(rows)} inputs)','',
        'FA2, DeepStack off, batch1, fixed8 outputs, same frozen source/input hashes. Vision and input preparation excluded; LLM and LM head included. Peak allocated GiB includes resident model weights/input buffers/LLM graph pools. All timed prefill and decode forwards verified as CUDA graph replays. Three repeated trials per input. Methods run serially on eight otherwise idle GPUs.',
        'Continuous decode includes greedy selection and all seven forwards with only a final synchronization. Matched-legacy decode measures each forward separately, excluding selection. No FLOP instrumentation in timed calls.','',
        '| Method | Peak GiB | Prefill ms | Continuous decode ms | Total ms | Total speedup |','|---|---:|---:|---:|---:|---:|']
    for r in result:
        lines.append(f"| {r['case']} | {r['peak_GiB']:.3f} | {r['continuous_prefill_ms']:.3f} | {r['continuous_decode_ms']:.3f} | {r['continuous_total_ms']:.3f} | {r['continuous_total_speedup']:.3f} |")
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
        for case in a.cases:
            children=[]
            for shard in range(8):
                log=(a.run/f'{case}_{shard}.log').open('w')
                cmd=[sys.executable,'-m','src.benchmarking','videomme','llm','--','worker','--run',str(a.run),'--previous',str(a.previous),'--case',case,'--shard',str(shard)]
                if a.indices:cmd+=['--indices',*[str(i) for i in a.indices[shard::8]]]
                p=subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=str(a.gpus[shard])),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
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
    p.add_argument('--case',choices=CASES);p.add_argument('--cases',nargs='+',choices=CASES,default=['base','adapter']);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));p.add_argument('--shard',type=int,default=0);p.add_argument('--indices',type=int,nargs='+');p.add_argument('--scheduler-pid',type=int)
    a=p.parse_args();a.run=a.run.resolve();a.previous=a.previous.resolve();a.run.mkdir(parents=True,exist_ok=True)
    os.environ.update(QWEN_VIDEO_NUM_FRAMES='8',QWEN_VIDEO_SAMPLING='full_timestamp_v1',TOKENIZERS_PARALLELISM='false')
    if a.phase=='worker':return worker(a)
    if a.phase=='report':return report(a)
    if len(a.gpus)!=8 or len(set(a.gpus))!=8:raise ValueError('Use eight distinct GPUs')
    if a.indices and len(a.indices)<8:raise ValueError('Queue needs at least one input per GPU')
    if a.indices and len(set(a.indices))!=len(a.indices):raise ValueError('Duplicate input indices')
    frozen=a.run/'driver.py'
    if not frozen.exists():shutil.copy2(__file__,frozen)
    dump(a.run/'protocol.json',dict(source_run=str(a.previous),cases=a.cases,driver_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        samples=len(a.indices) if a.indices else 999,indices=a.indices or list(range(999)),frames=8,tokens=8,deepstack=False,FA2=True,vision_in_timing=False,prefill_graph_capacity=1,decode_graph_capacity=8,
        repetitions=3,decode_modes=['legacy_separate_forward_timing','continuous_including_greedy_selection']))
    return queue(a)

if __name__=='__main__':main()
