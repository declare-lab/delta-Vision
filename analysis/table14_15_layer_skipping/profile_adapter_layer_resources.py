"""Separate warmed graph device execution from end-to-end Video-MME wall time."""
import argparse,json,os,signal,subprocess,sys,time,statistics
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]

def profile(run,case):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.benchmarking.engines import adapter as ab
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    from src.data import QwenBenchmarkDataset
    from src.model import load_qwen_embedding_adapter_checkpoint
    from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
    from src.attention import optimize_qwen_attention_metadata
    from src.model_setup import disable_qwen_deepstack
    from src.kernels import FusedQwenNorms
    from src.kernels import QwenExactRoPE
    from src.kernels import QwenFusedProjections
    from src.graphs import NativeDecoderGraphs
    from src.graphs import QwenWholePrefillGraphs
    from src.graphs import fixed_greedy
    protocol=json.loads((run/'protocol.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(42)
    device=torch.device('cuda:0')
    model,processor=load_baseline_model('base',protocol['model'],torch.bfloat16,device,1.,'flash_attention_2')
    disable_qwen_deepstack(model);optimize_qwen_attention_metadata(model)
    adapter=None;blocked=protocol['cases'][case]
    if case=='base':
        norms=FusedQwenNorms(model);rope=QwenExactRoPE(model);projections=QwenFusedProjections(model)
        native=NativeDecoderGraphs(model,max_shapes=8,prefill_layers=False,packed_kv=True,max_prefill_shapes=1)
        whole=QwenWholePrefillGraphs(model,max_shapes=1)
        def request(sync_steps=False):
            return fixed_greedy(model,inputs,8).sequences[0,-8:].tolist()
    else:
        adapter,meta=load_qwen_embedding_adapter_checkpoint(protocol['checkpoint'],model.model.language_model,device,torch.bfloat16)
        settings=SimpleNamespace(last_logits_only=True,attn_implementation='flash_attention_2',cuda_graph=True,cuda_graph_context=True,
            compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,adapter_decode_cache_mode='fast',
            adapter_exact_optimizations=True,adapter_max_optimizations=True,adapter_blocked_visual_layers=blocked)
        prefill=build_qwen_fast_adapter_prefill(model,adapter,settings);decoder=model._adapter_decode_graph_runner
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        def request(sync_steps=False):
            logits,_,_,_,cache=prefill(inputs)
            if sync_steps:torch.cuda.synchronize()
            generated=[]
            for step in range(8):
                scores=logits[:,-1].to(torch.float32,copy=True);scores[:,eos]=-float('inf')
                token=scores.argmax(-1).view(1,1);generated.append(token)
                if step<7:
                    if sync_steps:torch.cuda.synchronize()
                    logits,cache=decoder(model,adapter,token,cache,logits_to_keep=1)
                    if sync_steps:torch.cuda.synchronize()
            return torch.cat(generated,dim=1)[0].tolist()
    ds=QwenBenchmarkDataset(protocol['manifest'],processor,'videomme',data_root=str(ROOT/'data/benchmarks/videomme'),
        cache_dir=str(ROOT/'test/results/adapter_exact_20260915/inputs'))
    def graph_ms(graph,repeats=15):
        values=[]
        for _ in range(3):
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):graph.replay()
            end.record();end.synchronize();values.append(start.elapsed_time(end)/repeats)
        return statistics.median(values)
    out=[]
    with torch.inference_mode():
        for index in [0,333,666]:
            inputs=_qwen_inputs_from_item(ds[index],device)
            if case=='base':native.allow_capture=whole.allow_capture=True
            else:decoder.allow_capture=True
            expected=request();assert request()==expected
            if case=='base':native.allow_capture=whole.allow_capture=False
            else:decoder.allow_capture=False
            wall={}
            for sync_steps in ([False] if case=='base' else [False,True]):
                times=[]
                for _ in range(7):
                    torch.cuda.synchronize();t=time.perf_counter();got=request(sync_steps);torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-t));assert got==expected
                wall['request_wall_sync_steps_'+str(sync_steps)]=statistics.median(times)
            if case=='base':
                # Native graph entries consist of vision cache then whole-decode cache (no selectors or layer graphs).
                vision_entries=[list(e.values()) for e in native.entries if e and not hasattr(next(iter(e.values())),'packed_input')]
                vision=[e for entries in vision_entries for e in entries if type(e).__name__=='VisionGraph']
                if not vision:
                    vision=[e for entries in vision_entries for e in entries if type(e).__name__ not in ('PackedNativeDecodeGraph','NativeDecodeGraph')]
                assert len(vision)==1,[(type(e).__name__) for entries in vision_entries for e in entries]
                context=graph_ms(vision[0].graph)
                language=graph_ms(next(iter(whole.entries.values())).graph)
                decode_entries=[e for entries in native.entries for e in entries.values() if type(e).__name__=='PackedNativeDecodeGraph']
            else:
                context=graph_ms(prefill.graph_runners[0].graph)
                language=graph_ms(prefill.graph_runners[1].graph)
                decode_entries=[e for entries in decoder.native.entries for e in entries.values()]
            # Capacity eight may retain one old shape; use the seven newest entries in the decode LRU.
            decode_entries=decode_entries[-7:];assert len(decode_entries)==7
            decode_values=[graph_ms(e.graph) for e in decode_entries]
            row=dict(case=case,index=index,visual_tokens=int(inputs['mm_token_type_ids'].ne(0).sum()),
                text_tokens=int(inputs['mm_token_type_ids'].eq(0).sum()),tokens=expected,
                context_graph_gpu_ms=context,language_prefill_graph_gpu_ms=language,
                decode_graph_gpu_ms=sum(decode_values),decode_graph_gpu_ms_per_step=decode_values,
                blocked_layers=blocked,**wall)
            out.append(row);print(json.dumps(row),flush=True)
    dest=run/'stage_profile';dest.mkdir(exist_ok=True);(dest/f'{case}.json').write_text(json.dumps(out,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case');a=p.parse_args();run=a.run.resolve()
    os.environ.update(QWEN_VIDEO_NUM_FRAMES='8',QWEN_VIDEO_SAMPLING='full_timestamp_v1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if a.case:return profile(run,a.case)
    queue=int((run/'queue.pid').read_text());stage=run/'stage_profile';stage.mkdir(exist_ok=True);children=[]
    def interrupted(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(5)
        for case in ['base','adapter','first5_last10','first10_last10']:
            with (stage/f'{case}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,__file__,'--run',str(run),'--case',case],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
                children.append(child);code=child.wait();assert code==0,(case,code)
        (stage/'status.json').write_text(json.dumps(dict(state='complete'))+'\n')
    except BaseException as e:
        (stage/'status.json').write_text(json.dumps(dict(state='failed',error=repr(e)))+'\n');raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate();child.wait()
        os.kill(queue,signal.SIGCONT)
        (run/'AUDIT_PAUSE_RESTORED.json').write_text(json.dumps(dict(queue_pid=queue,resumed=time.time()))+'\n')
if __name__=='__main__':main()
