"""Video-MME 999: serial eight-GPU matched controls and visual-layer ablations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('RESOURCE_REPO', str(SOURCE)))
sys.path.insert(0, str(SOURCE))
from src.benchmarking.engines import adapter as ab
ab.ROOT = ROOT
ab.CHECKPOINT = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
ab.MANIFEST = ROOT / 'artifacts/diagnostics/video_balanced_base_adapter_20260914/videomme_selected.jsonl'
CASES = {'base': [], 'adapter': [], 'first5_last10': list(range(5))+list(range(26,36)),
         'first10_last10': list(range(10))+list(range(26,36))}
NAMES = {'base':'Qwen3-VL-4B', 'adapter':'Full embedding adapter',
         'first5_last10':'Adapter: first5 + last10 visual off',
         'first10_last10':'Adapter: first10 + last10 visual off'}
BURN = Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')


def rows(paths):
    result = {}
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['index'] not in result, (path,'duplicate')
            result[row['index']] = row
    return result


def worker(a):
    os.chdir(ROOT)
    protocol=a.run/'protocol.json'
    if protocol.exists():
        for name,digest in json.loads(protocol.read_text())['source_sha256'].items():
            assert ab.file_sha(SOURCE/name)==digest,(name,'source changed')
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8')
    blocked = CASES[a.case]
    if a.phase == 'memory':
        options = SimpleNamespace(model=ab.MODEL, checkpoint=str(ab.CHECKPOINT), output=str(a.run/a.case),
            input_cache=str(ROOT/'test/results/adapter_exact_20260915/inputs'), reference_results='',
            indices=a.indices, shard=a.shard, shards=8, tokens=8, runs=3, variant='optimized',
            optimized_level='max', reference_level='exact', packed_decode_kv=True,
            blocked_visual_layers=blocked, verify_eager_decode=a.smoke)
        if a.case == 'base':
            from src.benchmarking.engines.base import worker as run
        else:
            run = ab.worker
        return run(options)
    from src.benchmarking.videomme import resources as rr
    rr.ROOT, rr.CHECKPOINT, rr.MANIFEST = ROOT, ab.CHECKPOINT, ab.MANIFEST
    if blocked:
        from src.benchmarking.common import comparison as bc
        class BlockedRunner(bc.RequestRunner):
            def prefill(self, inputs):
                from src.model import (build_qwen_initial_context, prepare_qwen_embedding_adapter_inputs,
                                       qwen_embedding_adapter_prefill_cache_prepared)
                self.model.model.rope_deltas = None
                hidden, positions = build_qwen_initial_context(self.model, inputs)
                prepared = prepare_qwen_embedding_adapter_inputs(self.model,self.adapter,
                    inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,positions,
                    reuse_position_embeddings=True)
                logits, _, cache = qwen_embedding_adapter_prefill_cache_prepared(self.model,self.adapter,
                    **prepared, logits_to_keep=1,retain_prefix_states=False,blocked_visual_layers=blocked)
                return logits,cache,None
        bc.RequestRunner = BlockedRunner
    return rr.flops(SimpleNamespace(method='base' if a.case=='base' else 'adapter', retention=1.,
        output=a.run/a.case, indices=a.indices, shard=a.shard, shards=8, resume=False))


def audit_case(run,case,expected):
    mem=rows((run/case).glob('optimized_*.jsonl'))
    flop=rows((run/case).glob('flops_*.jsonl'))
    ref=rows((ROOT/'test/results/video_base_20260915/videomme999').glob('optimized_*.jsonl'))
    assert set(mem)==set(flop)==set(expected),(case,len(mem),len(flop))
    for i in expected:
        assert mem[i]['input_sha256']==flop[i]['input_sha256']==ref[i]['input_sha256'],(case,i,'input')
        assert mem[i]['tokens']==flop[i]['tokens'],(case,i,'tokens',mem[i]['tokens'],flop[i]['tokens'])
        assert mem[i]['timed_captures']==mem[i]['timed_fallbacks']==0
        assert len(mem[i]['tokens'])==8 and len(mem[i]['trials'])==3
        nv,nt=flop[i]['visual_tokens'],flop[i]['text_tokens']
        assert flop[i]['layer_lengths']==[nt if j in CASES[case] else nt+nv for j in range(36)]
    ab.dump(run/case/'AUDIT.json',dict(passed=True,samples=len(mem),input_hashes_and_generated_tokens_match=True,
        exact_layer_cache_lengths=True,timed_captures=0,timed_fallbacks=0))


def report(run, expected):
    reference=rows((ROOT/'test/results/video_base_20260915/videomme999').glob('optimized_*.jsonl'))
    result=[]
    base_flops=None
    for case,blocked in CASES.items():
        mem=rows((run/case).glob('optimized_*.jsonl'))
        flop=rows((run/case).glob('flops_*.jsonl'))
        assert set(mem)==set(flop)==set(expected),(case,len(mem),len(flop))
        if case=='base':base_flops=flop
        for i in expected:
            assert mem[i]['input_sha256']==flop[i]['input_sha256']==reference[i]['input_sha256'],(case,i,'input')
            assert mem[i]['tokens']==flop[i]['tokens'],(case,i,'tokens',mem[i]['tokens'],flop[i]['tokens'])
            assert mem[i]['timed_captures']==mem[i]['timed_fallbacks']==0
            nv,nt=flop[i]['visual_tokens'],flop[i]['text_tokens']
            assert flop[i]['layer_lengths']==[nt if j in blocked else nt+nv for j in range(36)],(case,i,'KV lengths')
            assert flop[i]['vision_matrix_flops']==base_flops[i]['vision_matrix_flops'],(case,i,'vision')
        keys=(['request_prefill_time_s','decode_time_s','total_time_s'] if case=='base'
              else ['prefill_s','decode_s','total_s'])
        r=dict(case=case,method=NAMES[case],samples=len(mem),blocked_visual_layers=blocked,
            peak_GiB=max(t['peak_memory_mb'] for row in mem.values() for t in row['trials'])/1024,
            flops_T=statistics.mean(row['request_matrix_flops']-base_flops[i]['vision_matrix_flops']
                for i,row in flop.items())/1e12)
        for name,key in zip(['prefill','decode','total'],keys):
            r[name+'_ms']=1000*statistics.mean(statistics.median(t[key] for t in row['trials']) for row in mem.values())
        r['request_wall_ms']=r['total_ms']
        r['total_ms']=r['prefill_ms']+r['decode_ms']
        result.append(r)
    for r in result:
        r['flops_pct_base']=100*r['flops_T']/result[0]['flops_T']
        for name in ['prefill','decode','total']:r[name+'_speedup']=result[0][name+'_ms']/r[name+'_ms']
    ab.dump(run/'RESULTS.json',result)
    lines=['# Video-MME layer-ablation resources','',
        '999 fixed inputs; FA2/BF16; DeepStack off; adapter fast path; CUDA graphs; batch1; 8 output tokens; 7 decode steps; 3 trials/input, median then mean.',
        'FLOPs exclude visual encoder for all methods. Timing includes fresh visual encoding. Total = Prefill + Decode, excluding between-stage token-selection/loop overhead. Peak is maximum allocated GiB including weights and graph pools. Methods run serially on the same eight GPUs. Input hashes, FLOP-run tokens and cache lengths checked for every input.','',
        '| Method | FLOPs (T) | FLOPs/base | Peak (GiB) | Prefill (ms) | Decode (ms) | Total (ms) | Prefill speedup | Decode speedup | Total speedup |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:
        lines.append(f"| {r['method']} | {r['flops_T']:.4f} | {r['flops_pct_base']:.2f}% | {r['peak_GiB']:.3f} | {r['prefill_ms']:.2f} | {r['decode_ms']:.2f} | {r['total_ms']:.2f} | {r['prefill_speedup']:.3f}× | {r['decode_speedup']:.3f}× | {r['total_speedup']:.3f}× |")
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    ab.dump(run/'AUDIT.json',dict(passed=True, samples_per_case=len(expected),cases=list(CASES),
        paired_inputs_and_tokens=True,layer_cache_lengths=True,timed_captures=0,timed_fallbacks=0))
    return result


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    files=(list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')))+list((ROOT/'baselines').rglob('*.py'))
    files += [ROOT/'analysis/table14_15_layer_skipping/measure_adapter_layer_resources.py',ROOT/'src/benchmarking/videomme/resources.py']
    for src in files:
        dest=run/'source'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    ab.dump(run/'protocol.json',dict(model=ab.MODEL,checkpoint=str(ab.CHECKPOINT),checkpoint_sha256=ab.file_sha(ab.CHECKPOINT),
        manifest=str(ab.MANIFEST),manifest_sha256=ab.file_sha(ab.MANIFEST),cases=CASES,indices=list(range(999)),
        attention='flash_attention_2',deepstack=False,dtype='bfloat16',batch_size=1,tokens=8,decode_steps=7,runs=3,
        video_frames=8,sampling='full_timestamp_v1',prompt_layout='media_first_v1',seed=42,
        gpus=list(range(8)),decode_graph_capacity=8,prefill_graph_capacity=1,vision_graph_capacity=1,
        adapter_fast_path=True,adapter_optimizations='max',flops_exclude_vision=True,timing_includes_vision=True,
        total_definition='sum of per-input prefill and decode stage medians',
        peak='Maximum allocated GiB including weights and graphs',
        historical_report=str(ROOT/'artifacts/reports/video_resources_999_final_20260922/ALL_METRICS.json'),
        source_sha256={str(p.relative_to(ROOT)):ab.file_sha(p) for p in files}))
    ab.dump(run/'status.json',dict(state='prepared'))


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def queue(a):
    run=a.run; children=[]; paused=False; burn_was_running=False
    start=time.time(); done=[]
    def status(state,**kw):ab.dump(run/'status.json',dict(state=state,elapsed_s=time.time()-start,completed=done,**kw))
    def stop_signal(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    scheduler=a.scheduler_pid
    try:
        if scheduler:
            cmd=Path(f'/proc/{scheduler}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'repair_baseline_suite.py queue' in cmd,cmd
            os.kill(scheduler,signal.SIGSTOP);paused=True
        if BURN.exists():
            pidfile=BURN.parent/'pid'
            burn_was_running=pidfile.exists() and alive(int(pidfile.read_text()))
            if burn_was_running:subprocess.run([sys.executable,str(BURN),'stop'],check=True)
        ab.dump(run/'isolation.json',dict(scheduler_pid=scheduler,scheduler_paused=paused,burn_was_running=burn_was_running))
        # Existing evaluation workers finish naturally; no evaluation work is killed.
        while True:
            usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            pids=[int(x.strip()) for x in usage.splitlines() if x.strip().isdigit()]
            status('waiting_for_gpu_drain',gpu_pids=pids)
            if not pids:break
            time.sleep(15)
        (run/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        env=dict(os.environ,RESOURCE_REPO=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            QWEN_VIDEO_NUM_FRAMES='8',QWEN_VIDEO_SAMPLING='full_timestamp_v1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        for case in CASES:
            for phase in ['memory','flops']:
                children=[]
                for shard in range(8):
                    logdir=run/'logs';logdir.mkdir(exist_ok=True)
                    log=(logdir/f'{case}_{phase}_{shard}.log').open('w')
                    cmd=[sys.executable,str(run/'source/analysis/table14_15_layer_skipping/measure_adapter_layer_resources.py'),phase,'--run',str(run),
                         '--case',case,'--shard',str(shard)]
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT)
                    log.close();children.append(proc)
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):raise RuntimeError(f'{case}/{phase} worker failed; see logs')
                    status('running',case=case,phase=phase,pids=[p.pid for p in children if p.poll() is None])
                    time.sleep(5)
                assert all(p.returncode==0 for p in children),(case,phase)
                done.append(f'{case}/{phase}');status('running',case=case,phase=phase)
            audit_case(run,case,range(999))
        report(run,range(999));status('complete')
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        for proc in children:
            if proc.poll() is None:proc.terminate()
        for proc in children:
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if paused and alive(scheduler):os.kill(scheduler,signal.SIGCONT)
        if burn_was_running:
            restore=subprocess.run([sys.executable,str(BURN),'start','--coexist'],capture_output=True,text=True)
            ab.dump(run/'restoration.json',dict(scheduler_resumed=paused,burn_returncode=restore.returncode,stdout=restore.stdout,stderr=restore.stderr))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','queue','memory','flops','report'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=CASES)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--smoke',action='store_true');p.add_argument('--scheduler-pid',type=int)
    a=p.parse_args();a.run=a.run.resolve()
    if a.phase=='prepare':return prepare(a.run)
    if a.phase=='queue':return queue(a)
    if a.phase=='report':return report(a.run,a.indices or range(999))
    return worker(a)

if __name__=='__main__':main()
