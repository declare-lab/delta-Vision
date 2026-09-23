"""Keep eight GPUs occupied by scheduling independent frozen evaluation shards."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    return path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--handoff',type=Path,required=True);a=p.parse_args()
    run=a.run_dir.resolve();root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(run/'source'))
    from src.qwen35_experiment import dump,sha
    from scripts.queue_qwen35_random44_eval import report
    config=json.loads((run/'config.json').read_text());plan=json.loads((run/'plan.json').read_text())
    methods=config['methods']
    assert methods and len(set(methods)) == len(methods)
    for name,digest in plan['source_sha256'].items():assert sha(run/'source'/name)==digest,name
    assert json.loads((run/'validation.json').read_text())['passed']
    handoff=json.loads(a.handoff.read_text());assert not alive(handoff['old_supervisor_pid'])
    env=dict(os.environ,PYTHONPATH=str(root/'artifacts/dependencies/qwen35_python')+os.pathsep+str(run/'source'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    def finished(method,shard):
        for name,info in config['evaluation'].items():
            path=run/'eval'/method/f'{name}.shard{shard}.jsonl'
            if not path.exists():return False
            rows=[json.loads(s) for s in path.read_text().splitlines() if s]
            if [r['index'] for r in rows]!=list(range(shard,info['samples'],8)):return False
        return True
    active={};done=set();pending=[];marked=set()
    for job in handoff['adopted']:
        if alive(job['pid']):active[job['gpu']]=dict(job,process=None,handle=None)
        else:assert finished(job['method'],job['shard'])
    adopted={(j['method'],j['shard']) for j in active.values()}
    pruning_methods=[m for m in methods if m not in ('native','adapter')]
    reference_methods=[m for m in methods if m in ('native','adapter')]
    task_order=[(m,s) for s in range(8) for m in pruning_methods]
    task_order += [(m,s) for s in range(8) for m in reference_methods]
    for key in task_order:
        if key in adopted:continue
        if finished(*key):done.add(key)
        else:pending.append(key)
    def report_partial():
        # Only expose fully completed methods to the reporter, avoiding reads
        # of an incomplete JSON line while another worker flushes a long answer.
        stage=run/'report_stage'
        (stage/'eval').mkdir(parents=True,exist_ok=True)
        (stage/'config.json').write_bytes((run/'config.json').read_bytes())
        for method in methods:
            if all((method,i) in done for i in range(8)):
                link=stage/'eval'/method
                if not link.exists():link.symlink_to(run/'eval'/method,target_is_directory=True)
        report(stage)
        for name in ('partial_summary.json','PARTIAL_RESULTS.md'):
            (run/name).write_bytes((stage/name).read_bytes())
    def status(state,**kwargs):
        dump(run/'status.json',dict(state=state,pid=os.getpid(),updated=time.time(),scheduler='independent_shards',
             workers=[j['pid'] for j in active.values()],
             assignments=[{k:j[k] for k in ('gpu','pid','method','shard')} for j in active.values()],
             completed_shards=len(done),total_shards=len(methods)*8,**kwargs))
    try:
        while pending or active:
            for gpu,job in list(active.items()):
                proc=job['process']
                running=proc.poll() is None if proc else alive(job['pid'])
                if running:continue
                if proc:assert proc.returncode==0,(job['method'],job['shard'],proc.returncode)
                assert finished(job['method'],job['shard']),('Incomplete exited worker',job['method'],job['shard'])
                done.add((job['method'],job['shard']))
                if job['handle']:job['handle'].close()
                del active[gpu]
            for gpu in range(8):
                if gpu in active or not pending:continue
                # Do not claim a GPU acquired by an unrelated workload.
                mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if mem>=1024:continue
                method,shard=pending.pop(0)
                worker='qwen35_worker.py' if method in ('native','adapter') else 'qwen35_pruning_worker.py'
                cmd=[sys.executable,str(run/'source/scripts'/worker),'eval','--run-dir',str(run),'--shard',str(shard)]
                if method in ('native','adapter'):cmd+=['--method',method]
                else:
                    algo,percent=method.rsplit('_',1);cmd+=['--method',algo,'--retention',str(int(percent)/100),'--shards','8']
                handle=(run/'logs'/f'{method}_{shard}.log').open('a')
                proc=subprocess.Popen(cmd,cwd=run/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=dict(gpu=gpu,pid=proc.pid,method=method,shard=shard,process=proc,handle=handle)
            for method in methods:
                if method not in marked and all((method,s) in done for s in range(8)):
                    report_partial()
                    dump(run/f'{method}.complete.json',dict(completed=time.time()))
                    marked.add(method)
            status('evaluating')
            if pending or active:time.sleep(10)
        report(run,complete=True);status('complete',report=str(run/'RESULTS.md'))
    except Exception as exc:
        status('failed',error=repr(exc))
        # Already running shards are left intact; their identity is saved above
        # so a repaired supervisor can safely adopt them again.
        raise

if __name__=='__main__':main()
