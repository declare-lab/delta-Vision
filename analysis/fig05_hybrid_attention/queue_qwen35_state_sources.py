import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCES=['analysis/fig05_hybrid_attention/qwen35_state_sources.py','analysis/fig05_hybrid_attention/qwen35_state_sources_worker.py',
         'analysis/fig05_hybrid_attention/queue_qwen35_state_sources.py','analysis/fig05_hybrid_attention/report_qwen35_state_sources.py',
         'test/diagnostics/test_qwen35_state_sources.py','src/qwen35.py', 'src/model_setup.py',
         'src/qwen35.py','src/benchmarks.py','src/model_setup.py']


def dump(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    plan=json.loads((run/'plan.json').read_text())
    assert {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}==plan['source_sha256']
    validation=json.loads((run/'validate/realworldqa.shard0.jsonl').read_text().splitlines()[0])
    assert validation['native_generation_exact'] and len(validation['layers'])==24
    status=dict(state='running',stage='analysis',started=time.time(),completed=0,total=765)
    running=[]
    try:
        for gpu in range(8):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            log=(run/'logs'/f'analysis{gpu}.log').open('a')
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','analysis/fig05_hybrid_attention/qwen35_state_sources_worker.py',
                '--run',str(run),'--shard',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running.append((gpu,proc,log))
        while True:
            failures=[(i,p.poll()) for i,p,_ in running if p.poll() not in (None,0)]
            if failures:raise RuntimeError(str(failures))
            progress={f.stem:json.loads(f.read_text())['completed'] for f in (run/'analysis').glob('progress*.json')}
            status.update(completed=sum(progress.values()),shard_progress=progress,elapsed_s=time.time()-status['started'],
                workers=[dict(gpu=i,pid=p.pid,returncode=p.poll()) for i,p,_ in running])
            dump(run/'status.json',status)
            if all(p.poll() is not None for _,p,_ in running):break
            time.sleep(10)
        assert status['completed']==765
        status['stage']='report';dump(run/'status.json',status)
        subprocess.run([str(ROOT/'.venv/bin/python'),'analysis/fig05_hybrid_attention/report_qwen35_state_sources.py','--run',str(run)],cwd=ROOT,check=True)
        status.update(state='complete',stage=None,elapsed_s=time.time()-status['started']);dump(run/'status.json',status)
    except BaseException as exc:
        for _,p,_ in running:
            if p.poll() is None:p.terminate()
        for _,p,_ in running:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        status.update(state='failed',error=repr(exc));dump(run/'status.json',status);raise
    finally:
        for _,_,log in running:log.close()


if __name__=='__main__':main()
