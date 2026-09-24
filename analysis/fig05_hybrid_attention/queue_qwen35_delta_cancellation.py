"""Eight single-GPU shards, persistent status, restore temporary RAM-only load."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    assert json.loads((run/'validation.json').read_text())['passed']
    started=time.time();workers=[];status=dict(state='running',started=started,total=2765)
    try:
        for gpu in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
            log=(run/'logs'/f'shard{gpu}.log').open('a')
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','analysis/fig05_hybrid_attention/qwen35_delta_cancellation_worker.py','--run',str(run),'--shard',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append((gpu,proc,log))
        while True:
            failures=[(g,p.poll()) for g,p,_ in workers if p.poll() not in (None,0)]
            if failures:raise RuntimeError(str(failures))
            count={f.name:sum(1 for _ in f.open()) for f in (run/'results').glob('*.jsonl')}
            status.update(completed=sum(count.values()),shards=count,elapsed_s=time.time()-started,
                workers=[dict(gpu=g,pid=p.pid,returncode=p.poll()) for g,p,_ in workers])
            dump(run/'status.json',status)
            if all(p.poll() is not None for _,p,_ in workers):break
            time.sleep(15)
        assert status['completed']==2765,status
        subprocess.run([str(ROOT/'.venv/bin/python'),'analysis/fig05_hybrid_attention/report_qwen35_delta_cancellation.py','--run',str(run)],cwd=ROOT,check=True)
        status.update(state='complete',elapsed_s=time.time()-started);dump(run/'status.json',status)
    except BaseException as e:
        for _,p,_ in workers:
            if p.poll() is None:p.terminate()
        for _,p,_ in workers:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        status.update(state='failed',error=repr(e));dump(run/'status.json',status);raise
    finally:
        for _,_,log in workers:log.close()
        control=Path('/dev/shm/qwen8b_adapter_load_20260921/control.py')
        if control.exists():
            r=subprocess.run([str(ROOT/'.venv/bin/python'),str(control),'start','--coexist'],capture_output=True,text=True)
            dump(run/'burn_restoration.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))


if __name__=='__main__':main()
