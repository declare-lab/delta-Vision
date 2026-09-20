"""Scoped watchdog for the baseline -> fixed-Q evaluation pipeline.

GPU idleness is an investigation trigger, not sufficient reason to kill work.
Only these explicitly configured evaluation processes may be restarted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MODULES=('src.multimodal_baseline_suite','src.fixed_q_visual_readout')

def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {}

def dump(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def scoped_processes(baseline,readout):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            parts=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if '-m' not in parts or '--output' not in parts:continue
            module=parts[parts.index('-m')+1]
            if module not in MODULES:continue
            output=Path(parts[parts.index('--output')+1])
            if not output.is_absolute():output=(p/'cwd').resolve()/output
            if output.resolve() not in (baseline,readout):continue
            result.append(dict(pid=int(p.name),module=module,mode=parts[parts.index('-m')+2],output=str(output.resolve())))
        except (OSError,ValueError,IndexError):continue
    return result

def gpu_state():
    try:
        proc=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        return [dict(index=int(a),utilization=int(b),memory_mib=int(c)) for a,b,c in (line.split(',') for line in proc.stdout.splitlines())]
    except (OSError,ValueError,subprocess.SubprocessError):return []

def progress(root,pattern):
    return tuple(sorted((p.name,p.stat().st_size) for p in root.glob(pattern)))

def errors(root):
    records=[]
    for p in root.glob('*.log'):
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-12000));tail=f.read().decode(errors='replace')
        if 'Traceback' in tail or 'OutOfMemoryError' in tail or 'AssertionError' in tail:
            records.append(dict(log=str(p),tail=tail[-5000:]))
    return records

def stop_scoped(baseline,readout):
    # Stop supervisors first, so they cannot replace a worker being terminated.
    processes=scoped_processes(baseline,readout)
    for mode in ('run','worker'):
        for p in processes:
            if p['mode']==mode:
                try:os.kill(p['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
    time.sleep(10)
    for p in scoped_processes(baseline,readout):
        try:os.kill(p['pid'],signal.SIGKILL)
        except ProcessLookupError:pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--readout',required=True)
    ap.add_argument('--idle-seconds',type=int,default=120);ap.add_argument('--stall-seconds',type=int,default=900)
    args=ap.parse_args();baseline=Path(args.baseline).resolve();readout=Path(args.readout).resolve()
    monitor=baseline/'monitor';monitor.mkdir(exist_ok=True)
    # Single listener, even if the user asks to resume monitoring repeatedly.
    import fcntl
    lock=(monitor/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None;last_progress=time.time();idle_since=None;restarts=0;last_launch=0.;phase=None
    children=[]
    while True:
        for proc,log in list(children):
            if proc.poll() is not None:log.close();children.remove((proc,log))
        bs=read_json(baseline/'status.json');rs=read_json(readout/'status.json')
        current='readout' if bs.get('state')=='complete' else 'baseline'
        if bs.get('state')=='complete' and rs.get('state')=='complete':
            dump(monitor/'status.json',dict(state='complete',time=time.time(),restarts=restarts));print('PIPELINE COMPLETE',flush=True);return
        if phase!=current:phase=current;previous=None;last_progress=time.time();idle_since=None
        path=baseline if phase=='baseline' else readout
        mark=progress(path,'*_shard*.jsonl' if phase=='baseline' else 'rows_*.jsonl')
        if mark!=previous:previous=mark;last_progress=time.time()
        gpus=gpu_state();processes=scoped_processes(baseline,readout)
        idle=bool(gpus) and all(g['utilization']<5 for g in gpus)
        idle_since=(idle_since or time.time()) if idle else None
        failures_reported=bool(bs.get('failure_events') or bs.get('failed')) if phase=='baseline' else rs.get('state')=='failed'
        investigate=(idle_since is not None and time.time()-idle_since>=args.idle_seconds) or not processes or failures_reported
        state=dict(state='monitoring',phase=phase,time=time.time(),processes=processes,gpus=gpus,
            seconds_without_progress=time.time()-last_progress,restarts=restarts,investigating=investigate)
        if investigate:
            diagnostic=dict(**state,baseline_status=bs,readout_status=rs,errors=errors(path))
            dump(monitor/'latest_investigation.json',diagnostic)
            # Idleness plus no result progress for 15 minutes is a scoped hang.
            if processes and idle_since is not None and time.time()-idle_since>=args.idle_seconds and time.time()-last_progress>=args.stall_seconds:
                print('SCOPED STALL: restarting evaluation processes',flush=True)
                stop_scoped(baseline,readout);processes=[]
            if not processes and time.time()-last_launch>=60:
                # Do not repeatedly re-run the same unknown deterministic error
                # forever; leave diagnostics and a live listener for repair.
                if restarts>=3:
                    state['state']='needs_code_repair'
                else:
                    if phase=='baseline':
                        cmd=[sys.executable,'-m',MODULES[0],'run','--output',str(baseline),'--methods','base','fastv','dart','divprune','zoo','sparsevlm','visionzip','--readout-after']
                    else:cmd=[sys.executable,'-m',MODULES[1],'run','--output',str(readout)]
                    log=(monitor/f'restart_{restarts}.log').open('a')
                    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    children.append((proc,log));restarts+=1;last_launch=time.time();last_progress=time.time();idle_since=None
                    print('RESUMED',phase,'pid',proc.pid,flush=True)
        dump(monitor/'status.json',state)
        time.sleep(15)

if __name__=='__main__':main()
