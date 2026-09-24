"""Priority RealWorldQA run; suspend/resume only the identified memory job group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def rows(run):
    out = []
    for path in (run/'accuracy').glob('*.jsonl'):
        out.extend(json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n'))
    assert len(out) == len({r['index'] for r in out})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    config = json.loads((run/'config.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    assert {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in plan['source_sha256']} == plan['source_sha256']
    parent = config['priority_preempts_pid']
    paused = False
    running = []
    status = dict(state='starting', started=time.time(), priority_preempts_pid=parent)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        command = Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'analysis/fig05_hybrid_attention/queue_qwen35_memory_probe.py' in command and config['priority_preempts_run'] in command
        assert os.getpgid(parent) == parent, 'Unexpected process group; will not suspend'
        # Pause the parent and all its workers together, preserving progress and caches.
        paused = True
        os.killpg(parent, signal.SIGSTOP)
        dump(run/'suspended_job.json', dict(pid=parent, process_group=parent, command=command,
                                          resume_after_run=True, supervisor_pid=os.getpid()))
        for gpu in range(8):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            log = (run/'logs'/f'worker_{gpu}.log').open('a')
            proc = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u',
                'analysis/fig05_hybrid_attention/qwen35_no_visual_write_worker.py', '--run', str(run), '--shard', str(gpu)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((gpu, proc, log))
        while True:
            failed = [(gpu, proc.poll()) for gpu, proc, _ in running if proc.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(str(failed))
            result = rows(run)
            status.update(state='running', completed=len(result), total=765,
                          elapsed_s=time.time()-status['started'],
                          workers=[dict(gpu=gpu, pid=proc.pid, returncode=proc.poll()) for gpu, proc, _ in running])
            dump(run/'status.json', status)
            if all(proc.poll() is not None for _, proc, _ in running):
                break
            time.sleep(5)
        assert {r['index'] for r in result} == set(range(765))
        baseline = 100*sum(r['native']['score'] for r in result)/765
        ablated = 100*sum(r['no_visual_write']['score'] for r in result)/765
        previous = [json.loads(line) for f in (Path(config['priority_preempts_run'])/'accuracy').glob('realworldqa.shard*.jsonl') for line in f.read_text().splitlines()]
        reference = {r['index']: r for r in previous}
        assert len(reference) == 765
        mismatches = []
        for r in result:
            old = reference[r['index']]
            assert r['input_ids_sha256'] == old['input_ids_sha256']
            v = next(v for v in old['variants'] if v['method'] == 'native' and v['rank'] is None)
            if r['native']['generated_token_ids'] != v['generated_token_ids']:
                mismatches.append(r['index'])
        summary = dict(samples=765, native_accuracy=baseline, no_visual_write_accuracy=ablated,
                       difference_pp=ablated-baseline, baseline_token_mismatch_vs_previous=mismatches,
                       improved=sum(r['no_visual_write']['score'] > r['native']['score'] for r in result),
                       worsened=sum(r['no_visual_write']['score'] < r['native']['score'] for r in result),
                       no_eos={k:sum(not r[k]['stopped_by_eos'] for r in result) for k in ['native','no_visual_write']})
        dump(run/'summary.json', summary)
        (run/'RESULTS.md').write_text(
            '# Qwen3.5-4B RealWorldQA: disable visual delta-rule writes\n\n'
            '765 fixed examples; native FA2/FLA; DeepStack off; unchanged prompt/scoring; max_new_tokens=8.\n'
            'All 24 LA layers: beta=0 at visual positions. Native decay, convolution, visual readout, residual/FFN and full attention retained.\n\n'
            '| Model | Accuracy (%) | Difference (pp) |\n|---|---:|---:|\n'
            f'| Native | {baseline:.2f} | — |\n| No visual writes | {ablated:.2f} | {ablated-baseline:+.2f} |\n')
        status.update(state='complete', elapsed_s=time.time()-status['started'])
        dump(run/'status.json', status)
    except BaseException as exc:
        for _, proc, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in running:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        status.update(state='failed', error=repr(exc))
        dump(run/'status.json', status)
        raise
    finally:
        for _, _, log in running:
            log.close()
        if paused:
            os.killpg(parent, signal.SIGCONT)
            dump(run/'resumed_job.json', dict(pid=parent, resumed_at=time.time()))


if __name__ == '__main__':
    main()
