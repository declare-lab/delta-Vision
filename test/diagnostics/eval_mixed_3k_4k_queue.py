"""Evaluate saved 3k then 4k checkpoints, reusing the completed 2k results."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT/'artifacts/experiments/qwen_mixed_adapter/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915'
OUT = ROOT/'test/results'/f'{TRAIN.name}_checkpoint_comparison'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'order': [2000,3000,4000], 'results': {}}
    def save(**updates):
        state.update(updates, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        tmp = OUT/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(OUT/'status.json')
    for step in (2000,3000,4000):
        result_dir = ROOT/'test/results'/f'{TRAIN.name}_step{step}_8gpu'
        if step != 2000:
            save(state='evaluating', step=step)
            with (OUT/f'step{step}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, str(ROOT/'test/diagnostics/eval_saved_mixed_checkpoint.py'),
                    '--train-dir',str(TRAIN),'--step',str(step),'--gpus','0,1,2,3,4,5,6,7','--memory-gib','64'],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    save(child_pid=child.pid)
                    time.sleep(10)
            if child.returncode:
                save(state='failed', exit_code=child.returncode)
                raise RuntimeError(f'Step {step} evaluation failed; see {OUT}/step{step}.log')
        result = json.loads((result_dir/'status.json').read_text())
        assert result['state']=='complete', (step,result)
        state['results'][str(step)] = result['results']
        save()
        print('COMPLETE',step,result['results'],flush=True)
    save(state='complete')
    (OUT/'results.json').write_text(json.dumps(state['results'],indent=2)+'\n')

if __name__=='__main__':
    main()
