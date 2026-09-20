"""Bounded profiling window; resumes the identified training workers on every exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/prefill_slowdown_investigation_20260915"
PIDS = list(range(4120620, 4120628))


def main():
    identities = {}
    for pid in PIDS:
        proc = Path(f"/proc/{pid}")
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "src.pixmo_objective_comparison" not in command or "m4multi64k_video64k" not in command:
            raise RuntimeError(f"Refusing to pause unrecognized PID {pid}")
        identities[pid] = (proc / "stat").read_text().split()[21]
        if (proc / "stat").read_text().split()[2] in ("T", "t"):
            raise RuntimeError(f"PID {pid} was already stopped")
    OUT.mkdir(parents=True, exist_ok=True)
    # Independent watchdog also resumes the same process identities if this parent is interrupted.
    watchdog_code = '''import json,os,signal,sys,time
from pathlib import Path
identities=json.loads(sys.argv[1]);time.sleep(150)
for pid,start in identities.items():
 try:
  if Path('/proc/'+pid+'/stat').read_text().split()[21]==start:os.kill(int(pid),signal.SIGCONT)
 except ProcessLookupError:pass
 except FileNotFoundError:pass
'''
    watchdog = subprocess.Popen([sys.executable, "-c", watchdog_code, json.dumps(identities)], start_new_session=True)
    paused = []
    record = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "pids": PIDS}
    try:
        for pid in PIDS:
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        # CUDA work already queued may finish after the host processes stop. Do not profile outstanding NCCL work.
        readings = []
        for _ in range(4):
            time.sleep(1)
            utilization = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip())
            readings.append(utilization)
        record["gpu0_utilization_after_stop"] = readings
        if any(value > 5 for value in readings[-2:]):
            record["status"] = "aborted_gpu_not_idle"
            print(json.dumps(record), flush=True)
            return
        record["status"] = "profiling"
        print(json.dumps(record), flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        with (OUT / "gpu_profile.log").open("w") as handle:
            result = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "test/diagnostics/profile_baseline_generation.py")],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=120)
        record["profile_exit_code"] = result.returncode
        record["status"] = "profile_complete" if result.returncode == 0 else "profile_failed"
    finally:
        for pid in paused:
            try:
                if Path(f"/proc/{pid}/stat").read_text().split()[21] == identities[pid]:
                    os.kill(pid, signal.SIGCONT)
            except (ProcessLookupError, FileNotFoundError):
                pass
        watchdog.terminate()
        watchdog.wait()
        record["resumed_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        (OUT / "pause_window.json").write_text(json.dumps(record, indent=2))
        print("Training workers resumed", flush=True)


if __name__ == "__main__":
    main()
