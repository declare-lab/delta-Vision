"""Evaluate saved checkpoints with the unchanged, audited 4000-step protocol.

The completed 4000-step evaluation is reused. Remaining (step, benchmark,
shard) jobs share eight GPU slots, so a short shard need not wait for the
slowest shard of its benchmark. Inference is delegated to the existing suite.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import multimodal_baseline_suite as suite

BENCHES = ("muirbench", "videomme", "mvbench")
COUNTS = dict(muirbench=1000, videomme=999, mvbench=950)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    opts = parser.parse_args()
    train = opts.train_dir.resolve()
    reference = train / "post_train_eval"
    root = train / "checkpoint_comparison"
    root.mkdir(exist_ok=True)
    lock = (root / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()

    def status(state, **kw):
        suite.dump(root / "status.json", dict(state=state, pid=os.getpid(),
                   elapsed_seconds=time.time()-started,
                   heartbeat_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw))

    # Do not compete with or restart the already running 4000-step evaluation.
    while read(reference / "status.json").get("state") != "complete":
        if (reference / "failure.json").exists():
            raise RuntimeError(read(reference / "failure.json"))
        status("waiting_for_existing_step4000", reference_status=read(reference / "status.json"))
        print("Waiting for existing step4000 evaluation", flush=True)
        time.sleep(10)

    import torch
    torch.set_num_threads(4)
    checkpoints = {}
    for step in (1000, 2000, 3000, 4000):
        path = train / f"checkpoints/qwen_embedding_adapter_step{step}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved.get("global_step", saved.get("step")) == step
        assert saved["args"]["output_mode"] == "embedding_adapter"
        assert all(torch.isfinite(value).all() for value in saved["state_dict"].values())
        checkpoints[step] = dict(path=str(path), sha256=sha(path), finite=True)
        del saved
    suite.dump(root / "checkpoints.json", checkpoints)

    originals = {}
    for bench in BENCHES:
        rows = [json.loads(line) for path in (reference / bench).glob("embedding_adapter_shard*.jsonl")
                for line in path.open()]
        assert len(rows) == COUNTS[bench]
        assert sorted(row["index"] for row in rows) == list(range(COUNTS[bench]))
        originals[bench] = {row["index"]: row for row in rows}

    arguments = {}
    jobs = []
    for step in (1000, 2000, 3000):
        for bench in BENCHES:
            out = root / f"step{step}" / bench
            out.mkdir(parents=True, exist_ok=True)
            args = suite.make_parser().parse_args([
                "run", "--output", str(out), "--methods", "embedding_adapter",
                "--checkpoint", checkpoints[step]["path"], "--benchmarks", bench,
                "--manifest-map", str(reference / "manifest_map.json"),
                "--limit", "1000", "--shards", "8", "--prompt-layout", "media_first_v1",
                "--max-new-tokens", "8" if bench == "muirbench" else "128", "--json-only"])
            # Clone the verified protocol, changing only checkpoint and output.
            plan = copy.deepcopy(read(reference / bench / "plan.json"))
            for source, digest in plan["source_sha256"].items():
                assert sha(source) == digest, ("Source changed", source)
            assert sha(suite.manifest_path(args, bench)) == plan["dataset_sha256"][bench]
            plan.update(args=vars(args), checkpoint=checkpoints[step]["path"],
                        checkpoint_sha256=checkpoints[step]["sha256"],
                        adapter_checkpoints={"embedding_adapter": {k: checkpoints[step][k] for k in ("path", "sha256")}},
                        after_baselines=None)
            if (out / "plan.json").exists():
                assert read(out / "plan.json") == plan, "Cannot mix changed protocols/checkpoints"
            else:
                suite.dump(out / "plan.json", plan)
            arguments[step, bench] = args
            for shard in range(8):
                if not suite.shard_complete(out, "embedding_adapter", shard, args):
                    jobs.append((step, bench, shard))

    # Source and preprocessing code are unchanged. Each slot runs a stock suite
    # worker with one explicit checkpoint and one dataset, preserving stop rules.
    running, attempts, failures = {}, {}, []
    def report(final=False):
        table = []
        for step in (1000, 2000, 3000, 4000):
            for bench in BENCHES:
                out = reference / bench if step == 4000 else Path(arguments[step, bench].output)
                rows = [json.loads(line) for path in out.glob("embedding_adapter_shard*.jsonl")
                        for line in path.open() if line.strip()]
                assert len({r["index"] for r in rows}) == len(rows)
                for row in rows:
                    old = originals[bench][row["index"]]
                    assert row["input_sha256"] == old["input_sha256"], (step, bench, row["index"])
                    assert row["dataset_sha256"] == old["dataset_sha256"]
                    assert row["deepstack_enabled"] is False
                    assert row["max_new_tokens"] == old["max_new_tokens"]
                    assert row["prompt_layout"] == old["prompt_layout"]
                if final:
                    assert len(rows) == COUNTS[bench], (step, bench, len(rows))
                table.append(dict(step=step, benchmark=bench, n=len(rows), expected=COUNTS[bench],
                                  accuracy=100*sum(r["score"] for r in rows)/len(rows) if rows else None,
                                  complete=len(rows)==COUNTS[bench], input_hashes_verified=True,
                                  result_directory=str(out)))
        suite.dump(root / "results.json", table)
        lines = ["# Embedding Adapter: 1000 / 2000 / 3000 / 4000 steps", "",
                 "Same frozen samples, physical image order and original M-RoPE; DeepStack OFF. "
                 "Greedy uncached decoding; MuirBench max 8 tokens, videos max 128. "
                 "MuirBench: random 1000, seed 42. Video-MME: existing 999 balanced samples. "
                 "MVBench: existing 950 samples (19 tasks × 50). No resampling. "
                 "Every input tensor hash is checked against the completed step-4000 evaluation.", "",
                 "| Step | MuirBench (1000) | Video-MME (999) | MVBench (950) |",
                 "|---:|---:|---:|---:|"]
        for step in (1000, 2000, 3000, 4000):
            cells = []
            for row in (r for r in table if r["step"] == step):
                cells.append(f'{row["accuracy"]:.2f}%' if row["complete"] else f'pending ({row["n"]}/{row["expected"]})')
            lines.append(f"| {step} | " + " | ".join(cells) + " |")
        (root / "README.md").write_text("\n".join(lines) + "\n")

    while jobs or running:
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        used = [int(x) for x in memory.splitlines()]
        assert len(used) == 8
        for gpu in range(8):
            if gpu in running or not jobs or used[gpu] >= 1024:
                continue
            job = jobs.pop(0)
            step, bench, shard = job
            args = arguments[step, bench]
            attempts[job] = attempts.get(job, 0) + 1
            log = (Path(args.output) / f"embedding_adapter_shard{shard}.log").open("a")
            cmd = [sys.executable, "-m", "src.multimodal_baseline_suite", "worker",
                   "--output", args.output, "--method", "embedding_adapter", "--checkpoint", args.checkpoint,
                   "--benchmarks", bench, "--shard", str(shard), "--shards", "8", "--limit", "1000",
                   "--manifest-map", args.manifest_map, "--prompt-layout", args.prompt_layout,
                   "--max-new-tokens", str(args.max_new_tokens)]
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"))
            running[gpu] = (proc, log, job)
            print("START", job, "GPU", gpu, "PID", proc.pid, flush=True)
        for gpu, (proc, log, job) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            args = arguments[job[:2]]
            okay = code == 0 and suite.shard_complete(Path(args.output), "embedding_adapter", job[2], args)
            print("FINISH", job, "code", code, "complete", okay, flush=True)
            if not okay:
                if attempts[job] < 2:
                    jobs.append(job)
                else:
                    failures.append(dict(job=job, code=code))
        # Workers append whole JSON records line-buffered. A partially observed
        # final write is retried on the next heartbeat, not treated as a result.
        try:
            report()
        except json.JSONDecodeError:
            pass
        status("running", pending=len(jobs), active=[dict(gpu=g, pid=v[0].pid, job=v[2]) for g, v in running.items()], failures=failures)
        if jobs or running:
            time.sleep(10)
    if failures:
        status("failed", failures=failures)
        raise RuntimeError(failures)
    for args in arguments.values():
        suite.aggregate(args, complete=True)
    report(final=True)
    status("complete")
    print("ALL COMPLETE", root / "README.md", flush=True)


if __name__ == "__main__":
    main()
