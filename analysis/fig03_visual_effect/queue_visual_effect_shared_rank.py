"""Queue the Table 6 rerun after the active Qwen3.5 evaluation completes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.fig03_visual_effect.visual_effect_shared_rank import dump

PRIOR = ROOT/'artifacts/experiments/qwen35_pruning/qwen35_random44_corrected_20260920_101318'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run, prior):
    from src.benchmarks import get_benchmark_spec
    from src.data import sample_evaluation_rows
    assert not run.exists()
    run.mkdir(parents=True)
    config = dict(repository=str(ROOT), prior=str(prior), seed=44,
        models=dict(qwen=str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
                    llava=str(Path(__file__).resolve().parents[2] / "model/llava-1.5-7b-hf")),
        ranks=[0, 16, 32, 64, 128, 256, 512, 1024],
        conditions=['native', 'full_effect', 'rank0', 'rank16', 'rank32', 'rank64', 'rank128', 'rank256', 'rank512', 'rank1024'],
        datasets={}, dtype='bfloat16', attention='flash_attention_2', deepstack='off', max_new_tokens=8,
        generation='Greedy; identical structured-answer/EOS stopping; original template and image processing; no KV cache',
        scoring='Per-question benchmark accuracy; all questions retained; generated text scored by pinned shared scorer',
        trajectory='Text-only rollout with external native-trajectory Delta at every layer; native trace recomputed per generated prefix',
        basis_protocol='Per model/dataset/layer shared basis; ALL selected prompt text positions; no labels; no centering; FP64 sum Delta.T@Delta/eigh; FP32 projection; no TF32',
        basis_scope='Transductive prompt-only; not held-out generalization; no token sampling',
        changes_vs_archived_table6=['Uncentered ALL-prompt second moments, replacing centered/capped token bank',
            'FP32 differences and reconstruction; native BF16 residual order',
            'Uniform seed44 sampling; FA2/DeepStack off; current pinned scorer',
            'One identical fresh-prefix generation/stopping procedure for native, full and all ranks'],
        numerical_control='Exact per-layer attention reconstruction; 3-question native/full smoke per model/dataset; logit relative RMS <0.03 and equal parsed answers',
        full_accuracy_control='Report native/full score gap and per-question mismatches; flag |gap|>1pp as failing full restoration control')
    for kind, model in config['models'].items():
        assert Path(model, 'config.json').is_file(), model
    for name in ['sqa', 'mmstar', 'realworldqa']:
        source = ROOT/get_benchmark_spec(name).default_data
        rows = [json.loads(s) for s in source.read_text().splitlines() if s]
        selected, indices = sample_evaluation_rows(rows, limit=1000, seed=44)
        for row in selected:
            root = Path(row.get('image_root') or source.parent).resolve()
            row['image_root'] = str(root)
            paths = row.get('images') or [row['image']]
            assert len(paths) == 1 and (root/paths[0]).is_file(), (name, paths)
        target = run/'eval_data'/f'{name}.jsonl'
        target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected))
        config['datasets'][name] = dict(path=str(target), sha256=sha(target), samples=len(selected), source=str(source),
            source_sha256=sha(source), source_indices=indices, image_root=str(source.parent), sampling='uniform_without_replacement')
    dump(run/'config.json', config)
    # Snapshot dependencies so later changes to the live scorer cannot affect this run.
    files = (list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')) + list((ROOT/'baselines').glob('*.py'))) + [Path(__file__).resolve(), ROOT/'test/diagnostics/test_visual_effect_shared_rank.py']
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = run/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha(target)
    dump(run/'plan.json', dict(source_sha256=hashes, model_config_sha256={k: sha(Path(v)/'config.json') for k,v in config['models'].items()},
        model_sample_pairs=2*sum(d['samples'] for d in config['datasets'].values()), phases=['wait_prior','smoke','basis','eval','report']))
    dump(run/'status.json', dict(state='prepared', prior=str(prior)))
    (run/'logs').mkdir()
    (run/'README.md').write_text('''# Table 6: shared uncentered visual-effect rank rerun

Two frozen models (Qwen3-VL-4B and LLaVA-1.5-7B), SQA/MMStar/RealWorldQA.
Ranks 0/16/32/64/128/256/512/1024; native and full-effect controls.
SQA/MMStar 1000, RealWorldQA 765; seed44 random subset, identical across models.
FA2, BF16; Qwen DeepStack off. Eight GPUs after the recorded preceding run completes.

Delta is native joint attention output at text positions minus text-only attention
on the SAME native states, after W_O and before residual. Replay evolves only text
states, while each layer's injected Delta remains sourced from the native trace.
Native traces are recomputed from each condition's generated prefix. Shared-prefix
trace/image-feature reuse is only within the same question and identical prefix.

All calibration prompts are used without answers or token subsampling. Per layer,
sum Delta.T@Delta in FP64 without centering, eigendecompose in FP64 and retain the
largest directions; FP32 projection. No mean is subtracted or restored. This aligns
the basis method with Table 7 while retaining Table 6's external-native trajectory.
The actual bases are separately built for this protocol; they are not claimed to
be the same tensors as historical Table 7 (sampling/backend/DeepStack differ).

Layer effects are reconstructed in FP32 before the native BF16 residual addition.
Native and all intervention modes use the same no-cache greedy generation, capped
at 8 tokens; stop at an extracted structured answer or EOS. All rows count in accuracy.
This is a mechanism experiment, not a latency or deployable compression benchmark.

CPU tests check full restoration, rank0 vs text-only at original positions, full-rank
projection, unchanged teacher trajectory, and a dominant-mean counterexample.
Real-model smoke checks run first for each model/dataset; failures stop the queue.
Final reports include the full-effect accuracy gap and numerical mismatch statistics.
''')


def report(run):
    config = json.loads((run/'config.json').read_text())
    records = []
    lines = ['# Uncentered visual-effect rank accuracy (%)', '',
        'FA2; DeepStack off; seed44; SQA/MMStar 1000, RealWorldQA 765.',
        'External native-trajectory effect; shared uncentered prompt-only basis; no src.training.', '',
        '| Model | Dataset | N | Native | Full effect | '+ ' | '.join(f'r={r}' for r in config['ranks'])+' |',
        '|---|---|---:|---:|---:|'+'---:|'*len(config['ranks'])]
    for model in config['models']:
        for name, info in config['datasets'].items():
            folder = run/f'{model}_{name}'
            if not all((folder/f'eval_{i}.done.json').exists() for i in range(8)):
                continue
            rows = sorted([json.loads(s) for i in range(8) for s in (folder/f'eval_{i}.jsonl').read_text().splitlines() if s], key=lambda r:r['sample'])
            assert [r['sample'] for r in rows] == list(range(info['samples']))
            scores = {mode: sum(r['results'][mode]['score'] for r in rows)*100/len(rows) for mode in config['conditions']}
            gap = scores['full_effect'] - scores['native']
            record = dict(model=model, benchmark=name, samples=len(rows), accuracy_pct=scores,
                full_minus_native_pp=gap, full_control_passed=abs(gap)<=1.,
                full_answer_mismatches=sum(r['results']['native']['prediction']!=r['results']['full_effect']['prediction'] for r in rows),
                max_full_logit_relative_rms=max(e['relative_rms'] for r in rows for e in r['full_reconstruction']))
            records.append(record)
            lines.append(f'| {model} | {name} | {len(rows)} | '+ ' | '.join(f'{scores[m]:.2f}' for m in config['conditions'])+' |')
    lines += ['', '## Full restoration control', '', '| Model | Dataset | Full − native (pp) | Answer mismatches | Max logit relative RMS | Pass (≤1 pp) |', '|---|---|---:|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['model']} | {r['benchmark']} | {r['full_minus_native_pp']:+.2f} | {r['full_answer_mismatches']} | {r['max_full_logit_relative_rms']:.6f} | {r['full_control_passed']} |")
    dump(run/'summary.json', records)
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return records


def execute(run):
    config = json.loads((run/'config.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    root = Path(config['repository'])
    prior = Path(config['prior'])
    def status(state, **kw):
        dump(run/'status.json', dict(state=state, pid=os.getpid(), updated=time.time(), prior=str(prior), **kw))
    for name, digest in plan['source_sha256'].items():
        assert sha(run/'source'/name) == digest, name
    for info in config['datasets'].values():
        assert sha(info['path']) == info['sha256']
    while True:
        old = json.loads((prior/'status.json').read_text())
        if old['state'] == 'complete':
            assert (prior/'RESULTS.md').exists()
            break
        status('waiting_previous_evaluation', predecessor_state=old['state'])
        time.sleep(20)
    env = dict(os.environ, PYTHONPATH=str(root/'artifacts/dependencies/qwen35_python')+os.pathsep+str(run/'source'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    jobs = [(phase, model, name, shard) for phase in ['smoke','basis','eval']
            for model in config['models'] for name in config['datasets'] for shard in (range(8) if phase == 'eval' else [0])]
    active = {}
    def marker(job):
        phase, model, name, shard = job
        return run/f'{model}_{name}'/f'{phase}_{shard}.done.json'
    def ready(job):
        phase, model, name, shard = job
        if phase == 'smoke': return True
        if not all(marker(j).exists() for j in jobs if j[0] == 'smoke'): return False
        return phase == 'basis' or marker(('basis', model, name, 0)).exists()
    try:
        while True:
            for gpu, (job, proc, handle) in list(active.items()):
                if proc.poll() is None: continue
                handle.close()
                assert proc.returncode == 0 and marker(job).exists(), (job, proc.returncode)
                del active[gpu]
                report(run)
            claimed = {item[0] for item in active.values()}
            pending = [j for j in jobs if not marker(j).exists() and j not in claimed]
            if not pending and not active: break
            for gpu in range(8):
                if gpu in active: continue
                job = next((j for j in pending if ready(j)), None)
                if job is None: break
                used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used >= 1024: continue
                phase, model, name, shard = job
                handle = (run/'logs'/f'{model}_{name}_{phase}_{shard}.log').open('a')
                command = [str(root/'.venv/bin/python'), '-m', 'analysis.fig03_visual_effect.visual_effect_shared_rank', '--run-dir', str(run),
                           '--model', model, '--benchmark', name, '--phase', phase, '--shard', str(shard)]
                proc = subprocess.Popen(command, cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = (job, proc, handle)
                pending.remove(job)
            status('running', completed=sum(marker(j).exists() for j in jobs), total_jobs=len(jobs),
                   assignments=[dict(gpu=g, job=j, pid=p.pid) for g,(j,p,h) in active.items()])
            time.sleep(10)
        records = report(run)
        assert len(records) == 6
        status('complete' if all(r['full_control_passed'] for r in records) else 'complete_with_full_control_failure',
               report=str(run/'RESULTS.md'))
    except Exception as exc:
        for job, proc, handle in active.values():
            if proc.poll() is None: proc.terminate()
            proc.wait(); handle.close()
        status('failed', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--start-prepared', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/diagnostics'/('visual_effect_uncentered_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if not args.start_prepared: prepare(run, args.prior.resolve())
    print('RUN_DIR='+str(run), flush=True)
    if not args.prepare_only: execute(run)


if __name__ == '__main__': main()
