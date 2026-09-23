"""Wait for the current recurrent experiment; validate, baseline, train, evaluate."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarks import DEFAULT_BENCHMARK_NAMES, get_benchmark_spec, score_prediction
from src.qwen35_experiment import dump, sha, score_evaluation_prediction
from src.evaluation_sampling import sample_evaluation_rows

PRIOR = ROOT/'artifacts/experiments/document_continued_recurrent_adapter/qwen3vl4b_document_continue_recurrent128_2000_20260920_052526'
DEPS = ROOT/'artifacts/dependencies/qwen35_python'
SOURCE = ['src/__init__.py', 'src/qwen35_embedding.py', 'src/qwen35_experiment.py', 'src/qwen_deepstack.py',
          'src/evaluation_sampling.py',
          'src/benchmarks.py', 'scripts/qwen35_worker.py', 'scripts/queue_qwen35_pixmo.py',
          'configs/qwen35_adapter_requirements.txt', 'test/diagnostics/test_qwen35_embedding.py']


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    model = ROOT/'models/Qwen3.5-4B'
    data = ROOT/'data/train/pixmo/pixmo_ama_full_valid.clean.jsonl'
    config = dict(model='Qwen/Qwen3.5-4B', model_path=str(model),
        model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        architecture='static_embedding_adapter', rank=128, num_layers=32, hidden_size=2560,
        trainable_parameters=20971520, init='new: default Linear down; zero up; residual E',
        teacher='frozen native Qwen3.5-4B, no DeepStack, thinking disabled',
        student='frozen native text path; independent E->residual MLP visual inputs at all32 layers; text-only FFN',
        linear_attention='24 native GatedDeltaNet mixers including all visual state writes and convolution context',
        full_attention='8 native gated full-attention mixers, FA2',
        data=str(data), data_sha256=sha(data), data_name='allenai/pixmo-ask-model-anything',
        image_root=str(data.parent), pixel_area_cache=str(data)+'.pixel_areas.json',
        max_steps=2000, save_every=500, log_every=5, seed=44, world_size=8,
        micro_batch_size_per_gpu=1, gradient_accumulation_steps=4, global_batch=32,
        lr=5e-5, betas=[.9, .95], weight_decay=.01, warmup_ratio=.03,
        min_lr_ratio=.1, lr_scheduler='cosine', grad_clip=1., dtype='bfloat16',
        supervision_loss='distill', kl_topk=1024, temperature=2., lambda_logit=1.,
        loss='Original answer-prefix KL only; teacher top1024 with gold-token inclusion; no hidden/state/attention loss',
        loss_normalization='token',
        accumulation_normalization='answer-token weighted across four samples per rank; then DDP rank mean, matching original PixMo microbatch4',
        batch_sampling='pixel_bucket', pixel_bucket_size=512, enable_thinking=False,
        input_resolution='native processor defaults, no custom pixel or token caps',
        wandb=True, wandb_mode='online', evaluation_sampling_seed=44,
        evaluation_sampling='uniform_without_replacement',
        evaluation_generation=dict(max_new_tokens=64, do_sample=False, unfinished_response='invalid_zero'),
        evaluation={})
    for name in DEFAULT_BENCHMARK_NAMES:
        spec = get_benchmark_spec(name)
        source = ROOT/spec.default_data
        all_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        rows, source_indices = sample_evaluation_rows(all_rows, limit=1000, seed=44)
        assert rows
        for row in rows:
            root = Path(row.get('image_root') or source.parent)
            paths = row.get('images') or [row['image']]
            assert len(paths) == 1 and (root/paths[0]).is_file(), (name, row)
        selected = run/'eval_data'/f'{name}.jsonl'
        selected.parent.mkdir(exist_ok=True)
        selected.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        config['evaluation'][name] = dict(path=str(selected), sha256=sha(selected), samples=len(rows),
            sampling='uniform_without_replacement', seed=44, source_indices=source_indices,
            source=str(source), source_sha256=sha(source), image_root=str(source.parent),
            max_new_tokens=config['evaluation_generation']['max_new_tokens'], metric=spec.metric,
            reported_metric='question_accuracy' if name in ('mme', 'pope') else spec.metric)
    dump(run/'config.json', config)
    sources = {}
    for name in SOURCE:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
        sources[name] = sha(dest)
    weights = {p.name: sha(p) for p in model.iterdir() if p.is_file()}
    distributions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(DEPS)])}
    dump(run/'plan.json', dict(created=time.time(), prior_run=str(PRIOR), source_sha256=sources,
        model_files_sha256=weights, dependencies=distributions,
        stages=['wait_prior_complete', 'hybrid_correctness_validation', 'native_9bench',
                'pixmo_KL_2000step', 'adapter_9bench', 'rescore_and_report'],
        report='per-question score x100; POPE/MME accuracy; VQAv2 soft accuracy; mean of nine unrounded scores',
        validation_failure='stop; never skip validation or fall back to Python DeltaNet'))
    (run/'README.md').write_text('''# Qwen3.5-4B / PixMo-AMA / static embedding rank128

32 independent residual MLPs, all initialized from scratch; 20,971,520 trainable parameters.
Frozen native teacher and backbone. Initial visual embeddings enter each layer's own MLP.
All 24 GatedDeltaNet layers retain original-order visual writes, text writes, causal
convolution and recurrent-state updates. Full attention uses FA2; linear attention
uses FLA and CUDA causal-conv1d. No DeepStack; thinking disabled for both paths.
Native visual mixer readouts are computed then discarded in this first integration;
visual FFNs are skipped. This run measures accuracy, not maximum inference speed.

Validation separately reports the BF16 rounding difference caused by text-only
FFN GEMMs. A positive control pads FFN inputs to the native GEMM shape and requires
exact equality of text hidden states and all hybrid-cache tensors. Production
training/evaluation use text-only FFN. Cache/full-prefix comparison also measures
the native model's own BF16 single-token-vs-full-sequence rounding as a control.
Two CPU tests check hybrid-cache equivalence and the original KL value/gradient.
Use the pinned dependencies in configs/qwen35_adapter_requirements.txt, installed
under artifacts/dependencies/qwen35_python. The shared .venv is left unchanged.

The ONLY training objective is the original answer-token top1024 KL, temperature2.
No hidden-MSE, attention-distribution or recurrent-state supervision was added.
2000 steps, global batch32, microbatch1 x accumulation4 x 8GPUs. W&B online.
Accumulated losses are weighted by answer-token counts to reproduce the original
PixMo microbatch4 token-mean objective and its per-rank sample groups exactly.

After the preceding recurrent run (including its evaluation) finishes:
1. Oracle hidden refill, hybrid cached-decode and real backward correctness checks.
2. Native Qwen3.5 baseline on the same 9 benchmark subsets.
3. PixMo-AMA training, then the adapter's 9 benchmark evaluations.
4. Rescore saved answers and write RESULTS.md and summary.json, including AVG.

Each benchmark uses its existing first1000 items (RealWorldQA765), frozen in eval_data.
POPE/MME are reported as question accuracy, VQAv2 as its existing soft-consensus score.
Processor defaults and original images are preserved. Failures stop the queue.
''')
    dump(run/'status.json', dict(state='prepared', updated=time.time()))


def report(run, config):
    summary, by_method = [], {}
    for method in ('native', 'adapter'):
        row = {'method': method}
        by_method[method] = {}
        for benchmark, info in config['evaluation'].items():
            predictions = []
            for shard in range(8):
                path = run/'eval'/method/f'{benchmark}.shard{shard}.jsonl'
                predictions.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            predictions.sort(key=lambda x: x['index'])
            assert [x['index'] for x in predictions] == list(range(info['samples']))
            rows = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
            scores = []
            for pred, original in zip(predictions, rows):
                scored = score_evaluation_prediction(pred, original, info['metric'])
                assert scored['score'] == pred['score']
                scores.append(scored['score'])
            row[benchmark] = 100*sum(scores)/len(scores)
            by_method[method][benchmark] = predictions
        row['avg'] = sum(row[b] for b in config['evaluation'])/9
        summary.append(row)
    for name in config['evaluation']:
        for native, adapted in zip(by_method['native'][name], by_method['adapter'][name]):
            assert native['input_ids_sha256'] == adapted['input_ids_sha256']
            assert native['image_grid_thw'] == adapted['image_grid_thw']
    dump(run/'summary.json', summary)
    names = list(config['evaluation']) + ['avg']
    lines = ['# Qwen3.5-4B: native vs PixMo static embedding adapter rank128', '',
             'Scores are percentages; MME/POPE use question accuracy. AVG averages nine unrounded scores.', '',
             '| Method | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for row in summary:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[b]:.2f}' for b in names)+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def execute(run):
    plan = json.loads((run/'plan.json').read_text())
    config = json.loads((run/'config.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(run/'source'/name) == digest, name
    (run/'logs').mkdir(exist_ok=True)
    def status(state, **more):
        dump(run/'status.json', dict(state=state, pid=os.getpid(), updated=time.time(), **more))
    status('waiting_prior', prior=str(PRIOR))
    while True:
        prior = json.loads((PRIOR/'status.json').read_text())
        if prior['state'] == 'complete': break
        if 'failed' in prior['state']:
            raise RuntimeError('Prior recurrent experiment failed; queue stopped: '+str(prior))
        time.sleep(30)
    status('waiting_gpus')
    while True:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        if len(used.splitlines()) == 8 and all(int(x) < 1024 for x in used.splitlines()): break
        time.sleep(30)
    env = dict(os.environ, PYTHONPATH=str(DEPS)+os.pathsep+str(run/'source'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='online', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_RUN_ID', None)
    worker = str(run/'source/scripts/qwen35_worker.py')
    def single(stage, command, gpu='0'):
        status(stage, command=command)
        with (run/'logs'/f'{stage}.log').open('a') as handle:
            subprocess.run(command, cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not (run/'validation.json').exists():
        single('validation', [sys.executable, worker, 'validate', '--run-dir', str(run)])
    assert json.loads((run/'validation.json').read_text())['passed']
    def evaluation(method):
        active = []
        status('evaluating_'+method)
        try:
            for shard in range(8):
                handle = (run/'logs'/f'eval_{method}_{shard}.log').open('a')
                cmd = [sys.executable, worker, 'eval', '--run-dir', str(run), '--method', method,
                       '--shard', str(shard)]
                process = subprocess.Popen(cmd, cwd=run/'source', env=dict(env, CUDA_VISIBLE_DEVICES=str(shard)),
                                           stdout=handle, stderr=subprocess.STDOUT)
                active.append((process, handle))
            while any(p.poll() is None for p, _ in active):
                assert all(p.poll() in (None, 0) for p, _ in active), 'Evaluation worker failed'
                time.sleep(10)
            assert all(p.returncode == 0 for p, _ in active)
        finally:
            for p, handle in active:
                if p.poll() is None: p.terminate()
                p.wait()
                handle.close()
    evaluation('native')
    final = run/'checkpoints/qwen35_embedding_adapter_step2000.pt'
    if not final.exists():
        single('training', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node', '8', worker, 'train', '--run-dir', str(run)], '0,1,2,3,4,5,6,7')
    assert final.exists()
    evaluation('adapter')
    report(run, config)
    status('complete', checkpoint=str(final), report=str(run/'RESULTS.md'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--start-prepared', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/qwen35_pixmo'/(
        'qwen35_4b_embedding128_pixmo2000_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if not args.start_prepared: prepare(run)
    print('RUN_DIR='+str(run), flush=True)
    if args.prepare_only: return
    try:
        execute(run)
    except Exception as error:
        dump(run/'status.json', dict(state='failed', error=repr(error), updated=time.time(), pid=os.getpid()))
        raise


if __name__ == '__main__': main()
