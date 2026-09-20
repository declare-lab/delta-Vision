"""Foreground 8-GPU training: existing embedding adapter, multi/video 50:50."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent / 'vision-kv-inject-attention-sink'
SOURCE = ROOT / 'data/train/multimodal_subset_20260911'
REFERENCE = ROOT / 'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/config.json'


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--data-root', type=Path, default=SOURCE,
                        help='Audited mixed_train.jsonl bundle; old data remains the default')
    parser.add_argument('--require-unique-samples', action='store_true',
                        help='Reject a schedule that repeats any QA during the 4000 steps')
    parser.add_argument('--wait-for-free-gpus', action='store_true')
    parser.add_argument('--wait-for-pid', type=int,
                        help='Wait for an existing job to exit; never stop it')
    parser.add_argument('--eval-after', action='store_true',
                        help='Evaluate the final checkpoint on the three frozen manifests')
    args = parser.parse_args()
    source = args.data_root.resolve()
    run_name = 'qwen3vl4b_embedding_multi50_video50_rank128_4000_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    dest = args.run_dir.resolve() if args.run_dir else ROOT / 'artifacts/experiments/qwen_mixed_adapter' / run_name
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    dest.mkdir(parents=True)
    run_name = dest.name
    assert json.loads((source / 'integrity_audit.json').read_text())['passed']
    original = [json.loads(line) for line in (source / 'mixed_train.jsonl').open() if line.strip()]
    areas = json.loads((source / 'mixed_train.jsonl.pixel_areas.json').read_text())['areas']
    assert len(original) == len(areas)
    pairs = [(r, a) for r, a in zip(original, areas) if r['mixture_source'] in ('multi_image', 'video')]
    rows = [r for r, _ in pairs]
    counts = Counter(r['mixture_source'] for r in rows)
    assert set(counts) == {'multi_image', 'video'} and min(counts.values()) > 0, counts
    if args.require_unique_samples:
        assert all(counts[k] >= 64000 for k in counts), counts
        assert len({r['id'] for r in rows}) == len(rows), 'Duplicate QA IDs'
    manifest = dest / 'train_multi50_video50.jsonl'
    with manifest.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    area_file = Path(str(manifest) + '.pixel_areas.json')
    dump(area_file, dict(data=str(manifest), count=len(rows), areas=[a for _, a in pairs]))
    sys.path.insert(0, str(WORK))
    from src.multimodal_training import mixed_pixel_bucket_order
    order = mixed_pixel_bucket_order(rows, [a for _, a in pairs], ratios='multi_image:0.5,video:0.5',
                                    batch_size=32, steps=4000, bucket_size=512, seed=44)
    scheduled = Counter(rows[i]['mixture_source'] for i in order)
    assert scheduled == {'multi_image': 64000, 'video': 64000}, scheduled
    if args.require_unique_samples:
        assert len(set(order)) == len(order), 'Schedule repeats QA'
    assert all(len({rows[i]['mixture_source'] for i in order[s:s+32]}) == 1 for s in range(0, len(order), 32))
    config = json.loads(REFERENCE.read_text())
    previous = dict(config)
    config.update(data=str(manifest), image_root=str(source), pixel_area_cache=str(area_file),
                  output_dir=str(dest / 'checkpoints'), metrics_jsonl=str(dest / 'checkpoints/train_metrics.jsonl'),
                  max_steps=4000, mixture_ratios='multi_image:0.5,video:0.5',
                  init_checkpoint='', resume_weights_only=False, wandb_run_id=None,
                  wandb=True, wandb_mode='online', wandb_run_name=run_name,
                  experiment='embedding_kl_multi50_video50')
    assert config['micro_batch_size_per_gpu'] == 4 and config['gradient_accumulation_steps'] == 1
    assert config['required_world_size'] == 8 and config['visual_adapter_rank'] == 128
    assert config['output_mode'] == 'embedding_adapter' and config['native_prefix_memory'] == 'legacy'
    assert config['adapter_start_layer'] == config['active_adapter_layers'] == 0
    assert config['supervision_loss'] == 'distill' and config['kl_topk'] == 1024
    dump(dest / 'config.json', config)
    dump(dest / 'plan.json', dict(reference_run=config['reference_run'], reference_config=str(REFERENCE),
        reference_config_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        differences={k:dict(before=previous.get(k),after=v) for k,v in config.items() if previous.get(k)!=v},
        qa_counts=dict(counts), scheduled_sample_presentations=dict(scheduled),
        data_root=str(source), repeated_sample_presentations=len(order)-len(set(order)),
        steps=4000, global_batch=32, optimizer_steps_per_modality=2000,
        sampler='Modality-homogeneous global batches; shuffled multi/video pair per two optimizer steps. Exact sample ratio 50:50.',
        no_pixmo=True, initialization='new adapter, no checkpoint resume',
        trainable_parameters=23592960, frozen='entire Qwen backbone; only adapter down/up trained',
        deepstack='Unchanged reference: native teacher; legacy embedding-adapter student has no DeepStack injection.',
        video='8 full-window frames, original timestamps, max262144 pixels/frame',
        multi_image='all2–5images in source order, max1048576 pixels/image; existing training formatter unchanged',
        warmup_steps=120, save_every=500, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    print(f'RUN_DIR={dest}', flush=True)
    print(f'SAMPLING={dict(scheduled)}; global_batch=32; trainable=23592960', flush=True)
    if args.prepare_only:
        dump(dest / 'status.json', dict(state='prepared'))
        return
    if args.wait_for_pid:
        while Path(f'/proc/{args.wait_for_pid}').exists():
            dump(dest/'status.json', dict(state='waiting_for_existing_job', launcher_pid=os.getpid(),
                 waiting_for_pid=args.wait_for_pid, eval_after=args.eval_after))
            time.sleep(10)
    while True:
        gpu_memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert len(gpu_memory.splitlines()) == 8
        if all(int(x.strip()) < 1024 for x in gpu_memory.splitlines()):
            break
        if not args.wait_for_free_gpus:
            raise RuntimeError('GPUs not idle')
        dump(dest/'status.json', dict(state='waiting_for_free_gpus', launcher_pid=os.getpid(),
             gpu_memory_mib=[int(x) for x in gpu_memory.splitlines()], eval_after=args.eval_after))
        time.sleep(10)
    env = dict(os.environ, PYTHONPATH=str(WORK), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8', WANDB_MODE='online')
    env.pop('WANDB_RUN_ID',None)
    command = [sys.executable, '-m','torch.distributed.run','--standalone','--nproc_per_node','8',
               '-m','src.pixmo_objective_comparison','--config',str(dest/'config.json')]
    started = time.monotonic()
    with (dest / 'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                dump(dest/'status.json',dict(state='training',launcher_pid=os.getpid(),torchrun_pid=child.pid,
                     elapsed_seconds=time.monotonic()-started,config=str(dest/'config.json'),wandb_run_name=run_name))
                time.sleep(10)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    dump(dest/'status.json',dict(state='complete' if child.returncode==0 else 'failed',exit_code=child.returncode,
         elapsed_seconds=time.monotonic()-started,wandb_run_name=run_name))
    if child.returncode:
        raise RuntimeError(f'Training failed, exit={child.returncode}; see {dest / "train.log"}')
    if args.eval_after:
        command = [sys.executable, str(ROOT/'scripts/watch_multi50_video50_eval.py'), '--train-dir', str(dest)]
        with (dest/'post_train_eval.log').open('a') as log:
            evaluation = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = evaluation.wait()
            finally:
                if evaluation.poll() is None:
                    evaluation.terminate()
                    evaluation.wait()
        if code:
            raise RuntimeError(f'Post-training evaluation failed; see {dest / "post_train_eval.log"}')


if __name__ == '__main__':
    main()
