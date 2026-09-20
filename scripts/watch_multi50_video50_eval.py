"""Wait for the exact 4000-step run, then evaluate three frozen manifests on 8 GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent/'vision-kv-inject-attention-sink'
BENCHMARKS=('muirbench','videomme','mvbench')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-dir',type=Path,required=True)
    opts=parser.parse_args()
    train=opts.train_dir.resolve()
    cfg=read(train/'config.json')
    assert cfg['max_steps']==4000 and cfg['mixture_ratios']=='multi_image:0.5,video:0.5'
    root=train/'post_train_eval';root.mkdir(exist_ok=True)
    lock=(root/'.watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=train/'checkpoints/qwen_embedding_adapter_step4000.pt'
    muir=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
    video=ROOT/'artifacts/diagnostics/video_balanced_base_adapter_20260914'
    matched_video=ROOT/'artifacts/diagnostics/mmiu_video_all_methods_matched_20260914'
    video_plan=read(video/'plan.json')
    manifests={'muirbench':muir/'muirbench_random1000.jsonl',
               **{b:Path(video_plan['datasets'][b]['path']) for b in ('videomme','mvbench')}}
    expected_counts={'muirbench':1000,'videomme':999,'mvbench':950}
    old_roots={'muirbench':muir,'videomme':matched_video,'mvbench':matched_video}
    frozen_map={};specs={}
    for b,path in manifests.items():
        previous=read(old_roots[b]/'plan.json')
        digest=sha(path)
        assert digest==previous['dataset_sha256'][b]
        count=sum(bool(x.strip()) for x in path.open())
        assert count==expected_counts[b],(b,count)
        dest=root/f'{b}_selected.jsonl'
        if dest.exists():assert sha(dest)==digest
        else:dest.write_bytes(path.read_bytes())
        frozen_map[b]=str(dest)
        specs[b]=dict(samples=count,sha256=digest,original_manifest=str(path),
                      max_new_tokens=8 if b=='muirbench' else 128,prior_result_root=str(old_roots[b]))
    dump(root/'manifest_map.json',frozen_map)
    sources=[ROOT/'src/multimodal_baseline_suite.py',ROOT/'src/multimodal_eval_inputs.py',
             ROOT/'baselines/eval_baselines.py',ROOT/'baselines/multimodal_pruning_utils.py',
             *[WORK/'src'/f for f in ('model.py','data.py','benchmarks.py','benchmark_video_sampling.py')]]
    plan=dict(training=str(train),training_config_sha256=sha(train/'config.json'),checkpoint=str(checkpoint),
              required_step=4000,order=list(BENCHMARKS),benchmarks=specs,deepstack_enabled=False,
              prompt_layout='media_first_v1',video_frames=8,video_subtitles=False,
              video_sampling='full_timestamp_v1',gpus=list(range(8)),
              source_sha256={str(p):sha(p) for p in sources},
              note='No new sampling, no collage or numbered-panel changes. Wait for successful training exit; no earlier checkpoint evaluation.')
    if (root/'plan.json').exists():assert read(root/'plan.json')==plan,'Plan differs; refusing to mix results'
    else:dump(root/'plan.json',plan)
    state=read(root/'status.json') if (root/'status.json').exists() else dict(completed=[])
    if state.get('state')=='complete':
        print('Already complete',root,flush=True)
        return
    state.update(state='waiting_for_step4000',watcher_pid=os.getpid(),checkpoint=str(checkpoint))
    def update(**kw):
        state.update(kw,heartbeat_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        dump(root/'status.json',state)
    update()
    print('WATCHING',train,flush=True)
    print('QUEUED',[(b,specs[b]['samples']) for b in BENCHMARKS],flush=True)
    recovered_completion=False
    while True:
        training=read(train/'status.json')
        if training['state']=='failed':raise RuntimeError('Training failed; no checkpoint will be evaluated')
        if training['state']=='complete':break
        pid=training.get('torchrun_pid')
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:
                # The training parent may be finalizing its success status.
                time.sleep(10)
                training=read(train/'status.json')
                if training['state']!='complete':
                    # A lost launcher heartbeat must not hide a successfully
                    # finished training child. Require every rank and both final
                    # checkpoint files, then verify the actual weights below.
                    assert checkpoint.is_file() and (train/'checkpoints/qwen_embedding_adapter_final.pt').is_file(), 'Missing final checkpoints'
                    for rank in range(8):
                        records=(train/f'checkpoints/timing_rank{rank}.jsonl').read_text().splitlines()
                        assert records and json.loads(records[-1])['step']==4000, ('Incomplete rank',rank)
                    recovered_completion=True
                    update(state='verifying_completed_training',recovered_after_lost_launcher=True)
                break
        metrics=train/'checkpoints/train_metrics.jsonl'
        if metrics.exists():
            entries=metrics.read_text().splitlines()
            if entries:
                try:update(training_logged_step=json.loads(entries[-1])['step'])
                except json.JSONDecodeError:update()
        else:update()
        time.sleep(20)
    if not recovered_completion:assert training.get('exit_code')==0
    assert sha(train/'config.json')==plan['training_config_sha256']
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved.get('global_step',saved.get('step'))==4000
    assert saved['args']['output_mode']=='embedding_adapter'
    if recovered_completion:
        final=torch.load(train/'checkpoints/qwen_embedding_adapter_final.pt',map_location='cpu',weights_only=False)
        assert final.get('global_step',final.get('step'))==4000
        assert saved['state_dict'].keys()==final['state_dict'].keys()
        assert all(torch.equal(v,final['state_dict'][k]) and torch.isfinite(v).all() for k,v in saved['state_dict'].items())
        dump(root/'training_completion_verification.json',dict(step=4000,completed_ranks=8,
             final_and_step4000_weights_identical=True,finite_weights=True,
             process_exit_code='unavailable; original launcher heartbeat was lost',
             checkpoint_sha256=sha(checkpoint)))
        del final
    del saved
    checkpoint_sha=sha(checkpoint)
    for file,digest in plan['source_sha256'].items():assert sha(file)==digest,('Evaluation source changed',file)
    for b in BENCHMARKS:
        if b in state['completed']:continue
        while True:
            memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if len(memory.splitlines())==8 and all(int(x.strip())<1024 for x in memory.splitlines()):break
            update(state='waiting_for_free_gpus',benchmark=b)
            time.sleep(20)
        assert sha(checkpoint)==checkpoint_sha
        assert sha(frozen_map[b])==specs[b]['sha256']
        out=root/b
        command=[sys.executable,'-m','src.multimodal_baseline_suite','run','--output',str(out),
                 '--methods','embedding_adapter','--checkpoint',str(checkpoint),'--benchmarks',b,
                 '--limit','1000','--shards','8','--gpus','0','1','2','3','4','5','6','7',
                 '--prompt-layout','media_first_v1','--max-new-tokens',str(specs[b]['max_new_tokens']),
                 '--manifest-map',str(root/'manifest_map.json'),'--json-only']
        update(state='evaluating',benchmark=b,checkpoint_sha256=checkpoint_sha)
        with (root/f'{b}.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'))
            try:
                while child.poll() is None:
                    update(evaluation_pid=child.pid)
                    time.sleep(10)
            finally:
                if child.poll() is None:child.terminate();child.wait()
        if child.returncode:raise RuntimeError(f'{b} evaluation failed, see {root/f"{b}.log"}')
        old={}
        for f in old_roots[b].glob('embedding_adapter_shard*.jsonl'):
            for line in f.open():
                row=json.loads(line)
                if row['benchmark']==b and row['method']=='embedding_adapter':old[row['index']]=row
        current=[json.loads(line) for f in out.glob('embedding_adapter_shard*.jsonl') for line in f.open()]
        assert len(old)==len(current)==expected_counts[b]
        for row in current:
            assert row['input_sha256']==old[row['index']]['input_sha256'],('Inputs differ from prior evaluation',b,row['index'])
        result=read(out/'results.json')
        assert len(result)==1 and result[0]['n']==result[0]['expected']==expected_counts[b]
        state.setdefault('results',{})[b]=dict(accuracy=result[0]['accuracy'],samples=len(current),
            previous_adapter_accuracy=100*sum(r['score'] for r in old.values())/len(old),matched_input_hashes=True)
        state['completed'].append(b)
        update()
        print('COMPLETE',b,state['results'][b],flush=True)
    dump(root/'results.json',state['results'])
    update(state='complete')
    print('ALL COMPLETE',root/'results.json',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Keep an explicit failure marker; never silently switch data/checkpoint.
        if '--train-dir' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--train-dir')+1]).resolve()/'post_train_eval'
            if folder.exists():
                dump(folder/'failure.json',dict(error=str(error),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise
