"""Five independent 2000-step static KL adapters; preserve the reference 8x4 batch."""
import argparse
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/config.json'
ORIGINAL=ROOT/'artifacts/experiments/qwen_postnorm_kl_2000_20260916/original_metadata.json'
TRAIN_COMMIT='28bc51c'
RANKS=[32,64,256,512,1024]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


ENTRY='''import argparse,json,os
from pathlib import Path
import torch
from src import train
from src.model_setup import disable_qwen_deepstack
p=argparse.ArgumentParser();p.add_argument('--config',required=True);cli=p.parse_args()
args=argparse.Namespace(**json.loads(Path(cli.config).read_text()))
assert args.output_mode=='embedding_adapter'
assert args.visual_adapter_rank in [32,64,256,512,1024]
assert args.supervision_loss=='distill' and args.kl_topk==1024 and args.temperature==2.
assert args.required_world_size==8 and args.micro_batch_size_per_gpu==4 and args.gradient_accumulation_steps==1
assert not args.init_checkpoint and args.max_steps==2000
loader=train.load_frozen_qwen3vl
def load_off(*a,**kw):
    processor,model=loader(*a,**kw)
    disable_qwen_deepstack(model)
    assert model.model.visual.deepstack_visual_indexes==[]
    assert not any(p.requires_grad for p in model.parameters())
    assert model.model.language_model.config.hidden_size==2560
    assert len(model.model.language_model.layers)==36
    if train.is_rank0():print('TEACHER_FROZEN_DEEPSTACK_OFF',flush=True)
    return processor,model
train.load_frozen_qwen3vl=load_off
original=train.trainable_parameters_for_mode
def checked(adapter):
    params=original(adapter);r=args.visual_adapter_rank
    assert adapter.mode=='embedding_adapter'
    assert adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.native_prefix_memory=='legacy' and not adapter.native_ffn_carriers
    assert len(adapter.visual_adapter_down)==len(adapter.visual_adapter_up)==36
    assert all(tuple(m.weight.shape)==(r,2560) for m in adapter.visual_adapter_down)
    assert all(tuple(m.weight.shape)==(2560,r) for m in adapter.visual_adapter_up)
    assert all(torch.count_nonzero(m.weight).item()==0 for m in adapter.visual_adapter_up)
    expected=36*2*2560*r
    assert sum(p.numel() for p in params)==expected
    assert all(name.startswith(('visual_adapter_down.','visual_adapter_up.')) for name,p in adapter.named_parameters() if p.requires_grad)
    if train.is_rank0():
        result=dict(rank=r,trainable_parameters=expected,all_36_layers=True,mode=adapter.mode,up_initialization='zero',checkpoint_resume=False,deepstack=False)
        (Path(args.output_dir).parent/'initialization_verified.json').write_text(json.dumps(result,indent=2))
        print('INITIALIZATION_VERIFIED',json.dumps(result),flush=True)
    return params
train.trainable_parameters_for_mode=checked
train.run_qwen(args)
'''


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    source=run/'training_source';source.mkdir()
    archive=subprocess.check_output(['git','archive',TRAIN_COMMIT,'src','configs'],cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:tar.extractall(source,filter='data')
    shutil.copy2(ROOT/'src/model_setup.py',source/'src/model_setup.py')
    (source/'run_config.py').write_text(ENTRY)
    reference=json.loads(REFERENCE.read_text())
    original=json.loads(ORIGINAL.read_text())['args']
    flags={original[i]:original[i+1] for i in range(len(original)-1) if original[i].startswith('--') and not original[i+1].startswith('--')}
    for key in ['max_steps','micro_batch_size_per_gpu','gradient_accumulation_steps','lr','warmup_ratio','min_lr_ratio','weight_decay','temperature','kl_topk','lambda_logit','seed','visual_adapter_rank','pixel_bucket_size','required_world_size','save_every','log_every']:
        assert float(reference[key])==float(flags['--'+key.replace('_','-')]),key
    for key in ['output_mode','supervision_loss','loss_normalization','batch_sampling','lr_scheduler','dtype','attn_implementation','distributed_engine']:
        assert reference[key]==flags['--'+key.replace('_','-')],key
    assert sha(reference['data'])=='0617827f2516706ba3a4b496a2c48ec60771b7c4308111082f27f2bfe1300207'
    rows=[json.loads(l) for l in Path(reference['data']).read_text().splitlines() if l.strip()]
    areas=json.loads(Path(reference['pixel_area_cache']).read_text())
    assert len(rows)==len(areas['areas'])==135995
    # The frozen full dataset was already used in the reference run; verify
    # paths across it before launching this queue without changing any records.
    missing=[r['image'] for r in rows if not (Path(reference['image_root'])/r['image']).is_file()]
    assert not missing,missing[:10]
    ranks={}
    for rank in RANKS:
        dest=run/f'rank{rank}';dest.mkdir()
        config=dict(reference)
        config.update(visual_adapter_rank=rank,teacher_deepstack=False,init_checkpoint='',resume_weights_only=False,
            output_dir=str(dest/'checkpoints'),metrics_jsonl=str(dest/'checkpoints/train_metrics.jsonl'),
            wandb=True,wandb_mode='online',wandb_run_id=None,wandb_run_name=f'{run.name}_rank{rank}',
            deepspeed_config=str(source/'configs/ds_zero2.json'),experiment=f'static_kl_rank{rank}')
        dump(dest/'config.json',config)
        dump(dest/'status.json',dict(state='queued',rank=rank))
        ranks[str(rank)]=dict(config=str(dest/'config.json'),trainable_parameters=36*2*2560*rank,
            differences={k:dict(before=reference.get(k),after=v) for k,v in config.items() if reference.get(k)!=v})
    dump(run/'plan.json',dict(reference_run=reference['reference_run'],reference_config=str(REFERENCE),reference_sha256=sha(REFERENCE),
        original_wandb_metadata=str(ORIGINAL),original_cli_hyperparameters_verified=True,
        source_commit=subprocess.check_output(['git','rev-parse',TRAIN_COMMIT],cwd=ROOT,text=True).strip(),
        source_note='Same isolated trainer revision as the successful recurrent/document retraining launchers; original historical dirty worktree is not claimed bitwise reproduced.',
        ranks=RANKS,runs=ranks,world_size=8,micro_batch=4,gradient_accumulation=1,global_batch=32,steps_each=2000,
        schedule='All 8 GPUs per run, sequential ascending ranks; stop queue on failure.',
        data_name='allenai/pixmo-ask-model-anything',data=str(reference['data']),data_sha256=sha(reference['data']),samples=len(rows),
        architecture='Independent pre-RMSNorm residual MLPs at all36 layers: E + Up_l(SiLU(Down_l(E))). No recurrent state.',
        loss='Original answer-token KL only; topk1024, temperature2, lambda1, token normalization. No MSE/hidden/attention losses.',
        deliberate_protocol_difference='Teacher/student DeepStack off project-wide. Original ds_zero2_coeff.json unavailable; verified existing ZeRO2 config with batch4/accum1/clip1 overrides.',
        initialization='From scratch for each rank; default down initialization, zero up; backbone frozen.',
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}))
    dump(run/'status.json',dict(state='prepared',ranks=RANKS,completed=[]))
    print('PREPARED',str(run),flush=True)


def latest_step(dest):
    path=dest/'checkpoints/train_metrics.jsonl'
    if not path.exists():return 0
    try:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-16384));lines=f.read().splitlines()
        for line in reversed(lines):
            try:return int(json.loads(line)['step'])
            except (ValueError,KeyError):pass
    except OSError:pass
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    name='qwen3vl4b_pixmo_static_kl_rank_sweep_2000_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run=args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/pixmo_static_rank_sweep'/name
    if not run.exists():prepare(run)
    if args.prepare_only:return
    plan=json.loads((run/'plan.json').read_text());source=run/'training_source'
    for rel,digest in plan['source_hashes'].items():assert sha(source/rel)==digest,rel
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(source),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_MODE='online',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_RUN_ID',None)
    completed=[];started=time.time()
    for rank in RANKS:
        dest=run/f'rank{rank}'
        if json.loads((dest/'status.json').read_text())['state']=='complete':completed.append(rank);continue
        if latest_step(dest)>0:raise RuntimeError('Interrupted training found; explicit optimizer-aware resume required, refusing to restart over it.')
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert len(memory.splitlines())==8 and all(int(x)<1024 for x in memory.splitlines()),memory
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(source/'run_config.py'),'--config',str(dest/'config.json')]
        dump(dest/'command.json',dict(command=cmd,cwd=str(source),gpus=list(range(8))))
        print('START_RANK',rank,flush=True)
        with (dest/'train.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    status=dict(state='training',rank=rank,step=latest_step(dest),max_steps=2000,torchrun_pid=child.pid,updated=time.time())
                    dump(dest/'status.json',status)
                    dump(run/'status.json',dict(status,launcher_pid=os.getpid(),completed=completed,remaining=[r for r in RANKS if r not in completed and r!=rank],elapsed_seconds=time.time()-started))
                    time.sleep(15)
            finally:
                if child.poll() is None:child.terminate();child.wait()
        checkpoint=dest/'checkpoints/qwen_embedding_adapter_step2000.pt'
        if child.returncode or not checkpoint.is_file() or latest_step(dest)!=2000:
            status=dict(state='failed',rank=rank,exit_code=child.returncode,step=latest_step(dest),completed=completed)
            dump(dest/'status.json',status);dump(run/'status.json',status);raise RuntimeError(f'Rank {rank} failed; see {dest}/train.log')
        # Check checkpoint metadata and all trainable projection shapes before
        # scheduling the next rank. No full model checkpoint is written.
        import torch
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['global_step']==2000 and ckpt['args']['visual_adapter_rank']==rank
        state=ckpt['state_dict']
        for layer in range(36):
            assert tuple(state[f'visual_adapter_down.{layer}.weight'].shape)==(rank,2560)
            assert tuple(state[f'visual_adapter_up.{layer}.weight'].shape)==(2560,rank)
        del ckpt,state
        completed.append(rank)
        dump(dest/'status.json',dict(state='complete',rank=rank,step=2000,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),completed_at=time.time()))
        print('COMPLETE_RANK',rank,flush=True)
    dump(run/'status.json',dict(state='complete',completed=completed,elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
