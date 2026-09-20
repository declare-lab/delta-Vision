"""Train the rank-1024 recurrent KL adapter, then evaluate frozen document subsets."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/recurrent_kl/config.json'
TRAIN_COMMIT = '28bc51c'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    source=run/'training_source';source.mkdir()
    archive=subprocess.check_output(['git','archive',TRAIN_COMMIT,'src','configs'],cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(source,filter='data')
    old=json.loads(REFERENCE.read_text());config=dict(old)
    config.update(visual_adapter_rank=1024,
        output_dir=str(run/'checkpoints'),metrics_jsonl=str(run/'checkpoints/train_metrics.jsonl'),
        wandb_run_name=run.name,wandb_run_id=None,init_checkpoint='',resume_weights_only=False,
        deepspeed_config=str(source/'configs/ds_zero2.json'),experiment='recurrent_kl_rank1024')
    assert config['output_mode']=='recurrent_embedding_adapter' and config['max_steps']==2000
    assert config['micro_batch_size_per_gpu']==4 and config['gradient_accumulation_steps']==1
    assert config['required_world_size']==8 and config['supervision_loss']=='distill'
    assert sha(config['data'])=='0617827f2516706ba3a4b496a2c48ec60771b7c4308111082f27f2bfe1300207'
    rows=[json.loads(l) for l in Path(config['data']).read_text().splitlines() if l.strip()]
    missing=[r['image'] for r in rows if not (Path(config['image_root'])/r['image']).is_file()]
    assert not missing,missing[:5]
    areas=json.loads(Path(config['pixel_area_cache']).read_text())
    assert len(areas['areas'])==len(rows)
    dump(run/'config.json',config)
    # The deleted objective-comparison wrapper only selected the KL/SFT/OPD
    # objective. This run invokes its maintained pure-KL core directly.
    entry=source/'run_config.py'
    entry.write_text('''import argparse,json,os
from pathlib import Path
import torch
from src import train
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
args=argparse.Namespace(**json.loads(Path(a.config).read_text()))
assert args.output_mode=='recurrent_embedding_adapter' and args.visual_adapter_rank==1024
original=train.trainable_parameters_for_mode
def checked(adapter):
    parameters=original(adapter)
    assert adapter.mode=='recurrent_embedding_adapter'
    assert adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert len(adapter.visual_adapter_down)==len(adapter.visual_adapter_up)==36
    assert all(tuple(m.weight.shape)==(1024,2560) for m in adapter.visual_adapter_down)
    assert all(tuple(m.weight.shape)==(2560,1024) for m in adapter.visual_adapter_up)
    assert all(torch.count_nonzero(m.weight).item()==0 for m in adapter.visual_adapter_up)
    assert sum(p.numel() for p in parameters)==188743680
    if train.is_rank0():
        print('INITIALIZATION_VERIFIED: recurrent, all36, rank1024, zero up projections, trainable188743680',flush=True)
    return parameters
train.trainable_parameters_for_mode=checked
train.run_qwen(args)
''')
    plan=dict(reference_config=str(REFERENCE),reference_sha256=sha(REFERENCE),
        train_commit=subprocess.check_output(['git','rev-parse',TRAIN_COMMIT],cwd=ROOT,text=True).strip(),
        differences={k:{'before':old.get(k),'after':v} for k,v in config.items() if old.get(k)!=v},
        initialization='From scratch; per-layer down projection default initialization, up projection zero; no rank128 resume.',
        architecture='36 separate recurrent residual MLPs: h_l = h_(l-1) + up_l(SiLU(down_l(h_(l-1)))); 2560 -> 1024 -> 2560.',
        trainable_parameters=188743680,backbone='Entire Qwen3-VL-4B, vision encoder and LM frozen.',
        teacher_deepstack='Native teacher, unchanged from reference rank128 training. Student legacy adapter has no DeepStack injection.',
        eval_deepstack='off for all methods, same completed document benchmark protocol',
        training_data_sha256=sha(config['data']),training_samples=len(rows),global_batch=32,
        steps=2000,wandb='online',eval_after=['chartqa','docvqa','infographicvqa'],eval_samples_each=1000,
        sources={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')})
    dump(run/'plan.json',plan)
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    name='qwen3vl4b_pixmo_recurrent_kl_rank1024_2000_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run=args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/pixmo_recurrent_rank1024'/name
    config=prepare(run)
    print('RUN_DIR='+str(run),flush=True)
    dump(run/'status.json',dict(state='prepared',run_dir=str(run)))
    if args.prepare_only:return
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(memory.splitlines())==8 and all(int(v)<1024 for v in memory.splitlines()),memory
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_MODE='online',
        PYTHONPATH=str(run/'training_source'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_RUN_ID',None)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','8',
        str(run/'training_source/run_config.py'),'--config',str(run/'config.json')]
    dump(run/'command.json',dict(command=command,cwd=str(run/'training_source')))
    started=time.time()
    with (run/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=run/'training_source',env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            dump(run/'status.json',dict(state='training',launcher_pid=os.getpid(),torchrun_pid=child.pid,started=started,updated=time.time()))
            time.sleep(10)
    if child.returncode:
        dump(run/'status.json',dict(state='training_failed',exit_code=child.returncode,updated=time.time()))
        raise RuntimeError('Training failed; see '+str(run/'train.log'))
    checkpoint=run/'checkpoints/qwen_recurrent_embedding_adapter_step2000.pt'
    assert checkpoint.is_file()
    dump(run/'status.json',dict(state='evaluating',training_wall_seconds=time.time()-started,checkpoint=str(checkpoint)))
    env.pop('PYTHONPATH',None)
    with (run/'eval.log').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/eval_document_recurrent_rank1024.py'),
            '--run-dir',str(run)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    dump(run/'status.json',dict(state='complete' if result.returncode==0 else 'evaluation_failed',
        checkpoint=str(checkpoint),eval_exit_code=result.returncode,finished=time.time()))
    assert result.returncode==0


if __name__=='__main__':main()
