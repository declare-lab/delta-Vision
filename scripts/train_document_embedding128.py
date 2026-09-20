"""Continue PixMo static rank128 on official document QA, then evaluate."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_pixmo_recurrent_rank1024 import dump, sha

DATA = ROOT/'data/train/document_qa_20260919'
REFERENCE = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl'
INITIAL = REFERENCE/'checkpoints/qwen_embedding_adapter_step2000.pt'
COMMIT = '28bc51c'


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    source = run/'training_source'
    source.mkdir()
    archive = subprocess.check_output(['git', 'archive', COMMIT, 'src', 'configs'], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(source, filter='data')
    # Use the existing explicit no-DeepStack guard in the isolated training source.
    (source/'src/qwen_deepstack.py').write_bytes((ROOT/'src/qwen_deepstack.py').read_bytes())
    for name in ['qwen_adapter_fa2.py', 'document_training_fa2.py']:
        (source/'src'/name).write_bytes((ROOT/'src'/name).read_bytes())
    old = json.loads((REFERENCE/'config.json').read_text())
    config = dict(old)
    config.update(data=str(DATA/'train.jsonl'), image_root=str(DATA), pixel_area_cache=str(DATA/'pixel_areas.json'),
        output_dir=str(run/'checkpoints'), metrics_jsonl=str(run/'checkpoints/train_metrics.jsonl'),
        init_checkpoint=str(INITIAL), resume_weights_only=True, parent_global_step=2000,
        micro_batch_size_per_gpu=1, gradient_accumulation_steps=4,
        wandb_run_name=run.name, wandb_run_id=None, experiment='document_continued_embedding128',
        deepspeed_config=str(source/'configs/ds_zero2.json'), teacher_deepstack=False,
        training_prompt_suffix='Answer the question using a single word or phrase.')
    assert config['output_mode']=='embedding_adapter' and config['visual_adapter_rank']==128
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert sha(config['data']) == manifest['manifest_sha256']
    dump(run/'config.json', config)
    entry = source/'run_config.py'
    entry.write_text('''import argparse,json
from pathlib import Path
import torch
from src import train
from src import model as training_model
from src.qwen_deepstack import disable_qwen_deepstack
from src.document_training_fa2 import DocumentTrainingAttention
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
args=argparse.Namespace(**json.loads(Path(a.config).read_text()))
load=train.load_frozen_qwen3vl
def load_off(*a,**kw):
    processor,model=load(*a,**kw)
    disable_qwen_deepstack(model)
    assert not any(p.requires_grad for p in model.parameters())
    assert model.model.visual.deepstack_visual_indexes==[]
    print('BACKBONE_FROZEN_DEEPSTACK_OFF_FA2',flush=True)
    return processor,model
train.load_frozen_qwen3vl=load_off
original=train.trainable_parameters_for_mode
def checked(adapter):
    parameters=original(adapter)
    ckpt=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['global_step']==2000
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    state=adapter.state_dict()
    assert set(state)==set(ckpt['state_dict'])
    for name,tensor in state.items():
        assert torch.equal(tensor.detach().cpu(),ckpt['state_dict'][name].to(dtype=tensor.dtype)),name
    assert sum(p.numel() for p in parameters)==23592960
    print('INITIALIZATION_VERIFIED: all72 tensors equal PixMo step2000; static rank128 all36; trainable23592960',flush=True)
    return parameters
train.trainable_parameters_for_mode=checked
attention=DocumentTrainingAttention()
training_model._efficient_prefix_causal_attention_heads=attention
prepare=train.prepare_qwen3vl_batch_inputs
def document_inputs(processor,rows,*a,**kw):
    rows=[dict(r,question=r['question']+'\\n'+args.training_prompt_suffix) for r in rows]
    result=prepare(processor,rows,*a,**kw)
    attention.prepare(result[0])
    return result
train.prepare_qwen3vl_batch_inputs=document_inputs
train.run_qwen(args)
assert attention.calls==36*args.max_steps*args.gradient_accumulation_steps,attention.calls
print('FA2_TRAINING_CALLS_VERIFIED',attention.calls,flush=True)
''')
    dump(run/'plan.json', dict(initial_checkpoint=str(INITIAL), initial_sha256=sha(INITIAL), parent_steps=2000,
        additional_steps=2000, optimizer='fresh AdamW; weights continued; fresh cosine schedule',
        train_commit=subprocess.check_output(['git','rev-parse',COMMIT],cwd=ROOT,text=True).strip(),
        data=manifest, data_provenance=str(DATA/'manifest.json'), teacher='frozen native Qwen3-VL-4B, same image and question as student, DeepStack OFF',
        student='all36 static residual embedding adapter, rank128, no cross-token mixing inside adapter',
        trainable_parameters=23592960, loss='answer-token top1024 KL on gold answer prefixes, temperature2, original token normalization',
        attention='FA2', global_batch=32, micro_batch=1, gradient_accumulation=4, world_size=8,
        differences={k:dict(before=old.get(k),after=v) for k,v in config.items() if old.get(k)!=v},
        sources={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}))
    return config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    name='qwen3vl4b_document_continue_embedding128_2000_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run=args.run_dir.resolve() if args.run_dir else ROOT/'artifacts/experiments/document_continued_adapter'/name
    prepare(run)
    print('RUN_DIR='+str(run),flush=True)
    if args.prepare_only:return
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(memory.splitlines())==8 and all(int(v)<1024 for v in memory.splitlines()),memory
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_MODE='online',
        PYTHONPATH=str(run/'training_source'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_RUN_ID',None)
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','8',
        str(run/'training_source/run_config.py'),'--config',str(run/'config.json')]
    dump(run/'command.json',dict(command=cmd,cwd=str(run/'training_source')))
    started=time.time()
    with (run/'train.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=run/'training_source',env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            dump(run/'status.json',dict(state='training',launcher_pid=os.getpid(),torchrun_pid=child.pid,started=started,updated=time.time()))
            time.sleep(10)
    if child.returncode:
        dump(run/'status.json',dict(state='training_failed',exit_code=child.returncode))
        raise RuntimeError('Training failed; see '+str(run/'train.log'))
    checkpoint=run/'checkpoints/qwen_embedding_adapter_step2000.pt'
    assert checkpoint.is_file()
    dump(run/'status.json',dict(state='evaluating',training_wall_seconds=time.time()-started,checkpoint=str(checkpoint)))
    env.pop('PYTHONPATH',None)
    with (run/'eval.log').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/eval_document_continued_adapter.py'),'--run-dir',str(run)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    dump(run/'status.json',dict(state='complete' if result.returncode==0 else 'evaluation_failed',checkpoint=str(checkpoint),eval_exit_code=result.returncode,finished=time.time()))
    assert result.returncode==0


if __name__=='__main__':main()
