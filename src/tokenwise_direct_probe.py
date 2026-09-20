"""Seven frozen-Qwen supervised probes, with held-out image auditing and replacement."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

from src.visual_cross_token_ablation import ROOT, MODEL, decompose, flatten_heads, prepare as benchmark_inputs

EXPERIMENTS = [('cross',8),('cross',16),('cross',24),('cross',35),('hidden',30),('hidden',34),('hidden',35)]
BENCHMARKS = {'mmstar':('data/benchmarks/mmstar/mmstar_val.jsonl',1000),
              'realworldqa':('data/benchmarks/realworldqa/test.jsonl',765)}
TRAIN = ROOT/'data/train/pixmo/pixmo_ama_full_valid.clean.jsonl'


def image_signature(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    basis=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64)
    dct=basis@gray@basis.T
    bits=dct.reshape(-1)>np.median(dct.reshape(-1)[1:]);bits[0]=False
    phash=sum(int(bit)<<i for i,bit in enumerate(bits))
    return digest,phash


def prepare_data(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        print('Existing prepared plan:',root/'plan.json',flush=True);return
    rows=[json.loads(l) for l in TRAIN.open()]
    area=json.loads(Path(str(TRAIN)+'.pixel_areas.json').read_text())['areas']
    assert len(area)==len(rows)
    sized=sorted((a,i) for i,a in enumerate(area))
    buckets=[[i for _,i in sized[j:j+512]] for j in range(0,len(sized),512)]
    rng=torch.Generator().manual_seed(44);order=[]
    for bi in torch.randperm(len(buckets),generator=rng).tolist():
        b=buckets[bi];order.extend(b[j] for j in torch.randperm(len(b),generator=rng).tolist())
    eval_paths=[];selections={}
    for name,(rel,n) in BENCHMARKS.items():
        path=ROOT/rel;rr=[json.loads(l) for l in path.open()][:n]
        selections[name]={'samples':len(rr),'rows_sha256':hashlib.sha256(json.dumps(rr,sort_keys=True).encode()).hexdigest()}
        for r in rr:
            image=r.get('image');assert isinstance(image,str)
            pp=Path(image);eval_paths.append(str(pp if pp.is_absolute() else path.parent/pp))
    eval_paths=sorted(set(eval_paths));hashes={};excluded=[];selected=[]
    required=args.steps*32
    with ThreadPoolExecutor(max_workers=32) as pool:
        es=list(pool.map(image_signature,eval_paths))
        eval_sha={s[0] for s in es};eval_phash={s[1] for s in es}
        print(f'Hashed {len(eval_paths)} held-out images; selecting {required} PixMo presentations',flush=True)
        for start in range(0,len(order),512):
            chunk=order[start:start+512]
            unseen=sorted({rows[i]['image'] for i in chunk if rows[i]['image'] not in hashes})
            paths=[str(TRAIN.parent/x) for x in unseen]
            for name,sig in zip(unseen,pool.map(image_signature,paths)):
                match='exact_pixels' if sig[0] in eval_sha else None
                if match is None and any((sig[1]^p).bit_count()<=4 for p in eval_phash):match='phash_distance_le4'
                hashes[name]={'sha256_rgb':sig[0],'phash':str(sig[1]),'excluded':match}
                if match:excluded.append({'image':name,'reason':match})
            for i in chunk:
                if hashes[rows[i]['image']]['excluded']:continue
                selected.append({'source_row':i,'image':rows[i]['image'],'question':rows[i]['question']})
                if len(selected)==required:break
            if start%4096==0:print(f'selected={len(selected)}/{required} unique_hashed={len(hashes)} excluded={len(excluded)}',flush=True)
            if len(selected)==required:break
    assert len(selected)==required
    with (root/'train_rows.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r)+'\n')
    train_images={r['image'] for r in selected}
    assert not ({hashes[x]['sha256_rgb'] for x in train_images}&eval_sha)
    (root/'image_audit.json').write_text(json.dumps({'train_unique_images':len(train_images),'eval_unique_paths':len(eval_paths),
        'exact_rgb_hash_overlap':0,'near_duplicate_filter':'32px grayscale DCT pHash, Hamming <=4',
        'excluded':excluded,'train_images':{x:hashes[x] for x in sorted(train_images)},'benchmark_selections':selections},indent=2))
    plan={'model':MODEL,'experiments':[{'kind':k,'layer':l,'name':f'{k}_l{l:02d}'} for k,l in EXPERIMENTS],
          'steps':args.steps,'rank':128,'rank_by_kind':{'cross':128,'hidden':1024},'world_size':8,'batch_per_gpu':4,'global_batch':32,'grad_acc':1,
          'lr':5e-5,'betas':[.9,.95],'weight_decay':.01,'grad_clip':1.,'warmup_ratio':.03,'min_lr_ratio':.1,'seed':44,
          'train_rows_sha256':hashlib.sha256((root/'train_rows.jsonl').read_bytes()).hexdigest(),
          'benchmarks':selections,'cross_input':'target layer input hidden before input RMSNorm',
          'cross_target':'visual j!=i contribution, concatenated heads, BEFORE W_O; no prefix contribution',
          'hidden_input':'initial merged visual embedding at language layer 0 input, before DeepStack injections',
          'hidden_target':'input of target language layer, zero-based indexing',
          'replacement':'single target layer only; full native prefix retained in BOTH experiment families',
          'architecture':'cross: down(2560,128), SiLU, up(128,4096), no identity; hidden: E+up(SiLU(down(E)))',
          'loss':'per-image mean elementwise MSE, averaged over images; no CE/KL/OPD/cosine term',
          'teacher':'frozen native Qwen3-VL-4B, native DeepStack enabled',
          'distributed':'DDP on trainable probe only; frozen teacher replicated, BF16 compute / FP32 probe parameters',
          'unseen_scope':'excluded exact decoded-pixel matches and near pHash matches from this adapter training; not a claim about backbone pretraining',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    print(f'PREPARED {root}',flush=True)


class Probe(nn.Module):
    def __init__(self,kind,rank=128):
        super().__init__();self.kind=kind
        self.down=nn.Linear(2560,rank,bias=False)
        self.up=nn.Linear(rank,4096 if kind=='cross' else 2560,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        y=self.up(F.silu(self.down(x)))
        return y if self.kind=='cross' else x+y


class Captured(Exception):
    pass


class TargetPath:
    """Teacher capture stops at target; eval patch recomputes native prefix per run."""
    def __init__(self,model,kind,layer):
        self.model=model;self.kind=kind;self.layer=layer;self.mode='capture';self.probe=None
        self.positions=[];self.lengths=[];self.initial=[];self.inputs=[];self.targets=[]
        self.parts=[];self.calls=0;self.max_error=0.
        layers=model.model.language_model.layers
        self.handles=[layers[0].register_forward_pre_hook(self.capture_initial,with_kwargs=True),
                      layers[layer].register_forward_pre_hook(self.layer_input,with_kwargs=True)]
        if kind=='cross':
            self.handles += [layers[layer].self_attn.register_forward_pre_hook(self.attention_input,with_kwargs=True),
                             layers[layer].self_attn.o_proj.register_forward_pre_hook(self.attention_output)]
    def reset(self,inputs,mode='capture',probe=None):
        self.mode=mode;self.probe=probe;self.initial=[];self.inputs=[];self.targets=[];self.parts=[];self.calls=0;self.max_error=0.
        self.positions=[(r==self.model.config.image_token_id).nonzero().flatten() for r in inputs['input_ids']]
        self.lengths=[int(m.sum()) for m in inputs['attention_mask']]
        for p,m,n in zip(self.positions,inputs['attention_mask'],self.lengths):
            assert len(p)>0 and int(p[-1])<n and bool(m[:n].all()) and not bool(m[n:].any())
    def capture_initial(self,module,args,kw):
        h=kw.get('hidden_states',args[0] if args else None)
        if h.shape[1]==1:return
        self.initial=[h[b,p].detach().clone() for b,p in enumerate(self.positions)]
    def layer_input(self,module,args,kw):
        h=kw.get('hidden_states',args[0] if args else None)
        if h.shape[1]==1:return
        current=[h[b,p].detach().clone() for b,p in enumerate(self.positions)]
        self.inputs=current if self.kind=='cross' else self.initial
        if self.kind!='hidden':return
        self.targets=current;self.calls+=1
        if self.mode=='capture':raise Captured
        if self.mode=='native':return
        new=h.clone()
        for b,p in enumerate(self.positions):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.probe(self.initial[b])
            new[b,p]=pred.to(h.dtype)
        if args:return (new,)+args[1:],kw
        kw=dict(kw);kw['hidden_states']=new;return args,kw
    def attention_input(self,module,args,kw):
        h=kw.get('hidden_states',args[0] if args else None)
        if h.shape[1]==1:return
        if self.mode=='native':self.calls+=1;return
        shape=(*h.shape[:-1],-1,module.head_dim)
        q=module.q_norm(module.q_proj(h).view(shape)).transpose(1,2)
        k=module.k_norm(module.k_proj(h).view(shape)).transpose(1,2)
        v=module.v_proj(h).view(shape).transpose(1,2)
        from src.model import qwen_apply_rotary_pos_emb
        q,k=qwen_apply_rotary_pos_emb(q,k,*kw['position_embeddings'])
        groups=q.shape[1]//k.shape[1]
        self.parts=[];self.targets=[]
        for b,(p,n) in enumerate(zip(self.positions,self.lengths)):
            own,cross,prefix,_=decompose(q[b,:,:n],k[b,:,:n].repeat_interleave(groups,0),
                                        v[b,:,:n].repeat_interleave(groups,0),p,float(module.scaling))
            self.parts.append((flatten_heads(own),flatten_heads(cross),flatten_heads(prefix)))
            self.targets.append(self.parts[-1][1])
        self.calls+=1
        if self.mode=='capture':raise Captured
    def attention_output(self,module,args):
        h=args[0]
        if h.shape[1]==1 or self.mode=='native':return
        new=h.clone()
        for b,p in enumerate(self.positions):
            own,cross,prefix=self.parts[b]
            re=own+cross+prefix
            self.max_error=max(self.max_error,float((re-h[b,p].float()).norm()/h[b,p].float().norm().clamp_min(1e-12)))
            if self.mode=='reconstructed':pred=cross
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.probe(self.inputs[b])
            new[b,p]=(own+pred.float()+prefix).to(h.dtype)
        assert self.max_error<.025
        return (new,)+args[1:]
    def collect(self,inputs):
        self.reset(inputs)
        self.model.model.rope_deltas=None
        with torch.no_grad():
            try:self.model.model(**inputs,use_cache=False,return_dict=True)
            except Captured:pass
            else:raise RuntimeError('Teacher target hook did not stop the forward')
        assert self.calls==1 and len(self.targets)==len(self.positions)
        return self.inputs,self.targets


class Metrics:
    def __init__(self,dim):
        self.n=0;self.sse=0.;self.y2=0.;self.sy=torch.zeros(dim,dtype=torch.float64)
        self.cos=0.;self.valid_cos=0;self.images=0;self.image_mse=0.
    def add(self,pred,target):
        p=pred.double();y=target.double();error=float((p-y).square().sum())
        self.n+=len(y);self.sse+=error;self.y2+=float(y.square().sum());self.sy+=y.sum(0).cpu()
        denom=p.norm(dim=-1)*y.norm(dim=-1);valid=denom>1e-12
        self.cos+=float(((p*y).sum(-1)[valid]/denom[valid]).sum());self.valid_cos+=int(valid.sum())
        self.images+=1;self.image_mse+=error/y.numel()
    def state(self):
        return dict(n=self.n,sse=self.sse,y2=self.y2,sy=self.sy.tolist(),cos=self.cos,valid_cos=self.valid_cos,
                    images=self.images,image_mse=self.image_mse)
    @staticmethod
    def merged(states):
        n=sum(s['n'] for s in states);sy=np.sum([s['sy'] for s in states],axis=0)
        sse=sum(s['sse'] for s in states);y2=sum(s['y2'] for s in states)
        sst=y2-float(sy@sy)/n;images=sum(s['images'] for s in states)
        count=sum(s['valid_cos'] for s in states)
        return {'images':images,'visual_tokens':n,'mse':sse/(n*len(sy)),
                'image_mean_mse':sum(s['image_mse'] for s in states)/images,
                'r2':1-sse/sst if sst>1e-20 else None,
                'cosine':sum(s['cos'] for s in states)/count if count else None,
                'cosine_valid_tokens':count,'target_centered_sst':sst}


def train(args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.model import load_frozen_qwen3vl,prepare_qwen3vl_batch_inputs
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==8
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.manual_seed(44)
    dist.init_process_group('nccl')
    root=Path(args.output);exp=root/f'{args.kind}_l{args.layer:02d}';exp.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());steps=plan['steps']
    rows=[json.loads(l) for l in (root/'train_rows.jsonl').open()]
    assert len(rows)==steps*32
    processor,model=load_frozen_qwen3vl(MODEL,torch.bfloat16,torch.device('cuda',local),'flash_attention_2')
    torch.manual_seed(44)
    probe_rank=int(plan.get('rank_by_kind',{}).get(args.kind,plan['rank']))
    plan={**plan,'rank':probe_rank,'runtime_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    probe=Probe(args.kind,rank=probe_rank).to(local);ddp=DDP(probe,device_ids=[local])
    optimizer=torch.optim.AdamW(probe.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=.01)
    target=TargetPath(model,args.kind,args.layer)
    wandb_run=None
    if rank==0:
        import wandb
        (exp/'train_config.json').write_text(json.dumps(plan,indent=2))
        wandb_run=wandb.init(project='vision-kv-inject',name=f'qwen3vl4b_pixmo_direct_{args.kind}_l{args.layer:02d}_r{probe_rank}_{steps}step',
                              group='pixmo-tokenwise-direct-20260912',config={**plan,'kind':args.kind,'layer':args.layer,
                                'trainable_parameters':sum(p.numel() for p in probe.parameters())},dir=str(exp),mode='online')
        (exp/'wandb.json').write_text(json.dumps({'url':wandb_run.url,'id':wandb_run.id}))
    log=(exp/f'train_rank{rank}.jsonl').open('w');start=time.time()
    for step in range(steps):
        begin=step*32+rank*4;batch=rows[begin:begin+4]
        inputs,*_=prepare_qwen3vl_batch_inputs(processor,batch,TRAIN.parent,torch.device('cuda',local),include_answers=False)
        xs,ys=target.collect(inputs)
        optimizer.zero_grad(set_to_none=True)
        warm=max(1,round(steps*.03))
        scale=(step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm-1)))
        optimizer.param_groups[0]['lr']=5e-5*scale
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=ddp(torch.cat(xs,0))
        predictions=prediction.split([len(x) for x in xs])
        loss=sum(F.mse_loss(p.float(),y.float()) for p,y in zip(predictions,ys))/4
        assert bool(torch.isfinite(loss)),('Nonfinite loss',step)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        assert bool(torch.isfinite(grad))
        optimizer.step()
        if step==0:
            assert float(grad)>0 and all(p.grad is None for p in model.parameters())
        number=loss.detach().clone();dist.all_reduce(number);number/=world
        torch.cuda.synchronize();elapsed=time.time()-start
        entry={'step':step+1,'mse':float(number),'local_mse':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],
               'grad_norm':float(grad),'elapsed_seconds':elapsed,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
        log.write(json.dumps(entry)+'\n');log.flush()
        if rank==0 and ((step+1)%5==0 or step==0):
            wandb_run.log(entry,step=step+1);print(f'{args.kind}-{args.layer} step={step+1}/{steps} mse={float(number):.6g} elapsed={elapsed:.1f}s',flush=True)
        if (step+1)%500==0 or step+1==steps:
            if rank==0:
                tmp=exp/f'step{step+1}.pt.tmp'
                torch.save({'probe':probe.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                            'kind':args.kind,'layer':args.layer,'plan':plan},tmp)
                os.replace(tmp,exp/f'step{step+1}.pt')
            dist.barrier()
        del inputs,xs,ys,prediction,predictions,loss
    log.close()
    (exp/f'train_rank{rank}.done.json').write_text(json.dumps({'steps':steps,'seconds':time.time()-start}))
    if rank==0:wandb_run.finish()
    dist.barrier();dist.destroy_process_group()


def evaluate(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec,score_prediction
    from src.eval_benchmarks import generate_teacher_qwen
    torch.set_num_threads(4);torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.output);exp=root/f'{args.kind}_l{args.layer:02d}';outdir=exp/'eval';outdir.mkdir(exist_ok=True)
    plan=json.loads((root/'plan.json').read_text())
    processor,model=load_frozen_qwen3vl(MODEL,torch.bfloat16,torch.device('cuda:0'),'flash_attention_2')
    saved=torch.load(exp/f'step{plan["steps"]}.pt',map_location='cpu',weights_only=False)
    probe_rank=int(saved['probe']['down.weight'].shape[0])
    assert probe_rank==int(plan.get('rank_by_kind',{}).get(args.kind,plan['rank']))
    probe=Probe(args.kind,rank=probe_rank).cuda().eval()
    probe.load_state_dict(saved['probe'])
    target=TargetPath(model,args.kind,args.layer)
    original_features=model.model.get_image_features;features=[]
    def cached_features(*a,**kw):
        if not features:features.append(original_features(*a,**kw))
        return features[0]
    model.model.get_image_features=cached_features
    states={};start=time.time()
    with (outdir/f'shard{args.shard}.jsonl').open('w') as f,torch.no_grad():
        for name,(rel,n) in BENCHMARKS.items():
            path=ROOT/rel;ds=QwenBenchmarkDataset(str(path),processor,name,data_root=str(path.parent),max_samples=n)
            sha=hashlib.sha256(json.dumps(ds.rows,sort_keys=True).encode()).hexdigest()
            assert sha==plan['benchmarks'][name]['rows_sha256']
            stats=Metrics(4096 if args.kind=='cross' else 2560)
            for i in range(args.shard,len(ds),8):
                item=ds[i];inputs=benchmark_inputs(item,torch.device('cuda:0'));features.clear()
                xs,ys=target.collect(inputs)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=probe(xs[0])
                stats.add(pred,ys[0])
                local=Metrics(pred.shape[-1]);local.add(pred,ys[0])
                lm=Metrics.merged([local.state()])
                modes=['native','reconstructed','replacement'] if args.kind=='cross' else ['native','replacement']
                results={}
                for mode in modes:
                    target.reset(inputs,mode=mode,probe=probe)
                    _,text=generate_teacher_qwen(model,processor,**inputs,max_new_tokens=8)
                    assert target.calls==1
                    score=score_prediction(metric=get_benchmark_spec(name).metric,prediction_text=text,answer=item['answer'],
                                           choices=item.get('choices'),question=item['row'].get('question'))
                    results[mode]={'text':text,**score}
                if args.kind=='cross' and args.layer==35:
                    assert all(x['text']==results['native']['text'] for x in results.values()), 'Final-layer visual attention output affected answer'
                record={'benchmark':name,'sample_position':i,'sample_id':item['index'],'metrics':lm,'predictions':results,
                        'selection_sha256':sha,'reconstruction_error':target.max_error}
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                if i//8%10==0:print(f'EVAL {args.kind}-{args.layer} {name} shard={args.shard} sample={i} elapsed={time.time()-start:.1f}s',flush=True)
            states[name]=stats.state()
    (outdir/f'shard{args.shard}.metrics.json').write_text(json.dumps(states))
    (outdir/f'shard{args.shard}.done.json').write_text(json.dumps({'seconds':time.time()-start}))


def merge_eval(root,kind,layer):
    exp=root/f'{kind}_l{layer:02d}';folder=exp/'eval'
    rows={}
    for f in folder.glob('shard[0-9]*.jsonl'):
        for line in f.open():
            r=json.loads(line);k=(r['benchmark'],r['sample_position']);assert k not in rows;rows[k]=r
    assert set(rows)=={(b,i) for b,(_,n) in BENCHMARKS.items() for i in range(n)}
    states=[json.loads(f.read_text()) for f in folder.glob('*.metrics.json')];assert len(states)==8
    report={}
    for name,(_,n) in BENCHMARKS.items():
        rr=[rows[(name,i)] for i in range(n)];summary=Metrics.merged([s[name] for s in states])
        modes=rr[0]['predictions'].keys()
        acc={m:100*sum(r['predictions'][m]['score'] for r in rr)/n for m in modes}
        ref='reconstructed' if kind=='cross' else 'native'
        diff=np.array([r['predictions'][ref]['score']-r['predictions']['replacement']['score'] for r in rr])
        boot=np.random.default_rng(44).integers(0,n,(10000,n));ci=np.quantile(diff[boot].mean(-1)*100,[.025,.975])
        report[name]={'fit':summary,'accuracy_pct':acc,'replacement_drop_pp':float(diff.mean()*100),
                      'drop_ci95_pp':ci.tolist(),'harmed':int((diff>0).sum()),'helped':int((diff<0).sum())}
    (exp/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def update_report(root):
    doc=['# PixMo token-wise direct supervision','',
         '7 个独立实验，顺序训练；原生 Qwen3-VL-4B 冻结。rank 按 plan.json 的 rank_by_kind；batch4/GPU ×8，1000 step，MSE。', '',
         '| 目标 | 层（0-based） | 测试集 | R² | cosine | MSE | 原生 acc | 重构 acc | 替换 acc | 匹配下降 pp |',
         '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,layer in EXPERIMENTS:
        p=root/f'{kind}_l{layer:02d}'/'results.json'
        if not p.exists():continue
        for name,r in json.loads(p.read_text()).items():
            f=r['fit'];a=r['accuracy_pct'];rc=f"{a['reconstructed']:.2f}" if 'reconstructed' in a else '—'
            doc.append(f"| {kind} | {layer} | {name} | {f['r2']:.5f} | {f['cosine']:.5f} | {f['mse']:.6g} | {a['native']:.2f} | {rc} | {a['replacement']:.2f} | {r['replacement_drop_pp']:.2f} |")
    doc += ['', '## 定义与边界','',
            '- cross：输入目标层 RMSNorm 前的当前视觉 hidden；预测各 head 的视觉 j≠i contribution，拼接为 4096 维，在 W_O 前替换。self 和前置模板读取保留，原分母不变。',
            '- hidden：输入初始 2560 维 merged visual embedding；预测目标层输入 hidden。只替换该层视觉输入，原生前缀继续计算，不宣称跳过或加速。',
            '- Cross head 不加 x 的 identity：它预测的是 attention contribution，不是 residual hidden。hidden head 沿用 E + up(SiLU(down(E)))。无 bias，无额外 norm。',
            '- Teacher 保留 native DeepStack，输出与输入均冻结。仅 MSE 训练，无最终答案损失、KL、OPD 或 LoRA。',
            '- R² = 1−SSE/SST，SST 使用测试集每通道均值，汇总所有视觉 token/channel；MSE 为同一总体的元素平均，另存每图平均 MSE。cosine 逐 token 平均，零范数项单独计数。',
            '- 测试图像不参与 adapter 训练；按解码像素 SHA256 和 DCT pHash 距离≤4 筛除训练重叠/近重复。这不是关于 backbone 预训练数据未见过这些图像的声明。',
            '- cross-35 在最后一层 W_O 前只改视觉输出，结构上不能影响文本答案；其准确率不作为预测有效性的证据，但拟合指标仍有效。hidden-35 在最后一层 attention 之前改视觉 K/V 来源，能影响文本预测。',
            '- 每项 results.json 保存配对 bootstrap CI 和答对/答错互换数。状态见 status.json；未产生 results.json 的实验尚未完成。', '',
            '[配置](plan.json) · [图像划分审计](image_audit.json)']
    (root/'README.md').write_text('\n'.join(doc)+'\n')


def launch(args):
    root=Path(args.output);prepare_data(args)
    update_report(root)
    env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status={'state':'running','completed':[]}
    def save():
        status['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        (root/'status.json').write_text(json.dumps(status,indent=2))
    try:
        for kind,layer in EXPERIMENTS:
            name=f'{kind}_l{layer:02d}';folder=root/name;folder.mkdir(exist_ok=True)
            if (folder/'results.json').exists():status['completed'].append(name);continue
            status.update(experiment=name,phase='train');save()
            with (folder/'train.log').open('w') as log:
                cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','src.tokenwise_direct_probe','train',
                     '--output',str(root),'--kind',kind,'--layer',str(layer)]
                print('START TRAIN',name,flush=True)
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['phase']='eval';save();procs=[]
            try:
                for shard in range(8):
                    log=(folder/f'eval_gpu{shard}.log').open('w')
                    cmd=[sys.executable,'-u','-m','src.tokenwise_direct_probe','eval','--output',str(root),
                         '--kind',kind,'--layer',str(layer),'--shard',str(shard)]
                    p=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT)
                    procs.append((p,log))
                while any(p.poll() is None for p,_ in procs):
                    for p,_ in procs:
                        if p.poll() not in (None,0):raise RuntimeError(f'Eval worker failed {p.pid}')
                    time.sleep(5)
                assert all(p.returncode==0 for p,_ in procs)
            finally:
                for p,f in procs:
                    if p.poll() is None:p.terminate()
                    f.close()
            print('RESULT',name,json.dumps(merge_eval(root,kind,layer)),flush=True)
            status['completed'].append(name);update_report(root);save()
        status.update(state='complete',phase='complete');save()
    except BaseException as e:
        status.update(state='failed',error=repr(e));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['prepare','train','eval','launch'])
    p.add_argument('--output',default=str(ROOT/'artifacts/diagnostics/pixmo_tokenwise_direct_1000_20260912'))
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--kind',choices=['cross','hidden'],default='cross')
    p.add_argument('--layer',type=int,default=8)
    p.add_argument('--shard',type=int,default=0)
    args=p.parse_args()
    {'prepare':prepare_data,'train':train,'eval':evaluate,'launch':launch}[args.command](args)
