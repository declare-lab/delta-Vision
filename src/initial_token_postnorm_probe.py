"""Independent residual prediction of native post-RMSNorm visual representations."""
import argparse,csv,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from src.initial_token_mlp_probe import Bank,PathHook,inputs_for,runtime,teacher
from src.initial_token_prediction_probe import atomic_save,vector_metrics
from src.visual_channel_native_cache import ROOT,dump_json,load_rows,digest
LAYERS=tuple(range(36))
KEYS=tuple(f'norm_{l}' for l in LAYERS)
OLD=ROOT/'artifacts/diagnostics/initial_token_mlp_qwen_20260916'
PROTOCOL='initial_E_independent_postnorm_residual_all36_v1'

class Capture(PathHook):
 def __init__(self,model):
  self.model=model;self.layers=model.model.language_model.layers;self.mode='off';self.handles=[]
  self.handles.append(self.layers[0].register_forward_pre_hook(self.initial_hook,with_kwargs=True))
  for l,block in enumerate(self.layers):
   self.handles.append(block.input_layernorm.register_forward_hook(self.norm_hook(l)))
 def initial_hook(self,module,args,kwargs):
  if self.mode!='off':self.initial=self.take(kwargs.get('hidden_states',args[0] if args else None)).detach().clone()
 def norm_hook(self,l):
  def hook(module,args,output):
   if self.mode=='off':return
   self.calls[l]=self.calls.get(l,0)+1
   target=self.take(output).detach()
   # Direct .forward bypasses hooks, avoiding recursive capture.
   base=module.forward(self.initial).detach()
   self.bases[f'norm_{l}']=base
   self.targets[f'norm_{l}']=target.float()-base.float()
  return hook
 def collect(self,inputs):
  self.begin(inputs,'capture',cache_native=False);self.bases={};self.model.model.rope_deltas=None
  with torch.no_grad():self.model.model(**inputs,use_cache=False,return_dict=True)
  assert self.calls=={l:1 for l in LAYERS}
  assert len(self.targets)==36 and torch.count_nonzero(self.targets['norm_0'])==0
  return self.initial,self.targets,self.sizes
 def close(self):
  self.mode='off'
  for h in self.handles:h.remove()

def prepare(root):
 root.mkdir(parents=True,exist_ok=True)
 if (root/'plan.json').exists():
  p=json.loads((root/'plan.json').read_text());assert p['protocol']==PROTOCOL and p['source_sha256']==digest(__file__)
  for m in p['manifests'].values():assert digest(root/m['file'])==m['sha256']
  return p
 old=json.loads((OLD/'plan.json').read_text());manifests=old['manifests']
 for m in manifests.values():
  assert digest(OLD/m['file'])==m['sha256'];shutil.copyfile(OLD/m['file'],root/m['file'])
 plan=dict(protocol=PROTOCOL,source_sha256=digest(__file__),model='Qwen3-VL-4B',attention='flash_attention_2',deepstack='off',steps=2000,world_size=8,batch_per_gpu=4,seed=44,lr=3e-4,
  manifests=manifests,image_split=old['image_split'],train_unique_images=old['train_unique_images'],
  layers=list(LAYERS),layer_numbering='zero-based language layer input RMSNorm output',
  parameterization='Zhat_l=RMSNorm_l(E)+MLP_l(E); E initial post-merger visual embedding, raw unstandardized input',
  mlp='36 independent 2560->2560->2560 SiLU heads with biases; tokenwise; no cross-layer or cross-token mixing',
  initialization='Xavier down, zero down bias, zero up weight/bias; physical residual exactly zero; no target mean offset',
  loss='plain postnorm representation MSE; equal per-image then per-layer mean, no extra channel standardization',
  optimizer='AdamW betas(.9,.95), wd.01, clip1, cosine3%warmup10%final LR',
  evaluation='final step2000 checkpoint; RQA765,MMStar1000,SQA1000; full postnorm representation cosine and MSE; equal image averaging',
  control='RMSNorm_l(E), residual=0',layer0='identity target, expected zero residual and zero gradient',
  teacher='frozen native current states, no predictions injected; labels prompt-only, no benchmark answers',
  validation='128 disjoint Pixmo images; diagnostic only, benchmark evaluated at final step2000')
 dump_json(root/'plan.json',plan);shutil.copyfile(__file__,root/'source_snapshot.py');return plan

def normalization(root,processor,model,rank,world):
 # Identity statistics adapt the existing MLP implementation without changing loss scale.
 return {'input':{'mean':torch.zeros(2560),'std':torch.ones(2560)},'targets':{k:{'mean':torch.zeros(2560),'std':torch.ones(2560)} for k in KEYS}}

def validation(bank, processor, model, hook, rows, rank, world):
    import torch.distributed as dist
    values = torch.zeros(len(KEYS) + 1, dtype=torch.float64, device=model.device)
    with torch.no_grad():
        for i in range(rank, len(rows), world):
            x, targets, sizes = hook.collect(inputs_for(processor, [rows[i]], model.device))
            for j, key in enumerate(KEYS):
                values[j] += bank.heads[key].loss(x, targets[key], sizes).double()
            values[-1] += 1
    dist.all_reduce(values)
    return dict(zip(KEYS, (values[:-1] / values[-1]).tolist()))


def train(root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    runtime()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256'], 'Source changed after the run was prepared'
    processor, model = teacher()
    rows, valrows = load_rows(root / 'train.jsonl'), load_rows(root / 'validation.jsonl')
    started = time.time()
    stats = normalization(root, processor, model, rank, world)
    hook = Capture(model)
    torch.manual_seed(plan['seed'])
    bank = Bank(stats, 'zero', keys=KEYS).cuda()
    ddp = DistributedDataParallel(bank, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=plan['lr'], betas=(.9, .95), weight_decay=.01, fused=True)
    first = 0
    best = {key: float('inf') for key in KEYS}
    best_steps = {key: 0 for key in KEYS}
    best_state = {}
    if (root / 'resume.pt').exists():
        saved = torch.load(root / 'resume.pt', weights_only=False, map_location='cpu', mmap=True)
        assert saved['plan'] == plan
        bank.load_state_dict(saved['bank']); optimizer.load_state_dict(saved['optimizer'])
        first, best, best_steps = saved['step'], saved['best'], saved['best_steps']
        if rank == 0:
            best_state = torch.load(root / 'best.pt', weights_only=False, map_location='cpu', mmap=True)['bank']
        del saved
    dump_json(root / f'runtime_rank{rank}.json', dict(source_sha256=digest(__file__),
              parameters=sum(p.numel() for p in bank.parameters()), world=world, rank=rank,
              processor_size=dict(processor.image_processor.size), start_step=first))
    with (root / f'train_rank{rank}.jsonl').open('a' if first else 'w', buffering=1) as log:
        for step in range(first, plan['steps']):
            tick = time.time()
            start = (step * 32 + rank * 4) % len(rows)
            batch = [rows[(start + j) % len(rows)] for j in range(4)]
            inp = inputs_for(processor, batch, model.device)
            x, targets, sizes = hook.collect(inp)
            warm = round(plan['steps'] * .03)
            scale = (step + 1) / warm if step < warm else .1 + .45 * (1 + math.cos(math.pi * (step-warm) / (plan['steps']-warm-1)))
            optimizer.param_groups[0]['lr'] = plan['lr'] * scale
            optimizer.zero_grad(set_to_none=True)
            loss, per = ddp(x, targets, sizes)
            assert bool(torch.isfinite(loss))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
            assert bool(torch.isfinite(grad))
            if step < 2:
                grads = {key: dict(down=float(head.down.weight.grad.norm()), up=float(head.up.weight.grad.norm()))
                         for key, head in bank.heads.items()}
                assert all(g['up'] > 0 and (step == 0 or g['down'] > 0) for k,g in grads.items() if k != 'norm_0')
                dump_json(root / f'initialization_step{step+1}_rank{rank}.json', grads)
                assert all(p.grad is None for p in model.parameters())
            optimizer.step()
            means = torch.stack([per[k] for k in KEYS]); dist.all_reduce(means); means /= world
            entry = dict(step=step+1, loss=float(means.mean()), by_target=dict(zip(KEYS, means.tolist())),
                         grad_norm=float(grad), lr=optimizer.param_groups[0]['lr'], visual_tokens=sizes,
                         seconds=time.time()-started, step_seconds=time.time()-tick,
                         peak_memory_bytes=torch.cuda.max_memory_allocated())
            if step == 0 or (step+1) % 500 == 0 or step+1 == plan['steps']:
                scores = validation(bank, processor, model, hook, valrows, rank, world)
                entry['validation'] = scores
                for key in KEYS:
                    if scores[key] < best[key]:
                        best[key], best_steps[key] = scores[key], step+1
                        if rank == 0:
                            best_state.update({f'heads.{key}.{k}': v.detach().cpu().clone()
                                               for k, v in bank.heads[key].state_dict().items()})
                if rank == 0:
                    atomic_save(dict(bank=best_state, stats=stats, best=best, best_steps=best_steps, plan=plan), root/'best.pt')
                    atomic_save(dict(bank=bank.state_dict(), optimizer=optimizer.state_dict(), stats=stats,
                                     step=step+1, best=best, best_steps=best_steps, plan=plan), root/'resume.pt')
                dist.barrier()
            log.write(json.dumps(entry, allow_nan=False)+'\n')
            if rank == 0 and (step == 0 or (step+1) % 10 == 0):
                print('TRAIN', json.dumps(entry), flush=True)
                dump_json(root / 'progress.json', dict(stage='training', **entry))
            del x, targets, loss, per, inp
    if rank == 0:
        atomic_save(dict(bank={k:v.detach().cpu() for k,v in bank.state_dict().items()},stats=stats,step=plan['steps'],plan=plan), root/'final.pt')
    dump_json(root / f'train_rank{rank}.done.json', dict(steps=plan['steps'], seconds=time.time()-started, best=best, best_steps=best_steps))
    hook.close()
    dist.barrier(); dist.destroy_process_group()


def evaluate(root,shard):
 from src.data import QwenBenchmarkDataset
 from src.visual_channel_rank_grid import _to_device_item
 runtime();processor,model=teacher();saved=torch.load(root/'final.pt',map_location='cpu',weights_only=False,mmap=True)
 bank=Bank(saved['stats'],'zero',keys=KEYS).cuda().eval();bank.load_state_dict(saved['bank']);del saved
 hook=Capture(model)
 with torch.no_grad(),(root/f'eval_{shard}.jsonl').open('w',buffering=1) as log:
  for benchmark in ('realworldqa','mmstar','sqa'):
   ds=QwenBenchmarkDataset(str(root/f'{benchmark}_eval.jsonl'),processor,benchmark)
   for i in range(shard,len(ds),8):
    item=ds[i];inp=_to_device_item(item,model.device);inp={k:v for k,v in inp.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')}
    x,targets,sizes=hook.collect(inp);metrics={}
    for key,head in bank.heads.items():
     base=hook.bases[key].float();target=base+targets[key];pred=base+head(x)
     metrics[key]={'mlp':vector_metrics(pred,target,base),'identity':vector_metrics(base,target,base)}
    log.write(json.dumps(dict(benchmark=benchmark,sample=i,sample_id=item['index'],visual_tokens=sizes[0],metrics=metrics),allow_nan=False)+'\n')
    if i//8%20==0:print('EVAL',shard,benchmark,i,flush=True)
 hook.close();dump_json(root/f'eval_{shard}.done.json',{'complete':True,'step':2000})

def report(root):
 plan=json.loads((root/'plan.json').read_text());rows=[]
 for shard in range(8):
  assert (root/f'eval_{shard}.done.json').exists();rows+=load_rows(root/f'eval_{shard}.jsonl')
 table=[]
 for b in ('realworldqa','mmstar','sqa'):
  rr=[r for r in rows if r['benchmark']==b];assert len(rr)==plan['manifests'][b]['samples'] and {r['sample'] for r in rr}==set(range(len(rr)))
  for l in LAYERS:
   row=dict(dataset=b,samples=len(rr),layer=l)
   for mode in ('mlp','identity'):
    for metric in ('cosine','mse'):
     row[f'{mode}_{metric}']=sum(r['metrics'][f'norm_{l}'][mode][metric] for r in rr)/len(rr)
   table.append(row)
 dump_json(root/'results.json',table)
 with (root/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 lines=['# All36 independent postnorm residual predictions','','Final step2000; Qwen3-VL-4B; FA2; DeepStack off; no intervened native forward.','','| Dataset | Layer | MLP cosine | MLP MSE | Identity cosine | Identity MSE |','|---|---:|---:|---:|---:|---:|']
 for r in table:lines.append(f"| {r['dataset']} | {r['layer']} | {r['mlp_cosine']:.6f} | {r['mlp_mse']:.6f} | {r['identity_cosine']:.6f} | {r['identity_mse']:.6f} |")
 (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def launch(root):
    prepare(root)
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_MODE='disabled')
    state = dict(protocol=PROTOCOL, state='running', stage='normalization_and_training', started=time.time(), pid=os.getpid())
    dump_json(root/'status.json', state)
    try:
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
             '-m','src.initial_token_postnorm_probe','train','--output',str(root)]
        with (root/'train.log').open('a') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['stage']='evaluation'; dump_json(root/'status.json',state)
        procs=[]
        try:
            for shard in range(8):
                log=(root/f'eval_{shard}.log').open('w')
                p=subprocess.Popen([sys.executable,'-u','-m','src.initial_token_postnorm_probe','eval','--output',str(root),'--shard',str(shard)],
                                   cwd=ROOT, env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
                procs.append((p,log))
            while any(p.poll() is None for p,_ in procs):
                if any(p.poll() not in (None,0) for p,_ in procs):
                    raise RuntimeError('An evaluation shard failed')
                time.sleep(5)
        finally:
            for p,log in procs:
                if p.poll() is None:p.terminate()
                log.close()
        report(root)
        state.update(state='complete',stage='complete',finished=time.time())
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),finished=time.time())
        raise
    finally:
        dump_json(root/'status.json',state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','eval','report','launch'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.action=='eval':evaluate(args.output,args.shard)
    else:globals()[args.action](args.output)
