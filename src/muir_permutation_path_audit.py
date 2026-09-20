"""Check image identity through projections and train/eval forward parity.

No weights, prompts, or formal benchmark results are changed. Eight fixed cases.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_permutation_path_audit_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from src.muir_binding_diagnostic import permute_row
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',max_samples=1000,prompt_layout='media_first_v1')
    index=[440,450,461,471,482,492,503,513][shard]
    original=ds.rows[index]
    results={};tensors={};prepared=[]
    def compare(a,b):
        a,b=a.float(),b.float()
        return dict(exact=bool(torch.equal(a,b)),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)))
    def distribution(a,b):
        a,b=a.float(),b.float()
        return dict(kl=float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum()),
            same_argmax=bool(a.argmax()==b.argmax()))
    with torch.inference_mode():
        for name in ['original','rotated','neighbor']:
            ds.rows[index]=permute_row(original,'rotate_media_fixed_choices') if name=='rotated' else original
            item=ds[index+1 if name=='neighbor' else index]
            inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            initial,pos=ref.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].ne(0)
            memory=adapter.all_visual_memories_batched(initial[:,visual])
            logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,position_ids=pos,logits_to_keep=1)[0][0,-1]
            if name!='rotated':prepared.append((inputs,initial,pos,logits))
            # ZeRO training uses separate layer MLP calls rather than stacked BMM.
            previous_uniform=adapter.uniform_visual_adapter_rank
            try:
                adapter.train();adapter.uniform_visual_adapter_rank=False
                train_path=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=initial,position_ids=pos,logits_to_keep=1)[0][0,-1]
            finally:
                adapter.eval();adapter.uniform_visual_adapter_rank=previous_uniform
            results[name]=dict(layerwise_train_vs_stacked_eval=distribution(logits,train_path))
            if name=='neighbor':continue
            counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
            chunks=list(initial[:,visual].split(counts,dim=1))
            if name=='rotated':chunks=chunks[-1:]+chunks[:-1]
            tensors[name]={'embedding':torch.cat(chunks,dim=1),'layers':{}}
            captured={};hooks=[]
            def hook(l):
                def save(module,args,kwargs):
                    h=args[0] if args else kwargs['hidden_states']
                    captured[l]=h[:,visual].clone()
                return save
            for l in range(12,18):
                hooks.append(model.model.language_model.layers[l].register_forward_pre_hook(hook(l),with_kwargs=True))
            try:model(**inputs,use_cache=False,logits_to_keep=1)
            finally:
                for h in hooks:h.remove()
            for l in range(12,18):
                layer=model.model.language_model.layers[l];attn=layer.self_attn
                tensors[name]['layers'][l]={}
                for method,hidden in [('native',captured[l]),('adapter',memory[l])]:
                    h=layer.input_layernorm(hidden)
                    k=attn.k_norm(attn.k_proj(h).view(1,h.shape[1],-1,attn.head_dim)).flatten(2)
                    v=attn.v_proj(h)
                    for kind,tensor in [('K',k),('V',v)]:
                        parts=list(tensor.split(counts,dim=1))
                        if name=='rotated':parts=parts[-1:]+parts[:-1]
                        tensors[name]['layers'][l][method+kind]=torch.cat(parts,dim=1)
                # Gathered RoPE agrees with full native positions; image identity
                # must be checked before RoPE, since moving an image changes phase.
                pack=ref.prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],
                    inputs['mm_token_type_ids'],initial,pos)
                full_rope=model.model.language_model.rotary_emb(initial,pos)
                assert all(torch.equal(a[:,visual],b) for a,b in zip(full_rope,pack['visual_position_embeddings']))
        ds.rows[index]=original
        results['permuted_embedding']=compare(tensors['original']['embedding'],tensors['rotated']['embedding'])
        results['permuted_KV_before_RoPE']={l:{k:compare(tensors['original']['layers'][l][k],tensors['rotated']['layers'][l][k])
            for k in ['nativeK','nativeV','adapterK','adapterV']} for l in range(12,18)}
        # Reproduce a padded training batch using the exact two independently
        # prepared sequences, then compare each last valid text logit to batch=1.
        length=max(x[0]['input_ids'].shape[1] for x in prepared)
        batch={k:torch.zeros((2,length),device='cuda',dtype=prepared[0][0][k].dtype)
               for k in ['input_ids','attention_mask','mm_token_type_ids']}
        h=torch.zeros((2,length,prepared[0][1].shape[-1]),device='cuda',dtype=prepared[0][1].dtype)
        p=torch.zeros((3,2,length),device='cuda',dtype=prepared[0][2].dtype)
        for i,(inputs,initial,pos,_) in enumerate(prepared):
            n=inputs['input_ids'].shape[1]
            for k in batch:batch[k][i,:n]=inputs[k][0]
            h[i,:n]=initial[0];p[:,i,:n]=pos[:,0]
        batched=ref.qwen_embedding_adapter_logits(model,adapter,batch,initial_hidden=h,position_ids=p,logits_to_keep=1)[0]
        results['padded_batch_vs_individual']=[distribution(row[3],batched[i,-1]) for i,row in enumerate(prepared)]
        results['index']=index
    (OUTPUT/f'row_{shard}.json').write_text(json.dumps(results,indent=2)+'\n')
    print('DONE',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_permutation_path_audit',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads((OUTPUT/f'row_{i}.json').read_text()) for i in range(8)]
    result=dict(samples=8,embedding_exact=all(r['permuted_embedding']['exact'] for r in rows),
        adapter_KV_exact=all(v[k]['exact'] for r in rows for v in r['permuted_KV_before_RoPE'].values() for k in ['adapterK','adapterV']),
        train_eval_max_kl=max(r[n]['layerwise_train_vs_stacked_eval']['kl'] for r in rows for n in ['original','rotated','neighbor']),
        train_eval_argmax_matches=sum(r[n]['layerwise_train_vs_stacked_eval']['same_argmax'] for r in rows for n in ['original','rotated','neighbor']),
        padded_batch_max_kl=max(x['kl'] for r in rows for x in r['padded_batch_vs_individual']),
        padded_batch_argmax_matches=sum(x['same_argmax'] for r in rows for x in r['padded_batch_vs_individual']))
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
