"""Native-only causal ablation of direct cross-image visual attention.

All token embeddings/positions and text-query causal masks stay original.
The explicit-causal control checks changes due to the SDPA mask/backend alone.
No adapter, training, token removal, or formal benchmark protocol modification.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_cross_image_route_audit_20260914'
MODES={'explicit_causal':(0,0),'block_all':(0,36),'block_0_11':(0,12),
       'block_12_17':(12,18),'block_18_35':(18,36)}
PREFIX_AUDIT=os.environ.get('MUIR_CROSS_IMAGE_PREFIX_AUDIT')=='1'
if PREFIX_AUDIT:
    OUTPUT=ROOT/'artifacts/diagnostics/muir_cross_image_prefix_route_20260914'
    MODES={'explicit_causal':(0,0),'text_prefix_all':(0,36),'text_prefix_middle':(12,18),
           'all_previous_all':(0,36),'all_previous_middle':(12,18)}


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    model.eval().requires_grad_(False)
    lm=model.model.language_model
    lm.register_forward_pre_hook(lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    lm._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(ds.rows) if r['task']=='Image-Text Matching']
    previous={}
    for p in (ROOT/'artifacts/diagnostics/muir_native_context_factors_20260914').glob('rows_*.jsonl'):
        for line in p.open():
            r=json.loads(line);previous[r['index']]=r['conditions']['native']['prediction']
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            item=ds[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            hidden,pos=ref.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].ne(0)
            counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
            visual_groups=visual.nonzero().flatten().split(counts)
            length=hidden.shape[1]
            ids=torch.full((length,),-1,device='cuda',dtype=torch.long)
            for i,g in enumerate(visual_groups):ids[g]=i
            assert bool((ids.ge(0)==visual).all())
            causal=torch.ones((length,length),device='cuda',dtype=torch.bool).tril()
            cross=(ids[:,None].ge(0)&ids[None,:].ge(0)&ids[:,None].ne(ids[None,:]))
            restricted=causal&~cross
            assert torch.equal(restricted[~visual],causal[~visual])
            # Same-image visual and all text keys remain causal and unchanged.
            assert torch.equal(restricted[:,~visual],causal[:,~visual])
            # For a visual query, keys between image 1's start and this image's
            # start are preceding-image context. Shared system/user prefix stays.
            own_start=torch.full((length,),length,device='cuda',dtype=torch.long)
            for g in visual_groups:own_start[g]=g[0]
            key_position=torch.arange(length,device='cuda')[None,:]
            previous=(visual[:,None] & key_position.ge(visual_groups[0][0]) & key_position.lt(own_start[:,None]))
            previous_text=previous & (~visual)[None,:]
            no_previous=causal&~previous
            no_previous_text=causal&~previous_text
            assert torch.equal(no_previous[~visual],causal[~visual])
            assert torch.equal(no_previous_text[~visual],causal[~visual])
            result=dict(index=index,gold=item['answer'],conditions={})
            def forward():
                output=lm(inputs_embeds=hidden,position_ids=pos,attention_mask=inputs['attention_mask'],use_cache=False)
                return model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
            native=forward()
            def save(name,logits):
                text=processor.tokenizer.decode([int(logits.argmax())]).strip()
                result['conditions'][name]=dict(prediction=text,correct=text==item['answer'],
                    native_kl=float((native.softmax(-1)*(native.log_softmax(-1)-logits.log_softmax(-1))).sum()))
            save('native',native)
            assert result['conditions']['native']['prediction']==previous[index]
            for name,(start,end) in MODES.items():
                handles=[]
                def make_hook(mask):
                    def patch(module,args,kwargs):return args,dict(kwargs,attention_mask=mask[None,None])
                    return patch
                try:
                    for i,l in enumerate(lm.layers):
                        restriction=(no_previous_text if name.startswith('text_prefix_') else no_previous) if PREFIX_AUDIT else restricted
                        mask=restriction if start<=i<end else causal
                        handles.append(l.self_attn.register_forward_pre_hook(make_hook(mask),with_kwargs=True))
                    save(name,forward())
                finally:
                    for h in handles:h.remove()
            out.write(json.dumps(result)+'\n');print('DONE',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_cross_image_route_audit',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==84
    summary={name:dict(samples=84,correct=sum(r['conditions'][name]['correct'] for r in rows),
        accuracy=100*sum(r['conditions'][name]['correct'] for r in rows)/84,
        mean_native_kl=sum(r['conditions'][name]['native_kl'] for r in rows)/84) for name in ['native',*MODES]}
    summary['explicit_mask_changed_predictions']=sum(r['conditions']['native']['prediction']!=r['conditions']['explicit_causal']['prediction'] for r in rows)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
