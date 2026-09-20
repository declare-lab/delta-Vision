"""Read-only runtime audit of visual/text packing vs original interleaved K/V.

No masking ablation or checkpoint modification. The original attention output
is always returned. FP32 calculations independently verify storage permutation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_kv_interleave_audit_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
                                                          torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
                           data_root=str(ROOT/'data/benchmarks/muirbench'),prompt_layout='media_first_v1')
    index=next(i for i,r in enumerate(ds.rows) if len(r['images'])==shard+2)
    with torch.inference_mode():
        item=ds[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
        hidden,pos=ref.build_qwen_initial_context(model,inputs)
        pack=ref.prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],inputs['attention_mask'],
                                                       inputs['mm_token_type_ids'],hidden,pos)
        vp=pack['image_positions'][0];tp=pack['text_positions'][0];n=hidden.shape[1]
        columns=torch.cat([vp,tp]);assert torch.equal(columns.sort().values,torch.arange(n,device=vp.device))
        restored=ref.scatter_qwen_text_visual_hidden(text_hidden=pack['h'],visual_hidden=pack['visual_memory'],
            text_positions=pack['text_positions'],image_positions=pack['image_positions'],
            text_mask=pack['text_mask'],image_mask=pack['image_mask'],seq_len=n)
        assert torch.equal(restored,hidden),'Gather/scatter lost or moved tokens'
        counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
        cuts=[0]+(torch.where(vp[1:]-vp[:-1]!=1)[0]+1).tolist()+[len(vp)]
        assert [b-a for a,b in zip(cuts[:-1],cuts[1:])]==counts
        reverse={p:j for j,p in enumerate(tp.tolist())}
        query_indices=[];boundaries=[]
        ids=inputs['input_ids'][0].tolist()
        for g,(start,end) in enumerate(zip(cuts[:-1],cuts[1:]),1):
            vs=int(vp[start]);ve=int(vp[end-1]);after=ve+1
            assert ids[vs-1]==processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            assert ids[after]==processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            gap_end=int(vp[cuts[g]])-1 if g<len(counts) else n
            gap=processor.tokenizer.decode(ids[after:gap_end],skip_special_tokens=False)
            assert f'[End of Image {g}]' in gap
            assert vs-1 in reverse and after in reverse
            assert all(p in reverse for p in range(after,gap_end))
            query_indices += [reverse[vs-1],reverse[after],reverse[min(after+5,gap_end-1)]]
            boundaries.append(dict(image=g,visual_first=vs,visual_last=ve,
                                   vision_start_position=vs-1,vision_end_position=after,
                                   gap_text=gap[:140],visual_tokens=end-start))
        query_indices=sorted(set(query_indices+[len(tp)-1]))
        qi=torch.tensor(query_indices,device=vp.device);qpositions=tp[qi]
        original=ref._efficient_prefix_causal_attention_heads
        checks=[]
        def attention(q,vk,vv,tk,tv,*,scaling,attention_mask):
            actual=original(q,vk,vv,tk,tv,scaling=scaling,attention_mask=attention_mask)
            fullk=vk.new_empty((1,vk.shape[1],n,vk.shape[-1]));fullv=torch.empty_like(fullk)
            fullk.index_copy_(2,vp,vk);fullk.index_copy_(2,tp,tk)
            fullv.index_copy_(2,vp,vv);fullv.index_copy_(2,tp,tv)
            packedk=torch.cat([vk,tk],2);packedv=torch.cat([vv,tv],2)
            assert torch.equal(fullk[:,:,columns],packedk) and torch.equal(fullv[:,:,columns],packedv)
            fullmask=qpositions[:,None]>=torch.arange(n,device=vp.device)[None,:]
            packedmask=attention_mask[0,0,qi]
            assert torch.equal(fullmask[:,columns],packedmask),'Mask did not follow K/V permutation'
            groups=q.shape[1]//vk.shape[1]
            query=q[:,:,qi].float()
            def read(k,v,mask):
                k=k.float().repeat_interleave(groups,1);v=v.float().repeat_interleave(groups,1)
                scores=query@k.transpose(-1,-2)*scaling
                return scores.masked_fill(~mask[None,None],float('-inf')).softmax(-1)@v
            fp=read(packedk,packedv,packedmask);fi=read(fullk,fullv,fullmask)
            relative=float((fi-fp).norm()/fp.norm().clamp_min(1e-12))
            prod=actual[:,qi].transpose(1,2).float()
            production_relative=float((prod-fi).norm()/fi.norm().clamp_min(1e-12))
            assert relative<1e-5,(len(checks),relative)
            assert production_relative<.025,(len(checks),production_relative)
            checks.append(dict(layer=len(checks),kv_roundtrip_exact=True,mask_permutation_exact=True,
                               fp32_interleaved_vs_packed_rel_error=relative,
                               bf16_production_vs_fp32_interleaved_rel_error=production_relative))
            return actual
        ref._efficient_prefix_causal_attention_heads=attention
        try:
            logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,
                                                     position_ids=pos,logits_to_keep=1)[0]
        finally:ref._efficient_prefix_causal_attention_heads=original
        assert len(checks)==36
        control=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,
                                                  position_ids=pos,logits_to_keep=1)[0]
        assert torch.equal(logits,control),'Audit changed model output'
        result=dict(index=index,images=len(counts),sequence_tokens=n,visual_tokens=len(vp),
                    gather_scatter_exact=True,audit_logits_exact=True,boundaries=boundaries,
                    query_positions=qpositions.tolist(),layers=checks)
        (OUT/f'row_{shard}.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS',index,len(counts),'images',n,'tokens',flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not list(OUT.glob('row_*.json')), 'Use a fresh output path if audit results already exist'
    logs=[];jobs=[]
    for s in range(8):
        log=(OUT/f'worker{s}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_kv_interleave_audit',str(s)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
            stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads((OUT/f'row_{s}.json').read_text()) for s in range(8)]
    summary=dict(samples=8,image_counts=[r['images'] for r in rows],layers_checked=288,
        max_fp32_permutation_rel_error=max(x['fp32_interleaved_vs_packed_rel_error'] for r in rows for x in r['layers']),
        max_bf16_vs_fp32_rel_error=max(x['bf16_production_vs_fp32_interleaved_rel_error'] for r in rows for x in r['layers']),
        gather_scatter_exact=all(r['gather_scatter_exact'] for r in rows),
        no_output_change=all(r['audit_logits_exact'] for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
