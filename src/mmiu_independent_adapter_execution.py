"""Implementation parity, not an intervention in the model being evaluated.

Reference uses native HF positions/causal attention and computes each adapter
MLP separately from native image embeddings. Candidate is the evaluation path.
Both implement the identical checkpoint and consume byte-identical inputs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/mmiu_random1000_seed42_all_methods_20260914'
OUT=ROOT/'artifacts/diagnostics/mmiu_independent_adapter_execution_20260914'
FP32=os.environ.get('MMIU_PARITY_FP32') == '1'
BF16_OUT=OUT
if FP32:
    OUT=ROOT/'artifacts/diagnostics/mmiu_independent_adapter_execution_fp32_20260914'


def selected_indices():
    if FP32:
        return sorted(json.loads((BF16_OUT/'summary.json').read_text())['differences'])
    return list(range(1000))


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import build_benchmark_prompt
    from src import model as ref
    from src.mmiu_binding_protocol_audit import MODEL,CHECKPOINT
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dtype=torch.float32 if FP32 else torch.bfloat16
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=dtype,
        device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
    from src.qwen_deepstack import disable_qwen_deepstack
    disable_qwen_deepstack(model)
    processor=AutoProcessor.from_pretrained(MODEL)
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),dtype)
    assert not meta['missing'] and not meta['unexpected']
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*args,**kwargs):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    ds=QwenBenchmarkDataset(str(SOURCE/'mmiu_random1000.jsonl'),processor,'mmiu',
        data_root=str(ROOT/'data/benchmarks/mmiu'),max_samples=1000,
        cache_dir=SOURCE/'processed/mmiu',prompt_layout='media_first_v1')
    old={r['index']:r for p in SOURCE.glob('embedding_adapter_shard*.jsonl')
        for line in p.open() if (r:=json.loads(line))}
    assert len(old)==1000 and all(r['generated_tokens']==1 for r in old.values())
    layers=model.model.language_model.layers
    with torch.inference_mode(),(OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in selected_indices()[shard::8]:
            row=ds.rows[index];item=ds[index]
            # Re-open original files in the published order, with no processed
            # cache and no dataset content-renderer call in the reference.
            images=[Image.open(ROOT/'data/benchmarks/mmiu'/p).convert('RGB') for p in row['images']]
            question=build_benchmark_prompt(row,ds.spec)
            assert '<|image_' not in question and row['source_question'].strip().casefold() in question.casefold()
            content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':question}]
            prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            independent=dict(processor(text=[prompt],images=images,return_tensors='pt',padding=True))
            for im in images:im.close()
            for key,value in independent.items():
                expected=item[key].unsqueeze(0) if key in ('input_ids','attention_mask','mm_token_type_ids') else item[key]
                assert torch.equal(value,expected),(index,key,'Cached/dataset input differs from independent input')
            inputs={k:v.cuda() for k,v in independent.items()}
            image_mask=inputs['input_ids'][0].eq(model.config.image_token_id)
            assert torch.equal(image_mask,inputs['mm_token_type_ids'][0].ne(0))
            starts=inputs['input_ids'][0].eq(model.config.vision_start_token_id).nonzero().flatten().tolist()
            ends=inputs['input_ids'][0].eq(model.config.vision_end_token_id).nonzero().flatten().tolist()
            counts=(inputs['image_grid_thw'].prod(-1)//model.model.visual.spatial_merge_size**2).tolist()
            assert len(starts)==len(ends)==len(counts)==len(row['images'])
            for start,end,n in zip(starts,ends,counts):
                assert end-start-1==n and image_mask[start+1:end].all()
            # Candidate: unchanged custom text-only execution.
            candidate_hidden={}
            def save_candidate(layer):
                def capture(m,args):
                    if layer not in candidate_hidden:
                        candidate_hidden[layer]=args[0].clone()
                return capture
            handles=[layer.input_layernorm.register_forward_pre_hook(save_candidate(l)) for l,layer in enumerate(layers)]
            try:
                model.model.rope_deltas=None
                candidate=ref.qwen_embedding_adapter_logits(model,adapter,inputs,logits_to_keep=1)[0][0,-1].float()
            finally:
                for handle in handles:handle.remove()
            candidate_pred=processor.tokenizer.decode([int(candidate.argmax())]).strip()
            if not FP32:
                assert candidate_pred==old[index]['text'].strip(),(index,'Candidate failed to reproduce')
            assert len(candidate_hidden)==36
            # Independent reference: anchor comes from the native decoder input,
            # never build_qwen_initial_context or the batched adapter helper.
            reference_state={};layer_errors=[]
            def substitute(layer):
                def replace(module,args,kwargs):
                    h=args[0] if args else kwargs['hidden_states']
                    if layer==0:reference_state['anchor']=h[:,image_mask].clone()
                    current=h[:,~image_mask].float();saved=candidate_hidden[layer].float()
                    assert current.shape==saved.shape
                    delta=current-saved
                    layer_errors.append(dict(layer=layer,max_abs=float(delta.abs().max()),
                        relative_l2=float(delta.norm()/current.norm().clamp_min(1e-12))))
                    anchor=reference_state['anchor']
                    down=F.linear(anchor,adapter.visual_adapter_down[layer].weight)
                    predicted=anchor+F.linear(F.silu(down),adapter.visual_adapter_up[layer].weight)
                    h=h.clone();h[:,image_mask]=predicted
                    return ((h,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=h))
                return replace
            handles=[layer.register_forward_pre_hook(substitute(l),with_kwargs=True) for l,layer in enumerate(layers)]
            try:
                model.model.rope_deltas=None
                reference=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            finally:
                for handle in handles:handle.remove()
            pred=processor.tokenizer.decode([int(reference.argmax())]).strip()
            out.write(json.dumps(dict(index=index,source_index=row['index'],task=row['task'],
                image_count=len(counts),visual_tokens=sum(counts),input_exact=True,image_boundaries_exact=True,
                candidate_prediction=candidate_pred,reference_prediction=pred,gold=row['answer'],
                candidate_correct=candidate_pred==row['answer'],reference_correct=pred==row['answer'],dtype=str(dtype),
                first_token_equal=int(candidate.argmax())==int(reference.argmax()),
                max_logit_difference=float((candidate-reference).abs().max()),
                output_kl=float((reference.softmax(-1)*(reference.log_softmax(-1)-candidate.log_softmax(-1))).sum()),
                layers=layer_errors,deepstack=False,same_checkpoint=True))+'\n')
            if FP32 or index%40==shard:print('DONE',index,flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.mmiu_independent_adapter_execution',str(shard)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==len(selected_indices())
    n=len(rows)
    summary=dict(n=n,fp32=FP32,tf32=False,candidate_accuracy=100*sum(r['candidate_correct'] for r in rows)/n,
        reference_accuracy=100*sum(r['reference_correct'] for r in rows)/n,
        differences=[r['index'] for r in rows if not r['first_token_equal']],
        inputs_and_boundaries_exact=all(r['input_exact'] and r['image_boundaries_exact'] for r in rows),
        mean_output_kl=sum(r['output_kl'] for r in rows)/n,
        max_output_kl=max(r['output_kl'] for r in rows),
        max_logit_difference=max(r['max_logit_difference'] for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
