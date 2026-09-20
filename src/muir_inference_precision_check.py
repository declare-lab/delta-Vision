"""FP32 parity check for BF16 inference disagreements, fixed weights/anchors."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=os.environ.get('INFERENCE_PARITY_BENCHMARK','muirbench')
PREFIX='muir' if BENCHMARK=='muirbench' else BENCHMARK
RANDOM1000=os.environ.get('INFERENCE_PARITY_RANDOM1000','0')=='1'
SUFFIX='_random1000' if RANDOM1000 else ''
PARENT=ROOT/f'artifacts/diagnostics/{PREFIX}_hf_adapter_inference_parity{SUFFIX}_20260914'
OUTPUT=ROOT/f'artifacts/diagnostics/{PREFIX}_inference_precision_check{SUFFIX}_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    rows=[json.loads(l) for p in PARENT.glob('rows_*.jsonl') for l in p.open()]
    selected=sorted([r for r in rows if not r['full_tokens_equal']],key=lambda x:x['index'])[shard::8]
    if not selected:
        return
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    manifest=ROOT/f'data/benchmarks/{BENCHMARK}/test.jsonl'
    if BENCHMARK=='mmiu':
        manifest=ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
    if RANDOM1000:
        assert BENCHMARK=='muirbench'
        manifest=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914/muirbench_random1000.jsonl'
    ds=QwenBenchmarkDataset(str(manifest),processor,BENCHMARK,max_samples=1000,prompt_layout='media_first_v1',
        data_root=str(ROOT/f'data/benchmarks/{BENCHMARK}'),
        **({'cache_dir':manifest.parent/'processed/muirbench'} if RANDOM1000 else {}))
    items=[]
    with torch.inference_mode():
        for row in selected:
            item=ds[row['index']];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            hidden,pos=ref.build_qwen_initial_context(model,inputs)
            # Preserve original BF16 vision-encoder outputs for both FP32 paths.
            items.append((row,inputs,hidden.float(),pos))
        model.float();adapter.float()
        for row,inputs,hidden,pos in items:
            a,b=row['candidate_tokens'],row['reference_tokens']
            divergence=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            if divergence:
                common=torch.tensor([a[:divergence]],device='cuda',dtype=inputs['input_ids'].dtype)
                inputs['input_ids']=torch.cat([inputs['input_ids'],common],1)
                inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(common)],1)
                hidden=torch.cat([hidden,model.model.get_input_embeddings()(common)],1)
                pos=torch.cat([pos,pos[:,:,-1:]+torch.arange(1,divergence+1,device='cuda').view(1,1,-1)],2)
            canonical=ref.qwen_position_ids(model,inputs,inputs_embeds=hidden)
            assert torch.equal(canonical,pos),(row['index'],'appended positions differ from HF')
            visual=inputs['mm_token_type_ids'][0].ne(0)
            memory=adapter.all_visual_memories_batched(hidden[:,visual])
            candidate=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0][0,-1].double()
            def make_hook(layer):
                def replace(module,args,kwargs):
                    h=(args[0] if args else kwargs['hidden_states']).clone()
                    h[:,visual]=memory[layer]
                    return ((h,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=h))
                return replace
            handles=[l.register_forward_pre_hook(make_hook(i),with_kwargs=True) for i,l in enumerate(model.model.language_model.layers)]
            try:
                reference=model(inputs_embeds=hidden,position_ids=pos,attention_mask=inputs['attention_mask'],
                    use_cache=False,logits_to_keep=1).logits[0,-1].double()
            finally:
                for h in handles:h.remove()
            centred_ref=reference-reference.mean();centred_cand=candidate-candidate.mean()
            result=dict(index=row['index'],first_divergence_token=divergence,
                same_argmax=bool(candidate.argmax()==reference.argmax()),
                candidate_token=processor.tokenizer.decode([int(candidate.argmax())]),
                reference_token=processor.tokenizer.decode([int(reference.argmax())]),
                kl=float((reference.softmax(-1)*(reference.log_softmax(-1)-candidate.log_softmax(-1))).sum()),
                centred_logit_relative_error=float((centred_ref-centred_cand).norm()/centred_ref.norm()),
                max_logit_abs_error=float((candidate-reference).abs().max()),positions_equal_HF=True)
            (OUTPUT/f'case_{row["index"]}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)


def run():
    assert json.loads((PARENT/'summary.json').read_text())['samples']==1000
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_inference_precision_check',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(p.read_text()) for p in OUTPUT.glob('case_*.json')]
    selected=[json.loads(line)['index'] for p in PARENT.glob('rows_*.jsonl') for line in p.open()
              if not json.loads(line)['full_tokens_equal']]
    assert len(rows)==len(selected) and {r['index'] for r in rows}==set(selected)
    summary=dict(samples=len(rows),matching_argmax=sum(r['same_argmax'] for r in rows),
        max_kl=max((r['kl'] for r in rows),default=0),max_centred_relative_logit_error=max((r['centred_logit_relative_error'] for r in rows),default=0),
        max_logit_abs_error=max((r['max_logit_abs_error'] for r in rows),default=0))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
