"""Replay identical prefixes at BF16/FP32 to localize cache-only discrepancies."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/multimodal_cache_precision_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    benchmark=('muirbench','mmiu','videomme')[shard]
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    path=ROOT/f'data/benchmarks/{benchmark}/test.jsonl'
    if benchmark=='mmiu':path=ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
    ds=QwenBenchmarkDataset(str(path),processor,benchmark,data_root=str(ROOT/f'data/benchmarks/{benchmark}'),
        max_samples=1000,prompt_layout='media_first_v1')
    inputs0=_qwen_inputs_from_item(ds[500],torch.device('cuda'))
    results=dict(benchmark=benchmark,index=500,tf32=False,shared_bf16_vision_anchor=True,modes={})
    with torch.inference_mode():
        anchor,pos0=ref.build_qwen_initial_context(model,inputs0)
        tokens=[]
        for precision in ('bf16','fp32'):
            if precision=='fp32':
                model.float()
                adapter.float()
            inputs={k:v.clone() for k,v in inputs0.items()}
            hidden=anchor.to(next(adapter.parameters()).dtype)
            pos=pos0.clone()
            first,_,cache=ref.qwen_embedding_adapter_prefill_cache(model,adapter,
                inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,pos)
            caches={name:copy.deepcopy(cache) for name in ('dynamic','hf_static','shape_exact')}
            del cache
            ref.qwen_embedding_adapter_attach_hf_static_cache(model,caches['hf_static'],max_new_tokens=8)
            methods=dict(dynamic=ref.qwen_embedding_adapter_decode_step,
                hf_static=ref.qwen_embedding_adapter_decode_step_hf_static,
                shape_exact=ref.qwen_embedding_adapter_decode_step_shape_exact)
            logits=first
            rows=[]
            for step in range(8):
                if precision=='bf16':tokens.append(int(logits[0,-1].argmax()))
                token=torch.tensor([[tokens[step]]],device='cuda',dtype=inputs['input_ids'].dtype)
                for key in ('input_ids','attention_mask','mm_token_type_ids'):
                    extra=token if key=='input_ids' else torch.ones_like(token) if key=='attention_mask' else torch.zeros_like(token)
                    inputs[key]=torch.cat([inputs[key],extra],1)
                hidden=torch.cat([hidden,model.model.get_input_embeddings()(token)],1)
                pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                native=ref.qwen_position_ids(model,inputs)
                assert torch.equal(pos,native)
                logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                    initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                a=logits[0,-1].double()
                current=dict(step=step,input_token=tokens[step],reference_argmax=int(a.argmax()),caches={})
                for name in caches:
                    b,caches[name]=methods[name](model,adapter,token,caches[name])
                    b=b[0,-1].double()
                    diff=(b-b.mean())-(a-a.mean())
                    current['caches'][name]=dict(same_argmax=bool(a.argmax()==b.argmax()),argmax=int(b.argmax()),
                        kl=float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum()),
                        max_abs=float((a-b).abs().max()),centered_relative_l2=float(diff.norm()/(a-a.mean()).norm()))
                rows.append(current)
            results['modes'][precision]=rows
            print('COMPLETE',benchmark,precision,flush=True)
            del caches,first,logits,b
        results['common_input_tokens']=tokens
    (OUTPUT/f'row_{shard}.json').write_text(json.dumps(results,indent=2)+'\n')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for shard in range(3):
        with (OUTPUT/f'worker{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-m','src.multimodal_cache_precision_audit',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
                stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    assert not any(codes),codes
    rows=[json.loads((OUTPUT/f'row_{i}.json').read_text()) for i in range(3)]
    summary={}
    for mode in ('bf16','fp32'):
        allchecks=[c for r in rows for s in r['modes'][mode] for c in s['caches'].values()]
        summary[mode]=dict(comparisons=len(allchecks),same_argmax=sum(c['same_argmax'] for c in allchecks),
            max_kl=max(c['kl'] for c in allchecks),max_centered_relative_l2=max(c['centered_relative_l2'] for c in allchecks))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
