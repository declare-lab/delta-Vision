"""Full-dataset FP32 text/adapter inference with unchanged BF16 vision features."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_fullprecision_inference_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    # Leave the vision module and its rotary buffers completely untouched.
    model.model.language_model.float()
    model.lm_head.float()
    adapter.float()
    adapter.precompute_stacked_weights()
    assert next(model.model.visual.parameters()).dtype==torch.bfloat16
    assert next(model.model.language_model.parameters()).dtype==torch.float32
    assert adapter.adapter_attention_backend=='efficient'
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',
        max_samples=1000,prompt_layout='media_first_v1')
    old={r['index']:r for p in (ROOT/'artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914').glob('rows_*.jsonl')
         for line in p.open() if (r:=json.loads(line))}
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in range(shard,1000,8):
            item=ds[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            hidden,pos=ref.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].ne(0)
            # A cast-up BF16 visual anchor must have exactly BF16-representable values.
            assert torch.equal(hidden[:,visual],hidden[:,visual].bfloat16().float())
            memory=adapter.all_visual_memories_batched(hidden[:,visual])
            original=adapter.all_visual_memories_batched
            adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
            generated=[]
            try:
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                for step in range(128):
                    logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                        initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                    assert logits.dtype==torch.float32
                    token=int(logits[0,-1].argmax());generated.append(token)
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                    if token in eos or text in [chr(65+j) for j in range(len(item['choices']))]:break
                    new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                    inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                    inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                    hidden=torch.cat([hidden,model.model.get_input_embeddings()(new)],1)
                    pos=torch.cat([pos,pos[:,:,-1:]+1],2)
            finally:adapter.all_visual_memories_batched=original
            score=score_prediction(metric='muirbench',prediction_text=text,answer=item['answer'],choices=item['choices'])
            out.write(json.dumps(dict(index=index,task=item['row']['task'],text=text,tokens=generated,fp32=score,
                bf16=old[index]['candidate'],bf16_text=old[index]['candidate_text']))+'\n')
            if index%80==shard:print('PROGRESS',index,flush=True)
            del hidden,memory


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for shard in range(8):
        with (OUTPUT/f'worker{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_fullprecision_inference',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
                stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert not any(codes),codes
    rows=[json.loads(line) for p in OUTPUT.glob('rows_*.jsonl') for line in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==1000
    result=dict(samples=len(rows),accuracy={m:sum(r[m]['score'] for r in rows)/10 for m in ('bf16','fp32')},
        changed_predictions=sum(r['bf16']['prediction']!=r['fp32']['prediction'] for r in rows),
        per_task={t:{m:100*sum(r[m]['score'] for r in rows if r['task']==t)/sum(r['task']==t for r in rows)
                     for m in ('bf16','fp32')} for t in sorted({r['task'] for r in rows})})
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
