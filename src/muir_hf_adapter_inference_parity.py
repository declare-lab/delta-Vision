"""Same adapter, two equivalent inference implementations on all 1000 inputs.

Reference: unmodified HF generation/cache/positions, with exactly the adapter's
visual states scattered at native layer inputs. Not a native-state ablation.
Candidate: the current explicit text-only recompute evaluation implementation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import hashlib

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=os.environ.get('INFERENCE_PARITY_BENCHMARK','muirbench')
LAYOUT=os.environ.get('INFERENCE_PARITY_LAYOUT','media_first_v1')
REFERENCE_MODEL=os.environ.get('INFERENCE_REFERENCE_MODEL','adapter')
RANDOM1000=os.environ.get('INFERENCE_PARITY_RANDOM1000','0')=='1'
MAX_NEW_TOKENS=8 if RANDOM1000 else 128
assert REFERENCE_MODEL in ('adapter','base')
REFERENCE_LETTER_STOP=os.environ.get('INFERENCE_REFERENCE_LETTER_STOP','1')=='1'
LAYOUT_SUFFIX='' if LAYOUT=='media_first_v1' else f'_{LAYOUT}'
RUN_KIND='hf_adapter_inference_parity' if REFERENCE_MODEL=='adapter' else 'matched_base_adapter'
if not REFERENCE_LETTER_STOP:RUN_KIND+='_eos_only'
if RANDOM1000:
    assert BENCHMARK=='muirbench' and LAYOUT=='media_first_v1' and REFERENCE_MODEL=='adapter'
    RUN_KIND+='_random1000'
OUTPUT=ROOT/f'artifacts/diagnostics/{"muir" if BENCHMARK == "muirbench" else BENCHMARK}_{RUN_KIND}{LAYOUT_SUFFIX}_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    manifest=ROOT/f'data/benchmarks/{BENCHMARK}/test.jsonl'
    if BENCHMARK=='mmiu':
        manifest=ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
    if RANDOM1000:
        manifest=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914/muirbench_random1000.jsonl'
    ds=QwenBenchmarkDataset(str(manifest),processor,BENCHMARK,max_samples=1000,prompt_layout=LAYOUT,
        data_root=str(ROOT/f'data/benchmarks/{BENCHMARK}'),
        **({'cache_dir':manifest.parent/'processed/muirbench'} if RANDOM1000 else {}))
    old={}
    prior_paths=(manifest.parent.glob('embedding_adapter_shard*.jsonl') if RANDOM1000 else
                 (ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914').glob('rows_*.jsonl'))
    for path in prior_paths:
        for line in path.open():
            r=json.loads(line)
            if r['benchmark']==BENCHMARK:old[r['index']]=r
    assert len(old)==1000
    class StopOnLetter(StoppingCriteria):
        def __init__(self,length,choices):self.length=length;self.choices=choices
        def __call__(self,input_ids,scores,**kwargs):
            text=processor.tokenizer.decode(input_ids[0,self.length:],skip_special_tokens=True).strip()
            return torch.tensor([text in self.choices],device=input_ids.device,dtype=torch.bool)
    outpath=OUTPUT/f'rows_{shard}.jsonl'
    done={json.loads(l)['index'] for l in outpath.open()} if outpath.exists() else set()
    with torch.inference_mode(),outpath.open('a',buffering=1) as out:
        for index in range(shard,1000,8):
            if index in done:continue
            item=ds[index];inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
            input_sha=None
            if RANDOM1000:
                digest=hashlib.sha256()
                for name in ('input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
                    if torch.is_tensor(item.get(name)):
                        value=item[name].contiguous().cpu()
                        digest.update(str((name,tuple(value.shape),str(value.dtype))).encode())
                        digest.update(value.view(torch.uint8).numpy().tobytes())
                input_sha=digest.hexdigest()
                assert input_sha==old[index]['input_sha256'],('Input changed',index)
                assert item['row']['index']==old[index]['source_index']
            hidden,pos=ref.build_qwen_initial_context(model,inputs0)
            visual=inputs0['mm_token_type_ids'][0].ne(0)
            memory=adapter.all_visual_memories_batched(hidden[:,visual])
            choices=[chr(65+j) for j in range(len(item['choices']))]
            original=adapter.all_visual_memories_batched
            adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,**kw:memory,adapter)
            generated=[];inputs=dict(inputs0);h=hidden;p=pos;first=None
            try:
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                for step in range(MAX_NEW_TOKENS):
                    model.model.rope_deltas=None
                    logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=h,position_ids=p,logits_to_keep=1)[0]
                    if first is None:first=logits[0,-1].float()
                    token=int(logits[0,-1].argmax());generated.append(token)
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                    if token in eos or text in choices:break
                    new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                    inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                    inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                    h=torch.cat([h,model.model.get_input_embeddings()(new)],1)
                    p=torch.cat([p,p[:,:,-1:]+1],2)
            finally:adapter.all_visual_memories_batched=original
            if LAYOUT=='media_first_v1':
                assert text==old[index]['text'],(index,text,old[index]['text'])
            calls=[0]*36;handles=[];prefix_len=inputs0['input_ids'].shape[1]
            def make_hook(layer):
                def replace(module,args,kwargs):
                    current=args[0] if args else kwargs['hidden_states']
                    if current.shape[1]==1:return
                    assert current.shape[1]==prefix_len
                    calls[layer]+=1
                    current=current.clone();current[:,visual]=memory[layer]
                    return ((current,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=current))
                return replace
            try:
                if REFERENCE_MODEL=='adapter':
                    for l,layer in enumerate(model.model.language_model.layers):
                        handles.append(layer.register_forward_pre_hook(make_hook(l),with_kwargs=True))
                model.model.rope_deltas=None
                result=model.generate(**inputs0,use_cache=True,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    stopping_criteria=StoppingCriteriaList([StopOnLetter(prefix_len,choices)] if REFERENCE_LETTER_STOP else []),
                    return_dict_in_generate=True,output_scores=True)
                native_tokens=result.sequences[0,prefix_len:].tolist()
                native_text=processor.tokenizer.decode(native_tokens,skip_special_tokens=True).strip()
                reference_first=result.scores[0][0].float()
            finally:
                for handle in handles:handle.remove()
            assert calls==([1]*36 if REFERENCE_MODEL=='adapter' else [0]*36),calls
            def score(t):return score_prediction(metric=ds.spec.metric,prediction_text=t,answer=item['answer'],choices=item['choices'],
                answers=item.get('answers'),question=item['row']['question'])
            record=dict(benchmark=BENCHMARK,prompt_layout=LAYOUT,index=index,task=item['row'].get('task'),visual_tokens=int(visual.sum()),
                image_count=len(item['row'].get('images',[])),candidate_text=text,reference_text=native_text,
                candidate_tokens=generated,reference_tokens=native_tokens,candidate=score(text),reference=score(native_text),
                first_token_equal=generated[0]==native_tokens[0],full_tokens_equal=generated==native_tokens,
                prefill_kl=float((reference_first.softmax(-1)*(reference_first.log_softmax(-1)-first.log_softmax(-1))).sum()),
                deepstack=False,same_adapter=REFERENCE_MODEL=='adapter',
                candidate_model='embedding_adapter_kl',reference_model=REFERENCE_MODEL,max_new_tokens=MAX_NEW_TOKENS,
                source_index=item['row'].get('index'),manifest=str(manifest),
                input_sha256=input_sha,
                reference_letter_stop=REFERENCE_LETTER_STOP)
            out.write(json.dumps(record)+'\n')
            if not record['full_tokens_equal']:print('DIFFERENCE',index,repr(text),repr(native_text),flush=True)
            if index%80==shard:print('PROGRESS',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if RANDOM1000:
        from src.embedding_adapter_corrected_eval import CHECKPOINT
        manifest=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914/muirbench_random1000.jsonl'
        plan=dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                  manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),max_new_tokens=MAX_NEW_TOKENS,
                  no_deepstack=True,protocol='Original unchanged media_first_v1; same adapter memories in full HF native layers')
        plan_path=OUTPUT/'plan.json'
        if plan_path.exists():assert json.loads(plan_path.read_text())==plan
        else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    jobs=[];logs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('a');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_hf_adapter_inference_parity',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==1000
    summary=dict(samples=1000,prompt_layout=LAYOUT,deepstack=False,max_new_tokens=MAX_NEW_TOKENS,random1000=RANDOM1000,
        candidate_model='embedding_adapter_kl',reference_model=REFERENCE_MODEL,reference_letter_stop=REFERENCE_LETTER_STOP,
        candidate_accuracy=sum(r['candidate']['score'] for r in rows)/10,
        reference_accuracy=sum(r['reference']['score'] for r in rows)/10,
        first_token_differences=[r['index'] for r in rows if not r['first_token_equal']],
        generation_differences=[r['index'] for r in rows if not r['full_tokens_equal']],
        answer_differences=[r['index'] for r in rows if r['candidate']['prediction']!=r['reference']['prediction']],
        mean_prefill_kl=sum(r['prefill_kl'] for r in rows)/1000,max_prefill_kl=max(r['prefill_kl'] for r in rows))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
