"""Paired RealWorldQA regression for project HF and vLLM inference paths."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch
from PIL import Image

from src.benchmarks import build_benchmark_prompt, get_benchmark_spec, score_prediction


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def disable_vllm_deepstack(model):
    # Load the native weights first, then disable both feature production and
    # injection. The native vLLM base kernels otherwise remain unchanged.
    model.config.vision_config.deepstack_visual_indexes=[]
    model.visual.deepstack_visual_indexes=[]
    model.visual.out_hidden_size=model.config.vision_config.out_hidden_size
    model.use_deepstack=False
    model.deepstack_num_level=0
    model.multiscale_dim=0
    return dict(deepstack=False,architecture=type(model).__name__)


def worker(args):
    root=Path(args.run).resolve()
    cfg=json.loads((root/'config.json').read_text())
    spec=get_benchmark_spec('realworldqa')
    rows=read(root/'data.jsonl')
    family=cfg['models'][args.family]
    export=Path(family['export'])
    exported=json.loads((export/'config.json').read_text())['delta_vision_adapter']
    dest=root/f'{args.family}.{args.backend}.{args.method}.jsonl'
    done=read(dest) if dest.exists() else []
    assert [x['index'] for x in done]==list(range(len(done)))
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    device=torch.device('cuda:0')
    controller=None
    if args.backend=='hf':
        if args.family=='qwen35':
            from src.qwen35 import load_model
            processor,model,adapter,controller=load_model(dict(
                model_path=exported['base_model'],rank=exported['rank']),device)
            if args.method=='adapter':
                saved=torch.load(exported['source_checkpoint'],map_location='cpu',weights_only=False)
                adapter.load_state_dict(saved['state_dict'],strict=True)
            adapter.eval()
        else:
            from src.model_setup import load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint
            processor,model=load_frozen_qwen3vl(exported['base_model'],torch.bfloat16,device,'flash_attention_2')
            model._adapter_attention_implementation='flash_attention_2'
            if args.method=='adapter':
                adapter,meta=load_qwen_embedding_adapter_checkpoint(exported['source_checkpoint'],
                    model.model.language_model,device,torch.bfloat16)
                assert not meta['missing'] and not meta['unexpected']
        if args.family=='qwen':
            from src.evaluate import _eos_token_ids
            eos=sorted(_eos_token_ids(processor.tokenizer))
        else:
            eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else list(eos)
    else:
        from transformers import AutoProcessor, GenerationConfig, AutoConfig
        from vllm import LLM, SamplingParams
        model_path=str(export) if args.method=='adapter' else exported['base_model']
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        if args.family=='qwen':
            from src.evaluate import _eos_token_ids
            eos=sorted(_eos_token_ids(processor.tokenizer))
        else:
            gc=GenerationConfig.from_model_config(AutoConfig.from_pretrained(model_path,local_files_only=True))
            eos=gc.eos_token_id
            eos=[eos] if isinstance(eos,int) else list(eos)
        llm=LLM(model=model_path,dtype='bfloat16',tensor_parallel_size=1,
            enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=False,
            async_scheduling=False,max_model_len=16384,max_num_batched_tokens=16384,
            max_num_seqs=1,gpu_memory_utilization=.35,
            kv_cache_memory_bytes=(4 if args.family=='qwen' else 2)*1024**3,
            skip_mm_profiling=True,mamba_ssm_cache_dtype='float32',
            limit_mm_per_prompt={'image':1,'video':0},seed=cfg['seed'],generation_config='vllm',
            attention_config={'backend':'FLASH_ATTN','flash_attn_version':2},
            gdn_prefill_backend='flashinfer')
        state=llm.apply_model(disable_vllm_deepstack)
        print('VLLM_MODEL',state,flush=True)
        sampling=SamplingParams(temperature=0,max_tokens=cfg['max_new_tokens'],
            stop_token_ids=eos,ignore_eos=True,skip_special_tokens=True,repetition_penalty=1.0)
    print('PROTOCOL',args.family,args.backend,args.method,'EOS',eos,'CAP',cfg['max_new_tokens'],flush=True)
    started=time.monotonic()
    with torch.inference_mode(),dest.open('a',buffering=1) as handle:
        for index in range(len(done),min(len(rows),args.limit or len(rows))):
            row=rows[index]
            question=build_benchmark_prompt(row,spec)
            content=[dict(type='image'),dict(type='text',text=question)]
            chat_kwargs=dict(tokenize=False,add_generation_prompt=True)
            if args.family=='qwen35':
                chat_kwargs['enable_thinking']=False
            prompt=processor.apply_chat_template([dict(role='user',content=content)],**chat_kwargs)
            with Image.open(Path(cfg['image_root'])/row['image']) as im:
                image=im.convert('RGB').copy()
            if args.backend=='vllm':
                result=llm.generate([dict(prompt=prompt,multi_modal_data={'image':image})],
                                    sampling,use_tqdm=False)[0]
                tokens=list(result.outputs[0].token_ids)
                prompt_ids=result.prompt_token_ids
            else:
                inputs=processor(text=[prompt],images=[image],return_tensors='pt').to(device)
                prompt_ids=inputs['input_ids'][0].tolist()
                if args.family=='qwen' and args.method=='adapter':
                    from src.evaluate import generate_adapter_qwen_decode_cache
                    metrics={}
                    generate_adapter_qwen_decode_cache(model,processor,adapter,inputs,
                        cfg['max_new_tokens'],decode_cache_mode='fast',
                        early_stop_metric=None,decode_step_metrics=metrics)
                    tokens=metrics['generated_token_ids'][0]
                else:
                    if hasattr(model.model,'rope_deltas'):
                        model.model.rope_deltas=None
                    ctx=controller.activate('adapter' if args.method=='adapter' else 'native',
                         inputs['mm_token_type_ids'].eq(1)) if controller else nullcontext()
                    with ctx:
                        generated=model.generate(**inputs,do_sample=False,
                            max_new_tokens=cfg['max_new_tokens'],use_cache=True,eos_token_id=eos,
                            pad_token_id=processor.tokenizer.pad_token_id,logits_to_keep=1)
                    tokens=generated[0,len(prompt_ids):].tolist()
            image.close()
            # Exactly the same current project scorer for every backend/model.
            text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
            scored=score_prediction(metric=spec.metric,prediction_text=text,
                answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),
                question=row.get('question'))
            finished=bool(tokens and tokens[-1] in eos)
            item=dict(index=index,source_index=row['source_index'],id=row.get('id'),
                prediction_text=text,generated_token_ids=tokens,**scored,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_ids_sha256=ids_hash(prompt_ids),prompt_tokens=len(prompt_ids),
                stopped_by_eos=finished,hit_generation_limit=not finished and len(tokens)>=cfg['max_new_tokens'])
            handle.write(json.dumps(item,ensure_ascii=False)+'\n')
            done.append(item)
            if len(done)%25==0 or len(done)==len(rows):
                print('PROGRESS',len(done),'/',len(rows),'acc',round(100*sum(x['score'] for x in done)/len(done),2),
                    'elapsed_s',round(time.monotonic()-started),flush=True)
    print('COMPLETE',dest,len(done),flush=True)


def summarize(args):
    root=Path(args.run); rows=read(root/'data.jsonl'); summary=[]; pairs=[]
    for family in ('qwen','qwen35'):
        for method in ('base','adapter'):
            runs={}
            for backend in ('hf','vllm'):
                path=root/f'{family}.{backend}.{method}.jsonl'
                data=read(path) if path.exists() else []
                for r in data:
                    if family=='qwen35':
                        from src.qwen35 import score_evaluation_prediction
                        r['project_score']=score_evaluation_prediction(r,rows[r['index']],'realworldqa')['score']
                    else:
                        r['project_score']=r['score']
                summary.append(dict(family=family,method=method,backend=backend,samples=len(data),
                    correct=sum(x['project_score'] for x in data),
                    accuracy=100*sum(x['project_score'] for x in data)/len(data) if data else None,
                    answer_correct=sum(x['score'] for x in data),
                    answer_accuracy=100*sum(x['score'] for x in data)/len(data) if data else None,
                    truncated=sum(x['hit_generation_limit'] for x in data),complete=len(data)==len(rows)))
                runs[backend]={r['index']:r for r in data}
            paired=[]
            for i in sorted(runs['hf'].keys() & runs['vllm'].keys()):
                a,b=runs['hf'][i],runs['vllm'][i]
                assert a['input_ids_sha256']==b['input_ids_sha256'],(family,method,i,'INPUT MISMATCH')
                assert a['prompt_sha256']==b['prompt_sha256']
                paired.append(dict(index=i,source_index=a['source_index'],hf_correct=bool(a['project_score']),
                    vllm_correct=bool(b['project_score']),
                    hf_answer_correct=bool(a['score']),vllm_answer_correct=bool(b['score']),
                    same_tokens=a['generated_token_ids']==b['generated_token_ids'],
                    hf_text=a['prediction_text'],vllm_text=b['prediction_text']))
            (root/f'{family}.{method}.paired.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in paired))
            pairs.append(dict(family=family,method=method,paired=len(paired),
                same_tokens=sum(x['same_tokens'] for x in paired),
                hf_correct_vllm_wrong=sum(x['hf_correct'] and not x['vllm_correct'] for x in paired),
                hf_wrong_vllm_correct=sum(not x['hf_correct'] and x['vllm_correct'] for x in paired)))
    result=dict(results=summary,paired=pairs,
        reported_scoring='Current project entrypoints: Qwen3-VL answer score; Qwen3.5 score_evaluation_prediction including unfinished-response invalid_zero. answer_accuracy also retained for common-parser comparison.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('worker','summarize'))
    p.add_argument('--run',required=True)
    p.add_argument('--family',choices=('qwen','qwen35'))
    p.add_argument('--backend',choices=('hf','vllm'))
    p.add_argument('--method',choices=('base','adapter'))
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    (worker if args.action=='worker' else summarize)(args)
