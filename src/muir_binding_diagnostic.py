"""All 184 geography/matching cases: image-to-choice binding format audit.

No answer-dependent prompts, no training, no benchmark manifest mutation.
Three paired layouts; native control and the same single KL adapter, DS off.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
PERMUTE=os.environ.get('MUIR_BINDING_MODE')=='permutation'
NATIVE_IDS=os.environ.get('MUIR_BINDING_MODE')=='native_ids'
OUTPUT=ROOT/('artifacts/diagnostics/muir_permutation_20260914' if PERMUTE else 'artifacts/diagnostics/muir_binding_20260914')
LAYOUTS=('rotate_media_fixed_choices','rotate_choices_fixed_media') if PERMUTE else ('end_marker','natural_number','explicit_option')
if NATIVE_IDS:
    OUTPUT=ROOT/'artifacts/diagnostics/muir_native_image_ids_20260914'
    LAYOUTS=('native_prefix_id',)


def permute_row(row,kind):
    if kind=='rotate_choices_fixed_media':
        n=len(row['choices'])
        return dict(row,choices=row['choices'][1:]+row['choices'][:1],answer=chr(65+(ord(row['answer'])-66)%n))
    assert kind=='rotate_media_fixed_choices'
    n=len(row['images'])
    def remap(text):
        return re.sub(r'<\|image_(\d+)\|>',lambda m:f'<|image_{(int(m.group(1))-2)%n+1}|>',text)
    return dict(row,images=row['images'][1:]+row['images'][:1],question=remap(row['question']),
                choices=[remap(c) for c in row['choices']])


def content_for(row,question,images,videos,layout):
    assert not videos
    if layout=='native_prefix_id':
        text=re.sub(r'<\|image_(\d+)\|>',lambda m:f'Picture {int(m.group(1))}',question)
        content=[]
        for i,image in enumerate(images,1):
            content.extend([{'type':'text','text':f'Picture {i}: '},{'type':'image','image':image}])
        return content+[{'type':'text','text':text}]
    assert layout in ('end_marker','natural_number','explicit_option')
    refs={}
    for j,choice in enumerate(row['choices']):
        match=re.fullmatch(r'<\|image_(\d+)\|>',choice.strip())
        if match:
            n=int(match.group(1));assert n not in refs;refs[n]=chr(65+j)
    # No access to row['answer']: only the published option/image association.
    text=re.sub(r'<\|image_(\d+)\|>',lambda m:f'Image {int(m.group(1))}',question)
    result=[]
    for i,image in enumerate(images,1):
        result.append({'type':'image','image':image})
        if layout=='end_marker':label=f'\n[End of Image {i}]\n'
        elif layout=='natural_number':label=f'\nThe image above is Image {i}.\n'
        else:
            label=f'\nThe image above is Image {i}.'
            if i in refs:label+=f' It corresponds to option {refs[i]}.'
            label+='\n'
        result.append({'type':'text','text':label})
    result.append({'type':'text','text':text})
    return result


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset,media_first_qwen_content
    from src.benchmarks import score_prediction
    from src.embedding_adapter_corrected_eval import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    def off(module,args,kwargs):return args,dict(kwargs,deepstack_visual_embeds=None)
    def reject(*a,**kw):raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off,with_kwargs=True)
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward;vision_cache={}
    def cached_vision(*a,**kw):
        if not vision_cache:
            value=original_vision(*a,**kw);vision_cache['value']=(type(value),dict(value))
        cls,fields=vision_cache['value'];return cls(**fields)
    model.model.visual.forward=cached_vision
    dataset=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',
        max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(dataset.rows) if r['task'] in ['Geographic Understanding','Image-Text Matching']]
    assert len(indices)==184
    if PERMUTE:indices=[i for i in indices if dataset.rows[i]['task']=='Image-Text Matching']
    old={(z['index'],z['method']):z for p in (ROOT/'artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913').glob('rows_*.jsonl')
        for l in p.open() if (z:=json.loads(l))['benchmark']=='muirbench' and z['method'] in ['base','static_kl']}
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            vision_cache.clear();reference_pixels=None;reference_grid=None
            original_row=dataset.rows[index]
            for layout in LAYOUTS:
                if PERMUTE:
                    dataset.rows[index]=permute_row(original_row,layout)
                    vision_cache.clear()
                def renderer(self,row,question,images,videos):
                    content=content_for(row,question,images,videos,'end_marker' if PERMUTE else layout)
                    if layout=='end_marker' or PERMUTE:assert content==media_first_qwen_content(question,images,videos)
                    if NATIVE_IDS:
                        native_content=[{'type':'image','image':im} for im in images]+[content[-1]]
                        native=processor.apply_chat_template([{'role':'user','content':native_content}],tokenize=False,
                            add_generation_prompt=True,add_vision_id=True)
                        manual=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                            add_generation_prompt=True)
                        assert native==manual,'Image labels must exactly match the model chat template'
                    return content
                dataset._qwen_message_content=types.MethodType(renderer,dataset)
                item=dataset[index];inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
                if reference_pixels is None:
                    reference_pixels=inputs0['pixel_values'].clone();reference_grid=inputs0['image_grid_thw'].clone()
                elif not PERMUTE:
                    assert torch.equal(reference_pixels,inputs0['pixel_values'])
                    assert torch.equal(reference_grid,inputs0['image_grid_thw'])
                initial,positions=ref.build_qwen_initial_context(model,inputs0)
                for method in ['base','static_kl']:
                    generated=[];inputs=dict(inputs0);hidden=initial;pos=positions
                    original_memories=None
                    if method=='static_kl':
                        memory=adapter.all_visual_memories_batched(initial[:,inputs0['mm_token_type_ids'][0].ne(0)])
                        original_memories=adapter.all_visual_memories_batched
                        adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memory,**kw:_m,adapter)
                    try:
                        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                        for step in range(128):
                            model.model.rope_deltas=None
                            if method=='base':logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                            else:logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                                initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
                            token=int(logits[0,-1].argmax());generated.append(token)
                            text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                            if token in eos or text in [chr(65+j) for j in range(len(item['choices']))]:break
                            new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                            inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                            inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                            if method=='static_kl':
                                hidden=torch.cat([hidden,model.model.get_input_embeddings()(new)],1)
                                pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                    finally:
                        if original_memories is not None:adapter.all_visual_memories_batched=original_memories
                    scored=score_prediction(metric='muirbench',prediction_text=text,answer=item['answer'],choices=item['choices'])
                    if layout=='end_marker':assert scored['prediction']==old[index,method]['prediction'],(index,method,scored,old[index,method])
                    previous=old[index,method]
                    expected=previous['prediction']
                    if layout=='rotate_choices_fixed_media':expected=chr(65+(ord(expected)-66)%len(item['choices']))
                    out.write(json.dumps(dict(index=index,task=item['row']['task'],layout=layout,method=method,text=text,
                        **scored,generated_tokens=len(generated),deepstack_enabled=False,
                        original_prediction=previous['prediction'],original_score=previous['score'],
                        same_choice_identity=scored['prediction']==expected,expected_prediction=expected,
                        image_order=item['row']['images'],choices=item['choices']),ensure_ascii=False)+'\n')
            dataset.rows[index]=original_row
            print('DONE',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[]
    for i in range(8):
        log=(OUTPUT/f'worker{i}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_binding_diagnostic',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({(r['index'],r['method'],r['layout']) for r in rows})==(336 if PERMUTE else 368 if NATIVE_IDS else 1104)
    result={}
    for task in ['Geographic Understanding','Image-Text Matching']:
        if PERMUTE and task!='Image-Text Matching':continue
        result[task]={}
        for layout in LAYOUTS:
            result[task][layout]={}
            for method in ['base','static_kl']:
                rs=[r for r in rows if (r['task'],r['layout'],r['method'])==(task,layout,method)]
                result[task][layout][method]=dict(n=len(rs),accuracy=100*sum(r['score'] for r in rs)/len(rs),
                    same_choice_identity=100*sum(r['same_choice_identity'] for r in rs)/len(rs),
                    predictions={x:sum(r['prediction']==x for r in rs) for x in sorted({r['prediction'] for r in rs})})
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
