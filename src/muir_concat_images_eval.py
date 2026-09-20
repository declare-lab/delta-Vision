"""Paired separate-image vs vertical-collage evaluation, frozen checkpoints.

Each source becomes an identical 512-square numbered panel in BOTH conditions.
The collage concatenates those panels before vision encoding. Total visual
tokens and even processed pixel patches match; image grids/positions differ.
No image is chosen using the answer. DeepStack is off for Base and Adapter.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_concat_images_20260914'
METHODS=('base','embedding_adapter')
MODES=('separate','concat')
SIDE=512


def prepare(ds,row,mode):
    import torch
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    from src.benchmarks import build_benchmark_prompt
    panels=[];sizes=[];collage=None
    font=ImageFont.load_default(size=24)
    try:
        for g,path in enumerate(ds._image_paths(row),1):
            with Image.open(path) as raw:
                rgb=raw.convert('RGB');sizes.append(list(rgb.size))
                scaled=ImageOps.contain(rgb,(SIDE,SIDE-32),Image.Resampling.BICUBIC)
                panel=Image.new('RGB',(SIDE,SIDE),'white')
                panel.paste(scaled,((SIDE-scaled.width)//2,32+(SIDE-32-scaled.height)//2))
                draw=ImageDraw.Draw(panel);draw.rectangle((0,0,SIDE-1,31),fill=(238,238,238))
                draw.text((8,3),f'Image {g}',font=font,fill='black')
                panels.append(panel);scaled.close();rgb.close()
        n=len(panels)
        if mode=='separate':images=panels
        else:
            assert mode=='concat'
            collage=Image.new('RGB',(SIDE,SIDE*n),'white')
            for j,panel in enumerate(panels):collage.paste(panel,(0,j*SIDE))
            images=[collage]
        question=build_benchmark_prompt(row,ds.spec,ds.answer_instruction)
        def reference(match):
            number=int(match[1]);assert 1<=number<=n
            return f'Image {number}'
        question=re.sub(r'<\|image_(\d+)\|>',reference,question)
        # Identical text in BOTH arms. The visible image labels identify panels
        # whether they occupy separate inputs or one concatenated canvas.
        question=('Each numbered panel is one picture. Use the visible labels Image 1 through '
                  f'Image {n} to identify the pictures. The panels are provided in numerical order.\n'+question)
        content=[dict(type='image',image=im) for im in images]+[dict(type='text',text=question)]
        prompt=ds.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        inputs=ds.processor(text=[prompt],images=images,images_kwargs={'do_resize':False},return_tensors='pt',padding=True)
        expected_grid=[[1,32,32]]*n if mode=='separate' else [[1,32*n,32]]
        assert inputs['image_grid_thw'].tolist()==expected_grid
        visual_tokens=int(inputs['mm_token_type_ids'].ne(0).sum())
        assert visual_tokens==n*256
        item=dict(inputs)
        for key in ('input_ids','attention_mask','mm_token_type_ids'):item[key]=item[key].squeeze(0)
        pixel_hash=hashlib.sha256(item['pixel_values'].contiguous().numpy().tobytes()).hexdigest()
        return item,dict(source_sizes=sizes,panels=n,processed_images=len(images),grid=expected_grid,
                         visual_tokens=visual_tokens,pixel_sha256=pixel_hash,question=question)
    finally:
        for panel in panels:panel.close()
        if collage is not None:collage.close()


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import score_prediction
    from src.multimodal_baseline_suite import MODEL,CHECKPOINT
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
                                                          torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==adapter.active_adapter_layers==0
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward;cache={}
    def cached_vision(*a,**k):
        pixels=a[0] if a else k['hidden_states'];grid=k['grid_thw']
        if not cache:
            value=original_vision(*a,**k)
            cache.update(cls=type(value),fields=dict(value),pixels=pixels.clone(),grid=grid.clone())
        else:
            assert torch.equal(pixels,cache['pixels']) and torch.equal(grid,cache['grid'])
        return cache['cls'](**cache['fields'])
    model.model.visual.forward=cached_vision
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
        data_root=str(ROOT/'data/benchmarks/muirbench'),max_samples=1000,prompt_layout='media_first_v1')
    path=OUT/f'rows_{shard}.jsonl'
    done={(r['index'],r['method'],r['mode']) for l in path.open() if (r:=json.loads(l))} if path.exists() else set()
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for index in range(shard,1000,8):
            if all((index,m,c) in done for m in METHODS for c in MODES):continue
            row=ds.rows[index];patches=None;reference_info=None
            for mode in MODES:
                item,info=prepare(ds,row,mode)
                if mode=='separate':patches=item['pixel_values'].clone();reference_info=info
                else:
                    assert torch.equal(item['pixel_values'],patches),'Pixels changed across the concat comparison'
                    assert info['pixel_sha256']==reference_info['pixel_sha256']
                    assert info['question']==reference_info['question']
                cache.clear()  # Never reuse separate-image vision features for a collage.
                inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
                hidden,positions=ref.build_qwen_initial_context(model,inputs0)
                for method in METHODS:
                    if (index,method,mode) in done:continue
                    started=time.time();inputs=dict(inputs0);h=hidden;pos=positions;generated=[]
                    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                    for step in range(8):
                        model.model.rope_deltas=None
                        if method=='base':logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                        else:logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,initial_hidden=h,
                                                                     position_ids=pos,logits_to_keep=1)[0]
                        token=int(logits[0,-1].argmax());generated.append(token)
                        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+j) for j in range(len(row['choices']))]:break
                        new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                        inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                        h=torch.cat([h,model.model.get_input_embeddings()(new)],1)
                        pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                    score=score_prediction(metric='muirbench',prediction_text=text,answer=row['answer'],
                                           choices=row['choices'],question=row['question'])
                    out.write(json.dumps(dict(index=index,source_index=row['index'],method=method,mode=mode,
                        task=row['task'],text=text,**score,**info,deepstack=False,generated_tokens=len(generated),
                        seconds=time.time()-started),ensure_ascii=False)+'\n')
            if index//8%10==0:print('DONE',shard,index,flush=True)


def aggregate(complete=False):
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open() if l.strip()]
    lookup={(r['index'],r['method'],r['mode']):r for r in rows}
    assert len(lookup)==len(rows)
    if complete:assert set(lookup)=={(i,m,c) for i in range(1000) for m in METHODS for c in MODES}
    results=[]
    for method in METHODS:
        for mode in MODES:
            rs=[r for r in rows if r['method']==method and r['mode']==mode]
            results.append(dict(method=method,mode=mode,n=len(rs),accuracy=100*sum(r['score'] for r in rs)/len(rs) if rs else None))
    result=dict(completed=len(rows),expected=4000,results=results)
    if complete:
        result['paired']={m:dict(improved=sum(lookup[i,m,'concat']['score']>lookup[i,m,'separate']['score'] for i in range(1000)),
                                worsened=sum(lookup[i,m,'concat']['score']<lookup[i,m,'separate']['score'] for i in range(1000))) for m in METHODS}
        result['tasks']={t:{f'{m}/{c}':dict(n=len(rs),accuracy=100*sum(r['score'] for r in rs)/len(rs))
                           for m in METHODS for c in MODES if (rs:=[r for r in rows if r['task']==t and r['method']==m and r['mode']==c])}
                         for t in sorted({r['task'] for r in rows})}
        for i in range(1000):
            rs=[lookup[i,m,c] for m in METHODS for c in MODES]
            assert len({r['pixel_sha256'] for r in rs})==len({r['question'] for r in rs})==1
            assert len({r['visual_tokens'] for r in rs})==1
    tmp=OUT/'summary.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(OUT/'summary.json')
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    from src.multimodal_baseline_suite import MODEL,CHECKPOINT
    plan=dict(model=MODEL,checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              manifest=str(SOURCE/'muirbench_random1000.jsonl'),manifest_sha256=hashlib.sha256((SOURCE/'muirbench_random1000.jsonl').read_bytes()).hexdigest(),
              panels='512x512; 32px numbered header; aspect-preserving fit within 512x480',
              collage='vertical concatenation before vision encoding, no resizing',
              tokens='256 per original image in both conditions',deepstack=False,max_new_tokens=8,seed=42,
              note='Matched panels and text in both arms; different from the historical original-resolution/end-marker prompt.')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    jobs={};logs=[];start=time.time()
    for s in range(8):
        log=(OUT/f'worker{s}.log').open('w');logs.append(log)
        jobs[s]=subprocess.Popen([sys.executable,'-m','src.muir_concat_images_eval',str(s)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
            stdout=log,stderr=subprocess.STDOUT)
    while any(p.poll() is None for p in jobs.values()):
        aggregate()
        (OUT/'status.json').write_text(json.dumps(dict(elapsed=time.time()-start,
            workers={s:dict(pid=p.pid,exit_code=p.poll()) for s,p in jobs.items()}),indent=2)+'\n')
        time.sleep(10)
    for log in logs:log.close()
    codes=[p.returncode for p in jobs.values()];assert not any(codes),codes
    result=aggregate(complete=True)
    lines=['# MuirBench: separate images vs concatenated single image','',
           'Same frozen Base/Embedding Adapter, random1000 seed42, no DeepStack, greedy ≤8 tokens. '
           'Every image receives the same 512-square numbered panel in both arms. '
           'The concat arm joins panels vertically before vision encoding. '
           'Processed pixel patches, total visual-token budget and question text match exactly. '
           'This is a new input-format comparison, not a replacement of the historical original-format scores.','',
           '| Model | Separate | Concat | Change |','|---|---:|---:|---:|']
    for m in METHODS:
        scores={r['mode']:r['accuracy'] for r in result['results'] if r['method']==m}
        lines.append(f"| {m} | {scores['separate']:.2f}% | {scores['concat']:.2f}% | {scores['concat']-scores['separate']:+.2f} pp |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    (OUT/'status.json').write_text(json.dumps(dict(state='complete',elapsed=time.time()-start,exit_codes=codes),indent=2)+'\n')
    print(json.dumps(result['results'],indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
