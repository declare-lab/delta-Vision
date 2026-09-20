"""Matched-input diagnostic: all images vs an ordered synthetic video.

Same 512x512 aspect-preserving letterbox images in both arms. Each image appears
twice consecutively in video to keep distinct images out of the same temporal
patch. Native timestamps/positions, no resampling, all images retained. Base and
Adapter both tested; no training or changes to the official benchmark protocol.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_images_as_video_20260914'
METHODS=('base','embedding_adapter')
MODES=('matched_images','as_video')
SIDE=512


def prepare(ds,row,mode):
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from src.benchmarks import build_benchmark_prompt
    images=[]
    sizes=[]
    for path in ds._image_paths(row):
        with Image.open(path) as raw:
            rgb=raw.convert('RGB')
            sizes.append(list(rgb.size))
            resized=ImageOps.contain(rgb,(SIDE,SIDE),Image.Resampling.BICUBIC)
            canvas=Image.new('RGB',(SIDE,SIDE),'white')
            canvas.paste(resized,((SIDE-resized.width)//2,(SIDE-resized.height)//2))
            images.append(canvas)
            resized.close();rgb.close()
    n=len(images)
    question=build_benchmark_prompt(row,ds.spec,ds.answer_instruction)
    try:
        if mode=='matched_images':
            content=ds._qwen_message_content(row,question,images,[])
            prompt=ds.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            inputs=ds.processor(text=[prompt],images=images,images_kwargs={'do_resize':False},return_tensors='pt',padding=True)
            grid=inputs['image_grid_thw']
            assert grid.tolist()==[[1,SIDE//16,SIDE//16]]*n
        else:
            assert mode=='as_video'
            # At 1 fps each duplicated pair is displayed at its mean timestamp:
            # 0.5, 2.5, 4.5, ... seconds in Qwen's native prompt expansion.
            timestamps=[2*i+.5 for i in range(n)]
            def replace(match):
                i=int(match[1])-1
                assert 0<=i<n
                return f'the frame at {timestamps[i]:.1f} seconds'
            question=re.sub(r'<\|image_(\d+)\|>',replace,question)
            question=('This video is a slideshow of the input images in their original order. '
                      'Each timestamped frame represents one separate input image.\n'+question)
            frames=np.stack([np.asarray(im) for im in images for _ in range(2)])
            assert all(np.array_equal(frames[2*i],frames[2*i+1]) for i in range(n))
            content=[dict(type='video',video=frames),dict(type='text',text=question)]
            prompt=ds.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            metadata={'total_num_frames':2*n,'fps':1.,'duration':float(2*n),'frames_indices':list(range(2*n))}
            inputs=ds.processor(text=[prompt],videos=[frames],video_metadata=[metadata],
                videos_kwargs={'do_resize':False,'do_sample_frames':False},return_tensors='pt',padding=True)
            grid=inputs['video_grid_thw']
            assert grid.tolist()==[[n,SIDE//16,SIDE//16]]
            rendered=ds.processor.tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=False)
            for t in timestamps:
                assert f'<{t:.1f} seconds>' in rendered
        expected=n*(SIDE//32)**2
        assert int((inputs['mm_token_type_ids']!=0).sum())==expected
        item=dict(inputs)
        for key in ('input_ids','attention_mask','mm_token_type_ids'):
            item[key]=item[key].squeeze(0)
        return item,dict(original_sizes=sizes,canvas=[SIDE,SIDE],image_count=n,
            visual_tokens=expected,grid=grid.tolist(),question=question)
    finally:
        for image in images:
            image.close()


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL,ADAPTER_CHECKPOINTS
    from src.audit_mmiu_random_results import extract_answer
    from baselines.eval_baselines import load_baseline_model,_qwen_inputs_from_item
    torch.set_num_threads(4);torch.manual_seed(42)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    model.eval().requires_grad_(False)
    assert processor.video_processor.temporal_patch_size==2
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(ADAPTER_CHECKPOINTS['embedding_adapter'],
        model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.mode=='embedding_adapter' and adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.native_prefix_memory=='legacy' and not adapter.native_ffn_carriers
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process=reject
    original_vision=model.model.visual.forward
    cache={}
    def cached_vision(*a,**k):
        if not cache:
            value=original_vision(*a,**k)
            cache['value']=(type(value),dict(value))
        cls,fields=cache['value']
        return cls(**fields)
    model.model.visual.forward=cached_vision
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
        data_root=str(ROOT/'data/benchmarks/muirbench'),max_samples=1000,prompt_layout='media_first_v1')
    assert len(ds.rows)==1000
    with torch.inference_mode(),(OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in range(shard,len(ds),8):
            row=ds.rows[index]
            image_patches=None
            for mode in MODES:
                item,info=prepare(ds,row,mode)
                if mode=='matched_images':
                    image_patches=item['pixel_values'].clone()
                else:
                    # Exact processed pixels: temporal patch of two identical
                    # frames equals the image processor's duplicated image patch.
                    assert image_patches.shape==item['pixel_values_videos'].shape
                    torch.testing.assert_close(image_patches,item['pixel_values_videos'],rtol=0,atol=1e-6)
                cache.clear()
                inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
                hidden,positions=ref.build_qwen_initial_context(model,inputs0)
                for method in METHODS:
                    inputs=dict(inputs0);h=hidden;pos=positions;generated=[]
                    eos=model.generation_config.eos_token_id
                    eos=eos if isinstance(eos,list) else [eos]
                    for _ in range(8):
                        model.model.rope_deltas=None
                        if method=='base':
                            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                        else:
                            logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                                initial_hidden=h,position_ids=pos,logits_to_keep=1)[0]
                        token=int(logits[0,-1].argmax());generated.append(token)
                        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+i) for i in range(len(row['choices']))]:break
                        new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                        inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                        h=torch.cat([h,model.model.get_input_embeddings()(new)],1)
                        pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                    prediction=extract_answer(text,row['choices'])
                    out.write(json.dumps(dict(index=index,source_index=row['index'],method=method,mode=mode,
                        task=row['task'],choices=row['choices'],images=row['images'],gold=row['answer'],
                        prediction=prediction,text=text,score=int(prediction==row['answer']),
                        generated_tokens=len(generated),deepstack=False,**info))+'\n')
            if index//8%5==0:print('DONE',shard,index,flush=True)


def summarize(codes):
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert not any(codes) and len(rows)==len({(r['index'],r['method'],r['mode']) for r in rows})==4000
    results=[]
    for task in ['all']+sorted({r['task'] for r in rows}):
        for method in METHODS:
            for mode in MODES:
                rs=[r for r in rows if r['method']==method and r['mode']==mode and (task=='all' or r['task']==task)]
                results.append(dict(task=task,method=method,mode=mode,n=len(rs),
                    accuracy=100*sum(r['score'] for r in rs)/len(rs)))
    payload=dict(expected=4000,completed=len(rows),exit_codes=codes,results=results,
        protocol='Same random1000 seed42; equal 512-square letterbox pixels and visual tokens, video repeats each image twice; no resampling; no DeepStack.')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs,logs=[],[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_images_as_video_audit',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    summarize(codes)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
