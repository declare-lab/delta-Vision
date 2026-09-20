"""Match the checkpoint's recorded vision backend without changing the adapter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PIXELS=os.environ.get('MUIR_SOURCE_PIXELS')=='1'
OUTPUT=ROOT/('artifacts/diagnostics/muir_source_pixels_20260914' if SOURCE_PIXELS else 'artifacts/diagnostics/muir_vision_backend_20260914')
BACKENDS=('reencoded_jpeg','source_pixels') if SOURCE_PIXELS else ('sdpa','flash_attention_2')


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
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    adapter,meta=ref.load_qwen_embedding_adapter_checkpoint(CHECKPOINT,model.model.language_model,
        torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False);adapter.eval().requires_grad_(False)
    training=json.loads((CHECKPOINT.parents[1]/'config.json').read_text())
    assert training['attn_implementation']=='flash_attention_2'
    assert adapter.adapter_attention_backend=='efficient'
    ds=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',
        max_samples=1000,prompt_layout='media_first_v1')
    raw=None
    if SOURCE_PIXELS:
        from datasets import load_dataset,DownloadConfig
        raw=load_dataset('MUIRBENCH/MUIRBENCH',split='test',download_config=DownloadConfig(local_files_only=True))
    historical={}
    for path in (ROOT/'artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914').glob('rows_*.jsonl'):
        for line in path.open():
            r=json.loads(line);historical[r['index']]=r
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in range(shard,1000,8):
            item=ds[index];inputs0=_qwen_inputs_from_item(item,torch.device('cuda'))
            record=dict(index=index,task=item['row']['task'],image_count=len(item['row']['images']),backends={})
            anchors={}
            for backend in BACKENDS:
                actual_backend='sdpa' if SOURCE_PIXELS else backend
                model.model.visual.set_attn_implementation(actual_backend)
                assert all(b.attn.config._attn_implementation==actual_backend for b in model.model.visual.blocks)
                selected_inputs=dict(inputs0)
                if SOURCE_PIXELS and backend=='source_pixels':
                    doc=raw[index]
                    assert doc['idx']==item['row']['idx'] and doc['answer']==item['answer']
                    images=[image.convert('RGB') for image in doc['image_list']]
                    pixels=processor.image_processor(images=images,return_tensors='pt')
                    assert torch.equal(pixels['image_grid_thw'].to('cuda'),inputs0['image_grid_thw'])
                    selected_inputs['pixel_values']=pixels['pixel_values'].to(inputs0['pixel_values'])
                    record['pixel_max_abs']=float((selected_inputs['pixel_values']-inputs0['pixel_values']).abs().max())
                    record['pixel_mean_abs']=float((selected_inputs['pixel_values']-inputs0['pixel_values']).abs().mean())
                    for image in images:image.close()
                initial,pos0=ref.build_qwen_initial_context(model,selected_inputs)
                visual=inputs0['mm_token_type_ids'][0].ne(0)
                anchors[backend]=initial[:,visual].clone()
                if backend==BACKENDS[0]:reference_positions=pos0.clone()
                else:assert torch.equal(reference_positions,pos0)
                memories=adapter.all_visual_memories_batched(initial[:,visual])
                original=adapter.all_visual_memories_batched
                adapter.all_visual_memories_batched=types.MethodType(lambda self,*a,_m=memories,**kw:_m,adapter)
                generated=[];inputs=dict(selected_inputs);h=initial;pos=pos0
                try:
                    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                    for step in range(128):
                        logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                            initial_hidden=h,position_ids=pos,logits_to_keep=1)[0]
                        token=int(logits[0,-1].argmax());generated.append(token)
                        text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+j) for j in range(len(item['choices']))]:break
                        new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                        inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                        h=torch.cat([h,model.model.get_input_embeddings()(new)],1)
                        pos=torch.cat([pos,pos[:,:,-1:]+1],2)
                finally:adapter.all_visual_memories_batched=original
                score=score_prediction(metric='muirbench',prediction_text=text,answer=item['answer'],choices=item['choices'])
                if backend==BACKENDS[0]:assert text==historical[index]['candidate_text'],(index,text,historical[index]['candidate_text'])
                record['backends'][backend]=dict(text=text,tokens=generated,**score)
            a,b=anchors[BACKENDS[0]].float(),anchors[BACKENDS[1]].float()
            record['anchor_relative_l2']=float((a-b).norm()/a.norm())
            record['anchor_max_abs']=float((a-b).abs().max())
            out.write(json.dumps(record)+'\n')
            if index%80==shard:print('PROGRESS',index,flush=True)
            del anchors,memories,a,b


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for shard in range(8):
        with (OUTPUT/f'worker{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_vision_backend_audit',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),
                stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert not any(codes),codes
    rows=[json.loads(line) for p in OUTPUT.glob('rows_*.jsonl') for line in p.open()]
    assert len(rows)==len({r['index'] for r in rows})==1000
    summary=dict(samples=len(rows),accuracy={b:sum(r['backends'][b]['score'] for r in rows)/10 for b in BACKENDS},
        changed_predictions=sum(r['backends'][BACKENDS[0]]['prediction']!=r['backends'][BACKENDS[1]]['prediction'] for r in rows),
        mean_anchor_relative_l2=sum(r['anchor_relative_l2'] for r in rows)/len(rows),
        max_anchor_relative_l2=max(r['anchor_relative_l2'] for r in rows))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
