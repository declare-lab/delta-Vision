"""Audit source images/options and cached prompts, without running inference."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_source_prompt_audit_20260914'
ARROW=Path('/lustre-data/leijingdi/cache/huggingface/datasets/MUIRBENCH___muirbench/default/0.0.0/4c393cffc985c77d28de3b9045e2e5186920df80')


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from datasets import Dataset,concatenate_datasets
    from transformers import AutoProcessor
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL
    torch.set_num_threads(2)
    os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1',QWEN_VIDEO_NUM_FRAMES='8')
    raw=concatenate_datasets([Dataset.from_file(str(p)) for p in sorted(ARROW.glob('*.arrow'))])
    processor=AutoProcessor.from_pretrained(MODEL)
    tok=processor.tokenizer
    ds=QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'),processor,'muirbench',
                           data_root=str(ROOT/'data/benchmarks/muirbench'),
                           cache_dir=SOURCE/'processed/muirbench',prompt_layout='media_first_v1')
    assert len(ds)==1000 and len(raw)==2600
    old={r['index']:r for p in SOURCE.glob('embedding_adapter_shard*.jsonl')
         for l in p.open() if (r:=json.loads(l))['retention']==1.}
    result_path=OUT/f'rows_{shard}.jsonl'
    done={json.loads(l)['index'] for l in result_path.open()} if result_path.exists() else set()
    with result_path.open('a',buffering=1) as out:
        for i in range(shard,1000,8):
            if i in done:continue
            row=ds.rows[i];doc=raw[row['source_manifest_index']]
            assert str(doc['idx'])==row['idx']
            for key in ('task','image_relation','image_type','counterpart_idx'):
                assert row[key]==doc[key],(i,key)
            assert row['answer']==str(doc['answer']).strip()==old[i]['gold']
            counter=0
            def number(match):
                nonlocal counter
                counter+=1;return f'<|image_{counter}|>'
            question=re.sub('<image>',number,doc['question'])
            choices=[re.sub('<image>',number,c) for c in doc['options']]
            assert question==row['question'] and choices==row['choices'],(i,'question/option mapping')
            assert counter==len(doc['image_list'])==len(row['images']),(i,'source placeholder/image count')
            image_paths=ds._image_paths(row)
            for g,(image,path) in enumerate(zip(doc['image_list'],image_paths)):
                expected=BytesIO();image.convert('RGB').save(expected,format='JPEG',quality=95)
                assert hashlib.sha256(expected.getvalue()).digest()==hashlib.sha256(path.read_bytes()).digest(),(i,g,'source image/file mismatch')
            # Independent prompt assembly: no production build_benchmark_prompt
            # or media_first_qwen_content call is used here.
            instruction="Answer with the option's letter from the given choices directly."
            prompt='\n'.join([question.strip()]+[f'{chr(65+j)}. {c.strip()}' for j,c in enumerate(choices)]+[instruction])
            path=ds._cache_path(row,image_paths,[],prompt)
            assert path.is_file(),(i,'cache missing',str(path))
            cached=torch.load(path,map_location='cpu',weights_only=False,mmap=True)['item']
            assert cached['index']==row['index'] and cached['answer']==row['answer']
            assert cached['choices']==choices
            assert cached['row']['images']==row['images'] and cached['row']['question']==question
            def textual(match):return f'Image {int(match.group(1))}'
            text_prompt=re.sub(r'<\|image_(\d+)\|>',textual,prompt)
            content=[]
            for g in range(1,counter+1):
                content.extend([dict(type='image',image=None),dict(type='text',text=f'\n[End of Image {g}]\n')])
            content.append(dict(type='text',text=text_prompt))
            rendered=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            counts=(cached['image_grid_thw'].prod(-1)//processor.image_processor.merge_size**2).tolist()
            assert len(counts)==counter
            parts=rendered.split(processor.image_token);assert len(parts)==counter+1
            expanded=parts[0]+''.join(processor.image_token*n+tail for n,tail in zip(counts,parts[1:]))
            ids=torch.tensor(tok.encode(expanded,add_special_tokens=False),dtype=cached['input_ids'].dtype)
            assert torch.equal(ids,cached['input_ids']),(i,'cached prompt differs from source')
            assert bool(cached['attention_mask'].all())
            assert torch.equal(cached['mm_token_type_ids'].ne(0),ids.eq(processor.image_token_id))
            # Text/input metadata match the actual completed evaluation, not
            # just another reconstructed copy of the annotations.
            digest=hashlib.sha256()
            for key in ('input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
                if torch.is_tensor(cached.get(key)):
                    value=cached[key].contiguous().cpu()
                    digest.update(str((key,tuple(value.shape),str(value.dtype))).encode())
                    digest.update(value.view(torch.uint8).numpy().tobytes())
            assert digest.hexdigest()==old[i]['input_sha256'],(i,'evaluated input mismatch')
            out.write(json.dumps(dict(index=i,source_index=row['index'],images=counter,
                    image_files_exact=True,source_choices_exact=True,cached_prompt_exact=True,
                    evaluated_input_hash_exact=True,sequence_tokens=len(ids)))+'\n')
            if i//8%20==0:print('PASS',shard,i,flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[];logs=[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_source_prompt_audit',str(shard)],cwd=ROOT,
            env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==1000 and len({r['index'] for r in rows})==1000
    summary=dict(samples=len(rows),images=sum(r['images'] for r in rows),
                 image_files_exact=True,source_choices_exact=True,cached_prompt_exact=True,evaluated_input_hash_exact=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
