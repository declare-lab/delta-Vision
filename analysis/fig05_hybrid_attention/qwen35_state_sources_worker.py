import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35 import load_model,prepare_inputs,generate_evaluation_answer,sha
from src.benchmarks import get_benchmark_spec,build_benchmark_prompt
from analysis.fig05_hybrid_attention.qwen35_state_sources import StateSourceTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--stage',choices=['validate','analysis'],default='analysis');args=p.parse_args()
    config=json.loads((args.run/'config.json').read_text());info=config['evaluation']['realworldqa']
    torch.set_num_threads(4);torch.manual_seed(44)
    processor,model,adapter,controller=load_model(config,torch.device('cuda:0'))
    torch.set_float32_matmul_precision('highest')
    spec=get_benchmark_spec('realworldqa')
    assert sha(info['path'])==info['sha256']
    rows=[json.loads(l) for l in Path(info['path']).read_text().splitlines()];assert len(rows)==765
    refs={r['index']:r for f in (Path(config['reference_run'])/'accuracy').glob('realworldqa.shard*.jsonl') for r in map(json.loads,f.read_text().splitlines())}
    if args.stage=='validate':
        loader=importlib.util.spec_from_file_location('source_tests',ROOT/'test/diagnostics/test_qwen35_state_sources.py')
        tests=importlib.util.module_from_spec(loader);loader.loader.exec_module(tests)
        print('KERNEL_TESTS',tests.run_tests(),flush=True)
    indices=[0] if args.stage=='validate' else list(range(args.shard,765,8))
    path=args.run/args.stage/f'realworldqa.shard{args.shard}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['index'] for r in old]==indices[:len(old)]
    with path.open('a') as handle,torch.inference_mode():
        for index in indices[len(old):]:
            started=time.time();row=rows[index]
            inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
            input_sha=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
            assert input_sha==refs[index]['input_ids_sha256']
            expected=next(v for v in refs[index]['variants'] if v['method']=='native' and v['condition']=='unmodified')
            mask=inputs['mm_token_type_ids'].eq(1);tracker=StateSourceTracker(model,mask)
            with tracker.activate():
                generated=generate_evaluation_answer(model,processor,inputs,row,spec,config,max_new_tokens=info['max_new_tokens'])
            assert generated['generated_token_ids']==expected['generated_token_ids'],(index,'Native generation changed')
            records=tracker.result()
            assert all(len(x['positions'])==int((~mask).sum())+generated['generated_tokens']-1 for x in records)
            result=dict(index=index,benchmark='realworldqa',input_ids_sha256=input_sha,
                prompt_length=mask.shape[1],visual_start=tracker.start,visual_end=tracker.end,
                input_token_ids=inputs['input_ids'][0].tolist(),generated=generated,native_generation_exact=True,
                layers=records,elapsed_s=time.time()-started)
            handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            progress=args.run/args.stage/f'progress{args.shard}.json'
            tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(dict(completed=indices.index(index)+1,index=index))+'\n');tmp.replace(progress)
            print(json.dumps(dict(index=index,stage=args.stage,elapsed_s=result['elapsed_s'])),flush=True)
            del inputs,tracker,result,records
    controller.close()


if __name__=='__main__':main()
