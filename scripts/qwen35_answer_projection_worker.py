import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35_experiment import load_model,prepare_inputs,sha
from src.benchmarks import get_benchmark_spec,build_benchmark_prompt,_realworldqa_choices_from_question
from src.qwen35_answer_projection import AnswerProjectionTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--stage',choices=['validate','analysis'],default='analysis');args=p.parse_args()
    config=json.loads((args.run/'config.json').read_text());info=config['evaluation']['realworldqa']
    torch.set_num_threads(4);torch.manual_seed(44)
    processor,model,adapter,controller=load_model(config,torch.device('cuda:0'))
    torch.set_float32_matmul_precision('highest');spec=get_benchmark_spec('realworldqa')
    assert sha(info['path'])==info['sha256']
    rows=[json.loads(l) for l in Path(info['path']).read_text().splitlines()];assert len(rows)==765
    refs={r['index']:r for f in (Path(config['reference_run'])/'accuracy').glob('realworldqa.shard*.jsonl') for r in map(json.loads,f.read_text().splitlines())}
    selected=config['selected_indices'];assert len(selected)==438
    indices=selected[:1] if args.stage=='validate' else selected[args.shard::8]
    path=args.run/args.stage/f'realworldqa.shard{args.shard}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['index'] for r in old]==indices[:len(old)]
    with path.open('a') as handle,torch.inference_mode():
        for index in indices[len(old):]:
            started=time.time();row=rows[index];choices=_realworldqa_choices_from_question(row['question'])
            labels=list('ABCD'[:len(choices)]);gold=str(row['answer']).strip().upper();assert gold in labels
            ids=[processor.tokenizer(label,add_special_tokens=False)['input_ids'] for label in labels]
            assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids]
            inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
            input_sha=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest();assert input_sha==refs[index]['input_ids_sha256']
            mask=inputs['mm_token_type_ids'].eq(1);tracker=AnswerProjectionTracker(model,mask)
            with tracker.activate():
                output=model(**inputs,use_cache=True,logits_to_keep=1)
            logits=output.logits[0,-1].float()
            reference=next(v for v in refs[index]['variants'] if v['method']=='native' and v['condition']=='unmodified')
            assert int(logits.argmax())==reference['generated_token_ids'][0]
            if args.stage=='validate':
                direct=model(**inputs,use_cache=True,logits_to_keep=1).logits[0,-1].float()
                torch.testing.assert_close(logits,direct,rtol=0,atol=0)
                del direct
            positive=labels.index(gold)
            negative=max((j for j in range(len(labels)) if j!=positive),key=lambda j:float(logits[ids[j]]))
            direction=model.lm_head.weight[ids[positive]].float()-model.lm_head.weight[ids[negative]].float()
            records=tracker.project(direction)
            result=dict(index=index,benchmark='realworldqa',input_ids_sha256=input_sha,
                gold=gold,negative=labels[negative],positive_token_id=ids[positive],negative_token_id=ids[negative],
                candidate_logits={label:float(logits[token]) for label,token in zip(labels,ids)},
                final_logit_margin=float(logits[ids[positive]]-logits[ids[negative]]),
                native_first_token_exact=True,full_logits_exact=args.stage=='validate',
                original_generation_score=reference['score'],layers=records,elapsed_s=time.time()-started)
            handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            progress=args.run/args.stage/f'progress{args.shard}.json';tmp=progress.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(completed=indices.index(index)+1,index=index))+'\n');tmp.replace(progress)
            print(json.dumps(dict(index=index,stage=args.stage,elapsed_s=result['elapsed_s'])),flush=True)
            del inputs,tracker,result,records,output,logits,direction
    controller.close()


if __name__=='__main__':main()
