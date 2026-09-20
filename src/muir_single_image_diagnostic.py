"""Read-only semantic control: judge each candidate image independently.

Forced A(Yes)/B(No) logit margin, not a replacement MuirBench accuracy.
Uses every image in all 84 matching scenes, including unanswerable scenes.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/diagnostics/muir_single_image_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
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
    dataset=QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),processor,'muirbench',
        max_samples=1000,prompt_layout='media_first_v1')
    indices=[i for i,r in enumerate(dataset.rows) if r['task']=='Image-Text Matching']
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in ['A','B']]
    assert all(len(x)==1 for x in ids);yes,no=[x[0] for x in ids]
    with torch.inference_mode(),(OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            original=dataset.rows[index]
            description=original['question'].split(':',1)[1].strip()
            gold_choice=original['choices'][ord(original['answer'])-65]
            match=re.fullmatch(r'<\|image_(\d+)\|>',gold_choice)
            gold_image=int(match.group(1))-1 if match else None
            assert match or gold_choice=='None of the choices provided'
            for image_index,path in enumerate(original['images']):
                # Prompt depends only on the published description, not gold.
                dataset.rows[index]=dict(original,images=[path],question=f'Does this image match the following description?\n{description}',
                    choices=['Yes','No'],answer='A' if image_index==gold_image else 'B')
                item=dataset[index];inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
                for method in ['base','static_kl']:
                    model.model.rope_deltas=None
                    if method=='base':logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                    else:logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,logits_to_keep=1)[0][0,-1].float()
                    margin=float(logits[yes]-logits[no])
                    out.write(json.dumps(dict(index=index,image_index=image_index,method=method,gold_image=gold_image,
                        positive=image_index==gold_image,margin=margin,forced_correct=(margin>0)==(image_index==gold_image),
                        greedy_token=processor.tokenizer.decode([int(logits.argmax())]),deepstack_enabled=False))+'\n')
            dataset.rows[index]=original
            print('DONE',index,flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    logs=[];jobs=[]
    for i in range(8):
        f=(OUTPUT/f'worker{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_single_image_diagnostic',str(i)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.open()]
    assert len(rows)==len({(r['index'],r['image_index'],r['method']) for r in rows})==504
    result={}
    for method in ['base','static_kl']:
        rs=[r for r in rows if r['method']==method];pos=[r for r in rs if r['positive']];neg=[r for r in rs if not r['positive']]
        groups=[[r for r in rs if r['index']==i] for i in sorted({r['index'] for r in rs})]
        answerable=[g for g in groups if g[0]['gold_image'] is not None]
        result[method]=dict(images=len(rs),positive_count=len(pos),negative_count=len(neg),
            forced_binary_accuracy=100*sum(r['forced_correct'] for r in rs)/len(rs),
            sensitivity=100*sum(r['margin']>0 for r in pos)/len(pos),
            specificity=100*sum(r['margin']<=0 for r in neg)/len(neg),
            auc=sum((a['margin']>b['margin'])+.5*(a['margin']==b['margin']) for a in pos for b in neg)/(len(pos)*len(neg)),
            answerable_scenes=len(answerable),
            answerable_top_image_accuracy=100*sum(max(g,key=lambda r:r['margin'])['positive'] for g in answerable)/len(answerable),
            greedy_outside_AB=sum(r['greedy_token'] not in ['A','B'] for r in rs))
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
