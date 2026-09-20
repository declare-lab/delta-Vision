"""Identify candidate interference vs final selection: paired single/multi reads.

All current 132 image-choice matching scenes, every candidate, native and the
same PixMo adapter. Counterbalance Yes/No letters. Not official Muir accuracy.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT = ROOT/'artifacts/diagnostics/muir_candidate_isolation_20260914'
METHODS = ('base', 'embedding_adapter')
CONDITIONS = ('single', 'multi_target')
CLEAR_VISION_PER_TARGET = False


def probe_row(original, target, condition, reverse):
    # No access to original gold in prompt construction. Retain original question
    # verbatim to avoid brittle colon-based extraction of the description.
    assert not re.search(r'<\|image_\d+\|>', original['question'])
    number = 1 if condition=='single' else target+1
    question = (f'Image-selection question: {original["question"]}\n'
                f'Is <|image_{number}|> a correct image to select for this question? '
                f'Judge only <|image_{number}|>. Answer Yes or No.')
    return dict(original, question=question,
                images=[original['images'][target]] if condition=='single' else original['images'],
                choices=['No','Yes'] if reverse else ['Yes','No'], answer=None)


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL, ADAPTER_CHECKPOINTS
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('base',MODEL,torch.bfloat16,'cuda:0',1.,'sdpa')
    model.eval().requires_grad_(False)
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(ADAPTER_CHECKPOINTS['embedding_adapter'],
        model.model.language_model,torch.device('cuda'),torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    assert adapter.adapter_start_layer==0 and adapter.active_adapter_layers==0
    assert adapter.mode=='embedding_adapter' and not adapter.native_ffn_carriers
    adapter.eval().requires_grad_(False)
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k:(a,dict(k,deepstack_visual_embeds=None)),with_kwargs=True)
    def reject(*a,**k):
        raise AssertionError('DeepStack executed')
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
    indices=[i for i,r in enumerate(ds.rows) if r['task']=='Image-Text Matching' and
        any(re.fullmatch(r'<\|image_\d+\|>',c.strip()) for c in r['choices'])]
    assert len(indices)==132 and sum(len(ds.rows[i]['images']) for i in indices)==441
    letters=[processor.tokenizer.encode(c,add_special_tokens=False) for c in ('A','B')]
    assert all(len(x)==1 for x in letters)
    with torch.inference_mode(),(OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for index in indices[shard::8]:
            original=ds.rows[index]
            gold=original['choices'][ord(original['answer'])-65]
            match=re.fullmatch(r'<\|image_(\d+)\|>',gold.strip())
            assert match or gold=='None of the choices provided'
            gold_image=int(match[1])-1 if match else None
            for condition in CONDITIONS:
                cache.clear()
                reference_pixels=None
                for target in range(len(original['images'])):
                    if condition=='single' or CLEAR_VISION_PER_TARGET:
                        cache.clear()
                        reference_pixels=None
                    for reverse in (False,True):
                        row=probe_row(original,target,condition,reverse)
                        ds.rows[index]=row
                        item=ds[index]
                        # Reuse vision only if all image pixels and grid match.
                        if reference_pixels is None:
                            reference_pixels=(item['pixel_values'].clone(),item['image_grid_thw'].clone())
                        else:
                            assert torch.equal(reference_pixels[0],item['pixel_values'])
                            assert torch.equal(reference_pixels[1],item['image_grid_thw'])
                        inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
                        hidden,positions=ref.build_qwen_initial_context(model,inputs)
                        yes=letters[int(reverse)][0]
                        no=letters[int(not reverse)][0]
                        for method in METHODS:
                            model.model.rope_deltas=None
                            if method=='base':
                                logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                            else:
                                logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                                    initial_hidden=hidden,position_ids=positions,logits_to_keep=1)[0][0,-1].float()
                            margin=float(logits[yes]-logits[no])
                            out.write(json.dumps(dict(index=index,source_index=original['index'],target=target,
                                images=original['images'],method=method,condition=condition,reverse=reverse,
                                gold_image=gold_image,positive=target==gold_image,margin=margin,
                                greedy_token=processor.tokenizer.decode([int(logits.argmax())]),
                                ab_mass=float(logits.softmax(-1)[yes]+logits.softmax(-1)[no]),
                                deepstack=False,question=row['question'],presented_images=row['images']))+'\n')
            ds.rows[index]=original
            print('DONE',shard,index,flush=True)


def summarize(codes):
    raw=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert not any(codes) and len(raw)==len({(r['index'],r['target'],r['method'],r['condition'],r['reverse']) for r in raw})==3528
    lookup={(r['index'],r['target'],r['method'],r['condition'],r['reverse']):r for r in raw}
    rows=[]
    for r in raw:
        if r['reverse']:
            continue
        other=lookup[r['index'],r['target'],r['method'],r['condition'],True]
        rows.append(dict(r,margin=(r['margin']+other['margin'])/2,
            answer_order_agreement=(r['margin']>0)==(other['margin']>0)))
    results=[]
    for method in METHODS:
        for condition in CONDITIONS:
            group=[r for r in rows if r['method']==method and r['condition']==condition]
            pos=[r for r in group if r['positive']]
            neg=[r for r in group if not r['positive']]
            scenes=[[r for r in group if r['index']==i] for i in sorted({r['index'] for r in group})]
            answerable=[g for g in scenes if g[0]['gold_image'] is not None]
            unanswerable=[g for g in scenes if g[0]['gold_image'] is None]
            def mean(xs):
                return sum(xs)/len(xs) if xs else None
            results.append(dict(method=method,condition=condition,candidates=len(group),
                positive_count=len(pos),negative_count=len(neg),
                sensitivity=100*mean([r['margin']>0 for r in pos]),
                specificity=100*mean([r['margin']<=0 for r in neg]),
                balanced_accuracy=50*(mean([r['margin']>0 for r in pos])+mean([r['margin']<=0 for r in neg])),
                answer_order_agreement=100*mean([r['answer_order_agreement'] for r in group]),
                answerable_scenes=len(answerable),
                top_candidate_accuracy=100*mean([max(g,key=lambda r:r['margin'])['positive'] for g in answerable]),
                mean_positive_margin=mean([r['margin'] for r in pos]),
                mean_negative_margin=mean([r['margin'] for r in neg]),
                unanswerable_scenes=len(unanswerable),
                reject_all_unanswerable=100*mean([all(r['margin']<=0 for r in g) for g in unanswerable])))
    paired=[]
    lookup={(r['index'],r['target'],r['method'],r['condition']):r for r in rows}
    for method in METHODS:
        single=[r for r in rows if r['method']==method and r['condition']=='single']
        pairs=[(r,lookup[r['index'],r['target'],method,'multi_target']) for r in single]
        paired.append(dict(method=method,
            correct_single_wrong_multi=sum(((a['margin']>0)==a['positive']) and ((b['margin']>0)!=b['positive']) for a,b in pairs),
            wrong_single_correct_multi=sum(((a['margin']>0)!=a['positive']) and ((b['margin']>0)==b['positive']) for a,b in pairs)))
    payload=dict(completed=len(raw),expected=3528,exit_codes=codes,results=results,paired=paired,
                 note='Counterbalanced mean Yes-minus-No logit margin, zero threshold; diagnostic, not official benchmark accuracy.')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs,logs=[],[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_candidate_isolation_audit',str(shard)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:
        log.close()
    summarize(codes)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
