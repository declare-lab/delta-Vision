"""Move the queried candidate first, not the gold image; keep all distractors.

Paired with prior binary probes. Full reads (Base + Adapter) and target-only
visual reads (Adapter). Changes presentation position and reference together,
so it is NOT a pure RoPE intervention. Never changes official benchmark inputs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/muir_target_front_20260914'
PREVIOUS=ROOT/'artifacts/diagnostics/muir_candidate_isolation_20260914'


def worker(mode,shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src import muir_candidate_isolation_audit as audit
    audit.OUT=OUT/mode
    audit.CONDITIONS=('multi_target',)
    audit.METHODS=('base','embedding_adapter') if mode=='full' else ('embedding_adapter',)
    audit.CLEAR_VISION_PER_TARGET=True
    original_row=audit.probe_row
    state={}
    def make_row(row,target,condition,reverse):
        images=row['images']
        presented=[images[target]]+images[:target]+images[target+1:]
        assert len(presented)==len(images) and sorted(presented)==sorted(images)
        state.update(n_images=len(images))
        result=original_row(dict(row,images=presented),0,'multi_target',reverse)
        assert result['images'][0]==images[target]
        return result
    audit.probe_row=make_row
    checks=[]
    if mode=='masked':
        original_prepare=ref.prepare_qwen_embedding_adapter_inputs
        def prepare(*args,**kwargs):
            pack=original_prepare(*args,**kwargs)
            assert pack['image_mask'].shape[0]==1 and bool(pack['image_mask'].all())
            positions=pack['image_positions'][0]
            boundaries=[0]+(torch.where(positions[1:]-positions[:-1]!=1)[0]+1).tolist()+[len(positions)]
            assert len(boundaries)-1==state['n_images']
            end=boundaries[1]
            original=pack['prefix_attention_mask']
            assert original.dtype==torch.bool
            mask=original.clone()
            mask[...,end:len(positions)]=False
            assert torch.equal(mask[...,len(positions):],original[...,len(positions):])
            assert torch.equal(mask[...,:end],original[...,:end])
            assert int(mask[0,0,-1,:len(positions)].sum())==end
            pack['prefix_attention_mask']=mask
            checks.append(dict(visual_tokens=len(positions),target_visual_tokens=end))
            return pack
        ref.prepare_qwen_embedding_adapter_inputs=prepare
    audit.worker(shard)
    (OUT/mode/f'checks_{shard}.json').write_text(json.dumps(checks)+'\n')


def summarize(mode,codes):
    folder=OUT/mode
    raw=[json.loads(l) for p in folder.glob('rows_*.jsonl') for l in p.open()]
    expected=1764 if mode=='full' else 882
    assert not any(codes) and len(raw)==len({(r['method'],r['index'],r['target'],r['reverse']) for r in raw})==expected
    oldfolder=PREVIOUS if mode=='full' else ROOT/'artifacts/diagnostics/muir_target_read_mask_20260914'
    old={(r['method'],r['index'],r['target'],r['reverse']):r
         for p in oldfolder.glob('rows_*.jsonl') for l in p.open()
         if (r:=json.loads(l))['condition']=='multi_target'}
    unchanged=[r for r in raw if r['target']==0]
    for r in unchanged:
        prior=old[r['method'],r['index'],0,r['reverse']]
        assert r['margin']==prior['margin'] and r['greedy_token']==prior['greedy_token'],(mode,r['index'],r['method'],'No-op control failed')
    for r in raw:
        images=r['images'];target=r['target']
        assert r['presented_images']==[images[target]]+images[:target]+images[target+1:]
    lookup={(r['method'],r['index'],r['target'],r['reverse']):r for r in raw}
    rows=[dict(r,margin=(r['margin']+lookup[r['method'],r['index'],r['target'],True]['margin'])/2) for r in raw if not r['reverse']]
    mean=lambda xs:sum(xs)/len(xs)
    results=[]
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        pos=[r for r in group if r['positive']]
        neg=[r for r in group if not r['positive']]
        scenes=[[r for r in group if r['index']==i] for i in sorted({r['index'] for r in group})]
        answerable=[g for g in scenes if g[0]['gold_image'] is not None]
        unanswerable=[g for g in scenes if g[0]['gold_image'] is None]
        results.append(dict(method=method,mode=mode,candidates=len(group),
            sensitivity=100*mean([r['margin']>0 for r in pos]),
            specificity=100*mean([r['margin']<=0 for r in neg]),
            answerable_scenes=len(answerable),
            top_candidate_accuracy=100*mean([max(g,key=lambda r:r['margin'])['positive'] for g in answerable]),
            mean_positive_margin=mean([r['margin'] for r in pos]),
            mean_negative_margin=mean([r['margin'] for r in neg]),
            top_candidate_ties=sum(sum(r['margin']==max(x['margin'] for x in g) for r in g)>1 for g in answerable),
            unanswerable_scenes=len(unanswerable),
            reject_all_unanswerable=100*mean([all(r['margin']<=0 for r in g) for g in unanswerable])))
    payload=dict(expected=expected,completed=len(raw),exit_codes=codes,
        unchanged_first_image_controls=len(unchanged),results=results)
    (folder/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    for mode in ('full','masked'):
        (OUT/mode).mkdir()
        jobs,logs=[],[]
        for shard in range(8):
            log=(OUT/mode/f'worker{shard}.log').open('w')
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_target_front_audit',mode,str(shard)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in jobs]
        for log in logs:
            log.close()
        summarize(mode,codes)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(sys.argv[1],int(sys.argv[2]))
