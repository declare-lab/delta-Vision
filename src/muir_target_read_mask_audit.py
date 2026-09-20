"""Diagnostic only: keep native positions/text, block non-target visual reads.

The target is the image explicitly requested by the binary question, never gold.
Uses the same 441 candidates and both Yes/No orders as the isolation experiment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/muir_target_read_mask_20260914'


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src import muir_candidate_isolation_audit as audit
    audit.OUT=OUT
    audit.METHODS=('embedding_adapter',)
    audit.CONDITIONS=('multi_target',)
    state={}
    original_row=audit.probe_row
    def make_row(row,target,condition,reverse):
        state.update(target=target,n_images=len(row['images']))
        return original_row(row,target,condition,reverse)
    audit.probe_row=make_row
    original_prepare=ref.prepare_qwen_embedding_adapter_inputs
    checks=[]
    def prepare(*args,**kwargs):
        pack=original_prepare(*args,**kwargs)
        assert pack['image_mask'].shape[0]==1 and bool(pack['image_mask'].all())
        positions=pack['image_positions'][0]
        boundaries=[0]+(torch.where(positions[1:]-positions[:-1]!=1)[0]+1).tolist()+[len(positions)]
        assert len(boundaries)-1==state['n_images']
        start,end=boundaries[state['target']:state['target']+2]
        original=pack['prefix_attention_mask']
        assert original.dtype==torch.bool
        mask=original.clone()
        mask[...,:start]=False
        mask[...,end:len(positions)]=False
        assert torch.equal(mask[...,len(positions):],original[...,len(positions):])
        assert torch.equal(mask[...,start:end],original[...,start:end])
        assert int(mask[0,0,-1,:len(positions)].sum())==end-start
        pack['prefix_attention_mask']=mask
        checks.append(dict(target=state['target'],visual_tokens=len(positions),
                           target_visual_tokens=end-start,other_visual_tokens_blocked=len(positions)-(end-start)))
        return pack
    ref.prepare_qwen_embedding_adapter_inputs=prepare
    audit.worker(shard)
    (OUT/f'mask_checks_{shard}.json').write_text(json.dumps(checks,indent=2)+'\n')


def summarize(codes):
    raw=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    assert not any(codes) and len(raw)==len({(r['index'],r['target'],r['reverse']) for r in raw})==882
    checks=[r for p in OUT.glob('mask_checks_*.json') for r in json.loads(p.read_text())]
    assert len(checks)==882 and all(r['other_visual_tokens_blocked']>0 for r in checks)
    lookup={(r['index'],r['target'],r['reverse']):r for r in raw}
    rows=[dict(r,margin=(r['margin']+lookup[r['index'],r['target'],True]['margin'])/2) for r in raw if not r['reverse']]
    pos=[r for r in rows if r['positive']]
    neg=[r for r in rows if not r['positive']]
    scenes=[[r for r in rows if r['index']==i] for i in sorted({r['index'] for r in rows})]
    answerable=[g for g in scenes if g[0]['gold_image'] is not None]
    unanswerable=[g for g in scenes if g[0]['gold_image'] is None]
    mean=lambda xs:sum(xs)/len(xs)
    payload=dict(expected=882,completed=len(raw),exit_codes=codes,candidates=len(rows),
        sensitivity=100*mean([r['margin']>0 for r in pos]),
        specificity=100*mean([r['margin']<=0 for r in neg]),
        balanced_accuracy=50*(mean([r['margin']>0 for r in pos])+mean([r['margin']<=0 for r in neg])),
        answerable_scenes=len(answerable),
        top_candidate_accuracy=100*mean([max(g,key=lambda r:r['margin'])['positive'] for g in answerable]),
        mean_positive_margin=mean([r['margin'] for r in pos]),
        mean_negative_margin=mean([r['margin'] for r in neg]),
        unanswerable_scenes=len(unanswerable),
        reject_all_unanswerable=100*mean([all(r['margin']<=0 for r in g) for g in unanswerable]),
        note='All 36 adapter layers: text queries cannot read non-target visual KV. Original pixels, positions, text and target KV retained. Diagnostic only.')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs,logs=[],[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_target_read_mask_audit',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:
        log.close()
    summarize(codes)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
