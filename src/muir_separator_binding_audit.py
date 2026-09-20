"""Diagnostic: localize visual reads of image-end markers, not answer queries.

Keep the existing checkpoint, pixels, input IDs, positions and all image K/V.
Only <vision_end> and the following explicit End-of-Image label's query rows
lose edges to other images. Text-to-text edges and question/answer rows stay
unchanged. This is an experimental routing intervention, NOT a formatting fix.
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
OLD=ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT=ROOT/'artifacts/diagnostics/muir_separator_binding_20260914'


def arguments(mode, shard=0):
    from src import multimodal_baseline_suite as suite
    return suite.make_parser().parse_args([
        mode,'--output',str(OUT),'--method','embedding_adapter',
        '--methods','embedding_adapter','--benchmarks','muirbench',
        '--manifest-map',str(OLD/'manifest_map.json'),'--limit','1000',
        '--shard',str(shard),'--shards','8','--max-new-tokens','8','--json-only'])


def worker(shard):
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from transformers import AutoTokenizer
    from src import model as ref, multimodal_baseline_suite as suite
    tokenizer=AutoTokenizer.from_pretrained(suite.MODEL)
    original=ref.prepare_qwen_embedding_adapter_inputs

    def prepare(model,adapter,input_ids,*args,**kwargs):
        pack=original(model,adapter,input_ids,*args,**kwargs)
        assert adapter.adapter_start_layer==0 and not adapter.active_adapter_layers
        vp=pack['image_positions'][0];tp=pack['text_positions'][0]
        cuts=[0]+(torch.where(vp[1:]-vp[:-1]!=1)[0]+1).tolist()+[len(vp)]
        ids=input_ids[0].tolist(); reverse={p:i for i,p in enumerate(tp.tolist())}
        old=pack['prefix_attention_mask'];mask=old.clone(); changed_rows=[]; audit=[]
        for image,(start,end) in enumerate(zip(cuts[:-1],cuts[1:]),1):
            # All gaps begin with the native image end delimiter. The last
            # gap includes the question, but ONLY the explicit label is patched.
            lo=int(vp[end-1])+1
            hi=int(vp[cuts[image]]) if image<len(cuts)-1 else len(ids)
            assert ids[lo]==tokenizer.convert_tokens_to_ids('<|vision_end|>')
            gap_ids=ids[lo:hi]
            text=tokenizer.decode(gap_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            assert encoded['input_ids']==gap_ids, 'Boundary token roundtrip mismatch'
            labels=list(re.finditer(r'\[End of Image (\d+)\]',text))
            assert len(labels)<=1, 'Ambiguous boundary label'
            queries=[lo]
            if labels:
                match=labels[0];assert int(match.group(1))==image,'Image label/order mismatch'
                queries += [lo+j for j,(a,b) in enumerate(encoded['offset_mapping'])
                            if a<match.end() and b>match.start()]
            qidx=[reverse[q] for q in queries]
            own=torch.zeros(len(vp),device=vp.device,dtype=torch.bool);own[start:end]=True
            before=mask[0,0,qidx,:len(vp)].clone()
            mask[0,0,qidx,:len(vp)]=before & own
            removed=int((before & ~own).sum())
            assert torch.equal(mask[0,0,qidx,start:end],old[0,0,qidx,start:end])
            changed_rows += qidx
            audit.append(dict(image=image,original_visual_start=int(vp[start]),
                              original_visual_end=int(vp[end-1]),visual_tokens=end-start,
                              separator_positions=queries,explicit_label=bool(labels),blocked_edges=removed))
        unchanged=torch.ones(len(tp),device=tp.device,dtype=torch.bool);unchanged[changed_rows]=False
        assert torch.equal(mask[:,:,unchanged],old[:,:,unchanged])
        assert torch.equal(mask[...,len(vp):],old[...,len(vp):])
        assert torch.equal(mask[:,:,-1],old[:,:,-1]), 'Answer query must still see all images'
        pack['prefix_attention_mask']=mask
        model.model._pruning_audit.append(dict(intervention='local_image_end_marker_reads',images=audit,
                                               answer_row_unchanged=True,text_edges_unchanged=True))
        return pack
    ref.prepare_qwen_embedding_adapter_inputs=prepare
    suite.worker(arguments('worker',shard))


def summarize():
    from src import multimodal_baseline_suite as suite
    suite.aggregate(arguments('aggregate'),complete=True)
    rows=[json.loads(l) for p in OUT.glob('embedding_adapter_shard*.jsonl') for l in p.open()]
    previous={r['index']:r for p in OLD.glob('embedding_adapter_shard*.jsonl')
              for l in p.open() if (r:=json.loads(l))['retention']==1.}
    assert len(rows)==len(previous)==1000
    for r in rows:
        for key in ('input_sha256','dataset_sha256','source_index','prompt_layout','max_new_tokens'):
            assert r[key]==previous[r['index']][key],(r['index'],key)
        assert len(r['audit'])==1 and r['audit'][0]['answer_row_unchanged']
    result=dict(n=1000,original_accuracy=sum(r['score'] for r in previous.values())/10,
                local_separator_accuracy=sum(r['score'] for r in rows)/10,
                improved=sum(r['score']>previous[r['index']]['score'] for r in rows),
                worsened=sum(r['score']<previous[r['index']]['score'] for r in rows),
                unchanged_input_hashes=True,deepstack=False,
                invalid=sum(not r['prediction'] for r in rows),
                tasks={t:dict(n=len(rs),original=100*sum(previous[r['index']]['score'] for r in rs)/len(rs),
                              patched=100*sum(r['score'] for r in rs)/len(rs))
                       for t in sorted({r['task'] for r in rows}) if (rs:=[r for r in rows if r['task']==t])})
    suite.dump(OUT/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


def run():
    from src import multimodal_baseline_suite as suite
    OUT.mkdir(parents=True,exist_ok=False)
    args=arguments('run')
    sources=[Path(__file__),Path(suite.__file__),suite.REFERENCE/'model.py',suite.REFERENCE/'data.py',
             suite.REFERENCE/'benchmarks.py',ROOT/'baselines/eval_baselines.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    suite.dump(OUT/'plan.json',dict(args=vars(args),source_sha256={str(p):sha(p) for p in sources},
        dataset_sha256={'muirbench':sha(OLD/'muirbench_random1000.jsonl')},
        adapter_checkpoints={'embedding_adapter':dict(path=str(suite.CHECKPOINT),sha256=sha(suite.CHECKPOINT))},
        deepstack_enabled=False,diagnostic='Only image-end marker text queries restricted to their own image; all text-key edges unchanged'))
    jobs={};logs=[];start=time.time()
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs[shard]=subprocess.Popen([sys.executable,'-m','src.muir_separator_binding_audit',str(shard)],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
                    stdout=log,stderr=subprocess.STDOUT)
    while any(p.poll() is None for p in jobs.values()):
        suite.aggregate(args)
        suite.dump(OUT/'status.json',dict(elapsed=time.time()-start,
                   workers={s:dict(pid=p.pid,exit_code=p.poll()) for s,p in jobs.items()}))
        time.sleep(10)
    for log in logs:log.close()
    codes=[p.returncode for p in jobs.values()]
    assert not any(codes),codes
    summarize()
    suite.dump(OUT/'status.json',dict(state='complete',elapsed=time.time()-start,exit_codes=codes))


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
