"""Evaluate the document-continued static rank128 adapter on the frozen document protocol."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_document_benchmarks as suite


def configure(run):
    suite.OUT=run/'eval_documents'
    suite.METHODS=['embedding_adapter']
    suite.CHECKPOINTS={'embedding_adapter':run/'checkpoints/qwen_embedding_adapter_step2000.pt'}


def report(run):
    from src.benchmarks import get_benchmark_spec,score_prediction
    records=[r for p in (suite.OUT/'full').glob('*.jsonl') for r in suite.load_rows(p)]
    assert len(records)==3000
    assert {(r['benchmark'],r['index']) for r in records}=={(b,i) for b in suite.BENCHES for i in range(1000)}
    previous=ROOT/'artifacts/eval/document_benchmarks_1000_20260918'
    oldplan=json.loads((previous/'plan.json').read_text())
    plan=json.loads((suite.OUT/'plan.json').read_text())
    assert oldplan['data']==plan['data']
    for key in ['model','attention','dtype','deepstack','prompt','max_new_tokens','decoding','image_resolution','metrics']:
        assert oldplan[key]==plan[key],key
    old_records={(r['benchmark'],r['index']):r for p in (previous/'full').glob('embedding_adapter_*.jsonl') for r in suite.load_rows(p)}
    data={b:[json.loads(l) for l in suite.manifest(b).read_text().splitlines()] for b in suite.BENCHES}
    for r in records:
        row=data[r['benchmark']][r['index']]
        assert r['input_sha256']==old_records[(r['benchmark'],r['index'])]['input_sha256']
        assert (r['source_index'],r['question_id'])==(row['index'],row['question_id'])
        score=score_prediction(metric=get_benchmark_spec(r['benchmark']).metric,
            prediction_text=r['prediction_text'],answer=row['answer'],answers=row['answers'])['score']
        assert abs(score-r['score'])<1e-12
    results=json.loads((previous/'full_summary.json').read_text())['rows']
    results=[r for r in results if r['method'] in ['base','embedding_adapter','recurrent_adapter']]
    rows=[]
    names={'base':'Base model','embedding_adapter':'Embedding adapter, rank128','recurrent_adapter':'Recurrent adapter, rank128'}
    for r in results:
        rows.append(dict(Method=names[r['method']],**{b:r[b] for b in suite.BENCHES},AVG=r['avg']))
    row={'Method':'Embedding adapter, rank128, document continued'}
    for b in suite.BENCHES:
        selected=[r for r in records if r['benchmark']==b]
        assert len(selected)==1000
        row[b]=100*sum(r['score'] for r in selected)/1000
    row['AVG']=sum(row[b] for b in suite.BENCHES)/3;rows.append(row)
    with (run/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Document continued training of embedding rank128','',
        'Qwen3-VL-4B-Instruct; FA2; DeepStack off at evaluation. Same 1000 questions and identical processed inputs per benchmark. ChartQA relaxed accuracy, DocVQA and InfographicVQA ANLS, all x100; AVG is their unrounded arithmetic mean.',
        'Rank128 initialized from PixMo static KL step2000, then trained for 2000 additional steps on official document training splits, global batch32, lr5e-5. Teacher and student DeepStack off; only adapter projections trained. Data/image overlap audit is saved with the training manifest.',
        'Base/rank128 rows are verified existing results from document_benchmarks_1000_20260918; document-continued rank128 is newly evaluated.','',
        '| Method | ChartQA | DocVQA | InfographicVQA | AVG |','|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['Method'],*[f'{r[b]:.2f}' for b in suite.BENCHES],f"{r['AVG']:.2f}"])+' |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    suite.dump(suite.OUT/'scoring_audit.json',dict(predictions=3000,all_input_hashes_match_rank128=True,
        all_scores_recomputed_and_equal=True,checkpoint_sha256=suite.sha(suite.CHECKPOINTS['embedding_adapter']),
        samples_each=1000,empty=sum(r['invalid'] for r in records),at_generation_limit=sum(r['truncated'] for r in records)))
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--stage',choices=['smoke','full'],default='full');args=parser.parse_args()
    run=args.run_dir.resolve();configure(run)
    if args.verify_only:
        from scripts import verify_document_adapter_decode as verify
        verify.OUT=suite.OUT
        verify.CHECKPOINTS=suite.CHECKPOINTS
        verify.main()
        return
    if args.worker:
        suite.worker(argparse.Namespace(method='embedding_adapter',shard=args.shard,shards=8,stage=args.stage))
        return
    import torch
    torch.set_num_threads(4)
    ckpt=torch.load(suite.CHECKPOINTS['embedding_adapter'],map_location='cpu',weights_only=False)
    assert ckpt['global_step']==2000 and ckpt['adapter_config']['visual_adapter_rank']==128
    assert ckpt['adapter_config']['output_mode']=='embedding_adapter'
    assert len(ckpt['state_dict'])==72
    for name,tensor in ckpt['state_dict'].items():
        assert tuple(tensor.shape)==((128,2560) if 'down' in name else (2560,128))
        assert torch.isfinite(tensor).all(),name
    del ckpt
    suite.preflight()
    # Original suite is the evaluator; this wrapper only supplies the new checkpoint and output root.
    suite.dump(suite.OUT/'wrapper_provenance.json',dict(file=str(Path(__file__).resolve()),sha256=suite.sha(__file__)))
    jobs=[]
    for stage in ['smoke','full']:
        for shard in range(1 if stage=='smoke' else 8):
            logfile=(suite.OUT/'logs'/f'{stage}_{shard}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(run),'--worker','--stage',stage,'--shard',str(shard)]
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT)
            jobs.append((p,logfile))
        while any(p.poll() is None for p,_ in jobs):
            n=suite.aggregate(stage)
            suite.dump(suite.OUT/'status.json',dict(state='running',stage=stage,predictions=n,updated=time.time()))
            time.sleep(15)
        for p,logfile in jobs:
            logfile.close()
            assert p.returncode==0, f'{stage} evaluator failed with code {p.returncode}'
        jobs=[]
        suite.aggregate(stage,require_complete=True)
        if stage=='smoke':
            with (suite.OUT/'logs/decode_parity.log').open('w') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(run),
                    '--verify-only'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
    report(run)
    suite.dump(suite.OUT/'status.json',dict(state='complete',predictions=3000,finished=time.time()))


if __name__=='__main__':main()
