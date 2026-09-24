"""Pinned three-benchmark delta correction experiment; fresh paired controls."""
import argparse
from contextlib import contextmanager
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
from analysis.fig05_hybrid_attention.qwen35_delta_cancellation import DeltaCancellation, relerr
from analysis.fig05_hybrid_attention.qwen35_no_visual_write import no_visual_write


@contextmanager
def capture(model,mask):
    result={'hidden':{}}
    handles=[]
    def layer_hook(i):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            if h.shape[1]==mask.shape[1]:
                result['hidden'][i]=h[:,~mask[0]].detach().float().clone()
        return hook
    for i,layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
    def model_hook(module,args,out):
        if 'logits' not in result:
            result['logits']=out.logits[:,-1].detach().float().clone()
    handles.append(model.register_forward_hook(model_hook))
    try:yield result
    finally:
        for h in handles:h.remove()


def compare(native,other):
    logp=native['logits'].log_softmax(-1)
    logq=other['logits'].log_softmax(-1)
    return dict(kl_to_native=(logp.exp()*(logp-logq)).sum().item(),
        logit_relative_error=relerr(other['logits'],native['logits']),
        text_hidden_relative_error={str(i):relerr(other['hidden'][i],native['hidden'][i]) for i in native['hidden']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--validate',action='store_true')
    args=p.parse_args();config=json.loads((args.run/'config.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(44)
    processor,model,adapter,controller=load_model(config,torch.device('cuda:0'))
    torch.set_float32_matmul_precision('highest')
    if args.validate:
        spec=importlib.util.spec_from_file_location('tests',ROOT/'test/diagnostics/test_qwen35_delta_cancellation.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        gate=mod.run_tests();print('KERNEL_TESTS',gate,flush=True)
    else:
        assert json.loads((args.run/'validation.json').read_text())['passed']
    stage='validation' if args.validate else 'results'
    outdir=args.run/stage;outdir.mkdir(parents=True,exist_ok=True)
    for benchmark,info in config['evaluation'].items():
        assert sha(info['path'])==info['sha256']
        rows=[json.loads(l) for l in Path(info['path']).read_text().splitlines()]
        assert len(rows)==info['samples']
        refs={r['index']:r for f in (Path(config['reference_run'])/'accuracy').glob(f'{benchmark}.shard*.jsonl') for r in map(json.loads,f.read_text().splitlines())}
        spec=get_benchmark_spec(benchmark)
        indices=[0] if args.validate else list(range(args.shard,len(rows),8))
        path=outdir/f'{benchmark}.shard{args.shard}.jsonl'
        old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        assert [r['index'] for r in old]==indices[:len(old)]
        with path.open('a') as f,torch.inference_mode():
            for index in indices[len(old):]:
                started=time.time();row=rows[index]
                inputs,_=prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),question=build_benchmark_prompt(row,spec))
                input_sha=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
                assert input_sha==refs[index]['input_ids_sha256']
                mask=inputs['mm_token_type_ids'].eq(1)
                def generate():
                    return generate_evaluation_answer(model,processor,inputs,row,spec,config,max_new_tokens=info['max_new_tokens'])
                observer=DeltaCancellation(model,mask,'observe')
                with observer.activate(),capture(model,mask) as base:
                    native=generate()
                expected=next(v for v in refs[index]['variants'] if v['method']=='native' and v['rank'] is None)
                assert native['generated_token_ids']==expected['generated_token_ids'],(benchmark,index,'Native token mismatch')
                variants={'native':native}
                modes=['replay_control','no_cancel','no_cancel_pure']
                if args.validate:modes+=['noop']
                for mode in modes:
                    tracker=DeltaCancellation(model,mask,mode)
                    with tracker.activate(),capture(model,mask) as observed:
                        answer=generate()
                    variants[mode]=dict(**answer,**compare(base,observed),
                        recurrent_state_relative_error={str(i):relerr(tracker.states[i],observer.states[i]) for i in tracker.states},
                        layer_checks=list(tracker.records.values()))
                    if mode=='noop':
                        assert answer['generated_token_ids']==native['generated_token_ids']
                        assert torch.equal(observed['logits'],base['logits'])
                    if mode=='no_cancel':changed_logits=observed['logits'].clone()
                    if mode=='no_cancel_pure':
                        variants[mode]['logit_relative_error_vs_paired_no_cancel']=relerr(observed['logits'],changed_logits)
                    del tracker,observed
                # Gate-only ablation retains native forgetting and convolution.
                with no_visual_write(model,mask),capture(model,mask) as observed:
                    answer=generate()
                variants['visual_beta_zero']=dict(**answer,**compare(base,observed))
                result=dict(index=index,benchmark=benchmark,input_ids_sha256=input_sha,
                    visual_start=observer.start,visual_end=observer.end,sequence_length=mask.shape[1],
                    native_generation_exact=True,layers=list(observer.records.values()),
                    variants=variants,elapsed_s=time.time()-started)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                print(json.dumps(dict(benchmark=benchmark,index=index,elapsed_s=result['elapsed_s'],
                    visual_c_mean=sum(r['visual']['cancellation_ratio'] for r in result['layers'])/24,
                    scores={m:v['score'] for m,v in variants.items()})),flush=True)
                del observer,inputs,base,observed,result,variants
    if args.validate:
        records=[json.loads(l) for p in outdir.glob('*.jsonl') for l in p.read_text().splitlines()]
        assert len(records)==3
        # Numerical controls are disclosed for every example in full evaluation;
        # require small logit KL in pilot before launching the expensive run.
        kl=max(r['variants']['replay_control']['kl_to_native'] for r in records)
        assert kl<.05,('Replay control KL too large',kl)
        (args.run/'validation.json').write_text(json.dumps(dict(passed=True,kernel=gate,
            native_generation_exact=True,noop_exact=True,max_replay_control_kl=kl),indent=2)+'\n')
    controller.close()


if __name__=='__main__':main()
