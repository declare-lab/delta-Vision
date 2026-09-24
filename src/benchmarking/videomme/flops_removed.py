"""Video-MME999 no-visual-token LLM FLOPs, aligned to archived resource inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'artifacts/diagnostics/video_adapter_layer_ablation_999_20260923'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def read(paths):
    result={}
    for p in paths:
        for line in p.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):continue
            r=json.loads(line);assert r['index'] not in result;result[r['index']]=r
    return result


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    for d in ['source/src','source/scripts','logs','rows','audits']:(run/d).mkdir(parents=True)
    old=json.loads((REFERENCE/'protocol.json').read_text())
    c={k:old[k] for k in ['model','manifest','manifest_sha256']}
    c.update(reference=str(REFERENCE),original_root=str(ROOT),samples=999,
        num_frames=8,video_sampling='full_timestamp_v1',deepstack=False,attention='flash_attention_2',dtype='bfloat16',
        tokens=8,decode_forwards=7,flops_scope='LLM+LM head only, no visual encoder',
        intervention='Drop all nonzero mm_token_type_ids before first LLM layer; preserve original text/template/timestamps/M-RoPE',
        accuracy_run=str(ROOT/'artifacts/eval/qwen4b_no_visual_tokens_20260923'),
        counter='Same executed dense matrix/conv/FA2 convention as reference;2 FLOPs per MAC; scalar ops excluded')
    assert sha(c['manifest'])==c['manifest_sha256']
    for p in (list((ROOT/'src').glob('*.py')) + list((ROOT/'analysis').rglob('*.py')) + list((ROOT/'src/benchmarking').rglob('*.py')) + list((ROOT/'src/training').rglob('*.py')))+[Path(__file__).resolve()]:
        dest=run/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(run/'config.json',c)
    dump(run/'source_hashes.json',{str(p.relative_to(run/'source')):sha(p) for p in (run/'source').rglob('*.py')})


def worker(run,shard,smoke):
    sys.path.insert(0,str(run/'source'))
    import torch
    from src.model import load_frozen_qwen3vl,build_qwen_initial_context
    from src.data import QwenBenchmarkDataset
    from src.benchmarking.common.resource_flops import matrix_flop_counter, counts
    from src.benchmarking.engines.adapter import tensor_sha
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root'])
    for p,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha(run/'source'/p)==h
    os.environ.update(QWEN_VIDEO_NUM_FRAMES='8',QWEN_VIDEO_SAMPLING='full_timestamp_v1')
    torch.set_num_threads(2);torch.manual_seed(42);device=torch.device('cuda:0')
    processor,model=load_frozen_qwen3vl(c['model'],torch.bfloat16,device,'flash_attention_2')
    assert model.model.visual.deepstack_visual_indexes==[]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    lm=model.model.language_model
    ds=QwenBenchmarkDataset(c['manifest'],processor,'videomme',data_root=str(root/'data/benchmarks/videomme'),
        cache_dir=str(root/'test/results/adapter_exact_20260915/inputs'))
    assert len(ds)==999 and sha(c['manifest'])==c['manifest_sha256']
    base=read((Path(c['reference'])/'base').glob('flops_*.jsonl'))
    inside=[False];vision=[0]
    def hook(*unused):
        assert not inside[0],'Visual encoder in FLOP-counted region'
        vision[0]+=1
    model.model.visual.register_forward_pre_hook(hook)

    def counted(hidden,pos):
        inside[0]=True
        try:
            with matrix_flop_counter() as pc:
                o=lm(inputs_embeds=hidden,position_ids=pos,attention_mask=None,use_cache=True)
                logits=model.lm_head(o.last_hidden_state[:,-1:])
            cache=o.past_key_values;lengths=[l.keys.shape[-2] for l in cache.layers]
            assert lengths==[hidden.shape[1]]*36
            first_logits=logits.clone();pre=int(pc.get_total_flops());dec=0;ops={};tokens=[]
            assert any('flash_attn' in k for k in counts(pc))
            for step in range(8):
                scores=logits[:,-1].float().clone();scores[:,eos]=-float('inf')
                token=scores.argmax(-1).view(1,1);tokens.append(int(token))
                if step==7:break
                with matrix_flop_counter() as dc:
                    o=lm(input_ids=token,position_ids=pos[:,:,-1:]+step+1,attention_mask=None,
                        past_key_values=cache,use_cache=True)
                    logits=model.lm_head(o.last_hidden_state[:,-1:])
                dec+=int(dc.get_total_flops())
                for k,v in counts(dc).items():ops[k]=ops.get(k,0)+v
            assert any('flash_attn' in k for k in ops)
            return dict(prefill_matrix_flops=pre,decode_matrix_flops=dec,request_matrix_flops=pre+dec,
                vision_matrix_flops=0,prefill_ops=counts(pc),decode_ops=ops,tokens=tokens,layer_lengths=lengths),first_logits
        finally:inside[0]=False

    tag='smoke' if smoke else f'shard{shard}'
    path=run/'rows'/f'{tag}.jsonl';prior=read([path]) if path.exists() else {}
    indices=[0] if smoke else list(range(shard,999,8))
    started=time.time();validations=[]
    with torch.inference_mode(),path.open('a',buffering=1) as out:
        for index in indices:
            if index in prior:continue
            item=ds[index]
            inputs={k:item[k].unsqueeze(0).to(device) for k in ['input_ids','attention_mask','mm_token_type_ids']}
            for k in ['pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw']:
                if torch.is_tensor(item.get(k)):inputs[k]=item[k].to(device)
            digest=tensor_sha([inputs[k] for k in sorted(inputs)])
            assert digest==base[index]['input_sha256'],(index,'Historical video inputs differ')
            model.model.rope_deltas=None;before=vision[0]
            hidden,pos=build_qwen_initial_context(model,inputs)
            assert vision[0]-before==1
            keep=inputs['mm_token_type_ids'][0].eq(0)
            assert int(keep.sum())==base[index]['text_tokens']
            assert int((~keep).sum())==base[index]['visual_tokens']
            if index==indices[0]:
                native,_=counted(hidden,pos)
                assert native['request_matrix_flops']==base[index]['request_matrix_flops']-base[index]['vision_matrix_flops']
                assert native['tokens']==base[index]['tokens'],(index,'Native control tokens changed')
            text_hidden=hidden[:,keep].contiguous();text_pos=pos[:,:,keep].contiguous()
            assert torch.equal(text_hidden,model.get_input_embeddings()(inputs['input_ids'][:,keep]))
            r,logits=counted(text_hidden,text_pos)
            if index==indices[0]:
                direct=lm(input_ids=inputs['input_ids'][:,keep],position_ids=text_pos,attention_mask=None,use_cache=True)
                torch.testing.assert_close(logits,model.lm_head(direct.last_hidden_state[:,-1:]),rtol=0,atol=0)
                validations.append(dict(index=index,native_flops_exact=True,native_tokens_exact=True,text_only_logits_exact=True))
                del direct
            r.update(index=index,input_sha256=digest,visual_tokens=int((~keep).sum()),text_tokens=int(keep.sum()),
                method='no_visual_tokens',encoder_in_counted_region=False)
            out.write(json.dumps(r)+'\n');prior[index]=r
            dump(run/f'progress_{tag}.json',dict(completed=len(prior),index=index,elapsed_s=time.time()-started))
            if smoke or len(prior)%25==0:print(tag,index,len(prior),round(time.time()-started),flush=True)
            del hidden,pos,text_hidden,text_pos,inputs,logits
    dump(run/'audits'/f'{tag}.json',dict(passed=True,completed=len(prior),validations=validations))


def report(run):
    c=json.loads((run/'config.json').read_text());fresh=read((run/'rows').glob('shard*.jsonl'))
    assert set(fresh)==set(range(999))
    result=[]
    for name in ['base','adapter','no_visual_tokens']:
        rr=fresh if name=='no_visual_tokens' else read((Path(c['reference'])/name).glob('flops_*.jsonl'))
        assert set(rr)==set(fresh)
        for i,r in rr.items():assert r['input_sha256']==fresh[i]['input_sha256']
        row=dict(method=name,samples=999,source='fresh' if name=='no_visual_tokens' else c['reference'])
        row['prefill_T']=sum(r['prefill_matrix_flops']-r['vision_matrix_flops'] for r in rr.values())/999/1e12
        row['decode_T']=sum(r['decode_matrix_flops'] for r in rr.values())/999/1e12
        row['total_T']=row['prefill_T']+row['decode_T'];result.append(row)
    for r in result:r['pct_base']=100*r['total_T']/result[0]['total_T']
    dump(run/'RESULTS.json',dict(complete=True,samples=999,exclude_vision=True,results=result))
    lines=['# Video-MME999 LLM-only FLOPs','',
        'Same999 inputs verified by full tensor hashes;8 frames, full_timestamp_v1;FA2/BF16;DeepStack off;fixed8 generated tokens/EOS suppressed. Encoder runs outside counting. Includes LLM and LM head;dense matrix/FA2 convention,2 FLOPs/MAC;scalar ops excluded.',
        'Base/adapter are archived matched-input resource results; no-visual is fresh. Each shard verifies native tokens and LLM FLOPs exactly against archived control. Every layer in the ablation has text-only KV. Nine image benchmarks remain the accuracy experiment.',
        '', '| Method | Prefill FLOPs(T) | Decode FLOPs(T) | Total FLOPs(T) | % Base |','|---|---:|---:|---:|---:|']
    for r in result:lines.append(f'| {r["method"]} | {r["prefill_T"]:.5f} | {r["decode_T"]:.5f} | {r["total_T"]:.5f} | {r["pct_base"]:.2f}% |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def queue(run):
    c=json.loads((run/'config.json').read_text());root=Path(c['original_root']);active=[];start=time.time()
    try:
        for smoke in [True,False]:
            active=[]
            for shard in ([0] if smoke else range(8)):
                tag='smoke' if smoke else f'shard{shard}'
                if (run/'audits'/f'{tag}.json').exists():continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
                cmd=[str(root/'.venv/bin/python'),'-u',str(run/'source/src/benchmarking/videomme/flops_removed.py'),'worker','--run',str(run),'--shard',str(shard)]
                if smoke:cmd+=['--smoke']
                log=(run/'logs'/f'{tag}.log').open('a');p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);active.append((p,log))
            while any(p.poll() is None for p,_ in active):
                if any(p.poll() not in (None,0) for p,_ in active):raise RuntimeError('Worker failure')
                done=sum(json.loads(p.read_text())['completed'] for p in run.glob('progress_shard*.json'))
                dump(run/'status.json',dict(state='validation' if smoke else 'running',completed=done,expected=999,elapsed_s=time.time()-start))
                time.sleep(10)
            assert all(p.returncode==0 for p,_ in active)
            for _,log in active:log.close()
        report(run);dump(run/'status.json',dict(state='complete',completed=999,expected=999,elapsed_s=time.time()-start))
    except BaseException as e:
        for p,_ in active:
            if p.poll() is None:p.terminate()
        for p,_ in active:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        dump(run/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','queue','report']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.run=a.run.resolve()
    if a.action=='worker':worker(a.run,a.shard,a.smoke)
    else:globals()[a.action](a.run)
