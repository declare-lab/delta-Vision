"""Actual vLLM GPU regression: reference, graph, repeat, batched, mixed requests."""
import argparse
import json
from pathlib import Path

import torch


def trace_logits(model):
    model._graph_test_logits = []
    original = model.compute_logits
    def wrapped(*args,**kwargs):
        result = original(*args,**kwargs)
        model._graph_test_logits.append(result.detach().float().cpu())
        forced = getattr(model, '_graph_test_forced_tokens', None)
        if forced is not None:
            token = forced[len(model._graph_test_logits)-1]
            result = torch.full_like(result, -float('inf'))
            result[:, token] = 0.
        return result
    model.compute_logits = wrapped
    return True


def reference_on(model):
    model._runtime_graphs.enabled = False
    model._graph_test_logits.clear()
    return True


def graphs_on(model):
    model._runtime_graphs.enabled = True
    model._runtime_graphs.allow_capture = True
    model._graph_test_logits.clear()
    return True


def main(a):
    from PIL import Image
    from transformers import AutoProcessor, AutoConfig, GenerationConfig
    from vllm import LLM, SamplingParams
    from src.benchmarks import build_benchmark_prompt, get_benchmark_spec
    from src.benchmarking.realworldqa_vllm import SOURCE, read, disable_deepstack
    from src.vllm_graphs import install_runtime_graphs, graph_stats
    cfg = json.loads((SOURCE/'config.json').read_text())
    family = cfg['models'][a.family]
    path = family['export'] if a.method=='adapter' else family['base_model']
    processor = AutoProcessor.from_pretrained(path)
    requests = []
    for i,row in enumerate(read(SOURCE/'data.jsonl')[:a.samples]):
        content = [dict(type='image'),dict(type='text',text=build_benchmark_prompt(row,get_benchmark_spec('realworldqa')))]
        if i==1: content.insert(0,dict(type='text',text='Look carefully at this image.'))
        prompt = processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,
            add_generation_prompt=True,**({'enable_thinking':False} if a.family=='qwen35' else {}))
        with Image.open(Path(cfg['image_root'])/row['image']) as im: image=im.convert('RGB').copy()
        requests.append(dict(prompt=prompt,multi_modal_data={'image':image}))
    llm = LLM(model=path,dtype='bfloat16',tensor_parallel_size=1,
        enforce_eager=False,compilation_config={'mode':0,'cudagraph_mode':'NONE'},
        enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
        max_model_len=16384,max_num_batched_tokens=16384,max_num_seqs=2,
        kv_cache_memory_bytes=(4 if a.family=='qwen' else 2)*1024**3,
        gpu_memory_utilization=.35,skip_mm_profiling=True,mamba_ssm_cache_dtype='float32',
        limit_mm_per_prompt={'image':1,'video':0},seed=44,generation_config='vllm',
        attention_config={'backend':'FLASH_ATTN','flash_attn_version':2},gdn_prefill_backend='flashinfer',
        hf_overrides={'delta_vision_hf_reference':a.hf_reference})
    llm.apply_model(disable_deepstack)
    llm.apply_model(install_runtime_graphs)
    llm.apply_model(trace_logits)
    eos = GenerationConfig.from_model_config(AutoConfig.from_pretrained(path)).eos_token_id
    eos = [eos] if isinstance(eos,int) else eos
    sampling = SamplingParams(temperature=0,max_tokens=8,min_tokens=8,ignore_eos=True,
        logit_bias={int(i):-100. for i in eos})
    output = Path(a.output); output.mkdir(parents=True,exist_ok=True)
    def run(requests, enabled, name, forced=None):
        llm.reset_mm_cache(); llm.llm_engine.reset_encoder_cache()
        llm.apply_model(graphs_on if enabled else reference_on)
        def set_forced(model): model._graph_test_forced_tokens = forced
        llm.apply_model(set_forced)
        results = llm.generate(requests,sampling,use_tqdm=False)
        dest = str((output/f'{name}.pt').absolute())
        def save(model):
            torch.save(model._graph_test_logits,dest)
            return True
        llm.apply_model(save)
        return results, torch.load(dest,map_location='cpu',weights_only=True)
    # BF16 kernel/layout changes need functional agreement, not bitwise parity.
    # Graph repeatability/cache isolation remain exact checks.
    report = []
    for i,request in enumerate(requests):
        eager,e = run([request],False,f'{i}.reference')
        graph,g = run([request],True,f'{i}.graph')
        repeated,r = run([request],True,f'{i}.repeat')
        assert len(e)==len(g)==len(r)==8
        assert eager[0].prompt_token_ids==graph[0].prompt_token_ids
        same = eager[0].outputs[0].token_ids==graph[0].outputs[0].token_ids==repeated[0].outputs[0].token_ids
        compared = g
        if getattr(a, 'allow_rounding_differences', False):
            _, compared = run([request],True,f'{i}.forced',forced=list(eager[0].outputs[0].token_ids))
        metrics=[]
        for aa,bb,cc,free in zip(e,compared,r,g):
            la,lb = aa.double().log_softmax(-1),bb.double().log_softmax(-1)
            metrics.append(dict(max_abs=float((aa-bb).abs().max()),
                kl=float((la.exp()*(la-lb)).sum(-1).max()),
                cosine=float(torch.nn.functional.cosine_similarity(aa,bb,dim=-1).min()),
                repeat_exact=torch.equal(free,cc)))
        row=dict(index=i,tokens_equal=same,max_kl=max(m['kl'] for m in metrics),
            min_cosine=min(m['cosine'] for m in metrics),repeat_exact=all(m['repeat_exact'] for m in metrics),steps=metrics)
        report.append(row)
        (output/'single_results.json').write_text(json.dumps(report,indent=2)+'\n')
        assert (same or getattr(a, 'allow_rounding_differences', False)) and row['max_kl']<.1 and row['min_cosine']>.995 and row['repeat_exact'],row
        print('SINGLE_VALID',row['index'],row['max_kl'],flush=True)
    assert sum(r['max_kl'] for r in report)/len(report)<.01, 'Mean KL gate'
    eager,_ = run(requests[:2],False,'batch.reference')
    graph,_ = run(requests[:2],True,'batch.graph')
    batch_match = [x.outputs[0].token_ids for x in eager]==[x.outputs[0].token_ids for x in graph]
    batch_repeat,_ = run(requests[:2],True,'batch.repeat')
    assert [x.outputs[0].token_ids for x in graph]==[x.outputs[0].token_ids for x in batch_repeat], 'Batched cache leak'
    assert batch_match or getattr(a, 'allow_rounding_differences', False),'Batched request mismatch'
    def mixed(enabled):
        llm.reset_mm_cache();llm.llm_engine.reset_encoder_cache()
        llm.apply_model(graphs_on if enabled else reference_on)
        llm.enqueue([requests[0]],sampling,use_tqdm=False)
        outputs=llm.llm_engine.step()
        llm.enqueue([requests[1]],sampling,use_tqdm=False)
        while llm.llm_engine.has_unfinished_requests(): outputs.extend(llm.llm_engine.step())
        final=[o for o in outputs if o.finished]
        return {tuple(o.prompt_token_ids):list(o.outputs[0].token_ids) for o in final}
    mixed_reference, mixed_graph = mixed(False), mixed(True)
    mixed_match = mixed_reference==mixed_graph
    assert len(mixed_graph)==2 and mixed_graph==mixed(True), 'Mixed cache leak'
    assert mixed_match or getattr(a, 'allow_rounding_differences', False),'Mixed decode/prefill mismatch'
    status=dict(family=a.family,method=a.method,samples=len(report),steps=8*len(report),
        passed=True,batch_match=batch_match,mixed_match=mixed_match,
        max_kl=max(r['max_kl'] for r in report),min_cosine=min(r['min_cosine'] for r in report),
        graph_stats=llm.apply_model(graph_stats)[0])
    (output/'validation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',required=True,choices=['qwen','qwen35'])
    p.add_argument('--method',required=True,choices=['base','adapter'])
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--hf-reference',action='store_true')
    p.add_argument('--allow-rounding-differences',action='store_true')
    main(p.parse_args())
