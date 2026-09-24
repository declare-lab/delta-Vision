"""Same-GPU, equal-length requests with genuine growing-cache decode (FA2)."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import os
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill
from src.data import QwenBenchmarkDataset
from src.benchmarking.common.generation_timing import GenerationStageTimer
from src.model import load_qwen_embedding_adapter_checkpoint, qwen_embedding_adapter_decode_step
from src.graphs import NativeDecoderGraphs, clone_tree

MODEL = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
CHECKPOINT = ROOT / "artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt"


def sync():
    torch.cuda.synchronize()


def assert_cache(a, b):
    if isinstance(a, dict) and '_native_cache' in a:
        a = a['_native_cache']
    if isinstance(b, dict) and '_native_cache' in b:
        b = b['_native_cache']
    if hasattr(a, 'layers') != hasattr(b, 'layers'):
        if hasattr(a, 'layers'):
            a, b = b, a
        assert len(a['layers']) == len(b.layers)
        for old, new in zip(a['layers'], b.layers):
            assert torch.equal(torch.cat([old['visual_key'], old['text_key']], dim=2), new.keys)
            assert torch.equal(torch.cat([old['visual_value'], old['text_value']], dim=2), new.values)
        return
    if hasattr(a, "layers"):
        assert len(a.layers) == len(b.layers)
        pairs = [(x.keys,y.keys) for x,y in zip(a.layers,b.layers)] + [(x.values,y.values) for x,y in zip(a.layers,b.layers)]
    else:
        pairs = [(x[key],y[key]) for x,y in zip(a["layers"],b["layers"]) for key in x]
    assert all(torch.equal(x,y) for x,y in pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["embedding_adapter","fastv","dart","divprune","zoo","sparsevlm","visionzip"])
    parser.add_argument("--samples", type=int, nargs="+", default=[0,25,125,133])
    parser.add_argument("--retentions", type=float, nargs="+", default=[.05,.2])
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--deepstack", choices=("off", "native"), default="off")
    parser.add_argument("--screenshot-fastv", action='store_true', help='Use the archived screenshot selection forward for FastV only.')
    parser.add_argument("--output", default="test/results/runtime_optimized_20260915/paired")
    args = parser.parse_args()
    assert args.tokens >= 2
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    occupants = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip().splitlines()
    occupants = [int(p.strip()) for p in occupants if p.strip().isdigit() and int(p.strip()) != os.getpid()]
    (out / "protocol.json").write_text(json.dumps(dict(vars(args), attention="flash_attention_2", shared_gpu=bool(occupants), other_gpu_processes_at_start=occupants,
        base="native FA2 vision/prefill-layer graphs plus whole native decode forward graphs",
        adapter="context/prefill graphs, no saved prefix activations, genuine q_len=1 decode graphs",
        stop="Exactly N tokens, EOS suppressed for ALL methods. This is an equal-work speed experiment; not the historical MMStar quality/stop protocol.",
        timing="Wall-clock model forwards synchronized on both sides. Capture/checks outside timing; alternate request order; all slow trials retained."), indent=2))
    base, processor = load_baseline_model("base",MODEL,torch.bfloat16,device,1.,"flash_attention_2")
    from src.model_setup import disable_qwen_deepstack
    if args.deepstack == "off":
        disable_qwen_deepstack(base)
    base_graphs = NativeDecoderGraphs(base,max_shapes=16)
    base_timer = GenerationStageTimer(base)
    path = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(path),processor,"mmstar",data_root=str(path.parent),max_samples=200)
    summaries = []

    def native_request(model, timer, inputs, capture=False):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model.model.rope_deltas = None
        sync()
        timer.begin()
        start = time.perf_counter()
        result = model.generate(**inputs,max_new_tokens=args.tokens,min_new_tokens=args.tokens,do_sample=False,
            return_dict_in_generate=True,output_logits=capture)
        sync()
        elapsed = time.perf_counter() - start
        measured = timer.finish(elapsed,args.tokens)
        measured["total_time_s"] = elapsed
        measured["tokens"] = result.sequences[0,inputs["input_ids"].shape[-1]:].tolist()
        assert len(measured["tokens"]) == args.tokens
        assert measured["decode_steps"] == args.tokens - 1
        return measured,result

    with torch.inference_mode():
        for method in args.methods:
            is_adapter = method == "embedding_adapter"
            model, _ = load_baseline_model("base" if is_adapter else method,MODEL,torch.bfloat16,device,.05,"flash_attention_2")
            if args.deepstack == "off":
                disable_qwen_deepstack(model)
            if method == 'fastv' and args.screenshot_fastv:
                from reproduce_screenshot_fastv import restore_screenshot_forward
                restore_screenshot_forward(model)
            graph = timer = adapter = prefill = eager_prefill = None
            if is_adapter:
                adapter,_ = load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT),model.model.language_model,device,torch.bfloat16)
                settings = SimpleNamespace(last_logits_only=True,attn_implementation="flash_attention_2",cuda_graph=False,
                    cuda_graph_context=True,compile_verify=True,compile_max_diff=0.,cuda_graph_warmup=3,adapter_decode_cache_mode="fast")
                eager_prefill = build_qwen_fast_adapter_prefill(model,adapter,settings)
                settings.cuda_graph = True
                prefill = build_qwen_fast_adapter_prefill(model,adapter,settings)
                graph = model._adapter_decode_graph_runner
            else:
                graph = NativeDecoderGraphs(model,max_shapes=16)
                timer = GenerationStageTimer(model)

            def adapter_request(inputs,capture=False,eager=False):
                sync()
                start = time.perf_counter()
                payload = (eager_prefill if eager else prefill)(inputs)
                sync()
                prefill_s = time.perf_counter() - start
                logits,mask,_,_,cache = payload
                times,tokens,all_logits = [],[],[]
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos,int) else eos
                for step in range(args.tokens):
                    if capture:
                        all_logits.append(logits.clone())
                    scores = logits[:, -1].to(dtype=torch.float32,copy=True)
                    scores[:,eos] = -float("inf")
                    token = scores.argmax(-1).view(1,1)
                    tokens.append(int(token))
                    if step + 1 == args.tokens:
                        break
                    sync()
                    begin = time.perf_counter()
                    logits,cache = (qwen_embedding_adapter_decode_step if eager else graph)(model,adapter,token,cache,logits_to_keep=1)
                    sync()
                    times.append(time.perf_counter() - begin)
                sync()
                total = time.perf_counter() - start
                return dict(total_time_s=total,generation_prefill_time_s=prefill_s,decode_time_s=sum(times),decode_steps=len(times),
                    generation_overhead_s=total-prefill_s-sum(times),tokens=tokens,decode_step_times_s=times), (all_logits,cache)

            for retention in ([None] if is_adapter else args.retentions):
                trials=[]
                for index in args.samples:
                    inputs=_qwen_inputs_from_item(dataset[index],device)
                    visual=inputs["mm_token_type_ids"][0].nonzero().flatten()
                    if not is_adapter:
                        configure_baseline(model,method,retention,int(visual[0]),len(visual))
                    base_graphs.enabled=False
                    base_ref,ref_result=native_request(base,base_timer,inputs,True)
                    base_graphs.enabled=base_graphs.allow_capture=True
                    base_check,check_result=native_request(base,base_timer,inputs,True)
                    assert base_ref["tokens"]==base_check["tokens"]
                    assert all(torch.equal(a,b) for a,b in zip(ref_result.logits,check_result.logits))
                    assert_cache(ref_result.past_key_values,check_result.past_key_values)
                    base_graphs.allow_capture=False
                    del ref_result,check_result
                    graph.enabled=False
                    if is_adapter:
                        reference,ref_result=adapter_request(inputs,True,True)
                    else:
                        reference,ref_result=native_request(model,timer,inputs,True)
                    graph.enabled=graph.allow_capture=True
                    candidate,check_result=adapter_request(inputs,True) if is_adapter else native_request(model,timer,inputs,True)
                    assert reference["tokens"]==candidate["tokens"],(method,index,"tokens")
                    a,b=(ref_result[0],check_result[0]) if is_adapter else (ref_result.logits,check_result.logits)
                    assert len(a)==len(b)==args.tokens
                    assert all(torch.equal(x,y) for x,y in zip(a,b)),(method,index,"logits")
                    assert_cache(ref_result[1] if is_adapter else ref_result.past_key_values,check_result[1] if is_adapter else check_result.past_key_values)
                    graph.allow_capture=False
                    del ref_result,check_result,a,b
                    before=(base_graphs.stats(),graph.stats())
                    for repetition in range(args.pairs):
                        pair=dict(index=index,repetition=repetition,exact_vs_eager=True)
                        for label in (["base","method"] if repetition%2==0 else ["method","base"]):
                            if label=="base":
                                pair[label],_=native_request(base,base_timer,inputs)
                                expected=base_ref["tokens"]
                            else:
                                pair[label],_=adapter_request(inputs) if is_adapter else native_request(model,timer,inputs)
                                expected=reference["tokens"]
                            assert pair[label]["tokens"]==expected
                        trials.append(pair)
                    for previous,current in zip(before,[base_graphs.stats(),graph.stats()]):
                        assert previous["captures"]==current["captures"]
                        key="cold_fallbacks" if "cold_fallbacks" in previous else "cold_layer_fallbacks"
                        assert previous[key]==current[key]
                    filename=f"{method}_ret{retention}.json"
                    (out/filename).write_text(json.dumps(trials,indent=2))
                    print(method,retention,index,"checked and timed",flush=True)
                summary=dict(method=method,retention=retention,pairs=len(trials),samples=len(args.samples),tokens_per_request=args.tokens)
                for key,name in [("total_time_s","total"),("generation_prefill_time_s","prefill"),("decode_time_s","decode")]:
                    ratios=[t["base"][key]/t["method"][key] for t in trials]
                    summary[name+"_paired_speedup"]=statistics.median(ratios)
                    summary[name+"_faster_pairs"]=sum(r>1 for r in ratios)
                    for label in ["base","method"]:
                        divisor=args.tokens-1 if name=="decode" else 1
                        summary[label+"_"+name+"_median_ms"]=statistics.median(t[label][key]*1000/divisor for t in trials)
                summaries.append(summary)
                (out/"summary.json").write_text(json.dumps(summaries,indent=2))
                print(json.dumps(summary),flush=True)
            if timer is not None:
                timer.remove()
                graph.remove()
            else:
                if hasattr(graph, "remove"):
                    graph.remove()
                model._adapter_decode_graph_runner=None
            del model,adapter,graph,timer,prefill,eager_prefill,adapter_request,_
            gc.collect()
            torch.cuda.empty_cache()
    base_graphs.remove()
    base_timer.remove()


if __name__ == "__main__":
    main()
