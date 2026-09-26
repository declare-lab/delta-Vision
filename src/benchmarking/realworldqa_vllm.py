"""Batch-one, fixed-eight-token RealWorldQA latency: project graphs versus vLLM."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/diagnostics/vllm_realworldqa_20260924'


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def execution_timer(model):
    """Measure worker-side model calls; scheduler, IPC and sampling stay outside."""
    import torch
    model._execution_events = []
    original_forward, original_logits = model.forward, model.compute_logits
    def interval(kind, fn, *args, **kwargs):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        model._execution_events.append((kind, start, end))
        return result
    def forward(*args, **kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None: x = kwargs.get('input_ids')
        model._execution_phase = 'prefill' if x.shape[0] > 1 else 'decode'
        return interval(model._execution_phase, original_forward, *args, **kwargs)
    def logits(*args, **kwargs):
        return interval(model._execution_phase+'_head', original_logits, *args, **kwargs)
    model.forward, model.compute_logits = forward, logits
    def before(*args):
        model._vision_start_event = torch.cuda.Event(enable_timing=True)
        model._vision_start_event.record()
    def after(*args):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        model._execution_events.append(('vision', model._vision_start_event, end))
    model.visual.register_forward_pre_hook(before)
    model.visual.register_forward_hook(after)
    return True


def take_execution_times(model):
    import torch
    torch.cuda.synchronize()
    values = {}
    for kind, start, end in model._execution_events:
        values.setdefault(kind, []).append(start.elapsed_time(end)/1000.)
    model._execution_events.clear()
    return values


def disable_deepstack(model):
    model.config.vision_config.deepstack_visual_indexes = []
    model.visual.deepstack_visual_indexes = []
    model.visual.out_hidden_size = model.config.vision_config.out_hidden_size
    model.use_deepstack = False
    model.deepstack_num_level = model.multiscale_dim = 0
    return True


class QwenGraphs:
    def __init__(self, model, adapter):
        from src.attention import optimize_qwen_attention_metadata
        from src.benchmarking.common.prefill import build_qwen_fast_adapter_prefill, QwenContextCudaGraphRunner
        from src.model import build_qwen_initial_context, qwen_position_ids, qwen_visual_grid_metadata
        from src.graphs import NativeDecoderGraphs, QwenWholePrefillGraphs
        from src.kernels import FusedQwenNorms, QwenExactRoPE, QwenFusedProjections
        self.model, self.adapter = model, adapter
        optimize_qwen_attention_metadata(model)
        self.context = QwenContextCudaGraphRunner(model=model, build_qwen_initial_context=build_qwen_initial_context,
            qwen_position_ids=qwen_position_ids, qwen_visual_grid_metadata=qwen_visual_grid_metadata,
            graph_warmup=3, verify=True, max_diff=0.)
        if adapter is not None:
            fast = build_qwen_fast_adapter_prefill(model, adapter, SimpleNamespace(
                last_logits_only=True, attn_implementation='flash_attention_2', cuda_graph=True,
                cuda_graph_context=False, compile_verify=True, compile_max_diff=0., cuda_graph_warmup=3,
                adapter_decode_cache_mode='fast', adapter_exact_optimizations=True, adapter_max_optimizations=True))
            self.prefill_graph = fast.graph_runners[1]
            self.decode_graph = model._adapter_decode_graph_runner
        else:
            self.optimizers = [FusedQwenNorms(model), QwenExactRoPE(model), QwenFusedProjections(model)]
            self.decode_graph = NativeDecoderGraphs(model, max_shapes=8, vision=False, prefill_layers=False, packed_kv=True)
            self.prefill_graph = QwenWholePrefillGraphs(model, max_shapes=1)

    def prepare(self, inputs):
        import torch
        self.inputs = inputs
        self.topology = torch.stack((inputs['attention_mask'][0], inputs['mm_token_type_ids'][0])).tolist()
        self.model.model.rope_deltas = None
        self.build_context()
        self.positions4 = self.model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
        self.next_positions = self.positions4[:, :, -1:] + torch.arange(1, 8, device='cuda')

    def build_context(self):
        x = self.inputs
        self.hidden, self.positions = self.context(x['input_ids'], x['attention_mask'], x['pixel_values'],
            x['image_grid_thw'], x['mm_token_type_ids'])

    def capture(self, enabled):
        self.decode_graph.allow_capture = enabled
        if self.adapter is None:
            self.prefill_graph.allow_capture = enabled

    def stats(self):
        d = self.decode_graph.stats()
        if self.adapter is not None:
            return (d['captures'], d['cold_fallbacks'], id(self.prefill_graph.graph))
        p = self.prefill_graph.stats()
        return (d['captures'], d['cold_layer_fallbacks'], p['captures'], p['fallbacks'])

    def prefill(self):
        x = self.inputs
        if self.adapter is not None:
            logits, _, cache = self.prefill_graph(x['input_ids'], x['attention_mask'], x['mm_token_type_ids'],
                self.hidden, self.positions, topology=self.topology)
            return logits, cache
        out = self.model(inputs_embeds=self.hidden, attention_mask=x['attention_mask'],
            position_ids=self.positions4, use_cache=True, logits_to_keep=1, return_dict=True)
        return out.logits, out.past_key_values

    def step(self, token, cache, i):
        if self.adapter is not None:
            return self.decode_graph(self.model, self.adapter, token, cache, logits_to_keep=1)
        out = self.model(input_ids=token, position_ids=self.next_positions[:, :, i:i+1],
            past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        return out.logits, out.past_key_values


def project_request(graphs, eos, check=False):
    import torch
    def measured(fn):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = fn()
        end.record()
        return result, (start, end)
    _, vision_events = measured(graphs.build_context)
    (logits, cache), prefill_events = measured(graphs.prefill)
    tokens, hashes, decode_events = [], [], []
    for i in range(8):
        if check: hashes.append(logits.detach().clone())
        # Sampling is excluded identically in both backends.
        scores = logits[:, -1].float().clone()
        scores[:, eos] = -float('inf')
        token = scores.argmax(-1).view(1, 1)
        tokens.append(token)
        if i != 7:
            (logits, cache), events = measured(lambda: graphs.step(token, cache, i))
            decode_events.append(events)
    torch.cuda.synchronize()
    seconds = lambda events: events[0].elapsed_time(events[1])/1000.
    vision, prefill = seconds(vision_events), seconds(prefill_events)
    decode = sum(seconds(events) for events in decode_events)
    return dict(vision_s=vision, prefill_s=prefill, decode_s=decode, total_s=prefill+decode,
        including_vision_s=vision+prefill+decode, tokens=torch.cat(tokens, 1)[0].tolist()), hashes


def worker(a):
    import torch
    from PIL import Image
    from src.benchmarks import build_benchmark_prompt, get_benchmark_spec
    torch.set_num_threads(4)
    torch.manual_seed(44)
    cfg = json.loads((SOURCE/'config.json').read_text())
    family = cfg['models'][a.family]
    rows = read(SOURCE/'data.jsonl')
    dest = Path(a.run)/f'{a.family}.{a.backend}.{a.method}.{a.shard}.jsonl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {r['index'] for r in read(dest)} if dest.exists() else set()
    device = torch.device('cuda:0')
    if a.backend == 'project':
        if a.family == 'qwen':
            from src.model_setup import load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint, disable_qwen_deepstack
            processor, model = load_frozen_qwen3vl(family['base_model'], torch.bfloat16, device, 'flash_attention_2')
            disable_qwen_deepstack(model)
            adapter = None
            if a.method == 'adapter':
                adapter, meta = load_qwen_embedding_adapter_checkpoint(family['source_checkpoint'], model.model.language_model, device, torch.bfloat16)
                assert not meta['missing'] and not meta['unexpected']
            graphs = QwenGraphs(model, adapter)
        else:
            from src.qwen35 import load_model
            processor, model, adapter, controller = load_model(dict(model_path=family['base_model'], rank=128), device)
            if a.method == 'adapter':
                saved = torch.load(family['source_checkpoint'], map_location='cpu', weights_only=False)
                adapter.load_state_dict(saved['state_dict'], strict=True)
            adapter.eval()
            from src.benchmarking.qwen35_graphs import Qwen35Graphs
            graphs = Qwen35Graphs(model, adapter if a.method == 'adapter' else None, controller)
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
    else:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        model_path = family['export'] if a.method == 'adapter' else family['base_model']
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        from src.vllm_graphs import install_runtime_graphs, graph_stats, capture_on, capture_off
        # Explicit shape-specific prefill/decode graphs share vLLM's live paged
        # caches. Keep torch.compile off; graph capture is handled by this runtime.
        llm = LLM(model=model_path, dtype='bfloat16', tensor_parallel_size=1,
            enforce_eager=False, compilation_config={'mode':0,'cudagraph_mode':'NONE'},
            enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, max_model_len=16384, max_num_batched_tokens=16384,
            max_num_seqs=1, gpu_memory_utilization=.35,
            kv_cache_memory_bytes=(4 if a.family=='qwen' else 2)*1024**3,
            skip_mm_profiling=True, mamba_ssm_cache_dtype='float32',
            limit_mm_per_prompt={'image':1,'video':0}, seed=44, generation_config='vllm',
            attention_config={'backend':'FLASH_ATTN','flash_attn_version':2}, gdn_prefill_backend='flashinfer')
        llm.apply_model(disable_deepstack)
        llm.apply_model(install_runtime_graphs)
        llm.apply_model(execution_timer)
        # Suppress the same EOS as project; fixed 8 output tokens, 7 cache steps.
        from transformers import AutoConfig, GenerationConfig
        eos = GenerationConfig.from_model_config(AutoConfig.from_pretrained(model_path)).eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        sampling = SamplingParams(temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True,
            logit_bias={int(i):-100. for i in eos}, repetition_penalty=1.)
        request_id = 0
        def vrequest(prompt, image):
            nonlocal request_id
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            request_id += 1
            # The engine still schedules the real request, but timing is collected
            # inside its worker around model execution only.
            engine = llm.llm_engine
            prepared = engine.input_processor.process_inputs(str(request_id),
                dict(prompt=prompt, multi_modal_data={'image':image}), sampling,
                supported_tasks=engine.get_supported_tasks())
            outputs, steps = [], 0
            engine.add_request(str(request_id), prepared, sampling)
            while engine.has_unfinished_requests():
                outputs.extend(engine.step())
                steps += 1
            result = outputs[-1]
            assert len(result.outputs[0].token_ids) == 8 and steps == 8, (steps, result)
            measured = llm.apply_model(take_execution_times)[0]
            expected_counts = dict(vision=1,prefill=1,prefill_head=1,decode=7,decode_head=7)
            assert {k:len(v) for k,v in measured.items()} == expected_counts, measured
            vision = sum(measured['vision'])
            prefill = sum(measured['prefill']) + sum(measured['prefill_head'])
            decode = sum(measured['decode']) + sum(measured['decode_head'])
            return dict(vision_s=vision, prefill_s=prefill, decode_s=decode,
                total_s=prefill+decode, including_vision_s=vision+prefill+decode,
                execution_times_s=measured, tokens=list(result.outputs[0].token_ids),
                prompt_ids=result.prompt_token_ids)
    spec = get_benchmark_spec('realworldqa')
    expected_rows = read(SOURCE/f'{a.family}.hf.{a.method}.jsonl')
    indices = list(range(a.shard, min(a.limit or len(rows), len(rows)), a.shards))
    with torch.inference_mode(), dest.open('a', buffering=1) as handle:
        for n, index in enumerate(indices):
            if index in done:
                continue
            row = rows[index]
            content = [dict(type='image'), dict(type='text', text=build_benchmark_prompt(row, spec))]
            prompt = processor.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                add_generation_prompt=True, **({'enable_thinking':False} if a.family=='qwen35' else {}))
            with Image.open(Path(cfg['image_root'])/row['image']) as im:
                image = im.convert('RGB').copy()
            if a.backend == 'project':
                inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(device)
                prompt_ids = inputs['input_ids'][0].tolist()
                graphs.prepare(inputs)
                graphs.capture(True)
                checked, logits = project_request(graphs, eos, check=True)
                warm, _ = project_request(graphs, eos)
                assert warm['tokens'] == checked['tokens']
                graphs.capture(False)
                before = graphs.stats()
                trials = [project_request(graphs, eos)[0] for _ in range(a.repeats)]
                assert before == graphs.stats(), 'Timed graph capture/fallback'
                if a.family == 'qwen35':
                    graphs.verify(logits, checked['tokens'], eos)
                del logits
            else:
                llm.apply_model(capture_on)
                warm = vrequest(prompt, image)
                before = llm.apply_model(capture_off)[0]
                trials = [vrequest(prompt, image) for _ in range(a.repeats)]
                after = llm.apply_model(graph_stats)[0]
                assert before['captures'] == after['captures'] and before['head_captures'] == after['head_captures']
                assert after['replays']['prefill'] - before['replays']['prefill'] == a.repeats
                assert after['replays']['decode'] - before['replays']['decode'] == 7*a.repeats
                assert after['head_replays'] - before['head_replays'] == 8*a.repeats
                prompt_ids = trials[0].pop('prompt_ids')
                for t in trials:
                    if 'prompt_ids' in t:
                        assert t.pop('prompt_ids') == prompt_ids
            assert all(t['tokens'] == warm['tokens'] for t in trials)
            expected = expected_rows[index]
            digest = hashlib.sha256(json.dumps(prompt_ids, separators=(',',':')).encode()).hexdigest()
            assert digest == expected['input_ids_sha256'], (index, 'Prompt mismatch')
            result = dict(index=index, family=a.family, backend=a.backend, method=a.method,
                shard=a.shard, input_ids_sha256=digest, prompt_tokens=len(prompt_ids), trials=trials,
                timing_scope='worker_model_cuda_events_no_scheduler_no_sampling',
                cuda_graph=True, fixed_output_tokens=8, decode_steps=7, timed_captures=0,
                graph_stats=after if a.backend=='vllm' else None)
            handle.write(json.dumps(result)+'\n')
            print(json.dumps(dict(index=index, done=n+1, total=len(indices),
                prefill_ms=statistics.median(t['prefill_s'] for t in trials)*1000,
                decode_ms=statistics.median(t['decode_s'] for t in trials)*1000)), flush=True)
            image.close()
            if a.family == 'qwen35' and a.backend == 'project':
                graphs.release()
                gc.collect()


def launch(a):
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    (run/'protocol.json').write_text(json.dumps(dict(samples=a.limit or 765, seed=44,
        input_manifest=str(SOURCE/'data.jsonl'), models=json.loads((SOURCE/'config.json').read_text())['models'],
        batch_size=1, generated_tokens=8, decode_steps=7, repeats=a.repeats, deepstack=False,
        project='Qwen3-VL previous maximum optimizations and prefill/decode CUDA graphs; Qwen3.5 verified hybrid CUDA graphs',
        vllm='Native vLLM FA2/GDN kernels, shape-specific prefill/decode/head CUDA graphs, no HF-exact arithmetic overrides; no prefix/encoder cache, TP1',
        timing='Warmed CUDA Events around model forward and LM head. Scheduler, IPC, CPU image processing, sampling and capture excluded in both backends. reported total=vision+prefill+decode; decoder-only total is also retained.',
        scheduling='One method across eight GPUs at a time, same sample assignment for every method',
        qwen3_optimizations=['fused Q/K RMSNorm + MRoPE', 'single-pass graph input inspection',
            'cached visual/text position indices', 'batched static visual K/V projection',
            'native FA2 split heuristic (no forced unsplit)'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/benchmarking/realworldqa_vllm.py', ROOT/'src/benchmarking/qwen35_graphs.py', ROOT/'src/vllm_adapter.py', ROOT/'src/vllm_graphs.py', ROOT/'src/kernels.py')}), indent=2)+'\n')
    for family in a.families:
        for backend in a.backends:
            for method in a.methods:
                processes = []
                for gpu in range(a.shards):
                    python = '.venv/bin/python' if backend=='project' else 'artifacts/dependencies/vllm023/bin/python'
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_PLUGINS='delta_vision', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                        PYTHONPATH=('artifacts/dependencies/vllm_hf_kernels:' if backend=='vllm' else '')+
                            ('artifacts/dependencies/qwen35_python:' if family=='qwen35' else '')+'.')
                    command = [str((ROOT/python).absolute()), '-m', 'src.benchmarking.realworldqa_vllm', 'worker',
                        '--run', str(run), '--family', family, '--backend', backend, '--method', method,
                        '--shard', str(gpu), '--shards', str(a.shards), '--repeats', str(a.repeats)]
                    if a.limit: command += ['--limit', str(a.limit)]
                    log = (run/f'{family}.{backend}.{method}.{gpu}.log').open('a')
                    processes.append(subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
                    log.close()
                codes = [p.wait() for p in processes]
                if any(codes): raise RuntimeError((family, backend, method, codes))
                print('COMPLETED', family, backend, method, flush=True)
    report(a)


def report(a):
    run = Path(a.run); summary = []
    for family in a.families:
        for method in a.methods:
            for backend in a.backends:
                rows = [r for p in run.glob(f'{family}.{backend}.{method}.*.jsonl') for r in read(p)]
                if not rows: continue
                assert len({r['index'] for r in rows}) == len(rows)
                scopes = {r.get('timing_scope') for r in rows}
                assert scopes == {'worker_model_cuda_events_no_scheduler_no_sampling'}, scopes
                result = dict(family=family, method=method, backend=backend, samples=len(rows),
                    complete=len(rows)==(a.limit or 765), cuda_graph=True,
                    timing_scope='worker_model_cuda_events_no_scheduler_no_sampling')
                for key in ('vision_s','prefill_s','decode_s'):
                    result[key.replace('_s','_mean_ms')] = statistics.mean(statistics.median(t[key] for t in r['trials']) for r in rows)*1000
                result['decoder_only_mean_ms'] = result['prefill_mean_ms'] + result['decode_mean_ms']
                result['total_mean_ms'] = result['vision_mean_ms'] + result['decoder_only_mean_ms']
                result['including_vision_mean_ms'] = result['total_mean_ms']
                result['total_includes_vision'] = True
                result['decode_ms_per_token'] = result['decode_mean_ms']/7
                summary.append(result)
    (run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (run/'summary_with_vision.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['worker','launch','report'])
    p.add_argument('--run', required=True)
    p.add_argument('--family', choices=['qwen','qwen35'])
    p.add_argument('--backend', choices=['project','vllm'])
    p.add_argument('--method', choices=['base','adapter'])
    p.add_argument('--families', nargs='+', default=['qwen','qwen35'])
    p.add_argument('--backends', nargs='+', default=['project','vllm'])
    p.add_argument('--methods', nargs='+', default=['base','adapter'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--limit', type=int)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    globals()[args.action](args)
