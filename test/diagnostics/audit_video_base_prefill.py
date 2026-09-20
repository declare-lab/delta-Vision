"""Audit saved base prefill boundaries, actual vision execution and fresh inputs.

This keeps the saved base optimizations. It does not enable adapter v2 kernels
or change the published 999-row results. Component events are a separate probe.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8',
                  HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false')
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.benchmark_adapter_optimizations import MODEL, MANIFEST, tensor_sha, file_sha
from src.data import QwenBenchmarkDataset
from src.qwen_attention_metadata import optimize_qwen_attention_metadata
from src.qwen_deepstack import disable_qwen_deepstack
from src.qwen_fused_norm import FusedQwenNorms
from src.qwen_exact_rope import QwenExactRoPE
from src.qwen_fused_projections import QwenFusedProjections
from src.qwen_native_graph import NativeDecoderGraphs
from src.qwen_native_prefill_graph import QwenWholePrefillGraphs

CACHE = ROOT/'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme'


def kv(cache):
    return [tensor for layer in cache.layers for tensor in (layer.keys, layer.values)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='test/results/video_base_20260915/prefill_boundary_audit')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 333, 666])
    parser.add_argument('--runs', type=int, default=7)
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, device, 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    metadata = optimize_qwen_attention_metadata(model)
    norms, rope, projections = FusedQwenNorms(model), QwenExactRoPE(model), QwenFusedProjections(model)
    graphs = NativeDecoderGraphs(model, max_shapes=8, prefill_layers=False)
    whole = QwenWholePrefillGraphs(model)
    data = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=str(CACHE))
    saved = {r['index']:r for p in (ROOT/'test/results/video_base_20260915/videomme999').glob('optimized_*.jsonl')
             for line in p.read_text().splitlines() if (r:=json.loads(line))}
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    active, events, handles = False, {}, []
    for module, label in [(model.model.visual, 'vision'), (model.model.language_model, 'language')]:
        def before(current, positional, kwargs, _label=label):
            if active:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                events.setdefault(_label, []).append((begin, end))
                begin.record()
        def after(current, positional, kwargs, result, _label=label):
            if active:
                events[_label][-1][1].record()
        handles += [module.register_forward_pre_hook(before, with_kwargs=True),
                    module.register_forward_hook(after, with_kwargs=True)]

    def enable(value, capture=False):
        for optimization in (metadata, norms, rope, projections):
            optimization.enabled = value
        graphs.enabled, graphs.allow_capture = value, capture
        whole.enabled, whole.allow_capture = value, capture

    def request(inputs, *, decode=True, split_boundary=True):
        model.model.rope_deltas = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        positions = model._prepare_position_ids_for_generation(inputs['input_ids'], dict(inputs))
        # This is the same synchronization as GenerationStageTimer._before in
        # the saved base benchmark. A separate trial omits it as a cross-check.
        if split_boundary:
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        result = model(**inputs, position_ids=positions, use_cache=True, logits_to_keep=1, return_dict=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        prefill_logits, prefill_cache = result.logits, result.past_key_values
        times = dict(request_prefill_ms=1000*(end-start),
                     position_preparation_ms=1000*(forward_start-start),
                     forward_prefill_ms=1000*(end-forward_start))
        tokens = []
        if decode:
            next_positions = positions[:, :, -1:] + torch.arange(1, 8, device=device)
            for step in range(8):
                scores = result.logits[:, -1].to(torch.float32, copy=True)
                scores[:, eos] = -float('inf')
                token = scores.argmax(-1).view(1, 1)
                tokens.append(token)
                if step < 7:
                    result = model(input_ids=token, position_ids=next_positions[:, :, step:step+1],
                        past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
            torch.cuda.synchronize()
        return times, prefill_logits, prefill_cache, result, tokens

    rows = []
    with torch.inference_mode():
        for index in args.indices:
            inputs = _qwen_inputs_from_item(data[index], device)
            assert tensor_sha([inputs[k] for k in sorted(inputs)]) == saved[index]['input_sha256']
            enable(False)
            _, ref_logits, ref_cache, _, _ = request(inputs, decode=False)
            ref_logits, ref_kv = ref_logits.clone(), [t.clone() for t in kv(ref_cache)]
            eager_times = [request(inputs, decode=False)[0]['request_prefill_ms'] for _ in range(3)]
            enable(True, True)
            _, actual_logits, _, result, tokens = request(inputs)
            assert torch.equal(actual_logits, ref_logits)
            assert torch.cat(tokens, 1)[0].tolist() == saved[index]['tokens']
            request(inputs)
            enable(True, False)
            _, actual_logits, actual_cache, _, _ = request(inputs, decode=False)
            assert torch.equal(actual_logits, ref_logits)
            assert all(torch.equal(a,b) for a,b in zip(kv(actual_cache), ref_kv))
            assert all(l.get_seq_length() == inputs['input_ids'].shape[1] for l in actual_cache.layers)
            initial = graphs.stats(), whole.stats()
            trials = [request(inputs)[0] for _ in range(args.runs)]
            no_middle_sync = [request(inputs, split_boundary=False)[0]['request_prefill_ms'] for _ in range(3)]
            final = graphs.stats(), whole.stats()
            assert initial[0]['captures'] == final[0]['captures']
            assert initial[0]['cold_layer_fallbacks'] == final[0]['cold_layer_fallbacks']
            assert initial[1]['captures'] == final[1]['captures'] and initial[1]['fallbacks'] == final[1]['fallbacks']
            component_trials = []
            for _ in range(3):
                active, events = True, {}
                request(inputs, decode=False)
                active = False
                assert len(events['vision']) == len(events['language']) == 1
                component_trials.append({key:pair[0][0].elapsed_time(pair[0][1]) for key,pair in events.items()})
            row = dict(index=index, duration=data[index]['row']['duration'],
                text_tokens=saved[index]['text_tokens'], visual_tokens=saved[index]['visual_tokens'],
                video_grid_thw=inputs['video_grid_thw'].tolist(), pixels_shape=list(inputs['pixel_values_videos'].shape),
                saved_forward_prefill_ms=1000*statistics.median(t['generation_prefill_time_s'] for t in saved[index]['trials']),
                median_ms={key:statistics.median(t[key] for t in trials) for key in trials[0]},
                uninstrumented_request_prefill_ms=statistics.median(no_middle_sync),
                eager_native_request_prefill_ms=statistics.median(eager_times),
                gpu_components_ms={key:statistics.median(t[key] for t in component_trials) for key in ['vision','language']},
                trials=trials, identical_native_logits_and_all_prefill_kv=True,
                identical_input_and_8_tokens_to_saved_base=True, timed_captures=0, timed_fallbacks=0)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del ref_cache, ref_kv, actual_cache, result

        # Change pixel values in the SAME tensor storage after graph warmup.
        # A replay must use the new content and own its output KV independently.
        index = args.indices[-1]
        inputs = _qwen_inputs_from_item(data[index], device)
        enable(True, False)
        _, old_logits, old_cache, _, _ = request(inputs, decode=False)
        old_hash, old_kv = tensor_sha([old_logits]), tensor_sha(kv(old_cache))
        pointer = inputs['pixel_values_videos'].data_ptr()
        inputs['pixel_values_videos'].mul_(0.9)
        _, changed_logits, changed_cache, _, _ = request(inputs, decode=False)
        assert inputs['pixel_values_videos'].data_ptr() == pointer
        assert tensor_sha([changed_logits]) != old_hash
        assert tensor_sha([old_logits]) == old_hash and tensor_sha(kv(old_cache)) == old_kv
        enable(False)
        _, eager_logits, eager_cache, _, _ = request(inputs, decode=False)
        assert torch.equal(changed_logits, eager_logits)
        assert all(torch.equal(a,b) for a,b in zip(kv(changed_cache), kv(eager_cache)))
    summary = dict(samples=len(rows), rows=rows, attention='flash_attention_2', deepstack='off',
        base_configuration='Saved base v1 exact kernels and whole-language/vision CUDA graphs; no adapter max optimizations',
        means_ms={key:statistics.mean(r['median_ms'][key] for r in rows) for key in rows[0]['median_ms']},
        mean_gpu_components_ms={key:statistics.mean(r['gpu_components_ms'][key] for r in rows) for key in ['vision','language']},
        same_storage_changed_frames_match_native_eager=True, previous_outputs_not_overwritten=True,
        fresh_vision_and_all_36_language_layers_verified=True,
        component_note='Separate CUDA-event diagnostics include queued copies/host launch gaps; not pure kernel time. Formal boundary trials have no active event hooks.',
        script_sha256=file_sha(Path(__file__)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
    for handle in handles: handle.remove()
    whole.remove(); graphs.remove(); projections.remove(); rope.remove(); norms.remove(); metadata.remove()


if __name__ == '__main__':
    main()
