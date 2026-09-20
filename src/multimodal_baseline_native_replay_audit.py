"""Independent native-decoder replay of actual pruning decisions.

This diagnoses execution, not the quality or upstream fidelity of a selector.
VisionZip also changes embeddings: its replay validates the decoder after the
merge, not the contextual-merging algorithm. No benchmark score is changed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BF16 = os.environ.get('BASELINE_REPLAY_BF16') == '1'
EXPLICIT_MASK = os.environ.get('BASELINE_REPLAY_NATIVE_CAUSAL') != '1'
OUT = ROOT / ('artifacts/diagnostics/baseline_native_replay_bf16_20260914' if BF16 else
              'artifacts/diagnostics/baseline_native_replay_20260914')
if not EXPLICIT_MASK:
    OUT = OUT.with_name(OUT.name + '_native_causal')
METHODS = ('fastv', 'dart', 'visionzip', 'sparsevlm', 'divprune', 'zoo')
MANIFESTS = {
    'muirbench': ROOT / 'artifacts/diagnostics/muir_random1000_seed42_matched_20260914/muirbench_random1000.jsonl',
    'videomme': ROOT / 'artifacts/diagnostics/video_balanced_base_adapter_20260914/videomme_selected.jsonl',
    'mvbench': ROOT / 'artifacts/diagnostics/video_balanced_base_adapter_20260914/mvbench_selected.jsonl',
}
INDICES = {'muirbench': (0, 666), 'videomme': (0, 666), 'mvbench': (0, 900)}


def worker(method):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    import torch.nn.functional as F
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from baselines import multimodal_pruning_utils as pu
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['QWEN_VIDEO_SAMPLING'] = 'full_timestamp_v1'
    os.environ['QWEN_VIDEO_NUM_FRAMES'] = '8'
    dtype = torch.bfloat16 if BF16 else torch.float32
    native, processor = load_baseline_model('base', MODEL, dtype, 'cuda:0', 1., 'sdpa')
    port, _ = load_baseline_model(method, MODEL, dtype, 'cuda:0', .2, 'sdpa')
    for model in (native, port):
        model.eval().requires_grad_(False)
        model.model.language_model.register_forward_pre_hook(
            lambda m, a, k: (a, dict(k, deepstack_visual_embeds=None)), with_kwargs=True)
        def reject(*a, **k):
            raise AssertionError('DeepStack executed')
        model.model.language_model._deepstack_process = reject

    # Observe kernel arguments only inside language attention, not vision ViT.
    flags = {'text': False, 'calls': []}
    real_sdpa = F.scaled_dot_product_attention
    def checked_sdpa(q, k, v, *args, **kwargs):
        if flags['text']:
            mask = kwargs.get('attn_mask', args[0] if args else None)
            causal = kwargs.get('is_causal', args[2] if len(args) > 2 else False)
            assert q.shape[-2] == k.shape[-2], 'Audit uses uncached prefill'
            n = q.shape[-2]
            if mask is None:
                assert causal or n == 1, 'Noncausal language attention'
            else:
                future = torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1)
                values = mask.expand(1, 1, n, n)[0, 0][future]
                assert (not values.any()) if mask.dtype == torch.bool else (values < -1e10).all()
            flags['calls'].append({'length': n, 'causal': bool(causal), 'explicit_mask': mask is not None})
        return real_sdpa(q, k, v, *args, **kwargs)
    F.scaled_dot_product_attention = checked_sdpa
    for layer in port.model.language_model.layers:
        layer.self_attn.register_forward_pre_hook(lambda m, a: flags.update(text=True))
        layer.self_attn.register_forward_hook(lambda m, a, o: flags.update(text=False))

    captured = {}
    def capture_input(tag):
        def hook(m, a, k):
            captured[tag] = {n: k[n].clone() for n in ('inputs_embeds', 'position_ids')}
        return hook
    native.model.language_model.register_forward_pre_hook(capture_input('native'), with_kwargs=True)
    port.model.language_model.register_forward_pre_hook(capture_input('port'), with_kwargs=True)
    events = []
    original_audit = pu.audit_prune
    def audit(m, n, visual, selected, layer):
        original_audit(m, n, visual, selected, layer)
        # Independently construct retained indices, without keep_visual_subset.
        keep = torch.ones(n, dtype=torch.bool, device=visual.device)
        keep[visual] = False
        keep[selected] = True
        events.append(dict(layer=int(layer), keep=keep.nonzero().flatten(),
                           visual=visual.clone(), selected=selected.clone()))
    pu.audit_prune = audit
    observed_layers = []
    def observe_layer(layer):
        def hook(m, a, k):
            h = a[0] if a else k['hidden_states']
            observed_layers.append((layer, h.shape[1], tuple(x.clone() for x in k['position_embeddings'])))
        return hook
    for layer, module in enumerate(port.model.language_model.layers):
        module.register_forward_pre_hook(observe_layer(layer), with_kwargs=True)

    with torch.inference_mode(), (OUT / f'{method}.jsonl').open('w', buffering=1) as output:
        for bench, manifest in MANIFESTS.items():
            ds = QwenBenchmarkDataset(str(manifest), processor, bench,
                data_root=str(ROOT / 'data/benchmarks' / bench), max_samples=1000,
                cache_dir=OUT / 'processed' / bench, prompt_layout='media_first_v1')
            for index in INDICES[bench]:
                item = ds[index]
                inputs = _qwen_inputs_from_item(item, torch.device('cuda:0'))
                assert inputs['attention_mask'].all(), 'Unpadded single-request audit'
                visual = inputs['mm_token_type_ids'][0].ne(0)
                n = visual.numel()
                # No target/answer tokens are passed to the model.
                assert 'labels' not in inputs
                native.model.rope_deltas = None
                base = native(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                native_input = captured['native']
                for retention in (1., .2, .05):
                    events.clear(); observed_layers.clear(); flags['calls'].clear()
                    torch.manual_seed(42000 + index)
                    configure_baseline(port, method, retention, int(visual.nonzero()[0]), int(visual.sum()))
                    port.model.rope_deltas = None
                    candidate = port(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    assert len(observed_layers) == len(flags['calls']) == 36
                    mapping = torch.arange(n, device='cuda')
                    per_layer = {}
                    event_stats = []
                    for event in events:
                        keep = event['keep']
                        before = mapping
                        mapping = mapping[keep]
                        assert torch.equal(mapping[~visual[mapping]], (~visual).nonzero().flatten())
                        event_stats.append(dict(layer=event['layer'], before=len(before), after=len(mapping),
                            visual_retained=int(visual[mapping].sum()), original_positions=mapping.cpu().tolist()))
                        per_layer.setdefault(event['layer'], []).append(keep)
                    assert int(visual[mapping].sum()) == pu.visual_budget(int(visual.sum()), retention)
                    assert len(mapping) == observed_layers[-1][1]
                    port_input = captured['port']
                    native_pos = native_input['position_ids']
                    # VisionZip prunes/merges before the language model. Others
                    # must present the exact native image embeddings and M-RoPE.
                    if method == 'visionzip':
                        assert torch.equal(port_input['position_ids'], native_pos[..., mapping])
                        torch.testing.assert_close(port_input['inputs_embeds'][:, ~visual[mapping]],
                            native_input['inputs_embeds'][:, ~visual], rtol=0, atol=0)
                        replay_input = port_input
                        start_mapping = mapping.clone()
                    else:
                        assert torch.equal(port_input['position_ids'], native_pos)
                        torch.testing.assert_close(port_input['inputs_embeds'], native_input['inputs_embeds'], rtol=1e-5, atol=1e-5)
                        replay_input = native_input
                        start_mapping = torch.arange(n, device='cuda')
                    state = {'mapping': start_mapping, 'mask': None, 'length': -1}
                    native_cos = native.model.language_model.rotary_emb(
                        native_input['inputs_embeds'], native_pos[1:] if native_pos.shape[0] == 4 else native_pos)
                    def replay_hook(layer):
                        def hook(m, a, k):
                            h = a[0] if a else k['hidden_states']
                            if method != 'visionzip':
                                for keep in per_layer.get(layer, []):
                                    h = h[:, keep]
                                    state['mapping'] = state['mapping'][keep]
                            ids = state['mapping']
                            cos = tuple(x[:, ids] for x in native_cos)
                            got_layer, length, port_cos = observed_layers[layer]
                            assert got_layer == layer and length == h.shape[1] == len(ids)
                            for expected, actual in zip(cos, port_cos):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            if state['length'] != len(ids) and EXPLICIT_MASK:
                                state['mask'] = torch.ones(len(ids), len(ids), device='cuda', dtype=torch.bool).tril()[None, None]
                                state['length'] = len(ids)
                            k = dict(k, attention_mask=state['mask'], position_embeddings=cos,
                                     position_ids=native_pos[0, :, ids] if native_pos.shape[0] == 4 else None)
                            return ((h,) + a[1:], k) if a else (a, dict(k, hidden_states=h))
                        return hook
                    handles = [m.register_forward_pre_hook(replay_hook(l), with_kwargs=True)
                               for l, m in enumerate(native.model.language_model.layers)]
                    try:
                        h = native.model.language_model(**replay_input, use_cache=False).last_hidden_state[:, -1:]
                        replay = native.lm_head(h)[0, -1].float()
                    finally:
                        for handle in handles: handle.remove()
                    delta = float((candidate - replay).abs().max())
                    kl = float((replay.softmax(-1) * (replay.log_softmax(-1) - candidate.log_softmax(-1))).sum())
                    base_delta = float((candidate - base).abs().max()) if retention == 1 else None
                    row = dict(method=method, benchmark=bench, index=index, retention=retention,
                        original_tokens=n, original_visual=int(visual.sum()), retained_visual=int(visual[mapping].sum()),
                        native_replay_max_logit_error=delta, native_replay_kl=kl,
                        native_replay_argmax_equal=int(candidate.argmax()) == int(replay.argmax()),
                        no_pruning_vs_base_max_error=base_delta,
                        numerical_pass=delta < .002 and kl < 1e-6 and (base_delta is None or base_delta < .002),
                        all_text_preserved=True, original_mrope_exact=True, causal_kernel_checked=True,
                        layer_lengths=[x[1] for x in observed_layers], sdpa_calls=flags['calls'], events=event_stats,
                        replay_scope='post_merge_decoder' if method == 'visionzip' else 'independent_native_pruning',
                        explicit_replay_mask=EXPLICIT_MASK,
                        dtype=str(dtype), deepstack=False)
                    output.write(json.dumps(row) + '\n')
                    print(method, bench, index, retention, delta, kl, row['numerical_pass'], flush=True)
                captured.clear(); observed_layers.clear(); events.clear()


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = []; logs = []
    for gpu, method in enumerate(METHODS):
        log = (OUT / f'{method}.log').open('w'); logs.append(log)
        jobs.append(subprocess.Popen([sys.executable, '-m', 'src.multimodal_baseline_native_replay_audit', method],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4'),
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    for log in logs: log.close()
    rows = [json.loads(line) for p in OUT.glob('*.jsonl') for line in p.open()]
    summary = dict(exit_codes=dict(zip(METHODS, codes)), completed=len(rows), expected=108,
                   dtype='bfloat16' if BF16 else 'float32',
                   explicit_replay_mask=EXPLICIT_MASK,
                   argmax_differences=sum(not r['native_replay_argmax_equal'] for r in rows),
                   failures=[{k:r[k] for k in ('method','benchmark','index','retention','native_replay_max_logit_error')}
                             for r in rows if not r['numerical_pass']])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if any(codes) or len(rows) != 108: raise SystemExit(1)


if __name__ == '__main__':
    run() if len(sys.argv) == 1 else worker(sys.argv[1])
