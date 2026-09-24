"""Frozen-native-trajectory visual-effect oracle with uncentered shared bases.

This preserves Table 6's external native source of each layer's Delta, not
Table 7's current-intervened-trajectory source. All text positions are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def uncentered_basis(moment, rank):
    """Columns are Table 7's descending eigenvectors of sum Delta.T @ Delta."""
    assert moment.dtype == torch.float64 and 0 < rank <= moment.shape[0]
    values, vectors = torch.linalg.eigh((moment + moment.T) * .5)
    basis = vectors[:, -rank:].flip(1).float().contiguous()
    error = (basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max()
    assert float(error) < 1e-4
    return basis, values.flip(0).clamp_min(0)


def project(delta, basis, rank):
    if rank == 0:
        return torch.zeros_like(delta, dtype=torch.float32)
    b = basis[:, :rank].float()
    return (delta.float() @ b) @ b.T


def reconstruction_error(joint, blocked, delta):
    # Even BF16 operands may span more than FP32's 24 significant bits:
    # subtracting a large blocked value can erase a very small joint value.
    # Check the FP32 arithmetic error bound, not bitwise cancellation identity.
    restored = blocked.float() + delta
    error = (restored - joint.float()).abs()
    bound = 4 * torch.finfo(torch.float32).eps * (blocked.float().abs() + joint.float().abs())
    assert bool((error <= bound + torch.finfo(torch.float32).tiny).all())
    return dict(max_abs=float(error.max()),
                relative_l2=float(error.norm()/joint.float().norm().clamp_min(1e-30)),
                nonidentical_elements=int((restored.to(joint.dtype) != joint).sum()))


def text_kwargs(kwargs, positions):
    out = dict(kwargs)
    assert out.get('past_key_values') is None
    out['hidden_states'] = out['hidden_states'].index_select(1, positions)
    out['position_embeddings'] = tuple(t.index_select(-2, positions) for t in out['position_embeddings'])
    mask = out.get('attention_mask')
    if mask is not None:
        assert mask.ndim == 4, 'Single unpadded sample required'
        out['attention_mask'] = mask.index_select(-2, positions).index_select(-1, positions)
    if out.get('position_ids') is not None:
        out['position_ids'] = out['position_ids'].index_select(-1, positions)
    if out.get('cache_position') is not None:
        out['cache_position'] = out['cache_position'].index_select(0, positions)
    return out


class NativeTraceOracle:
    def __init__(self, model):
        self.model = model
        self.language = model.model.language_model
        self.active = False
        self.handles = [self.language.layers[0].register_forward_pre_hook(self.initial, with_kwargs=True)]
        self.handles += [layer.self_attn.register_forward_hook(self.capture(i), with_kwargs=True)
                         for i, layer in enumerate(self.language.layers)]

    def initial(self, module, args, kwargs):
        if self.active:
            h = kwargs.get('hidden_states', args[0] if args else None)
            self.hidden0 = h.index_select(1, self.positions).detach()

    def capture(self, index):
        def hook(module, args, kwargs, output):
            if not self.active:
                return
            kw = dict(kwargs)
            if 'hidden_states' not in kw:
                kw['hidden_states'] = args[0]
            kw = text_kwargs(kw, self.positions)
            # Direct forward bypasses hooks. The native branch is not modified.
            blocked = module.forward(**kw)[0]
            joint = output[0].index_select(1, self.positions)
            delta = joint.float() - blocked.float()
            self.arithmetic[index] = reconstruction_error(joint, blocked, delta)
            kw.pop('hidden_states')
            self.effects[index] = delta
            self.kwargs[index] = kw
        return hook

    @torch.inference_mode()
    def trace(self, inputs, types, kind):
        assert types.shape[0] == 1 and bool(inputs['attention_mask'].all())
        self.positions = (types[0] == 0).nonzero().flatten()
        assert bool((types == 1).any()) and len(self.positions) > 0
        self.effects, self.kwargs, self.arithmetic = {}, {}, {}
        self.active = True
        if hasattr(self.model.model, 'rope_deltas'):
            self.model.model.rope_deltas = None
        actual = {k: v for k, v in inputs.items() if kind == 'qwen' or k != 'mm_token_type_ids'}
        try:
            result = self.model(**actual, use_cache=False, return_dict=True, logits_to_keep=1)
        finally:
            self.active = False
        assert len(self.effects) == len(self.language.layers)
        return dict(hidden0=self.hidden0, effects=self.effects, kwargs=self.kwargs,
                    native_logits=result.logits[:, -1].float(), arithmetic=self.arithmetic,
                    positions=self.positions, sequence_length=types.shape[1])

    @torch.inference_mode()
    def rollout(self, trace, bases, rank):
        h = trace['hidden0']
        for index, layer in enumerate(self.language.layers):
            blocked = layer.self_attn.forward(hidden_states=layer.input_layernorm(h),
                                              **trace['kwargs'][index])[0]
            delta = trace['effects'][index] if rank is None else project(trace['effects'][index], bases[index], rank)
            # Recombine attention in FP32 before the native residual addition.
            attention = (blocked.float() + delta).to(h.dtype)
            h = h + attention
            # Preserve native GEMM row count to avoid BF16 kernel-shape drift.
            # These are zero placeholders, not teacher visual hidden states;
            # tokenwise FFN outputs at placeholders are discarded.
            padded = h.new_zeros((h.shape[0], trace['sequence_length'], h.shape[-1]))
            padded.index_copy_(1, trace['positions'], h)
            ffn = layer.mlp(layer.post_attention_layernorm(padded))
            h = h + ffn.index_select(1, trace['positions'])
        return self.model.lm_head(self.language.norm(h[:, -1:]))[:, -1].float()


def append_prefix(inputs, types, prefix):
    current = dict(inputs)
    if prefix:
        ids = torch.tensor([prefix], device=inputs['input_ids'].device)
        current['input_ids'] = torch.cat((inputs['input_ids'], ids), 1)
        current['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(ids)), 1)
        types = torch.cat((types, torch.zeros_like(ids)), 1)
        if 'mm_token_type_ids' in current:
            current['mm_token_type_ids'] = types
    return current, types


@torch.inference_mode()
def generate_all(oracle, processor, inputs, types, kind, bases, ranks, spec, row, cap):
    from src.evaluate import _structured_answer_ready
    from src.benchmarks import score_prediction
    modes = {'native': 'native', 'full_effect': None, **{f'rank{r}': r for r in ranks}}
    ids, done, stops = {m: [] for m in modes}, set(), {}
    eos = oracle.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    errors = []
    for step in range(cap):
        groups = {}
        for mode in modes:
            if mode not in done:
                groups.setdefault(tuple(ids[mode]), []).append(mode)
        for prefix, names in groups.items():
            current, current_types = append_prefix(inputs, types, prefix)
            trace = oracle.trace(current, current_types, kind)
            for mode in names:
                logits = trace['native_logits'] if mode == 'native' else oracle.rollout(trace, bases, modes[mode])
                if mode == 'full_effect':
                    native = trace['native_logits']
                    errors.append(dict(step=step, relative_rms=float((logits-native).norm()/native.norm().clamp_min(1e-12)),
                                       argmax_equal=bool(logits.argmax() == native.argmax()),
                                       attention_arithmetic=trace['arithmetic']))
                token = int(logits[0].argmax())
                ids[mode].append(token)
                text = processor.tokenizer.decode(ids[mode], skip_special_tokens=True).strip()
                if token in eos or _structured_answer_ready(spec.metric, text, row.get('choices')):
                    done.add(mode)
                    stops[mode] = 'eos' if token in eos else 'structured_answer'
            del trace
        if len(done) == len(modes):
            break
    result = {}
    for mode, tokens in ids.items():
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        scored = score_prediction(metric=spec.metric, prediction_text=text, answer=row.get('answer'),
                                  answers=row.get('answers'), choices=row.get('choices'), question=row.get('question'))
        result[mode] = dict(text=text, token_ids=tokens, stop=stops.get(mode, 'length'), **scored)
    return result, errors


def worker(args):
    from src.model import load_frozen_qwen3vl, load_frozen_llava
    from src.data import QwenBenchmarkDataset, LlavaBenchmarkDataset
    from src.benchmarks import get_benchmark_spec
    from src.model_setup import disable_qwen_deepstack
    from analysis.table06_layer_effect.causal_effect_benchmark_suite import prepare
    run = Path(args.run_dir)
    config = json.loads((run/'config.json').read_text())
    info = config['datasets'][args.benchmark]
    destination = run/f'{args.model}_{args.benchmark}'
    destination.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = load_frozen_qwen3vl if args.model == 'qwen' else load_frozen_llava
    processor, model = loader(config['models'][args.model], torch.bfloat16, 'cuda:0', 'flash_attention_2')
    if args.model == 'qwen':
        disable_qwen_deepstack(model)
    assert model.model.language_model.config._attn_implementation == 'flash_attention_2'
    cls = QwenBenchmarkDataset if args.model == 'qwen' else LlavaBenchmarkDataset
    dataset = cls(info['path'], processor, args.benchmark, data_root=info['image_root'])
    assert len(dataset) == info['samples']
    oracle = NativeTraceOracle(model)
    # Reuse only image encoder features within one question, never decoder states.
    features = []
    original = model.model.get_image_features
    def cached(*a, **kw):
        if not features:
            features.append(original(*a, **kw))
        return features[0]
    model.model.get_image_features = cached
    bases = None
    if args.phase == 'eval':
        payload = torch.load(destination/'basis.pt', map_location='cpu', weights_only=False)
        assert payload['centered'] is False and payload['data_sha256'] == info['sha256']
        bases = {i: b.cuda().float() for i, b in payload['basis'].items()}
    covariances, counts = {}, {}
    indices = range(len(dataset)) if args.phase == 'basis' else range(args.shard, len(dataset), 8)
    if args.phase == 'smoke':
        indices = range(min(3, len(dataset)))
    output = destination/f'{args.phase}_{args.shard}.jsonl'
    started = time.time()
    with torch.inference_mode(), output.open('w') as handle:
        for j, index in enumerate(indices):
            item = dataset[index]
            inputs = prepare(item, model, args.model)
            types = inputs['mm_token_type_ids']
            features.clear()
            if args.phase == 'basis':
                trace = oracle.trace(inputs, types, args.model)
                for layer, delta in trace['effects'].items():
                    x = delta[0].double()
                    if layer not in covariances:
                        covariances[layer] = torch.zeros((x.shape[-1], x.shape[-1]), device=x.device, dtype=torch.float64)
                        counts[layer] = 0
                    covariances[layer].addmm_(x.T, x)
                    counts[layer] += len(x)
                record = dict(sample=index, text_tokens=len(oracle.positions))
                del trace
            else:
                results, errors = generate_all(oracle, processor, inputs, types, args.model, bases,
                    config['ranks'] if args.phase == 'eval' else [], get_benchmark_spec(args.benchmark),
                    item['row'], config['max_new_tokens'])
                record = dict(sample=index, original_index=info['source_indices'][index], results=results,
                    full_reconstruction=errors, input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest())
                if args.phase == 'smoke':
                    assert max(e['relative_rms'] for e in errors) < .03, ('Full restore numerical error', record)
                    assert results['native']['prediction'] == results['full_effect']['prediction'], ('Full restore answer mismatch', record)
            handle.write(json.dumps(record, ensure_ascii=False)+'\n'); handle.flush()
            if j % 10 == 0:
                print(args.model, args.benchmark, args.phase, args.shard, index, round(time.time()-started, 1), flush=True)
        if args.phase == 'basis':
            bases, energy = {}, {}
            for layer in range(len(oracle.language.layers)):
                b, spectrum = uncentered_basis(covariances.pop(layer), max(config['ranks']))
                bases[layer] = b.cpu()
                energy[layer] = {r: float(spectrum[:r].sum()/spectrum.sum().clamp_min(1e-30)) for r in config['ranks']}
                print('BASIS', layer, flush=True)
            torch.save(dict(basis=bases, energy=energy, counts=counts, centered=False,
                            data_sha256=info['sha256'], protocol=config['basis_protocol']), destination/'basis.pt')
    dump(destination/f'{args.phase}_{args.shard}.done.json', dict(passed=True, samples=len(indices), seconds=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model', choices=['qwen', 'llava'], required=True)
    parser.add_argument('--benchmark', choices=['sqa', 'mmstar', 'realworldqa'], required=True)
    parser.add_argument('--phase', choices=['smoke', 'basis', 'eval'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    worker(parser.parse_args())
