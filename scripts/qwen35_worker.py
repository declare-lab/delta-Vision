"""Qwen3.5 correctness gate, frozen-backbone PixMo KL training, and 9-task evaluation."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'artifacts/dependencies/qwen35_python'))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.benchmarks import build_benchmark_prompt, get_benchmark_spec, score_prediction
from src.qwen35_experiment import (answer_suffix, dump, initial_context, load_model, prepare_inputs, generate_evaluation_answer,
                                  sha, student_loss, teacher_targets)


def read_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train(config, run):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == config['world_size'] == 8
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    device = torch.device('cuda', local_rank)
    processor, model, adapter, controller = load_model(config, device)
    controller.adapter = DDP(adapter, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['lr'],
        betas=tuple(config['betas']), weight_decay=config['weight_decay'], fused=True)
    assert sha(config['data']) == config['data_sha256']
    rows = read_rows(config['data'])
    areas = json.loads(Path(config['pixel_area_cache']).read_text())
    areas = areas['areas'] if isinstance(areas, dict) else areas
    assert len(areas) == len(rows)
    sized = sorted(range(len(rows)), key=lambda i: (areas[i], i))
    buckets = [sized[i:i+512] for i in range(0, len(sized), 512)]
    generator = torch.Generator().manual_seed(config['seed'])
    order = []
    for i in torch.randperm(len(buckets), generator=generator).tolist():
        bucket = buckets[i]
        order.extend(bucket[j] for j in torch.randperm(len(bucket), generator=generator).tolist())
    wandb_run = None
    if rank == 0:
        import wandb
        wandb_run = wandb.init(project='vision-kv-inject', name=run.name, config=config,
                               mode='online', dir=str(run))
        dump(run/'wandb.json', {'id': wandb_run.id, 'url': wandb_run.url})
    accumulated = config['gradient_accumulation_steps']
    global_batch = world * accumulated
    started = time.time()
    metric_file = (run/'train_metrics.jsonl').open('a') if rank == 0 else None
    for step in range(config['max_steps']):
        warmup = math.ceil(config['max_steps'] * config['warmup_ratio'])
        factor = ((step+1) / warmup if step < warmup else
                  .1 + .9 * .5 * (1 + math.cos(math.pi * (step-warmup+1) / (config['max_steps']-warmup))))
        for group in optimizer.param_groups:
            group['lr'] = config['lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        local_loss = torch.zeros((), device=device)
        # Reproduce the original PixMo microbatch4 token mean, using sequential
        # microbatch1 forwards for hybrid-cache simplicity. Averaging four sample
        # means would silently change the relative weight of short/long answers.
        samples = [order[(step*global_batch + rank*accumulated + micro) % len(order)]
                   for micro in range(accumulated)]
        answer_counts = [len(processor.tokenizer(answer_suffix(processor, rows[i]),
                           add_special_tokens=False).input_ids) for i in samples]
        total_answer_tokens = sum(answer_counts)
        for micro in range(accumulated):
            sample = samples[micro]
            inputs, prompt_length = prepare_inputs(processor, rows[sample], config['image_root'],
                                                     device, training=True)
            context = initial_context(model, inputs)
            targets = inputs['input_ids'][:, prompt_length:]
            assert targets.shape[1] == answer_counts[micro] > 0
            indices, probabilities = teacher_targets(model, context, prompt_length, targets,
                                                       config['kl_topk'], config['temperature'])
            sync = nullcontext() if micro == accumulated-1 else controller.adapter.no_sync()
            with sync, controller.activate('adapter', inputs['mm_token_type_ids'].eq(1), checkpoint_layers=True):
                loss = student_loss(model, context, prompt_length, indices, probabilities, config['temperature'])
                assert bool(torch.isfinite(loss)), (step, rank, sample)
                (loss * (answer_counts[micro] / total_answer_tokens)).backward()
            local_loss += loss.detach() * (answer_counts[micro] / total_answer_tokens)
            del loss, inputs, context, indices, probabilities
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % config['log_every'] == 0:
            dist.all_reduce(local_loss)
            if rank == 0:
                metrics = {'step': step+1, 'kl_loss': local_loss.item()/world,
                           'lr': optimizer.param_groups[0]['lr'], 'grad_norm': norm.item(),
                           'elapsed_seconds': time.time()-started,
                           'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9}
                metric_file.write(json.dumps(metrics)+'\n')
                metric_file.flush()
                wandb_run.log(metrics, step=step+1)
                print(json.dumps(metrics), flush=True)
        if (step+1) % config['save_every'] == 0 or step+1 == config['max_steps']:
            if rank == 0:
                dest = run/'checkpoints'/f'qwen35_embedding_adapter_step{step+1}.pt'
                dest.parent.mkdir(exist_ok=True)
                torch.save({'state_dict': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                            'global_step': step+1, 'config': config}, dest.with_suffix('.tmp'))
                dest.with_suffix('.tmp').replace(dest)
            dist.barrier()
    if rank == 0:
        metric_file.close()
        wandb_run.finish()
    dist.destroy_process_group()


@torch.inference_mode()
def evaluate(config, run, method, shard):
    device = torch.device('cuda:0')
    processor, model, adapter, controller = load_model(config, device)
    if method == 'adapter':
        ckpt = torch.load(run/'checkpoints/qwen35_embedding_adapter_step2000.pt',
                          map_location='cpu', weights_only=False)
        assert ckpt['global_step'] == config['max_steps'] == 2000
        adapter.load_state_dict(ckpt['state_dict'], strict=True)
    dest = run/'eval'/method
    dest.mkdir(parents=True, exist_ok=True)
    for benchmark, info in config['evaluation'].items():
        rows = read_rows(info['path'])
        assert sha(info['path']) == info['sha256'] and len(rows) == info['samples']
        path = dest/f'{benchmark}.shard{shard}.jsonl'
        completed = read_rows(path) if path.exists() else []
        expected = list(range(shard, len(rows), 8))
        assert [r['index'] for r in completed] == expected[:len(completed)]
        spec = get_benchmark_spec(benchmark)
        with path.open('a') as handle:
            for index in expected[len(completed):]:
                row = rows[index]
                inputs, _ = prepare_inputs(processor, row, info['image_root'], device,
                    question=build_benchmark_prompt(row, spec))
                with controller.activate(method, inputs['mm_token_type_ids'].eq(1)):
                    generated = generate_evaluation_answer(model, processor, inputs, row, spec, config,
                                                          max_new_tokens=info['max_new_tokens'])
                result = {'index': index, 'benchmark': benchmark, **generated,
                    'input_ids_sha256': __import__('hashlib').sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                    'image_grid_thw': inputs['image_grid_thw'].tolist()}
                handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                handle.flush()
                if index % 80 < 8:
                    print(method, benchmark, index, flush=True)


def difference(a, b):
    a, b = a.float(), b.float()
    delta = a-b
    return {'max_abs': delta.abs().max().item(), 'rms': delta.square().mean().sqrt().item(),
            'relative_rms': (delta.square().mean()/a.square().mean().clamp_min(1e-12)).sqrt().item(),
            'argmax_equal': bool(torch.equal(a.argmax(-1), b.argmax(-1)))}


@contextmanager
def native_ffn_shape_for_oracle(model, mask):
    """Positive control ONLY: remove BF16 GEMM shape rounding as a confounder.

    FFN rows are independent. The production adapter always uses text-only FFN.
    Padding its input back to the native row count isolates wiring errors from
    the choice of GEMM algorithm. It must recover native outputs exactly.
    """
    originals = []
    text_idx = (~mask[0]).nonzero().flatten()
    for layer in model.model.language_model.layers:
        original = layer.mlp.forward
        originals.append(original)
        def padded(x, _original=original):
            if x.shape[1] == text_idx.numel():
                full = x.new_zeros((1, mask.shape[1], x.shape[-1])).index_copy(1, text_idx, x)
                return _original(full).index_select(1, text_idx)
            return _original(x)
        layer.mlp.forward = padded
    try:
        yield
    finally:
        for layer, original in zip(model.model.language_model.layers, originals):
            layer.mlp.forward = original


def compare_caches(first, second):
    checked = 0
    for a, b in zip(first.layers, second.layers):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            x, y = getattr(a, name, None), getattr(b, name, None)
            if x is None: continue
            xs, ys = (x, y) if isinstance(x, (list, tuple)) else ([x], [y])
            for u, v in zip(xs, ys):
                if u is not None:
                    torch.testing.assert_close(u, v, rtol=0, atol=0)
                    checked += 1
    assert checked >= 64, checked
    return checked


def validate(config, run):
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0')
    processor, model, adapter, controller = load_model(config, device)
    report = {'checks': []}
    # Two independent real inputs, including the historically sensitive RealWorldQA.
    for name in ('mmstar', 'realworldqa'):
        info = config['evaluation'][name]
        row = read_rows(info['path'])[0]
        inputs, _ = prepare_inputs(processor, row, info['image_root'], device,
                                   question=build_benchmark_prompt(row, get_benchmark_spec(name)))
        mask = inputs['mm_token_type_ids'].eq(1)
        with torch.no_grad():
            context = initial_context(model, inputs)
            text_idx = (~mask[0]).nonzero().flatten()
            with controller.activate('capture', mask):
                native_hidden = model.model.language_model(**context).last_hidden_state
            with controller.activate('oracle', mask):
                oracle_hidden = model.model.language_model(**context).last_hidden_state
            native_logits = model.lm_head(native_hidden.index_select(1, text_idx[-32:]))
            oracle_logits = model.lm_head(oracle_hidden.index_select(1, text_idx[-32:]))
            error = difference(native_logits, oracle_logits)
            report['checks'].append({'benchmark': name, 'test': 'text_only_FFN_bf16_rounding', **error})
            # Check exact native equivalence with the SAME GEMM dimensions.
            # Keep the unpadded result above visible; do not hide its rounding error.
            cache_context = dict(context, use_cache=True)
            with controller.activate('capture', mask):
                native_cached = model.model.language_model(**cache_context)
            with native_ffn_shape_for_oracle(model, mask), controller.activate('oracle', mask):
                oracle_cached = model.model.language_model(**cache_context)
            torch.testing.assert_close(native_cached.last_hidden_state[:, ~mask[0]],
                oracle_cached.last_hidden_state[:, ~mask[0]], rtol=0, atol=0)
            count = compare_caches(native_cached.past_key_values, oracle_cached.past_key_values)
            report['checks'].append({'benchmark': name, 'test': 'oracle_native_shape_hidden_and_all_caches',
                                     'exact_equal': True, 'cache_tensors_checked': count})
            del native_cached, oracle_cached
            del native_hidden, oracle_hidden, native_logits, oracle_logits
            # Native HF prepare_inputs_for_generation and hybrid cache, then full-prefix reference.
            native_decode_errors = []
            for mode in ('native', 'adapter'):
                seq = {k: v.clone() for k, v in inputs.items()}
                with controller.activate(mode, mask):
                    cached = model(**seq, use_cache=True, logits_to_keep=1)
                for step in range(3):
                    token = cached.logits[:, -1].argmax(-1, keepdim=True)
                    seq['input_ids'] = torch.cat((seq['input_ids'], token), 1)
                    seq['attention_mask'] = torch.cat((seq['attention_mask'], torch.ones_like(token)), 1)
                    seq['mm_token_type_ids'] = torch.cat((seq['mm_token_type_ids'], torch.zeros_like(token)), 1)
                    # Match HF generation: slice RoPE positions to the new token.
                    # Calling raw forward with the full attention mask but no
                    # explicit positions would construct positions for the entire prefix.
                    positions = model._prepare_position_ids_for_generation(seq['input_ids'],
                        {'attention_mask': seq['attention_mask'], 'past_key_values': cached.past_key_values})[..., -1:]
                    with controller.activate(mode, mask):
                        cached = model(input_ids=token, attention_mask=seq['attention_mask'],
                            position_ids=positions, past_key_values=cached.past_key_values,
                            use_cache=True, logits_to_keep=1)
                    with controller.activate(mode, seq['mm_token_type_ids'].eq(1)):
                        full = model(**seq, use_cache=False, logits_to_keep=1)
                    error = difference(full.logits, cached.logits)
                    report['checks'].append({'benchmark': name, 'test': mode+'_cached_decode', 'step': step, **error})
                    if mode == 'native':
                        native_decode_errors.append(error['relative_rms'])
                    else:
                        # BF16 single-token vs full-sequence GEMMs differ even for
                        # the unmodified model. Compare to that positive control.
                        limit = max(.02, 2*max(native_decode_errors))
                        assert error['relative_rms'] < limit, (error, limit)
                del cached, full
    # Exercise real FLA backward, both decoder types, and checkpoint recomputation.
    train_row = read_rows(config['data'])[0]
    inputs, plen = prepare_inputs(processor, train_row, config['image_root'], device, training=True)
    context = initial_context(model, inputs)
    idx, probs = teacher_targets(model, context, plen, inputs['input_ids'][:, plen:],
                                  config['kl_topk'], config['temperature'])
    with controller.activate('adapter', inputs['mm_token_type_ids'].eq(1), checkpoint_layers=True):
        loss = student_loss(model, context, plen, idx, probs, config['temperature'])
        assert torch.isfinite(loss)
        loss.backward()
    gradient_norms = [layer.weight.grad.float().norm().item() for layer in adapter.up]
    assert all(math.isfinite(x) and x > 0 for x in gradient_norms), gradient_norms
    assert all(p.grad is None for p in model.parameters())
    report.update(passed=True, kl_loss=loss.item(), per_layer_up_grad_norm=gradient_norms,
                  trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                  backbone_frozen=True, checkpoint_backward=True)
    dump(run/'validation.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['validate', 'train', 'eval'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', choices=['native', 'adapter'])
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    config = json.loads((args.run_dir/'config.json').read_text())
    if args.stage == 'validate': validate(config, args.run_dir)
    elif args.stage == 'train': train(config, args.run_dir)
    else: evaluate(config, args.run_dir, args.method, args.shard)


if __name__ == '__main__':
    main()
