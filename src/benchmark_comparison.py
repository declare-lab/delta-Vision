"""Matched dense Qwen3-VL speed benchmark, invoked by benchmark_prefill.

All methods use identical processed inputs,
fixed output lengths, and cached decode. Ports retain their own prefill selectors;
decode uses their native decoder layers with original (uncompressed) M-RoPE.
FLOPs are analytical decoder-core prefill FLOPs, not whole-VLM FLOPs.
Adapter prefill reuses benchmark_prefill's original CUDA graph fast path.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

METHODS = ("base", "embedding_adapter", "fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip")
ALIASES = {"zooprune": "zoo", "visionzup": "visionzip"}


def parse_methods(values: list[str]) -> list[str]:
    names = [ALIASES.get(v.lower(), v.lower()) for value in values for v in value.split(",")]
    if names == ["all"]:
        return list(METHODS)
    unknown = set(names) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown comparison methods: {sorted(unknown)}")
    return list(dict.fromkeys(["base", *names]))


def tensor_storage_bytes(value: Any) -> int:
    """Count backing storage once, including aliased views; exclude model weights."""
    seen: set[tuple] = set()

    def visit(obj):
        if torch.is_tensor(obj):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (tuple, list)):
            return sum(visit(v) for v in obj)
        if hasattr(obj, "layers"):
            return sum(visit({"keys": layer.keys, "values": layer.values}) for layer in obj.layers)
        return 0

    return visit(value)


def cache_metrics(cache, adapter: bool) -> dict[str, Any]:
    if adapter:
        kv = cache["layers"]
        lengths = [int(layer["text_key"].shape[-2] + layer["visual_key"].shape[-2]) for layer in kv]
    else:
        kv = [{"key": layer.keys, "value": layer.values} for layer in cache.layers]
        lengths = [int(layer.keys.shape[-2]) for layer in cache.layers]
    return {
        "kv_cache_mb": tensor_storage_bytes(kv) / 1024**2,
        "decode_cache_mb": tensor_storage_bytes(cache) / 1024**2,
        "layer_cache_lengths": lengths,
    }


def decoder_flops(config, lengths: list[int], *, text_tokens: int, image_tokens: int, adapter_rank=None) -> float:
    """2 FLOPs/MAC, dense attention square convention; excludes norm/softmax/head/vision/selectors."""
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    dim = int(config.head_dim)
    intermediate = int(config.intermediate_size)
    linear_per_token = 2 * (hidden * (2 * heads * dim + 2 * kv_heads * dim) + 3 * hidden * intermediate)
    if adapter_rank is None:
        return float(sum(linear_per_token * n + 4 * heads * dim * n * n for n in lengths))
    layers = len(lengths)
    text_core = layers * (linear_per_token * text_tokens + 4 * heads * dim * text_tokens**2)
    visual_kv = layers * 4 * image_tokens * hidden * kv_heads * dim
    cross_attention = layers * 4 * heads * dim * text_tokens * image_tokens
    adapter_projection = layers * 4 * image_tokens * hidden * max(1, int(adapter_rank))
    return float(text_core + visual_kv + cross_attention + adapter_projection)


@contextmanager
def disable_prune_audit():
    # audit_prune transfers selected indices to CPU for JSON logs. It is diagnostic
    # I/O, not part of token selection. Cache lengths are observed outside timing.
    from baselines import multimodal_pruning_utils as pruning
    original = pruning.audit_prune
    pruning.audit_prune = lambda *args, **kwargs: None
    try:
        yield
    finally:
        pruning.audit_prune = original


def native_cached_step(model, token: torch.Tensor, cache, position_ids: torch.Tensor):
    """Single unpadded token: attend to all cached keys in each individual layer.

    Pruning leaves different cache lengths across layers. Going through a shared
    HF generation mask or deriving RoPE from cache length is incorrect here.
    """
    language = model.model.language_model
    hidden = language.embed_tokens(token)
    position_embeddings = language.rotary_emb(hidden, position_ids)
    for layer in language.layers:
        hidden = layer(hidden, attention_mask=None, position_ids=None,
                       past_key_values=cache, position_embeddings=position_embeddings)
    return model.lm_head(language.norm(hidden)), cache


class RequestRunner:
    def __init__(self, model, method, adapter=None, decode_mode="fast", adapter_prefill_fn=None):
        self.model, self.method, self.adapter = model, method, adapter
        self.decode_mode = decode_mode
        self.adapter_prefill_fn = adapter_prefill_fn

    def prefill(self, inputs):
        self.model.model.rope_deltas = None
        if self.adapter is not None:
            from src.model import build_qwen_initial_context, qwen_embedding_adapter_prefill_cache
            if self.adapter_prefill_fn is not None:
                logits, _, _, _, cache = self.adapter_prefill_fn(inputs)
            else:
                hidden, positions = build_qwen_initial_context(self.model, inputs)
                logits, _, cache = qwen_embedding_adapter_prefill_cache(
                    self.model, self.adapter, inputs["input_ids"], inputs["attention_mask"],
                    inputs["mm_token_type_ids"], hidden, positions, logits_to_keep=1)
            if self.decode_mode == "fast":
                # Only shape_exact needs full prompt hidden states for replay.
                cache.pop("layer_inputs", None)
                cache.pop("layer_after_attention", None)
            return logits, cache, None
        output = self.model(**inputs, use_cache=True, logits_to_keep=1)
        if output.past_key_values is None:
            raise RuntimeError(f"{self.method} did not produce a KV cache")
        next_position = (inputs["input_ids"].shape[1] + self.model.model.rope_deltas).view(1, 1, 1).expand(3, 1, 1)
        return output.logits, output.past_key_values, next_position

    def step(self, token, cache, position):
        if self.adapter is not None:
            from src.model import qwen_embedding_adapter_decode_step, qwen_embedding_adapter_decode_step_shape_exact
            fn = qwen_embedding_adapter_decode_step if self.decode_mode == "fast" else qwen_embedding_adapter_decode_step_shape_exact
            logits, cache = fn(self.model, self.adapter, token, cache)
            return logits, cache, None
        logits, cache = native_cached_step(self.model, token, cache, position)
        return logits, cache, position + 1

    def request(self, inputs, new_tokens, *, measure=False):
        device = inputs["input_ids"].device
        if measure:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, cache, position = self.prefill(inputs)
        if measure:
            torch.cuda.synchronize(device)
        prefill_end = time.perf_counter()
        # Exclude resource inspection from both prefill and decode timings.
        initial = cache_metrics(cache, self.adapter is not None) if measure else None
        if measure:
            torch.cuda.synchronize(device)
        decode_start = time.perf_counter()
        tokens = []
        for i in range(new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(token)
            if i + 1 < new_tokens:
                logits, cache, position = self.step(token, cache, position)
        if measure:
            torch.cuda.synchronize(device)
        decode_end = time.perf_counter()
        if not measure:
            return None
        return {
            "prefill_time_s": prefill_end - start,
            "decode_time_s": decode_end - decode_start,
            "total_time_s": prefill_end - start + decode_end - decode_start,
            **initial,
            "final_kv_cache_mb": cache_metrics(cache, self.adapter is not None)["kv_cache_mb"],
            "generated_token_ids": torch.cat(tokens, dim=1)[0].cpu().tolist(),
        }


def verify_cached_decode(runner, inputs) -> dict:
    """Compare sequential cached decode to a causal two-token chunk on native layers.

    This specifically checks unequal per-layer KV lengths and original positions.
    Adapter fast decode is compared to its shape-exact reference, teacher forcing
    the same token on both paths. All checks happen outside measured requests.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(inputs["input_ids"].device)
    def reset_rng():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, inputs["input_ids"].device)

    logits, cache, position = runner.prefill(inputs)
    token = logits[:, -1].argmax(-1, keepdim=True)
    first, cache, next_position = runner.step(token, cache, position)
    if runner.adapter is not None:
        from src.model import qwen_embedding_adapter_decode_step_shape_exact
        reference_runner = RequestRunner(runner.model, runner.method, runner.adapter, "shape_exact")
        reset_rng()
        _, reference_cache, _ = reference_runner.prefill(inputs)
        expected, _ = qwen_embedding_adapter_decode_step_shape_exact(runner.model, runner.adapter, token, reference_cache)
        actual = first
    else:
        # Feed the same two tokens either sequentially or as a causal chunk.
        second_token = first[:, -1].argmax(-1, keepdim=True)
        second, _, _ = runner.step(second_token, cache, next_position)
        reset_rng()
        _, reference_cache, position = runner.prefill(inputs)
        language = runner.model.model.language_model
        hidden = language.embed_tokens(torch.cat([token, second_token], dim=1))
        positions = torch.cat([position, position + 1], dim=-1)
        rotary = language.rotary_emb(hidden, positions)
        for i, layer in enumerate(language.layers):
            past = reference_cache.layers[i].keys.shape[-2]
            mask = torch.ones((1, 1, 2, past + 2), device=hidden.device, dtype=torch.bool)
            mask[:, :, 0, -1] = False
            if language.config._attn_implementation == "flash_attention_2":
                # FlashAttention uses bottom-right causal alignment for a suffix;
                # it does not accept a 4D SDPA attention mask.
                mask = None
            hidden = layer(hidden, attention_mask=mask, position_ids=None,
                           past_key_values=reference_cache, position_embeddings=rotary)
        expected = runner.model.lm_head(language.norm(hidden))
        actual = torch.cat([first, second], dim=1)
    a, b = actual.float(), expected.float()
    relative = float((a - b).norm() / b.norm().clamp_min(1e-12))
    kl = float((b.softmax(-1) * (b.log_softmax(-1) - a.log_softmax(-1))).sum(-1).max())
    result = {"relative_logit_error": relative, "max_kl": kl,
              "same_argmax": bool(torch.equal(a.argmax(-1), b.argmax(-1)))}
    # BF16 single-token GEMMs differ numerically from chunk/full-text GEMMs.
    if not (relative < 0.04 and kl < 0.05):
        raise RuntimeError(f"Cached decode validation failed for {runner.method}: {result}")
    return result


def comparison_markdown(rows) -> str:
    lines = ["# Qwen3-VL matched speed comparison", "",
             "Adapter uses the original benchmark_prefill fast path (CUDA graphs by default); base/pruning use native forwards and cached decode. Batch size 1, fixed output token count. Times are sums of per-sample request medians. KV and FLOPs are per-sample means.", "",
             "FLOPs = analytical LLM prefill core + adapter projections (2 FLOPs/MAC, full-square attention convention). Vision encoder, selector/merge, LM head, normalization and softmax are excluded from FLOPs; their execution is included in timings. KV Cache is actual per-layer text + visual K/V storage, in MiB.", "",
             "Repository Qwen ports; these are not official upstream speed results. Generated text/quality is not scored in this fixed-length speed run.", "",
             "| Method | Retention | Total (s) | Prefill (s) | KV (MiB) | Prefill FLOPs (T) | Total speedup | Prefill speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['retention']:.0%} | {r['total_time_s']:.4f} | {r['prefill_time_s']:.4f} | {r['kv_cache_mb']:.2f} | {r['flops'] / 1e12:.4f} | {r['speedup_total']:.3f}x | {r['speedup_prefilling']:.3f}x |")
    return "\n".join(lines) + "\n"


def run_comparison(args):
    from transformers import AutoConfig, AutoProcessor
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from src.benchmark_prefill import build_qwen_fast_adapter_prefill, collect_checkpoint_specs, dtype_from_name, _resolve_repo_or_data_path, write_outputs
    from src.benchmarks import get_benchmark_spec
    from src.data import QwenBenchmarkDataset
    from src.model import load_qwen_embedding_adapter_checkpoint

    methods = parse_methods(args.compare_methods)
    if AutoConfig.from_pretrained(args.model_path).model_type != "qwen3_vl":
        raise ValueError("Matched comparison currently supports dense Qwen3-VL only")
    if args.compile or args.compile_teacher or args.compile_e2e or args.context_cache_dir or args.skip_e2e:
        raise ValueError("Comparison uses the metric-table CUDA graph path; compile and cached visual context are unsupported")
    if args.eval_batch_size != 1 or args.batch_size != 1:
        raise ValueError("Matched comparison currently requires batch size 1")
    if not args.last_logits_only:
        raise ValueError("Comparison requires last-logits-only for all methods")
    if args.attn_implementation not in ("auto", "sdpa", "flash_attention_2"):
        raise ValueError("Comparison supports auto, sdpa or flash_attention_2")
    attention = "flash_attention_2" if args.attn_implementation == "auto" else args.attn_implementation
    if args.cuda_graph and args.metric_prefill_warmup < 1:
        raise ValueError("CUDA graph comparison needs at least one unmeasured warmup per sample")
    if args.output_mode not in (None, "embedding_adapter"):
        raise ValueError("Matched comparison currently supports static embedding_adapter only")
    if min(args.metric_samples, args.comparison_runs, args.log_every) < 1 or args.metric_prefill_warmup < 0 or args.sample_index < 0:
        raise ValueError("Sample/run/log counts must be positive; warmup and sample index nonnegative")
    if any(not 0 < r <= 1 for r in args.retentions):
        raise ValueError("Retentions must lie in (0, 1]")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Matched speed measurements require CUDA")
    new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 32
    if new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    specs = collect_checkpoint_specs(args)
    if "embedding_adapter" in methods and len(specs) != 1:
        raise ValueError("Specify exactly one --checkpoint for embedding_adapter")
    checkpoint = specs[0][1] if "embedding_adapter" in methods else None
    spec = get_benchmark_spec(args.benchmark)
    data_path = _resolve_repo_or_data_path(args.sample_jsonl or spec.default_data, args.data_root)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dataset = QwenBenchmarkDataset(str(data_path), processor, spec.name, data_root=args.data_root,
                                   answer_instruction=args.answer_instruction, cache_dir=args.input_cache_dir or None)
    dataset.rows = dataset.rows[args.sample_index:args.sample_index + args.metric_samples]
    if not len(dataset):
        raise ValueError("No samples selected")
    # Process exactly once on CPU and reuse byte-identical tensors for every model.
    items = [dataset[i] for i in range(len(dataset))]
    input_hash = hashlib.sha256()
    for item in items:
        for key, value in sorted(item.items()):
            if torch.is_tensor(value):
                input_hash.update(key.encode())
                input_hash.update(str((tuple(value.shape), value.dtype)).encode())
                input_hash.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    dtype = dtype_from_name(args.dtype)
    rows, records = [], []
    base = None
    output = Path(args.output_json) if args.output_json else None
    protocol = {
        "model_path": args.model_path, "checkpoint": str(checkpoint), "data_path": str(data_path),
        "input_sha256": input_hash.hexdigest(), "sample_indices": [v.get("index") for v in items],
        "samples": len(items), "new_tokens": new_tokens, "runs": args.comparison_runs,
        "warmup_complete_requests_per_sample": args.metric_prefill_warmup,
        "dtype": args.dtype, "native_attention": attention, "adapter_prefix_attention": "sdpa",
        "base_pruning_deepstack": args.comparison_deepstack,
        "adapter_deepstack": "off",
        "adapter_prefill": "build_qwen_fast_adapter_prefill (shared with legacy metric table)",
        "adapter_cuda_graph": args.cuda_graph, "adapter_cuda_graph_context": args.cuda_graph_context,
        "base_baseline_execution": "native forward + cached decode",
        "adapter_decode_mode": args.comparison_decode_mode, "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "seed": args.seed,
        "timing_scope": "GPU-resident processed inputs through vision, selection/adapter, prefill and fixed-length greedy cached decode; excludes preprocessing, transfers, resource inspection and diagnostic audit I/O",
        "total_aggregation": "sum of per-sample medians of complete requests",
        "prefill_aggregation": "sum of per-sample medians of prefill portions",
        "flops_scope": "analytical decoder core prefill plus adapter projections; excludes vision, selectors, LM head, norms, softmax; 2 FLOPs/MAC; square attention convention",
        "cache_scope": "actual storage after prefill; kv_cache_mb = K/V only, decode_cache_mb = all retained decode state; MiB",
        "baseline_implementation": "repository Qwen ports with their prefill pruning decisions frozen for cached decode",
    }
    print("COMPARISON_PROTOCOL " + json.dumps(protocol), flush=True)
    with torch.inference_mode(), disable_prune_audit():
        for method in methods:
            load_method = "base" if method == "embedding_adapter" else method
            model, _ = load_baseline_model(load_method, args.model_path, dtype, str(device), 1.0, attention)
            model.eval().requires_grad_(False)
            handle = None
            if args.comparison_deepstack == "off":
                from src.qwen_deepstack import disable_qwen_deepstack
                disable_qwen_deepstack(model)
            adapter = None
            if method == "embedding_adapter":
                adapter, meta = load_qwen_embedding_adapter_checkpoint(checkpoint, model.model.language_model, device, dtype)
                if meta["missing"] or meta["unexpected"] or adapter.mode != "embedding_adapter":
                    raise ValueError(f"Checkpoint is not a matching static embedding adapter: {meta}")
                adapter.eval().requires_grad_(False)
            adapter_prefill_fn = build_qwen_fast_adapter_prefill(model, adapter, args) if adapter is not None else None
            runner = RequestRunner(model, method, adapter, args.comparison_decode_mode, adapter_prefill_fn)
            retentions = [1.0] if method in ("base", "embedding_adapter") else list(dict.fromkeys(args.retentions))
            for retention in retentions:
                method_records = []
                for i, item in enumerate(items):
                    inputs = _qwen_inputs_from_item(item, device)
                    visual = inputs["mm_token_type_ids"][0].ne(0)
                    image_tokens = int(visual.sum())
                    text_tokens = visual.numel() - image_tokens
                    if image_tokens == 0 or not bool(inputs["attention_mask"].all()):
                        raise ValueError("Comparison requires unpadded visual requests")
                    configure_baseline(model, load_method, retention, int(visual.nonzero()[0]), image_tokens)
                    seed = args.seed + args.sample_index + i
                    verification = None
                    if i == 0:
                        torch.manual_seed(seed)
                        verification = verify_cached_decode(runner, inputs)
                        print(f"VERIFY {method} retention={retention}: {verification}", flush=True)
                    for _ in range(args.metric_prefill_warmup):
                        torch.manual_seed(seed)
                        runner.request(inputs, new_tokens)
                    trials = []
                    for _ in range(args.comparison_runs):
                        torch.manual_seed(seed)
                        trials.append(runner.request(inputs, new_tokens, measure=True))
                    first = trials[0]
                    if any(t["generated_token_ids"] != first["generated_token_ids"] or
                           t["layer_cache_lengths"] != first["layer_cache_lengths"] for t in trials):
                        raise RuntimeError(f"Non-reproducible requests for {method}, sample {i}")
                    record = {**first, "method": method, "retention": retention,
                              "sample_index": args.sample_index + i, "source_index": item.get("index"),
                              "text_tokens": text_tokens, "image_tokens": image_tokens,
                              "verification": verification,
                              "flops": decoder_flops(model.model.language_model.config, first["layer_cache_lengths"],
                                                      text_tokens=text_tokens, image_tokens=image_tokens,
                                                      adapter_rank=adapter.visual_adapter_rank if adapter else None),
                              "trial_times": [{k: t[k] for k in ("total_time_s", "prefill_time_s", "decode_time_s")} for t in trials]}
                    for key in ("total_time_s", "prefill_time_s", "decode_time_s"):
                        record[key] = statistics.median(t[key] for t in trials)
                    method_records.append(record)
                    records.append(record)
                    if (i + 1) % args.log_every == 0:
                        print(f"{method} retention={retention:g} {i+1}/{len(items)} total={record['total_time_s']:.4f}s prefill={record['prefill_time_s']:.4f}s", flush=True)
                row = {"method": method, "retention": retention, "samples": len(items), "new_tokens": new_tokens,
                       "benchmark": spec.name, "input_sha256": input_hash.hexdigest(), "decode_mode": args.comparison_decode_mode if adapter else "native_cached",
                       "execution": "adapter_cuda_graph" if adapter is not None and args.cuda_graph else "eager",
                       "cuda_graph_context": bool(adapter is not None and args.cuda_graph and args.cuda_graph_context),
                       "attention": attention, "deepstack": "off"}
                for key in ("total_time_s", "prefill_time_s", "decode_time_s"):
                    row[key] = sum(r[key] for r in method_records)
                for key in ("kv_cache_mb", "decode_cache_mb", "final_kv_cache_mb", "flops", "text_tokens", "image_tokens"):
                    row[key] = statistics.mean(r[key] for r in method_records)
                row["prefilling_time_s"] = row["prefill_time_s"]
                if method == "base":
                    base = row
                row["speedup_total"] = base["total_time_s"] / row["total_time_s"]
                row["speedup_prefilling"] = base["prefill_time_s"] / row["prefill_time_s"]
                rows.append(row)
                write_outputs(rows, args)
                if output:
                    output.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
                    output.with_suffix(".samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                    output.with_suffix(".md").write_text(comparison_markdown(rows))
            if handle is not None:
                handle.remove()
            del runner, adapter_prefill_fn, adapter, model
            gc.collect()
            torch.cuda.empty_cache()
    print(comparison_markdown(rows), flush=True)
