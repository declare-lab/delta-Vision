"""Channel intervention with native visual states restored at every decoder layer.

Native visual layer inputs are captured once per prompt. Each intervened forward
replaces visual rows with a projection of that layer's native cache; text rows
always come from the intervened trajectory. This is an oracle diagnostic, not a
deployable compressed model or a measurement of inference acceleration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from PIL import Image

from src.benchmarks import get_benchmark_spec, score_prediction
from src.data import LlavaBenchmarkDataset, QwenBenchmarkDataset
from src.model_setup import disable_qwen_deepstack
from analysis.fig01a_hidden_channels.visual_channel_rank_grid import _layers, _to_device_item, load_model

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen": str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
    "llava": str(Path(__file__).resolve().parents[2] / "model/llava-1.5-7b-hf"),
}
BENCHMARKS = ("mmstar", "sqa", "realworldqa")
RANKS = (0, 32, 64, 128, 256, 512, 1024)
PROTOCOL = "native_visual_cache_shared_channel_pca_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(row, root):
    raw = row.get("images", [row.get("image")])
    if len(raw) != 1 or raw[0] is None:
        raise ValueError("This experiment requires one image per prompt")
    return (Path(row.get("image_root") or root) / str(raw[0])).resolve()


def pixel_digest(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        h = hashlib.sha256(str(rgb.size).encode())
        h.update(rgb.tobytes())
        return h.hexdigest()


def prepare(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["protocol"] == PROTOCOL
        assert plan["sample_limit"] == args.samples
        assert plan["calibration_samples"] == args.calibration_samples
        assert plan["seed"] == args.seed
        for name, info in plan["manifests"].items():
            assert digest(root / info["file"]) == info["sha256"], name
        return plan
    manifests = {}
    eval_hashes = set()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for benchmark in BENCHMARKS:
            source = ROOT / get_benchmark_spec(benchmark).default_data
            rows = load_rows(source)[:args.samples]
            paths = [image_path(row, source.parent) for row in rows]
            hashes = list(pool.map(pixel_digest, paths))
            eval_hashes.update(hashes)
            for row, path, image_hash in zip(rows, paths, hashes):
                row["image"] = str(path)
                row.pop("images", None)
                row["pixel_sha256"] = image_hash
            path = root / f"{benchmark}_eval.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            manifests[benchmark] = {"file": path.name, "sha256": digest(path), "samples": len(rows),
                                    "source": str(source)}
            print("MANIFEST", benchmark, len(rows), flush=True)
    source = ROOT / "data/train/pixmo/pixmo_ama_full_valid.clean.jsonl"
    rows = load_rows(source)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    selected, seen_paths, seen_pixels = [], set(), set()
    overlap = 0
    for idx in order:
        row = rows[idx]
        path = image_path(row, source.parent)
        if str(path) in seen_paths:
            continue
        seen_paths.add(str(path))
        image_hash = pixel_digest(path)
        if image_hash in eval_hashes:
            overlap += 1
            continue
        if image_hash in seen_pixels:
            continue
        seen_pixels.add(image_hash)
        selected.append({"index": idx, "image": str(path), "question": row["question"],
                         "pixel_sha256": image_hash})
        if len(selected) == args.calibration_samples:
            break
    if len(selected) != args.calibration_samples:
        raise ValueError("Insufficient disjoint calibration images")
    path = root / "pixmo_calibration.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    manifests["calibration"] = {"file": path.name, "sha256": digest(path), "samples": len(selected),
                                "source": str(source), "labels_included": False}
    plan = {"protocol": PROTOCOL, "created": time.time(), "models": MODELS,
            "sample_limit": args.samples, "calibration_samples": args.calibration_samples,
            "seed": args.seed, "ranks": list(RANKS), "manifests": manifests,
            "calibration_eval_pixel_overlap": 0, "excluded_overlapping_images": overlap,
            "attention": "flash_attention_2", "qwen_deepstack": "off", "dtype": "bfloat16",
            "basis": "one independent Pixmo basis per model and layer shared by all benchmarks",
            "covariance": "float64, all visual rows, no row sampling",
            "generation": "greedy; no KV cache; EOS or benchmark max_new_tokens; no label-based stopping",
            "intervention": "replace every layer's visual input with projected native visual cache; preserve current text"}
    dump_json(plan_path, plan)
    print("PREPARED", plan_path, flush=True)
    return plan


class NativeVisualHook:
    def __init__(self, layers):
        self.layers = list(layers)
        self.mode = "off"
        self.positions = None
        self.native = {}
        self.replacements = {}
        self.cov = {}
        self.sums = {}
        self.counts = {}
        self.calls = {}
        self.handles = [layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                        for i, layer in enumerate(self.layers)]

    def begin(self, mode):
        self.mode = mode
        self.calls = {}
        if mode == "capture":
            # Dictionaries can be aliased by a saved per-sample native cache.
            # Rebind them; never mutate a previously supplied read-only cache.
            self.native = {}
            self.replacements = {}

    def check(self):
        if self.mode != "off" and self.calls != {i: 1 for i in range(len(self.layers))}:
            raise AssertionError(f"Unexpected intervention calls: {self.calls}")

    def _hook(self, layer):
        def hook(module, args, kwargs):
            if self.mode == "off":
                return
            h = kwargs.get("hidden_states", args[0] if args else None)
            if h is None or h.ndim != 3 or h.shape[0] != 1:
                raise AssertionError("Expected one prompt with explicit visual token positions")
            self.calls[layer] = self.calls.get(layer, 0) + 1
            visual = h.index_select(1, self.positions)
            if self.mode == "capture":
                self.native[layer] = visual.detach().clone()
                return
            if self.mode == "collect":
                x = visual[0].double()
                if layer not in self.cov:
                    self.cov[layer] = torch.zeros((x.shape[-1], x.shape[-1]), device=x.device, dtype=torch.float64)
                    self.sums[layer] = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                    self.counts[layer] = 0
                self.cov[layer].addmm_(x.T, x)
                self.sums[layer].add_(x.sum(0))
                self.counts[layer] += len(x)
                return
            if self.mode != "replay":
                raise ValueError(self.mode)
            replacement = self.replacements[layer]
            if replacement.shape != visual.shape:
                raise AssertionError((replacement.shape, visual.shape))
            out = h.clone()
            out.index_copy_(1, self.positions, replacement)
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": out}
            return (out, *args[1:]), kwargs
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def configure_runtime():
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)


def get_model(kind):
    processor, model = load_model(kind, MODELS[kind], "bfloat16", "flash_attention_2", torch.device("cuda:0"))
    if kind == "qwen":
        disable_qwen_deepstack(model)
    assert model.config.text_config._attn_implementation == "flash_attention_2"
    return processor, model


def dataset(root, kind, processor, name):
    cls = QwenBenchmarkDataset if kind == "qwen" else LlavaBenchmarkDataset
    filename = "pixmo_calibration.jsonl" if name == "calibration" else f"{name}_eval.jsonl"
    return cls(str(root / filename), processor, "gqa" if name == "calibration" else name,
               answer_instruction="" if name == "calibration" else None)


def positions(kind, model, inputs):
    mask = inputs["mm_token_type_ids"].eq(1) if kind == "qwen" else inputs["input_ids"].eq(model.config.image_token_index)
    pos = mask[0].nonzero(as_tuple=True)[0]
    if not len(pos):
        raise ValueError("No visual tokens")
    return pos


def forward(model, inputs, hook, *, body_only=False):
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    hook.begin(hook.mode)
    kwargs = {k: v for k, v in inputs.items() if k in {
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids", "image_sizes"}}
    if body_only:
        out = model.model(**kwargs, use_cache=False, return_dict=True)
    else:
        out = model(**kwargs, use_cache=False, return_dict=True, logits_to_keep=1)
    hook.check()
    return out


def metadata(root, kind):
    return {"protocol": PROTOCOL, "model_kind": kind, "plan_sha256": digest(root / "plan.json"),
            "model_path": MODELS[kind], "model_config_sha256": digest(Path(MODELS[kind]) / "config.json"),
            "attention": "flash_attention_2", "qwen_deepstack": "off", "dtype": "bfloat16"}


@torch.inference_mode()
def collect(args):
    configure_runtime()
    root = Path(args.output)
    processor, model = get_model(args.model_kind)
    ds = dataset(root, args.model_kind, processor, "calibration")
    hook = NativeVisualHook(_layers(args.model_kind, model))
    indices = list(range(args.shard, len(ds), args.world))
    begin = time.time()
    for c, idx in enumerate(indices):
        inp = _to_device_item(ds[idx], torch.device("cuda:0"))
        hook.positions = positions(args.model_kind, model, inp)
        hook.mode = "collect"
        forward(model, inp, hook, body_only=True)
        if (c + 1) % 16 == 0 or c + 1 == len(indices):
            print("COLLECT", args.model_kind, args.shard, c + 1, len(indices), round(time.time() - begin, 1), flush=True)
    payload = {**metadata(root, args.model_kind), "shard": args.shard, "world": args.world,
               "indices": indices, "layers": {str(l): {"gram": hook.cov[l].cpu(), "sum": hook.sums[l].cpu(),
                                                      "count": hook.counts[l]} for l in hook.cov}}
    path = root / args.model_kind / f"cov_{args.shard}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    hook.close()
    print("SAVED", path, flush=True)


@torch.inference_mode()
def merge(args):
    configure_runtime()
    root = Path(args.output)
    shards = [torch.load(root / args.model_kind / f"cov_{s}.pt", map_location="cpu", weights_only=False, mmap=True)
              for s in range(args.world)]
    expected = metadata(root, args.model_kind)
    ids = []
    for s, shard in enumerate(shards):
        assert all(shard[k] == v for k, v in expected.items())
        assert shard["shard"] == s and shard["world"] == args.world
        ids.extend(shard["indices"])
    plan = json.loads((root / "plan.json").read_text())
    assert sorted(ids) == list(range(plan["calibration_samples"]))
    result = {**expected, "calibration_samples": len(ids), "basis_precision": "float64", "layers": {}}
    keys = sorted(shards[0]["layers"], key=int)[args.layer_shard::args.layer_world]
    for key in keys:
        entries = [s["layers"][key] for s in shards]
        gram = sum(e["gram"] for e in entries).cuda()
        sm = sum(e["sum"] for e in entries).cuda()
        count = sum(e["count"] for e in entries)
        centered = gram - torch.outer(sm, sm) / count
        centered = (centered + centered.T) * 0.5
        vals, vecs = torch.linalg.eigh(centered)
        result["layers"][key] = {"mean": (sm / count).cpu(),
                                 "basis": vecs.flip(1).contiguous().cpu(),
                                 "eigenvalues": vals.flip(0).clamp_min(0).cpu(), "rows": count}
        print("BASIS", args.model_kind, key, count, flush=True)
        del gram, sm, centered, vals, vecs
    filename = "basis.pt" if args.layer_world == 1 else f"basis_part_{args.layer_shard}.pt"
    path = root / args.model_kind / filename
    tmp = path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.replace(path)
    print("SAVED", path, flush=True)


def assemble(root, kind, layer_world):
    expected = metadata(root, kind)
    merged = {**expected, "basis_precision": "float64", "layers": {}}
    for part in range(layer_world):
        payload = torch.load(root / kind / f"basis_part_{part}.pt", map_location="cpu", weights_only=False, mmap=True)
        assert all(payload[k] == v for k, v in expected.items())
        assert payload["basis_precision"] == "float64"
        assert not (set(merged["layers"]) & set(payload["layers"]))
        merged["layers"].update(payload["layers"])
        merged["calibration_samples"] = payload["calibration_samples"]
    from transformers import AutoConfig
    # LLaVA's serialized text config can omit Llama defaults such as depth.
    config = AutoConfig.from_pretrained(MODELS[kind]).text_config
    assert sorted(map(int, merged["layers"])) == list(range(config.num_hidden_layers))
    tmp = root / kind / "basis.tmp"
    torch.save(merged, tmp)
    tmp.replace(root / kind / "basis.pt")


def extend(inputs, token):
    new = torch.tensor([[token]], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    out = {**inputs, "input_ids": torch.cat((inputs["input_ids"], new), 1),
           "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(new)), 1)}
    if "mm_token_type_ids" in out:
        out["mm_token_type_ids"] = torch.cat((out["mm_token_type_ids"], torch.zeros_like(new)), 1)
    return out


def generate(model, processor, inp, hook, first_logits, max_tokens):
    logits = first_logits
    eos = set()
    for value in [processor.tokenizer.eos_token_id, model.generation_config.eos_token_id]:
        eos.update(value if isinstance(value, list) else [value])
    generated = []
    current = inp
    for step in range(max_tokens):
        token = int(logits.argmax())
        generated.append(token)
        if token in eos or step + 1 == max_tokens:
            break
        current = extend(current, token)
        logits = forward(model, current, hook).logits[0, -1].float()
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip(), generated


def project_cache(native, bases, rank):
    projected = {}
    residual = 0.0
    energy = 0.0
    for l, x in native.items():
        mean, basis = (t.double() for t in bases[l])
        xc = x.double() - mean
        b = basis[:, :rank]
        y = (xc @ b) @ b.T + mean if rank else mean.expand_as(xc)
        projected[l] = y.to(x.dtype)
        residual += float((y - x.double()).square().sum())
        energy += float(xc.square().sum())
    return projected, residual / max(energy, 1e-30)


@torch.inference_mode()
def evaluate(args):
    configure_runtime()
    root = Path(args.output)
    meta = metadata(root, args.model_kind)
    processor, model = get_model(args.model_kind)
    ds = dataset(root, args.model_kind, processor, args.benchmark)
    payload = torch.load(root / args.model_kind / "basis.pt", map_location="cpu", weights_only=False, mmap=True)
    assert all(payload[k] == v for k, v in meta.items())
    assert payload["basis_precision"] == "float64"
    bases = {int(l): (e["mean"].cuda().view(1, 1, -1), e["basis"].cuda()) for l, e in payload["layers"].items()}
    hook = NativeVisualHook(_layers(args.model_kind, model))
    features = []
    original_features = model.model.get_image_features
    def cached_features(*a, **kw):
        if not features:
            features.append(original_features(*a, **kw))
        return features[0]
    model.model.get_image_features = cached_features
    path = root / args.model_kind / f"{args.benchmark}_{args.shard}.jsonl"
    meta_path = path.with_suffix(".meta.json")
    run_meta = {**meta, "benchmark": args.benchmark, "shard": args.shard, "world": args.world,
                "samples": len(ds), "ranks": list(RANKS), "projection_precision": "float64",
                "max_new_tokens": get_benchmark_spec(args.benchmark).max_new_tokens}
    completed = set()
    if path.exists():
        assert json.loads(meta_path.read_text()) == run_meta
        # Discard only an incomplete final write when resuming after interruption.
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n") + 1]
            path.write_bytes(raw)
        for row in load_rows(path):
            assert row["sample"] not in completed
            completed.add(row["sample"])
    else:
        dump_json(meta_path, run_meta)
    indices = list(range(args.shard, len(ds), args.world))
    assert completed.issubset(indices)
    started = time.time()
    n_new = 0
    spec = get_benchmark_spec(args.benchmark)
    with path.open("a", buffering=1) as f:
        for idx in indices:
            if idx in completed:
                continue
            item = ds[idx]
            inp = _to_device_item(item, torch.device("cuda:0"))
            hook.positions = positions(args.model_kind, model, inp)
            features.clear()
            hook.mode = "capture"
            native_logits = forward(model, inp, hook).logits[0, -1].float()
            native_cache = dict(hook.native)
            hook.mode = "off"
            native_text, native_tokens = generate(model, processor, inp, hook, native_logits, spec.max_new_tokens)
            native_logp = native_logits.log_softmax(-1)
            results = {}
            def score(text, tokens, logits, **extra):
                scored = score_prediction(metric=spec.metric, prediction_text=text, answer=item.get("answer"),
                                          answers=item.get("answers"), choices=item.get("choices"),
                                          question=item["row"].get("question"))
                kl = float((native_logp.exp() * (native_logp - logits.log_softmax(-1))).sum())
                return {"text": text, "tokens": tokens, **scored, "first_token_kl": kl, **extra}
            results["native"] = score(native_text, native_tokens, native_logits)
            # No compression: restoring native visual rows must reproduce native.
            hook.replacements = native_cache
            hook.mode = "replay"
            replay_logits = forward(model, inp, hook).logits[0, -1].float()
            replay_text, replay_tokens = generate(model, processor, inp, hook, replay_logits, spec.max_new_tokens)
            replay_diff = float((native_logits - replay_logits).abs().max())
            if replay_diff != 0 or replay_tokens != native_tokens:
                raise AssertionError(("Native cache identity failed", idx, replay_diff, native_tokens, replay_tokens))
            checks = {"native_replay_logit_max_abs": replay_diff, "native_replay_tokens_equal": True}
            # Test true full-width projection and native visual-cache invariance
            # to an appended text token once per worker and benchmark.
            if n_new == 0:
                width = next(iter(bases.values()))[1].shape[0]
                hook.replacements, err = project_cache(native_cache, bases, width)
                hook.mode = "replay"
                full_logits = forward(model, inp, hook).logits[0, -1].float()
                full_diff = float((native_logits - full_logits).abs().max())
                if err > 1e-10 or full_diff > 0.25 or int(full_logits.argmax()) != int(native_logits.argmax()):
                    raise AssertionError(("Full width projection failed", err, full_diff))
                checks.update(full_width_relative_squared_error=err, full_width_logit_max_abs=full_diff)
                extra = extend(inp, native_tokens[0])
                hook.mode = "capture"
                forward(model, extra, hook)
                cache_err = max(float((hook.native[l].float() - x.float()).abs().max()) for l, x in native_cache.items())
                checks["native_visual_cache_after_appended_token_max_abs"] = cache_err
                # Causal prefix should be invariant, within BF16 kernel rounding.
                if cache_err > 0.125:
                    raise AssertionError(("Native visual prefix changed", cache_err))
                hook.native = native_cache
            for rank in RANKS:
                hook.replacements, reconstruction = project_cache(native_cache, bases, rank)
                hook.mode = "replay"
                logits = forward(model, inp, hook).logits[0, -1].float()
                text, tokens = generate(model, processor, inp, hook, logits, spec.max_new_tokens)
                results[f"r{rank}"] = score(text, tokens, logits, relative_squared_reconstruction_error=reconstruction)
            rec = {"sample": idx, "index": item["index"], "visual_tokens": len(hook.positions),
                   "checks": checks, "results": results}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 10 == 0 or idx == indices[-1]:
                print("EVAL", args.model_kind, args.benchmark, args.shard, len(completed) + n_new, len(indices),
                      round(time.time() - started, 1), flush=True)
    hook.close()
    model.model.get_image_features = original_features
    print("SAVED", path, flush=True)


def report(args):
    import numpy as np
    root = Path(args.output)
    plan = json.loads((root / "plan.json").read_text())
    out = {"protocol": PROTOCOL, "plan_sha256": digest(root / "plan.json"), "results": {}}
    lines = ["# Native visual-cache channel intervention", "", "FA2; Qwen DeepStack off; all visual tokens retained.",
             "Each layer restores projected native visual states; text follows the intervened trajectory.",
             "Independent Pixmo calibration, no evaluation images or answer labels used for basis fitting.", "",
             "| Model | Dataset | N | Native | r0 | r32 | r64 | r128 | r256 | r512 | r1024 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in MODELS:
        for benchmark in BENCHMARKS:
            rows = []
            files = sorted((root / kind).glob(f"{benchmark}_[0-9]*.jsonl"))
            for path in files:
                rows.extend(load_rows(path))
            expected = plan["manifests"][benchmark]["samples"]
            if len(rows) != expected or sorted(row["sample"] for row in rows) != list(range(expected)):
                raise AssertionError(("Incomplete or duplicate results", kind, benchmark, len(rows), expected))
            native_scores = np.array([row["results"]["native"]["score"] for row in rows])
            summary = {}
            for condition in ("native", *(f"r{r}" for r in RANKS)):
                entries = [row["results"][condition] for row in rows]
                scores = np.array([e["score"] for e in entries])
                differences = scores - native_scores
                rng = np.random.default_rng(44)
                bootstrap = differences[rng.integers(0, len(rows), (5000, len(rows)))].mean(1) * 100
                summary[condition] = {"accuracy_pct": float(scores.mean() * 100), "samples": len(rows),
                                      "delta_accuracy_pp": float(differences.mean() * 100),
                                      "paired_bootstrap_delta_95ci_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
                                      "agreement_pct": 100 * sum(e["prediction"] == row["results"]["native"]["prediction"]
                                                                 for row, e in zip(rows, entries)) / len(rows),
                                      "mean_first_token_kl": sum(e["first_token_kl"] for e in entries) / len(rows)}
            out["results"][f"{kind}/{benchmark}"] = summary
            values = [f"{summary[c]['accuracy_pct']:.2f}" for c in ("native", *(f"r{r}" for r in RANKS))]
            lines.append("| " + " | ".join([kind, benchmark, str(len(rows)), *values]) + " |")
    lines.extend(["", "These results measure channel compressibility with native visual-state restoration. They do not establish that all visual layers can be compressed without native caches."])
    dump_json(root / "summary.json", out)
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def run_group(commands, root, phase):
    running = []
    dump_json(root / "status.json", {"phase": phase, "started": time.time(), "jobs": commands})
    for job in commands:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        path = root / "logs" / f"{job['name']}.log"
        path.parent.mkdir(exist_ok=True)
        handle = path.open("a")
        cmd = [sys.executable, "-u", "-m", 'analysis.fig01a_hidden_channels.visual_channel_native_cache', *job["args"], "--output", str(root)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        running.append((job, proc))
        print("START", job["name"], "gpu", job["gpu"], "pid", proc.pid, flush=True)
    failed = []
    for job, proc in running:
        rc = proc.wait()
        print("DONE", job["name"], rc, flush=True)
        if rc:
            failed.append({"job": job, "returncode": rc})
    if failed:
        dump_json(root / "status.json", {"phase": phase, "failed": failed})
        raise RuntimeError(failed)


def launch(args):
    root = Path(args.output).resolve()
    prepare(args)
    gpus = args.gpus
    if len(gpus) < 2 or len(gpus) % 2 or len(set(gpus)) != len(gpus):
        raise ValueError("Use an even number of distinct GPUs, split between the two models")
    world = len(gpus) // 2
    jobs = []
    for m, kind in enumerate(MODELS):
        for shard in range(world):
            path = root / kind / f"cov_{shard}.pt"
            if not path.exists():
                jobs.append({"name": f"collect_{kind}_{shard}", "gpu": gpus[m * world + shard],
                             "args": ["collect", "--model-kind", kind, "--shard", str(shard), "--world", str(world)]})
    if jobs:
        run_group(jobs, root, "calibration")
    jobs = [{"name": f"merge_{kind}_part{part}", "gpu": gpus[m * world + part],
             "args": ["merge", "--model-kind", kind, "--world", str(world),
                      "--layer-shard", str(part), "--layer-world", str(world)]}
            for m, kind in enumerate(MODELS) for part in range(world)
            if not (root / kind / "basis.pt").exists() and not (root / kind / f"basis_part_{part}.pt").exists()]
    if jobs:
        run_group(jobs, root, "basis_eigendecomposition")
    for kind in MODELS:
        if not (root / kind / "basis.pt").exists():
            assemble(root, kind, world)
    for benchmark in BENCHMARKS:
        jobs = [{"name": f"eval_{kind}_{benchmark}_{shard}", "gpu": gpus[m * world + shard],
                 "args": ["eval", "--model-kind", kind, "--benchmark", benchmark, "--shard", str(shard), "--world", str(world)]}
                for m, kind in enumerate(MODELS) for shard in range(world)]
        run_group(jobs, root, f"evaluation_{benchmark}")
    report(args)
    dump_json(root / "status.json", {"phase": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "collect", "merge", "eval", "report", "launch"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-kind", choices=list(MODELS))
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--layer-shard", type=int, default=0)
    parser.add_argument("--layer-world", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "merge": merge, "eval": evaluate,
     "report": report, "launch": launch}[args.command](args)


if __name__ == "__main__":
    main()
