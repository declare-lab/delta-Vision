"""Independent native-query comparisons of existing adapter visual memories.

Each sample has ONE untouched native forward, with DeepStack disabled. Every
tested layer reads that native forward's states. Diagnostic attention outputs
never enter another layer. There is no joint intervention or generation score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from src.data import QwenBenchmarkDataset
from src.model import load_qwen_embedding_adapter_checkpoint, qwen_apply_rotary_pos_emb
from src.visual_channel_native_cache import configure_runtime, dump_json, get_model, MODELS
from src.visual_channel_rank_grid import _to_device_item

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/diagnostics/channel_native_cache_20260916"
CHECKPOINT = ROOT / "artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt"
CHECKPOINT_SHA = "8c5097a49083947093d1105215849451517d5e237c679fcc4765dd34b52f9e74"
LAYERS = tuple(range(13, 23))
DATASETS = {"mmstar": 1000, "realworldqa": 765}
PROTOCOL = "adapter_independent_layer_native_query_fa2_v1"
INPUT_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root):
    sources = [Path(__file__), ROOT / "src/model.py", ROOT / "src/data.py",
               ROOT / "src/benchmarks.py", ROOT / "src/qwen_deepstack.py"]
    plan = dict(protocol=PROTOCOL, model=MODELS["qwen"], hidden_dimension=2560,
                checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
                layers=list(LAYERS), layer_indexing="zero based",
                attention="flash_attention_2", dtype="bfloat16", deepstack="off",
                adapter="existing static Pixmo KL 2000-step; initial E -> independent per-layer memory",
                sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                datasets={name: dict(path=str(DATA / f"{name}_eval.jsonl"), samples=n,
                                    sha256=sha(DATA / f"{name}_eval.jsonl"))
                          for name, n in DATASETS.items()},
                hidden="visual input to decoder layer, before input RMSNorm; same token positions",
                fixed="native text queries, native text K/V, M-RoPE, full causal softmax denominator",
                query_rows="all text after the last image token; answer-boundary token also reported separately",
                full_output="attention output after W_O, before residual addition",
                visual_contribution="W_O sum_visual softmax(Q K_all^T)_visual V_visual; no bias; text values zeroed while ALL keys remain in denominator",
                isolation="native forward only; each layer comparison is independent; diagnostic outputs never propagated",
                aggregation="primary: average per-token within sample then equal average of samples; also save pooled-token metrics",
                cosine="mean per-token vector cosine; zero-norm pairs excluded and counts reported",
                mse="mean squared element error; FP64 statistics",
                scope="single-layer hidden and attention comparisons only; no joint replacement or accuracy evaluation")
    assert plan["checkpoint_sha256"] == CHECKPOINT_SHA
    path = root / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan, "Existing output uses another protocol/source"
    else:
        dump_json(path, plan)
    return plan


def vector_stats(pred, native):
    """Sufficient statistics plus equal-sample metrics; reduce in float64."""
    assert pred.shape == native.shape and pred.ndim == 2
    p, y = pred.double(), native.double()
    p2, y2 = p.square().sum(-1), y.square().sum(-1)
    denom = (p2 * y2).sqrt()
    valid = denom > 1e-20
    cos = ((p * y).sum(-1) / denom.clamp_min(1e-20)).clamp(-1, 1)
    packed = torch.stack(((p - y).square().sum(), y2.sum(), p2.sum(),
                          cos.masked_fill(~valid, 0).sum(), valid.double().sum())).cpu().tolist()
    assert all(math.isfinite(x) for x in packed), packed
    sse, energy, pred_energy, cos_sum, nvalid = packed
    n, d = y.shape
    return dict(n=n, d=d, sse=sse, energy=energy, pred_energy=pred_energy,
                cosine_sum=cos_sum, valid_cosines=int(nvalid), mse=sse / (n * d),
                cosine=cos_sum / nvalid if nvalid else None,
                relative_l2=math.sqrt(sse / energy) if energy > 1e-20 else None,
                native_rms=math.sqrt(energy / (n * d)),
                pred_rms=math.sqrt(pred_energy / (n * d)))


class Capture:
    """Read-only native layer and attention inputs/outputs."""
    def __init__(self, model):
        self.layers = model.model.language_model.layers
        self.enabled = True
        self.handles = []
        for l in (0, *LAYERS):
            self.handles.append(self.layers[l].register_forward_pre_hook(self.layer_hook(l), with_kwargs=True))
        for l in LAYERS:
            self.handles.append(self.layers[l].self_attn.register_forward_pre_hook(self.attn_pre(l), with_kwargs=True))
            self.handles.append(self.layers[l].self_attn.register_forward_hook(self.attn_post(l)))
        self.reset()

    def reset(self):
        self.h, self.norm, self.rope, self.out, self.calls = {}, {}, {}, {}, {}

    def layer_hook(self, l):
        def hook(module, args, kwargs):
            if self.enabled:
                h = kwargs.get("hidden_states", args[0] if args else None)
                self.h[l] = h.detach().clone()
                self.calls[l] = self.calls.get(l, 0) + 1
        return hook

    def attn_pre(self, l):
        def hook(module, args, kwargs):
            if self.enabled:
                assert kwargs.get("attention_mask") is None, "Expected unpadded native FA2"
                self.norm[l] = kwargs["hidden_states"].detach().clone()
                self.rope[l] = tuple(x.detach().clone() for x in kwargs["position_embeddings"])
        return hook

    def attn_post(self, l):
        def hook(module, args, output):
            if self.enabled:
                self.out[l] = output[0].detach().clone()
        return hook

    def check(self):
        assert self.calls == {l: 1 for l in (0, *LAYERS)}, self.calls

    def close(self):
        for handle in self.handles:
            handle.remove()


def readout(attn, query, key, value, visual, *, visual_only=False):
    """Query is the final contiguous text run; FA2 bottom-right mask is exact."""
    from flash_attn.flash_attn_interface import flash_attn_func
    if visual_only:
        masked = torch.zeros_like(value)
        masked.index_copy_(2, visual, value.index_select(2, visual))
        value = masked
    heads = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                            dropout_p=0., softmax_scale=float(attn.scaling), causal=True)
    flat = heads.reshape(1, query.shape[2], -1)
    # An output-projection bias belongs to total output, not visual contribution.
    output = F.linear(flat, attn.o_proj.weight, None if visual_only else attn.o_proj.bias)
    return output[0]


def projected_kv(layer, memory, rope, visual):
    a = layer.self_attn
    n = layer.input_layernorm(memory)
    shape = (*n.shape[:-1], -1, a.head_dim)
    k = a.k_norm(a.k_proj(n).view(shape)).transpose(1, 2)
    v = a.v_proj(n).view(shape).transpose(1, 2)
    r = tuple(t.index_select(1, visual) for t in rope)
    # Same key normalization and rotary function as the maintained adapter.
    _, k = qwen_apply_rotary_pos_emb(k, k, *r)
    return k, v


def layer_comparison(cap, adapter, l, visual, query_start, *, validate=False):
    layer = cap.layers[l]
    a = layer.self_attn
    native_hidden = cap.h[l].index_select(1, visual)
    initial = cap.h[0].index_select(1, visual)
    memory = adapter.visual_memory_for_layer(initial, l)
    n = cap.norm[l]
    shape = (*n.shape[:-1], -1, a.head_dim)
    q = a.q_norm(a.q_proj(n).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(n).view(shape)).transpose(1, 2)
    v = a.v_proj(n).view(shape).transpose(1, 2)
    q, k = qwen_apply_rotary_pos_emb(q, k, *cap.rope[l])
    q = q[:, :, query_start:]
    ak, av = projected_kv(layer, memory, cap.rope[l], visual)
    ka, va = k.clone(), v.clone()
    ka.index_copy_(2, visual, ak)
    va.index_copy_(2, visual, av)
    text = torch.ones(k.shape[2], device=k.device, dtype=torch.bool)
    text[visual] = False
    assert torch.equal(k[:, :, text], ka[:, :, text])
    assert torch.equal(v[:, :, text], va[:, :, text])
    native_out = readout(a, q, k, v, visual)
    adapter_out = readout(a, q, ka, va, visual)
    native_vis = readout(a, q, k, v, visual, visual_only=True)
    adapter_vis = readout(a, q, ka, va, visual, visual_only=True)
    reconstruction = vector_stats(native_out, cap.out[l][0, query_start:])
    # Capture numerical error from query slicing and BF16 projection shapes.
    assert reconstruction["relative_l2"] < .01, (l, reconstruction)
    metrics = {"hidden": vector_stats(memory[0], native_hidden[0])}
    for group, sel in (("all_text", slice(None)), ("answer_boundary", slice(-1, None))):
        metrics[f"{group}/output"] = vector_stats(adapter_out[sel], native_out[sel])
        metrics[f"{group}/visual_contribution"] = vector_stats(adapter_vis[sel], native_vis[sel])
    checks = dict(native_reconstruction_relative_l2=reconstruction["relative_l2"])
    if validate:
        repeat = readout(a, q, k.clone(), v.clone(), visual)
        assert torch.equal(repeat, native_out), "Same memory must produce identical readout"
        ik, iv = projected_kv(layer, native_hidden, cap.rope[l], visual)
        ki, vi = k.clone(), v.clone()
        ki.index_copy_(2, visual, ik)
        vi.index_copy_(2, visual, iv)
        identity = readout(a, q, ki, vi, visual)
        identity_stats = vector_stats(identity, native_out)
        assert identity_stats["relative_l2"] < .01, (l, identity_stats)
        # Positive control: removing visual values must remove visual contribution.
        zero = v.clone()
        zero.index_fill_(2, visual, 0)
        empty = readout(a, q, k, zero, visual, visual_only=True)
        assert empty.abs().max() == 0 and native_vis.norm() > 0
        # Independent FP32 softmax validates causal indexing and full denominator.
        groups = q.shape[1] // k.shape[1]
        kr = k.repeat_interleave(groups, dim=1).float()
        vr = v.repeat_interleave(groups, dim=1).float()
        scores = q.float() @ kr.transpose(-1, -2) * float(a.scaling)
        visible = torch.arange(k.shape[2], device=k.device)[None, :] <= torch.arange(query_start, k.shape[2], device=k.device)[:, None]
        weights = scores.masked_fill(~visible[None, None], -float("inf")).softmax(-1)
        vh = (weights.index_select(-1, visual) @ vr.index_select(2, visual)).transpose(1, 2).reshape(1, q.shape[2], -1)
        reference = F.linear(vh, a.o_proj.weight.float(), None)[0]
        reference_stats = vector_stats(native_vis, reference)
        assert reference_stats["relative_l2"] < .01, (l, reference_stats)
        checks.update(exact_identity_max_abs=0., native_memory_reprojection_relative_l2=identity_stats["relative_l2"],
                      fp32_visual_reference_relative_l2=reference_stats["relative_l2"],
                      native_visual_softmax_mass=float(weights.index_select(-1, visual).sum(-1).mean()),
                      zero_visual_contribution_max_abs=0.)
    return metrics, checks


def setup():
    configure_runtime()
    processor, model = get_model("qwen")
    adapter, meta = load_qwen_embedding_adapter_checkpoint(CHECKPOINT, model.model.language_model,
                                                          torch.device("cuda:0"), torch.bfloat16)
    assert not meta["missing"] and not meta["unexpected"], meta
    assert adapter.mode == "embedding_adapter" and adapter.visual_adapter_rank == 128
    assert adapter.hidden_size == 2560 and adapter.num_layers == 36
    assert model._benchmark_deepstack == "off" and model.model.visual.deepstack_visual_indexes == []
    assert model.config.text_config._attn_implementation == "flash_attention_2"
    assert model.config.vision_config._attn_implementation == "flash_attention_2"
    return processor, model, adapter, Capture(model), meta


def native_forward(model, cap, inputs):
    cap.reset()
    model.model.rope_deltas = None
    result = model.model(**inputs, use_cache=False, return_dict=True)
    if cap.enabled:
        cap.check()
    return result.last_hidden_state


@torch.inference_mode()
def worker(args, smoke=False):
    root = Path(args.output)
    plan = prepare(root)
    processor, model, adapter, cap, meta = setup()
    start = time.time()
    path = root / ("smoke_rows.jsonl" if smoke else f"rows_{args.shard}.jsonl")
    assert not path.exists(), f"Refusing to overwrite {path}"
    checks, completed = [], 0
    with path.open("w", buffering=1) as f:
        for name, spec in plan["datasets"].items():
            ds = QwenBenchmarkDataset(spec["path"], processor, name)
            assert len(ds) == spec["samples"]
            indices = [0] if smoke else list(range(args.shard, len(ds), args.world))
            for count, index in enumerate(indices):
                item = _to_device_item(ds[index], torch.device("cuda:0"))
                inputs = {k: v for k, v in item.items() if k in INPUT_KEYS}
                assert inputs["input_ids"].shape[0] == 1 and inputs["attention_mask"].bool().all()
                types = inputs["mm_token_type_ids"][0]
                visual = types.eq(1).nonzero().flatten()
                assert len(visual) > 0 and types.max() == 1
                assert torch.equal(visual, torch.arange(visual[0], visual[-1] + 1, device=visual.device)), "Single contiguous image required"
                query_start = int(visual[-1]) + 1
                assert query_start < len(types) and (types[query_start:] == 0).all()
                native = native_forward(model, cap, inputs)
                this_check = dict(dataset=name, index=index, visual_tokens=len(visual), text_queries=len(types) - query_start)
                if smoke:
                    saved = (cap.h, cap.norm, cap.rope, cap.out, cap.calls)
                    cap.enabled = False
                    other = native_forward(model, cap, inputs)
                    diff = float((other.float() - native.float()).abs().max())
                    assert diff == 0, diff
                    cap.h, cap.norm, cap.rope, cap.out, cap.calls = saved
                    cap.enabled = True
                    this_check["capture_changes_native_max_abs"] = diff
                    # Current fast-path has both batched and per-layer memory APIs.
                    all_mem = adapter.all_visual_memories_batched(cap.h[0].index_select(1, visual))
                    this_check["batched_vs_per_layer_memory_max_abs"] = max(float((all_mem[l] - adapter.visual_memory_for_layer(cap.h[0].index_select(1, visual), l)).abs().max()) for l in LAYERS)
                    del all_mem
                for l in LAYERS:
                    metrics, validation = layer_comparison(cap, adapter, l, visual, query_start, validate=smoke)
                    f.write(json.dumps(dict(dataset=name, index=index, layer=l, visual_tokens=len(visual),
                                            text_queries=len(types) - query_start, metrics=metrics, checks=validation), allow_nan=False) + "\n")
                    if smoke:
                        this_check[str(l)] = validation
                if smoke:
                    checks.append(this_check)
                completed += 1
                if count % 10 == 0 or count + 1 == len(indices):
                    print(json.dumps(dict(shard=args.shard, dataset=name, done=count + 1, total=len(indices), elapsed=round(time.time() - start, 1))), flush=True)
    cap.close()
    dump_json(root / ("validation.json" if smoke else f"done_{args.shard}.json"),
              dict(protocol=PROTOCOL, source_sha256=sha(__file__), plan_sha256=sha(root / "plan.json"),
                   complete=True, samples=completed, elapsed_seconds=time.time() - start,
                   shard=args.shard, world=args.world, checkpoint_global_step=meta["global_step"], checks=checks))


def aggregate_metrics(stats):
    def average(key):
        values = [s[key] for s in stats if s[key] is not None]
        return sum(values) / len(values) if values else None
    n = sum(s["n"] for s in stats)
    valid = sum(s["valid_cosines"] for s in stats)
    sse, energy = sum(s["sse"] for s in stats), sum(s["energy"] for s in stats)
    return dict(samples=len(stats), tokens=n, valid_cosines=valid,
                macro={k: average(k) for k in ("mse", "cosine", "relative_l2", "native_rms", "pred_rms")},
                pooled=dict(mse=sse / (n * stats[0]["d"]), cosine=sum(s["cosine_sum"] for s in stats) / valid if valid else None,
                            relative_l2=math.sqrt(sse / energy) if energy else None))


def report(args):
    root = Path(args.output)
    plan = json.loads((root / "plan.json").read_text())
    all_rows = []
    for shard in range(args.world):
        done = json.loads((root / f"done_{shard}.json").read_text())
        assert done["complete"] and done["plan_sha256"] == sha(root / "plan.json")
        all_rows.extend(json.loads(s) for s in (root / f"rows_{shard}.jsonl").read_text().splitlines())
    keys = [(r["dataset"], r["index"], r["layer"]) for r in all_rows]
    expected = {(name, i, l) for name, spec in plan["datasets"].items() for i in range(spec["samples"]) for l in LAYERS}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    results = {}
    text = ["# Adapter 与原生模型：13–22 层独立比较", "",
            "Qwen3-VL-4B，hidden=2560；FA2，BF16，DeepStack 关闭。使用现有 Pixmo KL 2000-step adapter。", "",
            "每条样本只运行一次原生前向。第 ℓ 层始终用原生文本 Q 和文本 K/V，只换该层的视觉 memory；比较结果不传给下一层。", "",
            "hidden 在 input RMSNorm 前比较。attention output 在 W_O 后、残差相加前比较。视觉贡献为完整 causal softmax 下的视觉 V 加权和再经 W_O，保留文本 key 对分母的影响。", "",
            "主表对每条样本的 token 指标先取平均，再在样本间等权平均。统计使用 FP64；逐样本结果、token 加权统计及答案边界单 token 指标见 JSON。", ""]
    for name, spec in plan["datasets"].items():
        results[name] = {}
        text.extend([f"## {name}（{spec['samples']} 条）", "",
                     "| 层（0-based） | Hidden MSE | Hidden cosine | Attention MSE | Attention cosine | 视觉贡献 MSE | 视觉贡献 cosine | 视觉贡献相对 L2 |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for l in LAYERS:
            rows = [r for r in all_rows if r["dataset"] == name and r["layer"] == l]
            metrics = {key: aggregate_metrics([r["metrics"][key] for r in rows]) for key in rows[0]["metrics"]}
            metrics["max_native_reconstruction_relative_l2"] = max(r["checks"]["native_reconstruction_relative_l2"] for r in rows)
            results[name][str(l)] = metrics
            h, o, v = (metrics[k]["macro"] for k in ("hidden", "all_text/output", "all_text/visual_contribution"))
            text.append(f"| {l} | {h['mse']:.6g} | {h['cosine']:.4f} | {o['mse']:.6g} | {o['cosine']:.4f} | {v['mse']:.6g} | {v['cosine']:.4f} | {v['relative_l2']:.4f} |")
        text.append("")
    text.extend(["这些结果衡量单层、固定原生 query 时的读出接近程度，不等同于整体推理准确率或多层可替换性。", ""])
    dump_json(root / "summary.json", dict(protocol=PROTOCOL, results=results, samples=sum(DATASETS.values()), layer_sample_pairs=len(keys)))
    (root / "README.md").write_text("\n".join(text))
    dump_json(root / "status.json", dict(state="complete", samples=sum(DATASETS.values()), layer_sample_pairs=len(keys)))
    print("\n".join(text), flush=True)


def launch(args):
    root = Path(args.output)
    prepare(root)
    validation = json.loads((root / "validation.json").read_text())
    assert validation["complete"] and validation["source_sha256"] == sha(__file__)
    children, logs = [], []
    dump_json(root / "status.json", dict(state="running", gpus=list(range(args.world)), pid=os.getpid()))
    for shard in range(args.world):
        log = (root / f"worker_{shard}.log").open("w")
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        child = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "src.adapter_single_layer_similarity", "worker",
                                  "--output", str(root), "--shard", str(shard), "--world", str(args.world)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
    codes = [child.wait() for child in children]
    for log in logs:
        log.close()
    if any(codes):
        dump_json(root / "status.json", dict(state="failed", worker_exit_codes=codes))
        raise RuntimeError(codes)
    report(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "smoke", "worker", "launch", "report"))
    p.add_argument("--output", default=str(ROOT / "artifacts/diagnostics/adapter_single_layer_20260916"))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    if args.mode == "smoke":
        worker(args, smoke=True)
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "prepare":
        prepare(Path(args.output))
    elif args.mode == "report":
        report(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
