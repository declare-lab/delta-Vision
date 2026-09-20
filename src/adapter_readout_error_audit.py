"""Local K/V swap audit of independently measured adapter visual readout errors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from src.adapter_single_layer_similarity import (
    ROOT, DATA, DATASETS, LAYERS, INPUT_KEYS, CHECKPOINT_SHA, QwenBenchmarkDataset,
    _to_device_item, sha, setup, native_forward, projected_kv, vector_stats,
    aggregate_metrics, qwen_apply_rotary_pos_emb, dump_json,
)
from src.model import build_qwen_initial_context, qwen_embedding_adapter_prefill_cache_prepared
from src.qwen_adapter_prepare import prepare_fa2_inputs

ORIGINAL = ROOT / "artifacts/diagnostics/adapter_single_layer_20260916"


def vis_readout(a, q, k, v, visual):
    from flash_attn.flash_attn_interface import flash_attn_func
    vv = torch.zeros_like(v)
    vv.index_copy_(2, visual, v.index_select(2, visual))
    heads = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2),
                           dropout_p=0., softmax_scale=float(a.scaling), causal=True)
    flat = heads.reshape(q.shape[2], -1)
    return flat, F.linear(flat.unsqueeze(0), a.o_proj.weight, None)[0]


def one_layer(cap, adapter, l, visual, start, original, fast_cache=None):
    layer = cap.layers[l]
    a = layer.self_attn
    n = cap.norm[l]
    shape = (*n.shape[:-1], -1, a.head_dim)
    q = a.q_norm(a.q_proj(n).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(n).view(shape)).transpose(1, 2)
    v = a.v_proj(n).view(shape).transpose(1, 2)
    q, k = qwen_apply_rotary_pos_emb(q, k, *cap.rope[l])
    q = q[:, :, start:]
    memory = adapter.visual_memory_for_layer(cap.h[0].index_select(1, visual), l)
    ak, av = projected_kv(layer, memory, cap.rope[l], visual)
    checks = {}
    if fast_cache is not None:
        for key, value in (("visual_key", ak), ("visual_value", av)):
            diff = float((value.float() - fast_cache["layers"][l][key].float()).abs().max())
            assert diff == 0, (l, key, diff)
            checks[f"fast_path_{key}_max_abs"] = diff
    ka, va = k.clone(), v.clone()
    ka.index_copy_(2, visual, ak)
    va.index_copy_(2, visual, av)
    native_pre, native = vis_readout(a, q, k, v, visual)
    metrics = {}
    for name, kk, vv in (("K_only", ka, v), ("V_only", k, va), ("K_and_V", ka, va)):
        pre, out = vis_readout(a, q, kk, vv, visual)
        for group, sl in (("all_text", slice(None)), ("answer_boundary", slice(-1, None))):
            metrics[f"{name}/{group}/post_WO"] = vector_stats(out[sl], native[sl])
            metrics[f"{name}/{group}/pre_WO"] = vector_stats(pre[sl], native_pre[sl])
    for name, pred, target in (("key", ak, k.index_select(2, visual)), ("value", av, v.index_select(2, visual))):
        metrics[name] = vector_stats(pred[0].transpose(0, 1).flatten(1), target[0].transpose(0, 1).flatten(1))
    # The joint swap MUST reproduce the already reported experiment exactly.
    for group in ("all_text", "answer_boundary"):
        old = original["metrics"][f"{group}/visual_contribution"]
        new = metrics[f"K_and_V/{group}/post_WO"]
        assert old == new, (l, group, old, new)
    checks["original_measurement_exact_match"] = True
    return metrics, checks


@torch.inference_mode()
def worker(args):
    root = Path(args.output)
    plan = json.loads((ORIGINAL / "plan.json").read_text())
    assert plan["checkpoint_sha256"] == CHECKPOINT_SHA
    for name, info in plan["datasets"].items():
        assert sha(DATA / f"{name}_eval.jsonl") == info["sha256"]
    original = {(r["dataset"], r["index"], r["layer"]): r
                for line in (ORIGINAL / f"rows_{args.shard}.jsonl").read_text().splitlines()
                for r in [json.loads(line)]}
    processor, model, adapter, cap, meta = setup()
    start_time = time.time()
    samples, validation = 0, []
    output = root / f"rows_{args.shard}.jsonl"
    assert not output.exists(), output
    with output.open("w", buffering=1) as f:
        for name, count in DATASETS.items():
            ds = QwenBenchmarkDataset(str(DATA / f"{name}_eval.jsonl"), processor, name)
            assert len(ds) == count
            indices = list(range(args.shard, len(ds), args.world))
            for done, index in enumerate(indices):
                item = _to_device_item(ds[index], torch.device("cuda:0"))
                inputs = {k: v for k, v in item.items() if k in INPUT_KEYS}
                native_forward(model, cap, inputs)
                visual = inputs["mm_token_type_ids"][0].eq(1).nonzero().flatten()
                start = int(visual[-1]) + 1
                cache = None
                if args.shard == 0 and done == 0:
                    # Rebuild E with the actual fast-path helper, then compare its
                    # produced visual K/V against the diagnostic at all ten layers.
                    cap.enabled = False
                    initial, positions = build_qwen_initial_context(model, inputs)
                    diff = float((initial.float() - cap.h[0].float()).abs().max())
                    assert diff == 0, diff
                    prepared = prepare_fa2_inputs(model, adapter, inputs["input_ids"], inputs["attention_mask"],
                                                  inputs["mm_token_type_ids"], initial, positions)
                    _, _, cache = qwen_embedding_adapter_prefill_cache_prepared(
                        model, adapter, **prepared, retain_prefix_states=False)
                    assert cache["attention_implementation"] == "flash_attention_2"
                    cap.enabled = True
                    validation.append(dict(dataset=name, index=index, fast_path_initial_hidden_max_abs=diff))
                for l in LAYERS:
                    metrics, checks = one_layer(cap, adapter, l, visual, start,
                                               original[(name, index, l)], cache)
                    f.write(json.dumps(dict(dataset=name, index=index, layer=l, metrics=metrics, checks=checks), allow_nan=False) + "\n")
                samples += 1
                if done % 25 == 0 or done + 1 == len(indices):
                    print(json.dumps(dict(shard=args.shard, dataset=name, done=done + 1, total=len(indices),
                                          elapsed=round(time.time() - start_time, 1))), flush=True)
    dump_json(root / f"done_{args.shard}.json", dict(complete=True, samples=samples, validation=validation,
                                                    source_sha256=sha(__file__), original_plan_sha256=sha(ORIGINAL / "plan.json")))
    cap.close()


def report(args):
    root = Path(args.output)
    rows = []
    for shard in range(args.world):
        done = json.loads((root / f"done_{shard}.json").read_text())
        assert done["complete"] and done["source_sha256"] == sha(__file__)
        rows.extend(json.loads(line) for line in (root / f"rows_{shard}.jsonl").read_text().splitlines())
    keys = [(r["dataset"], r["index"], r["layer"]) for r in rows]
    expected = {(name, i, l) for name, n in DATASETS.items() for i in range(n) for l in LAYERS}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    text = ["# 单层视觉读出误差：K/V 分离诊断", "",
            "同一 checkpoint、FA2、DeepStack 关闭、固定原生文本 Q/K/V；沿用 MMStar1000、RealWorldQA765。", "",
            "K_only：只换视觉 K，保留原生视觉 V，检验注意力分配变化。V_only：只换视觉 V，保持原生注意力权重，检验内容变化。K_and_V：现有 adapter 的实际视觉 memory。", "",
            "所有 cosine 比较视觉贡献，after W_O，先在样本内平均再在样本间平均。此分解是混合反事实，不能把各项误差当作线性可加的归因。", ""]
    summary = {}
    for name, count in DATASETS.items():
        summary[name] = {}
        text.extend([f"## {name} ({count})", "", "| 层 | 只换 K cosine | 只换 V cosine | 同时换 K/V cosine | 同时换 K/V 相对 L2 |", "|---|---:|---:|---:|---:|"])
        for l in LAYERS:
            items = [r for r in rows if r["dataset"] == name and r["layer"] == l]
            metrics = {k: aggregate_metrics([r["metrics"][k] for r in items]) for k in items[0]["metrics"]}
            summary[name][str(l)] = metrics
            k, v, both = [metrics[f"{key}/all_text/post_WO"]["macro"] for key in ("K_only", "V_only", "K_and_V")]
            text.append(f"| {l} | {k['cosine']:.4f} | {v['cosine']:.4f} | {both['cosine']:.4f} | {both['relative_l2']:.4f} |")
        text.append("")
    dump_json(root / "summary.json", summary)
    (root / "README.md").write_text("\n".join(text))
    dump_json(root / "status.json", dict(state="complete", samples=sum(DATASETS.values()), pairs=len(rows),
                                         original_joint_swap_exact_matches=len(rows)))
    print("\n".join(text), flush=True)


def launch(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "plan.json", dict(original_plan=str(ORIGINAL / "plan.json"), original_plan_sha256=sha(ORIGINAL / "plan.json"),
                                       source_sha256=sha(__file__), layers=list(LAYERS), datasets=DATASETS,
                                       source_original_sha256=sha(ROOT / "src/adapter_single_layer_similarity.py"),
                                       intervention="independent K-only, V-only, both, native text Q/K/V fixed; no propagation"))
    dump_json(root / "status.json", dict(state="running", pid=os.getpid()))
    children, logs = [], []
    for shard in range(args.world):
        log = (root / f"worker_{shard}.log").open("w")
        logs.append(log)
        child = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "src.adapter_readout_error_audit", "worker",
                                  "--output", str(root), "--shard", str(shard), "--world", str(args.world)], cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"),
                                 stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
    codes = [child.wait() for child in children]
    for log in logs:
        log.close()
    if any(codes):
        dump_json(root / "status.json", dict(state="failed", codes=codes))
        raise RuntimeError(codes)
    report(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("worker", "launch", "report"))
    p.add_argument("--output", default=str(ROOT / "artifacts/diagnostics/adapter_readout_error_audit_20260916"))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    {"worker": worker, "launch": launch, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
