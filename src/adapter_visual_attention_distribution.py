"""Compare visual-token attention distributions for the SAME native text Q.

Every layer is independent. Native forward uses FA2, DeepStack off. Statistics
reconstruct probabilities from captured Q/K in FP32; they never enter the model.
No V multiplication, W_O, suffix intervention, generation, accuracy, or training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from src.adapter_single_layer_similarity import (
    ROOT, DATA, DATASETS, LAYERS, CHECKPOINT, CHECKPOINT_SHA, INPUT_KEYS,
    QwenBenchmarkDataset, _to_device_item, sha, setup, native_forward,
    projected_kv, qwen_apply_rotary_pos_emb, dump_json,
)

PROTOCOL = "fixed_native_text_Q_visual_attention_distribution_v1"
BASE_RESULTS = ROOT / "artifacts/diagnostics/adapter_single_layer_20260916"


def distribution_rows(native_scores, adapter_scores, native_log_z, adapter_log_z):
    """One row per head/query; visual scores include only causally visible keys.

Conditional p(j|visual) separates spatial allocation from total visual mass.
Full causal sequence denominators determine the separate total-mass metrics.
JS uses log base 2, hence [0,1]. Cosine is per head, never on a head-averaged map.
"""
    assert native_scores.shape == adapter_scores.shape
    lp = native_scores.log_softmax(-1)
    lq = adapter_scores.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    log_m = torch.logaddexp(lp, lq) - math.log(2)
    cosine = (p * q).sum(-1) / (p.square().sum(-1) * q.square().sum(-1)).sqrt()
    js = (.5 * ((p * (lp - log_m)).sum(-1) + (q * (lq - log_m)).sum(-1)) / math.log(2)).clamp_min(0)
    visual_n = native_scores.shape[-1]
    mass_p = (native_scores.logsumexp(-1) - native_log_z).double().exp()
    mass_q = (adapter_scores.logsumexp(-1) - adapter_log_z).double().exp()
    assert mass_p.max() <= 1.00001 and mass_q.max() <= 1.00001
    k10 = min(10, visual_n)
    kfrac = max(1, math.ceil(.1 * visual_n))
    maximum = max(k10, kfrac)
    pi = native_scores.topk(maximum, dim=-1).indices
    qi = adapter_scores.topk(maximum, dim=-1).indices

    def overlap(k):
        selected = torch.zeros_like(native_scores, dtype=torch.bool).scatter_(-1, pi[..., :k], True)
        return selected.gather(-1, qi[..., :k]).float().mean(-1)

    stats = dict(cosine=cosine.clamp(-1, 1), js_bits=js,
                 tv=.5 * (p - q).abs().sum(-1),
                 kl_native_to_adapter=(p * (lp - lq)).sum(-1).clamp_min(0),
                 kl_adapter_to_native=(q * (lq - lp)).sum(-1).clamp_min(0),
                 top1_match=(native_scores.argmax(-1) == adapter_scores.argmax(-1)).float(),
                 top10_overlap=overlap(k10), top10pct_overlap=overlap(kfrac),
                 native_visual_mass=mass_p, adapter_visual_mass=mass_q,
                 visual_mass_abs_gap=(mass_p - mass_q).abs(),
                 native_normalized_entropy=-(p * lp).sum(-1) / (math.log(visual_n) if visual_n > 1 else 1.),
                 adapter_normalized_entropy=-(q * lq).sum(-1) / (math.log(visual_n) if visual_n > 1 else 1.))
    return stats


def reduce_rows(stats):
    """Head/query mean within one sample; retain individual-head statistics."""
    names = list(stats)
    stacked = torch.stack([stats[k].double() for k in names])  # metric, head, query
    means = stacked.mean(dim=(1, 2)).cpu().tolist()
    per_head = stacked.mean(dim=2).cpu().tolist()
    assert all(math.isfinite(x) for x in means), means
    mass = stats["native_visual_mass"].double()
    extra = torch.stack(((mass * stats["cosine"]).sum(), (mass * stats["js_bits"]).sum(), mass.sum())).cpu().tolist()
    return dict(means=dict(zip(names, means)), per_head=dict(zip(names, per_head)),
                heads=mass.shape[0], queries=mass.shape[1],
                native_mass_weighted_cosine=extra[0] / extra[2] if extra[2] > 0 else None,
                native_mass_weighted_js_bits=extra[1] / extra[2] if extra[2] > 0 else None)


def compare_layer(cap, adapter, l, visual, start, validate=False):
    layer = cap.layers[l]
    a = layer.self_attn
    n = cap.norm[l]
    shape = (*n.shape[:-1], -1, a.head_dim)
    q = a.q_norm(a.q_proj(n).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(n).view(shape)).transpose(1, 2)
    q, k = qwen_apply_rotary_pos_emb(q, k, *cap.rope[l])
    query = q[:, :, start:]
    memory = adapter.visual_memory_for_layer(cap.h[0].index_select(1, visual), l)
    ak, _ = projected_kv(layer, memory, cap.rope[l], visual)
    ka = k.clone()
    ka.index_copy_(2, visual, ak)
    text = torch.ones(k.shape[2], dtype=torch.bool, device=k.device)
    text[visual] = False
    assert torch.equal(k[:, :, text], ka[:, :, text])
    groups = query.shape[1] // k.shape[1]
    kn, kp = (x.repeat_interleave(groups, dim=1).float() for x in (k, ka))
    length, qlength = k.shape[2], query.shape[2]
    collected, log_z_native, log_z_adapter = {}, [], []
    for offset in range(0, qlength, 32):
        qblock = query[:, :, offset:offset + 32].float()
        sn = qblock @ kn.transpose(-1, -2) * float(a.scaling)
        sp = qblock @ kp.transpose(-1, -2) * float(a.scaling)
        positions = torch.arange(start + offset, start + offset + qblock.shape[2], device=k.device)
        visible = torch.arange(length, device=k.device)[None, :] <= positions[:, None]
        sn = sn[0].masked_fill(~visible[None], -float("inf"))
        sp = sp[0].masked_fill(~visible[None], -float("inf"))
        zn, zp = sn.logsumexp(-1), sp.logsumexp(-1)
        vn, vp = sn.index_select(-1, visual), sp.index_select(-1, visual)
        assert torch.isfinite(vn).all() and torch.isfinite(vp).all()
        stats = distribution_rows(vn, vp, zn, zp)
        for key, value in stats.items():
            collected.setdefault(key, []).append(value)
        if validate:
            identity = distribution_rows(vn, vn, zn, zn)
            assert identity["js_bits"].abs().max() < 2e-7
            assert (identity["cosine"] - 1).abs().max() < 3e-7
            assert identity["top10_overlap"].min() == 1
            assert identity["visual_mass_abs_gap"].max() == 0
            log_z_native.append(zn)
            log_z_adapter.append(zp)
    all_stats = {key: torch.cat(values, dim=1) for key, values in collected.items()}
    result = {"all_text": reduce_rows(all_stats),
              "last_text_query": reduce_rows({key:value[:, -1:] for key,value in all_stats.items()})}
    checks = {}
    if validate:
        from flash_attn.flash_attn_interface import flash_attn_func
        # FA2's exposed log-sum-exp verifies the SAME causal softmax denominator.
        value = a.v_proj(n).view(shape).transpose(1, 2)
        for label, key, references in (("native", k, log_z_native), ("adapter_K", ka, log_z_adapter)):
            _, lse, _ = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                        dropout_p=0., softmax_scale=float(a.scaling), causal=True,
                                        return_attn_probs=True)
            expected = torch.cat(references, dim=1)
            error = float((expected - lse[0, :, :qlength]).abs().max())
            assert error < .001, (l, label, error)
            checks[f"{label}_FA2_logsumexp_max_abs"] = error
        checks["identity_js_below_2e_7"] = True
        checks["identity_top10_overlap"] = 1.
    return result, checks


def prepare(root):
    plan = dict(protocol=PROTOCOL, checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
                model="Qwen3-VL-4B-Instruct", layers=list(LAYERS), layer_indexing="zero based",
                attention="flash_attention_2", deepstack="off", model_dtype="bfloat16",
                probability_statistics="FP32 QK and log_softmax reconstructed from captured native/adapter QK; FP64 metric aggregation; does not alter native FA2 forward",
                fixed="native text Q, native text K, native RoPE and causal positions; change ONLY visual K",
                independence="one untouched native forward per example; each layer compares native and adapter independently",
                distribution="p(j|visual)=exp(qK_j)/sum_visual exp(qK); shape [head, text query, visual token]",
                visual_mass="sum_visual exp(qK)/sum_all_causally_visible_keys exp(qK); text keys included in denominator",
                cosine="cosine of per-head visual-token probability vectors, not hidden, V, AV, or head-averaged maps",
                js="Jensen-Shannon divergence, log base 2, range 0 to 1; lower is more similar",
                top10="intersection of 10 largest visual attention entries divided by min(10,N_visual)",
                top10pct="intersection of ceil(0.1*N_visual) largest entries divided by that count",
                queries="all text after final visual token; last text query separately; no generated answers",
                aggregation="equal heads and queries within sample, then equal samples; also retain individual heads and native-visual-mass-weighted diagnostics",
                datasets={n:dict(samples=c,path=str(DATA / f"{n}_eval.jsonl"),sha256=sha(DATA / f"{n}_eval.jsonl")) for n,c in DATASETS.items()},
                sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT / "src/adapter_single_layer_similarity.py",ROOT / "src/model.py")})
    assert plan["checkpoint_sha256"] == CHECKPOINT_SHA
    path = root / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        dump_json(path, plan)
    return plan


@torch.inference_mode()
def worker(args, smoke=False):
    root = Path(args.output)
    plan = prepare(root)
    processor, model, adapter, cap, meta = setup()
    start_time = time.time()
    path = root / ("smoke_rows.jsonl" if smoke else f"rows_{args.shard}.jsonl")
    assert not path.exists(), path
    checked, samples = [], 0
    with path.open("w", buffering=1) as stream:
        for name, spec in plan["datasets"].items():
            ds = QwenBenchmarkDataset(spec["path"], processor, name)
            assert len(ds) == spec["samples"]
            indices = [0] if smoke else list(range(args.shard, len(ds), args.world))
            for count, index in enumerate(indices):
                item = _to_device_item(ds[index], torch.device("cuda:0"))
                inputs = {k:v for k,v in item.items() if k in INPUT_KEYS}
                assert inputs["input_ids"].shape[0] == 1 and inputs["attention_mask"].bool().all()
                types = inputs["mm_token_type_ids"][0]
                visual = types.eq(1).nonzero().flatten()
                assert len(visual) and types.max() == 1
                assert torch.equal(visual, torch.arange(visual[0], visual[-1]+1, device=visual.device))
                start = int(visual[-1]) + 1
                assert start < len(types)
                native_forward(model, cap, inputs)
                for l in LAYERS:
                    groups, checks = compare_layer(cap, adapter, l, visual, start, validate=smoke)
                    stream.write(json.dumps(dict(dataset=name,index=index,layer=l,visual_tokens=len(visual),
                                                 text_queries=len(types)-start,groups=groups,checks=checks), allow_nan=False) + "\n")
                    if smoke:
                        checked.append(dict(dataset=name, layer=l, **checks))
                samples += 1
                if count % 10 == 0 or count+1 == len(indices):
                    print(json.dumps(dict(shard=args.shard,dataset=name,done=count+1,total=len(indices),elapsed=round(time.time()-start_time,1))), flush=True)
    cap.close()
    dump_json(root / ("validation.json" if smoke else f"done_{args.shard}.json"),
              dict(complete=True,samples=samples,source_sha256=sha(__file__),plan_sha256=sha(root / "plan.json"),
                   seconds=time.time()-start_time,checks=checked))


def summarize(groups):
    metrics = groups[0]["means"].keys()
    result = {key:sum(g["means"][key] for g in groups)/len(groups) for key in metrics}
    heads = groups[0]["heads"]
    result["per_head"] = {key:[sum(g["per_head"][key][h] for g in groups)/len(groups) for h in range(heads)] for key in metrics}
    for key in ("native_mass_weighted_cosine", "native_mass_weighted_js_bits"):
        values = [g[key] for g in groups if g[key] is not None]
        result[key] = sum(values)/len(values) if values else None
    result.update(samples=len(groups),heads=heads,total_text_queries=sum(g["queries"] for g in groups))
    return result


def report(args):
    root = Path(args.output)
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    for shard in range(args.world):
        done = json.loads((root / f"done_{shard}.json").read_text())
        assert done["complete"] and done["plan_sha256"] == sha(root / "plan.json")
        rows.extend(json.loads(x) for x in (root / f"rows_{shard}.jsonl").read_text().splitlines())
    keys = [(r["dataset"],r["index"],r["layer"]) for r in rows]
    expected = {(n,i,l) for n,s in plan["datasets"].items() for i in range(s["samples"]) for l in LAYERS}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    result = {}
    text = ["# 同一文本 query 对视觉 token 的 attention 分布", "",
            "Qwen3-VL-4B，现有 Pixmo KL 2000-step embedding adapter；FA2、BF16、DeepStack 关闭；0-based 13–22 层独立比较。", "",
            "固定原生文本 Q、文本 K 和位置，只替换视觉 K。逐 head、逐图像后文本 query 比较视觉 token 概率分布，先在样本内平均，再在样本间等权平均；没有先平均各个 head 的 attention map。", "",
            "视觉内分布 p(j|visual) 衡量关注哪些视觉 token；视觉总注意力 mass 衡量全部 causal softmax 中分配给视觉区域的比例。cosine 和 JS 针对视觉内分布，mass 差额是同一 head/query 上的绝对差再平均。", "",
            "JS 使用 log2，范围 [0,1]，0 表示完全一致。Top-10 重合率为两个 top-10 集合交集大小/10（视觉 token 少于10时采用实际数量）。", "",
            "原生模型前向始终为 FA2。由于 FA2 不输出完整概率矩阵，统计单独用捕获的同一 Q/K 以 FP32 重建 softmax；验证其 logsumexp 与 FA2 一致，统计结果不写回模型。", ""]
    for name,spec in plan["datasets"].items():
        result[name] = {}
        text.extend([f"## {name}（{spec['samples']} 条）", "",
                     "| 层 | Attention分布 cosine ↑ | JS ↓ | Top-10重合率 (%) ↑ | 原生视觉 mass (%) | Adapter视觉 mass (%) | mass绝对差 (pp) ↓ |",
                     "|---|---:|---:|---:|---:|---:|---:|"])
        for l in LAYERS:
            selected = [r for r in rows if r["dataset"] == name and r["layer"] == l]
            result[name][str(l)] = {group:summarize([r["groups"][group] for r in selected]) for group in ("all_text","last_text_query")}
            m = result[name][str(l)]["all_text"]
            text.append(f"| {l} | {m['cosine']:.4f} | {m['js_bits']:.4f} | {100*m['top10_overlap']:.2f} | {100*m['native_visual_mass']:.2f} | {100*m['adapter_visual_mass']:.2f} | {100*m['visual_mass_abs_gap']:.2f} |")
        text.append("")
    text.extend(["TV、双向视觉分布 KL、Top-1/Top-10% 重合率、熵、每个 head、最后一个文本 query 和按原生视觉 mass 加权的指标均保存在 summary.json。", "",
                 "本结果对应用户定义的 attention 分配功能相似性；不把它等同于 V/readout 向量重建、最终答案准确率或整模型功能等价。", ""])
    dump_json(root / "summary.json",result)
    (root / "README.md").write_text("\n".join(text))
    dump_json(root / "status.json",dict(state="complete",samples=sum(DATASETS.values()),layer_sample_pairs=len(rows)))
    print("\n".join(text),flush=True)


def launch(args):
    root = Path(args.output)
    prepare(root)
    validation = json.loads((root / "validation.json").read_text())
    assert validation["complete"] and validation["source_sha256"] == sha(__file__)
    dump_json(root / "status.json",dict(state="running",pid=os.getpid(),gpus=list(range(args.world))))
    children,logs = [],[]
    for shard in range(args.world):
        log = (root / f"worker_{shard}.log").open("w")
        logs.append(log)
        child = subprocess.Popen([str(ROOT / ".venv/bin/python"),"-m","src.adapter_visual_attention_distribution","worker",
                                  "--output",str(root),"--world",str(args.world),"--shard",str(shard)],cwd=ROOT,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false"),
                                 stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
    codes = [c.wait() for c in children]
    for log in logs: log.close()
    if any(codes):
        dump_json(root / "status.json",dict(state="failed",codes=codes))
        raise RuntimeError(codes)
    report(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","worker","launch","report"))
    p.add_argument("--output",default=str(ROOT / "artifacts/diagnostics/adapter_visual_attention_distribution_20260916"))
    p.add_argument("--world",type=int,default=8)
    p.add_argument("--shard",type=int,default=0)
    args=p.parse_args()
    if args.mode == "smoke": worker(args,smoke=True)
    else: {"worker":worker,"launch":launch,"report":report}[args.mode](args)


if __name__ == "__main__":
    main()
