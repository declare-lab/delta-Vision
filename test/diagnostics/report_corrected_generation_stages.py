"""Validate direct generation stages and compare every output with the saved original run."""
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/prefill_decode_corrected_20260915"
OLD = ROOT / "test/results/prefill_original_mmstar200_20260915/table_raw"


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def report_pairs():
    folder = OUT / "paired_decode"
    methods = ["fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"]
    paths = [(method, retention, folder / f"{method}_ret{int(retention*100):02d}.json")
             for method in methods for retention in [.05, .2]]
    if not all(path.exists() for _, _, path in paths):
        return
    rows = []
    lines = ["# 同输入交替运行的 decode 对照", "",
             "与训练共享 GPU；子集索引 0、25、125、133，每个索引预热 3 次、测量 10 对请求，每组共 40 对。",
             "每对交换 base/剪枝的运行先后，同 GPU 同输入，原生 generate，EOS 或最多 8 tokens，实际 cached forward 直接计时。",
             "这是一组短输出诊断，不能替代完整 MMStar 200 或独占 GPU 的性能结果。", "",
             "| 方法 | 保留率 | Base decode 中位数 ms/步 | 方法 decode 中位数 ms/步 | 逐对加速比中位数 | 方法更快的对数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, retention, path in paths:
        trials = json.loads(path.read_text())
        assert len(trials) == 40
        seen = {}
        metrics = {}
        for label in ["base", "method"]:
            for pair in trials:
                r = pair[label]
                close(r["total_time_s"], r["generation_prefill_time_s"] + r["decode_time_s"] + r["generation_overhead_s"])
                assert r["decode_steps"] == len(r["tokens"]) - 1
                assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in r["generation_stages"][1:])
                key = (pair["sample"], label)
                assert seen.setdefault(key, r["tokens"]) == r["tokens"]
            steps = [s["seconds"] * 1000 for pair in trials for s in pair[label]["generation_stages"][1:]]
            metrics[label + "_metrics"] = dict(decode_median_ms=statistics.median(steps),
                decode_mean_ms=statistics.mean(steps), decode_steps=len(steps),
                prefill_median_ms=statistics.median(p[label]["generation_prefill_time_s"] * 1000 for p in trials))
        ratios = [(p["base"]["decode_time_s"] / p["base"]["decode_steps"]) /
                  (p["method"]["decode_time_s"] / p["method"]["decode_steps"]) for p in trials]
        row = dict(method=method, retention=retention, pairs=len(trials), shared_gpu=True, **metrics,
                   paired_decode_speedup_median=statistics.median(ratios), method_faster_pairs=sum(r > 1 for r in ratios))
        rows.append(row)
        lines.append(f"| {method} | {retention:.0%} | {metrics['base_metrics']['decode_median_ms']:.3f} | "
                     f"{metrics['method_metrics']['decode_median_ms']:.3f} | {statistics.median(ratios):.4f}× | {sum(r > 1 for r in ratios)}/40 |")
    lines += ["", "逐对加速比先对同一对请求的 decode 每步耗时求比值，再取 40 对中位数；它不等于表中两个边际中位数的商。所有试次（含慢请求）均保留。",
              "", "这轮结果没有显示稳定的整步 decode 加速。剪枝减少 attention 的 KV 访问，但未减少每步线性层和 MLP 的权重计算；短输出、单请求及共享 GPU 环境下，不能用 KV 缩减率推算整步速度。",
              "", "Adapter 在原 MMStar 停止规则下没有实际 decode forward，因此未编造其 ms/步，也未强制延长输出来混入这张表。",
              "", "复跑：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/python test/diagnostics/paired_native_decode.py`。"]
    (folder / "summary.json").write_text(json.dumps(rows, indent=2))
    (folder / "README.zh.md").write_text("\n".join(lines) + "\n")
    print("PASS: 480 alternating pairs; stable repeated outputs and direct stage identities.")


def main():
    rows, validations = [], []
    for retention in [.05, .2]:
        folder = OUT / f"ret{int(retention*100):02d}"
        results = {d["method"]: (p, d) for p in folder.rglob("results.json") for d in [json.loads(p.read_text())]}
        assert len(results) == 7, (folder, len(results))
        base = results["base"][1]
        for method in ["base", "fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"]:
            path, summary = results[method]
            predictions = json.loads(path.with_name("predictions.json").read_text())
            assert len(predictions) == 200
            original_path = next((OLD / f"baseline_r{retention:g}" / method).rglob("predictions.json"))
            original = json.loads(original_path.read_text())
            changed = [i for i, (a, b) in enumerate(zip(original, predictions)) if a["prediction_text"] != b["prediction_text"]]
            assert not changed, (method, retention, changed)
            for pred in predictions:
                close(pred["total_time_s"], pred["generation_prefill_time_s"] + pred["decode_time_s"] + pred["generation_overhead_s"])
                assert pred["decode_steps"] == pred["generated_tokens"] - 1
                assert len(pred["generation_stages"]) == pred["generated_tokens"]
                assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in pred["generation_stages"][1:])
                close(pred["actual_prefill_kv_cache_mb"], pred["kv_cache_mb"])
            for key in ["total_time_s", "generation_prefill_time_s", "decode_time_s", "generation_overhead_s", "decode_steps"]:
                close(summary[key], sum(p[key] for p in predictions))
            close(summary["decode_ms_per_step"], 1000 * summary["decode_time_s"] / summary["decode_steps"])
            row = dict(method=method, retention=1. if method == "base" else retention, reference_group=f"ret{int(retention*100):02d}",
                samples=200, total_time_s=summary["total_time_s"], prefill_time_s=summary["generation_prefill_time_s"],
                standalone_prefill_time_s=summary["prefilling_time_s"], decode_time_s=summary["decode_time_s"],
                generation_overhead_s=summary["generation_overhead_s"], decode_steps=summary["decode_steps"],
                generated_tokens=summary["generated_tokens"], decode_ms_per_step=summary["decode_ms_per_step"],
                prefill_median_ms=statistics.median(p["generation_prefill_time_s"] * 1000 for p in predictions),
                decode_median_ms=statistics.median(s["seconds"] * 1000 for p in predictions for s in p["generation_stages"][1:]),
                kv_cache_mb=summary["actual_prefill_kv_cache_mb"], analytic_kv_cache_mb=summary["kv_cache_mb"], flops=summary["flops"],
                score=summary["score"], total_speedup=base["total_time_s"] / summary["total_time_s"],
                prefill_speedup=base["generation_prefill_time_s"] / summary["generation_prefill_time_s"],
                decode_speedup=base["decode_ms_per_step"] / summary["decode_ms_per_step"],
                source=str(path), shared_gpu=True, output_changes_vs_original=0)
            rows.append(row)
            validations.append(dict(method=method, retention=retention, samples=200, changed_outputs=changed,
                                    stage_sums_match=True, decode_tokens_and_cache_valid=True))
    adapter_rows = json.loads((OUT / "adapter.json").read_text())
    adapter_detail = json.loads((OUT / "adapter.details.json").read_text())
    predictions = adapter_detail["predictions"]
    assert len(predictions) == 200
    for pred in predictions:
        close(pred["adapter_total_s"], pred["adapter_prefill_s"] + pred["adapter_decode_forward_s"] + pred["adapter_generation_overhead_s"])
        assert pred["adapter_decode_steps"] == sum(len(ids) - 1 for ids in pred["adapter_generated_token_ids"])
    row = adapter_rows[1]
    rows.insert(0, dict(method="embedding_adapter", retention=None, reference_group="adapter", samples=200,
                       total_time_s=row["total_time_s"], prefill_time_s=row["prefilling_time_s"],
                       decode_time_s=row["decode_forward_time_s"], decode_steps=row["decode_steps"],
                       generation_overhead_s=row["generation_overhead_s"],
                       generated_tokens=sum(len(ids) for p in predictions for ids in p["adapter_generated_token_ids"]),
                       decode_ms_per_step=None if not row["decode_steps"] else row["decode_forward_time_s"] * 1000 / row["decode_steps"],
                       prefill_median_ms=statistics.median(p["adapter_prefill_s"] * 1000 for p in predictions),
                       decode_median_ms=None, kv_cache_mb=row["actual_prefill_kv_cache_mb"], analytic_kv_cache_mb=row["kv_cache_mb"],
                       flops=row["flops"], score=row["score"], total_speedup=row["speedup_total"], prefill_speedup=row["speedup_prefilling"],
                       decode_speedup=None, source=str(OUT / "adapter.json"), shared_gpu=True))
    for key in ["adapter_decode_forward_s", "adapter_decode_steps", "adapter_generation_overhead_s"]:
        close(adapter_detail["summary"]["timing"][key], sum(p[key] for p in predictions))
    (OUT / "table.json").write_text(json.dumps(rows, indent=2))
    with (OUT / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "validation.json").write_text(json.dumps({"status": "passed", "baseline_rows": validations,
        "all_2800_baseline_outputs_match_original": True, "adapter_samples": 200,
        "adapter_decode_steps": row["decode_steps"], "shared_gpu": True,
        "scope": "Valid direct stage accounting and output parity; shared-GPU timings do not reproduce isolated historical latency."}, indent=2))
    names = {"embedding_adapter": "Adapter fast-path", "base": "Base", "fastv": "FastV", "dart": "DART",
             "divprune": "DivPrune", "zoo": "ZooPrune", "sparsevlm": "SparseVLM", "visionzip": "VisionZip"}
    lines = ["# 修正后的 prefill / decode 分段结果", "",
             "Qwen3-VL-4B，截图对应 MMStar 200。按用户要求与训练共享 GPU 运行；这些绝对时间和 speedup 不代表独占 GPU 性能。", "",
             "保留原始 attention kernel、RoPE 和生成停止规则，只缓存重复计算的 FA2 序列元数据。", "",
             "## 直接分段计时", "", "时间列为 200 样本合计秒；每步 decode 毫秒按真正执行的 cached forward 数归一化。", "",
             "| 方法 | 组别 | Total s | Prefill s | Decode forward s | 其他生成开销 s | Decode 步数 | Decode ms/步 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ms = "N/A" if r["decode_ms_per_step"] is None else f"{r['decode_ms_per_step']:.3f}"
        lines.append(f"| {names[r['method']]} | {r['reference_group']} | {r['total_time_s']:.4f} | {r['prefill_time_s']:.4f} | "
                     f"{r['decode_time_s']:.4f} | {r['generation_overhead_s']:.4f} | {r['decode_steps']} | {ms} |")
    lines += ["", "每行满足 **Total = Prefill + Decode forward + 其他生成开销**。Baseline prefill 是同一次 generate 内首个模型 forward，decode 是后续逐 token forward；未用独立 prefill 测量值相减来冒充 decode。", "",
              "Adapter 按原规则识别到选项即停止。本次 200 请求都只生成首 token，decode forward 实际执行 0 次。其后续开销是选项解码及停止判断，不能宣称为零成本 decode。", "",
              "## 资源及配套速度比", "", "KV 为实际 retained K/V 的样本均值 MiB；FLOPs 为原 decoder prefill 解析均值，不含 vision/selector/head/norm/softmax。", "",
              "| 方法 | 组别 | 实际 KV MiB | FLOPs | Total speedup | Prefill speedup | Decode ms/步 speedup |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        speed = "N/A" if r["decode_speedup"] is None else f"{r['decode_speedup']:.4f}"
        lines.append(f"| {names[r['method']]} | {r['reference_group']} | {r['kv_cache_mb']:.2f} | {r['flops']:.4e} | "
                     f"{r['total_speedup']:.4f} | {r['prefill_speedup']:.4f} | {speed} |")
    lines += ["", "每组使用自己的 base 分母。Adapter 配套 base 的原始数据见 adapter.json；baseline 的 prefill speedup 使用 generate 内直接计时。共享 GPU 下不同请求遇到的竞争程度不同，速度比仅描述本次运行，不作为方法固有加速比。", "",
              "Adapter 实际 K/V 为 47.45 MiB；旧解析估算 11.41 MiB 另保留在 JSON 的 analytic_kv_cache_mb 中，不当作实际 cache。", "",
              "## 验证与复跑", "", "- 14 行 baseline × 200 样本的输出全部与修正前一致。", "- 全部逐样本分段加总、单 token decode、视觉输入只出现在 prefill、KV 计数均通过检查。",
              "- 细粒度 logits/KV 对照及 attention 操作 trace 见 ../prefill_slowdown_investigation_20260915/。",
              "- table.csv / table.json 含完整精度、prefill/decode 中位数、独立 prefill 旧字段和所有源文件路径。",
              "- protocol.json 保存 baseline 命令；adapter.protocol.json 保存 adapter 等价复跑命令与运行说明。adapter.details.json 保存逐样本真实 decode 次数与时间。",
              "- 另见 [同输入交替 decode 对照](paired_decode/README.zh.md)：4 个样本、每组 40 对请求，帮助检查分批计时中的环境波动。",
              "- 统一入口添加 --optimize-attention-metadata --measure-decode-steps；原 baseline 入口使用 --optimize-attention-metadata --measure-decode。"]
    (OUT / "README.zh.md").write_text("\n".join(lines) + "\n")
    print("PASS: all 2800 baseline outputs match; all sample stage identities/cache counts verified.")
    report_pairs()


if __name__ == "__main__":
    main()
