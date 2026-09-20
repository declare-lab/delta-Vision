"""Validate and summarize the FA2 runs without mixing continuation with decode."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test/results/all_fa2_20260915"
METHODS = ["fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"]


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)


def pairs():
    lines = ["# FA2：同卡交替运行 base 与剪枝方法", "",
        "Qwen3-VL-4B；MMStar 索引 0、25、125、133；每个输入 10 对、每组 40 对，合计 480 对。",
        "base 和剪枝都使用 FA2、原生 generate、视觉和 decoder CUDA Graph；与训练共享 H200。",
        "每对交换执行先后；原始 EOS/最多 8 tokens；decode 只计真实 cached forward。保留全部慢请求。",
        "下表是逐对加速比的中位数，分母和分子来自同一输入、同一 GPU。不是 200 条完整数据的加速比。", "",
        "| 方法 | 保留率 | Total 加速 | Prefill 加速 | Decode 加速 | Decode 更快的对数 |",
        "|---|---:|---:|---:|---:|---:|"]
    rows = []
    for method in METHODS:
        for retention in [.05, .2]:
            trials = json.loads((OUT / "paired" / f"{method}_ret{int(retention*100):02d}.json").read_text())
            assert len(trials) == 40
            for trial in trials:
                for label in ["base", "method"]:
                    value = trial[label]
                    close(value["total_time_s"], value["generation_prefill_time_s"] + value["decode_time_s"] + value["generation_overhead_s"])
                    assert value["decode_steps"] == len(value["tokens"]) - 1
                    assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in value["generation_stages"][1:])
            total = [t["base"]["total_time_s"] / t["method"]["total_time_s"] for t in trials]
            prefill = [t["base"]["generation_prefill_time_s"] / t["method"]["generation_prefill_time_s"] for t in trials]
            decode = [(t["base"]["decode_time_s"] / t["base"]["decode_steps"]) /
                      (t["method"]["decode_time_s"] / t["method"]["decode_steps"]) for t in trials]
            row = dict(method=method, retention=retention, pairs=40,
                total_speedup=statistics.median(total), prefill_speedup=statistics.median(prefill),
                decode_speedup=statistics.median(decode), faster_decode_pairs=sum(r > 1 for r in decode))
            rows.append(row)
            lines.append(f"| {method} | {retention:.0%} | {row['total_speedup']:.4f}× | {row['prefill_speedup']:.4f}× | {row['decode_speedup']:.4f}× | {row['faster_decode_pairs']}/40 |")
    lines += ["", "六种方法两档保留率的 decode 加速中位数均超过 1；prefill 包含各方法的选 token 开销，部分组仍慢于 base。",
        "", "复跑：", "```bash",
        ".venv/bin/python test/diagnostics/paired_native_decode.py --physical-gpu 0 --native-cuda-graphs --repetitions 10 --output-dir test/results/all_fa2_20260915/paired",
        "```"]
    (OUT / "paired/README.zh.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    paired = pairs()
    results, validations = {}, []
    for path in (OUT / "full").rglob("results.json"):
        summary = json.loads(path.read_text())
        retention = .05 if "_ret05" in str(path) else .2
        results[(summary["method"], retention)] = (path, summary)
    complete = len(results) == 14 and (OUT / "adapter200.json").exists()
    if args.require_complete:
        assert complete, len(results)
    rows = []
    for retention in [.05, .2]:
        if ("base", retention) not in results:
            continue
        base = results[("base", retention)][1]
        for method in ["base", *METHODS]:
            if (method, retention) not in results:
                continue
            path, summary = results[(method, retention)]
            predictions = json.loads(path.with_name("predictions.json").read_text())
            assert len(predictions) == 200
            assert summary["native_graph_logits_kv_exact"]
            assert summary["actual_attention_implementation"] == "flash_attention_2"
            for pred in predictions:
                close(pred["total_time_s"], pred["generation_prefill_time_s"] + pred["decode_time_s"] + pred["generation_overhead_s"])
                assert pred["native_graph_logits_kv_exact"]
                assert pred["decode_steps"] == pred["generated_tokens"] - 1
                assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in pred["generation_stages"][1:])
                close(pred["actual_prefill_kv_cache_mb"], pred["kv_cache_mb"])
            for key in ["total_time_s", "generation_prefill_time_s", "decode_time_s", "decode_steps"]:
                close(summary[key], sum(p[key] for p in predictions))
            old_path = next((ROOT / f"test/results/prefill_decode_corrected_20260915/ret{int(retention*100):02d}/{method}").rglob("predictions.json"))
            old = json.loads(old_path.read_text())
            changed = [i for i, (a, b) in enumerate(zip(old, predictions)) if a["prediction_text"] != b["prediction_text"]]
            validations.append(dict(method=method, retention=retention, exact_vs_eager=200, changed_vs_previous=changed))
            rows.append(dict(method=method, retention=1. if method == "base" else retention, group=f"ret{int(retention*100):02d}",
                samples=200, total_time_s=summary["total_time_s"], prefill_time_s=summary["generation_prefill_time_s"],
                standalone_prefill_time_s=summary["prefilling_time_s"], decode_time_s=summary["decode_time_s"],
                decode_steps=summary["decode_steps"], decode_ms_per_step=summary["decode_ms_per_step"],
                kv_cache_mib=summary["actual_prefill_kv_cache_mb"], flops=summary["flops"], score=summary["score"],
                preparation_s=summary["native_graph_prepare_s"],
                total_speedup=base["total_time_s"] / summary["total_time_s"],
                prefill_speedup=base["generation_prefill_time_s"] / summary["generation_prefill_time_s"],
                source=str(path.relative_to(OUT)), comparison="full-set shared-GPU cross-run ratio"))
    if (OUT / "adapter200.json").exists() and ("base", .05) in results:
        adapter = json.loads((OUT / "adapter200.json").read_text())[-1]
        base = results[("base", .05)][1]
        assert adapter["attention_implementation"] == "flash_attention_2"
        close(adapter["total_time_s"], adapter["prefilling_time_s"] + adapter["decode_forward_time_s"] + adapter["generation_overhead_s"])
        rows.append(dict(method="embedding_adapter", retention=None, group="ret05", samples=200,
            total_time_s=adapter["total_time_s"], prefill_time_s=adapter["prefilling_time_s"],
            standalone_prefill_time_s=adapter["prefilling_time_s"], decode_time_s=adapter["decode_forward_time_s"],
            decode_steps=adapter["decode_steps"], decode_ms_per_step=None if not adapter["decode_steps"] else 1000 * adapter["decode_forward_time_s"] / adapter["decode_steps"],
            kv_cache_mib=adapter["actual_prefill_kv_cache_mb"], flops=adapter["flops"], score=adapter["score"],
            total_speedup=base["total_time_s"] / adapter["total_time_s"],
            prefill_speedup=base["generation_prefill_time_s"] / adapter["prefilling_time_s"],
            source="adapter200.json", comparison="full-set shared-GPU cross-run ratio; original structured stop"))
        old = json.loads((ROOT / "test/results/prefill_decode_corrected_20260915/adapter.details.json").read_text())["predictions"]
        new = json.loads((OUT / "adapter200.details.json").read_text())["predictions"]
        assert len(new) == 200
        validations.append(dict(method="embedding_adapter", changed_vs_previous_sdpa=[i for i,(a,b) in enumerate(zip(old,new)) if a["adapter_text"] != b["adapter_text"]]))
    (OUT / "validation.json").write_text(json.dumps(dict(complete=complete, checks=validations, paired=paired), indent=2))
    if not complete:
        print(f"Validated {len(results)}/14 native rows; full report awaits completion")
        return
    (OUT / "table.json").write_text(json.dumps(rows, indent=2))
    with (OUT / "table.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Qwen3-VL-4B：统一 FA2 的 MMStar 200 复测", "",
        "当前 Pixmo static KL step2000；截图同一份 200 条数据；base、六种 baseline、adapter 视觉和文本 attention 都使用 FA2。",
        "adapter 保留 context + prefill-cache fast-path；base/剪枝使用原生 generate、DeepStack 和视觉/decoder CUDA Graph。",
        "全部 2800 条 native 请求在计时前逐条验证：图执行与 eager FA2 的 token、logits、KV 完全相同。",
        "修复了剪枝后 position IDs 导致的重复 FA2 元数据处理，并给原生视觉/decoder 接入图回放；VisionZip 显式传递序列长度和边界，图回放后恢复当前图像的选择器统计；ZooPrune 的额外预热不再消耗正式评测的随机数。",
        "六种方法各 4 条、8 对优化前后检查中，同方法 prefill 加速中位数为 2.48–2.82×，decode 为 3.24–4.26×；原始记录在 `*_check.json`，汇总在 `graph_optimization_pairs.json`。这是相对该方法 eager 执行的诊断，不是相对优化后 base 的加速比。",
        "", "## 同卡配对结果", "",
        "480 组交替请求中，六种 baseline 两档保留率的 decode 加速中位数为 1.015–1.044×。详见 [逐对结果](paired/README.zh.md)。",
        "", "## 全量耗时和资源", "",
        "本表为与训练共享 GPU 的全量运行记录。不同方法使用不同 GPU/时段，表中全量 speedup 是对应 base 的跨运行比值，含资源竞争差异；判定小幅加速请使用上面的同卡配对结果。",
        "时间为 200 条累计秒；prefill 是 generate 首次 forward（adapter 为 fast-path prefill）；decode ms/步只计后续 cached forward。",
        "KV 为每条样本的实际 K/V 张量平均 MiB。FLOPs 为原有 decoder prefill 解析估算、每条平均 TFLOPs，排除视觉、选择器、head、norm、softmax。",
        "原始停止规则保留：base/剪枝 EOS，adapter 识别到选项即停止；本轮 adapter 200 条均无 decode forward，N/A 表示没有此项观测。",
        "CUDA Graph 捕获、预热、逐条正确性核对不计入稳态时间；native preparation_s 单独保存在 JSON。", "",
        "| 组 | 方法 | 保留率 | Total s | Prefill s | Decode ms/步 | 实际 KV MiB | TFLOPs | Total 比值 | Prefill 比值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        retention = "—" if row["retention"] is None else f"{row['retention']:.0%}"
        decode = "N/A" if row["decode_ms_per_step"] is None else f"{row['decode_ms_per_step']:.3f}"
        lines.append(f"| {row['group']} | {row['method']} | {retention} | {row['total_time_s']:.4f} | {row['prefill_time_s']:.4f} | {decode} | {row['kv_cache_mib']:.2f} | {row['flops']/1e12:.3f} | {row['total_speedup']:.3f}× | {row['prefill_speedup']:.3f}× |")
    lines += ["", "## 实际算子核对", "",
        "`*_operators.json`：原生 base/六种 baseline 在禁止 SDPA 的检查中均观测到 FA2 算子。",
        "`adapter_runtime_check.json`：每次 prefill 60 次 FA2（24 视觉 + 36 文本）；两种 decode 实现连续 3 步各 108 次 FA2，SDPA 调用为零。",
        "`test_qwen_adapter_fa2.py`：图片前文本、多图交错、GQA、cached decode 和 CUDA Graph 回放验证。",
        "adapter 相比旧 SDPA 结果有 5/200 条答案变化，索引为 71、138、164、183、188。`adapter_backend_delta.json` 在同一视觉 hidden states 上仅切换 attention plan，逐条复现了两份结果；相关答案 logit 差距为 0 或 0.25。FA2 与 SDPA 的 BF16 结果不保证逐位相同，不能把旧 SDPA 准确率复制到 FA2 行。",
        "`full/protocol.json`、`full/gpu*.commands.json`、`adapter200.protocol.json` 保存命令、环境和源文件摘要；`validation.json` 保存旧结果对照和阶段校验。"]
    (OUT / "README.zh.md").write_text("\n".join(lines) + "\n")
    print("PASS: 14 native rows, adapter 200, and 480 paired trials")


if __name__ == "__main__":
    main()
