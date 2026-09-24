"""Validate and report the DeepStack-off rerun, preserving all base denominators."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmarking.common.peak_memory import summarize_peak_memory
OUT = ROOT / "test/results/deepstack_off_20260915"
FULL = OUT / "full"
METHODS = ["fastv", "dart", "visionzip", "sparsevlm", "divprune", "zoo"]
DISPLAY = dict(fastv="FastV", dart="DART", visionzip="VisionZip", sparsevlm="SparseVLM",
    divprune="DivPrune", zoo="ZooPrune", embedding_adapter="Embedding adapter")


def read(path):
    return json.loads(path.read_text())


def load_native(group, method):
    paths = [p for p in (FULL / group).rglob("results.json") if read(p)["method"] == method]
    assert len(paths) == 1, (group, method, paths)
    path = paths[0]
    result, predictions = read(path), read(path.with_name("predictions.json"))
    assert result["samples"] == len(predictions) == 200
    assert result["deepstack"] == "off"
    assert result["actual_attention_implementation"] == "flash_attention_2"
    assert result["native_graph_logits_kv_exact"]
    for p in predictions:
        assert p["native_graph_logits_kv_exact"]
        assert p["decode_steps"] == p["generated_tokens"] - 1
        assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in p["generation_stages"][1:])
        assert math.isclose(p["total_time_s"], p["generation_prefill_time_s"] + p["decode_time_s"] + p["generation_overhead_s"], abs_tol=1e-8)
    return result, predictions, str(path.relative_to(OUT))


def native_row(group, method, retention, result, ref, source):
    return dict(group=group, method=method, retention=retention, samples=200,
        total_time_s=result["total_time_s"], prefill_time_s=result["generation_prefill_time_s"],
        standalone_prefill_time_s=result["prefilling_time_s"],
        kv_cache_mib=result["actual_prefill_kv_cache_mb"], analytic_prefill_flops=result["flops"],
        decode_forward_time_s=result["decode_time_s"], decode_steps=result["decode_steps"],
        decode_ms_per_step=result["decode_ms_per_step"], generation_overhead_s=result["generation_overhead_s"],
        score=result["score"], reference_total_s=ref["total_time_s"], reference_prefill_s=ref["generation_prefill_time_s"],
        reference_standalone_prefill_s=ref["prefilling_time_s"],
        speedup_total=ref["total_time_s"]/result["total_time_s"],
        speedup_prefill=ref["generation_prefill_time_s"]/result["generation_prefill_time_s"],
        speedup_standalone_prefill=ref["prefilling_time_s"]/result["prefilling_time_s"], source=source)


def main():
    protocol = read(FULL / "protocol.json")
    assert protocol["status"] == "complete", protocol["status"]
    rows, references, base_answers, checks = [], [], [], []
    for retention in [.05, .2]:
        for method in METHODS:
            group = f"{method}_ret{int(retention*100):02d}"
            ref, base_predictions, ref_source = load_native(group, "base")
            result, predictions, source = load_native(group, method)
            references.append(native_row(group, "base", 1., ref, ref, ref_source))
            rows.append(native_row(group, method, retention, result, ref, source))
            base_answers.append([p["prediction_text"] for p in base_predictions])
            checks.append(dict(group=group, samples=200, logits_tokens_kv_exact_vs_eager=True))
    ref, predictions, ref_source = load_native("adapter_reference", "base")
    base_answers.append([p["prediction_text"] for p in predictions])
    references.append(native_row("adapter_reference", "base", 1., ref, ref, ref_source))
    adapter_source = FULL / "adapter200_native.json"
    adapter_command = read(FULL / "adapter200_native.command.json")
    assert adapter_command['status'] == 'complete' and adapter_command['exit_code'] == 0
    protocol['adapter_revision'] = adapter_command
    adapter = next(r for r in read(adapter_source) if r["method"] == "embedding_adapter")
    detail = read(adapter_source.with_suffix('.details.json'))
    assert adapter["samples"] == len(detail["predictions"]) == 200
    assert adapter["attention_implementation"] == "flash_attention_2"
    assert all(math.isclose(p["adapter_total_s"], p["adapter_prefill_s"]+p["adapter_decode_s"], abs_tol=1e-8) for p in detail["predictions"])
    rows.append(dict(group="adapter_reference", method="embedding_adapter", retention=None, samples=200,
        total_time_s=adapter["total_time_s"], prefill_time_s=adapter["prefilling_time_s"],
        standalone_prefill_time_s=adapter["prefilling_time_s"], kv_cache_mib=adapter["actual_prefill_kv_cache_mb"],
        analytic_prefill_flops=adapter["flops"], decode_forward_time_s=adapter["decode_forward_time_s"],
        decode_steps=adapter["decode_steps"], decode_ms_per_step=1000*adapter["decode_forward_time_s"]/adapter["decode_steps"] if adapter["decode_steps"] else None,
        generation_overhead_s=adapter["generation_overhead_s"], score=adapter["score"],
        reference_total_s=ref["total_time_s"], reference_prefill_s=ref["generation_prefill_time_s"],
        reference_standalone_prefill_s=ref["prefilling_time_s"],
        speedup_total=ref["total_time_s"]/adapter["total_time_s"],
        speedup_prefill=ref["generation_prefill_time_s"]/adapter["prefilling_time_s"],
        speedup_standalone_prefill=ref["prefilling_time_s"]/adapter["prefilling_time_s"], source=str(adapter_source.relative_to(OUT))))
    paired = read(OUT / "paired/summary.json")
    assert len(paired) == 13
    assert read(OUT / "paired/protocol.json")["deepstack"] == "off"
    paired = [r for r in paired if r['method'] != 'embedding_adapter'] + read(OUT / 'adapter_final/summary.json')
    pair_count = 0
    for r in paired:
        folder = 'adapter_final' if r['method'] == 'embedding_adapter' else 'paired'
        r['source'] = f"{folder}/{r['method']}_ret{r['retention']}.json"
        trials = read(OUT / r['source'])
        assert len(trials) == r['pairs']
        for trial in trials:
            assert trial["exact_vs_eager"]
            for label in ["base", "method"]:
                assert len(trial[label]["tokens"]) == 8 and trial[label]["decode_steps"] == 7
        pair_count += len(trials)
        for field, stage in [('generation_prefill_time_s', 'prefill'), ('decode_time_s', 'decode'), ('total_time_s', 'total')]:
            assert math.isclose(statistics.median(t['base'][field]/t['method'][field] for t in trials), r[stage+'_paired_speedup'])
    retention_rows = read(OUT / 'retention_interleaved/summary.json')
    assert len(retention_rows) == 6
    triple_count = 0
    for r in retention_rows:
        r['source'] = f"retention_interleaved/{r['method']}.json"
        trials = read(OUT / r['source'])
        assert len(trials) == r['triples'] == 48
        for t in trials:
            assert t['exact_vs_eager']
            for label in ['base', 'ret05', 'ret20']:
                assert len(t[label]['tokens']) == 8 and t[label]['decode_steps'] == 7
        for field, stage in [('generation_prefill_time_s', 'prefill'), ('decode_time_s', 'decode'), ('total_time_s', 'total')]:
            assert math.isclose(statistics.median(t['ret20'][field]/t['ret05'][field] for t in trials), r[stage+'_ret05_speedup_over_ret20'])
        triple_count += len(trials)
    old_adapter = read(ROOT / "test/results/all_fa2_20260915/adapter200.details.json")["predictions"]
    changes = [p["index"] for p,q in zip(detail["predictions"],old_adapter) if p["adapter_text"] != q["adapter_text"]]
    memory_protocol = read(OUT / 'memory200/protocol.json')
    assert memory_protocol['status'] == 'complete'
    protocol['peak_memory'] = memory_protocol
    memory_checks = []
    for row in rows + references:
        if row['method'] == 'embedding_adapter':
            row.update(summarize_peak_memory(detail['predictions'], prefix='adapter_'))
            row['peak_memory_source'] = str(adapter_source.with_suffix('.details.json').relative_to(OUT))
        else:
            group = 'base' if row['method'] == 'base' else row['group']
            files = list((OUT / 'memory200' / group).rglob('results.json'))
            assert len(files) == 1
            memory = read(files[0])
            observations = read(files[0].with_name('predictions.json'))
            assert memory['samples'] == len(observations) == 200
            assert memory['deepstack'] == 'off' and memory['actual_attention_implementation'] == 'flash_attention_2'
            assert all(p['native_graph_logits_kv_exact'] for p in observations)
            original = read((OUT / row['source']).with_name('predictions.json'))
            assert [p['prediction_text'] for p in observations] == [p['prediction_text'] for p in original]
            measured = summarize_peak_memory(observations)
            assert math.isclose(measured['peak_memory_mb'], memory['peak_memory_mb'])
            row.update(measured)
            row['peak_memory_source'] = str(files[0].relative_to(OUT))
            memory_checks.append(dict(group=row['group'], samples=200, answers_unchanged=True))
        assert row['peak_memory_samples'] == 200
    validation = dict(native_checks=checks, native_reference_samples=13*200, method_samples=13*200,
        fixed_work_pairs=pair_count, retention_triples=triple_count, fixed_work_tokens_per_request=8, fixed_work_cached_steps_per_request=7,
        base_answers_identical_across_references=all(a==base_answers[0] for a in base_answers),
        adapter_changed_vs_previous_fa2_indices=changes, deepstack="off", attention="flash_attention_2")
    validation['memory_checks'] = memory_checks
    (OUT / "validation.json").write_text(json.dumps(validation,indent=2))
    (OUT / "table.json").write_text(json.dumps(dict(protocol=protocol,rows=rows,references=references,paired=paired,retention_interleaved=retention_rows),indent=2))
    with (OUT / "table.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = ["# Qwen3-VL-4B：FA2、DeepStack 关闭后的复测", "",
        "## 本次配置", "",
        "- 数据：截图对应的 MMStar 200 子集；当前 Pixmo static KL 2000-step adapter。",
        "- base、六个 baseline、adapter 全部 FA2。DeepStack 的视觉侧 merger 分支和语言侧注入均关闭，禁止执行的检查贯穿预热与计时。主视觉编码器和主 merger 每个请求正常执行。",
        "- Adapter 使用原有 fast-path prefill，prefill 内一次性构建原生 KV，随后复用原生 Qwen 单 token FA2 decoder；base/baseline 也使用 vision、prefill layer、整段 cached decode CUDA Graph。捕获和一致性校验在计时之外。",
        "- 训练始终运行。各方法与自己的 base 在同一 GPU 上评测；完整 200 样本是分轮运行，固定 8 token 诊断才是同卡交替成对运行。这里不是独占 GPU 的绝对延迟结果。", "",
        "## 5% 为什么曾比 20% 慢：同输入直接交替复测", "",
        "每方法 4 个输入 ×12 轮，循环 base / 5% / 20% 的全部六种顺序。每请求固定 8 token，保留全部慢样本。下面是逐组三联请求中 20% 耗时 / 5% 耗时的中位数，大于 1 表示 5% 更快。", "",
        "| Method | 5% 相对 20% prefill speedup | Decode speedup | Total speedup |",
        "|---|---:|---:|---:|"]
    for r in retention_rows:
        lines.append(f"| {DISPLAY[r['method']]} | {r['prefill_ret05_speedup_over_ret20']:.4f}× | {r['decode_ret05_speedup_over_ret20']:.4f}× | {r['total_ret05_speedup_over_ret20']:.4f}× |")
    lines += ["", "六种方法的 5% prefill 都比 20% 快 4.5%–7.6%；decode 加速比仅 0.9967–1.0057×。分轮测量受并行训练负载变化影响，不能根据先后两轮总秒数判断 retention 的因果影响。这四个输入的诊断不等于 200 样本或独占卡的稳定排名。", "",
        "当前 adapter 固定 8 token 的 64 对交替请求，decode 逐对加速比中位数为 1.0133×，58/64 对快于 base；base / adapter 的 decode 中位耗时为 8.168 / 8.063 ms/step。Prefill 逐对加速比中位数仍为 0.9295×，没有证明 adapter prefill 已稳定快于 base。", "",
        "## 截图为什么不能共用一个 base 分母", "",
        "将截图的时间乘以 speedup，可反推出三组参照（四舍五入会有小误差）：", "",
        "| 截图分组 | base total s | base prefill s |", "|---|---:|---:|",
        "| 5% baseline | 19.2342 | 17.6140 |", "| 20% baseline | 17.2548 | 15.7758 |",
        "| δ-Vision | 15.8225 | 7.9274 |", "",
        "截图本身没有注明 DeepStack、attention backend、checkpoint 和停止规则，不能据此认定这些配置相同。当前 checkpoint 也不是先前截图记录中的旧 checkpoint。逐行转录及反算见 [screenshot_audit.json](screenshot_audit.json)。", "",
        "## MMStar 200：实际生成阶段", "",
        "Total、prefill 为 200 个请求的秒数之和；prefill 指生成中到首 token logits 的实际 forward，包含 vision。KV 是平均实际 prefill K/V 存储，单位 MiB。FLOPs 是平均解析 decoder-prefill FLOPs，不含 vision、选择器、head、norm、softmax；它不是端到端总 FLOPs。", "",
        "Peak Memory 是全部 200 个请求中的 CUDA allocated 最大峰值，包含该进程的模型权重、驻留图池、KV 和临时激活，单位 MiB；不包含其他训练进程或 CUDA allocator 之外的驱动内存。预热/捕获本身不参与取峰值，其保留下来的图缓冲区计入运行占用。base/baseline 在独立进程补测同一 200 样本，答案与速度实验完全一致；adapter 由最终 200 样本逐请求实测峰值汇总。", "",
        "| Method | Retention | Total s | Prefill s | KV MiB | Peak Memory MiB | FLOPs | Total speedup | Prefill speedup | 对应 base total / prefill s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        retention=f"{r['retention']:.0%}" if r['retention'] is not None else "未剪视觉 token"
        lines.append(f"| {DISPLAY[r['method']]} | {retention} | {r['total_time_s']:.4f} | {r['prefill_time_s']:.4f} | {r['kv_cache_mib']:.2f} | {r['peak_memory_mb']:.2f} | {r['analytic_prefill_flops']:.4e} | {r['speedup_total']:.4f}× | {r['speedup_prefill']:.4f}× | {r['reference_total_s']:.4f} / {r['reference_prefill_s']:.4f} |")
    lines += ["", f"独立 base 的 Peak Memory：{references[0]['peak_memory_mb']:.2f} MiB。JSON/CSV 同时提供 prefill/decode 阶段最大峰值、每请求峰值的平均值以及 reserved 最大值；adapter 自然停止无 decode 时，decode 阶段峰值为 null。",
        "", "原入口另测的 standalone prefill 及其分母/speedup 也完整保存在 table.json/CSV。它与 generate 内 prefill 来自不同的实际 forward，不能用 total 减去 standalone prefill 来推断 decode。", "",
        "## 固定 8 token：prefill / decode 速度", "",
        "每 baseline、每 retention：4 个输入（数据集偏移 0、25、125、133）×10 对交替请求；最终 native decoder 版 adapter 为同样 4 个输入 ×16 对。各方法各自与 base 成对，跨方法来自不同时间窗口。所有方法都禁止 EOS 提前结束，实际进行 1 次 prefill + 7 次增长 KV 的 decode。每个输入先核对全部 8 步 logits、token、最终 KV 与 eager 完全相同。Speedup 是逐对 base/method 比值的中位数，不是两列中位耗时相除。", "",
        "| Method | Retention | Prefill speedup | Decode speedup | Total speedup | Base / method decode ms/step |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        retention=f"{r['retention']:.0%}" if r['retention'] is not None else "未剪视觉 token"
        lines.append(f"| {DISPLAY[r['method']]} | {retention} | {r['prefill_paired_speedup']:.4f}× | {r['decode_paired_speedup']:.4f}× | {r['total_paired_speedup']:.4f}× | {r['base_decode_median_ms']:.3f} / {r['method_decode_median_ms']:.3f} |")
    lines += ["", "完整 MMStar 自然停止运行中，adapter 实际 decode steps = " + str(adapter['decode_steps']) + "。若为 0，则该协议没有 adapter 的单 token decode 速度，不能把少生成 token 的 total 优势当作 decode 优势。", "",
        "## 已修正的执行问题", "",
        "- DART 不再逐个把 GPU top-k 标量同步回 CPU；候选顺序、集合处理和 top-k tie 行为保持，邻居 tensor 运算可以 replay。",
        "- DivPrune/ZooPrune 贪心迭代移除了逐 token 标量索引同步，选择器使用预热图；计时中的 pruning 审计 CPU 拷贝关闭。",
        "- 全部方法使用整段单 token decode 图；KV 的输出复制合并分配，并保留独立数据所有权。",
        "- Adapter fast decode 不再重算全段 text prefix；prefill 不保留该旧模式的各层中间激活。无 padding 且全前缀可见时，decode 直接使用 dense FA2，跳过 varlen gather 和逐步 mask 的 CPU 回读。",
        "- Adapter 改为原生 Qwen FA2 decoder：prefill 输出时一次性打包 KV，直接构建独立 DynamicCache，避免先复制分段 KV 再拼接。显式传入序列位置和 M-RoPE 位置，保持 logits/KV 精确相同。其视觉 KV 数量仍与 base 相同，平均实际 prefill K/V 47.45 MiB。",
        "- 旧 decoder 对照实验：共享视觉 KV 减少逐步重复复制，decode 比旧版快 4.06%；进一步复用原生 decoder 比共享版本快 4.72%。这是各自同模型交替实验，不能把它们当作相对 base 的加速比。原始结果见 adapter_cache_strategies.json 和 native_adapter_probe.json。",
        "- 修复实际缓存统计：识别原生 DynamicCache；生成器更新外部最终缓存句柄，最终 KV 随 decode token 增长。主 KV 列采用实测，旧解析估计另存。", "",
        "## 校验与复现", "",
        f"固定工作量共 {pair_count} 对请求，另有 {triple_count} 组三联 retention 请求；各输入的图/eager token、logits、KV 一致。200 样本 adapter 相对上一版 FA2 答案变化数：{len(changes)}。所有 base 参照的答案是否完全一致：{validation['base_answers_identical_across_references']}。", "",
        "- [table.json](table.json)、[table.csv](table.csv)：完整数值、全部独立 base 参照、原始文件位置。",
        "- [validation.json](validation.json)：校验汇总。",
        "- [memory200/protocol.json](memory200/protocol.json)：Peak Memory 补测配置；各行 peak_memory_source 指向对应的逐样本记录或汇总。",
        "- [full/protocol.json](full/protocol.json)、full/*.command.json：精确命令、源码与数据/checkpoint SHA256。",
        "- [full/adapter200_native.command.json](full/adapter200_native.command.json)：最终 adapter 200 样本的新版本命令与源码哈希；其 base 分母仍为先前同卡原生 base 运行，因此完整表的 adapter 比值也受时间窗口影响。旧 adapter200.json 保留作历史记录。",
        "- [retention_interleaved/summary.json](retention_interleaved/summary.json)、[adapter_final/summary.json](adapter_final/summary.json)：最新交替复测。",
        "- `test/diagnostics/run_deepstack_off.py` 重跑完整 200；`test/diagnostics/paired_runtime_execution.py --deepstack off --pairs 10` 重跑固定工作量。"]
    (OUT / "README.zh.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    main()
