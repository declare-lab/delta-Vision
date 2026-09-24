"""Validate and report the second exact adapter optimization round, read-only inference."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'test/results/adapter_max_20260915/videomme999_final'
PREVIOUS = ROOT / 'test/results/adapter_exact_20260915/videomme999_final'
BASE = ROOT / 'test/results/video_base_20260915/videomme999'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def load_rows(directory, variant):
    rows = [json.loads(line) for path in directory.glob(variant + '_*.jsonl')
            for line in path.read_text().splitlines()]
    indexed = {row['index']: row for row in rows}
    assert len(rows) == len(indexed) == 999
    assert set(indexed) == set(range(999))
    return indexed


def totals(rows):
    return {key: sum(statistics.median(t[key] for t in row['trials']) for row in rows)
            for key in ['total_s', 'prefill_s', 'decode_s']}


def main():
    summary, protocol, validation = [read(OUTPUT / (name + '.json'))
                                     for name in ['summary', 'protocol', 'validation']]
    assert summary['samples'] == 999 and summary['tokens_per_request'] == 8
    assert summary['all_logits_and_kv_bitwise_equal'] and not summary['exact_mismatches']
    assert protocol['variants'] == {'original': 'exact', 'optimized': 'max'}
    assert protocol['runs'] == 3 and protocol['dtype'] == 'bfloat16'
    assert summary['attention'] == 'flash_attention_2' and summary['deepstack'] == 'off'
    assert validation['logits_checked'] == 7992 and validation['timed_requests'] == 5994
    assert validation['sdpa_calls'] == validation['timed_captures'] == validation['timed_fallbacks'] == 0
    assert read(OUTPUT / 'source_validation.json')['all_sources_unchanged']
    rows = {v: load_rows(OUTPUT, v) for v in ['original', 'optimized']}
    previous, base = load_rows(PREVIOUS, 'optimized'), load_rows(BASE, 'optimized')
    from src.benchmarking.engines.adapter import EXACT_FIELDS
    for index in range(999):
        for variant, data in rows.items():
            row = data[index]
            assert row['optimization_level'] == protocol['variants'][variant]
            assert len(row['trials']) == 3
            assert all(t['decode_steps'] == 7 for t in row['trials'])
            assert all(row[key] == previous[index][key] for key in EXACT_FIELDS), (index, variant)
            assert row['input_sha256'] == base[index]['input_sha256']
            assert row['flops'] == previous[index]['flops']
            assert row['trials'][0]['kv_cache_mb'] == previous[index]['trials'][0]['kv_cache_mb']
    cross_validation = dict(samples=999, all_current_variants_match_saved_v1_logits_and_kv=True,
        all_inputs_match_saved_base=True, saved_v1_result=str(PREVIOUS), saved_base_result=str(BASE),
        unchanged_fields=EXACT_FIELDS + ['flops', 'kv_cache_mb'],
        visual_token_counts=dict(sorted(Counter(r['visual_tokens'] for r in rows['optimized'].values()).items())))
    write(OUTPUT / 'previous_result_validation.json', cross_validation)
    groups = {}
    for field, values in [('duration', ['short', 'medium', 'long']), ('shard', range(8))]:
        groups[field] = {}
        for value in values:
            data = {}
            for variant, indexed in rows.items():
                selected = [r for r in indexed.values() if r[field] == value]
                if field == 'duration':
                    assert len(selected) == 333
                data[variant] = dict(samples=len(selected), **totals(selected))
            data['speedup_vs_v1'] = {k:data['original'][k]/data['optimized'][k]
                                     for k in ['total_s', 'prefill_s', 'decode_s']}
            groups[field][value] = data
    write(OUTPUT / 'group_summary.json', groups)
    a, b = [summary['variants'][v] for v in ['original', 'optimized']]
    speed = {key: a[key]/b[key] for key in ['total_s', 'prefill_s', 'decode_s']}
    table = [dict(method='adapter_' + protocol['variants'][variant], **summary['variants'][variant],
                  total_speedup_vs_v1=a['total_s']/summary['variants'][variant]['total_s'],
                  prefill_speedup_vs_v1=a['prefill_s']/summary['variants'][variant]['prefill_s'],
                  decode_speedup_vs_v1=a['decode_s']/summary['variants'][variant]['decode_s'])
             for variant in ['original', 'optimized']]
    with (OUTPUT / 'adapter_table.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    group_lines = '\n'.join(
        f"| {group} | {data['original']['prefill_s']*1000/333:.3f} → {data['optimized']['prefill_s']*1000/333:.3f} ms | "
        f"{data['original']['decode_s']*1000/(333*7):.3f} → {data['optimized']['decode_s']*1000/(333*7):.3f} ms/token |"
        for group, data in groups['duration'].items())
    report = f'''# Adapter 第二轮无损加速：Video-MME 999

## 完整结果

本表比较**上轮已优化的 adapter（exact / v1）和本轮 adapter（max / v2）**，两版均在本轮重新运行。
同一个 Qwen3-VL-4B-Instruct、Pixmo static KL step2000 checkpoint，FA2、BF16、DeepStack 关闭；Video-MME 原始 999 条，短/中/长各 333 条。

| 指标 | 上轮优化版，本轮复测 | 本轮优化版 | 相对上轮加速 |
|---|---:|---:|---:|
| Total，999 条累计 | {a['total_s']:.4f} s | {b['total_s']:.4f} s | {speed['total_s']:.4f}× |
| Prefill，999 条累计 | {a['prefill_s']:.4f} s | {b['prefill_s']:.4f} s | {speed['prefill_s']:.4f}× |
| Prefill，平均每条 | {a['prefill_s']*1000/999:.4f} ms | {b['prefill_s']*1000/999:.4f} ms | |
| Decode，999 条累计 | {a['decode_s']:.4f} s | {b['decode_s']:.4f} s | {speed['decode_s']:.4f}× |
| Decode，每 token | {a['decode_ms_per_token']:.4f} ms | {b['decode_ms_per_token']:.4f} ms | |
| Decode，token/s | {a['decode_tokens_per_second']:.2f} | {b['decode_tokens_per_second']:.2f} | |
| Prefill KV，平均每条 | {a['kv_cache_mb']:.4f} MiB | {b['kv_cache_mb']:.4f} MiB | |
| Decoder prefill FLOPs，平均每条 | {a['flops']:.6e} | {b['flops']:.6e} | |
| Peak memory，所有请求最大 | {a['peak_memory_mb']/1024:.4f} GiB | {b['peak_memory_mb']/1024:.4f} GiB | |

Total 耗时下降 {(1-b['total_s']/a['total_s'])*100:.2f}%；prefill 耗时下降 {(1-b['prefill_s']/a['prefill_s'])*100:.2f}%；decode 吞吐提升 {(speed['decode_s']-1)*100:.2f}%。
峰值显存下降 {(a['peak_memory_mb']-b['peak_memory_mb'])/1024:.4f} GiB。KV 和解析 FLOPs 相同，未减少输入帧、视觉 token、层数或生成步数。

| 视频长度，各 333 条 | Prefill，平均每条 | Decode |
|---|---:|---:|
{group_lines}

结果：[summary.json](videomme999_final/summary.json)、[CSV](videomme999_final/adapter_table.csv)、[按视频长度和 GPU 分组](videomme999_final/group_summary.json)。
`original_*.jsonl` 在本轮指 **v1 exact**，`optimized_*.jsonl` 指 **v2 max**；每行的 `optimization_level` 和协议均有明确记录。

## 本轮解决的开销

1. **Decode 整份 KV 的反复整理。** 36 层 K/V 使用一个连续输入缓冲区和一个连续输出缓冲区；native DynamicLayer 的相同 `torch.cat` 直接写入各层输出位置。相邻步输入由 72 个张量复制改为一次连续复制，输出仅克隆一次整块 KV。返回的 KV 独立拥有存储；若调用方替换某一层的视图，就回退为复制实际层张量。
2. **RMSNorm 多次启动内核。** 用一个 Triton 内核复现当前 PyTorch 的 FP32 mean 累加顺序，保留归一化和乘权重时的 BF16 舍入。Q/K RMSNorm 和 RoPE 进一步合为一个内核，仍保留全部中间舍入，关闭 FMA 合并。
3. **Video prefill 重复打包相同前缀。** 原方案为每段文本重复打包其可见的视觉/文本 KV。本轮仅存一份按原位置排序的 KV，各段传入可见长度 `seqused_k`，仍使用原来的 FA2 非分页内核和因果边界。
4. **视觉编码之后的 CPU/GPU 同步。** 在发起视觉计算之前读取文本/视觉位置拓扑；视觉计算期间在 CPU 准备索引，通过 pinned memory 异步上传，消除视觉计算后不必要的同步等待。每次请求仍读取当前输入并重新计算视觉特征与 KV。

入口为 `src/benchmarking/common/prefill.py::build_qwen_fast_adapter_prefill` 的 `adapter_max_optimizations=True` 分支。
`--adapter-max-optimizations` 自动包含上轮 `--adapter-exact-optimizations`，适用于本次验证的 BF16、FA2、batch size 1 推理。
配合 `--cuda-graph --cuda-graph-context --adapter-decode-cache --adapter-decode-cache-mode fast --last-logits-only` 使用；比较入口保持 `--comparison-deepstack off`。
使用 `--adapter-exact-optimizations --no-adapter-max-optimizations` 可回到本轮对照的 v1。

## 精度和请求独立性

- 全部 999 条的 7,992 份完整词表 logits、prefill 及最终所有 36 层 K/V SHA256、生成 token 与 v1 逐条相同；同时逐条匹配上轮保存的 v1 999 条结果。
- 输入张量 SHA256 与既有 base 的 999 条结果全部相同；实际视觉 token 数覆盖 880、1000、1008、1012、1056。
- 6 项 GPU 单元测试覆盖原生归约顺序、不同数值尺度、Q/K 布局、RoPE 中间舍入、视频因果关系、共享前缀、KV 布局与改变输入后的 Graph 重放。
- 额外用两份不同视频输入交错执行，15 项独立性检查通过；改动视频帧会改变输出；请求间 KV 不互相覆盖；替换单层缓存视图的回退正确。
- 每条生成 8 个 token，即 prefill 后 7 次 KV 真正增长的 decode。两版分别测 2,997 次完整请求；计时内 Graph 捕获、回退和 SDPA 调用均为 0。
- 归约顺序与 FA2 私有推理接口依赖当前本地库实现；换库版本或硬件后应重新执行逐位校验。

证据：[本轮校验](videomme999_final/validation.json)、[与旧结果交叉校验](videomme999_final/previous_result_validation.json)、[请求独立性](ownership.json)、[源文件未变更校验](videomme999_final/source_validation.json)。

## 测量口径和公平性

- 沿用 8 帧、`full_timestamp_v1`、`media_first_v1`、无字幕。输入缓存仅是 CPU processor 张量；每次计时重新执行视觉编码、adapter prefill、KV 构建和 decode。
- Prefill 从请求开始到首个 logits 与独立 KV 返回，包含位置准备、视觉编码和 adapter。Decode 只计 7 次单 token forward；Total 还包含 token 选择和同步。
- CPU 抽帧、processor、输入搬到 GPU、预热、Graph 捕获及一致性哈希均在计时外。这是**预热后的请求延迟**；没有测首次请求的编译/捕获耗时。
- 每项延迟各取每条 3 次完整请求的中位数，再对 999 条求和；阶段中位数相加不要求精确等于 Total 中位数。累计时间是请求延迟总和，不是 8 GPU 作业墙钟时间。
- 8 张 H200 按 index % 8 分片，每张 GPU 每次只有一版模型；两版在同一 GPU 测同一分片，奇偶分片反转运行顺序。两版 decode Graph 上限均为 12。
- Peak memory 为预热后单请求 CUDA allocated 峰值的最大值，包含模型、adapter、Graph 缓冲区；单位 GiB，KV 单位 MiB。
- FLOPs 为既有解析 decoder prefill 口径，乘加计 2 FLOPs，包括文本、视觉 K/V 和 adapter 投影；不包含视觉编码器、LM head、归一化、softmax、重复数据搬运。它不是完整请求的硬件计数。
- 本轮 speedup 的分母全部是**同协议复测的 v1 adapter**。本轮未重测 base；既有 base 的 prefill 不含生成侧位置准备，Graph 上限也不同，不能直接作为本表 prefill speedup 分母。
- Norm/RoPE 和 packed decode KV 也适用于 base 的原生 decoder。后续方法比较需要向 base 和其他 baseline 提供同等通用优化；本表的 decode 提升属于运行实现优化，不能据此宣称 adapter 架构本身比同等优化的 base decode 更快。

## 未采用的候选

- FA2 分页缓存：测试出现 BF16 输出差异，未采用；本轮采用的是输出逐位相同的非分页共享前缀。
- 再合并 36 层视觉归一化与 K/V 投影：实际仅约 0.1～0.2 ms 收益，额外权重/缓冲区成本较大，未合入；上轮已有的 adapter memory 批量投影仍保留。
- 未采用低精度量化、改变归约顺序的近似 RMSNorm、视觉特征跨请求复用或删减计算。

诊断：[分页与共享前缀对照](paged_fa2_probe.json)、[视觉批量投影](visual_batch_probe.json)。
保留的逐阶段 pilot 在 `packed_decode_pilot`、`norm_rope_pilot`、`shared_prefix_pilot`、`async_prepare_pilot_fixed`；完整表仅采用本轮 999 条结果。

## 复跑

```bash
.venv/bin/python -m src.benchmarking.engines.adapter \\
  --gpus 0 1 2 3 4 5 6 7 --runs 3 --tokens 8 \\
  --reference-level exact --optimized-level max \\
  --input-cache {protocol['input_cache']} \\
  --output test/results/adapter_max_rerun
```

输出目录必须新建。完整 [protocol.json](videomme999_final/protocol.json) 固定模型、checkpoint、数据清单、运行参数和源文件 SHA256；[environment.json](videomme999_final/environment.json) 记录本地环境。
Checkpoint：`{protocol['checkpoint']}`，SHA256 `{protocol['checkpoint_sha256']}`。
历史截图和上轮结果均保留。
'''
    (OUTPUT.parent / 'README.zh.md').write_text(report)
    print(json.dumps(dict(samples=999, saved_v1_match=True, base_input_match=True,
                         variants=summary['variants'], speedup_vs_v1=speed), indent=2))


if __name__ == '__main__':
    main()
