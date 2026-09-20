"""Summarize the completed paired adapter run without launching inference."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'test/results/adapter_exact_20260915/videomme999_final'


def main():
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    validation = json.loads((OUTPUT / 'validation.json').read_text())
    assert summary['samples'] == 999 and summary['all_logits_and_kv_bitwise_equal']
    assert summary['tokens_per_request'] == 8 and protocol['runs'] == 3
    assert validation['logits_checked'] == 7992 and validation['timed_requests'] == 5994
    assert json.loads((OUTPUT / 'source_validation.json').read_text())['all_sources_unchanged']
    rows = {v: {r['index']: r for p in OUTPUT.glob(v + '_*.jsonl')
                for r in map(json.loads, p.read_text().splitlines())}
            for v in ['original', 'optimized']}
    assert all(set(data) == set(range(999)) for data in rows.values())
    base_root = ROOT / 'test/results/video_base_20260915/videomme999'
    base = {r['index']: r for p in base_root.glob('optimized_*.jsonl')
            for r in map(json.loads, p.read_text().splitlines())}
    assert len(base) == 999
    assert all(base[i]['input_sha256'] == rows['original'][i]['input_sha256']
               == rows['optimized'][i]['input_sha256'] for i in base)
    duration = {}
    for group in ['short', 'medium', 'long']:
        duration[group] = {}
        for variant, data in rows.items():
            selected = [r for r in data.values() if r['duration'] == group]
            assert len(selected) == 333
            duration[group][variant] = dict(samples=len(selected), **{
                key: sum(statistics.median(t[key] for t in row['trials']) for row in selected)
                for key in ['total_s', 'prefill_s', 'decode_s']})
    (OUTPUT / 'duration_summary.json').write_text(json.dumps(duration, indent=2) + '\n')
    (OUTPUT / 'base_input_validation.json').write_text(json.dumps(dict(
        samples=999, all_input_tensors_identical_to_saved_base=True,
        base_result=str(base_root / 'optimized_summary.json')), indent=2) + '\n')
    a, b = (summary['variants'][v] for v in ['original', 'optimized'])
    speed = summary['speedup_vs_original_adapter']
    table = [dict(method='adapter_' + variant, **summary['variants'][variant],
                  total_speedup_vs_original_adapter=a['total_s']/summary['variants'][variant]['total_s'],
                  prefill_speedup_vs_original_adapter=a['prefill_s']/summary['variants'][variant]['prefill_s'],
                  decode_speedup_vs_original_adapter=a['decode_s']/summary['variants'][variant]['decode_s'])
             for variant in ['original', 'optimized']]
    with (OUTPUT / 'adapter_table.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report = f'''# Adapter fast-path 无损优化：Video-MME 999

## 完整结果

同一个 Pixmo static KL step2000 checkpoint，999 条原始清单（short / medium / long 各 333 条），FA2、BF16、DeepStack 关闭。
所有 7,992 步完整 logits、prefill 和 decode 后全部层 KV 的 SHA256 均相同，生成 token 逐条相同。
两版各测 2,997 个完整请求，计时内 Graph 捕获、回退及 SDPA 调用均为 0。全部输入张量也与已有 base 的 999 条结果逐条相同。

| 指标 | 原 adapter fast-path | 优化后 adapter | 加速 |
|---|---:|---:|---:|
| Total，999 条累计 | {a['total_s']:.4f} s | {b['total_s']:.4f} s | {speed['total_s']:.4f}× |
| Prefill，999 条累计 | {a['prefill_s']:.4f} s | {b['prefill_s']:.4f} s | {speed['prefill_s']:.4f}× |
| Decode，999 条累计 | {a['decode_s']:.4f} s | {b['decode_s']:.4f} s | {speed['decode_s']:.4f}× |
| Decode，平均每 token | {a['decode_ms_per_token']:.4f} ms | {b['decode_ms_per_token']:.4f} ms | |
| Decode，每秒 token 数 | {a['decode_tokens_per_second']:.2f} | {b['decode_tokens_per_second']:.2f} | |
| Prefill KV，平均每条 | {a['kv_cache_mb']:.4f} MiB | {b['kv_cache_mb']:.4f} MiB | |
| Decoder prefill FLOPs，平均每条 | {a['flops']:.6e} | {b['flops']:.6e} | |
| Peak memory，全部请求最大 | {a['peak_memory_mb']/1024:.4f} GiB | {b['peak_memory_mb']/1024:.4f} GiB | |

平均每条 prefill：{a['prefill_s']*1000/999:.4f} → {b['prefill_s']*1000/999:.4f} ms；平均每条总耗时：{a['total_s']*1000/999:.4f} → {b['total_s']*1000/999:.4f} ms。
每项延迟分别取每条 3 次完整请求的中位数，然后求和；不同阶段的中位数相加可能与 Total 中位数略有差异。

结果文件：[summary.json](videomme999_final/summary.json)、[adapter_table.csv](videomme999_final/adapter_table.csv)、[validation.json](videomme999_final/validation.json)、[完整协议和代码哈希](videomme999_final/protocol.json)、[按视频长度分组](videomme999_final/duration_summary.json)。
逐条原始数据在同目录 `original_0`～`original_7`、`optimized_0`～`optimized_7` 的 JSONL 中。

## 问题与修改

1. **逐层 RMSNorm、RoPE 小算子开销。** 融合逐元素运算，保留 PyTorch 原来的 FP32 mean 归约顺序和所有中间 BF16 舍入；RoPE 禁用 FMA 合并。
2. **36 层视觉 adapter 分别执行。** Static adapter 的各层视觉分支彼此独立，合并为 batched BMM，每个请求重新计算全部视觉 memory。
3. **FA2 输入先拼接再 gather。** 从分离的视觉/文本 K/V 直接打包到同一 FA2 varlen 布局；保留原有顺序、因果关系和 FA2 内核。
4. **Graph 外重复特征整理与位置计算。** CPU 只读取一次文本/视觉拓扑，将当前输入的特征 gather 和 RoPE 准备放进 prefill Graph。
5. **Prefill 后再次整理所有层 KV。** 每层直接写入最终 native decode 布局，Graph 返回后只做一次独立存储复制；后续请求不会覆盖已有 KV。
6. **Decode 分别启动 Q/K/V、gate/up 投影。** 单 token 时合并矩阵乘法；各权重是同一存储上的独立视图，参数数值不变。

优化接在 `src/benchmark_prefill.py::build_qwen_fast_adapter_prefill` 的 `adapter_exact_optimizations=True` 分支。
项目原入口可用 `--adapter-exact-optimizations` 开启，配合 `--cuda-graph --cuda-graph-context --adapter-decode-cache --adapter-decode-cache-mode fast --last-logits-only`；该优化要求 BF16、FA2、batch size 1。比较时仍指定 `--comparison-deepstack off`。
原有路径仍可用 `--no-adapter-exact-optimizations` 做逐位对照，历史截图结果文件未改写。

## 测量边界与解释

- 每条视频沿用 `full_timestamp_v1`、8 帧、无字幕和 `media_first_v1` 提示词。每次重新执行视觉编码、adapter prefill 和 KV 构建。
- 固定生成 8 个 token：一次 prefill 和 7 次 KV 真正增长的单 token decode；两版都屏蔽 EOS。
- Total 包含视觉编码、位置准备、adapter prefill、decode、token 选择与同步；不含 CPU 抽帧/processor、输入搬运、预热、捕获和一致性哈希。
- Prefill 从请求开始计时，到首个 logits 与独立 KV 返回为止，包含位置准备和视觉编码。Decode 只计 7 次 forward；token 选择计入 Total。
- 各 GPU 均只有一个模型进程，按 index % 8 分片；偶数分片先原版，奇数分片先优化版。Total 是单请求延迟累计，非 8 GPU 作业墙钟时间。
- Peak memory 是预热后单请求 CUDA allocated 峰值，包含模型、adapter、Graph 缓冲区；两个 adapter 版本的 decode Graph 上限均为 12。
- FLOPs 使用同一解析 decoder prefill 口径（乘加算 2 FLOPs），包括视觉 K/V、文本与 adapter 投影，不含视觉编码器、LM head、归一化、softmax 和 FA2 打包造成的重复数据搬运。
- 本次优化保持计算内容和视觉 token 数不变，KV 大小和解析 FLOPs 应当相同；速度改善来自执行效率。
- 已有 base 结果没有重跑。它的 prefill 记录是首次模型 forward，生成侧的位置准备计入 overhead，且 decode Graph 上限为 8。因此这里用完全同边界的 adapter 前后结果计算 speedup，没有将两种 prefill 边界混为一个比值。

## 复跑

```bash
.venv/bin/python -m src.benchmark_adapter_optimizations \\
  --gpus 0 1 2 3 4 5 6 7 --runs 3 --tokens 8 \\
  --input-cache {protocol['input_cache']} \\
  --output test/results/adapter_speed_rerun
```

输出目录必须为新目录。模型、checkpoint、清单路径及 SHA256 固定在协议中；输入缓存仅保存 CPU processor 张量。
完整测试由 launcher 自动启动原版和优化版、逐条比对并汇总，无需再跑 base。
新增/更新的 3 项 GPU 单元测试覆盖 BF16 RoPE、视频因果 FA2 打包、KV 布局与 Graph 重放；均已通过。
'''
    (OUTPUT.parent / 'README.zh.md').write_text(report)
    print(json.dumps(dict(samples=999, input_match_with_base=True,
        variants=summary['variants'], speedup=speed), indent=2))


if __name__ == '__main__':
    main()
