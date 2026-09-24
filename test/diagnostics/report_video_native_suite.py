"""Report native FA2 base, exact FA2 metadata pruning fixes and saved adapter."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'test/results/video_base_native_fa2_20260915/videomme999'
PRUNING = ROOT/'test/results/video_pruning_fa2_metadata_20260915/videomme999'
ADAPTER = ROOT/'test/results/adapter_max_20260915/videomme999_final'
NAMES = dict(base='Base · native FA2',fastv='FastV',dart='DART',visionzip='VisionZip',
             divprune='DivPrune',zoo='Zoo-Prune',sparsevlm='SparseVLM',adapter='Adapter · max')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_rows(directory, pattern):
    rows = [json.loads(line) for file in directory.glob(pattern) for line in file.read_text().splitlines()]
    indexed = {r['index']:r for r in rows}
    assert len(rows)==len(indexed)==999 and set(indexed)==set(range(999))
    return indexed


def main():
    base, summaries, adapter_summary = read(BASE/'native_summary.json'),read(PRUNING/'summary.json'),read(ADAPTER/'summary.json')
    assert base['samples']==999 and base['variant']=='native'
    assert len(summaries)==12 and all(s['samples']==999 for s in summaries)
    assert all(read(p/'source_validation.json')['all_sources_unchanged'] for p in [BASE,PRUNING,ADAPTER])
    base_rows = load_rows(BASE,'native_*.jsonl')
    adapter_rows = load_rows(ADAPTER,'optimized_*.jsonl')
    assert all(base_rows[i]['input_sha256']==adapter_rows[i]['input_sha256'] for i in base_rows)
    assert all(not r['cuda_graphs_enabled'] and r['saved_input_and_tokens_match'] for r in base_rows.values())
    groups, validations, fastv_before_after = {}, {}, []
    for s in summaries:
        directory=PRUNING/f"{s['method']}_r{round(s['retention']*100):03d}"
        data=load_rows(directory,'fa2_metadata_*.jsonl')
        assert all(r['input_sha256']==base_rows[i]['input_sha256'] for i,r in data.items())
        assert all(r['native_decode_logits_and_final_kv_bitwise_equal'] and not r['decoder_cuda_graphs_enabled']
                   and not r['vision_cuda_graphs_enabled'] and r['sdpa_calls']==0 for r in data.values())
        assert all(r['selector_graph_stats'].get('timed_captures',0)==r['selector_graph_stats'].get('timed_fallbacks',0)==0 for r in data.values())
        assert all(r['execution']=='fa2_metadata' and r['original_native_logits_prefix_and_final_kv_bitwise_equal']
                   and r['original_native_selected_positions_equal'] and r['exact_rmsnorm_enabled'] for r in data.values())
        assert all(len(r['trials'])==3 and all(t['decode_steps']==7 for t in r['trials']) for r in data.values())
        for r in data.values():
            assert all(t['request_prefill_time_s']>=t['generation_prefill_time_s'] for t in r['trials'])
            assert all(a['text_tokens']==r['text_tokens'] for a in r['pruning_audit'])
            assert r['pruning_audit'][-1]['after_visual']==r['retained_visual_tokens']
        group={}
        for duration in ['short','medium','long']:
            selected=[r for r in data.values() if r['duration']==duration]
            assert len(selected)==333
            group[duration]={k:sum(statistics.median(t[k] for t in r['trials']) for r in selected)
                             for k in ['total_time_s','request_prefill_time_s','decode_time_s']}
        groups[directory.name]=group
        validations[directory.name]=dict(samples=999,all_inputs_identical_to_native_base=True,
            visual_retained_counts=dict(sorted(Counter(r['retained_visual_tokens'] for r in data.values()).items())),
            all_text_and_timestamps_preserved=True,all_native_decode_logits_and_final_kv_bitwise_equal=True,
            all_original_native_logits_prefix_and_final_kv_bitwise_equal=True,
            all_original_native_selected_positions_equal=True,
            exact_native_order_rmsnorm_enabled=True,
            selector_graphs_enabled=next(iter(data.values()))['selector_cuda_graphs_enabled'],
            timed_selector_captures=0,timed_selector_fallbacks=0,
            original_fa2_prefill_metadata_builds=sorted({r['original_fa2_sequence_metadata_builds']['prefill'] for r in data.values()}),
            fa2_prefill_metadata_builds=sorted({r['fa2_sequence_metadata_builds']['prefill'] for r in data.values()}),
            fa2_decode_metadata_builds=sorted({r['fa2_sequence_metadata_builds']['decode'] for r in data.values()}))
        if s['method']=='fastv':
            old_path=ROOT/'test/results/video_pruning_native_fa2_20260915/videomme999'/directory.name
            old=load_rows(old_path,'native_*.jsonl')
            for i,r in data.items():
                for key in ['input_sha256','tokens','logits_sha256','prefill_kv_sha256','final_kv_sha256','pruning_audit','flops']:
                    assert r[key]==old[i][key], (i,s['retention'],key)
            before={k:sum(statistics.median(t[k] for t in r['trials']) for r in old.values())
                    for k in ['total_time_s','request_prefill_time_s','decode_time_s']}
            fastv_before_after.append(dict(retention=s['retention'],samples=999,
                all_saved_native_logits_and_kv_identical=True,before=before,
                after={k:s[k] for k in before},source=str(old_path)))
    adapter=adapter_summary['variants']['optimized']
    paired=read(PRUNING.parent/'paired_retention/fastv.json')
    assert paired['indices']==[0,333,666] and paired['runs']==12
    assert len(paired['trials'])==36 and all(t['exact_vs_full_results'] for t in paired['trials'])
    paired_prefill=paired['summaries']['request_prefill_time_s']
    paired_total=paired['summaries']['total_time_s']
    paired_decode=paired['summaries']['decode_time_s']
    table=[]
    def add(method,retention,total,prefill,decode,kv,flops,peak,source):
        table.append(dict(method=method,retention=retention,samples=999,total_time_s=total,
            prefill_time_s=prefill,prefill_ms_per_request=1000*prefill/999,
            decode_ms_per_token=decode,decode_tokens_per_s=1000/decode,
            kv_cache_mib=kv,decoder_prefill_flops=flops,peak_memory_gib=peak/1024,
            total_speedup=base['total_time_s']/total,prefill_speedup=base['prefill_time_s']/prefill,
            decode_speedup=base['decode_ms_per_token']/decode,source=str(source)))
    add('base',1.,base['total_time_s'],base['prefill_time_s'],base['decode_ms_per_token'],
        base['kv_cache_mb'],base['flops'],base['peak_memory_mb'],BASE/'native_summary.json')
    for retention in [.05,.2]:
        for method in ['fastv','dart','visionzip','divprune','zoo','sparsevlm']:
            s=next(s for s in summaries if s['method']==method and s['retention']==retention)
            add(method,retention,s['total_time_s'],s['prefill_time_s'],s['decode_ms_per_token'],
                s['kv_cache_mb'],s['flops'],s['peak_memory_mb'],PRUNING/f'{method}_r{round(retention*100):03d}/summary.json')
    add('adapter',None,adapter['total_s'],adapter['prefill_s'],adapter['decode_ms_per_token'],
        adapter['kv_cache_mb'],adapter['flops'],adapter['peak_memory_mb'],ADAPTER/'summary.json')
    write(PRUNING/'comparison.json',table)
    with (PRUNING/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    write(PRUNING/'group_summary.json',groups)
    write(PRUNING/'fastv_before_after.json',fastv_before_after)
    write(PRUNING/'validation.json',dict(method_configurations=12,samples_per_configuration=999,
        all_base_baseline_adapter_input_tensors_identical=True,
        baseline_timed_requests=12*999*3,baseline_decode_logits_checked=12*999*7,
        original_native_logits_compared=12*999*8,
        base_timed_requests=999*3,baseline_details=validations))
    table_lines=[]
    for r in table:
        retention='—' if r['retention'] is None else f"{r['retention']:.0%}"
        table_lines.append(f"| {NAMES[r['method']]} | {retention} | {r['total_time_s']:.3f} | {r['prefill_time_s']:.3f} | "
            f"{r['decode_ms_per_token']:.3f} | {r['kv_cache_mib']:.2f} | {r['peak_memory_gib']:.2f} | "
            f"{r['decoder_prefill_flops']/1e12:.4f} | {r['total_speedup']:.3f}× | {r['prefill_speedup']:.3f}× |")
    report='\n'.join([
        '# Video-MME 999：原生 FA2 base 与修正执行开销后的六种 baseline','',
        '按本轮要求，base 采用 Hugging Face 原生 FA2 `generate`。六个 baseline 使用项目现有 Qwen3-VL 移植版的 HF `generate`，统一复用 FA2 序列信息并使用精确 RMSNorm 内核；DART/DivPrune/Zoo-Prune 的选点张量运算采用 CUDA Graph 重放。视觉编码和 decoder 不加 Graph、compile 或融合 RoPE/投影。剪枝和 attention 算法不变。Adapter 行保留上一轮已经验证的 max 优化结果。','',
        'Qwen3-VL-4B-Instruct、BF16、FA2、DeepStack 关闭；同一份 Video-MME 999 条，短/中/长各 333 条，8 帧、完整时间戳、无字幕。输入张量哈希逐条一致。','',
        '## 完整指标','',
        '**Total/Prefill 是 999 条请求延迟累计，不是 8 GPU 作业墙钟时间。** 每条测 3 次，各指标取中位数再求和；每条固定生成 8 token，即 prefill 加 7 次 KV 真正增长的 decode。','',
        'KV 和 FLOPs 为单请求样本均值；Peak memory 取全部计时请求的最大值。','',
        '| 方法 | 视觉保留率 | Total (s) | Prefill (s) | Decode (ms/token) | KV (MiB) | Peak (GiB) | Decoder prefill FLOPs (T) | Total speedup | Prefill speedup |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*table_lines,'',
        '全部 speedup 均以本轮原生 FA2 base 为分母；CSV 还包含每条 prefill 的毫秒值、decode token/s 和 decode speedup。Adapter 保存全部视觉 KV，表中 5%/20% 是剪枝 baseline 的 token 保留率，不用于描述 adapter。','',
        '结果：[comparison.csv](videomme999/comparison.csv)、[comparison.json](videomme999/comparison.json)、[baseline 汇总](videomme999/summary.json)、[按视频长度分组](videomme999/group_summary.json)。','',
        '## 计时边界','',
        '- Prefill 从请求开始，到首个 logits 和独立 KV 就绪，包含生成初始化、位置准备、视觉编码、选点/合并和语言 prefill。旧版仅测首次 forward 的字段另存为 `generation_prefill_time_s`，不再用于完整 prefill speedup。',
        '- Decode 计 7 次单 token 模型 forward，Total 另含 token 选择和生成循环的开销。各阶段独立取中位数，阶段之和可能与 Total 中位数有小差异。',
        '- CPU 抽帧、processor、输入搬到 GPU、预热、Graph 捕获和一致性检查在计时外。每次请求都重新执行视觉编码与剪枝；CPU 缓存只保存输入张量。这是预热后的请求延迟，不是首次请求延迟。',
        '- Peak memory 是预热后单请求 CUDA allocated 峰值的最大值，包含模型权重和各自运行缓冲区。Adapter 的 Graph 缓冲区也计入，不是跨进程 nvidia-smi 显存。',
        '- KV 是 prefill 后实际 K/V 存储，按独立 storage 去重后取样本均值。FLOPs 是解析 decoder prefill 口径，乘加计 2 FLOPs，baseline 使用每层实际 KV 长度，adapter 包含视觉 K/V 和 adapter 投影；不包含视觉编码器、选点/合并、LM head、Norm 或 softmax，但这些操作的时间已计入。',
        '- 每个 GPU 一次只运行一个模型，按 index % 8 分片。各方法和两档保留率分别运行；base 已先单独完成。',
        '- Zoo-Prune 每条使用 seed=42+index，每次请求在计时前恢复同一 RNG 状态，重新计算相同的随机敏感度选点；不复用选点结果。','',
        '## 实现与正确性','',
        '- 这是项目内的 Qwen3-VL 移植实现测量，不能称为这些方法官方上游在 Video-MME 的结果。未改变本轮六种方法的选择或合并算法。',
        '- FastV 在第 2 层（从 0 开始）剪枝，使用前一层最后 query 对全体 key 归一化后的视觉分数；DART 同样在第 2 层，使用前层 post-RoPE keys。',
        '- SparseVLM 保留现有 [2,6,15] 配置和绝对目标预算：本协议在第 2 层达到目标，之后不重复删减。上述三种方法前两层保存完整 KV，后 34 层压缩。',
        '- VisionZip、DivPrune、Zoo-Prune 在语言第 0 层之前/入口剪枝，36 层均使用压缩 KV。VisionZip dominant/contextual 比例为 80:20，Zoo-Prune 保留 num_refine=64、noise_scale=0.01。',
        '- 所有方法使用整条请求的实际视觉位置集合；逐条检查选中数量、文本和时间戳完整保留，避免把视频交错的时间戳误当视觉 token。',
        '- 每条检查 36 层 prefill KV 长度以及 7 次真实增长。HF generate 的全部 7 步 decode logits 与独立逐层计算逐位一致，最终全部 K/V SHA256 一致；独立计算使用原始三轴 M-RoPE 位置，不由压缩 KV 长度推断位置。',
        '- 每条先运行未修正的原生移植实现，再启用序列信息复用。修正前后 8 份完整词表 logits、prefill 及最终全部 K/V 的 SHA256 和选点记录逐条相同。',
        '- 六种方法的 12 个配置全部 999 条均通过；总计 95,904 份修正前后 logits 对照、83,916 份独立 decode logits 检查、35,964 个计时请求。运行期间禁止调用 SDPA。',
        '- Base 的 999 条输入和生成 token 匹配既有记录；adapter 的输入同样逐条匹配。Adapter 本轮没有重跑，沿用此前 999 条 max 结果。','',
        '## 如何解读速度','',
        '同一份 FastV 999 条修正前后结果保存在 [fastv_before_after.json](videomme999/fastv_before_after.json)，逐条与之前保存的输入、全部 logits、prefill/最终 KV 和剪枝记录交叉匹配。','',
        f"FastV 保留率额外复核：在同一张 GPU 上，对索引 0/333/666 各交替测量 12 轮，每轮包括 5% 和 20% 两个请求，合计 72 次计时。5%/20% 的每条 prefill 分别为 {paired_prefill['ret05_mean_sample_median_ms']:.3f}/{paired_prefill['ret20_mean_sample_median_ms']:.3f} ms，total 为 {paired_total['ret05_mean_sample_median_ms']:.3f}/{paired_total['ret20_mean_sample_median_ms']:.3f} ms，decode 为 {paired_decode['ret05_mean_sample_median_ms']/7:.3f}/{paired_decode['ret20_mean_sample_median_ms']/7:.3f} ms/token。两档差异较小，prefill 没有呈严格单调关系。此小测仅诊断保留率排序，完整表仍采用各自 999 条结果。生成 token、全部 logits 和最终 KV 均匹配完整结果。[交替测量原始记录](paired_retention/fastv.json)。",'',
        '- 压缩 KV 和 decoder FLOPs 不保证原生 Python 执行的总耗时同比下降；选点、合并、张量整理和 FA2 序列信息准备都包含在请求中。',
        '- 修正前：剪枝保留原始位置后，text position ids 存在间隔，当前 Transformers FA2 会逐层走 varlen 序列信息准备。原生验证请求记录 `original_fa2_sequence_metadata_builds`，FastV/DART/SparseVLM 为 34 次，VisionZip/DivPrune/Zoo-Prune 为 36 次。',
        '- 修正后：对确认单条且位置递增的输入，每组位置只准备一次序列长度信息，仍用同一个 FA2 varlen kernel；真正的位置重置或 padding 保留原生处理。每次 forward 都清空 metadata，绝不跨请求复用剪枝或视觉结果。Decode 的单 token 不需要反复判断序列是否打包。',
        '- 诊断函数 `prepare_fa_kwargs_from_position_ids` 修正后调用次数为 0，因为等价信息由模型 hook 按位置组构建并传入；这不表示没有准备信息。调用计数只在未计时的校验阶段开启。',
        '- DivPrune/Zoo-Prune 的贪心循环每选一个 token 都重复启动多个小 GPU 算子，DART 的邻居搜索也有连续的小张量操作。复用既有 `NativeDecoderGraphs`，仅捕获这些选择函数的原始张量运算；保留 PyTorch topk/argmax、候选顺序、DART 的 Python 集合处理及 Zoo-Prune 的 64 次随机敏感度估计。每次重放都先复制当前输入，返回结果独立克隆。',
        '- Selector Graph 最多保留 12 个形状；捕获仅在每条的未计时校验/预热阶段允许，计时中的捕获和回退次数均为 0。全部输出与无 Graph 原生请求逐条验证相同。',
        '- RMSNorm 复用 adapter 已验证的 `native_order_rmsnorm`，保持当前 PyTorch FP32 mean 的累加顺序和 BF16 中间舍入，以单内核减少张量读写和启动次数。六个 baseline 同样启用；不改变权重、精度或网络结构。',
        '- Zoo-Prune 的单独逐阶段诊断显示，20% 保留率下敏感度估计约 6.88 → 3.61 ms，完整 prefill 约 54.62 → 47.16 ms/条（3 条 pilot，metadata/selector 已启用后比较 RMSNorm）。这些诊断数不作为完整 999 条表格数据。证据：[zoo_stage_diagnosis.json](zoo_stage_diagnosis.json)。',
        '- Decode 每步仍执行全部 36 层及相同权重投影；减少 KV 只减少其中一部分工作。原生调用开销也占用时间，因此不能预设 5% 或 20% 的速度提升幅度。',
        '- Adapter 行包含之前的 CUDA Graph、精确融合和连续 KV 优化；baseline 使用精确 RMSNorm、序列信息复用和上述 selector Graph；base 按要求保持原生。这里反映各实现速度，加速比同时包含算法与运行实现收益，不能单独归因于压缩算法。','',
        '## 证据与复跑','',
        '[完整协议和代码哈希](videomme999/protocol.json)、[跨方法输入与 KV/位置验证](videomme999/validation.json)、[源文件未变更校验](videomme999/source_validation.json)。每个方法目录保存全部 999 条审计、token、logits/KV 哈希和三次原始计时。','',
        'Base：`src/benchmarking/engines/base.py --variant native`，本轮 8 GPU 启动器为 `test/diagnostics/run_native_video_base.py`。原生基准位于 `test/results/video_base_native_fa2_20260915/videomme999`。','',
        '```bash',
        '.venv/bin/python -m src.benchmarking.engines.pruning \\',
        '  --gpus 0 1 2 3 4 5 6 7 \\',
        '  --methods fastv dart visionzip divprune zoo sparsevlm \\',
        '  --execution fa2_metadata --selector-graphs --exact-norms \\',
        '  --retentions 0.05 0.2 --runs 3 --tokens 8 \\',
        '  --output test/results/video_pruning_metadata_rerun',
        '```','',
        '输出使用新目录；回到修正前的执行时，用 `--execution native` 并移除 `--selector-graphs --exact-norms`。未优化 FastV 的完整 999 条结果保存在 `test/results/video_pruning_native_fa2_20260915/videomme999`，该轮在发现开销后停止，其余方法未完成。历史截图、旧的优化 base 和 adapter 结果保留。',''])
    (PRUNING.parent/'README.zh.md').write_text(report)
    print(json.dumps(table,indent=2))


if __name__=='__main__':
    main()
