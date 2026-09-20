# Baseline 与固定 Q 评测核查（2026-09-13）

## 完成核验

两组已按顺序完成。Baseline 共 52 个配置，每个配置 1000 条，合计 52000 条记录；独立复核所有记录的 DeepStack 禁用标记与实际视觉预算。固定 Q 共 1500 个独立样本、3 层、2 种 query 汇总，共 9000 条层级记录，每组索引严格覆盖 0–1499。原生 AV 重建最大相对误差为 0.0020171917（约 0.20%）。监听器确认两组 complete 后正常退出；最终这轮未触发异常自动重启。

固定 Q 全 text-query 结果：readout cosine 为 0.238974 / 0.178308 / 0.206997（层 33/34/35），relative error 为 1.266330 / 1.262133 / 1.254394。这组结果不支持“固定 native Q 下视觉 readout 高度相似”的假设；不能据此单独否定使用自身 query 的完整 adapter 模型的任务能力。

## 当前任务与结果目录

- Baseline：`artifacts/diagnostics/multimodal_baselines_nodeepstack_20260913/`。
- 固定 Q：`artifacts/diagnostics/fixed_q_readout_mmstar1500_20260912/`，仅在 baseline 完整通过后启动。
- 监听：baseline 目录下 `monitor/status.json`；异常诊断为 `monitor/latest_investigation.json`。
- 结果是否完整以每个配置的样本索引核验及最终 `status.json` 为准，不以 GPU 空闲或进程消失作为完成依据。

## 已定位并修复的问题

| 问题 | 修复与验证 |
|---|---|
| `1-(1-r)` 与 `r` 在半整数取整处得到不同 token 数 | 六个方法与评测断言统一使用十进制比例预算；最近整数、半整数取偶。1250 tokens、5% 一致得到 62；比例等价式覆盖 1–19999 个 token 的测试。 |
| 多图、视频的视觉位置不连续 | 使用真实视觉位置集合，只裁剪视觉位置，保留所有文本、分隔符、时间戳与原始 M-RoPE 坐标。 |
| Base 开 DeepStack、裁剪方法关 DeepStack | 按用户要求，这组 baseline 全部关闭 DeepStack。每次 forward 检查禁用入口执行；任何 DeepStack 注入都会报错。 |
| FastV 在视觉范围内单独 softmax，改变各 head 的相对权重 | 恢复完整 key 分母，再平均 head 并提取视觉分数。添加能区分两种排序的回归测试。 |
| FastV 使用裁剪层自己的分数 | 对照本地 upstream，改取前一层最后 query 的 attention；零起始 layer 2 裁剪，打分源为 layer 1。 |
| DART 使用当前层未经过 RoPE 的 K | 对照本地 Qwen2.5-VL upstream，改用前一层 post-RoPE K；diversity 仍使用前一层输出 hidden。 |
| VisionZip 多图/帧 attention 统计混在一起 | 统计按原生 ViT `cu_seqlens` 分段，避免跨图像/帧 softmax；上下文合并使用匹配的数据类型。 |
| 一个失败分片取消全部后续任务 | 分片重试一次；其他任务继续。全部有效样本核验成功后才能启动固定 Q。恢复时跳过已完成且版本有效的分片。 |

修复前 FastV/DART 的本轮输出保存在结果目录内 `archive_fastv_invalid/`、`archive_dart_invalid/`，未删除，不进入新表。更早开启 DeepStack 的 base 结果仍保留在旧目录，不混入本轮。

## 必须保留的移植边界

本实验评估的是仓库的 **Qwen3-VL 移植版本**，不是上游官方多图/视频成绩。

- 每条请求的全部视觉位置共享 20% 或 5% 的预算，不是逐图/逐帧保底预算。
- SparseVLM 当前端口采用固定最终预算，在 layer 2 达到目标后 layer 6/15 不再继续裁剪；它不是上游为 576 tokens 制定的逐层预算曲线，也没有声称完全复现上游 progressive merging。
- DART 使用精确 token 预算限制 pivot/diversity 选择，不能把它的预算取整当成上游近似配额的逐项复现。
- VisionZip 在各原生图像/帧内统计 saliency，但 contextual merging 使用请求内的视觉集合。
- 使用逐步无 cache 贪心前向，最多 8 tokens，选项/EOS 提前结束；每一步重新选择视觉位置。避免现有端口压缩 KV cache 后的生成位置歧义，不报告与官方 cached decode 等价的延迟。
- 每个 benchmark 前最多 1000 条；视频采用完整时间窗口内 8 帧、真实时间戳、无字幕，每帧最多 262144 pixels。所有方法使用同一输入协议。

## 固定 Q 实验

MMStar 全部 1500 条，完整 embedding adapter checkpoint，零起始 layer 33/34/35 的 **attention 输入**。这个实验仍使用原生 teacher（与此前 task-subspace 分析一致）；baseline 的 no-DeepStack 开关只作用在 baseline 进程中。

原生同层 text Q 固定；非视觉 K/V 固定。视觉 hidden 经相同原生 RMSNorm、K/V projection、K head norm、M-RoPE；按原生 GQA 对齐。完整因果序列参与 softmax，只在读出贡献中提取视觉部分，不对视觉 attention 重新归一化。

输出 hidden/K/V/QK similarity、全局 attention KL、visual attention cosine/top-10 overlap、视觉 readout 的 cosine/R²/relative error，并分别报告全部有视觉可见 key 的非视觉 prompt query 和最后答案边界 query。W_O 前后的视觉贡献都保留；投影后贡献不含 W_O bias。

验证：

1. 恒等视觉替换必须得到完全相同的 scores/readout 和零 KL。
2. 带 RoPE、GQA、交错视觉/文本位置的独立 SDPA 参考测试。
3. 只在可见视觉 logits 上选 top-k，避免概率下溢后的零值并列或未来位置混入。
4. 每个真实样本、每层重建全量原生 AV，与原生模型 W_O 输入比对。FP32 离线计算对 fused BF16 的 relative error 超过 0.02 时中止发布并保留错误。
5. 检查 checkpoint、参考 adapter 源码和数据文件 hash；拒绝 padded/batched 输入，防止用简化因果 mask 覆盖错误场景。

这是固定 native Q 下的读出诊断，不等于证明 student 实际产生的 Q 或端到端输出相同，也不预设实验必须支持功能等价。

## 监听与恢复边界

每 15 秒检查进程、GPU 利用率和结果文件进度。GPU 持续空闲 120 秒或报告失败会触发日志检查；仅当 GPU 持续空闲且结果 900 秒不增长时才按卡住处理。只允许终止/恢复这两个指定输出目录对应的评测进程，不操作其他训练。

已知退出/卡住可自动恢复，未知确定性错误最多恢复三次，然后保持监听并标记 `needs_code_repair`，由当前执行中的检查继续定位；监听脚本本身不能生成代码修复。不能将“挂了监听”描述为所有潜在代码错误都已解决。
