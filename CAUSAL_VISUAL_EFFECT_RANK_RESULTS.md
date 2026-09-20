# 视觉读取效应的低秩干预：方法与完整结果

实验日期：2026-09-12。模型：Qwen3-VL-4B-Instruct、LLaVA-1.5-7B。评测使用 8 卡分片完成，无模型或 adapter 训练。

## 1. 实验要回答什么

**文本通过 attention 读取视觉信息所造成的输出变化，保留多少个 hidden 通道方向，就足以维持最终回答能力？**

本实验压缩的是“正常读取视觉”与“屏蔽视觉读取”之间的 attention 输出差值，不是 Q/K、完整 hidden，也不是视觉 token 数。它与 visual self-only / cross-only 消融不是同一个实验。

## 2. 干预的具体定义

### 2.1 从同一份当前输入计算两个输出

对选中的第 $\ell$ 层，以当前轨迹上的层输入 $H_\ell$ 为起点：

1. 按原生 causal mask 计算正常 attention，得到 $O_\ell^{\mathrm{normal}}$。
2. 使用相同输入、权重、Q/K/V 和位置编码，仅额外屏蔽“文本 query → 视觉 key”的 attention logits，再计算 $O_\ell^{\mathrm{blocked}}$。

设 $\mathcal T$ 为文本位置、$\mathcal V$ 为视觉位置，屏蔽规则为：

$$
S_{ij}^{\mathrm{blocked}}=
\begin{cases}
-\infty,&i\in\mathcal T,\ j\in\mathcal V,\\
S_{ij}^{\mathrm{normal}},&\text{其他原本可见的位置}.
\end{cases}
$$

原本不可见的未来位置仍由 causal mask 屏蔽。代码对浮点 mask 使用该 dtype 的最小有限值实现屏蔽。

**屏蔽发生在 softmax 前，剩余可见位置的权重会重新归一化。**因此不是在原始 $AV$ 中直接减去视觉 value 的加权和。

两种 $O$ 均为 **multi-head 输出合并并经过原生 $W_O$ 之后、加 residual 之前**的张量，最后一维为模型 hidden width $d$。

### 2.2 定义视觉读取效应

$$
\boxed{\Delta_\ell=O_\ell^{\mathrm{normal}}-O_\ell^{\mathrm{blocked}}}
$$

这里的差值包含两部分的综合作用：允许读取视觉 value，以及允许这些视觉 key 参与 softmax 竞争。因此不能将它直接命名为“纯视觉 $AV$ contribution”。

### 2.3 对差值做通道投影，再恢复输出

每层使用一个正交通道基底：

$$
B_{\ell,r}\in\mathbb R^{d\times r},\qquad B_{\ell,r}^{\top}B_{\ell,r}=I.
$$

将差值投影到前 $r$ 个方向：

$$
\Delta_{\ell,r}=(\Delta_\ell B_{\ell,r})B_{\ell,r}^{\top}.
$$

仅对文本 query 的输出行进行替换：

$$
\boxed{\widetilde O_\ell=O_\ell^{\mathrm{blocked}}+\Delta_{\ell,r}}.
$$

然后执行原生 residual、MLP 和后续层。视觉 query 行保留本次正常 attention 的输出，不在该 hook 中替换；不删除视觉 token，不直接修改视觉 residual。

| 设置 | 指定层文本位置的 attention 输出 | 含义 |
|---|---|---|
| Native baseline | $O^{\mathrm{normal}}$ | 原模型 |
| rank 0 | $O^{\mathrm{blocked}}$ | 该层完全阻断文本直接读取视觉 token |
| rank 32 / 64 / 128 | $O^{\mathrm{blocked}}+\Delta B_rB_r^\top$ | 恢复相应数量的通道方向 |
| Full-effect | $O^{\mathrm{blocked}}+\Delta$ | 完整恢复，与正常输出恒等 |

Qwen 的完整通道宽度为 **2560**，LLaVA 为 **4096**。rank 128 分别是完整宽度的 **5%** 和 **3.125%**。这不是 KV cache 大小或实际计算量的压缩比例。

“视觉行不替换”只描述当前 hook 的直接操作，不表示任意输入布局下后续视觉 hidden 都与未干预模型完全相同；后续层仍接收已经干预过的整个序列状态。

### 2.4 多层干预采用当前轨迹，不回填原模型缓存

例如“所有层 rank 32”：

```text
当前第 0 层输入 → normal / blocked → rank32 恢复 → residual、MLP
                                                              ↓
当前第 1 层输入 → normal / blocked → rank32 恢复 → residual、MLP
                                                              ↓
                         ……直到最后一层与答案输出
```

每层的两个分支都从该层的**当前输入**重新计算；这个输入可能已被前面的干预改变。不会拿预先缓存的原模型 decoder hidden 或原模型差值替代当前差值。

每个层组、每个 rank 都是独立的一次评测轨迹。例如“前 5 层”与“最后 10 层”不是在同一次运行中叠加的两个设置。

## 3. 共享基底如何构造

先对相应数据集的输入做原模型前向。每层额外计算 blocked 分支以收集 $\Delta$，但收集阶段不替换输出、不改变原模型轨迹。

汇总该层所有样本、所有文本位置的差值，累积未中心化二阶矩：

$$
C_\ell=\sum_{\text{样本、文本位置}}\Delta_{\ell,i}^{\top}\Delta_{\ell,i}.
$$

对 $C_\ell$ 做特征分解，按特征值降序选择对应方向。这等价于对汇总差值矩阵取右奇异向量。

- 每个模型、每个数据集、每一层分别构造基底。
- 同一层所有样本共享基底，不是每条样本各自求最优 SVD。
- rank 32、64、128 使用同一谱的嵌套前缀方向。
- 只收集输入 prompt，不包含标准答案，也不收集答案生成轨迹。
- 二阶矩与特征分解使用 FP64；差值投影使用 FP32；关闭 TF32。

**基底使用的就是这次评测集的输入。因此这是同集合、transductive 的 oracle 压缩诊断，不是独立校准集到未见图像的泛化评测。**虽然没有读取答案标签，也不能因此称为 held-out 基底实验。

## 4. 模型、数据和生成设置

| 项目 | Qwen3-VL-4B-Instruct | LLaVA-1.5-7B |
|---|---|---|
| 语言模型层数 | 36 | 32 |
| attention 输出完整宽度 | 2560 | 4096 |
| 模型权重 | 冻结原模型，无 adapter | 冻结原模型，无 adapter |
| 视觉处理 | 原生视觉路径，保留 DeepStack | 原生视觉路径，576 个视觉位置 |

数据与评测约定：

- RealWorldQA：全部 **765** 条。
- MMStar：固定前 **1000** 条。
- 两个模型使用相同样本及顺序，但各自使用原生 processor、提示模板和图像预处理。
- BF16、SDPA、确定性 greedy 生成；最多生成 8 个新 token，识别出 A/B/C/D 或遇到 EOS 即停止。
- 每个生成步骤都对当前完整序列执行 fresh forward，`use_cache=False`。相同干预持续作用于原始文本位置和已生成的文本位置，不只是首次 prefill。
- 同一图像的视觉编码器结果可缓存；不同干预条件之间不复用 decoder hidden。
- 先完成四组原模型 baseline，再进行压缩干预评测。
- 评测按 8 个 GPU worker 分片；基底特征分解是单独阶段，不应将整套流程描述为所有阶段都使用 8 卡。

所有层号均为 **0-based**：

| 干预范围 | Qwen | LLaVA |
|---|---|---|
| 前 5 层 | 0–4 | 0–4 |
| 前 10 层 | 0–9 | 0–9 |
| 中间 10 层 | 13–22 | 11–20 |
| 最后 10 层 | 26–35 | 22–31 |
| 所有层 | 0–35 | 0–31 |

每个模型 × 数据集包含 5 个层组 × 4 个 rank，加 native 和 full-effect，共 **22 个条件**。

## 5. 完整准确率结果

下表均为 accuracy（%）。baseline 来自本次相同评测协议，不混用历史其他 attention backend 或生成设置的分数。

### 5.1 原模型与完整恢复自检

| 模型 | 数据集 | 样本数 | 原模型正确数 | 原模型 | Full-effect |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-4B | RealWorldQA | 765 | 547 | 71.50 | 71.50 |
| Qwen3-VL-4B | MMStar | 1000 | 648 | 64.80 | 64.80 |
| LLaVA-1.5-7B | RealWorldQA | 765 | 425 | 55.56 | 55.56 |
| LLaVA-1.5-7B | MMStar | 1000 | 369 | 36.90 | 36.90 |

### 5.2 Qwen3-VL-4B · RealWorldQA

| 干预范围 | 层号 | rank 0 | rank 32 | rank 64 | rank 128 |
|---|---|---:|---:|---:|---:|
| 前 5 层 | 0–4 | 72.55 | 71.90 | 71.76 | 71.50 |
| 前 10 层 | 0–9 | 72.94 | 71.90 | 72.03 | 72.03 |
| 中间 10 层 | 13–22 | 45.88 | 69.93 | 69.54 | 70.46 |
| 最后 10 层 | 26–35 | 71.11 | 70.98 | 71.11 | 71.50 |
| 所有层 | 0–35 | 46.14 | 68.37 | 69.15 | 70.72 |

### 5.3 Qwen3-VL-4B · MMStar

| 干预范围 | 层号 | rank 0 | rank 32 | rank 64 | rank 128 |
|---|---|---:|---:|---:|---:|
| 前 5 层 | 0–4 | 64.80 | 65.00 | 65.40 | 65.20 |
| 前 10 层 | 0–9 | 62.80 | 64.70 | 64.90 | 65.10 |
| 中间 10 层 | 13–22 | 40.50 | 61.00 | 63.30 | 64.40 |
| 最后 10 层 | 26–35 | 65.20 | 65.10 | 64.80 | 64.90 |
| 所有层 | 0–35 | 25.60 | 59.50 | 63.00 | 64.60 |

### 5.4 LLaVA-1.5-7B · RealWorldQA

| 干预范围 | 层号 | rank 0 | rank 32 | rank 64 | rank 128 |
|---|---|---:|---:|---:|---:|
| 前 5 层 | 0–4 | 53.20 | 55.82 | 56.21 | 55.95 |
| 前 10 层 | 0–9 | 49.41 | 54.90 | 55.16 | 55.82 |
| 中间 10 层 | 11–20 | 48.10 | 54.25 | 55.56 | 55.16 |
| 最后 10 层 | 22–31 | 55.95 | 55.03 | 55.16 | 55.16 |
| 所有层 | 0–31 | 42.35 | 53.20 | 54.90 | 54.77 |

### 5.5 LLaVA-1.5-7B · MMStar

| 干预范围 | 层号 | rank 0 | rank 32 | rank 64 | rank 128 |
|---|---|---:|---:|---:|---:|
| 前 5 层 | 0–4 | 37.30 | 35.80 | 36.40 | 36.70 |
| 前 10 层 | 0–9 | 31.00 | 34.80 | 34.90 | 35.60 |
| 中间 10 层 | 11–20 | 36.30 | 36.90 | 37.40 | 37.00 |
| 最后 10 层 | 22–31 | 36.40 | 36.80 | 36.80 | 37.00 |
| 所有层 | 0–31 | 24.90 | 34.00 | 34.90 | 35.80 |

### 5.6 所有层同时压缩的汇总

| 模型 | 数据集 | baseline | 所有层 rank 0 | 所有层 rank 128 | rank 128 相对 baseline 下降（百分点） |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-4B | RealWorldQA | 71.50 | 46.14 | 70.72 | 0.78 |
| Qwen3-VL-4B | MMStar | 64.80 | 25.60 | 64.60 | 0.20 |
| LLaVA-1.5-7B | RealWorldQA | 55.56 | 42.35 | 54.77 | 0.78 |
| LLaVA-1.5-7B | MMStar | 36.90 | 24.90 | 35.80 | 1.10 |

百分点差值按未四舍五入的正确数比例计算；因此个别行与已保留两位小数的分数直接相减相差 0.01。

## 6. 结果能说明什么，不能说明什么

### 6.1 当前观察

1. **完全阻断所有层的视觉读取会显著掉分，而恢复 rank 128 的差值方向能接近原模型。**这支持本次协议下，维持任务表现所需的视觉读取效应具有明显的通道可压缩性。
2. 层组敏感性不同。Qwen 对中间 10 层的 rank 0 干预尤其敏感；LLaVA 的层组分布不同，不能把 Qwen 的具体位置直接推广到所有模型。
3. 某些局部干预分数略高于 baseline，且准确率不必随 rank 单调上升。本次未报告配对显著性检验，不能把小幅上升直接认定为稳定增益。

### 6.2 必须保留的解释边界

- **不是 attention sink 因果机制的直接验证。**本实验没有按 sink mass 分组，也没有单独操纵 sink；不能仅据此断言可压缩性由 sink 导致。
- **不是 visual-to-visual mixing 消融。**干预的是文本 query 读取视觉 key 的路径。
- **不是“视觉信息没有用”。**所有层 rank 0 明显掉分；局部层组 rank 0 不掉分，也可能因为其他层或既有文本状态已经保留视觉信息。
- **不是 adapter 可预测性的证明。**评测时仍现场计算完整 normal 和 blocked 输出，得到真实 $\Delta$ 后再投影，没有训练从初始 embedding 预测 $\Delta$ 的模型。
- **不是推理加速实现。**每个干预层额外重算 blocked attention；通道压缩比例不能当作 FLOPs 或延迟收益。
- **不是独立测试集泛化证明。**共享基底由同一批评测输入统计，尽管不使用答案，仍属于 transductive 诊断。
- **不等于测出了唯一的“真实有效秩”。**这里报告固定 rank 下的功能性准确率，不是证明 rank 128 是最小充分维度，也不代表其他方向完全无信息。

最稳妥的总结是：**在两个模型、两个数据集的同集合共享基底实验中，将各层文本视觉读取效应限制到 128 个输出通道方向，能够在所有层同时干预时保留接近原模型的准确率。**

## 7. 完成与一致性核查

- 四组任务全部完成；每组 22 个条件。
- 已核查每组样本编号完整、无重复、无缺失，且每条样本都有 22 个结果。
- 干预评测中重复计算的 native 输出，与先前保存的 baseline 生成文本逐条一致。
- Full-effect 的生成文本与 native 逐条一致；不是仅总体分数相同。
- 本次共核查 3530 个模型—样本组合，完整恢复文本不一致数为 0。

## 8. 代码与原始结果

- [实验入口与两模型适配](src/causal_effect_benchmark_suite.py)
- [干预 hook 与共享基底构造](src/realworldqa_causal_effect_rank.py)
- [原始汇总 README](artifacts/diagnostics/causal_effect_2models_2bench_20260912/README.md)
- [实验计划、数据与代码记录](artifacts/diagnostics/causal_effect_2models_2bench_20260912/plan.json)
- [完成状态](artifacts/diagnostics/causal_effect_2models_2bench_20260912/status.json)
- [Qwen / RealWorldQA 结果](artifacts/diagnostics/causal_effect_2models_2bench_20260912/qwen_realworldqa/results.json)
- [Qwen / MMStar 结果](artifacts/diagnostics/causal_effect_2models_2bench_20260912/qwen_mmstar/results.json)
- [LLaVA / RealWorldQA 结果](artifacts/diagnostics/causal_effect_2models_2bench_20260912/llava_realworldqa/results.json)
- [LLaVA / MMStar 结果](artifacts/diagnostics/causal_effect_2models_2bench_20260912/llava_mmstar/results.json)

各结果子目录另外保留 `baseline_shard*.jsonl`、`eval_shard*.jsonl`、`basis.pt`、`basis_energy.json` 和执行日志，可追溯逐样本答案及基底。
