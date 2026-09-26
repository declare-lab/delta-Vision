# Vision KV Inject / δ-Vision

视觉 embedding adapter 的训练、评测、效率测量与论文分析。项目说明统一维护在本文件。

## 目录

| 位置 | 用途 |
|---|---|
| `src/model.py`、`src/model_setup.py` | Qwen3-VL／LLaVA adapter、模型构造和 checkpoint 加载 |
| `src/qwen35.py` | Qwen3.5 adapter 与 Full Attention／Gated DeltaNet 接入 |
| `src/vllm_adapter.py` | Qwen3-VL / Qwen3.5 embedding adapter 的 vLLM 导出与模型注册（正确性首版） |
| `src/training/` | 训练执行；Qwen3.5 当前训练和评测共用 worker 也在这里 |
| `src/evaluate.py` | 普通 benchmark 生成、逐题判分和汇总 |
| `src/benchmarking/` | Video-MME 计时、FLOPs、Peak Memory 和结果汇总 |
| `src/data.py`、`src/benchmarks.py`、`src/video.py` | 数据、prompt、评分、抽样与视频输入 |
| `src/attention.py`、`src/kernels.py`、`src/graphs.py` | FA2、底层算子、CUDA Graph 加速 |
| `baselines/` | baseline 方法仓库、模型接入、视觉 token 裁剪和专用 worker |
| `analysis/` | 按论文 Figure／Table 组织的分析实验 |
| `scripts/` | `train.sh`、`eval_benchmark.sh` 两个兼容启动脚本 |
| `configs/` | ZeRO2／ZeRO3、Qwen3.5 依赖版本、统一入口示例配置 |
| `test/diagnostics/` | 正确性回归和诊断工具 |
| `data/`、`model/` | 本地训练／评测数据与底座权重 |
| `artifacts/`、`test/results/`、`wandb/` | 实验产物、缓存、参考快照和日志 |

`baselines/` 与 `artifacts/` 按当前仓库设置由 Git 忽略，本地运行仍会使用其中内容。
共用模型加载默认关闭 DeepStack；Qwen3-VL adapter 训练的冻结 teacher 显式开启 DeepStack，student 与评测仍关闭。历史实验复现以其冻结配置为准。

## 环境

从项目根目录运行，使用 `.venv/bin/python`。基础依赖见 `pyproject.toml` 和 `uv.lock`。
Qwen3.5 的 FLA、causal-conv1d、Triton 版本见 `configs/qwen35_adapter_requirements.txt`；
其独立依赖位于 `artifacts/dependencies/qwen35_python`，统一入口在导入模型前加入路径。

## vLLM 接入试验：Qwen3-VL / Qwen3.5 + embedding adapter

使用独立环境，避免改变训练依赖：

```bash
uv venv artifacts/dependencies/vllm023 --python .venv/bin/python
uv pip install --python artifacts/dependencies/vllm023/bin/python vllm==0.23.0 transformers==5.15.0
uv pip install --python artifacts/dependencies/vllm023/bin/python --no-deps -e .

artifacts/dependencies/vllm023/bin/python -m src.vllm_adapter \
  --base /path/to/Qwen3-VL-4B-Instruct \
  --checkpoint /path/to/qwen_embedding_adapter_step2000.pt \
  --output /path/to/new-export-directory
```

导出目录链接底座文件、保存 adapter 权重和来源哈希，并关闭 DeepStack；不修改原 checkpoint。
安装后的 `delta_vision` entry point 自动注册自定义架构。运行示例：

```bash
VLLM_PLUGINS=delta_vision CUDA_VISIBLE_DEVICES=0 \
  artifacts/dependencies/vllm023/bin/vllm serve /path/to/new-export-directory \
  --dtype bfloat16 --enforce-eager --tensor-parallel-size 1 \
  --no-async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 2 \
  --gpu-memory-utilization 0.18 --kv-cache-memory-bytes 1G \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'
```

Qwen3.5 使用同一个导出命令，将 `--base` 换为 `model/Qwen3.5-4B`，
`--checkpoint` 换为训练产生的 `qwen35_embedding_adapter_step2000.pt`。
Qwen3.5 的请求模板需设置 `enable_thinking=False`，与当前测评协议一致。
Qwen3.5 启动时还需 `--mamba-ssm-cache-dtype float32`；Python API 对应
`mamba_ssm_cache_dtype='float32'`。原 HF 的 recurrent state 为 FP32，不能沿用
vLLM 的 `auto`（随 BF16 模型使用 BF16 state），否则 decode 的数值会改变。

首版支持 dense Qwen3-VL / Qwen3.5、static embedding adapter、TP=PP=1 和 eager 执行。
使用 vLLM 原生分页 KV 与位置编码，每层替换视觉 hidden，只有文本执行 FFN。
Qwen3.5 的 24 个 linear-attention 层保留全部视觉位置的 Q/K/V、门控、causal convolution
和 Gated DeltaNet state 更新；8 个 full-attention 层使用原生 gated attention。
后续文本 decode 继续使用 vLLM 管理的 convolution/recurrent state 和分页 KV。
视觉位置插值、视觉 RoPE 的 FP32 计算及 Conv3d patch 投影与本项目 Transformers 5.15 参考路径对齐。
Qwen3.5 另外对齐视觉／文本 RoPE、RMSNorm、gated norm 和 FFN 的 BF16 舍入边界；
这些兼容运算目前用 eager PyTorch 执行。
**Qwen3.5 的数值验收尚未通过**：两条真实输入的答案／EOS 一致，但仍有 decode KL
超过 `1e-3` 的诊断门槛，不能据此声称完整 logits 对齐或九项 benchmark 结果不变。
**仍计算随后丢弃的视觉 query 输出；尚未移植本项目 text-query-only fast-path，因此不能用此版本声称完整的速度收益。**
CUDA Graph、分块 prefill、prefix caching、异步调度、量化和多卡模型并行均未验证，当前显式拒绝相关配置。
Recurrent 递推代码与导出单元检查已接入，但因缺少训练 checkpoint，实际模型验证暂缓；不作为已验证组合。
CPU 导出回归：`.venv/bin/python -m unittest discover -s test/diagnostics -p test_vllm_adapter.py`。

GPU 对照分两步运行（同一 GPU 顺序执行）；第一步选输入清单前两条作诊断，包含图前文本测试，不是 benchmark 成绩：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 artifacts/dependencies/vllm023/bin/python \
  test/diagnostics/test_vllm_adapter.py --vllm \
  --export /path/to/new-export-directory --data /path/to/realworldqa.jsonl \
  --image-root data/benchmarks/realworldqa --output artifacts/diagnostics/vllm_check
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  test/diagnostics/test_vllm_adapter.py --reference --output artifacts/diagnostics/vllm_check
```

Qwen3.5 的第二步使用 `PYTHONPATH=artifacts/dependencies/qwen35_python:.`，加载与现有
HF 训练／评测相同的 FLA、causal-conv1d、Triton 依赖。第一步仍使用隔离的 vLLM 环境。

检查输入 token 完全一致、单请求／混合批次生成一致，并记录 prefill 和逐步 decode 的完整 logits RMSE、cosine、KL 及 argmax。
BF16 的跨引擎输出不能以“能生成答案”代替数值对照。此脚本不报告并发训练时的速度。
2026-09-24 首轮两条 RealWorldQA 诊断：输入像素／M-RoPE 完全一致，单请求／混合批次答案及 EOS 一致；
prefill KL 为 `1.43e-5 / 2.91e-5`，四个生成位置的 logits cosine 为 `0.99965–0.99991`。
记录：`artifacts/diagnostics/vllm_qwen_embedding/final_validation/comparison.json`。这不是九项 benchmark 的精度结论，也不是逐位等价证明。

## 训练与准确率评测

```bash
.venv/bin/python -m src.run train --family qwen -- --help
.venv/bin/python -m src.run eval --family qwen -- --help
.venv/bin/python -m src.run train --config configs/unified_qwen.example.json --dry-run
.venv/bin/python -m src.run eval --config configs/unified_qwen.example.json --dry-run
```

`--family` 支持 `qwen`、`llava`、`qwen35`。`--` 后传对应实现的参数。
参数优先级：训练默认配置 → 用户 JSON 的 `model`／当前任务配置 → 命令行显式参数。
示例配置用于展示格式，正式训练须填写实际模型、清单和输出路径。
Adapter 类型、rank 从 checkpoint 恢复；模型架构由训练／测评共同使用。

### 默认训练协议（后续未特别说明均沿用）

固定参考：`qwen_static_embedding_2000_wandb_delayed_20260820_145737_kl_only`。
以后提出 adapter 训练时默认沿用此配置，只覆盖明确指定的模型、rank、数据或实验变量；不自行增加 loss。
Qwen3-VL 的公开训练入口自动加载 [默认配置](configs/qwen_training.default.json)，`scripts/train.sh` 默认值同步对齐。

| 项目 | 默认设置 |
|---|---|
| 模型／结构 | Qwen3-VL-4B-Instruct，所有36层 static embedding adapter，rank128；底座冻结，只训练 adapter；teacher DeepStack 开启，student 关闭 |
| 数据 | PixMo-AMA，`data/train/pixmo/pixmo_ama_full_valid.clean.jsonl` |
| 训练规模 | 2000 optimizer steps，8卡，每卡 batch4，梯度累积1，有效 batch32；seed44 |
| Loss | 仅 logits KL；token mean；top-k1024；temperature2；λ_logit=1；无 hidden MSE／attention 辅助项 |
| 优化器 | AdamW，LR5e-5，betas=(0.9,0.95)，weight decay0.01，clip1 |
| 调度 | cosine，warmup3%（60步），起始LR比例0，最终LR比例0.1 |
| 执行 | BF16，FA2，DeepSpeed ZeRO2；pixel bucket512 |
| 保存／记录 | 每500步保存，每5步记录；W&B online 开启 |

来源为原实验归档的 W&B 启动参数。旧源码及 `ds_zero2_coeff.json` 可从 Git `32ceeda` 恢复，现用 `configs/ds_zero2.json` 并由训练参数覆盖 batch／梯度累积／clip，AdamW 显式构造。
**当前 teacher 与旧实验一致，保留 DeepStack；student 使用支持 padding 的 batch FA2，旧实验 student 使用 SDPA，因此不宣称 loss 曲线逐位复现。评测的原生模型与 adapter 仍关闭 DeepStack。**
更换为 Qwen3.5 等架构时继承训练超参数，通过对应 worker 配置接入。

分布式训练（只需指定新的输出目录）：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.run train --family qwen -- --output-dir RUN/checkpoints
```

显式参数可覆盖默认值，例如 `--visual-adapter-rank 256`；关闭 W&B 用 `--no-wandb`。
两个 `.sh` 的配置直接写在脚本里（如 `TOKENIZERS_PARALLELISM=false`），修改赋值即可；不再读取同名环境变量作默认值。
默认训练完成后用8卡评测九项单图 benchmark：MMStar、RealWorldQA、GQA、MMB-EN、MMB-CN、MME、POPE、SQA、VQAv2；每项按seed44随机抽最多1000题，不足则全量。
当前训练任务已串接此评测；独立启动 `eval_benchmark.sh` 时指定 `--run-dir RUN`。

默认 DeepSpeed 配置为 `configs/ds_zero2.json`；ZeRO3 通过 `--deepspeed-config configs/ds_zero3.json` 选择。

Qwen3.5 沿用 `RUN/config.json`，其中定义数据、模型、训练和生成协议：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.run train --family qwen35 -- --run-dir RUN
.venv/bin/python -m src.run eval --family qwen35 -- \
  --run-dir RUN --method adapter --shard 0
```

`--method native` 测原生模型；现有 Qwen3.5 worker 保留原来的 8 卡训练／8 分片评测和 checkpoint 约束。
普通判分实现位于 `src/benchmarks.py`；不同历史实验的生成上限、答案提取和评分版本可能不同，不能混合汇总。

### Baseline

```bash
.venv/bin/python -m src.run eval --family qwen --workflow baseline -- \
  --run RUN --model qwen3-vl-4b --method dart --suite image --shard 0
```

`RUN` 需包含 config、jobs、固定清单、scoring_reference、source_hashes 和冻结源码。
`image` 是单图，`multimodal` 是多图／视频。入口按已准备任务选择算法，在独立进程执行冻结 worker。
新快照使用 `source/baselines/`，旧 `source/evaluation/baselines/` 与 scripts 布局仍支持。

## Video-MME 测速

```bash
.venv/bin/python -m src.benchmarking videomme --list
.venv/bin/python -m src.benchmarking videomme llm -- --help
.venv/bin/python -m src.benchmarking videomme llm -- queue \
  --run artifacts/diagnostics/video_llm_NEW \
  --previous artifacts/diagnostics/video_adapter_layer_ablation_999_20260923 \
  --cases base adapter --gpus 0 1 2 3 4 5 6 7
```

| Profile | 用途 |
|---|---|
| `llm` | base／adapter／去掉视觉 token／关闭部分层视觉注入的统一计时 |
| `resources` | base、adapter 和六种 baseline 的资源复测 |
| `vision-removed` | 去掉视觉 token，prefill 包含视觉编码的复测 |
| `flops-removed` | 去掉视觉 token 后的 LLM FLOPs |
| `resource-report`、`layer-report` | 资源与层屏蔽实验汇总 |
| `base-diagnostic`、`adapter-diagnostic`、`pruning-diagnostic` | 各执行路径的数值和性能诊断 |
| `prefill` | 共用 prefill 工具接口 |

`llm` 的固定协议：Video-MME 999 条、8 帧，FA2／BF16／DeepStack 关闭／CUDA Graph，adapter fast-path 开启。
视觉 embedding 在计时前准备，prefill／decode 不包含视觉编码。固定生成 8 token：prefill 产生第一个 token，随后 7 次 decode。
单次 total=prefill+decode；各输入预热后测 3 次，先取该输入中位数，再对输入平均。
`continuous_trials` 包含选 token，decode 末尾同步；`trials` 保留逐 forward 同步且不含选 token 的旧口径，二者分开。
主 report 使用 continuous；`layer-report` 保留其历史汇总口径。独立求 total 中位数时，汇总值可能与阶段中位数之和略有差异。

Peak 使用 `max_memory_allocated`、单位 GiB，包含驻留权重、输入和图池。FLOPs 独立统计，不在计时中挂统计 hook。
`resources` 的原计时包含视觉编码；其排除视觉编码 FLOPs 字段及不同 profile 不能混报。
`--run` 使用新目录，`--previous` 指向冻结协议和数值参考。队列可暂停已知 RAM 空转负载并在结束后恢复，不终止未知任务。

## 论文分析

对应根目录 `ICLR_2027_Visual_KV.pdf` 当前版本，机器可读入口在 `analysis/catalog.json`。

```bash
.venv/bin/python -m analysis --list
.venv/bin/python -m analysis fig05_hybrid_attention --describe
.venv/bin/python -m analysis fig01a_hidden_channels run -- --help
```

| 论文位置 | 目录 | 实验 |
|---|---|---|
| Figure 1(a), Appendix H | `analysis/fig01a_hidden_channels/` | 视觉 hidden-channel 压缩 |
| Figure 1(b) | `analysis/fig01b_hidden_prediction/` | 独立残差 MLP 预测 |
| Table 5 | `analysis/table05_native_rank/` | 原生 Q / QKᵀ / attention output 秩统计 |
| Figure 3 | `analysis/fig03_visual_effect/` | 原生轨迹的视觉作用低秩恢复 |
| Table 6 | `analysis/table06_layer_effect/` | 选定层的因果视觉作用干预 |
| Table 10, Appendix E | `analysis/table10_training_objective/` | Init / SFT / OPD / Supervised KD |
| Figure 4, Appendix G | `analysis/fig04_adapter_rank/` | Adapter bottleneck rank |
| Table 13, Appendix I | `analysis/table13_pruning_adapter/` | DART / DivPrune + embedding adapter |
| Tables 14–15, Appendix J | `analysis/table14_15_layer_skipping/` | 关闭部分层视觉注入：准确率与资源 |
| Figure 5, Appendix K | `analysis/fig05_hybrid_attention/` | Qwen3.5 recurrent state 与 full attention 路径 |

下面保留各项实验定义、原始协议与入口。历史成绩与当前 DeepStack-off 重跑须区分。

<!-- analysis:fig01a_hidden_channels -->
### Figure 1(a), Appendix H：视觉 hidden-channel 压缩

Qwen3-VL-4B（hidden=2560）和 LLaVA-1.5-7B（4096）。1024 张 PixMo-AMA 图像拟合每层不中心化 PCA 基底；同层基底跨 token、样本共享，rank=0/32/64/128/256/512/1024。SQA、RealWorldQA、MMStar 各最多1000题。

每层视觉 hidden 从原生轨迹缓存恢复，再做通道投影；文本状态继续传播。视觉 token 数不变。`visual_channel_rank_grid.py` 同时含早期级联实验工具，论文的缓存恢复版本以 `visual_channel_native_cache.py` 为准。

运行：`python -m analysis fig01a_hidden_channels run -- launch --output NEW_RUN`。先用 `--help` 查看分片和模型参数。
<!-- /analysis:fig01a_hidden_channels -->

<!-- analysis:fig01b_hidden_prediction -->
### Figure 1(b)：独立残差 MLP 预测

36 个独立逐 token MLP，以初始视觉 embedding E 为输入。目标为第 l 层 input RMSNorm 输出；预测为 `RMSNorm_l(E)+MLP_l(E)`，训练的是残差，不是直接比较 E 和目标。2000 step，冻结 teacher，预测不注入 teacher。逐图、逐层平均 MSE；评测完整预测的 cosine/MSE。

数据：PixMo 训练，RealWorldQA 765、MMStar 1000、SQA 1000。运行：`python -m analysis fig01b_hidden_prediction run -- launch --output NEW_RUN`。

结果来源：`artifacts/diagnostics/initial_token_postnorm_all36_20260916/results.csv`。`plot` 按原脚本写入 `artifacts/figures/layerwise_residual_mlp_20260917`。辅助文件中更早的 prenorm/mixing probe 及 report_initial_token_mlp.py 不等于论文这张图。论文数值汇总使用 run -- report --output RUN。
<!-- /analysis:fig01b_hidden_prediction -->

<!-- analysis:table05_native_rank -->
### Table 5：原生 Q / QKᵀ / attention output 秩统计

计算实现已恢复：[visual_rank_statistics.py](analysis/table05_native_rank/visual_rank_statistics.py)。最初只搜索现存文件/Git，误判为源码缺失；随后从原始编辑记录恢复完整脚本及后续补丁。**恢复前的原文件 SHA256 与原实验 provenance 完全一致**：

`5fd6ab4650aae23164ea05a88f19c3a174c63a4eba8b7667e9e847e9e7960594`

#### 实验定义

Qwen3-VL-4B / LLaVA-1.5-7B，MMStar1000、RealWorldQA765、SQA1000，逐样本、逐层、逐head统计，没有 token/head 下采样。

- Q：原生 QNorm/RoPE 后视觉行，既统计拼接 heads，也逐 head 统计。
- QKᵀ：缩放后、加 causal mask 前的 visual-to-visual 分数矩阵；各 head 分开计算。
- Attention output：Wo 后、残差前的视觉行。
- r95 使用平方奇异值能量；effective rank 使用归一化奇异值的熵指数。
- 数值计算使用 FP64 Gram 特征值/QR；先平均样本，再等权平均层，QKᵀ 另外等权平均 head。
- 只挂观察 hook，原脚本会检查挂/卸 hook 后 logits 逐位一致。

#### 入口

```bash
.venv/bin/python -m analysis table05_native_rank run -- --help
### 8卡分片计算；新输出目录，避免覆盖原始结果
.venv/bin/python -m analysis table05_native_rank run -- launch --output NEW_RUN
.venv/bin/python -m analysis table05_native_rank run -- merge --output NEW_RUN
.venv/bin/python -m analysis table05_native_rank report
```

原结果：`artifacts/diagnostics/native_visual_rank_20260912/`。历史原生模型保留 DeepStack；现在调用共用 loader 遵守项目的 DeepStack-off 设置。因此恢复的是原统计算法，当前重新计算必须单列模型协议，不能直接声称全量数字已重新复现。

#### 恢复与验证

`artifacts/maintenance/paper_source_recovery_20260924/` 保存原文件、恢复来源及验证。规范路径版本只修改子进程模块路径，数值函数和 observer 不变。原始3项公式测试覆盖 r95/ER、QR 后 QKᵀ 谱、Gram/SVD 等价；另对完整历史逐题记录重新汇总并与原结果比较。具体结果见该目录 RESULTS.md。
<!-- /analysis:table05_native_rank -->

<!-- analysis:fig03_visual_effect -->
### Figure 3：原生轨迹的视觉作用低秩恢复

每层使用原生轨迹采集的文本 attention 视觉作用差值，投影后恢复到文本流。共享逐层基底由校准 token 的差值矩阵 SVD 得到。文本受干预后继续传播，但每层差值来自原生轨迹。

`native_trajectory.py` 及匹配 helper `native_effect_core.py` 从 Git **28bc51c** 恢复，保留原 generation 测评实现。先运行 `python -m analysis fig03_visual_effect native -- --help`，设置模型、benchmark、数据、rank、输出目录。

`rerun` 是后续 shared-rank 重测，不冒充原图数值：`artifacts/diagnostics/visual_effect_uncentered_20260920_104024/summary.json`。历史图与后续重测的模型协议/基准成绩有差别；本次只验证代码迁移，不宣称复跑全量论文成绩。
<!-- /analysis:fig03_visual_effect -->

<!-- analysis:table06_layer_effect -->
### Table 6：选定层的因果视觉作用干预

在当前已受前层干预影响的状态上，重新算正常文本 attention 与屏蔽视觉 KV 后的差值；投影并恢复该差值。与 Figure 3 的原生轨迹差值不同。基底按模型/数据集/层共享，在不含答案的 benchmark prompts 上校准（transductive oracle）。

rank=0/32/64/128；First5、First10、Middle10、Last10、All。Qwen 中间13–22、末尾26–35；LLaVA 中间11–20、末尾22–31（0起编号）。

历史结果：`artifacts/diagnostics/causal_effect_2models_2bench_20260912`。当时 SDPA / DeepStack 开启、生成上限8，RWQA765、MMStar前1000。当前项目模型加载关闭 DeepStack；不能直接把当前重跑称为历史数字的原协议复现。

入口：`python -m analysis table06_layer_effect run -- --help`。不自动启动旧目录任务。
<!-- /analysis:table06_layer_effect -->

<!-- analysis:table10_training_objective -->
### Table 10 / Appendix E：Init、SFT、OPD、Supervised KD

原始训练入口已找回：[pixmo_objective_comparison.py](analysis/table10_training_objective/pixmo_objective_comparison.py)。来自原始编辑记录的完整 Add File 补丁，连同原配套测试和实验说明一并恢复。恢复文件另与两次独立历史完整读回逐字节核对一致。此前“旧 objective trainer 缺失”的判断已撤回。

#### 原训练方法

Qwen3-VL-4B frozen backbone，rank128，全36层 adapter，PixMo-AMA，2000 step，seed44，8卡、每卡batch4、GA1。AdamW LR5e-5，betas(.9,.95)，WD .01，clip1；cosine，3%warmup，末端10%LR；BF16/FA2/ZeRO2。

- **Init**：不训练，逐层注入初始视觉 embedding，走原生完整 forward。
- **SFT**：gold answer 的 next-token CE，包含模板EOS，不跑 teacher。
- **Supervised KD**：gold answer 轨迹上的 teacher→student top1024 KL，temperature2，按答案token平均，系数1。
- **OPD**：当前 student 从初始视觉/问题输入采样，temperature1、无top-k/p、最多128token、EOS停止。直接回放采样的 token ID，不decode/re-tokenize、不把真实答案填回轨迹、不在截断处伪造EOS。原生 teacher 评估相同轨迹，计算 teacher→student KL。采样无梯度，回放更新 adapter。
- OPD 的 `FullVocabKL` 使用完整词表、FP32归一化、按token分块并使用解析梯度。旧配置另支持top1024，不能把两者混报；实际设置以各组config为准。

Static与recurrent是不同的记忆参数化，训练核心继续共用 src/training/engine.py；这里仅保留原实验的OPD loss/采样插件，不复制优化器或训练循环。

#### 入口与配置

```bash
.venv/bin/python -m analysis table10_training_objective train -- --help
.venv/bin/python -m analysis table10_training_objective train -- --config NEW_CONFIG.json
.venv/bin/python -m analysis table10_training_objective init -- --help
```

正式8卡训练在上述分析入口外加 torch.distributed.run。普通训练/测评仍通过 `python -m src.run train|eval`。

历史配置/日志/结果：`artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/`。重跑应复制配置到新目录，更新 output_dir、metrics_jsonl、W&B标识以及指向已删除工作目录的 deepspeed_config（本项目 configs/ds_zero2.json）。不要覆盖原实验。

规范入口补充当前共用 trainer 必需的 `teacher_deepstack=False` 缺省字段，沿用全项目关闭 DeepStack 的要求；没有更换 loss。原实验 teacher 的DeepStack设置与现在不同，历史完整训练成绩未在本次重跑。

#### 恢复与验证

原实现/原测试/当时实验说明保存在 `artifacts/maintenance/paper_source_recovery_20260924/reference/`。验证涵盖完整词表KL及梯度、EOS/padding轨迹、采样来自student而非gold，以及原实现与迁移实现的精确数值对照。结果见同目录 RESULTS.md。
<!-- /analysis:table10_training_objective -->

<!-- analysis:fig04_adapter_rank -->
### Figure 4, Appendix G：Adapter bottleneck rank

rank32/64/128/256/512/1024 的 PixMo static KL adapter，2000 step；rank128 使用对应参考实验。训练快照固定 Git28bc51c；测评使用已确认的7f266415答案提取。不要修改归档源码或把新判分偷偷代入旧结果。

训练：`artifacts/experiments/pixmo_static_rank_sweep/qwen3vl4b_pixmo_static_kl_rank_sweep_2000_20260921`。测评：`artifacts/eval/pixmo_static_rank_sweep_9image_7f266415_20260922`。

入口：`python -m analysis fig04_adapter_rank train -- --help` / `eval -- --help`。这些是论文批量实验编排；模型 forward 与通用训练实现不另复制。
<!-- /analysis:fig04_adapter_rank -->

<!-- analysis:table13_pruning_adapter -->
### Table 13, Appendix I：DART / DivPrune + embedding adapter

Qwen3-VL-4B；50%、20%、5%；九个单图 benchmark。DART 先进入 LLM 再裁剪；DivPrune 在进入 LLM 前选择。后续 adapter 对保留位置的初始 embedding 生成视觉 hidden。

结果：`artifacts/eval/qwen4b_pruned_embedding_adapter_20260923`。固定题目清单与评分源码，不改变 retention 口径。`qwen_pruned_embedding_adapter.py` 是唯一组合实现，被评测、测试共用。

入口：`python -m analysis table13_pruning_adapter run -- --help`。
<!-- /analysis:table13_pruning_adapter -->

<!-- analysis:table14_15_layer_skipping -->
### Tables 14–15, Appendix J：关闭部分层视觉注入：准确率与资源

关闭层0–4和26–35，或0–9和26–35的视觉 KV 注入；文本 attention/FFN 正常，中间层使用 adapter。准确率九个单图 benchmark，速度使用固定 Video-MME 999题。

历史 Table15 时间包含视觉编码，FLOPs 不包含视觉编码；报告必须写明两个口径，不因目录重构改变统计。资源执行复用 src/benchmarking/common/prefill.py。

准确率：`artifacts/eval/qwen4b_adapter_first5_last10_off_20260923` 及 first10 对应目录。资源：`artifacts/diagnostics/video_adapter_layer_ablation_999_20260923` 与 `artifacts/reports/video_adapter_layers_including_vision_20260923`。

入口：`python -m analysis table14_15_layer_skipping eval -- --help`，measure/profile/report 各沿用自己的参数。
<!-- /analysis:table14_15_layer_skipping -->

<!-- analysis:fig05_hybrid_attention -->
### Figure 5, Appendix K：Qwen3.5 recurrent state 与 full attention 路径

Qwen3.5-4B，24个LA、8个FA，DeepStack关闭。Boundary rank0 在视觉段结束时恢复该层视觉段前的 recurrent state；视觉 hidden/FFN 与 FA 保留，不能叫删除视觉 token 或 beta=0。

FA 干预只阻断 text query 对视觉 KV 的访问：关键11/15两层，或其余6层3/7/19/23/27/31。RWQA765、MMStar1500，seed44。论文比较使用 **8-token** 结果。boundary 脚本按历史逻辑生成64 token，并分别报告 cap64 与经过独立短生成核验的 cap8 前缀；本次整理未改这段逻辑。论文对照取 cap8，不能将默认主报告 cap64 混入。

结果：`artifacts/experiments/qwen35_fa_ablation/key11_15_vs_other6_full_cap8_20260923`；`artifacts/experiments/qwen35_memory/boundary_rank0_full_rwqa765_mmstar1500_cap64_20260923` 内 cap8 结果。

入口：`python -m analysis fig05_hybrid_attention boundary -- --help` 或 full-attention。共享状态与mask逻辑在本目录两个 qwen35 模块。

#### 同一机制问题的补充实验

原先散在 src/scripts 的 23 个机制分析与调度文件已归入本目录，不再保留旧副本。
这些是 Figure 5 的探索和控制实验，**不是论文新增的 Table/Figure**：

| 补充问题 | 统一入口 runner |
|---|---|
| recurrent state 秩、功能相似性、扰动敏感性 | `memory-queue` / `memory-worker` / `memory-report` |
| full-attention 视觉读取干预 | `fa-queue` / `fa-worker` |
| 视觉位置 beta=0，保留 decay 和 conv | `no-write-queue` / `no-write-worker` |
| 原生轨迹下 visual/text state 来源拆分 | `sources-queue` / `sources-worker` / `sources-report` |
| 正确选项与错误选项方向的投影 | `projection-queue` / `projection-worker` / `projection-report` |
| delta write 与 subtraction 拆分 | `cancellation-queue` / `cancellation-worker` / `cancellation-report` |
| no-write / no-forget / state-skip、双轨 readout 和分组控制 | `write-forget` / `write-forget-report` |

例如：`python -m analysis fig05_hybrid_attention write-forget -- --help`。
沿用各实验原参数、公式、清单和生成长度；不把这些不同干预都叫 boundary rank0。
迁移记录：`artifacts/maintenance/source_cleanup_20260924/`。
<!-- /analysis:fig05_hybrid_attention -->

## 验证

```bash
.venv/bin/python -m unittest discover -s test/diagnostics -p 'test_*.py' -v
```

2026-09-24 最新目录迁移验证：94 项测试中 93 通过；1 项依赖已删除外部历史目录，跳过。
真实 Qwen3-VL-4B 两步训练的 loss、全部 adapter 梯度和更新后参数，以及 base／adapter 各 8 步 logits 完全一致，最大差 0。
CUDA Graph 捕获／回放的 logits、tokens、KV hash 也一致。16 项入口检查通过。
记录和脚本在 `artifacts/maintenance/runtime_into_src_20260924/`。

这些是有限输入的代码回归，不代表完整 benchmark 或 2000-step 训练重跑；本次也没有在共享 GPU 上据此宣称速度数值完全相同。
Qwen3.5 和 LLaVA 的本轮测试范围为现有单元测试／入口检查，真实底座数值验证以各历史报告为准。
