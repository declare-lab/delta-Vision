命令行实现放在这里，按实验用途分层。

`scripts/*.py` 只保留兼容 wrapper；真正可复用的训练、评测、数据准备、oracle、benchmark、grounding 和 Qwen 实验入口都在本目录下。

目录约定：

```text
bench/        Sidecar kernel 和 generation speed benchmark
data/         数据集下载、格式转换和本地 image materialization
diagnostics/ 诊断 Teacher-forced、rollout、hidden trajectory 等问题
eval/         LLaVA shared 主线 MMStar/RWQA/hybrid 评测
grounding/   PixMo-Points grounding 训练和评测
oracle/      attention effect collection、PCA basis、low-rank oracle
qwen/         Qwen3-VL 训练、评测和 oracle 实验
train/        LLaVA shared 主线训练入口
```
