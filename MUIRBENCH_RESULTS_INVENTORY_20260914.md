# MuirBench 已有结果汇总（2026-09-14）

均为 Qwen3-VL-4B-Instruct 系列。分数单位为百分比。不同抽样、提示格式、像素处理、DeepStack 配置和干预不能混作同一排行榜。

范围：当前工作区已保存的完整模型测评分数、历史结果和 MuirBench 专项诊断。未完成/已废弃结果单列；小规模 smoke 不视为模型准确率。没有数值的张量一致性检查不冒充 benchmark 成绩。

## 1. 当前统一协议：随机1000条

seed42，从2600条无放回随机抽样；media_first_v1，原始图片处理，所有图片保留，DeepStack关闭，greedy最多8 tokens。两个后层 Adapter 是不同训练 checkpoint、rank512，不能当作只改变起始层的纯消融。其他方法没有当前协议下的结果时，不用历史成绩填补。

| 方法 | 题数 | Accuracy |
|---|---|---|
| Base | 1000 | 55.50 |
| Embedding Adapter + KL（全层，rank128） | 1000 | 40.70 |
| 混合数据训练 Embedding Adapter | 1000 | 44.10 |
| DART 保留20% | 1000 | 49.30 |
| DART 保留5% | 1000 | 43.40 |
| 原生0–6，Adapter7–34，原生35（rank512） | 1000 | 43.10 |
| 原生0–15，Adapter16–34，原生35（rank512） | 1000 | 51.60 |

来源：[artifacts/diagnostics/muir_random1000_seed42_matched_20260914/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_random1000_seed42_matched_20260914/results.json)

来源：[artifacts/diagnostics/muir_late_adapter_random1000_20260914/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_late_adapter_random1000_20260914/results.json)

## 2. 同一随机1000条的输入/干预实验

全部关闭 DeepStack。视频与拼图实验分别有自己的匹配输入对照；不可把43.40与原协议40.70之差归因于拼接。分隔符干预改变 attention，不是普通评测协议。

| 实验 | 条件 | 题数 | Base | Embedding Adapter |
|---|---|---|---|---|
| 等分辨率图像/视频对照 | 512方形多图，单独输入 | 1000 | 54.10 | 40.40 |
| 等分辨率图像/视频对照 | 相同图片按视频帧输入 | 1000 | 57.10 | 38.30 |
| 拼图对照 | 512方形带图号面板，单独输入 | 1000 | 60.90 | 43.40 |
| 拼图对照 | 相同带图号面板，纵向拼成单图 | 1000 | 56.90 | 40.50 |
| 分隔符读取干预 | 只限制分隔符/图号 query 读取本图视觉 KV | 1000 | 未测 | 40.30 |

来源：[artifacts/diagnostics/muir_images_as_video_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_images_as_video_20260914/summary.json)

来源：[artifacts/diagnostics/muir_concat_images_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_concat_images_20260914/summary.json)

来源：[artifacts/diagnostics/muir_separator_binding_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_separator_binding_20260914/summary.json)

## 3. 旧样本集：原始文件前1000条，统一图片放前

不是随机1000条；media_first_v1、DeepStack关闭。SFT原记录最多8 tokens为39.10；对159条截断回答延长到128 tokens后净多对10条，对应合并准确率40.10（其余841条沿用原记录）。

| 方法 | 题数 | Accuracy |
|---|---|---|
| Base | 1000 | 51.20 |
| Embedding Adapter + KL | 1000 | 40.10 |
| Recurrent Adapter + KL | 1000 | 40.70 |
| Embedding Adapter + SFT | 1000 | 39.10 |
| Embedding Adapter + OPD | 1000 | 40.30 |
| DART 保留20% | 1000 | 46.90 |
| DART 保留5% | 1000 | 41.10 |

来源：[artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913/results.json)

来源：[artifacts/diagnostics/muir_dart_mediafirst_20260914/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_dart_mediafirst_20260914/results.json)

来源：[artifacts/diagnostics/muir_length_audit_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_length_audit_20260914/summary.json)

## 4. 六个 pruning baseline 的旧完整记录

原始前1000条，每格1000条，DeepStack关闭；使用旧输入排布，不与第1/3节混比。同批 Base=53.50。这里比例是脚本的视觉token保留预算，不是跨全部层求和后的计算占比。

| 方法 | 保留20% | 保留5% |
|---|---|---|
| fastv | 48.60 | 42.10 |
| dart | 49.00 | 45.10 |
| divprune | 48.80 | 46.10 |
| zoo | 50.20 | 43.90 |
| sparsevlm | 47.00 | 42.20 |
| visionzip | 47.90 | 43.60 |

来源：[artifacts/diagnostics/multimodal_baselines_nodeepstack_20260913/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/multimodal_baselines_nodeepstack_20260913/results.json)

## 5. 最早训练输出目录中的历史测评

旧协议存档，未统一为当前 no-DeepStack/修正后的推理和提示协议，不作为当前公平对比。混合训练历史结果是全2600条；其余为旧前1000条。

| 训练版本 | 题数 | 配套Teacher/Base | Adapter |
|---|---|---|---|
| Embedding Adapter + OPD | 1000 | 49.50 | 37.90 |
| Recurrent Adapter + KL | 1000 | 49.50 | 37.50 |
| Embedding Adapter + SFT | 1000 | 49.50 | 36.90 |
| Embedding Adapter + KL | 1000 | 49.50 | 36.40 |
| 混合数据训练 Adapter | 2600 | 62.77 | 42.85 |

来源：[artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/opd/eval/muirbench/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/opd/eval/muirbench/results.json)

来源：[artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/recurrent_kl/eval/muirbench/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/recurrent_kl/eval/muirbench/results.json)

来源：[artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/sft/eval/muirbench/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/sft/eval/muirbench/results.json)

来源：[artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/eval/muirbench/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/eval/muirbench/results.json)

来源：[artifacts/experiments/qwen_mixed_adapter/mixed60_25_15_resume1500_2000_20260911/eval_final/muirbench/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/experiments/qwen_mixed_adapter/mixed60_25_15_resume1500_2000_20260911/eval_final/muirbench/results.json)

## 6. 同一 Adapter 的两套推理实现

两边都是同一 Adapter，完整路径并不是Base：只是在HF原生层前替换相同的视觉memory。最新随机1000的18条BF16预测差异在FP32复核时全部一致，最大输出KL=2.54e-10。

| 输入协议 | 题数 | 快速路径 | 原生完整序列路径 |
|---|---|---|---|
| 随机1000，原始图片放前 | 1000 | 40.70 | 40.90 |
| 旧前1000，图片放前 | 1000 | 40.10 | 39.90 |
| 旧前1000，图文交错 | 1000 | 36.60 | 36.80 |
| 旧前1000，完整路径仅EOS停止 | 1000 | 40.10 | 39.90 |

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_random1000_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_random1000_20260914/summary.json)

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914/summary.json)

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_interleaved_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_interleaved_20260914/summary.json)

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_eos_only_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_eos_only_20260914/summary.json)

## 7. 旧前1000条：数值精度与图像编码复核

全部是旧前1000条，不是当前随机1000；不和第1节直接比较。

| 检查 | 条件 | 题数 | Adapter Accuracy |
|---|---|---|---|
| muir_fullprecision_inference_20260914 | bf16 | 1000 | 40.10 |
| muir_fullprecision_inference_20260914 | fp32 | 1000 | 39.90 |
| muir_vision_backend_20260914 | sdpa | 1000 | 40.10 |
| muir_vision_backend_20260914 | flash_attention_2 | 1000 | 40.10 |
| muir_source_pixels_20260914 | reencoded_jpeg | 1000 | 40.10 |
| muir_source_pixels_20260914 | source_pixels | 1000 | 39.80 |

来源：[artifacts/diagnostics/muir_fullprecision_inference_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_fullprecision_inference_20260914/summary.json)

来源：[artifacts/diagnostics/muir_vision_backend_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_vision_backend_20260914/summary.json)

来源：[artifacts/diagnostics/muir_source_pixels_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_source_pixels_20260914/summary.json)

## 8. 单图/多图候选判断（不是普通benchmark accuracy）

随机1000中的132条图片选择题，441个候选图；73条有正确图片、59条None。先分别问每个候选Yes/No，平均两种答案顺序的logit margin。single只给当前候选图片；multi_target给所有图并问指定候选。

| 模型 | 条件 | 正确候选判Yes/73 | 错误候选判No/368 | 候选排名正确/73 |
|---|---|---|---|---|
| Base | single | 68.49 | 95.38 | 97.26 |
| Base | multi_target | 71.23 | 92.93 | 73.97 |
| Embedding Adapter + KL（全层，rank128） | single | 67.12 | 91.03 | 90.41 |
| Embedding Adapter + KL（全层，rank128） | multi_target | 23.29 | 89.95 | 34.25 |

来源：[artifacts/diagnostics/muir_candidate_isolation_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_candidate_isolation_20260914/summary.json)

## 9. 所有 MuirBench 专项 summary 的评分明细

下面按原始实验逐一列出，包括局部样本、oracle、排列/提示词、逐层替换、单图和读取干预。它们不是同一任务协议，不能把最高值当作当前Adapter的整体分数。`accuracy_change_points`是百分点变化，不是准确率；其余评分按原summary的百分比记录。分母未单列者请查看来源。

## muir_binding_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /Geographic Understanding/end_marker/base | accuracy | 100 | 42.00 |
| /Geographic Understanding/end_marker/static_kl | accuracy | 100 | 13.00 |
| /Geographic Understanding/natural_number/base | accuracy | 100 | 59.00 |
| /Geographic Understanding/natural_number/static_kl | accuracy | 100 | 13.00 |
| /Geographic Understanding/explicit_option/base | accuracy | 100 | 54.00 |
| /Geographic Understanding/explicit_option/static_kl | accuracy | 100 | 15.00 |
| /Image-Text Matching/end_marker/base | accuracy | 84 | 83.33 |
| /Image-Text Matching/end_marker/static_kl | accuracy | 84 | 38.10 |
| /Image-Text Matching/natural_number/base | accuracy | 84 | 84.52 |
| /Image-Text Matching/natural_number/static_kl | accuracy | 84 | 38.10 |
| /Image-Text Matching/explicit_option/base | accuracy | 84 | 80.95 |
| /Image-Text Matching/explicit_option/static_kl | accuracy | 84 | 38.10 |

来源：[artifacts/diagnostics/muir_binding_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_binding_20260914/summary.json)

## muir_candidate_isolation_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, condition=single] | sensitivity | 73 | 68.49 |
| /results/0 [method=base, condition=single] | specificity | 368 | 95.38 |
| /results/0 [method=base, condition=single] | balanced_accuracy | 441 | 81.94 |
| /results/0 [method=base, condition=single] | top_candidate_accuracy | 73 | 97.26 |
| /results/0 [method=base, condition=single] | reject_all_unanswerable | 59 | 91.53 |
| /results/1 [method=base, condition=multi_target] | sensitivity | 73 | 71.23 |
| /results/1 [method=base, condition=multi_target] | specificity | 368 | 92.93 |
| /results/1 [method=base, condition=multi_target] | balanced_accuracy | 441 | 82.08 |
| /results/1 [method=base, condition=multi_target] | top_candidate_accuracy | 73 | 73.97 |
| /results/1 [method=base, condition=multi_target] | reject_all_unanswerable | 59 | 89.83 |
| /results/2 [method=embedding_adapter, condition=single] | sensitivity | 73 | 67.12 |
| /results/2 [method=embedding_adapter, condition=single] | specificity | 368 | 91.03 |
| /results/2 [method=embedding_adapter, condition=single] | balanced_accuracy | 441 | 79.08 |
| /results/2 [method=embedding_adapter, condition=single] | top_candidate_accuracy | 73 | 90.41 |
| /results/2 [method=embedding_adapter, condition=single] | reject_all_unanswerable | 59 | 83.05 |
| /results/3 [method=embedding_adapter, condition=multi_target] | sensitivity | 73 | 23.29 |
| /results/3 [method=embedding_adapter, condition=multi_target] | specificity | 368 | 89.95 |
| /results/3 [method=embedding_adapter, condition=multi_target] | balanced_accuracy | 441 | 56.62 |
| /results/3 [method=embedding_adapter, condition=multi_target] | top_candidate_accuracy | 73 | 34.25 |
| /results/3 [method=embedding_adapter, condition=multi_target] | reject_all_unanswerable | 59 | 89.83 |

来源：[artifacts/diagnostics/muir_candidate_isolation_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_candidate_isolation_20260914/summary.json)

## muir_concat_images_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, mode=separate] | accuracy | 1000 | 60.90 |
| /results/1 [method=base, mode=concat] | accuracy | 1000 | 56.90 |
| /results/2 [method=embedding_adapter, mode=separate] | accuracy | 1000 | 43.40 |
| /results/3 [method=embedding_adapter, mode=concat] | accuracy | 1000 | 40.50 |
| /tasks/Action Understanding/base/separate | accuracy | 68 | 50.00 |
| /tasks/Action Understanding/base/concat | accuracy | 68 | 39.71 |
| /tasks/Action Understanding/embedding_adapter/separate | accuracy | 68 | 33.82 |
| /tasks/Action Understanding/embedding_adapter/concat | accuracy | 68 | 32.35 |
| /tasks/Attribute Similarity/base/separate | accuracy | 70 | 62.86 |
| /tasks/Attribute Similarity/base/concat | accuracy | 70 | 62.86 |
| /tasks/Attribute Similarity/embedding_adapter/separate | accuracy | 70 | 62.86 |
| /tasks/Attribute Similarity/embedding_adapter/concat | accuracy | 70 | 57.14 |
| /tasks/Cartoon Understanding/base/separate | accuracy | 29 | 48.28 |
| /tasks/Cartoon Understanding/base/concat | accuracy | 29 | 48.28 |
| /tasks/Cartoon Understanding/embedding_adapter/separate | accuracy | 29 | 44.83 |
| /tasks/Cartoon Understanding/embedding_adapter/concat | accuracy | 29 | 48.28 |
| /tasks/Counting/base/separate | accuracy | 95 | 38.95 |
| /tasks/Counting/base/concat | accuracy | 95 | 36.84 |
| /tasks/Counting/embedding_adapter/separate | accuracy | 95 | 38.95 |
| /tasks/Counting/embedding_adapter/concat | accuracy | 95 | 32.63 |
| /tasks/Diagram Understanding/base/separate | accuracy | 163 | 79.75 |
| /tasks/Diagram Understanding/base/concat | accuracy | 163 | 76.69 |
| /tasks/Diagram Understanding/embedding_adapter/separate | accuracy | 163 | 57.67 |
| /tasks/Diagram Understanding/embedding_adapter/concat | accuracy | 163 | 43.56 |
| /tasks/Difference Spotting/base/separate | accuracy | 126 | 43.65 |
| /tasks/Difference Spotting/base/concat | accuracy | 126 | 38.10 |
| /tasks/Difference Spotting/embedding_adapter/separate | accuracy | 126 | 33.33 |
| /tasks/Difference Spotting/embedding_adapter/concat | accuracy | 126 | 31.75 |
| /tasks/Geographic Understanding/base/separate | accuracy | 35 | 45.71 |
| /tasks/Geographic Understanding/base/concat | accuracy | 35 | 42.86 |
| /tasks/Geographic Understanding/embedding_adapter/separate | accuracy | 35 | 28.57 |
| /tasks/Geographic Understanding/embedding_adapter/concat | accuracy | 35 | 25.71 |
| /tasks/Image-Text Matching/base/separate | accuracy | 184 | 83.15 |
| /tasks/Image-Text Matching/base/concat | accuracy | 184 | 79.89 |
| /tasks/Image-Text Matching/embedding_adapter/separate | accuracy | 184 | 52.17 |
| /tasks/Image-Text Matching/embedding_adapter/concat | accuracy | 184 | 54.89 |
| /tasks/Ordering/base/separate | accuracy | 27 | 25.93 |
| /tasks/Ordering/base/concat | accuracy | 27 | 22.22 |
| /tasks/Ordering/embedding_adapter/separate | accuracy | 27 | 37.04 |
| /tasks/Ordering/embedding_adapter/concat | accuracy | 27 | 33.33 |
| /tasks/Scene Understanding/base/separate | accuracy | 65 | 58.46 |
| /tasks/Scene Understanding/base/concat | accuracy | 65 | 55.38 |
| /tasks/Scene Understanding/embedding_adapter/separate | accuracy | 65 | 61.54 |
| /tasks/Scene Understanding/embedding_adapter/concat | accuracy | 65 | 55.38 |
| /tasks/Visual Grounding/base/separate | accuracy | 35 | 37.14 |
| /tasks/Visual Grounding/base/concat | accuracy | 35 | 40.00 |
| /tasks/Visual Grounding/embedding_adapter/separate | accuracy | 35 | 34.29 |
| /tasks/Visual Grounding/embedding_adapter/concat | accuracy | 35 | 34.29 |
| /tasks/Visual Retrieval/base/separate | accuracy | 103 | 66.02 |
| /tasks/Visual Retrieval/base/concat | accuracy | 103 | 56.31 |
| /tasks/Visual Retrieval/embedding_adapter/separate | accuracy | 103 | 12.62 |
| /tasks/Visual Retrieval/embedding_adapter/concat | accuracy | 103 | 19.42 |

来源：[artifacts/diagnostics/muir_concat_images_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_concat_images_20260914/summary.json)

## muir_cross_image_route_audit_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /native | accuracy | 84 | 83.33 |
| /explicit_causal | accuracy | 84 | 83.33 |
| /block_all | accuracy | 84 | 61.90 |
| /block_0_11 | accuracy | 84 | 77.38 |
| /block_12_17 | accuracy | 84 | 66.67 |
| /block_18_35 | accuracy | 84 | 83.33 |

来源：[artifacts/diagnostics/muir_cross_image_route_audit_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_cross_image_route_audit_20260914/summary.json)

## muir_fullprecision_inference_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /bf16 | accuracy | 1000 | 40.10 |
| /fp32 | accuracy | 1000 | 39.90 |

来源：[artifacts/diagnostics/muir_fullprecision_inference_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_fullprecision_inference_20260914/summary.json)

## muir_hf_adapter_inference_parity_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | candidate_accuracy | 1000 | 40.10 |
| root | reference_accuracy | 1000 | 39.90 |

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_20260914/summary.json)

## muir_hf_adapter_inference_parity_eos_only_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | candidate_accuracy | 1000 | 40.10 |
| root | reference_accuracy | 1000 | 39.90 |

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_eos_only_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_eos_only_20260914/summary.json)

## muir_hf_adapter_inference_parity_interleaved_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | candidate_accuracy | 1000 | 36.60 |
| root | reference_accuracy | 1000 | 36.80 |

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_interleaved_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_interleaved_20260914/summary.json)

## muir_hf_adapter_inference_parity_random1000_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | candidate_accuracy | 1000 | 40.70 |
| root | reference_accuracy | 1000 | 40.90 |

来源：[artifacts/diagnostics/muir_hf_adapter_inference_parity_random1000_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_hf_adapter_inference_parity_random1000_20260914/summary.json)

## muir_images_as_video_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, mode=matched_images, task=all] | accuracy | 1000 | 54.10 |
| /results/1 [method=base, mode=as_video, task=all] | accuracy | 1000 | 57.10 |
| /results/2 [method=embedding_adapter, mode=matched_images, task=all] | accuracy | 1000 | 40.40 |
| /results/3 [method=embedding_adapter, mode=as_video, task=all] | accuracy | 1000 | 38.30 |
| /results/4 [method=base, mode=matched_images, task=Action Understanding] | accuracy | 68 | 51.47 |
| /results/5 [method=base, mode=as_video, task=Action Understanding] | accuracy | 68 | 51.47 |
| /results/6 [method=embedding_adapter, mode=matched_images, task=Action Understanding] | accuracy | 68 | 30.88 |
| /results/7 [method=embedding_adapter, mode=as_video, task=Action Understanding] | accuracy | 68 | 35.29 |
| /results/8 [method=base, mode=matched_images, task=Attribute Similarity] | accuracy | 70 | 58.57 |
| /results/9 [method=base, mode=as_video, task=Attribute Similarity] | accuracy | 70 | 57.14 |
| /results/10 [method=embedding_adapter, mode=matched_images, task=Attribute Similarity] | accuracy | 70 | 42.86 |
| /results/11 [method=embedding_adapter, mode=as_video, task=Attribute Similarity] | accuracy | 70 | 52.86 |
| /results/12 [method=base, mode=matched_images, task=Cartoon Understanding] | accuracy | 29 | 48.28 |
| /results/13 [method=base, mode=as_video, task=Cartoon Understanding] | accuracy | 29 | 41.38 |
| /results/14 [method=embedding_adapter, mode=matched_images, task=Cartoon Understanding] | accuracy | 29 | 55.17 |
| /results/15 [method=embedding_adapter, mode=as_video, task=Cartoon Understanding] | accuracy | 29 | 44.83 |
| /results/16 [method=base, mode=matched_images, task=Counting] | accuracy | 95 | 38.95 |
| /results/17 [method=base, mode=as_video, task=Counting] | accuracy | 95 | 35.79 |
| /results/18 [method=embedding_adapter, mode=matched_images, task=Counting] | accuracy | 95 | 32.63 |
| /results/19 [method=embedding_adapter, mode=as_video, task=Counting] | accuracy | 95 | 28.42 |
| /results/20 [method=base, mode=matched_images, task=Diagram Understanding] | accuracy | 163 | 74.85 |
| /results/21 [method=base, mode=as_video, task=Diagram Understanding] | accuracy | 163 | 84.05 |
| /results/22 [method=embedding_adapter, mode=matched_images, task=Diagram Understanding] | accuracy | 163 | 58.28 |
| /results/23 [method=embedding_adapter, mode=as_video, task=Diagram Understanding] | accuracy | 163 | 53.37 |
| /results/24 [method=base, mode=matched_images, task=Difference Spotting] | accuracy | 126 | 41.27 |
| /results/25 [method=base, mode=as_video, task=Difference Spotting] | accuracy | 126 | 35.71 |
| /results/26 [method=embedding_adapter, mode=matched_images, task=Difference Spotting] | accuracy | 126 | 30.16 |
| /results/27 [method=embedding_adapter, mode=as_video, task=Difference Spotting] | accuracy | 126 | 24.60 |
| /results/28 [method=base, mode=matched_images, task=Geographic Understanding] | accuracy | 35 | 34.29 |
| /results/29 [method=base, mode=as_video, task=Geographic Understanding] | accuracy | 35 | 31.43 |
| /results/30 [method=embedding_adapter, mode=matched_images, task=Geographic Understanding] | accuracy | 35 | 20.00 |
| /results/31 [method=embedding_adapter, mode=as_video, task=Geographic Understanding] | accuracy | 35 | 14.29 |
| /results/32 [method=base, mode=matched_images, task=Image-Text Matching] | accuracy | 184 | 74.46 |
| /results/33 [method=base, mode=as_video, task=Image-Text Matching] | accuracy | 184 | 79.35 |
| /results/34 [method=embedding_adapter, mode=matched_images, task=Image-Text Matching] | accuracy | 184 | 50.00 |
| /results/35 [method=embedding_adapter, mode=as_video, task=Image-Text Matching] | accuracy | 184 | 50.54 |
| /results/36 [method=base, mode=matched_images, task=Ordering] | accuracy | 27 | 18.52 |
| /results/37 [method=base, mode=as_video, task=Ordering] | accuracy | 27 | 25.93 |
| /results/38 [method=embedding_adapter, mode=matched_images, task=Ordering] | accuracy | 27 | 40.74 |
| /results/39 [method=embedding_adapter, mode=as_video, task=Ordering] | accuracy | 27 | 22.22 |
| /results/40 [method=base, mode=matched_images, task=Scene Understanding] | accuracy | 65 | 53.85 |
| /results/41 [method=base, mode=as_video, task=Scene Understanding] | accuracy | 65 | 66.15 |
| /results/42 [method=embedding_adapter, mode=matched_images, task=Scene Understanding] | accuracy | 65 | 58.46 |
| /results/43 [method=embedding_adapter, mode=as_video, task=Scene Understanding] | accuracy | 65 | 56.92 |
| /results/44 [method=base, mode=matched_images, task=Visual Grounding] | accuracy | 35 | 31.43 |
| /results/45 [method=base, mode=as_video, task=Visual Grounding] | accuracy | 35 | 34.29 |
| /results/46 [method=embedding_adapter, mode=matched_images, task=Visual Grounding] | accuracy | 35 | 28.57 |
| /results/47 [method=embedding_adapter, mode=as_video, task=Visual Grounding] | accuracy | 35 | 20.00 |
| /results/48 [method=base, mode=matched_images, task=Visual Retrieval] | accuracy | 103 | 38.83 |
| /results/49 [method=base, mode=as_video, task=Visual Retrieval] | accuracy | 103 | 47.57 |
| /results/50 [method=embedding_adapter, mode=matched_images, task=Visual Retrieval] | accuracy | 103 | 14.56 |
| /results/51 [method=embedding_adapter, mode=as_video, task=Visual Retrieval] | accuracy | 103 | 15.53 |

来源：[artifacts/diagnostics/muir_images_as_video_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_images_as_video_20260914/summary.json)

## muir_independent_native_memory_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /native | accuracy | 84 | 83.33 |
| /adapter | accuracy | 84 | 38.10 |
| /joint_native_middle | accuracy | 84 | 73.81 |
| /independent_native_middle | accuracy | 84 | 40.48 |
| /joint_native_all | accuracy | 84 | 83.33 |
| /independent_native_all | accuracy | 84 | 47.62 |

来源：[artifacts/diagnostics/muir_independent_native_memory_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_independent_native_memory_20260914/summary.json)

## muir_layer_kv_localization_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /native | accuracy | 84 | 83.33 |
| /adapter | accuracy | 84 | 38.10 |
| /prefix_36 | accuracy | 84 | 83.33 |
| /middle_12_18 | accuracy | 84 | 73.81 |
| /layer_12 | accuracy | 84 | 40.48 |
| /layer_13 | accuracy | 84 | 41.67 |
| /layer_14 | accuracy | 84 | 35.71 |
| /layer_15 | accuracy | 84 | 57.14 |
| /layer_16 | accuracy | 84 | 36.90 |
| /layer_17 | accuracy | 84 | 40.48 |
| /key_12_18 | accuracy | 84 | 46.43 |
| /value_12_18 | accuracy | 84 | 40.48 |

来源：[artifacts/diagnostics/muir_layer_kv_localization_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_layer_kv_localization_20260914/summary.json)

## muir_layer_localization_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /native | accuracy | 84 | 83.33 |
| /adapter | accuracy | 84 | 38.10 |
| /prefix_1 | accuracy | 84 | 39.29 |
| /prefix_4 | accuracy | 84 | 39.29 |
| /prefix_8 | accuracy | 84 | 36.90 |
| /prefix_12 | accuracy | 84 | 40.48 |
| /prefix_18 | accuracy | 84 | 76.19 |
| /prefix_24 | accuracy | 84 | 83.33 |
| /prefix_30 | accuracy | 84 | 83.33 |
| /prefix_36 | accuracy | 84 | 83.33 |
| /suffix_4 | accuracy | 84 | 83.33 |
| /suffix_8 | accuracy | 84 | 82.14 |
| /suffix_12 | accuracy | 84 | 82.14 |
| /suffix_18 | accuracy | 84 | 50.00 |
| /suffix_24 | accuracy | 84 | 38.10 |
| /suffix_30 | accuracy | 84 | 38.10 |

来源：[artifacts/diagnostics/muir_layer_localization_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_layer_localization_20260914/summary.json)

## muir_length_audit_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /base | accuracy_change_points | 未单列 | 0.00 |
| /recurrent_kl | accuracy_change_points | 未单列 | 0.00 |
| /sft | accuracy_change_points | 未单列 | 1.00 |
| /static_kl | accuracy_change_points | 未单列 | 0.00 |

来源：[artifacts/diagnostics/muir_length_audit_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_length_audit_20260914/summary.json)

## muir_matched_base_adapter_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | candidate_accuracy | 1000 | 40.10 |
| root | reference_accuracy | 1000 | 51.20 |

来源：[artifacts/diagnostics/muir_matched_base_adapter_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_matched_base_adapter_20260914/summary.json)

## muir_native_context_factors_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /native | accuracy | 84 | 83.33 |
| /adapter | accuracy | 84 | 38.10 |
| /joint_native_middle | accuracy | 84 | 73.81 |
| /independent_native_middle | accuracy | 84 | 40.48 |
| /joint_native_all | accuracy | 84 | 83.33 |
| /independent_native_all | accuracy | 84 | 47.62 |
| /original_position_middle | accuracy | 84 | 44.05 |
| /original_position_all | accuracy | 84 | 50.00 |
| /original_text_prefix_middle | accuracy | 84 | 40.48 |
| /original_text_prefix_all | accuracy | 84 | 50.00 |

来源：[artifacts/diagnostics/muir_native_context_factors_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_native_context_factors_20260914/summary.json)

## muir_native_image_ids_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /Geographic Understanding/native_prefix_id/base | accuracy | 100 | 57.00 |
| /Geographic Understanding/native_prefix_id/static_kl | accuracy | 100 | 20.00 |
| /Image-Text Matching/native_prefix_id/base | accuracy | 84 | 78.57 |
| /Image-Text Matching/native_prefix_id/static_kl | accuracy | 84 | 38.10 |

来源：[artifacts/diagnostics/muir_native_image_ids_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_native_image_ids_20260914/summary.json)

## muir_permutation_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /Image-Text Matching/rotate_media_fixed_choices/base | accuracy | 84 | 78.57 |
| /Image-Text Matching/rotate_media_fixed_choices/static_kl | accuracy | 84 | 27.38 |
| /Image-Text Matching/rotate_choices_fixed_media/base | accuracy | 84 | 80.95 |
| /Image-Text Matching/rotate_choices_fixed_media/static_kl | accuracy | 84 | 38.10 |

来源：[artifacts/diagnostics/muir_permutation_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_permutation_20260914/summary.json)

## muir_pixel_numbers_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, style=blank, order=original] | accuracy | 132 | 67.42 |
| /results/1 [method=base, style=blank, order=rotate_media_fixed_choices] | accuracy | 132 | 73.48 |
| /results/2 [method=base, style=numbered, order=original] | accuracy | 132 | 81.06 |
| /results/3 [method=base, style=numbered, order=rotate_media_fixed_choices] | accuracy | 132 | 81.06 |
| /results/4 [method=embedding_adapter, style=blank, order=original] | accuracy | 132 | 47.73 |
| /results/5 [method=embedding_adapter, style=blank, order=rotate_media_fixed_choices] | accuracy | 132 | 39.39 |
| /results/6 [method=embedding_adapter, style=numbered, order=original] | accuracy | 132 | 46.97 |
| /results/7 [method=embedding_adapter, style=numbered, order=rotate_media_fixed_choices] | accuracy | 132 | 42.42 |

来源：[artifacts/diagnostics/muir_pixel_numbers_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_pixel_numbers_20260914/summary.json)

## muir_random_matching_permutations_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, layout=original, subset=all] | accuracy | 184 | 75.00 |
| /results/1 [method=base, layout=rotate_choices_fixed_media, subset=all] | accuracy | 184 | 75.00 |
| /results/2 [method=base, layout=rotate_media_fixed_choices, subset=all] | accuracy | 184 | 78.26 |
| /results/3 [method=embedding_adapter, layout=original, subset=all] | accuracy | 184 | 51.09 |
| /results/4 [method=embedding_adapter, layout=rotate_choices_fixed_media, subset=all] | accuracy | 184 | 48.37 |
| /results/5 [method=embedding_adapter, layout=rotate_media_fixed_choices, subset=all] | accuracy | 184 | 50.00 |
| /results/6 [method=embedding_adapter_mixed, layout=original, subset=all] | accuracy | 184 | 45.11 |
| /results/7 [method=embedding_adapter_mixed, layout=rotate_choices_fixed_media, subset=all] | accuracy | 184 | 42.93 |
| /results/8 [method=embedding_adapter_mixed, layout=rotate_media_fixed_choices, subset=all] | accuracy | 184 | 54.89 |
| /results/9 [method=base, layout=original, subset=image_choices] | accuracy | 132 | 67.42 |
| /results/10 [method=base, layout=rotate_choices_fixed_media, subset=image_choices] | accuracy | 132 | 66.67 |
| /results/11 [method=base, layout=rotate_media_fixed_choices, subset=image_choices] | accuracy | 132 | 73.48 |
| /results/12 [method=embedding_adapter, layout=original, subset=image_choices] | accuracy | 132 | 43.18 |
| /results/13 [method=embedding_adapter, layout=rotate_choices_fixed_media, subset=image_choices] | accuracy | 132 | 41.67 |
| /results/14 [method=embedding_adapter, layout=rotate_media_fixed_choices, subset=image_choices] | accuracy | 132 | 40.91 |
| /results/15 [method=embedding_adapter_mixed, layout=original, subset=image_choices] | accuracy | 132 | 40.91 |
| /results/16 [method=embedding_adapter_mixed, layout=rotate_choices_fixed_media, subset=image_choices] | accuracy | 132 | 40.15 |
| /results/17 [method=embedding_adapter_mixed, layout=rotate_media_fixed_choices, subset=image_choices] | accuracy | 132 | 54.55 |
| /results/18 [method=base, layout=original, subset=text_choices] | accuracy | 52 | 94.23 |
| /results/19 [method=base, layout=rotate_choices_fixed_media, subset=text_choices] | accuracy | 52 | 96.15 |
| /results/20 [method=base, layout=rotate_media_fixed_choices, subset=text_choices] | accuracy | 52 | 90.38 |
| /results/21 [method=embedding_adapter, layout=original, subset=text_choices] | accuracy | 52 | 71.15 |
| /results/22 [method=embedding_adapter, layout=rotate_choices_fixed_media, subset=text_choices] | accuracy | 52 | 65.38 |
| /results/23 [method=embedding_adapter, layout=rotate_media_fixed_choices, subset=text_choices] | accuracy | 52 | 73.08 |
| /results/24 [method=embedding_adapter_mixed, layout=original, subset=text_choices] | accuracy | 52 | 55.77 |
| /results/25 [method=embedding_adapter_mixed, layout=rotate_choices_fixed_media, subset=text_choices] | accuracy | 52 | 50.00 |
| /results/26 [method=embedding_adapter_mixed, layout=rotate_media_fixed_choices, subset=text_choices] | accuracy | 52 | 55.77 |

来源：[artifacts/diagnostics/muir_random_matching_permutations_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_random_matching_permutations_20260914/summary.json)

## muir_separator_binding_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | original_accuracy | 1000 | 40.70 |
| root | local_separator_accuracy | 1000 | 40.30 |

来源：[artifacts/diagnostics/muir_separator_binding_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_separator_binding_20260914/summary.json)

## muir_single_image_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /base | forced_binary_accuracy | 未单列 | 92.86 |
| /base | sensitivity | 42 | 85.71 |
| /base | specificity | 210 | 94.29 |
| /base | answerable_top_image_accuracy | 42 | 95.24 |
| /static_kl | forced_binary_accuracy | 未单列 | 85.32 |
| /static_kl | sensitivity | 42 | 97.62 |
| /static_kl | specificity | 210 | 82.86 |
| /static_kl | answerable_top_image_accuracy | 42 | 95.24 |

来源：[artifacts/diagnostics/muir_single_image_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_single_image_20260914/summary.json)

## muir_source_pixels_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /reencoded_jpeg | accuracy | 1000 | 40.10 |
| /source_pixels | accuracy | 1000 | 39.80 |

来源：[artifacts/diagnostics/muir_source_pixels_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_source_pixels_20260914/summary.json)

## muir_target_front_20260914/full

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=base, mode=full] | sensitivity | 441 | 76.71 |
| /results/0 [method=base, mode=full] | specificity | 441 | 95.38 |
| /results/0 [method=base, mode=full] | top_candidate_accuracy | 73 | 90.41 |
| /results/0 [method=base, mode=full] | reject_all_unanswerable | 59 | 93.22 |
| /results/1 [method=embedding_adapter, mode=full] | sensitivity | 441 | 34.25 |
| /results/1 [method=embedding_adapter, mode=full] | specificity | 441 | 83.97 |
| /results/1 [method=embedding_adapter, mode=full] | top_candidate_accuracy | 73 | 36.99 |
| /results/1 [method=embedding_adapter, mode=full] | reject_all_unanswerable | 59 | 93.22 |

来源：[artifacts/diagnostics/muir_target_front_20260914/full/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_target_front_20260914/full/summary.json)

## muir_target_front_20260914/masked

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /results/0 [method=embedding_adapter, mode=masked] | sensitivity | 441 | 42.47 |
| /results/0 [method=embedding_adapter, mode=masked] | specificity | 441 | 92.93 |
| /results/0 [method=embedding_adapter, mode=masked] | top_candidate_accuracy | 73 | 86.30 |
| /results/0 [method=embedding_adapter, mode=masked] | reject_all_unanswerable | 59 | 88.14 |

来源：[artifacts/diagnostics/muir_target_front_20260914/masked/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_target_front_20260914/masked/summary.json)

## muir_target_read_mask_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| root | sensitivity | 441 | 31.51 |
| root | specificity | 441 | 93.75 |
| root | balanced_accuracy | 441 | 62.63 |
| root | top_candidate_accuracy | 73 | 80.82 |
| root | reject_all_unanswerable | 59 | 79.66 |

来源：[artifacts/diagnostics/muir_target_read_mask_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_target_read_mask_20260914/summary.json)

## muir_vision_backend_20260914

| 条件/字段路径 | 指标 | 分母/规模 | 数值 |
|---|---|---|---|
| /sdpa | accuracy | 1000 | 40.10 |
| /flash_attention_2 | accuracy | 1000 | 40.10 |

来源：[artifacts/diagnostics/muir_vision_backend_20260914/summary.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/muir_vision_backend_20260914/summary.json)

## 10. 早期未完成/已替代记录（不纳入比较）

9月12日旧实现：FastV仅部分样本完成，后续实现/协议已修订。没有结果的空记录和smoke不算有效benchmark结果。

| 方法 | 保留比例 | 已测题数 | 当时分数 |
|---|---|---|---|
| base | 100% | 1000 | 49.20 |
| fastv | 20% | 334 | 52.99 |
| fastv | 5% | 328 | 42.07 |

来源：[artifacts/diagnostics/multimodal_baselines_20260912/results.json](/lustre-data/leijingdi/code/vision-kv-inject/artifacts/diagnostics/multimodal_baselines_20260912/results.json)
