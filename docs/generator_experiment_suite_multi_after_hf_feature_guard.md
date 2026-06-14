# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-12T14:43:53.040082+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Clean n | Clean Macro-F1 | Clean/Positive AUC | Binary AUC | Gen P/R/F1 | Real P/R/F1 | Source Macro-F1 | Source Real FPR | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 多来源覆盖生成器归因 | 第二层多生成器来源线索，负责边界清楚的强归因和 unknown 退让。 | `235752e9-1708-4300-89b4-ef1ff1ef4d7a` | 120 | 0.568 | - | 1.000 | 1.000/0.451/0.622 | 0.243/1.000/0.391 | 0.065 | 0.333 | 未达标 | Label-covered Macro-F1=0.176 未达到 >= 0.300 的验收门槛。 |

## 逐轨细节

### 多来源覆盖生成器归因

- Profile: `multi_generator_label_covered`
- Candidate: `235752e9-1708-4300-89b4-ef1ff1ef4d7a`
- 目标: 只对跨多个 dataset_source 覆盖的生成器做强归因，减少单来源数据污染导致的虚高。
- 模型做法: open-set attribution；跨来源覆盖类别保留，单来源或小样本类别进入 unknown/other 兜底。
- 标签策略: real 保留；生成器类别至少覆盖 2 个 dataset_source 才保留原标签，否则映射 unknown。
- 激活政策: 只保存 component candidate；unknown 输出率必须随结果一起报告。
- 验收状态: `needs_improvement`；建议: `needs_label_covered_sources`；问题: Label-covered Macro-F1=0.176 未达到 >= 0.300 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.525` / `0.568` / `1.000`
- Positive label / positive AUC: `macro-only` / `-`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.338` / `1.000` / `0.451` / `0.000`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.065` / `0.226` / `0.429` / `0.333`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.176` / `0.238` / `0.333`
- Label distribution: `{"dall-e": 6, "dall-e-3": 12, "flux": 12, "gpt-image1": 6, "gpt-image2": 6, "midjourney": 12, "nano-banana": 6, "real": 18, "sd21": 6, "sd3": 6, "sdxl": 10, "seedream-4": 5, "stable-diffusion": 5, "unknown": 10}`
- Prediction distribution: `{"dall-e": 2, "dall-e-3": 7, "flux": 6, "gpt-image1": 7, "gpt-image2": 3, "midjourney": 3, "nano-banana": 4, "real": 74, "sd21": 3, "sd3": 4, "sdxl": 1, "seedream-4": 4, "unknown": 2}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Label-covered Macro-F1 | 0.176 | >= 0.300 | 未达标 |
| Unknown rate | 0.017 | 仅报告 | 仅报告 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | siddharthksah/DeepSafe-benchmark|siddharthksah/DeepSafe-benchmark:train | 2 | 1 | 0.500 |
| 真实图误报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 2 | 1 | 0.500 |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 2 | 1 | 0.500 |
| 生成图漏报 | Qwen/Qwen-Image-Bench|Qwen/Qwen-Image-Bench:test | 72 | 45 | 0.625 |
| 生成图漏报 | siddharthksah/DeepSafe-benchmark|siddharthksah/DeepSafe-benchmark:train | 58 | 32 | 0.552 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 45 | 30 | 0.667 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| dall-e | 1.000 | 0.333 | 0.500 | 6 |
| dall-e-3 | 1.000 | 0.583 | 0.737 | 12 |
| flux | 1.000 | 0.500 | 0.667 | 12 |
| gpt-image1 | 0.857 | 1.000 | 0.923 | 6 |
| gpt-image2 | 1.000 | 0.500 | 0.667 | 6 |
| midjourney | 1.000 | 0.250 | 0.400 | 12 |
| nano-banana | 1.000 | 0.667 | 0.800 | 6 |
| real | 0.243 | 1.000 | 0.391 | 18 |
| sd21 | 1.000 | 0.500 | 0.667 | 6 |
| sd3 | 1.000 | 0.667 | 0.800 | 6 |
| sdxl | 1.000 | 0.100 | 0.182 | 10 |
| seedream-4 | 1.000 | 0.800 | 0.889 | 5 |
| stable-diffusion | 0.000 | 0.000 | 0.000 | 5 |
| unknown | 1.000 | 0.200 | 0.333 | 10 |


## 解释口径

- `真实/生成鲁棒初筛` 主要看真实图误报率和 generated recall，不用多分类归因 Macro-F1 来硬评。
- `GPT-image2 专项识别` 是 one-vs-rest 组件，标签为 GPT-image2、real、other-generated。
- `多来源覆盖生成器归因` 不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- `Clean 原图归因上限` 与 `社交传播鲁棒性` 分开报告，用于说明 clean 到平台传播后的性能落差。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
