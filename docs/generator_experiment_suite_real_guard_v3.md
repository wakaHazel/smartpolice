# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-13T13:05:21.197899+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Source Macro-F1 | Label-covered Macro-F1 | Source Generated Recall | Source Real FPR | Clean sanity Macro-F1 | Unknown rate | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 真实/生成鲁棒初筛 | 两层可信输出的第一层，适合作为低误报初筛组件。 | `15870fd6-c242-4f01-8013-adb52d81e800` | 0.258 | 0.310 | 0.629 | 0.143 | 0.617 | 0.000 | 未达标 | Source Real FPR=0.143 未达到 <= 0.100 的验收门槛。 |

## 逐轨细节

### 真实/生成鲁棒初筛

- Profile: `binary_generated_gate`
- Candidate: `15870fd6-c242-4f01-8013-adb52d81e800`
- 目标: 先判断真实图 vs 疑似生成图，优先压低真实图误报，再把疑似生成图交给后续归因。
- 模型做法: generated/real 二分类 gate；使用 real-FPR-first 阈值校准和 source-balanced sample weights。
- 标签策略: 所有非 real 生成器合并为 generated；real-negative 与真实来源保留为 real。
- 激活政策: 只保存 component candidate；不得通过本 profile 直接替换 active。
- 验收状态: `needs_improvement`；建议: `needs_lower_real_fpr_or_higher_generated_recall`；问题: Source Real FPR=0.143 未达到 <= 0.100 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.925` / `0.617` / `0.967`
- Positive label / positive AUC: `generated` / `0.967`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.617` / `0.967` / `0.897` / `0.049`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.258` / `0.258` / `0.629` / `0.143`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.310` / `0.310` / `0.000`
- Label distribution: `{"generated": 39, "real": 41}`
- Prediction distribution: `{"generated": 37, "real": 43}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Source Real FPR | 0.143 | <= 0.100 | 未达标 |
| Source Generated Recall | 0.629 | >= 0.900 | 未达标 |
| Source Macro-F1 | 0.258 | >= 0.550 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 4 | 1 | 0.250 |
| 真实图误报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 5 | 1 | 0.200 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 114 | 75 | 0.658 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 92 | 67 | 0.728 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 78 | 55 | 0.705 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.946 | 0.897 | 0.921 | 39 |
| real | 0.907 | 0.951 | 0.929 | 41 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## 解释口径

- 主表优先看 source-holdout 和 label-covered 指标；Clean sanity 只说明训练视图是否自洽，不代表跨来源泛化满分。
- `五类主流生成器归因` 只强归因 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney；DALL-E、Flux、Imagen、Firefly 等先退到 unknown/other。
- 多生成器归因不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- 真实/生成初筛、GPT-image2 专项和五类主流归因仍是两层可信输出：先低误报筛生成，再给来源线索。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
