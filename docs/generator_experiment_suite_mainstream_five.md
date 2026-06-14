# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-12T15:41:38.839340+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Source Macro-F1 | Label-covered Macro-F1 | Source Generated Recall | Source Real FPR | Clean sanity Macro-F1 | Unknown rate | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 五类主流生成器归因 | 第二层主流来源线索，是后续归因汇报的主轨；低置信或非五类输出 unknown。 | `04e9e534-646d-4ab7-8753-631b2bbbef05` | 0.217 | 0.273 | 0.356 | 0.101 | 1.000 | 0.167 | 未达标 | Mainstream Macro-F1=0.273 未达到 >= 0.350 的验收门槛。 |

## 逐轨细节

### 五类主流生成器归因

- Profile: `mainstream_five_attribution`
- Candidate: `04e9e534-646d-4ab7-8753-631b2bbbef05`
- 目标: 把归因范围收束到 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney 五个主流来源，降低长尾小类和来源耦合带来的噪声。
- 模型做法: open-set 多分类归因；只强归因五个主流来源，real 保留，其他生成器统一 unknown/other。
- 标签策略: GPT-image2、nano-banana、seedream-4、stable-diffusion 系列、midjourney 保留；sd21/sd3/sdxl 合并到 stable-diffusion；其他生成器映射 unknown。
- 激活政策: 只保存 component candidate；不自动替换 active。
- 验收状态: `needs_improvement`；建议: `needs_stronger_mainstream_five_sources`；问题: Mainstream Macro-F1=0.273 未达到 >= 0.350 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `1.000` / `1.000` / `1.000`
- Positive label / positive AUC: `macro-only` / `-`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.667` / `1.000` / `1.000` / `0.000`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.217` / `0.332` / `0.356` / `0.101`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.273` / `0.386` / `0.000`
- Label distribution: `{"gpt-image2": 15, "midjourney": 30, "nano-banana": 15, "real": 45, "seedream-4": 15, "stable-diffusion": 30, "unknown": 30}`
- Prediction distribution: `{"gpt-image2": 15, "midjourney": 30, "nano-banana": 15, "real": 45, "seedream-4": 15, "stable-diffusion": 30, "unknown": 30}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Mainstream Macro-F1 | 0.273 | >= 0.350 | 未达标 |
| Source Macro-F1 | 0.217 | >= 0.250 | 未达标 |
| Real FPR | 0.101 | <= 0.200 | 达标 |
| Unknown rate | 0.167 | 仅报告 | 仅报告 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 19 | 5 | 0.263 |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 5 | 2 | 0.400 |
| 真实图误报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 18 | 2 | 0.111 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 477 | 229 | 0.480 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 135 | 69 | 0.511 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 64 | 0.496 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| gpt-image2 | 1.000 | 1.000 | 1.000 | 15 |
| midjourney | 1.000 | 1.000 | 1.000 | 30 |
| nano-banana | 1.000 | 1.000 | 1.000 | 15 |
| real | 1.000 | 1.000 | 1.000 | 45 |
| seedream-4 | 1.000 | 1.000 | 1.000 | 15 |
| stable-diffusion | 1.000 | 1.000 | 1.000 | 30 |
| unknown | 1.000 | 1.000 | 1.000 | 30 |


## 解释口径

- 主表优先看 source-holdout 和 label-covered 指标；Clean sanity 只说明训练视图是否自洽，不代表跨来源泛化满分。
- `五类主流生成器归因` 只强归因 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney；DALL-E、Flux、Imagen、Firefly 等先退到 unknown/other。
- 多生成器归因不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- 真实/生成初筛、GPT-image2 专项和五类主流归因仍是两层可信输出：先低误报筛生成，再给来源线索。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
