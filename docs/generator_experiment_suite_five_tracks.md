# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-12T16:26:32.865175+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Source Macro-F1 | Label-covered Macro-F1 | Source Generated Recall | Source Real FPR | Clean sanity Macro-F1 | Unknown rate | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 真实/生成鲁棒初筛 | 两层可信输出的第一层，适合作为低误报初筛组件。 | `489527cd-5cc5-4abb-a0b5-52080dd63ffe` | 0.331 | 0.559 | 0.391 | 0.179 | 0.595 | 0.000 | 未达标 | Source Real FPR=0.179 未达到 <= 0.100 的验收门槛。 |
| GPT-image2 专项识别 | 第二层来源线索组件，只输出疑似 GPT-image2，不做执法定论。 | `03f17119-309f-4590-a870-652f90d2077d` | 0.228 | 0.402 | 0.472 | 0.282 | 0.707 | 0.000 | 未达标 | Source Macro-F1=0.228 未达到 >= 0.450 的验收门槛。 |
| 五类主流生成器归因 | 第二层主流来源线索，是后续归因汇报的主轨；低置信或非五类输出 unknown。 | `62518979-ab88-4757-9a5c-f5f95bb15246` | 0.214 | 0.311 | 0.347 | 0.103 | 0.986 | 0.157 | 未达标 | Mainstream Macro-F1=0.311 未达到 >= 0.350 的验收门槛。 |
| Clean 原图归因上限 | 上限参照和误差分解，不承担上线组件角色。 | `edddcf6a-6c5a-4243-b833-887c0438996b` | 0.040 | 0.204 | 0.435 | 0.206 | 0.838 | 0.043 | 未达标 | Clean Macro-F1=0.838 未达到 >= 0.850 的验收门槛。 |
| 社交传播鲁棒性 | 鲁棒性证据和 hard-negative 来源，不直接证明生成器归因能力。 | `446a6b9c-f72b-4b54-a66c-0ccee9dc4b5e` | 0.268 | 0.498 | 0.289 | 0.091 | 0.667 | 0.000 | 未达标 | Source Real FPR=0.091 未达到 <= 0.050 的验收门槛。 |

## 逐轨细节

### 真实/生成鲁棒初筛

- Profile: `binary_generated_gate`
- Candidate: `489527cd-5cc5-4abb-a0b5-52080dd63ffe`
- 目标: 先判断真实图 vs 疑似生成图，优先压低真实图误报，再把疑似生成图交给后续归因。
- 模型做法: generated/real 二分类 gate；使用 real-FPR-first 阈值校准和 source-balanced sample weights。
- 标签策略: 所有非 real 生成器合并为 generated；real-negative 与真实来源保留为 real。
- 激活政策: 只保存 component candidate；不得通过本 profile 直接替换 active。
- 验收状态: `needs_improvement`；建议: `needs_lower_real_fpr_or_higher_generated_recall`；问题: Source Real FPR=0.179 未达到 <= 0.100 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.893` / `0.595` / `0.974`
- Positive label / positive AUC: `generated` / `0.974`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.595` / `0.974` / `0.986` / `0.197`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.331` / `0.331` / `0.391` / `0.179`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.559` / `0.559` / `0.095`
- Label distribution: `{"generated": 69, "real": 71}`
- Prediction distribution: `{"generated": 82, "real": 58}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Source Real FPR | 0.179 | <= 0.100 | 未达标 |
| Source Generated Recall | 0.391 | >= 0.900 | 未达标 |
| Source Macro-F1 | 0.331 | >= 0.550 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 15 | 7 | 0.467 |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 5 | 4 | 0.800 |
| 真实图误报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 15 | 2 | 0.133 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 352 | 157 | 0.446 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 135 | 66 | 0.489 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 55 | 0.426 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.829 | 0.986 | 0.901 | 69 |
| real | 0.983 | 0.803 | 0.884 | 71 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |

### GPT-image2 专项识别

- Profile: `gpt_image2_ovr`
- Candidate: `03f17119-309f-4590-a870-652f90d2077d`
- 目标: 把 GPT-image2 从 real 和 other-generated 中单独识别出来，解决多分类硬顶导致的召回塌陷。
- 模型做法: one-vs-rest 三分类：gpt-image2 / other-generated / real；后续补 Qwen 与 Scam-AI 来源互留评估。
- 标签策略: GPT-image2 为正类；真实图为 real；其他生成器合并为 other-generated。
- 激活政策: 只保存 component candidate；通过门槛后建议进入组合研判，不直接替换 active。
- 验收状态: `needs_improvement`；建议: `needs_more_cross_source_gpt_image2`；问题: Source Macro-F1=0.228 未达到 >= 0.450 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.929` / `0.707` / `0.982`
- Positive label / positive AUC: `gpt-image2` / `1.000`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.619` / `0.981` / `1.000` / `0.141`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.228` / `0.320` / `0.472` / `0.282`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.402` / `0.537` / `0.190`
- Label distribution: `{"gpt-image2": 23, "other-generated": 46, "real": 71}`
- Prediction distribution: `{"gpt-image2": 23, "other-generated": 56, "real": 61}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| GPT-image2 Recall | 1.000 | >= 0.600 | 达标 |
| GPT-image2 Precision | 1.000 | >= 0.700 | 达标 |
| Source Macro-F1 | 0.228 | >= 0.450 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 15 | 9 | 0.600 |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 5 | 5 | 1.000 |
| 真实图误报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 15 | 4 | 0.267 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 352 | 123 | 0.349 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 135 | 56 | 0.415 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 38 | 0.295 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| gpt-image2 | 1.000 | 1.000 | 1.000 | 23 |
| other-generated | 0.821 | 1.000 | 0.902 | 46 |
| real | 1.000 | 0.859 | 0.924 | 71 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |

### 五类主流生成器归因

- Profile: `mainstream_five_attribution`
- Candidate: `62518979-ab88-4757-9a5c-f5f95bb15246`
- 目标: 把归因范围收束到 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney 五个主流来源，降低长尾小类和来源耦合带来的噪声。
- 模型做法: open-set 多分类归因；只强归因五个主流来源，real 保留，其他生成器统一 unknown/other。
- 标签策略: GPT-image2、nano-banana、seedream-4、stable-diffusion 系列、midjourney 保留；sd21/sd3/sdxl 合并到 stable-diffusion；其他生成器映射 unknown。
- 激活政策: 只保存 component candidate；不自动替换 active。
- 验收状态: `needs_improvement`；建议: `needs_stronger_mainstream_five_sources`；问题: Mainstream Macro-F1=0.311 未达到 >= 0.350 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.986` / `0.986` / `0.999`
- Positive label / positive AUC: `macro-only` / `-`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.660` / `0.998` / `1.000` / `0.028`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.214` / `0.325` / `0.347` / `0.103`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.311` / `0.370` / `0.000`
- Label distribution: `{"gpt-image2": 12, "midjourney": 24, "nano-banana": 12, "real": 36, "seedream-4": 12, "stable-diffusion": 22, "unknown": 22}`
- Prediction distribution: `{"gpt-image2": 12, "midjourney": 24, "nano-banana": 13, "real": 35, "seedream-4": 12, "stable-diffusion": 22, "unknown": 22}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Mainstream Macro-F1 | 0.311 | >= 0.350 | 未达标 |
| Source Macro-F1 | 0.214 | >= 0.250 | 未达标 |
| Real FPR | 0.103 | <= 0.200 | 达标 |
| Unknown rate | 0.157 | 仅报告 | 仅报告 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 15 | 5 | 0.333 |
| 真实图误报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 15 | 2 | 0.133 |
| 真实图误报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 5 | 1 | 0.200 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 352 | 174 | 0.494 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 135 | 72 | 0.533 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 66 | 0.512 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| gpt-image2 | 1.000 | 1.000 | 1.000 | 12 |
| midjourney | 1.000 | 1.000 | 1.000 | 24 |
| nano-banana | 0.923 | 1.000 | 0.960 | 12 |
| real | 1.000 | 0.972 | 0.986 | 36 |
| seedream-4 | 1.000 | 1.000 | 1.000 | 12 |
| stable-diffusion | 1.000 | 1.000 | 1.000 | 22 |
| unknown | 0.955 | 0.955 | 0.955 | 22 |

### Clean 原图归因上限

- Profile: `clean_origin_attribution`
- Candidate: `edddcf6a-6c5a-4243-b833-887c0438996b`
- 目标: 测 clean/origin 图像条件下的归因上限，用来量化平台传播前后的性能落差。
- 模型做法: clean/origin 上限实验；不把 clean 高分当作社交平台泛化能力。
- 标签策略: 保留 clean_origin、多生成器 benchmark、GPT-image2 focus 样本，排除传播域样本。
- 激活政策: benchmark-only；不得直接激活。
- 验收状态: `needs_improvement`；建议: `upper_bound_only`；问题: Clean Macro-F1=0.838 未达到 >= 0.850 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.793` / `0.838` / `1.000`
- Positive label / positive AUC: `macro-only` / `-`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.464` / `1.000` / `0.766` / `0.000`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.040` / `0.205` / `0.435` / `0.206`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.204` / `0.195` / `0.250`
- Label distribution: `{"dall-e": 8, "dall-e-3": 8, "flux": 16, "gpt-image2": 8, "midjourney": 16, "nano-banana": 8, "real": 16, "sd21": 8, "sd3": 16, "sdxl": 15, "seedream-4": 7, "stable-diffusion": 7, "unknown": 7}`
- Prediction distribution: `{"dall-e": 8, "dall-e-3": 8, "flux": 10, "gpt-image2": 8, "midjourney": 9, "nano-banana": 8, "real": 45, "sd21": 5, "sd3": 12, "sdxl": 12, "seedream-4": 7, "stable-diffusion": 2, "unknown": 6}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Clean Macro-F1 | 0.838 | >= 0.850 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 14 | 5 | 0.357 |
| 真实图误报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 15 | 2 | 0.133 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 352 | 303 | 0.861 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 123 | 0.953 |
| 生成图漏报 | Rapidata/bananamark-dataset|Rapidata/bananamark-dataset:train | 195 | 85 | 0.436 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| dall-e | 1.000 | 1.000 | 1.000 | 8 |
| dall-e-3 | 1.000 | 1.000 | 1.000 | 8 |
| flux | 1.000 | 0.625 | 0.769 | 16 |
| gpt-image2 | 1.000 | 1.000 | 1.000 | 8 |
| midjourney | 1.000 | 0.562 | 0.720 | 16 |
| nano-banana | 1.000 | 1.000 | 1.000 | 8 |
| real | 0.356 | 1.000 | 0.525 | 16 |
| sd21 | 1.000 | 0.625 | 0.769 | 8 |
| sd3 | 1.000 | 0.750 | 0.857 | 16 |
| sdxl | 1.000 | 0.800 | 0.889 | 15 |
| seedream-4 | 1.000 | 1.000 | 1.000 | 7 |
| stable-diffusion | 1.000 | 0.286 | 0.444 | 7 |
| unknown | 1.000 | 0.857 | 0.923 | 7 |

### 社交传播鲁棒性

- Profile: `social_propagation_robustness`
- Candidate: `446a6b9c-f72b-4b54-a66c-0ccee9dc4b5e`
- 目标: 专门评估截图、压缩、重采样、水印、转发等传播扰动下的真实/生成识别稳定性。
- 模型做法: generated/real 鲁棒性 candidate；优先作为评测与 hard-negative mining 池。
- 标签策略: 传播域与 real-negative_pool 中的真实图保留 real；生成图映射 generated。
- 激活政策: benchmark-only/component candidate；不得直接替换 active。
- 验收状态: `needs_improvement`；建议: `benchmark_or_hard_negative_only`；问题: Source Real FPR=0.091 未达到 <= 0.050 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `1.000` / `0.667` / `1.000`
- Positive label / positive AUC: `generated` / `1.000`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.667` / `1.000` / `1.000` / `0.000`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.268` / `0.268` / `0.289` / `0.091`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.498` / `0.498` / `0.048`
- Label distribution: `{"generated": 105, "real": 35}`
- Prediction distribution: `{"generated": 105, "real": 35}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Source Real FPR | 0.091 | <= 0.050 | 未达标 |
| Source Generated Recall | 0.289 | >= 0.900 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 15 | 3 | 0.200 |
| 真实图误报 | AdoCleanCode/Fakeddit-FalseConnection-Fusion-LocalImages::real-negative-pool|HuggingFace dataset + downloaded public image URLs -> generator_attribution_real_negative | 15 | 1 | 0.067 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 352 | 136 | 0.386 |
| 生成图漏报 | marco-willi/synthbuster-plus|marco-willi/synthbuster-plus:train | 129 | 90 | 0.698 |
| 生成图漏报 | TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | 135 | 76 | 0.563 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 1.000 | 1.000 | 1.000 | 105 |
| real | 1.000 | 1.000 | 1.000 | 35 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## 解释口径

- 主表优先看 source-holdout 和 label-covered 指标；Clean sanity 只说明训练视图是否自洽，不代表跨来源泛化满分。
- `五类主流生成器归因` 只强归因 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney；DALL-E、Flux、Imagen、Firefly 等先退到 unknown/other。
- 多生成器归因不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- 真实/生成初筛、GPT-image2 专项和五类主流归因仍是两层可信输出：先低误报筛生成，再给来源线索。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
