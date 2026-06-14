# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-12T12:20:13.172060+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Clean n | Clean Macro-F1 | Clean/Positive AUC | Binary AUC | Gen P/R/F1 | Real P/R/F1 | Source Macro-F1 | Source Real FPR | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 真实/生成鲁棒初筛 | 两层可信输出的第一层，适合作为低误报初筛组件。 | `a249f85b-7497-447c-8659-762c37586465` | 40 | 0.615 | 0.979 | 0.979 | 0.957/0.917/0.936 | 0.882/0.938/0.909 | 0.319 | 0.200 | 未达标 | Source Real FPR=0.200 未达到 <= 0.100 的验收门槛。 |

## 逐轨细节

### 真实/生成鲁棒初筛

- Profile: `binary_generated_gate`
- Candidate: `a249f85b-7497-447c-8659-762c37586465`
- 目标: 先判断真实图 vs 疑似生成图，优先压低真实图误报，再把疑似生成图交给后续归因。
- 模型做法: generated/real 二分类 gate；使用 real-FPR-first 阈值校准和 source-balanced sample weights。
- 标签策略: 所有非 real 生成器合并为 generated；real-negative 与真实来源保留为 real。
- 激活政策: 只保存 component candidate；不得通过本 profile 直接替换 active。
- 验收状态: `needs_improvement`；建议: `needs_lower_real_fpr_or_higher_generated_recall`；问题: Source Real FPR=0.200 未达到 <= 0.100 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.925` / `0.615` / `0.979`
- Positive label / positive AUC: `generated` / `0.979`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.615` / `0.979` / `0.917` / `0.062`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.319` / `0.319` / `0.681` / `0.200`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.328` / `0.328` / `0.333`
- Label distribution: `{"generated": 24, "real": 16}`
- Prediction distribution: `{"generated": 23, "real": 17}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Source Real FPR | 0.200 | <= 0.100 | 未达标 |
| Source Generated Recall | 0.681 | >= 0.900 | 未达标 |
| Source Macro-F1 | 0.319 | >= 0.550 | 未达标 |

| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |
| --- | --- | ---: | ---: | ---: |
| 真实图误报 | Robo531/ai-detector-benchmark-test-data|Robo531/ai-detector-benchmark-test-data:train | 2 | 1 | 0.500 |
| 生成图漏报 | Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 24 | 20 | 0.833 |
| 生成图漏报 | siddharthksah/DeepSafe-benchmark|siddharthksah/DeepSafe-benchmark:train | 30 | 11 | 0.367 |
| 生成图漏报 | Robo531/ai-detector-benchmark-test-data|Robo531/ai-detector-benchmark-test-data:train | 22 | 1 | 0.045 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.957 | 0.917 | 0.936 | 24 |
| real | 0.882 | 0.938 | 0.909 | 16 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## 解释口径

- `真实/生成鲁棒初筛` 主要看真实图误报率和 generated recall，不用多分类归因 Macro-F1 来硬评。
- `GPT-image2 专项识别` 是 one-vs-rest 组件，标签为 GPT-image2、real、other-generated。
- `多来源覆盖生成器归因` 不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- `Clean 原图归因上限` 与 `社交传播鲁棒性` 分开报告，用于说明 clean 到平台传播后的性能落差。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
