# 分轨生成图检测与归因实验矩阵

- 生成时间: `2026-06-12T11:44:59.489532+00:00`
- 任务: `vision_generator_attribution`
- Active 是否保持不变: `True`
- Active 前后: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- 是否下载外部 benchmark: `False`

## 汇报主表

| 轨道 | 定位 | Candidate | Clean n | Clean Macro-F1 | Clean/Positive AUC | Binary AUC | Gen P/R/F1 | Real P/R/F1 | Source Macro-F1 | Source Real FPR | 验收状态 | 主要问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 社交传播鲁棒性 | 鲁棒性证据和 hard-negative 来源，不直接证明生成器归因能力。 | `2114c728-8e3f-4dc7-b411-2f8e532976e2` | 40 | 0.515 | 0.948 | 0.948 | 0.958/0.767/0.852 | 0.562/0.900/0.692 | 0.137 | 0.000 | 未达标 | Source Generated Recall=0.270 未达到 >= 0.900 的验收门槛。 |

## 逐轨细节

### 社交传播鲁棒性

- Profile: `social_propagation_robustness`
- Candidate: `2114c728-8e3f-4dc7-b411-2f8e532976e2`
- 目标: 专门评估截图、压缩、重采样、水印、转发等传播扰动下的真实/生成识别稳定性。
- 模型做法: generated/real 鲁棒性 candidate；优先作为评测与 hard-negative mining 池。
- 标签策略: 传播域与 real-negative_pool 中的真实图保留 real；生成图映射 generated。
- 激活政策: benchmark-only/component candidate；不得直接替换 active。
- 验收状态: `needs_improvement`；建议: `benchmark_or_hard_negative_only`；问题: Source Generated Recall=0.270 未达到 >= 0.900 的验收门槛。
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.800` / `0.515` / `0.948`
- Positive label / positive AUC: `generated` / `0.948`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.515` / `0.948` / `0.767` / `0.100`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.137` / `0.137` / `0.270` / `0.000`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.256` / `0.256` / `0.000`
- Label distribution: `{"generated": 30, "real": 10}`
- Prediction distribution: `{"generated": 24, "real": 16}`

| 验收项 | 当前值 | 门槛 | 状态 |
| --- | ---: | --- | --- |
| Source Real FPR | 0.000 | <= 0.050 | 达标 |
| Source Generated Recall | 0.270 | >= 0.900 | 未达标 |

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.958 | 0.767 | 0.852 | 30 |
| real | 0.562 | 0.900 | 0.692 | 10 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## 解释口径

- `真实/生成鲁棒初筛` 主要看真实图误报率和 generated recall，不用多分类归因 Macro-F1 来硬评。
- `GPT-image2 专项识别` 是 one-vs-rest 组件，标签为 GPT-image2、real、other-generated。
- `多来源覆盖生成器归因` 不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。
- `Clean 原图归因上限` 与 `社交传播鲁棒性` 分开报告，用于说明 clean 到平台传播后的性能落差。
- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。
