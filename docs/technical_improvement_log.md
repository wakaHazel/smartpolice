# GPT-image2 社交传播鲁棒检测技术改进日志

更新时间：2026-06-13

## 本轮已接入的学术方向

| 方向 | 当前落地状态 | 结果与边界 |
| --- | --- | --- |
| 多源 benchmark/source-holdout | 已接入实验套件，默认按 `dataset_source` 留出 | 仍是主要可信指标；clean/internal 高分不能替代该指标。 |
| 扰动增强 | 已在 candidate 训练中支持 JPEG、截图重存、裁剪、水印 | 能作为社交传播扰动证据，但当前 binary/source 召回仍不足。 |
| Open-set unknown | 已新增 `enable_open_set_unknown`、阈值倍率、top-2 margin 拒判 | GPT 专项 clean unknown rate 已从 0 提升到 0.075；跨来源 Macro-F1 未因此解决。 |
| 冻结视觉基础模型特征 | 已升级 `tools/run_frozen_embedding_baseline.py`，支持完整 CLIP image embedding、缓存预热、source-holdout、阈值扫描和可选取证特征融合 | 48 张 GPT-image2 OVR 小样本显示 clean AUC 很高，但 source-holdout 仍塌，说明纯 CLIP 语义 embedding 不能单独解决跨来源。 |
| 检测/归因分层 | 已在三轨报告继续保留 binary gate、GPT OVR、social robustness | 目前更可靠的是第一层低误报初筛和 GPT 疑似线索，不是闭集多模型归因。 |

## 最新 quick 对照

| 实验 | Candidate | Source Macro-F1 | Source Generated Recall | Source Real FPR | Clean Macro-F1 | Unknown rate | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 二元初筛 quick baseline | `efe3ff71-562a-49a5-ab98-48b05d93168a` | 0.271 | 0.543 | 0.000 | 0.548 | 0.000 | 低误报但漏检生成图。 |
| GPT-image2 + open-set | `c1ae8a35-6e76-4dd4-b36b-928046c1141a` | 0.129 | 0.868 | 0.250 | 0.715 | 0.075 | open-set 生效，但跨来源归因仍不稳。 |
| 社交传播鲁棒 quick | `32a8a25d-4bdb-4b6d-8750-149d3c933a91` | 0.237 | 0.565 | 0.000 | 0.644 | 0.000 | 适合作为 hard-negative/鲁棒性实验，不宜直接上线。 |
| 二元初筛 source-guard + real hard-negative 加权 | `71006884-51c7-4868-a81f-f66235e06cab` | 0.306 | 0.631 | 0.000 | 0.496 | 0.000 | source 召回小幅提高，但 clean real FPR 变差，暂不建议激活。 |
| CLIP image embedding + Logistic | `N/A` | 0.000 | 0.624 | 0.333 | 0.687 | 0.000 | Clean/GPT AUC 高，但 source-holdout 完全塌，真实图外部组误报严重。 |
| CLIP image embedding + 取证特征融合 | `N/A` | 0.000 | 0.602 | 0.333 | 0.617 | 0.000 | 融合未修复来源耦合，不能作为当前改进主线。 |
| 二元初筛保守 real-guard v2 | `047d3963-1d0c-4c77-8af6-6df620a0aad7` | 0.260 | 0.624 | 0.071 | 0.591 | 0.000 | 真实图误报降到验收线内，但生成图召回仍不足；适合作为警务保守初筛模式，不建议作为高召回检测器。 |
| 二元初筛 hard-positive v3 消融 | `15870fd6-c242-4f01-8013-adb52d81e800` | 0.258 | 0.629 | 0.143 | 0.617 | 0.000 | 弱生成来源加权只带来 `+0.005` 召回，但 real FPR 超过 0.10，判定为失败消融，默认关闭。 |
| 二元初筛 source 对齐 v4 | `614b54de-c9cf-409d-b1c7-d2da04588c37` | 0.260 | 0.624 | 0.071 | 0.591 | 0.000 | 修复 source 权重对齐后结果与 v2 一致；当前最佳强判定仍是保守 real-guard 策略。 |

Active 模型仍保持：`e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`。

## 最新代码层改进

- 修复 source-balanced sample weights 的 source 对齐逻辑：全量 `source_keys` 按 `train_indices` 取源，折内局部 `source_keys` 按局部位置取源，避免跨来源权重错配。
- 保留 `_is_generated_hard_positive_source` 与 `generated_hard_positive_multiplier` 消融入口，但默认倍率为 `1.0`，不再默认加权弱生成来源。
- 新增 binary gate 复核分层输出：`generated_strong`、`manual_review_generated_signal`、`low_generated_signal`。强判定阈值仍保护真实图；低一档生成概率只作为人工复核线索展示。
- 图像取证接口和前端已展示“生成复核线索”和“生成概率”，配合来源候选排名使用，避免把低置信样本写成确定归因。

## 阈值扫描结论

对二元初筛 candidate `efe3ff71-562a-49a5-ab98-48b05d93168a` 做 600 样本快速阈值扫描：

- 最低 real FPR 区间在阈值 `0.85` 左右，real FPR `0.142`，但 generated recall 只有 `0.550`。
- 召回较高的阈值区间 real FPR 明显偏高，例如阈值 `0.70` 时 generated recall `0.953`，real FPR `0.358`。
- 结论：单纯调阈值无法同时满足低误报和高召回，需要继续改训练数据、特征或 backbone。

## 下一步

1. 二元初筛继续主攻，不自动激活：扩大真实 hard-negative 和社交传播真实图，同时保持 generated 来源均衡。
2. 冻结视觉特征必须离线缓存，不再在训练/评测请求里同步抽 CLIP。
3. GPT-image2 继续作为“疑似来源线索”，保留 open-set unknown；不要写成确定归因。
4. 冻结 CLIP image embedding 已完成小样本反证：不能单独解决跨来源；下一步若继续 embedding，只做更多 backbone 对照或与 hard-negative mining 联动。
5. 保守 real-guard v2/v4 已把 Source Real FPR 压到 `0.071`，但 Source Generated Recall 只有 `0.624`；下一步不要再用简单弱来源加权，应优先做更多真实社交传播正负样本、平台转码仿真和更强 backbone/特征对照。
