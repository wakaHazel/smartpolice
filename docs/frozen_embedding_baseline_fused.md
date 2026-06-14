# 冻结视觉基础模型特征基线

- 生成时间: `2026-06-13T12:42:01.201939+00:00`
- Profile: `gpt_image2_ovr`
- 状态: `completed`
- Feature mode / view: `embedding` / `image`
- 融合取证特征: `True`
- 可用样本: `46` / `48`
- 缓存命中/新抽取/失败: `46` / `0` / `0`
- 特征版本: `local-clip-image-text-v1:openai/clip-vit-base-patch32:24`

| 指标 | Clean/随机留出 | Source-holdout |
| --- | ---: | ---: |
| Macro-F1 | 0.617 | 0.000 |
| Binary Macro-F1 | 0.667 | 0.211 |
| Generated Recall | 1.000 | 0.602 |
| Real FPR | 0.000 | 0.333 |
| Generated AUC | 1.000 | 0.000 |
| Macro OvR AUC | 1.000 | - |
| GPT-image2 AUC | 1.000 | - |

## Source-holdout 弱组

| 来源组 | 标签分布 | 预测分布 | Macro-F1 | Binary-F1 | Generated Recall | Real FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LukaDev13/Liminal-Dreamcore-1K|LukaDev13/Liminal-Dreamcore-1K:train | `{"gpt-image2": 16}` | `{"other-generated": 15, "real": 1}` | 0.000 | 0.323 | 0.938 | 0.000 |
| TheKernel01/AIGC-Detection-Benchmark|TheKernel01/AIGC-Detection-Benchmark:test | `{"real": 15}` | `{"gpt-image2": 10, "other-generated": 5}` | 0.000 | 0.000 | 0.000 | 1.000 |
| VIBE-Benchmark/VIBE-Seedream4.5|VIBE-Benchmark/VIBE-Seedream4.5:train | `{"other-generated": 15}` | `{"gpt-image2": 13, "real": 2}` | 0.000 | 0.310 | 0.867 | 0.000 |

## 阈值诊断

- 推荐阈值: `0.300`；Binary-F1 `0.258`；Generated Recall `0.935`；Real FPR `1.000`。

| Threshold | Binary-F1 | Generated Recall | Real FPR |
| ---: | ---: | ---: | ---: |
| 0.300 | 0.258 | 0.935 | 1.000 |
| 0.400 | 0.258 | 0.935 | 1.000 |
| 0.500 | 0.252 | 0.903 | 1.000 |
| 0.600 | 0.241 | 0.839 | 1.000 |
| 0.650 | 0.235 | 0.806 | 1.000 |
| 0.700 | 0.209 | 0.677 | 1.000 |
| 0.750 | 0.195 | 0.613 | 1.000 |
| 0.800 | 0.180 | 0.548 | 1.000 |
| 0.850 | 0.138 | 0.387 | 1.000 |
| 0.900 | 0.119 | 0.323 | 1.000 |

## 解释

- 这是冻结视觉基础模型路线的基线，不训练 CLIP/ViT 本体，只训练轻量分类头。
- 当前默认优先使用缓存；若缓存不足，结果会标记 skipped，避免拖慢主训练。
- 该结果用于和现有手工取证特征做对照，不能单独替代 source-holdout 和扰动评测。
