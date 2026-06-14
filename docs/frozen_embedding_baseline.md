# 冻结视觉基础模型特征基线

- 生成时间: `2026-06-13T12:37:44.003734+00:00`
- Profile: `gpt_image2_ovr`
- 状态: `completed`
- 可用样本: `46` / `48`
- 缓存命中/缺失: `0` / `46`
- 特征版本: `local-clip-image-text-v1:openai/clip-vit-base-patch32:24`

| 指标 | Clean/随机留出 | Source-holdout |
| --- | ---: | ---: |
| Macro-F1 | 0.687 | 0.000 |
| Binary Macro-F1 | 0.667 | 0.215 |
| Generated Recall | 1.000 | 0.624 |
| Real FPR | 0.000 | 0.333 |
| Macro OvR AUC | 1.000 | - |
| GPT-image2 AUC | 1.000 | - |

## 解释

- 这是冻结视觉基础模型路线的基线，不训练 CLIP/ViT 本体，只训练轻量分类头。
- 当前默认优先使用缓存；若缓存不足，结果会标记 skipped，避免拖慢主训练。
- 该结果用于和现有手工取证特征做对照，不能单独替代 source-holdout 和扰动评测。
