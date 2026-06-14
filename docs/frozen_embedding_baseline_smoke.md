# 冻结视觉基础模型特征基线

- 生成时间: `2026-06-13T12:36:22.499941+00:00`
- Profile: `gpt_image2_ovr`
- 状态: `skipped`
- 可用样本: `2` / `12`
- 缓存命中/缺失: `0` / `2`
- 特征版本: `local-clip-image-text-v1:openai/clip-vit-base-patch32:24`

| 指标 | Clean/随机留出 | Source-holdout |
| --- | ---: | ---: |
| Macro-F1 | - | - |
| Binary Macro-F1 | - | - |
| Generated Recall | - | - |
| Real FPR | - | - |
| Macro OvR AUC | - | - |
| GPT-image2 AUC | - | - |

## 解释

- 这是冻结视觉基础模型路线的基线，不训练 CLIP/ViT 本体，只训练轻量分类头。
- 当前默认优先使用缓存；若缓存不足，结果会标记 skipped，避免拖慢主训练。
- 该结果用于和现有手工取证特征做对照，不能单独替代 source-holdout 和扰动评测。

- 跳过原因: 冻结 embedding 缓存样本不足；可用 --allow-extract 先做小样本预热，或后续离线批量缓存。
