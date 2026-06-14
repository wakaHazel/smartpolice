# GPT-image2 专项技术审计摘要

更新时间：2026-06-13

## 当前结论

主线继续聚焦 GPT-image2，但当前可支撑的结论应拆成两层：

- Clean/internal GPT-image2 检测很强：最新 candidate `6ff5c9b3-40eb-481a-9bc1-0a192de14ac8` 的 clean GPT-image2 recall `0.962`，precision `1.000`，positive AUC `0.999`。
- 严格跨来源 GPT-image2 归因仍未达标：source-holdout mean GPT-image2 recall `0.002`，Source Macro-F1 `0.275`。不能把 clean 高分写成跨来源泛化能力。

Active 模型保持不变：`e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`。

## 新增 GPT-image2 公共来源

已接入并导入 `LukaDev13/Liminal-Dreamcore-1K`：

- 许可：MIT。
- 标签依据：README 明确说明 1000 张图均由 GPT Image 2 以 2K、medium quality 生成。
- 当前本地可用：`313` 张，其中新增导入 `273` 张，前 40 张来自 smoke import 去重。

当前 hard usable GPT-image2 分布：

| 来源 | 可用样本 |
| --- | ---: |
| `Scam-AI/gpt-image-2` | 738 |
| `LukaDev13/Liminal-Dreamcore-1K` | 313 |
| `Qwen/Qwen-Image-Bench` | 50 |

`JoyCN/ai-generated-ecommerce-images` 暂不导入为 GPT-image2：README 只说明混合使用 Gemini 与 OpenAI gpt-image-2，已检查 `annotations_ai.jsonl` 未发现逐图 generator 字段，不能安全标成 GPT-image2。

## 三轮改进对照

| 版本 | Candidate | Source Macro-F1 | Mean GPT Recall | Generated Recall | Real FPR | Label-covered Macro-F1 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPT 三桶均衡采样 | `fe471e0c-0b48-4c69-8210-f4837ff2d9c6` | 0.275 | 0.002 | 0.555 | 0.056 | 0.618 | 比旧抽样更可解释，但 GPT 来源互留仍几乎失败。 |
| Source-guard 特征过滤 | `7a86f1bf-25f1-4b25-991f-fd46fce664fe` | 0.187 | 0.020 | 0.279 | 0.197 | 0.426 | 单纯删除来源特征伤害整体能力，不作为主方案。 |
| GPT-vs-rest 二元 detector | `6ff5c9b3-40eb-481a-9bc1-0a192de14ac8` | 0.275 | 0.002 | 0.555 | 0.056 | 0.618 | detector 训练成功，但严格留出源概率不过阈值，未改善最终指标。 |

## 关键失败边界

严格留出时，GPT-image2 正样本在 detector 上的概率整体偏低：

- Liminal 留出：median `0.199`，阈值 `0.18` 时 recall `0.569`。
- Scam-AI 留出：median `0.152`，阈值 `0.18` 时 recall `0.342`。
- Qwen 留出：median `0.186`，阈值 `0.18` 时 recall `0.620`，但 other-generated FPR `0.634`。

因此问题不是“样本没导入”或“阈值随便调一下就好”，而是当前轻量特征把不同 GPT-image2 来源之间的共同指纹学得很弱，同时 Qwen 内部的其他 OpenAI/相近生成器会和 GPT-image2 混在一起。

## 下一步技术方向

短期不要再把 GPT-image2 归因写成强结论。更稳的主攻顺序：

1. 先强化 `generated/real` 初筛和社交传播扰动鲁棒性，目标是低真实图误报。
2. GPT-image2 输出改成“疑似 GPT-image2 线索”，仅在 clean/internal 或来源覆盖诊断通过时展示高置信。
3. 继续补充更多独立 GPT-image2 来源，尤其是普通手机照片风格、社交平台转码、截图重存、水印覆盖样本。
4. 下一轮模型不要只用当前手工/树模型特征，优先引入 CLIP/UnivFD 风格 embedding 或冻结视觉编码器特征，再做 source-holdout。

补充：本机已具备 `transformers 4.57.1`、`torch 2.9.0+cpu` 和 `openai/clip-vit-base-patch32` 缓存，但 3 张样本的 CLIP 特征烟测在 CPU 上超过 180 秒未完成。当前阶段不建议把 CLIP 直接塞进全量实验；应单独做离线 embedding 缓存或使用更轻量/预计算的 UnivFD baseline。

相关审计文件：

- `output/audits/gpt_image2_public_liminal_import.json`
- `output/audits/generator-experiment-suite-20260613T064813Z.json`
- `output/audits/generator-experiment-suite-20260613T070205Z.json`
- `output/audits/generator-experiment-suite-20260613T074547Z.json`
- `output/audits/gpt_image2_detector_source_probability_diagnostic.json`
