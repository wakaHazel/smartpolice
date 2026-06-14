# 多生成器归因数据缺口报告

本报告只审计 `vision_generator_attribution` 当前外部图片池，不训练、不激活模型。

## 结论

- 当前可用图片样本 `5008` 张，生成器标签 `15` 个。
- 多模型归因的主要瓶颈是来源覆盖和来源耦合，不是简单的总图片数。
- `max-per-label`/`max-per-class` 只能作为下载安全阀；GPT-image2 专项不应被 200 张这类冒烟上限限制。
- 多模型归因训练应先补到每个强归因标签至少 `3` 个独立来源，目标约 `300` 张/来源；这是规划目标，不是硬上限。

## 标签覆盖表

| 标签 | 样本数 | 来源数 | 有效来源数 | 最大来源占比 | 状态 | 建议新增来源 | 建议新增样本 | 推荐补源 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| dall-e | 33 | 2 | 1.20 | 90.9% | needs-more-sources | 1 | 300 | - |
| dall-e-3 | 353 | 3 | 1.69 | 74.5% | source-dominated | 0 | 0 | Synthbuster, MS COCOAI / CT2 |
| flux | 365 | 6 | 4.30 | 32.9% | usable | 0 | 0 | B-Free new generators, Qwen/Qwen-Image-Bench, Rapidata/bananamark |
| gpt-image1 | 163 | 2 | 1.90 | 61.4% | needs-more-sources | 1 | 300 | Qwen/Qwen-Image-Bench, DeepSafe-benchmark |
| gpt-image1.5 | 100 | 1 | 1.00 | 100.0% | critical-single-source | 2 | 600 | Qwen/Qwen-Image-Bench, user/local GPT-image1.5 pool |
| gpt-image2 | 838 | 2 | 1.27 | 88.1% | needs-more-sources | 1 | 300 | Scam-AI/gpt-image-2, Qwen/Qwen-Image-Bench, user/local GPT-image2 pool |
| imagegbt | 37 | 1 | 1.00 | 100.0% | critical-single-source | 2 | 600 | - |
| midjourney | 451 | 5 | 2.50 | 58.1% | usable | 0 | 0 | Synthbuster, GenImage, MS COCOAI / CT2 |
| nano-banana | 214 | 3 | 2.66 | 46.7% | usable | 0 | 0 | Qwen/Qwen-Image-Bench, Rapidata/bananamark |
| real | 1060 | 10 | 5.84 | 24.8% | negative-pool | 0 | 0 | - |
| sd21 | 326 | 2 | 1.45 | 80.7% | needs-more-sources | 1 | 300 | Synthbuster, MS COCOAI / CT2 |
| sd3 | 342 | 3 | 1.59 | 76.9% | source-dominated | 0 | 0 | MS COCOAI / CT2 |
| sdxl | 410 | 5 | 2.22 | 64.1% | source-dominated | 0 | 0 | Synthbuster, MS COCOAI / CT2, AIGCDetectBenchmark |
| seedream-4 | 178 | 3 | 2.43 | 56.2% | usable | 0 | 0 | Qwen/Qwen-Image-Bench, Rapidata/bananamark |
| stable-diffusion | 98 | 3 | 2.92 | 40.8% | usable | 0 | 0 | Synthbuster, GenImage |
| unknown | 40 | 2 | 1.60 | 75.0% | needs-more-sources | 1 | 300 | - |

## 主要问题标签

| 标签 | 问题 | Top 来源 |
| --- | --- | --- |
| dall-e | needs-more-sources | TheKernel01/AIGC-Detection-Benchmark:TheKernel01/AIGC-Detection-Benchmark:test = 30<br>marco-willi/synthbuster-plus:marco-willi/synthbuster-plus:train = 3 |
| dall-e-3 | source-dominated | Rajarshi-Roy-research/Defactify_Image_Dataset:Rajarshi-Roy-research/Defactify_Image_Dataset:train = 263<br>siddharthksah/DeepSafe-benchmark:siddharthksah/DeepSafe-benchmark:train = 63<br>Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset:Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset:train_0001 = 27 |
| gpt-image1 | needs-more-sources | Qwen/Qwen-Image-Bench:Qwen/Qwen-Image-Bench:test = 100<br>siddharthksah/DeepSafe-benchmark:siddharthksah/DeepSafe-benchmark:train = 63 |
| gpt-image1.5 | critical-single-source | Qwen/Qwen-Image-Bench:Qwen/Qwen-Image-Bench:test = 100 |
| gpt-image2 | needs-more-sources | Scam-AI/gpt-image-2:Scam-AI/gpt-image-2:train = 738<br>Qwen/Qwen-Image-Bench:Qwen/Qwen-Image-Bench:test = 100 |
| imagegbt | critical-single-source | Robo531/ai-detector-benchmark-test-data:Robo531/ai-detector-benchmark-test-data:train = 37 |
| sd21 | needs-more-sources | Rajarshi-Roy-research/Defactify_Image_Dataset:Rajarshi-Roy-research/Defactify_Image_Dataset:train = 263<br>siddharthksah/DeepSafe-benchmark:siddharthksah/DeepSafe-benchmark:train = 63 |
| sd3 | source-dominated | Rajarshi-Roy-research/Defactify_Image_Dataset:Rajarshi-Roy-research/Defactify_Image_Dataset:train = 263<br>siddharthksah/DeepSafe-benchmark:siddharthksah/DeepSafe-benchmark:train = 63<br>marco-willi/synthbuster-plus:marco-willi/synthbuster-plus:train = 16 |
| sdxl | source-dominated | Rajarshi-Roy-research/Defactify_Image_Dataset:Rajarshi-Roy-research/Defactify_Image_Dataset:train = 263<br>siddharthksah/DeepSafe-benchmark:siddharthksah/DeepSafe-benchmark:train = 63<br>Robo531/ai-detector-benchmark-test-data:Robo531/ai-detector-benchmark-test-data:train = 38<br>TheKernel01/AIGC-Detection-Benchmark:TheKernel01/AIGC-Detection-Benchmark:test = 30 |
| unknown | needs-more-sources | TheKernel01/AIGC-Detection-Benchmark:TheKernel01/AIGC-Detection-Benchmark:test = 30<br>marco-willi/synthbuster-plus:marco-willi/synthbuster-plus:train = 10 |

## 公开数据和 baseline 选择

| 名称 | 适配度 | 用法 | 标签帮助 | 注意事项 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Synthbuster | 高 | 多生成器归因评测和补源；每个生成器约 1K 张，覆盖 DALL-E 2/3、Midjourney v5、SD 1.x/2/XL、Firefly、Glide。 | dall-e-3, midjourney, stable-diffusion/sd21/sdxl, glide/unknown, firefly/unknown | 原始 real RAISE-1K 需单独下载；先作为 source-holdout，再决定是否入训练。 | https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images |
| GenImage | 中 | 大规模跨生成器检测和扰动评测；覆盖 Midjourney、Stable Diffusion、ADM、GLIDE、Wukong、VQDM、BigGAN。 | midjourney, stable-diffusion, adm/unknown, glide/unknown, wukong/unknown, vqdm/unknown, biggan/unknown | 很多类别不是我们当前强归因标签；优先做 detection/source-holdout，不直接把 unknown 类混入强归因。 | https://github.com/GenImage-Dataset/GenImage |
| MS COCOAI / CT2 | 高 | 专门补 SD3、SDXL、SD2.1、DALL-E 3、Midjourney 等多类归因来源，适合缓解来源耦合。 | sd3, sdxl, sd21, dall-e-3, midjourney | 竞赛平台数据需确认下载权限和许可；先导入为 benchmark_role=external_holdout。 | https://codalab.lisn.upsaclay.fr/competitions/20331 |
| AIGCDetectBenchmark | 中 | 复用其统一训练/评测口径和 CNNSpot/FreDect/DIRE/UnivFD/PatchCraft baseline；更适合二分类和鲁棒性对照。 | binary/generated first; generator labels depend on downloaded test split | 公开方法多为 real/fake 检测，不等价于多模型归因成绩。 | https://github.com/Ekko-zn/AIGCDetectBenchmark |
| UnivFD | 中 | 用 CLIP 特征 + 线性/近邻的思路增强跨生成器泛化；适合作为我们 binary gate 或 embedding baseline。 | method baseline, not a labeled attribution dataset | 不能直接解决 generator label 不均衡，但能减少只学压缩/尺寸等数据集伪特征。 | https://github.com/WisconsinAIVision/UniversalFakeDetect |
| Synthbuster baseline | 中 | Fourier-domain artifact 检测；可作为频域鲁棒检测 baseline 和特征设计参考。 | method baseline, binary synthetic detector | 检测 fake 不等于判断具体生成器；适合第一层初筛或辅助特征。 | https://github.com/qbammey/synthbuster |

## 本地 baseline 落地状态

| baseline | 本地路径 | 当前状态 | 可借鉴点 | 不解决的问题 |
| --- | --- | --- | --- | --- |
| AIGCDetectBenchmark | `D:\smartpolice\external_baselines\AIGCDetectBenchmark` | code-local | 统一评测 CNNSpot/FreDect/DIRE/UnivFD/PatchCraft；扰动参数覆盖 blur/jpeg/resize。 | 主要是 real/fake 检测，不是具体生成器归因。 |
| UniversalFakeDetect | `D:\smartpolice\external_baselines\UniversalFakeDetect` | code-local | CLIP ViT embedding + 线性头；适合提升跨生成器 fake/real 泛化，减少尺寸/压缩伪特征。 | 仍需我们的多源标签数据才能做多类归因。 |
| Synthbuster | `D:\smartpolice\external_baselines\synthbuster` | code-local | Fourier-domain artifact 检测和 Zenodo 多生成器数据；适合补 DALL-E/MJ/SD 来源。 | Fourier 检测是二分类辅助，不等于直接判断具体生成器。 |

## 下一步执行顺序

1. 多生成器归因先收束为 `mainstream_five_attribution`：GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney。
2. 优先补五类主流标签的跨 dataset_source 覆盖，暂不继续扩展 DALL-E、Flux、Imagen、Firefly 等长尾归因。
3. GPT-image2、Nano Banana、Seedream 尽量使用全部有效样本，并做来源互留；Stable Diffusion 系列合并评估。
4. 新数据先以 source-holdout/benchmark 角色导入，指标稳定后再进入训练；不把 clean 高分写成跨来源泛化。
5. 现成 baseline 先复用 UnivFD/AIGCDetectBenchmark/Synthbuster 的思路和评测，不把二分类 baseline 分数冒充多类归因。
