# 社交平台与扰动鲁棒性 Benchmark 接入说明

本项目当前主线是 `vision_generator_attribution`：对上传或取证图片给出“疑似生成来源线索”，并在报告中保留 hash、模型版本、审计 ID 和人工复核声明。下面这些外部 baseline 不建议一次性全量混入 active 训练池；更稳的用法是先作为独立 benchmark 或 source-holdout，用来证明系统知道自己的跨来源和传播扰动边界。

## 推荐接入顺序

| 优先级 | Benchmark | 当前用途 | 接入方式 |
| --- | --- | --- | --- |
| P0 | GenImage | 压缩、低分辨率、模糊等 degraded image classification 依据；适合作为扰动鲁棒性 benchmark | 本地下载后用 `tools/prepare_benchmark_manifest.py --format genimage` 抽样生成 JSONL，再通过 `/training/datasets/import` 导入 |
| P0 | AIGIBench | 覆盖更贴近真实场景的 AIGC 图像检测评测；适合做外部盲测和 source-holdout | 本地下载后用 `--format aigibench` 生成 manifest，优先作为 evaluation-only 样本 |
| P1 | SIDA / SID-Set | 社交媒体图像深伪检测，和“社交平台传播”叙事高度贴合 | 数据结构需按发布包确认；先写入研究依据，不直接混入 active |
| P1 | RRDataset / ITW-SM | 平台传播、重采样、重拍等 in-the-wild 扰动设计依据 | 作为鲁棒实验条件设计依据：JPEG、截图转存、裁剪、水印、重采样、重拍 |

## 本地导入流程

示例：将本地 GenImage 目录转为项目可导入 manifest，按类别抽样，避免把百万级图片直接灌进 SQLite。

```powershell
python D:\smartpolice\tools\prepare_benchmark_manifest.py `
  --dataset GenImage `
  --image-root D:\datasets\GenImage `
  --format genimage `
  --source-url https://github.com/GenImage-Dataset/GenImage `
  --max-per-label 200 `
  --output-dir D:\smartpolice\backend\data\benchmark_manifests
```

导入训练池：

```powershell
$payload = Get-Content D:\smartpolice\backend\data\benchmark_manifests\GenImage_import_payload.json -Raw
Invoke-RestMethod "http://127.0.0.1:8000/training/datasets/import" -Method Post -ContentType "application/json" -Body $payload
```

导入后先跑只读摘要和跨来源留出，不要马上替换 active：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/competition-summary?task_type=vision_generator_attribution"
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/source-holdout-run" -Method Post -ContentType "application/json" -Body '{"task_type":"vision_generator_attribution","holdout_key":"dataset_source","max_holdout_groups":8,"sample_limit":800}'
```

## 实验设计口径

- GenImage 适合支撑“扰动后检测能力”叙事，但它主要是 AIGC 检测 benchmark；用于本项目时应说明这是外部扰动鲁棒性评估，不等同于最终生成器来源归因能力。
- AIGIBench、RRDataset、ITW-SM 更适合作为真实场景评测依据：压缩、重采样、平台转发、截图转存、重拍会改变像素统计和元数据，因此模型输出只能是疑似来源线索。
- SID-Set/SIDA 贴合社交媒体深伪传播场景，但任务对象和标签体系可能偏二分类或深伪检测；导入前需要确认许可、标签含义、是否含人脸/身份敏感内容。
- 所有 benchmark 进入训练池前都应记录 `dataset_name`、`source`、`source_url`、`image_path`、`label`、`sha256`，并保留 source-holdout，避免模型学到数据集风格。

## 泛化能力借鉴方式

| 泛化风险 | 可参考 baseline | 本项目落地方式 | 当前状态 |
| --- | --- | --- | --- |
| 生成器迁移 | GenImage cross-generator | 按生成器标签保留多类别 Macro-F1、弱类召回和 unknown 边界 | 已在 clean validation 与 robustness matrix 披露 |
| 图像降质迁移 | GenImage degraded image、AIGIBench robustness | 固定 `clean/jpeg_q85/jpeg_q60/screenshot_resave/center_crop/watermark` 六条件 | 已跑 full baseline，最弱项为 screenshot-resave |
| 来源/数据集迁移 | AIGIBench 外部盲测 | `source-holdout-run` 按 `dataset_source` 留出，candidate 不自动替换 active | full baseline 均值 Macro-F1 `0.023`，作为泛化边界 |
| 社交媒体域迁移 | SIDA/SID-Set | 将社交平台图片视为单独 domain；先审许可证、标签和敏感内容 | 作为实验依据，暂不混入 active |
| 真实传播链迁移 | RRDataset / ITW-SM | 优先补平台重采样、截图转存、重拍、重复上传链样本 | 当前只有近似扰动，下一轮盲测补强 |

答辩时建议把这部分说成“借鉴公开 benchmark 的协议，不借 leaderboard 分数”。这样既能说明我们站在已有研究上，也能避免把未导入的大集包装成已经覆盖的能力。

## 参赛材料可用表达

本项目的鲁棒性实验参考 GenImage、AIGIBench、SID-Set/SIDA、RRDataset 和 ITW-SM 等外部 benchmark，把真实传播中常见的 JPEG 压缩、低分辨率、模糊、平台重采样、截图转存、裁剪、水印覆盖和重拍作为评测条件。项目提交阶段不声称覆盖全网分布，只把这些 benchmark 作为实验设计依据和后续扩展的外部盲测来源。
