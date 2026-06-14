# AI 生成图像取证研判与警务证据链系统

智警杯警用大模型作品赛半决赛原型。当前技术主线收束为“面向社交平台传播扰动的 GPT-image-2 生成图像鲁棒检测与视觉取证”：研究在元数据失效时，如何用视觉语义、频域、压缩痕迹和平台传播扰动特征识别疑似 GPT-image-2 图像，并把概率线索、hash、审计 ID、证据链和报告草稿接入警务辅助研判流程。多生成器归因保留为辅助线索，不作为主突破口。

## 运行

后端：

```powershell
cd D:\smartpolice\backend
$env:PYTHONPATH='D:\smartpolice\backend'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd D:\smartpolice\frontend
npm install
npm run dev -- --port 5173
```

访问：http://127.0.0.1:5173

## 验证

```powershell
$env:PYTHONPATH='D:\smartpolice\backend'
python -m pytest D:\smartpolice\backend\tests -q
cd D:\smartpolice\frontend
npm run build
```

## 本地视觉归因训练

内置四个方向案例只用于训练完成后的展示和评测，不进入训练集。当前 active 生成模型归因头为 `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`，训练池为 `4691` 张外部图片样本，训练 split 扰动增强为 `1200` 条临时特征。

- 外部导入数据集：通过 `/training/datasets/import` 导入 CSV、JSON、JSONL，图片类任务需要本地 `image_root` 和图片路径列。
- 视觉归因训练：`/training/vision/run` 支持 candidate、显式 activate 和门控激活。
- 鲁棒评估：`/training/vision/robustness-run` 输出 clean、JPEG、截图转存、裁剪、水印等分项结果。
- 比赛摘要：`/training/vision/competition-summary` 返回 active 模型、训练样本、增强协议、核心指标和模型边界。

材料表达必须保持边界：GPT-image-2 输出只作为“疑似来源线索”，不替代 C2PA、水印、平台元数据、发布链路和人工核验。

## 真实平台传播评测集

已构建 60 组 clean/平台回收配对样本：30 张 GPT-image2、30 张 real。平台回收条件包括微博下载、微博页面截图、小红书创作者后台返回图；小红书全量页面截图目前没有可靠回收入口，保持 unavailable，不用下载图伪造成截图。

复测当前 active 在真实平台传播条件下的表现：

```powershell
python D:\smartpolice\tools\run_platform_transcode_eval.py
```

生成可导入清单和 candidate-only 训练/评测 payload：

```powershell
python D:\smartpolice\tools\prepare_platform_transcode_manifest.py
```

关键输出：

- `D:\smartpolice\output\audits\platform_transcode_eval_latest.md`
- `D:\smartpolice\platform_eval\returned\SmartPolice_real-platform-transcode-60_manifest.jsonl`
- `D:\smartpolice\platform_eval\returned\SmartPolice_real-platform-transcode-60_import_payload.json`

该数据集用于真实传播扰动评测、candidate-only hard-positive/hard-negative 实验和失败边界披露；不自动替换 active 模型，不声称覆盖全网平台规则。

## 半决赛指标复现

只读查看当前 active 归因头、训练池、增强协议和核心指标：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/competition-summary?task_type=vision_generator_attribution"
```

预期核心输出包括：active 模型 `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`、训练池 `4691`、扰动增强 `1200`、clean validation Accuracy `0.708`、Macro-F1 `0.503`、GPT-image-2 Recall `0.915`。最新 full baseline 报告见 `D:\smartpolice\docs\benchmark_results.md`：active 鲁棒平均 Macro-F1 `0.655`，candidate `20ba33df-c5e0-49d5-8f02-8a430aad90ab` 未自动激活；修正 source-balanced 抽样和训练权重后，strict source-holdout 均值 Macro-F1 为 `0.124`，已见类别子集 Macro-F1 为 `0.139`，strict real false positive rate 为 `0.242`。按公开 baseline 更常见的标签覆盖 source-stratified 诊断，385 条 holdout 上归因 Macro-F1 为 `0.354`、binary Macro-F1 为 `0.464`、generated recall 为 `0.932`、真实图误报率为 `0.267`；该诊断更适合横向泛化叙事，但仍不能替代更大规模外部盲测。

生成图归因现在按“分层 candidate 矩阵”拆分实验，而不是只看单一大杂烩模型：`binary_generated_gate` 评估真实/生成初筛，`gpt_image2_ovr` 评估 GPT-image2 one-vs-rest，`multi_generator_label_covered` 只对跨来源覆盖类别做强归因，`clean_origin_attribution` 与 `social_propagation_robustness` 分别记录 clean 上限和社交传播鲁棒性。实验套件报告见 `D:\smartpolice\docs\generator_experiment_suite.md`，所有分轨模型默认只保存为 candidate，不自动替换 active。

泛化能力评估借鉴公开 benchmark 的协议，而不是借用 leaderboard 分数：GenImage 对应跨生成器与降质图像，AIGIBench 对应外部盲测和来源留出，SIDA/SID-Set 对应社交媒体域，RRDataset/ITW-SM 对应真实传播链、平台重采样和重拍。当前已落地为 source-holdout、六条件扰动鲁棒矩阵和 candidate 不自动激活的门控流程。

现场复测 clean/JPEG/截图/裁剪/水印分项鲁棒性：

```powershell
$body = @{
  task_type = "vision_generator_attribution"
  limit = 120
  conditions = @("clean", "jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark")
} | ConvertTo-Json
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/robustness-run" -Method Post -ContentType "application/json" -Body $body
```

跨来源留出和特征消融用于披露局限，不作为单独夸大的成绩：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/source-holdout-run" -Method Post -ContentType "application/json" -Body '{"task_type":"vision_generator_attribution","holdout_key":"dataset_source","sample_limit":1000,"max_holdout_groups":12}'
Invoke-RestMethod "http://127.0.0.1:8000/training/vision/feature-ablation-run" -Method Post -ContentType "application/json" -Body '{"task_type":"vision_generator_attribution","limit":120}'
```

## 半决赛材料

- 正式项目报告 Word：`D:\smartpolice\output\docx\semifinal_document.docx`
- 兼容文件名：`D:\smartpolice\output\docx\AIGC公共安全谣言治理智能研判系统项目报告.docx`
- PDF作品文档：`D:\smartpolice\output\pdf\semifinal_document.pdf`
- 5分钟演示脚本：`D:\smartpolice\docs\semifinal_video_script.md`
- 调研参考资料：`D:\smartpolice\docs\references.md`

## 演示案例

- 涉警公信力谣言
- 灾害险情谣言
- 群体对立煽动型谣言
- 低风险误传
