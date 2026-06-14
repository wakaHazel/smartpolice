from __future__ import annotations

TITLE = "AI 生成图像取证研判与警务证据链系统"
SUBTITLE = "智警杯警用大模型作品赛半决赛作品文档"
VERSION = "半决赛提交版 V4.0"
AUTHOR = "SmartPolice Project Team"


EXECUTIVE_SUMMARY = [
    "本作品面向 AIGC 图片在社交平台传播后的公共安全谣言核查需求，构建“本地视觉检测组件 + 传播扰动鲁棒评测 + 警务证据链 + 报告草稿”的一体化原型。技术主线不是泛化归因满分，而是先降低真实图片误报，再对 GPT-image-2 等生成图像给出可复核的疑似来源线索。",
    "系统把研究型模型输出接入警务证据链应用：上传图片和来源 URL 后，后端保留 hash、模型版本、审计 ID、证据条目和报告草稿。模型结论只表述为“疑似来源线索”，不替代 C2PA、水印、平台元数据、发布链路和人工核验。",
    "当前 active 归因头为 e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad。训练池包含 4691 张外部图片；验证使用 120 条 clean source-holdout；训练 split 额外生成 1200 条临时扰动增强特征。full baseline 显示，active 扰动平均 Macro-F1 为 0.655；strict source-holdout 均值 Macro-F1 为 0.124，label-covered 诊断 Macro-F1 为 0.354。最新阈值校准候选组件在 360 条扰动验证上实现 Real FPR 0.033、Generated Recall 0.692，说明低误报初筛方向可继续推进，但仍需更大独立盲测。",
]

REVIEW_ALIGNMENT_TABLE = [
    (
        "现实意义 / 应用价值",
        "社交平台涉警、灾害、群体事件图像传播后，元数据可能丢失，基层研判需要快速固定证据与输出可复核线索。",
        "证据链工作台、hash/审计 ID、报告草稿、人工复核声明。",
    ),
    (
        "科学性 / 方法合理性",
        "围绕可验证的小问题：传播扰动后先做真实/生成初筛，再给 GPT-image-2 疑似线索；设置 clean holdout、扰动增强、source-holdout、阈值校准和特征消融。",
        "4691 外部样本、120 clean validation、1200 增强特征、benchmark_results、阈值校准审计、模型卡和复现接口。",
    ),
    (
        "先进性 / 创新程度",
        "不依赖元数据单点溯源，而融合视觉语义、频域、压缩痕迹、文字覆盖代理和扰动增强特征。",
        "本地归因头、candidate/active 生命周期、门控激活、证据审计闭环。",
    ),
    (
        "实践性 / 可重复性",
        "使用 FastAPI、SQLite、pytest 和可复现本地训练链路；不把演示样例混进训练。",
        "README 复现命令、pytest 45 passed、Vite build 通过。",
    ),
]


PROJECT_GOALS_TABLE = [
    ("研究目标", "建立传播扰动后 AI 生成图像鲁棒检测方法，以低误报真实/生成初筛和 GPT-image-2 疑似线索为重点。"),
    ("工程目标", "形成可导入数据、可训练、可评估、可激活、可审计、可演示的本地视觉检测系统。"),
    ("应用目标", "把模型输出转换为警务证据链条目、处置建议和报告草稿，提高材料整理效率。"),
    ("边界目标", "固定使用“疑似来源线索”措辞，不替代 C2PA、水印、平台元数据、发布链路和人工核验。"),
]


COMPARISON_TABLE = [
    (
        "元数据/C2PA 检查",
        "速度快、证据意义强",
        "平台传播、截图和二次压缩后可能缺失",
        "本项目把元数据作为证据链一环，同时研究元数据失效后的视觉线索。",
    ),
    (
        "通用 AIGC 二分类检测",
        "实现简单，容易得到较高 Accuracy",
        "只能判断真/假，难以解释疑似来源和传播扰动影响",
        "本项目以低误报初筛为第一层，再提供 GPT-image-2 等疑似来源线索，并披露 Macro-F1 与跨源短板。",
    ),
    (
        "直接调用云端多模态模型",
        "OCR 和语义理解强",
        "成本高、网络依赖强，输出过程不一定可训练可复现",
        "本项目核心为本地可训练归因头，云/本地大模型只作辅助应用层。",
    ),
    (
        "单一频域/压缩痕迹检测",
        "对压缩加工敏感，可解释性较强",
        "对语义伪造和文本富集图像不足",
        "本项目融合语义、频域、压缩、文字覆盖和扰动增强特征。",
    ),
    (
        "本项目方案",
        "可训练、可审计、能展示证据链闭环",
        "跨来源泛化和强扰动召回仍需更多独立来源样本",
        "把不足写入模型卡和局限计划，避免把半成品包装成定性工具。",
    ),
]


FEASIBILITY_TABLE = [
    ("数据可行性", "已有 4691 张外部图片训练池，覆盖 real、GPT-image-2、Flux、SDXL、Midjourney、DALL-E、Seedream、Nano Banana 等。"),
    ("技术可行性", "使用本地特征抽取 + ExtraTrees，避免昂贵的基础大模型微调；训练、评估、激活都可在本地复现。"),
    ("工程可行性", "后端 FastAPI + SQLite + pytest，前端 React + Vite，当前回归测试和构建均通过。"),
    ("演示可行性", "工作台能展示上传图片、证据固定、来源线索、报告草稿和审计 ID，适合 5 分钟视频呈现。"),
    ("风险可控性", "输出边界固定为疑似来源线索；模型失败或未训练时不伪造分数。"),
]


ROADMAP_TABLE = [
    ("半决赛提交前", "完成 PDF/DOCX、5 分钟演示视频素材、演示截图、复现命令和答辩稿。", "可提交材料包"),
    ("半决赛后 2 周", "补 GPT-image-2 强扰动样本和真实图 hard negatives，重点是截图/裁剪/水印。", "更稳定的 GPT-image-2 Recall 与真实图误报控制"),
    ("总决赛前", "建立独立 source-holdout 盲测集，完善 feature-ablation 和置信度校准。", "更可信的跨来源泛化实验"),
    ("后续落地", "接入平台协查、C2PA/水印读取、取证链路导出和权限审计。", "从原型走向警务辅助工具"),
]


SUBMISSION_PACKAGE_TABLE = [
    ("作品文档 PDF", "已生成", "output/pdf/semifinal_document.pdf"),
    ("可编辑 DOCX", "已生成", "output/docx/semifinal_document.docx"),
    ("演示视频 MP4", "已生成", "output/video/semifinal_demo.mp4；脚本见 docs/semifinal_video_script.md"),
    ("系统演示截图", "已生成", "output/playwright/workspace-desktop.png 等"),
    ("复现命令", "已写入 README", "competition-summary、robustness-run、source-holdout、feature-ablation"),
    ("答辩问答", "已写入文档", "覆盖套壳、微调、指标、证据边界等问题"),
]


KEY_METRICS = [
    ("技术主线", "传播扰动后的 AI 生成图像鲁棒检测与证据链应用，不追求泛多生成器定性归因"),
    ("Active 归因头", "e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad"),
    ("模型形态", "本地视觉检测/归因头：115 维统计、频域、压缩痕迹、语义代理特征 + ExtraTrees + 类别原型"),
    ("训练池", "4691 张外部图片样本；内置四方向演示案例不进入训练"),
    ("验证协议", "120 条 source-holdout clean validation；增强样本不进入验证集"),
    ("验证指标", "Accuracy 0.708；Macro-F1 0.503；GPT-image-2 Recall 0.915"),
    ("扰动增强", "1200 条临时增强特征；5 类扰动条件各 240 条"),
    ("鲁棒摘要", "full baseline：active 扰动平均 Macro-F1 0.655；strict source-holdout 均值 Macro-F1 0.124；label-covered Macro-F1 0.354"),
    ("候选突破", "低误报阈值校准候选：360 条扰动验证 Real FPR 0.033、Generated Recall 0.692，仍需更大盲测"),
    ("工程状态", "后端 pytest 45 passed；前端 Vite build 通过"),
]


CLASS_DISTRIBUTION = [
    ("real", "1025", "真实照片 hard negatives，高于任一单生成器类别"),
    ("gpt-image2", "838", "内部标签；文档统一称 GPT-image-2"),
    ("midjourney", "389", "Midjourney 类别"),
    ("sdxl", "364", "SDXL 类别"),
    ("dall-e-3", "353", "DALL-E 3 类别"),
    ("flux", "350", "Flux 系列生成图"),
    ("sd21", "326", "Stable Diffusion 2.1 类别"),
    ("sd3", "326", "Stable Diffusion 3 类别"),
    ("nano-banana", "214", "Nano Banana 类别"),
    ("seedream-4", "178", "Seedream 4 类别"),
    ("gpt-image1", "163", "GPT-image-1 类别"),
    ("gpt-image1.5", "100", "GPT-image-1.5 类别"),
    ("stable-diffusion", "28", "Stable Diffusion 泛类"),
    ("imagegbt", "37", "原始导入标签归一化结果，支持数小，仅披露不强调"),
]


DATA_SOURCE_TABLE = [
    (
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "1577",
        "real、SD、DALL-E、Midjourney 等",
        "主干多类别训练",
        "公开 HuggingFace 数据集，记录 source_url；提交前需复核数据集卡许可。",
    ),
    (
        "Scam-AI/gpt-image-2",
        "738",
        "gpt-image2",
        "GPT-image-2 正样本补强",
        "公开 HuggingFace 数据集；包含平台传播环境样本，仍需跨源盲测。",
    ),
    (
        "Qwen/Qwen-Image-Bench",
        "600",
        "GPT-image-1/1.5/2、Seedream、Nano Banana、Flux",
        "可追溯 benchmark 样本",
        "单生成器目录样本有限，不代表真实平台全分布。",
    ),
    (
        "siddharthksah/DeepSafe-benchmark",
        "578",
        "real、Flux、SD、DALL-E、Midjourney 等",
        "多来源生成器补充",
        "公开数据集；用于扩展多类别归因覆盖。",
    ),
    (
        "Rapidata 系列公开集",
        "344",
        "Flux、DALL-E、Midjourney、Stable Diffusion、Seedream、Nano Banana",
        "生成器类别均衡补充",
        "记录原始 source_detail；不把来源名称作为模型特征。",
    ),
    (
        "Robo531 / Fakeddit / Tiny-GenImage / splicing real-negative pools",
        "854",
        "real、Flux、Nano Banana、Seedream、imagegbt 等",
        "真实图 hard negatives 与少量类别补充",
        "只保存本地路径、hash、标签和来源元数据，不做个人身份判断。",
    ),
    (
        "GenImage / AIGIBench / SIDA / RRDataset / ITW-SM",
        "0",
        "外部社交平台与扰动鲁棒性 benchmark",
        "实验设计依据与后续外部盲测来源",
        "当前不计入 active 训练池；本地下载后可用 tools/prepare_benchmark_manifest.py 抽样导入。",
    ),
    (
        "内置四方向演示样例",
        "0",
        "涉警公信力、灾害险情、群体对立、低风险误传",
        "演示与评测展示",
        "明确不进入训练、验证或特征缓存样本。",
    ),
]


VALIDATION_PROTOCOL_TABLE = [
    ("训练样本", "4571 条原始 train split", "来自 4691 外部图片训练池，排除 120 条 clean holdout。"),
    ("验证样本", "120 条 clean source-holdout", "从训练池内部按来源/类别构造，记录在模型卡；external_training_samples 不另写 validation split。"),
    ("特征维度", "115", "图像统计、频域、压缩痕迹、文字覆盖、清洗后的语义代理等。"),
    ("增强策略", "仅训练 split", "增强不写回数据集，不进入验证，不作为真实外部样本计数。"),
    ("生命周期", "candidate -> gate -> active", "生成模型归因任务默认 candidate；必须显式激活或通过门控才替换 active。"),
]


AUGMENTATION_PROTOCOL_TABLE = [
    ("jpeg_q85", "轻度 JPEG 转码压缩", "240", "模拟社交平台轻度压缩。"),
    ("jpeg_q60", "重度 JPEG 转码压缩", "240", "模拟多次转发或强压缩。"),
    ("screenshot_resave", "截图后重保存", "240", "加入边框和画布重排，是当前最不稳定扰动。"),
    ("center_crop", "中心裁剪", "240", "模拟平台裁剪或用户二次裁切。"),
    ("watermark", "角标水印覆盖", "240", "模拟平台水印、转发标识或角标覆盖。"),
]


VALIDATION_CLASS_TABLE = [
    ("gpt-image2", "47", "0.956", "0.915", "0.935", "核心关注类，召回较高但仍只作疑似来源线索"),
    ("real", "12", "0.588", "0.833", "0.690", "真实图召回较高，但验证支持数仍偏小"),
    ("flux", "10", "0.500", "0.300", "0.375", "召回偏低，说明跨来源泛化需继续补样本"),
    ("midjourney", "2", "0.000", "0.000", "0.000", "支持数很少，不夸大单类结论"),
    ("stable-diffusion", "1", "1.000", "1.000", "1.000", "支持数很少，不夸大单类结论"),
    ("nano-banana", "8", "0.667", "0.250", "0.364", "当前弱项，需要继续补强"),
    ("sd21", "4", "1.000", "0.250", "0.400", "支持数很少，不夸大单类结论"),
    ("sd3", "4", "0.167", "0.250", "0.200", "支持数很少，不夸大单类结论"),
    ("其他小类", "32", "-", "-", "-", "DALL-E、GPT-image-1/1.5、SDXL、Seedream、imagegbt 等合计"),
]


METRIC_PROVENANCE_TABLE = [
    (
        "当前 active",
        "e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad",
        "clean validation Accuracy 0.708、Macro-F1 0.503、GPT-image-2 Recall 0.915",
        "来源于 /training/vision/competition-summary 和模型卡；历史旧模型只保留在审计记录，不作为当前材料口径。",
    ),
    (
        "扰动集合",
        "clean、jpeg_q85、jpeg_q60、screenshot_resave、center_crop、watermark",
        "full baseline limit=120",
        "参考 GenImage/AIGIBench/SIDA/RRDataset/ITW-SM 的压缩、截图、裁剪、水印与平台传播扰动口径。",
    ),
    (
        "候选门控",
        "candidate 20ba33df-c5e0-49d5-8f02-8a430aad90ab",
        "active robust average Macro-F1 0.655；candidate 0.698；严格门控 suggest_activate",
        "evaluate-candidate 设置 activate_if_passes_gate=false，未改变 active。",
    ),
    (
        "低误报阈值校准",
        "candidate 489527cd-5cc5-4abb-a0b5-52080dd63ffe",
        "推荐 threshold 0.650；360 条扰动验证 Real FPR 0.033、Generated Recall 0.692、2-class Macro-F1 0.815",
        "说明真实图误报可被压低，但强扰动下生成召回仍不足，当前只作为 candidate/component 方向。",
    ),
    (
        "GPT-image-2 专项互留",
        "candidate d9e7b935-6371-4da2-ab4f-3b7187c34faa",
        "Clean GPT-image-2 AUC/Recall 1.000/1.000；Source Macro-F1 0.309；label-covered Macro-F1 0.712",
        "留出 Scam-AI 时 GPT-image-2 recall 为 0.000，留出 Qwen 时为 0.360，说明 clean 能力不能写成跨来源泛化。",
    ),
    (
        "跨来源留出",
        "dataset_source holdout sample_limit=1000、max_holdout_groups=12，实际完成 12 个来源组",
        "strict mean Macro-F1 0.124；seen-class Macro-F1 0.139；label-covered Macro-F1 0.354；binary Macro-F1 0.464",
        "该结果用于主动披露边界，不包装成最终能力。",
    ),
    (
        "摘要来源",
        "docs/benchmark_results.md 与 output/audits/baseline_matrix_latest.json",
        "run_baseline_matrix.py full baseline 后追加已有 candidate 评估",
        "报告落盘在 D:\\smartpolice，未下载 GenImage/AIGIBench 全量数据，未使用 E 盘；摘要接口和评估脚本不触发自动激活。",
    ),
]


ROBUSTNESS_SMOKE_TABLE = [
    ("clean", "原始 clean holdout", "0.933", "full baseline 复测；不等同于 120 条主验证 Macro-F1 0.503。"),
    ("jpeg_q85", "轻度 JPEG 压缩", "0.740", "压缩后仍保留部分来源线索。"),
    ("jpeg_q60", "重度 JPEG 压缩", "0.765", "压缩痕迹特征参与判别。"),
    ("screenshot_resave", "截图/转存近似", "0.496", "当前最不稳定，GPT-image-2 recall 降至 0.222。"),
    ("center_crop", "中心裁剪", "0.614", "裁剪扰动下仍保留局部统计线索，但召回下降。"),
    ("watermark", "角标水印覆盖", "0.659", "水印会改变局部边缘与压缩残差。"),
]


LEAKAGE_CONTROL_TABLE = [
    ("监督标签隔离", "label/source/source_detail 只作为监督目标或审计来源，不进入归因特征。"),
    ("文本清洗", "文本富集型图像上下文会移除 gpt-image2、midjourney、sdxl、flux、dall-e 等生成器名称，避免标签泄漏。"),
    ("特征范围", "归因分支使用图片统计、压缩、频域、文字覆盖代理和清洗后的视觉语义上下文。"),
    ("演示样例隔离", "内置四方向展示样例不进入训练、验证或特征缓存。"),
    ("增强隔离", "扰动图片写入临时目录，仅抽取训练特征；训练后临时文件删除。"),
]


TEST_RESULTS_TABLE = [
    ("后端回归", "python -m pytest D:\\smartpolice\\backend\\tests -q", "完整接口/单元测试", "45 passed"),
    ("前端构建", "npm run build", "React + TypeScript + Vite 生产构建", "build passed"),
    ("数据导入", "POST /training/datasets/import", "CSV/JSON/JSONL + image_root + label_column", "覆盖 fixture 与外部训练池"),
    ("候选训练", "POST /training/vision/run", "activation_mode=candidate", "候选模型不覆盖 active"),
    ("显式激活", "POST /training/vision/activate", "指定 run_id", "只切换同 task_type active 指针"),
    ("比赛摘要", "GET /training/vision/competition-summary", "只读 active 状态", "返回训练池、模型卡摘要、指标与局限"),
    ("鲁棒复测", "POST /training/vision/robustness-run", "clean/JPEG/截图/裁剪/水印", "输出分项 Macro-F1、召回率、混淆矩阵"),
    ("source-holdout", "POST /training/vision/source-holdout-run", "按 dataset/source 留出", "用于暴露跨来源泛化风险"),
    ("feature-ablation", "POST /training/vision/feature-ablation-run", "特征组消融", "用于解释语义、频域、压缩痕迹与扰动特征贡献"),
    ("正式研判", "POST /cases/{case_id}/real-analysis", "案例图片、URL 快照、证据链", "返回 hash、模型版本、审计 ID、报告草稿"),
]


REPRODUCTION_COMMANDS = [
    ("查询半决赛摘要", 'Invoke-RestMethod "http://127.0.0.1:8000/training/vision/competition-summary?task_type=vision_generator_attribution"'),
    ("鲁棒性复测", '$body=@{task_type="vision_generator_attribution";limit=120;conditions=@("clean","jpeg_q85","jpeg_q60","screenshot_resave","center_crop","watermark")} | ConvertTo-Json; Invoke-RestMethod "http://127.0.0.1:8000/training/vision/robustness-run" -Method Post -ContentType "application/json" -Body $body'),
    ("source-holdout", 'Invoke-RestMethod "http://127.0.0.1:8000/training/vision/source-holdout-run" -Method Post -ContentType "application/json" -Body \'{"task_type":"vision_generator_attribution","holdout_key":"dataset_source","sample_limit":1000,"max_holdout_groups":12}\''),
    ("feature-ablation", 'Invoke-RestMethod "http://127.0.0.1:8000/training/vision/feature-ablation-run" -Method Post -ContentType "application/json" -Body \'{"task_type":"vision_generator_attribution","limit":120}\''),
]


LIMITATION_ANALYSIS_TABLE = [
    ("真实/生成初筛未完全达标", "360 条扰动验证 Real FPR 0.033，但 Generated Recall 0.692 仍不足", "低误报方向有价值，但强扰动生成图会漏检", "补 Synthbuster、Defactify、AIGC-Detection-Benchmark 等 hard negatives 与强扰动生成样本"),
    ("GPT-image-2 跨来源不足", "clean AUC/Recall 可到 1.000，但 Source Macro-F1 0.309；Scam-AI 留出 recall 0.000", "不能把 clean 高分写成平台泛化能力", "用 Scam-AI、Qwen、自建平台转码样本做来源互留和域归一化"),
    ("真实平台扰动复杂", "本地 screenshot_resave 只是近似，active 在该条件下 GPT-image-2 recall 0.222", "真实平台还可能叠加缩放、滤镜、转码和二次截图", "补真实截图、裁剪、水印、多次压缩和平台转码配对样本"),
    ("跨来源泛化风险", "strict source-holdout mean Macro-F1 0.124，label-covered Macro-F1 0.354", "模型可能学到数据集风格或压缩习惯", "保留 source-holdout，新增独立盲测集"),
    ("证据边界", "模型只能输出疑似来源线索", "不能单独用于执法定性", "结合 C2PA、水印、平台元数据、发布链路和人工核验"),
    ("演示视频增强", "已生成 5 分钟以内自动讲解版 MP4", "自动成片可满足提交包完整性，但如现场答辩可补真人录屏版本", "保留当前 MP4，同时按脚本准备真人演示替换版"),
]


ARCHITECTURE_FLOW = [
    ("输入取证", "图片、URL、案例文本", "保存原始文件、正文、截图、sha256 和时间戳"),
    ("特征抽取", "视觉语义、频域、压缩痕迹、文字覆盖", "feature_cache 按 sha256 + extractor_version 复用"),
    ("归因训练", "ExtraTrees + 类别原型", "candidate 模型、模型卡、训练/验证指标"),
    ("鲁棒评估", "clean、JPEG、截图、裁剪、水印", "门控报告、source-holdout、feature-ablation"),
    ("警务应用", "证据链、处置建议、报告草稿", "hash、模型版本、审计 ID、人工复核声明"),
]


TRAINING_FLOW = [
    ("1. 外部样本导入", "CSV/JSON/JSONL + 本地图片路径；保存 dataset/source/label/sha256/raw_payload。"),
    ("2. 样本清洗与隔离", "排除演示样例；清洗生成器名称文本；去重并记录来源。"),
    ("3. clean holdout 构造", "从训练池内部按来源/类别抽取 120 条 clean validation，模型卡记录协议。"),
    ("4. 特征缓存", "提取 115 维统计、频域、压缩痕迹、文字覆盖和语义代理特征。"),
    ("5. 扰动增强", "对训练 split 生成 1200 条临时增强特征；验证集保持 clean。"),
    ("6. candidate 训练与门控", "默认保存候选；比较 clean、GPT-image-2 Recall、扰动平均 Macro-F1 和真实图误报。"),
    ("7. active 激活", "显式激活或通过门控后切换 active；不重训、不改变训练样本。"),
]


SCREENSHOTS = [
    ("图 1 取证研判中心首页与 active 模型摘要", "output/playwright/workspace-desktop.png"),
    ("图 2 证据链、复核与报告演示闭环", "output/playwright/ui-e2e-real-chain-final.png"),
]


TERMINOLOGY_TABLE = [
    ("GPT-image-2", "文档统一写法；数据库内部 label 仍保留 gpt-image2。"),
    ("active 归因头", "正式研判使用的本地视觉归因模型，不代表基础多模态大模型被微调。"),
    ("source-holdout", "按数据来源留出的验证/评估口径，用于暴露跨来源泛化风险。"),
    ("Macro-F1", "多类别平均 F1，避免被大类样本数掩盖；接口字段保留 macro_f1。"),
    ("疑似来源线索", "模型输出的法务边界措辞，不等同于确定证据或执法结论。"),
]


DOC_SECTIONS = [
    {
        "title": "一、研究背景与问题定义",
        "paragraphs": [
            "GPT-image-2 等生成模型显著降低了伪造公共安全场景图片的门槛。图片进入社交平台传播后，原始 C2PA 元数据、平台标识或水印可能被压缩、截图转存、裁剪、水印覆盖和再次上传等操作削弱甚至剥离。对公安机关而言，单靠元数据或人工肉眼判断很难在早期舆情扩散阶段形成可解释、可复核的研判依据。",
            "本项目聚焦一个更窄但更真实的问题：当元数据不可靠时，能否从传播扰动后的视觉内容、频域纹理、压缩痕迹和文字覆盖代理中提取稳定线索，先判断图片是否疑似生成，再对 GPT-image-2 等重点来源给出可复核线索，并把模型输出纳入警务证据链工作流。",
            "因此，平台工作台只是应用场景；技术主线是“传播扰动鲁棒检测与证据链化应用”。模型结论固定表述为疑似生成或疑似来源线索，最终处置必须结合 C2PA、水印、平台元数据、发布账号链路、原始文件流转和人工核验。",
        ],
        "bullets": [
            "研究对象：社交平台传播后的疑似 AI 生成图像，重点覆盖 GPT-image-2 与真实/生成低误报初筛。",
            "核心挑战：压缩、截图、裁剪、水印会改变像素统计和元数据，使常规溯源信号失效。",
            "应用边界：服务网络谣言核查、证据留存、平台协查和报告草稿，不替代执法定性。",
        ],
        "tables": ["review_alignment"],
    },
    {
        "title": "二、总体技术路线",
        "paragraphs": [
            "系统采用“外部数据建设 -> 特征抽取 -> 本地监督检测组件 -> 传播扰动增强 -> 候选门控 -> 警务证据链应用”的路线。当前版本不训练 Qwen3-VL 或其他基础多模态大模型本体；可训练核心是本地视觉检测/归因头。",
            "特征层同时使用视觉语义代理、频域纹理、JPEG/重压缩痕迹、字节分布、文字覆盖和水印代理。分类层使用 ExtraTrees 作为主分类器，并保留类别原型与 unknown 阈值，用于低置信输出和边界控制。",
            "警用大模型相关能力被放在应用辅助层：OCR、画面事实描述、文本复核和报告生成可作为证据整理工具，但文档不把这些写成基础大模型微调成果。",
        ],
        "bullets": [
            "鲁棒性设计：增强只进入训练 split，clean validation 保持原始样本，避免验证污染。",
            "可审计设计：模型卡记录训练来源、类别分布、增强协议、泄漏控制、生命周期和边界声明。",
            "生命周期设计：生成模型归因任务默认 candidate，不自动覆盖 active。",
        ],
        "tables": ["comparison"],
    },
    {
        "title": "三、数据集建设与合规边界",
        "paragraphs": [
            "训练池来自公开 HuggingFace 数据集、本地下载图片和转换后的公开图片样本。系统导入时记录 dataset/source/source_url/task/split/image_path/label/raw_payload/imported_at 和图片 sha256，便于去重、追溯和模型卡审计。",
            "真实照片 hard negatives 被刻意放大到 1025 张，高于任一单生成器类别，用于降低把普通新闻图、手机拍摄图、社交截图或海报误判为 GPT-image-2 的风险。",
            "数据合规方面，系统只保存图像文件路径、hash、标签和来源元数据，不进行个人身份识别。公开数据集的许可证和使用范围仍需在最终提交前由参赛者逐项确认，并在答辩中说明用途限定为科研比赛原型。",
        ],
        "bullets": [
            "排除：风格迁移类不作为主训练目标；内置四方向演示样例不进入训练池。",
            "记录：模型卡保留 source_url、导入时间、类别分布和排除声明。",
            "风险：公开数据集不能代表真实平台全分布，必须配合 source-holdout 和后续盲测。",
        ],
        "tables": ["data_sources"],
    },
    {
        "title": "四、训练过程与泄漏控制",
        "paragraphs": [
            "active 归因头 e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad 训练于 2026-06-11，模型类型为 local-generator-attribution-extratrees-v2。模型卡显示，训练池 4691 条，训练 split 4571 条，clean validation 120 条，特征数 115。",
            "训练过程中特别控制两类容易被质疑的泄漏：第一，label/source/source_detail 只作为监督目标或审计来源，不进入模型特征；第二，文本富集型图像上下文会移除 gpt-image2、midjourney、sdxl、flux、dall-e 等生成器名称，避免模型直接读取标签词。",
            "扰动增强通过 feature_cache 记录 sha256、扰动条件、清洗文本摘要和 extractor_version。增强图像写入临时目录，只抽取训练特征，训练后删除，不写回 external_training_samples。",
        ],
        "bullets": [
            "训练样本：4571 条原始 train split + 1200 条临时扰动增强特征。",
            "验证样本：120 条 clean source-holdout，不加入扰动增强。",
            "增强缓存：当前 active 复用了 1200 条已缓存增强特征，cache_hits 1200，cache_misses 0。",
        ],
        "tables": ["validation_protocol", "augmentation_protocol"],
    },
    {
        "title": "五、实验结果与指标解释",
        "paragraphs": [
            "active 归因头在 120 条 clean source-holdout 上取得 Accuracy 0.708、Macro-F1 0.503、GPT-image-2 Recall 0.915。这个结果说明模型对 GPT-image-2 类别已有较强内部召回，但多类别来源判别和跨来源场景仍不稳定，不能把 clean 高召回写成平台泛化能力。",
            "最新 full baseline 采用 clean、jpeg_q85、jpeg_q60、screenshot_resave、center_crop、watermark 六条件复测。active 扰动平均 Macro-F1 为 0.655，clean 到最弱 screenshot_resave 的 Macro-F1 下降 0.437；candidate 20ba33df 的扰动平均 Macro-F1 为 0.698，因此严格门控结论为 suggest_activate，但本轮未自动替换 active。",
            "技术线后续又拆出两个更适合提交叙事的候选组件：其一是低误报真实/生成初筛阈值校准，推荐 threshold 0.650，在 360 条 clean/JPEG/截图/裁剪/水印扰动验证中 Real FPR 为 0.033、Generated Recall 为 0.692；其二是 GPT-image-2 专项识别，clean GPT-image-2 AUC/Recall 为 1.000/1.000，但 Source Macro-F1 只有 0.309。两个结果共同说明：低误报初筛方向有价值，GPT-image-2 clean 能力强，但强扰动和跨来源仍是主要瓶颈。",
            "泛化能力部分不是单靠本项目内部切分自证，而是借鉴公开 benchmark 的评测协议：GenImage 对应 cross-generator 与 degraded image，AIGIBench 对应多来源外部盲测，SIDA/SID-Set 对应社交媒体域，RRDataset/ITW-SM 对应真实传播、平台重采样和重拍。当前项目已落成 source-holdout 与六条件扰动矩阵；尚未导入的大集只作为协议依据和下一轮盲测来源。",
            "训练集指标接近满分并不作为核心成绩展示，因为树模型在增强训练集上的拟合很容易偏高。提交材料优先展示 clean holdout、扰动复测、source-holdout 风险和局限分析，以避免看起来像指标作弊。",
        ],
        "bullets": [
            "主要卖点：低误报真实/生成初筛有可调空间，GPT-image-2 clean recall 较高，并能用 benchmark 矩阵披露扰动降级曲线。",
            "主要短板：多类别来源判别 Macro-F1 仍只有 0.503，strict source-holdout 均值 Macro-F1 仅 0.124，GPT-image-2 来源互留仍未达标。",
            "答辩口径：GPT-image-2 是疑似来源线索，不能作为单独证据。",
            "泛化口径：借鉴 GenImage 的 cross-generator/degraded image、AIGIBench 的外部盲测、SIDA/SID-Set 的社交媒体域、RRDataset/ITW-SM 的真实传播扰动协议，但不声称覆盖这些 leaderboard。",
        ],
        "tables": ["validation_class", "metric_provenance"],
    },
    {
        "title": "六、系统实现与应用演示",
        "paragraphs": [
            "系统采用 React + TypeScript + Vite 前端和 FastAPI + Pydantic + SQLite + httpx + pytest 后端。后端提供数据导入、训练、候选激活、鲁棒复测、source-holdout、feature-ablation、案例证据和正式研判接口。",
            "前端应用场景被收束为 AI 生成图像取证研判：上传图片或保存 URL 快照后，系统展示图片 hash、active 模型版本、疑似来源线索、扰动鲁棒性说明、证据条目、人工复核声明和报告草稿。",
            "这个工作台不是技术主线本身，而是把鲁棒归因研究成果转换为警务可读流程的展示层。它的价值在于让评委看到模型训练产物如何进入证据链，而不是停留在离线实验表格。",
        ],
        "bullets": [
            "核心接口：/training/vision/competition-summary、/training/vision/robustness-run、/cases/{case_id}/real-analysis。",
            "证据资产：case_assets、web_snapshots、evidence_items 记录原始输入、hash、截图和证据条目。",
            "审计输出：模型版本、审计 ID、人工复核声明必须出现在报告和页面中。",
        ],
        "tables": ["screenshots"],
    },
    {
        "title": "七、测试说明与复现结果",
        "paragraphs": [
            "测试目标分为工程正确性和研究复现性两层。工程正确性验证接口、构建和演示流程不崩；研究复现性验证训练池、active 模型、增强协议、指标口径和局限声明能从模型卡或接口中查到。",
            "需要注意的是，source-holdout 和 feature-ablation 在本地全量执行可能耗时较长，不应被包装成“秒级随手复测”。提交材料保留命令和接口，答辩现场可选择 competition-summary 和小 limit robustness-run 作为快速演示。",
        ],
        "bullets": [
            "已验证：python -m pytest D:\\smartpolice\\backend\\tests -q -> 45 passed。",
            "已验证：npm run build -> Vite production build passed。",
            "已验证：PDF/DOCX 结构包含测试说明、指标溯源、数据合规、局限分析和术语表。",
        ],
        "tables": ["reproduction_commands"],
    },
    {
        "title": "八、创新点与比赛契合度",
        "paragraphs": [
            "作品的创新点不在于简单调用大模型，而在于把生成图像检测放进传播扰动场景中：当元数据和水印可能被剥离时，尝试融合视觉语义、频域、压缩痕迹和传播扰动增强特征，提高传播后图片的疑似生成线索稳定性。",
            "与智警杯警用大模型主题的契合方式是“本地视觉检测能力 + 警务证据链应用”。本地检测/归因头承担可训练、可复现、可审计的核心；大模型能力可在 OCR、画面事实、文本复核和报告生成中辅助，但不冒充基础模型微调。",
            "工程创新体现在 candidate/active 生命周期、增强 cache、模型卡、source-holdout、feature-ablation、证据审计 ID 和报告边界声明。这些机制能防止半成品系统把不稳定模型直接包装成执法结论。",
        ],
        "bullets": [
            "研究创新：传播扰动后的 AI 生成图像鲁棒检测与 GPT-image-2 疑似线索。",
            "算法创新：语义、频域、压缩、文字覆盖、扰动增强融合。",
            "工程创新：候选门控、模型卡、证据链和审计闭环。",
            "应用创新：把疑似来源线索转化为可复核的警务报告草稿。",
        ],
        "tables": [],
    },
    {
        "title": "九、局限性与下一步计划",
        "paragraphs": [
            "当前作品仍是研究型半决赛原型，而不是可直接上线的执法系统。最核心的不足是强扰动生成图召回不足、GPT-image-2 跨来源互留未达标、真实平台扰动样本仍少，source-holdout 泛化风险需要更大盲测集验证。",
            "下一步数据建设应优先补 GPT-image-2 强扰动正样本，特别是截图转存、裁剪、水印覆盖和多次压缩样本；同时继续扩大真实照片 hard negatives，覆盖社交平台截图、新闻图、手机拍摄图和带文字海报。泛化盲测优先按 GenImage/AIGIBench 的 cross-source 抽样接入，再按 SIDA、RRDataset 和 ITW-SM 的社交平台传播口径补真实平台样本。",
            "半决赛提交包已生成 5 分钟以内 MP4 自动讲解演示片，覆盖首页指标、图片上传、证据链、报告页和边界声明；若后续需要更强现场感，可按同一脚本补录真人操作版。",
        ],
        "bullets": [
            "不宣称：不宣称训练基础多模态大模型本体。",
            "不替代：不替代 C2PA、水印、平台元数据、发布链路和人工核验。",
            "不定性：不把 GPT-image-2 线索写成确定证据。",
            "不混淆：GenImage、AIGIBench、SIDA/SID-Set、RRDataset、ITW-SM 当前作为泛化评测设计依据和后续外部盲测来源，未全量混入 active 训练池。",
        ],
        "tables": ["limitations"],
    },
]


DEFENSE_QA = [
    ("这是不是套壳？", "不是。核心是本地视觉检测/归因头训练、扰动增强、模型卡、候选门控、阈值校准和鲁棒评估；大模型/报告能力只在应用辅助层。"),
    ("是不是微调了 Qwen3-VL？", "没有。当前没有训练基础多模态大模型本体，训练的是本地监督视觉归因头。"),
    ("full baseline 会不会是硬编码？", "不是。报告由 tools/run_baseline_matrix.py 调用现有 robustness/source-holdout/feature-ablation/evaluate-candidate 路径生成，保存 JSON 和 Markdown；脚本不下载大集、不自动激活 candidate。"),
    ("为什么 Macro-F1 只有 0.503？", "因为这是多类别来源线索任务，不是简单二分类 AIGC 检测；小类支持数不足和跨来源分布差异会拉低 Macro-F1。当前提交把主线收束为低误报真实/生成初筛和 GPT-image-2 疑似线索，并主动披露跨来源不足。"),
    ("低误报初筛有什么进展？", "阈值校准候选 489527cd 在 360 条扰动验证上 Real FPR 为 0.033、Generated Recall 为 0.692，说明压低真实图误报有进展，但强扰动生成图召回还需要继续补样本。"),
    ("泛化能力借鉴了哪些 baseline？", "借鉴 GenImage 的跨生成器/降质评估、AIGIBench 的多来源外部盲测、SIDA/SID-Set 的社交媒体域定义、RRDataset/ITW-SM 的真实传播扰动；当前已落成 source-holdout 与六条件扰动矩阵，但不声称覆盖这些 leaderboard。"),
    ("GPT-image-2 结论能不能当证据？", "不能单独作为确定证据。它只能作为疑似来源线索，必须结合 C2PA、水印、平台元数据、发布链路和人工核验。"),
    ("警务价值在哪里？", "价值在快速固定 hash、来源快照、模型版本、审计 ID 和处置报告草稿，帮助研判人员提高证据整理效率。"),
]
