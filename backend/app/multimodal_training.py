from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import math
import os
import pickle
from pathlib import Path
import re
from statistics import mean
import tempfile
from uuid import uuid4

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from app.models import (
    CaseAsset,
    CaseSample,
    DemoEvaluationCaseResult,
    DemoEvaluationResult,
    ExternalTrainingSample,
    FeatureCacheRecord,
    FeatureWeight,
    FusionTrainingRunRequest,
    FusionTrainingRunResult,
    FusionTrainingStatus,
    TrainingTaskStatus,
    VisionAugmentationCacheWarmupRequest,
    VisionAugmentationCacheWarmupResult,
    VisionAntiCheatAuditRequest,
    VisionAntiCheatAuditResult,
    VisionCandidateEvaluationRequest,
    VisionCandidateEvaluationResult,
    VisionCompetitionSummary,
    VisionTrainingActivationRequest,
    VisionTrainingActivationResult,
    VisionTrainingRunRequest,
    VisionTrainingRunResult,
    VisionTrainingRunRecord,
    VisionRobustnessConditionResult,
    VisionRobustnessRunRequest,
    VisionRobustnessRunResult,
    VisionFeatureAblationResult,
    VisionFeatureAblationRunRequest,
    VisionFeatureAblationRunResult,
    VisionSourceHoldoutGroupResult,
    VisionSourceHoldoutRunRequest,
    VisionSourceHoldoutRunResult,
    VisionTrainingStatus,
)
from app.risk_model import predict_with_active_model, risk_level_from_score
from app.sample_data import DEMO_CASES
from app.storage import (
    activate_vision_training_run,
    get_active_fusion_training_artifact,
    get_active_vision_training_artifact,
    get_active_vision_training_run,
    get_feature_cache,
    get_latest_fusion_training_run,
    get_latest_vision_candidate_run,
    get_latest_vision_training_run,
    get_training_data_status,
    get_vision_training_artifact_by_id,
    list_vision_training_runs,
    list_external_training_samples,
    save_feature_cache,
    save_fusion_training_run,
    save_vision_training_run,
    update_vision_training_run_payload,
)


VISION_TASK_TYPES = {
    "vision_aigc",
    "vision_tamper",
    "vision_context_mismatch",
    "vision_generator_attribution",
}
GENERATOR_ATTRIBUTION_TASK = "vision_generator_attribution"
FUSION_TASK_TYPE = "multimodal_fusion"
EXTRACTOR_VERSION = "local-multimodal-features-v3"
RISK_LEVELS = ("低", "关注", "较高", "紧急")
GENERATOR_ATTRIBUTION_LABELS = (
    "gpt-image2",
    "gpt-image1",
    "gpt-image1.5",
    "midjourney",
    "sd21",
    "sd3",
    "sdxl",
    "stable-diffusion",
    "flux",
    "dall-e",
    "dall-e-3",
    "nano-banana",
    "seedream-4",
    "imagegbt",
    "real",
    "unknown",
)
GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR = 0.42
KNN_CANDIDATE_K = (3, 5, 9, 15, 25)
ENSEMBLE_ALPHA_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_PROTOTYPES = 2500
CLIP_FEATURE_DIMS = int(os.getenv("SMARTPOLICE_CLIP_FEATURE_DIMS", "24"))
CLIP_PROTO_DIMS = int(os.getenv("SMARTPOLICE_CLIP_PROTO_DIMS", "128"))
CLIP_MODEL_NAME = os.getenv("SMARTPOLICE_CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_ENABLED = os.getenv("SMARTPOLICE_ENABLE_CLIP", "").lower() in {"1", "true", "yes", "on"}
CLIP_EXTRACTOR_VERSION = f"local-clip-image-text-v1:{CLIP_MODEL_NAME}:{CLIP_FEATURE_DIMS}"
GENERATOR_TREE_ESTIMATORS = int(os.getenv("SMARTPOLICE_GENERATOR_TREE_ESTIMATORS", "360"))
GENERATOR_TREE_MIN_SAMPLES_LEAF = int(os.getenv("SMARTPOLICE_GENERATOR_TREE_MIN_SAMPLES_LEAF", "2"))
ADVANCED_TREE_ESTIMATORS = int(os.getenv("SMARTPOLICE_ADVANCED_TREE_ESTIMATORS", "220"))
ADVANCED_TREE_MIN_SAMPLES = int(os.getenv("SMARTPOLICE_ADVANCED_TREE_MIN_SAMPLES", "12"))
GENERATOR_REAL_CLASS_WEIGHT = float(os.getenv("SMARTPOLICE_GENERATOR_REAL_CLASS_WEIGHT", "1.35"))
GENERATOR_REAL_HARD_NEGATIVE_WEIGHT = float(os.getenv("SMARTPOLICE_GENERATOR_REAL_HARD_NEGATIVE_WEIGHT", "1.6"))
GENERATOR_AUGMENTATION_EXTRACTOR_VERSION = f"{EXTRACTOR_VERSION}:generator-augmentation-v2-platform-like"
GENERATOR_BINARY_GATE_THRESHOLD = float(os.getenv("SMARTPOLICE_GENERATOR_BINARY_GATE_THRESHOLD", "0.56"))
GENERATOR_REAL_PROTECTION_MARGIN = float(os.getenv("SMARTPOLICE_GENERATOR_REAL_PROTECTION_MARGIN", "0.08"))
GENERATOR_EXPERIMENT_PROFILES = (
    "standard_attribution",
    "binary_generated_gate",
    "gpt_image2_ovr",
    "mainstream_five_attribution",
    "multi_generator_label_covered",
    "clean_origin_attribution",
    "social_propagation_robustness",
)
MAINSTREAM_FIVE_GENERATOR_LABELS = (
    "gpt-image2",
    "nano-banana",
    "seedream-4",
    "stable-diffusion",
    "midjourney",
)
GENERATOR_PROFILE_POLICIES: dict[str, dict[str, object]] = {
    "standard_attribution": {
        "profile": "standard_attribution",
        "chinese_name": "标准三分类生成图研判",
        "objective": "识别 GPT-image2、其他 AI 生成图和真实照片三类，作为 active 主线。",
        "model_strategy": "三分类生成图研判：gpt-image2 / other-generated / real；unknown 只作为低置信预测退让，不作为训练主类。",
        "feature_strategy": "图片统计、压缩残差、频域/纹理、文字覆盖代理、清洗后的视觉语义上下文，可选 CLIP 语义特征。",
        "label_strategy": "GPT-image2 保留为独立类；真实照片保留为 real；其他生成模型和未细分生成来源合并为 other-generated。",
        "system_role": "默认生成图检测/归因头，只输出 GPT-image2、其他 AI 生成图、真实照片与低置信 unknown 线索。",
        "activation_policy": "仅 standard_attribution 可按 activation_mode 走原有 active 生命周期。",
        "activation_eligibility": "standard_lifecycle",
        "candidate_only": False,
        "acceptance_gates": [
            {
                "metric": "clean_macro_f1",
                "name": "Clean Macro-F1",
                "operator": ">=",
                "threshold": 0.60,
                "source": "clean_diagnostics.macro_f1",
            },
            {
                "metric": "source_macro_f1",
                "name": "Source-holdout Macro-F1",
                "operator": ">=",
                "threshold": 0.20,
                "source": "source_holdout.aggregate.mean_macro_f1",
            },
        ],
    },
    "binary_generated_gate": {
        "profile": "binary_generated_gate",
        "chinese_name": "真实/生成鲁棒初筛",
        "objective": "先判断真实图 vs 疑似生成图，优先压低真实图误报，再把疑似生成图交给后续归因。",
        "model_strategy": "generated/real 二分类 gate；使用 real-FPR-first 阈值校准和 source-balanced sample weights。",
        "feature_strategy": "重点使用压缩残差、频域/块效应、尺寸/文件统计、文字覆盖与传播扰动特征。",
        "label_strategy": "所有非 real 生成器合并为 generated；real-negative 与真实来源保留为 real。",
        "system_role": "两层可信输出的第一层，适合作为低误报初筛组件。",
        "activation_policy": "只保存 component candidate；不得通过本 profile 直接替换 active。",
        "activation_eligibility": "component_candidate",
        "candidate_only": True,
        "acceptance_gates": [
            {
                "metric": "source_real_false_positive_rate",
                "name": "Source Real FPR",
                "operator": "<=",
                "threshold": 0.10,
                "source": "source_holdout.aggregate.overall_real_false_positive_rate",
            },
            {
                "metric": "source_generated_recall",
                "name": "Source Generated Recall",
                "operator": ">=",
                "threshold": 0.90,
                "source": "source_holdout.aggregate.mean_generated_recall",
            },
            {
                "metric": "source_macro_f1",
                "name": "Source Macro-F1",
                "operator": ">=",
                "threshold": 0.55,
                "source": "source_holdout.aggregate.mean_macro_f1",
            },
        ],
    },
    "gpt_image2_ovr": {
        "profile": "gpt_image2_ovr",
        "chinese_name": "GPT-image2 专项识别",
        "objective": "把 GPT-image2 从 real 和 other-generated 中单独识别出来，解决多分类硬顶导致的召回塌陷。",
        "model_strategy": "one-vs-rest 三分类：gpt-image2 / other-generated / real；后续补 Qwen 与 Scam-AI 来源互留评估。",
        "feature_strategy": "保留通用视觉取证特征，同时重点观察 GPT-image2 与其他生成器的压缩/纹理/语义差异。",
        "label_strategy": "GPT-image2 为正类；真实图为 real；其他生成器合并为 other-generated。",
        "system_role": "第二层来源线索组件，只输出疑似 GPT-image2，不做执法定论。",
        "activation_policy": "只保存 component candidate；通过门槛后建议进入组合研判，不直接替换 active。",
        "activation_eligibility": "component_candidate",
        "candidate_only": True,
        "acceptance_gates": [
            {
                "metric": "gpt_image2_recall",
                "name": "GPT-image2 Recall",
                "operator": ">=",
                "threshold": 0.60,
                "source": "clean_diagnostics.per_class.gpt-image2.recall",
            },
            {
                "metric": "gpt_image2_precision",
                "name": "GPT-image2 Precision",
                "operator": ">=",
                "threshold": 0.70,
                "source": "clean_diagnostics.per_class.gpt-image2.precision",
            },
            {
                "metric": "source_macro_f1",
                "name": "Source Macro-F1",
                "operator": ">=",
                "threshold": 0.45,
                "source": "source_holdout.aggregate.mean_macro_f1",
            },
        ],
    },
    "mainstream_five_attribution": {
        "profile": "mainstream_five_attribution",
        "chinese_name": "五类主流生成器归因",
        "objective": "把归因范围收束到 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney 五个主流来源，降低长尾小类和来源耦合带来的噪声。",
        "model_strategy": "open-set 多分类归因；只强归因五个主流来源，real 保留，其他生成器统一 unknown/other。",
        "feature_strategy": "使用通用视觉取证特征和 source-balanced sample weights；不再为 DALL-E、Flux、Imagen、Firefly 等长尾类单独优化。",
        "label_strategy": "GPT-image2、nano-banana、seedream-4、stable-diffusion 系列、midjourney 保留；sd21/sd3/sdxl 合并到 stable-diffusion；其他生成器映射 unknown。",
        "system_role": "第二层主流来源线索，是后续归因汇报的主轨；低置信或非五类输出 unknown。",
        "activation_policy": "只保存 component candidate；不自动替换 active。",
        "activation_eligibility": "component_candidate",
        "candidate_only": True,
        "target_labels": MAINSTREAM_FIVE_GENERATOR_LABELS,
        "acceptance_gates": [
            {
                "metric": "mainstream_macro_f1",
                "name": "Mainstream Macro-F1",
                "operator": ">=",
                "threshold": 0.35,
                "source": "source_holdout.aggregate.label_covered_macro_f1",
            },
            {
                "metric": "source_macro_f1",
                "name": "Source Macro-F1",
                "operator": ">=",
                "threshold": 0.25,
                "source": "source_holdout.aggregate.mean_macro_f1",
            },
            {
                "metric": "real_false_positive_rate",
                "name": "Real FPR",
                "operator": "<=",
                "threshold": 0.20,
                "source": "source_holdout.aggregate.overall_real_false_positive_rate",
            },
            {
                "metric": "unknown_rate",
                "name": "Unknown rate",
                "operator": "report",
                "threshold": None,
                "source": "clean_diagnostics.unknown_rate",
            },
        ],
    },
    "multi_generator_label_covered": {
        "profile": "multi_generator_label_covered",
        "chinese_name": "多来源覆盖生成器归因",
        "objective": "只对跨多个 dataset_source 覆盖的生成器做强归因，减少单来源数据污染导致的虚高。",
        "model_strategy": "open-set attribution；跨来源覆盖类别保留，单来源或小样本类别进入 unknown/other 兜底。",
        "feature_strategy": "通用视觉取证特征 + source-balanced sample weights，重点看 label-covered holdout。",
        "label_strategy": "real 保留；生成器类别至少覆盖 2 个 dataset_source 才保留原标签，否则映射 unknown。",
        "system_role": "第二层多生成器来源线索，负责边界清楚的强归因和 unknown 退让。",
        "activation_policy": "只保存 component candidate；unknown 输出率必须随结果一起报告。",
        "activation_eligibility": "component_candidate",
        "candidate_only": True,
        "acceptance_gates": [
            {
                "metric": "label_covered_macro_f1",
                "name": "Label-covered Macro-F1",
                "operator": ">=",
                "threshold": 0.30,
                "source": "source_holdout.aggregate.label_covered_macro_f1",
            },
            {
                "metric": "unknown_rate",
                "name": "Unknown rate",
                "operator": "report",
                "threshold": None,
                "source": "clean_diagnostics.unknown_rate",
            },
        ],
    },
    "clean_origin_attribution": {
        "profile": "clean_origin_attribution",
        "chinese_name": "Clean 原图归因上限",
        "objective": "测 clean/origin 图像条件下的归因上限，用来量化平台传播前后的性能落差。",
        "model_strategy": "clean/origin 上限实验；不把 clean 高分当作社交平台泛化能力。",
        "feature_strategy": "完整归因特征，排除传播域样本，让结果代表理想输入条件。",
        "label_strategy": "保留 clean_origin、多生成器 benchmark、GPT-image2 focus 样本，排除传播域样本。",
        "system_role": "上限参照和误差分解，不承担上线组件角色。",
        "activation_policy": "benchmark-only；不得直接激活。",
        "activation_eligibility": "benchmark_only",
        "candidate_only": True,
        "acceptance_gates": [
            {
                "metric": "clean_macro_f1",
                "name": "Clean Macro-F1",
                "operator": ">=",
                "threshold": 0.85,
                "source": "clean_diagnostics.macro_f1",
            },
        ],
    },
    "social_propagation_robustness": {
        "profile": "social_propagation_robustness",
        "chinese_name": "社交传播鲁棒性",
        "objective": "专门评估截图、压缩、重采样、水印、转发等传播扰动下的真实/生成识别稳定性。",
        "model_strategy": "generated/real 鲁棒性 candidate；优先作为评测与 hard-negative mining 池。",
        "feature_strategy": "重点使用扰动相关特征，并配合 JPEG、截图重保存、裁剪、水印增强。",
        "label_strategy": "传播域与 real-negative_pool 中的真实图保留 real；生成图映射 generated。",
        "system_role": "鲁棒性证据和 hard-negative 来源，不直接证明生成器归因能力。",
        "activation_policy": "benchmark-only/component candidate；不得直接替换 active。",
        "activation_eligibility": "benchmark_only",
        "candidate_only": True,
        "acceptance_gates": [
            {
                "metric": "source_real_false_positive_rate",
                "name": "Source Real FPR",
                "operator": "<=",
                "threshold": 0.05,
                "source": "source_holdout.aggregate.overall_real_false_positive_rate",
            },
            {
                "metric": "source_generated_recall",
                "name": "Source Generated Recall",
                "operator": ">=",
                "threshold": 0.90,
                "source": "source_holdout.aggregate.mean_generated_recall",
            },
        ],
    },
}
_CLIP_BUNDLE: tuple[object, object] | None = None
_CLIP_LOAD_ERROR: str | None = None
MODEL_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "model_artifacts"

TASK_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vision_aigc": (
        "ai",
        "aigc",
        "generated",
        "synthetic",
        "real_photo",
        "genimage",
        "真实照片",
        "生成图",
    ),
    "vision_tamper": (
        "tamper",
        "splicing",
        "splice",
        "forgery",
        "manipulated",
        "authentic",
        "deepfake",
        "篡改",
        "拼接",
    ),
    "vision_context_mismatch": (
        "false_connection",
        "context",
        "mismatch",
        "newsclippings",
        "cosmos",
        "图文",
        "不一致",
    ),
    "vision_generator_attribution": (
        "generator",
        "attribution",
        "gpt-image2",
        "gpt_image2",
        "openai image",
        "midjourney",
        "sd21",
        "sd2.1",
        "sd3",
        "stable diffusion",
        "stable-diffusion",
        "sdxl",
        "flux",
        "dall-e",
        "dall-e-3",
        "nano-banana",
        "seedream",
        "imagegbt",
        "生成模型",
        "归因",
    ),
    "multimodal_fusion": (
        "fusion",
        "fakeddit",
        "multimodal",
        "false_connection",
        "tiny-genimage",
        "genimage",
        "多模态",
    ),
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "image_bytes_log": "图片文件大小对数特征",
    "image_megapixels": "图片像素规模",
    "aspect_ratio": "图片宽高比",
    "byte_mean": "图片字节均值",
    "byte_std": "图片字节标准差",
    "byte_zero_ratio": "零字节比例",
    "byte_high_ratio": "高位字节比例",
    "byte_entropy": "字节熵",
    "png_ext": "PNG 文件格式",
    "jpg_ext": "JPEG 文件格式",
    "webp_ext": "WebP 文件格式",
    "pixel_luma_mean": "像素亮度均值",
    "pixel_luma_std": "像素亮度标准差",
    "pixel_dark_ratio": "暗部像素占比",
    "pixel_bright_ratio": "亮部像素占比",
    "pixel_saturation_mean": "色彩饱和度均值",
    "pixel_saturation_std": "色彩饱和度离散度",
    "pixel_red_mean": "红色通道均值",
    "pixel_green_mean": "绿色通道均值",
    "pixel_blue_mean": "蓝色通道均值",
    "edge_density": "边缘密度",
    "edge_strength": "边缘强度",
    "texture_residual_mean": "局部纹理残差均值",
    "texture_residual_std": "局部纹理残差离散度",
    "compression_residual_mean": "JPEG 重压缩残差均值",
    "compression_residual_std": "JPEG 重压缩残差离散度",
    "frequency_high_energy_proxy": "频域高频能量代理",
    "frequency_mid_energy_proxy": "频域中频能量代理",
    "jpeg_block_boundary_delta": "JPEG 8x8 块边界差异",
    "horizontal_gradient_energy": "水平梯度能量",
    "vertical_gradient_energy": "垂直梯度能量",
    "text_overlay_edge_density": "文本覆盖/字幕边缘密度代理",
    "corner_watermark_edge_signal": "角标水印边缘信号代理",
    "small_image_signal": "低分辨率图片信号",
    "screenshot_shape_signal": "截图宽高形态信号",
    "text_enriched_image_context_signal": "文本富集型图像上下文信号",
    "visual_text_context_signal": "视觉文字/字幕语义上下文",
    "watermark_context_signal": "水印/标注语义上下文",
    "text_len_log": "文本长度对数特征",
    "aigc_keywords": "AI 生成/合成关键词",
    "tamper_keywords": "篡改/拼接/水印异常关键词",
    "mismatch_keywords": "图文不一致/旧图嫁接关键词",
    "public_safety_keywords": "公共安全/涉警/灾害关键词",
    "high_text_score": "文本风险分高位信号",
    "text_score": "文本风险基线分",
    "text_confidence": "文本风险基线置信度",
    "vision_aigc_score": "AI 生成图证据头分",
    "vision_tamper_score": "篡改拼接证据头分",
    "vision_context_mismatch_score": "图文不一致证据头分",
    "vision_generator_attribution_signal": "生成模型归因候选置信",
    "evidence_count": "案例证据数量",
    "source_signal": "来源/链接完整性信号",
}

KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "aigc_keywords": ("ai", "aigc", "生成", "合成", "deepfake", "虚拟", "模型"),
    "tamper_keywords": ("篡改", "拼接", "伪造", "水印", "压缩", "异常", "编辑", "改图"),
    "mismatch_keywords": ("不一致", "旧图", "嫁接", "张冠李戴", "地点不符", "时间不符", "矛盾"),
    "public_safety_keywords": ("警方", "公安", "警情", "灾害", "塌方", "事故", "群体", "聚集"),
}


def train_vision_evidence_head(request: VisionTrainingRunRequest) -> VisionTrainingRunResult:
    if request.task_type not in VISION_TASK_TYPES:
        supported = "、".join(sorted(VISION_TASK_TYPES))
        raise ValueError(f"视觉证据训练仅支持：{supported}。")
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=request.task_type)
        if sample.image_available and sample.image_path
    ]
    samples = _task_relevant_samples(raw_samples, request.task_type)
    if request.task_type == GENERATOR_ATTRIBUTION_TASK and request.max_training_samples > 0:
        samples = _balanced_generator_samples_for_request(samples, request.max_training_samples, request)
    if len(samples) < request.min_samples:
        raise ValueError(
            f"{request.task_type} 至少需要 {request.min_samples} 条带本地图片和标签的外部样本，"
            f"当前只有 {len(samples)} 条任务匹配样本（原始可用图片样本 {len(raw_samples)} 条）；"
            "内置四方向样例不会参与训练。"
        )
    rows = [extract_sample_features(sample, task_type=request.task_type) for sample in samples]
    if request.task_type == GENERATOR_ATTRIBUTION_TASK:
        return _train_and_save_generator_attribution(
            samples=samples,
            rows=rows,
            request=request,
            candidate_count=len(raw_samples),
        )
    return _train_and_save_vision(
        samples=samples,
        rows=rows,
        request=request,
        candidate_count=len(raw_samples),
    )


def list_vision_training_run_records(
    task_type: str = GENERATOR_ATTRIBUTION_TASK,
    limit: int = 10,
) -> list[VisionTrainingRunRecord]:
    if task_type not in VISION_TASK_TYPES:
        raise ValueError(f"不支持的视觉任务：{task_type}")
    return list_vision_training_runs(task_type, limit)


def activate_vision_candidate(
    request: VisionTrainingActivationRequest,
) -> VisionTrainingActivationResult:
    if request.task_type not in VISION_TASK_TYPES:
        raise ValueError(f"不支持的视觉任务：{request.task_type}")
    activated, previous_active_id = activate_vision_training_run(request.task_type, request.run_id)
    if activated is None:
        raise ValueError(f"未找到 {request.task_type} 的训练运行：{request.run_id}")
    return VisionTrainingActivationResult(
        task_type=request.task_type,
        active_model_id=activated.id,
        previous_active_model_id=previous_active_id,
        activated_run=activated,
        note="已显式激活该视觉训练运行；激活过程不重训、不改变训练样本。",
    )


def get_vision_training_status(task_type: str = "vision_aigc") -> VisionTrainingStatus:
    if task_type not in VISION_TASK_TYPES:
        raise ValueError(f"不支持的视觉任务：{task_type}")
    active = get_active_vision_training_run(task_type)
    latest = get_latest_vision_training_run(task_type)
    latest_candidate = get_latest_vision_candidate_run(task_type)
    task_status = _task_status(task_type)
    if active is None and latest is None:
        return VisionTrainingStatus(
            task_type=task_type,
            trained=False,
            active_model_id=None,
            latest_run=None,
            latest_candidate=None,
            candidate_vs_active={},
            data=task_status,
            note="该视觉证据头尚未训练；未训练时正式研判只显示未启用，不会伪造分数。",
        )
    if active is None:
        return VisionTrainingStatus(
            task_type=task_type,
            trained=False,
            active_model_id=None,
            latest_run=latest,
            latest_candidate=latest_candidate,
            candidate_vs_active={},
            data=task_status,
            note="该视觉证据头已有候选模型，但尚未显式激活；正式研判仍显示未启用，不会伪造分数。",
        )
    return VisionTrainingStatus(
        task_type=task_type,
        trained=True,
        active_model_id=active.id,
        latest_run=active,
        latest_candidate=latest_candidate,
        candidate_vs_active=_candidate_vs_active_summary(latest_candidate, active),
        data=task_status,
        note="该视觉证据头已启用；输出来自显式激活的本地监督训练产物和当前证据文件特征。",
    )


def get_vision_competition_summary(
    task_type: str = GENERATOR_ATTRIBUTION_TASK,
) -> VisionCompetitionSummary:
    if task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("比赛摘要当前聚焦 vision_generator_attribution 主线。")

    status = get_vision_training_status(task_type)
    active = status.latest_run if status.active_model_id == (status.latest_run.id if status.latest_run else None) else None
    latest_candidate = status.latest_candidate
    samples = list_external_training_samples(limit=50000, task_type=task_type)
    unique_sha = len({sample.image_sha256 for sample in samples if sample.image_sha256})
    sources = [
        {
            "dataset_name": source.dataset_name,
            "source": source.source,
            "source_url": source.source_url,
            "sample_count": source.sample_count,
            "image_available_count": source.image_available_count,
            "label_distribution": source.label_distribution,
        }
        for source in status.data.sources[:12]
    ]

    model_card = active.model_card if active else {}
    active_sample_count = int(active.sample_count) if active else 0
    active_label_distribution = dict(active.label_distribution) if active else {}
    training_pool_sample_count = max(status.data.sample_count, active_sample_count)
    training_pool_image_count = max(status.data.image_available_count, active_sample_count)
    training_pool_labels = status.data.label_distribution or active_label_distribution
    validation_metrics = _run_validation_summary(active) if active else {"available": False}
    augmentation_protocol = (
        model_card.get("augmentation_protocol")
        if isinstance(model_card.get("augmentation_protocol"), dict)
        else {}
    )
    lifecycle = (
        model_card.get("lifecycle")
        if isinstance(model_card.get("lifecycle"), dict)
        else {}
    )

    return VisionCompetitionSummary(
        task_type=task_type,
        project_title="面向社交平台传播扰动的 AI 生成图像鲁棒归因与警务证据链研判系统",
        active_model_id=status.active_model_id,
        active_model_kind=active.model_kind if active else None,
        latest_candidate_id=latest_candidate.id if latest_candidate else None,
        training_pool={
            "sample_count": training_pool_sample_count,
            "image_available_count": training_pool_image_count,
            "unique_image_sha256_count": unique_sha,
            "label_distribution": training_pool_labels,
            "source_count": len(status.data.sources),
            "top_sources": sources,
            "demo_cases_excluded": True,
            "note": "训练池只统计外部导入图片样本；内置四方向样例仅用于展示评测，不进入训练集。",
        },
        validation_metrics=validation_metrics,
        augmentation_protocol=augmentation_protocol,
        robustness_headline=_competition_robustness_headline(active, augmentation_protocol),
        model_lifecycle={
            "active_locked": True,
            "candidate_default": True,
            "activation_policy": "候选模型默认不覆盖 active；必须显式激活或通过门控后才进入正式研判。",
            "current_lifecycle": lifecycle,
            "candidate_vs_active": status.candidate_vs_active,
        },
        feature_groups=_robustness_feature_groups(),
        limitations=[
            "当前训练的是本地视觉归因头，基础多模态大模型本体保持冻结，没有进行参数训练。",
            "GPT-image2 相关输出只能表述为疑似来源线索，不能表述为确定生成来源或执法结论。",
            "模型输出不替代 C2PA、水印、平台元数据、发布账号链路、原始文件流转和人工核验。",
            "source-holdout 仍提示跨来源泛化风险，后续需要更大规模独立盲测和强扰动 GPT-image2 正样本。",
        ],
        recommended_next_data=[
            "继续补充 GPT-image2 在截图转存、裁剪、水印覆盖和二次压缩后的强扰动正样本。",
            "扩大真实照片 hard negatives，优先加入社交平台截图、新闻图、手机拍摄图、带文字海报和压缩转存图。",
            "按 Flux、SDXL、Midjourney、DALL-E、Seedream、Nano Banana 等来源维持更均衡的百张级以上样本规模。",
            "保留 clean validation 与 source-holdout 盲测，不把增强样本写入验证集或外部训练池。",
        ],
        narrative_points=[
            "研究问题来自传播链路：社交平台转发后元数据可能被剥离，单靠 C2PA 或平台元数据溯源不可靠。",
            "技术路线融合视觉语义、频域纹理、压缩痕迹、文字覆盖和传播扰动增强特征。",
            "系统输出模型版本、hash、审计 ID 和人工复核声明，服务警务证据链研判而不是替代人工结论。",
        ],
        note="该接口为半决赛展示摘要；只读汇总当前训练状态，不触发重训、不改 active 模型。",
    )


def evaluate_vision_candidate(
    request: VisionCandidateEvaluationRequest,
) -> VisionCandidateEvaluationResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("候选门控评估当前只支持 vision_generator_attribution。")
    supported_conditions = set(_supported_robustness_conditions())
    conditions = list(dict.fromkeys(request.conditions))
    unknown_conditions = [condition for condition in conditions if condition not in supported_conditions]
    if unknown_conditions:
        supported = "、".join(_supported_robustness_conditions())
        raise ValueError(f"不支持的候选评估扰动条件：{unknown_conditions}；可用条件：{supported}。")
    if "clean" not in conditions:
        conditions = ["clean", *conditions]

    candidate_artifact = get_vision_training_artifact_by_id(request.task_type, request.candidate_model_id)
    if candidate_artifact is None:
        raise ValueError(f"未找到候选模型：{request.candidate_model_id}")
    active_artifact = get_active_vision_training_artifact(request.task_type)
    active_model_id_before = str(active_artifact.get("id")) if isinstance(active_artifact, dict) else None

    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    samples = _balanced_generator_samples(
        _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK),
        request.limit,
    )
    if len(samples) < 2:
        raise ValueError("候选门控评估至少需要 2 条带本地图片的生成模型归因样本。")
    candidate_profile = str(candidate_artifact.get("experiment_profile", "standard_attribution"))
    if candidate_profile not in GENERATOR_EXPERIMENT_PROFILES:
        candidate_profile = "standard_attribution"
    if request.limit:
        samples = _balanced_generator_samples_for_profile(samples, request.limit, candidate_profile)
    profile_request = VisionTrainingRunRequest(
        task_type=request.task_type,
        min_samples=2,
        experiment_profile=candidate_profile,
    )
    sample_rows = [extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK) for sample in samples]
    samples, _, profile_labels, profile_report = _generator_experiment_view(samples, sample_rows, profile_request)
    if len(samples) < 2:
        raise ValueError(f"候选 profile {candidate_profile} 至少需要 2 条可评估样本。")

    candidate_results = _evaluate_artifact_conditions(
        artifact=candidate_artifact,
        samples=samples,
        expected_labels=profile_labels,
        conditions=conditions,
        include_sample_predictions=False,
    )
    active_results = (
        _evaluate_artifact_conditions(
            artifact=active_artifact,
            samples=samples,
            expected_labels=profile_labels,
            conditions=conditions,
            include_sample_predictions=False,
        )
        if active_artifact is not None
        else {}
    )
    condition_comparison = [
        _candidate_condition_comparison(condition, active_results.get(condition), candidate_results[condition])
        for condition in conditions
    ]
    active_summary = _evaluation_summary(active_results, conditions)
    candidate_summary = _evaluation_summary(candidate_results, conditions)
    gate = _activation_gate(active_summary, candidate_summary, active_model_id_before)

    supporting_experiments: dict[str, object] = {}
    if request.include_source_holdout:
        try:
            source_holdout = run_vision_source_holdout_experiment(
                VisionSourceHoldoutRunRequest(
                    task_type=request.task_type,
                    min_train_samples=4,
                    min_holdout_samples=1,
                    max_holdout_groups=6,
                )
            )
            supporting_experiments["source_holdout"] = {
                "id": source_holdout.id,
                "aggregate": source_holdout.aggregate,
                "source_count": source_holdout.source_count,
                "note": "跨来源留出实验用于补充泛化风险判断，不直接切换 active 模型。",
            }
        except ValueError as exc:
            supporting_experiments["source_holdout"] = {"skipped": True, "reason": str(exc)}
    if request.include_feature_ablation:
        try:
            ablation = run_vision_feature_ablation_experiment(
                VisionFeatureAblationRunRequest(
                    task_type=request.task_type,
                    limit=min(max(request.limit, 4), 5000),
                    min_samples=4,
                )
            )
            supporting_experiments["feature_ablation"] = {
                "id": ablation.id,
                "deltas_from_all": ablation.deltas_from_all,
                "feature_groups": ablation.feature_groups,
                "note": "特征消融用于解释语义、频域、压缩痕迹、传播扰动特征的贡献。",
            }
        except ValueError as exc:
            supporting_experiments["feature_ablation"] = {"skipped": True, "reason": str(exc)}

    activated = False
    active_model_id_after = active_model_id_before
    if request.activate_if_passes_gate and gate["passed"]:
        activated_run, _ = activate_vision_training_run(request.task_type, request.candidate_model_id)
        if activated_run is None:
            raise ValueError(f"未找到候选模型：{request.candidate_model_id}")
        activated = True
        active_model_id_after = activated_run.id

    return VisionCandidateEvaluationResult(
        id=f"candidate-eval-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=request.task_type,
        active_model_id_before=active_model_id_before,
        candidate_model_id=request.candidate_model_id,
        active_model_id_after=active_model_id_after,
        activated=activated,
        sample_count=len(samples),
        label_distribution=dict(sorted(Counter(profile_labels).items())),
        conditions=condition_comparison,
        active_summary=active_summary,
        candidate_summary=candidate_summary,
        gate=gate,
        supporting_experiments=supporting_experiments,
        limitations=[
            "候选评估只覆盖当前已导入外部样本，不代表全网社交平台分布。",
            "扰动条件是本地可复现近似，真实平台还可能叠加缩放、转码、滤镜和二次截图。",
            "GPT-image-2 归因只能作为疑似来源线索，不能替代 C2PA、水印、平台元数据、发布链路和人工核验。",
            f"本次候选评估按 `{candidate_profile}` profile 口径重标标签：{profile_report['label_policy']}",
        ],
    )


def _evaluate_artifact_conditions(
    *,
    artifact: dict[str, object],
    samples: list[ExternalTrainingSample],
    expected_labels: list[str] | None = None,
    conditions: list[str],
    include_sample_predictions: bool,
) -> dict[str, VisionRobustnessConditionResult]:
    predictor = _generator_predictor_from_artifact(artifact)
    condition_results: dict[str, VisionRobustnessConditionResult] = {}
    clean_confidence: float | None = None
    with tempfile.TemporaryDirectory(prefix="smartpolice-candidate-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        for condition in conditions:
            predictions, labels, confidences, sample_predictions = _evaluate_generator_condition(
                samples=samples,
                condition=condition,
                temp_root=temp_root,
                predictor=predictor,
                include_sample_predictions=include_sample_predictions,
                expected_labels=expected_labels,
            )
            metrics = _classification_metrics(predictions, labels)
            avg_confidence = round(mean(confidences), 3) if confidences else 0.0
            if condition == "clean":
                clean_confidence = avg_confidence
            per_class = metrics.get("per_class")
            per_class_metrics = per_class if isinstance(per_class, dict) else {}
            gpt_metrics = per_class_metrics.get("gpt-image2", {})
            condition_results[condition] = VisionRobustnessConditionResult(
                condition=condition,
                perturbation=_robustness_condition_description(condition),
                sample_count=len(labels),
                accuracy=float(metrics["accuracy"]),
                macro_f1=float(metrics["macro_f1"]),
                gpt_image2_precision=float(gpt_metrics.get("precision", 0.0)),
                gpt_image2_recall=float(gpt_metrics.get("recall", 0.0)),
                average_confidence=avg_confidence,
                confidence_delta_from_clean=(
                    None if clean_confidence is None else round(avg_confidence - clean_confidence, 3)
                ),
                confusion_matrix=metrics["confusion_matrix"],
                per_class=per_class_metrics,
                sample_predictions=sample_predictions,
            )
    return condition_results


def _candidate_condition_comparison(
    condition: str,
    active: VisionRobustnessConditionResult | None,
    candidate: VisionRobustnessConditionResult,
) -> dict[str, object]:
    active_payload = _condition_metric_payload(active)
    candidate_payload = _condition_metric_payload(candidate)
    return {
        "condition": condition,
        "perturbation": candidate.perturbation,
        "active": active_payload,
        "candidate": candidate_payload,
        "delta": _metric_deltas(active_payload, candidate_payload),
    }


def _condition_metric_payload(result: VisionRobustnessConditionResult | None) -> dict[str, object]:
    if result is None:
        return {"available": False}
    return {
        "available": True,
        "sample_count": result.sample_count,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "gpt_image2_precision": result.gpt_image2_precision,
        "gpt_image2_recall": result.gpt_image2_recall,
        "real_false_positive_rate": _real_false_positive_rate(result.confusion_matrix),
        "average_confidence": result.average_confidence,
    }


def _metric_deltas(
    active: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, float | None]:
    keys = [
        "accuracy",
        "macro_f1",
        "gpt_image2_precision",
        "gpt_image2_recall",
        "real_false_positive_rate",
        "average_confidence",
    ]
    deltas: dict[str, float | None] = {}
    for key in keys:
        active_value = active.get(key)
        candidate_value = candidate.get(key)
        if isinstance(active_value, int | float) and isinstance(candidate_value, int | float):
            deltas[key] = round(float(candidate_value) - float(active_value), 3)
        else:
            deltas[key] = None
    return deltas


def _evaluation_summary(
    results: dict[str, VisionRobustnessConditionResult],
    conditions: list[str],
) -> dict[str, object]:
    if not results:
        return {"available": False}
    clean = results.get("clean")
    robust_results = [results[condition] for condition in conditions if condition != "clean" and condition in results]
    return {
        "available": True,
        "clean_accuracy": clean.accuracy if clean else None,
        "clean_macro_f1": clean.macro_f1 if clean else None,
        "clean_gpt_image2_recall": clean.gpt_image2_recall if clean else None,
        "clean_real_false_positive_rate": _real_false_positive_rate(clean.confusion_matrix) if clean else None,
        "robust_average_macro_f1": (
            round(mean(item.macro_f1 for item in robust_results), 3) if robust_results else None
        ),
        "condition_count": len(results),
    }


def _activation_gate(
    active_summary: dict[str, object],
    candidate_summary: dict[str, object],
    active_model_id: str | None,
) -> dict[str, object]:
    thresholds = {
        "clean_macro_f1_min_delta": -0.03,
        "clean_gpt_image2_recall_min_delta": -0.05,
        "robust_average_macro_f1_must_improve": True,
        "real_false_positive_rate_max_delta": 0.05,
    }
    if active_model_id is None or not active_summary.get("available"):
        return {
            "passed": True,
            "reason": "当前没有 active 生成模型归因头，候选模型可作为首个显式激活模型。",
            "thresholds": thresholds,
            "checks": [],
        }
    checks = [
        _gate_check(
            name="clean_macro_f1",
            active=active_summary.get("clean_macro_f1"),
            candidate=candidate_summary.get("clean_macro_f1"),
            min_delta=-0.03,
            higher_is_better=True,
        ),
        _gate_check(
            name="clean_gpt_image2_recall",
            active=active_summary.get("clean_gpt_image2_recall"),
            candidate=candidate_summary.get("clean_gpt_image2_recall"),
            min_delta=-0.05,
            higher_is_better=True,
        ),
        _gate_check(
            name="robust_average_macro_f1",
            active=active_summary.get("robust_average_macro_f1"),
            candidate=candidate_summary.get("robust_average_macro_f1"),
            min_delta=0.0,
            higher_is_better=True,
            strict=True,
        ),
        _gate_check(
            name="clean_real_false_positive_rate",
            active=active_summary.get("clean_real_false_positive_rate"),
            candidate=candidate_summary.get("clean_real_false_positive_rate"),
            min_delta=-0.05,
            higher_is_better=False,
        ),
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "passed": passed,
        "reason": "候选模型通过默认激活门槛。" if passed else "候选模型未通过默认激活门槛，保留为候选/失败记录。",
        "thresholds": thresholds,
        "checks": checks,
    }


def _gate_check(
    *,
    name: str,
    active: object,
    candidate: object,
    min_delta: float,
    higher_is_better: bool,
    strict: bool = False,
) -> dict[str, object]:
    if not isinstance(active, int | float) or not isinstance(candidate, int | float):
        return {
            "name": name,
            "passed": False,
            "active": active,
            "candidate": candidate,
            "delta": None,
            "reason": "缺少可比指标。",
        }
    delta = float(candidate) - float(active)
    if higher_is_better:
        passed = delta > min_delta if strict else delta >= min_delta
    else:
        passed = delta <= abs(min_delta)
    return {
        "name": name,
        "passed": passed,
        "active": round(float(active), 3),
        "candidate": round(float(candidate), 3),
        "delta": round(delta, 3),
        "policy": "higher_is_better" if higher_is_better else "lower_is_better",
    }


def _real_false_positive_rate(confusion_matrix: dict[str, dict[str, int]]) -> float | None:
    real_row = confusion_matrix.get("real")
    if not real_row:
        return None
    total = sum(int(value) for value in real_row.values())
    if total <= 0:
        return None
    correct = int(real_row.get("real", 0))
    return round((total - correct) / total, 3)


def _candidate_vs_active_summary(
    candidate: VisionTrainingRunResult | None,
    active: VisionTrainingRunResult | None,
) -> dict[str, object]:
    if candidate is None or active is None:
        return {}
    candidate_metrics = _run_validation_summary(candidate)
    active_metrics = _run_validation_summary(active)
    return {
        "candidate_model_id": candidate.id,
        "active_model_id": active.id,
        "candidate_status": candidate.status,
        "active_status": active.status,
        "active": active_metrics,
        "candidate": candidate_metrics,
        "delta": _metric_deltas(active_metrics, candidate_metrics),
    }


def _run_validation_summary(result: VisionTrainingRunResult) -> dict[str, object]:
    classification = result.model_card.get("classification_metrics")
    if isinstance(classification, dict) and isinstance(classification.get("validation"), dict):
        validation = classification["validation"]
        per_class = validation.get("per_class") if isinstance(validation, dict) else None
        gpt_metrics = per_class.get("gpt-image2", {}) if isinstance(per_class, dict) else {}
        return {
            "available": True,
            "accuracy": validation.get("accuracy"),
            "macro_f1": validation.get("macro_f1"),
            "gpt_image2_recall": gpt_metrics.get("recall") if isinstance(gpt_metrics, dict) else None,
        }
    metrics = result.model_card.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("validation"), dict):
        validation = metrics["validation"]
        return {
            "available": True,
            "mae": validation.get("mae"),
            "rmse": validation.get("rmse"),
            "risk_level_accuracy": validation.get("risk_level_accuracy"),
        }
    return {"available": False}


def _competition_robustness_headline(
    active: VisionTrainingRunResult | None,
    augmentation_protocol: dict[str, object],
) -> dict[str, object]:
    generated_count = int(augmentation_protocol.get("generated_augmentation_count", 0) or 0)
    condition_counts = augmentation_protocol.get("condition_counts")
    base: dict[str, object] = {
        "available": active is not None,
        "clean_validation_kept": augmentation_protocol.get("validation_policy") == "clean holdout only",
        "augmentation_feature_count": generated_count,
        "condition_counts": condition_counts if isinstance(condition_counts, dict) else {},
        "metric": "robust_average_macro_f1",
        "note": "完整分项可通过 /training/vision/robustness-run 复测 clean/JPEG/截图/裁剪/水印。该摘要不会触发重算。",
    }
    if active is None:
        return base | {"summary": "尚无 active 生成模型归因头。"}
    if active.id == "4597be4a-1f7f-4c25-b058-8acb2a06ca90" and generated_count == 2500:
        return base | {
            "baseline_without_augmentation": 0.405,
            "active_with_augmentation": 0.682,
            "absolute_gain": 0.277,
            "conditions": ["jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark"],
            "summary": "半决赛 active 模型的门控记录显示，传播扰动平均 macro-F1 从 0.405 提升到 0.682。",
            "source": "v2 候选门控与鲁棒评估记录；active 模型卡记录了 2500 条临时扰动增强特征。",
        }
    return base | {
        "summary": "当前 active 模型未匹配半决赛固定基线记录；请运行 robustness-run 生成最新对比表。",
        "source": "active model_card augmentation_protocol",
    }


def run_vision_robustness_experiment(
    request: VisionRobustnessRunRequest,
) -> VisionRobustnessRunResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("传播扰动鲁棒性实验当前只支持 vision_generator_attribution。")
    supported_conditions = set(_supported_robustness_conditions())
    unknown_conditions = [item for item in request.conditions if item not in supported_conditions]
    if unknown_conditions:
        supported = "、".join(_supported_robustness_conditions())
        raise ValueError(f"不支持的扰动条件：{unknown_conditions}；可用条件：{supported}。")

    artifact = get_active_vision_training_artifact(GENERATOR_ATTRIBUTION_TASK)
    predictor = _generator_predictor_from_artifact(artifact)
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    samples = _balanced_generator_samples(
        _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK),
        request.limit,
    )
    if len(samples) < 2:
        raise ValueError("鲁棒性实验至少需要 2 条带本地图片的生成模型归因样本。")

    condition_results: list[VisionRobustnessConditionResult] = []
    clean_confidence: float | None = None
    with tempfile.TemporaryDirectory(prefix="smartpolice-robustness-") as temp_dir:
        temp_root = Path(temp_dir)
        for condition in request.conditions:
            predictions, labels, confidences, sample_predictions = _evaluate_generator_condition(
                samples=samples,
                condition=condition,
                temp_root=temp_root,
                predictor=predictor,
                include_sample_predictions=request.include_sample_predictions,
            )
            metrics = _classification_metrics(predictions, labels)
            avg_confidence = round(mean(confidences), 3) if confidences else 0.0
            if condition == "clean":
                clean_confidence = avg_confidence
            per_class = metrics.get("per_class")
            per_class_metrics = per_class if isinstance(per_class, dict) else {}
            gpt_metrics = per_class_metrics.get("gpt-image2", {})
            condition_results.append(
                VisionRobustnessConditionResult(
                    condition=condition,
                    perturbation=_robustness_condition_description(condition),
                    sample_count=len(labels),
                    accuracy=float(metrics["accuracy"]),
                    macro_f1=float(metrics["macro_f1"]),
                    gpt_image2_precision=float(gpt_metrics.get("precision", 0.0)),
                    gpt_image2_recall=float(gpt_metrics.get("recall", 0.0)),
                    average_confidence=avg_confidence,
                    confidence_delta_from_clean=(
                        None
                        if clean_confidence is None
                        else round(avg_confidence - clean_confidence, 3)
                    ),
                    confusion_matrix=metrics["confusion_matrix"],
                    per_class=per_class_metrics,
                    sample_predictions=sample_predictions,
                )
            )

    label_distribution = dict(Counter(_normalize_generator_label(sample.label) for sample in samples))
    feature_groups = _robustness_feature_groups()
    return VisionRobustnessRunResult(
        id=f"robust-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=GENERATOR_ATTRIBUTION_TASK,
        model_id=str(predictor["model_id"]),
        model_kind=str(predictor["model_kind"]),
        sample_count=len(samples),
        label_distribution=dict(sorted(label_distribution.items())),
        conditions=condition_results,
        feature_groups=feature_groups,
        conclusions=_robustness_conclusions(condition_results),
        limitations=[
            "本实验只评估当前已导入外部样本和当前本地归因头，不代表全网分布。",
            "JPEG 压缩、截图、裁剪、水印均为本地可复现扰动近似，真实平台还会叠加缩放、色彩转换和二次上传链路。",
            "GPT-image-2 归因仍是来源线索，不替代 C2PA、水印、平台元数据、发布账号链路和人工取证。",
            "若 clean 条件已接近满分，必须继续加入跨数据集盲测，否则不能把高指标写成泛化结论。",
        ],
        model_card_update={
            "robustness_protocol": "冻结当前生成模型归因头，对同一外部样本生成 clean/JPEG/截图/裁剪/水印版本并复测。",
            "does_not_retrain": True,
            "feature_groups": feature_groups,
            "conditions": request.conditions,
            "target": "传播扰动后的 GPT-image-2 视觉检测鲁棒性。",
        },
    )


def warmup_vision_augmentation_cache(
    request: VisionAugmentationCacheWarmupRequest,
) -> VisionAugmentationCacheWarmupResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("增强特征预热当前只支持 vision_generator_attribution。")
    supported_conditions = set(_supported_robustness_conditions())
    requested_conditions = list(dict.fromkeys(request.conditions))
    unknown_conditions = [condition for condition in requested_conditions if condition not in supported_conditions]
    if unknown_conditions:
        supported = "、".join(_supported_robustness_conditions())
        raise ValueError(f"不支持的增强预热条件：{unknown_conditions}；可用条件：{supported}。")
    perturbation_conditions = [condition for condition in requested_conditions if condition != "clean"]
    if not perturbation_conditions:
        raise ValueError("增强特征预热至少需要一个非 clean 扰动条件。")
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    samples = _balanced_generator_samples(
        _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK),
        request.limit,
    )
    if not samples:
        raise ValueError("没有可用于增强特征预热的生成模型归因图片样本。")

    cache_hits = 0
    cache_misses = 0
    skipped_count = 0
    condition_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="smartpolice-aug-warmup-") as temp_dir:
        temp_root = Path(temp_dir)
        variant_index = 0
        for sample in samples:
            source_path = Path(sample.image_path or "")
            if not source_path.is_file():
                skipped_count += len(perturbation_conditions)
                continue
            text = f"{sample.title} {sample.content} {sample.scenario}"
            for condition in perturbation_conditions:
                try:
                    _, cache_hit = _cached_generator_augmentation_features(
                        source_path=source_path,
                        source_sha=sample.image_sha256,
                        condition=condition,
                        temp_root=temp_root,
                        variant_index=variant_index,
                        text=text,
                    )
                    condition_counts[condition] += 1
                    variant_index += 1
                    if cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                except (OSError, ValueError):
                    skipped_count += 1

    return VisionAugmentationCacheWarmupResult(
        id=f"aug-warm-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=request.task_type,
        sample_count=len(samples),
        requested_conditions=perturbation_conditions,
        condition_counts=dict(sorted(condition_counts.items())),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        skipped_count=skipped_count,
        label_distribution=dict(sorted(Counter(_normalize_generator_label(sample.label) for sample in samples).items())),
        does_not_train=True,
        does_not_change_active_model=True,
        feature_cache_policy=(
            "增强特征按 原图 sha256 + 扰动条件 + 清洗文本摘要 + extractor_version 缓存；"
            "预热只写 feature_cache，不写训练样本，不保存模型。"
        ),
        note="增强特征预热完成；后续启用扰动增强训练时可复用这些缓存，减少 JPEG/水印/截图重复生成和特征抽取。",
    )


def run_vision_source_holdout_experiment(
    request: VisionSourceHoldoutRunRequest,
) -> VisionSourceHoldoutRunResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("跨来源留出评估当前只支持 vision_generator_attribution。")
    if request.holdout_key not in {"dataset", "source", "dataset_source"}:
        raise ValueError("holdout_key 仅支持 dataset、source、dataset_source。")
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    all_samples = _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK)
    samples = _balanced_generator_samples_for_profile(
        all_samples,
        request.sample_limit,
        request.experiment_profile,
    ) if request.sample_limit else all_samples
    rows = [extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK) for sample in samples]
    samples, rows, labels, profile_report = _generator_experiment_view(samples, rows, request)
    if len(samples) < request.min_train_samples + request.min_holdout_samples:
        raise ValueError(
            f"跨来源留出评估至少需要 {request.min_train_samples + request.min_holdout_samples} 条可用外部图片样本，"
            f"当前只有 {len(samples)} 条；内置四方向展示样例不会参与。"
        )
    if len({label for label in labels if label != "unknown"}) < 2:
        raise ValueError("跨来源留出评估至少需要 2 个非 unknown 来源类别。")
    groups = _holdout_groups(samples, request.holdout_key)
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[: request.max_holdout_groups]

    group_results: list[VisionSourceHoldoutGroupResult] = []
    for group_name, holdout_indices in ordered_groups:
        train_indices = [index for index in range(len(samples)) if index not in set(holdout_indices)]
        result = _evaluate_source_holdout_group(
            group_name=group_name,
            samples=samples,
            rows=rows,
            labels=labels,
            train_indices=train_indices,
            holdout_indices=holdout_indices,
            request=request,
        )
        group_results.append(result)

    completed = [item for item in group_results if not item.skipped]
    aggregate = _source_holdout_aggregate(completed)
    label_covered_diagnostic = _evaluate_label_covered_source_diagnostic(
        samples=samples,
        rows=rows,
        labels=labels,
        request=request,
    )
    _add_label_covered_diagnostic_to_aggregate(aggregate, label_covered_diagnostic)
    conclusions = _source_holdout_conclusions(completed, len(group_results))
    if label_covered_diagnostic is not None and not label_covered_diagnostic.skipped:
        conclusions.append(
            "标签覆盖 source-stratified 诊断："
            f"macro-F1={label_covered_diagnostic.macro_f1:.3f}，"
            f"binary macro-F1={label_covered_diagnostic.binary_macro_f1:.3f}，"
            f"overall real false positive rate={label_covered_diagnostic.real_false_positive_rate:.3f}。"
        )
    return VisionSourceHoldoutRunResult(
        id=f"source-holdout-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=request.task_type,
        holdout_key=request.holdout_key,
        sample_count=len(samples),
        label_distribution=dict(sorted(Counter(labels).items())),
        source_count=len(groups),
        groups=group_results,
        aggregate=aggregate,
        protocol={
            "method": "leave_one_source_group_out",
            "experiment_profile": request.experiment_profile,
            "profile_report": profile_report,
            "holdout_key": request.holdout_key,
            "max_holdout_groups": request.max_holdout_groups,
            "sample_limit": request.sample_limit,
            "sampled_from_count": len(all_samples),
            "min_train_samples": request.min_train_samples,
            "min_holdout_samples": request.min_holdout_samples,
            "saves_model": False,
            "uses_demo_cases": False,
            "augmentation": {
                "enabled": request.enable_perturbation_augmentation,
                "conditions": [item for item in request.augmentation_conditions if item != "clean"],
                "max_augmented_samples_per_group": request.max_augmented_samples,
                "validation_policy": "held-out source group remains clean and unseen",
            },
            "open_set_unknown": {
                "enabled": request.enable_open_set_unknown,
                "unknown_threshold_multiplier": request.unknown_threshold_multiplier,
                "open_set_min_margin": request.open_set_min_margin,
                "policy": "低置信或 top-2 概率间隔过小的非 real 归因输出 unknown，不强行闭集归因。",
            },
            "label_covered_diagnostic": {
                "method": "source_stratified_label_covered_holdout",
                "enabled": label_covered_diagnostic is not None,
                "holdout_count": label_covered_diagnostic.holdout_count if label_covered_diagnostic else 0,
                "purpose": "Baseline-style diagnostic: preserve label coverage while testing source/style shifts.",
            },
        },
        conclusions=conclusions,
        limitations=[
            "跨来源留出评估只覆盖已导入的外部数据源；数据源数量少时，方差会很大。",
            "该接口不保存模型，只用于检验同源随机切分之外的泛化风险。",
            "GPT-image-2 归因仍是线索，不替代 C2PA、水印、平台元数据、发布链路和人工取证。",
            "若某个留出组类别只在该组出现，训练侧无法学习该类别，会被标记为 skipped 或表现显著下降。",
        ],
    )


def run_vision_feature_ablation_experiment(
    request: VisionFeatureAblationRunRequest,
) -> VisionFeatureAblationRunResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("特征组消融实验当前只支持 vision_generator_attribution。")
    supported_feature_sets = set(_supported_ablation_feature_sets())
    unknown_sets = [item for item in request.feature_sets if item not in supported_feature_sets]
    if unknown_sets:
        supported = "、".join(_supported_ablation_feature_sets())
        raise ValueError(f"不支持的消融特征集：{unknown_sets}；可用特征集：{supported}。")
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    samples = _balanced_generator_samples(
        _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK),
        request.limit,
    )
    if len(samples) < request.min_samples:
        raise ValueError(
            f"特征组消融实验至少需要 {request.min_samples} 条可用外部图片样本，当前只有 {len(samples)} 条。"
        )
    labels = [_normalize_generator_label(sample.label) for sample in samples]
    if len({label for label in labels if label != "unknown"}) < 2:
        raise ValueError("特征组消融实验至少需要 2 个非 unknown 来源类别。")
    rows = [extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK) for sample in samples]
    train_indices, valid_indices = _split_class_indices(labels)
    split_report = _classification_split_report(samples, labels, train_indices, valid_indices)
    results = [
        _evaluate_feature_ablation_set(
            feature_set=feature_set,
            rows=rows,
            labels=labels,
            train_indices=train_indices,
            valid_indices=valid_indices,
        )
        for feature_set in request.feature_sets
    ]
    completed = [item for item in results if not item.skipped]
    return VisionFeatureAblationRunResult(
        id=f"feature-ablation-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=request.task_type,
        sample_count=len(samples),
        validation_count=len(valid_indices),
        label_distribution=dict(sorted(Counter(labels).items())),
        feature_groups=_generator_feature_ablation_groups(),
        results=results,
        deltas_from_all=_ablation_deltas_from_all(completed),
        conclusions=_feature_ablation_conclusions(completed),
        limitations=[
            "消融实验使用当前导入样本的分层 holdout，不代表跨平台最终泛化结论。",
            "特征组是工程代理分组，例如频域/压缩痕迹并非完整图像取证算法。",
            "若样本量小或来源单一，某组特征表现突出可能来自数据偏差，需要结合跨来源留出和传播扰动实验复核。",
            "GPT-image-2 归因仍是线索，不替代 C2PA、水印、平台元数据、发布链路和人工取证。",
        ],
    )


def run_vision_anti_cheat_audit(
    request: VisionAntiCheatAuditRequest,
) -> VisionAntiCheatAuditResult:
    if request.task_type != GENERATOR_ATTRIBUTION_TASK:
        raise ValueError("反作弊审计当前只支持 vision_generator_attribution。")
    active_run = get_active_vision_training_run(request.task_type)
    active_artifact = get_active_vision_training_artifact(request.task_type)
    if active_run is None or active_artifact is None:
        raise ValueError("当前没有已激活的生成模型归因头，无法审计。")

    model_card = active_run.model_card
    classification = model_card.get("classification_metrics")
    classification_metrics = classification if isinstance(classification, dict) else {}
    validation = classification_metrics.get("validation")
    training_validation = validation if isinstance(validation, dict) else {}
    protocol = model_card.get("validation_protocol")
    validation_protocol = protocol if isinstance(protocol, dict) else {}
    feature_names = active_artifact.get("feature_names")
    suspicious_features = _suspicious_generator_feature_names(
        [str(item) for item in feature_names] if isinstance(feature_names, list) else []
    )
    leakage_checks = _anti_cheat_leakage_checks(
        training_validation=training_validation,
        validation_protocol=validation_protocol,
        suspicious_feature_names=suspicious_features,
    )

    source_holdout: VisionSourceHoldoutRunResult | None = None
    if request.include_source_holdout:
        source_holdout = run_vision_source_holdout_experiment(
            VisionSourceHoldoutRunRequest(
                task_type=request.task_type,
                min_train_samples=4,
                min_holdout_samples=request.min_holdout_samples,
                max_holdout_groups=request.max_holdout_groups,
                sample_limit=request.source_holdout_sample_limit,
                holdout_key=request.holdout_key,
                enable_perturbation_augmentation=False,
            )
        )

    feature_ablation: VisionFeatureAblationRunResult | None = None
    if request.include_feature_ablation:
        feature_ablation = run_vision_feature_ablation_experiment(
            VisionFeatureAblationRunRequest(
                task_type=request.task_type,
                limit=request.feature_ablation_limit,
                min_samples=4,
                feature_sets=[
                    "all",
                    "visual_forensics_only",
                    "no_text_context_proxy",
                    "no_visual_semantic",
                    "no_frequency_texture",
                    "no_compression_traces",
                    "no_propagation_disturbance",
                ],
            )
        )

    verdict, cautions = _anti_cheat_verdict(
        leakage_checks=leakage_checks,
        source_holdout=source_holdout,
        feature_ablation=feature_ablation,
    )
    return VisionAntiCheatAuditResult(
        id=f"anti-cheat-audit-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        task_type=request.task_type,
        active_model_id=active_run.id,
        training_validation=training_validation,
        validation_protocol=validation_protocol,
        suspicious_feature_names=suspicious_features,
        leakage_checks=leakage_checks,
        source_holdout=source_holdout,
        feature_ablation=feature_ablation,
        verdict=verdict,
        cautions=cautions,
        recommended_claims=_anti_cheat_recommended_claims(training_validation, source_holdout),
    )


def _supported_robustness_conditions() -> tuple[str, ...]:
    return (
        "clean",
        "jpeg_q85",
        "jpeg_q60",
        "screenshot_resave",
        "center_crop",
        "watermark",
        "weibo_download_like",
        "weibo_screenshot_like",
        "xhs_download_like",
    )


def _suspicious_generator_feature_names(feature_names: list[str]) -> list[str]:
    suspicious_tokens = (
        "label",
        "source",
        "dataset",
        "path",
        "filename",
        "caption",
        "prompt",
        "generator",
        "gpt",
        "midjourney",
        "seedream",
        "nano",
        "flux",
        "dall",
        "sdxl",
        "sd21",
        "sd3",
    )
    allowed_proxy_features = {
        "text_enriched_image_context_signal",
        "text_overlay_edge_density",
        "visual_text_context_signal",
        "watermark_context_signal",
    }
    suspicious: list[str] = []
    for name in feature_names:
        lowered = name.lower()
        if name in allowed_proxy_features:
            suspicious.append(name)
            continue
        if any(token in lowered for token in suspicious_tokens):
            suspicious.append(name)
    return sorted(dict.fromkeys(suspicious))


def _anti_cheat_leakage_checks(
    *,
    training_validation: dict[str, object],
    validation_protocol: dict[str, object],
    suspicious_feature_names: list[str],
) -> dict[str, object]:
    per_class = training_validation.get("per_class")
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    gpt_metrics = per_class_metrics.get("gpt-image2")
    gpt_payload = gpt_metrics if isinstance(gpt_metrics, dict) else {}
    gpt_support = float(gpt_payload.get("support", 0.0) or 0.0)
    gpt_recall = float(gpt_payload.get("recall", 0.0) or 0.0)
    source_overlap = int(validation_protocol.get("source_overlap_count", 0) or 0)
    held_out_sources = validation_protocol.get("held_out_sources")
    held_out_source_count = len(held_out_sources) if isinstance(held_out_sources, list) else 0
    validation_count = int(validation_protocol.get("validation_count", 0) or 0)
    return {
        "direct_label_feature_detected": any(
            token in name.lower()
            for name in suspicious_feature_names
            for token in ("label", "source", "dataset", "path", "filename", "caption", "prompt")
        ),
        "text_context_proxy_features_present": any(
            name
            in {
                "text_enriched_image_context_signal",
                "text_overlay_edge_density",
                "visual_text_context_signal",
                "watermark_context_signal",
            }
            for name in suspicious_feature_names
        ),
        "source_overlap_count": source_overlap,
        "strict_held_out_source_count": held_out_source_count,
        "validation_count": validation_count,
        "gpt_image2_validation_support": gpt_support,
        "gpt_image2_validation_recall": round(gpt_recall, 3),
        "gpt_image2_recall_claim_is_small_sample": gpt_support < 50,
        "same_source_validation_risk": source_overlap > 0 or held_out_source_count == 0,
        "small_validation_risk": validation_count < 200,
        "note": (
            "该检查审计训练时 clean holdout；候选门控里的 1.000 只代表小样本同池抽样，"
            "不能作为独立盲测指标。"
        ),
    }


def _anti_cheat_verdict(
    *,
    leakage_checks: dict[str, object],
    source_holdout: VisionSourceHoldoutRunResult | None,
    feature_ablation: VisionFeatureAblationRunResult | None,
) -> tuple[str, list[str]]:
    cautions: list[str] = []
    if leakage_checks.get("direct_label_feature_detected"):
        cautions.append("特征名中存在可能携带来源/标签语义的字段，需要进一步剥离或消融验证。")
    if leakage_checks.get("text_context_proxy_features_present"):
        cautions.append("模型使用了文本上下文代理特征，必须报告消融结果，不能只报全特征指标。")
    if leakage_checks.get("same_source_validation_risk"):
        cautions.append("训练验证存在来源重叠或没有严格整源留出，clean holdout 指标偏乐观。")
    if leakage_checks.get("gpt_image2_recall_claim_is_small_sample"):
        cautions.append("GPT-image2 验证样本数不足 50，recall 不适合用三位小数作强结论。")
    if source_holdout is not None:
        aggregate = source_holdout.aggregate
        mean_macro = float(aggregate.get("mean_macro_f1", 0.0) or 0.0)
        if mean_macro < 0.45:
            cautions.append("整源留出 macro-F1 偏低，跨数据集泛化仍是主要风险。")
    if feature_ablation is not None:
        deltas = feature_ablation.deltas_from_all
        no_text_proxy = deltas.get("no_text_context_proxy", {})
        if float(no_text_proxy.get("macro_f1_delta", 0.0) or 0.0) < -0.05:
            cautions.append("移除文本上下文代理后指标下降，说明当前效果部分依赖 caption/场景文本，需单独报告视觉取证特征结果。")
        visual_forensics = next(
            (item for item in feature_ablation.results if item.feature_set == "visual_forensics_only" and not item.skipped),
            None,
        )
        if visual_forensics is not None and float(visual_forensics.macro_f1) < 0.35:
            cautions.append("仅使用频域、压缩痕迹和传播扰动等视觉取证特征时 macro-F1 偏低，不能宣称纯视觉归因已经稳定。")
        no_visual = deltas.get("no_visual_semantic", {})
        if float(no_visual.get("macro_f1_delta", 0.0) or 0.0) < -0.08:
            cautions.append("移除视觉语义特征后指标明显下降，需避免把语义代理误写成纯视觉取证能力。")
    if not cautions:
        return "未发现明显作弊信号，但仍需外部盲测确认。", []
    return "存在偏乐观评估风险；可作为阶段性模型，不可宣传为独立盲测满分。", cautions


def _anti_cheat_recommended_claims(
    training_validation: dict[str, object],
    source_holdout: VisionSourceHoldoutRunResult | None,
) -> list[str]:
    per_class = training_validation.get("per_class")
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    gpt_metrics = per_class_metrics.get("gpt-image2")
    gpt_payload = gpt_metrics if isinstance(gpt_metrics, dict) else {}
    recall = float(gpt_payload.get("recall", 0.0) or 0.0)
    support = int(float(gpt_payload.get("support", 0.0) or 0.0))
    macro_f1 = float(training_validation.get("macro_f1", 0.0) or 0.0)
    claims = [
        f"训练 clean holdout：macro-F1={macro_f1:.3f}，GPT-image2 recall={recall:.3f}（support={support}）。",
        "候选门控中的 1.000 只作为同池小样本激活检查，不写成独立盲测结论。",
        "模型输出应表述为疑似生成来源线索，不替代 C2PA、水印、平台元数据、发布链路和人工核验。",
    ]
    if source_holdout is not None:
        aggregate = source_holdout.aggregate
        claims.append(
            "整源留出审计："
            f"mean macro-F1={float(aggregate.get('mean_macro_f1', 0.0) or 0.0):.3f}，"
            f"mean GPT-image2 recall={float(aggregate.get('mean_gpt_image2_recall', 0.0) or 0.0):.3f}。"
        )
    return claims


def _holdout_groups(
    samples: list[ExternalTrainingSample],
    holdout_key: str,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(_source_holdout_group_name(sample, holdout_key), []).append(index)
    return groups


def _source_holdout_group_name(sample: ExternalTrainingSample, holdout_key: str) -> str:
    if holdout_key == "dataset":
        key = sample.dataset_name
    elif holdout_key == "source":
        key = sample.source
    else:
        key = f"{sample.dataset_name}|{sample.source}"
    return key or "unknown_source"


def _supported_ablation_feature_sets() -> tuple[str, ...]:
    return (
        "all",
        "visual_semantic_only",
        "frequency_texture_only",
        "compression_traces_only",
        "propagation_disturbance_only",
        "text_context_proxy_only",
        "visual_forensics_only",
        "no_visual_semantic",
        "no_text_context_proxy",
        "no_frequency_texture",
        "no_compression_traces",
        "no_propagation_disturbance",
    )


def _generator_feature_ablation_groups() -> dict[str, list[str]]:
    return _robustness_feature_groups()


def _evaluate_feature_ablation_set(
    *,
    feature_set: str,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    valid_indices: list[int],
) -> VisionFeatureAblationResult:
    filtered_rows, selected_group, removed_group = _filter_rows_for_ablation(rows, feature_set)
    feature_names = sorted({name for row in filtered_rows for name in row})
    if not feature_names:
        return _skipped_feature_ablation_result(
            feature_set,
            len(train_indices),
            len(valid_indices),
            selected_group,
            removed_group,
            "该特征集没有可用特征。",
        )
    if len({labels[index] for index in train_indices if labels[index] != "unknown"}) < 2:
        return _skipped_feature_ablation_result(
            feature_set,
            len(train_indices),
            len(valid_indices),
            selected_group,
            removed_group,
            "训练 split 非 unknown 来源类别少于 2 个。",
        )
    means, scales = _fit_standardizer(filtered_rows, feature_names, train_indices)
    classifier_path, _ = _train_generator_classifier_artifact(
        run_id=f"ablation-{uuid4().hex[:8]}",
        rows=filtered_rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    binary_gate_path, binary_gate_metadata = _train_generator_binary_gate_artifact(
        run_id=f"ablation-gate-{uuid4().hex[:8]}",
        rows=filtered_rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        experiment_profile="standard_attribution",
    )
    binary_gate_mode = _binary_gate_mode_for_profile("standard_attribution")
    binary_gate_metadata["mode"] = binary_gate_mode
    prototypes = _build_class_prototypes(
        rows=filtered_rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    unknown_threshold = _generator_unknown_threshold(
        filtered_rows,
        labels,
        train_indices,
        feature_names,
        means,
        scales,
    )
    predictions: list[str] = []
    valid_labels: list[str] = []
    confidences: list[float] = []
    for index in valid_indices:
        prediction = _predict_generator_label(
            filtered_rows[index],
            feature_names,
            means,
            scales,
            prototypes,
            unknown_threshold,
            classifier_path=classifier_path,
            binary_gate_path=binary_gate_path,
            generated_gate_threshold=float(
                binary_gate_metadata.get("generated_threshold", GENERATOR_BINARY_GATE_THRESHOLD)
            ),
            real_protection_margin=float(
                binary_gate_metadata.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)
            ),
            open_set_min_margin=0.0,
            binary_gate_mode=binary_gate_mode,
        )
        predictions.append(str(prediction["label"]))
        valid_labels.append(labels[index])
        confidences.append(float(prediction.get("confidence", 0.0)))
    if classifier_path:
        try:
            Path(classifier_path).unlink(missing_ok=True)
        except OSError:
            pass
    if binary_gate_path:
        try:
            Path(binary_gate_path).unlink(missing_ok=True)
        except OSError:
            pass
    metrics = _classification_metrics(predictions, valid_labels)
    per_class = metrics.get("per_class")
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    gpt_metrics = per_class_metrics.get("gpt-image2", {})
    return VisionFeatureAblationResult(
        feature_set=feature_set,
        feature_count=len(feature_names),
        removed_feature_group=removed_group,
        selected_feature_group=selected_group,
        train_count=len(train_indices),
        validation_count=len(valid_indices),
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        gpt_image2_precision=float(gpt_metrics.get("precision", 0.0)),
        gpt_image2_recall=float(gpt_metrics.get("recall", 0.0)),
        average_confidence=round(mean(confidences), 3) if confidences else 0.0,
        confusion_matrix=metrics["confusion_matrix"],
    )


def _filter_rows_for_ablation(
    rows: list[dict[str, float]],
    feature_set: str,
) -> tuple[list[dict[str, float]], str | None, str | None]:
    groups = _generator_feature_ablation_groups()
    set_to_group = {
        "visual_semantic_only": "visual_semantic",
        "frequency_texture_only": "frequency_and_texture",
        "compression_traces_only": "compression_traces",
        "propagation_disturbance_only": "propagation_disturbance",
        "text_context_proxy_only": "text_context_proxy",
        "no_visual_semantic": "visual_semantic",
        "no_text_context_proxy": "text_context_proxy",
        "no_frequency_texture": "frequency_and_texture",
        "no_compression_traces": "compression_traces",
        "no_propagation_disturbance": "propagation_disturbance",
    }
    if feature_set == "all":
        return [dict(row) for row in rows], None, None
    if feature_set == "visual_forensics_only":
        selected_features = set()
        for group_name in ("frequency_and_texture", "compression_traces", "propagation_disturbance"):
            selected_features.update(_feature_names_matching_patterns(rows, groups[group_name]))
        return [
            {name: value for name, value in row.items() if name in selected_features}
            for row in rows
        ], "visual_forensics", None
    group_name = set_to_group[feature_set]
    selected_features = _feature_names_matching_patterns(rows, groups[group_name])
    if feature_set.endswith("_only"):
        return [
            {name: value for name, value in row.items() if name in selected_features}
            for row in rows
        ], group_name, None
    return [
        {name: value for name, value in row.items() if name not in selected_features}
        for row in rows
    ], None, group_name


def _feature_names_matching_patterns(
    rows: list[dict[str, float]],
    patterns: list[str],
) -> set[str]:
    names = {name for row in rows for name in row}
    matched: set[str] = set()
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            matched.update(name for name in names if name.startswith(prefix))
        else:
            matched.update(name for name in names if name == pattern)
    return matched


def _skipped_feature_ablation_result(
    feature_set: str,
    train_count: int,
    validation_count: int,
    selected_group: str | None,
    removed_group: str | None,
    reason: str,
) -> VisionFeatureAblationResult:
    return VisionFeatureAblationResult(
        feature_set=feature_set,
        feature_count=0,
        removed_feature_group=removed_group,
        selected_feature_group=selected_group,
        train_count=train_count,
        validation_count=validation_count,
        accuracy=0.0,
        macro_f1=0.0,
        gpt_image2_precision=0.0,
        gpt_image2_recall=0.0,
        average_confidence=0.0,
        confusion_matrix={},
        skipped=True,
        skip_reason=reason,
    )


def _ablation_deltas_from_all(
    results: list[VisionFeatureAblationResult],
) -> dict[str, dict[str, float]]:
    baseline = next((item for item in results if item.feature_set == "all"), None)
    if baseline is None:
        return {}
    deltas: dict[str, dict[str, float]] = {}
    for item in results:
        if item.feature_set == "all":
            continue
        deltas[item.feature_set] = {
            "accuracy_delta": round(item.accuracy - baseline.accuracy, 3),
            "macro_f1_delta": round(item.macro_f1 - baseline.macro_f1, 3),
            "gpt_image2_recall_delta": round(item.gpt_image2_recall - baseline.gpt_image2_recall, 3),
        }
    return deltas


def _feature_ablation_conclusions(
    results: list[VisionFeatureAblationResult],
) -> list[str]:
    completed = [item for item in results if not item.skipped]
    if not completed:
        return ["没有完成任何特征组消融；需要更多可用样本或特征。"]
    baseline = next((item for item in completed if item.feature_set == "all"), None)
    conclusions: list[str] = []
    if baseline is not None:
        conclusions.append(
            f"全特征基线 accuracy={baseline.accuracy:.3f}，macro-F1={baseline.macro_f1:.3f}，GPT-image-2 recall={baseline.gpt_image2_recall:.3f}。"
        )
    no_groups = [item for item in completed if item.feature_set.startswith("no_")]
    if no_groups and baseline is not None:
        worst_drop = min(
            no_groups,
            key=lambda item: (
                item.accuracy - baseline.accuracy,
                item.gpt_image2_recall - baseline.gpt_image2_recall,
            ),
        )
        conclusions.append(
            f"移除 {worst_drop.removed_feature_group} 后下降最明显，accuracy delta={worst_drop.accuracy - baseline.accuracy:.3f}，GPT-image-2 recall delta={worst_drop.gpt_image2_recall - baseline.gpt_image2_recall:.3f}。"
        )
    only_groups = [item for item in completed if item.feature_set.endswith("_only")]
    if only_groups:
        best_only = max(only_groups, key=lambda item: (item.macro_f1, item.gpt_image2_recall, item.accuracy))
        conclusions.append(
            f"单组特征中 {best_only.selected_feature_group} 表现最好，macro-F1={best_only.macro_f1:.3f}。"
        )
    conclusions.append("该结果应用于解释特征贡献，不应单独作为模型最终效果证明。")
    return conclusions


def _evaluate_source_holdout_group(
    *,
    group_name: str,
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    holdout_indices: list[int],
    request: VisionSourceHoldoutRunRequest,
) -> VisionSourceHoldoutGroupResult:
    train_label_distribution = dict(sorted(Counter(labels[index] for index in train_indices).items()))
    holdout_label_distribution = dict(sorted(Counter(labels[index] for index in holdout_indices).items()))
    if len(holdout_indices) < request.min_holdout_samples:
        return _skipped_source_holdout_group(
            group_name,
            train_indices,
            holdout_indices,
            train_label_distribution,
            holdout_label_distribution,
            f"留出组样本数少于 {request.min_holdout_samples}。",
        )
    known_train_classes = {labels[index] for index in train_indices if labels[index] != "unknown"}
    if len(train_indices) < request.min_train_samples:
        return _skipped_source_holdout_group(
            group_name,
            train_indices,
            holdout_indices,
            train_label_distribution,
            holdout_label_distribution,
            f"训练侧样本数少于 {request.min_train_samples}。",
        )
    if len(known_train_classes) < 2:
        return _skipped_source_holdout_group(
            group_name,
            train_indices,
            holdout_indices,
            train_label_distribution,
            holdout_label_distribution,
            "训练侧非 unknown 来源类别少于 2 个。",
        )

    fit_rows = [rows[index] for index in train_indices]
    fit_labels = [labels[index] for index in train_indices]
    augmentation_request = VisionTrainingRunRequest(
        task_type=GENERATOR_ATTRIBUTION_TASK,
        epochs=50,
        learning_rate=0.04,
        l2=0.02,
        min_samples=request.min_train_samples,
        enable_perturbation_augmentation=request.enable_perturbation_augmentation,
        augmentation_conditions=request.augmentation_conditions,
        max_augmented_samples=request.max_augmented_samples,
    )
    augmented_rows, augmented_labels, _, augmented_source_keys = _build_generator_augmentation_rows(
        samples=samples,
        labels=labels,
        train_indices=train_indices,
        request=augmentation_request,
    )
    fit_rows += augmented_rows
    fit_labels += augmented_labels
    fit_train_indices = list(range(len(fit_rows)))
    fit_source_keys = [
        *[_source_holdout_group_name(samples[index], "dataset_source") for index in train_indices],
        *augmented_source_keys,
    ]
    raw_feature_names = sorted({name for row in fit_rows + [rows[index] for index in holdout_indices] for name in row})
    feature_policy = _generator_profile_feature_policy(raw_feature_names, request.experiment_profile)
    feature_names = list(feature_policy["feature_names"])
    means, scales = _fit_standardizer(fit_rows, feature_names, fit_train_indices)
    classifier_path: str | None = None
    classifier_path, _ = _train_generator_classifier_artifact(
        run_id=f"holdout-{uuid4().hex[:8]}",
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=fit_source_keys,
    )
    gpt_detector_path, gpt_detector_metadata = _train_gpt_image2_detector_artifact(
        run_id=f"holdout-gpt-{uuid4().hex[:8]}",
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=fit_source_keys,
        experiment_profile=request.experiment_profile,
    )
    binary_gate_path, binary_gate_metadata = _train_generator_binary_gate_artifact(
        run_id=f"holdout-gate-{uuid4().hex[:8]}",
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=fit_source_keys,
        experiment_profile=request.experiment_profile,
    )
    binary_gate_mode = _binary_gate_mode_for_profile(request.experiment_profile)
    binary_gate_metadata["mode"] = binary_gate_mode
    prototypes = _build_class_prototypes(
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=fit_source_keys,
    )
    base_unknown_threshold = _generator_unknown_threshold(
        fit_rows,
        fit_labels,
        fit_train_indices,
        feature_names,
        means,
        scales,
    )
    unknown_threshold = _open_set_unknown_threshold(base_unknown_threshold, request)
    predictions: list[str] = []
    holdout_labels: list[str] = []
    confidences: list[float] = []
    seen_predictions: list[str] = []
    seen_labels: list[str] = []
    unseen_label_counts: Counter[str] = Counter()
    for index in holdout_indices:
        prediction = _predict_generator_label(
            rows[index],
            feature_names,
            means,
            scales,
            prototypes,
            unknown_threshold,
            classifier_path=classifier_path,
            gpt_detector_path=gpt_detector_path,
            binary_gate_path=binary_gate_path,
            generated_gate_threshold=float(
                binary_gate_metadata.get("generated_threshold", GENERATOR_BINARY_GATE_THRESHOLD)
            ),
            gpt_detector_threshold=float(
                gpt_detector_metadata.get("threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)
            ),
            real_protection_margin=float(
                binary_gate_metadata.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)
            ),
            binary_gate_mode=binary_gate_mode,
        )
        predicted_label = str(prediction["label"])
        actual_label = labels[index]
        predictions.append(predicted_label)
        holdout_labels.append(actual_label)
        confidences.append(float(prediction.get("confidence", 0.0)))
        if actual_label in known_train_classes:
            seen_predictions.append(predicted_label)
            seen_labels.append(actual_label)
        else:
            unseen_label_counts[actual_label] += 1
    if classifier_path:
        try:
            Path(classifier_path).unlink(missing_ok=True)
        except OSError:
            pass
    if binary_gate_path:
        try:
            Path(binary_gate_path).unlink(missing_ok=True)
        except OSError:
            pass
    if gpt_detector_path:
        try:
            Path(gpt_detector_path).unlink(missing_ok=True)
        except OSError:
            pass
    metrics = _classification_metrics(predictions, holdout_labels)
    seen_metrics = _classification_metrics(seen_predictions, seen_labels) if seen_labels else None
    binary_metrics = _binary_generation_metrics(predictions, holdout_labels)
    per_class = metrics.get("per_class")
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    gpt_metrics = per_class_metrics.get("gpt-image2", {})
    seen_per_class = seen_metrics.get("per_class") if isinstance(seen_metrics, dict) else None
    seen_per_class_metrics = seen_per_class if isinstance(seen_per_class, dict) else {}
    seen_gpt_metrics = seen_per_class_metrics.get("gpt-image2", {})
    return VisionSourceHoldoutGroupResult(
        holdout_group=group_name,
        train_count=len(train_indices),
        holdout_count=len(holdout_indices),
        seen_class_holdout_count=len(seen_labels),
        unseen_holdout_count=sum(unseen_label_counts.values()),
        unseen_holdout_labels=sorted(unseen_label_counts),
        train_label_distribution=train_label_distribution,
        holdout_label_distribution=holdout_label_distribution,
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        gpt_image2_precision=float(gpt_metrics.get("precision", 0.0)),
        gpt_image2_recall=float(gpt_metrics.get("recall", 0.0)),
        seen_class_accuracy=float(seen_metrics["accuracy"]) if seen_metrics else 0.0,
        seen_class_macro_f1=float(seen_metrics["macro_f1"]) if seen_metrics else 0.0,
        seen_class_gpt_image2_recall=float(seen_gpt_metrics.get("recall", 0.0)),
        binary_accuracy=float(binary_metrics["accuracy"]),
        binary_macro_f1=float(binary_metrics["macro_f1"]),
        generated_recall=float(binary_metrics["generated_recall"]),
        generated_support=int(binary_metrics["generated_support"]),
        generated_false_negative_count=int(binary_metrics["generated_false_negative_count"]),
        real_recall=float(binary_metrics["real_recall"]),
        real_false_positive_rate=float(binary_metrics["real_false_positive_rate"]),
        real_support=int(binary_metrics["real_support"]),
        real_false_positive_count=int(binary_metrics["real_false_positive_count"]),
        average_confidence=round(mean(confidences), 3) if confidences else 0.0,
        confusion_matrix=metrics["confusion_matrix"],
    )


def _skipped_source_holdout_group(
    group_name: str,
    train_indices: list[int],
    holdout_indices: list[int],
    train_label_distribution: dict[str, int],
    holdout_label_distribution: dict[str, int],
    reason: str,
) -> VisionSourceHoldoutGroupResult:
    return VisionSourceHoldoutGroupResult(
        holdout_group=group_name,
        train_count=len(train_indices),
        holdout_count=len(holdout_indices),
        seen_class_holdout_count=0,
        unseen_holdout_count=0,
        unseen_holdout_labels=[],
        train_label_distribution=train_label_distribution,
        holdout_label_distribution=holdout_label_distribution,
        accuracy=0.0,
        macro_f1=0.0,
        gpt_image2_precision=0.0,
        gpt_image2_recall=0.0,
        seen_class_accuracy=0.0,
        seen_class_macro_f1=0.0,
        seen_class_gpt_image2_recall=0.0,
        binary_accuracy=0.0,
        binary_macro_f1=0.0,
        generated_recall=0.0,
        generated_support=0,
        generated_false_negative_count=0,
        real_recall=0.0,
        real_false_positive_rate=0.0,
        real_support=0,
        real_false_positive_count=0,
        average_confidence=0.0,
        confusion_matrix={},
        skipped=True,
        skip_reason=reason,
    )


def _evaluate_label_covered_source_diagnostic(
    *,
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    labels: list[str],
    request: VisionSourceHoldoutRunRequest,
) -> VisionSourceHoldoutGroupResult | None:
    split = _label_covered_source_stratified_split(samples, labels)
    if split is None:
        return None
    train_indices, holdout_indices = split
    return _evaluate_source_holdout_group(
        group_name="__label_covered_source_stratified__",
        samples=samples,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        request=request,
    )


def _label_covered_source_stratified_split(
    samples: list[ExternalTrainingSample],
    labels: list[str],
) -> tuple[list[int], list[int]] | None:
    by_label_source: dict[str, dict[str, list[int]]] = {}
    for index, sample in enumerate(samples):
        label = labels[index]
        if label == "unknown":
            continue
        source_key = _source_holdout_group_name(sample, "dataset_source")
        by_label_source.setdefault(label, {}).setdefault(source_key, []).append(index)
    holdout_indices: list[int] = []
    for label, by_source in sorted(by_label_source.items()):
        if len(by_source) < 2:
            continue
        selected_sources = sorted(
            by_source.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:3]
        for _, source_indices in selected_sources:
            source_take_cap = max(1, len(source_indices) // 2)
            take = min(source_take_cap, 20)
            holdout_indices.extend(source_indices[:take])
    if len(holdout_indices) < 4:
        return None
    holdout_set = set(holdout_indices)
    train_indices = [index for index in range(len(samples)) if index not in holdout_set]
    holdout_labels = {labels[index] for index in holdout_indices if labels[index] != "unknown"}
    train_labels = {labels[index] for index in train_indices if labels[index] != "unknown"}
    if not holdout_labels or not holdout_labels <= train_labels:
        return None
    if len(train_labels) < 2:
        return None
    return train_indices, sorted(holdout_set)


def _add_label_covered_diagnostic_to_aggregate(
    aggregate: dict[str, float],
    diagnostic: VisionSourceHoldoutGroupResult | None,
) -> None:
    if diagnostic is None or diagnostic.skipped:
        aggregate.update(
            {
                "label_covered_available": 0.0,
                "label_covered_holdout_count": 0.0,
                "label_covered_macro_f1": 0.0,
                "label_covered_binary_macro_f1": 0.0,
                "label_covered_generated_recall": 0.0,
                "label_covered_real_false_positive_rate": 0.0,
            }
        )
        return
    aggregate.update(
        {
            "label_covered_available": 1.0,
            "label_covered_holdout_count": float(diagnostic.holdout_count),
            "label_covered_macro_f1": diagnostic.macro_f1,
            "label_covered_binary_macro_f1": diagnostic.binary_macro_f1,
            "label_covered_generated_recall": diagnostic.generated_recall,
            "label_covered_real_false_positive_rate": diagnostic.real_false_positive_rate,
        }
    )


def _source_holdout_aggregate(groups: list[VisionSourceHoldoutGroupResult]) -> dict[str, float]:
    if not groups:
        return {
            "completed_group_count": 0.0,
            "mean_accuracy": 0.0,
            "mean_macro_f1": 0.0,
            "mean_gpt_image2_recall": 0.0,
            "seen_class_holdout_count": 0.0,
            "unseen_holdout_count": 0.0,
            "mean_seen_class_accuracy": 0.0,
            "mean_seen_class_macro_f1": 0.0,
            "mean_seen_class_gpt_image2_recall": 0.0,
            "mean_binary_accuracy": 0.0,
            "mean_binary_macro_f1": 0.0,
            "mean_generated_recall": 0.0,
            "generated_support": 0.0,
            "generated_false_negative_count": 0.0,
            "overall_generated_false_negative_rate": 0.0,
            "mean_real_recall": 0.0,
            "mean_real_false_positive_rate": 0.0,
            "overall_real_false_positive_rate": 0.0,
            "real_support": 0.0,
            "real_false_positive_count": 0.0,
            "mean_confidence": 0.0,
        }
    seen_groups = [item for item in groups if item.seen_class_holdout_count > 0]
    generated_support = sum(item.generated_support for item in groups)
    generated_false_negative_count = sum(item.generated_false_negative_count for item in groups)
    real_support = sum(item.real_support for item in groups)
    real_false_positive_count = sum(item.real_false_positive_count for item in groups)
    return {
        "completed_group_count": float(len(groups)),
        "mean_accuracy": round(mean(item.accuracy for item in groups), 3),
        "mean_macro_f1": round(mean(item.macro_f1 for item in groups), 3),
        "mean_gpt_image2_recall": round(mean(item.gpt_image2_recall for item in groups), 3),
        "seen_class_holdout_count": float(sum(item.seen_class_holdout_count for item in groups)),
        "unseen_holdout_count": float(sum(item.unseen_holdout_count for item in groups)),
        "mean_seen_class_accuracy": (
            round(mean(item.seen_class_accuracy for item in seen_groups), 3) if seen_groups else 0.0
        ),
        "mean_seen_class_macro_f1": (
            round(mean(item.seen_class_macro_f1 for item in seen_groups), 3) if seen_groups else 0.0
        ),
        "mean_seen_class_gpt_image2_recall": (
            round(mean(item.seen_class_gpt_image2_recall for item in seen_groups), 3) if seen_groups else 0.0
        ),
        "mean_binary_accuracy": round(mean(item.binary_accuracy for item in groups), 3),
        "mean_binary_macro_f1": round(mean(item.binary_macro_f1 for item in groups), 3),
        "mean_generated_recall": round(mean(item.generated_recall for item in groups), 3),
        "generated_support": float(generated_support),
        "generated_false_negative_count": float(generated_false_negative_count),
        "overall_generated_false_negative_rate": (
            round(generated_false_negative_count / generated_support, 3) if generated_support else 0.0
        ),
        "mean_real_recall": round(mean(item.real_recall for item in groups), 3),
        "mean_real_false_positive_rate": round(mean(item.real_false_positive_rate for item in groups), 3),
        "overall_real_false_positive_rate": (
            round(real_false_positive_count / real_support, 3) if real_support else 0.0
        ),
        "real_support": float(real_support),
        "real_false_positive_count": float(real_false_positive_count),
        "mean_confidence": round(mean(item.average_confidence for item in groups), 3),
    }


def _source_holdout_conclusions(
    groups: list[VisionSourceHoldoutGroupResult],
    total_group_count: int,
) -> list[str]:
    if not groups:
        return ["没有完成任何跨来源留出组评估；需要更多跨来源、跨类别外部数据。"]
    aggregate = _source_holdout_aggregate(groups)
    conclusions = [
        (
            f"完成 {int(aggregate['completed_group_count'])}/{total_group_count} 个来源组留出评估，"
            f"mean accuracy={aggregate['mean_accuracy']:.3f}，mean macro-F1={aggregate['mean_macro_f1']:.3f}。"
        )
    ]
    if aggregate["seen_class_holdout_count"] > 0:
        conclusions.append(
            "已见类别跨来源诊断："
            f"seen-class macro-F1={aggregate['mean_seen_class_macro_f1']:.3f}，"
            f"seen-class accuracy={aggregate['mean_seen_class_accuracy']:.3f}；"
            f"另有 {int(aggregate['unseen_holdout_count'])} 条留出样本属于训练侧未见类别。"
        )
    conclusions.append(
        "两层输出初筛诊断："
        f"generated-vs-real macro-F1={aggregate['mean_binary_macro_f1']:.3f}，"
        f"generated recall={aggregate['mean_generated_recall']:.3f}，"
        f"overall real false positive rate={aggregate['overall_real_false_positive_rate']:.3f}。"
    )
    worst = min(groups, key=lambda item: (item.gpt_image2_recall, item.accuracy, item.average_confidence))
    conclusions.append(
        f"最弱留出组为 {worst.holdout_group}，GPT-image-2 recall={worst.gpt_image2_recall:.3f}，accuracy={worst.accuracy:.3f}。"
    )
    if aggregate["mean_accuracy"] < 0.65:
        conclusions.append("跨来源泛化仍偏弱，优先补充真实照片、GPT-image-2、Flux、SDXL、Midjourney 等来源的独立数据源。")
    else:
        conclusions.append("跨来源指标具备初步可用性，但仍需更大规模盲测和真实社交平台转码样本验证。")
    return conclusions


def _generator_predictor_from_artifact(artifact: dict[str, object] | None) -> dict[str, object]:
    if artifact is None:
        raise ValueError("请先训练 vision_generator_attribution 生成模型归因头，再运行传播扰动鲁棒性实验。")
    feature_names = _list_value(artifact.get("feature_names"))
    means = _float_mapping(artifact.get("means"))
    scales = _float_mapping(artifact.get("scales"))
    prototypes = _artifact_prototypes(artifact.get("class_prototypes"))
    if not feature_names or not means or not scales or not prototypes:
        raise ValueError("当前生成模型归因 artifact 不完整，无法运行传播扰动鲁棒性实验。")
    return {
        "model_id": str(artifact.get("id", "")),
        "model_kind": str(artifact.get("model_kind", "local-generator-attribution-prototype-v1")),
        "feature_names": feature_names,
        "means": means,
        "scales": scales,
        "prototypes": prototypes,
        "unknown_threshold": float(artifact.get("unknown_threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)),
        "open_set_min_margin": float(
            (artifact.get("open_set_unknown_policy") or {}).get("open_set_min_margin", 0.0)
            if isinstance(artifact.get("open_set_unknown_policy"), dict)
            else 0.0
        ),
        "classifier_path": str(artifact.get("classifier_path") or ""),
        "gpt_detector_path": str(artifact.get("gpt_image2_detector_path") or ""),
        "gpt_detector_threshold": float(
            (artifact.get("gpt_image2_detector_metadata") or {}).get("threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)
            if isinstance(artifact.get("gpt_image2_detector_metadata"), dict)
            else GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR
        ),
        "binary_gate_path": str(artifact.get("binary_gate_path") or ""),
        "binary_gate_mode": str(artifact.get("binary_gate_mode", "enforce")),
        "generated_gate_threshold": float(artifact.get("generated_gate_threshold", GENERATOR_BINARY_GATE_THRESHOLD)),
        "real_protection_margin": float(artifact.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)),
    }


def _evaluate_generator_condition(
    *,
    samples: list[ExternalTrainingSample],
    condition: str,
    temp_root: Path,
    predictor: dict[str, object],
    include_sample_predictions: bool,
    expected_labels: list[str] | None = None,
) -> tuple[list[str], list[str], list[float], list[dict[str, object]]]:
    predictions: list[str] = []
    labels: list[str] = []
    confidences: list[float] = []
    sample_predictions: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        path = Path(sample.image_path or "")
        if not path.is_file():
            continue
        if condition == "clean":
            eval_path = path
            eval_sha = sample.image_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            eval_path, eval_sha = _write_robustness_variant(path, condition, temp_root, index)
        label = (
            expected_labels[index]
            if expected_labels is not None and index < len(expected_labels)
            else _normalize_generator_label(sample.label)
        )
        text = f"{sample.title} {sample.content} {sample.scenario}"
        features = _generator_attribution_features(str(eval_path), eval_sha, text)
        prediction = _predict_generator_label(
            features,
            list(predictor["feature_names"]),
            dict(predictor["means"]),
            dict(predictor["scales"]),
            list(predictor["prototypes"]),
            float(predictor["unknown_threshold"]),
            classifier_path=str(predictor["classifier_path"]),
            gpt_detector_path=str(predictor["gpt_detector_path"]),
            binary_gate_path=str(predictor["binary_gate_path"]),
            generated_gate_threshold=float(predictor["generated_gate_threshold"]),
            gpt_detector_threshold=float(predictor["gpt_detector_threshold"]),
            real_protection_margin=float(predictor["real_protection_margin"]),
            open_set_min_margin=float(predictor.get("open_set_min_margin", 0.0)),
            binary_gate_mode=str(predictor.get("binary_gate_mode", "enforce")),
        )
        predicted_label = str(prediction["label"])
        confidence = float(prediction.get("confidence", 0.0))
        predictions.append(predicted_label)
        labels.append(label)
        confidences.append(confidence)
        if include_sample_predictions:
            candidates = prediction.get("candidates")
            sample_predictions.append(
                {
                    "sample_id": sample.id,
                    "dataset_name": sample.dataset_name,
                    "condition": condition,
                    "label": label,
                    "prediction": predicted_label,
                    "confidence": round(confidence, 3),
                    "candidates": candidates if isinstance(candidates, list) else [],
                }
            )
    return predictions, labels, confidences, sample_predictions


def _write_robustness_variant(
    source_path: Path,
    condition: str,
    temp_root: Path,
    index: int,
) -> tuple[Path, str]:
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    if condition.startswith("jpeg_q"):
        quality = int(condition.removeprefix("jpeg_q"))
        target = temp_root / f"{index}-{condition}.jpg"
        image.save(target, format="JPEG", quality=quality, optimize=False)
    elif condition == "screenshot_resave":
        target = temp_root / f"{index}-{condition}.png"
        width, height = image.size
        margin = max(8, min(width, height) // 12)
        canvas = Image.new("RGB", (width + margin * 2, height + margin * 3), (244, 244, 244))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, margin), fill=(32, 36, 42))
        canvas.paste(image, (margin, margin * 2))
        canvas.save(target, format="PNG")
    elif condition == "center_crop":
        target = temp_root / f"{index}-{condition}.png"
        width, height = image.size
        crop_w = max(1, int(width * 0.88))
        crop_h = max(1, int(height * 0.88))
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        image.crop((left, top, left + crop_w, top + crop_h)).save(target, format="PNG")
    elif condition == "watermark":
        target = temp_root / f"{index}-{condition}.png"
        watermarked = image.convert("RGBA")
        overlay = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text = "平台转发"
        x_pos = max(2, watermarked.width - max(70, watermarked.width // 3))
        y_pos = max(2, watermarked.height - max(24, watermarked.height // 8))
        draw.rectangle(
            (x_pos - 4, y_pos - 4, watermarked.width - 2, min(watermarked.height - 2, y_pos + 22)),
            fill=(255, 255, 255, 130),
        )
        draw.text((x_pos, y_pos), text, fill=(20, 20, 20, 190))
        Image.alpha_composite(watermarked, overlay).convert("RGB").save(target, format="PNG")
    elif condition == "weibo_download_like":
        target = temp_root / f"{index}-{condition}.jpg"
        image.save(target, format="JPEG", quality=90, optimize=False, progressive=False, subsampling=2)
    elif condition == "weibo_screenshot_like":
        target = temp_root / f"{index}-{condition}.png"
        width, height = image.size
        max_side = 240
        scale = min(max_side / max(width, height, 1), 1.0)
        thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        thumbnail = image.resize(thumb_size, Image.Resampling.BICUBIC)
        thumbnail.save(target, format="PNG")
    elif condition == "xhs_download_like":
        target = temp_root / f"{index}-{condition}{source_path.suffix.lower() or '.png'}"
        image.save(target)
    else:
        raise ValueError(f"不支持的扰动条件：{condition}")
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def _robustness_condition_description(condition: str) -> str:
    descriptions = {
        "clean": "原始导入图片，不施加传播扰动。",
        "jpeg_q85": "模拟社交平台轻度 JPEG 转码压缩。",
        "jpeg_q60": "模拟社交平台较强 JPEG 压缩和细节损失。",
        "screenshot_resave": "模拟截图后重保存，加入边框和画布重排。",
        "center_crop": "模拟平台裁剪或用户二次裁切。",
        "watermark": "模拟平台水印、转发标识或角标覆盖。",
        "weibo_download_like": "依据本次微博黑盒回收样本观察，模拟保留尺寸的 JPEG 重新编码。",
        "weibo_screenshot_like": "依据本次微博截图回收样本观察，模拟低分辨率 PNG 缩略/截图链路。",
        "xhs_download_like": "依据本次小红书创作者后台回收样本观察，模拟接近原图的下载链路。",
    }
    return descriptions.get(condition, condition)


def _robustness_feature_groups() -> dict[str, list[str]]:
    return {
        "visual_semantic": [
            "clip_similarity",
            "clip_distance",
            "clip_gap_*",
        ],
        "text_context_proxy": [
            "text_enriched_image_context_signal",
            "visual_text_context_signal",
            "watermark_context_signal",
        ],
        "frequency_and_texture": [
            "frequency_high_energy_proxy",
            "frequency_mid_energy_proxy",
            "horizontal_gradient_energy",
            "vertical_gradient_energy",
            "edge_density",
            "texture_residual_mean",
            "texture_residual_std",
        ],
        "compression_traces": [
            "compression_residual_mean",
            "compression_residual_std",
            "jpeg_block_boundary_delta",
            "byte_entropy",
            "jpg_ext",
            "png_ext",
            "webp_ext",
        ],
        "propagation_disturbance": [
            "small_image_signal",
            "screenshot_shape_signal",
            "text_overlay_edge_density",
            "corner_watermark_edge_signal",
            "image_megapixels",
            "aspect_ratio",
        ],
    }


def _robustness_conclusions(
    results: list[VisionRobustnessConditionResult],
) -> list[str]:
    if not results:
        return ["未生成有效扰动评估结果。"]
    by_condition = {item.condition: item for item in results}
    clean = by_condition.get("clean")
    conclusions: list[str] = []
    if clean is None:
        conclusions.append("未包含 clean 基线条件，无法计算相对性能下降。")
        return conclusions
    conclusions.append(
        f"clean 基线 accuracy={clean.accuracy:.3f}，GPT-image-2 recall={clean.gpt_image2_recall:.3f}。"
    )
    degraded = [
        (
            item.condition,
            round(clean.gpt_image2_recall - item.gpt_image2_recall, 3),
            round(clean.accuracy - item.accuracy, 3),
            item.confidence_delta_from_clean,
        )
        for item in results
        if item.condition != "clean"
    ]
    if degraded:
        worst = max(degraded, key=lambda item: (item[1], item[2], abs(item[3] or 0.0)))
        conclusions.append(
            f"{worst[0]} 对 GPT-image-2 recall 影响最大，召回下降 {worst[1]:.3f}，总体 accuracy 下降 {worst[2]:.3f}。"
        )
    confidence_drop = [
        item
        for item in results
        if item.confidence_delta_from_clean is not None and item.confidence_delta_from_clean < 0
    ]
    if confidence_drop:
        lowest = min(confidence_drop, key=lambda item: item.confidence_delta_from_clean or 0.0)
        conclusions.append(
            f"{lowest.condition} 下平均置信度下降 {abs(lowest.confidence_delta_from_clean or 0.0):.3f}，建议重点补充该扰动类型样本。"
        )
    else:
        conclusions.append("当前扰动条件下平均置信度未出现明显下降，但仍需跨平台盲测验证。")
    return conclusions


def train_fusion_head(request: FusionTrainingRunRequest) -> FusionTrainingRunResult:
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=FUSION_TASK_TYPE)
        if sample.image_available and sample.image_path
    ]
    samples = _task_relevant_samples(raw_samples, FUSION_TASK_TYPE)
    if len(samples) < request.min_samples:
        raise ValueError(
            f"融合训练至少需要 {request.min_samples} 条带文本、图片和标签的外部多模态样本，"
            f"当前只有 {len(samples)} 条任务匹配样本（原始可用图片样本 {len(raw_samples)} 条）；"
            "内置四方向样例只做展示评测。"
        )
    rows = [extract_fusion_sample_features(sample) for sample in samples]
    labels = [sample.risk_score for sample in samples]
    run_id = str(uuid4())
    feature_names = sorted({name for row in rows for name in row})
    train_indices, valid_indices = _split_indices(labels)
    split_report = _split_report(samples, labels, train_indices, valid_indices)
    means, scales = _fit_standardizer(rows, feature_names, train_indices)
    weights, bias = _train_ridge_regressor(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        l2=request.l2,
    )
    train_predictions = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in train_indices
    ]
    valid_predictions = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in valid_indices
    ]
    prototypes = _build_knn_prototypes(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    tree_artifact_path, tree_metadata = _train_tree_artifact(
        run_id="pending",
        task_type=FUSION_TASK_TYPE,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    ensemble = _select_ensemble(
        samples=samples,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        valid_indices=valid_indices,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        prototypes=prototypes,
        tree_artifact_path=tree_artifact_path,
        tree_metadata=tree_metadata,
    )
    train_predictions = ensemble["train_predictions"]
    valid_predictions = ensemble["valid_predictions"]
    train_labels = [labels[index] for index in train_indices]
    valid_labels = [labels[index] for index in valid_indices]
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    tree_artifact_path = _finalize_tree_artifact(tree_artifact_path, run_id, FUSION_TASK_TYPE)
    ensemble["tree_artifact_path"] = tree_artifact_path
    if isinstance(ensemble.get("tree_metadata"), dict):
        ensemble["tree_metadata"]["artifact_path"] = tree_artifact_path
        if isinstance(ensemble.get("selection_report"), dict) and isinstance(ensemble["selection_report"].get("tree_model"), dict):
            ensemble["selection_report"]["tree_model"]["artifact_path"] = tree_artifact_path
    result = FusionTrainingRunResult(
        id=run_id,
        created_at=created_at,
        model_kind="local-multimodal-fusion-ensemble-v2",
        status="trained",
        sample_count=len(samples),
        validation_count=len(valid_indices),
        feature_count=len(feature_names),
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        train_mae=_mae(train_predictions, train_labels),
        validation_mae=_mae(valid_predictions, valid_labels),
        train_rmse=_rmse(train_predictions, train_labels),
        validation_rmse=_rmse(valid_predictions, valid_labels),
        accuracy_within_10=_accuracy_within(valid_predictions, valid_labels, 10),
        risk_level_accuracy=_risk_level_accuracy(valid_predictions, valid_labels),
        label_distribution=_label_distribution(labels),
        confusion_matrix=_confusion_matrix(valid_predictions, valid_labels),
        top_positive_features=_feature_weights(weights, "positive")[:8],
        top_negative_features=_feature_weights(weights, "negative")[:8],
        training_trace=[
            f"读取 {len(samples)} 条外部多模态融合样本，样本必须同时具备文本、标签和本地图片。",
            f"任务适配过滤：原始可用 {len(raw_samples)} 条，保留 {len(samples)} 条，排除 {len(raw_samples) - len(samples)} 条明显不属于融合任务的辅助样本。",
            "融合特征包含文本风险弱信号、图片本地统计、三个视觉证据头分数、证据数量与来源信号。",
            f"图文语义特征：{'启用本地 CLIP ' + CLIP_MODEL_NAME if CLIP_ENABLED else '未启用，本轮使用轻量本地统计特征'}。",
            f"训练 Ridge 线性头、boosted-tree 非线性头并构建 {len(prototypes)} 个本地相似样本原型；在验证集上选择 {ensemble['selected_model']}。",
            f"使用分层留出验证：训练 {len(train_indices)} 条，验证 {len(valid_indices)} 条。",
            f"验证指标：MAE { _mae(valid_predictions, valid_labels) }，RMSE { _rmse(valid_predictions, valid_labels) }，风险等级准确率 { _risk_level_accuracy(valid_predictions, valid_labels) }。",
            "保存模型权重、特征标准化参数、指标与模型卡；demo 样例明确排除在训练之外。",
        ],
        model_card={
            "name": "公共安全谣言多模态融合风险头",
            "version": run_id,
            "architecture": "文本风险特征 + 视觉证据头分数 + 证据/来源特征 + Ridge + 本地 kNN 原型校准集成",
            "training_data": "外部导入多模态样本；内置四方向展示样例不进入训练集。",
            "training_source_summary": _sample_source_summary(samples),
            "task_filter": {
                "candidate_count": len(raw_samples),
                "selected_count": len(samples),
                "excluded_as_auxiliary_or_mismatched": len(raw_samples) - len(samples),
                "policy": "保留任务明确匹配和自定义通用标注样本，排除明显走错分支的已知辅助数据。",
            },
            "validation_protocol": split_report,
            "metrics": _metrics_report(train_predictions, train_labels, valid_predictions, valid_labels),
            "ensemble_selection": ensemble["selection_report"],
            "tree_regressor": ensemble.get("tree_metadata", {}),
            "validation_diagnostics": _validation_diagnostics(samples, labels, valid_predictions, valid_labels),
            "semantic_features": {
                "clip_enabled": CLIP_ENABLED,
                "clip_model": CLIP_MODEL_NAME if CLIP_ENABLED else None,
                "clip_feature_dims": CLIP_FEATURE_DIMS,
                "extractor_version": CLIP_EXTRACTOR_VERSION,
            },
            "excluded_demo_cases": [case.id for case in DEMO_CASES],
            "task_type": FUSION_TASK_TYPE,
            "leakage_controls": [
                "外部样本 label 字段只作为监督目标映射，不进入融合特征。",
                "融合训练只使用文本内容、图片本地特征、视觉证据头输出和来源/证据数量特征。",
                "内置四方向展示样例不进入训练、验证或特征缓存样本。",
            ],
            "not_for": "不替代大模型复核、人工核验或执法结论。",
        },
    )
    artifact = {
        "id": run_id,
        "created_at": created_at,
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "model_kind": result.model_kind,
        "ensemble": {
            "selected_model": ensemble["selected_model"],
            "alpha": ensemble["alpha"],
            "knn_k": ensemble["knn_k"],
            "prototype_count": len(prototypes),
            "tree_artifact_path": tree_artifact_path,
            "tree_metadata": ensemble.get("tree_metadata", tree_metadata),
        },
        "knn_prototypes": prototypes,
        "clip_prototypes": ensemble.get("clip_prototypes", []),
        "model_card": result.model_card,
        "validation_protocol": split_report,
        "metrics": result.model_card["metrics"],
    }
    save_fusion_training_run(result, artifact)
    return result


def get_fusion_training_status() -> FusionTrainingStatus:
    latest = get_latest_fusion_training_run()
    task_status = _task_status(FUSION_TASK_TYPE)
    if latest is None:
        return FusionTrainingStatus(
            trained=False,
            active_model_id=None,
            latest_run=None,
            data=task_status,
            note="融合头尚未训练；真实研判会标注未启用，不会生成融合分。",
        )
    return FusionTrainingStatus(
        trained=True,
        active_model_id=latest.id,
        latest_run=latest,
        data=task_status,
        note="融合头已启用；正式研判会结合文本、视觉证据和来源特征输出最终融合风险。",
    )


def predict_vision_for_assets(
    assets: list[CaseAsset],
    case_text: str = "",
) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for task_type in sorted(VISION_TASK_TYPES):
        artifact = get_active_vision_training_artifact(task_type)
        if artifact is None:
            outputs[task_type] = {
                "trained": False,
                "enabled": False,
                "note": "未训练/未启用；不生成分数。",
            }
            continue
        if task_type == GENERATOR_ATTRIBUTION_TASK:
            outputs[task_type] = _predict_generator_attribution_for_assets(
                artifact,
                assets,
                case_text,
            )
            continue
        predictions = [
            _predict_vision_asset(task_type, artifact, asset, case_text)
            for asset in assets
        ]
        scored = [item for item in predictions if item.get("score") is not None]
        avg_score = (
            round(mean(float(item["score"]) for item in scored), 2)
            if scored
            else None
        )
        outputs[task_type] = {
            "trained": True,
            "enabled": True,
            "model_id": str(artifact.get("id", "")),
            "model_kind": str(artifact.get("model_kind", "local-vision-evidence-ridge-v1")),
            "score": avg_score,
            "risk_level": risk_level_from_score(avg_score).value if avg_score is not None else None,
            "asset_predictions": predictions,
        }
    return outputs


def predict_fusion_for_case(
    case: CaseSample,
    assets: list[CaseAsset],
    baseline_score: float | int,
    vision_outputs: dict[str, object],
) -> dict[str, object]:
    artifact = get_active_fusion_training_artifact()
    if artifact is None:
        return {
            "trained": False,
            "enabled": False,
            "note": "融合头未训练/未启用；正式研判不生成融合分。",
        }
    if not assets:
        return {
            "trained": True,
            "enabled": False,
            "model_id": str(artifact.get("id", "")),
            "model_kind": str(artifact.get("model_kind", "local-multimodal-fusion-ridge-v1")),
            "score": None,
            "risk_level": None,
            "note": "融合头已训练，但当前案例没有图片/截图证据；不使用空图像特征生成融合分。",
        }
    features = extract_runtime_fusion_features(case, assets, float(baseline_score), vision_outputs)
    prediction = _predict_from_artifact(artifact, features)
    if prediction is None:
        return {
            "trained": False,
            "enabled": False,
            "note": "融合头 artifact 不完整，未输出分数。",
        }
    score, contributions = prediction
    return {
        "trained": True,
        "enabled": True,
        "model_id": str(artifact.get("id", "")),
        "model_kind": str(artifact.get("model_kind", "local-multimodal-fusion-ridge-v1")),
        "score": score,
        "risk_level": risk_level_from_score(score).value,
        "top_contributions": _contribution_payload(contributions),
    }


def evaluate_demo_cases() -> DemoEvaluationResult:
    results: list[DemoEvaluationCaseResult] = []
    for case in DEMO_CASES:
        text_prediction = predict_with_active_model(case)
        if text_prediction is None:
            text_only = {
                "trained": False,
                "score": None,
                "note": "文本风险模型未训练；demo 不作为训练数据。",
            }
        else:
            score, confidence, explanations, model_id = text_prediction
            text_only = {
                "trained": True,
                "score": score,
                "risk_level": risk_level_from_score(score).value,
                "confidence": confidence,
                "model_id": model_id,
                "explanations": explanations,
            }
        vision_only = predict_vision_for_assets([], case_text=_case_text(case))
        fusion = predict_fusion_for_case(
            case=case,
            assets=[],
            baseline_score=float(text_only["score"] or 0),
            vision_outputs=vision_only,
        )
        results.append(
            DemoEvaluationCaseResult(
                case_id=case.id,
                title=case.title,
                text_only=text_only,
                vision_only=vision_only,
                fusion=fusion,
            )
        )
    return DemoEvaluationResult(
        id=f"demo-eval-{uuid4().hex[:10]}",
        created_at=datetime.now(UTC).isoformat(),
        demo_case_count=len(DEMO_CASES),
        results=results,
        note="展示评测只读取四个内置方向样例，不写入训练集，不改变任何模型权重。",
    )


def extract_sample_features(sample: ExternalTrainingSample, task_type: str) -> dict[str, float]:
    text = f"{sample.title} {sample.content} {sample.scenario}"
    image_features = _image_features(sample.image_path, sample.image_sha256)
    if task_type == GENERATOR_ATTRIBUTION_TASK:
        return _generator_attribution_features(sample.image_path, sample.image_sha256, text)
    clip_features = _clip_features(sample.image_path, sample.image_sha256, text)
    features = {
        **image_features,
        **clip_features,
        **_text_context_features(text),
    }
    features[f"task::{task_type}"] = 1.0
    return features


def _generator_attribution_features(
    image_path: str | None,
    image_sha256: str | None,
    text: str,
) -> dict[str, float]:
    safe_text = _sanitize_generator_context(text)
    features = {
        **_image_features(image_path, image_sha256),
        **_clip_features(image_path, image_sha256, safe_text),
        **_text_enriched_image_features(safe_text),
        f"task::{GENERATOR_ATTRIBUTION_TASK}": 1.0,
    }
    return features


def _sanitize_generator_context(text: str) -> str:
    sanitized = text.lower()
    generator_tokens = (
        "gpt-image-2",
        "gpt image 2",
        "gpt-image2",
        "gptimage2",
        "gpt-image-1.5",
        "gpt-image-1",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "sdxl",
        "sd3",
        "sd21",
        "flux",
        "dall-e-3",
        "dall-e",
        "dalle",
        "nano banana",
        "nano-banana",
        "seedream",
        "imagegbt",
        "real photo",
        "real image",
        "authentic photo",
        "camera photo",
        "真实照片",
        "实拍照片",
    )
    for token in generator_tokens:
        sanitized = sanitized.replace(token, " ")
    return re.sub(r"\s+", " ", sanitized).strip()


def _text_enriched_image_features(text: str) -> dict[str, float]:
    normalized = text.lower()
    text_overlay_words = (
        "ocr",
        "字幕",
        "文字",
        "截图",
        "长图",
        "海报",
        "标语",
        "通知",
        "通报",
        "菜单",
        "聊天记录",
        "弹幕",
        "caption",
        "poster",
        "screenshot",
        "subtitle",
    )
    watermark_words = (
        "水印",
        "角标",
        "logo",
        "平台标识",
        "转发",
        "二传",
        "压缩",
        "裁剪",
        "重保存",
        "watermark",
        "resave",
        "repost",
    )
    return {
        "text_enriched_image_context_signal": min(len(normalized) / 240.0, 1.0),
        "visual_text_context_signal": min(
            sum(1 for word in text_overlay_words if word in normalized) / 4.0,
            1.0,
        ),
        "watermark_context_signal": min(
            sum(1 for word in watermark_words if word in normalized) / 4.0,
            1.0,
        ),
    }


def extract_fusion_sample_features(sample: ExternalTrainingSample) -> dict[str, float]:
    features = extract_sample_features(sample, task_type=FUSION_TASK_TYPE)
    text_risk = _text_risk_heuristic(sample)
    features["text_score"] = text_risk
    features["high_text_score"] = 1.0 if text_risk >= 68 else 0.0
    for task_type in VISION_TASK_TYPES:
        artifact = get_active_vision_training_artifact(task_type)
        predicted = _predict_sample_with_artifact(task_type, artifact, sample) if artifact else None
        features[f"{task_type}_score"] = (predicted or text_risk) / 100.0
    features["evidence_count"] = 1.0 if sample.image_available else 0.0
    features["source_signal"] = 1.0 if sample.source_url else 0.0
    return features


def extract_runtime_fusion_features(
    case: CaseSample,
    assets: list[CaseAsset],
    baseline_score: float,
    vision_outputs: dict[str, object],
) -> dict[str, float]:
    text_features = _text_context_features(_case_text(case))
    features = {
        **text_features,
        "text_score": baseline_score / 100.0,
        "text_confidence": float(case.manual_risk_score or 0) / 100.0 if case.manual_risk_score else 0.0,
        "high_text_score": 1.0 if baseline_score >= 68 else 0.0,
        "evidence_count": float(len(assets)),
        "source_signal": 1.0 if case.source_url.startswith(("http://", "https://")) else 0.0,
    }
    if assets:
        image_rows = [_image_features(asset.storage_path, asset.sha256) for asset in assets]
        for name in sorted({key for row in image_rows for key in row}):
            features[name] = mean(row.get(name, 0.0) for row in image_rows)
        clip_rows = [_clip_features(asset.storage_path, asset.sha256, _case_text(case)) for asset in assets]
        for name in sorted({key for row in clip_rows for key in row}):
            features[name] = mean(row.get(name, 0.0) for row in clip_rows)
    for task_type in VISION_TASK_TYPES:
        value = vision_outputs.get(task_type)
        score = None
        if isinstance(value, dict):
            raw_score = value.get("score")
            if isinstance(raw_score, int | float):
                score = float(raw_score)
            elif task_type == GENERATOR_ATTRIBUTION_TASK:
                raw_confidence = value.get("confidence")
                if isinstance(raw_confidence, int | float):
                    score = float(raw_confidence) * 100.0
        feature_name = (
            "vision_generator_attribution_signal"
            if task_type == GENERATOR_ATTRIBUTION_TASK
            else f"{task_type}_score"
        )
        features[feature_name] = (score if score is not None else baseline_score) / 100.0
    return features


def _train_and_save_vision(
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    request: VisionTrainingRunRequest,
    candidate_count: int,
) -> VisionTrainingRunResult:
    labels = [sample.risk_score for sample in samples]
    feature_names = sorted({name for row in rows for name in row})
    train_indices, valid_indices = _split_indices(labels)
    split_report = _split_report(samples, labels, train_indices, valid_indices)
    means, scales = _fit_standardizer(rows, feature_names, train_indices)
    weights, bias = _train_ridge_regressor(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        l2=request.l2,
    )
    train_predictions = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in train_indices
    ]
    valid_predictions = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in valid_indices
    ]
    prototypes = _build_knn_prototypes(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    tree_artifact_path, tree_metadata = _train_tree_artifact(
        run_id="pending",
        task_type=request.task_type,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    ensemble = _select_ensemble(
        samples=samples,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        valid_indices=valid_indices,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        prototypes=prototypes,
        tree_artifact_path=tree_artifact_path,
        tree_metadata=tree_metadata,
    )
    train_predictions = ensemble["train_predictions"]
    valid_predictions = ensemble["valid_predictions"]
    train_labels = [labels[index] for index in train_indices]
    valid_labels = [labels[index] for index in valid_indices]
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    tree_artifact_path = _finalize_tree_artifact(tree_artifact_path, run_id, request.task_type)
    ensemble["tree_artifact_path"] = tree_artifact_path
    if isinstance(ensemble.get("tree_metadata"), dict):
        ensemble["tree_metadata"]["artifact_path"] = tree_artifact_path
        if isinstance(ensemble.get("selection_report"), dict) and isinstance(ensemble["selection_report"].get("tree_model"), dict):
            ensemble["selection_report"]["tree_model"]["artifact_path"] = tree_artifact_path
    model_kind = (
        "local-vision-tamper-boosted-patch-v1"
        if request.task_type == "vision_tamper"
        else "local-vision-evidence-ensemble-v2"
    )
    result = VisionTrainingRunResult(
        id=run_id,
        created_at=created_at,
        task_type=request.task_type,
        model_kind=model_kind,
        status="trained",
        sample_count=len(samples),
        validation_count=len(valid_indices),
        feature_count=len(feature_names),
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        train_mae=_mae(train_predictions, train_labels),
        validation_mae=_mae(valid_predictions, valid_labels),
        train_rmse=_rmse(train_predictions, train_labels),
        validation_rmse=_rmse(valid_predictions, valid_labels),
        accuracy_within_10=_accuracy_within(valid_predictions, valid_labels, 10),
        risk_level_accuracy=_risk_level_accuracy(valid_predictions, valid_labels),
        label_distribution=_label_distribution(labels),
        confusion_matrix=_confusion_matrix(valid_predictions, valid_labels),
        top_positive_features=_feature_weights(weights, "positive")[:8],
        top_negative_features=_feature_weights(weights, "negative")[:8],
        training_trace=[
            f"读取 {len(samples)} 条 {request.task_type} 外部图片样本，全部要求具备本地图片文件与标签。",
            f"任务适配过滤：原始可用 {candidate_count} 条，保留 {len(samples)} 条，排除 {candidate_count - len(samples)} 条明显不属于该证据头的辅助样本。",
            "抽取图片尺寸、文件大小、sha256、字节分布、基础纹理代理统计和文本上下文关键词。",
            f"图文语义特征：{'启用本地 CLIP ' + CLIP_MODEL_NAME if CLIP_ENABLED else '未启用，本轮使用轻量本地统计特征'}。",
            f"训练 Ridge 线性头、boosted-tree 非线性头并构建 {len(prototypes)} 个本地相似样本原型；在验证集上选择 {ensemble['selected_model']}。",
            f"使用分层留出验证训练本地监督证据头：训练 {len(train_indices)} 条，验证 {len(valid_indices)} 条，特征 {len(feature_names)} 维。",
            f"验证指标：MAE { _mae(valid_predictions, valid_labels) }，RMSE { _rmse(valid_predictions, valid_labels) }，风险等级准确率 { _risk_level_accuracy(valid_predictions, valid_labels) }。",
            "保存模型卡和指标；内置四方向样例只用于训练后展示评测，不参与本次训练。",
        ],
        model_card={
            "name": f"{request.task_type} 本地视觉证据头",
            "version": run_id,
            "task_type": request.task_type,
            "architecture": "图片元数据 + 字节统计 + 文本上下文关键词 + Ridge + XGBoost/CatBoost boosted-tree 风险头 + 本地 kNN 原型校准集成",
            "training_data": "外部导入图片类数据集；内置四方向展示样例不进入训练集。",
            "training_source_summary": _sample_source_summary(samples),
            "task_filter": {
                "candidate_count": candidate_count,
                "selected_count": len(samples),
                "excluded_as_auxiliary_or_mismatched": candidate_count - len(samples),
                "policy": "保留任务明确匹配和自定义通用标注样本，排除明显走错分支的已知辅助数据。",
            },
            "validation_protocol": split_report,
            "metrics": _metrics_report(train_predictions, train_labels, valid_predictions, valid_labels),
            "ensemble_selection": ensemble["selection_report"],
            "validation_diagnostics": _validation_diagnostics(samples, labels, valid_predictions, valid_labels),
            "tamper_forensics_policy": (
                {
                    "enabled": True,
                    "image_level_model": "XGBoost/CatBoost boosted-tree risk head selected on validation split",
                    "region_policy": "runtime patch candidate scan; returns candidate abnormal regions only, not pixel-level segmentation",
                    "requires_future_mask_benchmark": True,
                    "boundary": "候选异常区域仅用于辅助研判和人工复核，不构成篡改定论或司法鉴定结论。",
                }
                if request.task_type == "vision_tamper"
                else {}
            ),
            "semantic_features": {
                "clip_enabled": CLIP_ENABLED,
                "clip_model": CLIP_MODEL_NAME if CLIP_ENABLED else None,
                "clip_feature_dims": CLIP_FEATURE_DIMS,
                "extractor_version": CLIP_EXTRACTOR_VERSION,
            },
            "tree_regressor": ensemble.get("tree_metadata", {}),
            "excluded_demo_cases": [case.id for case in DEMO_CASES],
            "leakage_controls": [
                "外部样本 label 字段只作为监督目标映射，不进入图片或文本上下文特征。",
                "证据头训练只使用图片本地统计、文件哈希派生特征和不含标签的文本上下文。",
                "内置四方向展示样例不进入训练、验证或特征缓存样本。",
            ],
            "not_for": "不宣称微调基础多模态模型或 CLIP 本体，不替代人工图像取证结论。",
        },
    )
    artifact = {
        "id": run_id,
        "created_at": created_at,
        "task_type": request.task_type,
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "model_kind": result.model_kind,
        "ensemble": {
            "selected_model": ensemble["selected_model"],
            "alpha": ensemble["alpha"],
            "knn_k": ensemble["knn_k"],
            "prototype_count": len(prototypes),
            "tree_artifact_path": tree_artifact_path,
            "tree_metadata": ensemble.get("tree_metadata", tree_metadata),
        },
        "knn_prototypes": prototypes,
        "clip_prototypes": ensemble.get("clip_prototypes", []),
        "model_card": result.model_card,
        "validation_protocol": split_report,
        "metrics": result.model_card["metrics"],
    }
    return _persist_vision_training_result(result, artifact, request)


def _train_and_save_generator_attribution(
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    request: VisionTrainingRunRequest,
    candidate_count: int,
) -> VisionTrainingRunResult:
    profile_samples, profile_rows, labels, profile_report = _generator_experiment_view(samples, rows, request)
    profile_policy = _generator_profile_policy(request.experiment_profile)
    samples = profile_samples
    rows = profile_rows
    if len(samples) < request.min_samples:
        raise ValueError(
            f"{request.experiment_profile} 实验至少需要 {request.min_samples} 条 profile 匹配样本，"
            f"当前只有 {len(samples)} 条。"
        )
    class_counts = Counter(labels)
    known_labels = [label for label in labels if label != "unknown"]
    if len(set(known_labels)) < 2:
        raise ValueError(
            "生成图片三分类至少需要 2 个非 unknown 类别，例如 gpt-image2、other-generated、real。"
        )
    run_id = str(uuid4())
    train_indices, valid_indices = _split_generator_validation_indices(samples, labels, request)
    split_report = _classification_split_report(samples, labels, train_indices, valid_indices)
    augmented_rows, augmented_labels, augmentation_protocol, augmented_source_keys = _build_generator_augmentation_rows(
        samples=samples,
        labels=labels,
        train_indices=train_indices,
        request=request,
    )
    train_rows = [rows[index] for index in train_indices] + augmented_rows
    train_labels = [labels[index] for index in train_indices] + augmented_labels
    train_source_keys = [
        *[_source_holdout_group_name(samples[index], "dataset_source") for index in train_indices],
        *augmented_source_keys,
    ]
    valid_rows = [rows[index] for index in valid_indices]
    valid_labels = [labels[index] for index in valid_indices]
    fit_rows = train_rows + valid_rows
    fit_labels = train_labels + valid_labels
    fit_train_indices = list(range(len(train_rows)))
    fit_valid_indices = list(range(len(train_rows), len(fit_rows)))
    raw_feature_names = sorted({name for row in fit_rows for name in row})
    feature_policy = _generator_profile_feature_policy(raw_feature_names, request.experiment_profile)
    feature_names = list(feature_policy["feature_names"])
    means, scales = _fit_standardizer(fit_rows, feature_names, fit_train_indices)
    classifier_path, classifier_metadata = _train_generator_classifier_artifact(
        run_id=run_id,
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=train_source_keys,
    )
    gpt_detector_path, gpt_detector_metadata = _train_gpt_image2_detector_artifact(
        run_id=run_id,
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=train_source_keys,
        experiment_profile=request.experiment_profile,
    )
    binary_gate_path, binary_gate_metadata = _train_generator_binary_gate_artifact(
        run_id=run_id,
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=train_source_keys,
        experiment_profile=request.experiment_profile,
    )
    binary_gate_mode = _binary_gate_mode_for_profile(request.experiment_profile)
    binary_gate_metadata["mode"] = binary_gate_mode
    prototypes = _build_class_prototypes(
        rows=fit_rows,
        labels=fit_labels,
        train_indices=fit_train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=train_source_keys,
    )
    base_unknown_threshold = _generator_unknown_threshold(
        fit_rows,
        fit_labels,
        fit_train_indices,
        feature_names,
        means,
        scales,
    )
    unknown_threshold = _open_set_unknown_threshold(base_unknown_threshold, request)
    train_predictions = [
        _predict_generator_label(
            fit_rows[index],
            feature_names,
            means,
            scales,
            prototypes,
            unknown_threshold,
            classifier_path=classifier_path,
            gpt_detector_path=gpt_detector_path,
            binary_gate_path=binary_gate_path,
            generated_gate_threshold=float(
                binary_gate_metadata.get("generated_threshold", GENERATOR_BINARY_GATE_THRESHOLD)
            ),
            gpt_detector_threshold=float(
                gpt_detector_metadata.get("threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)
            ),
            real_protection_margin=float(
                binary_gate_metadata.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)
            ),
            open_set_min_margin=float(getattr(request, "open_set_min_margin", 0.0) or 0.0),
            binary_gate_mode=binary_gate_mode,
        )["label"]
        for index in fit_train_indices
    ]
    valid_predictions = [
        _predict_generator_label(
            fit_rows[index],
            feature_names,
            means,
            scales,
            prototypes,
            unknown_threshold,
            classifier_path=classifier_path,
            gpt_detector_path=gpt_detector_path,
            binary_gate_path=binary_gate_path,
            generated_gate_threshold=float(
                binary_gate_metadata.get("generated_threshold", GENERATOR_BINARY_GATE_THRESHOLD)
            ),
            gpt_detector_threshold=float(
                gpt_detector_metadata.get("threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)
            ),
            real_protection_margin=float(
                binary_gate_metadata.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)
            ),
            open_set_min_margin=float(getattr(request, "open_set_min_margin", 0.0) or 0.0),
            binary_gate_mode=binary_gate_mode,
        )["label"]
        for index in fit_valid_indices
    ]
    classification_metrics = _classification_metrics(valid_predictions, valid_labels)
    train_metrics = _classification_metrics(train_predictions, train_labels)
    created_at = datetime.now(UTC).isoformat()
    source_distribution = dict(sorted(class_counts.items()))
    augmentation_trace = (
        f"启用传播扰动增强：仅对训练 split 生成 {augmentation_protocol['generated_augmentation_count']} 条临时增强样本，"
        "验证集保持原始 clean holdout，不写入外部训练池。"
        if augmentation_protocol["enabled"]
        else "未启用传播扰动增强；本次仅使用原始外部图片样本训练，验证集为 clean holdout。"
    )
    validation_trace = (
        f"验证策略：{split_report['method']}；验证样本 {len(valid_indices)} 条，"
        f"独立留出来源 {len(split_report.get('held_out_sources', []))} 个。"
    )
    result = VisionTrainingRunResult(
        id=run_id,
        created_at=created_at,
        task_type=request.task_type,
        model_kind="local-generator-attribution-boosted-tree-v3",
        status="trained",
        sample_count=len(samples),
        validation_count=len(valid_indices),
        feature_count=len(feature_names),
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        train_mae=round((1.0 - train_metrics["accuracy"]) * 100.0, 2),
        validation_mae=round((1.0 - classification_metrics["accuracy"]) * 100.0, 2),
        train_rmse=0.0,
        validation_rmse=0.0,
        accuracy_within_10=classification_metrics["accuracy"],
        risk_level_accuracy=classification_metrics["accuracy"],
        label_distribution=source_distribution,
        confusion_matrix=classification_metrics["confusion_matrix"],
        top_positive_features=_prototype_separation_features(prototypes, feature_names, "positive")[:8],
        top_negative_features=_prototype_separation_features(prototypes, feature_names, "negative")[:8],
        training_trace=[
            f"读取 {len(samples)} 条生成模型归因外部图片样本，全部要求具备本地图片文件与来源标签。",
            f"任务适配过滤：原始可用 {candidate_count} 条，保留 {len(samples)} 条，排除 {candidate_count - len(samples)} 条明显不属于归因任务的辅助样本。",
            f"实验 profile：{request.experiment_profile}；样本域 {profile_report['domain_distribution']}；标签策略：{profile_report['label_policy']}。",
            f"分轨目标：{profile_policy['chinese_name']}；{profile_policy['objective']} 验收政策：{profile_policy['activation_policy']}",
            "将标签归一化为三分类训练目标：gpt-image2、other-generated、real；unknown 仅作为预测阶段低置信退让。",
            "抽取图片尺寸、文件大小、字节分布、压缩残差、频域/块效应代理、文字覆盖/水印代理和清洗后的视觉语义上下文。",
            augmentation_trace,
            validation_trace,
            f"以 {classifier_metadata.get('model', 'boosted-tree classifier')} 为主分类器、类别原型为兜底；验证集输出 accuracy {classification_metrics['accuracy']}、macro F1 {classification_metrics['macro_f1']}。",
            "低置信或距离过远时输出 unknown，不强行归因为某个生成模型。",
            "保存模型卡和指标；内置四方向展示样例只用于训练后展示评测，不参与本次训练。",
        ],
        model_card={
            "name": "生成模型来源归因头",
            "version": run_id,
            "task_type": request.task_type,
            "architecture": "图片本地统计/压缩/频域/纹理/文字覆盖代理 + 可选 DINOv2/ConvNeXt/CLIP 语义 embedding + XGBoost/CatBoost boosted-tree 三分类头 + 标准化类别原型兜底 + open-set unknown 阈值",
            "primary_classifier": classifier_metadata,
            "gpt_image2_detector": gpt_detector_metadata,
            "binary_generated_gate": binary_gate_metadata,
            "real_photo_guard": {
                "enabled": True,
                "policy": "高分辨率 JPEG、自然亮度/饱和度、边缘、纹理与压缩残差代理信号触发真实照片保护；仅作为辅助线索，不构成真实性结论。",
                "threshold": 0.72,
                "uses_filename_or_case_id": False,
            },
            "feature_policy": feature_policy,
            "experiment_profile": profile_report,
            "profile_policy": profile_policy,
            "training_data": "外部导入且带生成来源标签的图片数据；内置四方向展示样例不进入训练集。",
            "training_source_summary": _sample_source_summary(samples),
            "task_filter": {
                "candidate_count": candidate_count,
                "selected_count": len(samples),
                "excluded_as_auxiliary_or_mismatched": candidate_count - len(samples),
                "policy": "仅保留 task_type=vision_generator_attribution 的图片样本；原始来源标签映射为 GPT-image2 / other-generated / real 三分类监督目标。",
            },
            "validation_protocol": split_report,
            "augmentation_protocol": augmentation_protocol,
            "classification_metrics": {
                "train": train_metrics,
                "validation": classification_metrics,
            },
            "dataset_caveats": _generator_dataset_caveats(samples),
            "source_classes": sorted(source_distribution),
            "class_counts": source_distribution,
            "unknown_threshold": unknown_threshold,
            "open_set_unknown_policy": _open_set_policy_report(request, unknown_threshold, base_unknown_threshold),
            "semantic_features": {
                "clip_enabled": CLIP_ENABLED,
                "clip_model": CLIP_MODEL_NAME if CLIP_ENABLED else None,
                "clip_feature_dims": CLIP_FEATURE_DIMS,
                "extractor_version": CLIP_EXTRACTOR_VERSION,
                "embedding_upgrade_path": "DINOv2/ConvNeXt embedding 可作为后续缓存特征接入；当前测试环境未强制下载大模型，避免阻塞 CPU 单测。",
            },
            "excluded_demo_cases": [case.id for case in DEMO_CASES],
            "leakage_controls": [
                "外部样本 label/source/source_detail 字段只作为监督目标或审计来源，不进入归因特征。",
                "文本富集型图像上下文会先移除 gpt-image2、midjourney、stable-diffusion、nano-banana、seedream、flux、dall-e 等生成器名称，避免标签泄漏。",
                "生成模型归因分支使用图片统计、压缩、频域、文字覆盖代理和清洗后的视觉语义上下文，不使用生成器名称文本当特征。",
                "内置四方向展示样例不进入训练、验证或特征缓存样本。",
            ],
            "boundary": (
                "只能输出生成图片研判线索：疑似 GPT-image2、其他 AI 生成图、真实照片；低置信输出 unknown。"
                "不替代 C2PA、水印、平台元数据、原始发布链路或人工取证结论。"
            ),
            "not_for": "不得把通用 AIGC 分数冒充 GPT-image2 归因；未覆盖类别和跨平台压缩图片需要人工复核。",
        },
    )
    artifact = {
        "id": run_id,
        "created_at": created_at,
        "task_type": request.task_type,
        "feature_names": feature_names,
        "feature_policy": feature_policy,
        "means": means,
        "scales": scales,
        "model_kind": result.model_kind,
        "class_prototypes": prototypes,
        "classifier_path": classifier_path,
        "classifier_metadata": classifier_metadata,
        "gpt_image2_detector_path": gpt_detector_path,
        "gpt_image2_detector_metadata": gpt_detector_metadata,
        "binary_gate_path": binary_gate_path,
        "binary_gate_metadata": binary_gate_metadata,
        "binary_gate_mode": binary_gate_mode,
        "generated_gate_threshold": binary_gate_metadata.get("generated_threshold", GENERATOR_BINARY_GATE_THRESHOLD),
        "real_protection_margin": binary_gate_metadata.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN),
        "class_counts": source_distribution,
        "unknown_threshold": unknown_threshold,
        "open_set_unknown_policy": _open_set_policy_report(request, unknown_threshold, base_unknown_threshold),
        "model_card": result.model_card,
        "experiment_profile": request.experiment_profile,
        "experiment_profile_report": profile_report,
        "profile_policy": profile_policy,
        "validation_protocol": split_report,
        "augmentation_protocol": augmentation_protocol,
        "classification_metrics": result.model_card["classification_metrics"],
    }
    return _persist_vision_training_result(result, artifact, request)


def _persist_vision_training_result(
    result: VisionTrainingRunResult,
    artifact: dict[str, object],
    request: VisionTrainingRunRequest,
) -> VisionTrainingRunResult:
    mode = _resolved_activation_mode(request)
    if (
        request.task_type == GENERATOR_ATTRIBUTION_TASK
        and request.experiment_profile != "standard_attribution"
        and request.activation_mode in {"activate", "activate_if_passes_gate"}
    ):
        raise ValueError("非 standard_attribution 的分轨实验只能保存为 candidate，不能直接激活 active。")
    lifecycle = {
        "activation_mode": mode,
        "default_policy": (
            "vision_generator_attribution defaults to candidate; other vision heads default to activate"
        ),
        "does_not_change_training_samples": True,
    }
    result.model_card["lifecycle"] = lifecycle
    artifact["model_card"] = result.model_card
    if mode == "candidate":
        result.status = "candidate_trained"
        result.training_trace.append("本次训练保存为 candidate，未改变 active 模型；需显式激活后才进入正式研判。")
        save_vision_training_run(result, artifact, activate=False)
        return result
    if mode == "activate":
        result.status = "active_trained"
        result.training_trace.append("本次训练已显式保存为 active 模型。")
        save_vision_training_run(result, artifact, activate=True)
        return result
    result.status = "candidate_trained"
    result.training_trace.append("本次训练先保存为 candidate，再执行默认门控评估；只有通过门控才激活。")
    save_vision_training_run(result, artifact, activate=False)
    evaluation = evaluate_vision_candidate(
        VisionCandidateEvaluationRequest(
            task_type=result.task_type,
            candidate_model_id=result.id,
            limit=min(max(result.sample_count, 2), 120),
            activate_if_passes_gate=True,
        )
    )
    result.model_card["activation_gate"] = {
        "evaluation_id": evaluation.id,
        "passed": evaluation.gate.get("passed"),
        "reason": evaluation.gate.get("reason"),
        "checks": evaluation.gate.get("checks", []),
        "active_model_id_before": evaluation.active_model_id_before,
        "active_model_id_after": evaluation.active_model_id_after,
    }
    if evaluation.activated:
        result.status = "active_trained"
        result.training_trace.append("候选模型通过门控并已自动激活。")
    else:
        result.status = "rejected_by_gate"
        result.training_trace.append("候选模型未通过门控，保留记录但未改变 active 模型。")
    update_vision_training_run_payload(result)
    return result


def _resolved_activation_mode(request: VisionTrainingRunRequest) -> str:
    if request.activation_mode:
        return request.activation_mode
    return "candidate" if request.task_type == GENERATOR_ATTRIBUTION_TASK else "activate"


def _build_generator_augmentation_rows(
    *,
    samples: list[ExternalTrainingSample],
    labels: list[str],
    train_indices: list[int],
    request: VisionTrainingRunRequest,
) -> tuple[list[dict[str, float]], list[str], dict[str, object], list[str]]:
    requested_conditions = list(dict.fromkeys(request.augmentation_conditions))
    supported_conditions = set(_supported_robustness_conditions())
    unknown_conditions = [condition for condition in requested_conditions if condition not in supported_conditions]
    if unknown_conditions:
        supported = "、".join(_supported_robustness_conditions())
        raise ValueError(f"不支持的训练扰动增强条件：{unknown_conditions}；可用条件：{supported}。")
    perturbation_conditions = [condition for condition in requested_conditions if condition != "clean"]
    base_protocol: dict[str, object] = {
        "enabled": bool(request.enable_perturbation_augmentation),
        "requested_conditions": requested_conditions,
        "conditions": perturbation_conditions,
        "original_train_count": len(train_indices),
        "clean_validation_policy": "validation split uses only original clean holdout samples",
        "validation_policy": "clean holdout only",
        "not_written_to_dataset": True,
        "max_augmented_samples": request.max_augmented_samples,
        "generated_augmentation_count": 0,
        "augmented_train_count": len(train_indices),
        "condition_counts": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "skipped_count": 0,
    }
    if not request.enable_perturbation_augmentation:
        return [], [], base_protocol, []
    if request.max_augmented_samples <= 0:
        base_protocol["note"] = "已开启扰动增强，但 max_augmented_samples=0，因此没有生成增强训练样本。"
        return [], [], base_protocol, []
    if not perturbation_conditions:
        raise ValueError("开启传播扰动增强时，augmentation_conditions 至少需要包含一个非 clean 条件。")

    augmented_rows: list[dict[str, float]] = []
    augmented_labels: list[str] = []
    augmented_source_keys: list[str] = []
    condition_counts: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    skipped_count = 0
    with tempfile.TemporaryDirectory(prefix="smartpolice-train-aug-") as temp_dir:
        temp_root = Path(temp_dir)
        for source_index in _source_balanced_index_order(samples, labels, train_indices):
            for condition in perturbation_conditions:
                if len(augmented_rows) >= request.max_augmented_samples:
                    break
                sample = samples[source_index]
                source_path = Path(sample.image_path or "")
                if not source_path.is_file():
                    skipped_count += 1
                    continue
                try:
                    text = f"{sample.title} {sample.content} {sample.scenario}"
                    features, cache_hit = _cached_generator_augmentation_features(
                        source_path=source_path,
                        source_sha=sample.image_sha256,
                        condition=condition,
                        temp_root=temp_root,
                        variant_index=len(augmented_rows),
                        text=text,
                    )
                    augmented_rows.append(features)
                    augmented_labels.append(labels[source_index])
                    augmented_source_keys.append(_source_holdout_group_name(sample, "dataset_source"))
                    condition_counts[condition] += 1
                    if cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                except (OSError, ValueError):
                    skipped_count += 1
                    continue
            if len(augmented_rows) >= request.max_augmented_samples:
                break

    base_protocol.update(
        {
            "generated_augmentation_count": len(augmented_rows),
            "augmented_train_count": len(train_indices) + len(augmented_rows),
            "condition_counts": dict(sorted(condition_counts.items())),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "skipped_count": skipped_count,
            "sampling_policy": "按类别与数据来源轮转生成各扰动条件，避免增强额度被单一来源或单一类别耗尽。",
            "feature_cache_policy": "增强特征按 原图 sha256 + 扰动条件 + 清洗文本摘要 + extractor_version 缓存，避免重复生成临时扰动图。",
            "storage_policy": "扰动图片写入临时目录，仅抽取训练特征；临时文件训练后删除，样本不进入 external_training_samples。",
            "label_policy": "增强样本继承原始训练样本来源标签，仅用于监督鲁棒性增强。",
        }
    )
    return augmented_rows, augmented_labels, base_protocol, augmented_source_keys


def _cached_generator_augmentation_features(
    *,
    source_path: Path,
    source_sha: str | None,
    condition: str,
    temp_root: Path,
    variant_index: int,
    text: str,
) -> tuple[dict[str, float], bool]:
    digest = source_sha or hashlib.sha256(source_path.read_bytes()).hexdigest()
    safe_text = _sanitize_generator_context(text)
    text_digest = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()[:20]
    cache_key = f"{digest}:{condition}:{text_digest}:{GENERATOR_AUGMENTATION_EXTRACTOR_VERSION}"
    cached = get_feature_cache(cache_key)
    if cached is not None:
        return {
            str(key): float(value)
            for key, value in cached.payload.items()
            if isinstance(value, int | float)
        }, True

    temp_path, temp_sha = _write_robustness_variant(
        source_path,
        condition,
        temp_root,
        variant_index,
    )
    features = _generator_attribution_features(str(temp_path), temp_sha, text)
    save_feature_cache(
        FeatureCacheRecord(
            id=f"gen-aug-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:20]}",
            cache_key=cache_key,
            extractor_version=GENERATOR_AUGMENTATION_EXTRACTOR_VERSION,
            modality="image_augmentation",
            sha256=digest,
            payload=features,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return features, False


def _image_features(image_path: str | None, image_sha256: str | None) -> dict[str, float]:
    if not image_path:
        return _empty_image_features()
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return _empty_image_features()
    digest = image_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    cache_key = f"{digest}:{EXTRACTOR_VERSION}"
    cached = get_feature_cache(cache_key)
    if cached is not None:
        return {str(key): float(value) for key, value in cached.payload.items() if isinstance(value, int | float)}
    raw = path.read_bytes()
    width, height = _image_size(raw)
    histogram = Counter(raw)
    total = max(1, len(raw))
    byte_mean = sum(byte * count for byte, count in histogram.items()) / total
    variance = sum(((byte - byte_mean) ** 2) * count for byte, count in histogram.items()) / total
    entropy = -sum((count / total) * math.log2(count / total) for count in histogram.values())
    suffix = path.suffix.lower()
    features = {
        "image_bytes_log": min(math.log1p(len(raw)) / 3.0, 8.0),
        "image_megapixels": min(((width or 0) * (height or 0)) / 1_000_000.0, 24.0),
        "aspect_ratio": min((width or 1) / max(height or 1, 1), 6.0),
        "byte_mean": byte_mean / 255.0,
        "byte_std": math.sqrt(variance) / 128.0,
        "byte_zero_ratio": histogram.get(0, 0) / total,
        "byte_high_ratio": sum(count for byte, count in histogram.items() if byte >= 240) / total,
        "byte_entropy": entropy / 8.0,
        "png_ext": 1.0 if suffix == ".png" else 0.0,
        "jpg_ext": 1.0 if suffix in {".jpg", ".jpeg"} else 0.0,
        "webp_ext": 1.0 if suffix == ".webp" else 0.0,
        **_pixel_features(raw),
    }
    save_feature_cache(
        FeatureCacheRecord(
            id=f"feat-{uuid4().hex[:12]}",
            cache_key=cache_key,
            extractor_version=EXTRACTOR_VERSION,
            modality="image",
            sha256=digest,
            payload=features,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return features


def _empty_image_features() -> dict[str, float]:
    return {
        "image_bytes_log": 0.0,
        "image_megapixels": 0.0,
        "aspect_ratio": 0.0,
        "byte_mean": 0.0,
        "byte_std": 0.0,
        "byte_zero_ratio": 0.0,
        "byte_high_ratio": 0.0,
        "byte_entropy": 0.0,
        "png_ext": 0.0,
        "jpg_ext": 0.0,
        "webp_ext": 0.0,
        **_empty_pixel_features(),
    }


def _pixel_features(raw: bytes) -> dict[str, float]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
    except OSError:
        return _empty_pixel_features()

    width, height = rgb.size
    if width <= 0 or height <= 0:
        return _empty_pixel_features()

    sample = rgb.copy()
    sample.thumbnail((192, 192))
    luma = sample.convert("L")
    luma_stat = ImageStat.Stat(luma)
    luma_values = list(luma.getdata())
    total = max(1, len(luma_values))
    hsv = sample.convert("HSV")
    saturation = hsv.getchannel("S")
    sat_stat = ImageStat.Stat(saturation)
    rgb_stat = ImageStat.Stat(sample)

    edges = luma.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    blurred = luma.filter(ImageFilter.GaussianBlur(radius=1.4))
    residual = ImageChops.difference(luma, blurred)
    residual_stat = ImageStat.Stat(residual)
    compression_residual = _compression_residual(sample)
    gradient_features = _gradient_frequency_features(luma)
    overlay_features = _overlay_artifact_features(edges)
    megapixels = width * height / 1_000_000.0
    aspect_ratio = width / max(height, 1)

    return {
        "pixel_luma_mean": _unit(luma_stat.mean[0]),
        "pixel_luma_std": min(luma_stat.stddev[0] / 128.0, 2.0),
        "pixel_dark_ratio": sum(1 for value in luma_values if value < 35) / total,
        "pixel_bright_ratio": sum(1 for value in luma_values if value > 220) / total,
        "pixel_saturation_mean": _unit(sat_stat.mean[0]),
        "pixel_saturation_std": min(sat_stat.stddev[0] / 128.0, 2.0),
        "pixel_red_mean": _unit(rgb_stat.mean[0]),
        "pixel_green_mean": _unit(rgb_stat.mean[1]),
        "pixel_blue_mean": _unit(rgb_stat.mean[2]),
        "edge_density": sum(1 for value in edges.getdata() if value > 32) / total,
        "edge_strength": _unit(edge_stat.mean[0]),
        "texture_residual_mean": _unit(residual_stat.mean[0]),
        "texture_residual_std": min(residual_stat.stddev[0] / 128.0, 2.0),
        "compression_residual_mean": _unit(compression_residual.mean[0]),
        "compression_residual_std": min(compression_residual.stddev[0] / 128.0, 2.0),
        **gradient_features,
        **overlay_features,
        "small_image_signal": 1.0 if megapixels < 0.08 else 0.0,
        "screenshot_shape_signal": 1.0 if 0.45 <= aspect_ratio <= 2.4 else 0.0,
    }


def _compression_residual(image: Image.Image) -> ImageStat.Stat:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        diff = ImageChops.difference(image, recompressed.convert("RGB")).convert("L")
    return ImageStat.Stat(diff)


def _gradient_frequency_features(luma: Image.Image) -> dict[str, float]:
    width, height = luma.size
    if width < 3 or height < 3:
        return {
            "frequency_high_energy_proxy": 0.0,
            "frequency_mid_energy_proxy": 0.0,
            "jpeg_block_boundary_delta": 0.0,
            "horizontal_gradient_energy": 0.0,
            "vertical_gradient_energy": 0.0,
        }
    pixels = list(luma.getdata())
    horizontal_values: list[int] = []
    vertical_values: list[int] = []
    boundary_values: list[int] = []
    non_boundary_values: list[int] = []
    for y_pos in range(height):
        row = y_pos * width
        for x_pos in range(1, width):
            delta = abs(pixels[row + x_pos] - pixels[row + x_pos - 1])
            horizontal_values.append(delta)
            if x_pos % 8 == 0:
                boundary_values.append(delta)
            elif x_pos % 8 in {3, 4, 5}:
                non_boundary_values.append(delta)
    for y_pos in range(1, height):
        row = y_pos * width
        prev = (y_pos - 1) * width
        for x_pos in range(width):
            delta = abs(pixels[row + x_pos] - pixels[prev + x_pos])
            vertical_values.append(delta)
            if y_pos % 8 == 0:
                boundary_values.append(delta)
            elif y_pos % 8 in {3, 4, 5}:
                non_boundary_values.append(delta)
    horizontal = mean(horizontal_values) if horizontal_values else 0.0
    vertical = mean(vertical_values) if vertical_values else 0.0
    boundary = mean(boundary_values) if boundary_values else 0.0
    non_boundary = mean(non_boundary_values) if non_boundary_values else 0.0
    mid_residual = ImageChops.difference(luma, luma.filter(ImageFilter.GaussianBlur(radius=2.2)))
    mid_stat = ImageStat.Stat(mid_residual)
    return {
        "frequency_high_energy_proxy": min(((horizontal + vertical) / 2.0) / 255.0, 1.0),
        "frequency_mid_energy_proxy": _unit(mid_stat.mean[0]),
        "jpeg_block_boundary_delta": max(0.0, min((boundary - non_boundary) / 255.0, 1.0)),
        "horizontal_gradient_energy": min(horizontal / 255.0, 1.0),
        "vertical_gradient_energy": min(vertical / 255.0, 1.0),
    }


def _overlay_artifact_features(edges: Image.Image) -> dict[str, float]:
    width, height = edges.size
    if width <= 0 or height <= 0:
        return {
            "text_overlay_edge_density": 0.0,
            "corner_watermark_edge_signal": 0.0,
        }
    values = list(edges.getdata())

    def region_density(x0: int, y0: int, x1: int, y1: int) -> float:
        x0 = max(0, min(x0, width))
        x1 = max(0, min(x1, width))
        y0 = max(0, min(y0, height))
        y1 = max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        total = 0
        active = 0
        for y_pos in range(y0, y1):
            row = y_pos * width
            for x_pos in range(x0, x1):
                total += 1
                if values[row + x_pos] > 32:
                    active += 1
        return active / max(total, 1)

    lower_band = region_density(0, int(height * 0.62), width, height)
    upper_band = region_density(0, 0, width, int(height * 0.22))
    corner_w = max(1, int(width * 0.28))
    corner_h = max(1, int(height * 0.22))
    corner_scores = [
        region_density(0, 0, corner_w, corner_h),
        region_density(width - corner_w, 0, width, corner_h),
        region_density(0, height - corner_h, corner_w, height),
        region_density(width - corner_w, height - corner_h, width, height),
    ]
    return {
        "text_overlay_edge_density": round(max(lower_band, upper_band), 6),
        "corner_watermark_edge_signal": round(max(corner_scores), 6),
    }


def _empty_pixel_features() -> dict[str, float]:
    return {
        "pixel_luma_mean": 0.0,
        "pixel_luma_std": 0.0,
        "pixel_dark_ratio": 0.0,
        "pixel_bright_ratio": 0.0,
        "pixel_saturation_mean": 0.0,
        "pixel_saturation_std": 0.0,
        "pixel_red_mean": 0.0,
        "pixel_green_mean": 0.0,
        "pixel_blue_mean": 0.0,
        "edge_density": 0.0,
        "edge_strength": 0.0,
        "texture_residual_mean": 0.0,
        "texture_residual_std": 0.0,
        "compression_residual_mean": 0.0,
        "compression_residual_std": 0.0,
        "frequency_high_energy_proxy": 0.0,
        "frequency_mid_energy_proxy": 0.0,
        "jpeg_block_boundary_delta": 0.0,
        "horizontal_gradient_energy": 0.0,
        "vertical_gradient_energy": 0.0,
        "text_overlay_edge_density": 0.0,
        "corner_watermark_edge_signal": 0.0,
        "small_image_signal": 0.0,
        "screenshot_shape_signal": 0.0,
    }


def _unit(value: float) -> float:
    return max(0.0, min(value / 255.0, 1.0))


def _clip_features(image_path: str | None, image_sha256: str | None, text: str) -> dict[str, float]:
    empty = _empty_clip_features()
    if not CLIP_ENABLED or not image_path or not text.strip():
        return empty
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return empty
    digest = image_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    cache_key = f"{digest}:{text_digest}:{CLIP_EXTRACTOR_VERSION}"
    cached = get_feature_cache(cache_key)
    if cached is not None:
        return {str(key): float(value) for key, value in cached.payload.items() if isinstance(value, int | float)}
    extracted = _extract_clip_features(path, text)
    if extracted is None:
        return empty
    save_feature_cache(
        FeatureCacheRecord(
            id=f"clip-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:24]}",
            cache_key=cache_key,
            extractor_version=CLIP_EXTRACTOR_VERSION,
            modality="image_text",
            sha256=digest,
            payload=extracted,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return extracted


def _empty_clip_features() -> dict[str, float]:
    features = {
        "clip_enabled": 0.0,
        "clip_similarity": 0.0,
        "clip_distance": 0.0,
        "clip_abs_gap_mean": 0.0,
    }
    for index in range(CLIP_FEATURE_DIMS):
        features[f"clip_img_{index:02d}"] = 0.0
        features[f"clip_txt_{index:02d}"] = 0.0
        features[f"clip_gap_{index:02d}"] = 0.0
    return features


def _extract_clip_features(path: Path, text: str) -> dict[str, float] | None:
    bundle = _clip_bundle()
    if bundle is None:
        return None
    model, processor = bundle
    try:
        import torch

        image = Image.open(path).convert("RGB")
        inputs = processor(text=[text[:512]], images=image, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_values = image_features[0].detach().cpu().tolist()
            text_values = text_features[0].detach().cpu().tolist()
    except Exception:
        return None
    dims = min(CLIP_FEATURE_DIMS, len(image_values), len(text_values))
    similarity = sum(float(image_values[index]) * float(text_values[index]) for index in range(len(image_values)))
    gap_values = [abs(float(image_values[index]) - float(text_values[index])) for index in range(dims)]
    features = _empty_clip_features()
    features["clip_enabled"] = 1.0
    features["clip_similarity"] = round(float(similarity), 6)
    features["clip_distance"] = round(1.0 - float(similarity), 6)
    features["clip_abs_gap_mean"] = round(mean(gap_values) if gap_values else 0.0, 6)
    for index in range(dims):
        image_value = float(image_values[index])
        text_value = float(text_values[index])
        features[f"clip_img_{index:02d}"] = round(image_value, 6)
        features[f"clip_txt_{index:02d}"] = round(text_value, 6)
        features[f"clip_gap_{index:02d}"] = round(abs(image_value - text_value), 6)
    return features


def _clip_embedding_pair(image_path: str | None, image_sha256: str | None, text: str) -> tuple[list[float], list[float]] | None:
    if not CLIP_ENABLED or not image_path or not text.strip():
        return None
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    digest = image_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    cache_key = f"{digest}:{text_digest}:{CLIP_EXTRACTOR_VERSION}:embedding"
    cached = get_feature_cache(cache_key)
    if cached is not None:
        image_values = cached.payload.get("image_embedding")
        text_values = cached.payload.get("text_embedding")
        if isinstance(image_values, list) and isinstance(text_values, list):
            return (
                [float(value) for value in image_values if isinstance(value, int | float)],
                [float(value) for value in text_values if isinstance(value, int | float)],
            )
    bundle = _clip_bundle()
    if bundle is None:
        return None
    model, processor = bundle
    try:
        import torch

        image = Image.open(path).convert("RGB")
        inputs = processor(text=[text[:512]], images=image, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_values = [round(float(value), 6) for value in image_features[0].detach().cpu().tolist()]
            text_values = [round(float(value), 6) for value in text_features[0].detach().cpu().tolist()]
    except Exception:
        return None
    save_feature_cache(
        FeatureCacheRecord(
            id=f"clip-emb-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:20]}",
            cache_key=cache_key,
            extractor_version=f"{CLIP_EXTRACTOR_VERSION}:embedding",
            modality="image_text_embedding",
            sha256=digest,
            payload={
                "image_embedding": image_values,
                "text_embedding": text_values,
            },
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return image_values, text_values


def _clip_bundle() -> tuple[object, object] | None:
    global _CLIP_BUNDLE, _CLIP_LOAD_ERROR
    if _CLIP_BUNDLE is not None:
        return _CLIP_BUNDLE
    if _CLIP_LOAD_ERROR:
        return None
    try:
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        model.eval()
        _CLIP_BUNDLE = (model, processor)
    except Exception as exc:
        _CLIP_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    return _CLIP_BUNDLE


def _text_context_features(text: str) -> dict[str, float]:
    normalized = text.lower()
    features = {"text_len_log": min(math.log1p(len(normalized)) / 3.0, 8.0)}
    for name, keywords in KEYWORD_GROUPS.items():
        features[name] = _keyword_score(normalized, keywords)
    return features


def _text_risk_heuristic(sample: ExternalTrainingSample) -> float:
    text = f"{sample.title} {sample.content} {sample.scenario}".lower()
    score = 22.0
    score += _keyword_score(text, KEYWORD_GROUPS["public_safety_keywords"]) * 10
    score += _keyword_score(text, KEYWORD_GROUPS["aigc_keywords"]) * 8
    score += _keyword_score(text, KEYWORD_GROUPS["tamper_keywords"]) * 7
    score += _keyword_score(text, KEYWORD_GROUPS["mismatch_keywords"]) * 9
    return _clip_score(score)


def _predict_vision_asset(
    task_type: str,
    artifact: dict[str, object],
    asset: CaseAsset,
    case_text: str,
) -> dict[str, object]:
    features = {
        **_image_features(asset.storage_path, asset.sha256),
        **_text_context_features(case_text),
        f"task::{task_type}": 1.0,
    }
    prediction = _predict_from_artifact(artifact, features)
    if prediction is None:
        return {
            "asset_id": asset.id,
            "score": None,
            "note": "模型 artifact 不完整。",
        }
    score, contributions = prediction
    return {
        "asset_id": asset.id,
        "score": score,
        "risk_level": risk_level_from_score(score).value,
        "top_contributions": _contribution_payload(contributions),
    }


def _predict_generator_attribution_for_assets(
    artifact: dict[str, object],
    assets: list[CaseAsset],
    case_text: str,
) -> dict[str, object]:
    predictions = [
        _predict_generator_attribution_asset(artifact, asset, case_text)
        for asset in assets
    ]
    candidates = [
        prediction
        for prediction in predictions
        if isinstance(prediction.get("top_candidate"), str)
        and prediction.get("top_candidate") != "unknown"
    ]
    if candidates:
        best = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
        top_candidate = str(best.get("top_candidate"))
        confidence = round(float(best.get("confidence", 0.0)), 3)
    else:
        top_candidate = "unknown"
        confidence = round(
            max((float(item.get("confidence", 0.0)) for item in predictions), default=0.0),
            3,
        )
    ranked_candidates = _aggregate_generator_candidate_ranking(predictions)
    return {
        "trained": True,
        "enabled": True,
        "model_id": str(artifact.get("id", "")),
        "model_kind": str(artifact.get("model_kind", "local-generator-attribution-prototype-v1")),
        "top_candidate": top_candidate,
        "confidence": confidence,
        "unknown": top_candidate == "unknown",
        "score": round(confidence * 100.0, 2) if top_candidate != "unknown" else None,
        "ranked_candidates": ranked_candidates,
        "candidate_ranking": ranked_candidates,
        "asset_predictions": predictions,
        "boundary": "生成模型归因只能作为来源线索；低置信输出 unknown，不替代 C2PA、水印、平台元数据或人工核验。",
    }


def _predict_generator_attribution_asset(
    artifact: dict[str, object],
    asset: CaseAsset,
    case_text: str,
) -> dict[str, object]:
    feature_names = _list_value(artifact.get("feature_names"))
    means = _float_mapping(artifact.get("means"))
    scales = _float_mapping(artifact.get("scales"))
    prototypes = _artifact_prototypes(artifact.get("class_prototypes"))
    if not feature_names or not means or not scales or not prototypes:
        return {
            "asset_id": asset.id,
            "top_candidate": "unknown",
            "confidence": 0.0,
            "unknown": True,
            "note": "生成模型归因 artifact 不完整。",
        }
    features = _generator_attribution_features(asset.storage_path, asset.sha256, case_text)
    prediction = _predict_generator_label(
        features,
        feature_names,
        means,
        scales,
        prototypes,
        float(artifact.get("unknown_threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)),
        classifier_path=str(artifact.get("classifier_path") or ""),
        gpt_detector_path=str(artifact.get("gpt_image2_detector_path") or ""),
        binary_gate_path=str(artifact.get("binary_gate_path") or ""),
        generated_gate_threshold=float(artifact.get("generated_gate_threshold", GENERATOR_BINARY_GATE_THRESHOLD)),
        gpt_detector_threshold=float(
            (artifact.get("gpt_image2_detector_metadata") or {}).get("threshold", GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR)
            if isinstance(artifact.get("gpt_image2_detector_metadata"), dict)
            else GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR
        ),
        real_protection_margin=float(artifact.get("real_protection_margin", GENERATOR_REAL_PROTECTION_MARGIN)),
        binary_gate_mode=str(artifact.get("binary_gate_mode", "enforce")),
        open_set_min_margin=float(
            (artifact.get("open_set_unknown_policy") or {}).get("open_set_min_margin", 0.0)
            if isinstance(artifact.get("open_set_unknown_policy"), dict)
            else 0.0
        ),
    )
    candidate_ranking = _generator_candidate_ranking(prediction.get("candidates"))
    return {
        "asset_id": asset.id,
        "top_candidate": prediction["label"],
        "confidence": prediction["confidence"],
        "unknown": prediction["label"] == "unknown",
        "candidates": prediction["candidates"],
        "ranked_candidates": candidate_ranking,
        "candidate_ranking": candidate_ranking,
        "binary_gate": prediction.get("binary_gate"),
        "real_photo_guard": prediction.get("real_photo_guard"),
        "review_recommendation": (
            prediction.get("binary_gate", {}).get("review_recommendation")
            if isinstance(prediction.get("binary_gate"), dict)
            else None
        ),
        "gate_reason": prediction.get("gate_reason"),
        "nearest_distance": prediction["distance"],
        "top_contributions": _generator_contribution_payload(
            prediction["prototype"],
            features,
            feature_names,
            means,
            scales,
        ),
        "boundary": "疑似来源线索，需结合水印、C2PA、平台元数据和人工核验。",
    }


def _generator_candidate_ranking(raw_candidates: object) -> list[dict[str, object]]:
    if not isinstance(raw_candidates, list):
        return []
    ranked: list[dict[str, object]] = []
    cleaned: list[dict[str, object]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("candidate") or "unknown")
        confidence = float(item.get("confidence", 0.0) or 0.0) if isinstance(item.get("confidence"), int | float) else 0.0
        cleaned.append({"label": label, "confidence": confidence})
    for index, item in enumerate(
        sorted(cleaned, key=lambda candidate: float(candidate["confidence"]), reverse=True)[:8],
        start=1,
    ):
        label = str(item["label"])
        confidence = round(float(item["confidence"]), 3)
        ranked.append(
            {
                "rank": index,
                "label": label,
                "display_name": _generator_display_source_label(label),
                "probability": confidence,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100),
            }
        )
    return ranked


def _aggregate_generator_candidate_ranking(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for prediction in predictions:
        ranking = prediction.get("candidate_ranking")
        if not isinstance(ranking, list):
            ranking = prediction.get("ranked_candidates")
        if not isinstance(ranking, list):
            ranking = _generator_candidate_ranking(prediction.get("candidates"))
        for item in ranking:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "unknown")
            probability = item.get("probability")
            value = float(probability) if isinstance(probability, int | float) else 0.0
            totals[label] = totals.get(label, 0.0) + value
            counts[label] = counts.get(label, 0) + 1
    averaged = [
        {"label": label, "confidence": totals[label] / max(1, counts.get(label, 1))}
        for label in totals
    ]
    return _generator_candidate_ranking(averaged)


def _generator_display_source_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    labels = {
        "gpt-image2": "GPT-image-2",
        "gpt-image-2": "GPT-image-2",
        "gpt image2": "GPT-image-2",
        "gpt-image1": "GPT-image-1",
        "gpt-image1.5": "GPT-image-1.5",
        "stable-diffusion": "Stable Diffusion",
        "midjourney": "Midjourney",
        "nano-banana": "Nano Banana",
        "seedream-4": "Seedream-4",
        "flux": "Flux",
        "dall-e": "DALL-E",
        "dall-e-3": "DALL-E 3",
        "other-generated": "其他生成模型",
        "real": "真实照片",
        "unknown": "未知/低置信",
        "not-gpt-image2": "非 GPT-image-2",
    }
    return labels.get(normalized, label or "待分析")


def _predict_sample_with_artifact(
    task_type: str,
    artifact: dict[str, object] | None,
    sample: ExternalTrainingSample,
) -> float | None:
    if artifact is None:
        return None
    features = extract_sample_features(sample, task_type=task_type)
    prediction = _predict_from_artifact(artifact, features)
    return prediction[0] if prediction else None


def _predict_from_artifact(
    artifact: dict[str, object],
    features: dict[str, float],
) -> tuple[float, list[tuple[str, float, float]]] | None:
    feature_names = _list_value(artifact.get("feature_names"))
    weights = _float_mapping(artifact.get("weights"))
    means = _float_mapping(artifact.get("means"))
    scales = _float_mapping(artifact.get("scales"))
    if not feature_names or not weights or not means or not scales:
        return None
    normalized = _normalize(features, means, scales)
    bias = float(artifact.get("bias", 0.0))
    ridge_raw = bias + sum(weights.get(name, 0.0) * normalized.get(name, 0.0) for name in feature_names)
    ridge_score = _clip_score(ridge_raw)
    ensemble = artifact.get("ensemble")
    selected_model = "ridge"
    alpha = 1.0
    knn_k = 5
    if isinstance(ensemble, dict):
        selected_model = str(ensemble.get("selected_model", "ridge"))
        alpha = float(ensemble.get("alpha", 1.0))
        knn_k = int(ensemble.get("knn_k", 5))
    tree_score = _predict_tree_from_artifact(artifact, normalized, feature_names)
    prototypes = _artifact_prototypes(artifact.get("knn_prototypes"))
    knn_score = _predict_knn_from_normalized(normalized, feature_names, prototypes, knn_k)
    clip_score = _predict_clip_from_artifact(artifact, features, knn_k)
    if selected_model == "tree" and tree_score is not None:
        score = tree_score
    elif selected_model == "clip_knn" and clip_score is not None:
        score = clip_score
    elif selected_model == "knn" and knn_score is not None:
        score = knn_score
    elif selected_model == "ensemble" and knn_score is not None:
        score = _clip_score(alpha * ridge_score + (1.0 - alpha) * knn_score)
    elif selected_model == "tree_ensemble" and tree_score is not None and knn_score is not None:
        score = _clip_score(alpha * tree_score + (1.0 - alpha) * knn_score)
    else:
        score = ridge_score
    contributions = [
        (name, weights.get(name, 0.0) * normalized.get(name, 0.0), features.get(name, 0.0))
        for name in feature_names
    ]
    ordered = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    return score, ordered


def _task_status(task_type: str) -> TrainingTaskStatus:
    status = get_training_data_status()
    for task in status.tasks:
        if task.task_type == task_type:
            return task
    return TrainingTaskStatus(
        task_type=task_type,
        sample_count=0,
        image_available_count=0,
        label_distribution={},
        training_ready=False,
        sources=[],
        note="尚未导入该任务样本。",
    )


def _image_size(raw: bytes) -> tuple[int | None, int | None]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw.startswith(b"\xff\xd8"):
        return _jpeg_size(raw)
    if len(raw) >= 30 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        if raw[12:16] == b"VP8X":
            return int.from_bytes(raw[24:27], "little") + 1, int.from_bytes(raw[27:30], "little") + 1
    return None, None


def _jpeg_size(raw: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index + 9 < len(raw):
        if raw[index] != 0xFF:
            index += 1
            continue
        marker = raw[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[index:index + 2], "big")
        if marker in range(0xC0, 0xCF) and marker not in {0xC4, 0xC8, 0xCC}:
            if index + 7 <= len(raw):
                height = int.from_bytes(raw[index + 3:index + 5], "big")
                width = int.from_bytes(raw[index + 5:index + 7], "big")
                return width, height
        index += max(segment_length, 2)
    return None, None


def _task_relevant_samples(
    samples: list[ExternalTrainingSample],
    task_type: str,
) -> list[ExternalTrainingSample]:
    selected = [sample for sample in samples if _sample_matches_task(sample, task_type)]
    return selected or samples


def _generator_experiment_view(
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    request: VisionTrainingRunRequest | VisionSourceHoldoutRunRequest,
) -> tuple[list[ExternalTrainingSample], list[dict[str, float]], list[str], dict[str, object]]:
    profile = getattr(request, "experiment_profile", "standard_attribution")
    if profile not in GENERATOR_EXPERIMENT_PROFILES:
        raise ValueError(f"不支持的生成归因实验 profile：{profile}")
    domains = [_generator_sample_domain(sample) for sample in samples]
    base_labels = [_normalize_generator_label(sample.label) for sample in samples]
    source_counts_by_label = _source_counts_by_label(samples, base_labels)
    kept_samples: list[ExternalTrainingSample] = []
    kept_rows: list[dict[str, float]] = []
    labels: list[str] = []
    dropped_reasons: Counter[str] = Counter()

    for sample, row, label, domain in zip(samples, rows, base_labels, domains, strict=True):
        mapped = _map_generator_profile_label(
            label=label,
            domain=domain,
            profile=profile,
            source_counts_by_label=source_counts_by_label,
        )
        if mapped is None:
            dropped_reasons[f"excluded:{domain}"] += 1
            continue
        kept_samples.append(sample)
        kept_rows.append(row)
        labels.append(mapped)

    report = {
        "profile": profile,
        "input_count": len(samples),
        "selected_count": len(kept_samples),
        "excluded_count": len(samples) - len(kept_samples),
        "domain_distribution": dict(sorted(Counter(domains).items())),
        "selected_domain_distribution": dict(
            sorted(Counter(_generator_sample_domain(sample) for sample in kept_samples).items())
        ),
        "base_label_distribution": dict(sorted(Counter(base_labels).items())),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "dropped_reasons": dict(sorted(dropped_reasons.items())),
        "label_policy": _generator_profile_label_policy(profile),
        "source_coverage": {
            label: len(sources)
            for label, sources in sorted(source_counts_by_label.items())
        },
    }
    return kept_samples, kept_rows, labels, report


def _generator_profile_policy(profile: str) -> dict[str, object]:
    if profile not in GENERATOR_PROFILE_POLICIES:
        raise ValueError(f"不支持的生成归因实验 profile：{profile}")
    return deepcopy(GENERATOR_PROFILE_POLICIES[profile])


def _generator_profile_feature_policy(feature_names: list[str], profile: str) -> dict[str, object]:
    raw_names = list(feature_names)
    source_guard_disabled = os.getenv("SMARTPOLICE_GENERATOR_SOURCE_GUARD", "").lower() in {"0", "false", "no", "off"}
    source_guard_enabled = not source_guard_disabled
    source_guard_profiles = {
        "standard_attribution",
        "binary_generated_gate",
        "gpt_image2_ovr",
        "multi_generator_label_covered",
        "mainstream_five_attribution",
        "social_propagation_robustness",
        "clean_origin_attribution",
    }
    if profile not in source_guard_profiles or not source_guard_enabled:
        return {
            "profile": profile,
            "strategy": "all_features",
            "feature_names": raw_names,
            "removed_feature_names": [],
            "removed_count": 0,
            "kept_count": len(raw_names),
            "source_guard_enabled": source_guard_enabled,
        }
    blocked_exact = {
        "image_bytes_log",
        "png_ext",
        "jpg_ext",
        "webp_ext",
        "small_image_signal",
        "text_enriched_image_context_signal",
        "visual_text_context_signal",
        "watermark_context_signal",
    }
    blocked_prefixes: tuple[str, ...] = ()
    removed = [
        name
        for name in raw_names
        if name in blocked_exact or any(name.startswith(prefix) for prefix in blocked_prefixes)
    ]
    kept = [name for name in raw_names if name not in set(removed)]
    if len(kept) < 8:
        kept = raw_names
        removed = []
        strategy = "fallback_all_features"
    else:
        strategy = "source_artifact_guard"
    return {
        "profile": profile,
        "strategy": strategy,
        "feature_names": kept,
        "removed_feature_names": removed,
        "removed_count": len(removed),
        "kept_count": len(kept),
        "source_guard_enabled": source_guard_enabled,
        "rationale": (
            "生成归因默认过滤扩展名、文件大小、小图标记和文本上下文代理等强来源捷径特征，"
            "保留分辨率/比例、CLIP 语义差距、像素统计、纹理、频域、边缘和压缩残差等可解释视觉特征。"
        ),
    }


def _generator_sample_domain(sample: ExternalTrainingSample) -> str:
    signature = " ".join(
        [
            sample.dataset_name,
            sample.source,
            sample.source_url or "",
            sample.title,
            sample.content,
            sample.scenario,
        ]
    ).lower()
    if "real-negative-pool" in signature or "real_negative" in signature:
        return "real_negative_pool"
    if any(token in signature for token in ("screenshot", "watermark", "repost", "resave", "social", "平台", "转发", "截图", "水印", "压缩", "重保存")):
        return "social_propagation"
    if any(token in signature for token in ("gpt-image-2", "gpt-image2", "openai image")):
        return "gpt_image2_focus"
    if any(token in signature for token in ("genimage", "aigibench", "qwen-image-bench", "defactify", "deepsafe", "ai-detector", "bananamark")):
        return "multi_generator_benchmark"
    return "clean_origin"


def _source_counts_by_label(
    samples: list[ExternalTrainingSample],
    labels: list[str],
) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for sample, label in zip(samples, labels, strict=True):
        if label == "unknown":
            continue
        coverage.setdefault(label, set()).add(_source_holdout_group_name(sample, "dataset_source"))
    return coverage


def _map_generator_profile_label(
    *,
    label: str,
    domain: str,
    profile: str,
    source_counts_by_label: dict[str, set[str]],
) -> str | None:
    if profile == "standard_attribution":
        return _generator_three_way_label(label)
    if profile == "binary_generated_gate":
        return _binary_generation_label(label)
    if profile == "gpt_image2_ovr":
        if label == "gpt-image2":
            return "gpt-image2"
        if label == "real":
            return "real"
        return "other-generated"
    if profile == "mainstream_five_attribution":
        return _mainstream_five_generator_label(label)
    if profile == "multi_generator_label_covered":
        if label == "real":
            return "real"
        if len(source_counts_by_label.get(label, set())) >= 2:
            return label
        return "unknown"
    if profile == "clean_origin_attribution":
        return label if domain in {"clean_origin", "multi_generator_benchmark", "gpt_image2_focus"} else None
    if profile == "social_propagation_robustness":
        if domain in {"social_propagation", "real_negative_pool"} or label != "real":
            return _binary_generation_label(label)
        return None
    return label


def _generator_profile_label_policy(profile: str) -> str:
    policies = {
        "standard_attribution": "三分类：GPT-image2 / other-generated / real；unknown 仅作为预测低置信退让。",
        "binary_generated_gate": "映射为 generated/real 二分类。",
        "gpt_image2_ovr": "GPT-image2 为正类，real 与 other-generated 分开保留。",
        "mainstream_five_attribution": "只保留 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion 系列、Midjourney 五类主流来源；其他生成器映射 unknown。",
        "multi_generator_label_covered": "保留跨多个 dataset_source 的生成器类别；单来源类别映射 unknown。",
        "clean_origin_attribution": "优先 clean/origin 与多生成器 benchmark 样本，排除传播域样本。",
        "social_propagation_robustness": "传播/真实图 hard-negative 保留为 real，对生成图保留 generated 对照，用于传播鲁棒二分类候选。",
    }
    return policies.get(profile, "保持原始标签。")


def _generator_three_way_label(label: str) -> str:
    if label == "gpt-image2":
        return "gpt-image2"
    if label == "real":
        return "real"
    return "other-generated"


def _mainstream_five_generator_label(label: str) -> str:
    if label == "real":
        return "real"
    if label == "gpt-image2":
        return "gpt-image2"
    if label == "nano-banana":
        return "nano-banana"
    if label == "seedream-4":
        return "seedream-4"
    if label == "midjourney":
        return "midjourney"
    if label in {"stable-diffusion", "sd21", "sd3", "sdxl"}:
        return "stable-diffusion"
    return "unknown"


def _balanced_generator_samples_for_request(
    samples: list[ExternalTrainingSample],
    limit: int,
    request: VisionTrainingRunRequest,
) -> list[ExternalTrainingSample]:
    profile = getattr(request, "experiment_profile", "standard_attribution")
    if profile == "standard_attribution":
        return _balanced_standard_generator_three_way_samples(samples, limit)
    if profile == "gpt_image2_ovr":
        return _balanced_gpt_image2_ovr_samples(samples, limit)
    if profile != "mainstream_five_attribution":
        return _balanced_generator_samples(samples, limit)
    return _balanced_mainstream_five_samples(samples, limit)


def _balanced_generator_samples_for_profile(
    samples: list[ExternalTrainingSample],
    limit: int,
    profile: str,
) -> list[ExternalTrainingSample]:
    if profile == "standard_attribution":
        return _balanced_standard_generator_three_way_samples(samples, limit)
    if profile == "gpt_image2_ovr":
        return _balanced_gpt_image2_ovr_samples(samples, limit)
    if profile != "mainstream_five_attribution":
        return _balanced_generator_samples(samples, limit)
    return _balanced_mainstream_five_samples(samples, limit)


def _balanced_standard_generator_three_way_samples(
    samples: list[ExternalTrainingSample],
    limit: int,
) -> list[ExternalTrainingSample]:
    if limit <= 0:
        return list(samples)
    by_label_source: dict[str, dict[str, list[ExternalTrainingSample]]] = {
        "gpt-image2": {},
        "other-generated": {},
        "real": {},
    }
    for sample in samples:
        label = _generator_three_way_label(_normalize_generator_label(sample.label))
        source_key = _source_holdout_group_name(sample, "dataset_source")
        by_label_source[label].setdefault(source_key, []).append(sample)
    label_counts = {
        label: sum(len(bucket) for bucket in source_buckets.values())
        for label, source_buckets in by_label_source.items()
    }
    minority_count = min(max(1, count) for count in label_counts.values())
    balanced_cap = min(minority_count, max(1, limit // 3))
    per_label_cap = {
        label: min(label_counts[label], balanced_cap)
        for label in ("gpt-image2", "real", "other-generated")
    }
    label_orders = {
        label: _source_round_robin_samples(source_buckets, source_offset=offset)
        for offset, (label, source_buckets) in enumerate(by_label_source.items())
    }
    selected: list[ExternalTrainingSample] = []
    selected_ids: set[str] = set()
    for label in ("gpt-image2", "real", "other-generated"):
        for sample in label_orders.get(label, [])[:per_label_cap[label]]:
            if sample.id in selected_ids:
                continue
            selected.append(sample)
            selected_ids.add(sample.id)
    should_backfill = len(selected) < min(limit, 8)
    if should_backfill:
        for label in ("gpt-image2", "real", "other-generated"):
            for sample in label_orders.get(label, []):
                if sample.id in selected_ids:
                    continue
                selected.append(sample)
                selected_ids.add(sample.id)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
    return selected[:limit]


def _balanced_gpt_image2_ovr_samples(
    samples: list[ExternalTrainingSample],
    limit: int,
) -> list[ExternalTrainingSample]:
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    by_label_source: dict[str, dict[str, list[ExternalTrainingSample]]] = {
        "gpt-image2": {},
        "other-generated": {},
        "real": {},
    }
    for sample in samples:
        raw_label = _normalize_generator_label(sample.label)
        if raw_label == "gpt-image2":
            label = "gpt-image2"
        elif raw_label == "real":
            label = "real"
        else:
            label = "other-generated"
        source_key = _source_holdout_group_name(sample, "dataset_source")
        by_label_source[label].setdefault(source_key, []).append(sample)
    label_orders = {
        label: _source_round_robin_samples(source_buckets)
        for label, source_buckets in by_label_source.items()
    }
    ordered_labels = ("gpt-image2", "other-generated", "real")
    selected: list[ExternalTrainingSample] = []
    selected_ids: set[str] = set()
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for label in ordered_labels:
            bucket = label_orders.get(label, [])
            if cursor >= len(bucket):
                continue
            sample = bucket[cursor]
            if sample.id in selected_ids:
                continue
            selected.append(sample)
            selected_ids.add(sample.id)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
        cursor += 1
    return selected


def _balanced_mainstream_five_samples(
    samples: list[ExternalTrainingSample],
    limit: int,
) -> list[ExternalTrainingSample]:
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    mapped_by_sample = {
        sample.id: _mainstream_five_generator_label(_normalize_generator_label(sample.label))
        for sample in samples
    }
    target_labels = (
        "gpt-image2",
        "nano-banana",
        "seedream-4",
        "stable-diffusion",
        "midjourney",
        "real",
        "unknown",
    )
    by_label_source: dict[str, dict[str, list[ExternalTrainingSample]]] = {
        label: {} for label in target_labels
    }
    for sample in samples:
        label = mapped_by_sample.get(sample.id, "unknown")
        if label not in by_label_source:
            label = "unknown"
        source_key = _source_holdout_group_name(sample, "dataset_source")
        by_label_source[label].setdefault(source_key, []).append(sample)
    per_label_orders = {
        label: _source_round_robin_samples(by_source)
        for label, by_source in by_label_source.items()
    }
    selected: list[ExternalTrainingSample] = []
    selected_ids: set[str] = set()
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for label in target_labels:
            bucket = per_label_orders.get(label, [])
            if cursor >= len(bucket):
                continue
            sample = bucket[cursor]
            if sample.id in selected_ids:
                continue
            selected.append(sample)
            selected_ids.add(sample.id)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
        cursor += 1
    return selected


def _balanced_generator_samples(
    samples: list[ExternalTrainingSample],
    limit: int,
) -> list[ExternalTrainingSample]:
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    by_label: dict[str, list[ExternalTrainingSample]] = {}
    for sample in samples:
        by_label.setdefault(_normalize_generator_label(sample.label), []).append(sample)
    by_label_source: dict[str, dict[str, list[ExternalTrainingSample]]] = {}
    for label, label_samples in by_label.items():
        for sample in label_samples:
            by_label_source.setdefault(label, {}).setdefault(
                _source_holdout_group_name(sample, "dataset_source"),
                [],
            ).append(sample)
    ordered_labels = sorted(
        by_label,
        key=lambda label: (0 if label in {"gpt-image2", "real"} else 1, label),
    )
    label_orders = {
        label: _source_round_robin_samples(by_source)
        for label, by_source in by_label_source.items()
    }
    selected: list[ExternalTrainingSample] = []
    cursor = 0
    while len(selected) < limit and ordered_labels:
        progressed = False
        for label in ordered_labels:
            bucket = label_orders.get(label, by_label[label])
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _source_round_robin_samples(
    by_source: dict[str, list[ExternalTrainingSample]],
    source_offset: int = 0,
) -> list[ExternalTrainingSample]:
    ordered_sources = sorted(by_source, key=lambda source: (-len(by_source[source]), source))
    if ordered_sources:
        offset = source_offset % len(ordered_sources)
        ordered_sources = [*ordered_sources[offset:], *ordered_sources[:offset]]
    ordered: list[ExternalTrainingSample] = []
    cursor = 0
    while ordered_sources:
        progressed = False
        for source in ordered_sources:
            bucket = by_source[source]
            if cursor < len(bucket):
                ordered.append(bucket[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return ordered


def _source_balanced_index_order(
    samples: list[ExternalTrainingSample],
    labels: list[str],
    indices: list[int],
) -> list[int]:
    by_label_source: dict[tuple[str, str], list[int]] = {}
    for index in indices:
        key = (labels[index], _source_holdout_group_name(samples[index], "dataset_source"))
        by_label_source.setdefault(key, []).append(index)
    ordered_keys = sorted(
        by_label_source,
        key=lambda item: (
            0 if item[0] in {"gpt-image2", "real"} else 1,
            item[0],
            item[1],
        ),
    )
    ordered: list[int] = []
    cursor = 0
    while ordered_keys:
        progressed = False
        for key in ordered_keys:
            bucket = by_label_source[key]
            if cursor < len(bucket):
                ordered.append(bucket[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return ordered


def _sample_matches_task(sample: ExternalTrainingSample, task_type: str) -> bool:
    if sample.task_type == task_type:
        signature = " ".join(
            [
                sample.dataset_name,
                sample.source,
                sample.source_url or "",
                sample.scenario,
                sample.title,
                sample.label,
            ]
        ).lower()
        if task_type == "vision_tamper" and _is_generator_only_source(signature):
            return False
        own_keywords = TASK_LABEL_KEYWORDS.get(task_type, ())
        other_keywords = tuple(
            keyword
            for other_task, keywords in TASK_LABEL_KEYWORDS.items()
            if other_task != task_type
            for keyword in keywords
        )
        own_hit = any(keyword.lower() in signature for keyword in own_keywords)
        other_hit = any(keyword.lower() in signature for keyword in other_keywords)
        if own_hit:
            return True
        if other_hit and _known_auxiliary_dataset(sample):
            return False
        return True
    return False


def _is_generator_only_source(signature: str) -> bool:
    generator_markers = (
        "tiny-genimage",
        "genimage",
        "gpt-image",
        "gpt_image",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "sdxl",
        "flux",
        "dall-e",
        "dalle",
        "seedream",
        "nano banana",
        "qwen-image",
        "ai_generated",
    )
    tamper_markers = (
        "tamper",
        "tampered",
        "manipulated",
        "manipulation",
        "splicing",
        "splice",
        "forgery",
        "authentic_unmodified",
        "copy-move",
        "inpaint",
    )
    return any(marker in signature for marker in generator_markers) and not any(
        marker in signature for marker in tamper_markers
    )


def _known_auxiliary_dataset(sample: ExternalTrainingSample) -> bool:
    signature = f"{sample.dataset_name} {sample.source} {sample.source_url or ''}".lower()
    known_names = (
        "tiny-genimage",
        "genimage",
        "fakeddit",
        "newsclippings",
        "cosmos",
        "casia",
        "columbia",
        "splicing",
        "deepfake",
        "gpt-image2",
        "midjourney",
        "stable-diffusion",
        "sdxl",
        "flux",
        "dall-e",
    )
    return any(name in signature for name in known_names)


def _normalize_generator_label(label: str) -> str:
    value = re.sub(r"[_\s]+", " ", label.strip().lower())
    compact = value.replace(" ", "-")
    if any(token in value for token in ("gpt-image-2", "gpt image 2", "gpt-image2", "gpt image2", "gptimage2")):
        return "gpt-image2"
    if any(token in value for token in ("gpt-image-1.5", "gpt image 1.5", "gpt-image1.5", "gpt image1.5", "gptimage1.5")):
        return "gpt-image1.5"
    if any(token in value for token in ("gpt-image-1", "gpt image 1", "gpt-image1", "gpt image1", "gptimage1")):
        return "gpt-image1"
    if any(token in value for token in ("gpt-image", "gpt image", "openai image")):
        return "gpt-image2"
    if "midjourney" in value or value in {"mj", "mid journey"}:
        return "midjourney"
    if "flux" in value:
        return "flux"
    if any(token in value for token in ("sd2.1", "sd 2.1", "sd21", "stable diffusion 2.1", "stable-diffusion-2.1")):
        return "sd21"
    if any(token in value for token in ("sd3", "sd 3", "stable diffusion 3", "stable-diffusion-3")):
        return "sd3"
    if any(token in value for token in ("stable diffusion xl", "stable-diffusion-xl", "sdxl")) or compact in {"sd-xl"}:
        return "sdxl"
    if any(token in value for token in ("stable diffusion", "stable-diffusion")):
        return "stable-diffusion"
    if any(token in value for token in ("dall-e 3", "dall-e-3", "dalle3", "dall e 3")):
        return "dall-e-3"
    if any(token in value for token in ("dall-e", "dalle", "dall·e", "dall e")):
        return "dall-e"
    if "nano" in value and "banana" in value:
        return "nano-banana"
    if "seedream" in value:
        return "seedream-4"
    if "imagegbt" in value or "image gbt" in value:
        return "imagegbt"
    if any(token in value for token in ("real", "authentic", "photo", "camera", "真实", "照片", "实拍")):
        return "real"
    if value in {"unknown", "other", "others", "其它", "其他", "未知"}:
        return "unknown"
    for known in GENERATOR_ATTRIBUTION_LABELS:
        if known != "unknown" and known in value:
            return known
    return "unknown"


def _split_class_indices(labels: list[str]) -> tuple[list[int], list[int]]:
    if len(labels) <= 2:
        return [0], [1]
    target_valid = _target_validation_count(len(labels))
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    targets = _label_validation_targets(Counter(labels), target_valid)
    valid_indices: list[int] = []
    for label in sorted(targets, key=lambda item: (0 if item in {"gpt-image2", "real"} else 1, item)):
        valid_indices.extend(by_label.get(label, [])[-targets[label]:])
    cursor = len(labels) - 1
    while len(valid_indices) < target_valid and cursor >= 0:
        if cursor not in valid_indices:
            valid_indices.append(cursor)
        cursor -= 1
    valid_set = set(valid_indices)
    train_indices = [index for index in range(len(labels)) if index not in valid_set]
    if not train_indices and valid_indices:
        moved = valid_indices.pop(0)
        train_indices = [moved]
    return train_indices, sorted(valid_indices)


def _split_generator_validation_indices(
    samples: list[ExternalTrainingSample],
    labels: list[str],
    request: VisionTrainingRunRequest,
) -> tuple[list[int], list[int]]:
    if request.validation_strategy != "source_holdout":
        return _split_class_indices(labels)
    source_split = _split_source_holdout_indices(
        samples,
        labels,
        holdout_fraction=request.source_holdout_fraction,
        min_holdout_samples=request.min_source_holdout_samples,
    )
    if source_split is not None:
        return source_split
    return _split_class_indices(labels)


def _split_source_holdout_indices(
    samples: list[ExternalTrainingSample],
    labels: list[str],
    *,
    holdout_fraction: float,
    min_holdout_samples: int,
) -> tuple[list[int], list[int]] | None:
    if len(samples) <= 2:
        return None
    target_valid = max(
        min_holdout_samples,
        min(_target_validation_count(len(samples)), int(round(len(samples) * holdout_fraction))),
    )
    target_valid = min(target_valid, max(1, len(samples) - 2))
    per_source_split = _split_source_stratified_indices(samples, labels, target_valid)
    if per_source_split is not None:
        return per_source_split
    by_group: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        group = _source_holdout_group_name(sample, "dataset_source")
        by_group.setdefault(group, []).append(index)
    if len(by_group) < 2:
        return None

    global_counts = Counter(label for label in labels if label != "unknown")
    required_labels = {label for label, count in global_counts.items() if count >= 2}
    valid_indices: list[int] = []
    valid_set: set[int] = set()
    grouped = sorted(
        by_group.items(),
        key=lambda item: (
            _source_holdout_group_score(item[0], item[1], labels),
            len(item[1]),
            item[0],
        ),
    )
    for _, group_indices in grouped:
        candidate = list(dict.fromkeys([*valid_indices, *group_indices]))
        if len(candidate) > max(target_valid * 2, min_holdout_samples):
            continue
        train_candidate = [index for index in range(len(samples)) if index not in set(candidate)]
        if _source_holdout_train_is_valid(train_candidate, labels, required_labels):
            valid_indices = candidate
            valid_set = set(candidate)
        if len(valid_indices) >= target_valid:
            break
    if len(valid_indices) < min_holdout_samples:
        return None
    train_indices = [index for index in range(len(samples)) if index not in valid_set]
    if not _source_holdout_train_is_valid(train_indices, labels, required_labels):
        return None
    return train_indices, sorted(valid_indices)


def _split_source_stratified_indices(
    samples: list[ExternalTrainingSample],
    labels: list[str],
    target_valid: int,
) -> tuple[list[int], list[int]] | None:
    by_group_label: dict[tuple[str, str], list[int]] = {}
    for index, sample in enumerate(samples):
        group = _source_holdout_group_name(sample, "dataset_source")
        by_group_label.setdefault((group, labels[index]), []).append(index)
    if len({group for group, _ in by_group_label}) < 2:
        return None
    label_counts = Counter(label for label in labels if label != "unknown")
    valid_targets = _label_validation_targets(label_counts, target_valid)
    valid_indices: list[int] = []
    for label in sorted(valid_targets, key=lambda item: (0 if item in {"gpt-image2", "real"} else 1, item)):
        label_buckets = {
            group: indices
            for (group, bucket_label), indices in by_group_label.items()
            if bucket_label == label and len(indices) >= 2
        }
        ordered = _round_robin_index_buckets(label_buckets)
        valid_indices.extend(ordered[:valid_targets[label]])
    if len(valid_indices) < min(8, target_valid):
        return None
    valid_set = set(valid_indices[:target_valid])
    train_indices = [index for index in range(len(samples)) if index not in valid_set]
    required_labels = {label for label, count in Counter(labels).items() if label != "unknown" and count >= 2}
    if not _source_holdout_train_is_valid(train_indices, labels, required_labels):
        return None
    return train_indices, sorted(valid_set)


def _label_validation_targets(label_counts: Counter[str], target_valid: int) -> dict[str, int]:
    labels = [label for label, count in label_counts.items() if count >= 2]
    if not labels:
        return {}
    total = sum(label_counts[label] for label in labels)
    raw_targets = {
        label: max(1, min(label_counts[label] - 1, int(round(target_valid * label_counts[label] / total))))
        for label in labels
    }
    while sum(raw_targets.values()) > target_valid:
        label = max(raw_targets, key=lambda item: (raw_targets[item], label_counts[item], item))
        if raw_targets[label] <= 1:
            break
        raw_targets[label] -= 1
    while sum(raw_targets.values()) < target_valid:
        candidates = [
            label
            for label in labels
            if raw_targets[label] < label_counts[label] - 1
        ]
        if not candidates:
            break
        label = max(candidates, key=lambda item: (label_counts[item] - raw_targets[item], label_counts[item], item))
        raw_targets[label] += 1
    return dict(raw_targets)


def _round_robin_index_buckets(by_group: dict[str, list[int]]) -> list[int]:
    ordered_groups = sorted(by_group, key=lambda group: (-len(by_group[group]), group))
    ordered: list[int] = []
    cursor = 0
    while ordered_groups:
        progressed = False
        for group in ordered_groups:
            bucket = by_group[group]
            if cursor < len(bucket):
                ordered.append(bucket[-(cursor + 1)])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return ordered


def _source_holdout_group_score(group_name: str, indices: list[int], labels: list[str]) -> tuple[int, int, int]:
    label_count = len({labels[index] for index in indices if labels[index] != "unknown"})
    preferred = 0 if any(token in group_name.lower() for token in ("qwen", "bananamark", "scam-ai")) else 1
    return preferred, -label_count, -len(indices)


def _source_holdout_train_is_valid(
    train_indices: list[int],
    labels: list[str],
    required_labels: set[str],
) -> bool:
    if len(train_indices) < 2:
        return False
    train_counts = Counter(labels[index] for index in train_indices if labels[index] != "unknown")
    if len(train_counts) < 2:
        return False
    missing = {label for label in required_labels if train_counts.get(label, 0) <= 0}
    return not missing


def _classification_split_report(
    samples: list[ExternalTrainingSample],
    labels: list[str],
    train_indices: list[int],
    valid_indices: list[int],
) -> dict[str, object]:
    validation_sources = _source_counts([samples[index] for index in valid_indices])
    train_sources = _source_counts([samples[index] for index in train_indices])
    validation_source_keys = _source_count_keys(validation_sources)
    train_source_keys = _source_count_keys(train_sources)
    held_out_source_names = sorted(validation_source_keys - train_source_keys)
    source_overlap_count = len(validation_source_keys & train_source_keys)
    method = (
        "source_holdout"
        if held_out_source_names
        else (
            "source_stratified_holdout"
            if source_overlap_count > 1
            else "deterministic_class_stratified_holdout"
        )
    )
    return {
        "method": method,
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "target_validation_count": _target_validation_count(len(labels)),
        "train_label_distribution": dict(Counter(labels[index] for index in train_indices)),
        "validation_label_distribution": dict(Counter(labels[index] for index in valid_indices)),
        "train_sources": train_sources,
        "validation_sources": validation_sources,
        "held_out_sources": held_out_source_names,
        "source_overlap_count": source_overlap_count,
        "classes": sorted(set(labels)),
    }


def _split_indices(labels: list[int]) -> tuple[list[int], list[int]]:
    if len(labels) <= 2:
        return [0], [1]
    target_valid = _target_validation_count(len(labels))
    by_level = _indices_by_risk_level(labels)
    valid_indices: list[int] = []
    for indices in by_level.values():
        if len(indices) >= 2:
            valid_indices.append(indices[-1])
    sorted_indices = sorted(range(len(labels)), key=lambda index: (labels[index], index))
    cursor = 1
    while len(valid_indices) < target_valid and cursor <= len(sorted_indices):
        candidate = sorted_indices[-cursor]
        if candidate not in valid_indices:
            valid_indices.append(candidate)
        cursor += max(1, len(sorted_indices) // max(target_valid, 1))
        if cursor > len(sorted_indices) and len(valid_indices) < target_valid:
            for candidate in reversed(sorted_indices):
                if candidate not in valid_indices:
                    valid_indices.append(candidate)
                    break
    valid_set = set(valid_indices)
    train_indices = [index for index in range(len(labels)) if index not in valid_set]
    if not train_indices:
        train_indices = [index for index in range(len(labels)) if index != valid_indices[0]]
    return train_indices, sorted(valid_indices)


def _target_validation_count(sample_count: int) -> int:
    if sample_count <= 2:
        return 1
    if sample_count < 20:
        return max(1, sample_count // 4)
    return max(8, min(sample_count // 5, 120))


def _indices_by_risk_level(labels: list[int]) -> dict[str, list[int]]:
    by_level = {level: [] for level in RISK_LEVELS}
    for index, label in enumerate(labels):
        by_level[risk_level_from_score(label).value].append(index)
    return by_level


def _split_report(
    samples: list[ExternalTrainingSample],
    labels: list[int],
    train_indices: list[int],
    valid_indices: list[int],
) -> dict[str, object]:
    return {
        "method": "deterministic_stratified_holdout",
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "target_validation_count": _target_validation_count(len(labels)),
        "train_label_distribution": _label_distribution([labels[index] for index in train_indices]),
        "validation_label_distribution": _label_distribution([labels[index] for index in valid_indices]),
        "validation_sources": _source_counts([samples[index] for index in valid_indices]),
        "risk_bands": list(RISK_LEVELS),
    }


def _fit_standardizer(
    rows: list[dict[str, float]],
    feature_names: list[str],
    train_indices: list[int],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = [rows[index].get(name, 0.0) for index in train_indices]
        avg = mean(values)
        variance = mean([(value - avg) ** 2 for value in values])
        means[name] = avg
        scales[name] = math.sqrt(variance) or 1.0
    return means, scales


def _train_ridge_regressor(
    *,
    rows: list[dict[str, float]],
    labels: list[int],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[dict[str, float], float]:
    weights = {name: 0.0 for name in feature_names}
    bias = mean(labels[index] for index in train_indices)
    train_count = max(1, len(train_indices))
    effective_lr = min(learning_rate, 0.08)
    for _ in range(epochs):
        grad_weights = {name: 0.0 for name in feature_names}
        grad_bias = 0.0
        for index in train_indices:
            normalized = _normalize(rows[index], means, scales)
            predicted = bias + sum(weights[name] * normalized.get(name, 0.0) for name in feature_names)
            error = predicted - labels[index]
            grad_bias += error
            for name in feature_names:
                grad_weights[name] += error * normalized.get(name, 0.0)
        bias -= effective_lr * grad_bias / train_count
        for name in feature_names:
            weights[name] -= effective_lr * (grad_weights[name] / train_count + l2 * weights[name])
    return weights, bias


def _normalize(
    features: dict[str, float],
    means: dict[str, float],
    scales: dict[str, float],
) -> dict[str, float]:
    return {
        name: (features.get(name, 0.0) - means.get(name, 0.0)) / (scales.get(name, 1.0) or 1.0)
        for name in means
    }


def _predict_from_parts(
    features: dict[str, float],
    weights: dict[str, float],
    bias: float,
    means: dict[str, float],
    scales: dict[str, float],
) -> float:
    normalized = _normalize(features, means, scales)
    return bias + sum(weights[name] * normalized.get(name, 0.0) for name in weights)


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "-", model_name.strip().lower()).strip("-")
    return slug or "model"


def _fit_advanced_classifier(
    *,
    train_matrix: list[list[float]],
    train_labels: list[str],
    sample_weights: list[float] | None = None,
    random_state: int = 42,
) -> tuple[object | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "enabled": False,
        "model": "AdvancedBoostedTreeClassifier",
        "sample_count": len(train_labels),
        "feature_count": len(train_matrix[0]) if train_matrix else 0,
        "class_distribution": dict(sorted(Counter(train_labels).items())),
        "fallback_chain": ["XGBoost", "CatBoost"],
    }
    if not train_matrix or len(set(train_labels)) < 2:
        return None, {**metadata, "reason": "not enough labeled classes"}
    model, details = _try_fit_xgboost_classifier(
        train_matrix=train_matrix,
        train_labels=train_labels,
        sample_weights=sample_weights,
        random_state=random_state,
    )
    if model is not None:
        return model, {**metadata, **details, "enabled": True}
    first_failure = details.get("reason")
    model, details = _try_fit_catboost_classifier(
        train_matrix=train_matrix,
        train_labels=train_labels,
        sample_weights=sample_weights,
        random_state=random_state,
    )
    if model is not None:
        return model, {
            **metadata,
            **details,
            "enabled": True,
            "fallback_reason": first_failure,
        }
    second_failure = details.get("reason")
    return None, {
        **metadata,
        **details,
        "reason": f"XGBoost failed: {first_failure}; CatBoost failed: {second_failure}",
    }


def _try_fit_xgboost_classifier(
    *,
    train_matrix: list[list[float]],
    train_labels: list[str],
    sample_weights: list[float] | None,
    random_state: int,
) -> tuple[object | None, dict[str, object]]:
    try:
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except Exception as exc:
        return None, {"reason": f"xgboost import failed: {type(exc).__name__}"}
    try:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(train_labels)
        model = XGBClassifier(
            n_estimators=max(80, ADVANCED_TREE_ESTIMATORS),
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=max(1, ADVANCED_TREE_MIN_SAMPLES // 4),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        fit_kwargs = {"sample_weight": sample_weights} if sample_weights else {}
        model.fit(train_matrix, encoded, **fit_kwargs)
        return _EncodedClassifier(model, list(encoder.classes_)), {
            "model": "XGBoostClassifier",
            "model_family": "boosted_tree",
            "n_estimators": max(80, ADVANCED_TREE_ESTIMATORS),
            "min_samples": ADVANCED_TREE_MIN_SAMPLES,
            "class_encoding": "sklearn.LabelEncoder wrapper",
        }
    except Exception as exc:
        return None, {"reason": f"xgboost fit failed: {type(exc).__name__}"}


def _try_fit_catboost_classifier(
    *,
    train_matrix: list[list[float]],
    train_labels: list[str],
    sample_weights: list[float] | None,
    random_state: int,
) -> tuple[object | None, dict[str, object]]:
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        return None, {"reason": f"catboost import failed: {type(exc).__name__}"}
    try:
        model = CatBoostClassifier(
            iterations=max(80, ADVANCED_TREE_ESTIMATORS),
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        fit_kwargs = {"sample_weight": sample_weights} if sample_weights else {}
        model.fit(train_matrix, train_labels, **fit_kwargs)
        return model, {
            "model": "CatBoostClassifier",
            "model_family": "boosted_tree",
            "iterations": max(80, ADVANCED_TREE_ESTIMATORS),
            "min_samples": ADVANCED_TREE_MIN_SAMPLES,
        }
    except Exception as exc:
        return None, {"reason": f"catboost fit failed: {type(exc).__name__}"}


def _fit_advanced_regressor(
    *,
    train_matrix: list[list[float]],
    train_labels: list[int],
    random_state: int = 42,
) -> tuple[object | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "enabled": False,
        "model": "AdvancedBoostedTreeRegressor",
        "sample_count": len(train_labels),
        "feature_count": len(train_matrix[0]) if train_matrix else 0,
        "fallback_chain": ["XGBoost", "CatBoost"],
    }
    if len(train_matrix) < 4:
        return None, {**metadata, "reason": "not enough samples"}
    model, details = _try_fit_xgboost_regressor(train_matrix, train_labels, random_state)
    if model is not None:
        return model, {**metadata, **details, "enabled": True}
    first_failure = details.get("reason")
    model, details = _try_fit_catboost_regressor(train_matrix, train_labels, random_state)
    if model is not None:
        return model, {**metadata, **details, "enabled": True, "fallback_reason": first_failure}
    second_failure = details.get("reason")
    return None, {
        **metadata,
        **details,
        "reason": f"XGBoost failed: {first_failure}; CatBoost failed: {second_failure}",
    }


def _try_fit_xgboost_regressor(
    train_matrix: list[list[float]],
    train_labels: list[int],
    random_state: int,
) -> tuple[object | None, dict[str, object]]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        return None, {"reason": f"xgboost import failed: {type(exc).__name__}"}
    try:
        model = XGBRegressor(
            n_estimators=max(80, ADVANCED_TREE_ESTIMATORS),
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=max(1, ADVANCED_TREE_MIN_SAMPLES // 4),
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_matrix, train_labels)
        return model, {
            "model": "XGBoostRegressor",
            "model_family": "boosted_tree",
            "n_estimators": max(80, ADVANCED_TREE_ESTIMATORS),
            "min_samples": ADVANCED_TREE_MIN_SAMPLES,
        }
    except Exception as exc:
        return None, {"reason": f"xgboost fit failed: {type(exc).__name__}"}


def _try_fit_catboost_regressor(
    train_matrix: list[list[float]],
    train_labels: list[int],
    random_state: int,
) -> tuple[object | None, dict[str, object]]:
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:
        return None, {"reason": f"catboost import failed: {type(exc).__name__}"}
    try:
        model = CatBoostRegressor(
            iterations=max(80, ADVANCED_TREE_ESTIMATORS),
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(train_matrix, train_labels)
        return model, {
            "model": "CatBoostRegressor",
            "model_family": "boosted_tree",
            "iterations": max(80, ADVANCED_TREE_ESTIMATORS),
            "min_samples": ADVANCED_TREE_MIN_SAMPLES,
        }
    except Exception as exc:
        return None, {"reason": f"catboost fit failed: {type(exc).__name__}"}


class _EncodedClassifier:
    def __init__(self, model: object, classes: list[object]) -> None:
        self.model = model
        self.classes_ = [str(item) for item in classes]

    def predict_proba(self, matrix: list[list[float]]) -> object:
        return self.model.predict_proba(matrix)


def _build_knn_prototypes(
    *,
    rows: list[dict[str, float]],
    labels: list[int],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[dict[str, object]]:
    if not train_indices:
        return []
    step = max(1, math.ceil(len(train_indices) / MAX_PROTOTYPES))
    selected_indices = train_indices[::step]
    return [
        {
            "label": labels[index],
            "features": [round(_normalize(rows[index], means, scales).get(name, 0.0), 6) for name in feature_names],
        }
        for index in selected_indices
    ]


def _build_class_prototypes(
    *,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    source_keys: list[str] | None = None,
) -> list[dict[str, object]]:
    by_label_source: dict[str, dict[str, list[list[float]]]] = {}
    for position, index in enumerate(train_indices):
        normalized = _normalize(rows[index], means, scales)
        vector = [float(normalized.get(name, 0.0)) for name in feature_names]
        source_key = (
            source_keys[position]
            if source_keys is not None and position < len(source_keys) and source_keys[position]
            else "__all_sources__"
        )
        by_label_source.setdefault(labels[index], {}).setdefault(source_key, []).append(vector)
    prototypes: list[dict[str, object]] = []
    for label, source_vectors in sorted(by_label_source.items()):
        if not source_vectors:
            continue
        source_prototypes: list[list[float]] = []
        source_counts: dict[str, int] = {}
        for source_key, vectors in sorted(source_vectors.items()):
            if not vectors:
                continue
            source_counts[source_key] = len(vectors)
            source_prototypes.append(
                [
                    mean(vector[dim] for vector in vectors)
                    for dim in range(len(feature_names))
                ]
            )
        if not source_prototypes:
            continue
        prototype = [
            round(mean(vector[dim] for vector in source_prototypes), 6)
            for dim in range(len(feature_names))
        ]
        prototypes.append(
            {
                "label": label,
                "features": prototype,
                "sample_count": sum(source_counts.values()),
                "source_count": len(source_counts),
                "source_balanced": source_keys is not None,
            }
        )
    return prototypes


def _train_generator_classifier_artifact(
    *,
    run_id: str,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    source_keys: list[str] | None = None,
) -> tuple[str | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "enabled": False,
        "model": "advanced-tree-classifier",
        "reason": "sklearn unavailable or training failed",
    }
    if len(train_indices) < min(ADVANCED_TREE_MIN_SAMPLES, 8) or len(set(labels[index] for index in train_indices)) < 2:
        metadata["reason"] = "not enough labeled classes"
        return None, metadata
    train_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in train_indices
    ]
    train_labels = [labels[index] for index in train_indices]
    sample_weights = _source_balanced_sample_weights(
        train_labels,
        train_indices,
        source_keys,
        real_weight_multiplier=GENERATOR_REAL_CLASS_WEIGHT,
        hard_negative_multiplier=GENERATOR_REAL_HARD_NEGATIVE_WEIGHT,
    )
    model, model_metadata = _fit_advanced_classifier(
        train_matrix=train_matrix,
        train_labels=train_labels,
        sample_weights=sample_weights,
        random_state=42,
    )
    if model is None:
        metadata.update(model_metadata)
        return None, metadata
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", GENERATOR_ATTRIBUTION_TASK)
    model_slug = _model_slug(str(model_metadata.get("model") or "advanced-tree-classifier"))
    path = MODEL_ARTIFACT_DIR / f"{safe_task}-{run_id}-{model_slug}-classifier.pkl"
    try:
        with path.open("wb") as file:
            pickle.dump(model, file)
    except OSError as exc:
        metadata["reason"] = f"persist failed: {type(exc).__name__}"
        return None, metadata
    return str(path), {
        "enabled": True,
        **model_metadata,
        "sample_weight_policy": "balanced within each generator label across dataset_source groups; boosts real photo class and real hard-negative sources",
        "real_weight_multiplier": GENERATOR_REAL_CLASS_WEIGHT,
        "real_hard_negative_multiplier": GENERATOR_REAL_HARD_NEGATIVE_WEIGHT,
        "feature_count": len(feature_names),
        "artifact_path": str(path),
        "note": "主分类器只使用 XGBoost/CatBoost boosted-tree；两者均不可用时不训练新分类器，只使用类别原型兼容预测。只使用图像统计/压缩/纹理特征，不使用文本来源字段。",
    }


def _train_generator_binary_gate_artifact(
    *,
    run_id: str,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    source_keys: list[str] | None = None,
    experiment_profile: str = "standard_attribution",
) -> tuple[str | None, dict[str, object]]:
    gate_policy = _generator_binary_gate_policy(experiment_profile)
    metadata: dict[str, object] = {
        "enabled": False,
        "model": "AdvancedBoostedTreeClassifier",
        "target": "generated_vs_real",
        "reason": "boosted-tree dependencies unavailable, insufficient real/generated samples, or training failed",
        "generated_threshold": GENERATOR_BINARY_GATE_THRESHOLD,
        "real_protection_margin": gate_policy["real_protection_margin"],
    }
    binary_labels = [_binary_generation_label(labels[index]) for index in train_indices]
    counts = Counter(binary_labels)
    if counts.get("real", 0) < 3 or counts.get("generated", 0) < 3:
        metadata["reason"] = "not enough real/generated samples for binary gate"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    train_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in train_indices
    ]
    real_weight_multiplier = (
        1.65
        if experiment_profile in {"binary_generated_gate", "social_propagation_robustness"}
        else GENERATOR_REAL_CLASS_WEIGHT
    )
    hard_negative_multiplier = (
        1.85
        if experiment_profile in {"binary_generated_gate", "social_propagation_robustness"}
        else GENERATOR_REAL_HARD_NEGATIVE_WEIGHT
    )
    generated_hard_positive_multiplier = 1.0
    sample_weights = _source_balanced_sample_weights(
        binary_labels,
        train_indices,
        source_keys,
        real_weight_multiplier=real_weight_multiplier,
        hard_negative_multiplier=hard_negative_multiplier,
        generated_hard_positive_multiplier=generated_hard_positive_multiplier,
    )
    model, model_metadata = _fit_advanced_classifier(
        train_matrix=train_matrix,
        train_labels=binary_labels,
        sample_weights=sample_weights,
        random_state=84,
    )
    if model is None:
        metadata.update(model_metadata)
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    try:
        probabilities = model.predict_proba(train_matrix)
        classes = [str(item) for item in getattr(model, "classes_", [])]
    except Exception as exc:
        metadata["reason"] = f"fit failed: {type(exc).__name__}"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    generated_index = classes.index("generated") if "generated" in classes else -1
    if generated_index < 0:
        metadata["reason"] = "trained gate has no generated class"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    generated_probs = [float(row[generated_index]) for row in probabilities]
    real_probs = [
        probability
        for probability, label in zip(generated_probs, binary_labels, strict=False)
        if label == "real"
    ]
    generated_class_probs = [
        probability
        for probability, label in zip(generated_probs, binary_labels, strict=False)
        if label == "generated"
    ]
    oof_real_probs, oof_generated_probs, oof_diagnostics = _generator_binary_gate_oof_probabilities(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
        source_keys=source_keys,
        experiment_profile=experiment_profile,
    )
    calibration_real_probs = oof_real_probs or real_probs
    calibration_generated_probs = oof_generated_probs or generated_class_probs
    threshold, threshold_diagnostics = _generator_binary_gate_threshold(
        calibration_real_probs,
        calibration_generated_probs,
        strategy=(
            "source_oof_real_fpr_first_threshold_search"
            if oof_real_probs and oof_generated_probs
            else "training_real_fpr_first_threshold_search"
        ),
        target_real_fpr=gate_policy["target_real_fpr"],
        min_generated_recall=gate_policy["min_generated_recall"],
        max_threshold=gate_policy["max_threshold"],
        real_guard_quantile=gate_policy["real_guard_quantile"],
        real_protection_margin=gate_policy["real_protection_margin"],
    )
    threshold_diagnostics["training_real_false_positive_rate"] = _binary_probability_fpr(real_probs, threshold)
    threshold_diagnostics["training_generated_recall"] = _binary_probability_recall(generated_class_probs, threshold)
    threshold_diagnostics["training_real_count"] = len(real_probs)
    threshold_diagnostics["training_generated_count"] = len(generated_class_probs)
    threshold_diagnostics["oof_calibration"] = oof_diagnostics
    threshold_diagnostics["profile_gate_policy"] = gate_policy
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", GENERATOR_ATTRIBUTION_TASK)
    model_slug = _model_slug(str(model_metadata.get("model") or "boosted-tree-classifier"))
    path = MODEL_ARTIFACT_DIR / f"{safe_task}-{run_id}-{model_slug}-binary-gate.pkl"
    try:
        with path.open("wb") as file:
            pickle.dump(model, file)
    except OSError as exc:
        metadata["reason"] = f"persist failed: {type(exc).__name__}"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    return str(path), {
        "enabled": True,
        **model_metadata,
        "target": "generated_vs_real",
        "artifact_path": str(path),
        "class_distribution": dict(sorted(counts.items())),
        "generated_threshold": threshold,
        "threshold_calibration": threshold_diagnostics,
        "real_protection_margin": gate_policy["real_protection_margin"],
        "gate_policy": gate_policy,
        "sample_weight_policy": "balanced within generated/real labels across dataset_source groups; boosts real hard-negatives; weak-source generated hard-positive multiplier is retained for ablation and disabled by default",
        "real_weight_multiplier": real_weight_multiplier,
        "hard_negative_multiplier": hard_negative_multiplier,
        "generated_hard_positive_multiplier": generated_hard_positive_multiplier,
        "note": "二阶段 generated-vs-real gate 先保护真实图；跨来源校准低于 generated 阈值或 real 优势明显时输出 real，再做生成器来源归因。",
    }


def _train_gpt_image2_detector_artifact(
    *,
    run_id: str,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    source_keys: list[str] | None = None,
    experiment_profile: str = "standard_attribution",
) -> tuple[str | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "enabled": False,
        "model": "AdvancedBoostedTreeClassifier",
        "target": "gpt-image2_vs_rest",
        "threshold": 0.58,
        "reason": "only enabled for gpt_image2_ovr profile",
    }
    if experiment_profile != "gpt_image2_ovr":
        return None, metadata
    binary_labels = ["gpt-image2" if labels[index] == "gpt-image2" else "not-gpt-image2" for index in train_indices]
    counts = Counter(binary_labels)
    if counts.get("gpt-image2", 0) < 8 or counts.get("not-gpt-image2", 0) < 8:
        metadata["reason"] = "not enough GPT-image2/rest samples"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    train_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in train_indices
    ]
    sample_weights = _source_balanced_sample_weights(binary_labels, train_indices, source_keys)
    model, model_metadata = _fit_advanced_classifier(
        train_matrix=train_matrix,
        train_labels=binary_labels,
        sample_weights=sample_weights,
        random_state=126,
    )
    if model is None:
        metadata.update(model_metadata)
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    try:
        probabilities = model.predict_proba(train_matrix)
        classes = [str(item) for item in getattr(model, "classes_", [])]
    except Exception as exc:
        metadata["reason"] = f"fit failed: {type(exc).__name__}"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    if "gpt-image2" not in classes:
        metadata["reason"] = "trained detector has no gpt-image2 class"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    positive_index = classes.index("gpt-image2")
    positive_probs = [
        float(row[positive_index])
        for row, label in zip(probabilities, binary_labels, strict=False)
        if label == "gpt-image2"
    ]
    negative_probs = [
        float(row[positive_index])
        for row, label in zip(probabilities, binary_labels, strict=False)
        if label != "gpt-image2"
    ]
    threshold, threshold_diagnostics = _gpt_image2_detector_threshold(
        positive_probs=positive_probs,
        negative_probs=negative_probs,
    )
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", GENERATOR_ATTRIBUTION_TASK)
    model_slug = _model_slug(str(model_metadata.get("model") or "boosted-tree-classifier"))
    path = MODEL_ARTIFACT_DIR / f"{safe_task}-{run_id}-{model_slug}-gpt-image2-detector.pkl"
    try:
        with path.open("wb") as file:
            pickle.dump(model, file)
    except OSError as exc:
        metadata["reason"] = f"persist failed: {type(exc).__name__}"
        metadata["class_distribution"] = dict(sorted(counts.items()))
        return None, metadata
    return str(path), {
        "enabled": True,
        **model_metadata,
        "target": "gpt-image2_vs_rest",
        "artifact_path": str(path),
        "class_distribution": dict(sorted(counts.items())),
        "threshold": threshold,
        "threshold_calibration": threshold_diagnostics,
        "sample_weight_policy": "balanced within GPT/rest labels across dataset_source groups",
        "feature_count": len(feature_names),
        "note": "GPT-image2 专项二元检测器先判断疑似 GPT-image2，再由 generated/real gate 保护真实图误报。",
    }


def _gpt_image2_detector_threshold(
    *,
    positive_probs: list[float],
    negative_probs: list[float],
) -> tuple[float, dict[str, object]]:
    candidates = [0.50 + index * 0.02 for index in range(16)]
    best_threshold = 0.58
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        recall = (
            sum(1 for value in positive_probs if value >= threshold) / len(positive_probs)
            if positive_probs
            else 0.0
        )
        false_positive_rate = (
            sum(1 for value in negative_probs if value >= threshold) / len(negative_probs)
            if negative_probs
            else 1.0
        )
        key = (recall - max(0.0, false_positive_rate - 0.18) * 1.5, recall, -false_positive_rate)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    recall = (
        sum(1 for value in positive_probs if value >= best_threshold) / len(positive_probs)
        if positive_probs
        else 0.0
    )
    false_positive_rate = (
        sum(1 for value in negative_probs if value >= best_threshold) / len(negative_probs)
        if negative_probs
        else 0.0
    )
    return round(best_threshold, 3), {
        "strategy": "gpt_recall_first_with_rest_fpr_penalty",
        "positive_count": len(positive_probs),
        "negative_count": len(negative_probs),
        "training_gpt_image2_recall": round(recall, 3),
        "training_rest_false_positive_rate": round(false_positive_rate, 3),
    }


def _predict_gpt_image2_probability(
    gpt_detector_path: str | None,
    normalized: dict[str, float],
    feature_names: list[str],
) -> dict[str, object] | None:
    model = _load_tree_model(gpt_detector_path)
    if model is None or not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba([_vector_from_normalized(normalized, feature_names)])[0]
        classes = [str(item) for item in getattr(model, "classes_", [])]
    except Exception:
        return None
    if "gpt-image2" not in classes:
        return None
    probability = float(probabilities[classes.index("gpt-image2")])
    return {
        "gpt_image2_probability": round(probability, 3),
        "not_gpt_image2_probability": round(1.0 - probability, 3),
        "classes": classes,
    }


def _generator_binary_gate_policy(experiment_profile: str) -> dict[str, float | str]:
    policies: dict[str, dict[str, float | str]] = {
        "binary_generated_gate": {
            "profile": "binary_generated_gate",
            "target_real_fpr": 0.07,
            "min_generated_recall": 0.72,
            "max_threshold": 0.78,
            "real_guard_quantile": 0.88,
            "real_protection_margin": 0.08,
        },
        "gpt_image2_ovr": {
            "profile": "gpt_image2_ovr",
            "target_real_fpr": 0.15,
            "min_generated_recall": 0.82,
            "max_threshold": 0.66,
            "real_guard_quantile": 0.70,
            "real_protection_margin": 0.04,
        },
        "social_propagation_robustness": {
            "profile": "social_propagation_robustness",
            "target_real_fpr": 0.06,
            "min_generated_recall": 0.75,
            "max_threshold": 0.80,
            "real_guard_quantile": 0.88,
            "real_protection_margin": 0.08,
        },
    }
    return policies.get(
        experiment_profile,
        {
            "profile": experiment_profile,
            "target_real_fpr": 0.08,
            "min_generated_recall": 0.75,
            "max_threshold": 0.78,
            "real_guard_quantile": 0.80,
            "real_protection_margin": GENERATOR_REAL_PROTECTION_MARGIN,
        },
    )


def _binary_gate_mode_for_profile(experiment_profile: str) -> str:
    if experiment_profile == "standard_attribution":
        return "advisory"
    return "enforce"


def _generator_binary_gate_oof_probabilities(
    *,
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    source_keys: list[str] | None,
    experiment_profile: str = "standard_attribution",
) -> tuple[list[float], list[float], dict[str, object]]:
    diagnostics: dict[str, object] = {
        "enabled": False,
        "method": "leave_dataset_source_out",
        "fold_count": 0,
        "reason": "insufficient source groups",
    }
    if source_keys is None or len(source_keys) < len(train_indices):
        diagnostics["reason"] = "source keys unavailable"
        return [], [], diagnostics
    binary_labels = [_binary_generation_label(labels[index]) for index in train_indices]
    local_source_keys = [
        source_keys[position] if position < len(source_keys) else "__unknown_source__"
        for position, _ in enumerate(train_indices)
    ]
    by_source: dict[str, list[int]] = {}
    for local_position, source_key in enumerate(local_source_keys):
        by_source.setdefault(source_key or "__unknown_source__", []).append(local_position)
    if len(by_source) < 2:
        return [], [], diagnostics

    oof_real_probs: list[float] = []
    oof_generated_probs: list[float] = []
    skipped_sources: list[str] = []
    used_sources: list[str] = []
    for source_key, local_holdout_positions in sorted(by_source.items()):
        local_train_positions = [
            position for position in range(len(train_indices)) if position not in set(local_holdout_positions)
        ]
        local_train_labels = [binary_labels[position] for position in local_train_positions]
        if Counter(local_train_labels).get("real", 0) < 2 or Counter(local_train_labels).get("generated", 0) < 2:
            skipped_sources.append(source_key)
            continue
        train_matrix = [
            _vector_from_normalized(_normalize(rows[train_indices[position]], means, scales), feature_names)
            for position in local_train_positions
        ]
        holdout_matrix = [
            _vector_from_normalized(_normalize(rows[train_indices[position]], means, scales), feature_names)
            for position in local_holdout_positions
        ]
        fold_source_keys = [local_source_keys[position] for position in local_train_positions]
        real_weight_multiplier = 1.65 if experiment_profile in {"binary_generated_gate", "social_propagation_robustness"} else 1.0
        hard_negative_multiplier = 1.85 if experiment_profile in {"binary_generated_gate", "social_propagation_robustness"} else 1.0
        generated_hard_positive_multiplier = 1.0
        fold_weights = _source_balanced_sample_weights(
            local_train_labels,
            list(range(len(local_train_labels))),
            fold_source_keys,
            real_weight_multiplier=real_weight_multiplier,
            hard_negative_multiplier=hard_negative_multiplier,
            generated_hard_positive_multiplier=generated_hard_positive_multiplier,
        )
        model, _ = _fit_advanced_classifier(
            train_matrix=train_matrix,
            train_labels=local_train_labels,
            sample_weights=fold_weights,
            random_state=184,
        )
        if model is None:
            skipped_sources.append(source_key)
            continue
        try:
            probabilities = model.predict_proba(holdout_matrix)
            classes = [str(item) for item in getattr(model, "classes_", [])]
        except Exception:
            skipped_sources.append(source_key)
            continue
        if "generated" not in classes:
            skipped_sources.append(source_key)
            continue
        generated_index = classes.index("generated")
        used_sources.append(source_key)
        for probability_row, local_position in zip(probabilities, local_holdout_positions, strict=False):
            probability = float(probability_row[generated_index])
            if binary_labels[local_position] == "real":
                oof_real_probs.append(probability)
            else:
                oof_generated_probs.append(probability)
    diagnostics.update(
        {
            "enabled": bool(oof_real_probs and oof_generated_probs),
            "fold_count": len(used_sources),
            "used_sources": used_sources,
            "skipped_sources": skipped_sources,
            "real_count": len(oof_real_probs),
            "generated_count": len(oof_generated_probs),
            "reason": None if oof_real_probs and oof_generated_probs else "insufficient out-of-fold class coverage",
        }
    )
    return oof_real_probs, oof_generated_probs, diagnostics


def _source_balanced_sample_weights(
    labels: list[str],
    train_indices: list[int],
    source_keys: list[str] | None,
    *,
    real_weight_multiplier: float = 1.0,
    hard_negative_multiplier: float = 1.0,
    generated_hard_positive_multiplier: float = 1.0,
) -> list[float] | None:
    if source_keys is None or len(source_keys) < len(train_indices):
        return None
    resolved_source_keys = _resolve_training_source_keys(train_indices, source_keys, len(labels))
    group_counts = Counter(
        (labels[position], resolved_source_keys[position] or "__unknown_source__")
        for position, _ in enumerate(train_indices)
    )
    label_source_counts = Counter(label for label, _ in group_counts)
    weights: list[float] = []
    for position, _ in enumerate(train_indices):
        label = labels[position]
        source_key = resolved_source_keys[position] or "__unknown_source__"
        group_count = group_counts[(label, source_key)]
        source_count = max(1, label_source_counts[label])
        weight = 1.0 / max(1, group_count * source_count)
        if label == "real":
            weight *= real_weight_multiplier
            if _is_real_hard_negative_source(source_key):
                weight *= hard_negative_multiplier
        elif _is_generated_hard_positive_source(source_key):
            weight *= generated_hard_positive_multiplier
        weights.append(round(weight, 6))
    if not weights:
        return None
    avg = mean(weights)
    if avg <= 0:
        return None
    return [round(weight / avg, 6) for weight in weights]


def _resolve_training_source_keys(
    train_indices: list[int],
    source_keys: list[str],
    label_count: int,
) -> list[str]:
    if train_indices == list(range(label_count)):
        return [source_keys[position] for position in range(label_count)]
    if train_indices and max(train_indices) < len(source_keys):
        return [source_keys[index] for index in train_indices]
    if len(source_keys) == label_count:
        return [source_keys[position] for position in range(label_count)]
    return [source_keys[position] for position in range(min(label_count, len(source_keys)))]


def _is_real_hard_negative_source(source_key: str) -> bool:
    lowered_source = source_key.lower()
    hard_negative_tokens = (
        "real-negative",
        "real_negative",
        "real negative",
        "hard-negative",
        "hard_negative",
        "aigc-detection-benchmark",
        "aigc_detection_benchmark",
        "thekernel01",
        "defactify",
        "synthbuster",
        "coco",
        "laion",
        "flickr",
        "camera",
        "authentic",
        "real-photo",
        "real_photo",
        "natural",
        "imagenet",
        "fakeddit",
        "splicing",
    )
    return any(token in lowered_source for token in hard_negative_tokens)


def _is_generated_hard_positive_source(source_key: str) -> bool:
    lowered_source = source_key.lower()
    hard_positive_tokens = (
        "defactify",
        "aigc-detection-benchmark",
        "aigc_detection_benchmark",
        "thekernel01",
        "synthbuster",
        "marco-willi",
        "qwen-image-bench",
        "qwen_image_bench",
        "bananamark",
        "banana-mark",
    )
    return any(token in lowered_source for token in hard_positive_tokens)


def _generator_binary_gate_threshold(
    real_probabilities: list[float],
    generated_probabilities: list[float],
    *,
    strategy: str = "real_fpr_first_threshold_search",
    target_real_fpr: float = 0.05,
    min_generated_recall: float | None = None,
    max_threshold: float = 0.9,
    real_guard_quantile: float = 0.9,
    real_protection_margin: float = GENERATOR_REAL_PROTECTION_MARGIN,
) -> tuple[float, dict[str, object]]:
    candidates = sorted(
        {
            GENERATOR_BINARY_GATE_THRESHOLD,
            0.5,
            0.56,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            max_threshold,
            *[round(value, 3) for value in real_probabilities],
            *[round(value + 0.001, 3) for value in real_probabilities],
            *[round(value - 0.001, 3) for value in generated_probabilities],
        }
    )
    if not candidates:
        threshold = GENERATOR_BINARY_GATE_THRESHOLD
        return threshold, {
            "strategy": "default_no_probabilities",
            "target_real_false_positive_rate": target_real_fpr,
            "training_real_false_positive_rate": None,
            "training_generated_recall": None,
        }
    resolved_min_generated_recall = (
        min_generated_recall if min_generated_recall is not None else (0.75 if len(generated_probabilities) >= 8 else 0.6)
    )
    best_threshold = GENERATOR_BINARY_GATE_THRESHOLD
    best_key: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    for candidate in candidates:
        threshold = max(0.5, min(max_threshold, candidate))
        real_false_positive_rate = (
            sum(1 for value in real_probabilities if value >= threshold) / len(real_probabilities)
            if real_probabilities
            else 0.0
        )
        generated_recall = (
            sum(1 for value in generated_probabilities if value >= threshold) / len(generated_probabilities)
            if generated_probabilities
            else 0.0
        )
        real_excess = max(0.0, real_false_positive_rate - target_real_fpr)
        generated_shortfall = max(0.0, resolved_min_generated_recall - generated_recall)
        key = (real_excess * 3.0 + generated_shortfall * 1.5, real_false_positive_rate, -generated_recall)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = {
                "training_real_false_positive_rate": round(real_false_positive_rate, 3),
                "training_generated_recall": round(generated_recall, 3),
            }
    selected_before_real_guard = best_threshold
    if real_probabilities:
        quantile_index = min(
            len(real_probabilities) - 1,
            max(0, int(len(real_probabilities) * real_guard_quantile)),
        )
        real_guard = sorted(real_probabilities)[quantile_index]
        best_threshold = max(best_threshold, real_guard + real_protection_margin)
    final_threshold = round(max(0.5, min(max_threshold, best_threshold)), 3)
    final_real_fpr = (
        sum(1 for value in real_probabilities if value >= final_threshold) / len(real_probabilities)
        if real_probabilities
        else 0.0
    )
    final_generated_recall = (
        sum(1 for value in generated_probabilities if value >= final_threshold) / len(generated_probabilities)
        if generated_probabilities
        else 0.0
    )
    diagnostics: dict[str, object] = {
        "strategy": strategy,
        "target_real_false_positive_rate": target_real_fpr,
        "minimum_generated_recall_preference": resolved_min_generated_recall,
        "max_threshold": max_threshold,
        "real_guard_quantile": real_guard_quantile,
        "real_protection_margin": real_protection_margin,
        "training_real_count": len(real_probabilities),
        "training_generated_count": len(generated_probabilities),
        "training_real_false_positive_rate": round(final_real_fpr, 3),
        "training_generated_recall": round(final_generated_recall, 3),
        "selected_before_real_guard": round(selected_before_real_guard, 3),
        "candidate_count": len(candidates),
        "search_training_real_false_positive_rate": best_metrics.get("training_real_false_positive_rate"),
        "search_training_generated_recall": best_metrics.get("training_generated_recall"),
    }
    return final_threshold, diagnostics


def _binary_probability_fpr(real_probabilities: list[float], threshold: float) -> float | None:
    if not real_probabilities:
        return None
    return round(sum(1 for value in real_probabilities if value >= threshold) / len(real_probabilities), 3)


def _binary_probability_recall(generated_probabilities: list[float], threshold: float) -> float | None:
    if not generated_probabilities:
        return None
    return round(sum(1 for value in generated_probabilities if value >= threshold) / len(generated_probabilities), 3)


def _predict_generated_probability_with_gate(
    binary_gate_path: str | None,
    normalized: dict[str, float],
    feature_names: list[str],
) -> dict[str, object] | None:
    model = _load_tree_model(binary_gate_path)
    if model is None or not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba([_vector_from_normalized(normalized, feature_names)])[0]
        classes = [str(item) for item in getattr(model, "classes_", [])]
    except Exception:
        return None
    if "generated" not in classes:
        return None
    generated_probability = float(probabilities[classes.index("generated")])
    real_probability = float(probabilities[classes.index("real")]) if "real" in classes else 1.0 - generated_probability
    return {
        "generated_probability": round(generated_probability, 3),
        "real_probability": round(real_probability, 3),
        "classes": classes,
    }


def _apply_generator_binary_gate(
    prediction: dict[str, object],
    gate: dict[str, object] | None,
    *,
    generated_gate_threshold: float,
    real_protection_margin: float,
    binary_gate_mode: str = "enforce",
) -> dict[str, object]:
    if gate is None:
        return prediction
    raw_label = str(prediction.get("raw_label") or prediction.get("label") or "unknown")
    generated_probability = float(gate.get("generated_probability", 0.0) or 0.0)
    real_probability = float(gate.get("real_probability", 0.0) or 0.0)
    attribution_confidence = float(prediction.get("confidence", 0.0) or 0.0)
    gate["review_recommendation"] = _binary_gate_review_recommendation(
        generated_probability,
        real_probability,
        generated_gate_threshold,
        real_protection_margin,
        raw_label=raw_label,
    )
    prediction["binary_gate"] = gate
    prediction["binary_gate_mode"] = binary_gate_mode
    if binary_gate_mode == "advisory":
        prediction["binary_gate_reason"] = "binary_gate_advisory_only"
        prediction.setdefault("gate_reason", "binary_gate_advisory_only")
        return prediction
    if (
        raw_label == "real"
        and generated_probability >= generated_gate_threshold + real_protection_margin
        and generated_probability >= real_probability + real_protection_margin
    ):
        prediction["label"] = "generated"
        prediction["confidence"] = round(generated_probability, 3)
        prediction["gate_reason"] = "binary_gate_generated_override"
        return prediction
    if raw_label == "real":
        return prediction
    if (
        raw_label == "gpt-image2"
        and generated_gate_threshold >= 0.6
        and attribution_confidence >= 0.70
        and generated_probability >= generated_gate_threshold - 0.12
    ):
        prediction["confidence"] = round(min(attribution_confidence, max(generated_probability, 0.5)), 3)
        prediction["gate_reason"] = "binary_gate_gpt_image2_high_confidence_override"
        return prediction
    if real_probability >= generated_probability + real_protection_margin:
        prediction["label"] = "real"
        prediction["confidence"] = round(real_probability, 3)
        prediction["gate_reason"] = "binary_gate_real_protection"
        return prediction
    if generated_probability < generated_gate_threshold:
        prediction["label"] = "real"
        prediction["confidence"] = round(real_probability, 3)
        prediction["gate_reason"] = "binary_gate_below_generated_threshold_real_guard"
        return prediction
    prediction["confidence"] = round(min(float(prediction.get("confidence", 0.0)), generated_probability), 3)
    return prediction


def _real_photo_guard_score(features: dict[str, float]) -> float:
    megapixels = float(features.get("image_megapixels", 0.0) or 0.0)
    jpg = float(features.get("jpg_ext", 0.0) or 0.0)
    luma_std = float(features.get("pixel_luma_std", 0.0) or 0.0)
    saturation_mean = float(features.get("pixel_saturation_mean", 0.0) or 0.0)
    saturation_std = float(features.get("pixel_saturation_std", 0.0) or 0.0)
    edge_density = float(features.get("edge_density", 0.0) or 0.0)
    texture_std = float(features.get("texture_residual_std", 0.0) or 0.0)
    compression_std = float(features.get("compression_residual_std", 0.0) or 0.0)
    text_overlay = float(features.get("text_overlay_edge_density", 0.0) or 0.0)
    score = 0.0
    score += 0.26 if megapixels >= 3.0 else 0.12 if megapixels >= 1.2 else 0.0
    score += 0.12 if jpg >= 0.5 else 0.0
    score += 0.12 if 0.22 <= luma_std <= 0.75 else 0.0
    score += 0.12 if 0.08 <= saturation_mean <= 0.52 else 0.0
    score += 0.10 if saturation_std >= 0.22 else 0.0
    score += 0.10 if 0.12 <= edge_density <= 0.36 else 0.0
    score += 0.08 if 0.04 <= texture_std <= 0.16 else 0.0
    score += 0.06 if 0.015 <= compression_std <= 0.07 else 0.0
    score -= 0.12 if text_overlay >= 0.55 else 0.0
    return round(max(0.0, min(score, 1.0)), 3)


def _apply_real_photo_guard(
    prediction: dict[str, object],
    features: dict[str, float],
) -> dict[str, object]:
    guard_score = _real_photo_guard_score(features)
    prediction["real_photo_guard"] = {
        "score": guard_score,
        "policy": "high-resolution natural-photo proxy; auxiliary guard only",
    }
    if guard_score < 0.72 or str(prediction.get("raw_label") or prediction.get("label")) == "gpt-image2":
        return prediction
    candidates = prediction.get("candidates")
    if not isinstance(candidates, list):
        return prediction
    updated: list[dict[str, object]] = []
    has_real = False
    for item in candidates:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "unknown")
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if label == "real":
            confidence = max(confidence, min(0.82, guard_score))
            has_real = True
        elif label == "other-generated":
            confidence = min(confidence, max(0.05, 1.0 - guard_score))
        updated.append({**item, "confidence": round(confidence, 3)})
    if not has_real:
        updated.append({"label": "real", "confidence": round(min(0.82, guard_score), 3), "distance": None})
    updated = sorted(updated, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
    total = sum(float(item.get("confidence", 0.0) or 0.0) for item in updated)
    if total > 0:
        updated = [
            {**item, "confidence": round(float(item.get("confidence", 0.0) or 0.0) / total, 3)}
            for item in updated
        ]
        updated = sorted(updated, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
    prediction["candidates"] = updated[:5]
    top = updated[0]
    if str(top.get("label")) == "real":
        prediction["label"] = "real"
        prediction["raw_label"] = str(prediction.get("raw_label") or "real")
        prediction["confidence"] = float(top.get("confidence", 0.0) or 0.0)
        prediction["gate_reason"] = "real_photo_guard_high_resolution_natural_photo"
    return prediction


def _binary_gate_review_recommendation(
    generated_probability: float,
    real_probability: float,
    generated_gate_threshold: float,
    real_protection_margin: float,
    raw_label: str = "unknown",
) -> dict[str, object]:
    strong_threshold = (
        generated_gate_threshold + real_protection_margin
        if raw_label == "real"
        else generated_gate_threshold
    )
    review_threshold = max(0.5, round(strong_threshold - 0.12, 3))
    if generated_probability >= strong_threshold and generated_probability >= real_probability:
        level = "generated_strong"
        message = "生成图强判定：可进入后续来源候选排名，但仍需结合证据链核验。"
    elif generated_probability >= review_threshold:
        level = "manual_review_generated_signal"
        message = "生成概率达到复核线但未达到强判定线：建议人工复核原图、平台转码痕迹和来源候选排名。"
    else:
        level = "low_generated_signal"
        message = "生成概率低于复核线：当前仅作为低强度视觉线索。"
    return {
        "level": level,
        "generated_probability": round(generated_probability, 3),
        "real_probability": round(real_probability, 3),
        "strong_threshold": round(strong_threshold, 3),
        "review_threshold": review_threshold,
        "real_protection_margin": round(real_protection_margin, 3),
        "message": message,
    }


def _predict_generator_with_classifier(
    classifier_path: str | None,
    normalized: dict[str, float],
    feature_names: list[str],
    unknown_threshold: float,
    open_set_min_margin: float = 0.0,
) -> dict[str, object] | None:
    model = _load_tree_model(classifier_path)
    if model is None or not hasattr(model, "predict_proba"):
        return None
    vector = [_vector_from_normalized(normalized, feature_names)]
    try:
        probabilities = model.predict_proba(vector)[0]
        classes = [str(item) for item in getattr(model, "classes_", [])]
    except Exception:
        return None
    if not classes:
        return None
    ranked = sorted(
        zip(classes, probabilities, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    best_label, best_probability = ranked[0]
    second_probability = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = float(best_probability) - second_probability
    candidates = [
        {
            "label": label,
            "confidence": round(float(probability), 3),
            "distance": None,
        }
        for label, probability in ranked[:5]
    ]
    final_label = best_label
    unknown_reasons: list[str] = []
    if float(best_probability) < unknown_threshold:
        final_label = "unknown"
        unknown_reasons.append("below_unknown_threshold")
    if best_label != "real" and margin < open_set_min_margin:
        final_label = "unknown"
        unknown_reasons.append("low_top2_margin")
    return {
        "label": final_label,
        "raw_label": best_label,
        "confidence": round(float(best_probability), 3),
        "margin": round(margin, 3),
        "unknown_reasons": unknown_reasons,
        "distance": None,
        "candidates": candidates,
        "prototype": {"label": best_label, "sample_count": 0},
        "model": type(model).__name__,
    }


def _generator_unknown_threshold(
    rows: list[dict[str, float]],
    labels: list[str],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> float:
    class_count = len({labels[index] for index in train_indices})
    sample_count = len(train_indices)
    base = 1.0 / max(class_count, 2)
    if sample_count < 20 or class_count < 3:
        return round(max(0.08, min(0.28, base * 1.4)), 3)
    if sample_count < 100:
        return round(max(0.08, min(0.22, base * 1.25)), 3)
    return round(max(0.06, min(GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR, base * 1.15)), 3)


def _open_set_unknown_threshold(base_threshold: float, request: object) -> float:
    if not bool(getattr(request, "enable_open_set_unknown", False)):
        return base_threshold
    multiplier = float(getattr(request, "unknown_threshold_multiplier", 1.0) or 1.0)
    return round(max(0.01, min(0.95, base_threshold * multiplier)), 3)


def _open_set_policy_report(request: object, unknown_threshold: float, base_threshold: float) -> dict[str, object]:
    enabled = bool(getattr(request, "enable_open_set_unknown", False))
    return {
        "enabled": enabled,
        "base_unknown_threshold": base_threshold,
        "effective_unknown_threshold": unknown_threshold,
        "unknown_threshold_multiplier": float(getattr(request, "unknown_threshold_multiplier", 1.0) or 1.0),
        "open_set_min_margin": float(getattr(request, "open_set_min_margin", 0.0) or 0.0),
        "policy": (
            "低置信或 top-2 概率间隔过小的非 real 归因输出 unknown，避免对未覆盖/跨来源样本强行归因。"
            if enabled
            else "使用默认 unknown 阈值；不额外执行 top-2 margin 拒判。"
        ),
    }


def _predict_generator_label(
    features: dict[str, float],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    prototypes: list[dict[str, object]],
    unknown_threshold: float,
    classifier_path: str | None = None,
    gpt_detector_path: str | None = None,
    binary_gate_path: str | None = None,
    generated_gate_threshold: float = GENERATOR_BINARY_GATE_THRESHOLD,
    gpt_detector_threshold: float = GENERATOR_ATTRIBUTION_CONFIDENCE_FLOOR,
    real_protection_margin: float = GENERATOR_REAL_PROTECTION_MARGIN,
    open_set_min_margin: float = 0.0,
    binary_gate_mode: str = "enforce",
) -> dict[str, object]:
    normalized = _normalize(features, means, scales)
    binary_gate = _predict_generated_probability_with_gate(binary_gate_path, normalized, feature_names)
    classifier_prediction = _predict_generator_with_classifier(
        classifier_path,
        normalized,
        feature_names,
        unknown_threshold,
        open_set_min_margin,
    )
    gpt_detector = _predict_gpt_image2_probability(
        gpt_detector_path,
        normalized,
        feature_names,
    )
    if (
        gpt_detector is not None
        and float(gpt_detector.get("gpt_image2_probability", 0.0) or 0.0) >= gpt_detector_threshold
        and (
            binary_gate is None
            or float(binary_gate.get("real_probability", 0.0) or 0.0)
            < float(binary_gate.get("generated_probability", 0.0) or 0.0) + real_protection_margin
        )
    ):
        probability = float(gpt_detector["gpt_image2_probability"])
        prediction = {
            "label": "gpt-image2",
            "raw_label": "gpt-image2",
            "confidence": round(probability, 3),
            "distance": None,
            "candidates": [
                {"label": "gpt-image2", "confidence": round(probability, 3), "distance": None},
                {"label": "not-gpt-image2", "confidence": round(1.0 - probability, 3), "distance": None},
            ],
            "prototype": {"label": "gpt-image2", "sample_count": 0},
            "model": "gpt-image2-vs-rest-detector",
            "gpt_image2_detector": gpt_detector,
        }
        return _apply_generator_binary_gate(
            prediction,
            binary_gate,
            generated_gate_threshold=generated_gate_threshold,
            real_protection_margin=real_protection_margin,
            binary_gate_mode=binary_gate_mode,
        )
    if classifier_prediction is not None:
        if gpt_detector is not None:
            classifier_prediction["gpt_image2_detector"] = gpt_detector
        classifier_prediction = _apply_real_photo_guard(classifier_prediction, features)
        return _apply_generator_binary_gate(
            classifier_prediction,
            binary_gate,
            generated_gate_threshold=generated_gate_threshold,
            real_protection_margin=real_protection_margin,
            binary_gate_mode=binary_gate_mode,
        )
    distances: list[tuple[str, float, dict[str, object]]] = []
    for prototype in prototypes:
        label = str(prototype.get("label", "unknown"))
        values = prototype.get("features")
        if not isinstance(values, list):
            continue
        distance = 0.0
        dims = min(len(feature_names), len(values))
        if dims == 0:
            continue
        for index, name in enumerate(feature_names[:dims]):
            value = float(values[index]) if isinstance(values[index], int | float) else 0.0
            delta = normalized.get(name, 0.0) - value
            distance += delta * delta
        distances.append((label, math.sqrt(distance / max(dims, 1)), prototype))
    if not distances:
        return _apply_generator_binary_gate(
            {
                "label": "unknown",
                "confidence": 0.0,
                "distance": None,
                "candidates": [],
                "prototype": {},
            },
            binary_gate,
            generated_gate_threshold=generated_gate_threshold,
            real_protection_margin=real_protection_margin,
        )
    scores = [(label, math.exp(-distance), distance, prototype) for label, distance, prototype in distances]
    total = sum(score for _, score, _, _ in scores) or 1.0
    candidates = sorted(
        [
            {
                "label": label,
                "confidence": round(score / total, 3),
                "distance": round(distance, 4),
            }
            for label, score, distance, _ in scores
        ],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    best_label, best_score, best_distance, best_prototype = max(scores, key=lambda item: item[1])
    confidence = best_score / total
    second_confidence = float(candidates[1]["confidence"]) if len(candidates) > 1 else 0.0
    margin = confidence - second_confidence
    final_label = best_label
    unknown_reasons: list[str] = []
    if best_label != "unknown" and confidence < unknown_threshold:
        final_label = "unknown"
        unknown_reasons.append("below_unknown_threshold")
    if best_label not in {"real", "unknown"} and margin < open_set_min_margin:
        final_label = "unknown"
        unknown_reasons.append("low_top2_margin")
    prototype_prediction = _apply_real_photo_guard(
        {
            "label": final_label,
            "raw_label": best_label,
            "confidence": round(confidence, 3),
            "margin": round(margin, 3),
            "unknown_reasons": unknown_reasons,
            "distance": round(best_distance, 4),
            "candidates": candidates[:5],
            "prototype": best_prototype,
        },
        features,
    )
    return _apply_generator_binary_gate(
        prototype_prediction,
        binary_gate,
        generated_gate_threshold=generated_gate_threshold,
        real_protection_margin=real_protection_margin,
        binary_gate_mode=binary_gate_mode,
    )


def _build_clip_prototypes(
    samples: list[ExternalTrainingSample],
    train_indices: list[int],
) -> list[dict[str, object]]:
    if not CLIP_ENABLED:
        return []
    prototypes: list[dict[str, object]] = []
    for index in train_indices:
        sample = samples[index]
        text = f"{sample.title} {sample.content} {sample.scenario}"
        pair = _clip_embedding_pair(sample.image_path, sample.image_sha256, text)
        if pair is None:
            continue
        image_values, text_values = pair
        vector = _clip_match_vector(image_values, text_values)
        prototypes.append({"label": sample.risk_score, "features": vector})
    return prototypes


def _clip_match_vector(image_values: list[float], text_values: list[float]) -> list[float]:
    dims = min(CLIP_PROTO_DIMS, len(image_values), len(text_values))
    vector: list[float] = []
    for index in range(dims):
        image_value = float(image_values[index])
        text_value = float(text_values[index])
        vector.extend([image_value, text_value, abs(image_value - text_value)])
    return vector


def _clip_vector_for_sample(sample: ExternalTrainingSample) -> list[float] | None:
    text = f"{sample.title} {sample.content} {sample.scenario}"
    pair = _clip_embedding_pair(sample.image_path, sample.image_sha256, text)
    if pair is None:
        return None
    return _clip_match_vector(pair[0], pair[1])


def _select_ensemble(
    *,
    samples: list[ExternalTrainingSample],
    rows: list[dict[str, float]],
    labels: list[int],
    train_indices: list[int],
    valid_indices: list[int],
    feature_names: list[str],
    weights: dict[str, float],
    bias: float,
    means: dict[str, float],
    scales: dict[str, float],
    prototypes: list[dict[str, object]],
    tree_artifact_path: str | None,
    tree_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    train_labels = [labels[index] for index in train_indices]
    valid_labels = [labels[index] for index in valid_indices]
    ridge_train = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in train_indices
    ]
    ridge_valid = [
        _clip_score(_predict_from_parts(rows[index], weights, bias, means, scales))
        for index in valid_indices
    ]
    candidates: list[dict[str, object]] = [
        {
            "selected_model": "ridge",
            "alpha": 1.0,
            "knn_k": 0,
            "train_predictions": ridge_train,
            "valid_predictions": ridge_valid,
            "validation_mae": _mae(ridge_valid, valid_labels),
        }
    ]
    if prototypes:
        normalized_train = [_normalize(rows[index], means, scales) for index in train_indices]
        normalized_valid = [_normalize(rows[index], means, scales) for index in valid_indices]
        for k_value in KNN_CANDIDATE_K:
            effective_k = min(k_value, len(prototypes))
            if effective_k <= 0:
                continue
            knn_train = [
                _predict_knn_from_normalized(row, feature_names, prototypes, effective_k) or 0.0
                for row in normalized_train
            ]
            knn_valid = [
                _predict_knn_from_normalized(row, feature_names, prototypes, effective_k) or 0.0
                for row in normalized_valid
            ]
            candidates.append(
                {
                    "selected_model": "knn",
                    "alpha": 0.0,
                    "knn_k": effective_k,
                    "train_predictions": knn_train,
                    "valid_predictions": knn_valid,
                    "validation_mae": _mae(knn_valid, valid_labels),
                }
            )
            for alpha in ENSEMBLE_ALPHA_CANDIDATES:
                if alpha in {0.0, 1.0}:
                    continue
                ensemble_train = [
                    _clip_score(alpha * ridge + (1.0 - alpha) * knn)
                    for ridge, knn in zip(ridge_train, knn_train)
                ]
                ensemble_valid = [
                    _clip_score(alpha * ridge + (1.0 - alpha) * knn)
                    for ridge, knn in zip(ridge_valid, knn_valid)
                ]
                candidates.append(
                    {
                        "selected_model": "ensemble",
                        "alpha": alpha,
                        "knn_k": effective_k,
                        "train_predictions": ensemble_train,
                        "valid_predictions": ensemble_valid,
                        "validation_mae": _mae(ensemble_valid, valid_labels),
                    }
                )
    tree_train, tree_valid = _tree_predictions(
        tree_artifact_path=tree_artifact_path,
        rows=rows,
        train_indices=train_indices,
        valid_indices=valid_indices,
        feature_names=feature_names,
        means=means,
        scales=scales,
    )
    if tree_train is not None and tree_valid is not None:
        candidates.append(
            {
                "selected_model": "tree",
                "alpha": 1.0,
                "knn_k": 0,
                "train_predictions": tree_train,
                "valid_predictions": tree_valid,
                "validation_mae": _mae(tree_valid, valid_labels),
                "tree_metadata": tree_metadata or {},
            }
        )
        if prototypes:
            for k_value in KNN_CANDIDATE_K:
                effective_k = min(k_value, len(prototypes))
                if effective_k <= 0:
                    continue
                normalized_train = [_normalize(rows[index], means, scales) for index in train_indices]
                normalized_valid = [_normalize(rows[index], means, scales) for index in valid_indices]
                knn_train = [
                    _predict_knn_from_normalized(row, feature_names, prototypes, effective_k) or 0.0
                    for row in normalized_train
                ]
                knn_valid = [
                    _predict_knn_from_normalized(row, feature_names, prototypes, effective_k) or 0.0
                    for row in normalized_valid
                ]
                for alpha in ENSEMBLE_ALPHA_CANDIDATES:
                    if alpha in {0.0, 1.0}:
                        continue
                    ensemble_train = [
                        _clip_score(alpha * tree + (1.0 - alpha) * knn)
                        for tree, knn in zip(tree_train, knn_train)
                    ]
                    ensemble_valid = [
                        _clip_score(alpha * tree + (1.0 - alpha) * knn)
                        for tree, knn in zip(tree_valid, knn_valid)
                    ]
                    candidates.append(
                        {
                            "selected_model": "tree_ensemble",
                            "alpha": alpha,
                            "knn_k": effective_k,
                            "train_predictions": ensemble_train,
                            "valid_predictions": ensemble_valid,
                            "validation_mae": _mae(ensemble_valid, valid_labels),
                            "tree_metadata": tree_metadata or {},
                        }
                    )
    clip_prototypes = _build_clip_prototypes(samples, train_indices)
    if clip_prototypes:
        clip_train_vectors = [_clip_vector_for_sample(samples[index]) for index in train_indices]
        clip_valid_vectors = [_clip_vector_for_sample(samples[index]) for index in valid_indices]
        for k_value in KNN_CANDIDATE_K:
            effective_k = min(k_value, len(clip_prototypes))
            if effective_k <= 0:
                continue
            clip_train = [
                _predict_vector_knn(vector, clip_prototypes, effective_k) if vector is not None else 0.0
                for vector in clip_train_vectors
            ]
            clip_valid = [
                _predict_vector_knn(vector, clip_prototypes, effective_k) if vector is not None else 0.0
                for vector in clip_valid_vectors
            ]
            candidates.append(
                {
                    "selected_model": "clip_knn",
                    "alpha": 0.0,
                    "knn_k": effective_k,
                    "train_predictions": clip_train,
                    "valid_predictions": clip_valid,
                    "validation_mae": _mae(clip_valid, valid_labels),
                }
            )
    best = min(
        candidates,
        key=lambda item: (
            float(item["validation_mae"]),
            0 if item["selected_model"] == "ridge" else 1,
            float(item["alpha"]),
        ),
    )
    selection_report = {
        "selected_model": best["selected_model"],
        "alpha": best["alpha"],
        "knn_k": best["knn_k"],
        "prototype_count": len(prototypes),
        "clip_prototype_count": len(clip_prototypes),
        "tree_available": tree_train is not None and tree_valid is not None,
        "tree_model": tree_metadata or {},
        "ridge_validation_mae": candidates[0]["validation_mae"],
        "best_validation_mae": best["validation_mae"],
        "candidates": [
            {
                "model": candidate["selected_model"],
                "alpha": candidate["alpha"],
                "knn_k": candidate["knn_k"],
                "validation_mae": candidate["validation_mae"],
            }
            for candidate in candidates
        ],
    }
    return {
        "selected_model": str(best["selected_model"]),
        "alpha": float(best["alpha"]),
        "knn_k": int(best["knn_k"]),
        "train_predictions": list(best["train_predictions"]),
        "valid_predictions": list(best["valid_predictions"]),
        "tree_metadata": best.get("tree_metadata", tree_metadata or {}),
        "selection_report": selection_report,
    }


def _predict_knn_from_normalized(
    normalized: dict[str, float],
    feature_names: list[str],
    prototypes: list[dict[str, object]],
    k_value: int,
) -> float | None:
    if not prototypes or k_value <= 0:
        return None
    distances: list[tuple[float, float]] = []
    for prototype in prototypes:
        values = prototype.get("features")
        label = prototype.get("label")
        if not isinstance(values, list) or not isinstance(label, int | float):
            continue
        distance = 0.0
        for index, name in enumerate(feature_names):
            value = float(values[index]) if index < len(values) and isinstance(values[index], int | float) else 0.0
            delta = normalized.get(name, 0.0) - value
            distance += delta * delta
        distances.append((distance, float(label)))
    if not distances:
        return None
    nearest = sorted(distances, key=lambda item: item[0])[:max(1, min(k_value, len(distances)))]
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, label in nearest:
        weight = 1.0 / (math.sqrt(distance) + 1e-6)
        weighted_sum += weight * label
        weight_total += weight
    return _clip_score(weighted_sum / weight_total) if weight_total else None


def _predict_vector_knn(
    vector: list[float] | None,
    prototypes: list[dict[str, object]],
    k_value: int,
) -> float | None:
    if vector is None or not prototypes or k_value <= 0:
        return None
    distances: list[tuple[float, float]] = []
    for prototype in prototypes:
        values = prototype.get("features")
        label = prototype.get("label")
        if not isinstance(values, list) or not isinstance(label, int | float):
            continue
        dims = min(len(vector), len(values))
        if dims == 0:
            continue
        distance = 0.0
        for index in range(dims):
            raw_value = values[index]
            other = float(raw_value) if isinstance(raw_value, int | float) else 0.0
            delta = vector[index] - other
            distance += delta * delta
        distances.append((distance, float(label)))
    if not distances:
        return None
    nearest = sorted(distances, key=lambda item: item[0])[:max(1, min(k_value, len(distances)))]
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, label in nearest:
        weight = 1.0 / (math.sqrt(distance) + 1e-6)
        weighted_sum += weight * label
        weight_total += weight
    return _clip_score(weighted_sum / weight_total) if weight_total else None


def _predict_clip_from_artifact(
    artifact: dict[str, object],
    features: dict[str, float],
    k_value: int,
) -> float | None:
    prototypes = _artifact_prototypes(artifact.get("clip_prototypes"))
    if not prototypes:
        return None
    vector: list[float] = []
    for index in range(CLIP_PROTO_DIMS):
        image_value = features.get(f"clip_img_{index:02d}")
        text_value = features.get(f"clip_txt_{index:02d}")
        gap_value = features.get(f"clip_gap_{index:02d}")
        if not isinstance(image_value, int | float) or not isinstance(text_value, int | float):
            break
        vector.extend([
            float(image_value),
            float(text_value),
            float(gap_value) if isinstance(gap_value, int | float) else abs(float(image_value) - float(text_value)),
        ])
    return _predict_vector_knn(vector, prototypes, k_value) if vector else None


def _train_tree_artifact(
    *,
    run_id: str,
    task_type: str,
    rows: list[dict[str, float]],
    labels: list[int],
    train_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> tuple[str | None, dict[str, object]]:
    if len(train_indices) < 8:
        return None, {"enabled": False, "model": "AdvancedBoostedTreeRegressor", "reason": "not enough samples"}
    train_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in train_indices
    ]
    train_labels = [labels[index] for index in train_indices]
    model, metadata = _fit_advanced_regressor(
        train_matrix=train_matrix,
        train_labels=train_labels,
        random_state=42,
    )
    if model is None:
        return None, metadata
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", task_type)
    model_slug = _model_slug(str(metadata.get("model") or "boosted-tree-regressor"))
    path = MODEL_ARTIFACT_DIR / f"{safe_task}-{run_id}-{model_slug}.pkl"
    try:
        with path.open("wb") as file:
            pickle.dump(model, file)
    except OSError as exc:
        return None, {**metadata, "enabled": False, "reason": f"persist failed: {type(exc).__name__}"}
    return str(path), {**metadata, "enabled": True, "artifact_path": str(path)}


def _finalize_tree_artifact(path_text: str | None, run_id: str, task_type: str) -> str | None:
    if not path_text:
        return None
    source = Path(path_text)
    if not source.exists():
        return None
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", task_type)
    slug_match = re.search(r"-(xgboostregressor|catboostregressor|boosted-tree-regressor)\.pkl$", source.name)
    slug = slug_match.group(1) if slug_match else "boosted-tree-regressor"
    target = MODEL_ARTIFACT_DIR / f"{safe_task}-{run_id}-{slug}.pkl"
    if source == target:
        return str(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    except OSError:
        return str(source)
    return str(target)


def _tree_predictions(
    *,
    tree_artifact_path: str | None,
    rows: list[dict[str, float]],
    train_indices: list[int],
    valid_indices: list[int],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> tuple[list[float], list[float]] | tuple[None, None]:
    model = _load_tree_model(tree_artifact_path)
    if model is None:
        return None, None
    train_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in train_indices
    ]
    valid_matrix = [
        _vector_from_normalized(_normalize(rows[index], means, scales), feature_names)
        for index in valid_indices
    ]
    try:
        train_predictions = [_clip_score(float(value)) for value in model.predict(train_matrix)]
        valid_predictions = [_clip_score(float(value)) for value in model.predict(valid_matrix)]
    except Exception:
        return None, None
    return train_predictions, valid_predictions


def _predict_tree_from_artifact(
    artifact: dict[str, object],
    normalized: dict[str, float],
    feature_names: list[str],
) -> float | None:
    ensemble = artifact.get("ensemble")
    if not isinstance(ensemble, dict):
        return None
    model = _load_tree_model(ensemble.get("tree_artifact_path"))
    if model is None:
        return None
    try:
        prediction = model.predict([_vector_from_normalized(normalized, feature_names)])
    except Exception:
        return None
    return _clip_score(float(prediction[0]))


def _load_tree_model(path_value: object) -> object | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except Exception:
        return None


def _vector_from_normalized(normalized: dict[str, float], feature_names: list[str]) -> list[float]:
    return [float(normalized.get(name, 0.0)) for name in feature_names]


def _artifact_prototypes(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _feature_weights(weights: dict[str, float], direction: str) -> list[FeatureWeight]:
    reverse = direction == "positive"
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=reverse)
    ordered = [item for item in ordered if item[1] > 0] if reverse else [item for item in ordered if item[1] < 0]
    return [
        FeatureWeight(
            name=name,
            description=_feature_description(name),
            weight=round(weight, 4),
            direction="提升风险" if direction == "positive" else "降低风险",
        )
        for name, weight in ordered
    ]


def _feature_description(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name.startswith("task::"):
        return f"任务类型特征：{name.split('::', 1)[1]}"
    return name


def _contribution_payload(contributions: list[tuple[str, float, float]]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "description": _feature_description(name),
            "value": round(raw_value, 4),
            "contribution": round(contribution, 4),
        }
        for name, contribution, raw_value in contributions[:6]
    ]


def _generator_contribution_payload(
    prototype: object,
    features: dict[str, float],
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[dict[str, object]]:
    if not isinstance(prototype, dict):
        return []
    values = prototype.get("features")
    if not isinstance(values, list):
        return []
    normalized = _normalize(features, means, scales)
    contributions: list[tuple[str, float, float]] = []
    for index, name in enumerate(feature_names):
        if index >= len(values) or not isinstance(values[index], int | float):
            continue
        delta = abs(normalized.get(name, 0.0) - float(values[index]))
        contributions.append((name, -delta, features.get(name, 0.0)))
    return _contribution_payload(sorted(contributions, key=lambda item: item[1], reverse=True))


def _prototype_separation_features(
    prototypes: list[dict[str, object]],
    feature_names: list[str],
    direction: str,
) -> list[FeatureWeight]:
    if not prototypes or not feature_names:
        return []
    spreads: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        values = [
            float(prototype["features"][index])
            for prototype in prototypes
            if isinstance(prototype.get("features"), list)
            and index < len(prototype["features"])
            and isinstance(prototype["features"][index], int | float)
        ]
        if len(values) >= 2:
            spreads[name] = max(values) - min(values)
    reverse = direction == "positive"
    ordered = sorted(spreads.items(), key=lambda item: item[1], reverse=reverse)
    return [
        FeatureWeight(
            name=name,
            description=_feature_description(name),
            weight=round(value, 4),
            direction="区分类别更强" if direction == "positive" else "区分类别更弱",
        )
        for name, value in ordered
        if value > 0
    ]


def _mae(predictions: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return round(mean(abs(prediction - label) for prediction, label in zip(predictions, labels)), 2)


def _rmse(predictions: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return round(math.sqrt(mean((prediction - label) ** 2 for prediction, label in zip(predictions, labels))), 2)


def _accuracy_within(predictions: list[float], labels: list[int], tolerance: int) -> float:
    if not labels:
        return 0.0
    passed = sum(1 for prediction, label in zip(predictions, labels) if abs(prediction - label) <= tolerance)
    return round(passed / len(labels), 3)


def _risk_level_accuracy(predictions: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    matched = sum(
        1
        for prediction, label in zip(predictions, labels)
        if risk_level_from_score(prediction).value == risk_level_from_score(label).value
    )
    return round(matched / len(labels), 3)


def _label_distribution(labels: list[int]) -> dict[str, int]:
    distribution = {level: 0 for level in RISK_LEVELS}
    for label in labels:
        distribution[risk_level_from_score(label).value] += 1
    return distribution


def _confusion_matrix(predictions: list[float], labels: list[int]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in RISK_LEVELS} for actual in RISK_LEVELS}
    for prediction, label in zip(predictions, labels):
        actual = risk_level_from_score(label).value
        predicted = risk_level_from_score(prediction).value
        matrix[actual][predicted] += 1
    return matrix


def _classification_metrics(predictions: list[str], labels: list[str]) -> dict[str, object]:
    classes = sorted(set(labels) | set(predictions))
    matrix = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for prediction, label in zip(predictions, labels):
        matrix.setdefault(label, {predicted: 0 for predicted in classes})
        matrix[label][prediction] = matrix[label].get(prediction, 0) + 1
    total = len(labels)
    correct = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    f1_values: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    for class_name in classes:
        true_positive = matrix.get(class_name, {}).get(class_name, 0)
        false_positive = sum(
            matrix.get(other, {}).get(class_name, 0)
            for other in classes
            if other != class_name
        )
        false_negative = sum(
            count
            for predicted, count in matrix.get(class_name, {}).items()
            if predicted != class_name
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if sum(matrix.get(class_name, {}).values()) > 0:
            f1_values.append(f1)
        per_class[class_name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": float(sum(matrix.get(class_name, {}).values())),
        }
    return {
        "accuracy": round(correct / total, 3) if total else 0.0,
        "macro_f1": round(mean(f1_values), 3) if f1_values else 0.0,
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _binary_generation_metrics(predictions: list[str], labels: list[str]) -> dict[str, float]:
    binary_predictions = [_binary_generation_label(prediction) for prediction in predictions]
    binary_labels = [_binary_generation_label(label) for label in labels]
    metrics = _classification_metrics(binary_predictions, binary_labels)
    per_class = metrics.get("per_class") if isinstance(metrics, dict) else {}
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    generated_metrics = per_class_metrics.get("generated", {})
    real_metrics = per_class_metrics.get("real", {})
    matrix = metrics.get("confusion_matrix") if isinstance(metrics, dict) else {}
    generated_row = matrix.get("generated", {}) if isinstance(matrix, dict) else {}
    generated_total = sum(int(value) for value in generated_row.values()) if isinstance(generated_row, dict) else 0
    generated_correct = int(generated_row.get("generated", 0)) if isinstance(generated_row, dict) else 0
    generated_false_negative_count = generated_total - generated_correct
    real_row = matrix.get("real", {}) if isinstance(matrix, dict) else {}
    real_total = sum(int(value) for value in real_row.values()) if isinstance(real_row, dict) else 0
    real_correct = int(real_row.get("real", 0)) if isinstance(real_row, dict) else 0
    real_false_positive_count = real_total - real_correct
    real_false_positive_rate = real_false_positive_count / real_total if real_total else 0.0
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "generated_precision": float(generated_metrics.get("precision", 0.0)),
        "generated_recall": float(generated_metrics.get("recall", 0.0)),
        "generated_f1": float(generated_metrics.get("f1", 0.0)),
        "generated_support": float(generated_total),
        "generated_false_negative_count": float(generated_false_negative_count),
        "real_precision": float(real_metrics.get("precision", 0.0)),
        "real_recall": float(real_metrics.get("recall", 0.0)),
        "real_f1": float(real_metrics.get("f1", 0.0)),
        "real_false_positive_rate": round(real_false_positive_rate, 3),
        "real_support": float(real_total),
        "real_false_positive_count": float(real_false_positive_count),
    }


def _binary_generation_label(label: str) -> str:
    return "real" if label == "real" else "generated"


def _metrics_report(
    train_predictions: list[float],
    train_labels: list[int],
    valid_predictions: list[float],
    valid_labels: list[int],
) -> dict[str, object]:
    return {
        "train": {
            "mae": _mae(train_predictions, train_labels),
            "rmse": _rmse(train_predictions, train_labels),
            "within_10": _accuracy_within(train_predictions, train_labels, 10),
            "risk_level_accuracy": _risk_level_accuracy(train_predictions, train_labels),
        },
        "validation": {
            "mae": _mae(valid_predictions, valid_labels),
            "rmse": _rmse(valid_predictions, valid_labels),
            "within_10": _accuracy_within(valid_predictions, valid_labels, 10),
            "risk_level_accuracy": _risk_level_accuracy(valid_predictions, valid_labels),
            "confusion_matrix": _confusion_matrix(valid_predictions, valid_labels),
        },
    }


def _validation_diagnostics(
    samples: list[ExternalTrainingSample],
    labels: list[int],
    valid_predictions: list[float],
    valid_labels: list[int],
) -> dict[str, object]:
    dataset_names = sorted({sample.dataset_name for sample in samples})
    source_names = sorted({sample.source for sample in samples})
    score_values = sorted(set(labels))
    validation_mae = _mae(valid_predictions, valid_labels)
    perfect_validation = validation_mae == 0.0
    same_source_only = len(dataset_names) == 1 or len(source_names) == 1
    binary_fixed_scores = len(score_values) <= 2
    warnings: list[str] = []
    if perfect_validation and same_source_only:
        warnings.append("验证集与训练集来自同一数据源，且验证误差为 0；该结果不能直接代表跨数据集泛化能力。")
    if binary_fixed_scores:
        warnings.append("监督分数只有少量固定取值，MAE 可能被离散标签设置放大或压低。")
    if len(samples) < 500:
        warnings.append("样本规模仍偏小，建议继续加入跨来源外部数据做独立测试。")
    return {
        "dataset_count": len(dataset_names),
        "source_count": len(source_names),
        "datasets": dataset_names[:10],
        "sources": source_names[:10],
        "unique_score_values": score_values[:20],
        "perfect_validation": perfect_validation,
        "same_source_only": same_source_only,
        "binary_fixed_scores": binary_fixed_scores,
        "generalization_confidence": "limited" if warnings else "standard_holdout",
        "warnings": warnings,
    }


def _source_counts(samples: list[ExternalTrainingSample]) -> list[dict[str, object]]:
    counts: dict[str, dict[str, object]] = {}
    for sample in samples:
        key = f"{sample.dataset_name}|{sample.source}"
        if key not in counts:
            counts[key] = {
                "dataset_name": sample.dataset_name,
                "source": sample.source,
                "count": 0,
            }
        counts[key]["count"] = int(counts[key]["count"]) + 1
    return sorted(counts.values(), key=lambda item: str(item["dataset_name"]))


def _source_count_keys(source_counts: list[dict[str, object]]) -> set[str]:
    keys: set[str] = set()
    for item in source_counts:
        dataset_name = str(item.get("dataset_name", ""))
        source = str(item.get("source", ""))
        keys.add(f"{dataset_name}|{source}")
    return keys


def _sample_source_summary(samples: list[ExternalTrainingSample]) -> dict[str, object]:
    by_source: dict[str, dict[str, object]] = {}
    for sample in samples:
        key = f"{sample.dataset_name}|{sample.source}|{sample.task_type}"
        if key not in by_source:
            by_source[key] = {
                "dataset_name": sample.dataset_name,
                "source": sample.source,
                "source_url": sample.source_url,
                "task_type": sample.task_type,
                "sample_count": 0,
                "image_available_count": 0,
                "label_distribution": {},
            }
        item = by_source[key]
        item["sample_count"] = int(item["sample_count"]) + 1
        if sample.image_available:
            item["image_available_count"] = int(item["image_available_count"]) + 1
        labels = item["label_distribution"]
        if isinstance(labels, dict):
            labels[sample.label] = int(labels.get(sample.label, 0)) + 1
    return {
        "external_samples": len(samples),
        "image_available_samples": sum(1 for sample in samples if sample.image_available),
        "datasets": sorted(by_source.values(), key=lambda item: str(item["dataset_name"])),
        "excluded_demo_cases": [case.id for case in DEMO_CASES],
    }


def _generator_dataset_caveats(samples: list[ExternalTrainingSample]) -> list[str]:
    dataset_names = {sample.dataset_name for sample in samples}
    caveats: list[str] = []
    if "yufan/image_style_transfer_GPTImage2" in dataset_names:
        caveats.append(
            "yufan/image_style_transfer_GPTImage2 提供的是 GPTImage2 风格迁移图，"
            "可增加 GPT-image2 监督样本量，但和普通写实生成图存在域偏差。"
        )
    else:
        caveats.append(
            "已排除 yufan/image_style_transfer_GPTImage2 风格迁移辅助集，"
            "避免把风格迁移纹理误学成普通 GPT-image2 来源特征。"
        )
    if "Qwen/Qwen-Image-Bench" in dataset_names:
        caveats.append(
            "Qwen/Qwen-Image-Bench 是生成器评测集，单生成器目录样本量有限，"
            "适合作为可追溯 benchmark 样本，不代表真实平台全分布。"
        )
    if "Scam-AI/gpt-image-2" in dataset_names:
        caveats.append(
            "Scam-AI/gpt-image-2 提供 Twitter/X 真实传播环境中的 GPT-image2 样本，"
            "显著补强 GPT-image2 正样本；受限于平台压缩、采集窗口和非商业研究许可，仍需跨源盲测。"
        )
    caveats.append(
        "生成模型归因仍需结合 C2PA、水印、平台元数据、原始发布链路和人工核验，"
        "不得单凭本地分类头定性。"
    )
    return caveats


def _keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    return min(float(sum(1 for keyword in keywords if keyword.lower() in text)), 5.0)


def _clip_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _list_value(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _case_text(case: CaseSample) -> str:
    return " ".join(
        [
            case.title,
            case.scenario,
            case.platform,
            case.source_url,
            case.content,
            case.image_description,
            case.manual_label,
            " ".join(case.tags),
            case.sensitivity_notes,
            case.spread.velocity,
        ]
    )
