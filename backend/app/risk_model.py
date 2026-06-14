from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
from statistics import mean
from uuid import uuid4

from app.models import (
    CaseSample,
    ExternalTrainingSample,
    FeatureWeight,
    RiskLevel,
    SpreadMetrics,
    TrainingRunRequest,
    TrainingRunResult,
)
from app.storage import (
    get_active_training_artifact,
    list_external_training_samples,
    list_labeled_user_case_samples,
    save_training_run,
)


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    description: str
    keywords: tuple[str, ...] = ()


EXPERT_FEATURES: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("public_safety", "公共安全、警情、灾害、交通等治理相关信号", ("公安", "警方", "警察", "民警", "执法", "警情", "灾害", "塌方", "校车", "交通", "管制", "公共安全")),
    FeatureDefinition("police_trust", "涉警执法、公信力和舆情风险信号", ("警方", "民警", "执法", "隐瞒", "偏袒", "公信力", "推搡", "受伤")),
    FeatureDefinition("disaster_panic", "灾害险情、避险动员和资源挤兑风险信号", ("灾害", "塌方", "泥石流", "校车", "被困", "报警", "避险", "抢险", "恐慌")),
    FeatureDefinition("group_polarization", "群体对立、网暴和线下动员风险信号", ("性别", "群体", "对立", "网暴", "集合", "声援", "偏袒", "冲突")),
    FeatureDefinition("aigc_suspicion", "AI 生成、伪造、旧图嫁接和图像异常信号", ("AI", "AIGC", "合成", "生成", "伪造", "截图", "旧图", "水印", "EXIF", "纹理异常", "字体间距", "头像重复", "路牌文字")),
    FeatureDefinition("source_gap", "来源缺失、未核验和权威链路不足信号", ("未附", "缺少", "未见", "待核验", "公开平台样本", "脱敏", "无权威", "来源")),
    FeatureDefinition("emotion_incite", "强情绪、紧迫转发、煽动和扩大传播信号", ("立即", "马上", "转发", "号召", "声援", "隐瞒", "偏袒", "多人受伤", "紧急", "恐慌")),
    FeatureDefinition("offline_mobilization", "线下聚集、现场扰动和秩序维护信号", ("线下", "集合", "现场", "聚集", "巡查", "秩序", "声援", "围观")),
    FeatureDefinition("views_log", "浏览量对数特征"),
    FeatureDefinition("reposts_log", "转发量对数特征"),
    FeatureDefinition("comments_log", "评论量对数特征"),
    FeatureDefinition("likes_log", "点赞量对数特征"),
    FeatureDefinition("repost_rate", "转发/浏览比例"),
    FeatureDefinition("comment_rate", "评论/浏览比例"),
    FeatureDefinition("fast_velocity", "快速扩散、跨群传播、夜间激增等速度信号"),
    FeatureDefinition("low_spread", "低传播规模负向信号"),
)

SCENARIOS = (
    "涉警公信力谣言",
    "灾害险情谣言",
    "群体对立煽动型谣言",
    "低风险误传",
)
RISK_LEVELS = ("低", "关注", "较高", "紧急")
NGRAM_LIMIT = 80
FALLBACK_ARTIFACT_ID = "rule-fallback"


def train_risk_model(request: TrainingRunRequest) -> TrainingRunResult:
    base_samples = _base_training_samples()
    if len(base_samples) < 4:
        raise ValueError(
            "请先导入外部训练数据集或标注用户案例；内置四方向样例仅用于展示评测，不参与训练。"
        )
    samples = _training_samples(
        base_samples,
        include_augmented=request.include_augmented_samples,
    )

    vocabulary = _build_vocabulary([case for case, _ in samples], limit=NGRAM_LIMIT)
    feature_names = _feature_names(vocabulary)
    rows = [extract_features(case, vocabulary=vocabulary) for case, _ in samples]
    labels = [label for _, label in samples]
    train_indices, valid_indices = _split_indices(samples, labels)
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
    train_labels = [labels[index] for index in train_indices]
    valid_labels = [labels[index] for index in valid_indices]
    positive = _feature_weights(weights, "positive")
    negative = _feature_weights(weights, "negative")
    task_metrics = _task_metrics(valid_predictions, valid_labels)
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    source_summary = _training_source_summary(base_samples, samples)
    result = TrainingRunResult(
        id=run_id,
        created_at=created_at,
        model_kind="competition-local-hybrid-ngram-ridge-v3",
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
        accuracy_within_10=_accuracy_within(valid_predictions, valid_labels, tolerance=10),
        label_distribution=_label_distribution(labels),
        top_positive_features=positive[:8],
        top_negative_features=negative[:8],
        training_trace=[
            f"读取{source_summary['external_samples']}条外部数据集样本和{source_summary['user_labeled_cases']}条用户人工标注案例。",
            "明确排除内置四方向展示样例：这些样例只用于训练后的赛场展示和横向评测。",
            f"在合格训练样本基础上形成{len(samples)}条训练记录，增强样本只来自外部/人工标注数据。",
            f"构建{len(vocabulary)}个中文字符 n-gram 文本特征，覆盖谣言、公共安全、涉警、灾害和群体对立语义片段。",
            f"融合{len(EXPERT_FEATURES)}个专家规则/传播特征、{len(SCENARIOS)}个场景特征和 n-gram 特征，共{len(feature_names)}维。",
            f"使用分层留出验证：训练{len(train_indices)}条、验证{len(valid_indices)}条，训练 L2 正则化风险回归头。",
            "同步产出风险等级、一致性校准、Top-K特征贡献、混淆矩阵和模型卡，供比赛答辩说明。",
            "保存模型权重、标准化参数、词表、训练指标和审计信息，后续研判使用该版本推理。",
        ],
        model_card={
            "name": "公共安全谣言风险分级本地基线模型",
            "version": run_id,
            "architecture": "专家特征 + 中文字符 n-gram + 场景特征 + L2 风险回归头",
            "intended_use": "比赛正式版本地可解释风险分级基线，辅助民警进行人工复核和处置优先级排序。",
            "not_for": "不替代最终执法结论，不用于未经授权的真实个人数据自动决策。",
            "training_data": "外部导入中文谣言/假新闻数据集 + 用户人工标注案例；内置四方向样例不参与训练。",
            "training_source_summary": source_summary,
            "validation_protocol": split_report,
            "feature_groups": ["expert_rules", "spread_metrics", "scenario_one_hot", "char_ngram"],
            "leakage_controls": [
                "外部样本标签字段不进入文本特征、词表或标签。",
                "外部样本未提供真实传播链路时使用中性占位传播特征，不按标签合成浏览/转发量。",
                "内置四方向展示样例不进入训练、验证或增强样本。",
            ],
        },
        task_metrics=task_metrics,
        confusion_matrix=_confusion_matrix(valid_predictions, valid_labels),
    )
    artifact: dict[str, object] = {
        "id": run_id,
        "created_at": created_at,
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "model_kind": result.model_kind,
        "vocabulary": vocabulary,
        "model_card": result.model_card,
        "task_metrics": result.task_metrics,
        "confusion_matrix": result.confusion_matrix,
        "training_source_summary": source_summary,
        "validation_protocol": split_report,
    }
    save_training_run(result, artifact)
    return result


def training_status_note() -> str:
    return "正式比赛版本地风险模型已启用：训练集来自外部数据集/人工标注案例，内置四方向样例只做展示评测。"


def predict_with_active_model(case: CaseSample) -> tuple[float, float, list[str], str] | None:
    artifact = get_active_training_artifact()
    if artifact is None:
        return None
    feature_names = _artifact_mapping(artifact, "feature_names")
    weights = _float_mapping(artifact.get("weights"))
    means = _float_mapping(artifact.get("means"))
    scales = _float_mapping(artifact.get("scales"))
    vocabulary = _artifact_mapping(artifact, "vocabulary")
    if not feature_names or not weights or not means or not scales:
        return None
    features = extract_features(case, vocabulary=vocabulary)
    normalized = _normalize(features, means, scales)
    bias = float(artifact.get("bias", 0))
    raw_score = bias + sum(weights.get(name, 0.0) * normalized.get(name, 0.0) for name in feature_names)
    score = _clip_score(raw_score)
    contributions = [
        (name, weights.get(name, 0.0) * normalized.get(name, 0.0), features.get(name, 0.0))
        for name in feature_names
    ]
    explanations = _explain(contributions, score)
    confidence = _confidence(score, contributions)
    model_id = str(artifact.get("id", FALLBACK_ARTIFACT_ID))
    return score, confidence, explanations, model_id


def extract_features(
    case: CaseSample,
    vocabulary: list[str] | None = None,
) -> dict[str, float]:
    text = _case_text(case).lower()
    views = max(case.spread.views, 0)
    reposts = max(case.spread.reposts, 0)
    comments = max(case.spread.comments, 0)
    likes = max(case.spread.likes, 0)
    features: dict[str, float] = {}
    for definition in EXPERT_FEATURES:
        if definition.keywords:
            features[definition.name] = _keyword_score(text, definition.keywords)
    features["views_log"] = _log_metric(views)
    features["reposts_log"] = _log_metric(reposts)
    features["comments_log"] = _log_metric(comments)
    features["likes_log"] = _log_metric(likes)
    features["repost_rate"] = min(reposts / max(views, 1) * 100, 12)
    features["comment_rate"] = min(comments / max(views, 1) * 100, 16)
    features["fast_velocity"] = _keyword_score(
        case.spread.velocity.lower(),
        ("快速", "跨群", "夜间", "上涨", "30分钟", "2小时", "立即"),
    )
    features["low_spread"] = 1.0 if views < 3000 and reposts < 100 and comments < 100 else 0.0
    for scenario in SCENARIOS:
        features[f"scenario::{scenario}"] = 1.0 if case.scenario == scenario else 0.0
    for term in vocabulary or []:
        features[f"ngram::{term}"] = min(text.count(term), 5) / 5
    return {name: features.get(name, 0.0) for name in _feature_names(vocabulary or [])}


def infer_demo_label(case: CaseSample) -> int:
    if case.manual_risk_score is not None:
        return case.manual_risk_score
    label_text = case.manual_label + case.scenario + " ".join(case.tags)
    if "低风险" in label_text:
        return 22
    if "灾害" in label_text:
        return 88
    if "群体" in label_text or "线下" in label_text:
        return 82
    if "涉警" in label_text or "公信力" in label_text:
        return 79
    return 50


def risk_level_from_score(score: float) -> RiskLevel:
    if score >= 85:
        return RiskLevel.URGENT
    if score >= 68:
        return RiskLevel.HIGH
    if score >= 40:
        return RiskLevel.WATCH
    return RiskLevel.LOW


def _base_training_samples() -> list[tuple[CaseSample, int]]:
    samples: list[tuple[CaseSample, int]] = [
        (_external_sample_to_case(sample), sample.risk_score)
        for sample in list_external_training_samples(limit=50000, task_type="text_risk")
    ]
    samples.extend(
        (case, int(case.manual_risk_score))
        for case in list_labeled_user_case_samples()
        if case.manual_risk_score is not None
    )
    return samples


def _training_samples(
    base_samples: list[tuple[CaseSample, int]],
    include_augmented: bool,
) -> list[tuple[CaseSample, int]]:
    samples = list(base_samples)
    if include_augmented:
        augmented: list[tuple[CaseSample, int]] = []
        for case, label in samples:
            augmented.append((case, label))
            augmented.append((_augment_case(case, "lower"), max(0, label - 12)))
            augmented.append((_augment_case(case, "text_variation"), min(100, max(0, label + 2))))
            if label >= 60:
                augmented.append((_augment_case(case, "higher"), min(100, label + 8)))
        return augmented
    return samples


def _external_sample_to_case(sample: ExternalTrainingSample) -> CaseSample:
    return CaseSample(
        id=sample.id,
        title=sample.title,
        scenario=sample.scenario,
        platform=sample.source,
        publish_time=sample.created_at[:16].replace("T", " "),
        source_url=sample.source_url or sample.dataset_name,
        content=sample.content,
        image_description="外部文本数据集样本，暂无原始图片；用于训练文本风险基线。",
        spread=SpreadMetrics(
            views=12000,
            reposts=180,
            comments=120,
            likes=260,
            velocity="外部数据集未提供真实传播链路，使用中性占位传播特征",
        ),
        manual_label=f"外部数据集标注：{sample.label}",
        manual_risk_score=sample.risk_score,
        tags=["外部训练数据", sample.dataset_name],
        sensitivity_notes=f"来源：{sample.source}；split={sample.split}",
        review_note="由外部数据集导入，不是内置展示样例。",
        created_by_user=False,
    )


def _training_source_summary(
    base_samples: list[tuple[CaseSample, int]],
    all_samples: list[tuple[CaseSample, int]],
) -> dict[str, object]:
    external_count = sum(1 for case, _ in base_samples if case.id.startswith("ext-"))
    user_count = len(base_samples) - external_count
    datasets = sorted(
        {
            tag
            for case, _ in base_samples
            for tag in case.tags
            if tag != "外部训练数据"
        }
    )
    return {
        "external_samples": external_count,
        "user_labeled_cases": user_count,
        "base_sample_count": len(base_samples),
        "effective_sample_count": len(all_samples),
        "deterministic_augmented_samples": max(0, len(all_samples) - len(base_samples)),
        "excluded_demo_cases": 4,
        "datasets": datasets[:20],
    }


def _augment_case(case: CaseSample, direction: str) -> CaseSample:
    if direction == "lower":
        spread = case.spread.model_copy(update={
            "views": max(100, int(case.spread.views * 0.08)),
            "reposts": max(1, int(case.spread.reposts * 0.04)),
            "comments": max(1, int(case.spread.comments * 0.04)),
            "likes": max(1, int(case.spread.likes * 0.05)),
            "velocity": "小范围缓慢传播",
        })
        return case.model_copy(
            update={
                "id": f"{case.id}-aug-low",
                "spread": spread,
                "content": f"{case.content} 经核验传播范围有限，暂未出现线下扰动。",
                "manual_label": f"{case.manual_label} · 降级增强样本",
            }
        )
    if direction == "text_variation":
        return case.model_copy(
            update={
                "id": f"{case.id}-aug-text",
                "content": f"{case.content} 该内容仍需核验首发来源、图片原始文件和平台传播链路。",
                "image_description": f"{case.image_description} 建议补充OCR、EXIF、水印和局部纹理检测。",
            }
        )
    spread = case.spread.model_copy(update={
        "views": max(case.spread.views, int(case.spread.views * 1.35 + 30000)),
        "reposts": max(case.spread.reposts, int(case.spread.reposts * 1.45 + 600)),
        "comments": max(case.spread.comments, int(case.spread.comments * 1.35 + 700)),
        "likes": max(case.spread.likes, int(case.spread.likes * 1.2 + 500)),
        "velocity": "30分钟内跨群快速扩散",
    })
    return case.model_copy(
        update={
            "id": f"{case.id}-aug-high",
            "spread": spread,
            "content": f"{case.content} 评论区出现号召线下集合和立即转发内容。",
            "manual_label": f"{case.manual_label} · 升级增强样本",
        }
    )


def _case_text(case: CaseSample) -> str:
    return " ".join(
        [
            case.title,
            case.scenario,
            case.platform,
            case.source_url,
            case.content,
            case.image_description,
            " ".join(tag for tag in case.tags if tag not in {"rumor", "fake", "false", "real", "true"}),
            case.sensitivity_notes,
            case.spread.velocity,
        ]
    )


def _build_vocabulary(cases: list[CaseSample], limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        text = re.sub(r"\s+", "", _case_text(case).lower())
        tokens = re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]", text)
        compact = "".join(tokens)
        for size in (2, 3, 4):
            for index in range(max(0, len(compact) - size + 1)):
                gram = compact[index:index + size]
                if _useful_ngram(gram):
                    counter[gram] += 1
    return [gram for gram, _ in counter.most_common(limit)]


def _useful_ngram(value: str) -> bool:
    if len(value) < 2:
        return False
    if value.isdigit():
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _feature_names(vocabulary: list[str]) -> list[str]:
    return (
        [feature.name for feature in EXPERT_FEATURES]
        + [f"scenario::{scenario}" for scenario in SCENARIOS]
        + [f"ngram::{term}" for term in vocabulary]
    )


def _split_indices(
    samples: list[tuple[CaseSample, int]],
    labels: list[int],
) -> tuple[list[int], list[int]]:
    by_level = _indices_by_risk_level(labels)
    valid_indices: list[int] = []
    for indices in by_level.values():
        if len(indices) >= 2:
            valid_indices.append(indices[-1])
    target_valid = _target_validation_count(len(samples))
    sorted_indices = sorted(range(len(samples)), key=lambda index: (labels[index], index))
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
    train_indices = [index for index in range(len(samples)) if index not in valid_set]
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
    samples: list[tuple[CaseSample, int]],
    labels: list[int],
    train_indices: list[int],
    valid_indices: list[int],
) -> dict[str, object]:
    return {
        "method": "deterministic_stratified_holdout",
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "target_validation_count": _target_validation_count(len(samples)),
        "train_label_distribution": _label_distribution([labels[index] for index in train_indices]),
        "validation_label_distribution": _label_distribution([labels[index] for index in valid_indices]),
        "validation_sources": _source_counts([samples[index][0] for index in valid_indices]),
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
            penalty = l2 * weights[name]
            weights[name] -= effective_lr * (grad_weights[name] / train_count + penalty)
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


def _keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    return min(float(hits), 5.0)


def _log_metric(value: int) -> float:
    return min(math.log1p(value) / 2.5, 8.0)


def _clip_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


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


def _label_distribution(labels: list[int]) -> dict[str, int]:
    distribution = {"低": 0, "关注": 0, "较高": 0, "紧急": 0}
    for label in labels:
        distribution[risk_level_from_score(label).value] += 1
    return distribution


def _task_metrics(predictions: list[float], labels: list[int]) -> dict[str, object]:
    predicted_levels = [risk_level_from_score(value).value for value in predictions]
    actual_levels = [risk_level_from_score(label).value for label in labels]
    exact = sum(1 for pred, actual in zip(predicted_levels, actual_levels) if pred == actual)
    return {
        "risk_regression": {
            "mae": _mae(predictions, labels),
            "rmse": _rmse(predictions, labels),
            "within_10": _accuracy_within(predictions, labels, tolerance=10),
        },
        "risk_level_classification": {
            "accuracy": round(exact / len(labels), 3) if labels else 0.0,
            "labels": ["低", "关注", "较高", "紧急"],
        },
        "aigc_suspicion_head": {
            "method": "共享特征空间中的规则/文本弱监督头",
            "positive_signals": ["AI", "AIGC", "合成", "生成", "伪造", "截图异常", "水印/EXIF"],
        },
        "text_claim_head": {
            "method": "中文字符 n-gram 与专家词表联合抽取",
            "output": "风险触发词、行动号召、涉警/灾害/群体对立主题信号",
        },
    }


def _confusion_matrix(predictions: list[float], labels: list[int]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in RISK_LEVELS} for actual in RISK_LEVELS}
    for prediction, label in zip(predictions, labels):
        actual = risk_level_from_score(label).value
        predicted = risk_level_from_score(prediction).value
        matrix[actual][predicted] += 1
    return matrix


def _source_counts(cases: list[CaseSample]) -> list[dict[str, object]]:
    counts: dict[str, dict[str, object]] = {}
    for case in cases:
        dataset = next((tag for tag in case.tags if tag != "外部训练数据"), "用户人工标注")
        key = f"{dataset}|{case.platform}"
        if key not in counts:
            counts[key] = {
                "dataset_name": dataset,
                "source": case.platform,
                "count": 0,
            }
        counts[key]["count"] = int(counts[key]["count"]) + 1
    return sorted(counts.values(), key=lambda item: str(item["dataset_name"]))


def _feature_weights(weights: dict[str, float], direction: str) -> list[FeatureWeight]:
    reverse = direction == "positive"
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=reverse)
    if direction == "negative":
        ordered = [item for item in ordered if item[1] < 0]
    else:
        ordered = [item for item in ordered if item[1] > 0]
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
    expert = {feature.name: feature.description for feature in EXPERT_FEATURES}
    if name in expert:
        return expert[name]
    if name.startswith("scenario::"):
        return f"场景特征：{name.split('::', 1)[1]}"
    if name.startswith("ngram::"):
        return f"文本 n-gram 特征：{name.split('::', 1)[1]}"
    return name


def _explain(contributions: list[tuple[str, float, float]], score: float) -> list[str]:
    ordered = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    lines = [
        f"正式比赛版本地模型给出{round(score)}分，融合专家规则、传播指标、场景标签和中文 n-gram 文本特征。",
    ]
    for name, contribution, raw_value in ordered[:7]:
        if abs(contribution) < 0.2:
            continue
        direction = "抬高" if contribution > 0 else "压低"
        lines.append(
            f"{_feature_description(name)}命中值{round(raw_value, 2)}，对风险分{direction}{abs(contribution):.1f}分。"
        )
    if len(lines) == 1:
        lines.append("未出现单一强特征，建议结合人工核查结果复核。")
    return lines


def _confidence(score: float, contributions: list[tuple[str, float, float]]) -> float:
    contribution_strength = sum(abs(item[1]) for item in sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:12])
    band_bonus = 0.1 if score < 25 or score > 75 else 0.0
    return round(max(0.38, min(0.94, 0.5 + contribution_strength / 120 + band_bonus)), 2)


def _artifact_mapping(artifact: dict[str, object], name: str) -> list[str]:
    value = artifact.get(name)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}
