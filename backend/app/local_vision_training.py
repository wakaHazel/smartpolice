from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import math
import re
from statistics import mean
from uuid import uuid4

from app.models import (
    CaseAsset,
    FeatureWeight,
    LocalVisionCalibrationRunRequest,
    LocalVisionCalibrationRunResult,
    LocalVisionTrainingDataset,
    LocalVisionTrainingSample,
    LocalVisionTrainingStats,
    LocalVisionTrainingStatus,
)
from app.storage import (
    get_active_local_vision_training_artifact,
    get_latest_local_vision_training_run,
    list_case_assets,
    list_case_samples,
    list_llm_invocations,
    load_case_sample,
    save_local_vision_training_run,
)


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "vision_confidence": "视觉模型自报置信度",
    "ocr_count": "OCR 文本条目数量",
    "visual_fact_count": "画面事实条目数量",
    "tamper_signal_count": "合成/篡改迹象数量",
    "consistency_count": "图文一致性判断数量",
    "uncertainty_count": "不确定项数量",
    "generator_candidate_count": "生成/编辑模型候选数量",
    "max_generator_confidence": "生成/编辑候选最高置信度",
    "raw_length": "视觉回复文本长度",
    "evidence_density": "视觉输出证据密度",
    "image_megapixels": "图像像素规模",
}

KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "tamper_keywords": ("伪造", "篡改", "合成", "异常", "编辑", "拼接", "水印", "压缩", "失真"),
    "uncertainty_keywords": ("待核查", "无法确认", "不确定", "可能", "疑似", "需要", "建议补充"),
    "public_safety_keywords": ("警方", "公安", "警情", "灾害", "伤亡", "聚集", "冲突", "恐慌"),
    "consistency_keywords": ("一致", "矛盾", "不一致", "部分吻合", "无法对应", "图文"),
}


def build_local_vision_training_dataset(
    case_id: str | None = None,
    limit: int = 100,
) -> LocalVisionTrainingDataset:
    safe_limit = max(1, min(limit, 500))
    cases = [load_case_sample(case_id)] if case_id else list_case_samples()
    samples: list[LocalVisionTrainingSample] = []
    for case in cases:
        assets = {asset.id: asset for asset in list_case_assets(case.id)}
        invocations = list_llm_invocations(case_id=case.id, limit=safe_limit)
        for invocation in invocations:
            if invocation.provider != "LocalVision" or invocation.role != "视觉证据分析":
                continue
            if invocation.status not in {"success", "skipped_optional_local_vlm"} or not invocation.response_text:
                continue
            asset_id = _asset_id_from_payload(invocation.request_payload, assets)
            if asset_id is None:
                continue
            asset = assets[asset_id]
            parsed = _parse_raw_response(invocation.response_text)
            samples.append(
                LocalVisionTrainingSample(
                    case_id=case.id,
                    asset_id=asset.id,
                    image_path=asset.storage_path,
                    image_sha256=asset.sha256,
                    prompt=_prompt_from_payload(invocation.request_payload),
                    response={
                        "raw": invocation.response_text,
                        "parsed": parsed,
                    },
                    manual_label=case.manual_label,
                    manual_risk_score=case.manual_risk_score,
                    created_at=invocation.created_at,
                )
            )
            if len(samples) >= safe_limit:
                break
        if len(samples) >= safe_limit:
            break
    return LocalVisionTrainingDataset(
        id=f"local-vision-dataset-{uuid4().hex[:10]}",
        created_at=datetime.now(UTC).isoformat(),
        model_target="本地可选 VLM + 本地视觉风险校准头 / 通用 JSONL",
        sample_count=len(samples),
        format="jsonl-compatible image_instruction_response",
        samples=samples,
        note=(
            "该接口只导出 LocalVision 或本地离线视觉审计产生的样本；没有视觉审计记录就不会生成训练样本。"
            "当前系统先训练本地视觉风险校准头，样本积累后可导出给任意开源多模态基础模型继续微调。"
        ),
    )


def build_local_vision_training_stats(
    case_id: str | None = None,
    limit: int = 500,
) -> LocalVisionTrainingStats:
    dataset = build_local_vision_training_dataset(case_id=case_id, limit=limit)
    labeled = [
        sample
        for sample in dataset.samples
        if sample.manual_risk_score is not None
    ]
    scores = [float(sample.manual_risk_score) for sample in labeled if sample.manual_risk_score is not None]
    label_distribution = _label_distribution([int(score) for score in scores])
    return LocalVisionTrainingStats(
        sample_count=dataset.sample_count,
        labeled_sample_count=len(labeled),
        unlabeled_sample_count=dataset.sample_count - len(labeled),
        case_count=len({sample.case_id for sample in dataset.samples}),
        image_count=len({sample.image_sha256 for sample in dataset.samples}),
        average_manual_risk_score=round(mean(scores), 2) if scores else None,
        label_distribution=label_distribution,
        export_ready=dataset.sample_count > 0,
        training_ready=len(labeled) >= 4,
        note=_stats_note(dataset.sample_count, len(labeled)),
    )


def build_local_vision_jsonl(
    case_id: str | None = None,
    limit: int = 500,
) -> str:
    dataset = build_local_vision_training_dataset(case_id=case_id, limit=limit)
    lines: list[str] = []
    for sample in dataset.samples:
        parsed = _parsed_response(sample)
        payload = {
            "id": f"{sample.case_id}:{sample.asset_id}",
            "image": sample.image_path,
            "messages": [
                {"role": "user", "content": sample.prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(parsed, ensure_ascii=False),
                },
            ],
            "metadata": {
                "case_id": sample.case_id,
                "asset_id": sample.asset_id,
                "image_sha256": sample.image_sha256,
                "manual_label": sample.manual_label,
                "manual_risk_score": sample.manual_risk_score,
                "created_at": sample.created_at,
            },
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def train_local_vision_calibrator(
    request: LocalVisionCalibrationRunRequest,
) -> LocalVisionCalibrationRunResult:
    dataset = build_local_vision_training_dataset(limit=500)
    labeled_samples = [
        sample
        for sample in dataset.samples
        if sample.manual_risk_score is not None
    ]
    if len(labeled_samples) < request.min_samples:
        raise ValueError(
            f"本地视觉校准训练至少需要 {request.min_samples} 条带人工风险分的 LocalVision 样本，"
            f"当前只有 {len(labeled_samples)} 条。请先完成真实视觉核验并保存人工标注。"
        )

    assets = _assets_by_id()
    rows = [extract_local_vision_features(sample, assets.get(sample.asset_id)) for sample in labeled_samples]
    labels = [int(sample.manual_risk_score) for sample in labeled_samples if sample.manual_risk_score is not None]
    feature_names = sorted({name for row in rows for name in row})
    train_indices, valid_indices = _split_indices(labels)
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
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    result = LocalVisionCalibrationRunResult(
        id=run_id,
        created_at=created_at,
        model_kind="local-vision-risk-calibrator-ridge-v1",
        status="trained",
        sample_count=len(labeled_samples),
        validation_count=len(valid_indices),
        feature_count=len(feature_names),
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        train_mae=_mae(train_predictions, train_labels),
        validation_mae=_mae(valid_predictions, valid_labels),
        accuracy_within_10=_accuracy_within(valid_predictions, valid_labels, tolerance=10),
        label_distribution=_label_distribution(labels),
        top_positive_features=_feature_weights(weights, "positive")[:8],
        top_negative_features=_feature_weights(weights, "negative")[:8],
        training_trace=[
            f"读取 {len(dataset.samples)} 条 LocalVision 审计样本，其中 {len(labeled_samples)} 条具备人工风险分。",
            "解析视觉模型 JSON 输出，抽取 OCR、画面事实、篡改迹象、不确定性、候选生成模型和置信度特征。",
            f"训练本地 L2 风险校准头：训练 {len(train_indices)} 条，验证 {len(valid_indices)} 条，特征 {len(feature_names)} 维。",
            "保存校准权重、标准化参数、指标与模型卡；后续真实研判会把校准结果写入视觉结构化输出。",
        ],
        model_card={
            "name": "LocalVision 视觉风险校准头",
            "version": run_id,
            "base_model": "optional local VLM via Ollama/OpenAI-compatible API",
            "architecture": "LocalVision JSON 特征 + 图像元数据 + L2 风险回归校准头",
            "training_data": "真实 LocalVision 模型调用审计、图片哈希、人工风险标注。",
            "intended_use": "校准视觉证据对公共安全谣言风险的辅助贡献，提升本地模型链路可解释性。",
            "not_for": "不宣称微调基础多模态模型，不替代人工事实核验或执法结论。",
            "next_step": "样本量达到数十到数百条后，可将 JSONL 交给 LLaMA-Factory/PEFT 等工具微调本地开源多模态模型。",
        },
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
        "model_card": result.model_card,
    }
    save_local_vision_training_run(result, artifact)
    return result


def get_local_vision_training_status() -> LocalVisionTrainingStatus:
    latest = get_latest_local_vision_training_run()
    stats = build_local_vision_training_stats(limit=500)
    if latest is None:
        return LocalVisionTrainingStatus(
            trained=False,
            active_model_id=None,
            latest_run=None,
            dataset=stats,
            note="本地视觉模型已可真实推理；校准头尚未训练，需要先积累带人工风险分的视觉核验样本。",
        )
    return LocalVisionTrainingStatus(
        trained=True,
        active_model_id=latest.id,
        latest_run=latest,
        dataset=stats,
        note="LocalVision 视觉风险校准头已启用；真实研判会在视觉结构化结果中附加本地校准分。",
    )


def calibrate_local_vision_result(
    structured: dict[str, object],
    asset: CaseAsset | None = None,
) -> dict[str, object] | None:
    artifact = get_active_local_vision_training_artifact()
    if artifact is None:
        return None
    feature_names = _artifact_list(artifact, "feature_names")
    weights = _float_mapping(artifact.get("weights"))
    means = _float_mapping(artifact.get("means"))
    scales = _float_mapping(artifact.get("scales"))
    if not feature_names or not weights or not means or not scales:
        return None
    sample = LocalVisionTrainingSample(
        case_id="runtime",
        asset_id=asset.id if asset else "runtime-asset",
        image_path=asset.storage_path if asset else "",
        image_sha256=asset.sha256 if asset else "",
        prompt="runtime calibration",
        response={"parsed": structured, "raw": json.dumps(structured, ensure_ascii=False)},
        manual_label="runtime",
        manual_risk_score=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    features = extract_local_vision_features(sample, asset)
    normalized = _normalize(features, means, scales)
    bias = float(artifact.get("bias", 0))
    raw_score = bias + sum(
        weights.get(name, 0.0) * normalized.get(name, 0.0)
        for name in feature_names
    )
    score = _clip_score(raw_score)
    contributions = [
        (
            name,
            weights.get(name, 0.0) * normalized.get(name, 0.0),
            features.get(name, 0.0),
        )
        for name in feature_names
    ]
    ordered = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    return {
        "model_id": str(artifact.get("id", "")),
        "model_kind": str(artifact.get("model_kind", "local-vision-risk-calibrator-ridge-v1")),
        "score": score,
        "risk_level": _risk_level(score),
        "confidence": _confidence(score, ordered),
        "explanations": _explanations(score, ordered),
        "top_contributions": [
            {
                "name": name,
                "description": _feature_description(name),
                "value": round(raw_value, 4),
                "contribution": round(contribution, 4),
            }
            for name, contribution, raw_value in ordered[:6]
        ],
    }


def extract_local_vision_features(
    sample: LocalVisionTrainingSample,
    asset: CaseAsset | None = None,
) -> dict[str, float]:
    parsed = _parsed_response(sample)
    raw_text = _response_text(sample, parsed).lower()
    generator_candidates = parsed.get("generator_candidates")
    candidate_confidences = _candidate_confidences(generator_candidates)
    features: dict[str, float] = {
        "vision_confidence": _number(parsed.get("confidence"), default=0.0),
        "ocr_count": float(len(_list_value(parsed.get("ocr_text")))),
        "visual_fact_count": float(len(_list_value(parsed.get("visual_facts")))),
        "tamper_signal_count": float(len(_list_value(parsed.get("aigc_or_tamper_signals")))),
        "consistency_count": float(len(_list_value(parsed.get("text_image_consistency")))),
        "uncertainty_count": float(len(_list_value(parsed.get("uncertainties")))),
        "generator_candidate_count": float(len(candidate_confidences)),
        "max_generator_confidence": max(candidate_confidences) if candidate_confidences else 0.0,
        "raw_length": min(len(raw_text) / 300.0, 8.0),
    }
    evidence_items = (
        features["ocr_count"]
        + features["visual_fact_count"]
        + features["tamper_signal_count"]
        + features["consistency_count"]
    )
    features["evidence_density"] = min(evidence_items / max(features["uncertainty_count"], 1.0), 8.0)
    if asset and asset.width and asset.height:
        features["image_megapixels"] = min((asset.width * asset.height) / 1_000_000.0, 24.0)
    else:
        features["image_megapixels"] = 0.0
    for name, keywords in KEYWORD_GROUPS.items():
        features[name] = _keyword_score(raw_text, keywords)
    return features


def _asset_id_from_payload(
    payload: dict[str, object],
    assets: dict[str, CaseAsset],
) -> str | None:
    text = str(payload)
    for asset_id in assets:
        if asset_id in text:
            return asset_id
    if len(assets) == 1:
        return next(iter(assets))
    return None


def _prompt_from_payload(payload: dict[str, object]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    first = messages[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                value = item.get("text")
                if isinstance(value, str):
                    texts.append(value)
        return "\n".join(texts)
    return ""


def _parse_raw_response(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            return {"raw_text": text}
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw_text": text}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _parsed_response(sample: LocalVisionTrainingSample) -> dict[str, object]:
    parsed = sample.response.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    raw = sample.response.get("raw")
    return _parse_raw_response(raw if isinstance(raw, str) else "")


def _response_text(
    sample: LocalVisionTrainingSample,
    parsed: dict[str, object],
) -> str:
    raw = sample.response.get("raw")
    if isinstance(raw, str):
        return raw
    return json.dumps(parsed, ensure_ascii=False)


def _list_value(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _candidate_confidences(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    confidences: list[float] = []
    for item in value:
        if isinstance(item, dict):
            confidences.append(_number(item.get("confidence"), default=0.0))
    return confidences


def _number(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(float(value), 1.0))
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return default
        return max(0.0, min(parsed, 1.0))
    return default


def _assets_by_id() -> dict[str, CaseAsset]:
    assets: dict[str, CaseAsset] = {}
    for case in list_case_samples():
        for asset in list_case_assets(case.id):
            assets[asset.id] = asset
    return assets


def _split_indices(labels: list[int]) -> tuple[list[int], list[int]]:
    if len(labels) <= 2:
        return [0], [1]
    valid_count = max(1, min(len(labels) // 4, 8))
    sorted_indices = sorted(range(len(labels)), key=lambda index: labels[index])
    valid_indices = sorted(sorted_indices[1:: max(1, len(labels) // valid_count)][:valid_count])
    if not valid_indices:
        valid_indices = [len(labels) - 1]
    valid_set = set(valid_indices)
    train_indices = [index for index in range(len(labels)) if index not in valid_set]
    if not train_indices:
        train_indices = [index for index in range(len(labels)) if index != valid_indices[0]]
    return train_indices, valid_indices


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
            predicted = bias + sum(
                weights[name] * normalized.get(name, 0.0)
                for name in feature_names
            )
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
            direction="提升视觉风险" if direction == "positive" else "降低视觉风险",
        )
        for name, weight in ordered
    ]


def _feature_description(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name in KEYWORD_GROUPS:
        return f"视觉输出关键词组：{name}"
    return name


def _mae(predictions: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return round(mean(abs(prediction - label) for prediction, label in zip(predictions, labels)), 2)


def _accuracy_within(
    predictions: list[float],
    labels: list[int],
    tolerance: int,
) -> float:
    if not labels:
        return 0.0
    passed = sum(
        1
        for prediction, label in zip(predictions, labels)
        if abs(prediction - label) <= tolerance
    )
    return round(passed / len(labels), 3)


def _label_distribution(labels: list[int]) -> dict[str, int]:
    distribution = {"低": 0, "关注": 0, "较高": 0, "紧急": 0}
    for label in labels:
        distribution[_risk_level(label)] += 1
    return distribution


def _keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    return float(min(hits, 6))


def _risk_level(score: float) -> str:
    if score >= 85:
        return "紧急"
    if score >= 68:
        return "较高"
    if score >= 40:
        return "关注"
    return "低"


def _clip_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _confidence(
    score: float,
    ordered_contributions: list[tuple[str, float, float]],
) -> float:
    contribution_strength = sum(abs(item[1]) for item in ordered_contributions[:8])
    band_bonus = 0.08 if score < 30 or score > 72 else 0.0
    return round(max(0.36, min(0.93, 0.48 + contribution_strength / 100 + band_bonus)), 2)


def _explanations(
    score: float,
    ordered_contributions: list[tuple[str, float, float]],
) -> list[str]:
    lines = [
        f"本地视觉风险校准头给出 {round(score)} 分，依据 LocalVision 结构化输出和图像元数据计算。",
    ]
    for name, contribution, raw_value in ordered_contributions[:5]:
        if abs(contribution) < 0.15:
            continue
        direction = "抬高" if contribution > 0 else "压低"
        lines.append(
            f"{_feature_description(name)}取值 {round(raw_value, 2)}，{direction}视觉风险 {abs(contribution):.1f} 分。"
        )
    if len(lines) == 1:
        lines.append("当前样本未出现强单项视觉校准特征，应继续结合 URL 来源和人工核查。")
    return lines


def _artifact_list(artifact: dict[str, object], name: str) -> list[str]:
    value = artifact.get(name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _stats_note(sample_count: int, labeled_count: int) -> str:
    if sample_count == 0:
        return "尚无 LocalVision 真实调用样本；请先上传图片并完成证据链研判。"
    if labeled_count < 4:
        return "样本可以导出 JSONL，但带人工风险分的样本不足 4 条，暂不建议训练校准头。"
    return "样本已满足本地视觉风险校准头训练下限，可启动训练并保存 active 版本。"
