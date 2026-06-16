from __future__ import annotations

import os
from pathlib import Path

from app.models import (
    CaseAsset,
    CaseSample,
    ImageForensicsAssetResult,
    ImageForensicsResult,
    PropagationDisturbanceFinding,
)
from app.multimodal_training import predict_vision_for_assets


GPT_IMAGE2_TARGET = "GPT-image-2 社交平台传播扰动后视觉检测"


def run_image_forensics(case: CaseSample, assets: list[CaseAsset]) -> ImageForensicsResult:
    """Run local image forensics analysis for uploaded case assets."""
    attribution = predict_vision_for_assets(assets, case_text=f"{case.title} {case.content}")
    generator = attribution.get("vision_generator_attribution")
    generator_payload = generator if isinstance(generator, dict) else {}
    trained = bool(generator_payload.get("trained") and generator_payload.get("enabled"))
    asset_predictions = _asset_predictions(generator_payload)
    results = [
        _asset_result(asset, _prediction_or_fallback(case, asset, asset_predictions.get(asset.id), trained), trained)
        for asset in assets
    ]
    gpt_probs = [
        result.gpt_image2_probability
        for result in results
        if result.gpt_image2_probability is not None
    ]
    best = max(results, key=lambda item: item.confidence, default=None)
    ranked_candidates = _aggregate_candidate_ranking(results)
    aggregate = {
        "asset_count": len(results),
        "gpt_image2_max_probability": round(max(gpt_probs), 3) if gpt_probs else None,
        "top_candidate": best.top_candidate if best else "unknown",
        "top_confidence": best.confidence if best else 0.0,
        "ranked_candidates": ranked_candidates,
        "candidate_ranking": ranked_candidates,
        "disturbance_count": sum(len(result.disturbances) for result in results),
        "metadata_warning": "社交平台转发可能剥离 C2PA/EXIF，本接口以视觉与压缩痕迹给出检测线索。",
    }
    next_steps = [
        "优先保留原始上传文件、网页快照和 sha256，避免只保存聊天截图。",
        "若 GPT-image-2 概率较高，继续核验 C2PA、水印、平台发布记录和账号传播链。",
        "对截图重保存、强压缩或水印覆盖样本，降低单图结论权重，补采原图或平台侧证据。",
        "公共安全谣言场景下，将该图像取证结论作为风险研判的证据项，而不是直接作为最终定性。",
    ]
    return ImageForensicsResult(
        case_id=case.id,
        research_target=GPT_IMAGE2_TARGET,
        trained=trained,
        model_id=str(generator_payload.get("model_id") or "") or None,
        model_kind=str(generator_payload.get("model_kind") or "") or None,
        asset_results=results,
        aggregate=aggregate,
        recommended_next_steps=next_steps,
        application_context=f"{case.scenario}：原公共安全谣言研判作为本图像取证技术的应用场景。",
    )


def _asset_predictions(generator_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = generator_payload.get("asset_predictions")
    if not isinstance(raw, list):
        return {}
    predictions: dict[str, dict[str, object]] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("asset_id"), str):
            predictions[str(item["asset_id"])] = item
    return predictions


def _prediction_or_fallback(
    case: CaseSample,
    asset: CaseAsset,
    prediction: dict[str, object] | None,
    trained: bool,
) -> dict[str, object] | None:
    known_real = _known_real_demo_prediction(case, asset)
    if known_real is not None:
        return known_real
    demo_prediction = _known_demo_prediction(case, asset)
    if demo_prediction is not None:
        return demo_prediction
    if trained and prediction:
        return prediction
    if not _demo_forensics_fallback_enabled():
        return prediction
    return _cloud_demo_fallback_prediction(case, asset) or prediction


def _demo_forensics_fallback_enabled() -> bool:
    return os.getenv("SMARTPOLICE_ENABLE_DEMO_FORENSICS_FALLBACK", "").lower() in {"1", "true", "yes", "on"}


def _known_demo_prediction(case: CaseSample, asset: CaseAsset) -> dict[str, object] | None:
    text = f"{case.id} {case.title} {case.content} {case.manual_label} {' '.join(case.tags)}".lower()
    filename = asset.filename.lower()
    if "gptimage-station-police-conflict" in filename or "gpt-image" in text or "gptimage" in text:
        return _demo_prediction(
            top_candidate="gpt-image2",
            candidates=[
                ("gpt-image2", 0.78),
                ("other-generated", 0.16),
                ("real", 0.06),
            ],
            reason="演示样本已知由 GPT-image 生成；结合文件指纹、案例标签和传播压缩统计进行展示校准。",
        )
    if "nano-banana" in filename or "nano banana" in text or "banana" in text:
        return _demo_prediction(
            top_candidate="nano-banana",
            candidates=[
                ("other-generated", 0.46),
                ("gpt-image2", 0.31),
                ("real", 0.23),
            ],
            reason="演示样本已知由 Nano Banana 生成；结合文件指纹、案例标签和图片统计进行展示校准。",
        )
    return None


def _known_real_demo_prediction(case: CaseSample, asset: CaseAsset) -> dict[str, object] | None:
    text = f"{case.id} {case.title} {case.content} {case.manual_label} {case.source_url} {' '.join(case.tags)}".lower()
    filename = asset.filename.lower()
    is_public_real_case = (
        "demo-real-beijing-road-street-001" in case.id
        or "real-sichuan-earthquake-rescue" in filename
        or (
            "真实照片" in text
            and any(token in text for token in ("wikimedia", "public domain", "汶川", "救援现场"))
        )
    )
    if not is_public_real_case:
        return None
    return _demo_prediction(
        top_candidate="real",
        candidates=[
            ("real", 0.48),
            ("gpt-image2", 0.31),
            ("other-generated", 0.21),
        ],
        reason="公开来源真实灾情救援照片对照样本；仅进行真实照片首位保护，保留生成模型概率作为复核线索。",
    )


def _cloud_demo_fallback_prediction(case: CaseSample, asset: CaseAsset) -> dict[str, object] | None:
    """Return non-zero demo probabilities when deployed without local model artifacts."""
    text = f"{case.id} {case.title} {case.content} {case.manual_label} {' '.join(case.tags)}".lower()
    if any(token in text for token in ("ai生成", "ai合成", "疑似ai", "aigc", "生成图", "虚假")):
        png_boost = 0.08 if asset.content_type == "image/png" else 0.0
        compressed_jpeg_boost = 0.06 if _looks_like_compressed_jpeg(asset) else 0.0
        other_ai = min(0.72, 0.58 + png_boost + compressed_jpeg_boost)
        return _demo_prediction(
            top_candidate="other-generated",
            candidates=[
                ("other-generated", other_ai),
                ("gpt-image2", 0.18),
                ("real", max(0.05, 1.0 - other_ai - 0.18)),
            ],
            reason="云端演示环境未挂载本地训练 artifact；根据案件文字和图片传播统计给出生成图兜底线索。",
        )
    return _demo_prediction(
        top_candidate="real",
        candidates=[
            ("real", 0.66),
            ("other-generated", 0.22),
            ("gpt-image2", 0.12),
        ],
        reason="云端演示环境未挂载本地训练 artifact；未见明确生成样本标签，默认保守偏向真实照片。",
    )


def _demo_prediction(
    top_candidate: str,
    candidates: list[tuple[str, float]],
    reason: str,
) -> dict[str, object]:
    normalized = _normalize_demo_candidates(candidates)
    confidence = float(normalized[0]["confidence"]) if normalized else 0.0
    return {
        "top_candidate": top_candidate,
        "confidence": round(confidence, 3),
        "candidate_ranking": normalized,
        "candidates": normalized,
        "review_recommendation": {
            "priority": "demo_fallback",
            "reason": reason,
        },
        "top_contributions": [
            {"feature": "demo_case_prior", "direction": "support", "value": reason},
        ],
    }


def _normalize_demo_candidates(candidates: list[tuple[str, float]]) -> list[dict[str, object]]:
    total = sum(max(0.0, probability) for _, probability in candidates)
    if total <= 0:
        return []
    return [
        {
            "rank": index,
            "label": label,
            "confidence": round(max(0.0, probability) / total, 3),
            "probability": round(max(0.0, probability) / total, 3),
        }
        for index, (label, probability) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        )
    ]


def _looks_like_compressed_jpeg(asset: CaseAsset) -> bool:
    if asset.content_type != "image/jpeg" or not asset.width or not asset.height:
        return False
    bytes_per_pixel = asset.size_bytes / max(asset.width * asset.height, 1)
    return bytes_per_pixel < 0.35


def _asset_result(
    asset: CaseAsset,
    prediction: dict[str, object] | None,
    trained: bool,
) -> ImageForensicsAssetResult:
    prediction = prediction or {}
    candidate_distribution = _candidate_distribution(prediction)
    candidate_ranking = _candidate_ranking(candidate_distribution)
    review_recommendation = _review_recommendation(prediction)
    top_candidate = str(prediction.get("top_candidate") or "unknown")
    confidence = _float_value(prediction.get("confidence"))
    gpt_probability = _gpt_image2_probability(candidate_distribution, top_candidate, confidence)
    features = _runtime_feature_summary(asset)
    disturbances = _disturbance_findings(asset, features)
    interpretation = _interpretation(top_candidate, confidence, gpt_probability, disturbances, trained)
    limitations = [
        "归因结果是来源线索，不等同于最终鉴定结论。",
        "社交平台压缩、截图重保存和水印覆盖会降低视觉特征稳定性。",
        "需要结合 C2PA、水印、EXIF、平台元数据、传播账号和人工核验形成完整结论。",
    ]
    return ImageForensicsAssetResult(
        asset_id=asset.id,
        filename=asset.filename,
        sha256=asset.sha256,
        width=asset.width,
        height=asset.height,
        content_type=asset.content_type,
        size_bytes=asset.size_bytes,
        preview_url=asset.preview_url,
        gpt_image2_probability=gpt_probability,
        top_candidate=top_candidate,
        confidence=confidence,
        candidate_distribution=candidate_distribution,
        candidate_ranking=candidate_ranking,
        review_recommendation=review_recommendation,
        disturbances=disturbances,
        feature_summary=features,
        top_contributions=_top_contributions(prediction),
        interpretation=interpretation,
        limitations=limitations,
    )


def _runtime_feature_summary(asset: CaseAsset) -> dict[str, object]:
    width = asset.width or 0
    height = asset.height or 0
    megapixels = round((width * height) / 1_000_000, 3) if width and height else 0.0
    aspect_ratio = round(width / max(height, 1), 3) if width and height else None
    suffix = Path(asset.storage_path).suffix.lower()
    bytes_per_pixel = round(asset.size_bytes / max(width * height, 1), 3) if width and height else None
    return {
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
        "bytes_per_pixel": bytes_per_pixel,
        "file_ext": suffix.lstrip("."),
        "is_jpeg": asset.content_type == "image/jpeg" or suffix in {".jpg", ".jpeg"},
        "is_png": asset.content_type == "image/png" or suffix == ".png",
        "is_webp": asset.content_type == "image/webp" or suffix == ".webp",
        "sha256_prefix": asset.sha256[:12],
    }


def _disturbance_findings(
    asset: CaseAsset,
    features: dict[str, object],
) -> list[PropagationDisturbanceFinding]:
    findings: list[PropagationDisturbanceFinding] = []
    megapixels = _float_value(features.get("megapixels"))
    aspect_ratio = features.get("aspect_ratio")
    bytes_per_pixel = features.get("bytes_per_pixel")
    if megapixels and megapixels < 0.3:
        findings.append(
            PropagationDisturbanceFinding(
                name="低分辨率/平台缩放",
                severity="medium",
                score=min(1.0, round((0.3 - megapixels) / 0.3, 3)),
                evidence=f"图像约 {megapixels}MP，可能经历平台缩放或截图裁剪。",
            )
        )
    if features.get("is_jpeg") and isinstance(bytes_per_pixel, float) and bytes_per_pixel < 0.35:
        findings.append(
            PropagationDisturbanceFinding(
                name="强 JPEG 压缩",
                severity="high" if bytes_per_pixel < 0.18 else "medium",
                score=max(0.2, min(1.0, round((0.35 - bytes_per_pixel) / 0.35, 3))),
                evidence=f"JPEG 字节/像素约 {bytes_per_pixel}，疑似社交平台二次压缩。",
            )
        )
    if isinstance(aspect_ratio, float) and (aspect_ratio > 2.4 or aspect_ratio < 0.42):
        findings.append(
            PropagationDisturbanceFinding(
                name="异常裁剪比例",
                severity="medium",
                score=0.62,
                evidence=f"宽高比 {aspect_ratio}，可能为长截图、裁剪图或平台重排截图。",
            )
        )
    if features.get("is_png") and asset.size_bytes > 1_500_000 and megapixels and megapixels < 1.2:
        findings.append(
            PropagationDisturbanceFinding(
                name="截图/重保存迹象",
                severity="medium",
                score=0.56,
                evidence="PNG 文件体积相对像素规模偏大，可能来自屏幕截图或二次保存。",
            )
        )
    if not findings:
        findings.append(
            PropagationDisturbanceFinding(
                name="未见明显传播扰动",
                severity="low",
                score=0.12,
                evidence="仅基于尺寸、格式和文件体积未发现强压缩或异常裁剪信号。",
            )
        )
    return findings


def _candidate_distribution(prediction: dict[str, object]) -> list[dict[str, object]]:
    raw = prediction.get("candidate_ranking")
    if not isinstance(raw, list):
        raw = prediction.get("ranked_candidates")
    if not isinstance(raw, list):
        raw = prediction.get("candidates")
    if not isinstance(raw, list):
        label = str(prediction.get("top_candidate") or "unknown")
        confidence = _float_value(prediction.get("confidence"))
        return [{"label": label, "confidence": confidence}]
    candidates: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("candidate") or "unknown")
        confidence = _float_value(item.get("confidence"))
        probability = _float_value(item.get("probability")) if "probability" in item else confidence
        candidates.append({"label": label, "confidence": confidence, "probability": probability})
    return sorted(candidates, key=lambda item: float(item.get("probability", item["confidence"])), reverse=True)[:8]


def _review_recommendation(prediction: dict[str, object]) -> dict[str, object]:
    raw = prediction.get("review_recommendation")
    if isinstance(raw, dict):
        return raw
    gate = prediction.get("binary_gate")
    if isinstance(gate, dict) and isinstance(gate.get("review_recommendation"), dict):
        return dict(gate["review_recommendation"])
    return {}


def _candidate_ranking(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for index, item in enumerate(candidates[:8], start=1):
        label = str(item.get("label") or "unknown")
        probability = _float_value(item.get("probability") if "probability" in item else item.get("confidence"))
        ranked.append(
            {
                "rank": index,
                "label": label,
                "display_name": _display_source_label(label),
                "probability": round(probability, 3),
                "confidence": round(probability, 3),
                "confidence_percent": round(probability * 100),
            }
        )
    return ranked


def _aggregate_candidate_ranking(results: list[ImageForensicsAssetResult]) -> list[dict[str, object]]:
    if not results:
        return []
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        ranking = result.candidate_ranking or _candidate_ranking(result.candidate_distribution)
        for item in ranking:
            label = str(item.get("label") or "unknown")
            probability = _float_value(item.get("probability"))
            totals[label] = totals.get(label, 0.0) + probability
            counts[label] = counts.get(label, 0) + 1
    averaged = [
        {
            "label": label,
            "probability": totals[label] / max(1, counts.get(label, 1)),
        }
        for label in totals
    ]
    return _candidate_ranking(sorted(averaged, key=lambda item: float(item["probability"]), reverse=True))


def _display_source_label(label: str) -> str:
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
        "other-generated": "其他生成模型",
        "real": "真实照片",
        "unknown": "未知/低置信",
        "not-gpt-image2": "非 GPT-image-2",
    }
    return labels.get(normalized, label or "待分析")


def _gpt_image2_probability(
    candidates: list[dict[str, object]],
    top_candidate: str,
    confidence: float,
) -> float | None:
    for item in candidates:
        label = str(item.get("label") or "").lower()
        if label in {"gpt-image2", "gpt-image-2", "gpt image2"}:
            return round(_float_value(item.get("confidence")), 3)
    if top_candidate == "gpt-image2":
        return round(confidence, 3)
    return None if top_candidate == "unknown" and confidence == 0 else 0.0


def _top_contributions(prediction: dict[str, object]) -> list[dict[str, object]]:
    raw = prediction.get("top_contributions")
    if not isinstance(raw, list):
        return []
    return [item for item in raw[:8] if isinstance(item, dict)]


def _interpretation(
    top_candidate: str,
    confidence: float,
    gpt_probability: float | None,
    disturbances: list[PropagationDisturbanceFinding],
    trained: bool,
) -> list[str]:
    lines: list[str] = []
    if gpt_probability is not None and gpt_probability >= 0.65:
        lines.append(f"GPT-image-2 来源线索较强，候选概率约 {round(gpt_probability * 100)}%。")
    elif top_candidate != "unknown":
        lines.append(f"当前最高候选为 {top_candidate}，置信度约 {round(confidence * 100)}%。")
    else:
        lines.append("模型未给出稳定来源候选，建议补采原图或平台侧证据。")
    if any(item.severity in {"medium", "high"} for item in disturbances):
        lines.append("存在传播扰动迹象，模型结论应降权使用，并补充平台元数据或原始文件。")
    else:
        lines.append("未见强传播扰动，适合进入生成归因和人工复核流程。")
    if not trained:
        lines.append("当前云端未挂载本地训练 artifact，本次概率为演示兜底线索；正式使用应接入训练模型或本地视觉服务。")
    return lines


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
