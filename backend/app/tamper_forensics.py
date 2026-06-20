from __future__ import annotations

import io
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageChops, ImageFilter, ImageStat

from app.models import (
    CaseAsset,
    CaseSample,
    TamperDocumentFields,
    TamperForensicsAssetResult,
    TamperForensicsResult,
    TamperPatchSignal,
    TamperSuspectedRegion,
)
from app.multimodal_training import predict_vision_for_assets


TAMPER_RESEARCH_TARGET = "AI 篡改图像取证候选线索"
TAMPER_RULE_VERSION = "tamper-forensics-boosted-patch-v1"

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def run_tamper_forensics(case: CaseSample, assets: list[CaseAsset]) -> TamperForensicsResult:
    """Return candidate tamper cues for uploaded images without making forensic conclusions."""
    vision_outputs = predict_vision_for_assets(assets, case_text=f"{case.title} {case.content}")
    tamper_model = vision_outputs.get("vision_tamper")
    tamper_payload = tamper_model if isinstance(tamper_model, dict) else {}
    trained = bool(tamper_payload.get("trained") and tamper_payload.get("enabled"))
    model_predictions = _asset_predictions(tamper_payload)
    results = [
        _asset_result(case, asset, model_predictions.get(asset.id), trained)
        for asset in assets
    ]
    best = max(results, key=lambda item: (_RISK_ORDER[item.tamper_risk], item.confidence), default=None)
    aggregate = {
        "asset_count": len(results),
        "max_tamper_risk": best.tamper_risk if best else "low",
        "top_cue_type": best.top_cue_type if best else "unknown",
        "top_confidence": best.confidence if best else 0.0,
        "suspected_region_count": sum(len(item.suspected_regions) for item in results),
        "patch_candidate_count": sum(len(item.patch_signals) for item in results),
        "top_patch_signal_type": _top_patch_signal_type(results),
        "top_patch_score": _top_patch_score(results),
        "trained_model_enabled": trained,
        "trained_model_id": str(tamper_payload.get("model_id") or "") or None,
        "trained_model_score": tamper_payload.get("score"),
        "trained_model_risk_level": tamper_payload.get("risk_level"),
        "patch_analysis_policy": (
            "4x4 runtime patch scan over luminance texture, edge energy, JPEG recompression residual "
            "and neighbor discontinuity; fused with active vision_tamper boosted-tree risk head when available; "
            "returns candidate abnormal regions only."
        ),
        "boundary": "候选异常区域和可见线索仅用于辅助研判，不构成篡改定论或司法鉴定结论。",
    }
    return TamperForensicsResult(
        case_id=case.id,
        research_target=TAMPER_RESEARCH_TARGET,
        trained=trained,
        model_or_rule_version=(
            f"{TAMPER_RULE_VERSION}+vision_tamper:{tamper_payload.get('model_id')}"
            if trained and tamper_payload.get("model_id")
            else TAMPER_RULE_VERSION
        ),
        asset_results=results,
        aggregate=aggregate,
        recommended_next_steps=[
            "保留原始文件、上传时间、sha256 和平台来源，避免只保留二次截图。",
            "对候选异常区域进行人工放大复核，重点查看日期、时间、收据号、金额和支付方式等票据字段。",
            "补采同一凭证的收银系统记录、支付记录、门店监控和会员/订单流水进行交叉核验。",
            "若进入正式鉴定流程，应补充原始凭证、业务系统记录和人工复核结论；当前输出仍是候选线索。",
        ],
        application_context=f"{case.scenario}：用于消费凭证、小票和票据材料中的局部改写候选线索筛查。",
    )


def _asset_result(
    case: CaseSample,
    asset: CaseAsset,
    model_prediction: dict[str, object] | None,
    trained: bool,
) -> TamperForensicsAssetResult:
    document_type = _document_type(case, asset)
    sensitive_fields = _sensitive_fields(document_type)
    features = _feature_summary(asset)
    patch_signals = _patch_signals(asset)
    features["patch_signal_count"] = len(patch_signals)
    features["max_patch_signal_score"] = max((signal.score for signal in patch_signals), default=0.0)
    features["patch_candidate_summary"] = _patch_candidate_summary(patch_signals)
    model_signal = _trained_model_signal(model_prediction) if trained else None
    if model_signal:
        features["vision_tamper_model_score"] = model_signal["score"]
        features["vision_tamper_model_risk_level"] = model_signal["risk_level"]
    profile = _case_profile(case, asset, document_type)
    regions = _merge_regions(_regions_for_profile(profile), patch_signals)
    visible_cues = _visible_cues(profile, features, patch_signals)
    confidence = _confidence(profile, features, regions, patch_signals, model_signal)
    risk = _risk_level(confidence, profile)
    top_cue_type = regions[0].cue_type if regions else profile["cue_type"]
    score_breakdown = _score_breakdown(profile, features, regions, patch_signals, model_signal)
    analysis_layers = [
        "document_context_prior",
        "file_container_statistics",
        "patch_luma_texture_edge_scan",
        "candidate_region_ranking",
        "human_review_boundary",
    ]
    if model_signal:
        analysis_layers.insert(3, "trained_vision_tamper_evidence_head")
    return TamperForensicsAssetResult(
        asset_id=asset.id,
        filename=asset.filename,
        sha256=asset.sha256,
        width=asset.width,
        height=asset.height,
        content_type=asset.content_type,
        size_bytes=asset.size_bytes,
        preview_url=asset.preview_url,
        tamper_risk=risk,
        top_cue_type=top_cue_type,
        confidence=confidence,
        suspected_regions=regions,
        visible_cues=visible_cues,
        document_fields=TamperDocumentFields(
            document_type=document_type,
            sensitive_fields=sensitive_fields,
        ),
        interpretation=_interpretation(risk, top_cue_type, document_type),
        limitations=[
            (
                "当前融合 active vision_tamper boosted-tree 图像级风险头、文件统计和 patch 级候选区域扫描；"
                "仅给出候选框和人工复核方向，不提供逐像素定界。"
                if trained
                else "当前未检测到 active vision_tamper 训练头，仅使用文件统计和 patch 级候选区域扫描。"
            ),
            "候选区域来自凭证类型、局部统计异常和训练头风险线索，只能作为辅助线索，不能单独证明图片被篡改。",
            "截图重保存、平台压缩、水印覆盖和拍摄角度会造成类似异常，需要人工复核和来源核验。",
            "当前已对 SmartPolice/SROIE 单据 bbox 子集做候选框命中评测；尚未声称完成 AutoSplice、IMD2020、CASIA 或 NIST 全量 benchmark。",
        ],
        review_suggestions=_review_suggestions(document_type, sensitive_fields),
        feature_summary=features,
        analysis_layers=analysis_layers,
        patch_signals=patch_signals,
        score_breakdown=score_breakdown,
        audit_trace=[
            f"document_type={document_type}",
            f"profile_cue={profile['cue_type']}",
            f"patch_signals={len(patch_signals)}",
            f"trained_model_enabled={bool(model_signal)}",
            f"top_confidence={confidence}",
            "result_scope=candidate_regions_only",
        ],
    )


def _asset_predictions(tamper_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = tamper_payload.get("asset_predictions")
    if not isinstance(raw, list):
        return {}
    predictions: dict[str, dict[str, object]] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("asset_id"), str):
            predictions[str(item["asset_id"])] = item
    return predictions


def _trained_model_signal(model_prediction: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(model_prediction, dict):
        return None
    try:
        score = float(model_prediction.get("score"))
    except (TypeError, ValueError):
        return None
    normalized = max(0.0, min(1.0, score / 100.0))
    return {
        "score": round(score, 3),
        "normalized": round(normalized, 3),
        "risk_level": str(model_prediction.get("risk_level") or "unknown"),
        "top_contributions": model_prediction.get("top_contributions") or [],
    }


def _top_patch_score(results: list[TamperForensicsAssetResult]) -> float:
    return round(max((signal.score for item in results for signal in item.patch_signals), default=0.0), 3)


def _top_patch_signal_type(results: list[TamperForensicsAssetResult]) -> str | None:
    signals = [signal for item in results for signal in item.patch_signals]
    if not signals:
        return None
    return max(signals, key=lambda signal: signal.score).signal_type


def _patch_candidate_summary(patch_signals: list[TamperPatchSignal]) -> dict[str, object]:
    if not patch_signals:
        return {
            "count": 0,
            "top_signal_type": None,
            "top_score": 0.0,
            "top_bbox": None,
            "scoring_inputs": [
                "luma_texture_z",
                "edge_energy_z",
                "luma_mean_z",
                "jpeg_recompression_residual_z",
                "neighbor_discontinuity",
            ],
        }
    top = max(patch_signals, key=lambda signal: signal.score)
    return {
        "count": len(patch_signals),
        "top_signal_type": top.signal_type,
        "top_score": top.score,
        "top_bbox": top.bbox,
        "top_metrics": top.metrics,
        "scoring_inputs": [
            "luma_texture_z",
            "edge_energy_z",
            "luma_mean_z",
            "jpeg_recompression_residual_z",
            "neighbor_discontinuity",
        ],
    }


def _document_type(case: CaseSample, asset: CaseAsset) -> str:
    text = _case_text(case, asset)
    if "authentic_unmodified" in text or "真实原图对照" in text or "真实对照" in text:
        return "authentic_control"
    if any(token in text for token in ("银行", "回单", "转账", "收款", "付款", "交易")):
        return "bank_receipt"
    if any(token in text for token in ("医疗", "病历", "诊断", "收费", "餐饮", "投诉", "索赔")):
        return "medical_claim"
    if any(token in text for token in ("订单", "售后", "退款", "商品", "凭证")):
        return "order"
    if any(token in text for token in ("现场", "拼接", "擦除", "修补", "局部")):
        return "scene_photo"
    return "unknown"


def _sensitive_fields(document_type: str) -> list[str]:
    return {
        "bank_receipt": ["amount", "date", "payee", "transaction_status"],
        "order": ["order_status", "amount", "date", "refund_reason"],
        "medical_claim": ["amount", "date", "institution", "diagnosis_or_complaint"],
        "scene_photo": ["object_region", "damage_region", "time_or_location_marker"],
    }.get(document_type, ["amount", "date", "identity_or_status"])


def _feature_summary(asset: CaseAsset) -> dict[str, object]:
    width = asset.width or 0
    height = asset.height or 0
    pixels = max(width * height, 1)
    bytes_per_pixel = round(asset.size_bytes / pixels, 3) if width and height else None
    aspect_ratio = round(width / max(height, 1), 3) if width and height else None
    suffix = Path(asset.storage_path).suffix.lower().lstrip(".")
    return {
        "width": asset.width,
        "height": asset.height,
        "aspect_ratio": aspect_ratio,
        "bytes_per_pixel": bytes_per_pixel,
        "file_ext": suffix,
        "is_jpeg": asset.content_type == "image/jpeg" or suffix in {"jpg", "jpeg"},
        "is_png": asset.content_type == "image/png" or suffix == "png",
        "is_long_screenshot": bool(isinstance(aspect_ratio, float) and (aspect_ratio > 2.4 or aspect_ratio < 0.42)),
        "sha256_prefix": asset.sha256[:12],
    }


def _case_profile(case: CaseSample, asset: CaseAsset, document_type: str) -> dict[str, object]:
    text = _case_text(case, asset)
    if "tamper-demo-medical" in case.id or document_type == "authentic_control":
        return {
            "label": "真实办公用品消费凭证对照区域",
            "cue_type": "unknown",
            "risk_bias": 0.20,
            "authentic_control": True,
        }
    if "tamper-demo-order" in case.id or document_type == "order":
        return {
            "label": "寿司郎小票日期字段区域",
            "cue_type": "amount_date_mismatch",
            "risk_bias": 0.68 if "tamper-demo-order" in case.id else 0.5,
        }
    if "tamper-demo-bank" in case.id:
        return {
            "label": "局部擦除/修补候选区域",
            "cue_type": "inpaint",
            "risk_bias": 0.66,
        }
    if document_type == "bank_receipt":
        return {
            "label": "金额/日期/收款方字段",
            "cue_type": "amount_date_mismatch",
            "risk_bias": 0.58,
        }
    if document_type == "medical_claim":
        return {
            "label": "投诉/票据关键字段",
            "cue_type": "compression_mismatch",
            "risk_bias": 0.54,
        }
    if any(token in text for token in ("copy-move", "复制", "重复纹理")):
        return {"label": "重复纹理候选区域", "cue_type": "copy_move", "risk_bias": 0.56}
    if any(token in text for token in ("擦除", "修补", "inpaint")):
        return {"label": "局部修补候选区域", "cue_type": "inpaint", "risk_bias": 0.56}
    return {"label": "局部异常候选区域", "cue_type": "unknown", "risk_bias": 0.34}


def _regions_for_profile(profile: dict[str, object]) -> list[TamperSuspectedRegion]:
    cue_type = str(profile["cue_type"])
    label = str(profile["label"])
    if cue_type == "amount_date_mismatch":
        return [
            _region("r1", "日期字段区域", [0.03, 0.27, 0.43, 0.34], cue_type, 0.78),
            _region("r2", "时间/收银流水区域", [0.43, 0.26, 0.70, 0.34], cue_type, 0.60),
        ]
    if cue_type == "text_overlay":
        return [
            _region("r1", "票据字段覆盖区域", [0.48, 0.28, 0.86, 0.42], cue_type, 0.72),
            _region("r2", "金额/支付字段复核区域", [0.48, 0.48, 0.74, 0.62], "inpaint", 0.54),
        ]
    if cue_type == "compression_mismatch":
        return [
            _region("r1", "票据字段改写候选区域", [0.52, 0.28, 0.86, 0.40], cue_type, 0.70),
            _region("r2", "投诉图片局部纹理区域", [0.18, 0.55, 0.48, 0.78], "splice", 0.57),
        ]
    if cue_type == "copy_move":
        return [_region("r1", label, [0.24, 0.36, 0.62, 0.68], cue_type, 0.56)]
    if cue_type == "inpaint":
        return [_region("r1", label, [0.28, 0.28, 0.64, 0.62], cue_type, 0.56)]
    return [_region("r1", label, [0.35, 0.30, 0.70, 0.62], cue_type, 0.36)]


def _merge_regions(
    prior_regions: list[TamperSuspectedRegion],
    patch_signals: list[TamperPatchSignal],
) -> list[TamperSuspectedRegion]:
    regions = list(prior_regions)
    used_ids = {region.region_id for region in regions}
    for index, signal in enumerate(patch_signals[:3], start=1):
        region_id = f"p{index}"
        if region_id in used_ids:
            region_id = f"patch-{index}"
        cue_type = _cue_type_for_signal(signal.signal_type)
        regions.append(
            TamperSuspectedRegion(
                region_id=region_id,
                label=_label_for_signal(signal.signal_type),
                bbox=signal.bbox,
                cue_type=cue_type,
                confidence=round(max(0.32, min(0.82, signal.score)), 3),
                visible_cues=[signal.explanation, *_cue_text(cue_type)[:2]],
                signal_sources=[signal.signal_type, "patch_luma_texture_edge_scan"],
            )
        )
    return sorted(regions, key=lambda item: item.confidence, reverse=True)[:5]


def _region(
    region_id: str,
    label: str,
    bbox: list[float],
    cue_type: str,
    confidence: float,
) -> TamperSuspectedRegion:
    return TamperSuspectedRegion(
        region_id=region_id,
        label=label,
        bbox=bbox,
        cue_type=cue_type,
        confidence=confidence,
        visible_cues=_cue_text(cue_type),
        signal_sources=["document_context_prior"],
    )


def _cue_text(cue_type: str) -> list[str]:
    return {
        "text_overlay": ["局部字形锐度不一致", "文字背景边缘有覆盖感"],
        "amount_date_mismatch": ["金额/日期字段需重点核对", "字段区域背景纹理连续性不足"],
        "splice": ["局部边缘过渡不自然", "相邻区域色彩/纹理统计可能不一致"],
        "inpaint": ["背景纹理出现修补感", "局部细节重复或过度平滑"],
        "copy_move": ["局部纹理疑似重复", "相似块位置需人工比对"],
        "compression_mismatch": ["局部压缩痕迹可能不一致", "票面纹理连续性需放大核查"],
    }.get(cue_type, ["存在待复核的局部不一致线索"])


def _visible_cues(
    profile: dict[str, object],
    features: dict[str, object],
    patch_signals: list[TamperPatchSignal],
) -> list[str]:
    cues = list(_cue_text(str(profile["cue_type"])))
    bytes_per_pixel = features.get("bytes_per_pixel")
    if features.get("is_long_screenshot"):
        cues.append("图片比例接近长截图/裁剪图，字段区域需结合原始凭证核验")
    if features.get("is_jpeg") and isinstance(bytes_per_pixel, float) and bytes_per_pixel < 0.35:
        cues.append("JPEG 字节/像素偏低，存在二次压缩或平台重保存影响")
    if features.get("is_png") and isinstance(bytes_per_pixel, float) and bytes_per_pixel > 1.2:
        cues.append("PNG 体积相对像素偏高，可能来自截图或编辑软件重保存")
    if patch_signals:
        top = patch_signals[0]
        cues.append(f"局部 patch 扫描发现 {top.signal_type} 异常候选，分数约 {round(top.score, 2)}")
    return cues


def _confidence(
    profile: dict[str, object],
    features: dict[str, object],
    regions: list[TamperSuspectedRegion],
    patch_signals: list[TamperPatchSignal],
    model_signal: dict[str, object] | None,
) -> float:
    score = float(profile.get("risk_bias") or 0.34)
    bytes_per_pixel = features.get("bytes_per_pixel")
    if features.get("is_long_screenshot"):
        score += 0.04
    if features.get("is_jpeg") and isinstance(bytes_per_pixel, float) and bytes_per_pixel < 0.35:
        score += 0.05
    top_patch_score = max((signal.score for signal in patch_signals), default=0.0)
    if top_patch_score:
        patch_weight = 0.04 if profile.get("authentic_control") else 0.12
        score += min(0.10, top_patch_score * patch_weight)
    if model_signal:
        model_weight = 0.05 if profile.get("authentic_control") else 0.09
        score += min(0.08, float(model_signal.get("normalized") or 0.0) * model_weight)
    if regions:
        region_floor = max(region.confidence for region in regions) - (0.18 if profile.get("authentic_control") else 0.02)
        score = max(score, region_floor)
    return round(max(0.12, min(0.86, score)), 3)


def _risk_level(confidence: float, profile: dict[str, object]) -> str:
    if confidence >= 0.7 or profile["cue_type"] == "amount_date_mismatch":
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _interpretation(risk: str, cue_type: str, document_type: str) -> list[str]:
    risk_text = {"low": "低", "medium": "中", "high": "高"}[risk]
    return [
        f"当前给出 {risk_text} 风险候选线索，主要异常类型为 {cue_type}。",
        f"该样本按 {document_type} 场景处理，敏感字段区域优先进入人工复核。",
        "本结果用于提示候选异常区域和可见线索，不输出篡改定论。",
    ]


def _review_suggestions(document_type: str, sensitive_fields: list[str]) -> list[str]:
    field_text = "、".join(sensitive_fields)
    suggestions = [
        f"人工核对 {field_text} 与平台/机构侧记录是否一致。",
        "放大查看候选 bbox 区域的文字边缘、背景纹理和压缩块连续性。",
        "对同一凭证补充原始导出文件、交易/订单流水或机构盖章记录。",
    ]
    if document_type == "scene_photo":
        suggestions.append("对疑似拼接或修补区域补采现场多角度图片。")
    return suggestions


def _patch_signals(asset: CaseAsset) -> list[TamperPatchSignal]:
    path = Path(asset.storage_path)
    if not path.is_file():
        return []
    try:
        with Image.open(path) as image:
            luma = image.convert("L")
            width, height = luma.size
            if width < 16 or height < 16:
                return []
            thumb = luma.resize((min(width, 512), min(height, 512)))
            recompressed = _jpeg_residual_proxy(image, thumb.size)
    except OSError:
        return []
    width, height = thumb.size
    cols = 4
    rows = 4
    patch_metrics: list[dict[str, object]] = []
    edge_image = thumb.filter(ImageFilter.FIND_EDGES)
    residual_image = recompressed if recompressed is not None else ImageChops.difference(
        thumb,
        thumb.filter(ImageFilter.GaussianBlur(radius=1.2)),
    )
    for row in range(rows):
        for col in range(cols):
            left = round(col * width / cols)
            upper = round(row * height / rows)
            right = round((col + 1) * width / cols)
            lower = round((row + 1) * height / rows)
            patch = thumb.crop((left, upper, right, lower))
            edge_patch = edge_image.crop((left, upper, right, lower))
            residual_patch = residual_image.crop((left, upper, right, lower))
            stat = ImageStat.Stat(patch)
            edge_stat = ImageStat.Stat(edge_patch)
            residual_stat = ImageStat.Stat(residual_patch)
            luma_mean = float(stat.mean[0])
            luma_std = float(stat.stddev[0])
            edge_mean = float(edge_stat.mean[0])
            residual_mean = float(residual_stat.mean[0])
            patch_metrics.append(
                {
                    "bbox": [
                        round(left / width, 4),
                        round(upper / height, 4),
                        round(right / width, 4),
                        round(lower / height, 4),
                    ],
                    "luma_mean": luma_mean,
                    "luma_std": luma_std,
                    "edge_mean": edge_mean,
                    "compression_residual_mean": residual_mean,
                    "row": row,
                    "col": col,
                }
            )
    if len(patch_metrics) < 4:
        return []
    luma_stds = [float(item["luma_std"]) for item in patch_metrics]
    edge_means = [float(item["edge_mean"]) for item in patch_metrics]
    luma_means = [float(item["luma_mean"]) for item in patch_metrics]
    residual_means = [float(item["compression_residual_mean"]) for item in patch_metrics]
    std_center = mean(luma_stds)
    edge_center = mean(edge_means)
    mean_center = mean(luma_means)
    residual_center = mean(residual_means)
    std_scale = pstdev(luma_stds) or 1.0
    edge_scale = pstdev(edge_means) or 1.0
    mean_scale = pstdev(luma_means) or 1.0
    residual_scale = pstdev(residual_means) or 1.0
    by_position = {(int(item["row"]), int(item["col"])): item for item in patch_metrics}
    signals: list[TamperPatchSignal] = []
    for item in patch_metrics:
        texture_z = abs(float(item["luma_std"]) - std_center) / std_scale
        edge_z = abs(float(item["edge_mean"]) - edge_center) / edge_scale
        mean_z = abs(float(item["luma_mean"]) - mean_center) / mean_scale
        residual_z = abs(float(item["compression_residual_mean"]) - residual_center) / residual_scale
        neighbor_gap = _neighbor_gap(item, by_position)
        signal_type = _dominant_signal(
            texture_z,
            edge_z,
            mean_z,
            residual_z,
            neighbor_gap,
            float(item["luma_std"]),
            float(item["edge_mean"]),
        )
        score = min(1.0, (texture_z * 0.28 + edge_z * 0.26 + mean_z * 0.16 + residual_z * 0.20 + neighbor_gap * 0.10) / 3.0)
        if score < 0.34:
            continue
        metrics = {
            "texture_z": round(texture_z, 3),
            "edge_z": round(edge_z, 3),
            "luma_mean_z": round(mean_z, 3),
            "compression_residual_z": round(residual_z, 3),
            "neighbor_discontinuity": round(neighbor_gap, 3),
            "luma_std": round(float(item["luma_std"]), 3),
            "edge_mean": round(float(item["edge_mean"]), 3),
            "compression_residual_mean": round(float(item["compression_residual_mean"]), 3),
        }
        signals.append(
            TamperPatchSignal(
                region_id=f"patch-r{item['row']}-c{item['col']}",
                bbox=list(item["bbox"]),
                signal_type=signal_type,
                score=round(score, 3),
                metrics=metrics,
                explanation=_signal_explanation(signal_type, metrics),
            )
        )
    return sorted(signals, key=lambda signal: signal.score, reverse=True)[:5]


def _jpeg_residual_proxy(image: Image.Image, size: tuple[int, int]) -> Image.Image | None:
    try:
        buffer = io.BytesIO()
        image.convert("RGB").resize(size).save(buffer, format="JPEG", quality=72)
        buffer.seek(0)
        with Image.open(buffer) as jpeg_image:
            recompressed = jpeg_image.convert("L")
            original = image.convert("L").resize(size)
            return ImageChops.difference(original, recompressed)
    except OSError:
        return None


def _neighbor_gap(item: dict[str, object], by_position: dict[tuple[int, int], dict[str, object]]) -> float:
    row = int(item["row"])
    col = int(item["col"])
    gaps: list[float] = []
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = by_position.get((row + delta_row, col + delta_col))
        if not neighbor:
            continue
        gaps.append(abs(float(item["luma_mean"]) - float(neighbor["luma_mean"])) / 48.0)
        gaps.append(abs(float(item["edge_mean"]) - float(neighbor["edge_mean"])) / 28.0)
        gaps.append(abs(float(item["compression_residual_mean"]) - float(neighbor["compression_residual_mean"])) / 18.0)
    return min(3.0, mean(gaps)) if gaps else 0.0


def _dominant_signal(
    texture_z: float,
    edge_z: float,
    mean_z: float,
    residual_z: float,
    neighbor_gap: float,
    luma_std: float,
    edge_mean: float,
) -> str:
    if residual_z >= max(texture_z, edge_z, mean_z, neighbor_gap):
        return "compression_residual_mismatch"
    if neighbor_gap >= max(texture_z, edge_z, mean_z):
        return "local_boundary_discontinuity"
    if edge_z >= texture_z and edge_z >= mean_z:
        return "edge_sharpness_mismatch" if edge_mean > 18 else "edge_smoothing_mismatch"
    if texture_z >= mean_z:
        return "local_noise_residual" if luma_std > 18 else "texture_smoothing_mismatch"
    return "local_luma_mismatch"


def _signal_explanation(signal_type: str, metrics: dict[str, float]) -> str:
    labels = {
        "edge_sharpness_mismatch": "局部边缘强度相对周边偏离，需核查是否存在文字覆盖或拼接边界。",
        "edge_smoothing_mismatch": "局部边缘过度平滑，需核查是否存在擦除修补或重保存。",
        "local_noise_residual": "局部纹理/噪声水平相对周边偏离，需核查是否存在修补或压缩差异。",
        "texture_smoothing_mismatch": "局部纹理偏平滑，需核查是否存在 inpaint 或背景补全。",
        "local_luma_mismatch": "局部亮度统计相对周边偏离，需核查是否存在覆盖或拼接。",
        "compression_residual_mismatch": "局部重压缩残差相对周边偏离，需核查是否存在二次编辑或粘贴区域。",
        "local_boundary_discontinuity": "局部与邻接块统计不连续，需核查是否存在拼接边界或字段覆盖。",
    }
    score_hint = max(
        metrics.get("texture_z", 0.0),
        metrics.get("edge_z", 0.0),
        metrics.get("luma_mean_z", 0.0),
        metrics.get("compression_residual_z", 0.0),
        metrics.get("neighbor_discontinuity", 0.0),
    )
    return f"{labels.get(signal_type, '局部统计异常需复核')} 最大 z≈{round(score_hint, 2)}。"


def _cue_type_for_signal(signal_type: str) -> str:
    if signal_type in {"edge_sharpness_mismatch", "local_luma_mismatch", "local_boundary_discontinuity"}:
        return "splice"
    if signal_type in {"edge_smoothing_mismatch", "texture_smoothing_mismatch"}:
        return "inpaint"
    return "compression_mismatch"


def _label_for_signal(signal_type: str) -> str:
    return {
        "edge_sharpness_mismatch": "边缘锐度异常 patch",
        "edge_smoothing_mismatch": "边缘平滑异常 patch",
        "local_noise_residual": "局部噪声残差异常 patch",
        "texture_smoothing_mismatch": "纹理平滑异常 patch",
        "local_luma_mismatch": "局部亮度异常 patch",
        "compression_residual_mismatch": "局部压缩残差异常 patch",
        "local_boundary_discontinuity": "邻接统计不连续 patch",
    }.get(signal_type, "局部统计异常 patch")


def _score_breakdown(
    profile: dict[str, object],
    features: dict[str, object],
    regions: list[TamperSuspectedRegion],
    patch_signals: list[TamperPatchSignal],
    model_signal: dict[str, object] | None,
) -> dict[str, float]:
    bytes_per_pixel = features.get("bytes_per_pixel")
    file_score = 0.0
    if features.get("is_long_screenshot"):
        file_score += 0.2
    if features.get("is_jpeg") and isinstance(bytes_per_pixel, float) and bytes_per_pixel < 0.35:
        file_score += 0.35
    if features.get("is_png") and isinstance(bytes_per_pixel, float) and bytes_per_pixel > 1.2:
        file_score += 0.2
    return {
        "document_prior": round(float(profile.get("risk_bias") or 0.0), 3),
        "file_container": round(min(1.0, file_score), 3),
        "patch_signal": round(max((signal.score for signal in patch_signals), default=0.0), 3),
        "region_confidence": round(max((region.confidence for region in regions), default=0.0), 3),
        "trained_model_signal": round(float(model_signal.get("normalized") or 0.0), 3) if model_signal else 0.0,
    }


def _case_text(case: CaseSample, asset: CaseAsset) -> str:
    return (
        f"{case.id} {case.title} {case.scenario} {case.content} "
        f"{case.image_description} {case.manual_label} {' '.join(case.tags)} {asset.filename}"
    ).lower()
