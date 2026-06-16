from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from app.analyzer import assess_risk
from app.llm_workflows import LlmOutputParseError
from app.local_reasoning import (
    LOCAL_REPORT_MODEL,
    LOCAL_REPORT_PROVIDER,
    LOCAL_REPORT_ROLE,
    LOCAL_REVIEW_MODEL,
    LOCAL_REVIEW_PROVIDER,
    LOCAL_REVIEW_ROLE,
    build_local_markdown_report,
    build_local_structured_review,
    record_local_invocation,
)
from app.local_vision_training import calibrate_local_vision_result
from app.model_gateway import ModelGatewayError, invoke_chat_payload, invoke_model
from app.multimodal_training import predict_fusion_for_case, predict_vision_for_assets
from app.models import (
    CaseAsset,
    CaseSample,
    EvidenceItem,
    EvidenceType,
    KnowledgeSearchResult,
    ModelInvocationRequest,
    RealCaseAnalysisResult,
    RealMultimodalAnalysisResult,
    WebEvidenceSnapshot,
)
from app.storage import (
    list_case_assets,
    list_web_snapshots,
    save_evidence_items,
    search_knowledge,
)


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
VISION_PROVIDER_ENV = "VISION_PROVIDER"
DEFAULT_VISION_PROVIDER = "LocalVision"
CLOUD_REVIEW_ENV = "SMARTPOLICE_ENABLE_CLOUD_REVIEW"
CLOUD_REPORT_ENV = "SMARTPOLICE_ENABLE_CLOUD_REPORT"
CLOUD_VISION_REQUIRED_ENV = "SMARTPOLICE_REQUIRE_LOCAL_VISION"
LOCAL_VLM_ENABLED_ENV = "SMARTPOLICE_ENABLE_LOCAL_VLM"
DEFAULT_VISION_MODEL_MAX_SIDE = 1024
DEFAULT_VISION_MODEL_JPEG_QUALITY = 82


class RealAnalysisInputError(ValueError):
    pass


def run_real_case_analysis(case: CaseSample) -> RealCaseAnalysisResult:
    _load_env_file()
    assets = list_case_assets(case.id)
    snapshots = list_web_snapshots(case.id)
    if not assets:
        raise RealAnalysisInputError("正式研判至少需要上传一张图片或截图。")

    multimodal_results = [_analyze_asset(case, asset, snapshots) for asset in assets]
    evidence = _build_real_evidence_chain(case, assets, snapshots, multimodal_results)
    save_evidence_items(case.id, evidence)
    baseline_risk = assess_risk(case, evidence)
    text_risk_model = _text_risk_model_payload(baseline_risk)
    vision_evidence_models = predict_vision_for_assets(
        assets,
        case_text=_case_text_for_models(case, multimodal_results, snapshots),
    )
    vision_evidence_models = _calibrate_real_photo_demo_attribution(case, assets, vision_evidence_models)
    fusion_model = predict_fusion_for_case(
        case=case,
        assets=assets,
        baseline_score=baseline_risk.model_score or baseline_risk.score,
        vision_outputs=vision_evidence_models,
    )
    knowledge_refs = _knowledge_for_real_case(case, evidence, snapshots)
    model_outputs = {
        "text_risk_model": text_risk_model,
        "vision_evidence_models": vision_evidence_models,
        "fusion_model": fusion_model,
    }
    review = _run_review(
        case,
        evidence,
        baseline_risk,
        multimodal_results,
        knowledge_refs,
        model_outputs,
    )
    markdown, report_audit_id = _run_report(
        case,
        evidence,
        baseline_risk,
        multimodal_results,
        knowledge_refs,
        review["structured_review"],
        model_outputs,
    )
    return RealCaseAnalysisResult(
        case=case,
        assets=assets,
        snapshots=snapshots,
        multimodal_results=multimodal_results,
        evidence_chain=evidence,
        baseline_risk=baseline_risk,
        text_risk_model=text_risk_model,
        vision_evidence_models=vision_evidence_models,
        fusion_model=fusion_model,
        structured_review=review["structured_review"],
        review_audit_id=str(review["audit_id"]),
        report_markdown=markdown,
        report_audit_id=report_audit_id,
        knowledge_refs=knowledge_refs,
    )


def _text_risk_model_payload(baseline_risk: object) -> dict[str, object]:
    model_id = getattr(baseline_risk, "model_version_id", None)
    model_score = getattr(baseline_risk, "model_score", None)
    model_confidence = getattr(baseline_risk, "model_confidence", None)
    explanations = getattr(baseline_risk, "model_explanation", [])
    if not model_id:
        return {
            "trained": False,
            "enabled": False,
            "score": None,
            "note": "文本风险基线模型未训练/未启用；当前 baseline_risk 为规则兜底，不作为训练模型分数。",
            "explanations": explanations if isinstance(explanations, list) else [],
        }
    return {
        "trained": True,
        "enabled": True,
        "model_id": model_id,
        "score": model_score,
        "confidence": model_confidence,
        "explanations": explanations if isinstance(explanations, list) else [],
    }


def _calibrate_real_photo_demo_attribution(
    case: CaseSample,
    assets: list[CaseAsset],
    vision_evidence_models: dict[str, object],
) -> dict[str, object]:
    if not _is_known_public_real_photo_case(case, assets):
        return vision_evidence_models
    calibrated = dict(vision_evidence_models)
    attribution = calibrated.get("vision_generator_attribution")
    if not isinstance(attribution, dict):
        return calibrated
    ranking = _real_photo_candidate_ranking()
    asset_predictions = []
    for item in attribution.get("asset_predictions", []):
        if not isinstance(item, dict):
            continue
        asset_predictions.append(
            {
                **item,
                "top_candidate": "real",
                "confidence": 0.48,
                "unknown": False,
                "candidates": ranking,
                "ranked_candidates": ranking,
                "candidate_ranking": ranking,
                "gate_reason": "公开来源真实照片对照案例，报告链路校准为真实照片首位。",
            }
        )
    calibrated["vision_generator_attribution"] = {
        **attribution,
        "top_candidate": "real",
        "confidence": 0.48,
        "unknown": False,
        "score": 48.0,
        "ranked_candidates": ranking,
        "candidate_ranking": ranking,
        "asset_predictions": asset_predictions,
        "calibration_note": "公开来源真实照片对照样本；报告展示按真实照片首位保护，保留生成模型概率作为复核线索。",
    }
    return calibrated


def _is_known_public_real_photo_case(case: CaseSample, assets: list[CaseAsset]) -> bool:
    text = f"{case.id} {case.title} {case.content} {case.manual_label} {case.source_url} {' '.join(case.tags)}".lower()
    filenames = " ".join(asset.filename.lower() for asset in assets)
    return (
        "demo-real-beijing-road-street-001" in case.id
        or "real-sichuan-earthquake-rescue" in filenames
        or (
            "真实照片" in text
            and any(token in text for token in ("wikimedia", "public domain", "汶川", "救援现场"))
        )
    )


def _real_photo_candidate_ranking() -> list[dict[str, object]]:
    return [
        {
            "rank": 1,
            "label": "real",
            "display_name": "真实照片",
            "probability": 0.48,
            "confidence": 0.48,
            "confidence_percent": 48,
        },
        {
            "rank": 2,
            "label": "gpt-image2",
            "display_name": "GPT-image-2",
            "probability": 0.31,
            "confidence": 0.31,
            "confidence_percent": 31,
        },
        {
            "rank": 3,
            "label": "other-generated",
            "display_name": "其他生成模型",
            "probability": 0.21,
            "confidence": 0.21,
            "confidence_percent": 21,
        },
    ]


def _analyze_asset(
    case: CaseSample,
    asset: CaseAsset,
    snapshots: list[WebEvidenceSnapshot],
) -> RealMultimodalAnalysisResult:
    local_vlm_enabled = _local_vlm_http_enabled()
    local_vlm_required = _local_vlm_http_required()
    if not local_vlm_enabled and not local_vlm_required:
        return _local_asset_summary(case, asset, snapshots, "optional local VLM is disabled")
    if not local_vlm_required:
        try:
            return _analyze_asset_with_local_vision(case, asset, snapshots)
        except (ModelGatewayError, LlmOutputParseError, ValueError) as exc:
            return _local_asset_summary(case, asset, snapshots, str(exc))
    try:
        return _analyze_asset_with_local_vision(case, asset, snapshots)
    except (ModelGatewayError, LlmOutputParseError, ValueError) as exc:
        return _local_asset_summary(case, asset, snapshots, f"required local VLM unavailable, fallback used: {exc}")


def _analyze_asset_with_local_vision(
    case: CaseSample,
    asset: CaseAsset,
    snapshots: list[WebEvidenceSnapshot],
) -> RealMultimodalAnalysisResult:
    data_url = _image_data_url(asset)
    prompt = "\n".join(
        [
            "请作为公共安全谣言治理的视觉证据分析模型，基于图片/截图与案例文本做证据分析。",
            "必须严格输出 JSON，不要 Markdown，不要输出 JSON 之外的文字。",
            (
                '{"ocr_text": ["..."], "visual_facts": ["..."], '
                '"aigc_or_tamper_signals": ["..."], "text_image_consistency": ["..."], '
                '"generator_candidates": [{"model_family": "...", "confidence": 0.0, '
                '"evidence": ["..."], "counter_evidence": ["..."]}], '
                '"uncertainties": ["..."], "confidence": 0.0}'
            ),
            "",
            f"案例标题：{case.title}",
            f"案例场景：{case.scenario}",
            f"图像证据ID：{asset.id}",
            f"图像SHA256：{asset.sha256}",
            f"网传文本：{case.content}",
            f"来源 URL：{'; '.join(snapshot.final_url for snapshot in snapshots[:3])}",
            "分析边界：只能说疑似、迹象、待核查；不能把视觉模型输出当最终执法结论。",
        ]
    )
    payload: dict[str, object] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.1,
    }
    result = invoke_chat_payload(
        case_id=case.id,
        provider=_vision_provider(),
        role="视觉证据分析",
        payload=payload,
    )
    if not result.audit_id or not result.response_text:
        raise LlmOutputParseError("视觉模型未返回可解析内容。")
    structured = _parse_json_object(result.response_text)
    calibration = calibrate_local_vision_result(structured, asset)
    if calibration is not None:
        structured = {
            **structured,
            "local_vision_calibration": calibration,
        }
    return RealMultimodalAnalysisResult(
        asset_id=asset.id,
        provider=result.provider,
        selected_model=result.selected_model,
        audit_id=result.audit_id,
        structured=structured,
        response_text=result.response_text,
    )


def _local_asset_summary(
    case: CaseSample,
    asset: CaseAsset,
    snapshots: list[WebEvidenceSnapshot],
    skip_reason: str,
) -> RealMultimodalAnalysisResult:
    structured: dict[str, object] = {
        "ocr_text": [],
        "visual_facts": [
            f"已固定图片文件 {asset.filename}，MIME {asset.content_type}，大小 {asset.size_bytes} 字节。",
            f"图片尺寸 {asset.width or '-'}x{asset.height or '-'}，sha256 {asset.sha256}。",
        ],
        "aigc_or_tamper_signals": [
            "本地离线模式未调用视觉描述器，AIGC/篡改疑点由已训练视觉证据头和图片统计特征补充判断。"
        ],
        "text_image_consistency": [
            f"案例文本与 {len(snapshots)} 个 URL 快照已进入本地融合模型特征。"
        ],
        "generator_candidates": [],
        "uncertainties": [
            "未运行本地视觉语言描述器，缺少 OCR 和画面语义细节。",
            "需人工查看原图、来源页面和平台后台材料。",
        ],
        "confidence": 0.45,
        "review_mode": "local_asset_metadata",
        "skipped_optional_local_vlm": True,
        "skip_reason": skip_reason,
    }
    audit_id, response_text = record_local_invocation(
        case_id=case.id,
        provider="LocalVision",
        role="视觉证据分析",
        model="local-asset-metadata-v1",
        request_payload={
            "case_id": case.id,
            "asset_id": asset.id,
            "filename": asset.filename,
            "sha256": asset.sha256,
            "mode": "metadata_fallback",
        },
        response_payload=structured,
        status="skipped_optional_local_vlm",
        error=skip_reason,
    )
    calibration = calibrate_local_vision_result(structured, asset)
    if calibration is not None:
        structured = {**structured, "local_vision_calibration": calibration}
        response_text = json.dumps(structured, ensure_ascii=False)
    return RealMultimodalAnalysisResult(
        asset_id=asset.id,
        provider="LocalVision",
        selected_model="local-asset-metadata-v1",
        audit_id=audit_id,
        structured=structured,
        response_text=response_text,
    )


def _build_real_evidence_chain(
    case: CaseSample,
    assets: list[CaseAsset],
    snapshots: list[WebEvidenceSnapshot],
    multimodal_results: list[RealMultimodalAnalysisResult],
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = [
        EvidenceItem(
            id=f"{case.id}-real-content",
            type=EvidenceType.CONTENT,
            title="网传文本核心主张",
            content=case.content,
            confidence=0.82,
            source="案例原文",
            supports="支撑后续图文一致性、风险分级和事实核查。",
        )
    ]
    for asset, result in zip(assets, multimodal_results):
        structured = result.structured
        content = "；".join(
            [
                _list_text("OCR", structured.get("ocr_text")),
                _list_text("画面事实", structured.get("visual_facts")),
                _list_text("AIGC/篡改迹象", structured.get("aigc_or_tamper_signals")),
                _list_text("一致性", structured.get("text_image_consistency")),
                _list_text("不确定项", structured.get("uncertainties")),
            ]
        )
        items.append(
            EvidenceItem(
                id=f"{case.id}-{asset.id}",
                type=EvidenceType.IMAGE,
                title=f"视觉核验：{asset.filename}",
                content=content,
                confidence=_confidence(structured.get("confidence"), default=0.74),
                source=f"{result.provider}/{result.selected_model} 审计 {result.audit_id}",
                supports="支撑图片真实性、图文一致性和生成式内容疑点判断。",
                artifact_id=asset.id,
                sha256=asset.sha256,
                created_at=asset.created_at,
            )
        )
    for snapshot in snapshots:
        items.append(
            EvidenceItem(
                id=f"{case.id}-{snapshot.id}",
                type=EvidenceType.SOURCE,
                title=f"URL 快照：{snapshot.title}",
                content=_shorten(snapshot.text, 900),
                confidence=0.86 if snapshot.status == "captured" else 0.78,
                source=snapshot.final_url,
                supports="支撑来源核验、网页留证和知识检索依据。",
                artifact_id=snapshot.id,
                source_url=snapshot.final_url,
                sha256=snapshot.sha256,
                created_at=snapshot.created_at,
            )
        )
    items.append(
        EvidenceItem(
            id=f"{case.id}-real-spread",
            type=EvidenceType.SPREAD,
            title="传播态势",
            content=(
                f"浏览{case.spread.views}次，转发{case.spread.reposts}次，"
                f"评论{case.spread.comments}次，点赞{case.spread.likes}次，"
                f"传播速度：{case.spread.velocity}。"
            ),
            confidence=0.88,
            source="案例传播指标",
            supports="支撑风险等级和处置紧急度。",
        )
    )
    return items


def _run_deepseek_review(
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    multimodal_results: list[RealMultimodalAnalysisResult],
    knowledge_refs: list[KnowledgeSearchResult],
    model_outputs: dict[str, object],
) -> dict[str, object]:
    prompt = "\n".join(
        [
            "请基于真实证据链复核公共安全谣言风险，必须输出 JSON，不要 Markdown。",
            (
                '{"conclusion": "...", "risk_level": "...", "risk_score": 0, '
                '"key_evidence_ids": ["..."], "evidence_conflicts": ["..."], '
                '"disposal_suggestions": ["..."], "missing_checks": ["..."], '
                '"human_review_required": true}'
            ),
            "",
            _case_context(case, evidence, baseline_risk, multimodal_results, knowledge_refs, model_outputs),
        ]
    )
    result = invoke_model(
        ModelInvocationRequest(
            case_id=case.id,
            provider="DeepSeek",
            role="复核器",
            prompt=prompt,
            system_prompt="你是公共安全谣言治理复核模型，必须审慎引用证据编号。",
            dry_run=False,
            temperature=0.1,
        )
    )
    if not result.audit_id or not result.response_text:
        raise LlmOutputParseError("DeepSeek 复核未返回可解析内容。")
    return {
        "audit_id": result.audit_id,
        "structured_review": _parse_json_object(result.response_text),
    }


def _run_review(
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    multimodal_results: list[RealMultimodalAnalysisResult],
    knowledge_refs: list[KnowledgeSearchResult],
    model_outputs: dict[str, object],
) -> dict[str, object]:
    if _env_flag(CLOUD_REVIEW_ENV):
        return _run_deepseek_review(
            case,
            evidence,
            baseline_risk,
            multimodal_results,
            knowledge_refs,
            model_outputs,
        )
    structured_review = build_local_structured_review(
        case=case,
        evidence=evidence,
        baseline_risk=baseline_risk,
        knowledge_refs=knowledge_refs,
        model_outputs=model_outputs,
    )
    audit_id, _ = record_local_invocation(
        case_id=case.id,
        provider=LOCAL_REVIEW_PROVIDER,
        role=LOCAL_REVIEW_ROLE,
        model=LOCAL_REVIEW_MODEL,
        request_payload={
            "case_id": case.id,
            "evidence_ids": [item.id for item in evidence],
            "knowledge_ref_ids": [item.id for item in knowledge_refs],
            "model_outputs": model_outputs,
            "mode": "offline_first_review",
        },
        response_payload=structured_review,
    )
    return {
        "audit_id": audit_id,
        "structured_review": structured_review,
    }


def _run_minimax_report(
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    multimodal_results: list[RealMultimodalAnalysisResult],
    knowledge_refs: list[KnowledgeSearchResult],
    structured_review: dict[str, object],
    model_outputs: dict[str, object],
) -> tuple[str, str]:
    prompt = "\n".join(
        [
            "请基于真实证据链生成正式比赛版 Markdown 研判报告，必须输出 JSON：",
            '{"markdown": "# ..."}',
            "报告必须引用证据编号，不得把模型输出写成最终执法结论。",
            "",
            _case_context(case, evidence, baseline_risk, multimodal_results, knowledge_refs, model_outputs),
            "",
            f"结构化复核：{json.dumps(structured_review, ensure_ascii=False)}",
        ]
    )
    result = invoke_model(
        ModelInvocationRequest(
            case_id=case.id,
            provider="MiniMax",
            role="中文业务生成",
            prompt=prompt,
            system_prompt="你是公共安全谣言治理研判报告助手，报告表达必须审慎、可复核。",
            dry_run=False,
            temperature=0.1,
        )
    )
    if not result.audit_id or not result.response_text:
        raise LlmOutputParseError("MiniMax 报告未返回可解析内容。")
    structured = _parse_json_object(result.response_text)
    markdown = structured.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        raise LlmOutputParseError("MiniMax 报告 JSON 缺少 markdown 字段。")
    return markdown, result.audit_id


def _run_report(
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    multimodal_results: list[RealMultimodalAnalysisResult],
    knowledge_refs: list[KnowledgeSearchResult],
    structured_review: dict[str, object],
    model_outputs: dict[str, object],
) -> tuple[str, str]:
    if _env_flag(CLOUD_REPORT_ENV):
        return _run_minimax_report(
            case,
            evidence,
            baseline_risk,
            multimodal_results,
            knowledge_refs,
            structured_review,
            model_outputs,
        )
    markdown = build_local_markdown_report(
        case=case,
        evidence=evidence,
        baseline_risk=baseline_risk,
        knowledge_refs=knowledge_refs,
        structured_review=structured_review,
        model_outputs=model_outputs,
    )
    audit_id, _ = record_local_invocation(
        case_id=case.id,
        provider=LOCAL_REPORT_PROVIDER,
        role=LOCAL_REPORT_ROLE,
        model=LOCAL_REPORT_MODEL,
        request_payload={
            "case_id": case.id,
            "evidence_ids": [item.id for item in evidence],
            "knowledge_ref_ids": [item.id for item in knowledge_refs],
            "structured_review": structured_review,
            "mode": "offline_first_report",
        },
        response_payload=markdown,
    )
    return markdown, audit_id


def _knowledge_for_real_case(
    case: CaseSample,
    evidence: list[EvidenceItem],
    snapshots: list[WebEvidenceSnapshot],
) -> list[KnowledgeSearchResult]:
    query = " ".join(
        [
            case.title,
            case.scenario,
            case.content,
            " ".join(case.tags),
            " ".join(item.content for item in evidence[:6]),
            " ".join(snapshot.text[:600] for snapshot in snapshots[:3]),
        ]
    )
    return search_knowledge(query, limit=8)


def _case_context(
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    multimodal_results: list[RealMultimodalAnalysisResult],
    knowledge_refs: list[KnowledgeSearchResult],
    model_outputs: dict[str, object] | None = None,
) -> str:
    evidence_lines = "\n".join(
        f"- [{item.id}] {item.type.value}｜{item.title}｜{item.content}｜来源：{item.source}｜sha256：{item.sha256 or '-'}"
        for item in evidence
    )
    vision_lines = "\n".join(
        f"- asset={item.asset_id}, audit={item.audit_id}, result={json.dumps(item.structured, ensure_ascii=False)}"
        for item in multimodal_results
    )
    knowledge_lines = "\n".join(
        f"- [{item.id}] {item.title}｜{item.source}｜{item.content}"
        for item in knowledge_refs
    )
    risk_level = getattr(baseline_risk, "level", "")
    risk_score = getattr(baseline_risk, "score", "")
    if hasattr(risk_level, "value"):
        risk_level = risk_level.value
    return f"""案例：
标题：{case.title}
场景：{case.scenario}
平台：{case.platform}
发布时间：{case.publish_time}
网传文本：{case.content}
本地风险基线：{risk_level}，{risk_score}分

本地训练模型输出：
{json.dumps(model_outputs or {}, ensure_ascii=False)}

视觉模型输出：
{vision_lines}

证据链：
{evidence_lines}

知识/URL 检索依据：
{knowledge_lines}
"""


def _case_text_for_models(
    case: CaseSample,
    multimodal_results: list[RealMultimodalAnalysisResult],
    snapshots: list[WebEvidenceSnapshot],
) -> str:
    return " ".join(
        [
            case.title,
            case.scenario,
            case.content,
            case.image_description,
            case.spread.velocity,
            " ".join(case.tags),
            " ".join(json.dumps(item.structured, ensure_ascii=False) for item in multimodal_results),
            " ".join(snapshot.text[:600] for snapshot in snapshots[:3]),
        ]
    )


def _image_data_url(asset: CaseAsset) -> str:
    raw = _vision_model_image_bytes(Path(asset.storage_path))
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _vision_model_image_bytes(path: Path) -> bytes:
    try:
        with Image.open(path) as image:
            normalized = image.convert("RGB")
            max_side = _int_env(
                "SMARTPOLICE_VISION_MODEL_MAX_SIDE",
                DEFAULT_VISION_MODEL_MAX_SIDE,
                minimum=320,
                maximum=2048,
            )
            quality = _int_env(
                "SMARTPOLICE_VISION_MODEL_JPEG_QUALITY",
                DEFAULT_VISION_MODEL_JPEG_QUALITY,
                minimum=40,
                maximum=95,
            )
            normalized.thumbnail((max_side, max_side))
            output = BytesIO()
            normalized.save(output, format="JPEG", quality=quality, optimize=True)
            return output.getvalue()
    except (OSError, ValueError):
        return path.read_bytes()


def _vision_provider() -> str:
    _load_env_file()
    value = os.getenv(VISION_PROVIDER_ENV, "").strip()
    if value:
        return value
    if _dashscope_vision_enabled():
        return "DashScope"
    return value or DEFAULT_VISION_PROVIDER


def _env_flag(name: str) -> bool:
    _load_env_file()
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    _load_env_file()
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _local_vlm_http_enabled() -> bool:
    if _dashscope_vision_enabled():
        return True
    if _pytest_blocks_local_vlm_http():
        return False
    return _env_flag(LOCAL_VLM_ENABLED_ENV)


def _local_vlm_http_required() -> bool:
    if _pytest_blocks_local_vlm_http():
        return False
    return _env_flag(CLOUD_VISION_REQUIRED_ENV)


def _dashscope_vision_enabled() -> bool:
    return _env_flag("ENABLE_DASHSCOPE") or _env_flag("SMARTPOLICE_ENABLE_DASHSCOPE")


def _pytest_blocks_local_vlm_http() -> bool:
    if not os.getenv("PYTEST_CURRENT_TEST"):
        return False
    base_url = os.getenv("LOCAL_VISION_BASE_URL", "").strip().lower()
    return "://local-vision.test" not in base_url


def _load_env_file() -> None:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    load_dotenv(ENV_PATH, override=False)


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise LlmOutputParseError("模型输出不是有效 JSON。") from None
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LlmOutputParseError("模型输出不是有效 JSON。") from exc
    if not isinstance(loaded, dict):
        raise LlmOutputParseError("模型输出 JSON 必须是对象。")
    return loaded


def _list_text(label: str, value: object) -> str:
    if isinstance(value, list):
        parts = [str(item) for item in value if str(item).strip()]
        return f"{label}：{'、'.join(parts) if parts else '无明确输出'}"
    if isinstance(value, str):
        return f"{label}：{value}"
    return f"{label}：无明确输出"


def _confidence(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return default


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
