from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from uuid import uuid4

from app.models import (
    CaseSample,
    EvidenceItem,
    KnowledgeSearchResult,
    ModelInvocationAudit,
)
from app.storage import record_llm_invocation


LOCAL_REVIEW_PROVIDER = "LocalReview"
LOCAL_REPORT_PROVIDER = "LocalReport"
LOCAL_REVIEW_ROLE = "本地结构化复核"
LOCAL_REPORT_ROLE = "本地报告生成"
LOCAL_REVIEW_MODEL = "smartpolice-local-review-v1"
LOCAL_REPORT_MODEL = "smartpolice-local-report-v1"


def build_local_structured_review(
    *,
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    knowledge_refs: list[KnowledgeSearchResult],
    model_outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    risk_level = _risk_level_value(getattr(baseline_risk, "level", "待研判"))
    risk_score = _risk_score_value(getattr(baseline_risk, "model_score", None))
    if risk_score is None:
        risk_score = _risk_score_value(getattr(baseline_risk, "score", None)) or 0
    key_evidence_ids = [item.id for item in evidence[:6]]
    missing_checks = _missing_checks(evidence)
    disposal = [
        "固定原始图片、网页快照、发布账号和传播节点，保留 sha256 与审计编号。",
        "优先核验首发来源、权威通报和平台后台传播数据，避免仅凭模型分数定性。",
    ]
    if risk_score >= 68:
        disposal.append("建议进入平台协查和人工复核队列，必要时准备公开澄清口径。")
    return {
        "conclusion": _local_conclusion(risk_score, missing_checks),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "key_evidence_ids": key_evidence_ids,
        "evidence_conflicts": _evidence_conflicts(evidence),
        "disposal_suggestions": disposal,
        "missing_checks": missing_checks,
        "human_review_required": True,
        "review_mode": "local_offline",
        "model_outputs": model_outputs or {},
        "knowledge_ref_ids": [item.id for item in knowledge_refs[:6]],
        "boundary": "本地复核只做证据链一致性、风险分级和处置建议辅助，不替代人工核验或执法结论。",
    }


def build_local_markdown_report(
    *,
    case: CaseSample,
    evidence: list[EvidenceItem],
    baseline_risk: object,
    knowledge_refs: list[KnowledgeSearchResult],
    structured_review: dict[str, object],
    model_outputs: dict[str, object] | None = None,
) -> str:
    risk_level = str(structured_review.get("risk_level") or _risk_level_value(getattr(baseline_risk, "level", "待研判")))
    risk_score = structured_review.get("risk_score", getattr(baseline_risk, "score", "-"))
    evidence_lines = "\n".join(_display_evidence_line(index, item) for index, item in enumerate(evidence, start=1))
    knowledge_lines = "\n".join(
        f"- {item.title}：{item.content}"
        for item in knowledge_refs[:6]
    ) or "- 暂无命中的本地知识依据。"
    missing_lines = "\n".join(
        f"- {item}" for item in _string_list(structured_review.get("missing_checks"))
    ) or "- 暂无。"
    suggestions = "\n".join(
        f"- {item}" for item in _string_list(structured_review.get("disposal_suggestions"))
    )
    model_summary = _model_summary(model_outputs or {})
    return f"""# 公共安全谣言证据链研判报告

## 事件概况
案例名称：{case.title}

- 场景：{case.scenario}
- 平台：{case.platform}
- 发布时间：{case.publish_time}
- 核心主张：{case.content}

## 综合风险
本地复核结论：{structured_review.get("conclusion", "需人工复核")}

- 风险等级：{risk_level}
- 风险分：{risk_score}
- 复核方式：本地离线结构化复核

## 证据链
{evidence_lines}

## 模型辅助结论
{model_summary}

## 知识依据
{knowledge_lines}

## 处置建议
{suggestions}

## 待补充核查
{missing_lines}

## 人工复核声明
以上结论来自本地证据链、训练模型分支和规则化复核模板，只能作为辅助研判材料。最终定性需结合原始素材、平台协查、权威通报和人工复核。
"""


def record_local_invocation(
    *,
    case_id: str | None,
    provider: str,
    role: str,
    model: str,
    request_payload: dict[str, object],
    response_payload: dict[str, object] | str,
    status: str = "success",
    error: str | None = None,
) -> tuple[str, str]:
    response_text = (
        response_payload
        if isinstance(response_payload, str)
        else json.dumps(response_payload, ensure_ascii=False)
    )
    audit_id = str(uuid4())
    record_llm_invocation(
        ModelInvocationAudit(
            id=audit_id,
            case_id=case_id,
            provider=provider,
            role=role,
            model=model,
            status=status,
            request_payload=request_payload,
            response_text=response_text,
            error=error,
            latency_ms=0,
            token_usage={"local": True},
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return audit_id, response_text


def _display_evidence_line(index: int, item: EvidenceItem) -> str:
    fingerprint = f"，文件指纹：{item.sha256[:12]}" if item.sha256 else ""
    return f"- 证据{index}｜{item.title}：{_clean_report_text(item.content)}（来源：{_display_source(item.source)}{fingerprint}）"


def _display_source(value: str) -> str:
    if "审计" in value:
        return value.split(" 审计", 1)[0]
    return value


def _clean_report_text(value: str) -> str:
    text = re.sub(r"demo-[a-z0-9.-]+", "演示案例", value)
    text = re.sub(r"asset-demo-[a-z0-9.-]+-primary", "主图", text)
    text = re.sub(r"sha256 [0-9a-f]{16,}", "已固定文件指纹", text, flags=re.IGNORECASE)
    return text


def _model_summary(model_outputs: dict[str, object]) -> str:
    vision = model_outputs.get("vision_evidence_models")
    if not isinstance(vision, dict):
        return "- 当前报告使用本地证据链和规则化复核模板生成。"
    attribution = vision.get("vision_generator_attribution")
    if not isinstance(attribution, dict):
        return "- 当前报告使用本地证据链和规则化复核模板生成。"
    candidate = str(attribution.get("top_candidate") or "待研判")
    confidence = attribution.get("top_confidence")
    if isinstance(confidence, int | float):
        return f"- 图像来源研判最高候选：{_display_candidate(candidate)}，候选概率约 {round(float(confidence) * 100)}%。"
    return f"- 图像来源研判最高候选：{_display_candidate(candidate)}。"


def _display_candidate(value: str) -> str:
    labels = {
        "gpt-image2": "GPT-image-2",
        "gpt-image-2": "GPT-image-2",
        "nano-banana": "Nano Banana",
        "seedream-4": "Seedream-4",
        "stable-diffusion": "Stable Diffusion",
        "midjourney": "Midjourney",
        "flux": "Flux",
        "real": "真实照片",
        "unknown": "未知来源",
    }
    return labels.get(value.strip().lower(), value or "待研判")


def local_gateway_response(
    *,
    provider: str,
    role: str,
    prompt: str,
) -> dict[str, object] | str:
    if provider == LOCAL_REPORT_PROVIDER:
        markdown = (
            "# 本地报告草稿\n\n"
            "该结果由本地模板生成，用于在没有云端 API 时完成基础研判闭环。\n\n"
            f"## 输入摘要\n{prompt[:600]}\n\n"
            "## 边界\n本地报告不替代人工核验、平台协查或权威通报。"
        )
        return {"markdown": markdown, "role": role, "review_mode": "local_offline"}
    return {
        "conclusion": "本地复核建议进入人工核验队列。",
        "risk_level": "关注",
        "risk_score": 50,
        "key_evidence": [],
        "disposal_boundary": ["本地复核仅用于无云环境下的结构化辅助判断。"],
        "human_review_required": True,
        "missing_checks": ["原始素材", "权威来源", "平台传播数据"],
        "role": role,
        "review_mode": "local_offline",
    }


def _risk_level_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "待研判")


def _risk_score_value(value: object) -> int | None:
    if isinstance(value, int | float):
        return max(0, min(100, round(float(value))))
    return None


def _missing_checks(evidence: list[EvidenceItem]) -> list[str]:
    missing: list[str] = []
    if not any(item.source_url for item in evidence):
        missing.append("需补充公开 URL 或权威来源快照。")
    if not any(item.sha256 for item in evidence if item.type.value == "图像证据"):
        missing.append("需固定原始图片哈希并核验原图来源。")
    missing.append("需人工复核模型输出与原始素材的一致性。")
    return missing


def _evidence_conflicts(evidence: list[EvidenceItem]) -> list[str]:
    conflicts = []
    for item in evidence:
        text = item.content
        if "不确定" in text or "待核验" in text or "需核验" in text:
            conflicts.append(f"{item.id}: 存在待核查信息。")
    return conflicts[:5]


def _local_conclusion(score: int, missing_checks: list[str]) -> str:
    if score >= 85:
        return "证据链显示紧急风险信号，建议立即人工复核并启动协查。"
    if score >= 68:
        return "证据链显示较高风险信号，建议进入人工复核和平台协查。"
    if missing_checks:
        return "当前证据链仍有缺口，建议补充核查后再定性。"
    return "当前风险较低，但仍需保留证据并完成常规复核。"


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []
