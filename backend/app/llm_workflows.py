from __future__ import annotations

import json
import re

from app.analyzer import run_full_analysis
from app.model_gateway import invoke_model
from app.models import (
    CaseLlmReportRequest,
    CaseLlmReportResult,
    CaseLlmReviewRequest,
    CaseLlmReviewResult,
    CaseSample,
    FullAnalysis,
    KnowledgeSearchResult,
    ModelInvocationRequest,
)
from app.storage import search_knowledge


class LlmOutputParseError(ValueError):
    pass


def run_llm_review(
    case: CaseSample,
    request: CaseLlmReviewRequest,
) -> CaseLlmReviewResult:
    full = run_full_analysis(case)
    knowledge_refs = _knowledge_for_case(full)
    prompt = _review_prompt(full, knowledge_refs)
    system_prompt = (
        "你是公共安全谣言治理复核模型。只能基于输入案例、证据链和知识依据输出。"
        "必须用 JSON 返回，不要输出 Markdown。"
    )
    result = invoke_model(
        ModelInvocationRequest(
            case_id=case.id,
            provider=request.provider,
            role=request.role,
            prompt=prompt,
            system_prompt=system_prompt,
            dry_run=False,
            temperature=request.temperature,
        )
    )
    if not result.response_text or not result.audit_id:
        raise LlmOutputParseError("模型复核未返回可解析内容。")
    structured = _parse_json_object(result.response_text)
    return CaseLlmReviewResult(
        case_id=case.id,
        provider=result.provider,
        role=result.role,
        selected_model=result.selected_model,
        configured=result.configured,
        audit_id=result.audit_id,
        structured_review=structured,
        response_text=result.response_text,
    )


def run_llm_report(
    case: CaseSample,
    request: CaseLlmReportRequest,
) -> CaseLlmReportResult:
    full = run_full_analysis(case)
    knowledge_refs = _knowledge_for_case(full)
    prompt = _report_prompt(full, knowledge_refs)
    system_prompt = (
        "你是公共安全谣言治理研判报告助手。必须基于给定证据和知识依据生成报告。"
        "必须用 JSON 返回，字段为 markdown，不要添加 JSON 以外的内容。"
    )
    result = invoke_model(
        ModelInvocationRequest(
            case_id=case.id,
            provider=request.provider,
            role=request.role,
            prompt=prompt,
            system_prompt=system_prompt,
            dry_run=False,
            temperature=request.temperature,
        )
    )
    if not result.response_text or not result.audit_id:
        raise LlmOutputParseError("模型报告未返回可解析内容。")
    structured = _parse_json_object(result.response_text)
    markdown = structured.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        raise LlmOutputParseError("模型报告 JSON 缺少 markdown 字段。")
    return CaseLlmReportResult(
        case_id=case.id,
        provider=result.provider,
        role=result.role,
        selected_model=result.selected_model,
        configured=result.configured,
        audit_id=result.audit_id,
        markdown=markdown,
        knowledge_refs=knowledge_refs,
    )


def _knowledge_for_case(full: FullAnalysis) -> list[KnowledgeSearchResult]:
    query = " ".join(
        [
            full.case.title,
            full.case.scenario,
            full.case.content,
            " ".join(full.case.tags),
            full.risk.level.value,
            " ".join(full.disposal.verification),
        ]
    )
    return search_knowledge(query, limit=5)


def _review_prompt(
    full: FullAnalysis,
    knowledge_refs: list[KnowledgeSearchResult],
) -> str:
    return "\n".join(
        [
            "请复核以下公共安全谣言研判结果，并严格输出 JSON：",
            '{"conclusion": "...", "risk_level": "...", "risk_score": 0, '
            '"key_evidence": ["..."], "disposal_boundary": ["..."], '
            '"human_review_required": true, "missing_checks": ["..."]}',
            "",
            _case_block(full),
            _knowledge_block(knowledge_refs),
        ]
    )


def _report_prompt(
    full: FullAnalysis,
    knowledge_refs: list[KnowledgeSearchResult],
) -> str:
    return "\n".join(
        [
            "请生成一份审慎、可复制的研判报告，并严格输出 JSON：",
            '{"markdown": "# ..."}',
            "报告必须包含：事件概况、核心主张、多模态证据链、生成模型来源归因、风险评估、风险推演、处置建议、人工复核声明。",
            "不得下绝对化结论；涉及谣言性质或生成模型来源时，必须保留证据条件和人工复核口径。",
            "",
            _case_block(full),
            _knowledge_block(knowledge_refs),
        ]
    )


def _case_block(full: FullAnalysis) -> str:
    evidence_lines = "\n".join(
        f"- [{item.id}] {item.type.value}：{item.title}；{item.content}；置信度{round(item.confidence * 100)}%"
        for item in full.evidence_chain
    )
    dimensions = "\n".join(
        f"- {item.name}: {item.score}分，{item.reason}"
        for item in full.risk.dimensions
    )
    disposal = "\n".join(
        f"- {item}"
        for item in (
            full.disposal.verification
            + full.disposal.platform_coordination
            + full.disposal.public_response
            + full.disposal.local_coordination
            + full.disposal.evidence_preservation
        )
    )
    attribution = "\n".join(
        f"- {item.modality}: {item.candidate_model}，置信度{round(item.confidence * 100)}%，依据：{'；'.join(item.evidence)}，待核查：{'；'.join(item.verification_needed)}"
        for item in full.analysis.generator_attribution
    )
    return f"""案例：
标题：{full.case.title}
场景：{full.case.scenario}
平台：{full.case.platform}
内容：{full.case.content}
图片/截图描述：{full.case.image_description}
传播：浏览{full.case.spread.views}，转发{full.case.spread.reposts}，评论{full.case.spread.comments}，速度：{full.case.spread.velocity}
规则风险：{full.risk.level.value}，{full.risk.score}分

风险维度：
{dimensions}

证据链：
{evidence_lines}

生成模型来源归因：
{attribution}

规则处置建议：
{disposal}
"""


def _knowledge_block(knowledge_refs: list[KnowledgeSearchResult]) -> str:
    lines = "\n".join(
        f"- [{item.id}] {item.title}（{item.source}/{item.category}）：{item.content}"
        for item in knowledge_refs
    )
    return f"知识依据：\n{lines}"


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
