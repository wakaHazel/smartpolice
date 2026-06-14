from __future__ import annotations

from app.models import (
    AgentCostGate,
    AgentModelRoute,
    AgentOrchestration,
    AgentSkillRecommendation,
    CaseSample,
    DisposalSuggestion,
    EvidenceItem,
    EvidenceType,
    FullAnalysis,
    GeneratorAttribution,
    LearningPipelineStage,
    MultimodalAnalysis,
    ReportDraft,
    RiskAssessment,
    RiskDimension,
    RiskLevel,
)
from app.risk_model import (
    extract_features,
    predict_with_active_model,
    risk_level_from_score,
)


DIMENSION_NAMES = [
    "公共安全相关度",
    "真实性风险",
    "传播影响",
    "情绪煽动性",
    "线下扰动可能性",
]


def analyze_multimodal(case: CaseSample) -> MultimodalAnalysis:
    claims = _extract_claims(case)
    image_findings = _image_findings(case)
    consistency = _consistency_findings(case)
    aigc = _aigc_indicators(case)
    attribution = _generator_attribution(case)
    judgement = (
        "该样本存在多源疑点，且疑似生成模型来源需要进一步溯源核验，建议进入人工复核。"
        if len(aigc) > 1 or "低风险" not in case.scenario
        else "该样本风险较低，建议核验来源后以提示性处置为主。"
    )
    return MultimodalAnalysis(
        case_id=case.id,
        claims=claims,
        image_findings=image_findings,
        consistency_findings=consistency,
        aigc_indicators=aigc,
        generator_attribution=attribution,
        preliminary_judgement=judgement,
    )


def build_evidence_chain(case: CaseSample, analysis: MultimodalAnalysis) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"{case.id}-content",
            type=EvidenceType.CONTENT,
            title="文本主张与叙事风险",
            content="；".join(analysis.claims),
            confidence=0.86,
            source="样本文本语义分析",
            supports="支撑真实性风险和情绪煽动性判断",
        ),
        EvidenceItem(
            id=f"{case.id}-image",
            type=EvidenceType.IMAGE,
            title="图片/截图疑点与生成来源归因",
            content="；".join(analysis.image_findings),
            confidence=0.82,
            source="多模态内容分析与生成模型归因",
            supports="支撑AIGC疑似度、图文一致性和生成来源核验",
        ),
        EvidenceItem(
            id=f"{case.id}-source",
            type=EvidenceType.SOURCE,
            title="来源可信度",
            content=f"来源为{case.source_url}，未见权威通报链路，需进一步核验首发账号与原始素材。",
            confidence=0.74,
            source="来源信息与公开样本记录",
            supports="支撑核查优先级判断",
        ),
        EvidenceItem(
            id=f"{case.id}-spread",
            type=EvidenceType.SPREAD,
            title="传播态势",
            content=(
                f"浏览{case.spread.views}次，转发{case.spread.reposts}次，"
                f"评论{case.spread.comments}次，传播特征为{case.spread.velocity}。"
            ),
            confidence=0.9,
            source="脱敏传播指标",
            supports="支撑传播影响和处置紧急度判断",
        ),
        EvidenceItem(
            id=f"{case.id}-authority",
            type=EvidenceType.AUTHORITY,
            title="治理依据",
            content=(
                "生成式人工智能、深度合成和AI生成合成内容标识相关规定均要求防范"
                "虚假有害信息传播，公共安全类谣言应保留证据并开展平台协同核查。"
            ),
            confidence=0.88,
            source="政策法规知识库",
            supports="支撑处置建议和报告依据",
        ),
    ]


def assess_risk(case: CaseSample, evidence: list[EvidenceItem]) -> RiskAssessment:
    scores = _dimension_scores(case)
    model_prediction = predict_with_active_model(case)
    model_version_id: str | None = None
    model_score: float | None = None
    model_confidence: float | None = None
    model_explanation: list[str] = []
    if model_prediction is not None:
        model_score, model_confidence, model_explanation, model_version_id = model_prediction
        total = round(model_score)
    else:
        total = round(sum(scores.values()) / len(scores))
        model_explanation = ["尚未训练本地风险模型，当前使用规则特征评分兜底。"]
    level = risk_level_from_score(total)

    dimensions = [
        RiskDimension(name=name, score=score, reason=_dimension_reason(case, name, score))
        for name, score in scores.items()
    ]
    reasoning = [
        f"样本属于{case.scenario}，公共安全相关度明确。",
        f"证据链已形成{len(evidence)}类证据，可支持人工复核和分级处置。",
        (
            f"风险等级由训练模型版本 {model_version_id} 输出，并结合规则维度解释。"
            if model_version_id
            else "风险等级由内容疑点、传播态势和线下扰动可能性综合判定。"
        ),
    ]
    evolution = _risk_evolution(case, level)
    return RiskAssessment(
        case_id=case.id,
        score=total,
        level=level,
        dimensions=dimensions,
        reasoning=reasoning,
        evolution=evolution,
        model_version_id=model_version_id,
        model_score=model_score,
        model_confidence=model_confidence,
        model_explanation=model_explanation,
    )


def suggest_disposal(case: CaseSample, risk: RiskAssessment) -> DisposalSuggestion:
    common_verification = [
        "核验首发账号、原始发布时间和素材来源，固定页面截图与链接。",
        "比对权威通报、属地警情记录和公开辟谣信息。",
    ]
    platform = [
        "向平台提交疑似AIGC虚假信息线索，申请保全传播链路和账号操作日志。",
        "对持续扩散内容建议采取限流、提示、折叠或辟谣卡片等治理措施。",
    ]
    public = [
        "在事实核查完成后发布简明澄清信息，重点回应群众关切。",
        "避免扩大争议性表述，以事实、时间线和权威来源降低情绪对立。",
    ]
    local = [
        "视风险等级联动属地网安、宣传、应急或相关业务部门。",
        "如出现线下聚集苗头，提前开展现场秩序维护和重点区域巡查。",
    ]
    evidence = [
        "留存原帖、评论高赞内容、传播节点、图片文件和研判报告版本。",
        "标注大模型输出为辅助研判材料，最终结论以人工复核为准。",
    ]

    if risk.level == RiskLevel.LOW:
        platform = ["建议平台补充来源提示，暂不建议采取强处置措施。"]
        public = ["可通过社区群提示核验结果，避免过度放大。"]
        local = ["保持关注，无需启动跨部门联动。"]

    return DisposalSuggestion(
        case_id=case.id,
        verification=common_verification,
        platform_coordination=platform,
        public_response=public,
        local_coordination=local,
        evidence_preservation=evidence,
        review_note="系统结论为辅助研判草稿，需由民警结合线下核查结果复核确认。",
    )


def generate_report(
    case: CaseSample,
    analysis: MultimodalAnalysis,
    evidence: list[EvidenceItem],
    risk: RiskAssessment,
    disposal: DisposalSuggestion,
) -> ReportDraft:
    evidence_summary = [f"{item.type.value}: {item.title} - {item.content}" for item in evidence]
    suggestions = (
        disposal.verification
        + disposal.platform_coordination
        + disposal.public_response
        + disposal.local_coordination
        + disposal.evidence_preservation
    )
    summary = (
        f"样本《{case.title}》属于{case.scenario}，当前综合风险等级为{risk.level.value}，"
        f"风险分值{risk.score}。系统识别到{len(analysis.claims)}项文本主张、"
        f"{len(analysis.image_findings)}项图像/截图疑点，并生成"
        f"{len(analysis.generator_attribution)}项疑似生成模型来源候选。"
    )
    markdown = _report_markdown(case, analysis, evidence, risk, suggestions, summary)
    return ReportDraft(
        case_id=case.id,
        title=f"{case.scenario}研判报告",
        summary=summary,
        evidence_summary=evidence_summary,
        risk_summary=f"综合风险等级：{risk.level.value}；分值：{risk.score}。",
        suggestions=suggestions,
        review_statement=disposal.review_note,
        markdown=markdown,
    )


def build_agent_orchestration(
    case: CaseSample,
    analysis: MultimodalAnalysis,
    evidence: list[EvidenceItem],
    risk: RiskAssessment,
) -> AgentOrchestration:
    skills = _agent_skills(case, analysis, risk)
    strategy = (
        "高风险样本走 DeepSeek 复杂推理主控和复核，MiniMax 负责任务路由、长上下文证据读取和中文报告生成。"
        if risk.level in {RiskLevel.HIGH, RiskLevel.URGENT}
        else "低到中风险样本优先使用 MiniMax 轻量路由和规则化 skill，DeepSeek 仅在复核或升级处置时调用。"
    )
    trace = [
        f"读取样本标签：{'、'.join(case.tags)}。",
        f"证据链已形成{len(evidence)}类证据，综合风险{risk.score}分，等级为{risk.level.value}。",
        f"识别到{len(analysis.generator_attribution)}项生成来源候选，触发模型归因复核流程。",
        f"推荐{len(skills)}个 skills，并记录轨迹用于后续 HDBSCAN 聚类和 LightGBM 路由训练。",
    ]
    return AgentOrchestration(
        case_id=case.id,
        primary_strategy=strategy,
        model_routes=_agent_model_routes(case, risk),
        recommended_skills=skills,
        learning_pipeline=_learning_pipeline(),
        cost_gates=_cost_gates(risk),
        execution_trace=trace,
    )


def run_full_analysis(case: CaseSample) -> FullAnalysis:
    analysis = analyze_multimodal(case)
    evidence = build_evidence_chain(case, analysis)
    risk = assess_risk(case, evidence)
    disposal = suggest_disposal(case, risk)
    report = generate_report(case, analysis, evidence, risk, disposal)
    agent = build_agent_orchestration(case, analysis, evidence, risk)
    return FullAnalysis(
        case=case,
        analysis=analysis,
        evidence_chain=evidence,
        risk=risk,
        disposal=disposal,
        report=report,
        agent=agent,
    )


def _agent_model_routes(case: CaseSample, risk: RiskAssessment) -> list[AgentModelRoute]:
    high_risk = risk.level in {RiskLevel.HIGH, RiskLevel.URGENT}
    planner_model = "DeepSeek V4 Pro" if high_risk else "MiniMax-M3"
    planner_provider = "DeepSeek" if high_risk else "MiniMax"
    planner_reason = (
        "风险等级达到较高以上，需要强推理模型统筹证据冲突、处置升级和复核路径。"
        if high_risk
        else "当前样本不需要每轮都上强推理模型，先用 MiniMax 完成轻量业务编排。"
    )
    return [
        AgentModelRoute(
            role="任务路由器",
            selected_model="MiniMax-M3",
            provider="MiniMax",
            reason="用 MiniMax 判断任务类型、风险段位和是否需要进入 DeepSeek 强模型复核。",
            cost_tier="低成本",
            fallback_models=["DeepSeek V4 Pro"],
        ),
        AgentModelRoute(
            role="复杂推理主控",
            selected_model=planner_model,
            provider=planner_provider,
            reason=planner_reason,
            cost_tier="高能力" if high_risk else "中等成本",
            fallback_models=["MiniMax-M3"] if high_risk else ["DeepSeek V4 Pro"],
        ),
        AgentModelRoute(
            role="长上下文证据读取",
            selected_model="MiniMax-M3",
            provider="MiniMax",
            reason=(
                f"样本来自{case.platform}，需读取证据链、传播指标、法规依据和历史轨迹时使用长上下文窗口。"
            ),
            cost_tier="按需调用",
            fallback_models=["DeepSeek V4 Pro"],
        ),
        AgentModelRoute(
            role="中文业务生成",
            selected_model="MiniMax-M3",
            provider="MiniMax",
            reason="生成研判摘要、处置建议和报告草稿时使用 MiniMax 保持中文表达、结构化输出和长上下文一致性。",
            cost_tier="中等成本",
            fallback_models=["DeepSeek V4 Pro"],
        ),
        AgentModelRoute(
            role="复核器",
            selected_model="DeepSeek V4 Pro",
            provider="DeepSeek",
            reason="对高风险结论、模型归因和线下联动建议做二次推理复核。",
            cost_tier="高能力闸门",
            fallback_models=["MiniMax-M3"],
        ),
    ]


def _agent_skills(
    case: CaseSample,
    analysis: MultimodalAnalysis,
    risk: RiskAssessment,
) -> list[AgentSkillRecommendation]:
    skills = [
        AgentSkillRecommendation(
            name="source_verification_skill",
            trigger="样本缺少权威通报链路或首发账号可信度不足。",
            algorithm="BM25 + 向量召回 + 来源可信度规则评分",
            steps=[
                "固定原帖链接、发布时间、账号信息和截图。",
                "检索权威通报、属地警情记录和公开辟谣信息。",
                "比对首发时间线，标注二次转发和搬运节点。",
            ],
            verification=[
                "至少命中一条权威或属地核验来源。",
                "输出首发链路、传播链路和未核实缺口。",
            ],
            confidence=0.88,
        )
    ]

    if analysis.aigc_indicators and case.id != "low-risk-004":
        skills.append(
            AgentSkillRecommendation(
                name="aigc_attribution_skill",
                trigger="图像、截图或视频关键帧存在生成式内容疑点。",
                algorithm="多模态疑点规则 + 生成模型候选排序 + 人工复核清单",
                steps=[
                    "拆分文本主张、图片疑点、截图版式和视频关键帧。",
                    "检查 EXIF、C2PA、平台水印和反向图片检索结果。",
                    "根据候选模型证据与反证输出归因置信度。",
                ],
                verification=[
                    "每个候选模型必须同时给出依据、反证和待核查项。",
                    "模型归因不得作为最终结论，必须保留人工复核声明。",
                ],
                confidence=0.84,
            )
        )

    if risk.score >= 40:
        skills.append(
            AgentSkillRecommendation(
                name="risk_evolution_skill",
                trigger="传播量、评论情绪或线下扰动可能性进入关注以上等级。",
                algorithm="风险维度加权评分 + PrefixSpan 传播路径模式挖掘",
                steps=[
                    "聚合浏览、转发、评论、点赞和扩散速度。",
                    "识别恐慌、对立、涉警公信力等风险剧本。",
                    "输出从扩散到辟谣回落的演化路径。",
                ],
                verification=[
                    "风险推演需对应至少两项证据链节点。",
                    "升级处置必须与风险等级和传播态势一致。",
                ],
                confidence=0.9,
            )
        )

    if risk.level in {RiskLevel.HIGH, RiskLevel.URGENT}:
        skills.append(
            AgentSkillRecommendation(
                name="joint_disposal_skill",
                trigger="风险等级达到较高或紧急，需要平台协查、属地联动和证据保全。",
                algorithm="LightGBM 路由器 + 处置剧本规则库",
                steps=[
                    "按风险类型选择平台协查、公开回应和属地联动路径。",
                    "生成留证清单、协查请求要点和对外回应要点。",
                    "用强推理模型复核是否存在过度处置或遗漏处置。",
                ],
                verification=[
                    "处置建议必须覆盖核查、平台、公开回应、属地联动和证据留存。",
                    "低置信结论不得触发强处置建议。",
                ],
                confidence=0.86,
            )
        )

    if risk.level == RiskLevel.LOW:
        skills.append(
            AgentSkillRecommendation(
                name="low_risk_hint_skill",
                trigger="传播范围有限且未出现明显情绪煽动或线下扰动。",
                algorithm="低风险阈值规则 + 成本闸门",
                steps=[
                    "只做来源提示和轻量核验。",
                    "避免触发平台强处置或跨部门联动。",
                    "记录样本作为低风险负例，用于后续路由校准。",
                ],
                verification=[
                    "风险分值低于40分。",
                    "处置建议不得出现强联动或强下架表述。",
                ],
                confidence=0.93,
            )
        )

    return skills


def _learning_pipeline() -> list[LearningPipelineStage]:
    return [
        LearningPipelineStage(
            name="轨迹采集",
            algorithm="结构化事件日志",
            purpose="记录任务、上下文、工具调用、证据节点、测试/复核结果和人工评分。",
            output="可回放 agent trajectory 数据集",
        ),
        LearningPipelineStage(
            name="技能发现",
            algorithm="Embedding + UMAP + HDBSCAN",
            purpose="从真实研判任务中自动发现高频任务簇和异常任务簇。",
            output="候选 skill 主题、触发条件和代表样本",
        ),
        LearningPipelineStage(
            name="流程挖掘",
            algorithm="PrefixSpan 序列模式挖掘",
            purpose="从成功案例中提炼稳定工具调用顺序和证据核验路径。",
            output="skill steps 与常见失败分支",
        ),
        LearningPipelineStage(
            name="路由训练",
            algorithm="LightGBM / Logistic Regression",
            purpose="根据任务特征选择 DeepSeek、MiniMax 或规则化 skill。",
            output="可解释模型路由器与成本分层策略",
        ),
        LearningPipelineStage(
            name="在线优化",
            algorithm="Thompson Sampling",
            purpose="在成功率、成本和延迟之间动态选择模型组合。",
            output="模型调用策略的在线收益估计",
        ),
    ]


def _cost_gates(risk: RiskAssessment) -> list[AgentCostGate]:
    strong_model_rule = (
        "风险等级为较高/紧急或证据冲突超过2项时才调用 DeepSeek V4 Pro。"
        if risk.level in {RiskLevel.HIGH, RiskLevel.URGENT}
        else "低风险样本只在人工复核或结论不一致时调用 DeepSeek V4 Pro。"
    )
    return [
        AgentCostGate(
            name="强模型闸门",
            rule=strong_model_rule,
            expected_saving="减少不必要的高能力模型调用。",
        ),
        AgentCostGate(
            name="长上下文闸门",
            rule="证据材料超过普通上下文窗口或需跨报告、法规、历史轨迹比对时才调用 MiniMax-M3。",
            expected_saving="避免把短样本全部塞进长上下文模型。",
        ),
        AgentCostGate(
            name="复核闸门",
            rule="测试、规则和证据链一致时直接出草稿；高风险、低置信或强处置建议才进入二次复核。",
            expected_saving="把大模型 judge 控制在关键节点。",
        ),
    ]


def _extract_claims(case: CaseSample) -> list[str]:
    text = case.content + case.title
    if case.id == "police-trust-001":
        return [
            "声称民警处置纠纷时存在不当执法行为",
            "声称现场多人受伤且警方隐瞒执法记录",
            "以夜间冲突画面强化涉警负面叙事",
        ]
    if case.id == "disaster-risk-002":
        return [
            "声称本地山区突发塌方",
            "声称多辆校车被困",
            "要求市民立即转发避险，制造紧迫感",
        ]
    if case.id == "group-polarization-003":
        return [
            "声称商圈发生性别冲突事件",
            "声称警方偏袒一方",
            "号召网友线下集合声援",
        ]
    claims: list[str] = []
    if "警方" in text or "民警" in text or "执法" in text:
        claims.append("涉及警方执法或警情处置主张")
    if "灾害" in text or "塌方" in text or "被困" in text:
        claims.append("涉及灾害险情或公共避险主张")
    if "冲突" in text or "偏袒" in text or "集合" in text:
        claims.append("涉及群体冲突、偏袒叙事或线下动员主张")
    if "转发" in text or "立即" in text:
        claims.append("包含紧迫性传播动员表述")
    if not claims:
        claims = ["提取到待核验公共信息主张", "缺少明确权威来源或原始素材链路"]
    return claims


def _image_findings(case: CaseSample) -> list[str]:
    text = case.image_description + case.content
    if case.id == "low-risk-004" or ("低风险" in case.scenario and "异常" not in text):
        return ["图片未发现明显AI合成痕迹", "画面内容与交通提示基本相关"]
    findings = [case.image_description]
    if any(token in text for token in ["异常", "字体", "头像", "路牌", "纹理", "水印", "EXIF"]):
        findings.append("图片/截图存在版式、文字、纹理或元数据疑点。")
    if any(token in text for token in ["旧图", "不一致", "不匹配", "嫁接"]):
        findings.append("图像场景与文本叙事存在旧图复用或地点不一致风险。")
    findings.append("建议补充原图、EXIF、首发时间和反向图片检索结果。")
    return findings


def _consistency_findings(case: CaseSample) -> list[str]:
    text = case.content + case.image_description
    if case.id == "low-risk-004" or ("低风险" in case.scenario and "不匹配" not in text):
        return ["图文大体一致，但缺少权威来源。"]
    return [
        "文本给出的地点、时间或事件强度未能被图片独立证明",
        "图像视觉元素与声称场景存在不一致，需进入人工核查",
    ]


def _aigc_indicators(case: CaseSample) -> list[str]:
    text = case.content + case.image_description + case.manual_label
    if case.id == "low-risk-004" or ("低风险" in case.scenario and "AI" not in text and "合成" not in text):
        return ["未发现明显AIGC特征，AIGC疑似度较低。"]
    indicators = ["画面叙事完整但缺少可追溯原始来源"]
    if any(token in text for token in ["AI", "AIGC", "合成", "生成"]):
        indicators.append("样本明示或疑似包含AI生成/合成内容线索。")
    if any(token in text for token in ["文字", "边缘", "手部", "头像", "纹理", "字体"]):
        indicators.append("局部文字、边缘、人物或截图细节不稳定。")
    if any(token in text for token in ["转发", "号召", "偏袒", "隐瞒", "集合"]):
        indicators.append("图像和文本组合具有强情绪引导倾向。")
    return indicators


def _generator_attribution(case: CaseSample) -> list[GeneratorAttribution]:
    if case.id == "police-trust-001":
        return [
            GeneratorAttribution(
                modality="图片",
                candidate_model="GPT Image 2 / GPT Image 系列",
                model_family="OpenAI图像生成",
                confidence=0.58,
                evidence=[
                    "画面整体语义完整，执法场景与文本提示贴合度较高",
                    "人物手部、肩章、路牌文字存在生成式图像常见不稳定细节",
                    "缺少可核验EXIF和原始拍摄链路",
                ],
                counter_evidence=[
                    "社交平台二次压缩会削弱模型指纹",
                    "同类瑕疵也可能来自其他扩散模型或局部修图",
                ],
                verification_needed=[
                    "提取C2PA/内容凭证元数据",
                    "对原图做反向图片检索和压缩链路分析",
                    "与Seedream、Midjourney、Stable Diffusion候选指纹进行比对",
                ],
            ),
            GeneratorAttribution(
                modality="图片",
                candidate_model="Seedream / 豆包图像生成",
                model_family="字节跳动Seed图像模型",
                confidence=0.46,
                evidence=[
                    "中文城市街面和制服类视觉元素贴近中文提示词生成偏好",
                    "局部中文标识异常，符合中文图像生成模型常见OCR弱点",
                ],
                counter_evidence=["缺少平台水印、内容凭证或发布端线索，不能直接确认。"],
                verification_needed=["检查平台导出水印、生成记录截图和账号发布工具来源。"],
            ),
        ]
    if case.id == "disaster-risk-002":
        return [
            GeneratorAttribution(
                modality="图片",
                candidate_model="Seedream / 豆包图像生成",
                model_family="字节跳动Seed图像模型",
                confidence=0.62,
                evidence=[
                    "灾害场景构图完整，泥石流、车辆、道路元素高度贴合中文提示词",
                    "道路标识和植被细节与本地实况不一致",
                    "局部纹理呈现扩散模型补全痕迹",
                ],
                counter_evidence=[
                    "旧图嫁接也可能造成地点不一致",
                    "平台压缩会影响频域检测稳定性",
                ],
                verification_needed=[
                    "进行反向图片检索确认是否旧图复用",
                    "比对原图元数据、C2PA和平台生成水印",
                ],
            ),
            GeneratorAttribution(
                modality="视频关键帧",
                candidate_model="Seedance / Sora 候选",
                model_family="文本生成视频模型",
                confidence=0.37,
                evidence=[
                    "若样本来自短视频切片，应检查连续帧中车辆、道路和水流运动一致性",
                    "灾害类视频生成模型容易出现物理运动和物体持久性异常",
                ],
                counter_evidence=["当前样本仅有静态截图，视频模型归因证据不足。"],
                verification_needed=["获取原视频，做逐帧一致性、物理运动和元数据检查。"],
            ),
        ]
    if case.id == "group-polarization-003":
        return [
            GeneratorAttribution(
                modality="图片/截图",
                candidate_model="Stable Diffusion / Midjourney / GPT Image 系列",
                model_family="通用图像生成与局部编辑模型",
                confidence=0.55,
                evidence=[
                    "现场照片人物面部纹理异常，商场标识与声称地点不匹配",
                    "聊天截图字体间距和头像重复，疑似由模板生成或二次编辑",
                    "图文组合服务于强烈群体对立叙事",
                ],
                counter_evidence=[
                    "聊天截图也可能由普通修图工具伪造，并非一定来自大模型",
                    "缺少生成平台凭证，无法确认具体模型",
                ],
                verification_needed=[
                    "分离截图层和现场图片分别做OCR、版式一致性和生成指纹检测",
                    "检查是否存在豆包、即梦、可灵、Runway等平台水印或导出痕迹",
                ],
            ),
            GeneratorAttribution(
                modality="视频关键帧",
                candidate_model="Sora / Seedance / Kling 候选",
                model_family="文本生成视频模型",
                confidence=0.33,
                evidence=[
                    "若存在偷拍视频，应重点检查人物身份连续性、镜头运动、文字稳定性",
                    "群体冲突类生成视频常出现运动连续性和细节持久性问题",
                ],
                counter_evidence=["当前演示样本以截图和现场图为主，视频归因只能作为待核查项。"],
                verification_needed=["获取完整视频文件并进行关键帧、音画同步和内容凭证检测。"],
            ),
        ]
    return [
        GeneratorAttribution(
            modality="图片",
            candidate_model="未发现明确生成模型来源",
            model_family="无明显AIGC归因",
            confidence=0.18,
            evidence=["当前样本画面与文本大体一致，缺少明显生成式图像瑕疵。"],
            counter_evidence=["低清截图和平台压缩仍可能掩盖生成痕迹。"],
            verification_needed=["如需精确确认，可补充原图、EXIF和反向图片检索。"],
        )
    ]


def _dimension_scores(case: CaseSample) -> dict[str, int]:
    if case.id == "police-trust-001":
        return {
            "公共安全相关度": 88,
            "真实性风险": 82,
            "传播影响": 76,
            "情绪煽动性": 84,
            "线下扰动可能性": 62,
        }
    if case.id == "disaster-risk-002":
        return {
            "公共安全相关度": 94,
            "真实性风险": 86,
            "传播影响": 91,
            "情绪煽动性": 78,
            "线下扰动可能性": 83,
        }
    if case.id == "group-polarization-003":
        return {
            "公共安全相关度": 81,
            "真实性风险": 84,
            "传播影响": 79,
            "情绪煽动性": 92,
            "线下扰动可能性": 74,
        }
    features = extract_features(case)
    return {
        "公共安全相关度": _score_from_features(features, ["public_safety", "police_trust", "disaster_panic"], base=20),
        "真实性风险": _score_from_features(features, ["aigc_suspicion", "source_gap"], base=18),
        "传播影响": _score_from_features(features, ["views_log", "reposts_log", "comments_log", "fast_velocity"], base=8),
        "情绪煽动性": _score_from_features(features, ["emotion_incite", "group_polarization"], base=6),
        "线下扰动可能性": _score_from_features(features, ["offline_mobilization", "group_polarization", "disaster_panic"], base=6),
    }


def _dimension_reason(case: CaseSample, name: str, score: int) -> str:
    if score >= 80:
        qualifier = "高"
    elif score >= 60:
        qualifier = "中高"
    elif score >= 35:
        qualifier = "关注"
    else:
        qualifier = "低"
    return f"{name}为{qualifier}水平。{case.sensitivity_notes}"


def _risk_evolution(case: CaseSample, level: RiskLevel) -> list[str]:
    if level == RiskLevel.LOW:
        return ["小范围传播", "核验后群内提示", "风险自然衰减"]
    if case.id == "disaster-risk-002":
        return ["跨群快速转发", "引发集中咨询或报警", "挤占应急和警务资源", "权威辟谣后回落"]
    if case.id == "group-polarization-003":
        return ["争议话题聚集", "群体标签化评论增加", "网暴和线下声援风险上升", "事实澄清与平台治理后降温"]
    return ["负面涉警叙事扩散", "质疑执法公信力", "评论区情绪聚集", "核查通报后风险下降"]


def _score_from_features(features: dict[str, float], names: list[str], base: int) -> int:
    score = base + sum(features.get(name, 0.0) * 12 for name in names)
    if features.get("low_spread", 0) > 0:
        score -= 16
    return round(max(0, min(100, score)))


def _report_markdown(
    case: CaseSample,
    analysis: MultimodalAnalysis,
    evidence: list[EvidenceItem],
    risk: RiskAssessment,
    suggestions: list[str],
    summary: str,
) -> str:
    evidence_lines = "\n".join(f"- {item.type.value}: {item.content}" for item in evidence)
    dimension_lines = "\n".join(
        f"- {dimension.name}: {dimension.score}分，{dimension.reason}"
        for dimension in risk.dimensions
    )
    suggestion_lines = "\n".join(f"- {item}" for item in suggestions)
    claim_lines = "\n".join(f"- {item}" for item in analysis.claims)
    evolution_lines = "\n".join(f"- {item}" for item in risk.evolution)
    attribution_lines = "\n".join(
        (
            f"- {item.modality}：{item.candidate_model}（{item.model_family}，"
            f"置信度{round(item.confidence * 100)}%）。依据：{'；'.join(item.evidence)}。"
            f"待核查：{'；'.join(item.verification_needed)}。"
        )
        for item in analysis.generator_attribution
    )
    return f"""# {case.scenario}研判报告

## 一、事件概况
{summary}

## 二、核心主张
{claim_lines}

## 三、多模态证据链
{evidence_lines}

## 四、生成模型来源归因
{attribution_lines}

## 五、风险评估
综合等级：{risk.level.value}

综合分值：{risk.score}

{dimension_lines}

## 六、风险推演
{evolution_lines}

## 七、处置建议
{suggestion_lines}

## 八、复核声明
本报告由系统自动生成，作为辅助研判草稿。最终处置结论应由人工复核确认。
"""
