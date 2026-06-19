from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "低"
    WATCH = "关注"
    HIGH = "较高"
    URGENT = "紧急"


class EvidenceType(str, Enum):
    CONTENT = "内容证据"
    IMAGE = "图像证据"
    SOURCE = "来源证据"
    SPREAD = "传播证据"
    AUTHORITY = "权威依据"


class SpreadMetrics(BaseModel):
    views: int
    reposts: int
    comments: int
    likes: int
    velocity: str


class CaseCreateRequest(BaseModel):
    id: str | None = None
    title: str
    scenario: str
    platform: str
    publish_time: str
    source_url: str = "本地录入样本"
    content: str
    image_description: str
    spread: SpreadMetrics
    manual_label: str = "待人工复核"
    manual_risk_score: int | None = Field(default=None, ge=0, le=100)
    tags: list[str] = Field(default_factory=list)
    sensitivity_notes: str = ""
    review_note: str = ""


class CaseLabelRequest(BaseModel):
    manual_risk_score: int = Field(ge=0, le=100)
    manual_label: str = "人工复核标注"
    review_note: str = ""


class CaseSample(BaseModel):
    id: str
    title: str
    scenario: str
    platform: str
    publish_time: str
    source_url: str
    content: str
    image_description: str
    spread: SpreadMetrics
    manual_label: str
    manual_risk_score: int | None = Field(default=None, ge=0, le=100)
    tags: list[str]
    sensitivity_notes: str
    review_note: str = ""
    created_by_user: bool = False


class MultimodalAnalysis(BaseModel):
    case_id: str
    claims: list[str]
    image_findings: list[str]
    consistency_findings: list[str]
    aigc_indicators: list[str]
    generator_attribution: list["GeneratorAttribution"]
    preliminary_judgement: str


class GeneratorAttribution(BaseModel):
    modality: str
    candidate_model: str
    model_family: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    counter_evidence: list[str]
    verification_needed: list[str]


class EvidenceItem(BaseModel):
    id: str
    type: EvidenceType
    title: str
    content: str
    confidence: float = Field(ge=0, le=1)
    source: str
    supports: str
    artifact_id: str | None = None
    source_url: str | None = None
    sha256: str | None = None
    created_at: str | None = None


class RiskDimension(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    reason: str


class RiskAssessment(BaseModel):
    case_id: str
    score: int = Field(ge=0, le=100)
    level: RiskLevel
    dimensions: list[RiskDimension]
    reasoning: list[str]
    evolution: list[str]
    model_version_id: str | None = None
    model_score: float | None = Field(default=None, ge=0, le=100)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    model_explanation: list[str] = Field(default_factory=list)


class DisposalSuggestion(BaseModel):
    case_id: str
    verification: list[str]
    platform_coordination: list[str]
    public_response: list[str]
    local_coordination: list[str]
    evidence_preservation: list[str]
    review_note: str


class ReportDraft(BaseModel):
    case_id: str
    title: str
    summary: str
    evidence_summary: list[str]
    risk_summary: str
    suggestions: list[str]
    review_statement: str
    markdown: str


class AgentModelRoute(BaseModel):
    role: str
    selected_model: str
    provider: str
    reason: str
    cost_tier: str
    fallback_models: list[str]


class AgentSkillRecommendation(BaseModel):
    name: str
    trigger: str
    algorithm: str
    steps: list[str]
    verification: list[str]
    confidence: float = Field(ge=0, le=1)


class LearningPipelineStage(BaseModel):
    name: str
    algorithm: str
    purpose: str
    output: str


class AgentCostGate(BaseModel):
    name: str
    rule: str
    expected_saving: str


class AgentOrchestration(BaseModel):
    case_id: str
    primary_strategy: str
    model_routes: list[AgentModelRoute]
    recommended_skills: list[AgentSkillRecommendation]
    learning_pipeline: list[LearningPipelineStage]
    cost_gates: list[AgentCostGate]
    execution_trace: list[str]


class AgentRunRecord(BaseModel):
    id: str
    case_id: str
    created_at: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    model_routes: list[AgentModelRoute]
    skill_names: list[str]
    estimated_cost_units: int = Field(ge=0)
    primary_strategy: str


class UsageCount(BaseModel):
    name: str
    count: int = Field(ge=0)


class AgentMetrics(BaseModel):
    total_runs: int = Field(ge=0)
    average_cost_units: float = Field(ge=0)
    high_risk_runs: int = Field(ge=0)
    provider_usage: list[UsageCount]
    skill_usage: list[UsageCount]
    recent_runs: list[AgentRunRecord]


class ModelProviderStatus(BaseModel):
    provider: str
    configured: bool
    base_url: str | None
    default_models: dict[str, str]
    roles: list[str]
    missing_env: list[str]
    adapter: str
    health: str


class ModelGatewayStatus(BaseModel):
    providers: list[ModelProviderStatus]
    dry_run_default: bool
    note: str


class ModelInvocationRequest(BaseModel):
    case_id: str | None = None
    provider: str
    role: str
    prompt: str
    system_prompt: str = ""
    dry_run: bool = True
    temperature: float = Field(default=0.2, ge=0, le=2)


class ModelInvocationResult(BaseModel):
    provider: str
    role: str
    selected_model: str
    configured: bool
    dry_run: bool
    request_payload: dict[str, object]
    audit_id: str | None = None
    response_text: str | None = None
    error: str | None = None


class CaseLlmReviewRequest(BaseModel):
    provider: str = "LocalReview"
    role: str = "本地结构化复核"
    temperature: float = Field(default=0.2, ge=0, le=2)


class CaseLlmReviewResult(BaseModel):
    case_id: str
    provider: str
    role: str
    selected_model: str
    configured: bool
    audit_id: str
    structured_review: dict[str, object]
    response_text: str


class CaseLlmReportRequest(BaseModel):
    provider: str = "LocalReport"
    role: str = "本地报告生成"
    temperature: float = Field(default=0.2, ge=0, le=2)


class KnowledgeSearchResult(BaseModel):
    id: str
    title: str
    source: str
    category: str
    content: str
    score: float = Field(ge=0)
    evidence_id: str | None = None
    source_url: str | None = None


class CaseLlmReportResult(BaseModel):
    case_id: str
    provider: str
    role: str
    selected_model: str
    configured: bool
    audit_id: str
    markdown: str
    knowledge_refs: list[KnowledgeSearchResult]


class ModelInvocationAudit(BaseModel):
    id: str
    case_id: str | None
    provider: str
    role: str
    model: str
    status: str
    request_payload: dict[str, object]
    response_text: str | None = None
    error: str | None = None
    latency_ms: int = Field(ge=0)
    token_usage: dict[str, object] = Field(default_factory=dict)
    created_at: str


class FullAnalysis(BaseModel):
    case: CaseSample
    analysis: MultimodalAnalysis
    evidence_chain: list[EvidenceItem]
    risk: RiskAssessment
    disposal: DisposalSuggestion
    report: ReportDraft
    agent: AgentOrchestration


class CaseAsset(BaseModel):
    id: str
    case_id: str
    filename: str
    content_type: str
    size_bytes: int = Field(ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    sha256: str
    storage_path: str
    preview_url: str
    created_at: str


class UrlCaptureRequest(BaseModel):
    url: str


class WebEvidenceSnapshot(BaseModel):
    id: str
    case_id: str
    requested_url: str
    final_url: str
    title: str
    text: str
    sha256: str
    status: str
    error: str | None = None
    html_path: str
    text_path: str
    screenshot_path: str | None = None
    screenshot_url: str | None = None
    created_at: str


class CaseEvidenceBundle(BaseModel):
    case_id: str
    assets: list[CaseAsset]
    snapshots: list[WebEvidenceSnapshot]
    evidence_items: list[EvidenceItem]


class RealMultimodalAnalysisResult(BaseModel):
    asset_id: str
    provider: str
    selected_model: str
    audit_id: str
    structured: dict[str, object]
    response_text: str


class RealCaseAnalysisResult(BaseModel):
    case: CaseSample
    assets: list[CaseAsset]
    snapshots: list[WebEvidenceSnapshot]
    multimodal_results: list[RealMultimodalAnalysisResult]
    evidence_chain: list[EvidenceItem]
    baseline_risk: RiskAssessment
    text_risk_model: dict[str, object] = Field(default_factory=dict)
    vision_evidence_models: dict[str, object] = Field(default_factory=dict)
    fusion_model: dict[str, object] = Field(default_factory=dict)
    structured_review: dict[str, object]
    review_audit_id: str
    report_markdown: str
    report_audit_id: str
    knowledge_refs: list[KnowledgeSearchResult]


class PropagationDisturbanceFinding(BaseModel):
    name: str
    severity: str
    score: float = Field(ge=0, le=1)
    evidence: str


class ImageForensicsAssetResult(BaseModel):
    asset_id: str
    filename: str
    sha256: str
    width: int | None = None
    height: int | None = None
    content_type: str
    size_bytes: int = Field(ge=0)
    preview_url: str
    gpt_image2_probability: float | None = Field(default=None, ge=0, le=1)
    top_candidate: str
    confidence: float = Field(ge=0, le=1)
    candidate_distribution: list[dict[str, object]]
    candidate_ranking: list[dict[str, object]] = Field(default_factory=list)
    review_recommendation: dict[str, object] = Field(default_factory=dict)
    disturbances: list[PropagationDisturbanceFinding]
    feature_summary: dict[str, object]
    top_contributions: list[dict[str, object]]
    interpretation: list[str]
    limitations: list[str]


class ImageForensicsResult(BaseModel):
    case_id: str
    research_target: str
    trained: bool
    model_id: str | None = None
    model_kind: str | None = None
    asset_results: list[ImageForensicsAssetResult]
    aggregate: dict[str, object]
    recommended_next_steps: list[str]
    application_context: str


class TamperSuspectedRegion(BaseModel):
    region_id: str
    label: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    cue_type: str
    confidence: float = Field(ge=0, le=1)
    visible_cues: list[str]
    signal_sources: list[str] = Field(default_factory=list)


class TamperDocumentFields(BaseModel):
    document_type: str
    sensitive_fields: list[str]


class TamperPatchSignal(BaseModel):
    region_id: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    signal_type: str
    score: float = Field(ge=0, le=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    explanation: str


class TamperForensicsAssetResult(BaseModel):
    asset_id: str
    filename: str
    sha256: str
    width: int | None = None
    height: int | None = None
    content_type: str
    size_bytes: int = Field(ge=0)
    preview_url: str
    tamper_risk: Literal["low", "medium", "high"]
    top_cue_type: str
    confidence: float = Field(ge=0, le=1)
    suspected_regions: list[TamperSuspectedRegion]
    visible_cues: list[str]
    document_fields: TamperDocumentFields
    interpretation: list[str]
    limitations: list[str]
    review_suggestions: list[str]
    feature_summary: dict[str, object] = Field(default_factory=dict)
    analysis_layers: list[str] = Field(default_factory=list)
    patch_signals: list[TamperPatchSignal] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    audit_trace: list[str] = Field(default_factory=list)


class TamperForensicsResult(BaseModel):
    case_id: str
    research_target: str
    trained: bool
    model_or_rule_version: str
    asset_results: list[TamperForensicsAssetResult]
    aggregate: dict[str, object]
    recommended_next_steps: list[str]
    application_context: str


class CaseAuditBundle(BaseModel):
    case_id: str
    invocations: list[ModelInvocationAudit]
    assets: list[CaseAsset]
    snapshots: list[WebEvidenceSnapshot]


class LocalVisionTrainingSample(BaseModel):
    case_id: str
    asset_id: str
    image_path: str
    image_sha256: str
    prompt: str
    response: dict[str, object]
    manual_label: str
    manual_risk_score: int | None = Field(default=None, ge=0, le=100)
    created_at: str


class LocalVisionTrainingDataset(BaseModel):
    id: str
    created_at: str
    model_target: str
    sample_count: int = Field(ge=0)
    format: str
    samples: list[LocalVisionTrainingSample]
    note: str


class LocalVisionTrainingStats(BaseModel):
    sample_count: int = Field(ge=0)
    labeled_sample_count: int = Field(ge=0)
    unlabeled_sample_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    average_manual_risk_score: float | None = None
    label_distribution: dict[str, int]
    export_ready: bool
    training_ready: bool
    note: str


class CaseImportRequest(BaseModel):
    case_id: str


class CaseRequest(BaseModel):
    case_id: str


class FeatureWeight(BaseModel):
    name: str
    description: str
    weight: float
    direction: str


class LocalVisionCalibrationRunRequest(BaseModel):
    epochs: int = Field(default=700, ge=50, le=5000)
    learning_rate: float = Field(default=0.04, ge=0.001, le=0.5)
    l2: float = Field(default=0.02, ge=0, le=1)
    min_samples: int = Field(default=4, ge=2, le=200)


class LocalVisionCalibrationRunResult(BaseModel):
    id: str
    created_at: str
    model_kind: str
    status: str
    sample_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    epochs: int = Field(ge=0)
    learning_rate: float
    train_mae: float = Field(ge=0)
    validation_mae: float = Field(ge=0)
    accuracy_within_10: float = Field(ge=0, le=1)
    label_distribution: dict[str, int]
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    training_trace: list[str]
    model_card: dict[str, object] = Field(default_factory=dict)


class LocalVisionTrainingStatus(BaseModel):
    trained: bool
    active_model_id: str | None
    latest_run: LocalVisionCalibrationRunResult | None
    dataset: LocalVisionTrainingStats
    note: str


class ExternalTrainingSample(BaseModel):
    id: str
    dataset_name: str
    source: str
    source_url: str | None = None
    task_type: str = "text_risk"
    split: str = "train"
    title: str
    content: str
    image_path: str | None = None
    image_url: str | None = None
    image_sha256: str | None = None
    image_available: bool = False
    label: str
    risk_score: int = Field(ge=0, le=100)
    scenario: str
    raw_payload: dict[str, object] = Field(default_factory=dict)
    created_at: str


class ExternalDatasetImportRequest(BaseModel):
    dataset_name: str
    source: str = "HuggingFace/local"
    source_url: str | None = None
    source_path: str | None = None
    rows: list[dict[str, object]] = Field(default_factory=list)
    format: str = "auto"
    task_type: str = "text_risk"
    split: str = "train"
    image_root: str | None = None
    image_path_column: str | None = None
    image_url_column: str | None = None
    text_columns: list[str] = Field(
        default_factory=lambda: [
            "text",
            "content",
            "claim",
            "body",
            "news",
            "tweet",
            "微博正文",
            "文本",
        ]
    )
    title_column: str | None = "title"
    label_column: str = "label"
    label_schema: dict[str, int] = Field(default_factory=dict)
    risk_score_column: str | None = "risk_score"
    scenario_column: str | None = "scenario"
    positive_label_values: list[str] = Field(
        default_factory=lambda: ["1", "true", "fake", "false", "rumor", "谣言", "虚假", "不实"]
    )
    negative_label_values: list[str] = Field(
        default_factory=lambda: ["0", "real", "true news", "non-rumor", "nonrumor", "真实", "事实", "辟谣"]
    )
    default_positive_score: int = Field(default=82, ge=0, le=100)
    default_negative_score: int = Field(default=18, ge=0, le=100)
    limit: int = Field(default=5000, ge=1, le=50000)


class ExternalDatasetImportResult(BaseModel):
    dataset_name: str
    source: str
    task_type: str = "text_risk"
    imported_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    sample_count_after_import: int = Field(ge=0)
    image_available_count: int = Field(default=0, ge=0)
    label_distribution: dict[str, int]
    examples: list[ExternalTrainingSample]
    note: str


class ExternalDatasetSourceSummary(BaseModel):
    dataset_name: str
    source: str
    source_url: str | None = None
    task_type: str = "text_risk"
    sample_count: int = Field(ge=0)
    image_available_count: int = Field(default=0, ge=0)
    label_distribution: dict[str, int]
    latest_import_at: str | None = None


class TrainingTaskStatus(BaseModel):
    task_type: str
    sample_count: int = Field(ge=0)
    image_available_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    training_ready: bool
    sources: list[ExternalDatasetSourceSummary]
    note: str


class TrainingDataStatus(BaseModel):
    external_sample_count: int = Field(ge=0)
    labeled_user_case_count: int = Field(ge=0)
    eligible_sample_count: int = Field(ge=0)
    demo_case_count: int = Field(ge=0)
    training_ready: bool
    sources: list[ExternalDatasetSourceSummary]
    tasks: list[TrainingTaskStatus] = Field(default_factory=list)
    recommended_huggingface_datasets: list[dict[str, str]]
    note: str


class TrainingRunRequest(BaseModel):
    epochs: int = Field(default=900, ge=50, le=5000)
    learning_rate: float = Field(default=0.045, ge=0.001, le=0.5)
    l2: float = Field(default=0.015, ge=0, le=1)
    include_augmented_samples: bool = True


class TrainingRunResult(BaseModel):
    id: str
    created_at: str
    model_kind: str
    status: str
    sample_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    epochs: int = Field(ge=0)
    learning_rate: float
    train_mae: float = Field(ge=0)
    validation_mae: float = Field(ge=0)
    train_rmse: float = Field(ge=0)
    validation_rmse: float = Field(ge=0)
    accuracy_within_10: float = Field(ge=0, le=1)
    label_distribution: dict[str, int]
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    training_trace: list[str]
    model_card: dict[str, object] = Field(default_factory=dict)
    task_metrics: dict[str, object] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)


class TrainingStatus(BaseModel):
    trained: bool
    active_model_id: str | None
    latest_run: TrainingRunResult | None
    training_data: TrainingDataStatus
    note: str


class FeatureCacheRecord(BaseModel):
    id: str
    cache_key: str
    extractor_version: str
    modality: str
    sha256: str | None = None
    payload: dict[str, object]
    created_at: str


class VisionTrainingRunRequest(BaseModel):
    task_type: str = "vision_aigc"
    epochs: int = Field(default=700, ge=50, le=5000)
    learning_rate: float = Field(default=0.04, ge=0.001, le=0.5)
    l2: float = Field(default=0.02, ge=0, le=1)
    min_samples: int = Field(default=20, ge=2, le=5000)
    max_training_samples: int = Field(default=0, ge=0, le=50000)
    activation_mode: Literal["candidate", "activate", "activate_if_passes_gate"] | None = None
    experiment_profile: Literal[
        "standard_attribution",
        "binary_generated_gate",
        "gpt_image2_ovr",
        "mainstream_five_attribution",
        "multi_generator_label_covered",
        "clean_origin_attribution",
        "social_propagation_robustness",
    ] = "standard_attribution"
    validation_strategy: Literal["class_stratified", "source_holdout"] = "class_stratified"
    source_holdout_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    min_source_holdout_samples: int = Field(default=20, ge=2, le=1000)
    enable_perturbation_augmentation: bool = False
    augmentation_conditions: list[str] = Field(
        default_factory=lambda: ["jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark"]
    )
    max_augmented_samples: int = Field(default=2500, ge=0, le=50000)
    enable_open_set_unknown: bool = False
    unknown_threshold_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    open_set_min_margin: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionTrainingRunResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    model_kind: str
    status: str
    sample_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    epochs: int = Field(ge=0)
    learning_rate: float
    train_mae: float = Field(ge=0)
    validation_mae: float = Field(ge=0)
    train_rmse: float = Field(default=0.0, ge=0)
    validation_rmse: float = Field(default=0.0, ge=0)
    accuracy_within_10: float = Field(ge=0, le=1)
    risk_level_accuracy: float = Field(default=0.0, ge=0, le=1)
    label_distribution: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    training_trace: list[str]
    model_card: dict[str, object] = Field(default_factory=dict)


class VisionRobustnessRunRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    limit: int = Field(default=120, ge=2, le=1000)
    conditions: list[str] = Field(
        default_factory=lambda: [
            "clean",
            "jpeg_q85",
            "jpeg_q60",
            "screenshot_resave",
            "center_crop",
            "watermark",
        ]
    )
    include_sample_predictions: bool = False


class VisionRobustnessConditionResult(BaseModel):
    condition: str
    perturbation: str
    sample_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    gpt_image2_precision: float = Field(ge=0, le=1)
    gpt_image2_recall: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    confidence_delta_from_clean: float | None = None
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    sample_predictions: list[dict[str, object]] = Field(default_factory=list)


class VisionRobustnessRunResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    model_id: str
    model_kind: str
    sample_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    conditions: list[VisionRobustnessConditionResult]
    feature_groups: dict[str, list[str]]
    conclusions: list[str]
    limitations: list[str]
    model_card_update: dict[str, object] = Field(default_factory=dict)


class VisionAugmentationCacheWarmupRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    limit: int = Field(default=120, ge=1, le=5000)
    conditions: list[str] = Field(default_factory=lambda: ["jpeg_q85", "jpeg_q60", "watermark"])


class VisionAugmentationCacheWarmupResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    sample_count: int = Field(ge=0)
    requested_conditions: list[str]
    condition_counts: dict[str, int]
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    does_not_train: bool = True
    does_not_change_active_model: bool = True
    feature_cache_policy: str
    note: str


class VisionSourceHoldoutRunRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    experiment_profile: Literal[
        "standard_attribution",
        "binary_generated_gate",
        "gpt_image2_ovr",
        "mainstream_five_attribution",
        "multi_generator_label_covered",
        "clean_origin_attribution",
        "social_propagation_robustness",
    ] = "standard_attribution"
    min_train_samples: int = Field(default=4, ge=2, le=50000)
    min_holdout_samples: int = Field(default=1, ge=1, le=5000)
    max_holdout_groups: int = Field(default=12, ge=1, le=100)
    sample_limit: int = Field(default=500, ge=0, le=50000)
    holdout_key: str = Field(default="dataset_source")
    enable_perturbation_augmentation: bool = False
    augmentation_conditions: list[str] = Field(default_factory=lambda: ["jpeg_q85", "jpeg_q60"])
    max_augmented_samples: int = Field(default=1000, ge=0, le=50000)
    enable_open_set_unknown: bool = False
    unknown_threshold_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    open_set_min_margin: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionSourceHoldoutGroupResult(BaseModel):
    holdout_group: str
    train_count: int = Field(ge=0)
    holdout_count: int = Field(ge=0)
    seen_class_holdout_count: int = Field(default=0, ge=0)
    unseen_holdout_count: int = Field(default=0, ge=0)
    unseen_holdout_labels: list[str] = Field(default_factory=list)
    train_label_distribution: dict[str, int]
    holdout_label_distribution: dict[str, int]
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    gpt_image2_precision: float = Field(ge=0, le=1)
    gpt_image2_recall: float = Field(ge=0, le=1)
    seen_class_accuracy: float = Field(default=0.0, ge=0, le=1)
    seen_class_macro_f1: float = Field(default=0.0, ge=0, le=1)
    seen_class_gpt_image2_recall: float = Field(default=0.0, ge=0, le=1)
    binary_accuracy: float = Field(default=0.0, ge=0, le=1)
    binary_macro_f1: float = Field(default=0.0, ge=0, le=1)
    generated_recall: float = Field(default=0.0, ge=0, le=1)
    generated_support: int = Field(default=0, ge=0)
    generated_false_negative_count: int = Field(default=0, ge=0)
    real_recall: float = Field(default=0.0, ge=0, le=1)
    real_false_positive_rate: float = Field(default=0.0, ge=0, le=1)
    real_support: int = Field(default=0, ge=0)
    real_false_positive_count: int = Field(default=0, ge=0)
    average_confidence: float = Field(ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


class VisionSourceHoldoutRunResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    holdout_key: str
    sample_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    source_count: int = Field(ge=0)
    groups: list[VisionSourceHoldoutGroupResult]
    aggregate: dict[str, float]
    protocol: dict[str, object]
    conclusions: list[str]
    limitations: list[str]


class VisionFeatureAblationRunRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    limit: int = Field(default=500, ge=4, le=5000)
    min_samples: int = Field(default=4, ge=4, le=5000)
    feature_sets: list[str] = Field(
        default_factory=lambda: [
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
        ]
    )


class VisionFeatureAblationResult(BaseModel):
    feature_set: str
    feature_count: int = Field(ge=0)
    removed_feature_group: str | None = None
    selected_feature_group: str | None = None
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    gpt_image2_precision: float = Field(ge=0, le=1)
    gpt_image2_recall: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


class VisionFeatureAblationRunResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    sample_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    feature_groups: dict[str, list[str]]
    results: list[VisionFeatureAblationResult]
    deltas_from_all: dict[str, dict[str, float]]
    conclusions: list[str]
    limitations: list[str]


class VisionAntiCheatAuditRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    holdout_key: Literal["dataset", "source", "dataset_source"] = "dataset"
    max_holdout_groups: int = Field(default=6, ge=1, le=50)
    min_holdout_samples: int = Field(default=4, ge=1, le=5000)
    source_holdout_sample_limit: int = Field(default=320, ge=0, le=50000)
    feature_ablation_limit: int = Field(default=240, ge=4, le=5000)
    include_source_holdout: bool = True
    include_feature_ablation: bool = True


class VisionAntiCheatAuditResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    active_model_id: str | None
    training_validation: dict[str, object]
    validation_protocol: dict[str, object]
    suspicious_feature_names: list[str]
    leakage_checks: dict[str, object]
    source_holdout: VisionSourceHoldoutRunResult | None = None
    feature_ablation: VisionFeatureAblationRunResult | None = None
    verdict: str
    cautions: list[str]
    recommended_claims: list[str]


class VisionTrainingRunRecord(BaseModel):
    run: VisionTrainingRunResult
    is_active: bool


class VisionTrainingActivationRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    run_id: str


class VisionTrainingActivationResult(BaseModel):
    task_type: str
    active_model_id: str
    previous_active_model_id: str | None
    activated_run: VisionTrainingRunResult
    note: str


class VisionCandidateEvaluationRequest(BaseModel):
    task_type: str = "vision_generator_attribution"
    candidate_model_id: str
    limit: int = Field(default=120, ge=2, le=1000)
    conditions: list[str] = Field(
        default_factory=lambda: [
            "clean",
            "jpeg_q85",
            "jpeg_q60",
            "screenshot_resave",
            "center_crop",
            "watermark",
        ]
    )
    include_source_holdout: bool = True
    include_feature_ablation: bool = True
    activate_if_passes_gate: bool = False


class VisionCandidateEvaluationResult(BaseModel):
    id: str
    created_at: str
    task_type: str
    active_model_id_before: str | None
    candidate_model_id: str
    active_model_id_after: str | None
    activated: bool
    sample_count: int = Field(ge=0)
    label_distribution: dict[str, int]
    conditions: list[dict[str, object]]
    active_summary: dict[str, object]
    candidate_summary: dict[str, object]
    gate: dict[str, object]
    supporting_experiments: dict[str, object] = Field(default_factory=dict)
    limitations: list[str]


class VisionTrainingStatus(BaseModel):
    task_type: str
    trained: bool
    active_model_id: str | None
    latest_run: VisionTrainingRunResult | None
    latest_candidate: VisionTrainingRunResult | None = None
    candidate_vs_active: dict[str, object] = Field(default_factory=dict)
    data: TrainingTaskStatus
    note: str


class VisionCompetitionSummary(BaseModel):
    task_type: str
    project_title: str
    active_model_id: str | None
    active_model_kind: str | None
    latest_candidate_id: str | None
    training_pool: dict[str, object]
    validation_metrics: dict[str, object]
    augmentation_protocol: dict[str, object]
    robustness_headline: dict[str, object]
    model_lifecycle: dict[str, object]
    feature_groups: dict[str, list[str]]
    limitations: list[str]
    recommended_next_data: list[str]
    narrative_points: list[str]
    note: str


class FusionTrainingRunRequest(BaseModel):
    epochs: int = Field(default=800, ge=50, le=5000)
    learning_rate: float = Field(default=0.04, ge=0.001, le=0.5)
    l2: float = Field(default=0.02, ge=0, le=1)
    min_samples: int = Field(default=20, ge=2, le=5000)


class FusionTrainingRunResult(BaseModel):
    id: str
    created_at: str
    model_kind: str
    status: str
    sample_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    epochs: int = Field(ge=0)
    learning_rate: float
    train_mae: float = Field(ge=0)
    validation_mae: float = Field(ge=0)
    train_rmse: float = Field(default=0.0, ge=0)
    validation_rmse: float = Field(default=0.0, ge=0)
    accuracy_within_10: float = Field(ge=0, le=1)
    risk_level_accuracy: float = Field(default=0.0, ge=0, le=1)
    label_distribution: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    training_trace: list[str]
    model_card: dict[str, object] = Field(default_factory=dict)


class FusionTrainingStatus(BaseModel):
    trained: bool
    active_model_id: str | None
    latest_run: FusionTrainingRunResult | None
    data: TrainingTaskStatus
    note: str


class DemoEvaluationCaseResult(BaseModel):
    case_id: str
    title: str
    text_only: dict[str, object]
    vision_only: dict[str, object]
    fusion: dict[str, object]


class DemoEvaluationResult(BaseModel):
    id: str
    created_at: str
    demo_case_count: int = Field(ge=0)
    results: list[DemoEvaluationCaseResult]
    note: str
