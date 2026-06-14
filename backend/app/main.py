from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import (
    analyze_multimodal,
    assess_risk,
    build_agent_orchestration,
    build_evidence_chain,
    generate_report,
    run_full_analysis,
    suggest_disposal,
)
from app.dataset_importer import import_external_dataset
from app.evidence_service import (
    DATA_ROOT,
    EvidenceError,
    capture_url,
    save_uploaded_asset,
)
from app.image_forensics import run_image_forensics
from app.llm_workflows import LlmOutputParseError, run_llm_report, run_llm_review
from app.local_vision_training import (
    build_local_vision_jsonl,
    build_local_vision_training_dataset,
    build_local_vision_training_stats,
    get_local_vision_training_status,
    train_local_vision_calibrator,
)
from app.model_gateway import ModelGatewayError, get_model_gateway_status, invoke_model
from app.multimodal_training import (
    activate_vision_candidate,
    evaluate_demo_cases,
    evaluate_vision_candidate,
    get_fusion_training_status,
    get_vision_competition_summary,
    get_vision_training_status,
    list_vision_training_run_records,
    run_vision_feature_ablation_experiment,
    run_vision_anti_cheat_audit,
    run_vision_source_holdout_experiment,
    run_vision_robustness_experiment,
    train_fusion_head,
    train_vision_evidence_head,
    warmup_vision_augmentation_cache,
)
from app.models import (
    AgentMetrics,
    AgentOrchestration,
    AgentRunRecord,
    CaseAsset,
    CaseAuditBundle,
    CaseEvidenceBundle,
    CaseLlmReportRequest,
    CaseLlmReportResult,
    CaseLlmReviewRequest,
    CaseLlmReviewResult,
    CaseCreateRequest,
    ExternalDatasetImportRequest,
    ExternalDatasetImportResult,
    ExternalTrainingSample,
    CaseImportRequest,
    CaseLabelRequest,
    CaseRequest,
    CaseSample,
    DisposalSuggestion,
    DemoEvaluationResult,
    EvidenceItem,
    FullAnalysis,
    FusionTrainingRunRequest,
    FusionTrainingRunResult,
    FusionTrainingStatus,
    ImageForensicsResult,
    KnowledgeSearchResult,
    LocalVisionCalibrationRunRequest,
    LocalVisionCalibrationRunResult,
    LocalVisionTrainingDataset,
    LocalVisionTrainingStats,
    LocalVisionTrainingStatus,
    ModelGatewayStatus,
    ModelInvocationRequest,
    ModelInvocationResult,
    ModelInvocationAudit,
    MultimodalAnalysis,
    ReportDraft,
    RiskAssessment,
    RealCaseAnalysisResult,
    TrainingDataStatus,
    TrainingRunRequest,
    TrainingRunResult,
    TrainingStatus,
    UrlCaptureRequest,
    VisionAugmentationCacheWarmupRequest,
    VisionAugmentationCacheWarmupResult,
    VisionAntiCheatAuditRequest,
    VisionAntiCheatAuditResult,
    VisionCandidateEvaluationRequest,
    VisionCandidateEvaluationResult,
    VisionCompetitionSummary,
    VisionFeatureAblationRunRequest,
    VisionFeatureAblationRunResult,
    VisionRobustnessRunRequest,
    VisionRobustnessRunResult,
    VisionSourceHoldoutRunRequest,
    VisionSourceHoldoutRunResult,
    VisionTrainingActivationRequest,
    VisionTrainingActivationResult,
    VisionTrainingRunRequest,
    VisionTrainingRunResult,
    VisionTrainingRunRecord,
    VisionTrainingStatus,
    WebEvidenceSnapshot,
)
from app.real_analysis import RealAnalysisInputError, run_real_case_analysis
from app.risk_model import train_risk_model, training_status_note
from app.storage import (
    create_case_sample,
    delete_case_sample,
    delete_image_forensics_result,
    get_case_evidence_bundle,
    get_agent_metrics,
    get_latest_training_run,
    get_training_data_status,
    initialize_database,
    list_agent_runs,
    list_case_assets,
    list_case_samples,
    list_external_training_samples,
    list_llm_invocations,
    list_local_vision_training_runs,
    list_training_runs,
    list_web_snapshots,
    load_case_sample,
    load_image_forensics_result,
    record_agent_run,
    save_image_forensics_result,
    search_knowledge,
    update_case_label,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(
    title="AIGC公共安全谣言多模态证据链智能研判系统",
    version="0.1.0",
    description="Semi-final prototype for public-safety rumor risk assessment.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *[origin.strip() for origin in os.getenv("SMARTPOLICE_CORS_ORIGINS", "").split(",") if origin.strip()],
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/evidence/files", StaticFiles(directory=str(DATA_ROOT)), name="evidence-files")
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(index_path)


@app.get("/cases", response_model=list[CaseSample])
def list_cases() -> list[CaseSample]:
    return list_case_samples()


@app.post("/cases/import", response_model=CaseSample)
def import_case(payload: CaseImportRequest) -> CaseSample:
    return _case_or_404(payload.case_id)


@app.post("/cases", response_model=CaseSample)
def create_case(payload: CaseCreateRequest) -> CaseSample:
    try:
        return create_case_sample(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/cases/{case_id}", response_model=CaseSample)
def delete_case(case_id: str) -> CaseSample:
    try:
        return delete_case_sample(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc


@app.post("/cases/{case_id}/label", response_model=CaseSample)
def label_case(case_id: str, payload: CaseLabelRequest) -> CaseSample:
    try:
        return update_case_label(case_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc


@app.post("/cases/{case_id}/assets", response_model=CaseAsset)
async def upload_case_asset(case_id: str, file: UploadFile) -> CaseAsset:
    _case_or_404(case_id)
    try:
        asset = await save_uploaded_asset(case_id, file)
        delete_image_forensics_result(case_id)
        return asset
    except EvidenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/cases/{case_id}/sources/capture", response_model=WebEvidenceSnapshot)
def capture_case_source(
    case_id: str,
    payload: UrlCaptureRequest,
) -> WebEvidenceSnapshot:
    _case_or_404(case_id)
    try:
        return capture_url(case_id, payload.url)
    except EvidenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/cases/{case_id}/evidence", response_model=CaseEvidenceBundle)
def case_evidence(case_id: str) -> CaseEvidenceBundle:
    _case_or_404(case_id)
    return get_case_evidence_bundle(case_id)


@app.get("/cases/{case_id}/audit", response_model=CaseAuditBundle)
def case_audit(case_id: str, limit: int = 50) -> CaseAuditBundle:
    _case_or_404(case_id)
    return CaseAuditBundle(
        case_id=case_id,
        invocations=list_llm_invocations(case_id=case_id, limit=limit),
        assets=list_case_assets(case_id),
        snapshots=list_web_snapshots(case_id),
    )


@app.post("/cases/{case_id}/real-analysis", response_model=RealCaseAnalysisResult)
def case_real_analysis(case_id: str) -> RealCaseAnalysisResult:
    case = _case_or_404(case_id)
    try:
        return run_real_case_analysis(case)
    except RealAnalysisInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LlmOutputParseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc), "audit_id": exc.audit_id},
        ) from exc


@app.post("/cases/{case_id}/image-forensics", response_model=ImageForensicsResult)
def case_image_forensics(case_id: str) -> ImageForensicsResult:
    case = _case_or_404(case_id)
    assets = list_case_assets(case_id)
    if not assets:
        raise HTTPException(
            status_code=422,
            detail="请先上传至少一张图片/截图，再运行 GPT-image-2 图像取证检测。",
        )
    result = run_image_forensics(case, assets)
    save_image_forensics_result(result)
    return result


@app.get("/cases/{case_id}/image-forensics", response_model=ImageForensicsResult)
def cached_case_image_forensics(case_id: str) -> ImageForensicsResult:
    _case_or_404(case_id)
    result = load_image_forensics_result(case_id)
    if result is None:
        raise HTTPException(status_code=404, detail="暂无已保存的图像来源研判结果。")
    return result


@app.post("/analysis/multimodal", response_model=MultimodalAnalysis)
def multimodal_analysis(payload: CaseRequest) -> MultimodalAnalysis:
    case = _case_or_404(payload.case_id)
    return analyze_multimodal(case)


@app.post("/evidence/chain", response_model=list[EvidenceItem])
def evidence_chain(payload: CaseRequest) -> list[EvidenceItem]:
    case = _case_or_404(payload.case_id)
    analysis = analyze_multimodal(case)
    return build_evidence_chain(case, analysis)


@app.post("/risk/assess", response_model=RiskAssessment)
def risk_assessment(payload: CaseRequest) -> RiskAssessment:
    case = _case_or_404(payload.case_id)
    analysis = analyze_multimodal(case)
    evidence = build_evidence_chain(case, analysis)
    return assess_risk(case, evidence)


@app.post("/disposal/suggest", response_model=DisposalSuggestion)
def disposal_suggestion(payload: CaseRequest) -> DisposalSuggestion:
    case = _case_or_404(payload.case_id)
    analysis = analyze_multimodal(case)
    evidence = build_evidence_chain(case, analysis)
    risk = assess_risk(case, evidence)
    return suggest_disposal(case, risk)


@app.post("/reports/generate", response_model=ReportDraft)
def report_generation(payload: CaseRequest) -> ReportDraft:
    case = _case_or_404(payload.case_id)
    analysis = analyze_multimodal(case)
    evidence = build_evidence_chain(case, analysis)
    risk = assess_risk(case, evidence)
    disposal = suggest_disposal(case, risk)
    return generate_report(case, analysis, evidence, risk, disposal)


@app.post("/agent/orchestrate", response_model=AgentOrchestration)
def agent_orchestration(payload: CaseRequest) -> AgentOrchestration:
    case = _case_or_404(payload.case_id)
    analysis = analyze_multimodal(case)
    evidence = build_evidence_chain(case, analysis)
    risk = assess_risk(case, evidence)
    return build_agent_orchestration(case, analysis, evidence, risk)


@app.get("/agent/runs", response_model=list[AgentRunRecord])
def agent_runs(limit: int = 10) -> list[AgentRunRecord]:
    return list_agent_runs(limit)


@app.get("/agent/metrics", response_model=AgentMetrics)
def agent_metrics() -> AgentMetrics:
    return get_agent_metrics()


@app.get("/agent/model-gateway/status", response_model=ModelGatewayStatus)
def model_gateway_status() -> ModelGatewayStatus:
    return get_model_gateway_status()


@app.post("/agent/model-gateway/invoke", response_model=ModelInvocationResult)
def model_gateway_invoke(payload: ModelInvocationRequest) -> ModelInvocationResult:
    try:
        return invoke_model(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc), "audit_id": exc.audit_id},
        ) from exc


@app.get("/agent/model-gateway/invocations", response_model=list[ModelInvocationAudit])
def model_gateway_invocations(
    case_id: str | None = None,
    limit: int = 20,
) -> list[ModelInvocationAudit]:
    return list_llm_invocations(case_id=case_id, limit=limit)


@app.get("/knowledge/search", response_model=list[KnowledgeSearchResult])
def knowledge_search(query: str, limit: int = 5) -> list[KnowledgeSearchResult]:
    return search_knowledge(query, limit)


@app.post("/cases/{case_id}/llm-review", response_model=CaseLlmReviewResult)
def case_llm_review(
    case_id: str,
    payload: CaseLlmReviewRequest,
) -> CaseLlmReviewResult:
    case = _case_or_404(case_id)
    try:
        return run_llm_review(case, payload)
    except LlmOutputParseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc), "audit_id": exc.audit_id},
        ) from exc


@app.post("/cases/{case_id}/llm-report", response_model=CaseLlmReportResult)
def case_llm_report(
    case_id: str,
    payload: CaseLlmReportRequest,
) -> CaseLlmReportResult:
    case = _case_or_404(case_id)
    try:
        return run_llm_report(case, payload)
    except LlmOutputParseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc), "audit_id": exc.audit_id},
        ) from exc


@app.post("/analysis/full", response_model=FullAnalysis)
def full_analysis(payload: CaseRequest) -> FullAnalysis:
    case = _case_or_404(payload.case_id)
    result = run_full_analysis(case)
    record_agent_run(result)
    return result


@app.post("/training/run", response_model=TrainingRunResult)
def training_run(payload: TrainingRunRequest) -> TrainingRunResult:
    try:
        return train_risk_model(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/datasets/import", response_model=ExternalDatasetImportResult)
def training_dataset_import(
    payload: ExternalDatasetImportRequest,
) -> ExternalDatasetImportResult:
    try:
        return import_external_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/training/datasets/status", response_model=TrainingDataStatus)
def training_dataset_status(task_type: str | None = None) -> TrainingDataStatus:
    status = get_training_data_status()
    if not task_type:
        return status
    tasks = [task for task in status.tasks if task.task_type == task_type]
    sources = [source for source in status.sources if source.task_type == task_type]
    sample_count = sum(task.sample_count for task in tasks)
    image_count = sum(task.image_available_count for task in tasks)
    ready = any(task.training_ready for task in tasks)
    return status.model_copy(
        update={
            "external_sample_count": sample_count,
            "eligible_sample_count": sample_count,
            "training_ready": ready,
            "sources": sources,
            "tasks": tasks,
            "note": (
                tasks[0].note
                if tasks
                else f"尚未导入 {task_type} 任务样本；内置四方向样例不进入训练集。"
            ),
        }
    )


@app.get("/training/datasets/samples", response_model=list[ExternalTrainingSample])
def training_dataset_samples(
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    task_type: str | None = None,
    limit: int = 20,
) -> list[ExternalTrainingSample]:
    return list_external_training_samples(
        limit=limit,
        dataset_name=dataset_name or dataset_id,
        task_type=task_type,
    )


@app.post("/training/vision/run", response_model=VisionTrainingRunResult)
def vision_training_run(payload: VisionTrainingRunRequest) -> VisionTrainingRunResult:
    try:
        return train_vision_evidence_head(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/training/vision/status", response_model=VisionTrainingStatus)
def vision_training_status(task_type: str = "vision_aigc") -> VisionTrainingStatus:
    try:
        return get_vision_training_status(task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/training/vision/competition-summary", response_model=VisionCompetitionSummary)
def vision_competition_summary(
    task_type: str = "vision_generator_attribution",
) -> VisionCompetitionSummary:
    try:
        return get_vision_competition_summary(task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/training/vision/runs", response_model=list[VisionTrainingRunRecord])
def vision_training_runs(
    task_type: str = "vision_generator_attribution",
    limit: int = 10,
) -> list[VisionTrainingRunRecord]:
    try:
        return list_vision_training_run_records(task_type=task_type, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/activate", response_model=VisionTrainingActivationResult)
def vision_training_activate(payload: VisionTrainingActivationRequest) -> VisionTrainingActivationResult:
    try:
        return activate_vision_candidate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/evaluate-candidate", response_model=VisionCandidateEvaluationResult)
def vision_candidate_evaluation(payload: VisionCandidateEvaluationRequest) -> VisionCandidateEvaluationResult:
    try:
        return evaluate_vision_candidate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/robustness-run", response_model=VisionRobustnessRunResult)
def vision_robustness_run(payload: VisionRobustnessRunRequest) -> VisionRobustnessRunResult:
    try:
        return run_vision_robustness_experiment(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/augmentation-cache-warmup", response_model=VisionAugmentationCacheWarmupResult)
def vision_augmentation_cache_warmup(
    payload: VisionAugmentationCacheWarmupRequest,
) -> VisionAugmentationCacheWarmupResult:
    try:
        return warmup_vision_augmentation_cache(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/source-holdout-run", response_model=VisionSourceHoldoutRunResult)
def vision_source_holdout_run(payload: VisionSourceHoldoutRunRequest) -> VisionSourceHoldoutRunResult:
    try:
        return run_vision_source_holdout_experiment(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/feature-ablation-run", response_model=VisionFeatureAblationRunResult)
def vision_feature_ablation_run(payload: VisionFeatureAblationRunRequest) -> VisionFeatureAblationRunResult:
    try:
        return run_vision_feature_ablation_experiment(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/vision/anti-cheat-audit", response_model=VisionAntiCheatAuditResult)
def vision_anti_cheat_audit(payload: VisionAntiCheatAuditRequest) -> VisionAntiCheatAuditResult:
    try:
        return run_vision_anti_cheat_audit(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/fusion/run", response_model=FusionTrainingRunResult)
def fusion_training_run(payload: FusionTrainingRunRequest) -> FusionTrainingRunResult:
    try:
        return train_fusion_head(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/training/fusion/status", response_model=FusionTrainingStatus)
def fusion_training_status() -> FusionTrainingStatus:
    return get_fusion_training_status()


@app.post("/training/evaluation/demo-cases", response_model=DemoEvaluationResult)
def demo_case_evaluation() -> DemoEvaluationResult:
    return evaluate_demo_cases()


@app.get("/training/status", response_model=TrainingStatus)
def training_status() -> TrainingStatus:
    latest = get_latest_training_run()
    training_data = get_training_data_status()
    return TrainingStatus(
        trained=latest is not None,
        active_model_id=latest.id if latest else None,
        latest_run=latest,
        training_data=training_data,
        note=(
            training_status_note()
            if latest
            else f"尚未训练本地风险模型，研判使用规则特征兜底。{training_data.note}"
        ),
    )


@app.get("/training/runs", response_model=list[TrainingRunResult])
def training_runs(limit: int = 10) -> list[TrainingRunResult]:
    return list_training_runs(limit)


@app.get("/training/local-vision/dataset", response_model=LocalVisionTrainingDataset)
def local_vision_training_dataset(
    case_id: str | None = None,
    limit: int = 100,
) -> LocalVisionTrainingDataset:
    try:
        return build_local_vision_training_dataset(case_id=case_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc


@app.get("/training/local-vision/stats", response_model=LocalVisionTrainingStats)
def local_vision_training_stats(
    case_id: str | None = None,
    limit: int = 500,
) -> LocalVisionTrainingStats:
    try:
        return build_local_vision_training_stats(case_id=case_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc


@app.get("/training/local-vision/status", response_model=LocalVisionTrainingStatus)
def local_vision_training_status() -> LocalVisionTrainingStatus:
    return get_local_vision_training_status()


@app.get("/training/local-vision/dataset.jsonl", response_class=PlainTextResponse)
def local_vision_training_jsonl(
    case_id: str | None = None,
    limit: int = 500,
) -> PlainTextResponse:
    try:
        content = build_local_vision_jsonl(case_id=case_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc
    return PlainTextResponse(
        content=content,
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=local-vision-dataset.jsonl"},
    )


@app.post("/training/local-vision/run", response_model=LocalVisionCalibrationRunResult)
def local_vision_training_run(
    payload: LocalVisionCalibrationRunRequest,
) -> LocalVisionCalibrationRunResult:
    try:
        return train_local_vision_calibrator(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/training/local-vision/runs",
    response_model=list[LocalVisionCalibrationRunResult],
)
def local_vision_training_runs(
    limit: int = 10,
) -> list[LocalVisionCalibrationRunResult]:
    return list_local_vision_training_runs(limit)


def _case_or_404(case_id: str) -> CaseSample:
    try:
        return load_case_sample(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc


@app.get("/{path:path}", include_in_schema=False)
def frontend_fallback(path: str) -> FileResponse:
    if path.startswith(("api/", "cases/", "analysis/", "training/", "agent/", "models/", "evidence/")):
        raise HTTPException(status_code=404, detail="Not found")
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(index_path)
