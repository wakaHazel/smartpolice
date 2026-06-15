import type {
  AgentMetrics,
  CaseAsset,
  CaseAuditBundle,
  CaseCreateRequest,
  CaseEvidenceBundle,
  CaseLabelRequest,
  CaseSample,
  DemoEvaluationResult,
  FusionTrainingRunRequest,
  FusionTrainingRunResult,
  FusionTrainingStatus,
  FullAnalysis,
  ImageForensicsResult,
  LocalVisionCalibrationRunRequest,
  LocalVisionCalibrationRunResult,
  LocalVisionTrainingDataset,
  LocalVisionTrainingStatus,
  ExternalDatasetImportRequest,
  ExternalDatasetImportResult,
  TrainingDataStatus,
  ModelGatewayStatus,
  ModelInvocationRequest,
  ModelInvocationResult,
  RealCaseAnalysisResult,
  TrainingRunRequest,
  TrainingRunResult,
  TrainingStatus,
  VisionTrainingRunRequest,
  VisionTrainingRunResult,
  VisionCompetitionSummary,
  VisionTrainingStatus,
  WebEvidenceSnapshot,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.PROD ? "" : "/api");

export async function fetchCases(): Promise<CaseSample[]> {
  const response = await fetch(`${API_BASE}/cases`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载案例"));
  }
  return response.json() as Promise<CaseSample[]>;
}

export async function createCase(payload: CaseCreateRequest): Promise<CaseSample> {
  const response = await fetch(`${API_BASE}/cases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "样本录入失败"));
  }
  return response.json() as Promise<CaseSample>;
}

export async function deleteCase(caseId: string): Promise<CaseSample> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "案例删除失败"));
  }
  return response.json() as Promise<CaseSample>;
}

export async function labelCase(
  caseId: string,
  payload: CaseLabelRequest,
): Promise<CaseSample> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "人工标注保存失败"));
  }
  return response.json() as Promise<CaseSample>;
}

export async function runFullAnalysis(caseId: string): Promise<FullAnalysis> {
  const response = await fetch(`${API_BASE}/analysis/full`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId }),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "研判分析失败"));
  }
  return response.json() as Promise<FullAnalysis>;
}

export async function uploadCaseAsset(caseId: string, file: File): Promise<CaseAsset> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/assets`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "图片上传失败"));
  }
  return response.json() as Promise<CaseAsset>;
}

export async function captureCaseSource(
  caseId: string,
  url: string,
): Promise<WebEvidenceSnapshot> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/sources/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "URL 取证失败"));
  }
  return response.json() as Promise<WebEvidenceSnapshot>;
}

export async function fetchCaseEvidence(caseId: string): Promise<CaseEvidenceBundle> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/evidence`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载案例证据"));
  }
  return response.json() as Promise<CaseEvidenceBundle>;
}

export async function fetchCaseAudit(caseId: string): Promise<CaseAuditBundle> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/audit`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载案例审计"));
  }
  return response.json() as Promise<CaseAuditBundle>;
}

export async function runRealAnalysis(caseId: string): Promise<RealCaseAnalysisResult> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/real-analysis`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "真实研判失败"));
  }
  return response.json() as Promise<RealCaseAnalysisResult>;
}

export async function runImageForensics(caseId: string): Promise<ImageForensicsResult> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/image-forensics`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "图像来源研判失败"));
  }
  return response.json() as Promise<ImageForensicsResult>;
}

export async function fetchImageForensics(caseId: string): Promise<ImageForensicsResult | null> {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/image-forensics?_=${Date.now()}`, {
    cache: "no-store",
  });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载已保存的图像来源研判结果"));
  }
  return response.json() as Promise<ImageForensicsResult>;
}

export async function runTraining(
  payload: TrainingRunRequest,
): Promise<TrainingRunResult> {
  const response = await fetch(`${API_BASE}/training/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "训练任务失败"));
  }
  return response.json() as Promise<TrainingRunResult>;
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const response = await fetch(`${API_BASE}/training/status`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载训练状态"));
  }
  return response.json() as Promise<TrainingStatus>;
}

export async function fetchTrainingRuns(): Promise<TrainingRunResult[]> {
  const response = await fetch(`${API_BASE}/training/runs`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载训练历史"));
  }
  return response.json() as Promise<TrainingRunResult[]>;
}

export async function importExternalDataset(
  payload: ExternalDatasetImportRequest,
): Promise<ExternalDatasetImportResult> {
  const response = await fetch(`${API_BASE}/training/datasets/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "外部数据集导入失败"));
  }
  return response.json() as Promise<ExternalDatasetImportResult>;
}

export async function fetchTrainingDataStatus(taskType?: string): Promise<TrainingDataStatus> {
  const params = new URLSearchParams();
  if (taskType) {
    params.set("task_type", taskType);
  }
  const query = params.toString();
  const response = await fetch(`${API_BASE}/training/datasets/status${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载训练数据状态"));
  }
  return response.json() as Promise<TrainingDataStatus>;
}

export async function runVisionTraining(
  payload: VisionTrainingRunRequest,
): Promise<VisionTrainingRunResult> {
  const response = await fetch(`${API_BASE}/training/vision/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "视觉证据头训练失败"));
  }
  return response.json() as Promise<VisionTrainingRunResult>;
}

export async function fetchVisionTrainingStatus(
  taskType: string,
): Promise<VisionTrainingStatus> {
  const params = new URLSearchParams({ task_type: taskType });
  const response = await fetch(`${API_BASE}/training/vision/status?${params.toString()}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载视觉训练状态"));
  }
  return response.json() as Promise<VisionTrainingStatus>;
}

export async function fetchVisionCompetitionSummary(
  taskType = "vision_generator_attribution",
): Promise<VisionCompetitionSummary> {
  const params = new URLSearchParams({ task_type: taskType });
  const response = await fetch(`${API_BASE}/training/vision/competition-summary?${params.toString()}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载比赛训练摘要"));
  }
  return response.json() as Promise<VisionCompetitionSummary>;
}

export async function runFusionTraining(
  payload: FusionTrainingRunRequest,
): Promise<FusionTrainingRunResult> {
  const response = await fetch(`${API_BASE}/training/fusion/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "融合风险头训练失败"));
  }
  return response.json() as Promise<FusionTrainingRunResult>;
}

export async function fetchFusionTrainingStatus(): Promise<FusionTrainingStatus> {
  const response = await fetch(`${API_BASE}/training/fusion/status`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载融合训练状态"));
  }
  return response.json() as Promise<FusionTrainingStatus>;
}

export async function runDemoEvaluation(): Promise<DemoEvaluationResult> {
  const response = await fetch(`${API_BASE}/training/evaluation/demo-cases`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "展示评测失败"));
  }
  return response.json() as Promise<DemoEvaluationResult>;
}

export async function fetchLocalVisionTrainingStatus(): Promise<LocalVisionTrainingStatus> {
  const response = await fetch(`${API_BASE}/training/local-vision/status`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载本地视觉训练状态"));
  }
  return response.json() as Promise<LocalVisionTrainingStatus>;
}

export async function fetchLocalVisionDataset(
  caseId?: string,
): Promise<LocalVisionTrainingDataset> {
  const params = new URLSearchParams();
  if (caseId) {
    params.set("case_id", caseId);
  }
  const query = params.toString();
  const response = await fetch(`${API_BASE}/training/local-vision/dataset${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载本地视觉样本集"));
  }
  return response.json() as Promise<LocalVisionTrainingDataset>;
}

export async function runLocalVisionTraining(
  payload: LocalVisionCalibrationRunRequest,
): Promise<LocalVisionCalibrationRunResult> {
  const response = await fetch(`${API_BASE}/training/local-vision/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "本地视觉校准训练失败"));
  }
  return response.json() as Promise<LocalVisionCalibrationRunResult>;
}

export async function fetchLocalVisionTrainingRuns(): Promise<LocalVisionCalibrationRunResult[]> {
  const response = await fetch(`${API_BASE}/training/local-vision/runs`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载本地视觉训练历史"));
  }
  return response.json() as Promise<LocalVisionCalibrationRunResult[]>;
}

export function localVisionDatasetJsonlUrl(caseId?: string): string {
  const params = new URLSearchParams();
  if (caseId) {
    params.set("case_id", caseId);
  }
  const query = params.toString();
  return `${API_BASE}/training/local-vision/dataset.jsonl${query ? `?${query}` : ""}`;
}

export async function fetchAgentMetrics(): Promise<AgentMetrics> {
  const response = await fetch(`${API_BASE}/agent/metrics`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载 Agent 运行指标"));
  }
  return response.json() as Promise<AgentMetrics>;
}

export async function fetchModelGatewayStatus(): Promise<ModelGatewayStatus> {
  const response = await fetch(`${API_BASE}/agent/model-gateway/status`);
  if (!response.ok) {
    throw new Error(await errorMessage(response, "无法加载模型网关状态"));
  }
  return response.json() as Promise<ModelGatewayStatus>;
}

export async function invokeModelGateway(
  payload: ModelInvocationRequest,
): Promise<ModelInvocationResult> {
  const response = await fetch(`${API_BASE}/agent/model-gateway/invoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "模型网关调用失败"));
  }
  return response.json() as Promise<ModelInvocationResult>;
}

async function errorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const body = await response.json() as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (body.detail && typeof body.detail === "object") {
      const detail = body.detail as { message?: unknown; audit_id?: unknown };
      const message = typeof detail.message === "string" ? detail.message : JSON.stringify(body.detail);
      const audit = typeof detail.audit_id === "string" ? `（audit: ${detail.audit_id.slice(0, 8)}）` : "";
      return `${message}${audit}`;
    }
  } catch {
    return fallback;
  }
  return fallback;
}
