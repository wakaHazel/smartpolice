export type RiskLevel = "低" | "关注" | "较高" | "紧急";

export interface SpreadMetrics {
  views: number;
  reposts: number;
  comments: number;
  likes: number;
  velocity: string;
}

export interface CaseSample {
  id: string;
  title: string;
  scenario: string;
  platform: string;
  publish_time: string;
  source_url: string;
  content: string;
  image_description: string;
  spread: SpreadMetrics;
  manual_label: string;
  manual_risk_score: number | null;
  tags: string[];
  sensitivity_notes: string;
  review_note: string;
  created_by_user: boolean;
}

export interface CaseCreateRequest {
  id?: string;
  title: string;
  scenario: string;
  platform: string;
  publish_time: string;
  source_url: string;
  content: string;
  image_description: string;
  spread: SpreadMetrics;
  manual_label: string;
  manual_risk_score: number | null;
  tags: string[];
  sensitivity_notes: string;
  review_note: string;
}

export interface CaseLabelRequest {
  manual_risk_score: number;
  manual_label: string;
  review_note: string;
}

export interface MultimodalAnalysis {
  case_id: string;
  claims: string[];
  image_findings: string[];
  consistency_findings: string[];
  aigc_indicators: string[];
  generator_attribution: GeneratorAttribution[];
  preliminary_judgement: string;
}

export interface GeneratorAttribution {
  modality: string;
  candidate_model: string;
  model_family: string;
  confidence: number;
  evidence: string[];
  counter_evidence: string[];
  verification_needed: string[];
}

export interface EvidenceItem {
  id: string;
  type: string;
  title: string;
  content: string;
  confidence: number;
  source: string;
  supports: string;
  artifact_id: string | null;
  source_url: string | null;
  sha256: string | null;
  created_at: string | null;
}

export interface RiskDimension {
  name: string;
  score: number;
  reason: string;
}

export interface RiskAssessment {
  case_id: string;
  score: number;
  level: RiskLevel;
  dimensions: RiskDimension[];
  reasoning: string[];
  evolution: string[];
  model_version_id: string | null;
  model_score: number | null;
  model_confidence: number | null;
  model_explanation: string[];
}

export interface DisposalSuggestion {
  case_id: string;
  verification: string[];
  platform_coordination: string[];
  public_response: string[];
  local_coordination: string[];
  evidence_preservation: string[];
  review_note: string;
}

export interface ReportDraft {
  case_id: string;
  title: string;
  summary: string;
  evidence_summary: string[];
  risk_summary: string;
  suggestions: string[];
  review_statement: string;
  markdown: string;
}

export interface AgentModelRoute {
  role: string;
  selected_model: string;
  provider: string;
  reason: string;
  cost_tier: string;
  fallback_models: string[];
}

export interface AgentSkillRecommendation {
  name: string;
  trigger: string;
  algorithm: string;
  steps: string[];
  verification: string[];
  confidence: number;
}

export interface LearningPipelineStage {
  name: string;
  algorithm: string;
  purpose: string;
  output: string;
}

export interface AgentCostGate {
  name: string;
  rule: string;
  expected_saving: string;
}

export interface AgentOrchestration {
  case_id: string;
  primary_strategy: string;
  model_routes: AgentModelRoute[];
  recommended_skills: AgentSkillRecommendation[];
  learning_pipeline: LearningPipelineStage[];
  cost_gates: AgentCostGate[];
  execution_trace: string[];
}

export interface AgentRunRecord {
  id: string;
  case_id: string;
  created_at: string;
  risk_level: RiskLevel;
  risk_score: number;
  model_routes: AgentModelRoute[];
  skill_names: string[];
  estimated_cost_units: number;
  primary_strategy: string;
}

export interface UsageCount {
  name: string;
  count: number;
}

export interface AgentMetrics {
  total_runs: number;
  average_cost_units: number;
  high_risk_runs: number;
  provider_usage: UsageCount[];
  skill_usage: UsageCount[];
  recent_runs: AgentRunRecord[];
}

export interface ModelProviderStatus {
  provider: string;
  configured: boolean;
  base_url: string | null;
  default_models: Record<string, string>;
  roles: string[];
  missing_env: string[];
  adapter: string;
  health: string;
}

export interface ModelGatewayStatus {
  providers: ModelProviderStatus[];
  dry_run_default: boolean;
  note: string;
}

export interface ModelInvocationRequest {
  case_id?: string | null;
  provider: string;
  role: string;
  prompt: string;
  system_prompt: string;
  dry_run: boolean;
  temperature: number;
}

export interface ModelInvocationResult {
  provider: string;
  role: string;
  selected_model: string;
  configured: boolean;
  dry_run: boolean;
  request_payload: Record<string, unknown>;
  response_text: string | null;
  error: string | null;
  audit_id: string | null;
}

export interface FullAnalysis {
  case: CaseSample;
  analysis: MultimodalAnalysis;
  evidence_chain: EvidenceItem[];
  risk: RiskAssessment;
  disposal: DisposalSuggestion;
  report: ReportDraft;
  agent: AgentOrchestration;
}

export interface CaseAsset {
  id: string;
  case_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
  sha256: string;
  storage_path: string;
  preview_url: string;
  created_at: string;
}

export interface WebEvidenceSnapshot {
  id: string;
  case_id: string;
  requested_url: string;
  final_url: string;
  title: string;
  text: string;
  sha256: string;
  status: string;
  error: string | null;
  html_path: string;
  text_path: string;
  screenshot_path: string | null;
  screenshot_url: string | null;
  created_at: string;
}

export interface CaseEvidenceBundle {
  case_id: string;
  assets: CaseAsset[];
  snapshots: WebEvidenceSnapshot[];
  evidence_items: EvidenceItem[];
}

export interface RealMultimodalAnalysisResult {
  asset_id: string;
  provider: string;
  selected_model: string;
  audit_id: string;
  structured: Record<string, unknown>;
  response_text: string;
}

export interface KnowledgeSearchResult {
  id: string;
  title: string;
  source: string;
  category: string;
  content: string;
  score: number;
  evidence_id: string | null;
  source_url: string | null;
}

export interface RealCaseAnalysisResult {
  case: CaseSample;
  assets: CaseAsset[];
  snapshots: WebEvidenceSnapshot[];
  multimodal_results: RealMultimodalAnalysisResult[];
  evidence_chain: EvidenceItem[];
  baseline_risk: RiskAssessment;
  text_risk_model: Record<string, unknown>;
  vision_evidence_models: Record<string, unknown>;
  fusion_model: Record<string, unknown>;
  structured_review: Record<string, unknown>;
  review_audit_id: string;
  report_markdown: string;
  report_audit_id: string;
  knowledge_refs: KnowledgeSearchResult[];
}

export interface PropagationDisturbanceFinding {
  name: string;
  severity: string;
  score: number;
  evidence: string;
}

export interface ImageForensicsAssetResult {
  asset_id: string;
  filename: string;
  sha256: string;
  width: number | null;
  height: number | null;
  content_type: string;
  size_bytes: number;
  preview_url: string;
  gpt_image2_probability: number | null;
  top_candidate: string;
  confidence: number;
  candidate_distribution: Array<Record<string, unknown>>;
  candidate_ranking: Array<Record<string, unknown>>;
  review_recommendation: Record<string, unknown>;
  disturbances: PropagationDisturbanceFinding[];
  feature_summary: Record<string, unknown>;
  top_contributions: Array<Record<string, unknown>>;
  interpretation: string[];
  limitations: string[];
}

export interface ImageForensicsResult {
  case_id: string;
  research_target: string;
  trained: boolean;
  model_id: string | null;
  model_kind: string | null;
  asset_results: ImageForensicsAssetResult[];
  aggregate: Record<string, unknown>;
  recommended_next_steps: string[];
  application_context: string;
}

export interface ModelInvocationAudit {
  id: string;
  case_id: string | null;
  provider: string;
  role: string;
  model: string;
  status: string;
  request_payload: Record<string, unknown>;
  response_text: string | null;
  error: string | null;
  latency_ms: number;
  token_usage: Record<string, unknown>;
  created_at: string;
}

export interface CaseAuditBundle {
  case_id: string;
  invocations: ModelInvocationAudit[];
  assets: CaseAsset[];
  snapshots: WebEvidenceSnapshot[];
}

export interface FeatureWeight {
  name: string;
  description: string;
  weight: number;
  direction: string;
}

export interface LocalVisionTrainingSample {
  case_id: string;
  asset_id: string;
  image_path: string;
  image_sha256: string;
  prompt: string;
  response: Record<string, unknown>;
  manual_label: string;
  manual_risk_score: number | null;
  created_at: string;
}

export interface LocalVisionTrainingDataset {
  id: string;
  created_at: string;
  model_target: string;
  sample_count: number;
  format: string;
  samples: LocalVisionTrainingSample[];
  note: string;
}

export interface LocalVisionTrainingStats {
  sample_count: number;
  labeled_sample_count: number;
  unlabeled_sample_count: number;
  case_count: number;
  image_count: number;
  average_manual_risk_score: number | null;
  label_distribution: Record<string, number>;
  export_ready: boolean;
  training_ready: boolean;
  note: string;
}

export interface LocalVisionCalibrationRunRequest {
  epochs: number;
  learning_rate: number;
  l2: number;
  min_samples: number;
}

export interface LocalVisionCalibrationRunResult {
  id: string;
  created_at: string;
  model_kind: string;
  status: string;
  sample_count: number;
  validation_count: number;
  feature_count: number;
  epochs: number;
  learning_rate: number;
  train_mae: number;
  validation_mae: number;
  train_rmse: number;
  validation_rmse: number;
  accuracy_within_10: number;
  risk_level_accuracy: number;
  label_distribution: Record<string, number>;
  confusion_matrix: Record<string, Record<string, number>>;
  top_positive_features: FeatureWeight[];
  top_negative_features: FeatureWeight[];
  training_trace: string[];
  model_card: Record<string, unknown>;
}

export interface LocalVisionTrainingStatus {
  trained: boolean;
  active_model_id: string | null;
  latest_run: LocalVisionCalibrationRunResult | null;
  dataset: LocalVisionTrainingStats;
  note: string;
}

export interface ExternalTrainingSample {
  id: string;
  dataset_name: string;
  source: string;
  source_url: string | null;
  task_type: string;
  split: string;
  title: string;
  content: string;
  image_path: string | null;
  image_url: string | null;
  image_sha256: string | null;
  image_available: boolean;
  label: string;
  risk_score: number;
  scenario: string;
  raw_payload: Record<string, unknown>;
  created_at: string;
}

export interface ExternalDatasetImportRequest {
  dataset_name: string;
  source: string;
  source_url?: string | null;
  source_path?: string | null;
  rows?: Record<string, unknown>[];
  format?: string;
  task_type?: string;
  split?: string;
  image_root?: string | null;
  image_path_column?: string | null;
  image_url_column?: string | null;
  text_columns?: string[];
  title_column?: string | null;
  label_column: string;
  label_schema?: Record<string, number>;
  risk_score_column?: string | null;
  scenario_column?: string | null;
  positive_label_values?: string[];
  negative_label_values?: string[];
  default_positive_score?: number;
  default_negative_score?: number;
  limit?: number;
}

export interface ExternalDatasetImportResult {
  dataset_name: string;
  source: string;
  task_type: string;
  imported_count: number;
  skipped_count: number;
  sample_count_after_import: number;
  image_available_count: number;
  label_distribution: Record<string, number>;
  examples: ExternalTrainingSample[];
  note: string;
}

export interface ExternalDatasetSourceSummary {
  dataset_name: string;
  source: string;
  source_url: string | null;
  task_type: string;
  sample_count: number;
  image_available_count: number;
  label_distribution: Record<string, number>;
  latest_import_at: string | null;
}

export interface TrainingTaskStatus {
  task_type: string;
  sample_count: number;
  image_available_count: number;
  label_distribution: Record<string, number>;
  training_ready: boolean;
  sources: ExternalDatasetSourceSummary[];
  note: string;
}

export interface TrainingDataStatus {
  external_sample_count: number;
  labeled_user_case_count: number;
  eligible_sample_count: number;
  demo_case_count: number;
  training_ready: boolean;
  sources: ExternalDatasetSourceSummary[];
  tasks: TrainingTaskStatus[];
  recommended_huggingface_datasets: Array<Record<string, string>>;
  note: string;
}

export interface TrainingRunRequest {
  epochs: number;
  learning_rate: number;
  l2: number;
  include_augmented_samples: boolean;
}

export interface TrainingRunResult {
  id: string;
  created_at: string;
  model_kind: string;
  status: string;
  sample_count: number;
  validation_count: number;
  feature_count: number;
  epochs: number;
  learning_rate: number;
  train_mae: number;
  validation_mae: number;
  train_rmse: number;
  validation_rmse: number;
  accuracy_within_10: number;
  label_distribution: Record<string, number>;
  top_positive_features: FeatureWeight[];
  top_negative_features: FeatureWeight[];
  training_trace: string[];
  model_card: Record<string, unknown>;
  task_metrics: Record<string, unknown>;
  confusion_matrix: Record<string, Record<string, number>>;
}

export interface TrainingStatus {
  trained: boolean;
  active_model_id: string | null;
  latest_run: TrainingRunResult | null;
  training_data: TrainingDataStatus;
  note: string;
}

export interface VisionTrainingRunRequest {
  task_type: string;
  epochs: number;
  learning_rate: number;
  l2: number;
  min_samples: number;
}

export interface VisionTrainingRunResult {
  id: string;
  created_at: string;
  task_type: string;
  model_kind: string;
  status: string;
  sample_count: number;
  validation_count: number;
  feature_count: number;
  epochs: number;
  learning_rate: number;
  train_mae: number;
  validation_mae: number;
  train_rmse: number;
  validation_rmse: number;
  accuracy_within_10: number;
  risk_level_accuracy: number;
  label_distribution: Record<string, number>;
  confusion_matrix: Record<string, Record<string, number>>;
  top_positive_features: FeatureWeight[];
  top_negative_features: FeatureWeight[];
  training_trace: string[];
  model_card: Record<string, unknown>;
}

export interface VisionTrainingStatus {
  task_type: string;
  trained: boolean;
  active_model_id: string | null;
  latest_run: VisionTrainingRunResult | null;
  latest_candidate?: VisionTrainingRunResult | null;
  candidate_vs_active?: Record<string, unknown>;
  data: TrainingTaskStatus;
  note: string;
}

export interface VisionCompetitionSummary {
  task_type: string;
  project_title: string;
  active_model_id: string | null;
  active_model_kind: string | null;
  latest_candidate_id: string | null;
  training_pool: Record<string, unknown>;
  validation_metrics: Record<string, unknown>;
  augmentation_protocol: Record<string, unknown>;
  robustness_headline: Record<string, unknown>;
  model_lifecycle: Record<string, unknown>;
  feature_groups: Record<string, string[]>;
  limitations: string[];
  recommended_next_data: string[];
  narrative_points: string[];
  note: string;
}

export interface FusionTrainingRunRequest {
  epochs: number;
  learning_rate: number;
  l2: number;
  min_samples: number;
}

export interface FusionTrainingRunResult {
  id: string;
  created_at: string;
  model_kind: string;
  status: string;
  sample_count: number;
  validation_count: number;
  feature_count: number;
  epochs: number;
  learning_rate: number;
  train_mae: number;
  validation_mae: number;
  train_rmse: number;
  validation_rmse: number;
  accuracy_within_10: number;
  risk_level_accuracy: number;
  label_distribution: Record<string, number>;
  confusion_matrix: Record<string, Record<string, number>>;
  top_positive_features: FeatureWeight[];
  top_negative_features: FeatureWeight[];
  training_trace: string[];
  model_card: Record<string, unknown>;
}

export interface FusionTrainingStatus {
  trained: boolean;
  active_model_id: string | null;
  latest_run: FusionTrainingRunResult | null;
  data: TrainingTaskStatus;
  note: string;
}

export interface DemoEvaluationCaseResult {
  case_id: string;
  title: string;
  text_only: Record<string, unknown>;
  vision_only: Record<string, unknown>;
  fusion: Record<string, unknown>;
}

export interface DemoEvaluationResult {
  id: string;
  created_at: string;
  demo_case_count: number;
  results: DemoEvaluationCaseResult[];
  note: string;
}
