import {
  ClipboardList,
  Copy,
  Download,
  FileText,
  FileUp,
  Gauge,
  Globe2,
  Plus,
  RefreshCcw,
  Save,
  Search,
  ShieldAlert,
  Trash2,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  captureCaseSource,
  createCase,
  deleteCase,
  fetchCaseEvidence,
  fetchCases,
  fetchImageForensics,
  fetchRealAnalysis,
  fetchTamperForensics,
  fetchTrainingDataStatus,
  runFullAnalysis,
  runImageForensics,
  runRealAnalysis,
  runTamperForensics,
  uploadCaseAsset,
} from "./api";
import type {
  CaseCreateRequest,
  CaseEvidenceBundle,
  CaseSample,
  FullAnalysis,
  ImageForensicsResult,
  RealCaseAnalysisResult,
  RiskLevel,
  TamperForensicsResult,
  TrainingDataStatus,
} from "./types";

interface CandidateDistributionEntry {
  label: string;
  confidence: number;
  rank: number;
  displayName: string;
}

type WorkbenchTask = "generation" | "tamper";

type MarkdownBlock =
  | { kind: "code"; text: string }
  | { kind: "h1" | "h2" | "h3" | "p"; text: string }
  | { kind: "ul"; items: string[] };

const emptyCaseForm: CaseCreateRequest = {
  title: "",
  scenario: "涉警公信力谣言",
  platform: "短视频平台",
  publish_time: new Date().toISOString().slice(0, 16).replace("T", " "),
  source_url: "本地录入样本",
  content: "",
  image_description: "",
  spread: {
    views: 10000,
    reposts: 300,
    comments: 500,
    likes: 800,
    velocity: "小范围传播",
  },
  manual_label: "待人工复核",
  manual_risk_score: null,
  tags: ["待核验"],
  sensitivity_notes: "",
  review_note: "",
};

type DetailTab = "evidence" | "disposal" | "report";
const PRIMARY_DEMO_CASE_ID = "demo-doubao-collapse-disaster-001";
const VISIBLE_DEMO_CASE_IDS = new Set([
  PRIMARY_DEMO_CASE_ID,
  "demo-gptimage-station-police-conflict-001",
  "demo-real-beijing-road-street-001",
]);

const TAMPER_DEMO_CASE_IDS = new Set([
  "tamper-demo-order-after-sale-001",
  "tamper-demo-bank-transfer-001",
  "tamper-demo-medical-complaint-001",
]);

const PRIMARY_TAMPER_DEMO_CASE_ID = "tamper-demo-order-after-sale-001";

const GENERATION_POOL_STATS = [
  { label: "训练池图片", value: "5,914" },
  { label: "GPT-image-2", value: "1,928" },
  { label: "其他AI模型", value: "2,834" },
  { label: "真实照片", value: "1,152" },
];

export function App() {
  const [activeTask, setActiveTask] = useState<WorkbenchTask>("generation");
  const [cases, setCases] = useState<CaseSample[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [selectedTamperCaseId, setSelectedTamperCaseId] = useState<string>("");
  const [tamperAnalysis, setTamperAnalysis] = useState<FullAnalysis | null>(null);
  const [tamperEvidenceBundle, setTamperEvidenceBundle] = useState<CaseEvidenceBundle | null>(null);
  const [tamperForensics, setTamperForensics] = useState<TamperForensicsResult | null>(null);
  const [tamperDatasetStatus, setTamperDatasetStatus] = useState<TrainingDataStatus | null>(null);
  const [analysis, setAnalysis] = useState<FullAnalysis | null>(null);
  const [realAnalysis, setRealAnalysis] = useState<RealCaseAnalysisResult | null>(null);
  const [imageForensics, setImageForensics] = useState<ImageForensicsResult | null>(null);
  const [evidenceBundle, setEvidenceBundle] = useState<CaseEvidenceBundle | null>(null);
  const [searchText, setSearchText] = useState<string>("");
  const [levelFilter, setLevelFilter] = useState<string>("全部");
  const [showCreate, setShowCreate] = useState<boolean>(false);
  const [caseForm, setCaseForm] = useState<CaseCreateRequest>(emptyCaseForm);
  const [tagText, setTagText] = useState<string>("待核验");
  const [activeTab, setActiveTab] = useState<DetailTab>("evidence");
  const [isBusy, setIsBusy] = useState<boolean>(true);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isImageForensicsRunning, setIsImageForensicsRunning] = useState<boolean>(false);
  const [isRealRunning, setIsRealRunning] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [sourceUrlInput, setSourceUrlInput] = useState<string>("");

  const selectedCase = useMemo(
    () => cases.find((item) => item.id === selectedCaseId) ?? null,
    [cases, selectedCaseId],
  );
  const selectedTamperCase = useMemo(
    () => cases.find((item) => item.id === selectedTamperCaseId) ?? null,
    [cases, selectedTamperCaseId],
  );

  const currentImageForensics = imageForensics?.case_id === selectedCaseId ? imageForensics : null;
  const currentTamperForensics = tamperForensics?.case_id === selectedTamperCaseId ? tamperForensics : null;
  const selectedCaseAssetCount = evidenceBundle?.assets.length ?? 0;
  const selectedTamperAssetCount = tamperEvidenceBundle?.assets.length ?? 0;

  const visibleCases = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    return cases.filter((item) => {
      const isDemoVisible = VISIBLE_DEMO_CASE_IDS.has(item.id) || item.created_by_user;
      if (!isDemoVisible) {
        return false;
      }
      if (VISIBLE_DEMO_CASE_IDS.has(item.id)) {
        return true;
      }
      const hidesEngineeringCase =
        item.title.toLowerCase().includes("qwen") ||
        item.title.toLowerCase().includes("ocr") ||
        item.id.startsWith("pytest-") ||
        item.id.startsWith("ui-e2e-") ||
        item.id.startsWith("ui-drill-") ||
        item.id.startsWith("demo-video-") ||
        item.platform.includes("比赛工作台");
      if (hidesEngineeringCase) {
        return false;
      }
      const text = `${item.title} ${item.scenario} ${item.platform} ${item.tags.join(" ")}`.toLowerCase();
      const matchesText = !query || text.includes(query);
      const matchesLevel =
        levelFilter === "全部" ||
        (item.manual_risk_score !== null && scoreToLevel(item.manual_risk_score) === levelFilter);
      return matchesText && matchesLevel;
    });
  }, [cases, levelFilter, searchText]);

  const visibleTamperCases = useMemo(
    () => cases.filter((item) => TAMPER_DEMO_CASE_IDS.has(item.id) || item.scenario.includes("篡改")),
    [cases],
  );

  const loadAnalysis = useCallback(
    async (caseId: string) => {
      setIsBusy(true);
      setError("");
      const locallyCachedRealAnalysis = readCachedRealAnalysis(caseId);
      setRealAnalysis(locallyCachedRealAnalysis);
      try {
        const [result, bundle, cachedForensics, cachedRealAnalysis] = await Promise.all([
          runFullAnalysis(caseId),
          fetchCaseEvidence(caseId),
          fetchImageForensics(caseId).catch(() => null),
          fetchRealAnalysis(caseId).catch(() => null),
        ]);
        setAnalysis(result);
        setEvidenceBundle(bundle);
        if (!bundle.assets.length) {
          clearCachedRealAnalysis(caseId);
          setRealAnalysis(null);
        } else if (cachedRealAnalysis) {
          writeCachedRealAnalysis(cachedRealAnalysis);
          setRealAnalysis(cachedRealAnalysis);
        } else if (!locallyCachedRealAnalysis) {
          setRealAnalysis(null);
        }
        setImageForensics(cachedForensics);
        setSourceUrlInput(result.case.source_url.startsWith("http") ? result.case.source_url : "");
      } catch (err) {
        setError(err instanceof Error ? err.message : "研判失败");
      } finally {
        setIsBusy(false);
      }
    },
    [],
  );

  const loadTamperAnalysis = useCallback(
    async (caseId: string) => {
      setIsBusy(true);
      setError("");
      try {
        const [result, bundle, cachedTamper] = await Promise.all([
          runFullAnalysis(caseId),
          fetchCaseEvidence(caseId),
          fetchTamperForensics(caseId).catch(() => null),
        ]);
        setTamperAnalysis(result);
        setTamperEvidenceBundle(bundle);
        setTamperForensics(cachedTamper);
      } catch (err) {
        setError(err instanceof Error ? err.message : "篡改取证材料加载失败");
      } finally {
        setIsBusy(false);
      }
    },
    [],
  );

  const loadAll = useCallback(async () => {
    setIsBusy(true);
    setError("");
    try {
      const nextCases = await fetchCases();
      const tamperStatus = await fetchTrainingDataStatus("vision_tamper").catch(() => null);
      setCases(nextCases);
      setTamperDatasetStatus(tamperStatus);
      const preferredTamper =
        selectedTamperCaseId && nextCases.some((item) => item.id === selectedTamperCaseId)
          ? selectedTamperCaseId
          : nextCases.find((item) => item.id === PRIMARY_TAMPER_DEMO_CASE_ID)?.id ??
            nextCases.find((item) => TAMPER_DEMO_CASE_IDS.has(item.id))?.id ??
            "";
      setSelectedTamperCaseId(preferredTamper);
      const preferred =
        selectedCaseId && nextCases.some((item) => item.id === selectedCaseId)
          ? selectedCaseId
          : nextCases.find((item) => item.id === PRIMARY_DEMO_CASE_ID)?.id ??
            nextCases.find((item) => item.scenario === "灾害险情谣言")?.id ??
            nextCases.find((item) => item.id === "group-polarization-003")?.id ??
            nextCases[0]?.id ??
            "";
      setSelectedCaseId(preferred);
      if (preferred) {
        await loadAnalysis(preferred);
      } else {
        setIsBusy(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
      setIsBusy(false);
    }
  }, [loadAnalysis, selectedCaseId, selectedTamperCaseId]);

  useEffect(() => {
    void loadAll();
  }, []);

  useEffect(() => {
    if (activeTask === "tamper" && selectedTamperCaseId) {
      void loadTamperAnalysis(selectedTamperCaseId);
    }
  }, [activeTask, loadTamperAnalysis, selectedTamperCaseId]);

  const handleSelectCase = useCallback(
    (caseId: string) => {
      setSelectedCaseId(caseId);
      setAnalysis(null);
      setEvidenceBundle(null);
      setRealAnalysis(null);
      setImageForensics(null);
      void loadAnalysis(caseId);
    },
    [loadAnalysis],
  );

  const handleSelectTamperCase = useCallback((caseId: string) => {
    setSelectedTamperCaseId(caseId);
    setTamperAnalysis(null);
    setTamperEvidenceBundle(null);
    setTamperForensics(null);
    setMessage("");
    setError("");
    void loadTamperAnalysis(caseId);
  }, [loadTamperAnalysis]);

  const handleDeleteCase = useCallback(async (caseItem: CaseSample) => {
    const confirmed = window.confirm(`删除案例“${caseItem.title}”？相关图片、来源快照和报告记录也会一并移除。`);
    if (!confirmed) {
      return;
    }
    setError("");
    setMessage("");
    try {
      await deleteCase(caseItem.id);
      const nextCases = await fetchCases();
      setCases(nextCases);
      setMessage("案例已删除");
      clearCachedRealAnalysis(caseItem.id);
      const fallbackId = nextCases[0]?.id ?? "";
      setSelectedCaseId(fallbackId);
      setRealAnalysis(null);
      setImageForensics(null);
      setEvidenceBundle(null);
      if (fallbackId) {
        await loadAnalysis(fallbackId);
      } else {
        setAnalysis(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "案例删除失败");
    }
  }, [loadAnalysis]);

  const handleCreateCase = useCallback(async () => {
    setError("");
    setMessage("");
    try {
      const created = await createCase({
        ...caseForm,
        tags: tagText.split(/[，,]/).map((item) => item.trim()).filter(Boolean),
      });
      setMessage("案件已创建");
      setShowCreate(false);
      setCaseForm(emptyCaseForm);
      setTagText("待核验");
      const nextCases = await fetchCases();
      setCases(nextCases);
      setSelectedCaseId(created.id);
      await loadAnalysis(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "案件创建失败");
    }
  }, [caseForm, loadAnalysis, tagText]);

  const refreshEvidence = useCallback(async (caseId: string) => {
    const bundle = await fetchCaseEvidence(caseId);
    setEvidenceBundle(bundle);
  }, []);

  const handleUploadAsset = useCallback(async (file: File | undefined) => {
    if (!selectedCase || !file) {
      return;
    }
    setIsUploading(true);
    setError("");
    setMessage("");
    try {
      const asset = await uploadCaseAsset(selectedCase.id, file);
      setMessage(`图片已上传并固定：${asset.sha256.slice(0, 12)}`);
      setImageForensics(null);
      setRealAnalysis(null);
      clearCachedRealAnalysis(selectedCase.id);
      await refreshEvidence(selectedCase.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图片上传失败");
    } finally {
      setIsUploading(false);
    }
  }, [refreshEvidence, selectedCase]);

  const handleTamperUpload = useCallback(async (file: File | undefined) => {
    if (!selectedTamperCase || !file) {
      return;
    }
    setIsUploading(true);
    setError("");
    setMessage("");
    try {
      const asset = await uploadCaseAsset(selectedTamperCase.id, file);
      setMessage(`图片已上传并固定：${asset.sha256.slice(0, 12)}`);
      setTamperForensics(null);
      const bundle = await fetchCaseEvidence(selectedTamperCase.id);
      setTamperEvidenceBundle(bundle);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图片上传失败");
    } finally {
      setIsUploading(false);
    }
  }, [selectedTamperCase]);

  const handleTamperForensics = useCallback(async () => {
    if (!selectedTamperCase) {
      return;
    }
    setIsImageForensicsRunning(true);
    setError("");
    setMessage("");
    try {
      const result = await runTamperForensics(selectedTamperCase.id);
      setTamperForensics(result);
      const asset = result.asset_results[0];
      setMessage(asset ? `篡改取证完成：${tamperCueLabel(asset.top_cue_type)} ${Math.round(asset.confidence * 100)}%` : "篡改取证完成");
    } catch (err) {
      setError(err instanceof Error ? err.message : "图像篡改取证失败");
    } finally {
      setIsImageForensicsRunning(false);
    }
  }, [selectedTamperCase]);

  const handleImageForensics = useCallback(async () => {
    if (!selectedCase) {
      return;
    }
    setIsImageForensicsRunning(true);
    setError("");
    setMessage("");
    try {
      const result = await runImageForensics(selectedCase.id);
      setImageForensics(result);
      const ranking = candidateDistributionEntries(
        result.aggregate.candidate_ranking ??
          result.aggregate.ranked_candidates ??
          result.asset_results[0]?.candidate_ranking ??
          result.asset_results[0]?.candidate_distribution,
      );
      const grouped = groupedSourceDistribution(ranking);
      setMessage(`图片分析完成：${result.asset_results.length} 张，AI生成线索 ${Math.round(grouped.aiTotal * 100)}%`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图像来源研判失败");
    } finally {
      setIsImageForensicsRunning(false);
    }
  }, [selectedCase]);

  const handleCaptureUrl = useCallback(async () => {
    if (!selectedCase) {
      return;
    }
    setIsCapturing(true);
    setError("");
    setMessage("");
    try {
      const snapshot = await captureCaseSource(selectedCase.id, sourceUrlInput);
      setMessage(`来源页面已留证：${snapshot.sha256.slice(0, 12)}`);
      setRealAnalysis(null);
      clearCachedRealAnalysis(selectedCase.id);
      await refreshEvidence(selectedCase.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "URL 取证失败");
    } finally {
      setIsCapturing(false);
    }
  }, [refreshEvidence, selectedCase, sourceUrlInput]);

  const handleRealAnalysis = useCallback(async () => {
    if (!selectedCase) {
      return;
    }
    if (realAnalysis?.case.id === selectedCase.id) {
      setActiveTab("report");
      setMessage("已打开已保存的证据链报告");
      return;
    }
    setIsRealRunning(true);
    setError("");
    setMessage("");
    try {
      const result = await runRealAnalysis(selectedCase.id);
      writeCachedRealAnalysis(result);
      setRealAnalysis(result);
      setActiveTab("report");
      setMessage("证据链报告已生成");
      await refreshEvidence(selectedCase.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "真实研判失败");
      await refreshEvidence(selectedCase.id);
    } finally {
      setIsRealRunning(false);
    }
  }, [realAnalysis, refreshEvidence, selectedCase]);

  const hasSavedRealAnalysis = realAnalysis?.case.id === selectedCaseId;

  const handleCopyReport = useCallback(async () => {
    const markdown = realAnalysis?.report_markdown;
    if (!markdown) {
      setError("正式报告尚未生成，请先启动证据链研判。");
      return;
    }
    await navigator.clipboard.writeText(markdown);
    setMessage("报告已复制");
  }, [realAnalysis]);

  const handleDownloadReport = useCallback(() => {
    const markdown = realAnalysis?.report_markdown;
    if (!analysis || !markdown) {
      setError("正式报告尚未生成，请先启动证据链研判。");
      return;
    }
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${downloadSafeName(analysis.case.title)}-证据链报告.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [analysis, realAnalysis]);

  const taskCopy = activeTask === "generation"
    ? {
        badge: "选案件、传图片、看结论、出报告",
        caseTitle: "案例库",
      }
    : {
        badge: "选样例、传材料、看疑点、做复核",
        caseTitle: "篡改样例",
      };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">公安机关涉网谣言图像取证</p>
          <h1>公共安全AIGC图像取证研判台</h1>
          <div className="task-switch" aria-label="任务切换">
            {[
              ["generation", "AI 生成检测"],
              ["tamper", "AI 篡改取证"],
            ].map(([key, label]) => (
              <button
                className={activeTask === key ? "active" : ""}
                key={key}
                onClick={() => {
                  setActiveTask(key as WorkbenchTask);
                  setMessage("");
                  setError("");
                }}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="topbar-badge">
          <ShieldAlert size={18} />
          <span>{taskCopy.badge}</span>
        </div>
      </header>

      {message ? <div className="notice success">{message}</div> : null}
      {error ? <div className="notice error">{error}</div> : null}

      <div className="workspace">
        <aside className="case-rail" aria-label="案例库">
          <div className="section-heading">
            <ClipboardList size={18} />
            <span>{taskCopy.caseTitle}</span>
            {activeTask === "generation" ? (
              <button className="icon-action" onClick={() => setShowCreate((value) => !value)} title="新建案件" type="button">
                <Plus size={16} />
              </button>
            ) : null}
          </div>

          {activeTask === "generation" ? (
            <div className="filter-bar">
              <label>
                <Search size={15} />
                <input onChange={(event) => setSearchText(event.target.value)} placeholder="搜索案例" value={searchText} />
              </label>
              <select onChange={(event) => setLevelFilter(event.target.value)} value={levelFilter}>
                {["全部", "低", "关注", "较高", "紧急"].map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </div>
          ) : null}

          {activeTask === "generation" && showCreate ? (
            <CaseForm
              form={caseForm}
              onChange={setCaseForm}
              onSubmit={handleCreateCase}
              onTagTextChange={setTagText}
              tagText={tagText}
            />
          ) : null}

          <div className="case-list">
            {activeTask === "generation" ? (
              <>
                {visibleCases.map((item) => (
                  <article
                    className={`case-card ${item.id === selectedCaseId ? "active" : ""}`}
                    key={item.id}
                  >
                    <button className="case-card-main" onClick={() => handleSelectCase(item.id)} type="button">
                      <span className="case-scenario">{item.scenario}</span>
                      <strong>{item.title}</strong>
                      <span>{item.platform} · {item.spread.velocity}</span>
                      <em>待研判</em>
                    </button>
                    <button
                      aria-label={`删除案例：${item.title}`}
                      className="case-delete-button"
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDeleteCase(item);
                      }}
                      title="删除案例"
                      type="button"
                    >
                      <Trash2 size={15} />
                    </button>
                  </article>
                ))}
                {!visibleCases.length ? <p className="empty-state">暂无案例。点击上方加号新建案件。</p> : null}
              </>
            ) : (
              visibleTamperCases.map((item) => (
                <article
                  className={`case-card ${item.id === selectedTamperCaseId ? "active" : ""}`}
                  key={item.id}
                >
                  <button className="case-card-main" onClick={() => handleSelectTamperCase(item.id)} type="button">
                    <span className="case-scenario">{item.scenario}</span>
                    <strong>{item.title}</strong>
                    <span>{item.platform} · {item.spread.velocity}</span>
                    <em>演示样例</em>
                  </button>
                </article>
              ))
            )}
            {activeTask === "tamper" && !visibleTamperCases.length ? <p className="empty-state">篡改演示案例待同步。</p> : null}
          </div>
        </aside>

        <section className="analysis-panel">
          {activeTask === "tamper" && (isBusy || !tamperAnalysis || !selectedTamperCase) ? (
            <LoadingPanel />
          ) : activeTask === "tamper" && selectedTamperCase ? (
            <TamperWorkspace
              analysis={tamperAnalysis!}
              onAnalyze={handleTamperForensics}
              onUpload={handleTamperUpload}
              bundle={tamperEvidenceBundle}
              isAnalyzing={isImageForensicsRunning}
              isUploading={isUploading}
              result={currentTamperForensics}
              selectedCase={selectedTamperCase}
              tamperDatasetStatus={tamperDatasetStatus}
            />
          ) : !selectedCaseId && !isBusy ? (
            <EmptyFormalState
              title="暂无案例"
              body="点击案例库上方的加号新建案件，然后上传图片进行来源研判。"
            />
          ) : isBusy || !analysis ? (
            <LoadingPanel />
          ) : (
            <div className="analysis-grid">
              <section className="hero-panel">
                <div className="hero-copy">
                  <p className="eyebrow">当前案件</p>
                  <h2>{analysis.case.title}</h2>
                  <p>{analysis.case.content}</p>
                  <div className="tag-row">
                    {analysis.case.tags.map((tag) => <span key={tag}>{tag}</span>)}
                  </div>
                </div>
                <div className="case-progress-panel">
                  <SmallStat label="图片" value={`${selectedCaseAssetCount} 张`} />
                  <SmallStat label="分析" value={currentImageForensics ? "已完成" : selectedCaseAssetCount ? "待点击" : "待上传"} />
                  <SmallStat label="报告" value={realAnalysis ? "已生成" : "待生成"} />
                </div>
              </section>

              <PoolStatsStrip stats={GENERATION_POOL_STATS} />

              <section className="module module-span-2 real-workflow">
                <ModuleTitle
                  icon={<ShieldAlert size={18} />}
                  title="上传待检图片"
                  subtitle={`已上传 ${evidenceBundle?.assets.length ?? 0} 张`}
                />
                <div className="real-input-grid">
                  <label className="upload-box">
                    <UploadCloud size={22} />
                    <span>{isUploading ? "正在上传固定" : "上传图片"}</span>
                    <input
                      accept="image/png,image/jpeg,image/webp"
                      disabled={isUploading}
                      onChange={(event) => void handleUploadAsset(event.target.files?.[0])}
                      type="file"
                    />
                  </label>
                  <div className="url-capture">
                    <Globe2 size={18} />
                    <input
                      onChange={(event) => setSourceUrlInput(event.target.value)}
                      placeholder="来源链接，可选"
                      value={sourceUrlInput}
                    />
                    <button disabled={isCapturing || !sourceUrlInput.trim()} onClick={() => void handleCaptureUrl()} type="button">
                      <FileUp size={15} />{isCapturing ? "留证中" : "保存来源快照"}
                    </button>
                  </div>
                  <button
                    className="real-run"
                    disabled={isImageForensicsRunning || !(evidenceBundle?.assets.length)}
                    onClick={() => void handleImageForensics()}
                    type="button"
                  >
                    <Search size={16} />{isImageForensicsRunning ? "分析中" : "分析这张图"}
                  </button>
                </div>
                <AssetSnapshotStrip bundle={evidenceBundle} task="generation" />
              </section>

              <section className="module forensics-result-module">
                <ModuleTitle
                  icon={<Gauge size={18} />}
                  title="分析结果"
                  subtitle="三分类生成图研判 · 下一步"
                />
                <ImageForensicsPanel result={currentImageForensics} />
              </section>

              <section className="module action-module module-span-2">
                <ModuleTitle icon={<Wand2 size={18} />} title="报告操作" subtitle="生成、复制、导出办案材料" />
                <div className="action-grid">
                  <button onClick={() => void loadAnalysis(analysis.case.id)} type="button"><RefreshCcw size={15} />重新研判</button>
                  <button disabled={isRealRunning || !(evidenceBundle?.assets.length)} onClick={() => void handleRealAnalysis()} type="button">
                    <ShieldAlert size={15} />{isRealRunning ? "证据链生成中" : hasSavedRealAnalysis ? "打开证据链报告" : "生成证据链报告"}
                  </button>
                  <button onClick={() => void handleCopyReport()} type="button"><Copy size={15} />复制报告</button>
                  <button onClick={handleDownloadReport} type="button"><Download size={15} />导出报告</button>
                </div>
              </section>

              <section className="module module-span-2">
                <div className="tab-row">
                  {[ 
                    ["evidence", "材料"],
                    ["disposal", "处置建议"],
                    ["report", "报告"],
                  ].map(([key, label]) => (
                    <button
                      className={activeTab === key ? "active" : ""}
                      key={key}
                      onClick={() => setActiveTab(key as DetailTab)}
                      type="button"
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <DetailPanel
                  activeTab={activeTab}
                  evidenceBundle={evidenceBundle}
                  realAnalysis={realAnalysis}
                />
              </section>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function DetailPanel({
  activeTab,
  evidenceBundle,
  realAnalysis,
}: {
  activeTab: DetailTab;
  evidenceBundle: CaseEvidenceBundle | null;
  realAnalysis: RealCaseAnalysisResult | null;
}) {
  if (activeTab === "evidence") {
    const items = realAnalysis?.evidence_chain ?? evidenceBundle?.evidence_items ?? [];
    if (!items.length) {
      return (
        <EmptyFormalState
          title="材料待补充"
          body="上传图片后，这里会保存图片文件和来源快照。需要正式材料时，再生成证据链报告。"
        />
      );
    }
    return (
      <div className="evidence-timeline">
        {items.map((item) => (
          <article className="evidence-card" key={item.id}>
            <div className="evidence-card-head">
              <span>{item.type}</span>
              <b>{Math.round(item.confidence * 100)}%</b>
            </div>
            <strong>{item.title}</strong>
            <p>{item.content}</p>
            <footer>
              {item.sha256 ? <code>文件指纹 {item.sha256.slice(0, 12)}</code> : <code>案件文字</code>}
              <span>{item.supports}</span>
            </footer>
          </article>
        ))}
      </div>
    );
  }

  if (activeTab === "disposal") {
    if (!realAnalysis) {
      return (
        <EmptyFormalState
          title="处置建议待复核"
          body="上传图片并完成分析后，可以生成证据链报告，系统会整理下一步核查和处置建议。"
        />
      );
    }
    const suggestions = disposalSuggestionsFromReview(realAnalysis);
    return (
      <div className="disposal-grid">
        <ActionList accent="green" title="优先核查" items={suggestions.verification} />
        <ActionList accent="blue" title="平台协查" items={suggestions.platform} />
        <ActionList accent="amber" title="公开回应" items={suggestions.response} />
        <ActionList accent="red" title="不确定项" items={suggestions.uncertainties} />
      </div>
    );
  }

  return (
    <div className="report-panel">
      {realAnalysis?.report_markdown ? (
        <MarkdownReport markdown={realAnalysis.report_markdown} />
      ) : (
        <EmptyFormalState
          title="正式报告待生成"
          body="点击“生成证据链报告”后，系统会整理案件材料、图片分析结果、处置建议和可追溯记录。"
        />
      )}
    </div>
  );
}

function TamperWorkspace({
  analysis,
  bundle,
  isAnalyzing,
  isUploading,
  onAnalyze,
  onUpload,
  result,
  selectedCase,
  tamperDatasetStatus,
}: {
  analysis: FullAnalysis;
  bundle: CaseEvidenceBundle | null;
  isAnalyzing: boolean;
  isUploading: boolean;
  onAnalyze: () => Promise<void>;
  onUpload: (file: File | undefined) => Promise<void>;
  result: TamperForensicsResult | null;
  selectedCase: CaseSample;
  tamperDatasetStatus: TrainingDataStatus | null;
}) {
  const assetCount = bundle?.assets.length ?? 0;
  const tamperPoolStats = tamperDatasetStats(tamperDatasetStatus);
  return (
    <div className="analysis-grid">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">当前样例</p>
          <h2>{analysis.case.title}</h2>
          <p>{analysis.case.content}</p>
          <div className="tag-row">
            {analysis.case.tags.map((tag) => <span key={tag}>{tag}</span>)}
          </div>
        </div>
        <div className="case-progress-panel">
          <SmallStat label="图片" value={`${assetCount} 张`} />
          <SmallStat label="分析" value={result ? "已完成" : assetCount ? "待点击" : "待上传"} />
          <SmallStat label="接口" value="正式API" />
        </div>
      </section>

      <PoolStatsStrip stats={tamperPoolStats} />

      <section className="module module-span-2 real-workflow">
        <ModuleTitle
          icon={<ShieldAlert size={18} />}
          title="上传待检图片"
          subtitle={`已上传 ${assetCount} 张`}
        />
        <div className="real-input-grid tamper-input-grid">
          <label className="upload-box">
            <UploadCloud size={22} />
            <span>{isUploading ? "正在上传固定" : "上传图片"}</span>
            <input
              accept="image/png,image/jpeg,image/webp"
              disabled={isUploading}
              onChange={(event) => onUpload(event.target.files?.[0])}
              type="file"
            />
          </label>
          <div className="tamper-upload-note">
            <strong>{selectedCase.platform}</strong>
            <span>用于展示疑似异常区域、可见痕迹、辅助研判和人工复核建议。</span>
          </div>
          <button
            className="real-run"
            disabled={isAnalyzing || !assetCount}
            onClick={() => void onAnalyze()}
            type="button"
          >
            <Search size={16} />{isAnalyzing ? "分析中" : "分析这张图"}
          </button>
        </div>
        <AssetSnapshotStrip bundle={bundle} task="tamper" />
      </section>

      <section className="module forensics-result-module">
        <ModuleTitle
          icon={<Gauge size={18} />}
          title="分析结果"
          subtitle="结论 · 主要疑点 · 下一步"
        />
        <TamperForensicsPanel result={result} />
      </section>

      <section className="module action-module module-span-2">
        <ModuleTitle icon={<Wand2 size={18} />} title="复核建议" subtitle="原件、后台、时间戳" />
        <TamperSuggestions result={result} />
      </section>
    </div>
  );
}

function PoolStatsStrip({ stats }: { stats: Array<{ label: string; value: string }> }) {
  return (
    <section className="pool-strip module-span-2" aria-label="样本池概览">
      {stats.map((item) => (
        <SmallStat key={item.label} label={item.label} value={item.value} />
      ))}
    </section>
  );
}

function tamperDatasetStats(status: TrainingDataStatus | null): Array<{ label: string; value: string }> {
  const task = status?.tasks.find((item) => item.task_type === "vision_tamper");
  const labels = task?.label_distribution ?? {};
  const sourceCount = status?.sources.length ?? 0;
  const externalImages = task?.image_available_count ?? status?.external_sample_count ?? 0;
  return [
    { label: "篡改HF/外部图", value: String(externalImages) },
    { label: "数据源", value: String(sourceCount) },
    { label: "脱敏演示", value: "3" },
    { label: "标签类型", value: String(Object.keys(labels).length || 0) },
  ];
}

function TamperForensicsPanel({ result }: { result: TamperForensicsResult | null }) {
  const asset = result?.asset_results[0];
  if (!asset) {
    return (
      <EmptyFormalState
        title="等待图片分析"
        body="先上传待检图片，再点击“分析这张图”。这里会显示篡改风险、主要疑点、疑似区域和下一步核查建议。"
      />
    );
  }
  const riskLabel = tamperRiskLabel(asset.tamper_risk);
  const cueLabel = tamperCueLabel(asset.top_cue_type);
  const regionHint = asset.suspected_regions[0]?.label ?? "候选区域";
  const visibleCue = asset.visible_cues[0] ?? asset.suspected_regions[0]?.visible_cues[0] ?? "保留原始材料并进入人工复核。";
  const nextSteps = compactTamperNextSteps(asset, result);
  return (
    <div className="forensics-board">
      <div className="forensics-primary">
        <a className="forensics-preview" href={assetUrl(asset.preview_url)} rel="noreferrer" target="_blank">
          <img alt={asset.filename} src={assetUrl(asset.preview_url)} />
        </a>
        <div className="forensics-verdict">
          <span>初步结论</span>
          <strong>{riskLabel}风险</strong>
          <em>主要疑点 {cueLabel} · 可信度 {Math.round(asset.confidence * 100)}%</em>
        </div>
      </div>
      <div className="candidate-summary-list candidate-summary-list-three" aria-label="篡改风险摘要">
        {[
          { label: "风险等级", name: `${riskLabel}风险`, value: "" },
          { label: "主要疑点", name: cueLabel, value: "" },
          { label: "可信度", name: "辅助研判", value: `${Math.round(asset.confidence * 100)}%` },
        ].map((item) => (
          <div className="candidate-summary-item" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.name}</strong>
            <b>{item.value}</b>
          </div>
        ))}
      </div>
      <div className="next-step-list">
        {[`疑似区域：${regionHint}`, `可见线索：${visibleCue}`, ...nextSteps].map((item) => <span key={item}>{item}</span>)}
      </div>
    </div>
  );
}

function TamperScoreBreakdown({ scores }: { scores: Record<string, number> }) {
  const rows = [
    ["document_prior", "材料上下文"],
    ["file_container", "文件统计"],
    ["patch_signal", "Patch 信号"],
    ["region_confidence", "区域置信"],
  ].map(([key, label]) => ({
    key,
    label,
    value: Math.max(0, Math.min(1, scores[key] ?? 0)),
  }));
  return (
    <div className="tamper-score-list">
      <strong>分数拆解</strong>
      {rows.map((row) => (
        <span key={row.key}>
          <b>{row.label}</b>
          <i><em style={{ width: `${row.value * 100}%` }} /></i>
          <small>{Math.round(row.value * 100)}%</small>
        </span>
      ))}
    </div>
  );
}

function TamperSuggestions({ result }: { result: TamperForensicsResult | null }) {
  const asset = result?.asset_results[0];
  if (!asset) {
    return (
      <EmptyFormalState
        title="复核建议待生成"
        body="完成篡改分析后，这里会列出可见线索、后台核验方向和结论边界。"
      />
    );
  }
  const boundary = stringValue(result.aggregate.boundary) || "图像篡改分析为辅助线索，需结合原始文件、业务后台和人工复核。";
  const suggestions = asset.review_suggestions.length ? asset.review_suggestions : result.recommended_next_steps;
  return (
    <div className="disposal-grid">
      <ActionList accent="green" title="可见线索" items={asset.visible_cues} />
      <ActionList accent="blue" title="人工复核建议" items={suggestions} />
      <ActionList accent="amber" title="辅助研判边界" items={[boundary, ...asset.limitations.slice(0, 2)]} />
      <ActionList accent="red" title="候选区域" items={asset.suspected_regions.map((item) => `${item.label}：${item.visible_cues[0] ?? tamperCueLabel(item.cue_type)}`)} />
    </div>
  );
}

function ImageForensicsPanel({ result }: { result: ImageForensicsResult | null }) {
  if (!result || !result.asset_results.length) {
    return (
      <EmptyFormalState
        title="等待图片分析"
        body="先上传图片，再点击“分析这张图”。这里会显示三分类研判结果、置信度和下一步核查建议。"
      />
    );
  }
  const asset = result.asset_results[0];
  const allCandidates = candidateDistributionEntries(
    asset.candidate_ranking?.length ? asset.candidate_ranking : asset.candidate_distribution,
  );
  const groupedSources = groupedSourceDistribution(allCandidates);
  const displaySources = normalizeVisibleSourceDistribution(groupedSources);
  const groupedRows = groupedSourceRows(displaySources, asset.top_candidate);
  const generatedSignal = Math.round(displaySources.aiTotal * 100);
  const topSource = groupedRows[0];
  const verdictTitle = primaryForensicsVerdict(asset.top_candidate);
  const verdictSubtitle = forensicsVerdictSubtitle(topSource, displaySources);
  const modelHints = compactGeneratorModelHints(asset);
  const nextSteps = compactForensicsNextSteps(verdictTitle, generatedSignal);
  return (
    <div className="forensics-board">
      <div className="forensics-primary">
        <a className="forensics-preview" href={assetUrl(asset.preview_url)} rel="noreferrer" target="_blank">
          <img alt={asset.filename} src={assetUrl(asset.preview_url)} />
        </a>
        <div className="forensics-verdict">
          <span>初步结论</span>
          <strong>{verdictTitle}</strong>
          <em>{verdictSubtitle}</em>
        </div>
      </div>
      <div className="candidate-summary-list candidate-summary-list-three" aria-label="来源类别分布">
        {groupedRows.map((item) => (
          <div className="candidate-summary-item" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.name}</strong>
            <b>{Math.round(item.confidence * 100)}%</b>
          </div>
        ))}
      </div>
      <div className="next-step-list">
        {[...modelHints, ...nextSteps].map((item) => <span key={item}>{item}</span>)}
      </div>
    </div>
  );
}

function EmptyFormalState({ body, title }: { body: string; title: string }) {
  return (
    <div className="empty-formal-state">
      <ShieldAlert size={18} />
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  );
}

function AssetSnapshotStrip({ bundle, task }: { bundle: CaseEvidenceBundle | null; task: "generation" | "tamper" }) {
  const assets = bundle?.assets ?? [];
  const snapshots = bundle?.snapshots ?? [];
  if (!assets.length && !snapshots.length) {
    const text =
      task === "tamper"
        ? "篡改取证素材待同步。请上传单据/凭证/材料图片后再分析。"
        : "生成检测素材待同步。请上传待检图片或保存来源快照后再分析。";
    return <div className="empty-evidence">{text}</div>;
  }
  return (
    <div className="asset-strip">
      {assets.slice(0, 4).map((asset) => (
        <a href={assetUrl(asset.preview_url)} key={asset.id} rel="noreferrer" target="_blank">
          <img alt={asset.filename} src={assetUrl(asset.preview_url)} />
          <span>{asset.filename}</span>
          <b>文件指纹 {asset.sha256.slice(0, 10)}</b>
        </a>
      ))}
      {snapshots.slice(0, 4).map((snapshot) => (
        <a href={snapshot.screenshot_url ?? snapshot.final_url} key={snapshot.id} rel="noreferrer" target="_blank">
          <Globe2 size={15} />
          <span>{snapshot.title}</span>
          <b>{snapshot.status}</b>
        </a>
      ))}
    </div>
  );
}

function CaseForm({
  form,
  onChange,
  onSubmit,
  onTagTextChange,
  tagText,
}: {
  form: CaseCreateRequest;
  onChange: (value: CaseCreateRequest) => void;
  onSubmit: () => Promise<void>;
  onTagTextChange: (value: string) => void;
  tagText: string;
}) {
  const update = <K extends keyof CaseCreateRequest>(key: K, value: CaseCreateRequest[K]) => {
    onChange({ ...form, [key]: value });
  };
  const updateSpread = (key: keyof CaseCreateRequest["spread"], value: string) => {
    onChange({
      ...form,
      spread: {
        ...form.spread,
        [key]: key === "velocity" ? value : Number(value),
      },
    });
  };

  return (
    <section className="create-panel">
      <label className="field-block">
        <span>案件标题</span>
        <input onChange={(event) => update("title", event.target.value)} placeholder="例如：网传车站执法冲突图片核查" value={form.title} />
      </label>
      <label className="field-block">
        <span>案件类型</span>
        <select onChange={(event) => update("scenario", event.target.value)} value={form.scenario}>
          {["涉警公信力谣言", "灾害险情谣言", "群体对立煽动型谣言", "低风险误传"].map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
      </label>
      <label className="field-block">
        <span>网传内容</span>
        <textarea onChange={(event) => update("content", event.target.value)} placeholder="粘贴网传文字、标题或简要说法" rows={3} value={form.content} />
      </label>
      <label className="field-block">
        <span>图片情况</span>
        <textarea onChange={(event) => update("image_description", event.target.value)} placeholder="简单描述图片画面；不确定可以先写“待上传图片”" rows={2} value={form.image_description} />
      </label>
      <div className="form-row three">
        <label className="field-block compact">
          <span>浏览量</span>
          <input onChange={(event) => updateSpread("views", event.target.value)} type="number" value={form.spread.views} />
        </label>
        <label className="field-block compact">
          <span>转发量</span>
          <input onChange={(event) => updateSpread("reposts", event.target.value)} type="number" value={form.spread.reposts} />
        </label>
        <label className="field-block compact">
          <span>评论量</span>
          <input onChange={(event) => updateSpread("comments", event.target.value)} type="number" value={form.spread.comments} />
        </label>
      </div>
      <div className="form-row two">
        <label className="field-block compact">
          <span>点赞量</span>
          <input onChange={(event) => updateSpread("likes", event.target.value)} type="number" value={form.spread.likes} />
        </label>
        <label className="field-block compact">
          <span>传播速度</span>
          <input onChange={(event) => updateSpread("velocity", event.target.value)} placeholder="例如：同城群快速转发" value={form.spread.velocity} />
        </label>
      </div>
      <p className="form-hint">传播数据只用于风险背景；没有准确数字时可保留默认值，不影响图片上传和来源分析。</p>
      <label className="field-block">
        <span>标签</span>
        <input onChange={(event) => onTagTextChange(event.target.value)} placeholder="例如：涉警，AI生成，待核验" value={tagText} />
      </label>
      <button className="full-button" onClick={() => void onSubmit()} type="button"><Save size={15} />保存并研判</button>
    </section>
  );
}

function MarkdownReport({ markdown }: { markdown: string }) {
  const blocks = useMemo(() => parseMarkdownBlocks(markdown), [markdown]);
  return (
    <article className="markdown-report">
      {blocks.map((block, index) => {
        const key = `${block.kind}-${index}`;
        if (block.kind === "code") {
          return <pre className="report-code-block" key={key}>{block.text}</pre>;
        }
        if (block.kind === "ul") {
          return (
            <ul key={key}>
              {block.items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          );
        }
        if (block.kind === "h1") {
          return <h1 key={key}>{block.text}</h1>;
        }
        if (block.kind === "h2") {
          return <h2 key={key}>{block.text}</h2>;
        }
        if (block.kind === "h3") {
          return <h3 key={key}>{block.text}</h3>;
        }
        return <p key={key}>{block.text}</p>;
      })}
    </article>
  );
}

function ActionList({
  accent = "green",
  items,
  title,
}: {
  accent?: "green" | "blue" | "amber" | "red";
  items: string[];
  title: string;
}) {
  return (
    <article className={`action-list action-list-${accent}`}>
      <strong>{title}</strong>
      <ul>
        {items.length ? items.map((item) => <li key={item}>{item}</li>) : <li>待人工复核补充。</li>}
      </ul>
    </article>
  );
}

function ModuleTitle({ icon, subtitle, title }: { icon: ReactNode; subtitle: string; title: string }) {
  return (
    <div className="module-title">
      <div>
        {icon}
        <h3>{title}</h3>
      </div>
      <span>{subtitle}</span>
    </div>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="small-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LoadingPanel() {
  return (
    <div className="loading-panel">
      <ShieldAlert size={28} />
      <span>正在加载案件材料...</span>
    </div>
  );
}

function disposalSuggestionsFromReview(realAnalysis: RealCaseAnalysisResult): {
  verification: string[];
  platform: string[];
  response: string[];
  uncertainties: string[];
} {
  const review = realAnalysis.structured_review;
  const suggestions = stringList(review.disposal_suggestions);
  const verification = [
    ...stringList(review.verification_items),
    ...realAnalysis.knowledge_refs.slice(0, 2).map((item) => `对照知识依据：${item.title}`),
  ];
  return {
    verification: verification.length ? verification : suggestions.slice(0, 2),
    platform: suggestions.filter((item) => item.includes("平台") || item.includes("下架") || item.includes("限流")).slice(0, 3),
    response: suggestions.filter((item) => item.includes("通报") || item.includes("回应") || item.includes("澄清")).slice(0, 3),
    uncertainties: [
      ...stringList(review.uncertainties),
      ...realAnalysis.multimodal_results.flatMap((item) => stringList(item.structured.uncertainties)).slice(0, 3),
    ],
  };
}

function parseMarkdownBlocks(markdown: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  const lines = markdown.split(/\r?\n/);
  let paragraph: string[] = [];
  let listItems: string[] = [];
  let codeLines: string[] = [];
  let inCode = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: "p", text: paragraph.join(" ") });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      blocks.push({ kind: "ul", items: listItems });
      listItems = [];
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        blocks.push({ kind: "code", text: codeLines.join("\n") });
        codeLines = [];
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }
    if (trimmed.startsWith("### ")) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "h3", text: trimmed.slice(4) });
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "h2", text: trimmed.slice(3) });
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "h1", text: trimmed.slice(2) });
      continue;
    }
    if (trimmed.startsWith("- ")) {
      flushParagraph();
      listItems.push(trimmed.slice(2));
      continue;
    }
    flushList();
    paragraph.push(trimmed);
  }
  if (inCode && codeLines.length) {
    blocks.push({ kind: "code", text: codeLines.join("\n") });
  }
  flushParagraph();
  flushList();
  return blocks;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function downloadSafeName(value: string): string {
  const cleaned = value.trim().replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "");
  return cleaned || "图片来源研判";
}

function scoreToLevel(score: number): RiskLevel {
  if (score >= 85) {
    return "紧急";
  }
  if (score >= 68) {
    return "较高";
  }
  if (score >= 40) {
    return "关注";
  }
  return "低";
}

function tamperRiskLabel(value: string): string {
  const labels: Record<string, string> = {
    low: "低",
    medium: "中",
    high: "高",
  };
  return labels[value] ?? "待核验";
}

function tamperCueLabel(value: string): string {
  const labels: Record<string, string> = {
    text_overlay: "文字覆盖",
    amount_date_mismatch: "金额/日期不一致",
    splice: "局部拼接",
    inpaint: "擦除修补",
    copy_move: "复制移动",
    compression_mismatch: "压缩不一致",
    unknown: "未知疑点",
  };
  return labels[value] ?? (value ? value : "待分析");
}

function sourceLabel(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/_/g, "-");
  const labels: Record<string, string> = {
    "gpt-image2": "GPT-image2",
    "gpt-image-2": "GPT-image2",
    "gpt image2": "GPT-image2",
    "gpt-image1": "其他 AI 生成图",
    "gpt-image1.5": "其他 AI 生成图",
    "stable-diffusion": "其他 AI 生成图",
    "midjourney": "其他 AI 生成图",
    "nano-banana": "其他 AI 生成图",
    "seedream-4": "其他 AI 生成图",
    "dall-e": "其他 AI 生成图",
    "dall-e-3": "其他 AI 生成图",
    "flux": "其他 AI 生成图",
    "real": "真实照片",
    "other-generated": "其他 AI 生成图",
    "not-gpt-image2": "未知/低置信",
    "unknown": "未知/低置信",
  };
  return labels[normalized] ?? (value ? value : "待分析");
}

function primaryForensicsVerdict(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/_/g, "-");
  if (!normalized || normalized === "unknown") {
    return "未知/低置信";
  }
  if (normalized === "real") {
    return "真实照片";
  }
  if (normalized === "not-gpt-image2") {
    return "未知/低置信";
  }
  return "AI生成图";
}

interface GroupedSourceDistribution {
  gptImage2: number;
  otherAi: number;
  real: number;
  unknown: number;
  aiTotal: number;
}

function groupedSourceDistribution(candidates: CandidateDistributionEntry[]): GroupedSourceDistribution {
  const grouped = candidates.reduce(
    (totals, item) => {
      const label = item.label.trim().toLowerCase().replace(/_/g, "-");
      if (["gpt-image2", "gpt-image-2", "gpt image2"].includes(label)) {
        totals.gptImage2 += item.confidence;
      } else if (["real", "real-photo"].includes(label)) {
        totals.real += item.confidence;
      } else if (!label || ["unknown", "not-gpt-image2"].includes(label)) {
        totals.unknown += item.confidence;
      } else {
        totals.otherAi += item.confidence;
      }
      return totals;
    },
    { gptImage2: 0, otherAi: 0, real: 0, unknown: 0 },
  );
  const aiTotal = Math.max(0, Math.min(1, grouped.gptImage2 + grouped.otherAi));
  return {
    ...grouped,
    aiTotal,
  };
}

function normalizeVisibleSourceDistribution(grouped: GroupedSourceDistribution): GroupedSourceDistribution {
  const visibleTotal = grouped.gptImage2 + grouped.otherAi + grouped.real;
  if (visibleTotal <= 0) {
    return grouped;
  }
  const gptImage2 = grouped.gptImage2 / visibleTotal;
  const otherAi = grouped.otherAi / visibleTotal;
  const real = grouped.real / visibleTotal;
  return {
    gptImage2,
    otherAi,
    real,
    unknown: 0,
    aiTotal: Math.max(0, Math.min(1, gptImage2 + otherAi)),
  };
}

function groupedSourceRows(
  grouped: GroupedSourceDistribution,
  topCandidate?: string,
): Array<{ confidence: number; label: string; name: string }> {
  const topGroupName = visibleSourceGroupName(topCandidate ?? "");
  const sorted = [
    { name: "GPT-image2", confidence: grouped.gptImage2 },
    { name: "其他 AI 生成图", confidence: grouped.otherAi },
    { name: "真实照片", confidence: grouped.real },
  ].sort((left, right) => right.confidence - left.confidence);
  if (topGroupName) {
    const topIndex = sorted.findIndex((item) => item.name === topGroupName);
    if (topIndex > 0) {
      const [topItem] = sorted.splice(topIndex, 1);
      sorted.unshift(topItem);
    }
  }
  return sorted.map((item, index) => ({ ...item, label: `类别${index + 1}` }));
}

function visibleSourceGroupName(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/_/g, "-");
  if (["gpt-image2", "gpt-image-2", "gpt image2"].includes(normalized)) {
    return "GPT-image2";
  }
  if (["real", "real-photo"].includes(normalized)) {
    return "真实照片";
  }
  if (!normalized || ["unknown", "not-gpt-image2"].includes(normalized)) {
    return "";
  }
  return "其他 AI 生成图";
}

function forensicsVerdictSubtitle(
  topSource: { confidence: number; name: string } | undefined,
  grouped: GroupedSourceDistribution,
): string {
  if (!topSource) {
    return "等待来源类别概率";
  }
  const contrastName = topSource.name === "真实照片" ? "GPT-image2" : "真实照片";
  const contrastValue = topSource.name === "真实照片" ? grouped.gptImage2 : grouped.real;
  return `最高类别 ${topSource.name} ${Math.round(topSource.confidence * 100)}% · ${contrastName} ${Math.round(contrastValue * 100)}%`;
}

function compactForensicsNextSteps(verdictTitle: string, generatedSignal: number): string[] {
  const firstStep = verdictTitle === "真实照片"
    ? "按真实照片待核验处置，重点核对发布时间、原始出处和是否被移花接木。"
    : generatedSignal >= 50
      ? "按疑似AI生成图片处置，先固定原图、来源链接和发布账号信息。"
      : "先按待核验图片处置，补充原图、来源链接和发布账号信息。";
  return [
    firstStep,
    "联系平台调取首发记录、编辑记录和传播链，确认是否被二次转发或压缩。",
    "对外通报前结合现场、权威部门和平台证据复核，不单独依赖单图结论。",
  ];
}

function compactGeneratorModelHints(asset: ImageForensicsResult["asset_results"][number]): string[] {
  const hints: string[] = [];
  const realPhotoGuard = asRecord(asset.real_photo_guard) ?? {};
  const binaryGate = asRecord(asset.binary_gate) ?? {};
  const guardScore = numberValue(realPhotoGuard.score);
  if (guardScore !== null) {
    hints.push(`真实照片保护线索：${Math.round(guardScore * 100)}%，仅作辅助研判。`);
  }
  const generatedProbability = numberValue(binaryGate.generated_probability);
  const realProbability = numberValue(binaryGate.real_probability);
  if (generatedProbability !== null || realProbability !== null) {
    const generatedText = generatedProbability === null ? "待核验" : `${Math.round(generatedProbability * 100)}%`;
    const realText = realProbability === null ? "待核验" : `${Math.round(realProbability * 100)}%`;
    hints.push(`Generated-vs-real gate：生成 ${generatedText} · 真实 ${realText}。`);
  }
  return hints.slice(0, 2);
}

function compactTamperNextSteps(
  asset: TamperForensicsResult["asset_results"][number],
  result: TamperForensicsResult,
): string[] {
  const suggestions = asset.review_suggestions.length ? asset.review_suggestions : result.recommended_next_steps;
  return suggestions.slice(0, 3);
}

function generatedSignalProbability(candidates: CandidateDistributionEntry[]): number {
  const generated = candidates
    .filter((item) => {
      const label = item.label.trim().toLowerCase().replace(/_/g, "-");
      return label && !["real", "real-photo", "unknown", "not-gpt-image2"].includes(label);
    })
    .reduce((total, item) => total + item.confidence, 0);
  return Math.max(0, Math.min(1, generated));
}

function assetUrl(value: string): string {
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  const apiBase = import.meta.env.VITE_API_BASE ?? (import.meta.env.PROD ? "" : "/api");
  if (apiBase === "/api" && value.startsWith("/evidence/")) {
    return `/api${value}`;
  }
  if (apiBase && value.startsWith("/")) {
    return `${apiBase}${value}`;
  }
  return value;
}

const REAL_ANALYSIS_CACHE_PREFIX = "smartpolice.realAnalysis.";

function readCachedRealAnalysis(caseId: string): RealCaseAnalysisResult | null {
  try {
    const raw = window.localStorage.getItem(`${REAL_ANALYSIS_CACHE_PREFIX}${caseId}`);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as RealCaseAnalysisResult;
    return parsed?.case?.id === caseId && parsed.report_markdown ? parsed : null;
  } catch {
    clearCachedRealAnalysis(caseId);
    return null;
  }
}

function writeCachedRealAnalysis(result: RealCaseAnalysisResult): void {
  try {
    window.localStorage.setItem(`${REAL_ANALYSIS_CACHE_PREFIX}${result.case.id}`, JSON.stringify(result));
  } catch {
    // Browser storage is only a UI convenience; the backend remains the source of truth.
  }
}

function clearCachedRealAnalysis(caseId: string): void {
  try {
    window.localStorage.removeItem(`${REAL_ANALYSIS_CACHE_PREFIX}${caseId}`);
  } catch {
    // Ignore storage failures.
  }
}

function candidateDistributionEntries(value: unknown): CandidateDistributionEntry[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    if (!asRecord(item)) {
      return [];
    }
    const label = stringValue(item.label) || stringValue(item.candidate);
    const confidence = numberValue(item.probability) ?? numberValue(item.confidence);
    if (!label || confidence === null) {
      return [];
    }
    const rank = numberValue(item.rank);
    const displayName = sourceLabel(label);
    return [{ label, confidence, rank: rank === null ? 0 : Math.max(1, Math.round(rank)), displayName }];
  }).sort((left, right) => {
    if (left.rank && right.rank) {
      return left.rank - right.rank;
    }
    return right.confidence - left.confidence;
  }).map((item, index) => ({
    ...item,
    rank: item.rank || index + 1,
  }));
}
