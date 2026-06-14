# SmartPolice Claim Ledger

## Material Passport

- Origin skills: `research-project-lead`, `smartpolice-project-writing`, `academic-research-suite/academic-paper-reviewer`
- Created: 2026-06-13
- Verification status: INITIAL_LEDGER
- Purpose: keep competition, research, and demo claims aligned with evidence

## Claim Status Rules

- supported: backed by reproducible project evidence under stated conditions
- partial: true under narrow conditions, needs boundary wording
- weak: suggestive but not reliable enough to lead
- unsupported: do not use until evidence exists

## Ledger

| Claim | Evidence | Scope | Risk | Status | Safe wording |
| --- | --- | --- | --- | --- | --- |
| The project is not a simple wrapper. | Local attribution head, dataset import, perturbation augmentation, source-holdout, feature ablation, candidate/active lifecycle, evidence-chain workflow. | Engineering and research prototype. | Do not imply foundation-model training. | supported | "核心工作是本地视觉归因头与证据链工作流，不是简单套壳。" |
| The system can provide AIGC suspected-source clues. | Active model and component candidates output labels/confidence; GPT-image2 clean recall can be high in bounded runs. | Auxiliary clue generation. | Source-holdout weakness; not definitive. | partial | "提供疑似生成来源线索，供人工复核。" |
| The system can conclusively identify GPT-image2. | No evidence supports legal/conclusive identification. | None. | Overclaiming; forensic/legal risk. | unsupported | Do not say this. |
| Clean/sampled robustness evidence is meaningful. | `docs/benchmark_results.md` reports robust average Macro-F1 `0.655`; clean/watermark/JPEG/crop/screenshot rows exist. | Current sample and conditions only. | Clean rows can be mistaken for real-world generalization. | partial | "在当前样本和扰动条件下完成鲁棒性复测。" |
| Real platform propagation evidence exists. | 60 clean pairs plus recovered returned files: Weibo download, Weibo rendered screenshot, XHS creator-backend returned images. Latest active report is `output/audits/platform_transcode_eval_latest.md`; main follow-up now focuses on download/transcode conditions only. | Small paired black-box platform sample. | Do not infer exact platform codec rules or full network distribution. | partial | "构建了小规模真实平台传播配对评测集，主线聚焦微博/小红书下载转码链路；截图链路作为失败边界单独披露。" |
| Platform transcode analysis-driven augmentation improves download recall. | `output/audits/platform_transcode_enhancement_comparison_latest.md`: candidate `faa78335-c4c5-4825-9095-13779af5cfec` uses Weibo download-like synthetic augmentation on the larger pool and a target-recall GPT-image2 operating point. Threshold is calibrated on 30 platform pairs and reported on the opposite 30-pair holdout: Weibo download GPT-image2 recall improves `0.333 -> 0.867`, real FPR remains `0.000 -> 0.000`; reverse split also improves Weibo download `0.200 -> 0.933`; active unchanged. | Small paired black-box platform download set; component candidate, not active replacement. | Do not claim platform-wide guarantee or official transcode rules; screenshot remains parked as failure boundary. | partial | "通过分析微博/小红书回收图的可观察转码痕迹，并在大训练池上合成平台近似扰动，download 链路 GPT-image2 检出显著提升；阈值采用半数校准、半数 holdout 汇报，结果仍限定在当前小规模配对黑盒测试集。" |
| Broad cross-source generator attribution is solved. | Strict source-holdout Macro-F1 remains low; five-track source metrics below gates. | None. | Reviewer will reject. | unsupported | Do not lead with this claim. |
| Real-image false positives are under control. | Some clean evaluations have low real FPR; current source-holdout binary gate real FPR `0.179`. | Clean/sample-specific only. | Police workflow is sensitive to false positives. | weak | "真实图误报仍是下一阶段重点控制指标。" |
| Evidence-chain packaging is a practical contribution. | System preserves hash/audit ID/model version and generates evidence-chain/report draft. | Police auxiliary workflow prototype. | Needs human review and platform/original-file evidence. | supported | "把模型线索组织为可复核证据链和报告草稿。" |
| Benchmark families are fully reproduced. | Current docs borrow protocols; not all external benchmark datasets are fully present/downloaded. | Protocol reference. | Borrowed leaderboard impression. | partial | "借鉴公开 benchmark 的评测思想，不借用 leaderboard 分数。" |
| The project is ready for competition packaging if metrics are locked and video/source package are cleaned. | Prior audit found engineering tests pass but materials/packaging drift. | Submission readiness. | Missing MP4/source ZIP bloat/version drift. | partial | "工程基本盘可用，提交前需锁定口径并清理交付包。" |

## Forbidden Wording

- "确定来自 GPT-image2"
- "直接作为定案证据"
- "真实网络全场景可用"
- "已经掌握微博/小红书完整转码规则"
- "多生成器归因已经解决"
- "训练了 Qwen3-VL/基础多模态大模型"

## Preferred Wording

- "疑似生成来源线索"
- "辅助研判"
- "人工复核"
- "证据链草稿"
- "在当前样本和扰动条件下"
- "source-holdout 显示仍存在泛化边界"

## Reviewer Red Flags To Check Before Any Submission

1. Does the document cite one consistent metric snapshot?
2. Does any paragraph turn a clue into a conclusion?
3. Are weak source-holdout results disclosed?
4. Are demo video, PDF/DOCX, README, UI screenshots, and source package aligned?
5. Is false-positive risk discussed before police deployment claims?
