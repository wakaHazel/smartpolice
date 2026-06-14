# SmartPolice Experiment Backlog

## Material Passport

- Origin skills: `research-project-lead`, `academic-research-suite/experiment-agent`
- Created: 2026-06-13
- Verification status: METRIC_LOCKED
- Scope: experiments that can move `vision_generator_attribution` from metric chasing to evidence-driven progress

## Final Metric Lock

Use `D:\smartpolice\docs\final_metric_lock.md` as the source of truth for thesis/report metrics.

Primary table source:

- `D:\smartpolice\docs\platform_transcode_enhancement_comparison.md`
- `D:\smartpolice\output\audits\platform_like_augmentation_candidate_latest.md`
- `D:\smartpolice\output\audits\platform_like_augmentation_candidate_reverse_split.md`

Do not cite the earlier all-sample post-hoc `1.000` diagnostic as a strict main result. The locked result uses half-split threshold calibration and held-out reporting.

## Priority Rules

1. Prefer experiments that reduce uncertainty about source-holdout generalization.
2. Prefer false-positive reduction over headline attribution scores.
3. Keep multi-generator attribution secondary until label-covered Macro-F1 and generated recall move together.
4. Every experiment must have a stop condition before it runs.

## Sprint 1: False-Positive Reduction

### Experiment FP-1: Hard-Negative Real Guard

| Field | Content |
| --- | --- |
| Research question | Can real-negative expansion and threshold calibration reduce source real FPR below `0.10`? |
| Hypothesis | Hard-negative real samples plus real-FPR-first thresholding will reduce source real FPR without collapsing generated recall. |
| Dataset | current external pool plus available real-negative pools |
| Split | source-holdout by `dataset_source` |
| Baseline | current `binary_generated_gate` result: real FPR `0.179`, generated recall `0.391` |
| Experimental variable | real-negative sample coverage and generated gate threshold policy |
| Metrics | source real FPR, source generated recall, binary Macro-F1, label-covered real FPR |
| Success threshold | source real FPR `<=0.10`; generated recall should not fall below a clearly disclosed usable floor |
| Failure threshold | real FPR remains `>0.15` or generated recall collapses below `0.35` |
| Interpretation rule | If FPR improves but recall collapses, report as a high-precision review gate, not detector breakthrough. |

Suggested command shape:

```powershell
python D:\smartpolice\tools\run_generator_experiment_suite.py `
  --profiles binary_generated_gate `
  --training-sample-limit 1400 `
  --candidate-max-augmented-samples 800 `
  --candidate-eval-limit 140 `
  --source-sample-limit 1000 `
  --max-holdout-groups 8 `
  --docs-path D:\smartpolice\docs\generator_experiment_suite_binary_guard.md
```

### Experiment FP-2: Threshold Diagnostic Without Retraining

Goal: determine whether poor real FPR is a data/model problem or a threshold problem.

Output expected:

- per-source real probability distribution
- generated probability distribution
- candidate thresholds and their FPR/recall tradeoff
- recommended threshold for "review operating point"

Stop condition:

- if no threshold can reduce real FPR without unacceptable recall loss, prioritize data curation over more tuning.

## Sprint 2: GPT-image2 Clue Validation

### Experiment G2-1: GPT-image2 Cross-Source Split

| Field | Content |
| --- | --- |
| Research question | Does GPT-image2 recognition survive source changes, or is it source-artifact dependent? |
| Hypothesis | Clean GPT-image2 performance is high, but source-holdout failures reflect source/domain artifacts. |
| Baseline | current `gpt_image2_ovr`: clean GPT-image2 recall `1.000`, source Macro-F1 `0.228`, real FPR `0.282` |
| Metrics | per-source GPT-image2 recall, real FPR, other-generated confusion |
| Success threshold | identify at least one stable source condition and one failure condition with explanation |
| Interpretation rule | Use as suspected-source clue only; do not make conclusive attribution claim. |

Suggested command shape:

```powershell
python D:\smartpolice\tools\run_generator_experiment_suite.py `
  --profiles gpt_image2_ovr `
  --training-sample-limit 1400 `
  --candidate-max-augmented-samples 800 `
  --candidate-eval-limit 140 `
  --source-sample-limit 1000 `
  --max-holdout-groups 8 `
  --docs-path D:\smartpolice\docs\generator_experiment_suite_gpt_image2_clue.md
```

## Sprint 3: Propagation Robustness

### Experiment PR-1: Download/Transcode Platform Focus

| Field | Content |
| --- | --- |
| Research question | Can GPT-image2 detection survive domestic social-platform download/transcode propagation without using the small returned set as training data? |
| Baseline | active platform report shows Weibo download GPT-image2 recall `0.267`, XHS download recall `0.667`, real FPR `0.000` on this paired set. |
| Metrics | GPT-image2 recall, binary Macro-F1, real FPR, GPT-image2 AUC, average confidence |
| Success threshold | improve Weibo download recall while keeping real FPR at a reportable balanced level. |
| Interpretation rule | Treat Weibo/XHS as black-box returned samples; use observed artifacts to synthesize augmentation on the larger pool, not as official platform codec rules. |

### Experiment PR-2: Real Platform Transcode Holdout

| Field | Content |
| --- | --- |
| Research question | Does GPT-image2 detection survive real Weibo/Xiaohongshu download propagation rather than only synthetic JPEG perturbations? |
| Dataset | `D:\smartpolice\platform_eval\upload_batch_60` clean pairs plus returned files in `D:\smartpolice\platform_eval\returned` |
| Current collection | 60 clean pairs; main test uses 60 `weibo_download` and 60 `xhs_download`. `weibo_screenshot` is retained only as a documented failure-boundary subset; `xhs_screenshot` remains unavailable and is not fabricated. |
| Baseline | `output\audits\platform_transcode_eval_latest.md`: clean GPT-image2 recall `0.667`, Weibo download recall `0.267`, Weibo screenshot recall `0.000`, XHS download recall `0.667`; real FPR `0.000` in all four evaluated conditions. |
| Experimental variable | Platform-like synthetic augmentation inferred from Weibo/XHS download artifacts and applied to the larger external pool. |
| Metrics | GPT-image2 recall, generated recall, binary Macro-F1, real FPR, GPT-image2 AUC, confidence delta from clean. |
| Success threshold | Improve Weibo download GPT-image2 recall and keep XHS download stable; do not optimize on screenshot in the main run. |
| Stop condition | If download/transcode recall does not improve after platform-like augmentation on the larger pool, prefer threshold/calibration or data expansion over training on the 60-pair returned set. |
| Integration command | `python D:\smartpolice\tools\analyze_platform_transcode_artifacts.py`, then `python D:\smartpolice\tools\run_platform_like_augmentation_candidate.py`. |

Latest candidate-only run:

- Historical command: `python D:\smartpolice\tools\run_platform_candidate_experiment.py`
- Historical candidate id: `f618ea5f-ad23-4314-90dd-4f0feb756fb6`
- Active remained frozen: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Official operating point: GPT-image2 score threshold `0.60`, acceptable real FPR threshold `0.10`.
- Weibo download holdout recall improves `0.333 -> 0.800`, real FPR changes `0.000 -> 0.067`, Binary Macro-F1 improves `0.417 -> 0.577`.
- Clean and XHS download holdout recall also improve `0.667 -> 0.800`, real FPR changes `0.000 -> 0.067`.
- Weibo screenshot holdout remains `0.000 -> 0.000`; it is now parked as a documented failure boundary, not a main optimization target.
- Outputs: `D:\smartpolice\output\audits\platform_candidate_experiment_latest.json`, `.md`, `.csv`.

Locked main result:

- Command shape: `python D:\smartpolice\tools\run_platform_like_augmentation_candidate.py --candidate-id faa78335-c4c5-4825-9095-13779af5cfec --variants download --target-gpt-recall 0.95`
- Candidate id: `faa78335-c4c5-4825-9095-13779af5cfec`
- Main split: odd 30 pairs calibrate threshold, even 30 pairs report holdout.
- Weibo download recall improves `0.333 -> 0.867`, real FPR remains `0.000 -> 0.000`.
- Reverse split: even 30 pairs calibrate threshold, odd 30 pairs report holdout; Weibo download recall improves `0.200 -> 0.933`, real FPR remains `0.000 -> 0.000`.
- Use `docs\final_metric_lock.md` for exact locked values and safe wording.

## Parked: Mainstream Five Attribution

Current status:

- clean sanity is high
- source-holdout generated recall is weak
- label-covered Macro-F1 remains below target

Do not spend the next sprint here unless FP and GPT-image2 clue tracks have been stabilized. This track is useful for research narrative and future work, but it should not carry the competition claim now.

## Submission/Packaging Tasks

| Task | Acceptance criterion |
| --- | --- |
| Metric lock | README, DOCX/PDF, video script, and UI screenshots use one metric snapshot |
| Demo video | MP4 under 5 minutes exists and matches the script |
| Source ZIP hygiene | excludes datasets, tmp, logs, node_modules, LibreOffice, generated caches |
| Runbook | clean environment commands are documented |
| Boundary review | all model outputs are described as suspected clues |
