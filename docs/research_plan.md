# SmartPolice Research Plan

## Material Passport

- Origin skills: `research-project-lead`, `academic-research-suite/experiment-agent`, `smartpolice-project-writing`
- Origin mode: research diagnosis + experiment planning
- Created: 2026-06-13
- Verification status: DIRECTION_SET
- Scope: `vision_generator_attribution` research direction for `D:\smartpolice`

## Direction Correction

The project should stop treating "train another stronger multi-generator candidate" as the main path to breakthrough. The current evidence shows a stable pattern:

- clean and sampled perturbation metrics can look strong
- source-holdout generalization drops sharply
- real-image false positives remain operationally sensitive
- multi-generator attribution is not yet reliable enough to carry the project narrative

The corrected direction is:

> Build a trustworthy police-facing AIGC image verification workflow whose core contribution is not definitive generator attribution, but a bounded evidence chain: low-false-positive generated-image screening, GPT-image2 suspected-source clues, propagation robustness diagnostics, and auditable human-review reporting.

## Current Stage

Current project stage: evaluation and ablation, transitioning into competition packaging.

The project already has:

- runnable frontend/backend prototype
- local supervised visual attribution head
- external sample import
- perturbation augmentation
- robustness matrix
- source-holdout diagnostics
- feature ablation
- evidence-chain and report workflow

The project is not yet ready to claim broad real-world generator attribution.

## Strongest Current Claim

Supported, bounded claim:

> The system demonstrates a police-oriented AIGC image verification workflow that preserves image evidence, produces suspected generated-source clues, evaluates social-platform disturbances, and organizes model outputs into auditable evidence-chain and report drafts. It is useful as an auxiliary研判 prototype, while cross-source generalization remains the main research limitation.

## Weakest Current Link

The weakest link is source-holdout generalization, especially when generator labels, data sources, and real-negative domains shift. Current five-track evidence shows:

- `binary_generated_gate`: source real FPR `0.179`, generated recall `0.391`
- `gpt_image2_ovr`: clean GPT-image2 recall `1.000`, but source Macro-F1 `0.228`, real FPR `0.282`
- `mainstream_five_attribution`: clean Macro-F1 `0.986`, but source Macro-F1 `0.214`, label-covered Macro-F1 `0.311`

This means clean sanity is not the decision metric. Source-holdout and operational false-positive risk are the decision metrics.

## Research Questions

### RQ1: Low-False-Positive Screening

In social-platform disturbed AIGC image verification, can a generated-vs-real gate reduce real-image false positives below `0.10` on source-holdout while keeping generated recall useful enough for警务辅助研判?

Primary metrics:

- source-holdout real false positive rate
- source-holdout generated recall
- binary Macro-F1

### RQ2: GPT-image2 Suspected-Source Clue

Can GPT-image2 be treated as a high-precision suspected-source clue under bounded conditions, while explicitly refusing broad generator-source claims when source-holdout fails?

Primary metrics:

- GPT-image2 recall and precision on clean/candidate evaluation
- source-holdout Macro-F1
- source-holdout real false positive rate
- cross-source GPT-image2 subgroup behavior

### RQ3: Propagation Robustness

Which disturbance conditions most damage the workflow, and can screenshot-resave/crop/watermark robustness be improved without raising real-image false positives?

Primary metrics:

- per-condition Macro-F1
- worst-condition drop
- screenshot-resave GPT-image2 recall
- real false positive rate under disturbed conditions

### RQ4: Evidence-Chain Value

Does converting model outputs into hash-backed evidence-chain entries improve auditability and reviewer trust compared with a standalone detector score?

Evidence:

- report includes hash, audit ID, model version, clue type, uncertainty, and human-review statement
- demo shows upload -> model clue -> evidence chain -> disposal suggestion -> report draft

## Hypotheses

| ID | Hypothesis | Current status | Next evidence move |
| --- | --- | --- | --- |
| H1 | Real-FPR-first thresholding plus hard-negative real photos can reduce source real FPR to `<=0.10` without destroying generated recall. | partial; current binary gate real FPR `0.179` | run a focused real-negative/threshold sprint |
| H2 | GPT-image2 can be a high-confidence clue on clean and selected disturbed samples, but not a standalone cross-source attribution claim. | supported for clean, weak for source-holdout | keep as bounded clue; add cross-source GPT-image2 analysis |
| H3 | Five-class mainstream attribution is a research track, not a competition main claim yet. | current label-covered Macro-F1 `0.311`, below `0.350` target | park as secondary; only report as exploratory |
| H4 | Evidence-chain packaging is the project's practical breakthrough even while attribution metrics remain limited. | supported by system workflow | make it the competition narrative spine |

## Milestones

### Milestone 1: Freeze And Explain

Goal: freeze the current active model and current five-track evidence as a baseline, with no auto-activation.

Acceptance criteria:

- active model unchanged
- latest five-track report summarized
- no candidate is promoted from a failed profile
- claim ledger updated

### Milestone 2: False-Positive Sprint

Goal: improve police-facing trust by reducing real-image false positives.

Acceptance criteria:

- source real FPR `<=0.10` on the focused binary gate or clear failure explanation
- generated recall reported alongside FPR
- hard-negative groups identified
- failure cases documented

### Milestone 3: Competition Packaging

Goal: make materials internally consistent and judge-proof.

Acceptance criteria:

- README, report, video script, PDF/DOCX, and UI screenshots use one metric version
- 5-minute MP4 exists outside source package
- source ZIP excludes datasets, tmp, logs, node_modules, LibreOffice, and generated caches
- boundary wording is consistent: "suspected clue", not "final evidence"

## What To Stop Doing For Now

- Do not add new UI features unless needed for the demo.
- Do not run broad five-track retraining as the main path to progress.
- Do not promote any component candidate that fails its source-holdout gate.
- Do not lead with multi-generator attribution in competition materials.
- Do not hide source-holdout weakness; convert it into a credible limitation and research plan.

## Immediate Next Action

Let the running five-track process finish if it is still making progress, but treat it as evidence collection only. The next intentional research sprint should be the false-positive sprint, not another all-track sweep.
