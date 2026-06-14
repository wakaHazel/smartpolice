# Final Metric Lock

## Scope

This file locks the technical evidence for the current thesis/project line:

> GPT-image2 generated-image detection under metadata-loss and domestic social-platform download/transcode perturbations.

Do not use broad multi-generator attribution as the main document line. Keep it as future work only.

## Locked Model IDs

| Role | ID | Use |
| --- | --- | --- |
| Frozen active baseline | `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` | Historical/general active baseline; not replaced |
| Platform-enhanced GPT-image2 candidate | `faa78335-c4c5-4825-9095-13779af5cfec` | Component candidate for GPT-image2 download-transcode detection |

Active model remained unchanged in all final runs.

## Primary Report Sources

| Source | Path | Purpose |
| --- | --- | --- |
| Main comparison report | `D:\smartpolice\docs\platform_transcode_enhancement_comparison.md` | Thesis/table source for before-vs-after comparison |
| Platform artifact analysis | `D:\smartpolice\output\audits\platform_transcode_artifacts_latest.md` | Evidence for observed Weibo/XHS transcode artifacts |
| Main split candidate report | `D:\smartpolice\output\audits\platform_like_augmentation_candidate_latest.md` | odd calibration / even holdout result |
| Reverse split candidate report | `D:\smartpolice\output\audits\platform_like_augmentation_candidate_reverse_split.md` | even calibration / odd holdout sanity check |

## Locked Evaluation Protocol

- Platform returned samples are a small paired black-box test/analysis set, not the main training set.
- Thresholds are calibrated on 30 platform pairs and reported on the opposite 30-pair holdout.
- Official operating point: `condition_specific_gpt_score_threshold_target_recall`.
- Target GPT-image2 recall for threshold calibration: approximately `0.95`.
- Real FPR ceiling: `0.15`.
- Main conditions: `clean`, `weibo_download`, `xhs_download`.
- `weibo_screenshot` is a documented failure boundary, not a main optimization target.
- `xhs_screenshot` is unavailable and must not be fabricated.

## Locked Main Metrics

### odd calibration / even holdout

| Condition | Active Recall | Candidate Recall | Active FPR | Candidate FPR | Candidate Binary Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.667 | 0.933 | 0.000 | 0.000 | 0.644 |
| weibo_download | 0.333 | 0.867 | 0.000 | 0.000 | 0.622 |
| xhs_download | 0.667 | 0.933 | 0.000 | 0.000 | 0.644 |

### even calibration / odd holdout

| Condition | Active Recall | Candidate Recall | Active FPR | Candidate FPR | Candidate Binary Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.667 | 0.800 | 0.000 | 0.000 | 0.599 |
| weibo_download | 0.200 | 0.933 | 0.000 | 0.000 | 0.644 |
| xhs_download | 0.667 | 0.800 | 0.000 | 0.000 | 0.599 |

## Locked Artifact Findings

| Condition | Observation | Safe Interpretation |
| --- | --- | --- |
| `weibo_download` | 60/60 JPEG; dimensions preserved; median byte ratio `0.967`; JPEG quantization table changed for 22/60 pairs | In this black-box collection, Weibo download shows dimension-preserving JPEG re-encoding artifacts |
| `xhs_download` | 58/60 SHA-256 identical to clean; 60/60 dimensions preserved | In this creator-backend collection, XHS download is mostly clean-equivalent |
| `weibo_screenshot` | 60/60 low-resolution PNG; median area ratio `0.034` | Rendered thumbnail/screenshot chain; parked as failure boundary |

## Safe Claim

Use this wording:

> By analyzing observable download/transcode artifacts in a small paired Weibo/Xiaohongshu black-box returned set, the project parameterizes platform-like perturbation augmentation on the larger external training pool. Under half-split threshold calibration and held-out evaluation, the GPT-image2 platform-enhanced candidate substantially improves Weibo download recall while not increasing real-image false positives in this paired test.

## Do Not Claim

- Do not claim official Weibo/XHS codec rules.
- Do not claim full-network or all-platform robustness.
- Do not claim conclusive GPT-image2 attribution.
- Do not use broad multi-generator attribution as the main contribution.
- Do not present screenshot propagation as solved.
- Do not cite the earlier all-sample post-hoc `1.000` diagnostic as a strict main result.
