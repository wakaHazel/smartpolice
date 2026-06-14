# Platform Transcode Enhancement Comparison

- Created at: `2026-06-14T09:42:30.815436+00:00`
- Active model: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Enhancement candidate: `faa78335-c4c5-4825-9095-13779af5cfec`
- Active unchanged: `True`
- Operating point: `condition_specific_gpt_score_threshold_target_recall`

## Why This Matters

Real platform returned samples are used to infer observable download/transcode artifacts; the inferred perturbation is synthesized on the larger external pool, and the real returned download set is reserved for evaluation.

## Platform Artifact Findings

| Condition | N | Formats | Same SHA | Same Dim | Median Byte Ratio | Median Area Ratio | JPEG QTable Changed | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| weibo_download | 60 | {"JPEG": 60} | 0 | 60 | 0.967 | 1.000 | 22 | dimension-preserving JPEG re-encoding; suitable for synthetic platform-like augmentation |
| xhs_download | 60 | {"JPEG": 50, "PNG": 10} | 58 | 60 | 1.000 | 1.000 | 0 | mostly clean-equivalent in this creator-backend collection; useful as a stability check |
| weibo_screenshot | 60 | {"PNG": 60} | 0 | 0 | 0.267 | 0.034 | 50 | low-resolution PNG thumbnail/screenshot chain; parked as a failure boundary rather than main objective |

## Before vs After

| Condition | N | Recall Before | Recall After | Recall Delta | FPR Before | FPR After | F1 Before | F1 After | AUC Before | AUC After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 30 | 0.667 | 0.933 | +0.266 | 0.000 | 0.000 | 0.552 | 0.644 | 0.920 | 1.000 |
| weibo_download | 30 | 0.333 | 0.867 | +0.534 | 0.000 | 0.000 | 0.417 | 0.622 | 0.876 | 1.000 |
| xhs_download | 30 | 0.667 | 0.933 | +0.266 | 0.000 | 0.000 | 0.552 | 0.644 | 0.920 | 1.000 |

## Academic Takeaway

- The improvement is not only from adding more data; it comes from using the platform returned set to identify the actual download/transcode artifact family, then synthesizing that perturbation on the larger pool.
- Weibo download is the clearest example: dimension-preserving JPEG re-encoding was observed, and the enhanced candidate improves GPT-image2 recall from `0.333` to `0.867` on the held-out paired Weibo download test while keeping real FPR at `0.000` in this set.
- XHS download acts as a stability check because this collection is mostly clean-equivalent; the enhanced candidate reaches `0.933` recall there with real FPR `0.000`.
- Screenshot chains remain outside the main claim and should be written as a limitation/future-work branch.

## Limitations

- The paired platform set is small, so the result is a bounded black-box test rather than a platform-wide guarantee.
- The method infers observable artifacts from returned samples; it does not claim official Weibo/XHS codec rules.
- Screenshot propagation is intentionally excluded from the main objective because the recovered files are low-resolution rendered thumbnails and xhs_screenshot is unavailable.
- The candidate remains component-only; active model is not automatically replaced.
