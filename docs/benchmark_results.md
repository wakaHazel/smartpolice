# Baseline Benchmark Results

- Generated at: `2026-06-11T22:32:47.677721+00:00`
- Task: `vision_generator_attribution`
- Active model kept frozen: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Active unchanged after run: `True`
- Quick mode: `False`
- Temporary directory: `D:\smartpolice\tmp\baseline_matrix`
- External benchmark downloads in this run: `False`

## Active Baseline

| Item | Value |
| --- | --- |
| Training pool | 4691 samples |
| Active kind | local-generator-attribution-extratrees-v2 |
| Clean accuracy | 0.708 |
| Clean Macro-F1 | 0.503 |
| GPT-image-2 Recall | 0.915 |
| Augmentation features | 1200 |

## Robustness Matrix

| Condition | Accuracy | Macro-F1 | GPT-image-2 Recall | Avg Confidence |
| --- | ---: | ---: | ---: | ---: |
| clean | 1.000 | 0.933 | 1.000 | 0.645 |
| jpeg_q85 | 0.833 | 0.740 | 0.889 | 0.466 |
| jpeg_q60 | 0.858 | 0.765 | 0.889 | 0.489 |
| screenshot_resave | 0.583 | 0.496 | 0.222 | 0.370 |
| center_crop | 0.708 | 0.614 | 0.444 | 0.403 |
| watermark | 0.750 | 0.659 | 0.667 | 0.415 |

## Source Holdout

- Holdout key: `dataset_source`
- Sample count: `1000`
- Source groups: `14`
- Aggregate: `{"completed_group_count": 12.0, "mean_accuracy": 0.291, "mean_macro_f1": 0.124, "mean_gpt_image2_recall": 0.0, "seen_class_holdout_count": 845.0, "unseen_holdout_count": 143.0, "mean_seen_class_accuracy": 0.327, "mean_seen_class_macro_f1": 0.139, "mean_seen_class_gpt_image2_recall": 0.0, "mean_binary_accuracy": 0.862, "mean_binary_macro_f1": 0.401, "mean_generated_recall": 0.66, "mean_real_recall": 0.379, "mean_real_false_positive_rate": 0.121, "overall_real_false_positive_rate": 0.242, "real_support": 66.0, "real_false_positive_count": 16.0, "mean_confidence": 0.281, "label_covered_available": 1.0, "label_covered_holdout_count": 385.0, "label_covered_macro_f1": 0.354, "label_covered_binary_macro_f1": 0.464, "label_covered_generated_recall": 0.932, "label_covered_real_false_positive_rate": 0.267}`
- Seen-class diagnostic: `0.139` mean Macro-F1 on `845` holdout samples whose labels appear in training; `143` holdout samples use labels unseen by the training side.
- Binary screening diagnostic: `0.401` generated-vs-real Macro-F1, `0.660` generated recall, `0.242` overall real false positive rate (16/66 real samples).
- Baseline-style label-covered diagnostic: `0.354` attribution Macro-F1, `0.464` binary Macro-F1, `0.267` real false positive rate on `385` source-stratified holdout samples.

| Holdout group | Holdout samples | Macro-F1 | Seen-class Macro-F1 | Binary Macro-F1 | Real FP | Real FPR | Unseen samples | GPT-image-2 Recall | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen/Qwen-Image-Bench|Qwen/Qwen-Image-Bench:test | 224 | 0.000 | 0.000 | 0.500 | 0/0 | 0.000 | 78 | 0.000 | False |
| siddharthksah/DeepSafe-benchmark|siddharthksah/DeepSafe-benchmark:train | 222 | 0.069 | 0.069 | 0.421 | 1/11 | 0.091 | 0 | 0.000 | False |
| Rajarshi-Roy-research/Defactify_Image_Dataset|Rajarshi-Roy-research/Defactify_Image_Dataset:train | 167 | 0.142 | 0.142 | 0.349 | 0/11 | 0.000 | 0 | 0.000 | False |
| Robo531/ai-detector-benchmark-test-data|Robo531/ai-detector-benchmark-test-data:train | 140 | 0.144 | 0.188 | 0.425 | 9/11 | 0.818 | 37 | 0.000 | False |
| Rapidata/bananamark-dataset|Rapidata/bananamark-dataset:train | 68 | 0.122 | 0.122 | 0.500 | 0/0 | 0.000 | 0 | 0.000 | False |
| Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset|Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset:train_0001 | 41 | 0.028 | 0.028 | 0.500 | 0/0 | 0.000 | 0 | 0.000 | False |
| Scam-AI/gpt-image-2|Scam-AI/gpt-image-2:train | 39 | 0.000 | 0.000 | 0.233 | 0/0 | 0.000 | 0 | 0.000 | False |
| Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset|Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset:train_0010 | 28 | 0.000 | 0.000 | 0.500 | 0/0 | 0.000 | 28 | 0.000 | False |
| Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset|Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset:train_0005 | 26 | 0.000 | 0.000 | 0.320 | 0/0 | 0.000 | 0 | 0.000 | False |
| AdoCleanCode/Fakeddit-FalseConnection-Fusion-LocalImages::real-negative-pool|HuggingFace dataset + downloaded public image URLs -> generator_attribution_real_negative | 11 | 0.259 | 0.259 | 0.259 | 4/11 | 0.364 | 0 | 0.000 | False |
| TheKernel01/Tiny-GenImage::real-negative-pool|HuggingFace dataset -> generator_attribution_real_negative | 11 | 0.500 | 0.500 | 0.500 | 0/11 | 0.000 | 0 | 0.000 | False |
| lorenzo-morelli/image-splicing-deepfake-mix-test::real-negative-pool|HuggingFace dataset parquet image split -> generator_attribution_real_negative | 11 | 0.225 | 0.225 | 0.300 | 2/11 | 0.182 | 0 | 0.000 | False |

## Feature Ablation

| Feature set | Macro-F1 | GPT-image-2 Recall | Skipped |
| --- | ---: | ---: | --- |
| all | 0.444 | 1.000 | False |
| visual_semantic_only | 0.000 | 0.000 | False |
| frequency_texture_only | 0.106 | 0.000 | False |
| compression_traces_only | 0.100 | 0.500 | False |
| propagation_disturbance_only | 0.378 | 1.000 | False |
| text_context_proxy_only | 0.349 | 1.000 | False |
| visual_forensics_only | 0.335 | 1.000 | False |
| no_visual_semantic | 0.444 | 1.000 | False |
| no_text_context_proxy | 0.369 | 0.500 | False |
| no_frequency_texture | 0.562 | 1.000 | False |
| no_compression_traces | 0.540 | 1.000 | False |
| no_propagation_disturbance | 0.324 | 1.000 | False |

## Main Metrics

| Metric | Value |
| --- | ---: |
| Clean Macro-F1 | 0.933 |
| Robust average Macro-F1 | 0.655 |
| Clean-to-worst perturbation Macro-F1 drop | 0.437 |
| GPT-image-2 clean recall | 1.000 |
| Real clean false positive rate | 0.000 |

## Candidate Evaluation

- Candidate model: `20ba33df-c5e0-49d5-8f02-8a430aad90ab`
- Candidate status: `candidate_trained`
- Activated during this run: `False`
- Active before/after: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Strict recommendation: `suggest_activate`
- Reason: Candidate meets the stricter benchmark plan gate.

| Gate metric | Active | Candidate | Delta | Passed |
| --- | ---: | ---: | ---: | --- |
| clean_macro_f1 | 0.933 | 0.933 | 0.000 | True |
| clean_gpt_image2_recall | 1.000 | 1.000 | 0.000 | True |
| robust_average_macro_f1 | 0.655 | 0.698 | 0.043 | True |
| clean_real_false_positive_rate | 0.000 | 0.000 | 0.000 | True |

## Generalization Benchmark Borrowing

- GenImage contributes the cross-generator and degraded-image framing: evaluate generator shifts and perturbations such as compression, low-resolution variants, blur-like degradation, crop, and watermark without claiming full web coverage.
- AIGIBench contributes the external-blind-test framing: keep benchmark samples traceable by source and use source-holdout before considering any active replacement.
- SIDA/SID-Set contributes the social-media domain framing: treat social-platform images as a separate distribution with licensing, label, and sensitive-content checks before import.
- RRDataset and ITW-SM contribute the in-the-wild propagation framing: prioritize platform resampling, screenshot-resave, recapture/retake, and repeated upload chains as future robustness conditions.
- The project borrows these protocols, not their leaderboard claims: current output remains a suspected-source clue and weak source-holdout results are reported as a generalization boundary.
- Robustness rows that show `1.000` are bounded condition checks on the sampled robustness subset, not a full-score claim; the main clean validation Macro-F1 remains in the Active Baseline section.

| Baseline family | Borrowed generalization idea | Current project proxy | Current evidence |
| --- | --- | --- | --- |
| GenImage | Cross-generator and degraded-image testing | Generator labels plus `clean/jpeg/crop/watermark/screenshot_resave` robustness matrix | Robust average Macro-F1 `0.655`; screenshot-resave is the weakest condition |
| AIGIBench | External blind-test and source-aware evaluation | Strict `dataset_source` holdout plus label-covered source-stratified diagnostic | Strict mean Macro-F1 `0.124`; seen-class `0.139`; label-covered Macro-F1 `0.354` / binary `0.464`; strict real FPR `0.242` (`16/66`); `143` strict holdout samples have labels unseen by the training side |
| SIDA/SID-Set | Social-media distribution shift | Treat social-platform samples as a separate import domain after license and label checks | Protocol reference only; no sensitive social-media set is mixed into active |
| RRDataset / ITW-SM | In-the-wild propagation, resampling, recapture/retake | `screenshot_resave`, JPEG recompression, crop and watermark conditions | Recapture/retake and repeated upload chains remain next-round blind-test work |

## Interpretation

- These results use the currently imported external pool as the first benchmark matrix.
- GenImage and AIGIBench are prepared as local-data imports through `tools/prepare_benchmark_manifest.py`; they are not assumed to be present or fully downloaded.
- SIDA/SID-Set, RRDataset, and ITW-SM remain protocol references until licensing, labels, and sensitive-content constraints are confirmed.
- Model outputs remain suspected-source clues only; they do not replace C2PA, watermark checks, platform metadata, publication-chain evidence, or human review.
