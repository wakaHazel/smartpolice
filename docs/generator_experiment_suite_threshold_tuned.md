# Generator Experiment Suite

- Generated at: `2026-06-12T05:09:03.400558+00:00`
- Task: `vision_generator_attribution`
- Active unchanged: `True`
- Active before/after: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Downloads external benchmarks: `False`

## Profile Matrix

| Profile | Candidate | Clean n | Clean Macro-F1 | Clean AUC | Binary AUC | Gen P/R/F1 | Real P/R/F1 | Source Macro-F1 | Real FPR | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| binary_generated_gate | `36c54588-3b22-4bce-bde1-59e11beb6e26` | 20 | 0.633 | 0.979 | 0.979 | 1.000/0.917/0.957 | 0.889/1.000/0.941 | 0.422 | 0.500 | needs_more_data |
| social_propagation_robustness | `7f51ab4e-f08f-435e-9f68-4a6de5687fa6` | 20 | 0.562 | 0.920 | 0.920 | 0.882/1.000/0.938 | 1.000/0.600/0.750 | 0.500 | 0.000 | benchmark_only |

## Per-Profile Details

### binary_generated_gate

- Candidate: `36c54588-3b22-4bce-bde1-59e11beb6e26`
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.950` / `0.633` / `0.979`
- Positive label / positive AUC: `generated` / `0.979`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.633` / `0.979` / `0.917` / `0.000`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.422` / `0.427` / `0.839` / `0.500`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.443` / `0.443` / `0.667`
- Label distribution: `{"generated": 12, "real": 8}`
- Prediction distribution: `{"generated": 11, "real": 9}`

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 1.000 | 0.917 | 0.957 | 12 |
| real | 0.889 | 1.000 | 0.941 | 8 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |

### social_propagation_robustness

- Candidate: `7f51ab4e-f08f-435e-9f68-4a6de5687fa6`
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.900` / `0.562` / `0.920`
- Positive label / positive AUC: `generated` / `0.920`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.562` / `0.920` / `1.000` / `0.400`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.500` / `0.500` / `1.000` / `0.000`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.000` / `0.000` / `0.000`
- Label distribution: `{"generated": 15, "real": 5}`
- Prediction distribution: `{"generated": 17, "real": 3}`

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.882 | 1.000 | 0.938 | 15 |
| real | 1.000 | 0.600 | 0.750 | 5 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## Interpretation

- `binary_generated_gate` is judged by real false positive rate and generated recall, not attribution Macro-F1.
- `gpt_image2_ovr` is a one-vs-rest candidate; its labels are GPT-image2, real, and other-generated.
- `multi_generator_label_covered` excludes single-source generator classes from strong attribution claims.
- `clean_origin_attribution` and `social_propagation_robustness` separate clean upper-bound behavior from propagated-platform robustness.
- All rows are candidate experiments only; active replacement requires a separate explicit activation decision.
