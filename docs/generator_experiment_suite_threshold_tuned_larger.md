# Generator Experiment Suite

- Generated at: `2026-06-12T05:12:27.950132+00:00`
- Task: `vision_generator_attribution`
- Active unchanged: `True`
- Active before/after: `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad` -> `e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad`
- Downloads external benchmarks: `False`

## Profile Matrix

| Profile | Candidate | Clean n | Clean Macro-F1 | Clean AUC | Binary AUC | Gen P/R/F1 | Real P/R/F1 | Source Macro-F1 | Real FPR | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| binary_generated_gate | `c0c2370d-530a-4271-99d0-1c559c200534` | 40 | 0.558 | 0.990 | 0.990 | 0.800/1.000/0.889 | 1.000/0.625/0.769 | 0.475 | 0.143 | promising |
| social_propagation_robustness | `1d82bbb4-ca0b-4fe2-9ff9-80671826e541` | 40 | 0.562 | 0.938 | 0.938 | 0.882/1.000/0.938 | 1.000/0.600/0.750 | 0.466 | 0.000 | benchmark_only |

## Per-Profile Details

### binary_generated_gate

- Candidate: `c0c2370d-530a-4271-99d0-1c559c200534`
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.850` / `0.558` / `0.990`
- Positive label / positive AUC: `generated` / `0.990`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.553` / `0.990` / `1.000` / `0.375`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.475` / `0.476` / `0.912` / `0.143`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.482` / `0.482` / `0.000`
- Label distribution: `{"generated": 24, "real": 16}`
- Prediction distribution: `{"generated": 29, "real": 10, "unknown": 1}`

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.828 | 1.000 | 0.906 | 24 |
| real | 1.000 | 0.625 | 0.769 | 16 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |

### social_propagation_robustness

- Candidate: `1d82bbb4-ca0b-4fe2-9ff9-80671826e541`
- Clean accuracy / Macro-F1 / Macro OvR AUC: `0.900` / `0.562` / `0.938`
- Positive label / positive AUC: `generated` / `0.938`
- Binary Macro-F1 / binary AUC / generated recall / real FPR: `0.562` / `0.938` / `1.000` / `0.400`
- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `0.466` / `0.468` / `0.977` / `0.000`
- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `0.515` / `0.515` / `0.333`
- Label distribution: `{"generated": 30, "real": 10}`
- Prediction distribution: `{"generated": 34, "real": 6}`

| Clean class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| generated | 0.882 | 1.000 | 0.938 | 30 |
| real | 1.000 | 0.600 | 0.750 | 10 |
| unknown | 0.000 | 0.000 | 0.000 | 0 |


## Interpretation

- `binary_generated_gate` is judged by real false positive rate and generated recall, not attribution Macro-F1.
- `gpt_image2_ovr` is a one-vs-rest candidate; its labels are GPT-image2, real, and other-generated.
- `multi_generator_label_covered` excludes single-source generator classes from strong attribution claims.
- `clean_origin_attribution` and `social_propagation_robustness` separate clean upper-bound behavior from propagated-platform robustness.
- All rows are candidate experiments only; active replacement requires a separate explicit activation decision.
