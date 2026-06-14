# Generator Track Improvement Plan

本计划把五条生成图检测/归因轨道拆成可执行的文件级改造任务。所有改造默认只产生 candidate 和报告，不自动替换 active。

## Execution Order

| Order | Track | Files | Concrete Change | Acceptance Gate |
| ---: | --- | --- | --- | --- |
| 1 | 真实/生成鲁棒初筛 | `backend/app/multimodal_training.py`, `tools/run_generator_experiment_suite.py`, `backend/tests/test_api.py` | 写入 profile 策略元数据；使用 source-balanced sampling、扰动增强、real-FPR-first threshold calibration；报告 generated recall、real FPR、binary AUC | Source Real FPR <= 0.10, Generated Recall >= 0.90, Source Macro-F1 >= 0.55 |
| 2 | 社交传播鲁棒性 | `backend/app/multimodal_training.py`, `tools/run_generator_experiment_suite.py` | 定位为 robustness/hard-negative mining；训练为 generated/real 二分类；保留社交/截图/压缩/真实 hard-negative 口径 | Source Real FPR <= 0.05, Generated Recall >= 0.90; 不直接建议激活 |
| 3 | GPT-image2 专项识别 | `backend/app/multimodal_training.py`, `tools/run_generator_experiment_suite.py`, `docs/generator_experiment_suite.md` | 固定 one-vs-rest 三分类：gpt-image2 / other-generated / real；增加 Qwen vs Scam-AI 互留口径备注；低置信输出 unknown | GPT-image2 Recall >= 0.60, Precision >= 0.70, Source Macro-F1 >= 0.45 |
| 4 | 多生成器归因 | `backend/app/multimodal_training.py`, `tools/run_generator_experiment_suite.py` | 改为 open-set attribution；只强归因多 dataset_source 覆盖类别，单来源或小样本类归 unknown/other | Label-covered Macro-F1 >= 0.30; unknown rate 必须报告 |
| 5 | Clean 原图归因上限 | `backend/app/multimodal_training.py`, `docs/generator_experiment_suite.md` | 只作为 clean upper-bound，不作为泛化/上线候选；报告 clean 与 source-holdout 落差 | Clean Macro-F1 >= 0.85; source-holdout 不作为激活依据 |

## File-Level Tasks

| File | Task |
| --- | --- |
| `backend/app/multimodal_training.py` | 新增 profile policy registry；model_card 写入目标、特征策略、标签策略、验收门槛、激活策略；保留现有 CLIP/频域/压缩/纹理特征并显式映射到 profile。 |
| `tools/run_generator_experiment_suite.py` | 读取 profile policy；输出中文轨道名、目标指标、达标状态、主要问题；按 profile 给出 recommendation，不再只用统一阈值。 |
| `backend/tests/test_api.py` | 覆盖 profile policy 写入、非 standard 不可激活、binary threshold calibration、social profile 二分类标签、report gate 字段。 |
| `docs/generator_experiment_suite.md` | 实验脚本自动生成；用于汇报 candidate matrix。 |
| `docs/benchmark_results.md` | 保留 current active 与 benchmark 口径，不混用旧 active。 |

## Implementation Notes

- 第一阶段不引入大模型下载或新的大规模数据下载，先用现有 4691 张外部池。
- CLIP/ViT embedding 已有开关 `SMARTPOLICE_ENABLE_CLIP`，本阶段先在策略里显式记录，不强制启用。
- 所有非 `standard_attribution` profile 必须 `activation_mode="candidate"`。
- Source-holdout 与 clean diagnostics 必须同时报告；clean 高分不能单独写成泛化结论。
- `social_propagation_robustness` 默认是 benchmark/hard-negative track，不直接建议激活。

## Validation Commands

```powershell
python -m py_compile D:\smartpolice\backend\app\multimodal_training.py D:\smartpolice\tools\run_generator_experiment_suite.py D:\smartpolice\backend\tests\test_api.py
python -m pytest D:\smartpolice\backend\tests -q
npm run build
python D:\smartpolice\tools\run_generator_experiment_suite.py --profiles binary_generated_gate,gpt_image2_ovr,multi_generator_label_covered,clean_origin_attribution,social_propagation_robustness --training-sample-limit 80 --candidate-max-augmented-samples 10 --candidate-eval-limit 20 --source-sample-limit 80 --max-holdout-groups 3
```
