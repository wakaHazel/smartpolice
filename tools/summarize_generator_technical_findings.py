from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generator experiment facts for the technical track.")
    parser.add_argument(
        "--input",
        default=r"D:\smartpolice\output\audits\generator_experiment_suite_latest.json",
    )
    parser.add_argument(
        "--output",
        default=r"D:\smartpolice\output\audits\generator_technical_findings.md",
    )
    parser.add_argument(
        "--threshold-sweep",
        default=r"D:\smartpolice\output\audits\binary_gate_threshold_sweep_489527cd_quick.json",
    )
    parser.add_argument(
        "--calibration-eval",
        default=r"D:\smartpolice\output\audits\binary_gate_calibration_eval_489527cd_360.json",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sweep = None
    sweep_path = Path(args.threshold_sweep)
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    calibration = None
    calibration_path = Path(args.calibration_eval)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    output_path.write_text(render(payload, sweep, calibration), encoding="utf-8")
    print(f"wrote {output_path}")


def render(
    payload: dict[str, Any],
    threshold_sweep: dict[str, Any] | None = None,
    calibration_eval: dict[str, Any] | None = None,
) -> str:
    experiments = payload.get("experiments", [])
    lines = [
        "# 技术线实验事实摘要",
        "",
        f"- 任务: `{payload.get('task_type')}`",
        f"- Active: `{payload.get('active_model_id_before')}` -> `{payload.get('active_model_id_after')}`",
        f"- Active 保持不变: `{payload.get('active_unchanged')}`",
        "",
        "## 主线结论",
        "",
        "| 技术主线 | Candidate | Source Macro-F1 | Label-covered Macro-F1 | Generated Recall | Real FPR | Clean Macro-F1 | 结论 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for experiment in experiments:
        profile = str(experiment.get("profile") or "")
        if profile not in {"binary_generated_gate", "gpt_image2_ovr", "social_propagation_robustness"}:
            continue
        policy = experiment.get("profile_policy", {}) if isinstance(experiment.get("profile_policy"), dict) else {}
        candidate = experiment.get("candidate", {}) if isinstance(experiment.get("candidate"), dict) else {}
        aggregate = (
            experiment.get("source_holdout", {}).get("aggregate", {})
            if isinstance(experiment.get("source_holdout"), dict)
            else {}
        )
        diagnostics = experiment.get("clean_diagnostics", {}) if isinstance(experiment.get("clean_diagnostics"), dict) else {}
        lines.append(
            "| "
            f"{policy.get('chinese_name') or profile} | "
            f"`{candidate.get('id')}` | "
            f"{fmt(aggregate.get('mean_macro_f1'))} | "
            f"{fmt(aggregate.get('label_covered_macro_f1'))} | "
            f"{fmt(aggregate.get('mean_generated_recall'))} | "
            f"{fmt(aggregate.get('overall_real_false_positive_rate'))} | "
            f"{fmt(diagnostics.get('macro_f1'))} | "
            f"{technical_conclusion(profile, experiment)} |"
        )

    lines.extend(
        [
            "",
            "## 阈值校准诊断",
            "",
        ]
    )
    if threshold_sweep:
        recommended = threshold_sweep.get("recommended") if isinstance(threshold_sweep.get("recommended"), dict) else {}
        lines.extend(
            [
                f"- Candidate: `{threshold_sweep.get('candidate_id')}`",
                f"- 样本量: `{threshold_sweep.get('sample_count')}`；标签分布: `{json.dumps(threshold_sweep.get('label_distribution', {}), ensure_ascii=False)}`",
                f"- 当前 artifact 阈值/保护边距: `{fmt(threshold_sweep.get('current_artifact_threshold'))}` / `{fmt(threshold_sweep.get('current_artifact_margin'))}`",
                f"- 推荐扫描点: threshold `{fmt(recommended.get('threshold'))}`, margin `{fmt(recommended.get('real_protection_margin'))}`",
                f"- 推荐点指标: Real FPR `{fmt(recommended.get('real_false_positive_rate'))}`, Generated Recall `{fmt(recommended.get('generated_recall'))}`, Binary Macro-F1 `{fmt(recommended.get('binary_macro_f1'))}`",
                f"- 解释: {threshold_sweep.get('interpretation')}",
            ]
        )
    else:
        lines.append("- 未找到阈值扫描结果。")

    lines.extend(
        [
            "",
            "## 阈值校准扰动验证",
            "",
        ]
    )
    if calibration_eval:
        recommended = (
            calibration_eval.get("recommended")
            if isinstance(calibration_eval.get("recommended"), dict)
            else {}
        )
        clean = (
            recommended.get("summary", {}).get("clean", {})
            if isinstance(recommended.get("summary"), dict)
            else {}
        )
        robust = (
            recommended.get("summary", {}).get("robust_average", {})
            if isinstance(recommended.get("summary"), dict)
            else {}
        )
        lines.extend(
            [
                f"- Candidate: `{calibration_eval.get('candidate_id')}`",
                f"- 样本量: `{calibration_eval.get('sample_count')}`；条件: `{', '.join(calibration_eval.get('conditions', []))}`",
                f"- 推荐阈值/保护边距: `{fmt(recommended.get('threshold'))}` / `{fmt(recommended.get('real_protection_margin'))}`",
                f"- Clean: AUC `{fmt(clean.get('binary_auc'))}`，2-class Macro-F1 `{fmt(clean.get('binary_macro_f1'))}`，Generated Recall `{fmt(clean.get('generated_recall'))}`，Real FPR `{fmt(clean.get('real_false_positive_rate'))}`",
                f"- 扰动平均: AUC `{fmt(robust.get('binary_auc'))}`，2-class Macro-F1 `{fmt(robust.get('binary_macro_f1'))}`，Generated Recall `{fmt(robust.get('generated_recall'))}`，Real FPR `{fmt(robust.get('real_false_positive_rate'))}`",
                f"- Active 保持不变: `{calibration_eval.get('active_unchanged')}`",
                f"- 解释: {calibration_eval.get('interpretation')}",
            ]
        )
        weak_groups = recommended.get("weak_source_groups", [])
        if isinstance(weak_groups, list) and weak_groups:
            lines.extend(["", "| 主要问题 | 来源组 | 条件 | 错误数/Support | Rate |", "| --- | --- | --- | ---: | ---: |"])
            for item in weak_groups[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| "
                    f"{item.get('kind')} | "
                    f"`{item.get('group')}` | "
                    f"{item.get('condition')} | "
                    f"{item.get('error_count')}/{item.get('support')} | "
                    f"{fmt(item.get('rate'))} |"
                )
    else:
        lines.append("- 未找到扰动阈值验证结果。")

    lines.extend(
        [
            "",
            "## 失败边界",
            "",
            "- 低误报真假初筛：source-holdout 真实图误报仍高于目标，说明真实负样本 hard-negative 仍不足。",
            "- GPT-image2 专项：clean 内部识别很强，但跨来源 Macro-F1 未达标，不能把 clean 高分写成平台泛化能力。",
            "- 社交传播鲁棒性：真实误报接近目标，但生成召回不足，截图/转码/水印扰动下仍会漏检。",
            "- 多生成器归因：仅作为扩展诊断保留，不再作为当前论文/技术突破主线。",
            "",
            "## 下一步技术任务",
            "",
            "1. 先验证真实/生成初筛阈值校准：用独立 source-holdout/扰动子集确认 threshold≈0.65 是否稳定。",
            "2. GPT-image2 专项改做来源互留：Qwen、Scam-AI、本地池分别作为 holdout，报告 recall/precision。",
            "3. 社交扰动实验固定 clean、jpeg_q85、jpeg_q60、screenshot_resave、center_crop、watermark 六条件，并输出 clean-to-perturbation drop。",
            "4. 把 Defactify、AIGC-Detection-Benchmark、Synthbuster 中导致漏检/误报的来源组整理为 hard-negative pool。",
            "5. 继续冻结 active；所有新模型只作为 candidate/component candidate。",
            "",
        ]
    )
    return "\n".join(lines)


def technical_conclusion(profile: str, experiment: dict[str, Any]) -> str:
    aggregate = (
        experiment.get("source_holdout", {}).get("aggregate", {})
        if isinstance(experiment.get("source_holdout"), dict)
        else {}
    )
    diagnostics = experiment.get("clean_diagnostics", {}) if isinstance(experiment.get("clean_diagnostics"), dict) else {}
    if profile == "binary_generated_gate":
        return (
            "clean 可用，但跨来源生成召回不足；优先调阈值和补真实 hard-negative。"
            if number(aggregate.get("mean_generated_recall")) < 0.9
            else "可作为低误报初筛候选。"
        )
    if profile == "gpt_image2_ovr":
        return (
            "clean GPT-image2 识别强，跨来源不足；适合做专项互留实验。"
            if number(diagnostics.get("positive_auc")) >= 0.95
            else "专项特征仍不足。"
        )
    if profile == "social_propagation_robustness":
        return "真实误报较低但生成召回不足；适合作为扰动鲁棒性和 hard-negative 线。"
    return "扩展诊断。"


def number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def fmt(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "-"


if __name__ == "__main__":
    main()
