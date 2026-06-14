from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GPT-image2 specialist experiment facts.")
    parser.add_argument(
        "--suite-json",
        default=r"D:\smartpolice\output\audits\generator-experiment-suite-20260612T180326Z.json",
    )
    parser.add_argument(
        "--pretrain-holdout-json",
        default=r"D:\smartpolice\output\audits\gpt_image2_ovr_source_holdout_after_qwen_repair.json",
    )
    parser.add_argument(
        "--output",
        default=r"D:\smartpolice\output\audits\gpt_image2_specialist_findings.md",
    )
    args = parser.parse_args()

    suite = json.loads(Path(args.suite_json).read_text(encoding="utf-8"))
    pretrain = json.loads(Path(args.pretrain_holdout_json).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(suite, pretrain), encoding="utf-8")
    print(f"wrote {output}")


def render(suite: dict[str, Any], pretrain: dict[str, Any]) -> str:
    experiment = first_experiment(suite)
    candidate = experiment.get("candidate", {}) if isinstance(experiment.get("candidate"), dict) else {}
    diagnostics = experiment.get("clean_diagnostics", {}) if isinstance(experiment.get("clean_diagnostics"), dict) else {}
    source_holdout = experiment.get("source_holdout", {}) if isinstance(experiment.get("source_holdout"), dict) else {}
    aggregate = source_holdout.get("aggregate", {}) if isinstance(source_holdout.get("aggregate"), dict) else {}
    acceptance = experiment.get("acceptance", {}) if isinstance(experiment.get("acceptance"), dict) else {}
    scam_group = find_group(source_holdout, "Scam-AI/gpt-image-2")
    qwen_group = find_group(source_holdout, "Qwen/Qwen-Image-Bench")
    pre_scam_group = find_group(pretrain, "Scam-AI/gpt-image-2")
    pre_qwen_group = find_group(pretrain, "Qwen/Qwen-Image-Bench")

    lines = [
        "# GPT-image2 专项实验审计摘要",
        "",
        "## 数据状态",
        "",
        "- 已修复 `Qwen/Qwen-Image-Bench` 本地图片路径：300 条可用样本，其中 GPT-image2 为 50 条。",
        "- 当前 GPT-image2 可用来源变为 2 个：`Scam-AI/gpt-image-2` 与 `Qwen/Qwen-Image-Bench`。",
        "- 该修复只更新外部样本图片路径，不训练、不激活模型。",
        "",
        "## 新 Candidate",
        "",
        f"- Candidate: `{candidate.get('id')}`",
        f"- Active: `{suite.get('active_model_id_before')}` -> `{suite.get('active_model_id_after')}`；保持不变 `{suite.get('active_unchanged')}`",
        f"- Clean GPT-image2 AUC/Recall: `{fmt(diagnostics.get('positive_auc'))}` / `{fmt(per_class_recall(diagnostics, 'gpt-image2'))}`",
        f"- Clean Macro-F1: `{fmt(diagnostics.get('macro_f1'))}`；注意该值是 sanity check，不代表跨来源泛化。",
        f"- Source Macro-F1: `{fmt(aggregate.get('mean_macro_f1'))}`；Label-covered Macro-F1: `{fmt(aggregate.get('label_covered_macro_f1'))}`",
        f"- Mean GPT-image2 recall: `{fmt(aggregate.get('mean_gpt_image2_recall'))}`；Overall Real FPR: `{fmt(aggregate.get('overall_real_false_positive_rate'))}`",
        f"- 验收状态: `{acceptance.get('status')}`；主要问题: {acceptance.get('main_issue')}",
        "",
        "## 来源互留结果",
        "",
        "| 阶段 | 留出来源 | Holdout 标签 | GPT-image2 Precision | GPT-image2 Recall | Macro-F1 | Binary Macro-F1 | Real FPR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, group in (
        ("修复后诊断", pre_scam_group),
        ("修复后诊断", pre_qwen_group),
        ("新 candidate", scam_group),
        ("新 candidate", qwen_group),
    ):
        if group:
            lines.append(group_row(stage, group))
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 补回 Qwen 第二来源后，GPT-image2 专项仍未跨来源达标。",
            "- 留出 `Scam-AI/gpt-image-2` 时 GPT-image2 recall 仍为 `0.000`，说明当前特征/训练策略强依赖来源分布。",
            "- 留出 `Qwen/Qwen-Image-Bench` 时 GPT-image2 recall 为 `0.360`，有一定识别能力但 precision 只有 `0.340`。",
            "- 可对外表述为：GPT-image2 clean/internal 检测很强，但社交平台/跨来源泛化仍是主要技术瓶颈。",
            "",
            "## 下一步",
            "",
            "1. 不再追求多生成器归因满分，主线继续放在低误报真实/生成初筛。",
            "2. GPT-image2 需要增加第二、第三独立来源和真实社交传播扰动样本，再做互留训练。",
            "3. 对 `Scam-AI` 和 `Qwen` 分别做来源归一化或 domain-adversarial/leave-source-out 校准，不把 clean 满分写成泛化结果。",
            "4. 当前 candidate 仅保留为 component candidate，不建议激活。",
            "",
        ]
    )
    return "\n".join(lines)


def first_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    experiments = payload.get("experiments")
    if isinstance(experiments, list) and experiments and isinstance(experiments[0], dict):
        return experiments[0]
    return {}


def find_group(payload: dict[str, Any], needle: str) -> dict[str, Any]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        source_holdout = payload.get("source_holdout", {})
        groups = source_holdout.get("groups") if isinstance(source_holdout, dict) else []
    if not isinstance(groups, list):
        return {}
    for group in groups:
        if isinstance(group, dict) and needle in str(group.get("holdout_group", "")):
            return group
    return {}


def group_row(stage: str, group: dict[str, Any]) -> str:
    return (
        "| "
        f"{stage} | "
        f"`{group.get('holdout_group')}` | "
        f"`{json.dumps(group.get('holdout_label_distribution', {}), ensure_ascii=False)}` | "
        f"{fmt(group.get('gpt_image2_precision'))} | "
        f"{fmt(group.get('gpt_image2_recall'))} | "
        f"{fmt(group.get('macro_f1'))} | "
        f"{fmt(group.get('binary_macro_f1'))} | "
        f"{fmt(group.get('real_false_positive_rate'))} |"
    )


def per_class_recall(diagnostics: dict[str, Any], label: str) -> Any:
    per_class = diagnostics.get("per_class")
    if isinstance(per_class, dict):
        metrics = per_class.get(label)
        if isinstance(metrics, dict):
            return metrics.get("recall")
    return None


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "-"


if __name__ == "__main__":
    main()
