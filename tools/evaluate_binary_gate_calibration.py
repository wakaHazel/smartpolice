from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.multimodal_training import (  # noqa: E402
    GENERATOR_ATTRIBUTION_TASK,
    _binary_generation_metrics,
    _classification_metrics,
    _generator_attribution_features,
    _generator_predictor_from_artifact,
    _generator_sample_domain,
    _predict_generator_label,
    _robustness_condition_description,
    _source_holdout_group_name,
    _supported_robustness_conditions,
    _task_relevant_samples,
    _write_robustness_variant,
    get_vision_competition_summary,
)
from app.storage import (  # noqa: E402
    get_vision_training_artifact_by_id,
    initialize_database,
    list_external_training_samples,
)
from tools.run_generator_experiment_suite import (  # noqa: E402
    _balanced_profile_samples,
    _roc_auc,
    _score_for_label,
    _select_profile_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed binary gate thresholds across perturbations and source groups."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--profile", default="binary_generated_gate")
    parser.add_argument("--sample-limit", type=int, default=600)
    parser.add_argument(
        "--conditions",
        default="clean,jpeg_q85,jpeg_q60,screenshot_resave,center_crop,watermark",
    )
    parser.add_argument("--thresholds", default="0.567,0.65,0.75")
    parser.add_argument("--margins", default="0.00,0.06")
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "audits" / "binary_gate_calibration_eval.json"),
    )
    parser.add_argument(
        "--markdown-output",
        default=str(ROOT / "output" / "audits" / "binary_gate_calibration_eval.md"),
    )
    parser.add_argument(
        "--from-json",
        default="",
        help="Recompute recommendation and render markdown from an existing JSON payload without image evaluation.",
    )
    args = parser.parse_args()

    if args.from_json:
        payload = refresh_payload(json.loads(Path(args.from_json).read_text(encoding="utf-8")))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    initialize_database()
    active_before = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, args.candidate_id)
    if artifact is None:
        raise SystemExit(f"candidate not found: {args.candidate_id}")

    conditions = parse_conditions(args.conditions)
    thresholds = parse_floats(args.thresholds)
    margins = parse_floats(args.margins)
    configs = [{"threshold": threshold, "real_protection_margin": margin} for threshold in thresholds for margin in margins]
    samples, labels, profile_report = load_profile_samples(args.profile, args.sample_limit)
    predictor = _generator_predictor_from_artifact(artifact)
    condition_features = build_condition_features(samples, conditions)
    config_results = [
        evaluate_config(
            predictor=predictor,
            samples=samples,
            labels=labels,
            condition_features=condition_features,
            conditions=conditions,
            threshold=float(config["threshold"]),
            margin=float(config["real_protection_margin"]),
        )
        for config in configs
    ]
    recommendation = choose_recommendation(config_results)
    active_after = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    payload = {
        "candidate_id": args.candidate_id,
        "profile": args.profile,
        "sample_count": len(samples),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "conditions": conditions,
        "profile_report": profile_report,
        "artifact_threshold": predictor["generated_gate_threshold"],
        "artifact_real_protection_margin": predictor["real_protection_margin"],
        "evaluated_configs": config_results,
        "recommended": recommendation,
        "active_model_id_before": active_before,
        "active_model_id_after": active_after,
        "active_unchanged": active_before == active_after,
        "does_not_train": True,
        "does_not_activate": True,
        "interpretation": interpret(recommendation),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def refresh_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload.get("evaluated_configs", []):
        if not isinstance(row, dict):
            continue
        summary = row.setdefault("summary", {})
        if not isinstance(summary, dict):
            continue
        clean = summary.get("clean", {})
        robust = summary.get("robust_average", {})
        if isinstance(clean, dict) and isinstance(robust, dict):
            summary["meets_low_fpr_screening_target"] = meets_low_fpr_screening_target(clean, robust)
    evaluated = [row for row in payload.get("evaluated_configs", []) if isinstance(row, dict)]
    recommendation = choose_recommendation(evaluated)
    payload["recommended"] = recommendation
    payload["interpretation"] = interpret(recommendation)
    return payload


def parse_conditions(raw: str) -> list[str]:
    supported = set(_supported_robustness_conditions())
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in conditions if item not in supported]
    if unknown:
        raise SystemExit(f"unsupported conditions: {unknown}; supported={sorted(supported)}")
    if "clean" not in conditions:
        conditions = ["clean", *conditions]
    return list(dict.fromkeys(conditions))


def parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("at least one float value is required")
    return values


def load_profile_samples(profile: str, limit: int) -> tuple[list[Any], list[str], dict[str, object]]:
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    task_samples = _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK)
    samples, labels, report = _select_profile_samples(task_samples, profile)
    if limit and len(samples) > limit:
        domain_by_id = {sample.id: _generator_sample_domain(sample) for sample in samples}
        samples, labels = _balanced_profile_samples(samples, labels, domain_by_id, limit)
    labels = [binary_label(label) for label in labels]
    return samples, labels, report


def build_condition_features(samples: list[Any], conditions: list[str]) -> dict[str, list[dict[str, float] | None]]:
    feature_rows: dict[str, list[dict[str, float] | None]] = {condition: [] for condition in conditions}
    with tempfile.TemporaryDirectory(prefix="smartpolice-gate-calibration-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, sample in enumerate(samples):
            source_path = Path(sample.image_path or "")
            text = f"{sample.title} {sample.content} {sample.scenario}"
            for condition in conditions:
                if not source_path.is_file():
                    feature_rows[condition].append(None)
                    continue
                if condition == "clean":
                    eval_path = source_path
                    eval_sha = sample.image_sha256
                else:
                    eval_path, eval_sha = _write_robustness_variant(source_path, condition, temp_root, index)
                feature_rows[condition].append(_generator_attribution_features(str(eval_path), eval_sha, text))
    return feature_rows


def evaluate_config(
    *,
    predictor: dict[str, object],
    samples: list[Any],
    labels: list[str],
    condition_features: dict[str, list[dict[str, float] | None]],
    conditions: list[str],
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    condition_results: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    clean_macro_f1: float | None = None
    for condition in conditions:
        predictions: list[str] = []
        binary_predictions: list[str] = []
        used_labels: list[str] = []
        payloads: list[dict[str, object]] = []
        groups: list[str] = []
        confidences: list[float] = []
        for sample, expected, features in zip(samples, labels, condition_features[condition], strict=True):
            if features is None:
                continue
            prediction = _predict_generator_label(
                features,
                list(predictor["feature_names"]),
                dict(predictor["means"]),
                dict(predictor["scales"]),
                list(predictor["prototypes"]),
                float(predictor["unknown_threshold"]),
                classifier_path=str(predictor["classifier_path"]),
                binary_gate_path=str(predictor["binary_gate_path"]),
                generated_gate_threshold=threshold,
                real_protection_margin=margin,
            )
            predicted_label = str(prediction.get("label", "unknown"))
            binary_predicted = binary_label(predicted_label)
            predictions.append(predicted_label)
            binary_predictions.append(binary_predicted)
            used_labels.append(expected)
            payloads.append(prediction)
            groups.append(_source_holdout_group_name(sample, "dataset_source"))
            confidences.append(float(prediction.get("confidence", 0.0) or 0.0))
        class_metrics = _classification_metrics(binary_predictions, used_labels)
        standard_binary_macro_f1 = two_class_macro_f1(binary_predictions, used_labels)
        binary_metrics = _binary_generation_metrics(binary_predictions, used_labels)
        auc = _roc_auc(
            [_score_for_label(prediction, "generated") for prediction in payloads],
            [label == "generated" for label in used_labels],
        )
        project_macro_f1 = float(class_metrics.get("macro_f1", 0.0))
        if condition == "clean":
            clean_macro_f1 = standard_binary_macro_f1
        condition_results[condition] = {
            "condition": condition,
            "perturbation": _robustness_condition_description(condition),
            "sample_count": len(used_labels),
            "label_distribution": dict(sorted(Counter(used_labels).items())),
            "prediction_distribution": dict(sorted(Counter(binary_predictions).items())),
            "binary_auc": auc,
            "binary_macro_f1": standard_binary_macro_f1,
            "project_macro_f1_including_unknown": project_macro_f1,
            "accuracy": float(class_metrics.get("accuracy", 0.0)),
            "generated_precision": binary_metrics.get("generated_precision"),
            "generated_recall": binary_metrics.get("generated_recall"),
            "generated_f1": binary_metrics.get("generated_f1"),
            "real_recall": binary_metrics.get("real_recall"),
            "real_false_positive_rate": binary_metrics.get("real_false_positive_rate"),
            "average_confidence": round(sum(confidences) / max(1, len(confidences)), 3),
            "macro_f1_delta_from_clean": (
                None if clean_macro_f1 is None else round(standard_binary_macro_f1 - clean_macro_f1, 3)
            ),
            "confusion_matrix": class_metrics.get("confusion_matrix", {}),
        }
        source_rows.extend(
            {
                "group": group,
                "condition": condition,
                "label": label,
                "prediction": prediction,
            }
            for group, label, prediction in zip(groups, used_labels, binary_predictions, strict=True)
        )
    robust_conditions = [condition for condition in conditions if condition != "clean"]
    robust_average = average_condition_metrics(condition_results, robust_conditions or conditions)
    return {
        "threshold": threshold,
        "real_protection_margin": margin,
        "conditions": condition_results,
        "summary": {
            "clean": short_metrics(condition_results.get("clean", {})),
            "robust_average": robust_average,
            "worst_macro_f1_drop": worst_macro_f1_drop(condition_results),
            "meets_low_fpr_screening_target": meets_low_fpr_screening_target(
                condition_results.get("clean", {}),
                robust_average,
            ),
        },
        "weak_source_groups": weak_source_groups(source_rows),
    }


def average_condition_metrics(condition_results: dict[str, Any], conditions: list[str]) -> dict[str, float | None]:
    keys = [
        "binary_auc",
        "binary_macro_f1",
        "project_macro_f1_including_unknown",
        "accuracy",
        "generated_precision",
        "generated_recall",
        "generated_f1",
        "real_recall",
        "real_false_positive_rate",
    ]
    rows = [condition_results[condition] for condition in conditions if condition in condition_results]
    averaged: dict[str, float | None] = {}
    for key in keys:
        values = [number(row.get(key)) for row in rows if number(row.get(key)) is not None]
        averaged[key] = round(sum(values) / len(values), 3) if values else None
    return averaged


def short_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "binary_auc": number(row.get("binary_auc")),
        "binary_macro_f1": number(row.get("binary_macro_f1")),
        "project_macro_f1_including_unknown": number(row.get("project_macro_f1_including_unknown")),
        "generated_recall": number(row.get("generated_recall")),
        "real_false_positive_rate": number(row.get("real_false_positive_rate")),
    }


def meets_low_fpr_screening_target(clean: dict[str, Any], robust_average: dict[str, Any]) -> bool:
    clean_fpr = number(clean.get("real_false_positive_rate"))
    clean_recall = number(clean.get("generated_recall"))
    robust_fpr = number(robust_average.get("real_false_positive_rate"))
    robust_recall = number(robust_average.get("generated_recall"))
    if None in {clean_fpr, clean_recall, robust_fpr, robust_recall}:
        return False
    return bool(clean_fpr <= 0.10 and clean_recall >= 0.75 and robust_fpr <= 0.10 and robust_recall >= 0.65)


def worst_macro_f1_drop(condition_results: dict[str, Any]) -> float | None:
    clean = number(condition_results.get("clean", {}).get("binary_macro_f1"))
    if clean is None:
        return None
    drops = [
        clean - value
        for condition, metrics in condition_results.items()
        if condition != "clean" and (value := number(metrics.get("binary_macro_f1"))) is not None
    ]
    return round(max(drops), 3) if drops else 0.0


def weak_source_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), str(row["condition"]))].append(row)
    issues: list[dict[str, Any]] = []
    for (group, condition), group_rows in grouped.items():
        labels = [str(row["label"]) for row in group_rows]
        predictions = [str(row["prediction"]) for row in group_rows]
        metrics = _binary_generation_metrics(predictions, labels)
        real_support = int(metrics.get("real_support", 0.0))
        real_fp = int(metrics.get("real_false_positive_count", 0.0))
        generated_support = int(metrics.get("generated_support", 0.0))
        generated_fn = int(metrics.get("generated_false_negative_count", 0.0))
        if real_support and real_fp:
            issues.append(
                {
                    "kind": "真实图误报",
                    "group": group,
                    "condition": condition,
                    "support": real_support,
                    "error_count": real_fp,
                    "rate": metrics.get("real_false_positive_rate"),
                    "_rank": (real_fp, number(metrics.get("real_false_positive_rate")) or 0.0),
                }
            )
        if generated_support and generated_fn:
            miss_rate = generated_fn / generated_support
            issues.append(
                {
                    "kind": "生成图漏报",
                    "group": group,
                    "condition": condition,
                    "support": generated_support,
                    "error_count": generated_fn,
                    "rate": round(miss_rate, 3),
                    "_rank": (generated_fn, miss_rate),
                }
            )
    ranked = sorted(issues, key=lambda item: item["_rank"], reverse=True)[:10]
    for item in ranked:
        item.pop("_rank", None)
    return ranked


def choose_recommendation(config_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not config_results:
        return None
    return max(
        config_results,
        key=lambda row: (
            bool(row.get("summary", {}).get("meets_low_fpr_screening_target")),
            -(number(row.get("summary", {}).get("clean", {}).get("real_false_positive_rate")) or 1.0),
            -(number(row.get("summary", {}).get("robust_average", {}).get("real_false_positive_rate")) or 1.0),
            number(row.get("summary", {}).get("clean", {}).get("generated_recall")) or 0.0,
            number(row.get("summary", {}).get("robust_average", {}).get("generated_recall")) or 0.0,
            number(row.get("summary", {}).get("robust_average", {}).get("binary_macro_f1")) or 0.0,
        ),
    )


def interpret(recommendation: dict[str, Any] | None) -> str:
    if recommendation is None:
        return "没有可用的阈值校准结果。"
    robust = recommendation.get("summary", {}).get("robust_average", {})
    clean = recommendation.get("summary", {}).get("clean", {})
    fpr = number(robust.get("real_false_positive_rate"))
    recall = number(robust.get("generated_recall"))
    clean_fpr = number(clean.get("real_false_positive_rate"))
    clean_recall = number(clean.get("generated_recall"))
    if (
        clean_fpr is not None
        and clean_recall is not None
        and fpr is not None
        and recall is not None
        and clean_fpr <= 0.10
        and fpr <= 0.10
        and clean_recall >= 0.75
        and recall >= 0.65
    ):
        return "阈值校准同时压住 clean 与扰动真实误报，但扰动生成召回仍是主要瓶颈，需进入更大样本独立验证。"
    if clean_fpr is not None and fpr is not None and clean_fpr <= 0.10 and fpr <= 0.10:
        return "阈值校准能压低真实误报，但生成召回不足，需要补强生成侧扰动样本。"
    return "当前候选仅靠阈值校准仍未稳定满足低误报目标，需要继续做 hard-negative 与特征/训练策略改进。"


def render_markdown(payload: dict[str, Any]) -> str:
    recommended = payload.get("recommended") if isinstance(payload.get("recommended"), dict) else {}
    lines = [
        "# 二分类阈值校准扰动验证",
        "",
        f"- Candidate: `{payload.get('candidate_id')}`",
        f"- Profile: `{payload.get('profile')}`",
        f"- 样本量: `{payload.get('sample_count')}`；标签分布: `{json.dumps(payload.get('label_distribution', {}), ensure_ascii=False)}`",
        f"- Active 保持不变: `{payload.get('active_unchanged')}` (`{payload.get('active_model_id_before')}` -> `{payload.get('active_model_id_after')}`)",
        f"- 解释: {payload.get('interpretation')}",
        "",
        "## 阈值对比",
        "",
        "| Threshold | Margin | Clean AUC | Clean 2-class F1 | Clean 旧口径F1 | Clean Recall | Clean Real FPR | Robust AUC | Robust 2-class F1 | Robust Recall | Robust Real FPR | 最差 F1 下降 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("evaluated_configs", []):
        summary = row.get("summary", {}) if isinstance(row, dict) else {}
        clean = summary.get("clean", {}) if isinstance(summary.get("clean"), dict) else {}
        robust = summary.get("robust_average", {}) if isinstance(summary.get("robust_average"), dict) else {}
        marker = " **推荐**" if row is recommended else ""
        lines.append(
            "| "
            f"{fmt(row.get('threshold'))}{marker} | "
            f"{fmt(row.get('real_protection_margin'))} | "
            f"{fmt(clean.get('binary_auc'))} | "
            f"{fmt(clean.get('binary_macro_f1'))} | "
            f"{fmt(clean.get('project_macro_f1_including_unknown'))} | "
            f"{fmt(clean.get('generated_recall'))} | "
            f"{fmt(clean.get('real_false_positive_rate'))} | "
            f"{fmt(robust.get('binary_auc'))} | "
            f"{fmt(robust.get('binary_macro_f1'))} | "
            f"{fmt(robust.get('generated_recall'))} | "
            f"{fmt(robust.get('real_false_positive_rate'))} | "
            f"{fmt(summary.get('worst_macro_f1_drop'))} |"
        )
    if recommended:
        lines.extend(
            [
                "",
                "## 推荐阈值的分条件表现",
                "",
                "| 条件 | AUC | 2-class F1 | 旧口径F1 | Generated Recall | Generated Precision | Real FPR | Macro-F1 Δ |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition, metrics in recommended.get("conditions", {}).items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| "
                f"{condition} | "
                f"{fmt(metrics.get('binary_auc'))} | "
                f"{fmt(metrics.get('binary_macro_f1'))} | "
                f"{fmt(metrics.get('project_macro_f1_including_unknown'))} | "
                f"{fmt(metrics.get('generated_recall'))} | "
                f"{fmt(metrics.get('generated_precision'))} | "
                f"{fmt(metrics.get('real_false_positive_rate'))} | "
                f"{fmt(metrics.get('macro_f1_delta_from_clean'))} |"
            )
        lines.extend(["", "## 推荐阈值的弱来源组", ""])
        weak_groups = recommended.get("weak_source_groups", [])
        if weak_groups:
            lines.extend(
                [
                    "| 问题 | 来源组 | 条件 | Support | 错误数 | Rate |",
                    "| --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for item in weak_groups:
                lines.append(
                    "| "
                    f"{item.get('kind')} | "
                    f"`{item.get('group')}` | "
                    f"{item.get('condition')} | "
                    f"{item.get('support')} | "
                    f"{item.get('error_count')} | "
                    f"{fmt(item.get('rate'))} |"
                )
        else:
            lines.append("- 未发现可排序的来源组错误。")
    lines.append("")
    return "\n".join(lines)


def binary_label(label: object) -> str:
    return "real" if str(label) == "real" else "generated"


def two_class_macro_f1(predictions: list[str], labels: list[str]) -> float:
    values: list[float] = []
    for class_name in ("generated", "real"):
        true_positive = sum(1 for prediction, label in zip(predictions, labels, strict=False) if prediction == class_name and label == class_name)
        false_positive = sum(1 for prediction, label in zip(predictions, labels, strict=False) if prediction == class_name and label != class_name)
        false_negative = sum(1 for prediction, label in zip(predictions, labels, strict=False) if prediction != class_name and label == class_name)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(values) / len(values), 3) if values else 0.0


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def fmt(value: Any) -> str:
    numeric = number(value)
    if numeric is None:
        return "-"
    return f"{numeric:.3f}"


if __name__ == "__main__":
    main()
