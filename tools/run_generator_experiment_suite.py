from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import (  # noqa: E402
    VisionCandidateEvaluationRequest,
    VisionSourceHoldoutRunRequest,
    VisionTrainingRunRequest,
)
from app.multimodal_training import (  # noqa: E402
    GENERATOR_ATTRIBUTION_TASK,
    _balanced_generator_samples,
    _binary_generation_metrics,
    _classification_metrics,
    _generator_experiment_view,
    _generator_profile_policy,
    _generator_predictor_from_artifact,
    _generator_sample_domain,
    _normalize_generator_label,
    _predict_generator_label,
    _source_counts_by_label,
    _task_relevant_samples,
    extract_sample_features,
    evaluate_vision_candidate,
    get_vision_competition_summary,
    run_vision_source_holdout_experiment,
    train_vision_evidence_head,
)
from app.storage import (  # noqa: E402
    get_vision_training_artifact_by_id,
    initialize_database,
    list_external_training_samples,
)


PROFILES = (
    "binary_generated_gate",
    "gpt_image2_ovr",
    "social_propagation_robustness",
)
EXTENDED_PROFILES = (
    *PROFILES,
    "mainstream_five_attribution",
    "clean_origin_attribution",
)
ROBUSTNESS_CONDITIONS = ("clean", "jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark")
TRAIN_AUGMENTATION_CONDITIONS = ("jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark")
DEFAULT_TMP_DIR = ROOT / "tmp" / "generator_experiment_suite"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generator attribution split-profile candidate experiments.")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "audits"))
    parser.add_argument("--docs-path", default=str(ROOT / "docs" / "generator_experiment_suite.md"))
    parser.add_argument("--tmp-dir", default=os.environ.get("SMARTPOLICE_TMP_DIR", str(DEFAULT_TMP_DIR)))
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument(
        "--extended-five-track",
        action="store_true",
        help="Run the extended diagnostic matrix including five-generator attribution and clean-origin upper bound.",
    )
    parser.add_argument("--candidate-min-samples", type=int, default=20)
    parser.add_argument("--training-sample-limit", type=int, default=0)
    parser.add_argument("--candidate-max-augmented-samples", type=int, default=1200)
    parser.add_argument(
        "--no-candidate-augmentation",
        action="store_true",
        help="Disable perturbation augmentation during candidate training for fast data-coverage diagnostics.",
    )
    parser.add_argument("--candidate-eval-limit", type=int, default=160)
    parser.add_argument("--source-sample-limit", type=int, default=1000)
    parser.add_argument("--max-holdout-groups", type=int, default=12)
    parser.add_argument("--enable-open-set-unknown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unknown-threshold-multiplier", type=float, default=1.35)
    parser.add_argument("--open-set-min-margin", type=float, default=0.08)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only render the experiment-suite scaffold and verify active remains unchanged.",
    )
    args = parser.parse_args()

    if args.quick:
        args.training_sample_limit = min(args.training_sample_limit or 240, 240)
        args.candidate_max_augmented_samples = min(args.candidate_max_augmented_samples, 80)
        args.candidate_eval_limit = min(args.candidate_eval_limit, 40)
        args.source_sample_limit = min(args.source_sample_limit, 160)
        args.max_holdout_groups = min(args.max_holdout_groups, 4)

    configure_temp_dir(Path(args.tmp_dir))
    initialize_database()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path = Path(args.docs_path)

    if args.extended_five_track and args.profiles == ",".join(PROFILES):
        profiles = EXTENDED_PROFILES
    else:
        profiles = tuple(profile.strip() for profile in args.profiles.split(",") if profile.strip())
    baseline = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK)
    active_before = baseline.active_model_id
    payload: dict[str, Any] = {
        "id": f"generator-experiment-suite-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "task_type": GENERATOR_ATTRIBUTION_TASK,
        "active_model_id_before": active_before,
        "active_model_id_after": None,
        "active_unchanged": None,
        "run_config": vars(args),
        "baseline_summary": model_dump(baseline),
        "experiments": [],
        "disk_policy": {
            "tmp_dir": str(Path(args.tmp_dir).resolve()),
            "downloads_external_benchmarks": False,
            "uses_e_drive": False,
        },
    }

    latest_path = output_dir / "generator_experiment_suite_latest.json"
    flush_outputs(payload, latest_path, docs_path)
    print(f"active_before={active_before}")

    if args.skip_training:
        payload["note"] = "Training skipped by request; this run verifies suite scaffolding and active-model immutability only."
        print("training skipped")
    else:
        for profile in profiles:
            print(f"training profile candidate: {profile}")
            experiment = run_profile(profile, args)
            payload["experiments"].append(experiment)
            flush_outputs(payload, latest_path, docs_path)

    final_summary = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK)
    if final_summary.active_model_id != active_before:
        raise RuntimeError(f"active model changed unexpectedly: {active_before} -> {final_summary.active_model_id}")
    payload["active_model_id_after"] = final_summary.active_model_id
    payload["active_unchanged"] = final_summary.active_model_id == active_before
    flush_outputs(payload, latest_path, docs_path)
    stamped_path = output_dir / f"{payload['id']}.json"
    write_json(stamped_path, payload)
    print(f"wrote {latest_path}")
    print(f"wrote {docs_path}")


def run_profile(profile: str, args: argparse.Namespace) -> dict[str, Any]:
    profile_policy = _generator_profile_policy(profile)
    candidate = train_vision_evidence_head(
        VisionTrainingRunRequest(
            task_type=GENERATOR_ATTRIBUTION_TASK,
            min_samples=args.candidate_min_samples,
            max_training_samples=args.training_sample_limit,
            activation_mode="candidate",
            experiment_profile=profile,
            validation_strategy="source_holdout",
            source_holdout_fraction=0.2,
            min_source_holdout_samples=20,
            enable_perturbation_augmentation=(
                not args.no_candidate_augmentation
                and profile
                in {
                    "binary_generated_gate",
                    "gpt_image2_ovr",
                    "mainstream_five_attribution",
                    "social_propagation_robustness",
                }
            ),
            augmentation_conditions=list(TRAIN_AUGMENTATION_CONDITIONS),
            max_augmented_samples=args.candidate_max_augmented_samples,
            enable_open_set_unknown=bool(args.enable_open_set_unknown and profile != "binary_generated_gate"),
            unknown_threshold_multiplier=args.unknown_threshold_multiplier,
            open_set_min_margin=(0.0 if profile == "binary_generated_gate" else args.open_set_min_margin),
        )
    )
    evaluation = evaluate_vision_candidate(
        VisionCandidateEvaluationRequest(
            task_type=GENERATOR_ATTRIBUTION_TASK,
            candidate_model_id=candidate.id,
            limit=args.candidate_eval_limit,
            conditions=list(ROBUSTNESS_CONDITIONS),
            include_source_holdout=False,
            include_feature_ablation=False,
            activate_if_passes_gate=False,
        )
    )
    clean_diagnostics = evaluate_profile_clean_diagnostics(candidate.id, profile, args.candidate_eval_limit)
    source_holdout = run_vision_source_holdout_experiment(
        VisionSourceHoldoutRunRequest(
            task_type=GENERATOR_ATTRIBUTION_TASK,
            experiment_profile=profile,
            holdout_key="dataset_source",
            sample_limit=args.source_sample_limit,
            max_holdout_groups=args.max_holdout_groups,
            min_train_samples=4,
            min_holdout_samples=1,
            enable_perturbation_augmentation=False,
            enable_open_set_unknown=bool(args.enable_open_set_unknown and profile != "binary_generated_gate"),
            unknown_threshold_multiplier=args.unknown_threshold_multiplier,
            open_set_min_margin=(0.0 if profile == "binary_generated_gate" else args.open_set_min_margin),
        )
    )
    acceptance = evaluate_profile_acceptance(
        profile_policy,
        model_dump(evaluation),
        model_dump(source_holdout),
        clean_diagnostics,
    )
    return {
        "profile": profile,
        "profile_policy": profile_policy,
        "candidate": model_dump(candidate),
        "evaluation": model_dump(evaluation),
        "clean_diagnostics": clean_diagnostics,
        "source_holdout": model_dump(source_holdout),
        "acceptance": acceptance,
        "recommendation": profile_recommendation(
            profile,
            model_dump(evaluation),
            model_dump(source_holdout),
            clean_diagnostics,
            acceptance,
        ),
    }


def evaluate_profile_clean_diagnostics(candidate_id: str, profile: str, limit: int) -> dict[str, Any]:
    artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, candidate_id)
    if artifact is None:
        return {"available": False, "reason": "candidate artifact not found"}
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    task_samples = _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK)
    profile_request = VisionTrainingRunRequest(
        task_type=GENERATOR_ATTRIBUTION_TASK,
        min_samples=2,
        experiment_profile=profile,
    )
    samples, labels, profile_report = _select_profile_samples(task_samples, profile)
    if limit and len(samples) > limit:
        domain_by_id = {sample.id: _generator_sample_domain(sample) for sample in samples}
        samples, labels = _balanced_profile_samples(samples, labels, domain_by_id, limit)
    rows = [extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK) for sample in samples]
    predictor = _generator_predictor_from_artifact(artifact)
    predictions: list[str] = []
    prediction_payloads: list[dict[str, object]] = []
    confidences: list[float] = []
    for row in rows:
        prediction = _predict_generator_label(
            row,
            list(predictor["feature_names"]),
            dict(predictor["means"]),
            dict(predictor["scales"]),
            list(predictor["prototypes"]),
            float(predictor["unknown_threshold"]),
            classifier_path=str(predictor["classifier_path"]),
            gpt_detector_path=str(predictor.get("gpt_detector_path", "")),
            binary_gate_path=str(predictor["binary_gate_path"]),
            generated_gate_threshold=float(predictor["generated_gate_threshold"]),
            gpt_detector_threshold=float(predictor.get("gpt_detector_threshold", 0.18)),
            real_protection_margin=float(predictor["real_protection_margin"]),
        )
        predictions.append(str(prediction.get("label", "unknown")))
        prediction_payloads.append(prediction)
        confidences.append(float(prediction.get("confidence", 0.0) or 0.0))
    metrics = _classification_metrics(predictions, labels)
    binary_metrics = _binary_generation_metrics(predictions, labels)
    positive_label = _profile_auc_positive_label(profile)
    positive_auc = (
        _roc_auc(
            [_score_for_label(prediction, positive_label) for prediction in prediction_payloads],
            [label == positive_label for label in labels],
        )
        if positive_label
        else None
    )
    macro_auc = _macro_ovr_auc(labels, prediction_payloads)
    binary_auc = _roc_auc(
        [_score_for_label(prediction, "generated") for prediction in prediction_payloads],
        [label != "real" for label in labels],
    )
    per_class = metrics.get("per_class") if isinstance(metrics, dict) else {}
    per_class_metrics = per_class if isinstance(per_class, dict) else {}
    return {
        "available": True,
        "sample_count": len(labels),
        "profile_report": profile_report,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "macro_ovr_auc": macro_auc,
        "binary_macro_f1": float(binary_metrics.get("macro_f1", 0.0)),
        "binary_auc": binary_auc,
        "positive_label": positive_label,
        "positive_auc": positive_auc,
        "generated_recall": float(binary_metrics.get("generated_recall", 0.0)),
        "generated_precision": float(binary_metrics.get("generated_precision", 0.0)),
        "generated_f1": float(binary_metrics.get("generated_f1", 0.0)),
        "real_precision": float(binary_metrics.get("real_precision", 0.0)),
        "real_recall": float(binary_metrics.get("real_recall", 0.0)),
        "real_f1": float(binary_metrics.get("real_f1", 0.0)),
        "real_false_positive_rate": float(binary_metrics.get("real_false_positive_rate", 0.0)),
        "unknown_rate": round(Counter(predictions).get("unknown", 0) / max(1, len(predictions)), 3),
        "average_confidence": round(sum(confidences) / max(1, len(confidences)), 3),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "prediction_distribution": dict(sorted(Counter(predictions).items())),
        "per_class": per_class_metrics,
    }


def profile_recommendation(
    profile: str,
    evaluation: dict[str, Any],
    source_holdout: dict[str, Any],
    clean_diagnostics: dict[str, Any],
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = source_holdout.get("aggregate", {}) if isinstance(source_holdout, dict) else {}
    candidate_summary = evaluation.get("candidate_summary", {}) if isinstance(evaluation, dict) else {}
    real_fpr = number(aggregate.get("overall_real_false_positive_rate"))
    generated_recall = number(aggregate.get("mean_generated_recall"))
    source_macro = number(aggregate.get("mean_macro_f1"))
    label_covered_macro = number(aggregate.get("label_covered_macro_f1"))
    gpt_recall = number(candidate_summary.get("clean_gpt_image2_recall"))
    positive_auc = number(clean_diagnostics.get("positive_auc"))
    acceptance_status = str((acceptance or {}).get("status") or "")
    if acceptance_status == "passed":
        decision = "meets_gate_candidate_only"
    elif profile == "binary_generated_gate":
        decision = "needs_lower_real_fpr_or_higher_generated_recall"
    elif profile == "gpt_image2_ovr":
        decision = "needs_more_cross_source_gpt_image2"
    elif profile == "mainstream_five_attribution":
        decision = "needs_stronger_mainstream_five_sources"
    elif profile == "multi_generator_label_covered":
        decision = "needs_label_covered_sources"
    elif profile == "clean_origin_attribution":
        decision = "upper_bound_only"
    elif profile == "social_propagation_robustness":
        decision = "benchmark_or_hard_negative_only"
    else:
        decision = "needs_more_data"
    return {
        "decision": decision,
        "profile": profile,
        "source_macro_f1": source_macro,
        "label_covered_macro_f1": label_covered_macro,
        "overall_real_false_positive_rate": real_fpr,
        "generated_recall": generated_recall,
        "clean_gpt_image2_recall": gpt_recall,
        "positive_auc": positive_auc,
        "acceptance_status": acceptance_status or None,
        "main_issue": (acceptance or {}).get("main_issue") if isinstance(acceptance, dict) else None,
        "does_not_activate": True,
    }


def evaluate_profile_acceptance(
    profile_policy: dict[str, Any],
    evaluation: dict[str, Any],
    source_holdout: dict[str, Any],
    clean_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    context = {
        "evaluation": evaluation,
        "candidate_summary": evaluation.get("candidate_summary", {}) if isinstance(evaluation, dict) else {},
        "source_holdout": source_holdout,
        "clean_diagnostics": clean_diagnostics,
    }
    checks: list[dict[str, Any]] = []
    failing_checks: list[dict[str, Any]] = []
    for gate in profile_policy.get("acceptance_gates", []):
        if not isinstance(gate, dict):
            continue
        source = str(gate.get("source") or "")
        operator = str(gate.get("operator") or "report")
        threshold = gate.get("threshold")
        value = nested_get(context, source)
        check = {
            "metric": gate.get("metric"),
            "name": gate.get("name") or gate.get("metric"),
            "operator": operator,
            "threshold": threshold,
            "source": source,
            "value": value,
            "passed": None,
            "status": "reported",
        }
        if operator != "report":
            passed = compare_gate(value, operator, threshold)
            check["passed"] = passed
            check["status"] = "passed" if passed else "failed"
            if not passed:
                failing_checks.append(check)
        checks.append(check)
    if failing_checks:
        main_issue = gate_issue_text(failing_checks[0])
        status = "needs_improvement"
    else:
        main_issue = "已达到当前轨道验收门槛；仍保持 candidate-only，需人工复核后组合使用。"
        status = "passed"
    if profile_policy.get("activation_eligibility") == "benchmark_only" and status == "passed":
        main_issue = "达到 benchmark 口径门槛，但该轨道只作为评测/上限/鲁棒性证据，不直接激活。"
    return {
        "profile": profile_policy.get("profile"),
        "chinese_name": profile_policy.get("chinese_name"),
        "status": status,
        "passed": not failing_checks,
        "candidate_only": bool(profile_policy.get("candidate_only", True)),
        "activation_eligibility": profile_policy.get("activation_eligibility"),
        "main_issue": main_issue,
        "checks": checks,
    }


def nested_get(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def compare_gate(value: Any, operator: str, threshold: Any) -> bool:
    if not isinstance(value, int | float) or not isinstance(threshold, int | float):
        return False
    if operator == ">=":
        return float(value) >= float(threshold)
    if operator == ">":
        return float(value) > float(threshold)
    if operator == "<=":
        return float(value) <= float(threshold)
    if operator == "<":
        return float(value) < float(threshold)
    if operator == "==":
        return float(value) == float(threshold)
    return False


def gate_issue_text(check: dict[str, Any]) -> str:
    value = check.get("value")
    threshold = check.get("threshold")
    name = check.get("name") or check.get("metric")
    operator = check.get("operator")
    if value is None:
        return f"{name} 缺少可用数值，需要补评测或样本覆盖。"
    return f"{name}={fmt(value)} 未达到 {operator} {fmt(threshold)} 的验收门槛。"


def _profile_auc_positive_label(profile: str) -> str | None:
    if profile in {"binary_generated_gate", "social_propagation_robustness"}:
        return "generated"
    if profile == "gpt_image2_ovr":
        return "gpt-image2"
    return None


def _select_profile_samples(samples: list[Any], profile: str) -> tuple[list[Any], list[str], dict[str, object]]:
    domains = [_generator_sample_domain(sample) for sample in samples]
    base_labels = [_normalize_generator_label(sample.label) for sample in samples]
    source_counts_by_label = _source_counts_by_label(samples, base_labels)
    selected_samples: list[Any] = []
    labels: list[str] = []
    dropped_reasons: Counter[str] = Counter()
    for sample, label, domain in zip(samples, base_labels, domains, strict=True):
        mapped = _profile_label_from_metadata(
            label=label,
            domain=domain,
            profile=profile,
            source_counts_by_label=source_counts_by_label,
        )
        if mapped is None:
            dropped_reasons[f"excluded:{domain}"] += 1
            continue
        selected_samples.append(sample)
        labels.append(mapped)
    report = {
        "profile": profile,
        "input_count": len(samples),
        "selected_count": len(selected_samples),
        "excluded_count": len(samples) - len(selected_samples),
        "domain_distribution": dict(sorted(Counter(domains).items())),
        "selected_domain_distribution": dict(
            sorted(Counter(_generator_sample_domain(sample) for sample in selected_samples).items())
        ),
        "base_label_distribution": dict(sorted(Counter(base_labels).items())),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "dropped_reasons": dict(sorted(dropped_reasons.items())),
        "source_coverage": {label: len(sources) for label, sources in sorted(source_counts_by_label.items())},
    }
    return selected_samples, labels, report


def _profile_label_from_metadata(
    *,
    label: str,
    domain: str,
    profile: str,
    source_counts_by_label: dict[str, set[str]],
) -> str | None:
    if profile == "standard_attribution":
        return label
    if profile == "binary_generated_gate":
        return "real" if label == "real" else "generated"
    if profile == "gpt_image2_ovr":
        if label == "gpt-image2":
            return "gpt-image2"
        if label == "real":
            return "real"
        return "other-generated"
    if profile == "mainstream_five_attribution":
        return _mainstream_five_label_from_metadata(label)
    if profile == "multi_generator_label_covered":
        if label == "real":
            return "real"
        if len(source_counts_by_label.get(label, set())) >= 2:
            return label
        return "unknown"
    if profile == "clean_origin_attribution":
        return label if domain in {"clean_origin", "multi_generator_benchmark", "gpt_image2_focus"} else None
    if profile == "social_propagation_robustness":
        if domain in {"social_propagation", "real_negative_pool"} or label != "real":
            return "real" if label == "real" else "generated"
        return None
    return label


def _mainstream_five_label_from_metadata(label: str) -> str:
    if label == "real":
        return "real"
    if label == "gpt-image2":
        return "gpt-image2"
    if label == "nano-banana":
        return "nano-banana"
    if label == "seedream-4":
        return "seedream-4"
    if label == "midjourney":
        return "midjourney"
    if label in {"stable-diffusion", "sd21", "sd3", "sdxl"}:
        return "stable-diffusion"
    return "unknown"


def _balanced_profile_samples(
    samples: list[Any],
    labels: list[str],
    domain_by_id: dict[str, str],
    limit: int,
) -> tuple[list[Any], list[str]]:
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, (sample, label) in enumerate(zip(samples, labels, strict=True)):
        key = (label, domain_by_id.get(sample.id, "unknown"))
        buckets.setdefault(key, []).append(index)
    selected_indices: list[int] = []
    ordered_keys = sorted(buckets, key=lambda item: (item[0] != "real", item[0], item[1]))
    cursor = 0
    while len(selected_indices) < limit and ordered_keys:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if cursor < len(bucket):
                selected_indices.append(bucket[cursor])
                progressed = True
                if len(selected_indices) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    selected_indices = sorted(selected_indices)
    return (
        [samples[index] for index in selected_indices],
        [labels[index] for index in selected_indices],
    )


def _score_for_label(prediction: dict[str, object], label: str) -> float:
    gate = prediction.get("binary_gate")
    if label == "generated" and isinstance(gate, dict):
        value = gate.get("generated_probability")
        if isinstance(value, int | float):
            return float(value)
    if label == "real" and isinstance(gate, dict):
        value = gate.get("real_probability")
        if isinstance(value, int | float):
            return float(value)
    candidates = prediction.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("label") == label and isinstance(candidate.get("confidence"), int | float):
                return float(candidate["confidence"])
    raw_label = str(prediction.get("raw_label") or prediction.get("label") or "unknown")
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    return confidence if raw_label == label else 0.0


def _macro_ovr_auc(labels: list[str], predictions: list[dict[str, object]]) -> float | None:
    aucs = []
    for label in sorted(set(labels)):
        auc = _roc_auc(
            [_score_for_label(prediction, label) for prediction in predictions],
            [actual == label for actual in labels],
        )
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None
    return round(sum(aucs) / len(aucs), 3)


def _roc_auc(scores: list[float], positives: list[bool]) -> float | None:
    positive_scores = [score for score, is_positive in zip(scores, positives, strict=False) if is_positive]
    negative_scores = [score for score, is_positive in zip(scores, positives, strict=False) if not is_positive]
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for pos_score in positive_scores:
        for neg_score in negative_scores:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return round(wins / (len(positive_scores) * len(negative_scores)), 3)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 分轨生成图检测与归因实验矩阵",
        "",
        f"- 生成时间: `{payload.get('created_at')}`",
        f"- 任务: `{payload.get('task_type')}`",
        f"- Active 是否保持不变: `{payload.get('active_unchanged')}`",
        f"- Active 前后: `{payload.get('active_model_id_before')}` -> `{payload.get('active_model_id_after')}`",
        f"- 是否下载外部 benchmark: `{payload.get('disk_policy', {}).get('downloads_external_benchmarks')}`",
        "",
        "## 汇报主表",
        "",
        "| 轨道 | 定位 | Candidate | Source Macro-F1 | Label-covered Macro-F1 | Source Generated Recall | Source Real FPR | Clean sanity Macro-F1 | Unknown rate | 验收状态 | 主要问题 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for experiment in payload.get("experiments", []):
        candidate = experiment.get("candidate", {})
        evaluation = experiment.get("evaluation", {})
        source_holdout = experiment.get("source_holdout", {})
        aggregate = source_holdout.get("aggregate", {})
        summary = evaluation.get("candidate_summary", {})
        diagnostics = experiment.get("clean_diagnostics", {})
        policy = experiment.get("profile_policy", {})
        acceptance = experiment.get("acceptance", {})
        lines.append(
            "| "
            f"{policy.get('chinese_name') or experiment.get('profile')} | "
            f"{policy.get('system_role') or '-'} | "
            f"`{candidate.get('id')}` | "
            f"{fmt(aggregate.get('mean_macro_f1'))} | "
            f"{fmt(aggregate.get('label_covered_macro_f1'))} | "
            f"{fmt(aggregate.get('mean_generated_recall'))} | "
            f"{fmt(aggregate.get('overall_real_false_positive_rate'))} | "
            f"{fmt(diagnostics.get('macro_f1', summary.get('clean_macro_f1')))} | "
            f"{fmt(diagnostics.get('unknown_rate'))} | "
            f"{acceptance_label(acceptance)} | "
            f"{acceptance.get('main_issue') or '-'} |"
        )
    lines.extend(["", "## 逐轨细节", ""])
    for experiment in payload.get("experiments", []):
        profile = experiment.get("profile")
        diagnostics = experiment.get("clean_diagnostics", {})
        source_holdout = experiment.get("source_holdout", {})
        aggregate = source_holdout.get("aggregate", {})
        policy = experiment.get("profile_policy", {})
        acceptance = experiment.get("acceptance", {})
        recommendation = experiment.get("recommendation", {})
        lines.extend(
            [
                f"### {policy.get('chinese_name') or profile}",
                "",
                f"- Profile: `{profile}`",
                f"- Candidate: `{experiment.get('candidate', {}).get('id')}`",
                f"- 目标: {policy.get('objective') or '-'}",
                f"- 模型做法: {policy.get('model_strategy') or '-'}",
                f"- 标签策略: {policy.get('label_strategy') or '-'}",
                f"- 激活政策: {policy.get('activation_policy') or '-'}",
                f"- 验收状态: `{acceptance.get('status') or '-'}`；建议: `{recommendation.get('decision')}`；问题: {acceptance.get('main_issue') or '-'}",
                f"- Clean accuracy / Macro-F1 / Macro OvR AUC: `{fmt(diagnostics.get('accuracy'))}` / `{fmt(diagnostics.get('macro_f1'))}` / `{fmt(diagnostics.get('macro_ovr_auc'))}`",
                f"- Positive label / positive AUC: `{diagnostics.get('positive_label') or 'macro-only'}` / `{fmt(diagnostics.get('positive_auc'))}`",
                f"- Binary Macro-F1 / binary AUC / generated recall / real FPR: `{fmt(diagnostics.get('binary_macro_f1'))}` / `{fmt(diagnostics.get('binary_auc'))}` / `{fmt(diagnostics.get('generated_recall'))}` / `{fmt(diagnostics.get('real_false_positive_rate'))}`",
                f"- Source-holdout Macro-F1 / binary Macro-F1 / generated recall / real FPR: `{fmt(aggregate.get('mean_macro_f1'))}` / `{fmt(aggregate.get('mean_binary_macro_f1'))}` / `{fmt(aggregate.get('mean_generated_recall'))}` / `{fmt(aggregate.get('overall_real_false_positive_rate'))}`",
                f"- Label-covered Macro-F1 / binary Macro-F1 / real FPR: `{fmt(aggregate.get('label_covered_macro_f1'))}` / `{fmt(aggregate.get('label_covered_binary_macro_f1'))}` / `{fmt(aggregate.get('label_covered_real_false_positive_rate'))}`",
                f"- Label distribution: `{json.dumps(diagnostics.get('label_distribution', {}), ensure_ascii=False)}`",
                f"- Prediction distribution: `{json.dumps(diagnostics.get('prediction_distribution', {}), ensure_ascii=False)}`",
                "",
            ]
        )
        checks = acceptance.get("checks", [])
        if isinstance(checks, list) and checks:
            lines.extend(
                [
                    "| 验收项 | 当前值 | 门槛 | 状态 |",
                    "| --- | ---: | --- | --- |",
                ]
            )
            for check in checks:
                if not isinstance(check, dict):
                    continue
                lines.append(
                    "| "
                    f"{check.get('name') or check.get('metric')} | "
                    f"{fmt(check.get('value'))} | "
                    f"{gate_threshold_text(check)} | "
                    f"{check_status_label(check)} |"
                )
            lines.append("")
        weak_groups = source_holdout_weak_groups(source_holdout)
        if weak_groups:
            lines.extend(
                [
                    "| 跨来源薄弱点 | 来源组 | Support | 错分数 | 指标 |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for item in weak_groups:
                lines.append(
                    "| "
                    f"{item['kind']} | "
                    f"{item['group']} | "
                    f"{fmt_count(item['support'])} | "
                    f"{fmt_count(item['error_count'])} | "
                    f"{fmt(item['metric'])} |"
                )
            lines.append("")
        per_class = diagnostics.get("per_class", {})
        if isinstance(per_class, dict) and per_class:
            lines.extend(
                [
                    "| Clean class | Precision | Recall | F1 | Support |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for class_name, metrics in sorted(per_class.items()):
                if not isinstance(metrics, dict):
                    continue
                lines.append(
                    "| "
                    f"{class_name} | "
                    f"{fmt(metrics.get('precision'))} | "
                    f"{fmt(metrics.get('recall'))} | "
                    f"{fmt(metrics.get('f1'))} | "
                    f"{fmt_count(metrics.get('support'))} |"
                )
            lines.append("")
    lines.extend(
        [
            "",
            "## 解释口径",
            "",
        "- 主表优先看 source-holdout 和 label-covered 指标；Clean sanity 只说明训练视图是否自洽，不代表跨来源泛化满分。",
        "- `五类主流生成器归因` 只强归因 GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney；DALL-E、Flux、Imagen、Firefly 等先退到 unknown/other。",
        "- 多生成器归因不把单来源类别高分写成泛化能力，单来源/小样本类别退到 unknown。",
        "- 真实/生成初筛、GPT-image2 专项和五类主流归因仍是两层可信输出：先低误报筛生成，再给来源线索。",
        "- 所有分轨实验只产生 candidate/component candidate；active 替换必须另行显式决策。",
    ]
    )
    return "\n".join(lines) + "\n"


def configure_temp_dir(tmp_dir: Path) -> None:
    resolved = tmp_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(resolved)
    os.environ["TMP"] = str(resolved)
    tempfile.tempdir = str(resolved)


def model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flush_outputs(payload: dict[str, Any], latest_path: Path, docs_path: Path) -> None:
    write_json(latest_path, payload)
    docs_path.write_text(render_markdown(payload), encoding="utf-8")


def number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def fmt(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, int | float) else "-"


def fmt_count(value: object) -> str:
    return str(int(value)) if isinstance(value, int | float) else "-"


def fmt_triplet(metrics: dict[str, Any], prefix: str) -> str:
    return f"{fmt(metrics.get(f'{prefix}_precision'))}/{fmt(metrics.get(f'{prefix}_recall'))}/{fmt(metrics.get(f'{prefix}_f1'))}"


def source_holdout_weak_groups(source_holdout: dict[str, Any]) -> list[dict[str, Any]]:
    groups = source_holdout.get("groups", []) if isinstance(source_holdout, dict) else []
    if not isinstance(groups, list):
        return []
    real_failures: list[dict[str, Any]] = []
    generated_failures: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict) or group.get("skipped"):
            continue
        real_support = group.get("real_support", 0)
        real_fp_count = group.get("real_false_positive_count", 0)
        if isinstance(real_support, int | float) and isinstance(real_fp_count, int | float) and real_support > 0 and real_fp_count > 0:
            real_failures.append(
                {
                    "kind": "真实图误报",
                    "group": group.get("holdout_group"),
                    "support": real_support,
                    "error_count": real_fp_count,
                    "metric": group.get("real_false_positive_rate"),
                    "rank": (float(real_fp_count), float(group.get("real_false_positive_rate", 0.0) or 0.0)),
                }
            )
        generated_support = group.get("generated_support", 0)
        generated_fn_count = group.get("generated_false_negative_count", 0)
        if (
            isinstance(generated_support, int | float)
            and isinstance(generated_fn_count, int | float)
            and generated_support > 0
            and generated_fn_count > 0
        ):
            generated_failures.append(
                {
                    "kind": "生成图漏报",
                    "group": group.get("holdout_group"),
                    "support": generated_support,
                    "error_count": generated_fn_count,
                    "metric": 1.0 - float(group.get("generated_recall", 0.0) or 0.0),
                    "rank": (float(generated_fn_count), 1.0 - float(group.get("generated_recall", 0.0) or 0.0)),
                }
            )
    ranked = sorted(real_failures, key=lambda item: item["rank"], reverse=True)[:3]
    ranked.extend(sorted(generated_failures, key=lambda item: item["rank"], reverse=True)[:3])
    for item in ranked:
        item.pop("rank", None)
    return ranked


def acceptance_label(acceptance: dict[str, Any]) -> str:
    status = acceptance.get("status")
    if status == "passed":
        return "达标(candidate)"
    if status == "needs_improvement":
        return "未达标"
    return str(status or "-")


def check_status_label(check: dict[str, Any]) -> str:
    status = check.get("status")
    if status == "passed":
        return "达标"
    if status == "failed":
        return "未达标"
    return "仅报告"


def gate_threshold_text(check: dict[str, Any]) -> str:
    if check.get("operator") == "report":
        return "仅报告"
    return f"{check.get('operator')} {fmt(check.get('threshold'))}"


if __name__ == "__main__":
    main()
