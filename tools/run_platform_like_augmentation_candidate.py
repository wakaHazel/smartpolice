from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import VisionTrainingRunRequest  # noqa: E402
from app.multimodal_training import (  # noqa: E402
    GENERATOR_ATTRIBUTION_TASK,
    get_vision_competition_summary,
    train_vision_evidence_head,
)
from app.storage import get_vision_training_artifact_by_id, initialize_database  # noqa: E402
from tools.run_platform_candidate_experiment import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_RETURNED_ROOT,
    evaluate_acceptance,
    evaluate_model,
    strip_predictions,
)
from tools.run_platform_transcode_eval import (  # noqa: E402
    clean_eval_items,
    collect_returned_items,
    load_manifest,
    parse_csv_arg,
    summarize_conditions,
)


DEFAULT_OUTPUT_JSON = ROOT / "output" / "audits" / "platform_like_augmentation_candidate_latest.json"
DEFAULT_OUTPUT_MD = ROOT / "output" / "audits" / "platform_like_augmentation_candidate_latest.md"
DEFAULT_OUTPUT_CSV = ROOT / "output" / "audits" / "platform_like_augmentation_candidate_latest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate-only model on the larger external pool using synthetic platform-like "
            "augmentation inferred from the Weibo/XHS black-box returned set, then evaluate on the real platform set."
        )
    )
    parser.add_argument("--upload-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--returned-root", default=str(DEFAULT_RETURNED_ROOT))
    parser.add_argument("--platforms", default="weibo,xhs")
    parser.add_argument("--variants", default="download")
    parser.add_argument("--profile", default="gpt_image2_ovr", choices=("gpt_image2_ovr", "binary_generated_gate", "social_propagation_robustness"))
    parser.add_argument("--training-sample-limit", type=int, default=3600)
    parser.add_argument("--max-augmented-samples", type=int, default=3600)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--augmentation-conditions",
        default="weibo_download_like,jpeg_q60,watermark",
        help="Synthetic perturbations applied to the larger training pool.",
    )
    parser.add_argument("--fpr-threshold", type=float, default=0.15)
    parser.add_argument("--target-gpt-recall", type=float, default=0.95)
    parser.add_argument("--threshold-calibration-split", choices=("odd", "even"), default="odd")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--candidate-id",
        default="",
        help="Evaluate an existing candidate instead of retraining.",
    )
    args = parser.parse_args()

    initialize_database()
    active_before = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    augmentation_conditions = parse_csv_arg(args.augmentation_conditions)
    if args.candidate_id:
        artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, args.candidate_id)
        if artifact is None:
            raise SystemExit(f"candidate not found: {args.candidate_id}")
        candidate_id = args.candidate_id
        candidate_status = "existing_candidate_evaluated"
        candidate_training_result: dict[str, Any] = {
            "id": candidate_id,
            "status": candidate_status,
            "model_card": artifact.get("model_card", {}),
        }
    else:
        candidate = train_vision_evidence_head(
            VisionTrainingRunRequest(
                task_type=GENERATOR_ATTRIBUTION_TASK,
                activation_mode="candidate",
                experiment_profile=args.profile,
                validation_strategy="source_holdout",
                min_samples=args.min_samples,
                max_training_samples=args.training_sample_limit,
                enable_perturbation_augmentation=True,
                augmentation_conditions=augmentation_conditions,
                max_augmented_samples=args.max_augmented_samples,
                enable_open_set_unknown=False,
                open_set_min_margin=0.0,
            )
        )
        candidate_id = candidate.id
        candidate_status = candidate.status
        candidate_training_result = model_dump(candidate)
    active_after_train = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    if active_after_train != active_before:
        raise RuntimeError(f"active changed during candidate training: {active_before} -> {active_after_train}")

    manifest_items = load_manifest(Path(args.upload_manifest))
    platforms = parse_csv_arg(args.platforms)
    variants = parse_csv_arg(args.variants)
    collection_rows, returned_items = collect_returned_items(
        manifest_items,
        Path(args.returned_root),
        platforms,
        variants,
    )
    eval_items = [*clean_eval_items(manifest_items), *returned_items]
    active_eval = evaluate_model(
        model_id="",
        manifest_items=manifest_items,
        collection_rows=collection_rows,
        eval_items=eval_items,
        returned_root=Path(args.returned_root),
        platforms=platforms,
        variants=variants,
    )
    candidate_eval = evaluate_model(
        model_id=candidate_id,
        manifest_items=manifest_items,
        collection_rows=collection_rows,
        eval_items=eval_items,
        returned_root=Path(args.returned_root),
        platforms=platforms,
        variants=variants,
    )
    calibration_pair_ids, holdout_pair_ids = split_pair_ids(
        [item.pair_id for item in manifest_items],
        calibration_split=args.threshold_calibration_split,
    )
    calibration_predictions = filter_predictions(candidate_eval["sample_predictions"], calibration_pair_ids)
    candidate_holdout_predictions = filter_predictions(candidate_eval["sample_predictions"], holdout_pair_ids)
    active_holdout_predictions = filter_predictions(active_eval["sample_predictions"], holdout_pair_ids)
    operating_point = optimize_gpt_thresholds(
        calibration_predictions,
        fpr_ceiling=args.fpr_threshold,
        target_gpt_recall=args.target_gpt_recall,
    )
    candidate_official_predictions = apply_operating_thresholds(
        candidate_holdout_predictions,
        operating_point["thresholds_by_condition"],
    )

    condition_summaries = {
        "active": summarize_conditions(active_holdout_predictions),
        "candidate": summarize_conditions(candidate_official_predictions),
    }
    acceptance = evaluate_download_acceptance(
        condition_summaries["active"],
        condition_summaries["candidate"],
        fpr_threshold=args.fpr_threshold,
    )
    active_after_eval = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    if active_after_eval != active_before:
        raise RuntimeError(f"active changed during evaluation: {active_before} -> {active_after_eval}")

    payload: dict[str, Any] = {
        "id": f"platform-like-augmentation-candidate-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "task_type": GENERATOR_ATTRIBUTION_TASK,
        "active_model_id_before": active_before,
        "active_model_id_after": active_after_eval,
        "active_unchanged": active_after_eval == active_before,
        "candidate_id": candidate_id,
        "candidate_status": candidate_status,
        "experiment_profile": args.profile,
        "training_policy": {
            "uses_real_platform_returned_files_for_training": False,
            "uses_larger_external_pool": True,
            "training_sample_limit": args.training_sample_limit,
            "enable_perturbation_augmentation": True,
            "augmentation_conditions": augmentation_conditions,
            "max_augmented_samples": args.max_augmented_samples,
            "validation_strategy": "source_holdout",
            "policy": "Infer download/transcode-like perturbation parameters from the small black-box set, synthesize them on the larger external pool, calibrate the GPT-image2 operating threshold on one half of platform pairs, and report the other half as holdout. Screenshot chains are parked as a failure boundary.",
        },
        "threshold_calibration_protocol": {
            "calibration_split": args.threshold_calibration_split,
            "calibration_pair_count": len(calibration_pair_ids),
            "holdout_pair_count": len(holdout_pair_ids),
            "target_gpt_recall": args.target_gpt_recall,
            "calibration_policy": "condition-specific threshold search uses only calibration-pair labels and targets a conservative GPT-image2 recall instead of maximizing holdout appearance; official metrics use the opposite held-out pairs",
        },
        "fpr_threshold": args.fpr_threshold,
        "candidate_training_result": candidate_training_result,
        "active_eval": strip_predictions(active_eval),
        "candidate_eval": strip_predictions(candidate_eval),
        "raw_all_condition_summaries": {
            "active": summarize_conditions(active_eval["sample_predictions"]),
            "candidate": summarize_conditions(candidate_eval["sample_predictions"]),
        },
        "official_operating_point": operating_point,
        "condition_summaries": condition_summaries,
        "acceptance": acceptance,
        "recommendation": recommendation(acceptance),
        "limitations": [
            "The Weibo/XHS returned set is small and should be read as a black-box paired test, not a platform-wide rule.",
            "Synthetic platform-like augmentation is parameterized by observed artifacts; it is not an official codec model.",
            "Screenshot chains are not part of the main candidate objective because the recovered Weibo screenshot files are low-resolution rendered thumbnails and xhs_screenshot is unavailable.",
            "Candidate is not activated automatically.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    write_metric_csv(Path(args.output_csv), payload)

    print(json.dumps({"candidate_id": candidate_id, "active_unchanged": payload["active_unchanged"], "recommendation": payload["recommendation"]}, ensure_ascii=False, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(f"wrote {Path(args.output_csv)}")


def recommendation(acceptance: dict[str, Any]) -> str:
    if acceptance.get("passed_component_gate") is True:
        return "Candidate improves platform GPT-image2 recall under the configured FPR ceiling; keep as component candidate and do not auto-activate."
    return "Keep active frozen. Use this as an ablation unless platform recall improves without excessive real FPR."


def split_pair_ids(pair_ids: list[str], *, calibration_split: str) -> tuple[set[str], set[str]]:
    calibration: set[str] = set()
    holdout: set[str] = set()
    for pair_id in pair_ids:
        match = re.search(r"pair_(\d+)", pair_id)
        number = int(match.group(1)) if match else len(calibration) + len(holdout) + 1
        is_odd = number % 2 == 1
        goes_to_calibration = is_odd if calibration_split == "odd" else not is_odd
        if goes_to_calibration:
            calibration.add(pair_id)
        else:
            holdout.add(pair_id)
    return calibration, holdout


def filter_predictions(predictions: list[dict[str, Any]], pair_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row.get("pair_id") or "") in pair_ids]


def optimize_gpt_thresholds(
    predictions: list[dict[str, Any]],
    *,
    fpr_ceiling: float,
    target_gpt_recall: float,
) -> dict[str, Any]:
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    conditions = sorted({str(row.get("condition") or "") for row in predictions})
    for condition in conditions:
        rows = [row for row in predictions if row.get("condition") == condition]
        candidates = sorted({round(float(row.get("gpt_image2_score") or 0.0), 6) for row in rows})
        candidates = sorted({0.0, 1.001, *candidates})
        best: dict[str, Any] | None = None
        for threshold in candidates:
            simulated = apply_operating_thresholds(rows, {condition: threshold})
            summary = summarize_conditions(simulated).get(condition, {})
            fpr = float(summary.get("real_false_positive_rate") or 0.0)
            recall = float(summary.get("gpt_image2_recall") or 0.0)
            macro_f1 = float(summary.get("binary_macro_f1") or 0.0)
            if fpr > fpr_ceiling:
                continue
            candidate = {
                "threshold": round(float(threshold), 6),
                "gpt_image2_recall": round(recall, 3),
                "real_false_positive_rate": round(fpr, 3),
                "binary_macro_f1": round(macro_f1, 3),
            }
            if best is None or (
                -abs(candidate["gpt_image2_recall"] - target_gpt_recall),
                candidate["binary_macro_f1"],
                -candidate["real_false_positive_rate"],
                candidate["threshold"],
            ) > (
                -abs(best["gpt_image2_recall"] - target_gpt_recall),
                best["binary_macro_f1"],
                -best["real_false_positive_rate"],
                best["threshold"],
            ):
                best = candidate
        if best is None:
            best = {
                "threshold": 1.001,
                "gpt_image2_recall": 0.0,
                "real_false_positive_rate": 0.0,
                "binary_macro_f1": 0.0,
            }
        thresholds[condition] = float(best["threshold"])
        diagnostics[condition] = {
            **best,
            "fpr_ceiling": fpr_ceiling,
            "target_gpt_recall": target_gpt_recall,
            "policy": "choose the condition-specific threshold whose calibration recall is closest to the target GPT-image2 recall while keeping real FPR within the configured ceiling",
        }
    return {
        "method": "condition_specific_gpt_score_threshold_target_recall",
        "fpr_ceiling": fpr_ceiling,
        "target_gpt_recall": target_gpt_recall,
        "thresholds_by_condition": thresholds,
        "diagnostics_by_condition": diagnostics,
    }


def apply_operating_thresholds(
    predictions: list[dict[str, Any]],
    thresholds_by_condition: dict[str, float],
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in predictions:
        copied = dict(row)
        condition = str(copied.get("condition") or "")
        threshold = float(thresholds_by_condition.get(condition, 1.001))
        score = float(copied.get("gpt_image2_score") or 0.0)
        copied["raw_prediction_before_operating_point"] = copied.get("prediction")
        copied["prediction"] = "gpt-image2" if score >= threshold else "real"
        copied["predicted_binary"] = "generated" if copied["prediction"] != "real" else "real"
        copied["confidence"] = round(score if copied["prediction"] == "gpt-image2" else 1.0 - score, 3)
        copied["operating_point"] = {
            "method": "condition_specific_gpt_score_threshold",
            "threshold": threshold,
            "gpt_image2_score": score,
        }
        adjusted.append(copied)
    return adjusted


def evaluate_download_acceptance(
    active: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    fpr_threshold: float,
) -> dict[str, Any]:
    required_conditions = ("weibo_download", "xhs_download")
    checks: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    improvements: list[float] = []
    for condition in required_conditions:
        active_row = active.get(condition, {})
        candidate_row = candidate.get(condition, {})
        active_recall = number(active_row.get("gpt_image2_recall"))
        candidate_recall = number(candidate_row.get("gpt_image2_recall"))
        active_fpr = number(active_row.get("real_false_positive_rate"))
        candidate_fpr = number(candidate_row.get("real_false_positive_rate"))
        recall_delta = (
            round(candidate_recall - active_recall, 3)
            if active_recall is not None and candidate_recall is not None
            else None
        )
        fpr_delta = (
            round(candidate_fpr - active_fpr, 3)
            if active_fpr is not None and candidate_fpr is not None
            else None
        )
        if recall_delta is not None:
            improvements.append(recall_delta)
        passed_fpr = candidate_fpr is not None and candidate_fpr <= fpr_threshold
        if not passed_fpr:
            hard_failures.append(f"{condition} real FPR {candidate_fpr} exceeds {fpr_threshold:.3f}")
        checks.append(
            {
                "condition": condition,
                "active_gpt_image2_recall": active_recall,
                "candidate_gpt_image2_recall": candidate_recall,
                "recall_delta": recall_delta,
                "active_real_fpr": active_fpr,
                "candidate_real_fpr": candidate_fpr,
                "real_fpr_delta": fpr_delta,
                "passed_real_fpr_guard": passed_fpr,
                "improved_recall": recall_delta is not None and recall_delta > 0,
            }
        )
    improved_weibo = any(
        check["condition"] == "weibo_download" and check.get("improved_recall")
        for check in checks
    )
    passed = improved_weibo and not hard_failures
    return {
        "passed_component_gate": passed,
        "hard_real_fpr_threshold": fpr_threshold,
        "improved_weibo_download_recall": improved_weibo,
        "hard_failures": hard_failures,
        "checks": checks,
        "summary": (
            "Candidate improves Weibo download GPT-image2 recall under the configured FPR ceiling."
            if passed
            else "Candidate is not recommended under the current download-focused recall/FPR gate."
        ),
    }


def number(value: object) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def write_metric_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_role",
        "model_id",
        "condition",
        "sample_count",
        "binary_macro_f1",
        "generated_recall",
        "gpt_image2_recall",
        "real_false_positive_rate",
        "gpt_image2_auc",
        "average_confidence",
        "calibration_threshold",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_role, conditions in payload["condition_summaries"].items():
            model_id = payload["active_model_id_before"] if model_role == "active" else payload["candidate_id"]
            for condition, metrics in conditions.items():
                writer.writerow(
                    {
                        "model_role": model_role,
                        "model_id": model_id,
                        "condition": condition,
                        "sample_count": metrics.get("sample_count"),
                        "binary_macro_f1": metrics.get("binary_macro_f1"),
                        "generated_recall": metrics.get("generated_recall"),
                        "gpt_image2_recall": metrics.get("gpt_image2_recall"),
                        "real_false_positive_rate": metrics.get("real_false_positive_rate"),
                        "gpt_image2_auc": metrics.get("gpt_image2_auc"),
                        "average_confidence": metrics.get("average_confidence"),
                        "calibration_threshold": "",
                    }
                )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Platform-Like Augmentation Candidate",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Active before: `{payload['active_model_id_before']}`",
        f"- Active after: `{payload['active_model_id_after']}`",
        f"- Active unchanged: `{payload['active_unchanged']}`",
        f"- Candidate id: `{payload['candidate_id']}`",
        f"- Experiment profile: `{payload['experiment_profile']}`",
        f"- Training policy: {payload['training_policy']['policy']}",
        f"- Augmentation conditions: `{', '.join(payload['training_policy']['augmentation_conditions'])}`",
        f"- Official operating point: `{payload['official_operating_point']['method']}`, FPR ceiling `{payload['official_operating_point']['fpr_ceiling']}`",
        f"- Threshold protocol: calibration split `{payload['threshold_calibration_protocol']['calibration_split']}`, calibration pairs `{payload['threshold_calibration_protocol']['calibration_pair_count']}`, holdout pairs `{payload['threshold_calibration_protocol']['holdout_pair_count']}`",
        f"- Recommendation: {payload['recommendation']}",
        "",
        "## Metrics",
        "",
        "| Condition | Model | N | Binary Macro-F1 | GPT-image2 Recall | Real FPR | GPT AUC | Avg Conf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    condition_order = ["clean", "weibo_download", "xhs_download", "weibo_screenshot", "xhs_screenshot"]
    summaries = payload["condition_summaries"]
    for condition in condition_order:
        for model_key, label in (("active", "active"), ("candidate", "candidate")):
            metrics = summaries.get(model_key, {}).get(condition)
            if not metrics:
                continue
            lines.append(
                "| {condition} | {model} | {n} | {macro:.3f} | {recall:.3f} | {fpr:.3f} | {auc:.3f} | {conf:.3f} |".format(
                    condition=condition,
                    model=label,
                    n=int(metrics.get("sample_count", 0)),
                    macro=float(metrics.get("binary_macro_f1", 0.0)),
                    recall=float(metrics.get("gpt_image2_recall", 0.0)),
                    fpr=float(metrics.get("real_false_positive_rate", 0.0)),
                    auc=float(metrics.get("gpt_image2_auc", 0.0)),
                    conf=float(metrics.get("average_confidence", 0.0)),
                )
            )
    lines.extend(["", "## Gate", ""])
    gate = payload["acceptance"]
    lines.append(f"- passed_component_gate: `{gate.get('passed_component_gate')}`")
    lines.append(f"- summary: {gate.get('summary')}")
    lines.extend(["", "## Operating Thresholds", ""])
    lines.append("| Condition | Threshold | Recall | Real FPR | Binary Macro-F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    diagnostics = payload["official_operating_point"].get("diagnostics_by_condition", {})
    for condition in condition_order:
        row = diagnostics.get(condition)
        if not row:
            continue
        lines.append(
            "| {condition} | {threshold:.3f} | {recall:.3f} | {fpr:.3f} | {macro:.3f} |".format(
                condition=condition,
                threshold=float(row.get("threshold", 0.0)),
                recall=float(row.get("gpt_image2_recall", 0.0)),
                fpr=float(row.get("real_false_positive_rate", 0.0)),
                macro=float(row.get("binary_macro_f1", 0.0)),
            )
        )
    lines.extend(["", "## Limitations", ""])
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


if __name__ == "__main__":
    main()
