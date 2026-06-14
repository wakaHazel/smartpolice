from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import ExternalTrainingSample, VisionTrainingRunRequest  # noqa: E402
from app.multimodal_training import (  # noqa: E402
    GENERATOR_ATTRIBUTION_TASK,
    _classification_metrics,
    _train_and_save_generator_attribution,
    extract_sample_features,
    get_vision_competition_summary,
)
from app.storage import get_vision_training_artifact_by_id, initialize_database  # noqa: E402
from tools.run_platform_transcode_eval import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_RETURNED_ROOT,
    build_payload,
    clean_eval_items,
    collect_returned_items,
    evaluate_items,
    load_artifact,
    load_manifest,
    parse_csv_arg,
    summarize_conditions,
)


DEFAULT_PLATFORM_MANIFEST = (
    ROOT / "platform_eval" / "returned" / "SmartPolice_real-platform-transcode-60_manifest.jsonl"
)
DEFAULT_OUTPUT_JSON = ROOT / "output" / "audits" / "platform_candidate_experiment_latest.json"
DEFAULT_OUTPUT_MD = ROOT / "output" / "audits" / "platform_candidate_experiment_latest.md"
DEFAULT_OUTPUT_CSV = ROOT / "output" / "audits" / "platform_candidate_experiment_latest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate-only GPT-image2 detector on real platform transcode hard perturbations "
            "and compare it with the frozen active model."
        )
    )
    parser.add_argument("--platform-manifest", default=str(DEFAULT_PLATFORM_MANIFEST))
    parser.add_argument("--upload-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--returned-root", default=str(DEFAULT_RETURNED_ROOT))
    parser.add_argument("--platforms", default="weibo,xhs")
    parser.add_argument("--variants", default="download,screenshot")
    parser.add_argument("--train-split", choices=("odd", "even"), default="odd")
    parser.add_argument("--profile", default="gpt_image2_ovr", choices=("gpt_image2_ovr", "binary_generated_gate"))
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--calibration-floor",
        type=float,
        default=0.60,
        help="Pre-registered GPT-image2 score floor. Lower values trade more recall for more real-image FPR.",
    )
    parser.add_argument(
        "--fpr-threshold",
        type=float,
        default=0.10,
        help="Maximum acceptable real false-positive rate for the component gate.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    initialize_database()
    active_before = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    platform_rows = load_platform_rows(Path(args.platform_manifest))
    train_rows, holdout_rows = split_platform_rows(platform_rows, args.train_split)
    train_samples = [sample_from_manifest_row(row) for row in train_rows]
    train_feature_rows = [
        extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK)
        for sample in train_samples
    ]

    candidate = _train_and_save_generator_attribution(
        samples=train_samples,
        rows=train_feature_rows,
        request=VisionTrainingRunRequest(
            task_type=GENERATOR_ATTRIBUTION_TASK,
            activation_mode="candidate",
            experiment_profile=args.profile,
            validation_strategy="class_stratified",
            min_samples=args.min_samples,
            enable_perturbation_augmentation=False,
            max_augmented_samples=0,
            enable_open_set_unknown=False,
            open_set_min_margin=0.0,
        ),
        candidate_count=len(train_samples),
    )
    candidate_artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, candidate.id)
    if candidate_artifact is None:
        raise SystemExit(f"candidate artifact not found after training: {candidate.id}")

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
        model_id=candidate.id,
        manifest_items=manifest_items,
        collection_rows=collection_rows,
        eval_items=eval_items,
        returned_root=Path(args.returned_root),
        platforms=platforms,
        variants=variants,
    )

    train_pair_ids = {str(row["pair_id"]) for row in train_rows}
    holdout_pair_ids = {str(row["pair_id"]) for row in holdout_rows}
    scoped = {
        "all": {
            "active": summarize_conditions(active_eval["sample_predictions"]),
            "candidate": summarize_conditions(candidate_eval["sample_predictions"]),
        },
        "train_pairs": {
            "active": summarize_conditions(filter_predictions(active_eval["sample_predictions"], train_pair_ids)),
            "candidate": summarize_conditions(filter_predictions(candidate_eval["sample_predictions"], train_pair_ids)),
        },
        "holdout_pairs": {
            "active": summarize_conditions(filter_predictions(active_eval["sample_predictions"], holdout_pair_ids)),
            "candidate": summarize_conditions(filter_predictions(candidate_eval["sample_predictions"], holdout_pair_ids)),
        },
    }
    calibration = fit_balanced_gpt_thresholds(
        candidate_eval["sample_predictions"],
        train_pair_ids,
        threshold_floor=args.calibration_floor,
    )
    calibrated_candidate_predictions = apply_gpt_thresholds(
        candidate_eval["sample_predictions"],
        calibration["thresholds_by_condition"],
    )
    calibrated_scoped = {
        "all": summarize_conditions(calibrated_candidate_predictions),
        "train_pairs": summarize_conditions(filter_predictions(calibrated_candidate_predictions, train_pair_ids)),
        "holdout_pairs": summarize_conditions(filter_predictions(calibrated_candidate_predictions, holdout_pair_ids)),
    }
    acceptance_raw = evaluate_acceptance(
        scoped["holdout_pairs"]["active"],
        scoped["holdout_pairs"]["candidate"],
        fpr_threshold=args.fpr_threshold,
    )
    acceptance_calibrated = evaluate_acceptance(
        scoped["holdout_pairs"]["active"],
        calibrated_scoped["holdout_pairs"],
        fpr_threshold=args.fpr_threshold,
    )

    payload: dict[str, Any] = {
        "id": f"platform-candidate-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "task_type": GENERATOR_ATTRIBUTION_TASK,
        "active_model_id_before": active_before,
        "active_model_id_after": get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id,
        "active_unchanged": get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id == active_before,
        "candidate_id": candidate.id,
        "candidate_status": candidate.status,
        "experiment_profile": args.profile,
        "calibration_floor": args.calibration_floor,
        "fpr_threshold": args.fpr_threshold,
        "split_policy": {
            "pair_split": args.train_split,
            "train_pair_count": len(train_pair_ids),
            "holdout_pair_count": len(holdout_pair_ids),
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "train_label_distribution": dict(sorted(Counter(row["label"] for row in train_rows).items())),
            "holdout_label_distribution": dict(sorted(Counter(row["label"] for row in holdout_rows).items())),
            "train_condition_distribution": dict(sorted(Counter(row["condition"] for row in train_rows).items())),
            "holdout_condition_distribution": dict(sorted(Counter(row["condition"] for row in holdout_rows).items())),
            "policy": "Train on one parity of platform pairs and report the other parity as platform holdout.",
        },
        "platform_manifest": str(Path(args.platform_manifest)),
        "active_eval": strip_predictions(active_eval),
        "candidate_eval": strip_predictions(candidate_eval),
        "scoped_condition_summaries": scoped,
        "calibration": calibration,
        "calibrated_candidate_condition_summaries": calibrated_scoped,
        "acceptance": {
            "raw_candidate": acceptance_raw,
            "calibrated_candidate": acceptance_calibrated,
        },
        "activation_recommendation": activation_recommendation(acceptance_calibrated),
        "limitations": [
            "This is a candidate-only component experiment; it does not replace the active model.",
            "The platform set has 60 clean pairs and 180 returned files, so confidence intervals are wide.",
            "Odd/even pair holdout prevents direct training-on-the-same-file reporting, but the clean source images may overlap historical external pools.",
            "Balanced calibration thresholds are fit from train-pair real samples with a fixed score floor, then applied to holdout pairs.",
            "The experiment treats Weibo/XHS as black-box propagation conditions; it does not infer exact platform transcode rules.",
            "xhs_screenshot remains unavailable and is not fabricated from downloaded images.",
        ],
    }
    if payload["active_model_id_after"] != active_before:
        raise RuntimeError(
            f"active changed unexpectedly: {active_before} -> {payload['active_model_id_after']}"
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    write_csv(output_csv, payload)
    print(json.dumps({
        "candidate_id": candidate.id,
        "active_unchanged": payload["active_unchanged"],
        "acceptance": payload["acceptance"],
        "calibration": calibration,
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_md),
            "csv": str(output_csv),
        },
    }, ensure_ascii=False, indent=2))


def load_platform_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"platform manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("condition") == "xhs_screenshot":
                continue
            image_path = resolve_image(row)
            if not image_path.is_file():
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no usable platform rows found in {path}")
    return rows


def split_platform_rows(
    rows: list[dict[str, Any]],
    train_split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for row in rows:
        number = pair_number(str(row.get("pair_id") or ""))
        is_odd = number % 2 == 1
        use_for_train = is_odd if train_split == "odd" else not is_odd
        (train_rows if use_for_train else holdout_rows).append(row)
    if not train_rows or not holdout_rows:
        raise SystemExit("platform split produced empty train or holdout rows")
    return train_rows, holdout_rows


def pair_number(pair_id: str) -> int:
    match = re.search(r"pair_(\d+)", pair_id)
    if not match:
        raise ValueError(f"cannot parse pair number from {pair_id!r}")
    return int(match.group(1))


def sample_from_manifest_row(row: dict[str, Any]) -> ExternalTrainingSample:
    image_path = resolve_image(row)
    label = str(row.get("label") or "unknown")
    condition = str(row.get("condition") or "platform")
    pair_id = str(row.get("pair_id") or image_path.stem)
    content = " ".join(
        str(row.get(key) or "")
        for key in ("caption", "title", "source", "source_detail", "benchmark_role")
    ).strip()
    sha = file_sha256(image_path)
    digest = hashlib.sha256(
        f"platform-candidate|{condition}|{pair_id}|{label}|{sha}".encode("utf-8")
    ).hexdigest()[:18]
    return ExternalTrainingSample(
        id=f"platform-candidate-{digest}",
        dataset_name="SmartPolice platform candidate train split",
        source=f"SmartPolice/platform_candidate:{condition}",
        source_url=str(row.get("source_url") or ""),
        task_type=GENERATOR_ATTRIBUTION_TASK,
        split="platform_candidate_train",
        title=str(row.get("title") or f"{pair_id} {condition}"),
        content=content or f"{condition} platform transcode candidate sample",
        image_path=str(image_path),
        image_url=None,
        image_sha256=sha,
        image_available=True,
        label=label,
        risk_score=82 if label != "real" else 18,
        scenario="社交平台传播扰动 candidate-only 训练",
        raw_payload=dict(row),
        created_at=datetime.now(UTC).isoformat(),
    )


def resolve_image(row: dict[str, Any]) -> Path:
    raw = Path(str(row.get("image") or ""))
    return raw if raw.is_absolute() else ROOT / raw


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model(
    *,
    model_id: str,
    manifest_items: list[Any],
    collection_rows: list[dict[str, object]],
    eval_items: list[Any],
    returned_root: Path,
    platforms: list[str],
    variants: list[str],
) -> dict[str, Any]:
    artifact, model_scope = load_artifact(model_id)
    predictor = _predictor = None
    from app.multimodal_training import _generator_predictor_from_artifact  # local import keeps script startup clear

    predictor = _generator_predictor_from_artifact(artifact)
    predictions = evaluate_items(eval_items, predictor)
    return build_payload(
        manifest_items=manifest_items,
        collection_rows=collection_rows,
        predictions=predictions,
        artifact=artifact,
        model_scope=model_scope,
        returned_root=returned_root,
        platforms=platforms,
        variants=variants,
        include_sample_predictions=True,
    )


def filter_predictions(predictions: list[dict[str, Any]], pair_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row.get("pair_id") or "") in pair_ids]


def fit_balanced_gpt_thresholds(
    predictions: list[dict[str, Any]],
    train_pair_ids: set[str],
    threshold_floor: float,
) -> dict[str, Any]:
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    train_rows = filter_predictions(predictions, train_pair_ids)
    for condition in sorted({str(row.get("condition") or "") for row in train_rows}):
        rows = [row for row in train_rows if row.get("condition") == condition]
        real_scores = [
            float(row.get("gpt_image2_score") or 0.0)
            for row in rows
            if row.get("label") == "real"
        ]
        gpt_scores = [
            float(row.get("gpt_image2_score") or 0.0)
            for row in rows
            if row.get("label") == "gpt-image2"
        ]
        threshold = round(max((max(real_scores) if real_scores else 1.0) + 0.001, threshold_floor), 3)
        threshold = min(max(threshold, 0.0), 1.001)
        diagnostics[condition] = {
            "threshold": threshold,
            "train_real_count": len(real_scores),
            "train_gpt_image2_count": len(gpt_scores),
            "train_real_max_score": round(max(real_scores), 3) if real_scores else None,
            "train_gpt_image2_max_score": round(max(gpt_scores), 3) if gpt_scores else None,
            "train_gpt_image2_recall_at_threshold": (
                round(sum(1 for score in gpt_scores if score >= threshold) / len(gpt_scores), 3)
                if gpt_scores
                else None
            ),
            "threshold_floor": threshold_floor,
            "policy": "threshold = max(train-pair real GPT-image2 score + 0.001, fixed balanced score floor)",
        }
        thresholds[condition] = threshold
    return {
        "method": "condition_specific_balanced_gpt_score_threshold",
        "threshold_floor": threshold_floor,
        "thresholds_by_condition": thresholds,
        "diagnostics_by_condition": diagnostics,
        "target": "preserve real FPR while recovering GPT-image2 platform recall",
    }


def apply_gpt_thresholds(
    predictions: list[dict[str, Any]],
    thresholds_by_condition: dict[str, float],
) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for row in predictions:
        copied = dict(row)
        condition = str(copied.get("condition") or "")
        threshold = float(thresholds_by_condition.get(condition, 1.001))
        score = float(copied.get("gpt_image2_score") or 0.0)
        copied["raw_prediction_before_calibration"] = copied.get("prediction")
        copied["prediction"] = "gpt-image2" if score >= threshold else "real"
        copied["predicted_binary"] = "generated" if copied["prediction"] != "real" else "real"
        copied["confidence"] = round(score if copied["prediction"] == "gpt-image2" else 1.0 - score, 3)
        copied["calibration"] = {
            "method": "condition_specific_gpt_score_threshold",
            "threshold": threshold,
            "gpt_image2_score": score,
        }
        calibrated.append(copied)
    return calibrated


def strip_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    stripped["sample_predictions"] = []
    return stripped


def evaluate_acceptance(
    active: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    fpr_threshold: float,
) -> dict[str, Any]:
    required_conditions = ("weibo_download", "weibo_screenshot")
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
        passed_recall = recall_delta is not None and recall_delta > 0
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
                "improved_recall": passed_recall,
            }
        )
    improved_any = any(delta > 0 for delta in improvements)
    passed = improved_any and not hard_failures
    return {
        "passed_component_gate": passed,
        "hard_real_fpr_threshold": fpr_threshold,
        "improved_any_weibo_recall": improved_any,
        "hard_failures": hard_failures,
        "checks": checks,
        "summary": (
            "Candidate improves at least one Weibo GPT-image2 recall condition while preserving the real-FPR guard."
            if passed
            else "Candidate is not recommended for activation/component use under the current hard real-FPR guard."
        ),
    }


def activation_recommendation(acceptance: dict[str, Any]) -> dict[str, Any]:
    return {
        "replace_active": False,
        "component_candidate_use": bool(acceptance.get("passed_component_gate")),
        "reason": (
            "Keep active frozen. This candidate can only be considered as a platform-perturbation component."
            if acceptance.get("passed_component_gate")
            else "Keep active frozen. Candidate did not pass the platform hard-FPR/recall improvement gate."
        ),
    }


def number(value: object) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Platform Candidate Experiment",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Active before: `{payload['active_model_id_before']}`",
        f"- Active after: `{payload['active_model_id_after']}`",
        f"- Active unchanged: `{payload['active_unchanged']}`",
        f"- Candidate id: `{payload['candidate_id']}`",
        f"- Experiment profile: `{payload['experiment_profile']}`",
        f"- Calibration floor: `{payload['calibration_floor']}`",
        f"- FPR threshold: `{payload['fpr_threshold']}`",
        f"- Recommendation: `{payload['activation_recommendation']['reason']}`",
        "",
        "## Split",
        "",
        "| Split | Pairs | Rows | Labels | Conditions |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    split = payload["split_policy"]
    lines.append(
        f"| Train | {split['train_pair_count']} | {split['train_row_count']} | "
        f"{json.dumps(split['train_label_distribution'], ensure_ascii=False)} | "
        f"{json.dumps(split['train_condition_distribution'], ensure_ascii=False)} |"
    )
    lines.append(
        f"| Holdout | {split['holdout_pair_count']} | {split['holdout_row_count']} | "
        f"{json.dumps(split['holdout_label_distribution'], ensure_ascii=False)} | "
        f"{json.dumps(split['holdout_condition_distribution'], ensure_ascii=False)} |"
    )
    for scope in ("all", "holdout_pairs"):
        lines.extend(["", f"## Metrics: {scope}", ""])
        lines.extend(
            [
                "| Condition | Model | N | Binary Macro-F1 | GPT-image2 Recall | Real FPR | GPT AUC | Avg Conf |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        scoped = payload["scoped_condition_summaries"][scope]
        for condition in ("clean", "weibo_download", "weibo_screenshot", "xhs_download"):
            for role in ("active", "candidate"):
                row = scoped.get(role, {}).get(condition, {})
                if not row:
                    continue
                lines.append(
                    f"| {condition} | {role} | {row.get('sample_count')} | "
                    f"{fmt(row.get('binary_macro_f1'))} | {fmt(row.get('gpt_image2_recall'))} | "
                    f"{fmt(row.get('real_false_positive_rate'))} | {fmt(row.get('gpt_image2_auc'))} | "
                    f"{fmt(row.get('average_confidence'))} |"
                )
        if scope in payload["calibrated_candidate_condition_summaries"]:
            lines.extend(["", f"## Calibrated Candidate Metrics: {scope}", ""])
            lines.extend(
                [
                    "| Condition | N | Threshold | Binary Macro-F1 | GPT-image2 Recall | Real FPR | GPT AUC | Avg Conf |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            calibrated = payload["calibrated_candidate_condition_summaries"][scope]
            thresholds = payload["calibration"]["thresholds_by_condition"]
            for condition in ("clean", "weibo_download", "weibo_screenshot", "xhs_download"):
                row = calibrated.get(condition, {})
                if not row:
                    continue
                lines.append(
                    f"| {condition} | {row.get('sample_count')} | {fmt(thresholds.get(condition))} | "
                    f"{fmt(row.get('binary_macro_f1'))} | {fmt(row.get('gpt_image2_recall'))} | "
                    f"{fmt(row.get('real_false_positive_rate'))} | {fmt(row.get('gpt_image2_auc'))} | "
                    f"{fmt(row.get('average_confidence'))} |"
                )
    lines.extend(["", "## Gate", ""])
    lines.append("Raw candidate:")
    for check in payload["acceptance"]["raw_candidate"]["checks"]:
        lines.append(
            f"- `{check['condition']}`: recall {fmt(check['active_gpt_image2_recall'])} -> "
            f"{fmt(check['candidate_gpt_image2_recall'])}, real FPR "
            f"{fmt(check['active_real_fpr'])} -> {fmt(check['candidate_real_fpr'])}."
        )
    lines.append("")
    lines.append("Calibrated candidate:")
    for check in payload["acceptance"]["calibrated_candidate"]["checks"]:
        lines.append(
            f"- `{check['condition']}`: recall {fmt(check['active_gpt_image2_recall'])} -> "
            f"{fmt(check['candidate_gpt_image2_recall'])}, real FPR "
            f"{fmt(check['active_real_fpr'])} -> {fmt(check['candidate_real_fpr'])}."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "condition",
        "model_role",
        "model_id",
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
        for scope, scoped in payload["scoped_condition_summaries"].items():
            for model_role, conditions in scoped.items():
                model_id = payload["active_model_id_before"] if model_role == "active" else payload["candidate_id"]
                for condition, row in conditions.items():
                    writer.writerow(
                        {
                            "scope": scope,
                            "condition": condition,
                            "model_role": model_role,
                            "model_id": model_id,
                            "sample_count": row.get("sample_count"),
                            "binary_macro_f1": row.get("binary_macro_f1"),
                            "generated_recall": row.get("generated_recall"),
                            "gpt_image2_recall": row.get("gpt_image2_recall"),
                            "real_false_positive_rate": row.get("real_false_positive_rate"),
                            "gpt_image2_auc": row.get("gpt_image2_auc"),
                            "average_confidence": row.get("average_confidence"),
                            "calibration_threshold": "",
                        }
                    )
        for scope, conditions in payload["calibrated_candidate_condition_summaries"].items():
            for condition, row in conditions.items():
                writer.writerow(
                    {
                        "scope": scope,
                        "condition": condition,
                        "model_role": "candidate_calibrated",
                        "model_id": payload["candidate_id"],
                        "sample_count": row.get("sample_count"),
                        "binary_macro_f1": row.get("binary_macro_f1"),
                        "generated_recall": row.get("generated_recall"),
                        "gpt_image2_recall": row.get("gpt_image2_recall"),
                        "real_false_positive_rate": row.get("real_false_positive_rate"),
                        "gpt_image2_auc": row.get("gpt_image2_auc"),
                        "average_confidence": row.get("average_confidence"),
                        "calibration_threshold": payload["calibration"]["thresholds_by_condition"].get(condition),
                    }
                )


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
