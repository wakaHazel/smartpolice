from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import (  # noqa: E402
    VisionCandidateEvaluationRequest,
    VisionFeatureAblationRunRequest,
    VisionRobustnessRunRequest,
    VisionSourceHoldoutRunRequest,
    VisionTrainingRunRequest,
)
from app.multimodal_training import (  # noqa: E402
    evaluate_vision_candidate,
    get_vision_competition_summary,
    run_vision_feature_ablation_experiment,
    run_vision_robustness_experiment,
    run_vision_source_holdout_experiment,
    train_vision_evidence_head,
)
from app.storage import (  # noqa: E402
    get_vision_training_artifact_by_id,
    initialize_database,
    list_vision_training_runs,
)


TASK_TYPE = "vision_generator_attribution"
ROBUSTNESS_CONDITIONS = [
    "clean",
    "jpeg_q85",
    "jpeg_q60",
    "screenshot_resave",
    "center_crop",
    "watermark",
]
TRAIN_AUGMENTATION_CONDITIONS = [
    "jpeg_q85",
    "jpeg_q60",
    "screenshot_resave",
    "center_crop",
    "watermark",
]
DEFAULT_TMP_DIR = ROOT / "tmp" / "baseline_matrix"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the baseline-driven robustness matrix without activating candidate models."
    )
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "audits"))
    parser.add_argument("--docs-path", default=str(ROOT / "docs" / "benchmark_results.md"))
    parser.add_argument(
        "--tmp-dir",
        default=os.environ.get("SMARTPOLICE_TMP_DIR", str(DEFAULT_TMP_DIR)),
        help="Temporary directory for perturbation variants. Defaults to the D: workspace tmp folder.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use bounded limits suitable for a disk-safe smoke baseline run.",
    )
    parser.add_argument("--robustness-limit", type=int, default=120)
    parser.add_argument("--source-sample-limit", type=int, default=1000)
    parser.add_argument("--max-holdout-groups", type=int, default=12)
    parser.add_argument("--feature-ablation-limit", type=int, default=120)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-source-holdout", action="store_true")
    parser.add_argument("--skip-feature-ablation", action="store_true")
    parser.add_argument(
        "--train-candidate",
        action="store_true",
        help="Train a non-active candidate. Omitted by default to keep the baseline run light.",
    )
    parser.add_argument("--skip-candidate", action="store_true")
    parser.add_argument("--candidate-min-samples", type=int, default=20)
    parser.add_argument("--candidate-max-augmented-samples", type=int, default=2500)
    parser.add_argument("--candidate-eval-limit", type=int, default=120)
    parser.add_argument("--candidate-eval-include-source-holdout", action="store_true")
    parser.add_argument("--candidate-eval-include-feature-ablation", action="store_true")
    parser.add_argument(
        "--candidate-model-id",
        default="",
        help="Evaluate an existing candidate model without retraining it.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Update output/audits/baseline_matrix_latest.json instead of starting a fresh report.",
    )
    args = parser.parse_args()

    if args.quick:
        args.robustness_limit = min(args.robustness_limit, 80)
        args.source_sample_limit = min(args.source_sample_limit, 320)
        args.max_holdout_groups = min(args.max_holdout_groups, 6)
        args.feature_ablation_limit = min(args.feature_ablation_limit, 80)

    configure_temp_dir(Path(args.tmp_dir))
    initialize_database()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(UTC).isoformat()
    baseline_summary = get_vision_competition_summary(TASK_TYPE)
    active_before = baseline_summary.active_model_id
    latest_path = output_dir / "baseline_matrix_latest.json"
    docs_path = Path(args.docs_path)
    if args.resume_latest:
        if not latest_path.exists():
            raise SystemExit(f"cannot resume because {latest_path} does not exist")
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        if payload.get("task_type") != TASK_TYPE:
            raise SystemExit(f"cannot resume report for a different task: {payload.get('task_type')}")
        previous_active = payload.get("active_model_id_before")
        if previous_active and previous_active != active_before:
            raise SystemExit(f"active model changed since the report was created: {previous_active} -> {active_before}")
        payload["baseline_summary"] = model_dump(baseline_summary)
        payload["resumed_at"] = created_at
        payload["last_run_config"] = normalized_args(args)
    else:
        payload: dict[str, Any] = {
            "id": f"baseline-matrix-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
            "created_at": created_at,
            "task_type": TASK_TYPE,
            "active_model_id_before": active_before,
            "active_model_id_after": None,
            "active_unchanged": None,
            "run_config": normalized_args(args),
            "disk_policy": {
                "tmp_dir": str(Path(args.tmp_dir).resolve()),
                "uses_e_drive": False,
                "downloads_external_benchmarks": False,
            },
            "baseline_summary": model_dump(baseline_summary),
            "robustness": None,
            "source_holdout": None,
            "feature_ablation": None,
            "candidate": None,
            "candidate_evaluation": None,
            "strict_activation_recommendation": None,
            "benchmark_role": {
                "current_pool": "Existing 4691 external samples are used as the first benchmark matrix.",
                "genimage_aigibench": "Import only when local data is available; default use is evaluation/source-holdout.",
                "sida_rr_itw": "Research and perturbation-protocol references until licensing and labels are confirmed.",
            },
        }
        flush_outputs(payload, latest_path, docs_path)

    print(f"active_before={active_before}")
    if not args.skip_robustness:
        print("running robustness matrix...")
        robustness = run_vision_robustness_experiment(
            VisionRobustnessRunRequest(
                task_type=TASK_TYPE,
                limit=args.robustness_limit,
                conditions=ROBUSTNESS_CONDITIONS,
                include_sample_predictions=False,
            )
        )
        payload["robustness"] = model_dump(robustness)
        flush_outputs(payload, latest_path, docs_path)
    else:
        print("skipping robustness matrix")

    if not args.skip_source_holdout:
        print("running source-holdout matrix...")
        source_holdout = run_vision_source_holdout_experiment(
            VisionSourceHoldoutRunRequest(
                task_type=TASK_TYPE,
                holdout_key="dataset_source",
                sample_limit=args.source_sample_limit,
                max_holdout_groups=args.max_holdout_groups,
                min_train_samples=4,
                min_holdout_samples=1,
                enable_perturbation_augmentation=False,
            )
        )
        payload["source_holdout"] = model_dump(source_holdout)
        flush_outputs(payload, latest_path, docs_path)
    else:
        print("skipping source-holdout matrix")

    if not args.skip_feature_ablation:
        print("running feature ablation...")
        feature_ablation = run_vision_feature_ablation_experiment(
            VisionFeatureAblationRunRequest(
                task_type=TASK_TYPE,
                limit=args.feature_ablation_limit,
                min_samples=4,
            )
        )
        payload["feature_ablation"] = model_dump(feature_ablation)
        flush_outputs(payload, latest_path, docs_path)
    else:
        print("skipping feature ablation")

    candidate = None
    candidate_evaluation = None
    plan_gate = None
    if args.train_candidate and not args.skip_candidate:
        print("training candidate model without activation...")
        candidate = train_vision_evidence_head(
            VisionTrainingRunRequest(
                task_type=TASK_TYPE,
                min_samples=args.candidate_min_samples,
                activation_mode="candidate",
                validation_strategy="source_holdout",
                source_holdout_fraction=0.2,
                min_source_holdout_samples=20,
                enable_perturbation_augmentation=True,
                augmentation_conditions=TRAIN_AUGMENTATION_CONDITIONS,
                max_augmented_samples=args.candidate_max_augmented_samples,
            )
        )
        print(f"candidate={candidate.id}")
        print("evaluating candidate without activation...")
        candidate_evaluation = evaluate_vision_candidate(
            VisionCandidateEvaluationRequest(
                task_type=TASK_TYPE,
                candidate_model_id=candidate.id,
                limit=args.candidate_eval_limit,
                conditions=ROBUSTNESS_CONDITIONS,
                include_source_holdout=args.candidate_eval_include_source_holdout,
                include_feature_ablation=args.candidate_eval_include_feature_ablation,
                activate_if_passes_gate=False,
            )
        )
        plan_gate = strict_activation_recommendation(
            candidate_evaluation.active_summary,
            candidate_evaluation.candidate_summary,
        )
        payload["candidate"] = model_dump(candidate)
        payload["candidate_evaluation"] = model_dump(candidate_evaluation)
        payload["strict_activation_recommendation"] = plan_gate
        flush_outputs(payload, latest_path, docs_path)
    elif args.candidate_model_id and not args.skip_candidate:
        candidate_model_id = args.candidate_model_id
        candidate_artifact = get_vision_training_artifact_by_id(TASK_TYPE, candidate_model_id)
        if candidate_artifact is None:
            raise SystemExit(f"candidate model not found: {candidate_model_id}")
        print(f"evaluating existing candidate without activation: {candidate_model_id}")
        candidate = existing_run_payload(candidate_model_id) or candidate_payload_from_artifact(candidate_artifact)
        candidate_evaluation = evaluate_vision_candidate(
            VisionCandidateEvaluationRequest(
                task_type=TASK_TYPE,
                candidate_model_id=candidate_model_id,
                limit=args.candidate_eval_limit,
                conditions=ROBUSTNESS_CONDITIONS,
                include_source_holdout=args.candidate_eval_include_source_holdout,
                include_feature_ablation=args.candidate_eval_include_feature_ablation,
                activate_if_passes_gate=False,
            )
        )
        plan_gate = strict_activation_recommendation(
            candidate_evaluation.active_summary,
            candidate_evaluation.candidate_summary,
        )
        payload["candidate"] = model_dump(candidate)
        payload["candidate_evaluation"] = model_dump(candidate_evaluation)
        payload["strict_activation_recommendation"] = plan_gate
        flush_outputs(payload, latest_path, docs_path)
    else:
        print("candidate training skipped; use --train-candidate to opt in")

    final_summary = get_vision_competition_summary(TASK_TYPE)
    if final_summary.active_model_id != active_before:
        raise RuntimeError(
            f"active model changed unexpectedly: {active_before} -> {final_summary.active_model_id}"
        )
    payload["active_model_id_after"] = final_summary.active_model_id
    payload["active_unchanged"] = final_summary.active_model_id == active_before
    stamped_path = output_dir / f"{payload['id']}.json"
    flush_outputs(payload, latest_path, docs_path)
    write_json(stamped_path, payload)
    Path(args.docs_path).write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {latest_path}")
    print(f"wrote {args.docs_path}")


def configure_temp_dir(tmp_dir: Path) -> None:
    resolved = tmp_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(resolved)
    os.environ["TMP"] = str(resolved)
    tempfile.tempdir = str(resolved)


def normalized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in sorted(vars(args).items())}


def existing_run_payload(run_id: str) -> dict[str, Any] | None:
    for record in list_vision_training_runs(TASK_TYPE, limit=100):
        if record.run.id == run_id:
            return model_dump(record.run)
    return None


def candidate_payload_from_artifact(artifact: dict[str, object]) -> dict[str, Any]:
    model_card = artifact.get("model_card")
    class_counts = artifact.get("class_counts")
    feature_names = artifact.get("feature_names")
    validation_protocol = artifact.get("validation_protocol")
    return {
        "id": artifact.get("id"),
        "created_at": artifact.get("created_at"),
        "task_type": artifact.get("task_type"),
        "model_kind": artifact.get("model_kind"),
        "status": "candidate_trained",
        "sample_count": (
            sum(int(value) for value in class_counts.values()) if isinstance(class_counts, dict) else 0
        ),
        "validation_count": (
            int(validation_protocol.get("validation_count", 0)) if isinstance(validation_protocol, dict) else 0
        ),
        "feature_count": len(feature_names) if isinstance(feature_names, list) else 0,
        "model_card": model_card if isinstance(model_card, dict) else {},
    }


def flush_outputs(payload: dict[str, Any], latest_path: Path, docs_path: Path) -> None:
    write_json(latest_path, payload)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_markdown(payload), encoding="utf-8")


def model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def strict_activation_recommendation(
    active_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = []
    checks.append(delta_check("clean_macro_f1", active_summary, candidate_summary, min_delta=-0.02))
    checks.append(delta_check("clean_gpt_image2_recall", active_summary, candidate_summary, min_delta=-0.02))
    checks.append(
        delta_check(
            "robust_average_macro_f1",
            active_summary,
            candidate_summary,
            min_delta=0.03,
        )
    )
    real_fp = lower_is_better_check(
        "clean_real_false_positive_rate",
        active_summary,
        candidate_summary,
        max_worse_delta=0.02,
        improvement_delta=0.03,
    )
    checks.append(real_fp)

    by_name = {check["metric"]: check for check in checks}
    no_core_regression = (
        by_name["clean_macro_f1"]["passed"]
        and by_name["clean_gpt_image2_recall"]["passed"]
        and by_name["clean_real_false_positive_rate"]["not_worse"]
    )
    meaningful_improvement = (
        by_name["robust_average_macro_f1"]["passed"]
        or by_name["clean_real_false_positive_rate"]["meaningfully_improved"]
    )
    suggest_activation = no_core_regression and meaningful_improvement
    return {
        "suggest_activation": suggest_activation,
        "decision": "suggest_activate" if suggest_activation else "keep_candidate",
        "reason": (
            "Candidate meets the stricter benchmark plan gate."
            if suggest_activation
            else "Candidate remains a non-active benchmark candidate under the stricter plan gate."
        ),
        "thresholds": {
            "clean_macro_f1_min_delta": -0.02,
            "clean_gpt_image2_recall_min_delta": -0.02,
            "robust_average_macro_f1_min_delta": 0.03,
            "real_false_positive_rate_max_worse_delta": 0.02,
            "real_false_positive_rate_meaningful_improvement_delta": 0.03,
        },
        "checks": checks,
    }


def delta_check(
    metric: str,
    active_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    min_delta: float,
) -> dict[str, Any]:
    active = number(active_summary.get(metric))
    candidate = number(candidate_summary.get(metric))
    delta = safe_delta(candidate, active)
    return {
        "metric": metric,
        "active": active,
        "candidate": candidate,
        "delta": delta,
        "min_delta": min_delta,
        "passed": delta is not None and delta >= min_delta,
    }


def lower_is_better_check(
    metric: str,
    active_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    max_worse_delta: float,
    improvement_delta: float,
) -> dict[str, Any]:
    active = number(active_summary.get(metric))
    candidate = number(candidate_summary.get(metric))
    delta = safe_delta(candidate, active)
    not_worse = delta is not None and delta <= max_worse_delta
    meaningfully_improved = delta is not None and -delta >= improvement_delta
    return {
        "metric": metric,
        "active": active,
        "candidate": candidate,
        "delta": delta,
        "max_worse_delta": max_worse_delta,
        "improvement_delta": improvement_delta,
        "not_worse": not_worse,
        "meaningfully_improved": meaningfully_improved,
        "passed": not_worse,
    }


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return round(float(value), 3)
    return None


def safe_delta(candidate: float | None, active: float | None) -> float | None:
    if candidate is None or active is None:
        return None
    return round(candidate - active, 3)


def derived_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    robustness = payload.get("robustness")
    if not isinstance(robustness, dict):
        return {}
    conditions = robustness.get("conditions")
    if not isinstance(conditions, list):
        return {}
    clean = next((item for item in conditions if item.get("condition") == "clean"), None)
    nonclean = [item for item in conditions if item.get("condition") != "clean"]
    clean_macro = number(clean.get("macro_f1")) if isinstance(clean, dict) else None
    nonclean_macro = [number(item.get("macro_f1")) for item in nonclean if isinstance(item, dict)]
    nonclean_macro = [item for item in nonclean_macro if item is not None]
    robust_average = round(sum(nonclean_macro) / len(nonclean_macro), 3) if nonclean_macro else None
    worst_nonclean = min(nonclean_macro) if nonclean_macro else None
    return {
        "clean_macro_f1": clean_macro,
        "robust_average_macro_f1": robust_average,
        "clean_to_worst_macro_f1_drop": (
            round(clean_macro - worst_nonclean, 3) if clean_macro is not None and worst_nonclean is not None else None
        ),
        "gpt_image2_clean_recall": number(clean.get("gpt_image2_recall")) if isinstance(clean, dict) else None,
        "real_clean_false_positive_rate": (
            real_false_positive_rate(clean.get("confusion_matrix", {})) if isinstance(clean, dict) else None
        ),
    }


def real_false_positive_rate(confusion_matrix: dict[str, Any]) -> float | None:
    real_row = confusion_matrix.get("real")
    if not isinstance(real_row, dict):
        return None
    total = sum(int(value) for value in real_row.values())
    if total <= 0:
        return None
    correct = int(real_row.get("real", 0))
    return round((total - correct) / total, 3)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["baseline_summary"]
    active_id = payload["active_model_id_before"] or "None"
    pool = summary.get("training_pool", {})
    validation = summary.get("validation_metrics", {})
    robustness = payload.get("robustness")
    source_holdout = payload.get("source_holdout")
    feature_ablation = payload.get("feature_ablation")
    candidate = payload.get("candidate")
    candidate_eval = payload.get("candidate_evaluation")
    plan_gate = payload.get("strict_activation_recommendation")
    run_config = payload.get("run_config", {})
    disk_policy = payload.get("disk_policy", {})

    lines = [
        "# Baseline Benchmark Results",
        "",
        f"- Generated at: `{payload['created_at']}`",
        f"- Task: `{payload['task_type']}`",
        f"- Active model kept frozen: `{active_id}`",
        f"- Active unchanged after run: `{payload['active_unchanged']}`",
        f"- Quick mode: `{run_config.get('quick', False)}`",
        f"- Temporary directory: `{disk_policy.get('tmp_dir', '-')}`",
        f"- External benchmark downloads in this run: `{disk_policy.get('downloads_external_benchmarks', False)}`",
        "",
        "## Active Baseline",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Training pool | {pool.get('sample_count', '-')} samples |",
        f"| Active kind | {summary.get('active_model_kind', '-')} |",
        f"| Clean accuracy | {fmt(validation.get('accuracy'))} |",
        f"| Clean Macro-F1 | {fmt(validation.get('macro_f1'))} |",
        f"| GPT-image-2 Recall | {fmt(validation.get('gpt_image2_recall'))} |",
        f"| Augmentation features | {summary.get('robustness_headline', {}).get('augmentation_feature_count', '-')} |",
        "",
        "## Robustness Matrix",
        "",
        "| Condition | Accuracy | Macro-F1 | GPT-image-2 Recall | Avg Confidence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if robustness:
        for item in robustness.get("conditions", []):
            lines.append(
                f"| {item.get('condition')} | {fmt(item.get('accuracy'))} | {fmt(item.get('macro_f1'))} | "
                f"{fmt(item.get('gpt_image2_recall'))} | {fmt(item.get('average_confidence'))} |"
            )
    else:
        lines.append("| Not run yet | - | - | - | - |")

    lines.extend(["", "## Source Holdout", ""])
    if source_holdout:
        aggregate = source_holdout.get("aggregate", {})
        lines.extend(
            [
                f"- Holdout key: `{source_holdout.get('holdout_key')}`",
                f"- Sample count: `{source_holdout.get('sample_count')}`",
                f"- Source groups: `{source_holdout.get('source_count')}`",
                f"- Aggregate: `{json.dumps(aggregate, ensure_ascii=False)}`",
                (
                    "- Seen-class diagnostic: "
                    f"`{fmt(aggregate.get('mean_seen_class_macro_f1'))}` mean Macro-F1 on "
                    f"`{int(number(aggregate.get('seen_class_holdout_count')) or 0)}` holdout samples whose labels appear in training; "
                    f"`{int(number(aggregate.get('unseen_holdout_count')) or 0)}` holdout samples use labels unseen by the training side."
                ),
                (
                    "- Binary screening diagnostic: "
                    f"`{fmt(aggregate.get('mean_binary_macro_f1'))}` generated-vs-real Macro-F1, "
                    f"`{fmt(aggregate.get('mean_generated_recall'))}` generated recall, "
                    f"`{fmt(aggregate.get('overall_real_false_positive_rate'))}` overall real false positive rate "
                    f"({int(number(aggregate.get('real_false_positive_count')) or 0)}/"
                    f"{int(number(aggregate.get('real_support')) or 0)} real samples)."
                ),
                (
                    "- Baseline-style label-covered diagnostic: "
                    f"`{fmt(aggregate.get('label_covered_macro_f1'))}` attribution Macro-F1, "
                    f"`{fmt(aggregate.get('label_covered_binary_macro_f1'))}` binary Macro-F1, "
                    f"`{fmt(aggregate.get('label_covered_real_false_positive_rate'))}` real false positive rate "
                    f"on `{int(number(aggregate.get('label_covered_holdout_count')) or 0)}` source-stratified holdout samples."
                ),
                "",
                "| Holdout group | Holdout samples | Macro-F1 | Seen-class Macro-F1 | Binary Macro-F1 | Real FP | Real FPR | Unseen samples | GPT-image-2 Recall | Skipped |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in source_holdout.get("groups", []):
            lines.append(
                f"| {item.get('holdout_group')} | {item.get('holdout_count', '-')} | "
                f"{fmt(item.get('macro_f1'))} | {fmt(item.get('seen_class_macro_f1'))} | "
                f"{fmt(item.get('binary_macro_f1'))} | {item.get('real_false_positive_count', 0)}/{item.get('real_support', 0)} | "
                f"{fmt(item.get('real_false_positive_rate'))} | "
                f"{item.get('unseen_holdout_count', 0)} | {fmt(item.get('gpt_image2_recall'))} | "
                f"{item.get('skipped', False)} |"
            )
    else:
        lines.append("- Source-holdout has not been run in this baseline file yet.")

    lines.extend(
        [
            "",
            "## Feature Ablation",
            "",
            "| Feature set | Macro-F1 | GPT-image-2 Recall | Skipped |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    if feature_ablation:
        for item in feature_ablation.get("results", []):
            lines.append(
                f"| {item.get('feature_set')} | {fmt(item.get('macro_f1'))} | "
                f"{fmt(item.get('gpt_image2_recall'))} | {item.get('skipped', False)} |"
            )
    else:
        lines.append("| Not run yet | - | - | True |")

    lines.extend(["", "## Main Metrics", ""])
    metrics = derived_metrics(payload)
    if metrics:
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Clean Macro-F1 | {fmt(metrics.get('clean_macro_f1'))} |",
                f"| Robust average Macro-F1 | {fmt(metrics.get('robust_average_macro_f1'))} |",
                f"| Clean-to-worst perturbation Macro-F1 drop | {fmt(metrics.get('clean_to_worst_macro_f1_drop'))} |",
                f"| GPT-image-2 clean recall | {fmt(metrics.get('gpt_image2_clean_recall'))} |",
                f"| Real clean false positive rate | {fmt(metrics.get('real_clean_false_positive_rate'))} |",
            ]
        )
    else:
        lines.append("- Derived metrics will appear after the robustness matrix is available.")

    lines.extend(["", "## Candidate Evaluation", ""])
    if candidate and candidate_eval and plan_gate:
        lines.extend(
            [
                f"- Candidate model: `{candidate.get('id')}`",
                f"- Candidate status: `{candidate.get('status')}`",
                f"- Activated during this run: `{candidate_eval.get('activated')}`",
                f"- Active before/after: `{candidate_eval.get('active_model_id_before')}` -> `{candidate_eval.get('active_model_id_after')}`",
                f"- Strict recommendation: `{plan_gate.get('decision')}`",
                f"- Reason: {plan_gate.get('reason')}",
                "",
                "| Gate metric | Active | Candidate | Delta | Passed |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for check in plan_gate.get("checks", []):
            lines.append(
                f"| {check.get('metric')} | {fmt(check.get('active'))} | {fmt(check.get('candidate'))} | "
                f"{fmt(check.get('delta'))} | {check.get('passed', check.get('not_worse'))} |"
            )
    else:
        lines.append("- Candidate training was skipped for this run.")

    robust_average_text = fmt(metrics.get("robust_average_macro_f1")) if metrics else "-"
    source_macro_text = "-"
    seen_source_macro_text = "-"
    unseen_count_text = 0
    binary_macro_text = "-"
    real_fpr_text = "-"
    label_covered_macro_text = "-"
    label_covered_binary_text = "-"
    real_fp_count_text = 0
    real_support_text = 0
    if source_holdout:
        aggregate = source_holdout.get("aggregate", {})
        source_macro_text = fmt(aggregate.get("mean_macro_f1"))
        seen_source_macro_text = fmt(aggregate.get("mean_seen_class_macro_f1"))
        unseen_count_text = int(number(aggregate.get("unseen_holdout_count")) or 0)
        binary_macro_text = fmt(aggregate.get("mean_binary_macro_f1"))
        real_fpr_text = fmt(aggregate.get("overall_real_false_positive_rate"))
        label_covered_macro_text = fmt(aggregate.get("label_covered_macro_f1"))
        label_covered_binary_text = fmt(aggregate.get("label_covered_binary_macro_f1"))
        real_fp_count_text = int(number(aggregate.get("real_false_positive_count")) or 0)
        real_support_text = int(number(aggregate.get("real_support")) or 0)

    lines.extend(
        [
            "",
            "## Generalization Benchmark Borrowing",
            "",
            "- GenImage contributes the cross-generator and degraded-image framing: evaluate generator shifts and perturbations such as compression, low-resolution variants, blur-like degradation, crop, and watermark without claiming full web coverage.",
            "- AIGIBench contributes the external-blind-test framing: keep benchmark samples traceable by source and use source-holdout before considering any active replacement.",
            "- SIDA/SID-Set contributes the social-media domain framing: treat social-platform images as a separate distribution with licensing, label, and sensitive-content checks before import.",
            "- RRDataset and ITW-SM contribute the in-the-wild propagation framing: prioritize platform resampling, screenshot-resave, recapture/retake, and repeated upload chains as future robustness conditions.",
            "- The project borrows these protocols, not their leaderboard claims: current output remains a suspected-source clue and weak source-holdout results are reported as a generalization boundary.",
            "- Robustness rows that show `1.000` are bounded condition checks on the sampled robustness subset, not a full-score claim; the main clean validation Macro-F1 remains in the Active Baseline section.",
            "",
            "| Baseline family | Borrowed generalization idea | Current project proxy | Current evidence |",
            "| --- | --- | --- | --- |",
            f"| GenImage | Cross-generator and degraded-image testing | Generator labels plus `clean/jpeg/crop/watermark/screenshot_resave` robustness matrix | Robust average Macro-F1 `{robust_average_text}`; screenshot-resave is the weakest condition |",
            f"| AIGIBench | External blind-test and source-aware evaluation | Strict `dataset_source` holdout plus label-covered source-stratified diagnostic | Strict mean Macro-F1 `{source_macro_text}`; seen-class `{seen_source_macro_text}`; label-covered Macro-F1 `{label_covered_macro_text}` / binary `{label_covered_binary_text}`; strict real FPR `{real_fpr_text}` (`{real_fp_count_text}/{real_support_text}`); `{unseen_count_text}` strict holdout samples have labels unseen by the training side |",
            "| SIDA/SID-Set | Social-media distribution shift | Treat social-platform samples as a separate import domain after license and label checks | Protocol reference only; no sensitive social-media set is mixed into active |",
            "| RRDataset / ITW-SM | In-the-wild propagation, resampling, recapture/retake | `screenshot_resave`, JPEG recompression, crop and watermark conditions | Recapture/retake and repeated upload chains remain next-round blind-test work |",
            "",
            "## Interpretation",
            "",
            "- These results use the currently imported external pool as the first benchmark matrix.",
            "- GenImage and AIGIBench are prepared as local-data imports through `tools/prepare_benchmark_manifest.py`; they are not assumed to be present or fully downloaded.",
            "- SIDA/SID-Set, RRDataset, and ITW-SM remain protocol references until licensing, labels, and sensitive-content constraints are confirmed.",
            "- Model outputs remain suspected-source clues only; they do not replace C2PA, watermark checks, platform metadata, publication-chain evidence, or human review.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    parsed = number(value)
    return "-" if parsed is None else f"{parsed:.3f}"


if __name__ == "__main__":
    main()
