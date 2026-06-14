from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
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
    _binary_generation_metrics,
    _generator_predictor_from_artifact,
    _generator_sample_domain,
    _normalize,
    _predict_generator_label,
    _predict_generated_probability_with_gate,
    _task_relevant_samples,
    extract_sample_features,
)
from app.storage import (  # noqa: E402
    get_vision_training_artifact_by_id,
    initialize_database,
    list_external_training_samples,
)
from tools.run_generator_experiment_suite import _balanced_profile_samples, _select_profile_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep generated-vs-real gate thresholds for a candidate artifact.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--profile", default="binary_generated_gate")
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--output", default=str(ROOT / "output" / "audits" / "binary_gate_threshold_sweep.json"))
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--margins", default="0.00,0.04,0.08,0.12")
    args = parser.parse_args()

    initialize_database()
    artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, args.candidate_id)
    if artifact is None:
        raise SystemExit(f"candidate not found: {args.candidate_id}")
    predictor = _generator_predictor_from_artifact(artifact)
    samples, labels, profile_report = load_profile_samples(args.profile, args.sample_limit)
    rows = [extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK) for sample in samples]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    margins = [float(item) for item in args.margins.split(",") if item.strip()]

    base_predictions = [
        cached_binary_gate_prediction(row, predictor)
        for row in rows
    ]
    sweeps = []
    for threshold in thresholds:
        for margin in margins:
            predictions = [
                swept_binary_label(base_prediction, threshold, margin)
                for base_prediction in base_predictions
            ]
            metrics = _binary_generation_metrics(predictions, labels)
            sweeps.append(
                {
                    "threshold": threshold,
                    "real_protection_margin": margin,
                    "binary_macro_f1": metrics.get("macro_f1"),
                    "generated_recall": metrics.get("generated_recall"),
                    "generated_precision": metrics.get("generated_precision"),
                    "real_false_positive_rate": metrics.get("real_false_positive_rate"),
                    "real_recall": metrics.get("real_recall"),
                    "prediction_distribution": dict(sorted(Counter(predictions).items())),
                }
            )

    recommended = choose_recommendation(sweeps)
    payload = {
        "candidate_id": args.candidate_id,
        "profile": args.profile,
        "sample_count": len(samples),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "profile_report": profile_report,
        "current_artifact_threshold": predictor["generated_gate_threshold"],
        "current_artifact_margin": predictor["real_protection_margin"],
        "prediction_cache_policy": "每张样本只抽特征并读取 binary gate 概率一次，随后离线扫描 threshold/margin。",
        "recommended": recommended,
        "sweeps": sweeps,
        "interpretation": interpret(recommended),
        "does_not_train": True,
        "does_not_activate": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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


def binary_label(label: object) -> str:
    return "real" if str(label) == "real" else "generated"


def cached_binary_gate_prediction(row: dict[str, float], predictor: dict[str, object]) -> dict[str, object]:
    feature_names = list(predictor["feature_names"])
    means = dict(predictor["means"])
    scales = dict(predictor["scales"])
    normalized = _normalize(row, means, scales)
    gate = _predict_generated_probability_with_gate(
        str(predictor["binary_gate_path"]),
        normalized,
        feature_names,
    )
    prediction = _predict_generator_label(
        row,
        feature_names,
        means,
        scales,
        list(predictor["prototypes"]),
        float(predictor["unknown_threshold"]),
        classifier_path=str(predictor["classifier_path"]),
        gpt_detector_path=str(predictor.get("gpt_detector_path", "")),
        binary_gate_path=str(predictor["binary_gate_path"]),
        generated_gate_threshold=0.99,
        gpt_detector_threshold=float(predictor.get("gpt_detector_threshold", 0.42)),
        real_protection_margin=0.0,
        open_set_min_margin=float(predictor.get("open_set_min_margin", 0.0)),
    )
    return {
        "raw_label": str(prediction.get("raw_label") or prediction.get("label") or "unknown"),
        "confidence": float(prediction.get("confidence", 0.0) or 0.0),
        "gate": gate,
    }


def swept_binary_label(base_prediction: dict[str, object], threshold: float, margin: float) -> str:
    gate = base_prediction.get("gate")
    raw_label = str(base_prediction.get("raw_label") or "unknown")
    confidence = float(base_prediction.get("confidence", 0.0) or 0.0)
    if not isinstance(gate, dict):
        return binary_label(raw_label)
    generated_probability = float(gate.get("generated_probability", 0.0) or 0.0)
    real_probability = float(gate.get("real_probability", 0.0) or 0.0)
    if (
        raw_label == "real"
        and generated_probability >= threshold + margin
        and generated_probability >= real_probability + margin
    ):
        return "generated"
    if raw_label == "real":
        return "real"
    if (
        raw_label == "gpt-image2"
        and threshold >= 0.6
        and confidence >= 0.70
        and generated_probability >= threshold - 0.12
    ):
        return "generated"
    if real_probability >= generated_probability + margin:
        return "real"
    if generated_probability < threshold:
        return "real"
    return binary_label(raw_label)


def choose_recommendation(sweeps: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row
        for row in sweeps
        if number(row.get("real_false_positive_rate")) <= 0.10
        and number(row.get("generated_recall")) >= 0.50
    ]
    pool = feasible or sweeps
    if not pool:
        return None
    return max(
        pool,
        key=lambda row: (
            number(row.get("real_false_positive_rate")) <= 0.10,
            number(row.get("generated_recall")),
            number(row.get("binary_macro_f1")),
            -number(row.get("real_false_positive_rate")),
        ),
    )


def interpret(recommended: dict[str, Any] | None) -> str:
    if recommended is None:
        return "没有可解释的阈值扫描结果。"
    fpr = number(recommended.get("real_false_positive_rate"))
    recall = number(recommended.get("generated_recall"))
    if fpr <= 0.10 and recall >= 0.75:
        return "阈值校准有希望同时满足低误报和生成召回，可优先做无重训阈值更新实验。"
    if fpr <= 0.10:
        return "阈值可以压低真实误报，但生成召回不足；需要补生成侧扰动样本或改特征。"
    return "单纯阈值扫描仍无法压低真实误报；下一步应补真实 hard-negative 和做特征/训练策略改动。"


def number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()
