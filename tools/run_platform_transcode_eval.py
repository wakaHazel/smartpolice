from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
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
    _predict_generator_label,
    get_vision_competition_summary,
)
from app.storage import (  # noqa: E402
    get_active_vision_training_artifact,
    get_vision_training_artifact_by_id,
    initialize_database,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MANIFEST = ROOT / "platform_eval" / "upload_batch_60" / "upload_manifest.csv"
DEFAULT_RETURNED_ROOT = ROOT / "platform_eval" / "returned"
DEFAULT_COLLECTION_MANIFEST = DEFAULT_RETURNED_ROOT / "collection_manifest.csv"
DEFAULT_OUTPUT = ROOT / "output" / "audits" / "platform_transcode_eval_latest.json"
DEFAULT_MARKDOWN = ROOT / "output" / "audits" / "platform_transcode_eval_latest.md"


@dataclass(frozen=True)
class ManifestItem:
    pair_id: str
    label: str
    clean_path: Path
    original_sha256: str
    dataset_name: str
    source: str
    source_url: str
    width: str
    height: str


@dataclass(frozen=True)
class EvalItem:
    pair_id: str
    label: str
    condition: str
    platform: str
    variant: str
    path: Path
    dataset_name: str
    source: str
    clean_sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clean vs real Weibo/Xiaohongshu returned images for GPT-image2 detection."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--returned-root", default=str(DEFAULT_RETURNED_ROOT))
    parser.add_argument("--collection-manifest", default=str(DEFAULT_COLLECTION_MANIFEST))
    parser.add_argument("--model-id", default="", help="Candidate/run id. Defaults to current active model.")
    parser.add_argument("--platforms", default="weibo,xhs")
    parser.add_argument("--variants", default="download,screenshot")
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-sample-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    initialize_database()
    manifest_items = load_manifest(Path(args.manifest))
    platforms = parse_csv_arg(args.platforms)
    variants = parse_csv_arg(args.variants)
    returned_root = Path(args.returned_root)
    collection_rows, returned_items = collect_returned_items(
        manifest_items,
        returned_root,
        platforms,
        variants,
    )
    write_collection_manifest(Path(args.collection_manifest), collection_rows)
    eval_items: list[EvalItem] = []
    if args.include_clean:
        eval_items.extend(clean_eval_items(manifest_items))
    eval_items.extend(returned_items)

    artifact, model_scope = load_artifact(args.model_id)
    predictor = _generator_predictor_from_artifact(artifact)
    predictions = evaluate_items(eval_items, predictor)
    payload = build_payload(
        manifest_items=manifest_items,
        collection_rows=collection_rows,
        predictions=predictions,
        artifact=artifact,
        model_scope=model_scope,
        returned_root=returned_root,
        platforms=platforms,
        variants=variants,
        include_sample_predictions=args.include_sample_predictions,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {Path(args.collection_manifest)}")


def parse_csv_arg(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("CSV argument must contain at least one value")
    return list(dict.fromkeys(values))


def load_manifest(path: Path) -> list[ManifestItem]:
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    items: list[ManifestItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            pair_id = str(row.get("pair_id") or "").strip()
            clean_path = Path(str(row.get("upload_path") or ""))
            if not pair_id or not clean_path.is_file():
                continue
            items.append(
                ManifestItem(
                    pair_id=pair_id,
                    label=normalize_label(str(row.get("label") or "unknown")),
                    clean_path=clean_path,
                    original_sha256=str(row.get("original_sha256") or ""),
                    dataset_name=str(row.get("dataset_name") or ""),
                    source=str(row.get("source") or ""),
                    source_url=str(row.get("source_url") or ""),
                    width=str(row.get("width") or ""),
                    height=str(row.get("height") or ""),
                )
            )
    if not items:
        raise SystemExit(f"no usable rows found in manifest: {path}")
    return sorted(items, key=lambda item: item.pair_id)


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    if normalized in {"gpt-image-2", "gptimage2", "gpt image 2"}:
        return "gpt-image2"
    if normalized in {"real", "authentic", "camera", "photo"}:
        return "real"
    return normalized or "unknown"


def collect_returned_items(
    manifest_items: list[ManifestItem],
    returned_root: Path,
    platforms: list[str],
    variants: list[str],
) -> tuple[list[dict[str, object]], list[EvalItem]]:
    by_pair = {item.pair_id: item for item in manifest_items}
    collection_rows: list[dict[str, object]] = []
    eval_items: list[EvalItem] = []
    for platform in platforms:
        for variant in variants:
            folder = returned_root / f"{platform}_{variant}"
            matches = index_returned_files(folder, by_pair.keys())
            for item in manifest_items:
                files = matches.get(item.pair_id, [])
                chosen = choose_returned_file(files)
                suggested = f"{item.pair_id}_{platform}_{variant}{item.clean_path.suffix.lower()}"
                row = {
                    "pair_id": item.pair_id,
                    "label": item.label,
                    "platform": platform,
                    "variant": variant,
                    "condition": condition_name(platform, variant),
                    "expected_folder": str(folder),
                    "suggested_filename": suggested,
                    "matched_path": str(chosen) if chosen else "",
                    "match_count": len(files),
                    "status": "matched" if chosen else "missing",
                }
                collection_rows.append(row)
                if chosen is None:
                    continue
                eval_items.append(
                    EvalItem(
                        pair_id=item.pair_id,
                        label=item.label,
                        condition=condition_name(platform, variant),
                        platform=platform,
                        variant=variant,
                        path=chosen,
                        dataset_name=item.dataset_name,
                        source=item.source,
                        clean_sha256=item.original_sha256,
                    )
                )
    return collection_rows, eval_items


def index_returned_files(folder: Path, pair_ids: Any) -> dict[str, list[Path]]:
    pairs = [str(pair_id).lower() for pair_id in pair_ids]
    original_by_lower = {str(pair_id).lower(): str(pair_id) for pair_id in pair_ids}
    indexed: dict[str, list[Path]] = defaultdict(list)
    if not folder.is_dir():
        return indexed
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    for path in files:
        lower_name = path.name.lower()
        for pair_lower in pairs:
            if pair_lower in lower_name:
                indexed[original_by_lower[pair_lower]].append(path)
                break
    return indexed


def choose_returned_file(files: list[Path]) -> Path | None:
    existing = [path for path in files if path.is_file()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: (-path.stat().st_size, path.name.lower()))[0]


def condition_name(platform: str, variant: str) -> str:
    return f"{platform}_{variant}"


def write_collection_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "label",
        "platform",
        "variant",
        "condition",
        "expected_folder",
        "suggested_filename",
        "matched_path",
        "match_count",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_eval_items(manifest_items: list[ManifestItem]) -> list[EvalItem]:
    return [
        EvalItem(
            pair_id=item.pair_id,
            label=item.label,
            condition="clean",
            platform="none",
            variant="clean",
            path=item.clean_path,
            dataset_name=item.dataset_name,
            source=item.source,
            clean_sha256=item.original_sha256,
        )
        for item in manifest_items
    ]


def load_artifact(model_id: str) -> tuple[dict[str, object], dict[str, object]]:
    active_id = get_vision_competition_summary(GENERATOR_ATTRIBUTION_TASK).active_model_id
    if model_id:
        artifact = get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, model_id)
        if artifact is None:
            raise SystemExit(f"vision_generator_attribution model not found: {model_id}")
        return artifact, {"requested_model_id": model_id, "active_model_id": active_id, "uses_active": model_id == active_id}
    artifact = get_active_vision_training_artifact(GENERATOR_ATTRIBUTION_TASK)
    if artifact is None:
        raise SystemExit("no active vision_generator_attribution artifact found")
    return artifact, {"requested_model_id": "", "active_model_id": active_id, "uses_active": True}


def evaluate_items(items: list[EvalItem], predictor: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in items:
        sha256 = file_sha256(item.path)
        features = _generator_attribution_features(
            str(item.path),
            sha256,
            f"metadata unavailable platform transcode condition {item.condition}",
        )
        prediction = _predict_generator_label(
            features,
            list(predictor["feature_names"]),
            dict(predictor["means"]),
            dict(predictor["scales"]),
            list(predictor["prototypes"]),
            float(predictor["unknown_threshold"]),
            classifier_path=str(predictor["classifier_path"]),
            gpt_detector_path=str(predictor.get("gpt_detector_path", "")),
            binary_gate_path=str(predictor["binary_gate_path"]),
            generated_gate_threshold=float(predictor["generated_gate_threshold"]),
            gpt_detector_threshold=float(predictor.get("gpt_detector_threshold", 0.42)),
            real_protection_margin=float(predictor["real_protection_margin"]),
            open_set_min_margin=float(predictor.get("open_set_min_margin", 0.0)),
        )
        predicted_label = str(prediction.get("label") or "unknown")
        binary_gate = prediction.get("binary_gate") if isinstance(prediction.get("binary_gate"), dict) else {}
        rows.append(
            {
                "pair_id": item.pair_id,
                "label": item.label,
                "expected_binary": binary_label(item.label),
                "condition": item.condition,
                "platform": item.platform,
                "variant": item.variant,
                "path": str(item.path),
                "sha256": sha256,
                "clean_sha256": item.clean_sha256,
                "dataset_name": item.dataset_name,
                "source": item.source,
                "prediction": predicted_label,
                "predicted_binary": binary_label(predicted_label),
                "raw_label": str(prediction.get("raw_label") or predicted_label),
                "confidence": round(float(prediction.get("confidence", 0.0) or 0.0), 3),
                "generated_probability": number(binary_gate.get("generated_probability")) if isinstance(binary_gate, dict) else None,
                "real_probability": number(binary_gate.get("real_probability")) if isinstance(binary_gate, dict) else None,
                "gpt_image2_score": score_for_label(prediction, "gpt-image2"),
                "candidate_ranking": compact_candidates(prediction.get("candidates")),
                "gate_reason": str(prediction.get("gate_reason") or ""),
            }
        )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_label(label: str) -> str:
    return "real" if label == "real" else "generated"


def number(value: object) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def score_for_label(prediction: dict[str, object], label: str) -> float:
    if label == "generated":
        gate = prediction.get("binary_gate")
        if isinstance(gate, dict) and isinstance(gate.get("generated_probability"), int | float):
            return round(float(gate["generated_probability"]), 3)
    if label == "real":
        gate = prediction.get("binary_gate")
        if isinstance(gate, dict) and isinstance(gate.get("real_probability"), int | float):
            return round(float(gate["real_probability"]), 3)
    detector = prediction.get("gpt_image2_detector")
    if label == "gpt-image2" and isinstance(detector, dict):
        value = detector.get("gpt_image2_probability")
        if isinstance(value, int | float):
            return round(float(value), 3)
    candidates = prediction.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("label") == label and isinstance(candidate.get("confidence"), int | float):
                return round(float(candidate["confidence"]), 3)
    raw_label = str(prediction.get("raw_label") or prediction.get("label") or "unknown")
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    return round(confidence if raw_label == label else 0.0, 3)


def compact_candidates(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    compacted: list[dict[str, object]] = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "label": str(item.get("label") or "unknown"),
                "confidence": number(item.get("confidence")),
            }
        )
    return compacted


def build_payload(
    *,
    manifest_items: list[ManifestItem],
    collection_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    artifact: dict[str, object],
    model_scope: dict[str, object],
    returned_root: Path,
    platforms: list[str],
    variants: list[str],
    include_sample_predictions: bool,
) -> dict[str, object]:
    condition_summaries = summarize_conditions(predictions)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": str(artifact.get("id") or model_scope.get("active_model_id") or ""),
        "active_model_id": model_scope.get("active_model_id"),
        "uses_active": model_scope.get("uses_active"),
        "manifest_sample_count": len(manifest_items),
        "expected_returned_count": len(collection_rows),
        "matched_returned_count": sum(1 for row in collection_rows if row.get("status") == "matched"),
        "missing_returned_count": sum(1 for row in collection_rows if row.get("status") == "missing"),
        "condition_count": len(condition_summaries),
        "returned_root": str(returned_root),
    }
    return {
        "summary": summary,
        "task_type": GENERATOR_ATTRIBUTION_TASK,
        "model_scope": model_scope,
        "model_card_hint": {
            "id": artifact.get("id"),
            "model_kind": artifact.get("model_kind"),
            "unknown_threshold": artifact.get("unknown_threshold"),
            "generated_gate_threshold": artifact.get("generated_gate_threshold"),
            "real_protection_margin": artifact.get("real_protection_margin"),
        },
        "platforms": platforms,
        "variants": variants,
        "collection_status": collection_status(collection_rows),
        "condition_summaries": condition_summaries,
        "deltas_from_clean": deltas_from_clean(condition_summaries),
        "sample_predictions": predictions if include_sample_predictions else [],
        "collection_rows": collection_rows,
        "interpretation": interpretation(condition_summaries, collection_rows),
    }


def summarize_conditions(predictions: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        by_condition[str(row["condition"])].append(row)
    summaries: dict[str, dict[str, object]] = {}
    for condition, rows in sorted(by_condition.items()):
        labels = [str(row["label"]) for row in rows]
        predicted = [str(row["prediction"]) for row in rows]
        binary_metrics = _binary_generation_metrics(predicted, labels)
        class_metrics = _classification_metrics(predicted, labels)
        gpt = gpt_image2_metrics(rows)
        summaries[condition] = {
            "sample_count": len(rows),
            "label_distribution": dict(sorted(Counter(labels).items())),
            "prediction_distribution": dict(sorted(Counter(predicted).items())),
            "binary_macro_f1": binary_metrics.get("macro_f1"),
            "binary_accuracy": binary_metrics.get("accuracy"),
            "generated_precision": binary_metrics.get("generated_precision"),
            "generated_recall": binary_metrics.get("generated_recall"),
            "generated_f1": binary_metrics.get("generated_f1"),
            "real_false_positive_rate": binary_metrics.get("real_false_positive_rate"),
            "real_recall": binary_metrics.get("real_recall"),
            "strict_macro_f1": class_metrics.get("macro_f1"),
            "strict_accuracy": class_metrics.get("accuracy"),
            "gpt_image2_precision": gpt["precision"],
            "gpt_image2_recall": gpt["recall"],
            "gpt_image2_f1": gpt["f1"],
            "gpt_image2_support": gpt["support"],
            "binary_auc": roc_auc(
                [float(row.get("generated_probability") or 0.0) for row in rows],
                [row.get("expected_binary") == "generated" for row in rows],
            ),
            "gpt_image2_auc": roc_auc(
                [float(row.get("gpt_image2_score") or 0.0) for row in rows],
                [row.get("label") == "gpt-image2" for row in rows],
            ),
            "average_confidence": average([float(row.get("confidence") or 0.0) for row in rows]),
            "average_generated_probability": average(
                [float(row.get("generated_probability") or 0.0) for row in rows if row.get("generated_probability") is not None]
            ),
            "average_gpt_image2_score": average([float(row.get("gpt_image2_score") or 0.0) for row in rows]),
            "source_breakdown": source_breakdown(rows),
        }
    return summaries


def gpt_image2_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    true_positive = sum(1 for row in rows if row.get("label") == "gpt-image2" and row.get("prediction") == "gpt-image2")
    false_positive = sum(1 for row in rows if row.get("label") != "gpt-image2" and row.get("prediction") == "gpt-image2")
    false_negative = sum(1 for row in rows if row.get("label") == "gpt-image2" and row.get("prediction") != "gpt-image2")
    support = sum(1 for row in rows if row.get("label") == "gpt-image2")
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "support": float(support),
    }


def roc_auc(scores: list[float], positives: list[bool]) -> float | None:
    positive_scores = [score for score, positive in zip(scores, positives, strict=False) if positive]
    negative_scores = [score for score, positive in zip(scores, positives, strict=False) if not positive]
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for positive_score in positive_scores:
        for negative_score in negative_scores:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return round(wins / (len(positive_scores) * len(negative_scores)), 3)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def source_breakdown(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("source") or row.get("dataset_name") or "unknown")
        by_source[key].append(row)
    breakdown: list[dict[str, object]] = []
    for source, source_rows in sorted(by_source.items(), key=lambda item: (-len(item[1]), item[0]))[:12]:
        labels = [str(row["label"]) for row in source_rows]
        predicted = [str(row["prediction"]) for row in source_rows]
        metrics = _binary_generation_metrics(predicted, labels)
        breakdown.append(
            {
                "source": source,
                "sample_count": len(source_rows),
                "label_distribution": dict(sorted(Counter(labels).items())),
                "generated_recall": metrics.get("generated_recall"),
                "real_false_positive_rate": metrics.get("real_false_positive_rate"),
                "gpt_image2_recall": gpt_image2_metrics(source_rows)["recall"],
            }
        )
    return breakdown


def collection_status(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["platform"]), str(row["variant"]))].append(row)
    status: list[dict[str, object]] = []
    for (platform, variant), group_rows in sorted(grouped.items()):
        status.append(
            {
                "platform": platform,
                "variant": variant,
                "condition": condition_name(platform, variant),
                "expected": len(group_rows),
                "matched": sum(1 for row in group_rows if row.get("status") == "matched"),
                "missing": sum(1 for row in group_rows if row.get("status") == "missing"),
                "duplicate_pairs": sum(1 for row in group_rows if int(row.get("match_count") or 0) > 1),
            }
        )
    return status


def deltas_from_clean(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, float | None]]:
    clean = summaries.get("clean")
    if not clean:
        return {}
    deltas: dict[str, dict[str, float | None]] = {}
    for condition, row in summaries.items():
        if condition == "clean":
            continue
        deltas[condition] = {
            "binary_macro_f1_delta": subtract(row.get("binary_macro_f1"), clean.get("binary_macro_f1")),
            "gpt_image2_recall_delta": subtract(row.get("gpt_image2_recall"), clean.get("gpt_image2_recall")),
            "real_fpr_delta": subtract(row.get("real_false_positive_rate"), clean.get("real_false_positive_rate")),
            "average_confidence_delta": subtract(row.get("average_confidence"), clean.get("average_confidence")),
        }
    return deltas


def subtract(value: object, baseline: object) -> float | None:
    if not isinstance(value, int | float) or not isinstance(baseline, int | float):
        return None
    return round(float(value) - float(baseline), 3)


def interpretation(summaries: dict[str, dict[str, object]], collection_rows: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    missing = sum(1 for row in collection_rows if row.get("status") == "missing")
    matched = sum(1 for row in collection_rows if row.get("status") == "matched")
    if matched == 0:
        notes.append("平台返回样本尚未回收；当前报告只给出 clean 基线和待回收清单。")
    elif missing:
        status = collection_status(collection_rows)
        missing_conditions = [
            f"{row['condition']}={row['missing']}"
            for row in status
            if int(row.get("missing") or 0) > 0
        ]
        notes.append(
            f"平台返回样本已回收 {matched} 个文件；缺失条件：{', '.join(missing_conditions)}。"
        )
        notes.append(
            "缺失条件不自动阻塞已匹配条件的黑盒评测；例如 xhs_screenshot 目前没有可靠全量渲染截图入口，"
            "保持 unavailable，不用本地下载图伪造成截图。"
        )
    else:
        notes.append("平台返回样本已全部匹配，可将微博/小红书转码作为真实传播扰动条件报告。")
    clean = summaries.get("clean")
    if clean:
        notes.append(
            "clean 行是上传前原图基线，用于计算平台转码后的 recall/FPR/置信度下降；不要把它写成平台鲁棒性结果。"
        )
    notes.append("本工具只评测，不训练、不激活模型；模型输出仍是辅助线索，需要原始文件、平台链路和人工复核共同支撑。")
    return notes


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Platform Transcode Evaluation",
        "",
        f"- Generated at: `{summary.get('created_at')}`",
        f"- Task: `{payload.get('task_type')}`",
        f"- Model id: `{summary.get('model_id')}`",
        f"- Active id: `{summary.get('active_model_id')}`",
        f"- Uses active: `{summary.get('uses_active')}`",
        f"- Returned root: `{summary.get('returned_root')}`",
        "",
        "## Collection Status",
        "",
        "| Condition | Expected | Matched | Missing | Duplicate pairs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("collection_status", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('condition')} | {row.get('expected')} | {row.get('matched')} | "
            f"{row.get('missing')} | {row.get('duplicate_pairs')} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Condition | N | Binary Macro-F1 | Gen Recall | GPT-image2 Recall | Real FPR | Binary AUC | GPT AUC | Avg Conf |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    condition_summaries = payload.get("condition_summaries", {})
    if isinstance(condition_summaries, dict):
        for condition, row in condition_summaries.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {condition} | {row.get('sample_count')} | {fmt(row.get('binary_macro_f1'))} | "
                f"{fmt(row.get('generated_recall'))} | {fmt(row.get('gpt_image2_recall'))} | "
                f"{fmt(row.get('real_false_positive_rate'))} | {fmt(row.get('binary_auc'))} | "
                f"{fmt(row.get('gpt_image2_auc'))} | {fmt(row.get('average_confidence'))} |"
            )
    lines.extend(["", "## Deltas From Clean", ""])
    lines.extend(
        [
            "| Condition | Binary Macro-F1 Delta | GPT Recall Delta | Real FPR Delta | Confidence Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    deltas = payload.get("deltas_from_clean", {})
    if isinstance(deltas, dict):
        for condition, row in deltas.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {condition} | {fmt(row.get('binary_macro_f1_delta'))} | "
                f"{fmt(row.get('gpt_image2_recall_delta'))} | {fmt(row.get('real_fpr_delta'))} | "
                f"{fmt(row.get('average_confidence_delta'))} |"
            )
    lines.extend(["", "## Notes", ""])
    for note in payload.get("interpretation", []):
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "```powershell",
            "python tools\\run_platform_transcode_eval.py",
            "```",
            "",
            "Returned files are matched by `pair_id` appearing anywhere in the filename.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
