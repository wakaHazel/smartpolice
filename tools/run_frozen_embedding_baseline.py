from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import FeatureCacheRecord, VisionTrainingRunRequest  # noqa: E402
from app.multimodal_training import (  # noqa: E402
    CLIP_EXTRACTOR_VERSION,
    CLIP_FEATURE_DIMS,
    CLIP_MODEL_NAME,
    GENERATOR_ATTRIBUTION_TASK,
    _binary_generation_metrics,
    _classification_metrics,
    _clip_features,
    extract_sample_features,
    _generator_experiment_view,
    _task_relevant_samples,
)
from app.storage import (  # noqa: E402
    get_feature_cache,
    get_vision_training_artifact_by_id,
    initialize_database,
    list_external_training_samples,
    save_feature_cache,
)

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
EMBEDDING_EXTRACTOR_VERSION = f"local-clip-image-embedding-v1:{CLIP_MODEL_NAME}"
_LOCAL_CLIP_BUNDLES: dict[bool, tuple[object, object]] = {}
_LOCAL_CLIP_LOAD_ERRORS: dict[bool, str] = {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen-vision-feature baseline for GPT-image2/social robustness diagnostics."
    )
    parser.add_argument("--profile", default="gpt_image2_ovr")
    parser.add_argument("--sample-limit", type=int, default=420)
    parser.add_argument("--source-sample-limit", type=int, default=900)
    parser.add_argument("--max-holdout-groups", type=int, default=8)
    parser.add_argument(
        "--feature-mode",
        choices=("embedding", "summary", "combined"),
        default="embedding",
        help="Use full frozen CLIP embeddings, the old low-dimensional summary, or both.",
    )
    parser.add_argument(
        "--include-forensic-features",
        action="store_true",
        help="Fuse frozen embedding with existing compression/texture/propagation forensic features.",
    )
    parser.add_argument(
        "--embedding-view",
        choices=("image", "image_text", "image_gap", "all"),
        default="image",
        help="Embedding feature view. Default image avoids text-caption leakage.",
    )
    parser.add_argument("--allow-extract", action="store_true", help="Run local CLIP extraction for missing cache rows.")
    parser.add_argument("--extract-limit", type=int, default=0, help="Maximum cache-missing samples to extract in this run.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download CLIP weights; use local HF cache only.")
    parser.add_argument("--prewarm-only", action="store_true", help="Only populate/read embedding cache; do not train.")
    parser.add_argument("--candidate-id", default="", help="Optional existing artifact to reuse cached feature_names style only.")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "audits"))
    parser.add_argument("--docs-path", default=str(ROOT / "docs" / "frozen_embedding_baseline.md"))
    args = parser.parse_args()

    initialize_database()
    load_limit = max(args.sample_limit, args.source_sample_limit)
    samples = _load_profile_samples(args.profile, load_limit)
    payload: dict[str, Any] = {
        "id": f"frozen-embedding-baseline-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "sample_limit": args.sample_limit,
        "source_sample_limit": args.source_sample_limit,
        "feature_mode": args.feature_mode,
        "embedding_view": args.embedding_view,
        "include_forensic_features": args.include_forensic_features,
        "allow_extract": args.allow_extract,
        "extract_limit": args.extract_limit,
        "local_files_only": args.local_files_only,
        "prewarm_only": args.prewarm_only,
        "extractor": {
            "name": "CLIP frozen image embedding",
            "version": CLIP_EXTRACTOR_VERSION,
            "embedding_version": EMBEDDING_EXTRACTOR_VERSION,
            "summary_feature_dims": CLIP_FEATURE_DIMS,
            "embedding_feature_policy": "默认使用完整 image embedding；text embedding 只在显式选择 image_text/image_gap/all 时进入实验。",
            "policy": "冻结视觉基础模型，只训练轻量 Logistic/校准头；默认优先读缓存，缺缓存不拖慢主训练。",
        },
        "candidate_artifact_available": bool(
            get_vision_training_artifact_by_id(GENERATOR_ATTRIBUTION_TASK, args.candidate_id)
        )
        if args.candidate_id
        else False,
    }
    result = run_embedding_experiment(
        samples=samples,
        profile=args.profile,
        allow_extract=args.allow_extract,
        extract_limit=args.extract_limit,
        local_files_only=args.local_files_only,
        prewarm_only=args.prewarm_only,
        clean_sample_limit=args.sample_limit,
        source_sample_limit=args.source_sample_limit,
        max_holdout_groups=args.max_holdout_groups,
        feature_mode=args.feature_mode,
        embedding_view=args.embedding_view,
        include_forensic_features=args.include_forensic_features,
    )
    payload.update(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "frozen_embedding_baseline_latest.json"
    stamped_path = output_dir / f"{payload['id']}.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stamped_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.docs_path).write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {latest_path}")
    print(f"wrote {args.docs_path}")


def _load_profile_samples(profile: str, limit: int) -> list[Any]:
    raw_samples = [
        sample
        for sample in list_external_training_samples(limit=50000, task_type=GENERATOR_ATTRIBUTION_TASK)
        if sample.image_available and sample.image_path and Path(sample.image_path).is_file()
    ]
    task_samples = _task_relevant_samples(raw_samples, GENERATOR_ATTRIBUTION_TASK)
    rows = [{} for _ in task_samples]
    request = VisionTrainingRunRequest(
        task_type=GENERATOR_ATTRIBUTION_TASK,
        min_samples=2,
        experiment_profile=profile,  # type: ignore[arg-type]
    )
    selected_samples, _, selected_labels, _ = _generator_experiment_view(task_samples, rows, request)
    if not limit or len(selected_samples) <= limit:
        return selected_samples
    buckets: dict[str, list[Any]] = {}
    for sample, label in zip(selected_samples, selected_labels, strict=False):
        buckets.setdefault(str(label), []).append(sample)
    chosen: list[Any] = []
    cursor = 0
    labels = sorted(buckets)
    while len(chosen) < limit and labels:
        progressed = False
        for label in labels:
            bucket = buckets[label]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return chosen


def run_embedding_experiment(
    *,
    samples: list[Any],
    profile: str,
    allow_extract: bool,
    extract_limit: int,
    local_files_only: bool,
    prewarm_only: bool,
    clean_sample_limit: int,
    source_sample_limit: int,
    max_holdout_groups: int,
    feature_mode: str,
    embedding_view: str,
    include_forensic_features: bool,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    cache_hits = 0
    cache_misses = 0
    skipped = 0
    extract_attempts = 0
    extract_failures = 0
    extracted_count = 0
    usable_samples: list[Any] = []
    for sample in samples:
        remaining_extracts = None
        if allow_extract and extract_limit > 0:
            remaining_extracts = max(0, extract_limit - extracted_count)
        row, hit = frozen_clip_row(
            sample,
            allow_extract=allow_extract,
            remaining_extracts=remaining_extracts,
            local_files_only=local_files_only,
            feature_mode=feature_mode,
            embedding_view=embedding_view,
            include_forensic_features=include_forensic_features,
        )
        if not row:
            skipped += 1
            if allow_extract and (remaining_extracts is None or remaining_extracts > 0):
                extract_attempts += 1
                extract_failures += 1
            continue
        usable_samples.append(sample)
        rows.append(row)
        if hit:
            cache_hits += 1
        else:
            cache_misses += 1
            extract_attempts += 1
            extracted_count += 1
    if prewarm_only:
        return {
            "sample_count": len(samples),
            "usable_count": len(usable_samples),
            "skipped_count": skipped,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "extract_attempts": extract_attempts,
            "extracted_count": extracted_count,
            "extract_failures": extract_failures,
            "clip_load_errors": _LOCAL_CLIP_LOAD_ERRORS,
            "status": "prewarmed",
            "reason": "已完成缓存预热/检查，未训练轻量分类头。",
        }
    request = VisionTrainingRunRequest(
        task_type=GENERATOR_ATTRIBUTION_TASK,
        min_samples=2,
        experiment_profile=profile,  # type: ignore[arg-type]
    )
    usable_samples, rows, labels, profile_report = _generator_experiment_view(usable_samples, rows, request)
    base = {
        "sample_count": len(samples),
        "usable_count": len(usable_samples),
        "skipped_count": skipped,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "extract_attempts": extract_attempts,
        "extracted_count": extracted_count,
        "extract_failures": extract_failures,
        "clip_load_errors": _LOCAL_CLIP_LOAD_ERRORS,
        "profile_report": profile_report,
    }
    if len(usable_samples) < 12 or len(set(labels)) < 2:
        return {
            **base,
            "status": "skipped",
            "reason": "冻结 embedding 缓存样本不足；可用 --allow-extract 先做小样本预热，或后续离线批量缓存。",
        }
    clean_rows, clean_labels = balanced_rows(rows, labels, clean_sample_limit)
    clean = train_eval_logistic(clean_rows, clean_labels)
    source_samples, source_rows, source_labels = balanced_samples_rows(usable_samples, rows, labels, source_sample_limit)
    source = source_holdout_logistic(
        samples=source_samples,
        rows=source_rows,
        labels=source_labels,
        max_holdout_groups=max_holdout_groups,
    )
    return {
        **base,
        "status": "completed",
        "feature_mode": feature_mode,
        "embedding_view": embedding_view,
        "include_forensic_features": include_forensic_features,
        "clean_sample_count": len(clean_rows),
        "source_sample_count": len(source_rows),
        "clean": clean,
        "source_holdout": source,
        "interpretation": [
            "该实验只训练轻量 LogisticRegression，视觉编码器保持冻结。",
            "若 source-holdout 明显好于手工特征，下一步应扩大 embedding 缓存并纳入 candidate 对照。",
            "若 clean 高而 source 低，说明视觉基础模型特征仍受来源/风格耦合影响，需要更多独立来源和 unknown 门控。",
        ],
    }


def frozen_clip_row(
    sample: Any,
    *,
    allow_extract: bool,
    remaining_extracts: int | None,
    local_files_only: bool,
    feature_mode: str,
    embedding_view: str,
    include_forensic_features: bool,
) -> tuple[dict[str, float], bool]:
    path = Path(sample.image_path or "")
    if not path.is_file():
        return {}, False
    digest = sample.image_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    text = f"{sample.title} {sample.content} {sample.scenario}"
    safe_text = text[:512]
    row: dict[str, float] = {}
    cache_hit = False
    if feature_mode in {"embedding", "combined"}:
        embedding_row, embedding_hit = frozen_clip_embedding_row(
            path=path,
            digest=digest,
            text=safe_text,
            allow_extract=allow_extract,
            remaining_extracts=remaining_extracts,
            local_files_only=local_files_only,
            embedding_view=embedding_view,
        )
        row.update(embedding_row)
        cache_hit = cache_hit or embedding_hit
    if feature_mode in {"summary", "combined"}:
        summary_row, summary_hit = frozen_clip_summary_row(
            path=path,
            digest=digest,
            text=safe_text,
            allow_extract=allow_extract,
        )
        row.update(summary_row)
        cache_hit = cache_hit or summary_hit
    if not row:
        return {}, False
    if include_forensic_features:
        row.update(forensic_feature_row(sample))
    return row, cache_hit


def frozen_clip_embedding_row(
    *,
    path: Path,
    digest: str,
    text: str,
    allow_extract: bool,
    remaining_extracts: int | None,
    local_files_only: bool,
    embedding_view: str,
) -> tuple[dict[str, float], bool]:
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    cache_key = f"{digest}:{text_digest}:{EMBEDDING_EXTRACTOR_VERSION}"
    cached = get_feature_cache(cache_key) or get_feature_cache(f"{digest}:{text_digest}:{CLIP_EXTRACTOR_VERSION}:embedding")
    if cached is not None:
        row = clip_embedding_features(cached.payload, embedding_view=embedding_view)
        return row, bool(row)
    if not allow_extract:
        return {}, False
    if remaining_extracts == 0:
        return {}, False
    pair = extract_clip_embedding_pair(path, digest, text, local_files_only=local_files_only)
    if pair is None:
        return {}, False
    image_values, text_values = pair
    payload = {"image_embedding": image_values, "text_embedding": text_values}
    save_feature_cache(
        FeatureCacheRecord(
            id=f"clip-emb-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:20]}",
            cache_key=cache_key,
            extractor_version=EMBEDDING_EXTRACTOR_VERSION,
            modality="image_text_embedding",
            sha256=digest,
            payload=payload,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return clip_embedding_features(payload, embedding_view=embedding_view), False


def extract_clip_embedding_pair(
    path: Path,
    digest: str,
    text: str,
    *,
    local_files_only: bool,
) -> tuple[list[float], list[float]] | None:
    bundle = local_clip_bundle(local_files_only=local_files_only)
    if bundle is None:
        return None
    model, processor = bundle
    try:
        import torch

        image = Image.open(path).convert("RGB")
        inputs = processor(text=[text[:512]], images=image, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_values = [round(float(value), 6) for value in image_features[0].detach().cpu().tolist()]
            text_values = [round(float(value), 6) for value in text_features[0].detach().cpu().tolist()]
    except Exception as exc:
        _LOCAL_CLIP_LOAD_ERRORS[local_files_only] = f"{type(exc).__name__}: {exc}"
        return None
    return image_values, text_values


def local_clip_bundle(*, local_files_only: bool) -> tuple[object, object] | None:
    if local_files_only in _LOCAL_CLIP_BUNDLES:
        return _LOCAL_CLIP_BUNDLES[local_files_only]
    if local_files_only in _LOCAL_CLIP_LOAD_ERRORS:
        return None
    try:
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=local_files_only)
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=local_files_only)
        model.eval()
        _LOCAL_CLIP_BUNDLES[local_files_only] = (model, processor)
    except Exception as exc:
        _LOCAL_CLIP_LOAD_ERRORS[local_files_only] = f"{type(exc).__name__}: {exc}"
        return None
    return _LOCAL_CLIP_BUNDLES[local_files_only]


def frozen_clip_summary_row(
    *,
    path: Path,
    digest: str,
    text: str,
    allow_extract: bool,
) -> tuple[dict[str, float], bool]:
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    cache_key = f"{digest}:{text_digest}:{CLIP_EXTRACTOR_VERSION}"
    cached = get_feature_cache(cache_key)
    if cached is not None:
        row = clip_image_only_features(cached.payload)
        return row, bool(row)
    if not allow_extract:
        return {}, False
    features = _clip_features(str(path), digest, text)
    row = clip_image_only_features(features)
    if row:
        save_feature_cache(
            FeatureCacheRecord(
                id=f"frozen-clip-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:20]}",
                cache_key=cache_key,
                extractor_version=CLIP_EXTRACTOR_VERSION,
                modality="frozen_clip_image",
                sha256=digest,
                payload=features,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
    return row, False


def clip_embedding_features(payload: dict[str, object], *, embedding_view: str) -> dict[str, float]:
    image_values = float_list(payload.get("image_embedding"))
    text_values = float_list(payload.get("text_embedding"))
    if not image_values:
        return {}
    row: dict[str, float] = {}
    if embedding_view in {"image", "image_text", "image_gap", "all"}:
        for index, value in enumerate(image_values):
            row[f"clip_emb_img_{index:03d}"] = value
    if embedding_view in {"image_text", "all"} and text_values:
        for index, value in enumerate(text_values):
            row[f"clip_emb_txt_{index:03d}"] = value
    if embedding_view in {"image_gap", "all"} and text_values:
        for index, (image_value, text_value) in enumerate(zip(image_values, text_values, strict=False)):
            row[f"clip_emb_gap_{index:03d}"] = abs(image_value - text_value)
    return row


def float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def clip_image_only_features(payload: dict[str, object]) -> dict[str, float]:
    row: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(value, int | float):
            continue
        name = str(key)
        if name.startswith("clip_img_") or name in {"clip_similarity", "clip_distance", "clip_abs_gap_mean"}:
            row[name] = float(value)
    return row


def forensic_feature_row(sample: Any) -> dict[str, float]:
    raw = extract_sample_features(sample, task_type=GENERATOR_ATTRIBUTION_TASK)
    blocked_prefixes = ("clip_",)
    return {
        f"forensic_{name}": float(value)
        for name, value in raw.items()
        if isinstance(value, int | float) and not str(name).startswith(blocked_prefixes)
    }


def balanced_rows(rows: list[dict[str, float]], labels: list[str], limit: int) -> tuple[list[dict[str, float]], list[str]]:
    indices = balanced_indices(labels, limit)
    return [rows[index] for index in indices], [labels[index] for index in indices]


def balanced_samples_rows(
    samples: list[Any],
    rows: list[dict[str, float]],
    labels: list[str],
    limit: int,
) -> tuple[list[Any], list[dict[str, float]], list[str]]:
    indices = balanced_indices(labels, limit)
    return [samples[index] for index in indices], [rows[index] for index in indices], [labels[index] for index in indices]


def balanced_indices(labels: list[str], limit: int) -> list[int]:
    if not limit or len(labels) <= limit:
        return list(range(len(labels)))
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(str(label), []).append(index)
    selected: list[int] = []
    cursor = 0
    ordered_labels = sorted(buckets)
    while len(selected) < limit and ordered_labels:
        progressed = False
        for label in ordered_labels:
            bucket = buckets[label]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected)


def train_eval_logistic(rows: list[dict[str, float]], labels: list[str]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_names = sorted({name for row in rows for name in row})
    matrix = [[row.get(name, 0.0) for name in feature_names] for row in rows]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    try:
        train_indices, valid_indices = next(splitter.split(matrix, labels))
    except ValueError:
        valid_indices = list(range(max(1, len(labels) // 4)))
        train_indices = [index for index in range(len(labels)) if index not in set(valid_indices)]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"),
    )
    model.fit([matrix[index] for index in train_indices], [labels[index] for index in train_indices])
    predictions = [str(item) for item in model.predict([matrix[index] for index in valid_indices])]
    valid_labels = [labels[index] for index in valid_indices]
    probabilities = model.predict_proba([matrix[index] for index in valid_indices])
    classes = [str(item) for item in model.classes_]
    payloads = probability_payloads(classes, probabilities)
    metrics = _classification_metrics(predictions, valid_labels)
    binary = _binary_generation_metrics(predictions, valid_labels)
    generated_scores = [generated_probability(payload) for payload in payloads]
    return {
        "feature_count": len(feature_names),
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "metrics": metrics,
        "binary": binary,
        "macro_ovr_auc": macro_auc(valid_labels, payloads),
        "gpt_image2_auc": auc_for_label(valid_labels, payloads, "gpt-image2"),
        "generated_auc": roc_auc(generated_scores, [label != "real" for label in valid_labels]),
        "threshold_scan": threshold_scan(generated_scores, valid_labels),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "prediction_distribution": dict(sorted(Counter(predictions).items())),
    }


def source_holdout_logistic(
    *,
    samples: list[Any],
    rows: list[dict[str, float]],
    labels: list[str],
    max_holdout_groups: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_names = sorted({name for row in rows for name in row})
    matrix = [[row.get(name, 0.0) for name in feature_names] for row in rows]
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(f"{sample.dataset_name}|{sample.source}", []).append(index)
    results: list[dict[str, Any]] = []
    all_binary_labels: list[str] = []
    all_generated_scores: list[float] = []
    all_binary_predictions_by_threshold: dict[str, list[str]] = {
        f"{threshold:.2f}": [] for threshold in DEFAULT_THRESHOLDS
    }
    for group, holdout_indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:max_holdout_groups]:
        train_indices = [index for index in range(len(labels)) if index not in set(holdout_indices)]
        train_labels = [labels[index] for index in train_indices]
        if len(set(train_labels)) < 2:
            results.append({"group": group, "skipped": True, "reason": "训练侧类别少于 2 个。"})
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"),
        )
        model.fit([matrix[index] for index in train_indices], train_labels)
        predictions = [str(item) for item in model.predict([matrix[index] for index in holdout_indices])]
        holdout_labels = [labels[index] for index in holdout_indices]
        probabilities = model.predict_proba([matrix[index] for index in holdout_indices])
        classes = [str(item) for item in model.classes_]
        payloads = probability_payloads(classes, probabilities)
        generated_scores = [generated_probability(payload) for payload in payloads]
        binary_labels = [binary_label(label) for label in holdout_labels]
        all_binary_labels.extend(binary_labels)
        all_generated_scores.extend(generated_scores)
        for threshold in DEFAULT_THRESHOLDS:
            key = f"{threshold:.2f}"
            all_binary_predictions_by_threshold[key].extend(
                ["generated" if score >= threshold else "real" for score in generated_scores]
            )
        metrics = _classification_metrics(predictions, holdout_labels)
        binary = _binary_generation_metrics(predictions, holdout_labels)
        results.append(
            {
                "group": group,
                "skipped": False,
                "holdout_count": len(holdout_indices),
                "holdout_label_distribution": dict(sorted(Counter(holdout_labels).items())),
                "prediction_distribution": dict(sorted(Counter(predictions).items())),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "binary_macro_f1": binary["macro_f1"],
                "generated_recall": binary["generated_recall"],
                "real_false_positive_rate": binary["real_false_positive_rate"],
                "generated_auc": roc_auc(generated_scores, [label != "real" for label in holdout_labels]),
                "gpt_image2_auc": auc_for_label(holdout_labels, payloads, "gpt-image2"),
                "mean_generated_probability": round(sum(generated_scores) / len(generated_scores), 3)
                if generated_scores
                else 0.0,
                "per_class": metrics["per_class"],
            }
        )
    completed = [item for item in results if not item.get("skipped")]
    threshold_rows = []
    for key, predictions in all_binary_predictions_by_threshold.items():
        if not predictions:
            continue
        metrics = _binary_generation_metrics(predictions, all_binary_labels)
        threshold_rows.append(
            {
                "threshold": float(key),
                "binary_macro_f1": metrics["macro_f1"],
                "generated_recall": metrics["generated_recall"],
                "real_false_positive_rate": metrics["real_false_positive_rate"],
            }
        )
    return {
        "groups": results,
        "completed_group_count": len(completed),
        "mean_macro_f1": round(sum(float(item["macro_f1"]) for item in completed) / len(completed), 3)
        if completed
        else 0.0,
        "mean_binary_macro_f1": round(
            sum(float(item["binary_macro_f1"]) for item in completed) / len(completed),
            3,
        )
        if completed
        else 0.0,
        "mean_generated_recall": round(
            sum(float(item["generated_recall"]) for item in completed) / len(completed),
            3,
        )
        if completed
        else 0.0,
        "overall_real_false_positive_rate": round(
            sum(float(item["real_false_positive_rate"]) for item in completed) / len(completed),
            3,
        )
        if completed
        else 0.0,
        "generated_auc": roc_auc(all_generated_scores, [label == "generated" for label in all_binary_labels]),
        "threshold_scan": threshold_rows,
        "recommended_threshold": recommended_threshold(threshold_rows),
    }


def macro_auc(labels: list[str], payloads: list[dict[str, object]]) -> float | None:
    aucs = [auc_for_label(labels, payloads, label) for label in sorted(set(labels))]
    valid = [value for value in aucs if value is not None]
    return round(sum(valid) / len(valid), 3) if valid else None


def probability_payloads(classes: list[str], probabilities: Any) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for row in probabilities:
        payloads.append(
            {
                "candidates": [
                    {"label": label, "confidence": float(probability)}
                    for label, probability in zip(classes, row, strict=False)
                ]
            }
        )
    return payloads


def generated_probability(prediction: dict[str, object]) -> float:
    real_score = score_for_label(prediction, "real")
    if real_score > 0:
        return max(0.0, min(1.0, 1.0 - real_score))
    candidates = prediction.get("candidates")
    if not isinstance(candidates, list):
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            sum(
                float(candidate.get("confidence", 0.0))
                for candidate in candidates
                if isinstance(candidate, dict)
                and candidate.get("label") != "real"
                and isinstance(candidate.get("confidence"), int | float)
            ),
        ),
    )


def binary_label(label: object) -> str:
    return "real" if str(label) == "real" else "generated"


def threshold_scan(scores: list[float], labels: list[str]) -> list[dict[str, float]]:
    binary_labels = [binary_label(label) for label in labels]
    rows: list[dict[str, float]] = []
    for threshold in DEFAULT_THRESHOLDS:
        predictions = ["generated" if score >= threshold else "real" for score in scores]
        metrics = _binary_generation_metrics(predictions, binary_labels)
        rows.append(
            {
                "threshold": threshold,
                "binary_macro_f1": metrics["macro_f1"],
                "generated_recall": metrics["generated_recall"],
                "real_false_positive_rate": metrics["real_false_positive_rate"],
            }
        )
    return rows


def recommended_threshold(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    conservative = [
        row
        for row in rows
        if row["real_false_positive_rate"] <= 0.10 and row["generated_recall"] >= 0.50
    ]
    candidates = conservative or rows
    return max(
        candidates,
        key=lambda row: (
            row["binary_macro_f1"],
            -row["real_false_positive_rate"],
            row["generated_recall"],
        ),
    )


def auc_for_label(labels: list[str], payloads: list[dict[str, object]], label: str) -> float | None:
    scores = [score_for_label(payload, label) for payload in payloads]
    return roc_auc(scores, [item == label for item in labels])


def score_for_label(prediction: dict[str, object], label: str) -> float:
    candidates = prediction.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if (
                isinstance(candidate, dict)
                and candidate.get("label") == label
                and isinstance(candidate.get("confidence"), int | float)
            ):
                return float(candidate["confidence"])
    return 0.0


def roc_auc(scores: list[float], positives: list[bool]) -> float | None:
    positive_scores = [score for score, is_positive in zip(scores, positives, strict=False) if is_positive]
    negative_scores = [score for score, is_positive in zip(scores, positives, strict=False) if not is_positive]
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


def render_markdown(payload: dict[str, Any]) -> str:
    clean = payload.get("clean", {}) if isinstance(payload.get("clean"), dict) else {}
    source = payload.get("source_holdout", {}) if isinstance(payload.get("source_holdout"), dict) else {}
    clean_metrics = clean.get("metrics", {}) if isinstance(clean.get("metrics"), dict) else {}
    binary = clean.get("binary", {}) if isinstance(clean.get("binary"), dict) else {}
    recommended = source.get("recommended_threshold") if isinstance(source.get("recommended_threshold"), dict) else {}
    lines = [
        "# 冻结视觉基础模型特征基线",
        "",
        f"- 生成时间: `{payload.get('created_at')}`",
        f"- Profile: `{payload.get('profile')}`",
        f"- 状态: `{payload.get('status')}`",
        f"- Feature mode / view: `{payload.get('feature_mode')}` / `{payload.get('embedding_view')}`",
        f"- 融合取证特征: `{payload.get('include_forensic_features', False)}`",
        f"- 可用样本: `{payload.get('usable_count')}` / `{payload.get('sample_count')}`",
        f"- 缓存命中/新抽取/失败: `{payload.get('cache_hits')}` / `{payload.get('extracted_count', payload.get('cache_misses'))}` / `{payload.get('extract_failures', 0)}`",
        f"- 特征版本: `{payload.get('extractor', {}).get('version')}`",
        "",
        "| 指标 | Clean/随机留出 | Source-holdout |",
        "| --- | ---: | ---: |",
        f"| Macro-F1 | {fmt(clean_metrics.get('macro_f1'))} | {fmt(source.get('mean_macro_f1'))} |",
        f"| Binary Macro-F1 | {fmt(binary.get('macro_f1'))} | {fmt(source.get('mean_binary_macro_f1'))} |",
        f"| Generated Recall | {fmt(binary.get('generated_recall'))} | {fmt(source.get('mean_generated_recall'))} |",
        f"| Real FPR | {fmt(binary.get('real_false_positive_rate'))} | {fmt(source.get('overall_real_false_positive_rate'))} |",
        f"| Generated AUC | {fmt(clean.get('generated_auc'))} | {fmt(source.get('generated_auc'))} |",
        f"| Macro OvR AUC | {fmt(clean.get('macro_ovr_auc'))} | - |",
        f"| GPT-image2 AUC | {fmt(clean.get('gpt_image2_auc'))} | - |",
        "",
        "## Source-holdout 弱组",
        "",
        "| 来源组 | 标签分布 | 预测分布 | Macro-F1 | Binary-F1 | Generated Recall | Real FPR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    groups = source.get("groups") if isinstance(source.get("groups"), list) else []
    for group in groups[:8]:
        if not isinstance(group, dict) or group.get("skipped"):
            continue
        lines.append(
            "| {group} | `{labels}` | `{predictions}` | {macro} | {binary_f1} | {recall} | {fpr} |".format(
                group=group.get("group"),
                labels=json.dumps(group.get("holdout_label_distribution", {}), ensure_ascii=False),
                predictions=json.dumps(group.get("prediction_distribution", {}), ensure_ascii=False),
                macro=fmt(group.get("macro_f1")),
                binary_f1=fmt(group.get("binary_macro_f1")),
                recall=fmt(group.get("generated_recall")),
                fpr=fmt(group.get("real_false_positive_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## 阈值诊断",
            "",
            f"- 推荐阈值: `{fmt(recommended.get('threshold'))}`；Binary-F1 `{fmt(recommended.get('binary_macro_f1'))}`；Generated Recall `{fmt(recommended.get('generated_recall'))}`；Real FPR `{fmt(recommended.get('real_false_positive_rate'))}`。",
            "",
            "| Threshold | Binary-F1 | Generated Recall | Real FPR |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    threshold_rows = source.get("threshold_scan") if isinstance(source.get("threshold_scan"), list) else []
    for row in threshold_rows:
        if isinstance(row, dict):
            lines.append(
                f"| {fmt(row.get('threshold'))} | {fmt(row.get('binary_macro_f1'))} | {fmt(row.get('generated_recall'))} | {fmt(row.get('real_false_positive_rate'))} |"
            )
    lines.extend(
        [
            "",
        "## 解释",
        "",
        "- 这是冻结视觉基础模型路线的基线，不训练 CLIP/ViT 本体，只训练轻量分类头。",
        "- 当前默认优先使用缓存；若缓存不足，结果会标记 skipped，避免拖慢主训练。",
        "- 该结果用于和现有手工取证特征做对照，不能单独替代 source-holdout 和扰动评测。",
        ]
    )
    if payload.get("reason"):
        lines.extend(["", f"- 跳过原因: {payload.get('reason')}"])
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
