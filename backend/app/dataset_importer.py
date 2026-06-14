from __future__ import annotations

from datetime import UTC, datetime
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from app.models import (
    ExternalDatasetImportRequest,
    ExternalDatasetImportResult,
    ExternalTrainingSample,
)
from app.storage import (
    count_external_training_samples,
    filter_new_external_training_samples,
    get_external_training_sample_count,
    save_external_training_samples,
)


SUPPORTED_FILE_FORMATS = {".csv", ".json", ".jsonl", ".ndjson"}
SUPPORTED_TASK_TYPES = {
    "text_risk",
    "vision_aigc",
    "vision_tamper",
    "vision_context_mismatch",
    "vision_generator_attribution",
    "multimodal_fusion",
}
IMAGE_REQUIRED_TASKS = {
    "vision_aigc",
    "vision_tamper",
    "vision_context_mismatch",
    "vision_generator_attribution",
}
GENERATOR_ATTRIBUTION_TASK = "vision_generator_attribution"
GENERATED_SOURCE_HINTS = {
    "gpt-image1",
    "gpt_image1",
    "gpt image1",
    "gpt-image2",
    "gpt_image2",
    "gpt image2",
    "gpt-image-1",
    "gpt image 1",
    "gpt-image-2",
    "gpt image 2",
    "gpt-image",
    "gpt image",
    "openai image",
    "openai",
    "midjourney",
    "mj",
    "sd21",
    "sd2.1",
    "stable diffusion 2.1",
    "sd3",
    "stable diffusion 3",
    "sdxl",
    "stable diffusion",
    "stable-diffusion",
    "sd",
    "flux",
    "dall-e",
    "dall-e-3",
    "dalle3",
    "dalle",
    "dall·e",
    "nano",
    "seedream",
    "imagegbt",
}
REAL_SOURCE_HINTS = {"real", "authentic", "photo", "camera", "真实", "照片", "实拍"}


def import_external_dataset(
    payload: ExternalDatasetImportRequest,
) -> ExternalDatasetImportResult:
    _validate_payload(payload)
    rows = _load_rows(payload)
    imported: list[ExternalTrainingSample] = []
    skipped = 0
    created_at = datetime.now(UTC).isoformat()

    for row in rows[: payload.limit]:
        sample = _sample_from_row(row, payload, created_at)
        if sample is None:
            skipped += 1
            continue
        imported.append(sample)

    new_samples = filter_new_external_training_samples(imported)
    inserted_count = save_external_training_samples(new_samples)
    skipped += len(imported) - inserted_count
    return ExternalDatasetImportResult(
        dataset_name=payload.dataset_name.strip(),
        source=payload.source.strip() or "HuggingFace/local",
        task_type=payload.task_type,
        imported_count=inserted_count,
        skipped_count=skipped,
        sample_count_after_import=(
            count_external_training_samples(payload.task_type)
            if payload.task_type
            else get_external_training_sample_count()
        ),
        image_available_count=sum(1 for sample in new_samples if sample.image_available),
        label_distribution=_label_distribution(new_samples),
        examples=new_samples[:5],
        note=(
            "外部数据集已进入对应任务训练池；内置四方向样例仍只用于展示/评测。"
            if imported
            else "没有可导入样本，请检查文本列、标签列、图片路径和文件格式。"
        ),
    )


def _validate_payload(payload: ExternalDatasetImportRequest) -> None:
    task_type = payload.task_type.strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        supported = "、".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"不支持的 task_type：{payload.task_type}；支持：{supported}。")
    if task_type in IMAGE_REQUIRED_TASKS:
        if not payload.image_root or not payload.image_path_column:
            raise ValueError(
                f"{task_type} 需要提供 image_root 与 image_path_column，系统只读取本地图片，不自动批量爬取远程图片。"
            )


def _load_rows(payload: ExternalDatasetImportRequest) -> list[dict[str, object]]:
    if payload.rows:
        return [dict(row) for row in payload.rows]
    if not payload.source_path:
        raise ValueError("请提供 rows 或 source_path；source_path 支持 CSV/JSON/JSONL。")

    source_path = Path(payload.source_path).expanduser()
    if not source_path.exists() or not source_path.is_file():
        raise ValueError(f"外部数据集文件不存在：{source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".parquet":
        raise ValueError("当前轻量版不直接解析 parquet；请先从 HuggingFace 导出为 CSV 或 JSONL。")
    if suffix not in SUPPORTED_FILE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FILE_FORMATS))
        raise ValueError(f"不支持的数据集格式 {suffix}；请使用 {supported}。")

    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, object]] = []
        with source_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                loaded = json.loads(stripped)
                if isinstance(loaded, dict):
                    rows.append(loaded)
        return rows

    with source_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict):
        for key in ("data", "rows", "train", "validation", "test"):
            value = loaded.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    raise ValueError("JSON 数据集需要是对象数组，或包含 data/rows/train/test 数组字段。")


def _sample_from_row(
    row: dict[str, object],
    payload: ExternalDatasetImportRequest,
    created_at: str,
) -> ExternalTrainingSample | None:
    content = _content_from_row(row, payload.text_columns)
    image_path, image_url, image_sha256, image_available = _image_fields(row, payload)
    if not content and payload.task_type not in IMAGE_REQUIRED_TASKS:
        return None
    if payload.task_type in IMAGE_REQUIRED_TASKS and not image_available:
        return None
    if not content:
        content = _vision_content(row, payload, image_path, image_url)

    label_value = row.get(payload.label_column)
    risk_score = _risk_score(row, label_value, payload)
    if risk_score is None:
        return None

    label = _label_text(label_value, risk_score)
    dataset_name = _row_text_or_default(row, "dataset_name", payload.dataset_name.strip())
    source = _row_text_or_default(row, "source", payload.source.strip() or "HuggingFace/local")
    source_url = _row_text_or_default(row, "source_url", payload.source_url)
    title = _clean_text(row.get(payload.title_column) if payload.title_column else None)
    if not title:
        title = _title_from_content(content)
    scenario = _scenario_from_row(row, payload.scenario_column, content, risk_score)
    sample_id = _sample_id(
        dataset_name,
        source,
        payload.task_type,
        payload.split,
        title,
        content,
        label,
        image_sha256 or image_path or image_url or "",
    )
    return ExternalTrainingSample(
        id=sample_id,
        dataset_name=dataset_name,
        source=source,
        source_url=source_url,
        task_type=payload.task_type.strip() or "text_risk",
        split=payload.split.strip() or "train",
        title=title,
        content=content,
        image_path=image_path,
        image_url=image_url,
        image_sha256=image_sha256,
        image_available=image_available,
        label=label,
        risk_score=risk_score,
        scenario=scenario,
        raw_payload=_jsonable_row(row),
        created_at=created_at,
    )


def _content_from_row(row: dict[str, object], columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = _clean_text(row.get(column))
        if value:
            parts.append(value)
    if not parts:
        for value in row.values():
            text = _clean_text(value)
            if len(text) >= 12 and re.search(r"[\u4e00-\u9fffA-Za-z]", text):
                parts.append(text)
                break
    content = " ".join(parts)
    return re.sub(r"\s+", " ", content).strip()


def _risk_score(
    row: dict[str, object],
    label_value: object,
    payload: ExternalDatasetImportRequest,
) -> int | None:
    if payload.risk_score_column and payload.risk_score_column in row:
        parsed = _parse_score(row[payload.risk_score_column])
        if parsed is not None:
            return parsed
    if label_value is None:
        return None
    if payload.task_type == GENERATOR_ATTRIBUTION_TASK:
        source_score = _generator_attribution_risk_score(label_value, payload)
        if source_score is not None:
            return source_score
    parsed_label_score = _parse_score(label_value)
    if parsed_label_score is not None and str(label_value).strip() not in {"0", "1"}:
        return parsed_label_score

    normalized = str(label_value).strip().lower()
    schema_key = str(label_value).strip()
    if schema_key in payload.label_schema:
        return _clamp_score(payload.label_schema[schema_key])
    if normalized in payload.label_schema:
        return _clamp_score(payload.label_schema[normalized])
    positives = {item.strip().lower() for item in payload.positive_label_values}
    negatives = {item.strip().lower() for item in payload.negative_label_values}
    if normalized in positives:
        return payload.default_positive_score
    if normalized in negatives:
        return payload.default_negative_score
    if normalized == "1":
        return payload.default_positive_score
    if normalized == "0":
        return payload.default_negative_score
    if any(token in normalized for token in ("rumor", "fake", "false", "谣", "虚假", "不实")):
        return payload.default_positive_score
    if any(token in normalized for token in ("real", "true", "事实", "真实")):
        return payload.default_negative_score
    return None


def _generator_attribution_risk_score(
    label_value: object,
    payload: ExternalDatasetImportRequest,
) -> int | None:
    normalized = str(label_value).strip().lower()
    if not normalized:
        return None
    schema_key = str(label_value).strip()
    if schema_key in payload.label_schema:
        return _clamp_score(payload.label_schema[schema_key])
    if normalized in payload.label_schema:
        return _clamp_score(payload.label_schema[normalized])
    if any(token in normalized for token in REAL_SOURCE_HINTS):
        return payload.default_negative_score
    if any(token in normalized for token in GENERATED_SOURCE_HINTS):
        return payload.default_positive_score
    if normalized in {"unknown", "other", "其它", "其他", "未知"}:
        return 50
    return None


def _image_fields(
    row: dict[str, object],
    payload: ExternalDatasetImportRequest,
) -> tuple[str | None, str | None, str | None, bool]:
    image_url = _clean_text(row.get(payload.image_url_column)) if payload.image_url_column else ""
    image_path_text = _clean_text(row.get(payload.image_path_column)) if payload.image_path_column else ""
    if not image_path_text:
        return None, image_url or None, None, False
    candidate = Path(image_path_text).expanduser()
    if not candidate.is_absolute():
        if not payload.image_root:
            return None, image_url or None, None, False
        candidate = Path(payload.image_root).expanduser() / image_path_text
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return str(candidate), image_url or None, None, False
    if not resolved.is_file():
        return str(resolved), image_url or None, None, False
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return str(resolved), image_url or None, digest, True


def _vision_content(
    row: dict[str, object],
    payload: ExternalDatasetImportRequest,
    image_path: str | None,
    image_url: str | None,
) -> str:
    title = _clean_text(row.get(payload.title_column) if payload.title_column else None)
    source = image_path or image_url or "未命名图片"
    return title or f"{payload.task_type} 图片训练样本：{source}"


def _parse_score(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        score = float(str(value).strip())
    except ValueError:
        return None
    if 0 <= score <= 1:
        score *= 100
    if 0 <= score <= 100:
        return int(round(score))
    return None


def _label_text(value: object, score: int) -> str:
    text = _clean_text(value)
    if text:
        return text[:80]
    return "高风险谣言样本" if score >= 60 else "低风险真实/误传样本"


def _scenario_from_row(
    row: dict[str, object],
    scenario_column: str | None,
    content: str,
    risk_score: int,
) -> str:
    if scenario_column:
        scenario = _clean_text(row.get(scenario_column))
        if scenario:
            return scenario[:60]
    text = content.lower()
    if "警" in text or "police" in text:
        return "涉警公信力谣言"
    if any(token in text for token in ("灾", "洪", "地震", "事故", "塌方", "fire", "flood")):
        return "灾害险情谣言"
    if any(token in text for token in ("群体", "冲突", "对立", "集合", "性别")):
        return "群体对立煽动型谣言"
    return "外部谣言检测样本"


def _title_from_content(content: str) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    return compact[:42] or "外部训练样本"


def _sample_id(
    dataset_name: str,
    source: str,
    task_type: str,
    split: str,
    title: str,
    content: str,
    label: str,
    image_identity: str,
) -> str:
    digest = hashlib.sha256(
        f"{dataset_name}|{source}|{task_type}|{split}|{title}|{content}|{label}|{image_identity}".encode("utf-8")
    ).hexdigest()[:18]
    return f"ext-{digest}"


def _clamp_score(value: object) -> int | None:
    parsed = _parse_score(value)
    if parsed is None:
        return None
    return max(0, min(100, parsed))


def _label_distribution(samples: list[ExternalTrainingSample]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for sample in samples:
        distribution[sample.label] = distribution.get(sample.label, 0) + 1
    return distribution


def _row_text_or_default(
    row: dict[str, object],
    key: str,
    default: str | None,
) -> str | None:
    value = _clean_text(row.get(key))
    if value:
        return value
    return default


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _jsonable_row(row: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = _clean_text(value)
    return safe
