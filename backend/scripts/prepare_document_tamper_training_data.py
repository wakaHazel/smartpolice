from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from uuid import uuid4

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.dataset_importer import import_external_dataset
from app.models import ExternalDatasetImportRequest, VisionTrainingRunRequest
from app.multimodal_training import train_vision_evidence_head
from app.storage import get_training_data_status, initialize_database


TASK_TYPE = "vision_tamper"
DATASET_NAME = "SmartPolice document-tamper receipts from HF SROIE sources"
SOURCE = "HF document receipt datasets: Voxel51/scanned_receipts + jsdnrs/ICDAR2019-SROIE"
SOURCE_URL = "https://huggingface.co/datasets/Voxel51/scanned_receipts"
RTH_SROIE_URL = "https://huggingface.co/datasets/rth/sroie-2019-v2"
RTH_SROIE_MIRROR_PARQUET = (
    "https://hf-mirror.com/datasets/rth/sroie-2019-v2/resolve/main/data/train-00000-of-00001.parquet"
)
GITHUB_SROIE_RAW = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a document-domain tamper pool for vision_tamper from real public SROIE receipt images. "
            "The authentic source documents are public receipt/document datasets; tampered copies are "
            "programmatic local overlays with bbox labels for demo training, not user-provided evidence."
        )
    )
    parser.add_argument("--source-count", type=int, default=300)
    parser.add_argument("--tamper-variants", type=int, default=2)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--min-samples", type=int, default=200)
    args = parser.parse_args()

    initialize_database()
    prepared = prepare_document_tamper_pool(
        source_count=args.source_count,
        tamper_variants=args.tamper_variants,
    )
    print(json.dumps(prepared, ensure_ascii=False, indent=2))

    import_result = import_external_dataset(
        ExternalDatasetImportRequest(
            dataset_name=DATASET_NAME,
            source="document-tamper receipt manifest",
            source_url=SOURCE_URL,
            source_path=str(prepared["manifest_path"]),
            task_type=TASK_TYPE,
            split="train",
            image_root=str(prepared["image_root"]),
            image_path_column="image",
            text_columns=["caption", "source_detail", "scenario", "bbox_json"],
            title_column="title",
            label_column="label",
            risk_score_column="risk_score",
            scenario_column="scenario",
        )
    )
    print(import_result.model_dump_json(indent=2))

    status = get_training_data_status()
    tamper_task = next((task for task in status.tasks if task.task_type == TASK_TYPE), None)
    print(
        json.dumps(
            {
                "task_type": TASK_TYPE,
                "sample_count": tamper_task.sample_count if tamper_task else 0,
                "image_available_count": tamper_task.image_available_count if tamper_task else 0,
                "label_distribution": tamper_task.label_distribution if tamper_task else {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.train:
        run = train_vision_evidence_head(
            VisionTrainingRunRequest(
                task_type=TASK_TYPE,
                epochs=args.epochs,
                learning_rate=0.04,
                l2=0.02,
                min_samples=args.min_samples,
                activation_mode="activate",
            )
        )
        print(run.model_dump_json(indent=2))


def prepare_document_tamper_pool(*, source_count: int, tamper_variants: int) -> dict[str, object]:
    output_root = ROOT / "backend" / "data" / "hf_tamper_document_forensics"
    image_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    rows: list[dict[str, object]] = []
    source_images = _load_or_download_document_receipts(output_root, image_root, source_count)
    for index, source_path in enumerate(source_images):
        authentic_name = f"doc_authentic_{source_path.name}"
        authentic_path = image_root / authentic_name
        if not authentic_path.exists():
            authentic_path.write_bytes(source_path.read_bytes())
        rows.append(_row(authentic_name, index, "authentic_unmodified", 18, None, "真实公开收据原图"))
        for variant in range(tamper_variants):
            tampered_name, bbox = _create_tampered_receipt(source_path, image_root, index, variant)
            rows.append(_row(tampered_name, index, "document_field_tampered", 88, bbox, "真实收据原图的程序化字段覆盖篡改样本"))

    _write_manifest(manifest_path, rows)
    return {
        "manifest_path": str(manifest_path),
        "image_root": str(image_root),
        "source_receipt_count": len(source_images),
        "row_count": len(rows),
        "label_distribution": _label_distribution(rows),
        "source_boundary": (
            "真实单据原图来自公开 HF SROIE/receipt 数据源；tampered 样本为本地程序化字段覆盖，"
            "带 bbox_json，专用于 document-tamper 训练，不混用生成图数据集。"
        ),
    }


def _load_or_download_document_receipts(output_root: Path, image_root: Path, limit: int) -> list[Path]:
    parquet_paths = [
        ROOT / "tmp" / "sroie-rth-train.parquet",
        output_root / "rth-sroie-2019-v2-train.parquet",
    ]
    for parquet_path in parquet_paths:
        if parquet_path.exists() and parquet_path.stat().st_size > 1024 * 1024:
            exported = _export_sroie_parquet_images(parquet_path, image_root, limit)
            if len(exported) >= limit:
                return exported[:limit]

    downloaded = output_root / "rth-sroie-2019-v2-train.parquet"
    if _download_file(RTH_SROIE_MIRROR_PARQUET, downloaded):
        exported = _export_sroie_parquet_images(downloaded, image_root, limit)
        if exported:
            return exported[:limit]
    return _download_sroie_receipts(image_root, limit)


def _export_sroie_parquet_images(parquet_path: Path, image_root: Path, limit: int) -> list[Path]:
    exported = [path for path in sorted(image_root.glob("sroie_rth_source_*.jpg")) if _is_valid_image(path)]
    if len(exported) >= limit:
        return exported[:limit]
    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(json.dumps({"warning": "sroie_parquet_read_failed", "path": str(parquet_path), "error": str(exc)}, ensure_ascii=False))
        return exported[:limit]
    seen_sha = {_sha256(path) for path in exported}
    for row_index, row in frame.iterrows():
        if len(exported) >= limit:
            break
        image_value = row.get("image")
        if not isinstance(image_value, dict):
            continue
        raw = image_value.get("bytes")
        if not isinstance(raw, bytes) or len(raw) < 1024:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_sha:
            continue
        target = image_root / f"sroie_rth_source_{int(row_index):04d}_{digest[:12]}.jpg"
        if not target.exists():
            target.write_bytes(raw)
        if _is_valid_image(target):
            seen_sha.add(digest)
            exported.append(target)
        else:
            target.unlink(missing_ok=True)
    return exported[:limit]


def _download_file(url: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return True
    try:
        with requests.get(url, timeout=90, stream=True, headers={"User-Agent": "SmartPolice-document-tamper-prep/1.0"}) as response:
            response.raise_for_status()
            with target.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        return target.exists() and target.stat().st_size > 1024 * 1024
    except Exception as exc:
        target.unlink(missing_ok=True)
        print(json.dumps({"warning": "document_dataset_download_failed", "url": url, "error": str(exc)}, ensure_ascii=False))
        return False


def _download_sroie_receipts(image_root: Path, limit: int) -> list[Path]:
    for broken in sorted(image_root.glob("sroie_source_*.jpg")):
        if not _is_valid_image(broken):
            broken.unlink(missing_ok=True)
    cached = [path for path in sorted(image_root.glob("sroie_source_*.jpg")) if _is_valid_image(path)]
    if len(cached) >= limit:
        return cached[:limit]
    session = requests.Session()
    session.headers.update({"User-Agent": "SmartPolice-document-tamper-prep/1.0"})
    paths = cached[:]
    index = 0
    misses = 0
    while len(paths) < limit and index < 626:
        target = image_root / f"sroie_source_{index:03d}.jpg"
        if target.exists() and _is_valid_image(target):
            if target not in paths:
                paths.append(target)
            index += 1
            continue
        if target.exists():
            target.unlink(missing_ok=True)
        remote_name = f"{index:03d}.jpg"
        url = GITHUB_SROIE_RAW.format(name=remote_name)
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
            if _is_valid_image(target):
                paths.append(target)
            else:
                target.unlink(missing_ok=True)
                misses += 1
        except Exception:
            target.unlink(missing_ok=True)
            misses += 1
        index += 1
    if len(paths) < limit:
        print(
            json.dumps(
                {"warning": "sroie_source_shortfall", "requested": limit, "available": len(paths), "misses": misses},
                ensure_ascii=False,
            )
        )
    return paths[:limit]


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width >= 64 and image.height >= 64
    except Exception:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_tampered_receipt(source: Path, image_root: Path, index: int, variant: int) -> tuple[str, dict[str, float]]:
    with Image.open(source) as original:
        image = original.convert("RGB")
    width, height = image.size
    rng = random.Random(f"{source.name}:{variant}")
    box_w = max(80, int(width * rng.uniform(0.28, 0.42)))
    box_h = max(28, int(height * rng.uniform(0.045, 0.075)))
    left = int(width * rng.uniform(0.48, 0.60))
    top = int(height * rng.choice([0.28, 0.36, 0.44, 0.58]))
    right = min(width - 2, left + box_w)
    lower = min(height - 2, top + box_h)
    left = max(1, min(left, right - 8))
    top = max(1, min(top, lower - 8))

    draw = ImageDraw.Draw(image)
    patch = image.crop((left, top, right, lower)).filter(ImageFilter.GaussianBlur(radius=1.2))
    image.paste(patch, (left, top))
    fill = _nearby_background_color(image, left, top, right, lower)
    draw.rectangle((left, top, right, lower), fill=fill)
    text = rng.choice(["TOTAL 128.90", "PAID 2026-06-18", "REFUND OK", "AMOUNT 899.00", "DATE 06/18"])
    font = ImageFont.load_default()
    draw.text((left + 6, top + max(4, box_h // 4)), text, fill=(20, 20, 20), font=font)

    raw = image.tobytes()
    digest = hashlib.sha256(raw + source.name.encode() + str(variant).encode()).hexdigest()[:16]
    filename = f"doc_tampered_{index:03d}_{variant}_{digest}.jpg"
    image.save(image_root / filename, format="JPEG", quality=88)
    bbox = {
        "x1": round(left / width, 4),
        "y1": round(top / height, 4),
        "x2": round(right / width, 4),
        "y2": round(lower / height, 4),
    }
    return filename, bbox


def _nearby_background_color(image: Image.Image, left: int, top: int, right: int, lower: int) -> tuple[int, int, int]:
    sample_left = max(0, left - 8)
    sample_top = max(0, top - 8)
    sample_right = min(image.width, right + 8)
    sample_lower = min(image.height, lower + 8)
    crop = image.crop((sample_left, sample_top, sample_right, sample_lower))
    pixels = list(crop.getdata())
    if not pixels:
        return (245, 245, 245)
    channels = list(zip(*pixels, strict=False))
    return tuple(int(sorted(channel)[len(channel) // 2]) for channel in channels[:3])


def _row(
    image_name: str,
    index: int,
    label: str,
    risk_score: int,
    bbox: dict[str, float] | None,
    source_detail: str,
) -> dict[str, object]:
    return {
        "image": image_name,
        "title": f"Document tamper receipt sample {index:03d}",
        "caption": "receipt document tamper / authentic control sample with amount/date/status field review",
        "label": label,
        "risk_score": risk_score,
        "scenario": "单据/凭证字段局部篡改候选区域训练",
        "dataset_name": DATASET_NAME,
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "source_detail": source_detail,
        "bbox_json": json.dumps(bbox or {}, ensure_ascii=False),
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image",
        "title",
        "caption",
        "label",
        "risk_score",
        "scenario",
        "dataset_name",
        "source",
        "source_url",
        "source_detail",
        "bbox_json",
        "manifest_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "manifest_id": str(uuid4())})


def _label_distribution(rows: list[dict[str, object]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label") or "unknown")
        labels[label] = labels.get(label, 0) + 1
    return labels


if __name__ == "__main__":
    main()
