from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.dataset_importer import import_external_dataset
from app.models import ExternalDatasetImportRequest, VisionTrainingRunRequest
from app.multimodal_training import train_vision_evidence_head
from app.storage import get_training_data_status, initialize_database
from app.storage import delete_external_training_samples_by_source_patterns


TASK_TYPE = "vision_tamper"
DATASET_NAME = "lorenzo-morelli/image-splicing-deepfake-mix"
DATASET_URL = "https://huggingface.co/datasets/lorenzo-morelli/image-splicing-deepfake-mix"
DATASET_SOURCE = "HuggingFace:image-splicing-deepfake-mix:test"
LOCAL_FAKEDDIT_NAME = "AdoCleanCode/Fakeddit manipulated image subset"
LOCAL_FAKEDDIT_URL = "https://huggingface.co/datasets/AdoCleanCode/Fakeddit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, import and optionally train the SmartPolice vision_tamper line from "
            "tamper-specific HF/local image pools. This script never imports generator "
            "attribution datasets."
        )
    )
    parser.add_argument("--limit", type=int, default=360, help="Maximum total manifest rows to prepare.")
    parser.add_argument("--splicing-limit", type=int, default=240, help="Rows to export from the HF splicing parquet.")
    parser.add_argument("--local-limit", type=int, default=120, help="Rows to add from local manipulated-image pool.")
    parser.add_argument("--train", action="store_true", help="Train and activate vision_tamper after import.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.02)
    args = parser.parse_args()

    initialize_database()
    removed = delete_external_training_samples_by_source_patterns(
        task_type=TASK_TYPE,
        patterns=[
            "tiny-genimage",
            "genimage",
            "gpt-image",
            "midjourney",
            "stable diffusion",
            "stable-diffusion",
            "sdxl",
            "flux",
            "dall-e",
            "seedream",
            "nano banana",
            "qwen-image",
        ],
    )
    if removed:
        print(json.dumps({"removed_cross_line_tamper_samples": removed}, ensure_ascii=False))
    prepared = prepare_manifest(
        total_limit=args.limit,
        splicing_limit=args.splicing_limit,
        local_limit=args.local_limit,
    )
    print(json.dumps(prepared, ensure_ascii=False, indent=2))

    import_result = import_external_dataset(
        ExternalDatasetImportRequest(
            dataset_name="SmartPolice tamper HF/local supervised pool",
            source="tamper-only manifest",
            source_url=DATASET_URL,
            source_path=str(prepared["manifest_path"]),
            task_type=TASK_TYPE,
            split="train",
            image_root=str(prepared["image_root"]),
            image_path_column="image",
            text_columns=["caption", "source_detail", "scenario"],
            title_column="title",
            label_column="label",
            risk_score_column="risk_score",
            scenario_column="scenario",
            limit=args.limit,
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
                "sources": [
                    source.model_dump()
                    for source in status.sources
                    if source.task_type == TASK_TYPE
                ],
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
                learning_rate=args.learning_rate,
                l2=args.l2,
                min_samples=args.min_samples,
                activation_mode="activate",
            )
        )
        print(run.model_dump_json(indent=2))


def prepare_manifest(
    *,
    total_limit: int,
    splicing_limit: int,
    local_limit: int,
) -> dict[str, object]:
    output_root = ROOT / "backend" / "data" / "hf_tamper_forensics"
    image_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    rows: list[dict[str, object]] = []
    rows.extend(_rows_from_existing_splicing_jsonl(image_root, max(0, splicing_limit)))
    if not rows:
        rows.extend(_rows_from_splicing_parquet(image_root, max(0, splicing_limit)))
    remaining = max(0, min(total_limit, splicing_limit + local_limit) - len(rows))
    if remaining:
        rows.extend(_rows_from_local_manipulated_pool(image_root, min(local_limit, remaining)))
    rows = rows[:total_limit]
    if not rows:
        raise RuntimeError(
            "No tamper rows prepared. Expected local HF cache at tmp/hf_datasets/"
            "lorenzo-morelli__image-splicing-deepfake-mix or local manipulated images under "
            "tmp/training_real/fakeddit_manipulated_tamper_images."
        )
    _write_manifest(manifest_path, rows)
    return {
        "manifest_path": str(manifest_path),
        "image_root": str(image_root),
        "row_count": len(rows),
        "label_distribution": _label_distribution(rows),
        "dataset_boundary": (
            "篡改线专属导入：HF image splicing/deepfake mix 与本地 manipulated tamper 图片池；"
            "不导入 GPT-image2、Midjourney、Stable Diffusion 等生成归因数据。"
        ),
    }


def _rows_from_splicing_parquet(image_root: Path, limit: int) -> list[dict[str, object]]:
    parquet_path = (
        ROOT
        / "tmp"
        / "hf_datasets"
        / "lorenzo-morelli__image-splicing-deepfake-mix"
        / "data"
        / "test-00000-of-00001.parquet"
    )
    if limit <= 0 or not parquet_path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise RuntimeError("pyarrow is required to export the cached HF parquet dataset.") from exc

    rows: list[dict[str, object]] = []
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=64):
        for item in batch.to_pylist():
            image = item.get("image") if isinstance(item, dict) else None
            if not isinstance(image, dict) or not image.get("bytes"):
                continue
            image_bytes = bytes(image["bytes"])
            digest = hashlib.sha256(image_bytes).hexdigest()
            filename = f"splicing_mix_{digest[:16]}.jpg"
            target = image_root / filename
            if not target.exists():
                target.write_bytes(image_bytes)
            caption = str(item.get("caption") or "HF image splicing/deepfake tamper sample")
            rows.append(
                {
                    "image": filename,
                    "title": f"HF splicing tamper sample {len(rows) + 1}",
                    "caption": caption,
                    "label": "tampered",
                    "risk_score": 86,
                    "scenario": "通用图像局部拼接/插入篡改",
                    "dataset_name": DATASET_NAME,
                    "source": DATASET_SOURCE,
                    "source_url": DATASET_URL,
                    "source_detail": "HF parquet image bytes exported into backend/data/hf_tamper_forensics/images",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _rows_from_existing_splicing_jsonl(image_root: Path, limit: int) -> list[dict[str, object]]:
    manifest_path = ROOT / "tmp" / "training_real" / "splicing_deepfake_mix_tamper.jsonl"
    source_root = ROOT / "tmp" / "training_real" / "splicing_deepfake_mix_images"
    if limit <= 0 or not manifest_path.exists() or not source_root.exists():
        return []
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                continue
            source_name = str(item.get("image_path") or "")
            source = source_root / source_name
            if not source.is_file():
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            filename = f"splicing_mix_{digest[:16]}{source.suffix.lower()}"
            target = image_root / filename
            if not target.exists():
                shutil.copy2(source, target)
            label = str(item.get("label") or "")
            is_authentic = "authentic" in label.lower() or int(item.get("risk_score") or 0) < 50
            rows.append(
                {
                    "image": filename,
                    "title": str(item.get("title") or f"Splicing mix tamper sample {len(rows) + 1}")[:120],
                    "caption": str(item.get("text") or item.get("source_caption") or ""),
                    "label": "authentic_unmodified" if is_authentic else "splicing_tampered",
                    "risk_score": 18 if is_authentic else 86,
                    "scenario": "外部图像篡改/拼接训练样本",
                    "dataset_name": DATASET_NAME,
                    "source": DATASET_SOURCE,
                    "source_url": DATASET_URL,
                    "source_detail": "Existing local export from HF image-splicing-deepfake-mix with authentic/tampered labels.",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _rows_from_local_manipulated_pool(image_root: Path, limit: int) -> list[dict[str, object]]:
    source_root = ROOT / "tmp" / "training_real" / "fakeddit_manipulated_tamper_images"
    if limit <= 0 or not source_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for source in sorted(source_root.glob("*")):
        if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        filename = f"fakeddit_manipulated_{digest[:16]}{source.suffix.lower()}"
        target = image_root / filename
        if not target.exists():
            shutil.copy2(source, target)
        rows.append(
            {
                "image": filename,
                "title": f"Fakeddit manipulated tamper image {len(rows) + 1}",
                "caption": "Generic manipulated image from local Fakeddit-derived tamper pool.",
                "label": "tampered",
                "risk_score": 82,
                "scenario": "通用图像篡改/拼接素材，不作为单据类数据集宣称",
                "dataset_name": LOCAL_FAKEDDIT_NAME,
                "source": "local extracted manipulated image pool",
                "source_url": LOCAL_FAKEDDIT_URL,
                "source_detail": "Local extracted manipulated images; generic tamper support, not receipt/document-specific.",
            }
        )
        if len(rows) >= limit:
            return rows
    return rows


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
