from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_MANIFEST = ROOT / "platform_eval" / "upload_batch_60" / "upload_manifest.csv"
DEFAULT_RETURNED_ROOT = ROOT / "platform_eval" / "returned"
DEFAULT_OUTPUT_DIR = ROOT / "platform_eval" / "returned"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a /training/datasets/import manifest from the real Weibo/Xiaohongshu "
            "returned platform-transcode image set."
        )
    )
    parser.add_argument("--upload-manifest", default=str(DEFAULT_UPLOAD_MANIFEST))
    parser.add_argument("--returned-root", default=str(DEFAULT_RETURNED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-name", default="SmartPolice real-platform-transcode-60")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--platforms", default="weibo,xhs")
    parser.add_argument("--variants", default="download,screenshot")
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any requested platform/variant pair is missing; off by default because xhs_screenshot is currently unavailable.",
    )
    args = parser.parse_args()

    rows, status = build_rows(
        upload_manifest=Path(args.upload_manifest),
        returned_root=Path(args.returned_root),
        dataset_name=args.dataset_name,
        source_url=args.source_url,
        platforms=parse_csv_arg(args.platforms),
        variants=parse_csv_arg(args.variants),
        include_clean=args.include_clean,
    )
    if args.require_complete and any(row["missing"] for row in status):
        missing = ", ".join(
            f"{row['condition']}={row['missing']}" for row in status if row["missing"]
        )
        raise SystemExit(f"missing requested returned files: {missing}")
    write_outputs(rows, status, Path(args.output_dir), args.dataset_name)


def parse_csv_arg(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("CSV argument must contain at least one value")
    return list(dict.fromkeys(values))


def build_rows(
    *,
    upload_manifest: Path,
    returned_root: Path,
    dataset_name: str,
    source_url: str,
    platforms: list[str],
    variants: list[str],
    include_clean: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    upload_items = load_upload_manifest(upload_manifest)
    pair_ids = [str(item["pair_id"]) for item in upload_items]
    rows: list[dict[str, object]] = []
    status: list[dict[str, object]] = []

    if include_clean:
        clean_count = 0
        for item in upload_items:
            clean_path = Path(str(item["upload_path"]))
            if not clean_path.is_file():
                continue
            rows.append(
                row_from_item(
                    item=item,
                    image_path=clean_path,
                    dataset_name=dataset_name,
                    source_url=source_url,
                    platform="clean",
                    variant="clean",
                    condition="clean",
                    split="platform_clean_reference",
                    role="clean_reference",
                )
            )
            clean_count += 1
        status.append(
            {
                "condition": "clean",
                "expected": len(upload_items),
                "matched": clean_count,
                "missing": len(upload_items) - clean_count,
            }
        )

    for platform in platforms:
        for variant in variants:
            condition = f"{platform}_{variant}"
            folder = returned_root / condition
            matches = index_returned_files(folder, pair_ids)
            matched = 0
            for item in upload_items:
                files = matches.get(str(item["pair_id"]), [])
                chosen = choose_returned_file(files)
                if chosen is None:
                    continue
                matched += 1
                rows.append(
                    row_from_item(
                        item=item,
                        image_path=chosen,
                        dataset_name=dataset_name,
                        source_url=source_url,
                        platform=platform,
                        variant=variant,
                        condition=condition,
                        split=f"platform_{condition}",
                        role="real_platform_transcode",
                    )
                )
            status.append(
                {
                    "condition": condition,
                    "expected": len(upload_items),
                    "matched": matched,
                    "missing": len(upload_items) - matched,
                }
            )
    return rows, status


def load_upload_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"upload manifest not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for raw in csv.DictReader(file):
            pair_id = str(raw.get("pair_id") or "").strip()
            upload_path = Path(str(raw.get("upload_path") or ""))
            if not pair_id or not upload_path.is_file():
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "label": normalize_label(str(raw.get("label") or "")),
                    "upload_path": str(upload_path),
                    "original_sha256": str(raw.get("original_sha256") or ""),
                    "original_dataset_name": str(raw.get("dataset_name") or ""),
                    "original_source": str(raw.get("source") or ""),
                    "original_source_url": str(raw.get("source_url") or ""),
                    "width": str(raw.get("width") or ""),
                    "height": str(raw.get("height") or ""),
                }
            )
    if not rows:
        raise SystemExit(f"no usable rows found in upload manifest: {path}")
    return sorted(rows, key=lambda row: str(row["pair_id"]))


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    if normalized in {"gpt-image-2", "gptimage2", "gpt image 2"}:
        return "gpt-image2"
    if normalized in {"real", "authentic", "camera", "photo"}:
        return "real"
    return normalized or "unknown"


def index_returned_files(folder: Path, pair_ids: Iterable[str]) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    if not folder.is_dir():
        return indexed
    pair_lookup = {pair_id.lower(): pair_id for pair_id in pair_ids}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lower_name = path.name.lower()
        for pair_lower, pair_id in pair_lookup.items():
            if pair_lower in lower_name:
                indexed[pair_id].append(path)
                break
    return indexed


def choose_returned_file(files: list[Path]) -> Path | None:
    existing = [path for path in files if path.is_file()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: (-path.stat().st_size, path.name.lower()))[0]


def row_from_item(
    *,
    item: dict[str, object],
    image_path: Path,
    dataset_name: str,
    source_url: str,
    platform: str,
    variant: str,
    condition: str,
    split: str,
    role: str,
) -> dict[str, object]:
    label = str(item["label"])
    pair_id = str(item["pair_id"])
    original_source = str(item.get("original_source") or "")
    title = f"{pair_id} {label} {condition} platform transcode sample"
    caption = (
        f"社交平台传播扰动样本；condition={condition}；platform={platform}；"
        f"variant={variant}；original_source={original_source or 'unknown'}。"
    )
    return {
        "dataset_name": dataset_name,
        "source": f"SmartPolice/platform_transcode:{condition}",
        "source_url": source_url or str(item.get("original_source_url") or ""),
        "source_detail": original_source,
        "split": split,
        "image": relative_to_root(image_path),
        "caption": caption,
        "title": title,
        "label": label,
        "task_type": "vision_generator_attribution",
        "scenario": "社交平台传播扰动鲁棒性评测",
        "benchmark_role": role,
        "condition": condition,
        "platform": platform,
        "variant": variant,
        "pair_id": pair_id,
        "original_sha256": str(item.get("original_sha256") or ""),
        "original_dataset_name": str(item.get("original_dataset_name") or ""),
        "original_source": original_source,
        "width": str(item.get("width") or ""),
        "height": str(item.get("height") or ""),
        "use_policy": "candidate_only_or_evaluation; never auto-activate active model",
    }


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def write_outputs(
    rows: list[dict[str, object]],
    status: list[dict[str, object]],
    output_dir: Path,
    dataset_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", dataset_name).strip("_") or "platform_transcode"
    manifest_path = output_dir / f"{safe_name}_manifest.jsonl"
    payload_path = output_dir / f"{safe_name}_import_payload.json"
    summary_path = output_dir / f"{safe_name}_summary.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = Counter(str(row["label"]) for row in rows)
    condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        condition_counts[str(row["condition"])][str(row["label"])] += 1

    payload = {
        "dataset_name": dataset_name,
        "source": "SmartPolice/platform_transcode",
        "task_type": "vision_generator_attribution",
        "source_path": str(manifest_path),
        "image_root": str(ROOT),
        "image_path_column": "image",
        "text_columns": ["caption", "title", "source", "source_detail", "benchmark_role"],
        "title_column": "title",
        "label_column": "label",
        "scenario_column": "scenario",
        "format": "jsonl",
        "split": "platform_transcode",
        "limit": len(rows),
    }
    summary = {
        "manifest_path": str(manifest_path),
        "import_payload_path": str(payload_path),
        "dataset_name": dataset_name,
        "row_count": len(rows),
        "label_distribution": dict(sorted(label_counts.items())),
        "condition_label_distribution": {
            condition: dict(sorted(counts.items())) for condition, counts in sorted(condition_counts.items())
        },
        "collection_status": status,
        "import_command": (
            "$payload = Get-Content "
            f"'{payload_path}' -Raw | ConvertFrom-Json; "
            "Invoke-RestMethod 'http://127.0.0.1:8000/training/datasets/import' "
            "-Method Post -ContentType 'application/json' -Body ($payload | ConvertTo-Json -Depth 8)"
        ),
        "note": (
            "This manifest is a real-platform propagation perturbation set. "
            "Use it for candidate-only training/evaluation or reporting; do not claim platform codec rules."
        ),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
