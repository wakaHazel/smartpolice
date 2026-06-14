from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_MANIFEST = ROOT / "platform_eval" / "upload_batch_60" / "upload_manifest.csv"
DEFAULT_COLLECTION_MANIFEST = ROOT / "platform_eval" / "returned" / "collection_manifest.csv"
DEFAULT_OUTPUT_JSON = ROOT / "output" / "audits" / "platform_transcode_artifacts_latest.json"
DEFAULT_OUTPUT_MD = ROOT / "output" / "audits" / "platform_transcode_artifacts_latest.md"
DEFAULT_OUTPUT_CSV = ROOT / "output" / "audits" / "platform_transcode_artifacts_latest.csv"


@dataclass(frozen=True)
class ImageArtifact:
    pair_id: str
    label: str
    condition: str
    path: Path
    sha256: str
    suffix: str
    format: str
    width: int
    height: int
    mode: str
    bytes_size: int
    jpeg_qtable_hash: str
    jpeg_qtable_sum: int
    jpeg_qtable_count: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze observable black-box platform transcode artifacts for Weibo/XHS returned images."
    )
    parser.add_argument("--upload-manifest", default=str(DEFAULT_UPLOAD_MANIFEST))
    parser.add_argument("--collection-manifest", default=str(DEFAULT_COLLECTION_MANIFEST))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    clean_items = load_clean_items(Path(args.upload_manifest))
    returned_items = load_returned_items(Path(args.collection_manifest))
    all_items = [*clean_items, *returned_items]
    by_pair_clean = {item.pair_id: item for item in clean_items}
    paired_rows = build_paired_rows(returned_items, by_pair_clean)
    payload = build_payload(all_items, paired_rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, paired_rows)

    print(json.dumps(payload["condition_summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(f"wrote {output_csv}")


def load_clean_items(path: Path) -> list[ImageArtifact]:
    if not path.is_file():
        raise SystemExit(f"upload manifest not found: {path}")
    items: list[ImageArtifact] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            image_path = Path(str(row.get("upload_path") or ""))
            if not image_path.is_file():
                continue
            items.append(read_artifact(str(row.get("pair_id") or ""), str(row.get("label") or ""), "clean", image_path))
    if not items:
        raise SystemExit(f"no clean images found in {path}")
    return sorted(items, key=lambda item: item.pair_id)


def load_returned_items(path: Path) -> list[ImageArtifact]:
    if not path.is_file():
        raise SystemExit(f"collection manifest not found: {path}")
    items: list[ImageArtifact] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("status") or "") != "matched":
                continue
            image_path = Path(str(row.get("matched_path") or ""))
            if not image_path.is_file():
                continue
            items.append(
                read_artifact(
                    str(row.get("pair_id") or ""),
                    str(row.get("label") or ""),
                    str(row.get("condition") or ""),
                    image_path,
                )
            )
    if not items:
        raise SystemExit(f"no returned images found in {path}")
    return sorted(items, key=lambda item: (item.condition, item.pair_id))


def read_artifact(pair_id: str, label: str, condition: str, path: Path) -> ImageArtifact:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as image:
        fmt = str(image.format or path.suffix.lstrip(".")).upper()
        width, height = image.size
        mode = image.mode
        quantization = getattr(image, "quantization", None)
        q_hash, q_sum, q_count = jpeg_qtable_stats(quantization)
    return ImageArtifact(
        pair_id=pair_id,
        label=normalize_label(label),
        condition=condition,
        path=path,
        sha256=digest,
        suffix=path.suffix.lower(),
        format=fmt,
        width=int(width),
        height=int(height),
        mode=mode,
        bytes_size=len(raw),
        jpeg_qtable_hash=q_hash,
        jpeg_qtable_sum=q_sum,
        jpeg_qtable_count=q_count,
    )


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    if normalized in {"gpt-image-2", "gptimage2", "gpt image 2"}:
        return "gpt-image2"
    if normalized in {"authentic", "camera", "photo"}:
        return "real"
    return normalized or "unknown"


def jpeg_qtable_stats(quantization: object) -> tuple[str, int, int]:
    if not isinstance(quantization, dict) or not quantization:
        return "", 0, 0
    flat: list[int] = []
    for key in sorted(quantization):
        value = quantization[key]
        if isinstance(value, list):
            flat.extend(int(item) for item in value)
    encoded = ",".join(str(item) for item in flat)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16], sum(flat), len(flat)


def build_paired_rows(
    returned_items: list[ImageArtifact],
    clean_by_pair: dict[str, ImageArtifact],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for returned in returned_items:
        clean = clean_by_pair.get(returned.pair_id)
        if clean is None:
            continue
        rows.append(
            {
                "pair_id": returned.pair_id,
                "label": returned.label,
                "condition": returned.condition,
                "clean_path": str(clean.path),
                "returned_path": str(returned.path),
                "clean_format": clean.format,
                "returned_format": returned.format,
                "clean_suffix": clean.suffix,
                "returned_suffix": returned.suffix,
                "clean_width": clean.width,
                "clean_height": clean.height,
                "returned_width": returned.width,
                "returned_height": returned.height,
                "clean_bytes": clean.bytes_size,
                "returned_bytes": returned.bytes_size,
                "byte_ratio": safe_ratio(returned.bytes_size, clean.bytes_size),
                "width_ratio": safe_ratio(returned.width, clean.width),
                "height_ratio": safe_ratio(returned.height, clean.height),
                "area_ratio": safe_ratio(returned.width * returned.height, clean.width * clean.height),
                "same_sha256": returned.sha256 == clean.sha256,
                "same_dimensions": returned.width == clean.width and returned.height == clean.height,
                "jpeg_qtable_changed": bool(clean.jpeg_qtable_hash or returned.jpeg_qtable_hash)
                and clean.jpeg_qtable_hash != returned.jpeg_qtable_hash,
                "clean_qtable_hash": clean.jpeg_qtable_hash,
                "returned_qtable_hash": returned.jpeg_qtable_hash,
                "clean_qtable_sum": clean.jpeg_qtable_sum,
                "returned_qtable_sum": returned.jpeg_qtable_sum,
            }
        )
    return rows


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def build_payload(all_items: list[ImageArtifact], paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[ImageArtifact]] = defaultdict(list)
    paired_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        by_condition[item.condition].append(item)
    for row in paired_rows:
        paired_by_condition[str(row["condition"])].append(row)

    condition_summary = {
        condition: summarize_condition(items, paired_by_condition.get(condition, []))
        for condition, items in sorted(by_condition.items())
    }
    return {
        "id": f"platform-transcode-artifacts-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "small paired black-box platform returned set; observations are not official platform rules",
        "condition_summary": condition_summary,
        "inferred_augmentation_recipe": inferred_recipes(condition_summary),
        "paired_rows": paired_rows,
    }


def summarize_condition(items: list[ImageArtifact], paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [item.bytes_size for item in items]
    widths = [item.width for item in items]
    heights = [item.height for item in items]
    areas = [item.width * item.height for item in items]
    return {
        "count": len(items),
        "label_distribution": dict(sorted(Counter(item.label for item in items).items())),
        "format_distribution": dict(sorted(Counter(item.format for item in items).items())),
        "extension_distribution": dict(sorted(Counter(item.suffix for item in items).items())),
        "width": stats(widths),
        "height": stats(heights),
        "megapixels": stats([area / 1_000_000 for area in areas]),
        "bytes": stats(sizes),
        "jpeg_qtable_hash_distribution": dict(
            sorted(Counter(item.jpeg_qtable_hash or "none" for item in items).items())
        ),
        "paired": summarize_paired_rows(paired_rows),
    }


def summarize_paired_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "count": len(rows),
        "same_sha256": sum(1 for row in rows if row["same_sha256"]),
        "same_dimensions": sum(1 for row in rows if row["same_dimensions"]),
        "format_changed": sum(1 for row in rows if row["clean_format"] != row["returned_format"]),
        "jpeg_qtable_changed": sum(1 for row in rows if row["jpeg_qtable_changed"]),
        "byte_ratio": stats([float(row["byte_ratio"]) for row in rows]),
        "width_ratio": stats([float(row["width_ratio"]) for row in rows]),
        "height_ratio": stats([float(row["height_ratio"]) for row in rows]),
        "area_ratio": stats([float(row["area_ratio"]) for row in rows]),
    }


def stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(float(min(values)), 6),
        "median": round(float(statistics.median(values)), 6),
        "max": round(float(max(values)), 6),
        "mean": round(float(statistics.fmean(values)), 6),
    }


def inferred_recipes(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "weibo_download_like": {
            "source_observation": "weibo_download",
            "recipe": "preserve dimensions and encode as JPEG with medium-high quality around q88-q92",
            "training_use": "synthetic augmentation on the larger external training pool, not direct reuse of returned files",
            "evidence": compact_evidence(summary.get("weibo_download", {})),
        },
        "weibo_screenshot_like": {
            "source_observation": "weibo_screenshot",
            "recipe": "downsample to thumbnail scale with max side about 240 px and save as PNG",
            "training_use": "hard perturbation augmentation and explicit failure-boundary evaluation",
            "evidence": compact_evidence(summary.get("weibo_screenshot", {})),
        },
        "xhs_download_like": {
            "source_observation": "xhs_download",
            "recipe": "identity/equivalent returned file in this creator-backend collection",
            "training_use": "do not over-weight; keep as black-box test observation unless future samples show stronger transcode",
            "evidence": compact_evidence(summary.get("xhs_download", {})),
        },
    }


def compact_evidence(condition: dict[str, Any]) -> dict[str, Any]:
    paired = condition.get("paired") if isinstance(condition, dict) else {}
    if not isinstance(paired, dict):
        paired = {}
    return {
        "format_distribution": condition.get("format_distribution", {}) if isinstance(condition, dict) else {},
        "extension_distribution": condition.get("extension_distribution", {}) if isinstance(condition, dict) else {},
        "same_sha256": paired.get("same_sha256", 0),
        "same_dimensions": paired.get("same_dimensions", 0),
        "byte_ratio_median": (paired.get("byte_ratio") or {}).get("median", 0) if isinstance(paired.get("byte_ratio"), dict) else 0,
        "area_ratio_median": (paired.get("area_ratio") or {}).get("median", 0) if isinstance(paired.get("area_ratio"), dict) else 0,
        "jpeg_qtable_changed": paired.get("jpeg_qtable_changed", 0),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Platform Transcode Artifact Analysis",
        "",
        f"- Created at: `{payload['created_at']}`",
        "- Scope: small paired black-box platform returned set; observations are not official platform rules.",
        "- xhs_screenshot remains unavailable and is not fabricated.",
        "",
        "## Condition Summary",
        "",
        "| Condition | N | Formats | Same SHA | Same Dim | Median Byte Ratio | Median Area Ratio | JPEG QTable Changed |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, summary in payload["condition_summary"].items():
        paired = summary.get("paired") or {}
        lines.append(
            "| {condition} | {count} | {formats} | {same_sha} | {same_dim} | {byte_ratio:.3f} | {area_ratio:.3f} | {qchanged} |".format(
                condition=condition,
                count=summary["count"],
                formats=json.dumps(summary["format_distribution"], ensure_ascii=False),
                same_sha=int(paired.get("same_sha256", 0)),
                same_dim=int(paired.get("same_dimensions", 0)),
                byte_ratio=float((paired.get("byte_ratio") or {}).get("median", 0) if paired else 0),
                area_ratio=float((paired.get("area_ratio") or {}).get("median", 0) if paired else 0),
                qchanged=int(paired.get("jpeg_qtable_changed", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Observable Rules For Augmentation",
            "",
            "| Synthetic condition | Observation | Proposed augmentation | Use policy |",
            "| --- | --- | --- | --- |",
        ]
    )
    for condition, recipe in payload["inferred_augmentation_recipe"].items():
        lines.append(
            f"| `{condition}` | `{recipe['source_observation']}` | {recipe['recipe']} | {recipe['training_use']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Weibo download in this sample behaves like dimension-preserving JPEG re-encoding, with changed JPEG coding traces for many pairs.",
            "- Weibo screenshot in this sample behaves like a low-resolution rendered thumbnail/screenshot saved as PNG; this explains the severe GPT-image2 recall collapse.",
            "- XHS download in this creator-backend collection is mostly clean-equivalent; it should be kept as an observation, not generalized to all Xiaohongshu user-facing downloads.",
            "- These observations should parameterize synthetic perturbation augmentation on the larger training pool; the 60-pair platform set should remain a black-box test/analysis set.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "label",
        "condition",
        "clean_format",
        "returned_format",
        "clean_width",
        "clean_height",
        "returned_width",
        "returned_height",
        "clean_bytes",
        "returned_bytes",
        "byte_ratio",
        "width_ratio",
        "height_ratio",
        "area_ratio",
        "same_sha256",
        "same_dimensions",
        "jpeg_qtable_changed",
        "clean_qtable_hash",
        "returned_qtable_hash",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


if __name__ == "__main__":
    main()
