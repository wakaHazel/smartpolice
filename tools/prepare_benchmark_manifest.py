from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

REAL_HINTS = {
    "0",
    "0_real",
    "authentic",
    "camera",
    "human",
    "nature",
    "photo",
    "real",
    "real_images",
}

FAKE_HINTS = {
    "1",
    "1_fake",
    "ai",
    "aigc",
    "fake",
    "generated",
    "synthetic",
}

GENERATOR_ALIASES = {
    "adm": "unknown",
    "biggan": "unknown",
    "dall-e": "dall-e",
    "dall-e-2": "dall-e",
    "dall-e-3": "dall-e-3",
    "dalle-2": "dall-e",
    "dalle-3": "dall-e-3",
    "dalle": "dall-e",
    "dalle2": "dall-e",
    "dalle3": "dall-e-3",
    "firefly": "unknown",
    "flux": "flux",
    "glide": "unknown",
    "gpt-image-1": "gpt-image1",
    "gpt-image-1.5": "gpt-image1.5",
    "gpt-image-2": "gpt-image2",
    "gpt-image1": "gpt-image1",
    "gpt-image1.5": "gpt-image1.5",
    "gpt-image2": "gpt-image2",
    "midjourney": "midjourney",
    "mj": "midjourney",
    "nano-banana": "nano-banana",
    "sd": "stable-diffusion",
    "sd1": "stable-diffusion",
    "sd1-3": "stable-diffusion",
    "sd1-4": "stable-diffusion",
    "sd15": "stable-diffusion",
    "sd2": "sd21",
    "sd21": "sd21",
    "sd3": "sd3",
    "sdxl": "sdxl",
    "seedream": "seedream-4",
    "stable-diffusion": "stable-diffusion",
    "stable-diffusion-1": "stable-diffusion",
    "stable-diffusion-1-3": "stable-diffusion",
    "stable-diffusion-1-4": "stable-diffusion",
    "stable-diffusion-2": "sd21",
    "stable-diffusion-v-1-4": "stable-diffusion",
    "stable-diffusion-v-1-5": "stable-diffusion",
    "stable-diffusion-v-2": "sd21",
    "stable-diffusion-v2": "sd21",
    "stable-diffusion-xl": "sdxl",
    "vqdm": "unknown",
    "wukong": "unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a locally downloaded image benchmark folder into the JSONL "
            "manifest accepted by /training/datasets/import."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name stored in the manifest.")
    parser.add_argument("--image-root", required=True, help="Root directory containing benchmark images.")
    parser.add_argument(
        "--format",
        choices=("genimage", "aigibench", "generic-tree"),
        default="generic-tree",
        help="Directory convention to infer labels from.",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/data/benchmark_manifests",
        help="Directory for generated JSONL manifest and import payload.",
    )
    parser.add_argument("--split", default="", help="Optional split name to store in each row.")
    parser.add_argument("--source-url", default="", help="Paper, GitHub, or dataset URL for provenance.")
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional cap per normalized label.")
    parser.add_argument(
        "--task-type",
        default="vision_generator_attribution",
        choices=("vision_generator_attribution", "vision_aigc"),
        help="Target SmartPolice vision task.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("generator", "binary"),
        default="generator",
        help="Use generator labels or binary real/generated labels.",
    )
    args = parser.parse_args()

    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.exists():
        raise SystemExit(f"image root not found: {image_root}")

    rows = build_rows(
        dataset=args.dataset,
        image_root=image_root,
        source_url=args.source_url,
        split=args.split,
        benchmark_format=args.format,
        task_type=args.task_type,
        label_mode=args.label_mode,
        max_per_label=args.max_per_label,
    )
    write_outputs(
        rows=rows,
        output_dir=Path(args.output_dir),
        image_root=image_root,
        dataset=args.dataset,
        task_type=args.task_type,
    )


def build_rows(
    *,
    dataset: str,
    image_root: Path,
    source_url: str,
    split: str,
    benchmark_format: str,
    task_type: str,
    label_mode: str,
    max_per_label: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for image_path in iter_images(image_root):
        relative_path = image_path.relative_to(image_root)
        inferred = infer_label(relative_path, benchmark_format, label_mode)
        if inferred is None:
            continue
        label, source_detail = inferred
        if max_per_label > 0 and counts[label] >= max_per_label:
            continue
        counts[label] += 1
        rows.append(
            {
                "dataset_name": dataset,
                "source": f"{dataset}:{split or top_level(relative_path)}",
                "source_detail": source_detail,
                "source_url": source_url,
                "split": split or split_from_path(relative_path),
                "image": str(relative_path).replace("\\", "/"),
                "caption": f"{label} sample from {dataset} benchmark",
                "label": label,
                "task_type": task_type,
                "benchmark_role": benchmark_format,
            }
        )
    return rows


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_label(relative_path: Path, benchmark_format: str, label_mode: str) -> tuple[str, str] | None:
    parts = [slug(part) for part in relative_path.parts[:-1]]
    if not parts:
        return None

    real_index = first_index(parts, REAL_HINTS)
    fake_index = first_index(parts, FAKE_HINTS)
    if real_index is not None and (fake_index is None or real_index >= fake_index):
        label = "real"
    elif label_mode == "binary" and fake_index is not None:
        label = "generated"
    else:
        label = generator_label(parts, benchmark_format)

    source_detail = "/".join(parts)
    return label, source_detail


def generator_label(parts: list[str], benchmark_format: str) -> str:
    ordered = generator_candidates(parts, benchmark_format)
    for candidate in ordered:
        normalized = normalize_generator(candidate)
        if normalized:
            return normalized
    return "unknown"


def generator_candidates(parts: list[str], benchmark_format: str) -> list[str]:
    if benchmark_format == "genimage":
        return parts
    if benchmark_format == "aigibench":
        return list(reversed(parts))
    return list(reversed(parts))


def normalize_generator(value: str) -> str | None:
    cleaned = slug(value)
    if cleaned in REAL_HINTS or cleaned in FAKE_HINTS:
        return None
    if cleaned in GENERATOR_ALIASES:
        return GENERATOR_ALIASES[cleaned]
    for alias, label in GENERATOR_ALIASES.items():
        if alias in cleaned:
            return label
    if "stable" in cleaned and "diffusion" in cleaned:
        if "xl" in cleaned:
            return "sdxl"
        if "3" in cleaned:
            return "sd3"
        if "2-1" in cleaned or "21" in cleaned:
            return "sd21"
        return "stable-diffusion"
    return "unknown"


def first_index(parts: list[str], hints: set[str]) -> int | None:
    for index, part in enumerate(parts):
        if part in hints:
            return index
    return None


def slug(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", value.strip().lower()).strip("-")


def top_level(relative_path: Path) -> str:
    return relative_path.parts[0] if relative_path.parts else "root"


def split_from_path(relative_path: Path) -> str:
    for part in relative_path.parts:
        normalized = slug(part)
        if normalized in {"train", "test", "val", "valid", "validation"}:
            return normalized
    return ""


def write_outputs(
    *,
    rows: list[dict[str, object]],
    output_dir: Path,
    image_root: Path,
    dataset: str,
    task_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = re.sub(r"[^0-9A-Za-z_-]+", "_", dataset).strip("_") or "benchmark"
    manifest_path = output_dir / f"{safe_dataset}_manifest.jsonl"
    payload_path = output_dir / f"{safe_dataset}_import_payload.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts: Counter[str] = Counter(str(row["label"]) for row in rows)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source_counts[str(row["source"])][str(row["label"])] += 1

    payload = {
        "dataset_name": dataset,
        "source": "local benchmark manifest",
        "task_type": task_type,
        "source_path": str(manifest_path),
        "image_root": str(image_root),
        "image_path_column": "image",
        "text_columns": ["caption", "source", "dataset_name", "source_detail"],
        "label_column": "label",
        "format": "jsonl",
        "limit": len(rows),
        "label_distribution": dict(sorted(label_counts.items())),
        "source_label_distribution": {
            source: dict(sorted(labels.items())) for source, labels in sorted(source_counts.items())
        },
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
