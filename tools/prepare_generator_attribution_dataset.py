from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from requests import Response


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_RAW_URL = "https://huggingface.co/datasets/{dataset}/resolve/main/{path}"

DEFAC_LABELS = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dall-e-3",
    5: "midjourney",
}

ROBO531_LABELS = {"flux", "imagegbt", "nano-banana", "real", "sdxl", "seedream-4"}

DEFAULT_SPECS = [
    {
        "dataset": "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "split": "train",
        "kind": "rows",
    },
    {
        "dataset": "Robo531/ai-detector-benchmark-test-data",
        "split": "train",
        "kind": "rows",
    },
    {
        "dataset": "Qwen/Qwen-Image-Bench",
        "split": "test",
        "kind": "qwen_bench",
    },
    {
        "dataset": "siddharthksah/DeepSafe-benchmark",
        "split": "train",
        "kind": "deepsafe",
    },
    {
        "dataset": "Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset",
        "split": "train_0001",
        "kind": "rapidata_pairwise",
    },
    {
        "dataset": "Rapidata/bananamark-dataset",
        "split": "train",
        "kind": "bananamark_pairwise",
    },
    {
        "dataset": "nlphuji/flickr30k",
        "split": "test",
        "kind": "flickr30k_real",
        "config": "TEST",
    },
]

OPTIONAL_SPECS = {
    "gptimage2_style_transfer": {
        "dataset": "yufan/image_style_transfer_GPTImage2",
        "split": "train",
        "kind": "gptimage2_style_transfer",
    },
    "scam_ai_gpt_image_2": {
        "dataset": "Scam-AI/gpt-image-2",
        "split": "train",
        "kind": "scam_ai_gpt_image_2",
    },
    "liminal_dreamcore_gpt_image2": {
        "dataset": "LukaDev13/Liminal-Dreamcore-1K",
        "split": "train",
        "kind": "repo_image_files_single_label",
        "label": "gpt-image2",
        "benchmark": "liminal-dreamcore-1k",
        "caption_file": "captions.md",
        "evidence_note": "README states all images were generated using GPT Image 2 at 2K resolution.",
    },
    "synthbuster_plus": {
        "dataset": "marco-willi/synthbuster-plus",
        "split": "train",
        "kind": "synthbuster_plus",
    },
    "aigc_detection_benchmark": {
        "dataset": "TheKernel01/AIGC-Detection-Benchmark",
        "split": "test",
        "kind": "aigc_detection_benchmark",
    },
    "bitmind_nano_banana": {
        "dataset": "bitmind/nano-banana",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "nano-banana",
        "benchmark": "bitmind-nano-banana",
    },
    "ash_nano_banana_pro": {
        "dataset": "ash12321/nano-banana-pro-generated-1k",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "nano-banana",
        "benchmark": "nano-banana-pro-generated",
    },
    "vibe_banana_flash": {
        "dataset": "VIBE-Benchmark/VIBE-Banana-Flash",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "nano-banana",
        "benchmark": "vibe-banana-flash",
    },
    "vibe_banana_pro": {
        "dataset": "VIBE-Benchmark/VIBE-Banana-Pro",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "nano-banana",
        "benchmark": "vibe-banana-pro",
    },
    "ash_seedream_45": {
        "dataset": "ash12321/seedream-4.5-generated-2k",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "seedream-4",
        "benchmark": "seedream-4.5-generated",
    },
    "vibe_seedream_40": {
        "dataset": "VIBE-Benchmark/VIBE-Seedream4.0",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "seedream-4",
        "benchmark": "vibe-seedream4.0",
    },
    "vibe_seedream_45": {
        "dataset": "VIBE-Benchmark/VIBE-Seedream4.5",
        "split": "train",
        "kind": "single_label_image_rows",
        "label": "seedream-4",
        "benchmark": "vibe-seedream4.5",
    },
    "rapidata_seedream3_pairwise": {
        "dataset": "Rapidata/Seedream-3_t2i_human_preference",
        "split": "train",
        "kind": "rapidata_seedream_pairwise",
    },
    "tellif_semantic_seedream": {
        "dataset": "tellif/ai_vs_real_image_semantically_similar",
        "split": "test",
        "kind": "class_label_image_rows",
        "benchmark": "ai-vs-real-semantically-similar",
    },
}

NANO_BANANA_PUBLIC_SPEC_KEYS = (
    "bitmind_nano_banana",
    "ash_nano_banana_pro",
    "vibe_banana_flash",
    "vibe_banana_pro",
)
SEEDREAM_PUBLIC_SPEC_KEYS = (
    "vibe_seedream_40",
    "vibe_seedream_45",
    "ash_seedream_45",
    "rapidata_seedream3_pairwise",
    "tellif_semantic_seedream",
)
GPT_IMAGE2_PUBLIC_SPEC_KEYS = (
    "liminal_dreamcore_gpt_image2",
)

QWEN_BENCH_GENERATOR_COLUMNS = {
    "gpt-image-2": "gpt-image2",
    "GPT-Image-1": "gpt-image1",
    "GPT-Image-1.5": "gpt-image1.5",
    "FLUX.2_max": "flux",
    "FLUX.2-pro": "flux",
    "nano-banana-2.0": "nano-banana",
    "nano-banana-pro": "nano-banana",
    "Seedream-4.0": "seedream-4",
    "Seedream-4.5": "seedream-4",
    "Seedream-5.0": "seedream-4",
}

DEEPSAFE_LABELS = {
    2: "real",
    4: "dall-e-3",
    6: "flux",
    9: "gpt-image1",
    18: "midjourney",
    19: "midjourney",
    20: "midjourney",
    28: "sd21",
    32: "sd3",
    33: "sdxl",
}

DEEPSAFE_LABEL_NAMES = {
    2: "coco",
    4: "dalle_3",
    6: "flux_1",
    9: "gpt_image_1",
    18: "midjourney_6",
    19: "midjourney_7",
    20: "midjourney_v5",
    28: "sd_2.1",
    32: "stable_diffusion_3",
    33: "stable_diffusion_xl",
}

RAPIDATA_SPLITS = tuple(f"train_{index:04d}" for index in range(1, 27))
BANANAMARK_TARGET_LABELS = {"flux", "nano-banana", "seedream-4"}
AIGC_DETECTION_GENERATOR_LABELS = {
    0: "real",
    1: "unknown",  # ADM
    2: "unknown",  # BigGAN
    3: "unknown",  # CycleGAN
    4: "dall-e",
    5: "unknown",  # GauGAN
    6: "unknown",  # GLIDE
    7: "midjourney",
    8: "unknown",  # ProGAN
    9: "stable-diffusion",
    10: "stable-diffusion",
    11: "sdxl",
    12: "unknown",  # StarGAN
    13: "unknown",  # StyleGAN
    14: "unknown",  # StyleGAN2
    15: "unknown",  # VQDM
    16: "real",  # WhichFaceIsReal
    17: "unknown",  # Wukong
}
AIGC_DETECTION_GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "CycleGAN",
    4: "DALLE2",
    5: "GauGAN",
    6: "GLIDE",
    7: "Midjourney",
    8: "ProGAN",
    9: "SD14",
    10: "SD15",
    11: "SDXL",
    12: "StarGAN",
    13: "StyleGAN",
    14: "StyleGAN2",
    15: "VQDM",
    16: "WhichFaceIsReal",
    17: "Wukong",
}
SYNTHBUSTER_PLUS_SEED_OFFSETS = {
    "train": (
        0,
        800,
        900,
        1000,
        1100,
        1200,
        1600,
        2000,
        2800,
        3200,
        4000,
        4800,
        5200,
        6000,
        6400,
        7200,
        8000,
        8400,
        8600,
        8800,
    ),
    "validation": (0, 200, 400, 600, 800, 1000, 1200, 1400, 1800, 2000, 2200),
    "test": (0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a local generator-attribution image dataset for SmartPolice training.",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/data/hf_generator_attribution",
        help="Directory for downloaded images and JSONL manifest.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=24,
        help=(
            "Download safety cap per label for this manifest build. This is not a training cap: "
            "GPT-image2-focused experiments can use far more effective samples, while "
            "multi-generator attribution should balance by (label, source)."
        ),
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument(
        "--rows-per-seed-window",
        type=int,
        default=0,
        help=(
            "Optional row count per seeded Synthbuster window. Use a small value such as 8-12 "
            "to avoid downloading many adjacent samples from the same generator block."
        ),
    )
    parser.add_argument(
        "--synthbuster-seed-offsets",
        default="",
        help="Comma-separated Synthbuster row offsets to sample before default offsets, e.g. 2800,3200,4800,5200,8400.",
    )
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--style-transfer-per-column",
        type=int,
        default=50,
        help="Maximum GPTImage2 style-transfer samples per generated style column.",
    )
    parser.add_argument(
        "--skip-qwen-bench",
        action="store_true",
        help="Skip Qwen/Qwen-Image-Bench generator-bucket samples.",
    )
    parser.add_argument(
        "--skip-style-transfer",
        action="store_true",
        help="Deprecated compatibility flag; style-transfer samples are skipped by default.",
    )
    parser.add_argument(
        "--include-style-transfer",
        action="store_true",
        help="Include yufan/image_style_transfer_GPTImage2 auxiliary style-transfer samples. Off by default.",
    )
    parser.add_argument(
        "--include-gated-gpt-image2",
        action="store_true",
        help="Try Scam-AI/gpt-image-2 gated Twitter samples. Requires HF_TOKEN and accepted access.",
    )
    parser.add_argument(
        "--include-synthbuster-plus",
        action="store_true",
        help=(
            "Include marco-willi/synthbuster-plus via Hugging Face rows. Useful for DALL-E/Midjourney/SD "
            "source coverage; off by default because it is large."
        ),
    )
    parser.add_argument(
        "--include-aigc-detection-benchmark",
        action="store_true",
        help=(
            "Include TheKernel01/AIGC-Detection-Benchmark fake generator labels and real negatives. "
            "Off by default because it is large and mainly a detection benchmark."
        ),
    )
    parser.add_argument(
        "--only-gated-gpt-image2",
        action="store_true",
        help="Download only Scam-AI/gpt-image-2 gated samples; skip all default negative datasets.",
    )
    parser.add_argument(
        "--only-gpt-image2-public",
        action="store_true",
        help="Download only public GPT-image2 supplement datasets whose generator labels are source-documented.",
    )
    parser.add_argument(
        "--only-balancing-sources",
        action="store_true",
        help="Download only added balancing sources for real/Flux/Nano Banana/Seedream classes.",
    )
    parser.add_argument(
        "--only-hf-baseline-sources",
        action="store_true",
        help="Download only optional HF baseline sources selected by --include-synthbuster-plus/--include-aigc-detection-benchmark.",
    )
    parser.add_argument(
        "--include-nano-banana-public",
        action="store_true",
        help="Include public Nano Banana HF datasets for weak-class supplementation.",
    )
    parser.add_argument(
        "--include-seedream-public",
        action="store_true",
        help="Include public Seedream/Doubao-family HF datasets for weak-class supplementation.",
    )
    parser.add_argument(
        "--include-gpt-image2-public",
        action="store_true",
        help="Include public source-documented GPT-image2 HF datasets for the GPT-image2 specialist track.",
    )
    parser.add_argument(
        "--only-weak-mainstream-sources",
        action="store_true",
        help="Download only selected Nano Banana and Seedream public supplements.",
    )
    parser.add_argument(
        "--optional-specs",
        default="",
        help=(
            "Comma-separated OPTIONAL_SPECS keys to include directly, e.g. "
            "vibe_banana_flash,vibe_banana_pro,vibe_seedream_45."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=default_hf_token(),
        help="Hugging Face token for gated datasets; defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--rebuild-from-existing",
        action="store_true",
        help="Rebuild JSONL/summary from images already downloaded in output-dir/images, without network calls.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_from_existing:
        rows = rebuild_rows_from_existing_images(image_dir)
        write_manifest_summary(
            output_dir=output_dir,
            image_dir=image_dir,
            rows=rows,
            note=(
                "Manifest rebuilt from successfully downloaded HF images; "
                "yufan/image_style_transfer_GPTImage2 is excluded by default."
            ),
        )
        return

    rows: list[dict[str, Any]] = []
    skipped = 0
    counts: Counter[str] = Counter()

    if (
        args.only_gated_gpt_image2
        or args.only_gpt_image2_public
        or args.only_hf_baseline_sources
        or args.only_weak_mainstream_sources
    ):
        specs = []
    elif args.only_balancing_sources:
        specs = [spec for spec in DEFAULT_SPECS if spec["kind"] in {"bananamark_pairwise", "flickr30k_real"}]
    else:
        specs = list(DEFAULT_SPECS)
    if (
        not args.only_gated_gpt_image2
        and not args.only_gpt_image2_public
        and args.include_style_transfer
        and not args.skip_style_transfer
    ):
        specs.append(OPTIONAL_SPECS["gptimage2_style_transfer"])
    if args.include_gated_gpt_image2 or args.only_gated_gpt_image2:
        specs.append(OPTIONAL_SPECS["scam_ai_gpt_image_2"])
    if not args.only_gated_gpt_image2 and not args.only_gpt_image2_public and args.include_synthbuster_plus:
        specs.append(OPTIONAL_SPECS["synthbuster_plus"])
    if not args.only_gated_gpt_image2 and not args.only_gpt_image2_public and args.include_aigc_detection_benchmark:
        specs.append(OPTIONAL_SPECS["aigc_detection_benchmark"])
    optional_spec_keys: list[str] = []
    if args.include_gpt_image2_public or args.only_gpt_image2_public:
        optional_spec_keys.extend(GPT_IMAGE2_PUBLIC_SPEC_KEYS)
    if not args.only_gated_gpt_image2 and not args.only_gpt_image2_public and args.include_nano_banana_public:
        optional_spec_keys.extend(NANO_BANANA_PUBLIC_SPEC_KEYS)
    if not args.only_gated_gpt_image2 and not args.only_gpt_image2_public and args.include_seedream_public:
        optional_spec_keys.extend(SEEDREAM_PUBLIC_SPEC_KEYS)
    if not args.only_gated_gpt_image2 and args.optional_specs:
        optional_spec_keys.extend(parse_optional_spec_keys(args.optional_specs))
    seen_spec_keys: set[str] = set()
    for key in optional_spec_keys:
        if key in seen_spec_keys:
            continue
        seen_spec_keys.add(key)
        specs.append(OPTIONAL_SPECS[key])

    for spec in specs:
        try:
            if spec["kind"] == "qwen_bench" and args.skip_qwen_bench:
                continue
            target = int(args.max_per_class)
            if spec["kind"] == "qwen_bench":
                fetched = collect_qwen_image_bench(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "gptimage2_style_transfer":
                fetched = collect_gptimage2_style_transfer(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    per_column=int(args.style_transfer_per_column),
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "deepsafe":
                fetched = collect_deepsafe(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "rapidata_pairwise":
                fetched = collect_rapidata_pairwise(
                    dataset=str(spec["dataset"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "bananamark_pairwise":
                fetched = collect_bananamark_pairwise(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "flickr30k_real":
                fetched = collect_real_image_rows(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    config=str(spec.get("config", "default")),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "scam_ai_gpt_image_2":
                fetched = collect_scam_ai_gpt_image2(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "synthbuster_plus":
                fetched = collect_synthbuster_plus(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    rows_per_seed_window=args.rows_per_seed_window,
                    seed_offsets=parse_seed_offsets(args.synthbuster_seed_offsets),
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "aigc_detection_benchmark":
                fetched = collect_aigc_detection_benchmark(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "single_label_image_rows":
                fetched = collect_single_label_image_rows(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    label=str(spec["label"]),
                    benchmark=str(spec.get("benchmark", spec["dataset"])),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "repo_image_files_single_label":
                fetched = collect_repo_image_files_single_label(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    label=str(spec["label"]),
                    benchmark=str(spec.get("benchmark", spec["dataset"])),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                    caption_file=str(spec.get("caption_file", "")),
                    evidence_note=str(spec.get("evidence_note", "")),
                )
            elif spec["kind"] == "rapidata_seedream_pairwise":
                fetched = collect_rapidata_seedream_pairwise(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            elif spec["kind"] == "class_label_image_rows":
                fetched = collect_class_label_image_rows(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    benchmark=str(spec.get("benchmark", spec["dataset"])),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
            else:
                fetched = collect_dataset(
                    dataset=str(spec["dataset"]),
                    split=str(spec["split"]),
                    max_per_class=target,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                    image_dir=image_dir,
                    hf_token=args.hf_token,
                )
        except (requests.RequestException, OSError, RuntimeError) as exc:
            print(f"Skipped dataset spec after repeated failure: {spec['dataset']} ({exc})")
            fetched = []
        rows.extend(fetched)
        counts.update(row["label"] for row in fetched)

    write_manifest_summary(output_dir=output_dir, image_dir=image_dir, rows=rows)


def write_manifest_summary(
    *,
    output_dir: Path,
    image_dir: Path,
    rows: list[dict[str, Any]],
    note: str | None = None,
) -> None:
    counts: Counter[str] = Counter(str(row.get("label", "unknown")) for row in rows)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source_counts[str(row.get("dataset_name", "unknown"))][str(row.get("label", "unknown"))] += 1

    manifest_path = output_dir / "generator_attribution_train.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "image_root": str(image_dir),
        "sample_count": len(rows),
        "label_distribution": dict(sorted(counts.items())),
        "source_label_distribution": {
            source: dict(sorted(labels.items()))
            for source, labels in sorted(source_counts.items())
        },
        "task_type": "vision_generator_attribution",
        "label_column": "label",
        "image_path_column": "image",
        "text_columns": ["caption", "source", "dataset_name"],
        "sampling_note": (
            "max-per-class is only a manifest/download safety cap. Do not interpret it as the "
            "maximum useful training size. GPT-image2专项 can use all valid GPT-image2 samples; "
            "multi-generator attribution should instead control per-source dominance."
        ),
        "excluded_by_default": [
            {
                "dataset_name": "yufan/image_style_transfer_GPTImage2",
                "reason": "style-transfer auxiliary images are not normal in-the-wild GPT-image2 outputs.",
            }
        ],
    }
    if note:
        summary["note"] = note
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def default_hf_token() -> str:
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    for path in (Path.home() / ".cache" / "huggingface" / "token", Path.home() / ".huggingface" / "token"):
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return ""


def collect_dataset(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    targets = expected_labels(dataset)
    for page in range(max_pages):
        if targets and all(counts[label] >= max_per_class for label in targets):
            break
        data = try_fetch_rows(dataset, split, page * page_size, page_size, timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            parsed = parse_row(dataset, split, item)
            if parsed is None:
                continue
            label = parsed["label"]
            if counts[label] >= max_per_class:
                continue
            image_url = parsed["image_url"]
            key = f"{dataset}|{split}|{item.get('row_idx')}|{label}"
            if key in seen:
                continue
            seen.add(key)
            filename = safe_filename(dataset, split, int(item.get("row_idx", len(rows))), label, image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": parsed["caption"],
                    "label": label,
                    "source": parsed["source"],
                    "source_detail": parsed["source_detail"],
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": int(item.get("row_idx", -1)),
                    "image_url": image_url,
                }
            )
            counts[label] += 1
    return rows


def rebuild_rows_from_existing_images(image_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(sorted(image_dir.iterdir())):
        if not image_path.is_file():
            continue
        parsed = parse_existing_image_name(image_path.name)
        if parsed is None:
            continue
        dataset, split, label, source_detail, source_url = parsed
        rows.append(
            {
                "dataset_name": dataset,
                "split": split,
                "image": image_path.name,
                "caption": f"{label} generator attribution sample from {dataset}: {image_path.name}",
                "label": label,
                "source": f"{dataset}:{split}",
                "source_detail": source_detail,
                "source_url": source_url,
                "hf_row_idx": index,
                "image_url": None,
                "rebuilt_from_downloaded_file": True,
            }
        )
    return rows


def parse_existing_image_name(name: str) -> tuple[str, str, str, str, str] | None:
    if name.startswith("Qwen__Qwen_Image_Bench__"):
        token = name.split("__", 3)[3].rsplit("__", 1)[0]
        label = label_from_qwen_token(token)
        if not label:
            return None
        dataset = "Qwen/Qwen-Image-Bench"
        split = "test"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("Rajarshi_Roy_research__Defactify_Image_Dataset__"):
        token = name.split("__", 4)[3]
        label = token.replace("dall_e_3", "dall-e-3")
        dataset = "Rajarshi-Roy-research/Defactify_Image_Dataset"
        split = "train"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("Robo531__ai_detector_benchmark_test_data__"):
        token = name.split("__", 4)[3]
        label = token.replace("nano_banana", "nano-banana").replace("seedream_4", "seedream-4")
        dataset = "Robo531/ai-detector-benchmark-test-data"
        split = "train"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    weak_public_prefixes = {
        "bitmind__nano_banana__": ("bitmind/nano-banana", "nano-banana"),
        "ash12321__nano_banana_pro_generated_1k__": (
            "ash12321/nano-banana-pro-generated-1k",
            "nano-banana",
        ),
        "VIBE_Benchmark__VIBE_Banana_Flash__": (
            "VIBE-Benchmark/VIBE-Banana-Flash",
            "nano-banana",
        ),
        "VIBE_Benchmark__VIBE_Banana_Pro__": (
            "VIBE-Benchmark/VIBE-Banana-Pro",
            "nano-banana",
        ),
        "ash12321__seedream_4.5_generated_2k__": (
            "ash12321/seedream-4.5-generated-2k",
            "seedream-4",
        ),
        "VIBE_Benchmark__VIBE_Seedream4.0__": (
            "VIBE-Benchmark/VIBE-Seedream4.0",
            "seedream-4",
        ),
        "VIBE_Benchmark__VIBE_Seedream4.5__": (
            "VIBE-Benchmark/VIBE-Seedream4.5",
            "seedream-4",
        ),
        "Rapidata__Seedream_3_t2i_human_preference__": (
            "Rapidata/Seedream-3_t2i_human_preference",
            "seedream-4",
        ),
        "tellif__ai_vs_real_image_semantically_similar__": (
            "tellif/ai_vs_real_image_semantically_similar",
            "",
        ),
    }
    for prefix, (dataset, default_label) in weak_public_prefixes.items():
        if not name.startswith(prefix):
            continue
        parts = name.split("__", 4)
        if len(parts) < 4:
            return None
        split = parts[2]
        token = parts[3]
        label = default_label or normalize_label(token)
        if not label:
            return None
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("Rapidata__Flux_SD3_MJ_Dalle_Human_Alignment_Dataset__"):
        parts = name.split("__", 4)
        if len(parts) < 4:
            return None
        split = parts[2]
        token = parts[3]
        label = label_from_rapidata_token(token)
        if not label:
            return None
        dataset = "Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("Scam_AI__gpt_image_2__"):
        dataset = "Scam-AI/gpt-image-2"
        split = "train"
        token = name.rsplit("__", 1)[0]
        return (
            dataset,
            split,
            "gpt-image2",
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("LukaDev13__Liminal_Dreamcore_1K__"):
        dataset = "LukaDev13/Liminal-Dreamcore-1K"
        split = "train"
        token = name.rsplit("__", 1)[0]
        return (
            dataset,
            split,
            "gpt-image2",
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("siddharthksah__DeepSafe_benchmark__"):
        token = name.split("__", 4)[3]
        label = label_from_deepsafe_token(token)
        if not label:
            return None
        dataset = "siddharthksah/DeepSafe-benchmark"
        split = "train"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("marco_willi__synthbuster_plus__"):
        parts = name.split("__", 4)
        if len(parts) < 4:
            return None
        split = parts[2]
        token = parts[3]
        label = normalize_label(token)
        dataset = "marco-willi/synthbuster-plus"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    if name.startswith("TheKernel01__AIGC_Detection_Benchmark__"):
        parts = name.split("__", 4)
        if len(parts) < 4:
            return None
        split = parts[2]
        token = parts[3]
        label = normalize_label(token)
        dataset = "TheKernel01/AIGC-Detection-Benchmark"
        return (
            dataset,
            split,
            label,
            f"{dataset}:{split}:{token}",
            f"https://huggingface.co/datasets/{dataset}",
        )
    return None


def label_from_qwen_token(token: str) -> str | None:
    if token.startswith("gpt_image2"):
        return "gpt-image2"
    if token.startswith("gpt_image1_5"):
        return "gpt-image1.5"
    if token.startswith("gpt_image1"):
        return "gpt-image1"
    if token.startswith("nano_banana"):
        return "nano-banana"
    if token.startswith("seedream_4"):
        return "seedream-4"
    if token.startswith("flux"):
        return "flux"
    return None


def label_from_deepsafe_token(token: str) -> str | None:
    if token.startswith("dall_e_3"):
        return "dall-e-3"
    if token.startswith("gpt_image1"):
        return "gpt-image1"
    if token.startswith("midjourney"):
        return "midjourney"
    if token.startswith("sdxl"):
        return "sdxl"
    if token.startswith("sd3"):
        return "sd3"
    if token.startswith("sd21"):
        return "sd21"
    if token.startswith("flux"):
        return "flux"
    if token.startswith("real"):
        return "real"
    return None


def label_from_rapidata_token(token: str) -> str | None:
    if token.startswith("dall_e_3"):
        return "dall-e-3"
    if token.startswith("flux"):
        return "flux"
    if token.startswith("midjourney") or token.startswith("mj"):
        return "midjourney"
    if token.startswith("stable_diffusion"):
        return "stable-diffusion"
    return None


def collect_qwen_image_bench(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    for page in range(max_pages):
        if all(counts[label] >= max_per_class for label in set(QWEN_BENCH_GENERATOR_COLUMNS.values())):
            break
        data = try_fetch_rows(dataset, split, page * page_size, page_size, timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            row_idx = int(item.get("row_idx", len(rows)))
            prompt = str(row.get("prompt_cn") or row.get("prompt_en") or "")
            for column, label in QWEN_BENCH_GENERATOR_COLUMNS.items():
                if counts[label] >= max_per_class:
                    continue
                repo_path = row.get(column)
                if not isinstance(repo_path, str) or not repo_path.startswith("images/"):
                    continue
                if repo_path in seen_paths:
                    continue
                seen_paths.add(repo_path)
                image_url = HF_RAW_URL.format(dataset=dataset, path=repo_path)
                filename = safe_filename(
                    dataset,
                    split,
                    row_idx,
                    f"{label}-{column}",
                    image_url,
                )
                image_path = image_dir / filename
                if not image_path.exists():
                    if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                        continue
                    time.sleep(sleep)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "split": split,
                        "image": filename,
                        "caption": prompt or f"{label} generator attribution benchmark sample",
                        "label": label,
                        "source": f"{dataset}:{split}",
                        "source_detail": f"{dataset}:{split}:{column}",
                        "source_url": f"https://huggingface.co/datasets/{dataset}",
                        "hf_row_idx": row_idx,
                        "image_url": image_url,
                        "generator_column": column,
                        "prompt_en": str(row.get("prompt_en") or ""),
                        "prompt_cn": str(row.get("prompt_cn") or ""),
                    }
                )
                counts[label] += 1
    return rows


def collect_deepsafe(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    targets = set(DEEPSAFE_LABELS.values())
    for page in range(max_pages):
        if targets and all(counts[label] >= max_per_class for label in targets):
            break
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            try:
                raw_label = int(row.get("label", -1))
            except (TypeError, ValueError):
                continue
            label = DEEPSAFE_LABELS.get(raw_label)
            if not label or counts[label] >= max_per_class:
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http"):
                continue
            row_idx = int(item.get("row_idx", len(rows)))
            key = f"{dataset}|{split}|{row_idx}|{raw_label}|{image_url}"
            if key in seen:
                continue
            seen.add(key)
            label_name = DEEPSAFE_LABEL_NAMES.get(raw_label, str(raw_label))
            filename = safe_filename(dataset, split, row_idx, f"{label}-{label_name}", image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": f"{label} generator attribution sample from DeepSafe label {label_name}",
                    "label": label,
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:label={label_name}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "raw_label": raw_label,
                    "raw_label_name": label_name,
                }
            )
            counts[label] += 1
    return rows


def collect_rapidata_pairwise(
    *,
    dataset: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_urls: set[str] = set()
    targets = {"dall-e-3", "flux", "midjourney", "stable-diffusion"}
    split_pages = 0
    for split in RAPIDATA_SPLITS:
        if all(counts[label] >= max_per_class for label in targets):
            break
        for page in range(min(max_pages, 3)):
            if all(counts[label] >= max_per_class for label in targets):
                break
            data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
            if data is None:
                continue
            split_pages += 1
            page_rows = data.get("rows", [])
            if not page_rows:
                break
            for item in page_rows:
                row = item.get("row")
                if not isinstance(row, dict):
                    continue
                row_idx = int(item.get("row_idx", len(rows)))
                prompt = str(row.get("prompt") or "")
                for image_key, model_key, path_key in (
                    ("image1", "model1", "image1_path"),
                    ("image2", "model2", "image2_path"),
                ):
                    raw_model = str(row.get(model_key) or "")
                    label = normalize_label(raw_model)
                    if label not in targets or counts[label] >= max_per_class:
                        continue
                    image = row.get(image_key)
                    if not isinstance(image, dict):
                        continue
                    image_url = image.get("src")
                    if not isinstance(image_url, str) or not image_url.startswith("http"):
                        continue
                    if image_url in seen_urls:
                        continue
                    seen_urls.add(image_url)
                    repo_path = str(row.get(path_key) or "")
                    filename = safe_filename(dataset, split, row_idx, f"{label}-{image_key}", image_url)
                    image_path = image_dir / filename
                    if not image_path.exists():
                        if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                            continue
                        time.sleep(sleep)
                    rows.append(
                        {
                            "dataset_name": dataset,
                            "split": split,
                            "image": filename,
                            "caption": prompt or f"{label} generator attribution sample",
                            "label": label,
                            "source": f"{dataset}:{split}",
                            "source_detail": f"{dataset}:{split}:{raw_model}:{repo_path}",
                            "source_url": f"https://huggingface.co/datasets/{dataset}",
                            "hf_row_idx": row_idx,
                            "image_url": image_url,
                            "raw_model": raw_model,
                            "repo_path": repo_path,
                        }
                    )
                    counts[label] += 1
            if len(page_rows) < min(page_size, 100):
                break
        if split_pages >= max_pages * len(RAPIDATA_SPLITS):
            break
    return rows


def collect_bananamark_pairwise(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_urls: set[str] = set()
    for page in range(max_pages):
        if all(counts[label] >= max_per_class for label in BANANAMARK_TARGET_LABELS):
            break
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            row_idx = int(item.get("row_idx", len(rows)))
            prompt = str(row.get("prompt") or "")
            for image_key, model_key in (("image1", "model1"), ("image2", "model2")):
                raw_model = str(row.get(model_key) or "")
                label = normalize_label(raw_model)
                if label not in BANANAMARK_TARGET_LABELS or counts[label] >= max_per_class:
                    continue
                image = row.get(image_key)
                if not isinstance(image, dict):
                    continue
                image_url = image.get("src")
                if not isinstance(image_url, str) or not image_url.startswith("http"):
                    continue
                if image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                filename = safe_filename(dataset, split, row_idx, f"{label}-{image_key}", image_url)
                image_path = image_dir / filename
                if not image_path.exists():
                    if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                        continue
                    time.sleep(sleep)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "split": split,
                        "image": filename,
                        "caption": prompt or f"{label} generator attribution sample from Bananamark",
                        "label": label,
                        "source": f"{dataset}:{split}",
                        "source_detail": f"{dataset}:{split}:{raw_model}:{image_key}",
                        "source_url": f"https://huggingface.co/datasets/{dataset}",
                        "hf_row_idx": row_idx,
                        "image_url": image_url,
                        "raw_model": raw_model,
                        "benchmark": "bananamark",
                    }
                )
                counts[label] += 1
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_single_label_image_rows(
    *,
    dataset: str,
    split: str,
    label: str,
    benchmark: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    label = normalize_label(label)
    for page in range(max_pages):
        if len(rows) >= max_per_class:
            break
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            if len(rows) >= max_per_class:
                break
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            row_idx = int(item.get("row_idx", len(rows)))
            caption = str(row.get("prompt") or row.get("caption") or row.get("id") or "")
            filename = safe_filename(dataset, split, row_idx, label, image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": caption or f"{label} generator attribution sample from {benchmark}",
                    "label": label,
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:row={row_idx}:benchmark={benchmark}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "benchmark": benchmark,
                    "public_weak_mainstream_supplement": True,
                }
            )
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_repo_image_files_single_label(
    *,
    dataset: str,
    split: str,
    label: str,
    benchmark: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
    caption_file: str = "",
    evidence_note: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label = normalize_label(label)
    captions = (
        read_caption_markdown(dataset, caption_file, timeout, retries, hf_token)
        if caption_file
        else {}
    )
    image_paths = [
        path
        for path in list_repo_files(dataset, "", timeout, retries, hf_token)
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and "/" not in path.strip("/")
    ]
    max_images = min(max_per_class, max_pages * max(1, page_size), len(image_paths))
    for row_idx, repo_path in enumerate(image_paths[:max_images]):
        image_url = HF_RAW_URL.format(dataset=dataset, path=repo_path)
        filename = safe_filename(dataset, split, row_idx, f"{label}-{repo_path}", image_url)
        image_path = image_dir / filename
        if not image_path.exists():
            if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                continue
            time.sleep(sleep)
        caption = captions.get(repo_path, "")
        rows.append(
            {
                "dataset_name": dataset,
                "split": split,
                "image": filename,
                "caption": caption or f"{label} generator attribution sample from {benchmark}",
                "label": label,
                "source": f"{dataset}:{split}",
                "source_detail": f"{dataset}:{split}:{repo_path}:benchmark={benchmark}",
                "source_url": f"https://huggingface.co/datasets/{dataset}",
                "hf_row_idx": row_idx,
                "image_url": image_url,
                "repo_path": repo_path,
                "benchmark": benchmark,
                "public_gpt_image2_supplement": label == "gpt-image2",
                "source_documentation": evidence_note,
            }
        )
    return rows


def read_caption_markdown(
    dataset: str,
    caption_file: str,
    timeout: int,
    retries: int,
    hf_token: str = "",
) -> dict[str, str]:
    if not caption_file:
        return {}
    url = HF_RAW_URL.format(dataset=dataset, path=caption_file)
    try:
        response = request_with_retries(url, timeout=timeout, retries=retries, hf_token=hf_token)
    except requests.RequestException:
        return {}
    captions: dict[str, str] = {}
    current_name = ""
    current_lines: list[str] = []
    heading_pattern = re.compile(r"^###\s+([^\s#]+\.(?:jpe?g|png|webp))\s*$", re.IGNORECASE)
    for raw_line in response.text.splitlines():
        line = raw_line.rstrip()
        match = heading_pattern.match(line.strip())
        if match:
            if current_name and current_lines:
                captions[current_name] = " ".join(part.strip() for part in current_lines if part.strip())[:500]
            current_name = match.group(1)
            current_lines = []
            continue
        if current_name:
            stripped = line.strip()
            if stripped and stripped != "---":
                current_lines.append(stripped)
    if current_name and current_lines:
        captions[current_name] = " ".join(part.strip() for part in current_lines if part.strip())[:500]
    return captions


def collect_rapidata_seedream_pairwise(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    label = "seedream-4"
    for page in range(max_pages):
        if len(rows) >= max_per_class:
            break
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            if len(rows) >= max_per_class:
                break
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            row_idx = int(item.get("row_idx", len(rows)))
            prompt = str(row.get("prompt") or "")
            for image_key, model_key in (("image1", "model1"), ("image2", "model2")):
                if len(rows) >= max_per_class:
                    break
                raw_model = str(row.get(model_key) or "")
                if "seedream" not in raw_model.lower():
                    continue
                image = row.get(image_key)
                if not isinstance(image, dict):
                    continue
                image_url = image.get("src")
                if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                filename = safe_filename(dataset, split, row_idx, f"{label}-{image_key}", image_url)
                image_path = image_dir / filename
                if not image_path.exists():
                    if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                        continue
                    time.sleep(sleep)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "split": split,
                        "image": filename,
                        "caption": prompt or "Seedream family pairwise preference sample",
                        "label": label,
                        "source": f"{dataset}:{split}",
                        "source_detail": f"{dataset}:{split}:{raw_model}:{image_key}",
                        "source_url": f"https://huggingface.co/datasets/{dataset}",
                        "hf_row_idx": row_idx,
                        "image_url": image_url,
                        "raw_model": raw_model,
                        "benchmark": "rapidata-seedream3-human-preference",
                        "public_weak_mainstream_supplement": True,
                    }
                )
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_class_label_image_rows(
    *,
    dataset: str,
    split: str,
    benchmark: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_urls: set[str] = set()
    label_names: list[str] = []
    for page in range(max_pages):
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        if not label_names:
            label_names = class_label_names(data, "label")
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            raw_label = row.get("label")
            try:
                label_index = int(raw_label)
            except (TypeError, ValueError):
                continue
            raw_name = label_names[label_index] if 0 <= label_index < len(label_names) else str(raw_label)
            label = normalize_label(raw_name)
            if label not in {"seedream-4", "real"}:
                continue
            if counts[label] >= max_per_class:
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            row_idx = int(item.get("row_idx", len(rows)))
            filename = safe_filename(dataset, split, row_idx, f"{label}-{raw_name}", image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": f"{label} semantically similar benchmark sample",
                    "label": label,
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:class={raw_name}:row={row_idx}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "raw_label": raw_label,
                    "raw_label_name": raw_name,
                    "benchmark": benchmark,
                    "public_weak_mainstream_supplement": True,
                }
            )
            counts[label] += 1
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_real_image_rows(
    *,
    dataset: str,
    split: str,
    config: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for page in range(max_pages):
        if len(rows) >= max_per_class:
            break
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token, config)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            if len(rows) >= max_per_class:
                break
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            row_idx = int(item.get("row_idx", len(rows)))
            caption_value = row.get("caption")
            if isinstance(caption_value, list):
                caption = " ".join(str(item) for item in caption_value[:2])
            else:
                caption = str(caption_value or row.get("filename") or "")
            filename = safe_filename(dataset, split, row_idx, "real", image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": caption or "real photographic negative sample",
                    "label": "real",
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:row={row_idx}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "real_negative": True,
                }
            )
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_scam_ai_gpt_image2(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    if not hf_token:
        raise RuntimeError("Scam-AI/gpt-image-2 is gated; set HF_TOKEN or pass --hf-token after accepting access.")
    rows: list[dict[str, Any]] = []
    skipped = 0
    image_paths = [
        path
        for path in list_repo_files(dataset, "images/", timeout, retries, hf_token)
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    max_images = min(max_per_class, max_pages * max(1, page_size), len(image_paths))
    for row_idx, repo_path in enumerate(image_paths[:max_images]):
        image_url = HF_RAW_URL.format(dataset=dataset, path=repo_path)
        filename = safe_filename(dataset, split, row_idx, "gpt-image2-twitter", image_url)
        image_path = image_dir / filename
        if not image_path.exists():
            try:
                download_image(image_url, image_path, timeout, retries, hf_token)
                time.sleep(sleep)
            except requests.RequestException:
                skipped += 1
                continue
        rows.append(
            {
                "dataset_name": dataset,
                "split": split,
                "image": filename,
                "caption": "GPT-image2 in-the-wild Twitter/X sample from Scam-AI gated dataset",
                "label": "gpt-image2",
                "source": f"{dataset}:{split}",
                "source_detail": f"{dataset}:{split}:{repo_path}",
                "source_url": f"https://huggingface.co/datasets/{dataset}",
                "hf_row_idx": row_idx,
                "image_url": image_url,
                "repo_path": repo_path,
                "gated": True,
                "in_the_wild": True,
            }
        )
        if len(rows) >= max_images:
            break
    if skipped:
        print(f"Skipped {skipped} Scam-AI GPT-image2 images after repeated download failures.")
    return rows


def collect_synthbuster_plus(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    rows_per_seed_window: int,
    seed_offsets: list[int],
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_urls: set[str] = set()
    page_offsets = synthbuster_plus_offsets(split, page_size, max_pages, seed_offsets)
    fetch_size = rows_per_seed_window if rows_per_seed_window > 0 else page_size
    for offset in page_offsets:
        data = try_fetch_rows(dataset, split, offset, min(fetch_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            continue
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            label = normalize_label(str(row.get("source") or "unknown"))
            try:
                binary_label = int(row.get("label", -1))
            except (TypeError, ValueError):
                binary_label = -1
            if binary_label == 0:
                label = "real"
            if counts[label] >= max_per_class:
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            row_idx = int(item.get("row_idx", len(rows)))
            source_name = str(row.get("source") or "")
            image_id = str(row.get("image_id") or row_idx)
            filename = safe_filename(dataset, split, row_idx, f"{label}-{source_name}", image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": f"{label} generator attribution sample from Synthbuster source={source_name}",
                    "label": label,
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:source={source_name}:image_id={image_id}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "raw_source": source_name,
                    "raw_binary_label": binary_label,
                    "benchmark": "synthbuster-plus",
                }
            )
            counts[label] += 1
        if len(page_rows) < min(fetch_size, 100):
            continue
    return rows


def synthbuster_plus_offsets(
    split: str,
    page_size: int,
    max_pages: int,
    seed_offsets: list[int] | None = None,
) -> list[int]:
    bounded_page_size = max(1, min(page_size, 100))
    offsets: list[int] = []
    seeds = tuple(seed_offsets or ()) + tuple(SYNTHBUSTER_PLUS_SEED_OFFSETS.get(split, ()))
    for seed in seeds:
        offsets.append(seed)
        if bounded_page_size <= 50:
            offsets.append(seed + bounded_page_size)
    offsets.extend(index * bounded_page_size for index in range(max_pages))
    deduped: list[int] = []
    seen: set[int] = set()
    for offset in offsets:
        if offset < 0 or offset in seen:
            continue
        seen.add(offset)
        deduped.append(offset)
        if len(deduped) >= max_pages:
            break
    return deduped


def parse_seed_offsets(value: str) -> list[int]:
    offsets: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            offsets.append(int(stripped))
        except ValueError:
            raise SystemExit(f"Invalid --synthbuster-seed-offsets item: {stripped}") from None
    return offsets


def parse_optional_spec_keys(value: str) -> list[str]:
    keys: list[str] = []
    for token in value.split(","):
        key = token.strip()
        if not key:
            continue
        if key not in OPTIONAL_SPECS:
            valid = ", ".join(sorted(OPTIONAL_SPECS))
            raise SystemExit(f"Invalid --optional-specs item: {key}. Valid keys: {valid}") from None
        keys.append(key)
    return keys


def collect_aigc_detection_benchmark(
    *,
    dataset: str,
    split: str,
    max_per_class: int,
    page_size: int,
    max_pages: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_urls: set[str] = set()
    for page in range(max_pages):
        data = try_fetch_rows(dataset, split, page * page_size, min(page_size, 100), timeout, retries, hf_token)
        if data is None:
            continue
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for item in page_rows:
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            try:
                binary_label = int(row.get("label", -1))
                raw_generator = int(row.get("generator", -1))
            except (TypeError, ValueError):
                continue
            label = "real" if binary_label == 0 else AIGC_DETECTION_GENERATOR_LABELS.get(raw_generator, "unknown")
            if counts[label] >= max_per_class:
                continue
            image = row.get("image")
            if not isinstance(image, dict):
                continue
            image_url = image.get("src")
            if not isinstance(image_url, str) or not image_url.startswith("http") or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            row_idx = int(item.get("row_idx", len(rows)))
            generator_name = AIGC_DETECTION_GENERATOR_NAMES.get(raw_generator, str(raw_generator))
            filename = safe_filename(dataset, split, row_idx, f"{label}-{generator_name}", image_url)
            image_path = image_dir / filename
            if not image_path.exists():
                if not try_download_image(image_url, image_path, timeout, retries, hf_token):
                    continue
                time.sleep(sleep)
            rows.append(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "image": filename,
                    "caption": f"{label} attribution sample from AIGC-Detection-Benchmark generator={generator_name}",
                    "label": label,
                    "source": f"{dataset}:{split}",
                    "source_detail": f"{dataset}:{split}:generator={generator_name}:label={binary_label}",
                    "source_url": f"https://huggingface.co/datasets/{dataset}",
                    "hf_row_idx": row_idx,
                    "image_url": image_url,
                    "raw_generator": raw_generator,
                    "raw_generator_name": generator_name,
                    "raw_binary_label": binary_label,
                    "benchmark": "aigc-detection-benchmark",
                }
            )
            counts[label] += 1
        if len(page_rows) < min(page_size, 100):
            break
    return rows


def collect_gptimage2_style_transfer(
    *,
    dataset: str,
    split: str,
    per_column: int,
    timeout: int,
    retries: int,
    sleep: float,
    image_dir: Path,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    style_columns = ("American_Cartoon", "Pixel_Art", "Oil_Painting", "Ghibli")
    counts: Counter[str] = Counter()
    for parquet_path in list_repo_files(dataset, "data/", timeout, retries, hf_token):
        if not parquet_path.endswith(".parquet"):
            continue
        table = read_hf_parquet(dataset, parquet_path, timeout, retries, hf_token)
        column_names = set(table.column_names)
        if not set(style_columns) & column_names:
            continue
        row_count = table.num_rows
        for row_idx in range(row_count):
            if all(counts[column] >= per_column for column in style_columns):
                break
            row = table.slice(row_idx, 1).to_pylist()[0]
            raw_id = str(row.get("id", row_idx))
            for column in style_columns:
                if counts[column] >= per_column:
                    continue
                image_payload = row.get(column)
                image_bytes = image_bytes_from_parquet_value(image_payload)
                if not image_bytes:
                    continue
                filename = safe_filename(dataset, split, row_idx, f"gpt-image2-{column}", f"{column}.jpg")
                image_path = image_dir / filename
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                    time.sleep(sleep)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "split": split,
                        "image": filename,
                        "caption": f"GPTImage2 style-transfer sample, style={column}, source_id={raw_id}",
                        "label": "gpt-image2",
                        "source": f"{dataset}:{split}",
                        "source_detail": f"{dataset}:{split}:{column}",
                        "source_url": f"https://huggingface.co/datasets/{dataset}",
                        "hf_row_idx": row_idx,
                        "image_url": None,
                        "generator_column": column,
                        "style_transfer": True,
                    }
                )
                counts[column] += 1
    return rows


def fetch_rows(
    dataset: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
    retries: int,
    hf_token: str = "",
    config: str = "default",
) -> dict[str, Any]:
    response = request_with_retries(
        HF_ROWS_URL,
        retries=retries,
        timeout=timeout,
        hf_token=hf_token,
        params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        },
    )
    response.raise_for_status()
    loaded = response.json()
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected rows payload for {dataset}:{split}")
    return loaded


def try_fetch_rows(
    dataset: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
    retries: int,
    hf_token: str = "",
    config: str = "default",
) -> dict[str, Any] | None:
    try:
        return fetch_rows(dataset, split, offset, length, timeout, retries, hf_token, config)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"Skipped rows page after repeated fetch failure: {dataset}:{split}:{offset}-{offset + length} ({exc})")
        return None


def class_label_names(rows_payload: dict[str, Any], feature_name: str) -> list[str]:
    features = rows_payload.get("features")
    if not isinstance(features, list):
        return []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("name") != feature_name:
            continue
        type_info = feature.get("type")
        if not isinstance(type_info, dict):
            return []
        names = type_info.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
    return []


def list_repo_files(dataset: str, prefix: str, timeout: int, retries: int, hf_token: str = "") -> list[str]:
    response = request_with_retries(
        f"https://huggingface.co/api/datasets/{dataset}",
        timeout=timeout,
        retries=retries,
        hf_token=hf_token,
    )
    loaded = response.json()
    if not isinstance(loaded, dict):
        return []
    siblings = loaded.get("siblings")
    if not isinstance(siblings, list):
        return []
    files: list[str] = []
    for item in siblings:
        if isinstance(item, dict):
            name = item.get("rfilename")
            if isinstance(name, str) and name.startswith(prefix):
                files.append(name)
    return sorted(files)


def read_hf_parquet(dataset: str, repo_path: str, timeout: int, retries: int, hf_token: str = "") -> "pq.Table":
    url = HF_RAW_URL.format(dataset=dataset, path=repo_path)
    response = request_with_retries(url, timeout=timeout, retries=retries, hf_token=hf_token)
    return pq.read_table(BytesIO(response.content))


def image_bytes_from_parquet_value(value: object) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
        path = value.get("path")
        if isinstance(path, bytes):
            return path
    if isinstance(value, bytes):
        return value
    return None


def first_image_cell(row: dict[str, Any]) -> dict[str, Any] | None:
    for value in row.values():
        if isinstance(value, dict) and isinstance(value.get("src"), str):
            return value
    return None


def parse_row(dataset: str, split: str, item: dict[str, Any]) -> dict[str, str] | None:
    row = item.get("row")
    if not isinstance(row, dict):
        return None
    if dataset == "Rajarshi-Roy-research/Defactify_Image_Dataset":
        image = row.get("Image")
        label = DEFAC_LABELS.get(int(row.get("Label_B", -1)))
        caption = str(row.get("Caption") or "")
        source = f"{dataset}:{split}"
        source_detail = f"{dataset}:{split}:Label_B={row.get('Label_B')}"
    else:
        image = row.get("image")
        label = normalize_label(str(row.get("generator") or row.get("label") or ""))
        caption = str(row.get("filename") or row.get("source") or "")
        source = f"{dataset}:{split}"
        source_detail = f"{dataset}:{split}:{row.get('source') or ''}"
    if not isinstance(image, dict) or not label:
        return None
    image_url = image.get("src")
    if not isinstance(image_url, str) or not image_url.startswith("http"):
        return None
    return {
        "image_url": image_url,
        "label": label,
        "caption": caption or f"{label} generator attribution sample",
        "source": source,
        "source_detail": source_detail,
    }


def normalize_label(value: str) -> str:
    label = value.strip().lower().replace("_", "-").replace(" ", "-")
    if label.startswith("unknown"):
        return "unknown"
    if "gpt-image-2" in label or "gptimage2" in label:
        return "gpt-image2"
    if "gpt-image-1.5" in label or "gptimage1.5" in label:
        return "gpt-image1.5"
    if "gpt-image-1" in label or "gptimage1" in label:
        return "gpt-image1"
    if "dalle3" in label or "dalle-3" in label or "dall-e-3" in label:
        return "dall-e-3"
    if "dalle" in label or "dall-e" in label:
        return "dall-e"
    if "glide" in label or "firefly" in label or "imagen" in label:
        return "unknown"
    if "midjourney" in label or label == "mj" or label.startswith("mj-"):
        return "midjourney"
    if "flux" in label:
        return "flux"
    if "stable-diffusion-xl" in label or "sd-xl" in label or "sdxl" in label:
        return "sdxl"
    if "stable-diffusion-3" in label or "sd-3" in label or "sd3" in label:
        return "sd3"
    if "stable-diffusion-2.1" in label or "sd-2.1" in label or "sd21" in label:
        return "sd21"
    if "stable-diffusion" in label:
        return "stable-diffusion"
    if "sdxl" in label:
        return "sdxl"
    if "nano" in label:
        return "nano-banana"
    if "seedream" in label:
        return "seedream-4"
    if "imagegbt" in label or "image-gbt" in label:
        return "imagegbt"
    if "real" in label:
        return "real"
    return label or "unknown"


def expected_labels(dataset: str) -> set[str]:
    if dataset == "Rajarshi-Roy-research/Defactify_Image_Dataset":
        return set(DEFAC_LABELS.values())
    if dataset == "Robo531/ai-detector-benchmark-test-data":
        return ROBO531_LABELS
    return set()


def safe_filename(dataset: str, split: str, row_idx: int, label: str, image_url: str) -> str:
    suffix = Path(urlparse(image_url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    dataset_slug = dataset.replace("/", "__").replace("-", "_")
    label_slug = safe_slug(label)
    return f"{dataset_slug}__{split}__{label_slug}__{row_idx:06d}{suffix}"


def download_image(url: str, path: Path, timeout: int, retries: int, hf_token: str = "") -> None:
    response = request_with_retries(url, timeout=timeout, retries=retries, hf_token=hf_token)
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"Empty image response: {url}")
    path.write_bytes(response.content)


def try_download_image(url: str, path: Path, timeout: int, retries: int, hf_token: str = "") -> bool:
    try:
        download_image(url, path, timeout, retries, hf_token)
        return True
    except (requests.RequestException, OSError, RuntimeError) as exc:
        print(f"Skipped image after repeated download failure: {url} ({exc})")
        return False


def request_with_retries(
    url: str,
    *,
    timeout: int,
    retries: int,
    hf_token: str = "",
    params: dict[str, object] | None = None,
) -> Response:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
            response = requests.get(url, params=params, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed without an exception: {url}")


def safe_slug(value: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


if __name__ == "__main__":
    main()
