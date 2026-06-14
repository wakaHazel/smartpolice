from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_benchmark_manifest import infer_label, normalize_generator
from tools.prepare_platform_transcode_manifest import build_rows


def test_synthbuster_style_generator_aliases() -> None:
    assert normalize_generator("dalle3") == "dall-e-3"
    assert normalize_generator("stable-diffusion-2") == "sd21"
    assert normalize_generator("stable-diffusion-xl") == "sdxl"
    assert normalize_generator("firefly") == "unknown"


def test_generic_tree_infers_generator_label_from_path() -> None:
    label, source_detail = infer_label(
        Path("synthbuster/generated/stable-diffusion-xl/example.png"),
        "generic-tree",
        "generator",
    ) or ("", "")

    assert label == "sdxl"
    assert source_detail == "synthbuster/generated/stable-diffusion-xl"


def test_platform_transcode_manifest_keeps_condition_and_label_balance(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    returned_root = tmp_path / "returned"
    upload_dir.mkdir()
    (returned_root / "weibo_download").mkdir(parents=True)
    (returned_root / "weibo_screenshot").mkdir()
    (returned_root / "xhs_download").mkdir()
    (returned_root / "xhs_screenshot").mkdir()

    for name, color in {
        "pair_0001_gpt_image2_clean.png": (220, 50, 50),
        "pair_0002_real_clean.png": (50, 120, 220),
    }.items():
        Image.new("RGB", (12, 12), color).save(upload_dir / name)

    manifest = upload_dir / "upload_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "pair_id,label,upload_path,original_sha256,dataset_name,source,source_url,width,height",
                f"pair_0001,gpt-image2,{upload_dir / 'pair_0001_gpt_image2_clean.png'},sha-gpt,source-a,src-a,,12,12",
                f"pair_0002,real,{upload_dir / 'pair_0002_real_clean.png'},sha-real,source-b,src-b,,12,12",
            ]
        ),
        encoding="utf-8",
    )

    for condition in ("weibo_download", "weibo_screenshot", "xhs_download"):
        for pair_id, label, color in (
            ("pair_0001", "gpt_image2", (200, 40, 40)),
            ("pair_0002", "real", (40, 100, 200)),
        ):
            Image.new("RGB", (12, 12), color).save(
                returned_root / condition / f"{pair_id}_{label}_{condition}.jpg"
            )

    rows, status = build_rows(
        upload_manifest=manifest,
        returned_root=returned_root,
        dataset_name="pytest platform",
        source_url="",
        platforms=["weibo", "xhs"],
        variants=["download", "screenshot"],
        include_clean=False,
    )

    assert len(rows) == 6
    assert {row["condition"] for row in rows} == {"weibo_download", "weibo_screenshot", "xhs_download"}
    assert [row for row in status if row["condition"] == "xhs_screenshot"][0]["missing"] == 2
    assert {row["benchmark_role"] for row in rows} == {"real_platform_transcode"}
    assert sum(1 for row in rows if row["label"] == "gpt-image2") == 3
    assert sum(1 for row in rows if row["label"] == "real") == 3
