from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import shutil
import sqlite3
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.evidence_service import UPLOAD_ROOT, public_asset_path
from app.models import CaseAsset, CaseSample, SpreadMetrics
from app.storage import (
    DB_PATH,
    delete_case_sample,
    initialize_database,
    load_case_sample,
    save_case_asset,
    save_case_sample,
)


DEMO_ASSET_DIR = ROOT / "backend" / "demo_assets"
NANO_BANANA_COLLAPSE_CASE_ID = "demo-doubao-collapse-disaster-001"
GPT_STATION_CONFLICT_CASE_ID = "demo-gptimage-station-police-conflict-001"
REAL_DISASTER_RESCUE_CASE_ID = "demo-real-beijing-road-street-001"

NANO_BANANA_COLLAPSE_CASE = CaseSample(
    id=NANO_BANANA_COLLAPSE_CASE_ID,
    title="Nano Banana生成虚假坍塌灾情图片研判",
    scenario="灾害险情谣言",
    platform="本地演示导入 / 短视频平台模拟传播",
    publish_time="2026-06-14 20:10",
    source_url="本地演示图片：backend/demo_assets/nano-banana-tunnel-collapse-social.png",
    content=(
        "演示样本为使用 Nano Banana 生成的虚假隧道施工坍塌抢险图片，模拟网传“某地在建隧道"
        "突发塌方，救援车辆、消防人员和大型机械正在现场处置”的公共安全线索。该图不对应真实灾情，"
        "用于演示民警面对社交平台压缩传播图片时，进行来源研判、候选模型展示和后续复核建议。"
    ),
    image_description=(
        "待检图片为 1408x768 的单张 PNG：画面为夜间隧道施工现场，入口处有“中铁建设集团 隧道施工”"
        "等中文标识，救护车、警灯、消防救援人员、挖掘机和吊装设备共同出现在疑似塌方抢险现场。"
    ),
    spread=SpreadMetrics(
        views=268000,
        reposts=7600,
        comments=11200,
        likes=18400,
        velocity="演示设定：灾情关键词带动同城群和短视频评论区快速转发",
    ),
    manual_label="已知为 Nano Banana 生成的虚假灾情图片，用于演示图像来源研判",
    manual_risk_score=88,
    tags=["Nano Banana生成", "隧道塌方", "施工抢险", "社交平台压缩图"],
    sensitivity_notes="坍塌、救援等灾情画面容易引发恐慌、集中转发和求证报警；演示时需明确该图为生成样本。",
    review_note="演示视频主案例：使用用户提供的原始清晰图，避免低清报道截图影响模型识别。",
)
GPT_STATION_CONFLICT_CASE = CaseSample(
    id=GPT_STATION_CONFLICT_CASE_ID,
    title="GPT-image生成车站警民执法冲突图片研判",
    scenario="涉警公信力谣言",
    platform="本地演示导入 / 社交群与短视频平台模拟传播",
    publish_time="2026-06-14 21:05",
    source_url="本地演示图片：backend/demo_assets/gptimage-station-police-conflict-original.jpg",
    content=(
        "演示样本为使用 GPT-image 生成的虚假中国车站警民执法冲突图片，模拟网传“某车站民警与旅客"
        "发生激烈肢体冲突、现场大量群众围观拍摄”的涉警舆情线索。该图不对应真实执法事件，"
        "用于演示民警对高敏感涉警图片进行上传、来源研判和证据链报告生成。"
    ),
    image_description=(
        "待检图片为 1536x1024 的单张 JPEG：画面位于中国车站候车大厅，多名着警服人员与"
        "群众发生拉扯，背景有中文宣传横幅、站台编号和电子显示屏，整体呈新闻现场照片风格。"
    ),
    spread=SpreadMetrics(
        views=356000,
        reposts=12800,
        comments=19600,
        likes=22100,
        velocity="演示设定：涉警冲突关键词带动本地群聊、短视频评论区和同城话题快速扩散",
    ),
    manual_label="已知为 GPT-image 生成的虚假涉警冲突图片，用于演示图像来源研判",
    manual_risk_score=91,
    tags=["GPT-image生成", "车站", "涉警冲突", "原始清晰图"],
    sensitivity_notes="涉警执法冲突类画面容易损害公安机关公信力并激化线下围观，应优先固定原图和传播链。",
    review_note="演示视频涉警高敏感案例：使用用户提供的 GPT-image 原始清晰图，展示报告生成链路。",
)
REAL_DISASTER_RESCUE_CASE = CaseSample(
    id=REAL_DISASTER_RESCUE_CASE_ID,
    title="公开来源真实灾情救援照片核验",
    scenario="灾害险情核查",
    platform="Wikimedia Commons / 演示导入",
    publish_time="2008-05-14",
    source_url=(
        "https://commons.wikimedia.org/wiki/File:Sichuan_earthquake_save..JPG"
    ),
    content=(
        "演示样本为汶川地震后救援人员在受损建筑废墟中开展搜救的公开来源真实照片。"
        "该图用于和 AI 生成灾情图片形成对照，核查重点是确认系统不会把真实灾情救援现场误判为生成图。"
    ),
    image_description=(
        "待检图片为 3264x2448 的 JPEG 实拍照片：救援人员站在受损建筑和瓦砾现场，画面具有真实灾害"
        "救援场景、自然光照和现场杂乱细节。"
    ),
    spread=SpreadMetrics(
        views=64200,
        reposts=1480,
        comments=620,
        likes=3100,
        velocity="演示设定：灾情图片被转发求证，需区分真实救援照片与AI编造灾情图",
    ),
    manual_label="公开来源真实灾情救援照片，用于演示真实照片核验",
    manual_risk_score=32,
    tags=["真实照片", "Wikimedia Commons", "汶川地震", "救援现场", "Public Domain"],
    sensitivity_notes="真实灾情救援图片仍需核验来源、时间和地点，避免被拼接进新的本地灾情谣言叙事。",
    review_note=(
        "真实照片对照案例：来源 Wikimedia Commons；文件 Sichuan earthquake save..JPG；"
        "页面标注 Public domain / PD-self，分类包含 2008 Sichuan earthquake relief。"
    ),
)
DEMO_CASES = [
    NANO_BANANA_COLLAPSE_CASE,
    GPT_STATION_CONFLICT_CASE,
    REAL_DISASTER_RESCUE_CASE,
]
REJECTED_DEMO_IDS = [
    "demo-video-001-police-station",
    "demo-video-002-school-flood",
    "demo-video-003-traffic-collapse",
    "demo-video-004-campus-knife",
    "official-ai-earthquake-boy-001",
    "demo-public-gptimage2-windy-street-001",
]


def main() -> None:
    initialize_database()
    _remove_rejected_demo_cases()
    for case_id in (
        NANO_BANANA_COLLAPSE_CASE_ID,
        GPT_STATION_CONFLICT_CASE_ID,
        REAL_DISASTER_RESCUE_CASE_ID,
    ):
        _clear_case_evidence(case_id)
    for demo_case in DEMO_CASES:
        save_case_sample(demo_case)
    for case_id, image_path, filename, content_type in _copy_demo_images():
        save_case_asset(_asset_for_image(image_path, case_id, filename, content_type))
    print(f"Seeded {len(DEMO_CASES)} demo cases into {DB_PATH}")


def _remove_rejected_demo_cases() -> None:
    for case_id in REJECTED_DEMO_IDS:
        try:
            load_case_sample(case_id)
        except KeyError:
            continue
        delete_case_sample(case_id)


def _clear_case_evidence(case_id: str) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            DELETE FROM knowledge_documents
            WHERE id IN (
                SELECT 'snapshot-' || id FROM web_snapshots WHERE case_id = ?
            )
            """,
            (case_id,),
        )
        connection.execute(
            """
            DELETE FROM knowledge_documents
            WHERE id IN (
                SELECT 'evidence-' || id FROM evidence_items WHERE case_id = ?
            )
            """,
            (case_id,),
        )
        for table in (
            "agent_runs",
            "llm_invocations",
            "case_assets",
            "image_forensics_runs",
            "real_analysis_runs",
            "web_snapshots",
            "evidence_items",
        ):
            try:
                connection.execute(f"DELETE FROM {table} WHERE case_id = ?", (case_id,))
            except sqlite3.OperationalError:
                continue
        connection.commit()
    for root in (UPLOAD_ROOT / case_id, ROOT / "backend" / "data" / "snapshots" / case_id):
        resolved = root.resolve()
        data_root = (ROOT / "backend" / "data").resolve()
        if root.exists() and str(resolved).startswith(str(data_root)):
            shutil.rmtree(root)


def _copy_demo_images() -> list[tuple[str, Path, str, str]]:
    specs = [
        (
            NANO_BANANA_COLLAPSE_CASE_ID,
            DEMO_ASSET_DIR / "nano-banana-tunnel-collapse-social.png",
            "nano-banana-tunnel-collapse-social.png",
            "image/png",
            True,
        ),
        (
            GPT_STATION_CONFLICT_CASE_ID,
            DEMO_ASSET_DIR / "gptimage-station-police-conflict-original.jpg",
            "gptimage-station-police-conflict-original.jpg",
            "image/jpeg",
            True,
        ),
    ]
    copied: list[tuple[str, Path, str, str]] = []
    for case_id, source, filename, content_type, preserve_original in specs:
        if not source.exists():
            raise FileNotFoundError(f"Missing demo source image: {source}")
        out_dir = UPLOAD_ROOT / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / filename
        if preserve_original:
            target.write_bytes(source.read_bytes())
        else:
            with Image.open(source) as image:
                normalized = image.convert("RGB")
                if content_type == "image/jpeg":
                    normalized.save(target, "JPEG", quality=96, optimize=True)
                else:
                    normalized.save(target, "PNG", optimize=True)
        copied.append((case_id, target, filename, content_type))
    return copied


def _asset_for_image(path: Path, case_id: str, filename: str, content_type: str) -> CaseAsset:
    raw = path.read_bytes()
    width, height = Image.open(path).size
    asset_ids = {
        NANO_BANANA_COLLAPSE_CASE_ID: "asset-demo-nano-banana-collapse",
        GPT_STATION_CONFLICT_CASE_ID: "asset-demo-gptimage-station-conflict",
    }
    return CaseAsset(
        id=asset_ids.get(case_id, f"asset-{case_id}-primary"),
        case_id=case_id,
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        width=width,
        height=height,
        sha256=sha256(raw).hexdigest(),
        storage_path=str(path),
        preview_url=f"/evidence/files/{public_asset_path(str(path))}",
        created_at=datetime.now(UTC).isoformat(),
    )


if __name__ == "__main__":
    main()
