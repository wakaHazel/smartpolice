from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.storage import DB_PATH, initialize_database  # noqa: E402

TASK_TYPE = "vision_generator_attribution"
ACTIVE_MODEL_ID = "e19d4bb3-c8fc-4fe3-b58d-8b6def12a3ad"
KEEP_PROFILES = {
    "binary_generated_gate",
    "gpt_image2_ovr",
    "mainstream_five_attribution",
    "clean_origin_attribution",
    "social_propagation_robustness",
}
ARTIFACT_DIR = BACKEND / "data" / "model_artifacts"
TMP_SUITE_DIR = ROOT / "tmp" / "generator_experiment_suite"
OLD_DOWNLOAD_DIRS = (
    "hf_generator_attribution_v2",
    "hf_generator_attribution_v3",
    "hf_generator_attribution_v4",
    "hf_generator_attribution_v5_balancing",
    "hf_generator_attribution_smoke",
    "hf_generator_attribution_smoke_v5",
    "hf_generator_attribution_hf_baseline_smoke",
    "hf_generator_attribution_hf_baseline_smoke2",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean obsolete generator-attribution experiments safely.")
    parser.add_argument("--execute", action="store_true", help="Actually delete rows/files. Default is dry-run.")
    parser.add_argument("--keep-latest-per-profile", type=int, default=1)
    parser.add_argument("--remove-old-downloads", action="store_true")
    args = parser.parse_args()

    initialize_database()
    rows = _load_generator_runs()
    keep_ids = _kept_run_ids(rows, args.keep_latest_per_profile)
    delete_ids = [row["id"] for row in rows if row["id"] not in keep_ids]
    artifact_files = _artifact_files_for(delete_ids)
    temp_files = _files_under(TMP_SUITE_DIR)
    old_download_files = []
    if args.remove_old_downloads:
        for name in OLD_DOWNLOAD_DIRS:
            old_download_files.extend(_files_under(BACKEND / "data" / name))

    payload = {
        "mode": "execute" if args.execute else "dry-run",
        "db_path": str(DB_PATH),
        "active_model_id": ACTIVE_MODEL_ID,
        "generator_runs_total": len(rows),
        "kept_run_ids": sorted(keep_ids),
        "deleted_run_ids": delete_ids,
        "artifact_files": [str(path) for path in artifact_files],
        "artifact_file_count": len(artifact_files),
        "artifact_mb": round(sum(path.stat().st_size for path in artifact_files if path.exists()) / 1024 / 1024, 2),
        "tmp_suite_file_count": len(temp_files),
        "tmp_suite_mb": round(sum(path.stat().st_size for path in temp_files if path.exists()) / 1024 / 1024, 2),
        "old_download_file_count": len(old_download_files),
        "old_download_mb": round(sum(path.stat().st_size for path in old_download_files if path.exists()) / 1024 / 1024, 2),
        "old_download_dirs": [str(BACKEND / "data" / name) for name in OLD_DOWNLOAD_DIRS if (BACKEND / "data" / name).exists()],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return

    _delete_db_rows(delete_ids)
    for path in artifact_files:
        path.unlink(missing_ok=True)
    if TMP_SUITE_DIR.exists():
        _remove_tree_contents(TMP_SUITE_DIR)
    if args.remove_old_downloads:
        for name in OLD_DOWNLOAD_DIRS:
            _remove_dir(BACKEND / "data" / name)


def _load_generator_runs() -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, is_active, payload
            FROM vision_training_runs
            WHERE task_type = ?
            ORDER BY created_at DESC
            """,
            (TASK_TYPE,),
        ).fetchall()
    loaded: list[dict[str, Any]] = []
    for run_id, created_at, is_active, payload_text in rows:
        payload = json.loads(str(payload_text))
        card = payload.get("model_card") if isinstance(payload, dict) else {}
        profile = None
        if isinstance(card, dict):
            profile_payload = card.get("experiment_profile")
            if isinstance(profile_payload, dict):
                profile = profile_payload.get("profile")
            elif isinstance(profile_payload, str):
                profile = profile_payload
        loaded.append(
            {
                "id": str(run_id),
                "created_at": str(created_at),
                "is_active": bool(is_active),
                "status": payload.get("status") if isinstance(payload, dict) else None,
                "profile": str(profile or "standard_attribution"),
            }
        )
    return loaded


def _kept_run_ids(rows: list[dict[str, Any]], keep_latest_per_profile: int) -> set[str]:
    keep = {row["id"] for row in rows if row["is_active"] or row["id"] == ACTIVE_MODEL_ID}
    per_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["profile"] in KEEP_PROFILES and not row["is_active"]:
            per_profile.setdefault(row["profile"], []).append(row)
    for profile_rows in per_profile.values():
        keep.update(row["id"] for row in profile_rows[: max(0, keep_latest_per_profile)])
    return keep


def _artifact_files_for(run_ids: list[str]) -> list[Path]:
    if not ARTIFACT_DIR.exists():
        return []
    escaped = "|".join(re.escape(run_id) for run_id in run_ids)
    if not escaped:
        return []
    pattern = re.compile(escaped)
    return [path for path in ARTIFACT_DIR.iterdir() if path.is_file() and pattern.search(path.name)]


def _files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file()]


def _delete_db_rows(run_ids: list[str]) -> None:
    if not run_ids:
        return
    placeholders = ",".join("?" for _ in run_ids)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            f"DELETE FROM vision_training_runs WHERE task_type = ? AND is_active = 0 AND id IN ({placeholders})",
            (TASK_TYPE, *run_ids),
        )
        connection.commit()


def _remove_tree_contents(path: Path) -> None:
    for item in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            item.rmdir()


def _remove_dir(path: Path) -> None:
    if not path.exists():
        return
    _remove_tree_contents(path)
    path.rmdir()


if __name__ == "__main__":
    main()
