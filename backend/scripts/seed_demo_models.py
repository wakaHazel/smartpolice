from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.storage import DB_PATH, initialize_database


SEED_PATH = ROOT / "backend" / "demo_model_seed" / "active_runs.json"


def main() -> None:
    initialize_database()
    if not SEED_PATH.exists():
        print(f"Demo model seed not found: {SEED_PATH}")
        return
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Invalid demo model seed: runs must be a list.")
    with sqlite3.connect(DB_PATH) as connection:
        for run in runs:
            if not isinstance(run, dict):
                continue
            table = str(run.get("table") or "")
            if table not in {"vision_training_runs", "fusion_training_runs", "training_runs"}:
                continue
            run_id = str(run.get("id") or "")
            created_at = str(run.get("created_at") or "")
            run_payload = run.get("payload")
            artifact = run.get("artifact")
            is_active = int(run.get("is_active") or 0)
            if not run_id or not created_at or not isinstance(run_payload, dict) or not isinstance(artifact, dict):
                continue
            artifact = _resolve_packaged_artifact_paths(artifact)
            if is_active:
                if table == "vision_training_runs":
                    task_type = str(run_payload.get("task_type") or artifact.get("task_type") or "")
                    connection.execute(
                        "UPDATE vision_training_runs SET is_active = 0 WHERE task_type = ?",
                        (task_type,),
                    )
                else:
                    connection.execute(f"UPDATE {table} SET is_active = 0")
            if table == "vision_training_runs":
                task_type = str(run_payload.get("task_type") or artifact.get("task_type") or "")
                connection.execute(
                    """
                    INSERT OR REPLACE INTO vision_training_runs (id, task_type, created_at, payload, artifact, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        task_type,
                        created_at,
                        json.dumps(run_payload, ensure_ascii=False),
                        json.dumps(artifact, ensure_ascii=False),
                        is_active,
                    ),
                )
            else:
                connection.execute(
                    f"""
                    INSERT OR REPLACE INTO {table} (id, created_at, payload, artifact, is_active)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        created_at,
                        json.dumps(run_payload, ensure_ascii=False),
                        json.dumps(artifact, ensure_ascii=False),
                        is_active,
                    ),
                )
        connection.commit()
    print(f"Seeded demo model runs into {DB_PATH}")


def _resolve_packaged_artifact_paths(artifact: dict[str, object]) -> dict[str, object]:
    resolved = dict(artifact)
    for key in ("classifier_path", "binary_gate_path", "gpt_image2_detector_path"):
        value = resolved.get(key)
        if isinstance(value, str) and value:
            resolved_path = _resolve_seed_path(value)
            if resolved_path is not None:
                resolved[key] = str(resolved_path)
                metadata_key = {
                    "classifier_path": "classifier_metadata",
                    "binary_gate_path": "binary_gate_metadata",
                    "gpt_image2_detector_path": "gpt_image2_detector_metadata",
                }[key]
                metadata = resolved.get(metadata_key)
                if isinstance(metadata, dict):
                    metadata = dict(metadata)
                    metadata["artifact_path"] = str(resolved_path)
                    resolved[metadata_key] = metadata
    return resolved


def _resolve_seed_path(value: str) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path if path.exists() else None
    candidates = [
        ROOT / path,
        ROOT / "backend" / path,
        SEED_PATH.parent / path.name,
        SEED_PATH.parent / "model_artifacts" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


if __name__ == "__main__":
    main()
