from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "backend" / "data" / "smartpolice.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair stale external sample image paths by filename.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task-type", default="vision_generator_attribution")
    parser.add_argument("--new-image-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    image_root = Path(args.new_image_root).resolve()
    if not image_root.is_dir():
        raise SystemExit(f"image root not found: {image_root}")
    rows = load_rows(args.dataset_name, args.task_type)
    repairs = []
    missing = 0
    already_valid = 0
    label_counts: Counter[str] = Counter()
    for row in rows:
        current = Path(str(row["image_path"] or ""))
        if current.is_file():
            already_valid += 1
            continue
        candidate = image_root / current.name
        if not candidate.is_file():
            missing += 1
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        repairs.append((str(candidate), digest, row["id"]))
        label_counts[str(row["label"])] += 1

    if args.apply and repairs:
        with sqlite3.connect(DB_PATH) as connection:
            connection.executemany(
                """
                UPDATE external_training_samples
                SET image_path = ?, image_sha256 = ?, image_available = 1
                WHERE id = ?
                """,
                repairs,
            )

    print(
        {
            "dataset_name": args.dataset_name,
            "task_type": args.task_type,
            "new_image_root": str(image_root),
            "rows": len(rows),
            "already_valid": already_valid,
            "repairable": len(repairs),
            "missing": missing,
            "label_distribution_repaired": dict(sorted(label_counts.items())),
            "applied": bool(args.apply),
        }
    )


def load_rows(dataset_name: str, task_type: str) -> list[sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return list(
            connection.execute(
                """
                SELECT id, label, image_path
                FROM external_training_samples
                WHERE dataset_name = ? AND task_type = ?
                """,
                (dataset_name, task_type),
            )
        )


if __name__ == "__main__":
    main()
