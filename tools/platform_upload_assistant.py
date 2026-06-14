from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "platform_eval" / "upload_batch_60" / "upload_manifest.csv"
DEFAULT_STATE = ROOT / "platform_eval" / "upload_state.json"
DEFAULT_LOG = ROOT / "platform_eval" / "platform_upload_log.csv"

PLATFORM_URLS = {
    "weibo": "https://weibo.com",
    "xhs": "https://creator.xiaohongshu.com/publish/publish?from=menu&target=image",
}


@dataclass(frozen=True)
class UploadItem:
    pair_id: str
    label: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semi-automatic helper for platform transcode upload experiments."
    )
    parser.add_argument("command", choices=("plan", "open-folder", "assist", "fill-cdp", "log"))
    parser.add_argument("--platform", choices=tuple(PLATFORM_URLS), default="weibo")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--batch-index", type=int, default=1, help="1-based batch index.")
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--url", default="", help="Override platform publish/home URL.")
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9223", help="Existing Chrome CDP URL for fill-cdp.")
    parser.add_argument("--post-url", default="", help="Recorded URL after manual/semi-auto posting.")
    parser.add_argument("--status", default="posted", help="Status stored by the log command.")
    parser.add_argument("--headless", action="store_true", help="Run browser headless. Not recommended.")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    state = Path(args.state)
    log = Path(args.log)
    items = load_items(manifest)
    batches = make_batches(items, args.batch_size)
    if args.batch_index < 1 or args.batch_index > len(batches):
        raise SystemExit(f"batch-index must be 1..{len(batches)}")
    batch = batches[args.batch_index - 1]

    if args.command == "plan":
        write_state(state, args.platform, args.batch_size, batches)
        print_plan(args.platform, args.batch_size, batches, state)
    elif args.command == "open-folder":
        open_folder(batch)
        print_batch(args.platform, args.batch_index, len(batches), batch)
    elif args.command == "assist":
        assist_upload(
            platform=args.platform,
            batch_index=args.batch_index,
            total_batches=len(batches),
            batch=batch,
            url=args.url or PLATFORM_URLS[args.platform],
            headless=args.headless,
        )
    elif args.command == "fill-cdp":
        fill_existing_chrome(
            platform=args.platform,
            batch_index=args.batch_index,
            total_batches=len(batches),
            batch=batch,
            cdp_url=args.cdp_url,
        )
    elif args.command == "log":
        append_log(log, args.platform, args.batch_index, batch, args.post_url, args.status)
        print(f"logged batch {args.batch_index} to {log}")


def load_items(manifest: Path) -> list[UploadItem]:
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    items: list[UploadItem] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            path = Path(str(row.get("upload_path") or ""))
            if not path.exists():
                continue
            items.append(
                UploadItem(
                    pair_id=str(row.get("pair_id") or path.stem),
                    label=str(row.get("label") or "unknown"),
                    path=path,
                )
            )
    if not items:
        raise SystemExit(f"no existing images found in manifest: {manifest}")
    return items


def make_batches(items: list[UploadItem], batch_size: int) -> list[list[UploadItem]]:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    # Interleave labels so each batch carries both positive and real examples.
    by_label: dict[str, list[UploadItem]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    ordered: list[UploadItem] = []
    labels = sorted(by_label)
    cursors = {label: 0 for label in labels}
    while len(ordered) < len(items):
        advanced = False
        for label in labels:
            cursor = cursors[label]
            if cursor < len(by_label[label]):
                ordered.append(by_label[label][cursor])
                cursors[label] += 1
                advanced = True
        if not advanced:
            break
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def write_state(state: Path, platform: str, batch_size: int, batches: list[list[UploadItem]]) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "platform": platform,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batches": [
            {
                "batch_index": index,
                "count": len(batch),
                "pair_ids": [item.pair_id for item in batch],
                "paths": [str(item.path) for item in batch],
            }
            for index, batch in enumerate(batches, start=1)
        ],
    }
    state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_plan(platform: str, batch_size: int, batches: list[list[UploadItem]], state: Path) -> None:
    print(f"platform={platform} batch_size={batch_size} batches={len(batches)} state={state}")
    for index, batch in enumerate(batches, start=1):
        counts: dict[str, int] = {}
        for item in batch:
            counts[item.label] = counts.get(item.label, 0) + 1
        print(f"batch {index:02d}: {len(batch)} images {counts}")
    print("\nNext:")
    print(f"  python tools\\platform_upload_assistant.py open-folder --platform {platform} --batch-index 1")
    print(f"  python tools\\platform_upload_assistant.py assist --platform {platform} --batch-index 1")


def print_batch(platform: str, batch_index: int, total_batches: int, batch: list[UploadItem]) -> None:
    print(f"{platform} batch {batch_index}/{total_batches}:")
    print("caption:")
    print(f"platform transcode eval {platform} batch {batch_index:02d}")
    print("images:")
    for item in batch:
        print(f"  {item.path}")


def open_folder(batch: list[UploadItem]) -> None:
    folder = batch[0].path.parent
    if sys.platform.startswith("win"):
        os.startfile(str(folder))  # type: ignore[attr-defined]
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(folder)], check=False)


def append_log(log: Path, platform: str, batch_index: int, batch: list[UploadItem], post_url: str, status: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    exists = log.exists()
    with log.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["created_at", "platform", "batch_index", "count", "pair_ids", "post_url", "status"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "platform": platform,
                "batch_index": batch_index,
                "count": len(batch),
                "pair_ids": ";".join(item.pair_id for item in batch),
                "post_url": post_url,
                "status": status,
            }
        )


def assist_upload(
    *,
    platform: str,
    batch_index: int,
    total_batches: int,
    batch: list[UploadItem],
    url: str,
    headless: bool,
) -> None:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    user_data_dir = ROOT / "platform_eval" / f"browser_profile_{platform}"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    print_batch(platform, batch_index, total_batches, batch)
    print("\nOpening browser. Log in manually if needed; the script will not bypass verification.")
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(user_data_dir),
            headless=headless,
            accept_downloads=True,
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        print("Browser opened. Navigate to the post composer if this is not already it.")
        input("Press Enter here after the composer is visible and any login/verification is complete...")
        file_inputs = page.locator("input[type='file']")
        try:
            count = file_inputs.count()
        except Exception:
            count = 0
        paths = [str(item.path) for item in batch]
        if count <= 0:
            print("No file input found. Please click the platform image upload button manually.")
            input("After the file picker/upload input appears or upload area is ready, press Enter to retry...")
            try:
                count = file_inputs.count()
            except Exception:
                count = 0
        if count > 0:
            try:
                file_inputs.first.set_input_files(paths, timeout=60000)
                print(f"Set {len(paths)} files on the first upload input.")
            except PlaywrightTimeoutError:
                print("Timed out while setting files. Use the printed paths for manual selection.")
            except Exception as exc:
                print(f"Could not set files automatically: {type(exc).__name__}: {exc}")
        else:
            print("Still no file input found; use manual selection with the printed paths.")
        caption = f"platform transcode eval {platform} batch {batch_index:02d}"
        for selector in [
            "textarea",
            "[contenteditable='true']",
            "input[type='text']",
        ]:
            locator = page.locator(selector)
            try:
                if locator.count() > 0:
                    locator.first.fill(caption, timeout=3000)
                    print(f"Filled caption via selector {selector!r}.")
                    break
            except Exception:
                continue
        print("\nReview the post manually. Publish it as private/only-visible-to-self if the platform supports it.")
        input("After publishing or cancelling, press Enter to close this browser context...")
        context.close()


def fill_existing_chrome(
    *,
    platform: str,
    batch_index: int,
    total_batches: int,
    batch: list[UploadItem],
    cdp_url: str,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    caption = f"platform transcode eval {platform} batch {batch_index:02d}"
    paths = [str(item.path) for item in batch]
    print_batch(platform, batch_index, total_batches, batch)
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if not contexts or not contexts[0].pages:
            browser.close()
            raise SystemExit(f"no pages available from {cdp_url}")
        page = contexts[0].pages[0]
        if platform == "xhs":
            page.goto(PLATFORM_URLS["xhs"], wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)
            file_input = page.locator(
                "input[type='file'][accept*='jpg'], "
                "input[type='file'][accept*='jpeg'], "
                "input[type='file'][accept*='png'], "
                "input[type='file'][accept*='webp']"
            )
        else:
            file_input = page.locator("input[type='file']")
        if file_input.count() <= 0:
            browser.close()
            raise SystemExit("no file input found; open/click the image upload area and retry")
        file_input.first.set_input_files(paths, timeout=120000)
        caption_selectors = [
            "input[placeholder*='标题']",
            "textarea[placeholder*='标题']",
            "textarea[placeholder*='正文']",
            "textarea[placeholder*='描述']",
            "textarea",
            "[contenteditable='true']",
            "input[type='text']",
        ]
        for selector in caption_selectors:
            locator = page.locator(selector)
            try:
                if locator.count() > 0:
                    locator.first.fill(caption, timeout=3000)
                    break
            except Exception:
                continue
        print(f"filled caption and set {len(paths)} files on {page.url}")
        print("Review visibility and publish manually. This command does not click the publish button.")
        browser.close()


if __name__ == "__main__":
    main()
