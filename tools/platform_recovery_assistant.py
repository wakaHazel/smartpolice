from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "platform_eval" / "platform_upload_log.csv"
DEFAULT_RETURNED_ROOT = ROOT / "platform_eval" / "returned"
DEFAULT_AUDIT = ROOT / "platform_eval" / "returned" / "recovery_audit.json"


@dataclass(frozen=True)
class BatchRecord:
    platform: str
    batch_index: int
    pair_ids: list[str]


@dataclass(frozen=True)
class DownloadedImage:
    url: str
    path: Path
    sha256: str
    byte_count: int
    width: int
    height: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover platform-rendered images from logged Weibo/Xiaohongshu experiment posts."
    )
    parser.add_argument("command", choices=("recover-weibo", "recover-weibo-screenshots", "recover-xhs", "inspect-xhs"))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--returned-root", default=str(DEFAULT_RETURNED_ROOT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9223")
    parser.add_argument("--batch", type=int, default=0, help="Recover only one batch; 0 means all visible batches.")
    parser.add_argument("--quality", default="large,mw2000,mw1024,mw690,orj360")
    args = parser.parse_args()

    records = load_records(Path(args.log))
    if args.command == "recover-weibo":
        recover_weibo(
            records=[record for record in records if record.platform == "weibo"],
            cdp_url=args.cdp_url,
            returned_root=Path(args.returned_root),
            audit_path=Path(args.audit),
            only_batch=args.batch,
            quality_candidates=[item.strip() for item in args.quality.split(",") if item.strip()],
        )
    elif args.command == "recover-weibo-screenshots":
        recover_weibo_screenshots(
            records=[record for record in records if record.platform == "weibo"],
            cdp_url=args.cdp_url,
            returned_root=Path(args.returned_root),
            audit_path=Path(args.audit),
            only_batch=args.batch,
        )
    elif args.command == "recover-xhs":
        recover_xhs(
            records=[record for record in records if record.platform == "xhs"],
            cdp_url=args.cdp_url,
            returned_root=Path(args.returned_root),
            audit_path=Path(args.audit),
            only_batch=args.batch,
        )
    elif args.command == "inspect-xhs":
        inspect_xhs(cdp_url=args.cdp_url)


def load_records(path: Path) -> list[BatchRecord]:
    if not path.is_file():
        raise SystemExit(f"log not found: {path}")
    records: list[BatchRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            pair_ids = [item for item in str(row.get("pair_ids") or "").split(";") if item]
            if not pair_ids:
                continue
            records.append(
                BatchRecord(
                    platform=str(row.get("platform") or ""),
                    batch_index=int(row.get("batch_index") or 0),
                    pair_ids=pair_ids,
                )
            )
    return records


def recover_weibo(
    *,
    records: list[BatchRecord],
    cdp_url: str,
    returned_root: Path,
    audit_path: Path,
    only_batch: int,
    quality_candidates: list[str],
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    target_dir = returned_root / "weibo_download"
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = {record.batch_index: record for record in records if only_batch in {0, record.batch_index}}
    recovered: list[dict[str, object]] = []
    missing_batches: list[int] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        page.goto("https://weibo.com/u/7789574150", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        visible = collect_weibo_visible_batches(page)
        # The personal feed is virtualized; scan down to catch all batches.
        for _ in range(24):
            if set(expected).issubset(set(visible)):
                break
            page.mouse.wheel(0, 700)
            page.wait_for_timeout(900)
            visible.update(collect_weibo_visible_batches(page))
        browser.close()

    for batch_index, record in sorted(expected.items()):
        urls = visible.get(batch_index, [])
        if len(urls) < len(record.pair_ids):
            missing_batches.append(batch_index)
        for position, pair_id in enumerate(record.pair_ids):
            if position >= len(urls):
                recovered.append(
                    {
                        "platform": "weibo",
                        "batch_index": batch_index,
                        "pair_id": pair_id,
                        "position": position + 1,
                        "status": "missing_url",
                    }
                )
                continue
            downloaded = download_best_weibo_image(
                urls[position],
                target_dir / f"{pair_id}_weibo_download.jpg",
                quality_candidates,
            )
            recovered.append(
                {
                    "platform": "weibo",
                    "batch_index": batch_index,
                    "pair_id": pair_id,
                    "position": position + 1,
                    "status": "downloaded",
                    "url": downloaded.url,
                    "path": str(downloaded.path),
                    "sha256": downloaded.sha256,
                    "byte_count": downloaded.byte_count,
                    "width": downloaded.width,
                    "height": downloaded.height,
                }
            )
    audit = {
        "command": "recover-weibo",
        "cdp_url": cdp_url,
        "expected_batches": sorted(expected),
        "visible_batches": {str(key): len(value) for key, value in sorted(visible.items())},
        "missing_batches": missing_batches,
        "downloaded_count": sum(1 for row in recovered if row.get("status") == "downloaded"),
        "rows": recovered,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("expected_batches", "visible_batches", "missing_batches", "downloaded_count")}, ensure_ascii=False, indent=2))
    print(f"wrote {audit_path}")


def collect_weibo_visible_batches(page: Any) -> dict[int, list[str]]:
    raw = page.evaluate(
        r"""
() => {
  const out = {};
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while (node = walker.nextNode()) {
    const text = node.nodeValue || "";
    const match = text.match(/platform transcode eval weibo batch (\d{2})/);
    if (!match) continue;
    let element = node.parentElement;
    for (let level = 0; level < 8 && element; level += 1, element = element.parentElement) {
      if (!element.classList || !element.classList.contains("wbpro-feed-content")) continue;
      const urls = [...element.querySelectorAll("img")]
        .map((img) => img.currentSrc || img.src || "")
        .filter((src) => /sinaimg\.cn\/(orj|mw|large|bmiddle|thumb)/.test(src))
        .filter((src) => !/crop\./.test(src))
        .filter((src, index, arr) => arr.indexOf(src) === index);
      if (urls.length) {
        out[String(parseInt(match[1], 10))] = urls;
      }
      break;
    }
  }
  return out;
}
"""
    )
    result: dict[int, list[str]] = {}
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        if isinstance(value, list):
            result[int(key)] = [str(item) for item in value if item]
    return result


def recover_weibo_screenshots(
    *,
    records: list[BatchRecord],
    cdp_url: str,
    returned_root: Path,
    audit_path: Path,
    only_batch: int,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    target_dir = returned_root / "weibo_screenshot"
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = {record.batch_index: record for record in records if only_batch in {0, record.batch_index}}
    recovered: list[dict[str, object]] = []
    seen_batches: set[int] = set()
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        page.goto("https://weibo.com/u/7789574150", wait_until="domcontentloaded", timeout=60000)
        page.set_viewport_size({"width": 1440, "height": 1400})
        page.wait_for_timeout(2500)
        for _ in range(28):
            visible_batch_indices = sorted(collect_weibo_visible_batches(page), reverse=True)
            for batch_index in visible_batch_indices:
                if batch_index not in expected or batch_index in seen_batches:
                    continue
                rows = screenshot_weibo_batch(page, expected[batch_index], target_dir)
                recovered.extend(rows)
                seen_batches.add(batch_index)
            if set(expected).issubset(seen_batches):
                break
            page.mouse.wheel(0, 700)
            page.wait_for_timeout(900)
        browser.close()
    missing_batches = sorted(set(expected) - seen_batches)
    audit = {
        "command": "recover-weibo-screenshots",
        "cdp_url": cdp_url,
        "expected_batches": sorted(expected),
        "recovered_batches": sorted(seen_batches),
        "missing_batches": missing_batches,
        "screenshot_count": sum(1 for row in recovered if row.get("status") == "screenshot"),
        "rows": recovered,
        "note": "Screenshots are taken from image elements rendered in the Weibo personal feed grid.",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("expected_batches", "recovered_batches", "missing_batches", "screenshot_count")}, ensure_ascii=False, indent=2))
    print(f"wrote {audit_path}")


def screenshot_weibo_batch(page: Any, record: BatchRecord, target_dir: Path) -> list[dict[str, object]]:
    handle = page.evaluate_handle(
        r"""
(batchIndex) => {
  const pattern = new RegExp(`platform transcode eval weibo batch ${String(batchIndex).padStart(2, "0")}`);
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while (node = walker.nextNode()) {
    if (!pattern.test(node.nodeValue || "")) continue;
    let element = node.parentElement;
    for (let level = 0; level < 8 && element; level += 1, element = element.parentElement) {
      if (element.classList && element.classList.contains("wbpro-feed-content")) return element;
    }
  }
  return null;
}
""",
        record.batch_index,
    )
    element = handle.as_element()
    if element is None:
        return [
            {
                "platform": "weibo",
                "batch_index": record.batch_index,
                "status": "missing_container",
            }
        ]
    image_handles = element.query_selector_all("img")
    usable = []
    for image in image_handles:
        meta = image.evaluate(
            r"""
(img) => ({
  src: img.currentSrc || img.src || "",
  width: img.naturalWidth || img.width || 0,
  height: img.naturalHeight || img.height || 0,
  clientWidth: img.clientWidth || 0,
  clientHeight: img.clientHeight || 0
})
"""
        )
        if not isinstance(meta, dict):
            continue
        src = str(meta.get("src") or "")
        if not re.search(r"sinaimg\.cn/(orj|mw|large|bmiddle|thumb)", src):
            continue
        if "crop." in src:
            continue
        if int(meta.get("clientWidth") or 0) < 30 or int(meta.get("clientHeight") or 0) < 30:
            continue
        usable.append((image, meta))
    rows: list[dict[str, object]] = []
    for index, pair_id in enumerate(record.pair_ids):
        if index >= len(usable):
            rows.append(
                {
                    "platform": "weibo",
                    "batch_index": record.batch_index,
                    "pair_id": pair_id,
                    "position": index + 1,
                    "status": "missing_image_element",
                }
            )
            continue
        image, meta = usable[index]
        path = target_dir / f"{pair_id}_weibo_screenshot.png"
        image.screenshot(path=str(path), timeout=30000)
        width, height = image_size(path.read_bytes())
        rows.append(
            {
                "platform": "weibo",
                "batch_index": record.batch_index,
                "pair_id": pair_id,
                "position": index + 1,
                "status": "screenshot",
                "path": str(path),
                "src": meta.get("src"),
                "rendered_width": meta.get("clientWidth"),
                "rendered_height": meta.get("clientHeight"),
                "screenshot_width": width,
                "screenshot_height": height,
            }
        )
    return rows


def download_best_weibo_image(url: str, target: Path, quality_candidates: list[str]) -> DownloadedImage:
    attempts = weibo_quality_urls(url, quality_candidates)
    best: DownloadedImage | None = None
    failures: list[str] = []
    for attempt in attempts:
        try:
            data = fetch_bytes(attempt)
            if len(data) < 1000:
                failures.append(f"{attempt}: too small")
                continue
            width, height = image_size(data)
            suffix = suffix_from_url(attempt, data)
            candidate_target = target.with_suffix(suffix)
            sha256 = hashlib.sha256(data).hexdigest()
            downloaded = DownloadedImage(
                url=attempt,
                path=candidate_target,
                sha256=sha256,
                byte_count=len(data),
                width=width,
                height=height,
            )
            if best is None or (downloaded.width * downloaded.height, downloaded.byte_count) > (
                best.width * best.height,
                best.byte_count,
            ):
                best = downloaded
                candidate_target.write_bytes(data)
        except Exception as exc:
            failures.append(f"{attempt}: {type(exc).__name__}")
    if best is None:
        raise RuntimeError(f"could not download {url}: {failures[:3]}")
    # Remove lower-quality files with the same stem that may have been written.
    for sibling in best.path.parent.glob(f"{target.stem}.*"):
        if sibling != best.path:
            sibling.unlink(missing_ok=True)
    return best


def weibo_quality_urls(url: str, quality_candidates: list[str]) -> list[str]:
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    urls: list[str] = []
    for quality in quality_candidates:
        if len(parts) >= 3:
            next_parts = parts[:]
            next_parts[1] = quality
            urls.append(urlunparse(parsed._replace(path="/".join(next_parts), query="")))
    urls.append(urlunparse(parsed._replace(query="")))
    deduped: list[str] = []
    for item in urls:
        if item not in deduped:
            deduped.append(item)
    return deduped


def fetch_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 platform-transcode-eval",
            "Referer": "https://weibo.com/",
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def fetch_bytes_with_referer(url: str, referer: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 platform-transcode-eval",
            "Referer": referer,
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def image_size(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        return image.size


def suffix_from_url(url: str, data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    guessed = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "")
    if guessed:
        return guessed
    return ".jpg"


def inspect_xhs(cdp_url: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        page.goto("https://creator.xiaohongshu.com/creator/notes", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        payload = page.evaluate(
            r"""
() => {
  const text = document.body.innerText || "";
  const imgs = [...document.querySelectorAll("img")].map((img) => ({
    src: img.currentSrc || img.src || "",
    alt: img.alt || "",
    w: img.naturalWidth || img.width || 0,
    h: img.naturalHeight || img.height || 0,
    cw: img.clientWidth || 0,
    ch: img.clientHeight || 0,
  })).filter((item) => item.src);
  const anchors = [...document.querySelectorAll("a")].map((a) => ({
    text: (a.innerText || "").slice(0, 120),
    href: a.href || "",
  })).filter((item) => item.text || item.href);
  return {url: location.href, title: document.title, text: text.slice(0, 4000), imgs: imgs.slice(0, 80), anchors: anchors.slice(0, 80)};
}
"""
        )
        browser.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def recover_xhs(
    *,
    records: list[BatchRecord],
    cdp_url: str,
    returned_root: Path,
    audit_path: Path,
    only_batch: int,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit(f"Python playwright is not available: {type(exc).__name__}: {exc}") from exc

    target_dir = returned_root / "xhs_download"
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = {record.batch_index: record for record in records if only_batch in {0, record.batch_index}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        with page.expect_response(lambda response: "creator/note/user/posted" in response.url, timeout=20000) as response_info:
            page.goto("https://creator.xiaohongshu.com/new/note-manager", wait_until="domcontentloaded", timeout=60000)
        response = response_info.value
        payload = response.json()
        browser.close()

    notes = extract_xhs_notes(payload)
    batch_to_urls = map_xhs_batches(notes, expected)
    recovered: list[dict[str, object]] = []
    missing_batches: list[int] = []
    for batch_index, record in sorted(expected.items()):
        urls = batch_to_urls.get(batch_index, [])
        if len(urls) < len(record.pair_ids):
            missing_batches.append(batch_index)
        for position, pair_id in enumerate(record.pair_ids):
            if position >= len(urls):
                recovered.append(
                    {
                        "platform": "xhs",
                        "batch_index": batch_index,
                        "pair_id": pair_id,
                        "position": position + 1,
                        "status": "missing_url",
                    }
                )
                continue
            downloaded = download_xhs_image(urls[position], target_dir / f"{pair_id}_xhs_download.jpg")
            recovered.append(
                {
                    "platform": "xhs",
                    "batch_index": batch_index,
                    "pair_id": pair_id,
                    "position": position + 1,
                    "status": "downloaded",
                    "url": downloaded.url,
                    "path": str(downloaded.path),
                    "sha256": downloaded.sha256,
                    "byte_count": downloaded.byte_count,
                    "width": downloaded.width,
                    "height": downloaded.height,
                }
            )
    audit = {
        "command": "recover-xhs",
        "cdp_url": cdp_url,
        "expected_batches": sorted(expected),
        "visible_batches": {str(key): len(value) for key, value in sorted(batch_to_urls.items())},
        "missing_batches": missing_batches,
        "downloaded_count": sum(1 for row in recovered if row.get("status") == "downloaded"),
        "note_mapping": [
            {
                "batch_index": item.get("batch_index"),
                "note_id": item.get("id"),
                "title": item.get("display_title"),
                "time": item.get("time"),
                "image_count": len(item.get("images_list", [])) if isinstance(item.get("images_list"), list) else 0,
                "permission_msg": item.get("permission_msg"),
            }
            for item in notes
            if item.get("batch_index") in expected
        ],
        "rows": recovered,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("expected_batches", "visible_batches", "missing_batches", "downloaded_count")}, ensure_ascii=False, indent=2))
    print(f"wrote {audit_path}")


def extract_xhs_notes(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    notes = data.get("notes")
    if not isinstance(notes, list):
        return []
    return [note for note in notes if isinstance(note, dict)]


def map_xhs_batches(notes: list[dict[str, object]], expected: dict[int, BatchRecord]) -> dict[int, list[str]]:
    # Batch 1 and 2 kept titles. Later batches were posted with empty titles, so
    # use the upload chronology recorded in note-manager: batch 7 is newest.
    xhs_notes = [
        note
        for note in notes
        if (str(note.get("display_title") or "").startswith("platform transcode eval xhs batch") or is_xhs_experiment_note(note))
    ]
    xhs_notes.sort(key=lambda note: str(note.get("time") or ""))
    for index, note in enumerate(xhs_notes, start=1):
        title = str(note.get("display_title") or "")
        match = re.search(r"batch\s+(\d{2})", title)
        note["batch_index"] = int(match.group(1)) if match else index
    mapped: dict[int, list[str]] = {}
    for note in xhs_notes:
        batch_index = int(note.get("batch_index") or 0)
        if batch_index not in expected:
            continue
        images = note.get("images_list")
        if not isinstance(images, list):
            continue
        urls = []
        for image in images:
            if not isinstance(image, dict):
                continue
            url = str(image.get("url") or "")
            if url:
                urls.append(url.replace("http://", "https://", 1))
        mapped[batch_index] = urls
    return mapped


def is_xhs_experiment_note(note: dict[str, object]) -> bool:
    images = note.get("images_list")
    if not isinstance(images, list):
        return False
    count = len(images)
    if count not in {6, 9}:
        return False
    time = str(note.get("time") or "")
    return time.startswith("2026-06-14 14:")


def download_xhs_image(url: str, target: Path) -> DownloadedImage:
    candidates = xhs_quality_urls(url)
    best: DownloadedImage | None = None
    failures: list[str] = []
    for candidate in candidates:
        try:
            data = fetch_bytes_with_referer(candidate, "https://creator.xiaohongshu.com/new/note-manager")
            if len(data) < 1000:
                failures.append(f"{candidate}: too small")
                continue
            width, height = image_size(data)
            suffix = suffix_from_url(candidate, data)
            candidate_target = target.with_suffix(suffix)
            downloaded = DownloadedImage(
                url=candidate,
                path=candidate_target,
                sha256=hashlib.sha256(data).hexdigest(),
                byte_count=len(data),
                width=width,
                height=height,
            )
            if best is None or (downloaded.width * downloaded.height, downloaded.byte_count) > (
                best.width * best.height,
                best.byte_count,
            ):
                best = downloaded
                candidate_target.write_bytes(data)
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}")
    if best is None:
        raise RuntimeError(f"could not download {url}: {failures[:3]}")
    for sibling in best.path.parent.glob(f"{target.stem}.*"):
        if sibling != best.path:
            sibling.unlink(missing_ok=True)
    return best


def xhs_quality_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    base = urlunparse(parsed._replace(query=""))
    queries = [
        "imageView2/2/w/2160/format/jpg&origin=0",
        "imageView2/2/w/1080/format/jpg&origin=0",
        parsed.query,
        "",
    ]
    urls = [urlunparse(parsed._replace(query=query)) if query else base for query in queries]
    deduped: list[str] = []
    for item in urls:
        if item not in deduped:
            deduped.append(item)
    return deduped


if __name__ == "__main__":
    main()
