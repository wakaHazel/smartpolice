from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
import ipaddress
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile
import httpx

from app.models import CaseAsset, WebEvidenceSnapshot
from app.storage import save_case_asset, save_web_snapshot


DATA_ROOT = Path(os.getenv("SMARTPOLICE_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
UPLOAD_ROOT = DATA_ROOT / "uploads"
SNAPSHOT_ROOT = DATA_ROOT / "snapshots"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_HTML_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


class EvidenceError(ValueError):
    status_code = 400


class EvidenceValidationError(EvidenceError):
    status_code = 422


class UrlCaptureError(EvidenceError):
    status_code = 502


def public_asset_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(DATA_ROOT.resolve())
    except ValueError:
        raise EvidenceValidationError("文件不在证据存储目录内。") from None
    return str(relative).replace("\\", "/")


async def save_uploaded_asset(case_id: str, file: UploadFile) -> CaseAsset:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise EvidenceValidationError("只支持 PNG、JPEG、WebP 图片或截图。")
    raw = await file.read()
    if not raw:
        raise EvidenceValidationError("上传文件为空。")
    if len(raw) > MAX_IMAGE_BYTES:
        raise EvidenceValidationError("图片超过 10MB 上限。")

    digest = sha256(raw).hexdigest()
    extension = _extension_for_content_type(file.content_type)
    asset_id = f"asset-{uuid4().hex[:12]}"
    case_dir = UPLOAD_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{asset_id}{extension}"
    storage_path = case_dir / stored_name
    storage_path.write_bytes(raw)
    width, height = _image_size(raw, file.content_type)
    asset = CaseAsset(
        id=asset_id,
        case_id=case_id,
        filename=file.filename or stored_name,
        content_type=file.content_type,
        size_bytes=len(raw),
        width=width,
        height=height,
        sha256=digest,
        storage_path=str(storage_path),
        preview_url=f"/evidence/files/{public_asset_path(str(storage_path))}",
        created_at=datetime.now(UTC).isoformat(),
    )
    save_case_asset(asset)
    return asset


def capture_url(case_id: str, url: str) -> WebEvidenceSnapshot:
    normalized = url.strip()
    _validate_public_url(normalized)
    snapshot_id = f"snapshot-{uuid4().hex[:12]}"
    case_dir = SNAPSHOT_ROOT / case_id / snapshot_id
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "SmartPoliceEvidenceBot/1.0"},
        ) as client:
            response = client.get(normalized)
            response.raise_for_status()
            body = response.content[: MAX_HTML_BYTES + 1]
    except httpx.HTTPError as exc:
        raise UrlCaptureError(f"URL 抓取失败：{exc}") from exc

    if len(body) > MAX_HTML_BYTES:
        raise EvidenceValidationError("网页内容超过 5MB 上限。")

    final_url = str(response.url)
    _validate_public_url(final_url)
    html = body.decode(response.encoding or "utf-8", errors="replace")
    digest = sha256(body).hexdigest()
    title, text = extract_html_text(html)
    html_path = case_dir / "page.html"
    text_path = case_dir / "content.txt"
    screenshot_path = case_dir / "screenshot.png"
    html_path.write_text(html, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    screenshot_ok = _capture_screenshot(final_url, screenshot_path)
    snapshot = WebEvidenceSnapshot(
        id=snapshot_id,
        case_id=case_id,
        requested_url=normalized,
        final_url=final_url,
        title=title or final_url,
        text=text,
        sha256=digest,
        status="captured" if screenshot_ok else "captured_without_screenshot",
        error=None if screenshot_ok else "Playwright 截图不可用，已保存 HTML 与正文快照。",
        html_path=str(html_path),
        text_path=str(text_path),
        screenshot_path=str(screenshot_path) if screenshot_ok else None,
        screenshot_url=(
            f"/evidence/files/{public_asset_path(str(screenshot_path))}"
            if screenshot_ok
            else None
        ),
        created_at=datetime.now(UTC).isoformat(),
    )
    save_web_snapshot(snapshot)
    return snapshot


def extract_html_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(part.strip() for part in parser.text_parts if part.strip())
    return title[:240], " ".join(text.split())[:12000]


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise EvidenceValidationError("URL 只允许 http/https。")
    if not parsed.hostname:
        raise EvidenceValidationError("URL 缺少 hostname。")
    hostname = parsed.hostname.strip().lower()
    if hostname in {"localhost"} or hostname.endswith(".localhost"):
        raise EvidenceValidationError("URL 不允许指向本机地址。")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise EvidenceValidationError("URL 域名无法解析。") from exc
    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise EvidenceValidationError("URL 不允许指向内网、本机或保留地址。")


def _extension_for_content_type(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }[content_type]


def _image_size(raw: bytes, content_type: str) -> tuple[int | None, int | None]:
    if content_type == "image/png" and raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if content_type == "image/jpeg":
        return _jpeg_size(raw)
    if content_type == "image/webp" and len(raw) >= 30 and raw[:4] == b"RIFF":
        return _webp_size(raw)
    return None, None


def _jpeg_size(raw: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index + 9 < len(raw):
        if raw[index] != 0xFF:
            index += 1
            continue
        marker = raw[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[index : index + 2], "big")
        if marker in range(0xC0, 0xCF) and marker not in {0xC4, 0xC8, 0xCC}:
            if index + 7 <= len(raw):
                height = int.from_bytes(raw[index + 3 : index + 5], "big")
                width = int.from_bytes(raw[index + 5 : index + 7], "big")
                return width, height
        index += max(segment_length, 2)
    return None, None


def _webp_size(raw: bytes) -> tuple[int | None, int | None]:
    if raw[12:16] == b"VP8X" and len(raw) >= 30:
        width = int.from_bytes(raw[24:27], "little") + 1
        height = int.from_bytes(raw[27:30], "little") + 1
        return width, height
    return None, None


def _capture_screenshot(url: str, output_path: Path) -> bool:
    script = (
        "from pathlib import Path\n"
        "from playwright.sync_api import sync_playwright\n"
        f"url = {url!r}\n"
        f"out = Path({str(output_path)!r})\n"
        "with sync_playwright() as p:\n"
        "    browser = p.chromium.launch(headless=True)\n"
        "    page = browser.new_page(viewport={\"width\": 1365, \"height\": 900})\n"
        "    page.goto(url, wait_until=\"networkidle\", timeout=15000)\n"
        "    page.screenshot(path=str(out), full_page=True)\n"
        "    browser.close()\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            timeout=25,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        elif len(text) > 1:
            self.text_parts.append(text)
