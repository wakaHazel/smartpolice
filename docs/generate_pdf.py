from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import generate_project_report_docx
from semifinal_paper_content import ROOT


DOCX_PATH = ROOT / "output" / "docx" / "semifinal_document.docx"
OUTPUT = ROOT / "output" / "pdf" / "semifinal_document.pdf"
LO_PATH = ROOT / "tools" / "LibreOffice" / "program" / "soffice.exe"


def _export_with_word(docx_path: Path, pdf_path: Path) -> bool:
    script = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
  $doc = $word.Documents.Open('{docx_path}')
  $doc.ExportAsFixedFormat('{pdf_path}', 17)
  $doc.Close($false)
}} finally {{
  $word.Quit()
}}
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    return pdf_path.exists() and pdf_path.stat().st_size > 0


def _export_with_libreoffice(docx_path: Path, pdf_path: Path) -> bool:
    soffice = LO_PATH if LO_PATH.exists() else shutil.which("soffice")
    if not soffice:
        return False
    out_dir = pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                str(soffice),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(out_dir),
                str(docx_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    converted = out_dir / f"{docx_path.stem}.pdf"
    if converted.exists() and converted != pdf_path:
        if pdf_path.exists():
            pdf_path.unlink()
        converted.rename(pdf_path)
    return pdf_path.exists() and pdf_path.stat().st_size > 0


def main() -> None:
    generate_project_report_docx.main()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    if _export_with_word(DOCX_PATH, OUTPUT):
        print(OUTPUT)
        return
    if _export_with_libreoffice(DOCX_PATH, OUTPUT):
        print(OUTPUT)
        return
    raise RuntimeError("DOCX 已生成，但未能通过 Word 或 LibreOffice 导出 PDF。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line reporting
        print(f"PDF export failed: {exc}", file=sys.stderr)
        raise
