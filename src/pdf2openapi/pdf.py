"""Read PDF pages as text, using OCR for screenshots and scanned pages."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

MIN_IMAGE_WIDTH = 150
MIN_IMAGE_HEIGHT = 40
SCANNED_PAGE_CHARS = 20


@dataclass
class Page:
    number: int
    text: str
    ocr_text: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(t for t in (self.text, self.ocr_text) if t.strip())


@dataclass
class Document:
    path: str
    title: str
    pages: list[Page] = field(default_factory=list)
    ocr_used: bool = False


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr_png(png: bytes) -> str:
    try:
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=png,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else ""


def _page_lines(page: pymupdf.Page) -> str:
    """Rebuild visual lines from word boxes so table rows stay on one line.

    Wide horizontal gaps become three spaces, which the extractor treats as a column break.
    """
    words = page.get_text("words")
    rows: list[list] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        mid = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - mid) <= 3:
            rows[-1][1].append(w)
        else:
            rows.append([mid, [w]])
    lines = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[0])
        parts = [ws[0][4]]
        for prev, cur in zip(ws, ws[1:], strict=False):
            height = max(cur[3] - cur[1], 1)
            parts.append("   " if cur[0] - prev[2] > 0.9 * height else " ")
            parts.append(cur[4])
        lines.append("".join(parts))
    return "\n".join(lines)


def _ocr_images(doc: pymupdf.Document, page: pymupdf.Page) -> str:
    texts = []
    for img in page.get_images(full=True):
        try:
            pix = pymupdf.Pixmap(doc, img[0])
        except (RuntimeError, ValueError):
            continue
        if pix.width < MIN_IMAGE_WIDTH or pix.height < MIN_IMAGE_HEIGHT:
            continue
        if pix.alpha or pix.n - pix.alpha >= 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix, 0)
        text = _ocr_png(pix.tobytes("png"))
        if text.strip():
            texts.append(text)
    return "\n".join(texts)


def load(path: str | Path, ocr: bool = True) -> Document:
    path = str(path)
    use_ocr = ocr and ocr_available()
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        result = Document(path=path, title=title.strip())
        for index, page in enumerate(doc):
            text = _page_lines(page)
            ocr_text = ""
            if use_ocr:
                if len(text.strip()) < SCANNED_PAGE_CHARS:
                    ocr_text = _ocr_png(page.get_pixmap(dpi=300).tobytes("png"))
                else:
                    ocr_text = _ocr_images(doc, page)
                result.ocr_used = result.ocr_used or bool(ocr_text.strip())
            result.pages.append(Page(number=index + 1, text=text, ocr_text=ocr_text))
    return result


def page_text(doc: Document, pages: list[int] | None = None) -> str:
    wanted = set(pages or [])
    chunks = []
    for page in doc.pages:
        if wanted and page.number not in wanted:
            continue
        chunks.append(f"--- page {page.number} ---\n{page.full_text}")
    return "\n\n".join(chunks)
