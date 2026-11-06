"""Read PDF pages as text, using OCR for screenshots and scanned pages."""

from __future__ import annotations

import re
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
    cells = [_cells(sorted(ws, key=lambda w: w[0])) for _, ws in rows]
    _join_wrapped_cells(cells)
    return "\n".join("   ".join(" ".join(w[4] for w in c) for c in row) for row in cells if row)


def _cells(ws: list) -> list[list]:
    out = [[ws[0]]]
    for prev, cur in zip(ws, ws[1:], strict=False):
        if cur[0] - prev[2] > 0.9 * max(cur[3] - cur[1], 1):
            out.append([cur])
        else:
            out[-1].append(cur)
    return out


METHOD_WORDS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
WRAPPED_RE = re.compile(r"^(?:https?://\S+)?/\S*(?:[/-]|\{[^}]*)$")
PATH_WORD_RE = re.compile(r"^(?:https?://\S+)?/[\w{}$\-.~%:/]*$")
CONTINUATION_RE = re.compile(r"[\w{}$\-.~%:/]+")


def _join_wrapped_cells(rows: list[list[list]]) -> None:
    """Rejoin URL paths that wrap inside a table cell ('POST /v1/{id}/' over 'orders').

    The rest of the path starts a cell on a following row, at the same x as the path's cell.
    """
    for r, row in enumerate(rows):
        for cell in row:
            for k, w in enumerate(cell):
                strong = bool(WRAPPED_RE.match(w[4]))
                weak = k == len(cell) - 1 and bool(PATH_WORD_RE.match(w[4]))
                if not (strong or weak):
                    continue
                if k and cell[k - 1][4] not in METHOD_WORDS and not cell[0][4].startswith("/"):
                    continue
                left, text, joined = cell[0][0], w[4], False
                for nxt in rows[r + 1 : r + 8]:
                    match = next((c for c in nxt if abs(c[0][0] - left) <= 3), None)
                    if not match:
                        continue
                    word = match[0][4]
                    if len(match) > 1 and "/" not in word:
                        break
                    if not _continues(text, word, strong, joined):
                        break
                    text, joined = text + word, True
                    match.pop(0)
                    if not match:
                        nxt.remove(match)
                    strong = bool(WRAPPED_RE.match(text))
                cell[k] = (*w[:4], text, *w[5:])


def _continues(path: str, word: str, strong: bool, joined: bool) -> bool:
    if not CONTINUATION_RE.fullmatch(word) or word.upper() in METHOD_WORDS:
        return False
    if not strong:
        return "/" in word and "://" not in word and (joined or not word.startswith("/"))
    if path.count("{") > path.count("}"):
        return True
    return word[0] in "{/$" or word[0].islower() or "/" in word or "{" in word


def _ocr_images(doc: pymupdf.Document, page: pymupdf.Page) -> str:
    texts = []
    for img in page.get_images(full=True):
        try:
            pix = pymupdf.Pixmap(doc, img[0])
        except (RuntimeError, ValueError):
            continue
        if pix.width < MIN_IMAGE_WIDTH or pix.height < MIN_IMAGE_HEIGHT:
            continue
        try:
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            if pix.n >= 4 or (pix.colorspace and pix.colorspace.n not in (1, 3)):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            png = pix.tobytes("png")
        except (RuntimeError, ValueError):
            continue
        text = _ocr_png(png)
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
