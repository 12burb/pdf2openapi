"""Entry points for the in-browser build (Pyodide).

The page runs this package in the visitor's browser: pdf.js supplies the words of each page
and nothing is uploaded anywhere. PyMuPDF, OCR and AI refinement are not available there.
"""

from __future__ import annotations

import json
from typing import Any

from . import pdf
from .diff import diff_specs
from .extract import ai_hint, extract_document
from .output import dump


def document_from_words(name: str, title: str, pages: list[list[list[Any]]]) -> pdf.Document:
    """Build a Document from per-page word boxes: [[x0, y0, x1, y1, text], ...] per page."""
    doc = pdf.Document(path=name, title=(title or "").strip())
    for number, words in enumerate(pages, start=1):
        doc.pages.append(pdf.Page(number=number, text=pdf.lines_from_words(words)))
    return doc


def extract_json(name: str, title: str, pages_json: str, fmt: str = "yaml") -> str:
    """Extract from pdf.js words (JSON) and return spec, text, warnings and tip as JSON."""
    result = extract_document(document_from_words(name, title, json.loads(pages_json)))
    return json.dumps(
        {
            "spec": result.spec,
            "text": dump(result.spec, fmt),
            "warnings": result.warnings,
            "tip": ai_hint(result.spec),
            "count": sum(len(item) for item in result.spec["paths"].values()),
        }
    )


def diff_json(old_spec_json: str, new_spec_json: str) -> str:
    report = diff_specs(json.loads(old_spec_json), json.loads(new_spec_json))
    return json.dumps({"markdown": report.to_markdown(), **report.to_dict()})
