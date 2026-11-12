from __future__ import annotations

import json

import pymupdf

from pdf2openapi import browser
from pdf2openapi.extract import extract


def _words(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        return title, [[list(w[:5]) for w in page.get_text("words")] for page in doc]


def test_words_path_matches_pymupdf_path(vendor_pdf, vendor_pdf_v2):
    title, pages = _words(vendor_pdf)
    out = json.loads(browser.extract_json(vendor_pdf.name, title, json.dumps(pages)))
    expected = extract(vendor_pdf, ocr=False).spec
    assert out["spec"] == expected
    assert out["count"] == sum(len(v) for v in expected["paths"].values())
    assert out["text"].startswith("openapi: 3.0.3")

    title2, pages2 = _words(vendor_pdf_v2)
    new = json.loads(browser.extract_json(vendor_pdf_v2.name, title2, json.dumps(pages2)))
    report = json.loads(browser.diff_json(json.dumps(out["spec"]), json.dumps(new["spec"])))
    assert report["breaking_count"] >= 1
    assert "BREAKING" in report["markdown"]


def test_browser_module_does_not_need_pymupdf_or_httpx(monkeypatch):
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pymupdf'] = None; sys.modules['httpx'] = None; "
        "import pdf2openapi.browser as b; "
        "print(b.extract_json('x.pdf', 't', '[[[0,0,50,10,\"GET\"],[60,0,120,10,\"/v1/a\"]]]'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '"/v1/a"' in out.stdout
