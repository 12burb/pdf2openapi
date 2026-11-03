from __future__ import annotations

import httpx
import yaml

from pdf2openapi.cli import main
from pdf2openapi.diff import diff_specs, load_spec
from pdf2openapi.extract import extract
from pdf2openapi.validate import probe


def test_diff_between_pdf_versions(vendor_pdf, vendor_pdf_v2):
    report = diff_specs(load_spec(str(vendor_pdf)), load_spec(str(vendor_pdf_v2)))
    assert report.added == ["GET /v2/refunds"]
    assert report.removed == []
    details = {c.detail: c.breaking for c in report.changes}
    assert details["body field 'amount' type integer -> number"] is True
    assert details["body field 'reference' added (required)"] is True
    assert len(report.breaking) == 2
    md = report.to_markdown()
    assert "## API changes (2 breaking)" in md
    assert "`GET /v2/refunds`" in md


def test_diff_same_doc_is_empty(vendor_pdf):
    spec = load_spec(str(vendor_pdf))
    assert diff_specs(spec, spec).empty


def test_diff_detects_removed_endpoint_and_renamed_path_param():
    old = {
        "paths": {
            "/a/{id}": {"get": {"responses": {"200": {"description": "ok"}}}},
            "/b": {"get": {"responses": {"200": {"description": "ok"}}}},
        }
    }
    new = {"paths": {"/a/{thing_id}": {"get": {"responses": {"200": {"description": "ok"}}}}}}
    report = diff_specs(old, new)
    assert report.removed == ["GET /b"]
    assert report.changes == []


def test_cli_extract_and_diff_exit_codes(tmp_path, vendor_pdf, vendor_pdf_v2, capsys):
    out = tmp_path / "spec.yaml"
    assert main(["extract", str(vendor_pdf), "-o", str(out), "--no-ocr"]) == 0
    assert "/v2/orders" in yaml.safe_load(out.read_text())["paths"]
    assert main(["validate", str(out)]) == 0
    assert main(["diff", str(out), str(out)]) == 0
    assert main(["diff", str(out), str(vendor_pdf_v2), "--fail-on-breaking"]) == 1
    assert "BREAKING" in capsys.readouterr().out


def test_cli_ai_without_key_fails_cleanly(vendor_pdf, monkeypatch, capsys):
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PDF2OPENAPI_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    assert main(["extract", str(vendor_pdf), "--ai"]) == 2
    assert "no AI provider configured" in capsys.readouterr().err


def test_probe_reports_wrong_paths(vendor_pdf_v2):
    spec = extract(vendor_pdf_v2, ocr=False).spec

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Acme-Key"] == "k"
        return httpx.Response(404)

    results = probe(
        spec, "https://sandbox.test", {"X-Acme-Key": "k"}, transport=httpx.MockTransport(handler)
    )
    by_key = {r.endpoint: r for r in results}
    assert by_key["GET /v2/refunds"].status == 404
    assert "wrong path" in by_key["GET /v2/refunds"].verdict
    assert by_key["GET /v2/orders/{order_id}"].status is None
