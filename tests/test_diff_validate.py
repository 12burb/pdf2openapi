from __future__ import annotations

import httpx
import yaml
from conftest import PdfBuilder

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


def _op_spec(op, security=None, schemes=None):
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "t", "version": "1"},
        "paths": {"/items": {"post": op}},
    }
    if schemes:
        spec["components"] = {"securitySchemes": schemes}
        spec["security"] = security or [{k: []} for k in schemes]
    return spec


def _body(schema_, required=True):
    return {
        "requestBody": {"required": required, "content": {"application/json": {"schema": schema_}}},
        "responses": {"200": {"description": "ok"}},
    }


def _resp(schema_):
    return {
        "responses": {
            "200": {"description": "ok", "content": {"application/json": {"schema": schema_}}}
        }
    }


def test_diff_auth_definition_change_is_breaking():
    old = _op_spec(
        _resp({}), schemes={"apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-Old-Key"}}
    )
    new = _op_spec(
        _resp({}), schemes={"apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-New-Key"}}
    )
    report = diff_specs(old, new)
    assert any("X-New-Key" in c.detail and c.breaking for c in report.changes)


def test_diff_response_field_becoming_optional_is_breaking():
    obj = {"type": "object", "properties": {"id": {"type": "string"}}}
    report = diff_specs(_op_spec(_resp({**obj, "required": ["id"]})), _op_spec(_resp(obj)))
    assert [c.detail for c in report.breaking] == ["response field 'id' is now optional"]


def test_diff_array_body_and_body_requiredness():
    old = _op_spec(_body({"type": "array", "items": {"type": "string"}}, required=False))
    new = _op_spec(_body({"type": "array", "items": {"type": "integer"}}, required=True))
    details = {c.detail for c in diff_specs(old, new).breaking}
    assert "request body type array<string> -> array<integer>" in details
    assert "request body is now required" in details


def test_diff_removed_input_is_breaking():
    def spec(params):
        return _op_spec({"parameters": params, "responses": {"200": {"description": "ok"}}})

    old = spec([{"name": "include", "in": "query", "schema": {"type": "string"}}])
    report = diff_specs(old, spec([]))
    assert [c.detail for c in report.breaking] == ["query param 'include' removed"]


def test_server_prefix_not_duplicated(vendor_pdf):
    spec = extract(vendor_pdf, ocr=False).spec
    assert spec["servers"] == [{"url": "https://api.acme.test"}]
    assert all(p.startswith("/v2/") for p in spec["paths"])


def test_probe_skips_network_path_urls():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200)

    spec = {
        "paths": {
            "//attacker.example/collect": {"get": {"responses": {}}},
            "/ok": {"get": {"responses": {"200": {"description": "ok"}}}},
        }
    }
    results = probe(spec, "https://sandbox.test", transport=httpx.MockTransport(handler))
    assert calls == ["https://sandbox.test/ok"]
    assert results[0].verdict.startswith("skipped")


def test_load_spec_refuses_private_urls(monkeypatch):
    import pytest

    monkeypatch.delenv("PDF2OPENAPI_ALLOW_PRIVATE_URLS", raising=False)
    with pytest.raises(ValueError, match="non-public"):
        load_spec("http://127.0.0.1:9/spec.yaml")


def test_load_spec_url_follows_redirect_and_cleans_up(vendor_pdf, tmp_path, monkeypatch):
    import tempfile

    tmpdir = tmp_path / "downloads"
    tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
    pdf_bytes = vendor_pdf.read_bytes()

    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/docs.pdf"})
        return httpx.Response(200, content=pdf_bytes)

    spec = load_spec("https://vendor.test/old", ocr=False, transport=httpx.MockTransport(handler))
    assert "/v2/orders" in spec["paths"]
    assert list(tmpdir.iterdir()) == []


def test_diff_oauth_scope_change_is_breaking():
    old = {"security": [{"oauth": ["read"]}], "paths": {}}
    new = {"security": [{"oauth": ["write"]}], "paths": {}}
    report = diff_specs(old, new)
    assert any("oauth[read]" in c.detail and c.breaking for c in report.changes)


def test_fetch_pins_checked_ip(monkeypatch):
    import socket

    from pdf2openapi.diff import _request_target

    monkeypatch.delenv("PDF2OPENAPI_ALLOW_PRIVATE_URLS", raising=False)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )
    url, headers, ext = _request_target("https://u:p@docs.vendor.com:8443/a.pdf", pin=True)
    assert url == "https://93.184.216.34:8443/a.pdf"
    assert headers == {"Host": "docs.vendor.com:8443"}
    assert ext == {"sni_hostname": "docs.vendor.com"}


def test_fetch_credentials_not_sent_to_redirect_target():
    from pdf2openapi.diff import fetch_url

    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("authorization")))
        if request.url.host == "docs.vendor.test":
            return httpx.Response(302, headers={"location": "https://cdn.other.test/a.pdf"})
        return httpx.Response(200, content=b"ok")

    body = fetch_url("https://me:pw@docs.vendor.test/a.pdf", transport=httpx.MockTransport(handler))
    assert body == b"ok"
    assert seen[0][0] == "docs.vendor.test" and seen[0][1].startswith("Basic ")
    assert seen[1] == ("cdn.other.test", None)


def test_cli_extract_tip_only_without_ai(tmp_path, capsys):
    b = PdfBuilder().new_page()
    b.line("POST /v1/things")
    pdf_path = b.save(tmp_path / "thin.pdf")
    assert main(["extract", str(pdf_path), "-o", str(tmp_path / "t.yaml"), "--no-ocr"]) == 0
    err = capsys.readouterr().err
    assert "tip: 1 of 1 endpoints look incomplete" in err and "--provider ollama" in err


def test_inferred_response_warnings_are_collapsed():
    from pdf2openapi.validate import validate_spec

    op = {"responses": {"200": {"description": "ok"}}, "x-inferred-responses": True}
    spec = {"openapi": "3.0.3", "info": {"title": "t", "version": "1"}, "paths": {}}
    spec["paths"] = {f"/p{i}": {"get": dict(op)} for i in range(5)}
    assert validate_spec(spec)[1] == [
        "5 endpoints have no status codes in the docs; assumed 200 (e.g. GET /p0)"
    ]
    spec["paths"] = {"/p0": {"get": dict(op)}}
    assert validate_spec(spec)[1] == ["GET /p0: no status codes in the docs; assumed 200"]
