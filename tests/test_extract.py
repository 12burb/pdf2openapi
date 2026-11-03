from __future__ import annotations

import pytest
from conftest import build_vendor_pdf

from pdf2openapi import extract
from pdf2openapi.extract import normalize_path
from pdf2openapi.pdf import ocr_available
from pdf2openapi.schema import flatten, parse_type
from pdf2openapi.validate import validate_spec


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/v2/orders/:order_id", "/v2/orders/{order_id}"),
        ("/v2/orders/<int:id>.", "/v2/orders/{id}"),
        ("/v2/orders/{{id}}/items/", "/v2/orders/{id}/items"),
        ("/search?q=x", "/search"),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("int64", {"type": "integer", "format": "int64"}),
        ("string[]", {"type": "array", "items": {"type": "string"}}),
        ("array of strings", {"type": "array", "items": {"type": "string"}}),
        ("Description", None),
    ],
)
def test_parse_type(label, expected):
    assert parse_type(label) == expected


def test_extracts_endpoints_with_citations(vendor_pdf):
    result = extract(vendor_pdf, ocr=False)
    spec = result.spec

    assert spec["info"]["title"] == "Acme Payments API Reference"
    assert spec["info"]["version"] == "2.1"
    assert spec["servers"] == [{"url": "https://api.acme.test/v2"}]
    assert spec["components"]["securitySchemes"] == {
        "apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-Acme-Key"}
    }
    assert set(spec["paths"]) == {"/v2/orders", "/v2/orders/{order_id}"}

    create = spec["paths"]["/v2/orders"]["post"]
    assert create["summary"] == "Create an order"
    assert create["x-source-pages"] == [2]
    body = create["requestBody"]["content"]["application/json"]["schema"]
    assert flatten(body) == {
        "amount": {"type": "integer", "required": True},
        "currency": {"type": "string", "required": True},
        "customer": {"type": "object", "required": False},
        "customer.email": {"type": "string", "required": False},
    }
    assert set(create["responses"]) == {"201", "400"}
    created = create["responses"]["201"]["content"]["application/json"]
    assert created["example"]["id"] == "ord_123"
    assert set(created["schema"]["properties"]) == {"id", "amount", "status"}

    get = spec["paths"]["/v2/orders/{order_id}"]["get"]
    assert get["x-source-pages"] == [3]
    params = {(p["in"], p["name"]): p for p in get["parameters"]}
    assert params[("path", "order_id")]["required"] is True
    assert "required" not in params[("query", "expand")]
    assert set(get["responses"]) == {"200", "404"}


def test_output_is_valid_openapi(vendor_pdf):
    errors, _ = validate_spec(extract(vendor_pdf, ocr=False).spec)
    assert errors == []


def test_unique_operation_ids(vendor_pdf_v2):
    spec = extract(vendor_pdf_v2, ocr=False).spec
    ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()]
    assert len(ids) == len(set(ids))
    assert "postOrders" in ids


@pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
def test_ocr_finds_endpoint_in_screenshot(tmp_path):
    pdf_path = build_vendor_pdf(tmp_path / "shot.pdf", screenshot=True)
    spec = extract(pdf_path).spec
    op = spec["paths"]["/v2/orders/{order_id}"]["delete"]
    assert op["x-source-pages"] == [4]
    assert "204" in op["responses"]


def test_no_endpoints_warns(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((60, 60), "Just a brochure with no API in it.")
    doc.save(tmp_path / "x.pdf")
    result = extract(tmp_path / "x.pdf", ocr=False)
    assert result.spec["paths"] == {}
    assert any("no endpoints found" in w for w in result.warnings)
