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
    assert spec["servers"] == [{"url": "https://api.acme.test"}]
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


def test_real_world_layouts(tmp_path):
    """Patterns seen in real vendor PDFs: label rows, reordered columns, wrapped paths."""
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("2.2. SendSms")
    b.row("Method", "URL")
    b.line("POST   https://api.vendor.test/v1/SendSMS")
    b.line("2.2.1   Parameters")
    b.row("Parameter", "Mandatory", "Type", "Description")
    b.row("toNumber", "Yes", "String", "Phone to send to")
    b.row("text", "No", "String", "Message")
    b.line("2.2.2   Responses")
    b.row("Name", "Type", "Description")
    b.row("status", "string", "Result")
    b.line("Lookup a number")
    b.line("GET /v1/AssignNumber?Number=xxx&format=json")
    b.line("Download media")
    b.line("GET /v1/media/{file.ext}/")
    b.line("content")
    path = b.save(tmp_path / "real.pdf")

    spec = extract(path, ocr=False).spec
    send = spec["paths"]["/v1/SendSMS"]["post"]
    assert send["summary"] == "SendSms"
    assert "parameters" not in send
    body = send["requestBody"]["content"]["application/json"]["schema"]
    assert flatten(body) == {
        "toNumber": {"type": "string", "required": True},
        "text": {"type": "string", "required": False},
    }
    resp = send["responses"]["200"]["content"]["application/json"]["schema"]
    assert list(resp["properties"]) == ["status"]

    lookup = spec["paths"]["/v1/AssignNumber"]["get"]
    assert [p["name"] for p in lookup["parameters"]] == ["Number", "format"]
    assert "/v1/media/{file_ext}/content" in spec["paths"]
    assert validate_spec(spec)[0] == []


def test_heading_after_trailing_slash_is_not_joined(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("List orders")
    b.line("GET /orders/")
    b.line("Response")
    b.line("200 OK")
    b.line("Wrapped")
    b.line("GET /v1/accounts/{account")
    b.line("_id}/balance")
    path = b.save(tmp_path / "slash.pdf")
    paths = extract(path, ocr=False).spec["paths"]
    assert "/orders" in paths
    assert "/v1/accounts/{account_id}/balance" in paths
    assert not any("Response" in p for p in paths)


def test_method_and_url_on_separate_lines(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("Get PAR URL")
    b.line("- Method: GET")
    b.line("- Path: /objectstore/getParURL")
    b.row("Parameter", "Value Type", "Description")
    b.row("filename", "String", "File name")
    b.line("Payment request")
    b.line("API URL")
    b.line("https://tt.pay.test/plugins/payments/request")
    b.line("HTTP")
    b.line("POST")
    b.row("Name", "Type", "Required")
    b.row("total", "number", "Yes")
    path = b.save(tmp_path / "split.pdf")
    spec = extract(path, ocr=False).spec
    assert [p["name"] for p in spec["paths"]["/objectstore/getParURL"]["get"]["parameters"]] == [
        "filename"
    ]
    post = spec["paths"]["/plugins/payments/request"]["post"]
    assert "total" in post["requestBody"]["content"]["application/json"]["schema"]["properties"]
    assert {"url": "https://tt.pay.test"} in spec["servers"]


def test_uri_without_method_is_guessed(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("4.1.1 All Active Bids Report")
    b.line("4.1.1.2.1 URI to access the Web API")
    b.line("/api-AllActiveBidsReport/v1")
    b.line("4.1.1.3.1 API Request Parameters")
    b.row("Name", "Type", "Required")
    b.row("gasDay", "date", "Yes")
    path = b.save(tmp_path / "guess.pdf")
    result = extract(path, ocr=False)
    op = result.spec["paths"]["/api-AllActiveBidsReport/v1"]["post"]
    assert op["x-method-guessed"] is True
    assert any("guessed" in w for w in result.warnings)


def test_xsd_table_rows_become_body_fields(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("4.1.1 All Active Bids Report")
    b.line("URI to access the Web API")
    b.line("/api-AllActiveBidsReport/v1")
    b.line("API Request Parameters")
    b.row("xml", "json")
    b.row("1", "MOS", "xs:string", "Yes")
    b.row("1", "GAS_DAY", "xs:date", "No")
    path = b.save(tmp_path / "xsd.pdf")
    op = extract(path, ocr=False).spec["paths"]["/api-AllActiveBidsReport/v1"]["post"]
    body = op["requestBody"]["content"]["application/json"]["schema"]
    assert body["properties"]["GAS_DAY"]["format"] == "date"
    assert set(body["properties"]) == {"MOS", "GAS_DAY"}
    assert body["required"] == ["MOS"]
    assert op["summary"] == "All Active Bids Report"
